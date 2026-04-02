"""
BTTS (Both Teams To Score) Backtest: Walk-forward evaluation using footiqo odds.

Reuses the same 4-model ensemble architecture as Over/Under 2.5:
  - XGBoost + LightGBM + Logistic Regression + Dixon-Coles
  - Per-model logit-shift calibration
  - Model-market blend with agreement gating
  - Regime detection + refined Kelly staking + two-phase early season

Key differences from Over/Under:
  - Target: BTTS (both teams scored) instead of Over 2.5
  - Odds: BTTSY/BTTSN from footiqo instead of B365
  - Dixon-Coles: P(BTTS) via inclusion-exclusion instead of P(Over 2.5)
  - Sides: "yes"/"no" instead of "over"/"under"
  - Features: same pipeline features (BTTS-relevant ones dominate naturally via XGB)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import run_pipeline
from btts_data import load_btts_odds, merge_btts_odds
from config import BTTS_ALL_FEATURES
from model import (train_xgb_btts, train_lgb_btts, train_logreg, _fill_nan_median,
                    _clip_scaled, DixonColesPredictor, tune_dc_params)

# Reuse helpers from backtest.py
from backtest import (
    _calibrate, _calibrate_single, _lr_predict,
    RegimeDetector, refined_kelly, compute_drawdown_factor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# BTTS Default Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BTTS_DEFAULT_CONFIG = {
    "blend_weight": 0.50,       # Higher than O/U: BTTS model has private features
    "min_edge": 0.015,          # Lower than O/U: BTTS market less efficient
    "min_agree": 2,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.05,
    "regime_detection": True,
    "refined_staking": True,
    # Two-phase early-season strategy
    "early_season": True,
    "early_season_matches": 80, # BTTS needs more warmup (rolling FTS/CS features)
    "early_blend_weight": 0.30, # Still conservative early but not crippled
    "early_min_edge": 0.02,     # Tighter early filter
    "early_kelly_fraction": 0.15,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Core BTTS backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

def btts_backtest_season(train_df, test_df, features, config=None, dc_kwargs=None,
                         cumulative_bankroll=1.0, peak_bankroll=1.0):
    """Backtest one season for BTTS market.

    Returns (bets_df, metrics, cumulative_bankroll, peak_bankroll).
    """
    if config is None:
        config = BTTS_DEFAULT_CONFIG.copy()

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)
    use_regime = config.get("regime_detection", True)
    use_refined_staking = config.get("refined_staking", True)

    # Two-phase early-season settings
    use_early = config.get("early_season", True)
    early_matches = config.get("early_season_matches", 60)
    early_blend_w = config.get("early_blend_weight", 0.20)
    early_min_edge = config.get("early_min_edge", 0.03)
    early_kelly = config.get("early_kelly_fraction", 0.15)

    # --- Target: BTTS ---
    y_train = ((train_df["Home_Goals"] > 0) & (train_df["Away_Goals"] > 0)).astype(int).values
    y_test = ((test_df["Home_Goals"] > 0) & (test_df["Away_Goals"] > 0)).astype(int).values

    X_train = train_df[features].values

    # Early stopping split
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = ((train_df.loc[es_train_mask, "Home_Goals"] > 0) &
                  (train_df.loc[es_train_mask, "Away_Goals"] > 0)).astype(int).values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = ((train_df.loc[es_val_mask, "Home_Goals"] > 0) &
                (train_df.loc[es_val_mask, "Away_Goals"] > 0)).astype(int).values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    if dc_kwargs is None:
        dc_kwargs = {}

    # --- Train 4 base models (on BTTS target, BTTS-tuned hyperparams) ---
    xgb_m = train_xgb_btts(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb_btts(X_es_train, y_es_train, X_es_val, y_es_val,
                            feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)

    # Dixon-Coles: predict P(BTTS) instead of P(Over 2.5)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Raw predictions on test set ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
    dc_raw = dc_m.predict_proba_btts_df(test_df)  # BTTS-specific

    # --- Calibration ---
    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    if recent_mask.sum() >= 100:
        recent_train = train_df.loc[recent_mask]
        base_rate = ((recent_train["Home_Goals"] > 0) &
                     (recent_train["Away_Goals"] > 0)).mean()
    else:
        base_rate = y_train.mean()

    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, lr_shift = _calibrate(lr_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    xgb_cal = np.array([_calibrate_single(p, xgb_shift) for p in xgb_raw])
    lgb_cal = np.array([_calibrate_single(p, lgb_shift) for p in lgb_raw])
    lr_cal = np.array([_calibrate_single(p, lr_shift) for p in lr_raw])
    dc_cal = np.array([_calibrate_single(p, dc_shift) for p in dc_raw])

    # Weighted ensemble: LR and DC produce wider prediction spreads for BTTS,
    # so they get more weight than compressed tree models.
    # Weights: XGB=0.20, LGB=0.20, LR=0.30, DC=0.30
    ensemble_p = 0.20 * xgb_cal + 0.20 * lgb_cal + 0.30 * lr_cal + 0.30 * dc_cal

    try:
        auc = roc_auc_score(y_test, ensemble_p)
    except ValueError:
        auc = 0.5

    # --- Regime detector (tracks BTTS rate instead of Over rate) ---
    regime = RegimeDetector(
        prior_base_rate=base_rate,
        window=config.get("regime_window", 40),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.04),
        min_matches=config.get("regime_min_matches", 15),
    )
    regime_shifts = 0

    # --- Bet simulation ---
    bets = []
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        actual = y_test[pred_idx]

        # Update regime detector with BTTS outcome
        regime.update(actual)

        # Recalibrate if regime shifted
        if use_regime and regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            _, xgb_s_adj = _calibrate(xgb_raw, adj_rate)
            _, lgb_s_adj = _calibrate(lgb_raw, adj_rate)
            _, lr_s_adj = _calibrate(lr_raw, adj_rate)
            _, dc_s_adj = _calibrate(dc_raw, adj_rate)

            xgb_p = _calibrate_single(xgb_raw[pred_idx], xgb_s_adj)
            lgb_p = _calibrate_single(lgb_raw[pred_idx], lgb_s_adj)
            lr_p = _calibrate_single(lr_raw[pred_idx], lr_s_adj)
            dc_p = _calibrate_single(dc_raw[pred_idx], dc_s_adj)
            regime_shifts += 1
        else:
            xgb_p = xgb_cal[pred_idx]
            lgb_p = lgb_cal[pred_idx]
            lr_p = lr_cal[pred_idx]
            dc_p = dc_cal[pred_idx]

        per_model = np.array([xgb_p, lgb_p, lr_p, dc_p])
        # Weighted ensemble matching calibrated weights (0.20, 0.20, 0.30, 0.30)
        model_weights = np.array([0.20, 0.20, 0.30, 0.30])
        model_yes = np.dot(per_model, model_weights)   # P(BTTS = Yes)
        model_no = 1 - model_yes                       # P(BTTS = No)

        odds_yes = row.get("BTTSY", np.nan)
        odds_no = row.get("BTTSN", np.nan)

        if pd.isna(odds_yes) or pd.isna(odds_no):
            continue
        if odds_yes <= 1.0 or odds_no <= 1.0:
            continue

        # Fair market probabilities (overround removed)
        raw_y = 1.0 / odds_yes
        raw_n = 1.0 / odds_no
        overround = raw_y + raw_n
        fair_yes = raw_y / overround
        fair_no = raw_n / overround

        # Drawdown factor
        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll) if use_refined_staking else 1.0

        # Two-phase
        is_early = use_early and match_num < early_matches
        active_blend_w = early_blend_w if is_early else blend_w
        active_min_edge = early_min_edge if is_early else min_edge
        active_kelly = early_kelly if is_early else kelly_fraction

        for side, model_p, fair_p, odds in [
            ("yes", model_yes, fair_yes, odds_yes),
            ("no", model_no, fair_no, odds_no),
        ]:
            blended_p = active_blend_w * model_p + (1 - active_blend_w) * fair_p
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            # Model agreement
            if side == "yes":
                n_agree = int(np.sum(per_model > fair_yes))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_no))

            if ev <= 0 or edge < active_min_edge or n_agree < min_agree:
                continue

            # Staking
            if use_refined_staking:
                stake = refined_kelly(
                    blended_p, odds, n_agree, edge,
                    kelly_fraction=active_kelly,
                    max_stake_pct=max_stake_pct,
                    drawdown_factor=dd_factor,
                )
            else:
                kelly = (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0
                stake = min(max_stake_pct, max(0, kelly * active_kelly))

            if stake <= 0:
                continue

            won = (side == "yes" and actual == 1) or (side == "no" and actual == 0)
            profit = stake * (odds - 1) if won else -stake

            # Update bankroll
            actual_stake = stake * cumulative_bankroll
            if won:
                cumulative_bankroll += actual_stake * (odds - 1)
            else:
                cumulative_bankroll -= actual_stake
            peak_bankroll = max(peak_bankroll, cumulative_bankroll)

            bets.append({
                "season": row.get("SeasonIndex", 0),
                "home": row.get("Home_Team", ""),
                "away": row.get("Away_Team", ""),
                "date": row.get("Date", ""),
                "side": side,
                "model_prob": model_p,
                "blended_prob": blended_p,
                "fair_prob": fair_p,
                "odds": odds,
                "edge": edge,
                "ev": ev,
                "n_agree": n_agree,
                "kelly": (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0,
                "stake_pct": stake,
                "won": won,
                "profit_pct": profit,
                "actual_btts": actual,
                "regime_rate": regime.get_adjusted_base_rate(),
                "dd_factor": dd_factor,
                "phase": "early" if is_early else "regime",
            })

    bets_df = pd.DataFrame(bets)

    metrics = {
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
        "n_matches": len(test_df),
        "n_bets": len(bets_df),
        "auc": auc,
        "base_rate": base_rate,
        "actual_btts_rate": float(y_test.mean()),
        "model_mean": float(ensemble_p.mean()),
        "regime_shifts": regime_shifts,
        "final_regime_rate": regime.get_adjusted_base_rate(),
        "cumulative_bankroll": cumulative_bankroll,
        "peak_bankroll": peak_bankroll,
    }

    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["avg_edge"] = bets_df["edge"].mean()
        metrics["avg_odds"] = bets_df["odds"].mean()
        metrics["avg_stake"] = bets_df["stake_pct"].mean()
        metrics["roi"] = bets_df["profit_pct"].sum() / bets_df["stake_pct"].sum()
        for side in ["yes", "no"]:
            sb = bets_df[bets_df["side"] == side]
            if len(sb) > 0:
                metrics[f"n_{side}"] = len(sb)
                metrics[f"{side}_roi"] = sb["profit_pct"].sum() / sb["stake_pct"].sum()
    else:
        metrics.update({"total_profit_pct": 0, "win_rate": 0, "roi": 0})

    return bets_df, metrics, cumulative_bankroll, peak_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward BTTS backtest
# ═══════════════════════════════════════════════════════════════════════════════

def run_btts_backtest(config=None, start_season=19, end_season=25, verbose=True):
    """Walk-forward BTTS backtest across multiple seasons.

    BTTS odds are only available from S15+, but we backtest S19-S25
    (matching Over/Under range) for fair comparison.
    """
    if config is None:
        config = BTTS_DEFAULT_CONFIG.copy()

    bw = config.get("blend_weight", 0.35)

    # Load pipeline data — use BTTS-specific feature set
    data = run_pipeline(verbose=verbose)
    full_df = data["full_df"]
    features = [f for f in BTTS_ALL_FEATURES if f in full_df.columns]
    if verbose:
        print(f"BTTS features: {len(features)} (vs {len(data['features'])} O/U features)")

    # Merge BTTS odds
    btts_odds = load_btts_odds()
    full_df = merge_btts_odds(full_df, btts_odds)

    # Tune Dixon-Coles params
    if verbose:
        print("Tuning Dixon-Coles hyperparameters...")
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)

    all_bets = []
    all_metrics = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    if verbose:
        print(f"\n{'='*85}")
        print(f"BTTS WALK-FORWARD BACKTEST ({bw:.0%} model / {1-bw:.0%} market blend)")
        print(f"{'='*85}")
        print(f"{'Season':>8s} {'Year':>8s} {'AUC':>5s} {'Bets':>5s} "
              f"{'Y/N':>7s} {'Win%':>5s} {'ROI':>6s} "
              f"{'Y_ROI':>6s} {'N_ROI':>6s} {'Regime':>8s} {'Bank':>9s}")

    for season in range(start_season, end_season + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        # Need BTTS odds for test season
        has_odds = test_df["BTTSY"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            if verbose:
                print(f"S{season:>5d}  skipped (odds={has_odds}, train={len(train_df)})")
            continue

        bets_df, metrics, cumulative_bankroll, peak_bankroll = btts_backtest_season(
            train_df, test_df, features, config=config, dc_kwargs=dc_kwargs,
            cumulative_bankroll=cumulative_bankroll, peak_bankroll=peak_bankroll,
        )

        all_bets.append(bets_df)
        all_metrics.append(metrics)

        if verbose:
            m = metrics
            year = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
            n_y = m.get("n_yes", 0)
            n_n = m.get("n_no", 0)
            y_roi = m.get("yes_roi", 0)
            n_roi = m.get("no_roi", 0)
            regime_str = f"{m['regime_shifts']:>2d}x>{m['final_regime_rate']:.0%}"
            print(f"S{m['season']:>5d}  {year:>8s} {m['auc']:>5.3f} {m['n_bets']:>5d} "
                  f"{n_y:>3d}Y/{n_n:<3d}N {m['win_rate']:>5.1%} {m['roi']:>+6.1%} "
                  f"{y_roi:>+6.1%} {n_roi:>+6.1%} {regime_str:>8s} "
                  f"{cumulative_bankroll:>9.4f}")

    # Summary
    if all_bets and any(len(b) > 0 for b in all_bets):
        total_bets = pd.concat(all_bets, ignore_index=True)

        if verbose and len(total_bets) > 0:
            total_staked = total_bets["stake_pct"].sum()
            total_profit = total_bets["profit_pct"].sum()
            overall_roi = total_profit / total_staked if total_staked > 0 else 0

            print(f"\n{'='*85}")
            print(f"BTTS BACKTEST SUMMARY ({bw:.0%} model / {1-bw:.0%} market blend)")
            print(f"{'='*85}")
            print(f"  Seasons: S{start_season}-S{end_season} ({len(all_metrics)} tested)")
            print(f"  Total bets: {len(total_bets)} ({len(total_bets)/len(all_metrics):.0f}/season)")
            print(f"  Win rate: {total_bets['won'].mean():.1%}")
            print(f"  Avg edge: {total_bets['edge'].mean():.2%}")
            print(f"  Avg odds: {total_bets['odds'].mean():.2f}")
            print(f"  Overall ROI: {overall_roi:+.1%}")
            print(f"  Final bankroll: {cumulative_bankroll:.4f} ({(cumulative_bankroll - 1)*100:+.1f}%)")

            for side in ["yes", "no"]:
                sb = total_bets[total_bets["side"] == side]
                if len(sb) > 0:
                    sroi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"\n  {side.upper():>5s}: {len(sb)} bets, "
                          f"win {sb['won'].mean():.1%}, ROI {sroi:+.1%}")

            profitable = sum(1 for m in all_metrics
                             if m.get("total_profit_pct", 0) > 0)
            print(f"\n  Profitable seasons: {profitable}/{len(all_metrics)}")

            # BTTS rate diagnostics
            print(f"\n  Season diagnostics:")
            for m in all_metrics:
                yr = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
                regime_info = ""
                if m.get("regime_shifts", 0) > 0:
                    regime_info = (f" REGIME: {m['regime_shifts']}x -> "
                                   f"{m['final_regime_rate']:.1%}")
                print(f"    {yr}: base_rate={m['base_rate']:.1%}, "
                      f"actual_btts={m['actual_btts_rate']:.1%}, "
                      f"model_mean={m['model_mean']:.1%}{regime_info}")

            # Max drawdown
            if "profit_pct" in total_bets.columns:
                cum_pl = total_bets["profit_pct"].cumsum()
                running_max = cum_pl.cummax()
                drawdowns = cum_pl - running_max
                max_dd = drawdowns.min()
                print(f"\n  Max drawdown: {max_dd*100:.1f}% of bankroll")

        return total_bets, all_metrics, cumulative_bankroll

    return None, all_metrics, cumulative_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--help" in sys.argv:
        print("Usage: python btts_backtest.py [--start N] [--end N]")
        print("  --start N   Start season (default 19)")
        print("  --end N     End season (default 25)")
        sys.exit(0)

    start = 19
    end = 25
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])
    if "--end" in sys.argv:
        end = int(sys.argv[sys.argv.index("--end") + 1])

    print(f"Running BTTS backtest S{start}-S{end}...")
    total_bets, metrics, bank = run_btts_backtest(
        config=BTTS_DEFAULT_CONFIG,
        start_season=start,
        end_season=end,
        verbose=True,
    )
