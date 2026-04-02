"""
Corners Over/Under 10.5 Backtest: Walk-forward evaluation using Betfair exchange odds.

Reuses the same 4-model ensemble architecture as Over/Under 2.5 and BTTS:
  - XGBoost + LightGBM + Logistic Regression + Dixon-Coles
  - Per-model logit-shift calibration
  - Model-market blend with agreement gating
  - Regime detection + refined Kelly staking + two-phase early season

Key differences from Over/Under 2.5 and BTTS:
  - Target: total corners > 10.5 (Home_Corners + Away_Corners > 10.5)
  - Odds: Corner_Over_Odds / Corner_Under_Odds from Betfair exchange
  - Dixon-Coles: P(Over 10.5 corners) via Poisson with corner-specific ratings
  - Sides: "over" / "under"
  - Features: CORNERS_ALL_FEATURES (pressing, corner rolling stats, set piece style)
  - S25 settlement: uses Betfair Corner_Winner column (no corner counts in CSV)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import run_pipeline
from corners_data import load_corner_odds, merge_corner_odds
from config import CORNERS_ALL_FEATURES
from model import (train_xgb_corners, train_lgb_corners, train_logreg, _fill_nan_median,
                    _clip_scaled, DixonColesPredictor, tune_dc_params_corners)

# Reuse helpers from backtest.py
from backtest import (
    _calibrate, _calibrate_single, _lr_predict,
    RegimeDetector, refined_kelly, compute_drawdown_factor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Corners Default Configuration
# ═══════════════════════════════════════════════════════════════════════════════

CORNERS_DEFAULT_CONFIG = {
    "blend_weight": 0.40,       # Exchange odds are sharper, need more model weight
    "min_edge": 0.02,           # Conservative for new market
    "min_agree": 2,
    "kelly_fraction": 0.20,     # Conservative for new market
    "max_stake_pct": 0.04,      # Exchange liquidity concerns
    "regime_detection": False,   # Disabled: 50% base rate causes noise-driven triggers
    "regime_trigger": 0.15,     # 15pp = ~2 sigma for 50% base rate (if enabled)
    "refined_staking": True,
    "calibration_strength": 0.5, # Dampen logit-shift calibration (unstable at 50%)
    # Ensemble weights: XGB, LGB, LR, DC (DC downweighted — Poisson weak for corners)
    "model_weights": [0.30, 0.30, 0.25, 0.15],
    # Two-phase early-season strategy
    "early_season": True,
    "early_season_matches": 80,
    "early_blend_weight": 0.25,
    "early_min_edge": 0.025,
    "early_kelly_fraction": 0.12,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration helper (dampened for ~50% base rate stability)
# ═══════════════════════════════════════════════════════════════════════════════

def _calibrate_damped(probs: np.ndarray, base_rate: float,
                      strength: float = 0.5) -> tuple[np.ndarray, float]:
    """Logit-shift calibration with dampening for ~50% base rate stability.

    At 50% base rate, small fluctuations create large logit shifts that
    destabilize predictions. The strength parameter shrinks the shift toward
    zero to reduce overreaction while still correcting systematic bias.

    Args:
        probs: Model probability predictions.
        base_rate: Target base rate.
        strength: Shrinkage factor (0.0 = no calibration, 1.0 = full shift).

    Returns:
        Tuple of (calibrated probs, damped shift).
    """
    _, raw_shift = _calibrate(probs, base_rate)
    damped_shift = raw_shift * strength
    logits = np.log(probs / (1 - probs + 1e-10))
    calibrated = 1 / (1 + np.exp(-(logits - damped_shift)))
    return calibrated, damped_shift


# ═══════════════════════════════════════════════════════════════════════════════
# Core corners backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

def corners_backtest_season(train_df: pd.DataFrame, test_df: pd.DataFrame,
                            features: list[str], config: dict | None = None,
                            dc_kwargs: dict | None = None,
                            cumulative_bankroll: float = 1.0,
                            peak_bankroll: float = 1.0):
    """Backtest one season for Corners O/U 10.5 market.

    Args:
        train_df: Training data with features and corner counts.
        test_df: Test season data with features and corner odds.
        features: Feature column names.
        config: Betting configuration parameters.
        dc_kwargs: Dixon-Coles constructor kwargs.
        cumulative_bankroll: Running bankroll from previous seasons.
        peak_bankroll: Peak bankroll for drawdown calculation.

    Returns:
        Tuple of (bets_df, metrics, cumulative_bankroll, peak_bankroll).
    """
    if config is None:
        config = CORNERS_DEFAULT_CONFIG.copy()

    blend_w = config.get("blend_weight", 0.40)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.20)
    max_stake_pct = config.get("max_stake_pct", 0.04)
    use_regime = config.get("regime_detection", True)
    use_refined_staking = config.get("refined_staking", True)

    # Two-phase early-season settings
    use_early = config.get("early_season", True)
    early_matches = config.get("early_season_matches", 80)
    early_blend_w = config.get("early_blend_weight", 0.25)
    early_min_edge = config.get("early_min_edge", 0.025)
    early_kelly = config.get("early_kelly_fraction", 0.12)

    # --- Target: Over 10.5 total corners ---
    # For training: use actual corner counts
    train_has_corners = train_df["Home_Corners"].notna() & train_df["Away_Corners"].notna()
    train_use = train_df[train_has_corners].copy()
    y_train = ((train_use["Home_Corners"] + train_use["Away_Corners"]) > 10.5).astype(int).values

    # For test: use corner counts if available, else use Betfair settlement
    test_has_corners = test_df["Home_Corners"].notna() & test_df["Away_Corners"].notna()
    if test_has_corners.all():
        y_test = ((test_df["Home_Corners"] + test_df["Away_Corners"]) > 10.5).astype(int).values
        settlement_source = "corners"
    elif "Corner_Winner" in test_df.columns:
        # S25: use Betfair settlement column
        y_test = (test_df["Corner_Winner"] == "over").astype(int).values
        settlement_source = "betfair"
    else:
        # No settlement data at all — can't backtest
        return pd.DataFrame(), {"season": 0, "n_bets": 0, "roi": 0}, cumulative_bankroll, peak_bankroll

    X_train = train_use[features].values

    # Early stopping split
    train_seasons = sorted(train_use["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_use["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_use.loc[es_train_mask, features].values
    y_es_train = ((train_use.loc[es_train_mask, "Home_Corners"] +
                    train_use.loc[es_train_mask, "Away_Corners"]) > 10.5).astype(int).values
    X_es_val = train_use.loc[es_val_mask, features].values
    y_es_val = ((train_use.loc[es_val_mask, "Home_Corners"] +
                  train_use.loc[es_val_mask, "Away_Corners"]) > 10.5).astype(int).values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_use) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    if dc_kwargs is None:
        dc_kwargs = {}

    # --- Train 4 base models (on corners target) ---
    xgb_m = train_xgb_corners(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb_corners(X_es_train, y_es_train, X_es_val, y_es_val,
                               feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)

    # Dixon-Coles: predict P(Over 10.5 corners) using corner-specific ratings
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_use)           # fit goals (needed for infrastructure)
    dc_m.fit_corners(train_use)   # fit corner-specific ratings

    # --- Raw predictions on test set ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
    dc_raw = dc_m.predict_proba_corners_df(test_df)  # Corners-specific

    # --- Calibration (dampened for ~50% base rate stability) ---
    cal_strength = config.get("calibration_strength", 0.5)
    recent_s = sorted(train_seasons)[-3:]  # 3 seasons for stability at 50%
    recent_mask = train_use["SeasonIndex"].isin(recent_s)
    if recent_mask.sum() >= 100:
        recent_train = train_use.loc[recent_mask]
        base_rate = ((recent_train["Home_Corners"] + recent_train["Away_Corners"]) > 10.5).mean()
    else:
        base_rate = y_train.mean()

    xgb_cal, xgb_shift = _calibrate_damped(xgb_raw, base_rate, cal_strength)
    lgb_cal, lgb_shift = _calibrate_damped(lgb_raw, base_rate, cal_strength)
    lr_cal, lr_shift = _calibrate_damped(lr_raw, base_rate, cal_strength)
    dc_cal, dc_shift = _calibrate_damped(dc_raw, base_rate, cal_strength)

    # Configurable ensemble weights (default: XGB=0.30, LGB=0.30, LR=0.25, DC=0.15)
    mw = config.get("model_weights", [0.30, 0.30, 0.25, 0.15])
    ensemble_p = mw[0] * xgb_cal + mw[1] * lgb_cal + mw[2] * lr_cal + mw[3] * dc_cal

    try:
        auc = roc_auc_score(y_test, ensemble_p)
    except ValueError:
        auc = 0.5

    # --- Regime detector (tracks Over 10.5 rate) ---
    regime = RegimeDetector(
        prior_base_rate=base_rate,
        window=config.get("regime_window", 40),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.15),
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

        # Update regime detector
        regime.update(actual)

        # Recalibrate if regime shifted (dampened)
        if use_regime and regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            _, xgb_s_adj = _calibrate_damped(xgb_raw, adj_rate, cal_strength)
            _, lgb_s_adj = _calibrate_damped(lgb_raw, adj_rate, cal_strength)
            _, lr_s_adj = _calibrate_damped(lr_raw, adj_rate, cal_strength)
            _, dc_s_adj = _calibrate_damped(dc_raw, adj_rate, cal_strength)

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
        model_over = np.dot(per_model, mw)   # P(Over 10.5)
        model_under = 1 - model_over                     # P(Under 10.5)

        odds_over = row.get("Corner_Over_Odds", np.nan)
        odds_under = row.get("Corner_Under_Odds", np.nan)

        if pd.isna(odds_over) or pd.isna(odds_under):
            continue
        if odds_over <= 1.0 or odds_under <= 1.0:
            continue

        # Fair market probabilities (Betfair exchange ~2% commission, close to fair)
        raw_o = 1.0 / odds_over
        raw_u = 1.0 / odds_under
        overround = raw_o + raw_u
        fair_over = raw_o / overround
        fair_under = raw_u / overround

        # Drawdown factor
        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll) if use_refined_staking else 1.0

        # Two-phase
        is_early = use_early and match_num < early_matches
        active_blend_w = early_blend_w if is_early else blend_w
        active_min_edge = early_min_edge if is_early else min_edge
        active_kelly = early_kelly if is_early else kelly_fraction

        for side, model_p, fair_p, odds in [
            ("over", model_over, fair_over, odds_over),
            ("under", model_under, fair_under, odds_under),
        ]:
            blended_p = active_blend_w * model_p + (1 - active_blend_w) * fair_p
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            # Model agreement
            if side == "over":
                n_agree = int(np.sum(per_model > fair_over))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_under))

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

            won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
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
                "actual_over105": actual,
                "regime_rate": regime.get_adjusted_base_rate(),
                "dd_factor": dd_factor,
                "phase": "early" if is_early else "regime",
                "settlement": settlement_source,
            })

    bets_df = pd.DataFrame(bets)

    metrics = {
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
        "n_matches": len(test_df),
        "n_bets": len(bets_df),
        "auc": auc,
        "base_rate": base_rate,
        "actual_over105_rate": float(y_test.mean()),
        "model_mean": float(ensemble_p.mean()),
        "regime_shifts": regime_shifts,
        "final_regime_rate": regime.get_adjusted_base_rate(),
        "cumulative_bankroll": cumulative_bankroll,
        "peak_bankroll": peak_bankroll,
        "settlement_source": settlement_source,
    }

    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["avg_edge"] = bets_df["edge"].mean()
        metrics["avg_odds"] = bets_df["odds"].mean()
        metrics["avg_stake"] = bets_df["stake_pct"].mean()
        metrics["roi"] = bets_df["profit_pct"].sum() / bets_df["stake_pct"].sum()
        for side in ["over", "under"]:
            sb = bets_df[bets_df["side"] == side]
            if len(sb) > 0:
                metrics[f"n_{side}"] = len(sb)
                metrics[f"{side}_roi"] = sb["profit_pct"].sum() / sb["stake_pct"].sum()
    else:
        metrics.update({"total_profit_pct": 0, "win_rate": 0, "roi": 0})

    return bets_df, metrics, cumulative_bankroll, peak_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward corners backtest
# ═══════════════════════════════════════════════════════════════════════════════

def run_corners_backtest(config: dict | None = None, start_season: int = 19,
                         end_season: int = 25, verbose: bool = True):
    """Walk-forward Corners O/U 10.5 backtest across multiple seasons.

    Corner odds from Betfair available from S16+, but we backtest S19-S25
    (matching O/U and BTTS range) for fair comparison.

    Args:
        config: Betting configuration. Uses CORNERS_DEFAULT_CONFIG if None.
        start_season: First season to test (default 19 = 2019/20).
        end_season: Last season to test (default 25 = 2025/26).
        verbose: Print progress and results.

    Returns:
        Tuple of (total_bets_df, all_metrics, final_bankroll).
    """
    if config is None:
        config = CORNERS_DEFAULT_CONFIG.copy()

    bw = config.get("blend_weight", 0.40)

    # Load pipeline data — use corners-specific feature set
    data = run_pipeline(verbose=verbose)
    full_df = data["full_df"]
    features = [f for f in CORNERS_ALL_FEATURES if f in full_df.columns]
    if verbose:
        print(f"Corners features: {len(features)} (vs {len(data['features'])} O/U features)")

    # Merge corner odds
    corner_odds = load_corner_odds()
    full_df = merge_corner_odds(full_df, corner_odds)

    # Tune Dixon-Coles params (uses goals — corner DC uses separate fit_corners)
    if verbose:
        print("Tuning Dixon-Coles hyperparameters...")
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_corners(tune_df)

    all_bets = []
    all_metrics = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    if verbose:
        print(f"\n{'='*90}")
        print(f"CORNERS O/U 10.5 WALK-FORWARD BACKTEST ({bw:.0%} model / {1-bw:.0%} market blend)")
        print(f"{'='*90}")
        print(f"{'Season':>8s} {'Year':>8s} {'AUC':>5s} {'Bets':>5s} "
              f"{'O/U':>7s} {'Win%':>5s} {'ROI':>6s} "
              f"{'O_ROI':>6s} {'U_ROI':>6s} {'Regime':>8s} {'Bank':>9s}")

    for season in range(start_season, end_season + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        # Need corner odds for test season
        has_odds = test_df["Corner_Over_Odds"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            if verbose:
                print(f"S{season:>5d}  skipped (odds={has_odds}, train={len(train_df)})")
            continue

        bets_df, metrics, cumulative_bankroll, peak_bankroll = corners_backtest_season(
            train_df, test_df, features, config=config, dc_kwargs=dc_kwargs,
            cumulative_bankroll=cumulative_bankroll, peak_bankroll=peak_bankroll,
        )

        all_bets.append(bets_df)
        all_metrics.append(metrics)

        if verbose:
            m = metrics
            year = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
            n_o = m.get("n_over", 0)
            n_u = m.get("n_under", 0)
            o_roi = m.get("over_roi", 0)
            u_roi = m.get("under_roi", 0)
            regime_str = f"{m['regime_shifts']:>2d}x>{m['final_regime_rate']:.0%}"
            print(f"S{m['season']:>5d}  {year:>8s} {m['auc']:>5.3f} {m['n_bets']:>5d} "
                  f"{n_o:>3d}O/{n_u:<3d}U {m['win_rate']:>5.1%} {m['roi']:>+6.1%} "
                  f"{o_roi:>+6.1%} {u_roi:>+6.1%} {regime_str:>8s} "
                  f"{cumulative_bankroll:>9.4f}")

    # Summary
    if all_bets and any(len(b) > 0 for b in all_bets):
        total_bets = pd.concat(all_bets, ignore_index=True)

        if verbose and len(total_bets) > 0:
            total_staked = total_bets["stake_pct"].sum()
            total_profit = total_bets["profit_pct"].sum()
            overall_roi = total_profit / total_staked if total_staked > 0 else 0

            print(f"\n{'='*90}")
            print(f"CORNERS O/U 10.5 BACKTEST SUMMARY ({bw:.0%} model / {1-bw:.0%} market blend)")
            print(f"{'='*90}")
            print(f"  Seasons: S{start_season}-S{end_season} ({len(all_metrics)} tested)")
            print(f"  Total bets: {len(total_bets)} ({len(total_bets)/len(all_metrics):.0f}/season)")
            print(f"  Win rate: {total_bets['won'].mean():.1%}")
            print(f"  Avg edge: {total_bets['edge'].mean():.2%}")
            print(f"  Avg odds: {total_bets['odds'].mean():.2f}")
            print(f"  Overall ROI: {overall_roi:+.1%}")
            print(f"  Final bankroll: {cumulative_bankroll:.4f} ({(cumulative_bankroll - 1)*100:+.1f}%)")

            for side in ["over", "under"]:
                sb = total_bets[total_bets["side"] == side]
                if len(sb) > 0:
                    sroi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"\n  {side.upper():>5s}: {len(sb)} bets, "
                          f"win {sb['won'].mean():.1%}, ROI {sroi:+.1%}")

            profitable = sum(1 for m in all_metrics
                             if m.get("total_profit_pct", 0) > 0)
            print(f"\n  Profitable seasons: {profitable}/{len(all_metrics)}")

            # Corner rate diagnostics
            print(f"\n  Season diagnostics:")
            for m in all_metrics:
                yr = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
                regime_info = ""
                if m.get("regime_shifts", 0) > 0:
                    regime_info = (f" REGIME: {m['regime_shifts']}x -> "
                                   f"{m['final_regime_rate']:.1%}")
                print(f"    {yr}: base_rate={m['base_rate']:.1%}, "
                      f"actual_over105={m['actual_over105_rate']:.1%}, "
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
        print("Usage: python corners_backtest.py [--start N] [--end N]")
        print("  --start N   Start season (default 19)")
        print("  --end N     End season (default 25)")
        sys.exit(0)

    start = 19
    end = 25
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])
    if "--end" in sys.argv:
        end = int(sys.argv[sys.argv.index("--end") + 1])

    print(f"Running Corners O/U 10.5 backtest S{start}-S{end}...")
    total_bets, metrics, bank = run_corners_backtest(
        config=CORNERS_DEFAULT_CONFIG,
        start_season=start,
        end_season=end,
        verbose=True,
    )
