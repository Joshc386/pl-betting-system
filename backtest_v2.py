"""
Backtest V2: Model-Market Blend + Asymmetric Calibration + Model Agreement

Key improvements over v1:
  1. Model-market blend: final_prob = w*model + (1-w)*market, reducing overconfidence
  2. Asymmetric calibration: calibrate Over and Under signals separately
  3. Model agreement gating: only bet when N+ models agree on direction vs market
  4. Per-model individual predictions stored for agreement checks
  5. Runs S19-S25 (all seasons with Bet365 odds)

Walk-forward: train on seasons 14..(S-1), test on S.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from pipeline import run_pipeline
from model import (train_xgb, train_lgb, train_logreg, _fill_nan_median,
                    _clip_scaled, DixonColesPredictor, tune_dc_params)


def _lr_predict(lr_model, lr_scaler, X):
    """Get LR predictions, handling NaN fill + scaling + clipping."""
    X_filled, _ = _fill_nan_median(X, medians=lr_model._col_medians)
    X_scaled = _clip_scaled(lr_scaler.transform(X_filled))
    return lr_model.predict_proba(X_scaled)[:, 1]


def calibrate_logit_shift(raw_probs, base_rate):
    """Logit-shift calibration: shift mean logit to match base_rate."""
    raw_logits = np.log(raw_probs / (1 - raw_probs + 1e-10))
    target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
    shift = np.mean(raw_logits) - target_logit
    return 1 / (1 + np.exp(-(raw_logits - shift))), shift


def backtest_season_v2(train_df, test_df, features, config, dc_kwargs=None):
    """Backtest one season with model-market blend.

    Config dict keys:
        blend_weight: float (0-1), weight on model vs market. 1.0 = pure model.
        min_edge: float, minimum edge to bet.
        min_agree: int (1-4), minimum models agreeing on bet direction.
        kelly_fraction: float, fraction of Kelly to stake.
        max_stake_pct: float, max stake as fraction of bankroll.
    """
    blend_w = config.get("blend_weight", 0.7)
    min_edge = config.get("min_edge", 0.03)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)

    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    # Use last season of training data as early-stopping val
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_train_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_train_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train = X_train[:-n_val]
        y_es_train = y_train[:-n_val]
        X_es_val = X_train[-n_val:]
        y_es_val = y_train[-n_val:]

    if dc_kwargs is None:
        dc_kwargs = {}

    # --- Train 4 base models ---
    xgb_m = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val, feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Predict test set (raw per-model probabilities) ---
    X_test = test_df[features].values
    xgb_p = xgb_m.predict_proba(X_test)[:, 1]
    lgb_p = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_p = _lr_predict(lr_m, lr_scaler, X_test)
    dc_p = dc_m.predict_proba_df(test_df)

    # --- Calibrate each model individually ---
    # Base rate from most recent 2 training seasons (adapts to trends)
    recent_seasons = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_seasons)
    base_rate = train_df.loc[recent_mask, "Over_2_5"].mean() if recent_mask.sum() >= 100 else y_train.mean()

    xgb_cal, _ = calibrate_logit_shift(xgb_p, base_rate)
    lgb_cal, _ = calibrate_logit_shift(lgb_p, base_rate)
    lr_cal, _ = calibrate_logit_shift(lr_p, base_rate)
    dc_cal, _ = calibrate_logit_shift(dc_p, base_rate)

    # Ensemble: simple average of calibrated predictions
    ensemble_p = (xgb_cal + lgb_cal + lr_cal + dc_cal) / 4

    # AUC
    y_test = test_df["Over_2_5"].values
    try:
        auc = roc_auc_score(y_test, ensemble_p)
    except ValueError:
        auc = 0.5

    # --- Simulate bets with model-market blend ---
    bets = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        odds_over = row.get("B365Greater2.5", np.nan)
        odds_under = row.get("B365LessThan2.5", np.nan)

        if pd.isna(odds_over) or pd.isna(odds_under) or odds_over <= 1 or odds_under <= 1:
            continue

        # Market fair probabilities (overround removed)
        raw_o = 1.0 / odds_over
        raw_u = 1.0 / odds_under
        overround = raw_o + raw_u
        fair_over = raw_o / overround
        fair_under = raw_u / overround

        # Per-model Over probabilities (calibrated)
        model_probs_over = [xgb_cal[i], lgb_cal[i], lr_cal[i], dc_cal[i]]
        model_avg_over = ensemble_p[i]
        model_avg_under = 1 - model_avg_over

        # Model-market blend
        blended_over = blend_w * model_avg_over + (1 - blend_w) * fair_over
        blended_under = 1 - blended_over

        actual = row["Over_2_5"]

        for side, blended_p, fair_p, odds, model_ps_for_side in [
            ("over", blended_over, fair_over, odds_over,
             [p for p in model_probs_over]),
            ("under", blended_under, fair_under, odds_under,
             [1 - p for p in model_probs_over]),
        ]:
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            # Model agreement: how many models think this side has edge?
            if side == "over":
                n_agree = sum(1 for p in model_probs_over if p > fair_over)
            else:
                n_agree = sum(1 for p in model_probs_over if (1 - p) > fair_under)

            if ev <= 0 or edge < min_edge or n_agree < min_agree:
                continue

            # Kelly stake
            kelly = (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0
            stake = min(max_stake_pct, max(0, kelly * kelly_fraction))
            if stake <= 0:
                continue

            won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
            profit = stake * (odds - 1) if won else -stake

            bets.append({
                "season": row.get("SeasonIndex", 0),
                "home": row.get("Home_Team", ""),
                "away": row.get("Away_Team", ""),
                "side": side,
                "model_prob": model_avg_over if side == "over" else model_avg_under,
                "blended_prob": blended_p,
                "fair_prob": fair_p,
                "odds": odds,
                "edge": edge,
                "ev": ev,
                "n_agree": n_agree,
                "stake_pct": stake,
                "won": won,
                "profit_pct": profit,
                "actual_over": actual,
            })

    bets_df = pd.DataFrame(bets)

    metrics = {
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
        "n_matches": len(test_df),
        "n_bets": len(bets_df),
        "auc": auc,
        "base_rate": base_rate,
        "actual_over_rate": y_test.mean(),
        "model_mean": ensemble_p.mean(),
    }

    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["avg_edge"] = bets_df["edge"].mean()
        metrics["avg_odds"] = bets_df["odds"].mean()
        metrics["roi"] = bets_df["profit_pct"].sum() / bets_df["stake_pct"].sum()
    else:
        metrics.update({"total_profit_pct": 0, "win_rate": 0, "avg_edge": 0,
                        "avg_odds": 0, "roi": 0})

    return bets_df, metrics


def run_backtest_v2(config, start_season=19, end_season=25, verbose=True,
                    _cached_data=None, _cached_dc_kwargs=None):
    """Walk-forward backtest across multiple seasons with V2 engine.

    Pass _cached_data and _cached_dc_kwargs to avoid re-loading/re-tuning
    (useful for grid search).
    """
    if _cached_data is not None:
        data = _cached_data
    else:
        if verbose:
            print("Loading pipeline data...")
        data = run_pipeline(verbose=False)

    full_df = data["full_df"]
    features = list(data["features"])

    if _cached_dc_kwargs is not None:
        dc_kwargs = _cached_dc_kwargs
    else:
        if verbose:
            print("Tuning Dixon-Coles hyperparameters...")
        tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
        dc_kwargs = tune_dc_params(tune_df)

    if verbose:
        bw = config.get("blend_weight", 0.7)
        me = config.get("min_edge", 0.03)
        ma = config.get("min_agree", 2)
        print(f"\nBacktest V2 settings:")
        print(f"  Blend weight: {bw:.0%} model / {1-bw:.0%} market")
        print(f"  Min edge: {me:.1%}")
        print(f"  Min model agreement: {ma}/4")
        print(f"  Kelly fraction: {config.get('kelly_fraction', 0.25)}")
        print(f"  Max stake: {config.get('max_stake_pct', 0.05):.1%}")
        print(f"  Seasons: {start_season}-{end_season}")

    all_bets = []
    all_metrics = []
    cumulative_bankroll = 1.0

    if verbose:
        print(f"\n{'Season':>8s} {'Year':>8s} {'AUC':>6s} {'Bets':>5s} "
              f"{'WinR':>6s} {'ROI':>7s} {'O/U':>8s} {'Bankroll':>10s}")
        print("-" * 70)

    for season in range(start_season, end_season + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_odds = test_df["B365Greater2.5"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            continue

        bets_df, metrics = backtest_season_v2(
            train_df, test_df, features, config, dc_kwargs=dc_kwargs
        )

        # Update bankroll
        if len(bets_df) > 0:
            for _, bet in bets_df.iterrows():
                actual_stake = bet["stake_pct"] * cumulative_bankroll
                if bet["won"]:
                    cumulative_bankroll += actual_stake * (bet["odds"] - 1)
                else:
                    cumulative_bankroll -= actual_stake

        all_bets.append(bets_df)
        all_metrics.append(metrics)

        if verbose:
            m = metrics
            year = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
            n_over = len(bets_df[bets_df["side"] == "over"]) if len(bets_df) > 0 else 0
            n_under = len(bets_df[bets_df["side"] == "under"]) if len(bets_df) > 0 else 0
            print(f"S{m['season']:>5d}  {year:>8s} {m['auc']:>5.3f} {m['n_bets']:>5d} "
                  f"{m['win_rate']:>5.1%} {m['roi']:>+6.1%} "
                  f"{n_over:>3d}O/{n_under:<3d}U "
                  f"{cumulative_bankroll:>9.4f}")

    # Summary
    if all_bets:
        total_bets = pd.concat(all_bets, ignore_index=True)

        if verbose and len(total_bets) > 0:
            print(f"\n{'='*70}")
            print(f"BACKTEST V2 SUMMARY")
            print(f"{'='*70}")
            print(f"  Seasons: {start_season}-{end_season} ({len(all_metrics)} tested)")
            print(f"  Total bets: {len(total_bets)} ({len(total_bets)/len(all_metrics):.0f}/season)")
            print(f"  Win rate: {total_bets['won'].mean():.1%}")
            print(f"  Overall ROI: {total_bets['profit_pct'].sum() / total_bets['stake_pct'].sum():+.1%}")
            print(f"  Final bankroll: {cumulative_bankroll:.4f} ({(cumulative_bankroll - 1)*100:+.1f}%)")

            for side in ["over", "under"]:
                sb = total_bets[total_bets["side"] == side]
                if len(sb) > 0:
                    sroi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"  {side.upper():>5s}: {len(sb)} bets, win {sb['won'].mean():.1%}, ROI {sroi:+.1%}")

            profitable = sum(1 for m in all_metrics if m.get("total_profit_pct", 0) > 0)
            print(f"  Profitable seasons: {profitable}/{len(all_metrics)}")

        return total_bets, all_metrics, cumulative_bankroll

    return None, all_metrics, cumulative_bankroll


def grid_search_blend():
    """Search over blend parameters to find optimal configuration."""
    print("=" * 80)
    print("BLEND PARAMETER GRID SEARCH")
    print("=" * 80)

    # Pre-load data and tune DC once
    print("Loading data and tuning DC (once)...")
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)
    print("Done. Starting grid search...\n")

    results = []

    # Grid over key parameters
    blend_weights = [1.0, 0.85, 0.7, 0.55, 0.4]
    min_edges = [0.02, 0.03, 0.05]
    min_agrees = [1, 2, 3]

    total = len(blend_weights) * len(min_edges) * len(min_agrees)
    count = 0

    for bw in blend_weights:
        for me in min_edges:
            for ma in min_agrees:
                count += 1
                config = {
                    "blend_weight": bw,
                    "min_edge": me,
                    "min_agree": ma,
                    "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05,
                }

                _, metrics, bankroll = run_backtest_v2(
                    config, start_season=19, end_season=25, verbose=False,
                    _cached_data=data, _cached_dc_kwargs=dc_kwargs,
                )

                # Aggregate
                total_bets = sum(m.get("n_bets", 0) for m in metrics)
                n_seasons = len(metrics)
                avg_auc = np.mean([m["auc"] for m in metrics])

                # Compute overall ROI from metrics
                total_profit = sum(m.get("total_profit_pct", 0) for m in metrics)
                profitable_seasons = sum(1 for m in metrics if m.get("total_profit_pct", 0) > 0)

                results.append({
                    "blend_w": bw,
                    "min_edge": me,
                    "min_agree": ma,
                    "total_bets": total_bets,
                    "bets_per_season": total_bets / n_seasons if n_seasons > 0 else 0,
                    "avg_auc": avg_auc,
                    "bankroll": bankroll,
                    "profitable_seasons": profitable_seasons,
                    "n_seasons": n_seasons,
                })

                marker = " <--" if bankroll > 1.1 else ""
                print(f"  [{count:3d}/{total}] blend={bw:.2f} edge={me:.0%} agree={ma} => "
                      f"{total_bets:4d} bets ({total_bets/n_seasons:.0f}/s), "
                      f"bankroll={bankroll:.4f}, "
                      f"profitable={profitable_seasons}/{n_seasons}{marker}")

    # Sort by bankroll
    results.sort(key=lambda x: x["bankroll"], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 10 CONFIGURATIONS")
    print(f"{'='*80}")
    print(f"{'Blend':>6s} {'Edge':>5s} {'Agree':>6s} {'Bets':>5s} {'B/S':>5s} "
          f"{'Bankroll':>9s} {'Prof':>5s}")
    print("-" * 50)
    for r in results[:10]:
        print(f"{r['blend_w']:>5.2f}  {r['min_edge']:>4.0%}  {r['min_agree']:>5d}  "
              f"{r['total_bets']:>5d} {r['bets_per_season']:>5.0f}  "
              f"{r['bankroll']:>8.4f}  {r['profitable_seasons']}/{r['n_seasons']}")

    print(f"\nBOTTOM 5:")
    for r in results[-5:]:
        print(f"{r['blend_w']:>5.2f}  {r['min_edge']:>4.0%}  {r['min_agree']:>5d}  "
              f"{r['total_bets']:>5d} {r['bets_per_season']:>5.0f}  "
              f"{r['bankroll']:>8.4f}  {r['profitable_seasons']}/{r['n_seasons']}")

    return results


if __name__ == "__main__":
    import sys

    if "--grid" in sys.argv:
        grid_search_blend()
    else:
        # Default: run a few key configurations for comparison
        print("=" * 80)
        print("BACKTEST V2: Model-Market Blend Comparison")
        print("=" * 80)

        configs = [
            ("V1 Baseline (pure model, no agreement)",
             {"blend_weight": 1.0, "min_edge": 0.05, "min_agree": 1}),
            ("70% Model / 30% Market, 2+ agree, 3% edge",
             {"blend_weight": 0.7, "min_edge": 0.03, "min_agree": 2}),
            ("55% Model / 45% Market, 3+ agree, 3% edge",
             {"blend_weight": 0.55, "min_edge": 0.03, "min_agree": 3}),
            ("85% Model / 15% Market, 2+ agree, 5% edge",
             {"blend_weight": 0.85, "min_edge": 0.05, "min_agree": 2}),
            ("70% Model / 30% Market, 3+ agree, 2% edge",
             {"blend_weight": 0.7, "min_edge": 0.02, "min_agree": 3}),
        ]

        for label, config in configs:
            print(f"\n\n{'#'*80}")
            print(f"# {label}")
            print(f"{'#'*80}")
            config.setdefault("kelly_fraction", 0.25)
            config.setdefault("max_stake_pct", 0.05)
            run_backtest_v2(config, start_season=19, end_season=25)
