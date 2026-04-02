"""
Fast Corners O/U 10.5 grid search: load pipeline once, train models once per season,
replay betting logic across parameter combos.
S19-S24 for optimisation, S25 held out for validation.
"""
import pandas as pd
import numpy as np
from pipeline import run_pipeline
from corners_data import load_corner_odds, merge_corner_odds
from config import CORNERS_ALL_FEATURES
from model import (train_xgb_corners, train_lgb_corners, train_logreg,
                    DixonColesPredictor, tune_dc_params_corners)
from backtest import (_calibrate, _calibrate_single, _lr_predict,
                       RegimeDetector, refined_kelly, compute_drawdown_factor)
from corners_backtest import CORNERS_DEFAULT_CONFIG, _calibrate_damped


def main():
    print("=== FAST CORNERS O/U 10.5 GRID SEARCH (S19-S24, S25 held out) ===")
    print("Step 1: Load pipeline + tune DC (once)...")

    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = [f for f in CORNERS_ALL_FEATURES if f in full_df.columns]
    corner_odds = load_corner_odds()
    full_df = merge_corner_odds(full_df, corner_odds)
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_corners(tune_df)

    print(f"  Features: {len(features)}")
    print(f"  DC params: {dc_kwargs}")

    # Step 2: Pre-compute per-season model predictions
    print("Step 2: Training models per season (S19-S24)...")
    season_cache = {}

    for season in range(19, 25):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_odds = test_df["Corner_Over_Odds"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipped (odds={has_odds}, train={len(train_df)})")
            continue

        # Filter training to rows with corner data
        train_has_corners = train_df["Home_Corners"].notna() & train_df["Away_Corners"].notna()
        train_use = train_df[train_has_corners].copy()

        y_train = ((train_use["Home_Corners"] + train_use["Away_Corners"]) > 10.5).astype(int).values
        y_test = ((test_df["Home_Corners"] + test_df["Away_Corners"]) > 10.5).astype(int).values
        X_train = train_use[features].values

        # Early stopping split
        train_seasons = sorted(train_use["SeasonIndex"].unique())
        last_s = train_seasons[-1]
        es_val_mask = train_use["SeasonIndex"] == last_s
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

        # Train models
        xgb_m = train_xgb_corners(X_es_train, y_es_train, X_es_val, y_es_val)
        lgb_m = train_lgb_corners(X_es_train, y_es_train, X_es_val, y_es_val,
                                   feature_names=features)
        lr_m, lr_scaler = train_logreg(X_train, y_train)
        dc_m = DixonColesPredictor(**dc_kwargs)
        dc_m.fit(train_use)
        dc_m.fit_corners(train_use)

        X_test = test_df[features].values
        xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
        lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
        dc_raw = dc_m.predict_proba_corners_df(test_df)

        # Base rate (last 3 seasons for stability at ~50% base rate)
        recent_s = sorted(train_seasons)[-3:]
        recent_mask = train_use["SeasonIndex"].isin(recent_s)
        if recent_mask.sum() >= 100:
            recent_train = train_use.loc[recent_mask]
            base_rate = ((recent_train["Home_Corners"] + recent_train["Away_Corners"]) > 10.5).mean()
        else:
            base_rate = y_train.mean()

        # Sort test_df by date
        test_sorted = test_df.sort_values("Date").reset_index(drop=True)
        sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

        season_cache[season] = {
            "xgb_raw": xgb_raw, "lgb_raw": lgb_raw, "lr_raw": lr_raw, "dc_raw": dc_raw,
            "y_test": y_test, "base_rate": base_rate,
            "test_sorted": test_sorted, "sorted_positions": sorted_positions,
        }
        print(f"  S{season}: cached ({len(test_df)} matches, {has_odds} with odds)")

    print(f"Cached {len(season_cache)} seasons")

    # Step 3: Fast replay function (with dampened calibration + configurable weights)
    def replay_corners(config: dict):
        blend_w = config.get("blend_weight", 0.40)
        min_edge = config.get("min_edge", 0.02)
        min_agree = config.get("min_agree", 2)
        kelly_fraction = config.get("kelly_fraction", 0.20)
        max_stake_pct = config.get("max_stake_pct", 0.04)
        use_early = config.get("early_season", True)
        early_matches = config.get("early_season_matches", 80)
        early_blend_w = config.get("early_blend_weight", 0.25)
        early_min_edge = config.get("early_min_edge", 0.025)
        early_kelly = config.get("early_kelly_fraction", 0.12)
        cal_strength = config.get("calibration_strength", 0.5)
        mw = config.get("model_weights", [0.30, 0.30, 0.25, 0.15])
        use_regime = config.get("regime_detection", False)

        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in sorted(season_cache.keys()):
            sc = season_cache[season]
            base_rate = sc["base_rate"]

            # Dampened calibration for ~50% base rate stability
            xgb_cal, xgb_shift = _calibrate_damped(sc["xgb_raw"], base_rate, cal_strength)
            lgb_cal, lgb_shift = _calibrate_damped(sc["lgb_raw"], base_rate, cal_strength)
            lr_cal, lr_shift = _calibrate_damped(sc["lr_raw"], base_rate, cal_strength)
            dc_cal, dc_shift = _calibrate_damped(sc["dc_raw"], base_rate, cal_strength)

            regime = RegimeDetector(
                prior_base_rate=base_rate,
                window=config.get("regime_window", 40),
                blend_speed=config.get("regime_blend_speed", 0.4),
                trigger_threshold=config.get("regime_trigger", 0.15),
                min_matches=config.get("regime_min_matches", 15),
            )

            test_sorted = sc["test_sorted"]
            sorted_positions = sc["sorted_positions"]

            for match_num, (_, row) in enumerate(test_sorted.iterrows()):
                pred_idx = sorted_positions[match_num]
                actual = sc["y_test"][pred_idx]
                regime.update(actual)

                if use_regime and regime.regime_shift_detected():
                    adj_rate = regime.get_adjusted_base_rate()
                    _, xs = _calibrate_damped(sc["xgb_raw"], adj_rate, cal_strength)
                    _, ls = _calibrate_damped(sc["lgb_raw"], adj_rate, cal_strength)
                    _, lrs = _calibrate_damped(sc["lr_raw"], adj_rate, cal_strength)
                    _, ds = _calibrate_damped(sc["dc_raw"], adj_rate, cal_strength)
                    xgb_p = _calibrate_single(sc["xgb_raw"][pred_idx], xs)
                    lgb_p = _calibrate_single(sc["lgb_raw"][pred_idx], ls)
                    lr_p = _calibrate_single(sc["lr_raw"][pred_idx], lrs)
                    dc_p = _calibrate_single(sc["dc_raw"][pred_idx], ds)
                else:
                    xgb_p = xgb_cal[pred_idx]
                    lgb_p = lgb_cal[pred_idx]
                    lr_p = lr_cal[pred_idx]
                    dc_p = dc_cal[pred_idx]

                per_model = np.array([xgb_p, lgb_p, lr_p, dc_p])
                model_over = np.dot(per_model, mw)
                model_under = 1 - model_over

                odds_over = row.get("Corner_Over_Odds", np.nan)
                odds_under = row.get("Corner_Under_Odds", np.nan)
                if pd.isna(odds_over) or pd.isna(odds_under) or odds_over <= 1 or odds_under <= 1:
                    continue

                raw_o = 1.0 / odds_over
                raw_u = 1.0 / odds_under
                overround = raw_o + raw_u
                fair_over = raw_o / overround
                fair_under = raw_u / overround

                dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)

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

                    if side == "over":
                        n_agree = int(np.sum(per_model > fair_over))
                    else:
                        n_agree = int(np.sum((1 - per_model) > fair_under))

                    if ev <= 0 or edge < active_min_edge or n_agree < min_agree:
                        continue

                    stake = refined_kelly(
                        blended_p, odds, n_agree, edge,
                        kelly_fraction=active_kelly,
                        max_stake_pct=max_stake_pct,
                        drawdown_factor=dd_factor,
                    )
                    if stake <= 0:
                        continue

                    won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
                    profit = stake * (odds - 1) if won else -stake
                    actual_stake = stake * cumulative_bankroll
                    if won:
                        cumulative_bankroll += actual_stake * (odds - 1)
                    else:
                        cumulative_bankroll -= actual_stake
                    peak_bankroll = max(peak_bankroll, cumulative_bankroll)

                    all_bets.append({
                        "season": season, "side": side, "odds": odds, "edge": edge,
                        "stake_pct": stake, "won": won, "profit_pct": profit,
                        "phase": "early" if is_early else "regime",
                    })

        return pd.DataFrame(all_bets), cumulative_bankroll

    # Step 4: Expanded grid search
    blend_weights = [0.30, 0.40, 0.50]
    min_edges = [0.015, 0.020, 0.025, 0.030]
    kelly_fracs = [0.15, 0.20, 0.25]
    min_agrees = [1, 2]
    weight_configs = [
        [0.25, 0.25, 0.25, 0.25],
        [0.30, 0.30, 0.25, 0.15],
        [0.35, 0.35, 0.20, 0.10],
    ]
    cal_strengths = [0.3, 0.5, 0.7, 1.0]
    regime_opts = [False, True]

    total = (len(blend_weights) * len(min_edges) * len(kelly_fracs) * len(min_agrees)
             * len(weight_configs) * len(cal_strengths) * len(regime_opts))
    print(f"\nStep 3: Grid search ({total} combos)...")

    results = []
    for bw in blend_weights:
        for me in min_edges:
            for kf in kelly_fracs:
                for ma in min_agrees:
                    for wc in weight_configs:
                        for cs in cal_strengths:
                            for rd in regime_opts:
                                cfg = {**CORNERS_DEFAULT_CONFIG,
                                       "blend_weight": bw, "min_edge": me,
                                       "kelly_fraction": kf, "min_agree": ma,
                                       "model_weights": wc, "calibration_strength": cs,
                                       "regime_detection": rd}
                                bets, bank = replay_corners(cfg)
                                if len(bets) > 0:
                                    staked = bets["stake_pct"].sum()
                                    profit = bets["profit_pct"].sum()
                                    roi = profit / staked if staked > 0 else 0
                                    season_rois = []
                                    for s in bets["season"].unique():
                                        sb = bets[bets["season"] == s]
                                        sr = (sb["profit_pct"].sum() / sb["stake_pct"].sum()
                                              if sb["stake_pct"].sum() > 0 else 0)
                                        season_rois.append(sr)
                                    results.append({
                                        "blend": bw, "edge": me, "kelly": kf, "agree": ma,
                                        "weights": str(wc), "cal_str": cs, "regime": rd,
                                        "bets": len(bets), "roi": roi, "bank": bank,
                                        "win_rate": bets["won"].mean(),
                                        "profitable_seasons": sum(1 for r in season_rois if r > 0),
                                        "total_seasons": len(season_rois),
                                        "worst_season": min(season_rois) if season_rois else -1,
                                        "over_pct": (bets["side"] == "over").mean(),
                                        "bets_per_season": len(bets) / len(season_rois) if season_rois else 0,
                                    })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No profitable configs found!")
        return

    # Score: ROI (40%), consistency (30%), balance (15%), volume (15%)
    df["score"] = (
        df["roi"].clip(-0.5, 1.0) * 0.4
        + (df["profitable_seasons"] / df["total_seasons"]) * 0.3
        + (1 - abs(df["over_pct"] - 0.5)) * 0.15
        + (df["bets_per_season"] / df["bets_per_season"].max()).clip(0, 1) * 0.15
    )
    df = df.sort_values("score", ascending=False)

    print(f"\nTested {len(results)} combinations")
    print()
    hdr = (f"{'#':>3s} {'blend':>5s} {'edge':>5s} {'kelly':>5s} {'agree':>5s} "
           f"{'cal':>4s} {'rgm':>3s} {'wts':>16s} | "
           f"{'Bets':>5s} {'B/Szn':>5s} {'ROI':>7s} {'Bank':>6s} "
           f"{'Win%':>5s} {'Szns':>5s} {'O%':>5s} {'Worst':>7s}")
    print("TOP 15 CONFIGS:")
    print(hdr)
    for i, (_, row) in enumerate(df.head(15).iterrows()):
        print(f"{i+1:>3d} {row['blend']:>5.2f} {row['edge']:>5.3f} "
              f"{row['kelly']:>5.2f} {row['agree']:>5.0f} "
              f"{row['cal_str']:>4.1f} {'Y' if row['regime'] else 'N':>3s} "
              f"{row['weights']:>16s} | "
              f"{row['bets']:>5.0f} {row['bets_per_season']:>5.0f} "
              f"{row['roi']:>+6.1%} {row['bank']:>5.2f}x "
              f"{row['win_rate']:>5.1%} "
              f"{row['profitable_seasons']:>2.0f}/{row['total_seasons']:>1.0f} "
              f"{row['over_pct']:>5.0%} {row['worst_season']:>+6.1%}")

    print()
    print("=== BEST CORNERS CONFIG ===")
    best = df.iloc[0]
    print(f"blend_weight: {best['blend']:.2f}")
    print(f"min_edge: {best['edge']:.3f}")
    print(f"kelly_fraction: {best['kelly']:.2f}")
    print(f"min_agree: {best['agree']:.0f}")
    print(f"model_weights: {best['weights']}")
    print(f"calibration_strength: {best['cal_str']:.1f}")
    print(f"regime_detection: {best['regime']}")
    print(f"ROI: {best['roi']:+.1%} | Bets: {best['bets']:.0f} "
          f"({best['bets_per_season']:.0f}/season) | Bank: {best['bank']:.2f}x")
    print(f"Win rate: {best['win_rate']:.1%} | "
          f"Seasons: {best['profitable_seasons']:.0f}/{best['total_seasons']:.0f} | "
          f"Over%: {best['over_pct']:.0%} | Worst: {best['worst_season']:+.1%}")

    # Go/no-go gate
    print()
    if best['roi'] > -0.05:
        print("GO/NO-GO: PASS — best ROI > -5%, corners O/U worth continuing")
    else:
        print("GO/NO-GO: FAIL — best ROI still worse than -5%, "
              "corners O/U on Betfair exchange may be too efficient")


if __name__ == "__main__":
    main()
