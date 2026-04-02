"""
Fast BTTS grid search: load pipeline once, train models once per season,
replay betting logic 72 times with different configs.
S19-S24 for training, S25 held out for validation.
"""
import pandas as pd
import numpy as np
from pipeline import run_pipeline
from btts_data import load_btts_odds, merge_btts_odds
from config import BTTS_ALL_FEATURES
from model import (train_xgb_btts, train_lgb_btts, train_logreg,
                    DixonColesPredictor, tune_dc_params)
from backtest import (_calibrate, _calibrate_single, _lr_predict,
                       RegimeDetector, refined_kelly, compute_drawdown_factor)
from btts_backtest import BTTS_DEFAULT_CONFIG


def main():
    print("=== FAST BTTS GRID SEARCH (S19-S24, S25 held out) ===")
    print("Step 1: Load pipeline + tune DC (once)...")

    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = [f for f in BTTS_ALL_FEATURES if f in full_df.columns]
    btts_odds = load_btts_odds()
    full_df = merge_btts_odds(full_df, btts_odds)
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)

    print(f"  Features: {len(features)}")
    print(f"  DC params: {dc_kwargs}")

    # Step 2: Pre-compute per-season model predictions
    print("Step 2: Training models per season (S19-S24)...")
    season_cache = {}

    for season in range(19, 25):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_odds = test_df["BTTSY"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipped (odds={has_odds}, train={len(train_df)})")
            continue

        y_train = ((train_df["Home_Goals"] > 0) & (train_df["Away_Goals"] > 0)).astype(int).values
        y_test = ((test_df["Home_Goals"] > 0) & (test_df["Away_Goals"] > 0)).astype(int).values
        X_train = train_df[features].values

        # Early stopping split
        train_seasons = sorted(train_df["SeasonIndex"].unique())
        last_s = train_seasons[-1]
        es_val_mask = train_df["SeasonIndex"] == last_s
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

        # Train models
        xgb_m = train_xgb_btts(X_es_train, y_es_train, X_es_val, y_es_val)
        lgb_m = train_lgb_btts(X_es_train, y_es_train, X_es_val, y_es_val,
                                feature_names=features)
        lr_m, lr_scaler = train_logreg(X_train, y_train)
        dc_m = DixonColesPredictor(**dc_kwargs)
        dc_m.fit(train_df)

        X_test = test_df[features].values
        xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
        lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
        dc_raw = dc_m.predict_proba_btts_df(test_df)

        # Base rate
        recent_s = sorted(train_seasons)[-2:]
        recent_mask = train_df["SeasonIndex"].isin(recent_s)
        if recent_mask.sum() >= 100:
            recent_train = train_df.loc[recent_mask]
            base_rate = ((recent_train["Home_Goals"] > 0) &
                         (recent_train["Away_Goals"] > 0)).mean()
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

    # Step 3: Fast replay function
    def replay_btts(config):
        blend_w = config.get("blend_weight", 0.50)
        min_edge = config.get("min_edge", 0.015)
        min_agree = config.get("min_agree", 2)
        kelly_fraction = config.get("kelly_fraction", 0.25)
        max_stake_pct = config.get("max_stake_pct", 0.05)
        use_early = config.get("early_season", True)
        early_matches = config.get("early_season_matches", 80)
        early_blend_w = config.get("early_blend_weight", 0.30)
        early_min_edge = config.get("early_min_edge", 0.02)
        early_kelly = config.get("early_kelly_fraction", 0.15)

        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in sorted(season_cache.keys()):
            sc = season_cache[season]
            base_rate = sc["base_rate"]

            _, xgb_shift = _calibrate(sc["xgb_raw"], base_rate)
            _, lgb_shift = _calibrate(sc["lgb_raw"], base_rate)
            _, lr_shift = _calibrate(sc["lr_raw"], base_rate)
            _, dc_shift = _calibrate(sc["dc_raw"], base_rate)

            xgb_cal = np.array([_calibrate_single(p, xgb_shift) for p in sc["xgb_raw"]])
            lgb_cal = np.array([_calibrate_single(p, lgb_shift) for p in sc["lgb_raw"]])
            lr_cal = np.array([_calibrate_single(p, lr_shift) for p in sc["lr_raw"]])
            dc_cal = np.array([_calibrate_single(p, dc_shift) for p in sc["dc_raw"]])

            regime = RegimeDetector(
                prior_base_rate=base_rate,
                window=config.get("regime_window", 40),
                blend_speed=config.get("regime_blend_speed", 0.4),
                trigger_threshold=config.get("regime_trigger", 0.04),
                min_matches=config.get("regime_min_matches", 15),
            )

            test_sorted = sc["test_sorted"]
            sorted_positions = sc["sorted_positions"]

            for match_num, (_, row) in enumerate(test_sorted.iterrows()):
                pred_idx = sorted_positions[match_num]
                actual = sc["y_test"][pred_idx]
                regime.update(actual)

                if regime.regime_shift_detected():
                    adj_rate = regime.get_adjusted_base_rate()
                    _, xs = _calibrate(sc["xgb_raw"], adj_rate)
                    _, ls = _calibrate(sc["lgb_raw"], adj_rate)
                    _, lrs = _calibrate(sc["lr_raw"], adj_rate)
                    _, ds = _calibrate(sc["dc_raw"], adj_rate)
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
                model_weights = np.array([0.20, 0.20, 0.30, 0.30])
                model_yes = np.dot(per_model, model_weights)
                model_no = 1 - model_yes

                odds_yes = row.get("BTTSY", np.nan)
                odds_no = row.get("BTTSN", np.nan)
                if pd.isna(odds_yes) or pd.isna(odds_no) or odds_yes <= 1 or odds_no <= 1:
                    continue

                raw_y = 1.0 / odds_yes
                raw_n = 1.0 / odds_no
                overround = raw_y + raw_n
                fair_yes = raw_y / overround
                fair_no = raw_n / overround

                dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)

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

                    if side == "yes":
                        n_agree = int(np.sum(per_model > fair_yes))
                    else:
                        n_agree = int(np.sum((1 - per_model) > fair_no))

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

                    won = (side == "yes" and actual == 1) or (side == "no" and actual == 0)
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

    # Step 4: Grid search
    print("\nStep 3: Grid search (72 combos)...")
    blend_weights = [0.40, 0.50, 0.60]
    min_edges = [0.010, 0.015, 0.020, 0.025]
    kelly_fracs = [0.20, 0.25, 0.30]
    min_agrees = [1, 2]

    results = []
    for bw in blend_weights:
        for me in min_edges:
            for kf in kelly_fracs:
                for ma in min_agrees:
                    cfg = {**BTTS_DEFAULT_CONFIG,
                           "blend_weight": bw, "min_edge": me,
                           "kelly_fraction": kf, "min_agree": ma}
                    bets, bank = replay_btts(cfg)
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
                            "bets": len(bets), "roi": roi, "bank": bank,
                            "win_rate": bets["won"].mean(),
                            "profitable_seasons": sum(1 for r in season_rois if r > 0),
                            "total_seasons": len(season_rois),
                            "worst_season": min(season_rois),
                            "yes_pct": (bets["side"] == "yes").mean(),
                            "bets_per_season": len(bets) / len(season_rois),
                        })

    df = pd.DataFrame(results)
    # Score: ROI (40%), consistency (30%), balance (15%), volume (15%)
    df["score"] = (
        df["roi"].clip(-0.5, 1.0) * 0.4
        + (df["profitable_seasons"] / df["total_seasons"]) * 0.3
        + (1 - abs(df["yes_pct"] - 0.5)) * 0.15
        + (df["bets_per_season"] / df["bets_per_season"].max()).clip(0, 1) * 0.15
    )
    df = df.sort_values("score", ascending=False)

    print(f"\nTested {len(results)} combinations")
    print()
    hdr = (f"{'#':>3s} {'blend':>5s} {'edge':>5s} {'kelly':>5s} {'agree':>5s} | "
           f"{'Bets':>5s} {'B/Szn':>5s} {'ROI':>7s} {'Bank':>6s} "
           f"{'Win%':>5s} {'Szns':>5s} {'Yes%':>5s} {'Worst':>7s}")
    print("TOP 15 CONFIGS:")
    print(hdr)
    for i, (_, row) in enumerate(df.head(15).iterrows()):
        print(f"{i+1:>3d} {row['blend']:>5.2f} {row['edge']:>5.3f} "
              f"{row['kelly']:>5.2f} {row['agree']:>5.0f} | "
              f"{row['bets']:>5.0f} {row['bets_per_season']:>5.0f} "
              f"{row['roi']:>+6.1%} {row['bank']:>5.2f}x "
              f"{row['win_rate']:>5.1%} "
              f"{row['profitable_seasons']:>2.0f}/{row['total_seasons']:>1.0f} "
              f"{row['yes_pct']:>5.0%} {row['worst_season']:>+6.1%}")

    print()
    print("=== BEST CONFIG ===")
    best = df.iloc[0]
    print(f"blend_weight: {best['blend']:.2f}")
    print(f"min_edge: {best['edge']:.3f}")
    print(f"kelly_fraction: {best['kelly']:.2f}")
    print(f"min_agree: {best['agree']:.0f}")
    print(f"ROI: {best['roi']:+.1%} | Bets: {best['bets']:.0f} "
          f"({best['bets_per_season']:.0f}/season) | Bank: {best['bank']:.2f}x")
    print(f"Win rate: {best['win_rate']:.1%} | "
          f"Seasons: {best['profitable_seasons']:.0f}/{best['total_seasons']:.0f} | "
          f"Yes%: {best['yes_pct']:.0%} | Worst: {best['worst_season']:+.1%}")


if __name__ == "__main__":
    main()
