"""
Focused grid search for Band_13plus corner strategy.

Stress-tests the 13+ signal across parameter dimensions:
  - min_edge: how much model-market gap is needed
  - kelly_fraction: staking aggressiveness
  - ensemble_gate_agree: how many models must agree on "over"
  - min_odds / max_odds: odds range filtering
  - lambda_source: DC-only vs DC+feature blend
  - feature_lambda_weight: blend weight if using feature lambda
  - min_prob_gap: minimum probability gap

Caches pipeline + model training, replays betting logic per combo.
S19-S24 for optimisation, S25 held out.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pipeline import run_pipeline
from corners_data import load_corner_odds, merge_corner_odds
from corners_bands_data import load_corner_bands, merge_corner_bands
from config import CORNERS_ALL_FEATURES
from model import (train_xgb_corners, train_lgb_corners, train_logreg,
                    DixonColesPredictor, tune_dc_params_corners)
from backtest import (_calibrate, _lr_predict, refined_kelly, compute_drawdown_factor)
from corners_backtest import _calibrate_damped


def main() -> None:
    print("=== BAND 13+ FOCUSED GRID SEARCH (S19-S24, S25 held out) ===")
    print("Step 1: Load pipeline + tune DC...")

    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = [f for f in CORNERS_ALL_FEATURES if f in full_df.columns]

    corner_odds = load_corner_odds()
    full_df = merge_corner_odds(full_df, corner_odds)
    band_odds = load_corner_bands()
    full_df = merge_corner_bands(full_df, band_odds)

    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_corners(tune_df)
    print(f"  DC params: {dc_kwargs}")
    print(f"  Features: {len(features)}")

    # Step 2: Cache per-season model predictions + DC lambdas
    print("Step 2: Training models per season...")
    season_cache = {}

    for season in range(19, 25):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_bands = (test_df["Band_9orLess_Odds"].notna() &
                     test_df["Band_10to12_Odds"].notna() &
                     test_df["Band_13plus_Odds"].notna()).sum()
        if has_bands < 30 or len(train_df) < 500:
            print(f"  S{season}: skipped (bands={has_bands}, train={len(train_df)})")
            continue

        train_has_corners = (train_df["Home_Corners"].notna() &
                             train_df["Away_Corners"].notna())
        train_use = train_df[train_has_corners].copy()
        y_train = ((train_use["Home_Corners"] + train_use["Away_Corners"]) > 10.5).astype(int).values
        X_train = train_use[features].values

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

        xgb_m = train_xgb_corners(X_es_train, y_es_train, X_es_val, y_es_val)
        lgb_m = train_lgb_corners(X_es_train, y_es_train, X_es_val, y_es_val,
                                   feature_names=features)
        lr_m, lr_scaler = train_logreg(X_train, y_train)
        dc = DixonColesPredictor(**dc_kwargs)
        dc.fit(train_use)
        dc.fit_corners(train_use)

        X_test = test_df[features].values

        # Ensemble O/U predictions
        xgb_over = xgb_m.predict_proba(X_test)[:, 1]
        lgb_over = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        lr_over = _lr_predict(lr_m, lr_scaler, X_test)
        dc_over = dc.predict_proba_corners_df(test_df)

        # Calibrate
        recent_s = sorted(train_seasons)[-3:]
        recent_mask = train_use["SeasonIndex"].isin(recent_s)
        if recent_mask.sum() >= 100:
            recent_train = train_use.loc[recent_mask]
            base_rate = ((recent_train["Home_Corners"] +
                          recent_train["Away_Corners"]) > 10.5).mean()
        else:
            base_rate = y_train.mean()

        xgb_cal, _ = _calibrate_damped(xgb_over, base_rate, 0.5)
        lgb_cal, _ = _calibrate_damped(lgb_over, base_rate, 0.5)
        lr_cal, _ = _calibrate_damped(lr_over, base_rate, 0.5)
        dc_cal, _ = _calibrate_damped(dc_over, base_rate, 0.5)

        # Compute DC lambda per match
        dc_lambdas = []
        for _, row in test_df.iterrows():
            h_att = dc.corner_attack_home.get(row["Home_Team"])
            a_def = dc.corner_defence_away.get(row["Away_Team"])
            a_att = dc.corner_attack_away.get(row["Away_Team"])
            h_def = dc.corner_defence_home.get(row["Home_Team"])
            if any(v is None for v in [h_att, a_def, a_att, h_def]):
                dc_lambdas.append(np.nan)
                continue
            sqrt_gamma = np.sqrt(dc.corner_gamma)
            h_lam = np.clip(h_att * a_def * dc.corner_mu * sqrt_gamma, 1.0, 15.0)
            a_lam = np.clip(a_att * h_def * dc.corner_mu / sqrt_gamma, 1.0, 15.0)
            dc_lambdas.append(h_lam + a_lam)

        dc_lambdas = np.array(dc_lambdas)

        test_sorted = test_df.sort_values("Date").reset_index(drop=True)
        sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

        season_cache[season] = {
            "xgb_cal": xgb_cal, "lgb_cal": lgb_cal,
            "lr_cal": lr_cal, "dc_cal": dc_cal,
            "dc_lambdas": dc_lambdas,
            "test_df": test_df,
            "test_sorted": test_sorted,
            "sorted_positions": sorted_positions,
        }
        print(f"  S{season}: cached ({len(test_df)} matches, {has_bands} with band odds)")

    print(f"Cached {len(season_cache)} seasons")

    # Step 3: Fast replay — 13+ band only
    def replay_13plus(config: dict) -> tuple[pd.DataFrame, float]:
        min_edge = config["min_edge"]
        kelly_f = config["kelly_fraction"]
        max_stake = config["max_stake_pct"]
        gate_agree = config["ensemble_gate_agree"]
        min_odds = config["min_odds"]
        max_odds = config["max_odds"]
        use_feat_blend = config["lambda_source"] == "blend"
        feat_w = config.get("feature_lambda_weight", 0.3)

        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in sorted(season_cache.keys()):
            sc = season_cache[season]
            test_sorted = sc["test_sorted"]
            sorted_positions = sc["sorted_positions"]

            for match_num, (_, row) in enumerate(test_sorted.iterrows()):
                pred_idx = sorted_positions[match_num]

                odds_9 = row.get("Band_9orLess_Odds", np.nan)
                odds_10 = row.get("Band_10to12_Odds", np.nan)
                odds_13 = row.get("Band_13plus_Odds", np.nan)
                if pd.isna(odds_9) or pd.isna(odds_10) or pd.isna(odds_13):
                    continue
                if odds_9 <= 1 or odds_10 <= 1 or odds_13 <= 1:
                    continue
                if odds_13 < min_odds or odds_13 > max_odds:
                    continue

                # DC lambda
                total_lambda = sc["dc_lambdas"][pred_idx]
                if np.isnan(total_lambda):
                    continue

                # Optional feature blend
                if use_feat_blend:
                    h_avg = row.get("Home_CornersAvg_10", np.nan)
                    a_avg = row.get("Away_CornersAvg_10", np.nan)
                    if not pd.isna(h_avg) and not pd.isna(a_avg):
                        feat_lam = h_avg + a_avg
                        total_lambda = (1 - feat_w) * total_lambda + feat_w * feat_lam

                # Band probabilities
                p_13 = 1.0 - poisson.cdf(12, total_lambda)

                # Implied prob (fair, overround-removed)
                raw_9 = 1.0 / odds_9
                raw_10 = 1.0 / odds_10
                raw_13 = 1.0 / odds_13
                overround = raw_9 + raw_10 + raw_13
                implied_13 = raw_13 / overround

                edge = p_13 - implied_13
                ev = p_13 * odds_13 - 1
                if edge < min_edge or ev <= 0:
                    continue

                # Ensemble gate
                n_over = int(
                    (sc["xgb_cal"][pred_idx] > 0.5) +
                    (sc["lgb_cal"][pred_idx] > 0.5) +
                    (sc["lr_cal"][pred_idx] > 0.5) +
                    (sc["dc_cal"][pred_idx] > 0.5)
                )
                if n_over < gate_agree:
                    continue

                # Kelly staking
                dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)
                raw_kelly = (p_13 * odds_13 - 1) / (odds_13 - 1)
                stake = kelly_f * raw_kelly * dd_factor
                stake = min(stake, max_stake)
                if stake <= 0.001:
                    continue

                winner = row.get("Band_Winner", "")
                won = (winner == "Band_13plus")
                profit = stake * (odds_13 - 1) if won else -stake

                actual_stake = stake * cumulative_bankroll
                if won:
                    cumulative_bankroll += actual_stake * (odds_13 - 1)
                else:
                    cumulative_bankroll -= actual_stake
                peak_bankroll = max(peak_bankroll, cumulative_bankroll)

                all_bets.append({
                    "season": season, "odds": odds_13, "model_prob": p_13,
                    "implied_prob": implied_13, "edge": edge, "stake_pct": stake,
                    "won": won, "profit_pct": profit, "lambda": total_lambda,
                    "n_over": n_over,
                })

        return pd.DataFrame(all_bets), cumulative_bankroll

    # Step 4: Grid
    min_edges = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    kelly_fracs = [0.10, 0.15, 0.20, 0.25]
    gate_agrees = [0, 1, 2, 3]
    min_odds_list = [1.50, 2.00, 2.50]
    max_odds_list = [6.00, 8.00, 12.00]
    lambda_sources = ["dc", "blend"]
    feat_weights = [0.2, 0.3, 0.4]

    # Build config list (skip feat_weight combos when source=dc)
    configs = []
    for me in min_edges:
        for kf in kelly_fracs:
            for ga in gate_agrees:
                for mi in min_odds_list:
                    for mx in max_odds_list:
                        if mi >= mx:
                            continue
                        for ls in lambda_sources:
                            if ls == "dc":
                                configs.append({
                                    "min_edge": me, "kelly_fraction": kf,
                                    "max_stake_pct": 0.03, "ensemble_gate_agree": ga,
                                    "min_odds": mi, "max_odds": mx,
                                    "lambda_source": ls, "feature_lambda_weight": 0.0,
                                })
                            else:
                                for fw in feat_weights:
                                    configs.append({
                                        "min_edge": me, "kelly_fraction": kf,
                                        "max_stake_pct": 0.03, "ensemble_gate_agree": ga,
                                        "min_odds": mi, "max_odds": mx,
                                        "lambda_source": ls, "feature_lambda_weight": fw,
                                    })

    print(f"\nStep 3: Grid search ({len(configs)} combos)...")

    results = []
    for cfg in configs:
        bets, bank = replay_13plus(cfg)
        if len(bets) == 0:
            continue

        staked = bets["stake_pct"].sum()
        profit = bets["profit_pct"].sum()
        roi = profit / staked if staked > 0 else 0
        season_rois = []
        for s in bets["season"].unique():
            sb = bets[bets["season"] == s]
            sr = sb["profit_pct"].sum() / sb["stake_pct"].sum() if sb["stake_pct"].sum() > 0 else 0
            season_rois.append(sr)

        results.append({
            "edge": cfg["min_edge"], "kelly": cfg["kelly_fraction"],
            "gate": cfg["ensemble_gate_agree"],
            "min_o": cfg["min_odds"], "max_o": cfg["max_odds"],
            "src": cfg["lambda_source"], "fw": cfg["feature_lambda_weight"],
            "bets": len(bets), "roi": roi, "bank": bank,
            "win_rate": bets["won"].mean(),
            "profitable_seasons": sum(1 for r in season_rois if r > 0),
            "total_seasons": len(season_rois),
            "worst_season": min(season_rois) if season_rois else -1,
            "best_season": max(season_rois) if season_rois else 0,
            "bets_per_season": len(bets) / len(season_rois) if season_rois else 0,
            "avg_edge": bets["edge"].mean(),
            "avg_odds": bets["odds"].mean(),
            "avg_lambda": bets["lambda"].mean(),
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No configs produced any bets!")
        return

    # Score: ROI (50%), consistency (30%), volume (20%)
    df["score"] = (
        df["roi"].clip(-0.5, 2.0) * 0.50
        + (df["profitable_seasons"] / df["total_seasons"]) * 0.30
        + (df["bets_per_season"] / df["bets_per_season"].max()).clip(0, 1) * 0.20
    )
    df = df.sort_values("score", ascending=False)

    print(f"\nTested {len(results)} combinations with bets")
    print(f"Configs with positive ROI: {(df['roi'] > 0).sum()}/{len(df)}")
    print()

    hdr = (f"{'#':>3s} {'edge':>5s} {'kelly':>5s} {'gate':>4s} "
           f"{'odds':>9s} {'src':>5s} {'fw':>4s} | "
           f"{'Bets':>4s} {'B/S':>4s} {'ROI':>7s} {'Bank':>6s} "
           f"{'Win%':>5s} {'Szns':>5s} {'Best':>7s} {'Worst':>7s}")
    print("TOP 20 CONFIGS:")
    print(hdr)
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        print(f"{i+1:>3d} {row['edge']:>5.2f} {row['kelly']:>5.2f} {row['gate']:>4.0f} "
              f"{row['min_o']:>4.1f}-{row['max_o']:>3.0f} {row['src']:>5s} {row['fw']:>4.1f} | "
              f"{row['bets']:>4.0f} {row['bets_per_season']:>4.0f} "
              f"{row['roi']:>+6.1%} {row['bank']:>5.2f}x "
              f"{row['win_rate']:>5.1%} "
              f"{row['profitable_seasons']:>2.0f}/{row['total_seasons']:>1.0f} "
              f"{row['best_season']:>+6.1%} {row['worst_season']:>+6.1%}")

    # Bottom 5 (worst configs)
    print(f"\nBOTTOM 5:")
    for i, (_, row) in enumerate(df.tail(5).iterrows()):
        print(f"     {row['edge']:>5.2f} {row['kelly']:>5.2f} {row['gate']:>4.0f} "
              f"{row['min_o']:>4.1f}-{row['max_o']:>3.0f} {row['src']:>5s} {row['fw']:>4.1f} | "
              f"{row['bets']:>4.0f} {row['bets_per_season']:>4.0f} "
              f"{row['roi']:>+6.1%} {row['bank']:>5.2f}x "
              f"{row['win_rate']:>5.1%} "
              f"{row['profitable_seasons']:>2.0f}/{row['total_seasons']:>1.0f} "
              f"{row['best_season']:>+6.1%} {row['worst_season']:>+6.1%}")

    print()
    print("=== BEST BAND 13+ CONFIG ===")
    best = df.iloc[0]
    print(f"min_edge: {best['edge']:.2f}")
    print(f"kelly_fraction: {best['kelly']:.2f}")
    print(f"ensemble_gate_agree: {best['gate']:.0f}")
    print(f"odds_range: {best['min_o']:.1f} - {best['max_o']:.0f}")
    print(f"lambda_source: {best['src']}, feature_weight: {best['fw']:.1f}")
    print(f"ROI: {best['roi']:+.1%} | Bets: {best['bets']:.0f} "
          f"({best['bets_per_season']:.0f}/season) | Bank: {best['bank']:.2f}x")
    print(f"Win rate: {best['win_rate']:.1%} | "
          f"Seasons: {best['profitable_seasons']:.0f}/{best['total_seasons']:.0f} | "
          f"Best: {best['best_season']:+.1%} | Worst: {best['worst_season']:+.1%}")
    print(f"Avg edge: {best['avg_edge']:.3f} | Avg odds: {best['avg_odds']:.2f} | "
          f"Avg lambda: {best['avg_lambda']:.1f}")

    # Robustness check: what % of configs with 10+ bets are profitable?
    viable = df[df["bets"] >= 10]
    if len(viable) > 0:
        pct_profitable = (viable["roi"] > 0).mean()
        print(f"\nROBUSTNESS: {pct_profitable:.0%} of configs with 10+ bets are profitable "
              f"({(viable['roi'] > 0).sum()}/{len(viable)})")
        median_roi = viable["roi"].median()
        print(f"Median ROI across viable configs: {median_roi:+.1%}")


if __name__ == "__main__":
    main()
