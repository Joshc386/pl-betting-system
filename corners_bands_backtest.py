"""
Corners Band Market Backtest: Walk-forward evaluation using Betfair CORNER_ODDS data.

Bands partition total corners into:
  - "9 or less" (0-9 corners)
  - "10 - 12" (10-12 corners)
  - "13 or more" (13+ corners)

Model probabilities from Poisson:
  P(0-9)   = poisson.cdf(9, lambda)
  P(10-12) = poisson.cdf(12, lambda) - poisson.cdf(9, lambda)
  P(13+)   = 1 - poisson.cdf(12, lambda)

Uses DC Poisson corner lambda as the primary signal, with ensemble O/U agreement
as a confidence gate. Bets where model band probability exceeds implied odds probability.

S19-S24 for evaluation, S25 held out.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from pipeline import run_pipeline
from corners_data import load_corner_odds, merge_corner_odds
from corners_bands_data import load_corner_bands, merge_corner_bands
from config import CORNERS_ALL_FEATURES
from model import (train_xgb_corners, train_lgb_corners, train_logreg, _fill_nan_median,
                    _clip_scaled, DixonColesPredictor, tune_dc_params_corners)
from backtest import (
    _calibrate, _calibrate_single, _lr_predict,
    refined_kelly, compute_drawdown_factor,
)
from corners_backtest import _calibrate_damped


BANDS_DEFAULT_CONFIG = {
    # Minimum edge (model prob - implied prob) to bet
    "min_edge": 0.03,
    # Kelly staking
    "kelly_fraction": 0.15,
    "max_stake_pct": 0.03,
    # Minimum model-market probability gap to consider a band
    "min_prob_gap": 0.02,
    # Ensemble O/U gating: require N models to agree on direction
    # before weighting the 13+ or 9orLess band more heavily
    "use_ensemble_gate": True,
    "ensemble_gate_agree": 3,
    # Poisson lambda source: "dc" (Dixon-Coles only) or "blend" (DC + feature-based)
    "lambda_source": "dc",
    # Blend weight for combining DC lambda with feature-based lambda
    "feature_lambda_weight": 0.3,
    # Minimum odds to bet (avoid very short prices)
    "min_odds": 1.50,
    # Maximum odds to bet (avoid extreme longshots)
    "max_odds": 8.00,
}


def _poisson_band_probs(lam: float) -> dict[str, float]:
    """Compute Poisson band probabilities for total corners.

    Args:
        lam: Expected total corners (Poisson lambda).

    Returns:
        Dict with P(0-9), P(10-12), P(13+).
    """
    p_9_or_less = poisson.cdf(9, lam)
    p_10_to_12 = poisson.cdf(12, lam) - poisson.cdf(9, lam)
    p_13_plus = 1.0 - poisson.cdf(12, lam)
    return {
        "Band_9orLess": p_9_or_less,
        "Band_10to12": p_10_to_12,
        "Band_13plus": p_13_plus,
    }


def _implied_probs_from_odds(odds_9: float, odds_10: float, odds_13: float
                             ) -> dict[str, float]:
    """Convert band odds to fair implied probabilities (removing overround).

    Args:
        odds_9: Decimal odds for "9 or less".
        odds_10: Decimal odds for "10 - 12".
        odds_13: Decimal odds for "13 or more".

    Returns:
        Dict with fair implied probabilities for each band.
    """
    raw_9 = 1.0 / odds_9
    raw_10 = 1.0 / odds_10
    raw_13 = 1.0 / odds_13
    overround = raw_9 + raw_10 + raw_13
    return {
        "Band_9orLess": raw_9 / overround,
        "Band_10to12": raw_10 / overround,
        "Band_13plus": raw_13 / overround,
    }


def bands_backtest(config: dict | None = None) -> pd.DataFrame:
    """Run walk-forward band market backtest across S19-S24.

    Args:
        config: Override default config parameters.

    Returns:
        DataFrame of all bets placed with columns:
            season, band, odds, model_prob, implied_prob, edge, stake_pct,
            won, profit_pct
    """
    cfg = {**BANDS_DEFAULT_CONFIG, **(config or {})}

    print("=== CORNERS BAND MARKET BACKTEST (S19-S24, S25 held out) ===")
    print("Step 1: Load pipeline + tune DC...")

    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = [f for f in CORNERS_ALL_FEATURES if f in full_df.columns]

    # Merge corner O/U odds (for ensemble gating)
    corner_odds = load_corner_odds()
    full_df = merge_corner_odds(full_df, corner_odds)

    # Merge band odds
    band_odds = load_corner_bands()
    full_df = merge_corner_bands(full_df, band_odds)

    # Tune DC for corners
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_corners(tune_df)
    print(f"  DC params: {dc_kwargs}")
    print(f"  Features: {len(features)}")

    all_bets: list[dict] = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    for season in range(19, 25):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        # Need band odds for betting
        has_bands = (test_df["Band_9orLess_Odds"].notna() &
                     test_df["Band_10to12_Odds"].notna() &
                     test_df["Band_13plus_Odds"].notna()).sum()
        if has_bands < 30 or len(train_df) < 500:
            print(f"  S{season}: skipped (bands={has_bands}, train={len(train_df)})")
            continue

        # Filter training to rows with corner data
        train_has_corners = (train_df["Home_Corners"].notna() &
                             train_df["Away_Corners"].notna())
        train_use = train_df[train_has_corners].copy()

        # Train DC model for Poisson lambda
        dc = DixonColesPredictor(**dc_kwargs)
        dc.fit(train_use)
        dc.fit_corners(train_use)

        # Train ensemble models (for O/U gating signal)
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

        X_test = test_df[features].values

        # Ensemble O/U predictions (for gating)
        xgb_over = xgb_m.predict_proba(X_test)[:, 1]
        lgb_over = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        lr_over = _lr_predict(lr_m, lr_scaler, X_test)
        dc_over = dc.predict_proba_corners_df(test_df)

        # Calibrate ensemble
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

        season_bets = 0
        test_sorted = test_df.sort_values("Date").reset_index(drop=True)
        sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

        for match_num, (_, row) in enumerate(test_sorted.iterrows()):
            pred_idx = sorted_positions[match_num]

            # Check all 3 band odds present
            odds_9 = row.get("Band_9orLess_Odds", np.nan)
            odds_10 = row.get("Band_10to12_Odds", np.nan)
            odds_13 = row.get("Band_13plus_Odds", np.nan)
            if pd.isna(odds_9) or pd.isna(odds_10) or pd.isna(odds_13):
                continue
            if odds_9 <= 1 or odds_10 <= 1 or odds_13 <= 1:
                continue

            # Get DC Poisson lambda for this match
            home_team = row.get("Home_Team", "")
            away_team = row.get("Away_Team", "")

            h_att = dc.corner_attack_home.get(home_team)
            a_def = dc.corner_defence_away.get(away_team)
            a_att = dc.corner_attack_away.get(away_team)
            h_def = dc.corner_defence_home.get(home_team)

            if any(v is None for v in [h_att, a_def, a_att, h_def]):
                continue

            sqrt_gamma = np.sqrt(dc.corner_gamma)
            home_lambda = np.clip(h_att * a_def * dc.corner_mu * sqrt_gamma, 1.0, 15.0)
            away_lambda = np.clip(a_att * h_def * dc.corner_mu / sqrt_gamma, 1.0, 15.0)
            total_lambda = home_lambda + away_lambda

            # Feature-based lambda (from pipeline Poisson features)
            if cfg["lambda_source"] == "blend":
                home_avg = row.get("Home_CornersAvg_10", np.nan)
                away_avg = row.get("Away_CornersAvg_10", np.nan)
                if not pd.isna(home_avg) and not pd.isna(away_avg):
                    feat_lambda = home_avg + away_avg
                    w = cfg["feature_lambda_weight"]
                    total_lambda = (1 - w) * total_lambda + w * feat_lambda

            # Band probabilities from Poisson
            model_probs = _poisson_band_probs(total_lambda)
            implied_probs = _implied_probs_from_odds(odds_9, odds_10, odds_13)

            # Ensemble gate: count how many models agree on Over/Under 10.5
            ensemble_preds = np.array([
                xgb_cal[pred_idx], lgb_cal[pred_idx],
                lr_cal[pred_idx], dc_cal[pred_idx],
            ])
            n_over = int(np.sum(ensemble_preds > 0.5))
            n_under = 4 - n_over

            dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)

            # Evaluate each band
            band_odds_map = {
                "Band_9orLess": odds_9,
                "Band_10to12": odds_10,
                "Band_13plus": odds_13,
            }

            for band, odds in band_odds_map.items():
                if odds < cfg["min_odds"] or odds > cfg["max_odds"]:
                    continue

                model_p = model_probs[band]
                implied_p = implied_probs[band]
                edge = model_p - implied_p

                if edge < cfg["min_edge"]:
                    continue

                ev = model_p * odds - 1
                if ev <= 0:
                    continue

                # Ensemble gating: if models strongly agree on direction,
                # boost confidence in aligned bands
                if cfg["use_ensemble_gate"]:
                    gate_agree = cfg["ensemble_gate_agree"]
                    # 13+ aligns with "over" consensus
                    if band == "Band_13plus" and n_over < gate_agree:
                        continue
                    # 9 or less aligns with "under" consensus
                    if band == "Band_9orLess" and n_under < gate_agree:
                        continue
                    # 10-12 is always allowed (middle band)

                # Kelly staking
                kelly_f = cfg["kelly_fraction"]
                raw_kelly = (model_p * odds - 1) / (odds - 1)
                stake = kelly_f * raw_kelly * dd_factor
                stake = min(stake, cfg["max_stake_pct"])
                if stake <= 0.001:
                    continue

                # Settlement
                winner = row.get("Band_Winner", "")
                won = (winner == band)
                profit = stake * (odds - 1) if won else -stake

                actual_stake = stake * cumulative_bankroll
                if won:
                    cumulative_bankroll += actual_stake * (odds - 1)
                else:
                    cumulative_bankroll -= actual_stake
                peak_bankroll = max(peak_bankroll, cumulative_bankroll)

                all_bets.append({
                    "season": season,
                    "band": band,
                    "odds": odds,
                    "model_prob": model_p,
                    "implied_prob": implied_p,
                    "edge": edge,
                    "stake_pct": stake,
                    "won": won,
                    "profit_pct": profit,
                    "lambda": total_lambda,
                })
                season_bets += 1

        print(f"  S{season}: {season_bets} bets | Bankroll: {cumulative_bankroll:.3f}x")

    bets_df = pd.DataFrame(all_bets)

    # Summary
    print(f"\n{'='*60}")
    if len(bets_df) == 0:
        print("No bets placed!")
        return bets_df

    total_staked = bets_df["stake_pct"].sum()
    total_profit = bets_df["profit_pct"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0
    win_rate = bets_df["won"].mean()

    print(f"CORNERS BAND MARKET RESULTS (S19-S24)")
    print(f"  Total bets: {len(bets_df)}")
    print(f"  ROI: {roi:+.1%}")
    print(f"  Final bankroll: {cumulative_bankroll:.3f}x")
    print(f"  Win rate: {win_rate:.1%}")

    # Per-band breakdown
    print(f"\nPer-band breakdown:")
    for band in ["Band_9orLess", "Band_10to12", "Band_13plus"]:
        bb = bets_df[bets_df["band"] == band]
        if len(bb) > 0:
            b_roi = bb["profit_pct"].sum() / bb["stake_pct"].sum() if bb["stake_pct"].sum() > 0 else 0
            print(f"  {band}: {len(bb)} bets, ROI {b_roi:+.1%}, "
                  f"Win {bb['won'].mean():.1%}, Avg odds {bb['odds'].mean():.2f}")

    # Per-season breakdown
    print(f"\nPer-season breakdown:")
    for s in sorted(bets_df["season"].unique()):
        sb = bets_df[bets_df["season"] == s]
        s_roi = sb["profit_pct"].sum() / sb["stake_pct"].sum() if sb["stake_pct"].sum() > 0 else 0
        print(f"  S{s}: {len(sb)} bets, ROI {s_roi:+.1%}, Win {sb['won'].mean():.1%}")

    # Average edge and lambda
    print(f"\nDiagnostics:")
    print(f"  Avg edge: {bets_df['edge'].mean():.3f}")
    print(f"  Avg lambda: {bets_df['lambda'].mean():.1f}")
    print(f"  Avg odds: {bets_df['odds'].mean():.2f}")

    return bets_df


if __name__ == "__main__":
    bets = bands_backtest()
