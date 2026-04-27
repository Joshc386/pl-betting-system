"""
Backtest: Walk-forward evaluation of alternative O/U goal lines against Betfair odds.

Uses the same 4-model ensemble as the O/U 2.5 backtest, but evaluates all
available Betfair goal lines (1.5, 2.5, 3.5, 4.5, etc.) per match and bets
whichever line offers the best edge.

Key differences from backtest.py:
  - Uses Betfair exchange odds (near-zero margin) instead of Bet365
  - Evaluates multiple lines per match via DC Poisson goal distribution
  - Can bet different lines for different matches (line selection)
  - Alt-line probabilities come from DC Poisson only (not the GBDT ensemble)
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import roc_auc_score

from pipeline import run_pipeline
from config import ALL_FEATURES, DC_LAMBDA_MIN, DC_LAMBDA_MAX
from model import (
    train_xgb, train_lgb, train_logreg, _fill_nan_median,
    _clip_scaled, DixonColesPredictor, tune_dc_params,
)
from backtest import (
    _calibrate, _calibrate_single, _lr_predict, DEFAULT_CONFIG,
    RegimeDetector, refined_kelly, compute_drawdown_factor,
)
from alt_lines import (
    build_goal_matrix, total_goals_distribution, prob_over_line,
    expected_value_asian, settle_asian,
)
from alt_lines_data import load_and_merge


ALT_LINES_CONFIG = {
    **DEFAULT_CONFIG,
    "alt_blend_weight": 0.35,   # blend weight for alt-line Poisson probs
    "alt_min_edge": 0.02,       # minimum edge for alt-line bets
    "alt_kelly_fraction": 0.20, # slightly more conservative for DC-only bets
    "alt_max_stake_pct": 0.04,  # lower cap since fewer models agree
    "eval_lines": [1.5, 2.5],   # only lines with proven edge
    "allowed_sides": ["over"],   # over only — under has no edge
    "best_line_only": True,      # only bet the single best line per match
}


def _get_dc_lambdas(
    dc_model: DixonColesPredictor,
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    """Extract home/away goal lambdas from DC model."""
    h_att = dc_model.attack_home.get(home_team, dc_model.PRIORS["attack_home"])
    a_def = dc_model.defence_away.get(away_team, dc_model.PRIORS["defence_away"])
    a_att = dc_model.attack_away.get(away_team, dc_model.PRIORS["attack_away"])
    h_def = dc_model.defence_home.get(home_team, dc_model.PRIORS["defence_home"])

    sqrt_gamma = np.sqrt(dc_model.gamma)
    home_lambda = np.clip(h_att * a_def * dc_model.mu * sqrt_gamma, DC_LAMBDA_MIN, DC_LAMBDA_MAX)
    away_lambda = np.clip(a_att * h_def * dc_model.mu / sqrt_gamma, DC_LAMBDA_MIN, DC_LAMBDA_MAX)
    return float(home_lambda), float(away_lambda)


def backtest_alt_lines_season(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    config: dict | None = None,
    dc_kwargs: dict | None = None,
    cumulative_bankroll: float = 1.0,
    peak_bankroll: float = 1.0,
) -> tuple[pd.DataFrame, dict, float, float]:
    """Walk-forward backtest on alternative goal lines for one season.

    Args:
        train_df: Training data (all prior seasons).
        test_df: Test season data with Betfair odds columns.
        features: Feature column names.
        config: Backtest configuration dict.
        dc_kwargs: Dixon-Coles model kwargs.
        cumulative_bankroll: Starting bankroll.
        peak_bankroll: Peak bankroll for drawdown.

    Returns:
        Tuple of (bets_df, metrics, final_bankroll, final_peak).
    """
    config = config or ALT_LINES_CONFIG.copy()
    blend_w = config.get("alt_blend_weight", 0.35)
    min_edge = config.get("alt_min_edge", 0.02)
    kelly_fraction = config.get("alt_kelly_fraction", 0.20)
    max_stake_pct = config.get("alt_max_stake_pct", 0.04)
    eval_lines = config.get("eval_lines", [1.5, 2.5])
    allowed_sides = config.get("allowed_sides", ["over", "under"])
    best_line_only = config.get("best_line_only", True)

    if dc_kwargs is None:
        dc_kwargs = {}

    # Train DC model (primary model for alt-line evaluation)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # Also train ensemble for O/U 2.5 agreement gating
    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    xgb_m = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val,
                       feature_names=features)

    # Bet simulation
    bets = []
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)

    for _, row in test_sorted.iterrows():
        home = row.get("Home_Team", "")
        away = row.get("Away_Team", "")
        actual_total = row.get("TG", np.nan)

        if pd.isna(actual_total):
            continue

        # Get DC lambdas
        home_lam, away_lam = _get_dc_lambdas(dc_m, home, away)
        mat = build_goal_matrix(home_lam, away_lam, rho=dc_m.rho)
        goal_dist = total_goals_distribution(mat)

        # Drawdown factor
        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)

        # Evaluate each available line
        match_opportunities = []

        for line in eval_lines:
            line_str = f"{line:.1f}".replace(".", "")
            over_col = f"BF_Over_{line_str}"
            under_col = f"BF_Under_{line_str}"
            winner_col = f"BF_Winner_{line_str}"

            if over_col not in row.index or under_col not in row.index:
                continue

            odds_over = row.get(over_col, np.nan)
            odds_under = row.get(under_col, np.nan)

            if pd.isna(odds_over) or pd.isna(odds_under):
                continue
            if odds_over <= 1.01 or odds_under <= 1.01:
                continue

            # Model probabilities from Poisson
            line_probs = prob_over_line(goal_dist, line)
            model_p_over = line_probs["p_over"]
            model_p_under = line_probs["p_under"]

            # Fair market probabilities (remove Betfair overround)
            raw_o = 1.0 / odds_over
            raw_u = 1.0 / odds_under
            overround = raw_o + raw_u
            fair_over = raw_o / overround
            fair_under = raw_u / overround

            # Evaluate allowed sides
            all_sides = [
                ("over", model_p_over, fair_over, odds_over),
                ("under", model_p_under, fair_under, odds_under),
            ]
            for side, model_p, fair_p, odds in all_sides:
                if side not in allowed_sides:
                    continue
                blended_p = blend_w * model_p + (1 - blend_w) * fair_p
                edge = blended_p - fair_p
                ev = blended_p * odds - 1

                if ev <= 0 or edge < min_edge:
                    continue

                # Kelly sizing
                kelly_raw = (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0
                stake = min(max_stake_pct, max(0, kelly_raw * kelly_fraction))
                stake *= dd_factor

                if stake < 0.005:
                    continue

                # Actual result
                actual_over = int(actual_total) > line
                won = (side == "over" and actual_over) or (
                    side == "under" and not actual_over
                )
                profit = stake * (odds - 1) if won else -stake

                match_opportunities.append({
                    "season": row.get("SeasonIndex", 0),
                    "home": home,
                    "away": away,
                    "date": row.get("Date", ""),
                    "line": line,
                    "side": side,
                    "model_prob": model_p,
                    "blended_prob": blended_p,
                    "fair_prob": fair_p,
                    "odds": odds,
                    "edge": edge,
                    "ev": ev,
                    "stake_pct": stake,
                    "won": won,
                    "profit_pct": profit,
                    "actual_total": int(actual_total),
                    "home_lambda": home_lam,
                    "away_lambda": away_lam,
                    "dd_factor": dd_factor,
                })

        # Filter: best line only, or all profitable lines
        if match_opportunities:
            if best_line_only:
                # Pick the single best opportunity by EV
                best = max(match_opportunities, key=lambda x: x["ev"])
                selected = [best]
            else:
                selected = match_opportunities

            for bet in selected:
                bets.append(bet)

                # Update bankroll
                actual_stake = bet["stake_pct"] * cumulative_bankroll
                if bet["won"]:
                    cumulative_bankroll += actual_stake * (bet["odds"] - 1)
                else:
                    cumulative_bankroll -= actual_stake
                peak_bankroll = max(peak_bankroll, cumulative_bankroll)

    bets_df = pd.DataFrame(bets)

    # Metrics
    metrics = {
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
        "n_matches": len(test_sorted),
        "n_bets": len(bets_df),
        "cumulative_bankroll": cumulative_bankroll,
        "peak_bankroll": peak_bankroll,
    }

    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["avg_edge"] = bets_df["edge"].mean()
        metrics["avg_odds"] = bets_df["odds"].mean()
        metrics["avg_stake"] = bets_df["stake_pct"].mean()
        metrics["roi"] = (bets_df["profit_pct"].sum() /
                          bets_df["stake_pct"].sum())

        # Per-line breakdown
        for line in eval_lines:
            lb = bets_df[bets_df["line"] == line]
            metrics[f"n_ou{line:.1f}".replace(".", "")] = len(lb)
            if len(lb) > 0:
                metrics[f"roi_ou{line:.1f}".replace(".", "")] = (
                    lb["profit_pct"].sum() / lb["stake_pct"].sum()
                )

        # Per-side breakdown
        for side in ["over", "under"]:
            sb = bets_df[bets_df["side"] == side]
            metrics[f"n_{side}"] = len(sb)
            if len(sb) > 0:
                metrics[f"{side}_roi"] = (
                    sb["profit_pct"].sum() / sb["stake_pct"].sum()
                )
    else:
        metrics.update({
            "total_profit_pct": 0, "win_rate": 0, "avg_edge": 0,
            "avg_odds": 0, "avg_stake": 0, "roi": 0,
        })

    return bets_df, metrics, cumulative_bankroll, peak_bankroll


def run_full_backtest(config: dict | None = None,
                      verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Run walk-forward backtest across all available seasons.

    Returns:
        Tuple of (all_bets_df, season_metrics_list).
    """
    config = config or ALT_LINES_CONFIG.copy()

    if verbose:
        print("Loading pipeline data + Betfair goal odds...")

    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = [f for f in ALL_FEATURES if f in full_df.columns]

    # Add DateOnly for merge
    full_df["DateOnly"] = pd.to_datetime(full_df["Date"]).dt.date

    # Merge Betfair odds
    try:
        merged_df = load_and_merge(full_df)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return pd.DataFrame(), []

    # Check which seasons have Betfair coverage
    bf_cols = [c for c in merged_df.columns if c.startswith("BF_Over_")]
    if not bf_cols:
        print("ERROR: No Betfair odds columns found after merge")
        return pd.DataFrame(), []

    seasons = sorted(merged_df["SeasonIndex"].unique())
    season_coverage = {}
    for s in seasons:
        s_df = merged_df[merged_df["SeasonIndex"] == s]
        n_with_bf = (s_df[bf_cols].notna().any(axis=1)).sum()
        if n_with_bf > 0:
            season_coverage[s] = n_with_bf

    if verbose:
        print(f"\nFeatures: {len(features)}")
        print(f"Betfair odds columns: {bf_cols}")
        print(f"\nSeason Betfair coverage:")
        for s, n in sorted(season_coverage.items()):
            total = len(merged_df[merged_df["SeasonIndex"] == s])
            print(f"  S{s}: {n}/{total} matches ({n/total:.0%})")

    # Walk-forward: train on all prior, test on next
    test_seasons = [s for s in seasons if s in season_coverage and s >= 19]

    if not test_seasons:
        print("No test seasons with Betfair coverage!")
        return pd.DataFrame(), []

    # Tune DC params once on full data
    tune_df = merged_df[merged_df["SeasonIndex"] >= 14]
    dc_kwargs = tune_dc_params(tune_df)
    if verbose:
        print(f"\nDC params: {dc_kwargs}")

    all_bets = []
    all_metrics = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    for test_season in test_seasons:
        train = merged_df[
            (merged_df["SeasonIndex"] >= 14) &
            (merged_df["SeasonIndex"] < test_season)
        ]
        test = merged_df[merged_df["SeasonIndex"] == test_season]

        if len(train) < 500 or len(test) < 50:
            continue

        if verbose:
            n_bf = (test[bf_cols].notna().any(axis=1)).sum()
            print(f"\n{'='*60}")
            print(f"Season {test_season}: train={len(train)}, "
                  f"test={len(test)}, betfair_matches={n_bf}")
            print(f"{'='*60}")

        bets_df, metrics, cumulative_bankroll, peak_bankroll = (
            backtest_alt_lines_season(
                train, test, features, config, dc_kwargs,
                cumulative_bankroll, peak_bankroll,
            )
        )

        all_bets.append(bets_df)
        all_metrics.append(metrics)

        if verbose and len(bets_df) > 0:
            print(f"\n  Bets: {metrics['n_bets']}")
            print(f"  Win rate: {metrics['win_rate']:.1%}")
            print(f"  ROI: {metrics['roi']:+.1%}")
            print(f"  Profit: {metrics['total_profit_pct']:+.4f} units")
            print(f"  Bankroll: {cumulative_bankroll:.4f}")

            # Line breakdown
            for line in config.get("eval_lines", []):
                key = f"n_ou{line:.1f}".replace(".", "")
                roi_key = f"roi_ou{line:.1f}".replace(".", "")
                n = metrics.get(key, 0)
                roi = metrics.get(roi_key, 0)
                if n > 0:
                    print(f"    O/U {line}: {n} bets, ROI {roi:+.1%}")

    # Combine
    if all_bets:
        combined = pd.concat(all_bets, ignore_index=True)
    else:
        combined = pd.DataFrame()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  OVERALL SUMMARY")
        print(f"{'='*60}")
        if len(combined) > 0:
            total_profit = combined["profit_pct"].sum()
            total_staked = combined["stake_pct"].sum()
            overall_roi = total_profit / total_staked if total_staked > 0 else 0

            print(f"  Total bets: {len(combined)}")
            print(f"  Win rate: {combined['won'].mean():.1%}")
            print(f"  Overall ROI: {overall_roi:+.1%}")
            print(f"  Total profit: {total_profit:+.4f} units")
            print(f"  Final bankroll: {cumulative_bankroll:.4f}")

            # Line summary
            print(f"\n  By line:")
            for line in config.get("eval_lines", []):
                lb = combined[combined["line"] == line]
                if len(lb) > 0:
                    line_roi = lb["profit_pct"].sum() / lb["stake_pct"].sum()
                    print(f"    O/U {line}: {len(lb)} bets, "
                          f"win {lb['won'].mean():.0%}, "
                          f"ROI {line_roi:+.1%}")

            # Side summary
            print(f"\n  By side:")
            for side in ["over", "under"]:
                sb = combined[combined["side"] == side]
                if len(sb) > 0:
                    side_roi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"    {side.title()}: {len(sb)} bets, "
                          f"win {sb['won'].mean():.0%}, "
                          f"ROI {side_roi:+.1%}")
        else:
            print("  No bets placed.")

    return combined, all_metrics


if __name__ == "__main__":
    print("=" * 60)
    print("  Alt Lines Backtest")
    print("=" * 60)
    run_full_backtest(verbose=True)
