"""
Edge analytics: validates model predictions against actual outcomes.

Works on two data sources:
  1. Backtest bets_df (walk-forward, historical Bet365 odds)
  2. Live settled recommendations (from dashboard DB)

Key analyses:
  - Hit rate by edge bucket (does higher edge = higher win rate?)
  - Calibration curve (predicted probability vs actual win rate)
  - ROI by edge bucket (money-weighted edge validation)
  - Per-model accuracy (Brier score per base model)
  - Confidence level validation (high/medium/low accuracy)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Bucket Analysis
# ═══════════════════════════════════════════════════════════════════════════════

EDGE_BUCKETS = [
    (0.00, 0.02, "0-2%"),
    (0.02, 0.04, "2-4%"),
    (0.04, 0.06, "4-6%"),
    (0.06, 0.08, "6-8%"),
    (0.08, 0.10, "8-10%"),
    (0.10, 1.00, "10%+"),
]


def edge_bucket_analysis(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute hit rate and ROI by edge bucket.

    Args:
        bets_df: DataFrame with columns: edge, won, profit_pct, stake_pct, odds.

    Returns:
        DataFrame with one row per edge bucket:
          bucket, n_bets, win_rate, roi, avg_edge, avg_odds, profit.
    """
    if bets_df.empty:
        return pd.DataFrame()

    rows = []
    for lo, hi, label in EDGE_BUCKETS:
        mask = (bets_df["edge"] >= lo) & (bets_df["edge"] < hi)
        bucket = bets_df[mask]
        if len(bucket) == 0:
            continue

        staked = bucket["stake_pct"].sum()
        profit = bucket["profit_pct"].sum()
        rows.append({
            "bucket": label,
            "n_bets": len(bucket),
            "win_rate": bucket["won"].mean(),
            "roi": profit / staked if staked > 0 else 0.0,
            "avg_edge": bucket["edge"].mean(),
            "avg_odds": bucket["odds"].mean(),
            "profit": profit,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Curve
# ═══════════════════════════════════════════════════════════════════════════════

CALIBRATION_BINS = [
    (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
    (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80),
    (0.80, 0.85), (0.85, 0.90), (0.90, 1.00),
]


def calibration_curve(
    bets_df: pd.DataFrame,
    prob_col: str = "model_prob",
) -> pd.DataFrame:
    """Compute calibration: predicted probability vs actual win rate.

    Groups bets by predicted probability bucket and compares the
    average prediction against the actual win rate in that bucket.

    Args:
        bets_df: DataFrame with prob_col and 'won' columns.
        prob_col: Column name for predicted probability.

    Returns:
        DataFrame with: bin_mid, predicted, actual, n_bets, gap.
    """
    if bets_df.empty or prob_col not in bets_df.columns:
        return pd.DataFrame()

    rows = []
    for lo, hi in CALIBRATION_BINS:
        mask = (bets_df[prob_col] >= lo) & (bets_df[prob_col] < hi)
        bucket = bets_df[mask]
        if len(bucket) < 3:  # Need minimum sample for meaningful stat
            continue

        predicted = bucket[prob_col].mean()
        actual = bucket["won"].mean()
        rows.append({
            "bin_label": f"{lo:.0%}-{hi:.0%}",
            "bin_mid": (lo + hi) / 2,
            "predicted": predicted,
            "actual": actual,
            "n_bets": len(bucket),
            "gap": actual - predicted,  # Positive = model underestimates
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Model Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute Brier score (lower = better, 0 = perfect).

    Args:
        probs: Predicted probabilities (0-1).
        outcomes: Binary outcomes (0 or 1).

    Returns:
        Mean squared error of probability predictions.
    """
    return float(np.mean((probs - outcomes) ** 2))


def per_model_accuracy(
    all_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Brier score and accuracy for each base model.

    Args:
        all_predictions: DataFrame with columns for each model's probability
            and the actual outcome. Expected columns:
            - xgb_prob, lgb_prob, lr_prob, dc_prob (whichever are available)
            - ensemble_prob
            - actual (binary outcome: 1 = over/yes, 0 = under/no)

    Returns:
        DataFrame with: model, brier_score, mean_prob, accuracy.
    """
    if all_predictions.empty or "actual" not in all_predictions.columns:
        return pd.DataFrame()

    outcomes = all_predictions["actual"].values
    rows = []

    model_cols = {
        "XGBoost": "xgb_prob",
        "LightGBM": "lgb_prob",
        "LogReg": "lr_prob",
        "Dixon-Coles": "dc_prob",
        "Ensemble": "ensemble_prob",
    }

    for name, col in model_cols.items():
        if col not in all_predictions.columns:
            continue
        probs = all_predictions[col].dropna().values
        if len(probs) == 0:
            continue
        valid = all_predictions[col].notna()
        p = all_predictions.loc[valid, col].values
        o = outcomes[valid.values]

        rows.append({
            "model": name,
            "brier_score": brier_score(p, o),
            "mean_prob": float(p.mean()),
            "accuracy": float(((p > 0.5) == o).mean()),
            "n_predictions": len(p),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence Level Validation
# ═══════════════════════════════════════════════════════════════════════════════

def confidence_validation(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Validate confidence levels against actual outcomes.

    Args:
        bets_df: DataFrame with 'confidence' and 'won' columns.

    Returns:
        DataFrame with: confidence, n_bets, win_rate, roi, avg_edge.
    """
    if bets_df.empty or "confidence" not in bets_df.columns:
        return pd.DataFrame()

    rows = []
    for level in ["high", "medium", "low"]:
        mask = bets_df["confidence"] == level
        bucket = bets_df[mask]
        if len(bucket) == 0:
            continue

        staked = bucket["stake_pct"].sum()
        profit = bucket["profit_pct"].sum()
        rows.append({
            "confidence": level,
            "n_bets": len(bucket),
            "win_rate": bucket["won"].mean(),
            "roi": profit / staked if staked > 0 else 0.0,
            "avg_edge": bucket["edge"].mean(),
            "avg_odds": bucket["odds"].mean(),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Side Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def side_analysis(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Break down performance by bet side (over/under or yes/no).

    Args:
        bets_df: DataFrame with 'side', 'won', 'profit_pct', 'stake_pct'.

    Returns:
        DataFrame with: side, n_bets, win_rate, roi, avg_edge, avg_odds.
    """
    if bets_df.empty or "side" not in bets_df.columns:
        return pd.DataFrame()

    rows = []
    for side in sorted(bets_df["side"].unique()):
        bucket = bets_df[bets_df["side"] == side]
        staked = bucket["stake_pct"].sum()
        profit = bucket["profit_pct"].sum()
        rows.append({
            "side": side,
            "n_bets": len(bucket),
            "win_rate": bucket["won"].mean(),
            "roi": profit / staked if staked > 0 else 0.0,
            "avg_edge": bucket["edge"].mean(),
            "avg_odds": bucket["odds"].mean(),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Season-over-Season Trends
# ═══════════════════════════════════════════════════════════════════════════════

def season_trends(bets_df: pd.DataFrame) -> pd.DataFrame:
    """Compute key metrics per season for trend analysis.

    Args:
        bets_df: DataFrame with 'season', 'won', 'profit_pct', 'stake_pct',
                 'edge', 'model_prob'.

    Returns:
        DataFrame with per-season: n_bets, win_rate, roi, avg_edge,
            brier_score, bankroll_growth.
    """
    if bets_df.empty or "season" not in bets_df.columns:
        return pd.DataFrame()

    rows = []
    cumulative_bankroll = 1.0

    for season in sorted(bets_df["season"].unique()):
        s = bets_df[bets_df["season"] == season]
        staked = s["stake_pct"].sum()
        profit = s["profit_pct"].sum()
        cumulative_bankroll *= (1 + profit / staked) if staked > 0 else 1.0

        row = {
            "season": int(season),
            "year": f"{2000+int(season)}/{str(2001+int(season))[-2:]}",
            "n_bets": len(s),
            "win_rate": s["won"].mean(),
            "roi": profit / staked if staked > 0 else 0.0,
            "avg_edge": s["edge"].mean(),
            "avg_odds": s["odds"].mean(),
            "profit": profit,
            "bankroll": cumulative_bankroll,
        }

        # Brier score if model_prob available
        if "model_prob" in s.columns and "won" in s.columns:
            valid = s["model_prob"].notna()
            if valid.sum() > 0:
                row["brier"] = brier_score(
                    s.loc[valid, "model_prob"].values,
                    s.loc[valid, "won"].values.astype(float),
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Full Analytics Report (from backtest)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_analytics(
    bets_df: pd.DataFrame,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run all analytics on a backtest bets DataFrame.

    Args:
        bets_df: Combined bets from walk-forward backtest.
        verbose: Print summary tables.

    Returns:
        Dict of analysis name → DataFrame.
    """
    results: dict[str, pd.DataFrame] = {}

    # 1. Edge buckets
    edge_df = edge_bucket_analysis(bets_df)
    results["edge_buckets"] = edge_df
    if verbose and not edge_df.empty:
        print("\n" + "=" * 70)
        print("EDGE BUCKET ANALYSIS")
        print("Does higher edge = higher win rate?")
        print("=" * 70)
        for _, row in edge_df.iterrows():
            print(f"  {row['bucket']:>6s}  "
                  f"{row['n_bets']:>4d} bets  "
                  f"Win: {row['win_rate']:>5.1%}  "
                  f"ROI: {row['roi']:>+6.1%}  "
                  f"Avg edge: {row['avg_edge']:.1%}  "
                  f"Avg odds: {row['avg_odds']:.2f}")

    # 2. Calibration
    cal_df = calibration_curve(bets_df)
    results["calibration"] = cal_df
    if verbose and not cal_df.empty:
        print("\n" + "=" * 70)
        print("CALIBRATION CURVE")
        print("Model predicted % vs actual win %")
        print("=" * 70)
        for _, row in cal_df.iterrows():
            direction = "+" if row["gap"] > 0 else ""
            print(f"  {row['bin_label']:>9s}  "
                  f"Predicted: {row['predicted']:>5.1%}  "
                  f"Actual: {row['actual']:>5.1%}  "
                  f"Gap: {direction}{row['gap']:.1%}  "
                  f"({row['n_bets']} bets)")

    # 3. Confidence validation
    conf_df = confidence_validation(bets_df)
    results["confidence"] = conf_df
    if verbose and not conf_df.empty:
        print("\n" + "=" * 70)
        print("CONFIDENCE LEVEL VALIDATION")
        print("=" * 70)
        for _, row in conf_df.iterrows():
            print(f"  {row['confidence']:>6s}  "
                  f"{row['n_bets']:>4d} bets  "
                  f"Win: {row['win_rate']:>5.1%}  "
                  f"ROI: {row['roi']:>+6.1%}  "
                  f"Avg edge: {row['avg_edge']:.1%}")

    # 4. Side analysis
    side_df = side_analysis(bets_df)
    results["sides"] = side_df
    if verbose and not side_df.empty:
        print("\n" + "=" * 70)
        print("SIDE BREAKDOWN")
        print("=" * 70)
        for _, row in side_df.iterrows():
            print(f"  {row['side']:>5s}  "
                  f"{row['n_bets']:>4d} bets  "
                  f"Win: {row['win_rate']:>5.1%}  "
                  f"ROI: {row['roi']:>+6.1%}  "
                  f"Avg odds: {row['avg_odds']:.2f}")

    # 5. Season trends
    trend_df = season_trends(bets_df)
    results["seasons"] = trend_df
    if verbose and not trend_df.empty:
        print("\n" + "=" * 70)
        print("SEASON TRENDS")
        print("=" * 70)
        for _, row in trend_df.iterrows():
            brier_str = f"  Brier: {row['brier']:.4f}" if "brier" in row and pd.notna(row.get("brier")) else ""
            print(f"  {row['year']}  "
                  f"{row['n_bets']:>4d} bets  "
                  f"Win: {row['win_rate']:>5.1%}  "
                  f"ROI: {row['roi']:>+6.1%}  "
                  f"Bankroll: {row['bankroll']:.3f}"
                  f"{brier_str}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Live Recommendations Analytics (from DB)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_live_recommendations(
    league: str = "PL",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run analytics on settled live recommendations from the dashboard DB.

    Args:
        league: League key ("PL" or "EFL").
        verbose: Print summary tables.

    Returns:
        Dict of analysis name → DataFrame.
    """
    try:
        from dashboard import _get_db
    except ImportError:
        print("Cannot import dashboard — run from project root.")
        return {}

    with _get_db(league) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM recommendations WHERE settled = 1",
            conn,
        )

    if df.empty:
        if verbose:
            print(f"No settled bets found for {league}.")
        return {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"LIVE RECOMMENDATIONS ANALYTICS — {league}")
        print(f"Settled bets: {len(df)}")
        print(f"{'='*70}")

    # Normalise column names to match backtest format
    df = df.rename(columns={
        "profit_pct": "profit_pct",
        "stake_pct": "stake_pct",
    })

    # Ensure numeric types
    for col in ["model_prob", "edge", "odds", "stake_pct", "profit_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "won" in df.columns:
        df["won"] = df["won"].astype(int)

    # Derive confidence if missing
    if "confidence" not in df.columns or df["confidence"].isna().all():
        df["confidence"] = df["edge"].apply(
            lambda e: "high" if e > 0.04 else
                      "medium" if e > 0.025 else "low"
        )

    return run_full_analytics(df, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_analytics(
    market: str = "ou25",
    start_season: int = 19,
    end_season: int = 25,
) -> dict[str, pd.DataFrame]:
    """Run walk-forward backtest and then full analytics on the results.

    Args:
        market: "ou25" or "btts".
        start_season: First test season.
        end_season: Last test season.

    Returns:
        Dict of analysis DataFrames.
    """
    if market == "ou25":
        from backtest import run_backtest
        print("Running O/U 2.5 walk-forward backtest...\n")
        bets_df = run_backtest(
            start_season=start_season,
            end_season=end_season,
            verbose=True,
        )
        # run_backtest returns None but prints; we need the bets.
        # Re-run to capture bets_df properly.
        from backtest import run_backtest as _rb, DEFAULT_CONFIG
        from pipeline import run_pipeline
        from model import tune_dc_params

        data = run_pipeline(verbose=False)
        full_df = data["full_df"]
        features = list(data["features"])
        tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
        dc_kwargs = tune_dc_params(tune_df)

        from backtest import backtest_season
        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in range(start_season, end_season + 1):
            train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                               (full_df["SeasonIndex"] < season)].copy()
            test_df = full_df[full_df["SeasonIndex"] == season].copy()

            has_odds = test_df["B365Greater2.5"].notna().sum()
            if has_odds < 50 or len(train_df) < 500:
                continue

            bets_df, metrics, cumulative_bankroll, peak_bankroll = backtest_season(
                train_df, test_df, features, dc_kwargs=dc_kwargs,
                cumulative_bankroll=cumulative_bankroll,
                peak_bankroll=peak_bankroll,
            )
            all_bets.append(bets_df)

        if all_bets:
            total_bets = pd.concat(all_bets, ignore_index=True)
            return run_full_analytics(total_bets, verbose=True)

    elif market == "btts":
        from btts_backtest import run_btts_backtest, btts_backtest_season
        from pipeline import run_pipeline
        from model import tune_dc_params

        print("Running BTTS walk-forward backtest...\n")
        data = run_pipeline(verbose=False)
        full_df = data["full_df"]
        features = list(data["features"])
        tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
        dc_kwargs = tune_dc_params(tune_df)

        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in range(start_season, end_season + 1):
            train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                               (full_df["SeasonIndex"] < season)].copy()
            test_df = full_df[full_df["SeasonIndex"] == season].copy()

            if len(train_df) < 500:
                continue

            bets_df, metrics, cumulative_bankroll, peak_bankroll = btts_backtest_season(
                train_df, test_df, features, dc_kwargs=dc_kwargs,
                cumulative_bankroll=cumulative_bankroll,
                peak_bankroll=peak_bankroll,
            )
            all_bets.append(bets_df)

        if all_bets:
            total_bets = pd.concat(all_bets, ignore_index=True)
            return run_full_analytics(total_bets, verbose=True)

    return {}


if __name__ == "__main__":
    import sys

    market = sys.argv[1] if len(sys.argv) > 1 else "ou25"
    source = sys.argv[2] if len(sys.argv) > 2 else "backtest"

    if source == "live":
        league = sys.argv[3] if len(sys.argv) > 3 else "PL"
        analyse_live_recommendations(league=league)
    else:
        run_backtest_analytics(market=market)
