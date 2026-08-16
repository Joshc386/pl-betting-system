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
# Confidence-interval helpers (Performance tab)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These exist so that headline numbers on the dashboard can show "is this
# actually trustworthy?" alongside the point estimate. Two methods:
#
#   wilson_ci   — for binomial proportions (win rate). Closed form, always
#                 valid even at n=1 or wins=0/n. Standard in the literature
#                 for hit-rate confidence intervals.
#
#   bootstrap_ci — for arbitrary distributions (ROI, P/L). Resamples the
#                 underlying profit array N times and reads percentiles.
#                 More expensive but works for ratios and skewed
#                 distributions where parametric formulas would lie.
# ═══════════════════════════════════════════════════════════════════════════════

def wilson_ci(wins: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the textbook normal-approximation interval at small
    n, and always returns a valid [0, 1] range (the normal approximation
    can produce negative lower bounds at low win counts).

    Args:
        wins: count of successful trials
        n: total trials
        alpha: significance level (default 0.05 for 95% CI)

    Returns:
        (lo, hi) tuple, both in [0, 1]. Returns (0.0, 1.0) when n == 0.
    """
    if n <= 0:
        return (0.0, 1.0)
    # 1.96 for 95% CI; computed from inverse normal CDF
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p_hat = wins / n
    denom = 1 + (z ** 2) / n
    centre = (p_hat + (z ** 2) / (2 * n)) / denom
    half = (z * np.sqrt(p_hat * (1 - p_hat) / n + (z ** 2) / (4 * n ** 2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap percentile CI for the mean of an array.

    Used for ROI and P/L confidence intervals where the underlying
    profit-per-bet distribution is heavy-tailed (a single +6.0u win
    drags the mean far more than a -1u loss). Bootstrap is robust to
    that shape; a normal-approximation CI would be too narrow.

    Args:
        values: 1-D array of per-bet outcomes (e.g. profit_pct values)
        n_resamples: bootstrap iterations (2000 = solid for 95% CI)
        alpha: significance level
        seed: RNG seed for reproducibility — same data, same CI

    Returns:
        (lo, hi, mean) tuple. Returns (nan, nan, nan) for empty input.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        # single observation — can't bootstrap, return point with infinite CI
        return (float("nan"), float("nan"), float(arr[0]))

    rng = np.random.default_rng(seed)
    n = arr.size
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi, float(arr.mean()))


def adequacy_label(n: int, ci_lo: float, ci_hi: float) -> str:
    """Three-state badge for whether a result is interpretable.

    Used in the Performance-tab per-market table to flag which rows are
    statistically meaningful vs which are noise. The thresholds are
    pragmatic (not derived from a power calc) — the goal is to stop the
    eye reading a +28% ROI on 18 bets as a real edge.

    Returns:
        "ok" | "marginal" | "noise"
    """
    if n < 30:
        return "noise"
    # CI width on ROI / P/L — wider than ±15pp means we can't tell
    # apart "+5%" from "-5%" so calling the result anything is dishonest
    if not np.isnan(ci_lo) and not np.isnan(ci_hi):
        width = ci_hi - ci_lo
        if width > 0.30:
            return "noise"
        # CI doesn't straddle 0 → effect is detectable
        if ci_lo > 0 or ci_hi < 0:
            return "ok"
    return "marginal"


# ═══════════════════════════════════════════════════════════════════════════════
# Historical agreement (ADR 0010)
# ═══════════════════════════════════════════════════════════════════════════════


def realised_edge(won, fair_prob) -> float:
    """What a set of bets actually beat the market by.

    ``mean(won) - mean(fair_prob)``. Where ``edge`` is the claim the model
    makes before kickoff, this is that claim graded against outcomes.

    Unlike hit rate it is comparable across markets, which is the whole
    reason it exists: PL O/U 1.5 Over wins ~75% of the time and EFL O/U 3.5
    ~35%, so a raw hit rate mostly reports which market you are looking at.
    Subtracting the fair price removes the base rate, leaving an unskilled
    bet at ~0 in any market. It is also not an ROI estimand, so it is
    immune to the historical-``_first``-price vs best-of-14-books mismatch
    that makes backtest and live ROI unpoolable.

    Meaningful only over a set — for one bet this is a noisy Bernoulli
    residual.

    Returns:
        The realised edge, or nan for empty input.
    """
    w = np.asarray(won, dtype=float)
    f = np.asarray(fair_prob, dtype=float)
    if w.size == 0:
        return float("nan")
    return float(w.mean() - f.mean())


def clustered_bootstrap_ci(won, fair_prob, fixture_ids,
                           n_resamples: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> tuple[float, float]:
    """Percentile CI for ``realised_edge``, resampling fixtures not rows.

    One fixture can produce several bets — an O/U 2.5 bet and a BTTS bet on
    the same match — and their outcomes are driven by the same goals. O/U
    2.5 Over is outright *nested* inside O/U 1.5 Over: if the total beat
    2.5 it has already beaten 1.5. Treating those as independent trials,
    which ``wilson_ci`` and ``bootstrap_ci`` both do, reports an interval
    narrower than the evidence supports.

    Drawing whole fixtures keeps the interval honest under that correlation
    without having to model its shape.

    Args:
        won: 1/0 outcome per bet
        fair_prob: market fair probability per bet
        fixture_ids: cluster key per bet — bets sharing one settle together
        n_resamples: bootstrap iterations
        alpha: significance level
        seed: RNG seed — same data, same CI

    Returns:
        (lo, hi). Returns (nan, nan) when there are fewer than two fixtures.
    """
    w = np.asarray(won, dtype=float)
    f = np.asarray(fair_prob, dtype=float)
    if w.size == 0:
        return (float("nan"), float("nan"))

    # Positional row indices per fixture, so a drawn fixture brings all
    # of its bets along.
    groups: dict = {}
    for pos, key in enumerate(fixture_ids):
        groups.setdefault(key, []).append(pos)
    clusters = [np.asarray(v, dtype=int) for v in groups.values()]
    if len(clusters) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    n_clusters = len(clusters)
    for i in range(n_resamples):
        pick = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([clusters[j] for j in pick])
        stats[i] = w[idx].mean() - f[idx].mean()
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


_MODEL_NAMES = ("xgb", "lgb", "dc", "lr")

# The six (league, market) cells with an OOF cache.
OOF_CELLS = (("PL", "ou25"), ("PL", "btts"), ("PL", "ou15"),
             ("EFL", "ou25"), ("EFL", "btts"), ("EFL", "ou15"))


def load_oof_cell(league: str, market: str,
                  cache_dir=None) -> Optional[pd.DataFrame]:
    """Load one OOF cache, or None when it has not been generated.

    ``reports/`` is gitignored, so these are build artefacts that may
    legitimately be absent. None is returned rather than an empty frame so
    callers can tell "not generated yet" from "generated and empty" — the
    dashboard must say which, never render a silent blank.
    """
    from pathlib import Path

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent / "reports" / \
            "roi_validate" / "oof_cache"
    path = Path(cache_dir) / f"{league.lower()}_{market}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def replay_oof_gate(oof_df: pd.DataFrame,
                    config: Optional[dict] = None) -> pd.DataFrame:
    """Replay an OOF cache through the live gate, keeping both sides.

    The backtest runners drop rejected bets before recording them, so the
    agreement levels the gate turns down leave no trace. The OOF cache
    holds every fixture the model priced, so the gate can be applied here
    and its threshold moved — which is what makes "where should min_agree
    sit?" answerable at all (ADR 0010).

    Arithmetic mirrors ``backtest.py`` exactly: per-model probabilities are
    logit-shift calibrated, the market is de-vigged proportionally, and
    ``n_agree`` counts models beating the fair price on the side being
    evaluated.

    Both sides are returned. Callers wanting a pre-gate view must keep one
    side per fixture — see ``agreement_bins`` — because the two sides are
    antisymmetric by construction.

    Returns:
        One row per (fixture, side) with fixture, season, side, side_col,
        n_models, n_agree, model_prob, fair_prob, odds, edge, ev, won and
        passes.
    """
    from scripts.roi_validate import (DEFAULT_CONFIG, _calibrate_single,
                                      _implied_fair_prob)

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    bw = cfg["blend_weight"]
    min_edge = cfg["min_edge"]
    min_agree = cfg["min_agree"]

    rows: list[dict] = []
    for _, r in oof_df.iterrows():
        per_model = [
            _calibrate_single(float(r[f"{m}_prob"]), float(r[f"{m}_shift"]))
            for m in _MODEL_NAMES
            if r.get(f"{m}_prob") is not None and pd.notna(r.get(f"{m}_prob"))
        ]
        if not per_model:
            continue
        pm = np.array(per_model, dtype=float)

        fair_a, fair_b = _implied_fair_prob(r.get("odds_a"), r.get("odds_b"))
        if np.isnan(fair_a) or np.isnan(fair_b):
            continue
        model_a = float(pm.mean())
        fixture = f'{r["date"]}|{r["home_team"]}|{r["away_team"]}'

        for label, model_p, fair_p, odds, col in (
            (r["side_a_label"], model_a, fair_a, r["odds_a"], "a"),
            (r["side_b_label"], 1.0 - model_a, fair_b, r["odds_b"], "b"),
        ):
            if odds is None or pd.isna(odds) or float(odds) <= 1:
                continue
            odds = float(odds)
            # per_model must reflect the side being evaluated
            pm_side = pm if col == "a" else 1.0 - pm
            n_agree = int(np.sum(pm_side > fair_p))
            blended = bw * model_p + (1 - bw) * fair_p
            edge = blended - fair_p
            ev = blended * odds - 1
            won = ((int(r["outcome"]) == 1 and col == "a")
                   or (int(r["outcome"]) == 0 and col == "b"))
            rows.append({
                "fixture": fixture, "season": int(r["season"]),
                "side": label, "side_col": col,
                "n_models": int(pm.size), "n_agree": n_agree,
                "model_prob": model_p, "fair_prob": fair_p, "odds": odds,
                "edge": edge, "ev": ev, "won": bool(won),
                "passes": bool(ev > 0 and edge >= min_edge
                               and n_agree >= min_agree),
            })
    return pd.DataFrame(rows)


def agreement_bins(replayed: pd.DataFrame, gated: bool = False,
                   seed: int = 0) -> pd.DataFrame:
    """Bin replayed OOF rows by agreement count, with clustered CIs.

    Two views, and the difference matters:

    ``gated=True`` keeps only bets the live gate would place. At most one
    side of a fixture can pass (the two sides' edges sum to zero and
    ``min_edge`` is positive), so no de-duplication is needed. This view
    answers "how have the bins we bet performed?" and structurally cannot
    contain agreement below ``min_agree``.

    ``gated=False`` keeps **side_a only** — one row per fixture. Both sides
    would make bins 0 and N the same fixtures mirrored, with realised edges
    that are exact negatives and a self-mirroring middle bin pinned to
    +0.00%; that is arithmetic, not evidence. Restricted to one side, the
    full 0..N range becomes readable and answers the question the gated
    view cannot: where *should* the threshold sit?

    Never pool cells into one call — agreement pays in PL O/U 2.5 and PL
    BTTS and in no other cell, and averaging them together hides it
    (ADR 0010).

    Returns:
        One row per agreement level: n_agree, n_rows, n_fixtures,
        realised_edge, ci_lo, ci_hi, claimed_edge, hit_rate, adequacy.
    """
    if replayed.empty:
        return pd.DataFrame()
    sub = (replayed[replayed["passes"]] if gated
           else replayed[replayed["side_col"] == "a"])
    if sub.empty:
        return pd.DataFrame()

    out = []
    for n in sorted(sub["n_agree"].unique()):
        b = sub[sub["n_agree"] == n]
        lo, hi = clustered_bootstrap_ci(
            b["won"].to_numpy(dtype=float),
            b["fair_prob"].to_numpy(dtype=float),
            b["fixture"].tolist(), seed=seed)
        out.append({
            "n_agree": int(n),
            "n_rows": len(b),
            "n_fixtures": b["fixture"].nunique(),
            "realised_edge": realised_edge(b["won"], b["fair_prob"]),
            "ci_lo": lo,
            "ci_hi": hi,
            "claimed_edge": float(b["edge"].mean()),
            "hit_rate": float(b["won"].mean()),
            "adequacy": adequacy_label(len(b), lo, hi),
        })
    return pd.DataFrame(out)


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
        from db import get_db
    except ImportError:
        print("Cannot import db — run from project root.")
        return {}

    with get_db(league) as conn:
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

            has_odds = test_df["Odds_Over_2.5"].notna().sum()
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


# ═══════════════════════════════════════════════════════════════════════════════
# Counterfactual strategy comparison (Performance tab Section 2)
# ═══════════════════════════════════════════════════════════════════════════════

def counterfactual_strategies(predictions_df: pd.DataFrame,
                              recs_df: pd.DataFrame) -> pd.DataFrame:
    """Compare three betting strategies on the same settled outcomes.

    The point of this table is to disentangle two decisions the bot makes:
      1. Which markets to bet (the "recommendation" filter — n_agree, EV
         after vig, Kelly stake > 0)
      2. How much to stake (Kelly vs flat)

    Three rows answer those questions independently:

      Row 1  Recommended only — Kelly stake
             What the bot actually did. Real ROI / P/L.

      Row 2  Recommended only — flat stake (avg Kelly)
             Same selection, but flat-staked. Comparing row 1 vs row 2
             tells you whether Kelly sizing earned its variance.

      Row 3  All positive edge — flat stake (avg Kelly)
             Wider selection, same flat stake. Comparing row 2 vs row 3
             tells you whether the recommendation filter picks winners.

    All three rows use the SAME flat stake (the bot's average Kelly stake)
    so absolute stake size is held constant — the only differences are
    selection and sizing.

    Args:
        predictions_df: settled `predictions` rows (positive-edge, no
            recommendation filter applied). Needs: best_odds, won.
        recs_df: settled `recommendations` rows (filter passed).
            Needs: odds, won, profit_pct, stake_pct.

    Returns:
        DataFrame with columns: strategy, n_bets, win_rate, win_rate_lo,
        win_rate_hi, roi, roi_lo, roi_hi, pl_pct, pl_lo, pl_hi, avg_stake.
        Empty DataFrame when neither input has any settled rows.
    """
    rows: list[dict] = []

    # Average Kelly stake from settled recommendations — the flat-stake
    # baseline for rows 2 and 3. If no settled recs yet, fall back to 1%
    # so the comparison at least renders something.
    if not recs_df.empty and recs_df["stake_pct"].notna().any():
        avg_kelly = float(recs_df["stake_pct"].dropna().mean())
        if avg_kelly <= 0:
            avg_kelly = 0.01
    else:
        avg_kelly = 0.01

    def _summarise(profits_arr: np.ndarray, stakes_arr: np.ndarray,
                   wins: int, n: int, label: str) -> dict:
        """One row of the output table — point estimates plus CIs."""
        if n == 0:
            return {
                "strategy": label, "n_bets": 0,
                "win_rate": np.nan, "win_rate_lo": np.nan, "win_rate_hi": np.nan,
                "roi": np.nan, "roi_lo": np.nan, "roi_hi": np.nan,
                "pl_pct": np.nan, "pl_lo": np.nan, "pl_hi": np.nan,
                "avg_stake": np.nan,
            }
        # Win rate + Wilson CI
        wr = wins / n
        wr_lo, wr_hi = wilson_ci(wins, n)
        # ROI = profit / staked, bootstrapped on per-bet ROI
        total_staked = float(stakes_arr.sum())
        total_profit = float(profits_arr.sum())
        roi = total_profit / total_staked if total_staked > 0 else np.nan
        # ROI CI: bootstrap on per-bet ROI ratios for stability
        per_bet_roi = profits_arr / np.where(stakes_arr > 0, stakes_arr, np.nan)
        roi_lo, roi_hi, _ = bootstrap_ci(per_bet_roi)
        # P/L = bankroll change as %, bootstrap on per-bet profit
        pl_lo, pl_hi, pl_mean = bootstrap_ci(profits_arr)
        return {
            "strategy": label, "n_bets": n,
            "win_rate": wr, "win_rate_lo": wr_lo, "win_rate_hi": wr_hi,
            "roi": roi, "roi_lo": roi_lo, "roi_hi": roi_hi,
            "pl_pct": total_profit, "pl_lo": pl_lo * n, "pl_hi": pl_hi * n,
            "avg_stake": float(stakes_arr.mean()),
        }

    # Row 1 — recommended, Kelly stake (uses precomputed profit_pct from DB)
    rec_settled = recs_df[recs_df.get("settled", 0) == 1].copy() if not recs_df.empty else pd.DataFrame()
    if not rec_settled.empty:
        # profit_pct in the DB is already normalised to bankroll fraction.
        # Coerce to numeric first to dodge the pandas FutureWarning about
        # downcasting object dtypes on .fillna — settled-bet columns can
        # arrive as object dtype if any row had a NULL at insert time.
        profits_kelly = pd.to_numeric(
            rec_settled["profit_pct"], errors="coerce").fillna(0).to_numpy(dtype=float)
        stakes_kelly = pd.to_numeric(
            rec_settled["stake_pct"], errors="coerce").fillna(0).to_numpy(dtype=float)
        wins_kelly = int(pd.to_numeric(
            rec_settled["won"], errors="coerce").fillna(0).sum())
        rows.append(_summarise(
            profits_kelly, stakes_kelly, wins_kelly, len(rec_settled),
            f"Recommended — Kelly (avg {avg_kelly*100:.2f}%)"))

        # Row 2 — same selection, flat stake at avg Kelly
        odds_arr = pd.to_numeric(
            rec_settled["odds"], errors="coerce").fillna(0).to_numpy(dtype=float)
        won_arr = pd.to_numeric(
            rec_settled["won"], errors="coerce").fillna(0).to_numpy(dtype=float)
        # profit at flat stake: (odds-1)*stake on win, -stake on loss
        profits_flat = np.where(
            won_arr == 1,
            (odds_arr - 1) * avg_kelly,
            -avg_kelly,
        )
        stakes_flat = np.full(len(rec_settled), avg_kelly, dtype=float)
        rows.append(_summarise(
            profits_flat, stakes_flat, wins_kelly, len(rec_settled),
            f"Recommended — flat ({avg_kelly*100:.2f}%)"))
    else:
        rows.append(_summarise(np.array([]), np.array([]), 0, 0,
                               "Recommended — Kelly"))
        rows.append(_summarise(np.array([]), np.array([]), 0, 0,
                               "Recommended — flat"))

    # Row 3 — all settled positive-edge predictions, flat stake at avg Kelly
    if not predictions_df.empty:
        pred_settled = predictions_df[
            (predictions_df.get("settled", 0) == 1)
            & (predictions_df["edge_pct"].fillna(-1) > 0)
        ].copy()
    else:
        pred_settled = pd.DataFrame()
    if not pred_settled.empty:
        odds_arr = pd.to_numeric(
            pred_settled["best_odds"], errors="coerce").fillna(0).to_numpy(dtype=float)
        won_arr = pd.to_numeric(
            pred_settled["won"], errors="coerce").fillna(0).to_numpy(dtype=float)
        profits_all = np.where(
            won_arr == 1,
            (odds_arr - 1) * avg_kelly,
            -avg_kelly,
        )
        stakes_all = np.full(len(pred_settled), avg_kelly, dtype=float)
        wins_all = int(won_arr.sum())
        rows.append(_summarise(
            profits_all, stakes_all, wins_all, len(pred_settled),
            f"All positive edge — flat ({avg_kelly*100:.2f}%)"))
    else:
        rows.append(_summarise(np.array([]), np.array([]), 0, 0,
                               "All positive edge — flat"))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    market = sys.argv[1] if len(sys.argv) > 1 else "ou25"
    source = sys.argv[2] if len(sys.argv) > 2 else "backtest"

    if source == "live":
        league = sys.argv[3] if len(sys.argv) > 3 else "PL"
        analyse_live_recommendations(league=league)
    else:
        run_backtest_analytics(market=market)
