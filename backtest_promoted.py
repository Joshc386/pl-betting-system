"""
Backtest: evaluate the impact of initialize_promoted_features() on model accuracy.

Runs the Championship pipeline twice (with and without promoted feature
initialisation), trains a walk-forward XGBoost model on each, and compares
metrics specifically for matches involving newly promoted/relegated teams.

Usage:
    python backtest_promoted.py
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

from championship_pipeline import (
    CHAMP_ALL_FEATURES,
    _detect_new_teams,
    initialize_promoted_features,
    run_pipeline,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Configuration ────────────────────────────────────────────────────────────

MIN_TEST_SEASON: int = 8  # first season used as test fold
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
    "verbosity": 0,
}
TARGET: str = "Over_2_5"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_promoted_mask(
    df: pd.DataFrame,
    new_teams: dict[int, set[str]],
) -> pd.Series:
    """Return a boolean mask: True for rows where Home or Away is a new team."""
    mask = pd.Series(False, index=df.index)
    for season_idx, teams in new_teams.items():
        season_rows = df["SeasonIndex"] == season_idx
        for team in teams:
            mask |= season_rows & (
                (df["Home_Team"] == team) | (df["Away_Team"] == team)
            )
    return mask


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Return AUC or None if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    """Return log loss or None if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(log_loss(y_true, y_prob))


def walk_forward_evaluate(
    df: pd.DataFrame,
    features: list[str],
    new_teams: dict[int, set[str]],
) -> dict[str, list[dict]]:
    """Run walk-forward CV and collect per-fold predictions.

    Args:
        df: Full pipeline DataFrame.
        features: Feature columns to use.
        new_teams: Mapping of SeasonIndex -> set of new team names.

    Returns:
        Dict with 'all', 'promoted', 'non_promoted' keys, each containing
        a list of per-fold metric dicts.
    """
    seasons = sorted(df["SeasonIndex"].unique())
    test_seasons = [s for s in seasons if s >= MIN_TEST_SEASON]

    promoted_mask = _build_promoted_mask(df, new_teams)

    all_preds: list[np.ndarray] = []
    all_trues: list[np.ndarray] = []
    all_promo_flags: list[np.ndarray] = []
    fold_results: list[dict] = []

    for test_season in test_seasons:
        train_mask = df["SeasonIndex"] < test_season
        test_mask = df["SeasonIndex"] == test_season

        X_train = df.loc[train_mask, features].copy()
        y_train = df.loc[train_mask, TARGET].copy()
        X_test = df.loc[test_mask, features].copy()
        y_test = df.loc[test_mask, TARGET].copy()

        # Drop rows with NaN target
        train_valid = y_train.notna()
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]

        test_valid = y_test.notna()
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        promo_flags = promoted_mask.loc[X_test.index].values

        all_preds.append(y_prob)
        all_trues.append(y_test.values)
        all_promo_flags.append(promo_flags)

        # Per-fold metrics
        fold_results.append({
            "season": int(test_season),
            "n_total": len(y_test),
            "n_promoted": int(promo_flags.sum()),
            "auc_all": _safe_auc(y_test.values, y_prob),
            "acc_all": float(accuracy_score(y_test, y_pred)),
        })

    # Aggregate across all folds
    y_true_all = np.concatenate(all_trues)
    y_prob_all = np.concatenate(all_preds)
    promo_all = np.concatenate(all_promo_flags).astype(bool)

    results: dict[str, Any] = {}

    # All matches
    results["all"] = {
        "n": len(y_true_all),
        "auc": _safe_auc(y_true_all, y_prob_all),
        "accuracy": float(accuracy_score(y_true_all, (y_prob_all >= 0.5).astype(int))),
        "log_loss": _safe_logloss(y_true_all, y_prob_all),
    }

    # Promoted team matches
    if promo_all.sum() > 0:
        results["promoted"] = {
            "n": int(promo_all.sum()),
            "auc": _safe_auc(y_true_all[promo_all], y_prob_all[promo_all]),
            "accuracy": float(
                accuracy_score(
                    y_true_all[promo_all],
                    (y_prob_all[promo_all] >= 0.5).astype(int),
                )
            ),
            "log_loss": _safe_logloss(y_true_all[promo_all], y_prob_all[promo_all]),
        }
    else:
        results["promoted"] = {"n": 0, "auc": None, "accuracy": None, "log_loss": None}

    # Non-promoted team matches
    non_promo = ~promo_all
    if non_promo.sum() > 0:
        results["non_promoted"] = {
            "n": int(non_promo.sum()),
            "auc": _safe_auc(y_true_all[non_promo], y_prob_all[non_promo]),
            "accuracy": float(
                accuracy_score(
                    y_true_all[non_promo],
                    (y_prob_all[non_promo] >= 0.5).astype(int),
                )
            ),
            "log_loss": _safe_logloss(y_true_all[non_promo], y_prob_all[non_promo]),
        }
    else:
        results["non_promoted"] = {
            "n": 0, "auc": None, "accuracy": None, "log_loss": None,
        }

    results["folds"] = fold_results
    return results


def _strip_promoted_init(df: pd.DataFrame) -> pd.DataFrame:
    """Re-run pipeline steps but skip initialize_promoted_features().

    Since run_pipeline() already applies the initialization, we need a
    separate path. We re-run the pipeline and return the DataFrame from
    just BEFORE initialize_promoted_features() is called.
    """
    # Import the individual pipeline steps
    from championship_pipeline import (
        add_advanced_features,
        add_congestion_features,
        add_context_features,
        add_derived_features,
        add_discipline_features,
        add_elo,
        add_halftime_features,
        add_poisson_features,
        load_championship_data,
    )
    from api.computed_strengths import compute_team_strengths, merge_strengths

    df = load_championship_data()
    df = add_derived_features(df)
    df = add_congestion_features(df)
    df = add_discipline_features(df)
    df = add_halftime_features(df)
    df = add_advanced_features(df)
    df = add_elo(df)
    df = add_poisson_features(df)
    strengths = compute_team_strengths(df)
    df = merge_strengths(df, strengths)
    df = add_context_features(df)
    # Deliberately skip initialize_promoted_features()
    return df


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    """Format a metric value, handling None."""
    if val is None:
        return "  N/A  "
    return f"{val:{fmt}}"


def print_comparison(
    with_init: dict[str, Any],
    without_init: dict[str, Any],
) -> None:
    """Print a clear comparison table of the two variants."""
    sep = "=" * 82
    thin_sep = "-" * 82

    print(f"\n{sep}")
    print("  PROMOTED TEAM FEATURE INITIALISATION — BACKTEST COMPARISON")
    print(f"{sep}\n")

    # Per-fold summary
    print("  Per-fold summary (WITH initialisation):")
    print(f"  {'Season':>8s}  {'N total':>8s}  {'N promo':>8s}  {'AUC':>8s}  {'Acc':>8s}")
    print(f"  {thin_sep[:50]}")
    for fold in with_init["folds"]:
        print(
            f"  {fold['season']:>8d}  {fold['n_total']:>8d}  "
            f"{fold['n_promoted']:>8d}  {_fmt(fold['auc_all']):>8s}  "
            f"{fold['acc_all']:>8.4f}"
        )
    print()

    # Main comparison table
    categories = [
        ("ALL MATCHES", "all"),
        ("PROMOTED TEAM MATCHES", "promoted"),
        ("NON-PROMOTED MATCHES", "non_promoted"),
    ]

    header = (
        f"  {'Category':<24s}  {'Variant':<12s}  "
        f"{'N':>6s}  {'AUC':>8s}  {'Acc':>8s}  {'LogLoss':>8s}"
    )
    print(header)
    print(f"  {thin_sep}")

    for label, key in categories:
        w = with_init[key]
        wo = without_init[key]

        print(
            f"  {label:<24s}  {'WITH':>12s}  "
            f"{w['n']:>6d}  {_fmt(w['auc']):>8s}  "
            f"{_fmt(w['accuracy']):>8s}  {_fmt(w['log_loss']):>8s}"
        )
        print(
            f"  {'':<24s}  {'WITHOUT':>12s}  "
            f"{wo['n']:>6d}  {_fmt(wo['auc']):>8s}  "
            f"{_fmt(wo['accuracy']):>8s}  {_fmt(wo['log_loss']):>8s}"
        )

        # Delta row
        def _delta(a: float | None, b: float | None) -> str:
            if a is None or b is None:
                return "    -   "
            diff = a - b
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.4f}"

        auc_d = _delta(w["auc"], wo["auc"])
        acc_d = _delta(w["accuracy"], wo["accuracy"])
        # For log loss, lower is better, so invert sign for interpretation
        ll_d = _delta(w["log_loss"], wo["log_loss"])

        print(
            f"  {'':<24s}  {'DELTA':>12s}  "
            f"{'':>6s}  {auc_d:>8s}  {acc_d:>8s}  {ll_d:>8s}"
        )
        print(f"  {thin_sep}")

    # Interpretation
    promo_w = with_init["promoted"]
    promo_wo = without_init["promoted"]
    print(f"\n  Interpretation:")
    print(f"  - Promoted match count: {promo_w['n']}")

    if promo_w["auc"] is not None and promo_wo["auc"] is not None:
        auc_diff = promo_w["auc"] - promo_wo["auc"]
        direction = "IMPROVES" if auc_diff > 0 else "WORSENS" if auc_diff < 0 else "NO CHANGE"
        print(f"  - Promoted team AUC: {direction} by {abs(auc_diff):.4f}")

    if promo_w["accuracy"] is not None and promo_wo["accuracy"] is not None:
        acc_diff = promo_w["accuracy"] - promo_wo["accuracy"]
        direction = "IMPROVES" if acc_diff > 0 else "WORSENS" if acc_diff < 0 else "NO CHANGE"
        print(f"  - Promoted team accuracy: {direction} by {abs(acc_diff):.4f}")

    # Regression check on non-promoted
    np_w = with_init["non_promoted"]
    np_wo = without_init["non_promoted"]
    if np_w["auc"] is not None and np_wo["auc"] is not None:
        np_diff = np_w["auc"] - np_wo["auc"]
        if abs(np_diff) < 0.005:
            print(f"  - Non-promoted AUC: STABLE (delta {np_diff:+.4f})")
        elif np_diff < 0:
            print(f"  - Non-promoted AUC: REGRESSION of {abs(np_diff):.4f} — investigate")
        else:
            print(f"  - Non-promoted AUC: IMPROVED by {np_diff:.4f}")

    print(f"\n{sep}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full promoted-features backtest comparison."""
    print("Running Championship pipeline WITH promoted feature initialisation...")
    result_with = run_pipeline(verbose=False)
    df_with = result_with["full_df"]
    features_with = result_with["features"]

    print("Running Championship pipeline WITHOUT promoted feature initialisation...")
    df_without = _strip_promoted_init(df_with)
    features_without = [
        f for f in CHAMP_ALL_FEATURES
        if f in df_without.columns and df_without[f].notna().any()
    ]

    # Use the intersection of features so the comparison is fair
    common_features = sorted(set(features_with) & set(features_without))
    print(f"Common features: {len(common_features)}")

    # Detect promoted teams (same for both variants — based on raw team presence)
    new_teams = _detect_new_teams(df_with)
    total_new = sum(len(v) for v in new_teams.values())
    print(f"Detected {total_new} newly entered teams across {len(new_teams)} seasons")

    for season_idx in sorted(new_teams.keys()):
        teams = new_teams[season_idx]
        print(f"  Season {season_idx}: {', '.join(sorted(teams))}")

    print("\nTraining walk-forward models (WITH initialisation)...")
    results_with = walk_forward_evaluate(df_with, common_features, new_teams)

    print("Training walk-forward models (WITHOUT initialisation)...")
    results_without = walk_forward_evaluate(df_without, common_features, new_teams)

    print_comparison(results_with, results_without)


if __name__ == "__main__":
    main()
