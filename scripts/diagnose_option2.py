"""
Option 2 diagnostic — isolated DC-only comparison.

Compares three Dixon-Coles configurations on walk-forward validation folds
for each (league, market) combination, to see whether Option 2's changes
(Step 1: per-market tuning, Step 2: partial-pooling shrinkage) actually
improve the model in isolation before we commit to a full backtest.

Variants:
  A. BASELINE      = tune once on O/U 2.5, reuse for all markets, no shrinkage
                     (equivalent to pre-Option-2 behaviour)
  B. STEP 1 ONLY   = tune per market, no shrinkage
  C. STEPS 1 + 2   = tune per market, with partial-pooling shrinkage

Reports AUC and Brier score per fold and aggregated across folds.
AUC is the primary ranking metric (ROI cannot be computed from DC alone
without the full ensemble + odds — that's the next step's job).

Runtime: ~5 min total (tuning dominates).
Run from repo root:  python scripts/diagnose_option2.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from model import DixonColesPredictor, tune_dc_params
from championship_model import tune_dc_params_champ, MIN_TRAIN_SEASON
from pipeline import run_pipeline as pl_pipeline
from championship_pipeline import run_pipeline as efl_pipeline


class MarketSpec(NamedTuple):
    """One market to evaluate DC on."""
    league: str           # "PL" or "EFL"
    market: str           # "ou25", "ou15", "btts"
    target_col: str       # column in full_df holding the binary outcome
    predict_fn: str       # DixonColesPredictor method name


def walk_forward_folds(
    full_df: pd.DataFrame,
    min_train_season: int,
    start_val_season: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Generate (train, val, season_index) folds in chronological order."""
    seasons = sorted(full_df["SeasonIndex"].unique())
    folds = []
    for vs in range(start_val_season, max(seasons) + 1):
        train = full_df[(full_df["SeasonIndex"] >= min_train_season) &
                        (full_df["SeasonIndex"] < vs)]
        val = full_df[full_df["SeasonIndex"] == vs]
        if len(train) >= 100 and len(val) >= 50:
            folds.append((train, val, vs))
    return folds


def evaluate_dc(
    dc: DixonColesPredictor,
    val_df: pd.DataFrame,
    target_col: str,
    predict_fn: str,
) -> tuple[float, float] | None:
    """Return (auc, brier) for a fitted DC on a validation fold, or None if
    there's no class variance in the val fold."""
    preds = getattr(dc, predict_fn)(val_df)
    y = val_df[target_col].values
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)
    return float(auc), float(brier)


def run_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dc_kwargs: dict,
    use_shrinkage: bool,
    target_col: str,
    predict_fn: str,
) -> tuple[float, float] | None:
    """Train a DC with the given kwargs + shrinkage flag, return (auc, brier)."""
    dc = DixonColesPredictor(
        rho=dc_kwargs["rho"],
        half_life=dc_kwargs["half_life"],
        use_shrinkage=use_shrinkage,
    )
    dc.fit(train_df)
    return evaluate_dc(dc, val_df, target_col, predict_fn)


def diagnose_market(
    spec: MarketSpec,
    full_df: pd.DataFrame,
    shared_kwargs: dict,
    market_kwargs: dict,
) -> dict:
    """Run all three variants across all folds for a single market."""
    print(f"\n{'='*70}")
    print(f"  {spec.league} / {spec.market}  (target={spec.target_col})")
    print(f"{'='*70}")
    print(f"  Shared kwargs (O/U 2.5 tune): {shared_kwargs}")
    print(f"  Market kwargs (per-market tune): {market_kwargs}")

    if spec.league == "PL":
        folds = walk_forward_folds(full_df, min_train_season=14, start_val_season=19)
    else:
        folds = walk_forward_folds(
            full_df, min_train_season=MIN_TRAIN_SEASON, start_val_season=19)

    results: dict[str, list[tuple[int, float, float]]] = {
        "A_baseline": [],
        "B_step1": [],
        "C_steps1_2": [],
    }

    print(f"\n  Fold-by-fold AUC / Brier:")
    print(f"  {'Season':<8} {'A_baseline':<20} {'B_step1':<20} {'C_1+2':<20}")

    for train_df, val_df, vs in folds:
        # Ensure BTTS column exists for the PL case if needed
        if spec.target_col == "BTTS" and "BTTS" not in val_df.columns:
            val_df = val_df.copy()
            val_df["BTTS"] = ((val_df["Home_Goals"] > 0)
                              & (val_df["Away_Goals"] > 0)).astype(int)
        if spec.target_col == "BTTS" and "BTTS" not in train_df.columns:
            train_df = train_df.copy()
            train_df["BTTS"] = ((train_df["Home_Goals"] > 0)
                                & (train_df["Away_Goals"] > 0)).astype(int)

        a = run_variant(train_df, val_df, shared_kwargs, use_shrinkage=False,
                         target_col=spec.target_col, predict_fn=spec.predict_fn)
        b = run_variant(train_df, val_df, market_kwargs, use_shrinkage=False,
                         target_col=spec.target_col, predict_fn=spec.predict_fn)
        c = run_variant(train_df, val_df, market_kwargs, use_shrinkage=True,
                         target_col=spec.target_col, predict_fn=spec.predict_fn)

        def _fmt(res):
            if res is None:
                return "  (no variance)    "
            return f"AUC={res[0]:.4f} Br={res[1]:.4f}"

        print(f"  S{vs:<7} {_fmt(a):<20} {_fmt(b):<20} {_fmt(c):<20}")

        if a: results["A_baseline"].append((vs, a[0], a[1]))
        if b: results["B_step1"].append((vs, b[0], b[1]))
        if c: results["C_steps1_2"].append((vs, c[0], c[1]))

    # Aggregated means
    print(f"\n  Mean across folds:")
    print(f"  {'Variant':<20} {'AUC':<12} {'Brier':<12} {'n_folds':<8}")
    summary: dict[str, dict] = {}
    for variant, rows in results.items():
        if not rows:
            continue
        mean_auc = np.mean([r[1] for r in rows])
        mean_brier = np.mean([r[2] for r in rows])
        summary[variant] = {"auc": float(mean_auc),
                            "brier": float(mean_brier),
                            "n": len(rows)}
        print(f"  {variant:<20} {mean_auc:.4f}       {mean_brier:.4f}       {len(rows)}")

    # Deltas vs baseline
    if "A_baseline" in summary:
        base_auc = summary["A_baseline"]["auc"]
        base_brier = summary["A_baseline"]["brier"]
        print(f"\n  Delta vs baseline (A):")
        for variant in ("B_step1", "C_steps1_2"):
            if variant in summary:
                d_auc = summary[variant]["auc"] - base_auc
                d_brier = summary[variant]["brier"] - base_brier
                auc_sym = "+" if d_auc > 0 else ""
                brier_sym = "+" if d_brier > 0 else ""
                print(f"  {variant:<20} dAUC={auc_sym}{d_auc:+.4f}  "
                      f"dBrier={brier_sym}{d_brier:+.4f}  "
                      f"({'better' if d_auc > 0 else 'worse'} AUC, "
                      f"{'worse' if d_brier > 0 else 'better'} Brier)")

    return summary


def main():
    print("\n" + "#" * 70)
    print("#  Option 2 Diagnostic — DC-only comparison")
    print("#" * 70)

    # ── Load pipelines ──
    print("\n[1/3] Loading PL pipeline...")
    pl_data = pl_pipeline(verbose=False)
    pl_df = pl_data["full_df"]
    if "BTTS" not in pl_df.columns:
        pl_df["BTTS"] = ((pl_df["Home_Goals"] > 0)
                         & (pl_df["Away_Goals"] > 0)).astype(int)
    print(f"  PL: {len(pl_df)} rows, seasons "
          f"{pl_df['SeasonIndex'].min()}-{pl_df['SeasonIndex'].max()}")

    print("[2/3] Loading EFL pipeline...")
    efl_data = efl_pipeline(verbose=False)
    efl_df = efl_data["full_df"]
    print(f"  EFL: {len(efl_df)} rows, seasons "
          f"{efl_df['SeasonIndex'].min()}-{efl_df['SeasonIndex'].max()}")

    # ── Tune per-league shared (O/U 2.5) + per-market kwargs ──
    print("\n[3/3] Tuning DC hyperparameters...")

    print("\n  PL O/U 2.5 (shared baseline + market-specific)...")
    pl_ou25 = tune_dc_params(pl_df, target_col="Over_2_5",
                              predict_fn_name="predict_proba_df")
    print(f"    -> {pl_ou25}")

    print("\n  PL BTTS (market-specific)...")
    pl_btts = tune_dc_params(pl_df, target_col="BTTS",
                              predict_fn_name="predict_proba_btts_df")
    print(f"    -> {pl_btts}")

    print("\n  EFL O/U 2.5 (shared baseline + market-specific)...")
    efl_ou25 = tune_dc_params_champ(efl_df, target="Over_2_5",
                                      predict_fn="predict_proba_df")
    print(f"    -> {efl_ou25}")

    print("\n  EFL O/U 1.5 (market-specific)...")
    efl_ou15 = tune_dc_params_champ(efl_df, target="Over_1_5",
                                      predict_fn="predict_proba_ou15_df")
    print(f"    -> {efl_ou15}")

    print("\n  EFL BTTS (market-specific)...")
    efl_btts = tune_dc_params_champ(efl_df, target="BTTS",
                                      predict_fn="predict_proba_btts_df")
    print(f"    -> {efl_btts}")

    # ── Run diagnostic per market ──
    all_results: dict[str, dict] = {}

    specs_and_kwargs: list[tuple[MarketSpec, pd.DataFrame, dict, dict]] = [
        # (spec, df, shared_kwargs, market_kwargs)
        (MarketSpec("PL", "ou25", "Over_2_5", "predict_proba_df"),
         pl_df, pl_ou25, pl_ou25),  # shared and market are the same for O/U 2.5
        (MarketSpec("PL", "btts", "BTTS", "predict_proba_btts_df"),
         pl_df, pl_ou25, pl_btts),
        (MarketSpec("EFL", "ou25", "Over_2_5", "predict_proba_df"),
         efl_df, efl_ou25, efl_ou25),
        (MarketSpec("EFL", "ou15", "Over_1_5", "predict_proba_ou15_df"),
         efl_df, efl_ou25, efl_ou15),
        (MarketSpec("EFL", "btts", "BTTS", "predict_proba_btts_df"),
         efl_df, efl_ou25, efl_btts),
    ]

    for spec, df, shared, market in specs_and_kwargs:
        key = f"{spec.league}_{spec.market}"
        all_results[key] = diagnose_market(spec, df, shared, market)

    # ── Final summary table ──
    print("\n" + "#" * 70)
    print("#  SUMMARY — mean AUC across walk-forward folds")
    print("#" * 70)
    print(f"\n  {'Market':<12} {'A_base':<10} {'B_step1':<10} {'C_1+2':<10} "
          f"{'dAUC (C-A)':<12}")
    for key, summary in all_results.items():
        a = summary.get("A_baseline", {}).get("auc", float("nan"))
        b = summary.get("B_step1", {}).get("auc", float("nan"))
        c = summary.get("C_steps1_2", {}).get("auc", float("nan"))
        d = c - a if not (np.isnan(a) or np.isnan(c)) else float("nan")
        d_sym = "+" if d > 0 else ""
        print(f"  {key:<12} {a:.4f}     {b:.4f}     {c:.4f}     {d_sym}{d:+.4f}")

    print("\n" + "#" * 70)
    print("#  Interpretation guide:")
    print("#    dAUC > +0.005  = clear uplift from Option 2")
    print("#    dAUC ± 0.002   = noise (no meaningful change)")
    print("#    dAUC < -0.005  = regression (something broken)")
    print("#" * 70)


if __name__ == "__main__":
    main()
