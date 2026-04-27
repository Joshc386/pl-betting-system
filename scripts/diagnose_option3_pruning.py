"""
Option 3 Step 4 (partial): pruning-only A/B comparison.

Runs walk-forward AUC + Brier across all 5 live markets with and
without the feature pruning flag (use_sparse_features). Tells us whether
removing the 62 zero-permutation-importance features improves, hurts, or
has no effect on model quality.

This is an INTERIM diagnostic — the full Option 3 Step 4 diagnostic
(baseline -> pruned -> +new features) will follow once Step 2 additions
land.

Runtime: ~15-20 minutes.
Run:     python scripts/diagnose_option3_pruning.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import (
    ALL_FEATURES, BTTS_ALL_FEATURES, get_active_features,
)
from model import (
    train_xgb, train_lgb, DixonColesPredictor,
)


# =============================================================================
# Local walk-forward evaluator
# =============================================================================

def _run_walk_forward(
    full_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_xgb_fn,
    train_lgb_fn,
    dc_predict_fn: str,
    min_train_season: int = 14,
    start_val_season: int = 19,
    label: str = "",
) -> list[dict]:
    """Run walk-forward CV and return per-fold AUC + Brier.

    Uses the ensemble average (XGB + LGB + DC) / 3 as the prediction.
    """
    all_seasons = sorted(full_df["SeasonIndex"].unique())
    max_season = max(all_seasons)
    fold_metrics: list[dict] = []

    for val_season in range(start_val_season, max_season + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= min_train_season)
                            & (full_df["SeasonIndex"] < val_season)]
        val_df = full_df[full_df["SeasonIndex"] == val_season]
        if len(train_df) < 100 or len(val_df) < 50:
            continue

        # Filter to features present in the DataFrame (defensive)
        present = [f for f in features if f in train_df.columns]

        X_tr = train_df[present].values
        y_tr = train_df[target_col].values
        X_v = val_df[present].values
        y_v = val_df[target_col].values

        if len(np.unique(y_v)) < 2:
            continue

        xgb_m = train_xgb_fn(X_tr, y_tr, X_v, y_v)
        xgb_p = xgb_m.predict_proba(X_v)[:, 1]

        lgb_m = train_lgb_fn(X_tr, y_tr, X_v, y_v, feature_names=present)
        lgb_p = lgb_m.predict_proba(pd.DataFrame(X_v, columns=present))[:, 1]

        dc_m = DixonColesPredictor()  # default rho/half_life for clean A/B
        dc_m.fit(train_df)
        dc_p = getattr(dc_m, dc_predict_fn)(val_df)

        ensemble = (xgb_p + lgb_p + dc_p) / 3.0
        fold_auc = roc_auc_score(y_v, ensemble)
        fold_brier = brier_score_loss(y_v, ensemble)

        fold_metrics.append({
            "val_season": int(val_season),
            "n_train": len(train_df), "n_val": len(val_df),
            "auc": float(fold_auc), "brier": float(fold_brier),
            "n_features": len(present),
        })
        print(f"    {label} S{val_season}: n_feat={len(present)} "
              f"AUC={fold_auc:.4f} Brier={fold_brier:.4f}")

    return fold_metrics


def _ensure_btts_col(df: pd.DataFrame) -> pd.DataFrame:
    """Materialise BTTS column from goals if missing (PL pipeline quirk)."""
    if "BTTS" in df.columns:
        return df
    df = df.copy()
    df["BTTS"] = ((df["Home_Goals"] > 0) & (df["Away_Goals"] > 0)).astype(int)
    return df


# =============================================================================
# Market runners
# =============================================================================

def run_combo(
    league: str,
    market: str,
    full_df: pd.DataFrame,
) -> dict:
    """Run baseline (use_sparse=True) and pruned (use_sparse=False) variants
    for one market; return the comparison result.
    """
    # Resolve features + target + trainers + dc function per combo
    if league == "PL":
        from model import train_xgb as _xgb, train_lgb as _lgb
        from model import train_xgb_btts as _xgb_b, train_lgb_btts as _lgb_b
        if market == "ou25":
            base_features = ALL_FEATURES
            target_col = "Over_2_5"
            xgb_fn, lgb_fn = _xgb, _lgb
            dc_fn = "predict_proba_df"
        elif market == "btts":
            base_features = BTTS_ALL_FEATURES
            target_col = "BTTS"
            xgb_fn, lgb_fn = _xgb_b, _lgb_b
            dc_fn = "predict_proba_btts_df"
        else:
            raise ValueError(f"Unknown PL market: {market}")
    elif league == "EFL":
        from championship_model import (
            train_xgb_champ, train_lgb_champ,
            train_xgb_ou15_champ, train_lgb_ou15_champ,
            train_xgb_btts_champ, train_lgb_btts_champ,
        )
        from championship_pipeline import (
            CHAMP_ALL_FEATURES, CHAMP_OU15_FEATURES, CHAMP_BTTS_FEATURES,
        )
        if market == "ou25":
            base_features = CHAMP_ALL_FEATURES
            target_col = "Over_2_5"
            xgb_fn, lgb_fn = train_xgb_champ, train_lgb_champ
            dc_fn = "predict_proba_df"
        elif market == "ou15":
            base_features = CHAMP_OU15_FEATURES
            target_col = "Over_1_5"
            xgb_fn, lgb_fn = train_xgb_ou15_champ, train_lgb_ou15_champ
            dc_fn = "predict_proba_ou15_df"
        elif market == "btts":
            base_features = CHAMP_BTTS_FEATURES
            target_col = "BTTS"
            xgb_fn, lgb_fn = train_xgb_btts_champ, train_lgb_btts_champ
            dc_fn = "predict_proba_btts_df"
        else:
            raise ValueError(f"Unknown EFL market: {market}")
    else:
        raise ValueError(f"Unknown league: {league}")

    df = _ensure_btts_col(full_df) if market == "btts" else full_df

    baseline_features = get_active_features(base_features, use_sparse=True)
    pruned_features = get_active_features(base_features, use_sparse=False)

    print(f"\n{'='*70}")
    print(f"  {league} / {market}")
    print(f"{'='*70}")
    print(f"  baseline: {len(baseline_features)} features")
    print(f"  pruned:   {len(pruned_features)} features "
          f"(removed {len(baseline_features) - len(pruned_features)})")

    # Pick sensible start_val_season per league
    start_val = 19 if league == "PL" else 19

    print(f"\n  [A] BASELINE (use_sparse=True)")
    base_folds = _run_walk_forward(
        df, baseline_features, target_col,
        xgb_fn, lgb_fn, dc_fn,
        start_val_season=start_val, label="   [A]",
    )

    print(f"\n  [B] PRUNED   (use_sparse=False)")
    pruned_folds = _run_walk_forward(
        df, pruned_features, target_col,
        xgb_fn, lgb_fn, dc_fn,
        start_val_season=start_val, label="   [B]",
    )

    # Aggregate
    def _agg(folds, key):
        vals = [f[key] for f in folds]
        return np.mean(vals) if vals else float("nan")

    base_auc = _agg(base_folds, "auc")
    pruned_auc = _agg(pruned_folds, "auc")
    base_brier = _agg(base_folds, "brier")
    pruned_brier = _agg(pruned_folds, "brier")

    d_auc = pruned_auc - base_auc
    d_brier = pruned_brier - base_brier

    return {
        "league": league, "market": market,
        "n_base_features": len(baseline_features),
        "n_pruned_features": len(pruned_features),
        "base_auc_mean": base_auc,
        "pruned_auc_mean": pruned_auc,
        "base_brier_mean": base_brier,
        "pruned_brier_mean": pruned_brier,
        "delta_auc": d_auc,
        "delta_brier": d_brier,
        "base_folds": base_folds,
        "pruned_folds": pruned_folds,
    }


# =============================================================================
# Entry point
# =============================================================================

def main():
    print("\n" + "#" * 70)
    print("#  Option 3 pruning diagnostic  -  baseline vs use_sparse=False")
    print("#" * 70)

    results: list[dict] = []

    # PL
    print("\n[pipeline] Loading PL...")
    from pipeline import run_pipeline as pl_pipeline
    pl_df = pl_pipeline(verbose=True)["full_df"]
    print(f"[pipeline] PL: {len(pl_df)} rows")

    for market in ("ou25", "btts"):
        results.append(run_combo("PL", market, pl_df))

    # EFL
    print("\n[pipeline] Loading EFL...")
    from championship_pipeline import run_pipeline as efl_pipeline
    efl_df = efl_pipeline(verbose=True)["full_df"]
    print(f"[pipeline] EFL: {len(efl_df)} rows")

    for market in ("ou25", "ou15", "btts"):
        results.append(run_combo("EFL", market, efl_df))

    # Summary
    print("\n" + "#" * 70)
    print("#  SUMMARY - pruning effect per market")
    print("#" * 70)
    print(f"\n  {'Combo':<12} {'nFeat':>10} {'AUC-base':>9} {'AUC-prune':>9} "
          f"{'dAUC':>9} {'Brier-base':>10} {'Brier-prune':>11} {'dBrier':>9}")
    for r in results:
        combo = f"{r['league']}/{r['market']}"
        nfeat = f"{r['n_base_features']}->{r['n_pruned_features']}"
        print(f"  {combo:<12} {nfeat:>10} "
              f"{r['base_auc_mean']:>9.4f} {r['pruned_auc_mean']:>9.4f} "
              f"{r['delta_auc']:>+9.4f} "
              f"{r['base_brier_mean']:>10.4f} {r['pruned_brier_mean']:>11.4f} "
              f"{r['delta_brier']:>+9.4f}")

    # Write results file
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/diagnose_option3_pruning.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Option 3 pruning diagnostic results\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"\n{r['league']}/{r['market']}\n")
            f.write(f"  Baseline features: {r['n_base_features']}\n")
            f.write(f"  Pruned features:   {r['n_pruned_features']}\n")
            f.write(f"  Baseline AUC:   {r['base_auc_mean']:.4f}\n")
            f.write(f"  Pruned AUC:     {r['pruned_auc_mean']:.4f}\n")
            f.write(f"  Delta AUC:      {r['delta_auc']:+.4f}\n")
            f.write(f"  Baseline Brier: {r['base_brier_mean']:.4f}\n")
            f.write(f"  Pruned Brier:   {r['pruned_brier_mean']:.4f}\n")
            f.write(f"  Delta Brier:    {r['delta_brier']:+.4f}\n")
            f.write("  Per-fold (baseline):\n")
            for fold in r["base_folds"]:
                f.write(f"    S{fold['val_season']}: AUC={fold['auc']:.4f} "
                        f"Brier={fold['brier']:.4f}\n")
            f.write("  Per-fold (pruned):\n")
            for fold in r["pruned_folds"]:
                f.write(f"    S{fold['val_season']}: AUC={fold['auc']:.4f} "
                        f"Brier={fold['brier']:.4f}\n")

    print(f"\n  Full results saved to: {out_path}")

    # Interpretation
    print("\n" + "#" * 70)
    print("#  Interpretation:")
    print("#    dAUC > +0.002  AND  dBrier <= 0:  pruning helps - populate SPARSE_FEATURE_GROUPS permanently")
    print("#    |dAUC| <= 0.001 AND |dBrier| <= 0.001: neutral - pruning is safe, decide on simplicity grounds")
    print("#    dAUC < -0.002:  pruning hurts - the groups carried hidden correlated signal")
    print("#" * 70)


if __name__ == "__main__":
    main()
