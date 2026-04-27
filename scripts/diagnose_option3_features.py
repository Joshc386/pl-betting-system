"""
Option 3 Step 4: feature-addition A/B comparison.

For each market:
  Variant A (pre-Option-3): feature list stripped of 2a/2b/2c additions
  Variant B (post-Option-3): current feature list with all 2a/2b/2c

Runs walk-forward CV on both, reports per-market dAUC + dBrier.

Runtime: ~20-25 minutes.
Run:     python scripts/diagnose_option3_features.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from config import ALL_FEATURES, BTTS_ALL_FEATURES
from model import train_xgb, train_lgb, DixonColesPredictor


# =============================================================================
# The exact set of features added in Option 3 Step 2a/2b/2c — used to
# *strip back* a given feature list to its pre-Option-3 state.
# =============================================================================

_OPTION3_NEW_FEATURES_PL = {
    # 2a: Corner efficiency (PL has both _5 and _10)
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    "Home_CornerEfficiency_10", "Away_CornerEfficiency_10",
    # 2b: Set-play xG ratio (PL only)
    "Home_SetPieceXG_Ratio_8", "Away_SetPieceXG_Ratio_8",
    # 2c: _3 rolling windows
    "Home_Over25_3", "Away_Over25_3",
    "Home_BTTS_3", "Away_BTTS_3",
    "Home_TGAvg_3", "Away_TGAvg_3",
    "Home_Past3Goals", "Away_Past3Goals",
    "Home_CornersAvg_3", "Away_CornersAvg_3",
}

_OPTION3_NEW_FEATURES_EFL = {
    # 2a: Corner efficiency (EFL only has _5)
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    # 2b: not applicable — EFL lacks OpenPlayXG/SetPieceXG components
    # 2c: _3 rolling windows
    "Home_Over25_3", "Away_Over25_3",
    "Home_BTTS_3", "Away_BTTS_3",
    "Home_TGAvg_3", "Away_TGAvg_3",
    "Home_Past3Goals", "Away_Past3Goals",
    "Home_CornersAvg_3", "Away_CornersAvg_3",
}


def _strip_option3(features: list[str], league: str) -> list[str]:
    """Return a feature list with all Option 3 additions removed."""
    to_remove = (_OPTION3_NEW_FEATURES_PL if league == "PL"
                 else _OPTION3_NEW_FEATURES_EFL)
    return [f for f in features if f not in to_remove]


# =============================================================================
# Local walk-forward evaluator — reused pattern from Option 2/3 diagnostics
# =============================================================================

def _run_walk_forward(
    full_df: pd.DataFrame,
    features: list[str],
    target_col: str,
    train_xgb_fn,
    train_lgb_fn,
    dc_predict_fn: str,
    label: str,
    min_train_season: int = 14,
    start_val_season: int = 19,
) -> list[dict]:
    """Run walk-forward CV and return per-fold AUC + Brier.

    Uses (XGB + LGB + DC) / 3 simple-average ensemble as the prediction.
    DC is constant across variants (no feature input) so it doesn't
    contaminate the comparison.
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

        dc_m = DixonColesPredictor()
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
    """Materialise BTTS column from goals if missing (PL quirk)."""
    if "BTTS" in df.columns:
        return df
    df = df.copy()
    df["BTTS"] = ((df["Home_Goals"] > 0) & (df["Away_Goals"] > 0)).astype(int)
    return df


# =============================================================================
# Market runners
# =============================================================================

def run_combo(league: str, market: str, full_df: pd.DataFrame) -> dict:
    """Run variant A (no Option 3) and B (with Option 3) for one market."""
    if league == "PL":
        from model import (
            train_xgb as _xgb, train_lgb as _lgb,
            train_xgb_btts as _xgb_b, train_lgb_btts as _lgb_b,
        )
        if market == "ou25":
            current_features = ALL_FEATURES
            target_col = "Over_2_5"
            xgb_fn, lgb_fn = _xgb, _lgb
            dc_fn = "predict_proba_df"
        elif market == "btts":
            current_features = BTTS_ALL_FEATURES
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
            current_features = CHAMP_ALL_FEATURES
            target_col = "Over_2_5"
            xgb_fn, lgb_fn = train_xgb_champ, train_lgb_champ
            dc_fn = "predict_proba_df"
        elif market == "ou15":
            current_features = CHAMP_OU15_FEATURES
            target_col = "Over_1_5"
            xgb_fn, lgb_fn = train_xgb_ou15_champ, train_lgb_ou15_champ
            dc_fn = "predict_proba_ou15_df"
        elif market == "btts":
            current_features = CHAMP_BTTS_FEATURES
            target_col = "BTTS"
            xgb_fn, lgb_fn = train_xgb_btts_champ, train_lgb_btts_champ
            dc_fn = "predict_proba_btts_df"
        else:
            raise ValueError(f"Unknown EFL market: {market}")
    else:
        raise ValueError(f"Unknown league: {league}")

    df = _ensure_btts_col(full_df) if market == "btts" else full_df

    post_features = list(current_features)
    pre_features = _strip_option3(current_features, league)

    print(f"\n{'='*70}")
    print(f"  {league} / {market}")
    print(f"{'='*70}")
    print(f"  pre-Option-3:  {len(pre_features)} features")
    print(f"  post-Option-3: {len(post_features)} features "
          f"(+{len(post_features) - len(pre_features)})")

    start_val = 19

    print(f"\n  [A] PRE-Option-3 (stripped)")
    pre_folds = _run_walk_forward(
        df, pre_features, target_col, xgb_fn, lgb_fn, dc_fn,
        label="   [A]", start_val_season=start_val,
    )

    print(f"\n  [B] POST-Option-3 (current)")
    post_folds = _run_walk_forward(
        df, post_features, target_col, xgb_fn, lgb_fn, dc_fn,
        label="   [B]", start_val_season=start_val,
    )

    def _agg(folds, key):
        vals = [f[key] for f in folds]
        return np.mean(vals) if vals else float("nan")

    pre_auc = _agg(pre_folds, "auc")
    post_auc = _agg(post_folds, "auc")
    pre_brier = _agg(pre_folds, "brier")
    post_brier = _agg(post_folds, "brier")

    return {
        "league": league, "market": market,
        "n_pre": len(pre_features), "n_post": len(post_features),
        "pre_auc": pre_auc, "post_auc": post_auc,
        "pre_brier": pre_brier, "post_brier": post_brier,
        "delta_auc": post_auc - pre_auc,
        "delta_brier": post_brier - pre_brier,
        "pre_folds": pre_folds,
        "post_folds": post_folds,
    }


def main():
    print("\n" + "#" * 70)
    print("#  Option 3 Step 4: feature-addition A/B")
    print("#  Variant A = pre-Option-3 features")
    print("#  Variant B = current (with 2a corner efficiency + 2b set-play")
    print("#             xG ratio + 2c _3 rolling windows)")
    print("#" * 70)

    results: list[dict] = []

    print("\n[pipeline] Loading PL...")
    from pipeline import run_pipeline as pl_pipeline
    pl_df = pl_pipeline(verbose=True)["full_df"]
    print(f"[pipeline] PL: {len(pl_df)} rows")

    for market in ("ou25", "btts"):
        results.append(run_combo("PL", market, pl_df))

    print("\n[pipeline] Loading EFL...")
    from championship_pipeline import run_pipeline as efl_pipeline
    efl_df = efl_pipeline(verbose=True)["full_df"]
    print(f"[pipeline] EFL: {len(efl_df)} rows")

    for market in ("ou25", "ou15", "btts"):
        results.append(run_combo("EFL", market, efl_df))

    # Summary
    print("\n" + "#" * 70)
    print("#  SUMMARY - Option 3 feature additions")
    print("#" * 70)
    print(f"\n  {'Combo':<12} {'nFeat':>10} {'AUC-pre':>8} {'AUC-post':>9} "
          f"{'dAUC':>9} {'Brier-pre':>10} {'Brier-post':>11} {'dBrier':>9}")
    for r in results:
        combo = f"{r['league']}/{r['market']}"
        nfeat = f"{r['n_pre']}->{r['n_post']}"
        print(f"  {combo:<12} {nfeat:>10} "
              f"{r['pre_auc']:>8.4f} {r['post_auc']:>9.4f} "
              f"{r['delta_auc']:>+9.4f} "
              f"{r['pre_brier']:>10.4f} {r['post_brier']:>11.4f} "
              f"{r['delta_brier']:>+9.4f}")

    # Persist
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/diagnose_option3_features.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Option 3 feature-addition diagnostic results\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"\n{r['league']}/{r['market']}\n")
            f.write(f"  Pre-Option-3 features:  {r['n_pre']}\n")
            f.write(f"  Post-Option-3 features: {r['n_post']}\n")
            f.write(f"  Pre  AUC:   {r['pre_auc']:.4f}\n")
            f.write(f"  Post AUC:   {r['post_auc']:.4f}\n")
            f.write(f"  Delta AUC:  {r['delta_auc']:+.4f}\n")
            f.write(f"  Pre  Brier: {r['pre_brier']:.4f}\n")
            f.write(f"  Post Brier: {r['post_brier']:.4f}\n")
            f.write(f"  Delta Brier:{r['delta_brier']:+.4f}\n")
            f.write("  Per-fold (pre-Option-3):\n")
            for fold in r["pre_folds"]:
                f.write(f"    S{fold['val_season']}: AUC={fold['auc']:.4f} "
                        f"Brier={fold['brier']:.4f}\n")
            f.write("  Per-fold (post-Option-3):\n")
            for fold in r["post_folds"]:
                f.write(f"    S{fold['val_season']}: AUC={fold['auc']:.4f} "
                        f"Brier={fold['brier']:.4f}\n")
    print(f"\n  Full results saved to: {out_path}")

    print("\n" + "#" * 70)
    print("#  Interpretation:")
    print("#    any market dAUC > +0.003     -> features help, keep them")
    print("#    all markets |dAUC| <= 0.001  -> neutral, keep (no harm)")
    print("#    any market dAUC < -0.002     -> feature adds hurt; consider revert")
    print("#" * 70)


if __name__ == "__main__":
    main()
