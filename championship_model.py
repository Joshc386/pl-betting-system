"""
Championship Over/Under & BTTS prediction model.

Architecture mirrors model.py but adapted for Championship:
  1. Walk-forward CV (train on seasons 0..N, validate on N+1)
  2. Three base models: XGBoost, LightGBM, Dixon-Coles Poisson
  3. Logistic regression stacker trained on OOF predictions
  4. Logit-shift calibration from validation data
  5. Three separate ensembles: O/U 2.5, O/U 1.5, BTTS

Key differences from PL model:
  - No xG data for Dixon-Coles (goals-only)
  - ~552 matches/season (24 teams × 46 games / 2) vs PL's 380
  - Different base rates: O/U 2.5 ≈ 47.5%, O/U 1.5 ≈ 73.0%, BTTS ≈ 51.7%
  - Walk-forward starts from season 15 (15 seasons ≈ 8000+ training matches)
  - Championship-specific BTTS hyperparameters (more physical league)

Imports base model trainers and DixonColesPredictor from model.py to avoid
code duplication.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score,
)
from sklearn.linear_model import LogisticRegression

# Import base classes from PL model to avoid duplication
from model import (
    DixonColesPredictor,
    EnsembleModel,
    IsotonicWrapper,
    prune_features,
)
from championship_pipeline import (
    run_pipeline,
    CHAMP_ALL_FEATURES,
    CHAMP_OU15_FEATURES,
    CHAMP_BTTS_FEATURES,
)
from league_config import get_league_config

LEAGUE_CFG = get_league_config("EFL")

# ═══════════════════════════════════════════════════════════════════════════════
# Model directory and paths
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "championship")

# O/U 2.5 model paths
OU25_MODEL_PATH = os.path.join(MODEL_DIR, "champ_ou25_model.pkl")
OU25_FEATURES_PATH = os.path.join(MODEL_DIR, "champ_ou25_features.pkl")

# O/U 1.5 model paths
OU15_MODEL_PATH = os.path.join(MODEL_DIR, "champ_ou15_model.pkl")
OU15_FEATURES_PATH = os.path.join(MODEL_DIR, "champ_ou15_features.pkl")

# BTTS model paths
BTTS_MODEL_PATH = os.path.join(MODEL_DIR, "champ_btts_model.pkl")
BTTS_FEATURES_PATH = os.path.join(MODEL_DIR, "champ_btts_features.pkl")

# Walk-forward parameters — RESEARCH PATH ONLY (ADR 0009)
# These govern this module's main(), not the live EFL models.
# championship_predict.py's train() — what scheduler.py retrains — filters on
# SeasonIndex >= MIN_TRAIN_SEASON and derives its Early-Stopping Season from the
# data; it never reads TEST_SEASON. See CONTEXT.md "Training Path".
#
# Note the partition here is `< TEST_SEASON` / `== TEST_SEASON`, which is
# exhaustive between those two rules but not above them: any season past
# TEST_SEASON falls outside both. At TEST_SEASON=24 with data through 25,
# season 25 was in neither wf_df nor test_df.
MIN_TRAIN_SEASON = 0     # Earliest season (2000/01)
START_VAL_SEASON = 15    # First validation season (~8000 training matches)
TEST_SEASON = 26         # Held-out test (2026/27, in progress)


# ═══════════════════════════════════════════════════════════════════════════════
# Base model training (Championship-tuned hyperparameters)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb_champ(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost for Championship O/U 2.5.

    Slightly deeper trees than PL (more teams/patterns), moderate regularisation.
    Championship has more training data per fold so can afford slightly more complexity.
    """
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        gamma=0.1, reg_alpha=0.5, reg_lambda=3.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_champ(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     feature_names: list[str] | None = None) -> lgb.LGBMClassifier:
    """LightGBM for Championship O/U 2.5."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_xgb_btts_champ(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost for Championship BTTS.

    Championship is more physical/high-tempo than PL — deeper trees to capture
    discipline × defence interactions. BTTS base rate ~51.7% (balanced).
    """
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_btts_champ(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          feature_names: list[str] | None = None) -> lgb.LGBMClassifier:
    """LightGBM for Championship BTTS."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_xgb_ou15_champ(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
    """XGBoost for Championship O/U 1.5.

    O/U 1.5 has a high base rate (~73%) — imbalanced toward Over. Deeper trees
    and lower min_child_weight to capture Under 1.5 patterns (rare, valuable).
    """
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
        gamma=0.1, reg_alpha=0.3, reg_lambda=2.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_ou15_champ(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          feature_names: list[str] | None = None) -> lgb.LGBMClassifier:
    """LightGBM for Championship O/U 1.5."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=2.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Dixon-Coles hyperparameter tuning (Championship-adapted)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_dc_params_champ(full_df: pd.DataFrame, target: str = "Over_2_5",
                          predict_fn: str = "predict_proba_df") -> dict:
    """Grid-search half_life and rho for Championship Dixon-Coles.

    Championship has 64 teams (vs PL's 20), making MLE optimisation impractical
    (259 parameters, fails to converge). Uses weighted-average only.

    Args:
        full_df: Full pipeline DataFrame.
        target: Target column name.
        predict_fn: DixonColesPredictor method name for predictions.

    Returns:
        Dict of best params for DixonColesPredictor constructor.
    """
    # half_life=10 added at low end for markets that favour very recent form
    # (BTTS, O/U 1.5 tend to respond to short-term team defensive/scoring streaks)
    half_life_grid = [10, 15, 20, 25, 30, 40, 50, 70]
    rho_grid = [-0.20, -0.15, -0.13, -0.10, -0.07, -0.03, 0.0]

    val_seasons = list(range(START_VAL_SEASON, TEST_SEASON))

    # Pre-split folds
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for vs in val_seasons:
        train = full_df[(full_df["SeasonIndex"] >= MIN_TRAIN_SEASON) &
                        (full_df["SeasonIndex"] < vs)]
        val = full_df[full_df["SeasonIndex"] == vs]
        if len(train) >= 100 and len(val) >= 50:
            folds.append((train, val))

    if not folds:
        print("  [DC tune] Not enough folds, using defaults")
        return {"half_life": 30, "rho": -0.13}

    best_auc = -1.0
    best_params: dict = {"half_life": 30, "rho": -0.13}
    results: list[tuple] = []

    for hl in half_life_grid:
        for rho in rho_grid:
            fold_aucs: list[float] = []
            for train_df, val_df in folds:
                dc = DixonColesPredictor(rho=rho, half_life=hl)
                dc.fit(train_df)
                preds = getattr(dc, predict_fn)(val_df)
                y = val_df[target].values
                if len(np.unique(y)) < 2:
                    continue
                fold_aucs.append(roc_auc_score(y, preds))

            if fold_aucs:
                mean_auc = float(np.mean(fold_aucs))
                results.append((hl, rho, mean_auc))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"half_life": hl, "rho": rho}

    print(f"\n{'='*60}")
    print(f"Dixon-Coles Tuning (Championship, target={target})")
    print(f"{'='*60}")
    print(f"  Grid: {len(half_life_grid)}x{len(rho_grid)} = {len(results)} combos, {len(folds)} folds")
    print(f"  Top 5:")
    for hl, rho, auc in sorted(results, key=lambda x: -x[2])[:5]:
        marker = " <-- BEST" if hl == best_params["half_life"] and rho == best_params["rho"] else ""
        print(f"    half_life={hl:3d}  rho={rho:+.2f}  AUC={auc:.4f}{marker}")

    # Skip MLE for Championship: 64 teams = 259 parameters, L-BFGS-B can't converge.
    # Weighted-average DC is sufficient as one component of the 3-model ensemble.
    print(f"  (MLE skipped: {len(set(full_df['Home_Team']))} teams makes MLE impractical)")

    final = {"half_life": best_params["half_life"], "rho": best_params["rho"]}
    print(f"\n  Final: {final}, AUC={best_auc:.4f}")
    print(f"{'='*60}\n")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(full_df: pd.DataFrame, features: list[str],
                     target: str, train_xgb_fn, train_lgb_fn,
                     dc_kwargs: dict, dc_predict_fn: str = "predict_proba_df",
                     ) -> tuple[list[dict], pd.DataFrame]:
    """Walk-forward CV for Championship.

    Trains on seasons MIN_TRAIN_SEASON..N, validates on N+1.
    Returns fold metrics and OOF predictions for stacker training.

    Args:
        full_df: Full pipeline DataFrame.
        features: Feature column names.
        target: Target column name (Over_2_5, Over_1_5, BTTS).
        train_xgb_fn: XGBoost training function.
        train_lgb_fn: LightGBM training function.
        dc_kwargs: Dixon-Coles constructor kwargs.
        dc_predict_fn: Method name on DixonColesPredictor for predictions.

    Returns:
        Tuple of (fold_metrics, oof_df).
    """
    all_seasons = sorted(full_df["SeasonIndex"].unique())
    fold_metrics: list[dict] = []
    oof_records: list[dict] = []

    for val_season in range(START_VAL_SEASON, TEST_SEASON):
        train_df = full_df[(full_df["SeasonIndex"] >= MIN_TRAIN_SEASON) &
                           (full_df["SeasonIndex"] < val_season)].copy()
        val_df = full_df[full_df["SeasonIndex"] == val_season].copy()

        if len(train_df) < 100 or len(val_df) < 50:
            continue

        X_tr = train_df[features].values
        y_tr = train_df[target].values
        X_v = val_df[features].values
        y_v = val_df[target].values

        # XGBoost
        xgb_m = train_xgb_fn(X_tr, y_tr, X_v, y_v)
        xgb_p = xgb_m.predict_proba(X_v)[:, 1]

        # LightGBM
        lgb_m = train_lgb_fn(X_tr, y_tr, X_v, y_v, feature_names=features)
        lgb_p = lgb_m.predict_proba(pd.DataFrame(X_v, columns=features))[:, 1]

        # Dixon-Coles
        dc_m = DixonColesPredictor(**dc_kwargs)
        dc_m.fit(train_df)
        dc_p = getattr(dc_m, dc_predict_fn)(val_df)

        # Average for fold evaluation
        avg_p = (xgb_p + lgb_p + dc_p) / 3
        fold_auc = roc_auc_score(y_v, avg_p)
        fold_acc = accuracy_score(y_v, (avg_p > 0.5).astype(int))
        fold_brier = brier_score_loss(y_v, avg_p)

        fold_metrics.append({
            "val_season": val_season, "n_train": len(train_df),
            "n_val": len(val_df), "auc": fold_auc, "acc": fold_acc,
            "brier": fold_brier,
        })
        print(f"  Fold S{val_season}: train={len(train_df):5d} val={len(val_df):3d} "
              f"AUC={fold_auc:.4f} Acc={fold_acc:.4f} Brier={fold_brier:.4f}")

        for i, idx in enumerate(val_df.index):
            oof_records.append({
                "idx": idx, "season": val_season,
                "xgb": xgb_p[i], "lgb": lgb_p[i], "dc": dc_p[i],
                "y": y_v[i],
            })

    oof_df = pd.DataFrame(oof_records)
    return fold_metrics, oof_df


# ═══════════════════════════════════════════════════════════════════════════════
# Model saving/loading
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(model: EnsembleModel, features: list[str],
               model_path: str, features_path: str) -> None:
    """Save ensemble model and feature list."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(features, features_path)
    print(f"  Model saved to {model_path}")


def load_model(model_path: str, features_path: str) -> tuple:
    """Load ensemble model and feature list."""
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features


# ═══════════════════════════════════════════════════════════════════════════════
# Single market training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def train_market(full_df: pd.DataFrame, features: list[str],
                  target: str, market_name: str,
                  train_xgb_fn, train_lgb_fn,
                  dc_kwargs: dict,
                  dc_predict_fn: str = "predict_proba_df",
                  model_path: str | None = None,
                  features_path: str | None = None,
                  ) -> tuple[EnsembleModel, list[str], dict]:
    """Train a complete ensemble for one market (O/U 2.5, O/U 1.5, or BTTS).

    Phases:
      1. Tune Dixon-Coles hyperparameters
      2. Walk-forward CV (OOF predictions for stacker)
      3. Feature pruning via initial XGBoost
      4. Train final base models on train+val
      5. Train stacker on walk-forward OOF predictions
      6. Logit-shift calibration
      7. Evaluate on held-out test season
      8. Save

    Args:
        full_df: Full pipeline DataFrame.
        features: Feature column names.
        target: Target column name.
        market_name: Display name (e.g. "O/U 2.5").
        train_xgb_fn: XGBoost training function.
        train_lgb_fn: LightGBM training function.
        dc_kwargs: Dixon-Coles constructor kwargs (or empty for auto-tune).
        dc_predict_fn: DC prediction method name.
        model_path: Path to save model pickle.
        features_path: Path to save feature list pickle.

    Returns:
        Tuple of (ensemble, features, test_metrics).
    """
    print(f"\n{'#'*70}")
    print(f"# CHAMPIONSHIP {market_name} MODEL")
    print(f"{'#'*70}")

    base_rate = full_df[target].mean()
    print(f"\n  Base rate ({target}): {base_rate:.3f}")
    print(f"  Total matches: {len(full_df)}")
    print(f"  Initial features: {len(features)}")

    # Exclude test season from tuning/CV
    wf_df = full_df[full_df["SeasonIndex"] < TEST_SEASON].copy()
    test_df = full_df[full_df["SeasonIndex"] == TEST_SEASON].copy()
    print(f"  Train+Val: {len(wf_df)} matches (seasons {MIN_TRAIN_SEASON}-{TEST_SEASON - 1})")
    print(f"  Test: {len(test_df)} matches (season {TEST_SEASON})")

    if len(test_df) < 50:
        print(f"  WARNING: Test set only has {len(test_df)} matches")

    # ── Phase 1: Tune Dixon-Coles ──
    print(f"\n{'='*60}")
    print(f"Phase 1: Tuning Dixon-Coles ({market_name})")
    print(f"{'='*60}")

    if not dc_kwargs:
        dc_kwargs = tune_dc_params_champ(wf_df, target=target, predict_fn=dc_predict_fn)

    # ── Phase 2: Walk-forward CV ──
    print(f"\n{'='*60}")
    print(f"Phase 2: Walk-forward CV ({market_name})")
    print(f"{'='*60}")

    fold_metrics, oof_df = walk_forward_cv(
        wf_df, features, target,
        train_xgb_fn, train_lgb_fn,
        dc_kwargs, dc_predict_fn,
    )

    if fold_metrics:
        avg_auc = np.mean([f["auc"] for f in fold_metrics])
        avg_brier = np.mean([f["brier"] for f in fold_metrics])
        print(f"\n  Walk-forward summary ({len(fold_metrics)} folds):")
        print(f"    Avg AUC:   {avg_auc:.4f}")
        print(f"    Avg Brier: {avg_brier:.4f}")

    # ── Phase 3: Feature pruning ──
    print(f"\n{'='*60}")
    print(f"Phase 3: Feature pruning ({market_name})")
    print(f"{'='*60}")

    # Use second-to-last season as val for early stopping + pruning
    prune_val_season = TEST_SEASON - 1
    prune_train = wf_df[wf_df["SeasonIndex"] < prune_val_season]
    prune_val = wf_df[wf_df["SeasonIndex"] == prune_val_season]

    X_ptr = prune_train[features].values
    y_ptr = prune_train[target].values
    X_pv = prune_val[features].values
    y_pv = prune_val[target].values

    xgb_prune = train_xgb_fn(X_ptr, y_ptr, X_pv, y_pv)
    kept_features, dropped = prune_features(xgb_prune, features)
    if dropped:
        features = kept_features
        print(f"  Pruned to {len(features)} features")
    else:
        print(f"  No features pruned, keeping all {len(features)}")

    # ── Phase 4: Train final base models on all non-test data ──
    print(f"\n{'='*60}")
    print(f"Phase 4: Final base models ({market_name})")
    print(f"{'='*60}")

    # Split for early stopping: train on seasons 0..22, val on season 23
    final_val_season = TEST_SEASON - 1
    train_only = wf_df[wf_df["SeasonIndex"] < final_val_season]
    val_only = wf_df[wf_df["SeasonIndex"] == final_val_season]

    X_train = train_only[features].values
    y_train = train_only[target].values
    X_val = val_only[features].values
    y_val = val_only[target].values

    # XGBoost: early stopping then retrain on full train+val
    print(f"\n  Training XGBoost...")
    xgb_temp = train_xgb_fn(X_train, y_train, X_val, y_val)
    best_xgb_trees = xgb_temp.best_iteration
    print(f"    Best iteration: {best_xgb_trees}")

    X_all = wf_df[features].values
    y_all = wf_df[target].values

    # Get the hyperparams from the training function (match what train_xgb_fn uses)
    xgb_params = xgb_temp.get_params()
    xgb_model = xgb.XGBClassifier(
        n_estimators=best_xgb_trees,
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        min_child_weight=xgb_params["min_child_weight"],
        gamma=xgb_params["gamma"],
        reg_alpha=xgb_params["reg_alpha"],
        reg_lambda=xgb_params["reg_lambda"],
        eval_metric="logloss", random_state=42,
    )
    xgb_model.fit(X_all, y_all, verbose=False)
    print(f"    Retrained on {len(wf_df)} matches with {best_xgb_trees} trees")

    # LightGBM: same approach
    print(f"\n  Training LightGBM...")
    lgb_temp = train_lgb_fn(X_train, y_train, X_val, y_val, feature_names=features)
    best_lgb_trees = lgb_temp.best_iteration_
    print(f"    Best iteration: {best_lgb_trees}")

    lgb_params = lgb_temp.get_params()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=best_lgb_trees,
        max_depth=lgb_params["max_depth"],
        learning_rate=lgb_params["learning_rate"],
        subsample=lgb_params["subsample"],
        colsample_bytree=lgb_params["colsample_bytree"],
        min_child_weight=lgb_params["min_child_weight"],
        reg_alpha=lgb_params["reg_alpha"],
        reg_lambda=lgb_params["reg_lambda"],
        random_state=42, verbose=-1,
    )
    lgb_model.fit(pd.DataFrame(X_all, columns=features), y_all)
    print(f"    Retrained on {len(wf_df)} matches with {best_lgb_trees} trees")

    # Dixon-Coles on full train+val
    print(f"\n  Training Dixon-Coles (kwargs={dc_kwargs})...")
    dc_model = DixonColesPredictor(**dc_kwargs)
    dc_model.fit(wf_df)

    # ── Phase 5: Stacker ──
    print(f"\n{'='*60}")
    print(f"Phase 5: Stacker ({market_name})")
    print(f"{'='*60}")

    if len(oof_df) > 100:
        print(f"\n  Using {len(oof_df)} walk-forward OOF predictions")
        oof_stack = oof_df[["xgb", "lgb", "dc"]].values
        oof_y = oof_df["y"].values

        stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        stacker.fit(oof_stack, oof_y)

        print(f"  Stacker coefficients: XGB={stacker.coef_[0][0]:.3f}, "
              f"LGB={stacker.coef_[0][1]:.3f}, DC={stacker.coef_[0][2]:.3f}")
        print(f"  Stacker intercept: {stacker.intercept_[0]:.3f}")

        oof_stacker_p = stacker.predict_proba(oof_stack)[:, 1]
        oof_auc = roc_auc_score(oof_y, oof_stacker_p)
        print(f"  OOF stacker AUC: {oof_auc:.4f}")
    else:
        print("  WARNING: Insufficient OOF data, stacker will be None")
        stacker = None
        oof_stack = None
        oof_y = None

    # ── Phase 6: Calibration ──
    print(f"\n{'='*60}")
    print(f"Phase 6: Calibration ({market_name})")
    print(f"{'='*60}")

    xgb_val_p = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_p = lgb_model.predict_proba(pd.DataFrame(X_val, columns=features))[:, 1]
    dc_val_p = getattr(dc_model, dc_predict_fn)(val_only)

    if stacker is not None:
        val_stack = np.column_stack([xgb_val_p, lgb_val_p, dc_val_p])
        stacker_val_raw = stacker.predict_proba(val_stack)[:, 1]
    else:
        stacker_val_raw = (xgb_val_p + lgb_val_p + dc_val_p) / 3

    val_mean_logit = float(np.mean(np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10))))
    train_base_rate = float(y_all.mean())
    target_logit = float(np.log(train_base_rate / (1 - train_base_rate + 1e-10)))
    logit_shift = val_mean_logit - target_logit

    print(f"\n  Val predictions: mean={stacker_val_raw.mean():.4f}, actual={y_val.mean():.4f}")
    print(f"  Target base rate: {train_base_rate:.4f}")
    print(f"  Logit shift: {logit_shift:.4f}")

    # Verify
    val_logits = np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10))
    val_corrected = 1 / (1 + np.exp(-(val_logits - logit_shift)))
    print(f"  Val corrected mean: {val_corrected.mean():.4f}")
    print(f"  Val Brier: before={brier_score_loss(y_val, stacker_val_raw):.4f}, "
          f"after={brier_score_loss(y_val, val_corrected):.4f}")

    # ── Build ensemble ──
    ensemble = EnsembleModel(
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        dc_model=dc_model,
        stacker=stacker,
        feature_names=features,
        threshold=0.50,
        platt_scaler=logit_shift,
    )

    # ── Phase 7: Test evaluation ──
    print(f"\n{'='*60}")
    print(f"Phase 7: Test evaluation ({market_name})")
    print(f"{'='*60}")

    test_metrics: dict = {}
    if len(test_df) >= 20:
        X_test = test_df[features].values
        y_test = test_df[target].values

        xgb_test_p = xgb_model.predict_proba(X_test)[:, 1]
        lgb_test_p = lgb_model.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
        dc_test_p = getattr(dc_model, dc_predict_fn)(test_df)

        print(f"\n  Individual models:")
        for name, probs in [("XGBoost", xgb_test_p), ("LightGBM", lgb_test_p),
                             ("Dixon-Coles", dc_test_p)]:
            auc = roc_auc_score(y_test, probs)
            brier = brier_score_loss(y_test, probs)
            print(f"    {name:12s}: AUC={auc:.4f}  Brier={brier:.4f}")

        # Stacker ensemble
        ensemble_p = ensemble.predict_proba(X_test, dc_probs=dc_test_p)[:, 1]
        test_auc = roc_auc_score(y_test, ensemble_p)
        test_brier = brier_score_loss(y_test, ensemble_p)
        test_acc = accuracy_score(y_test, (ensemble_p >= 0.5).astype(int))

        print(f"\n  Ensemble (calibrated):")
        print(f"    AUC:   {test_auc:.4f}")
        print(f"    Brier: {test_brier:.4f}")
        print(f"    Acc:   {test_acc:.4f}")
        print(f"    Pred mean: {ensemble_p.mean():.4f}, Actual: {y_test.mean():.4f}")

        # Overfitting diagnostic
        if fold_metrics:
            wf_avg_auc = np.mean([f["auc"] for f in fold_metrics])
            print(f"\n  Walk-forward AUC: {wf_avg_auc:.4f} vs Test AUC: {test_auc:.4f} "
                  f"(gap: {abs(wf_avg_auc - test_auc):.4f})")

        test_metrics = {
            "auc": test_auc, "brier": test_brier, "accuracy": test_acc,
            "log_loss": log_loss(y_test, ensemble_p),
        }
    else:
        print("  Skipping test evaluation (insufficient data)")

    # ── Save ──
    if model_path and features_path:
        save_model(ensemble, features, model_path, features_path)

    return ensemble, features, test_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main(markets: list[str] | None = None) -> dict:
    """Train Championship models for specified markets.

    Args:
        markets: List of markets to train. Options: "ou25", "ou15", "btts".
                 If None, trains all three.

    Returns:
        Dict of {market_name: (ensemble, features, test_metrics)}.
    """
    if markets is None:
        markets = ["ou25", "ou15", "btts"]

    print("=" * 70)
    print("CHAMPIONSHIP MODEL TRAINING")
    print("=" * 70)

    # Run pipeline
    result = run_pipeline(verbose=True)
    full_df = result["full_df"]
    ou25_features = list(result["features"])
    ou15_features = list(result["ou15_features"])
    btts_features = list(result["btts_features"])

    results: dict = {}

    # ── O/U 2.5 ──
    if "ou25" in markets:
        ensemble, feats, metrics = train_market(
            full_df, ou25_features, "Over_2_5", "O/U 2.5",
            train_xgb_champ, train_lgb_champ,
            dc_kwargs={},  # auto-tune
            dc_predict_fn="predict_proba_df",
            model_path=OU25_MODEL_PATH,
            features_path=OU25_FEATURES_PATH,
        )
        results["ou25"] = (ensemble, feats, metrics)

    # ── O/U 1.5 ──
    if "ou15" in markets:
        # Dixon-Coles O/U 1.5: need a custom predict function
        # DixonColesPredictor.predict_match() computes P(Over 2.5)
        # For O/U 1.5, we need P(Over 1.5) which is different.
        # We'll tune DC params separately for O/U 1.5 using a wrapper.
        ensemble, feats, metrics = train_market(
            full_df, ou15_features, "Over_1_5", "O/U 1.5",
            train_xgb_ou15_champ, train_lgb_ou15_champ,
            dc_kwargs={},  # auto-tune
            dc_predict_fn="predict_proba_ou15_df",
            model_path=OU15_MODEL_PATH,
            features_path=OU15_FEATURES_PATH,
        )
        results["ou15"] = (ensemble, feats, metrics)

    # ── BTTS ──
    if "btts" in markets:
        ensemble, feats, metrics = train_market(
            full_df, btts_features, "BTTS", "BTTS",
            train_xgb_btts_champ, train_lgb_btts_champ,
            dc_kwargs={},  # auto-tune
            dc_predict_fn="predict_proba_btts_df",
            model_path=BTTS_MODEL_PATH,
            features_path=BTTS_FEATURES_PATH,
        )
        results["btts"] = (ensemble, feats, metrics)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE — SUMMARY")
    print(f"{'='*70}")
    for market, (_, feats, metrics) in results.items():
        auc = metrics.get("auc", "N/A")
        brier = metrics.get("brier", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
        brier_str = f"{brier:.4f}" if isinstance(brier, float) else brier
        print(f"  {market:8s}: {len(feats):3d} features, Test AUC={auc_str}, Brier={brier_str}")

    return results


if __name__ == "__main__":
    # Parse command-line args for which markets to train
    args = sys.argv[1:]
    if args:
        valid = {"ou25", "ou15", "btts"}
        markets = [a for a in args if a in valid]
        if not markets:
            print(f"Usage: python championship_model.py [ou25] [ou15] [btts]")
            sys.exit(1)
    else:
        markets = None  # Train all

    main(markets=markets)
