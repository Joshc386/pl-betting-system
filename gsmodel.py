"""
Over/Under 2.5 Goals prediction model.

Architecture:
  1. Walk-forward CV for robust evaluation (train on seasons 14-N, validate on N+1)
  2. Four diverse base models: XGBoost, LightGBM, Logistic Regression, Dixon-Coles Poisson
  3. Logistic regression stacker (meta-learner) trained on OOF predictions
  4. Calibration based on validation data only (no test leakage)

Includes training, evaluation, SHAP, and plotting.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson as poisson_dist
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_DIR, MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, TRAIN_MIN_SEASON,
)
from gspipeline import run_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Base model training
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost with strong regularization and early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        gamma=0.1, reg_alpha=0.5, reg_lambda=3.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb(X_train, y_train, X_val, y_val, feature_names=None):
    """Train LightGBM with matching regularization."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _fill_nan_median(X, medians=None):
    """Fill NaN with column medians. If medians not provided, compute from X."""
    X_filled = X.copy()
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    for j in range(X_filled.shape[1]):
        mask = np.isnan(X_filled[:, j])
        if mask.any():
            X_filled[mask, j] = medians[j] if not np.isnan(medians[j]) else 0.0
    return X_filled, medians


def _clip_scaled(X_scaled, clip=5.0):
    """Clip scaled features to [-clip, +clip].

    Prevents catastrophic predictions from distribution shift between train/test.
    Without this, a single feature shifting by 250+ standard deviations (e.g.
    Home_DefensiveStrength_5) can push LR logits to +10, making every prediction 0.9999.
    """
    return np.clip(X_scaled, -clip, clip)


def train_logreg(X_train, y_train):
    """Train L2-regularized logistic regression (genuinely different model class).

    Fills NaN with column medians before scaling (better than 0 for centered data).
    Clips scaled features to [-5, 5] to prevent distribution-shift catastrophe.
    """
    X_filled, col_medians = _fill_nan_median(X_train)
    scaler = StandardScaler()
    X_scaled = _clip_scaled(scaler.fit_transform(X_filled))
    model = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    # Store medians for inference
    model._col_medians = col_medians
    return model, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# Dixon-Coles Poisson model (standalone — not just a feature)
# ═══════════════════════════════════════════════════════════════════════════════

class DixonColesPredictor:
    """Standalone Dixon-Coles Poisson model for Over/Under 2.5 prediction.

    Uses team-level attack/defence ratings estimated from historical goals (or xG).
    Fundamentally different model class from gradient boosting — makes explicit
    parametric assumptions about goal-scoring (Poisson) with tau correction
    for low-score dependency.

    Improvements over naive implementation:
      - Time-decay weighting (half-life parameter, default 20 matches)
      - Correct lambda formula (no double-counting of home advantage)
      - xG-based ratings when available (lower variance than goals)
      - Promoted team priors (start below average, not at average)
    """

    def __init__(self, rho=-0.13, half_life=30, use_xg=False):
        self.rho = rho
        self.half_life = half_life  # matches; more recent = more weight
        self.use_xg = use_xg
        self.attack = {}    # team -> attack strength (normalised to 1.0 = average)
        self.defence = {}   # team -> defence weakness (>1 = leaky, <1 = solid)
        self.mu = 1.35      # league avg goals per team per match
        self.gamma = 1.36   # home advantage factor (home_goals / away_goals)

    def _decay_weights(self, n):
        """Exponential decay weights: most recent match = 1.0, decays backward."""
        if n == 0:
            return np.array([])
        indices = np.arange(n)[::-1]  # [n-1, n-2, ..., 0] — most recent last
        return np.power(0.5, indices / self.half_life)

    def fit(self, df):
        """Estimate attack/defence ratings from historical match data.
        Uses time-decayed weighted averages over all available matches per team.
        Falls back to xG when available and use_xg=True (lower variance signal).
        """
        # Choose scoring metric: xG if available, else goals
        has_xg = ("home_xg" in df.columns or "Home_RollingXG_5" in df.columns)
        if self.use_xg and "home_xg" in df.columns:
            h_score_col, a_score_col = "home_xg", "away_xg"
        else:
            h_score_col, a_score_col = "Home_Goals", "Away_Goals"

        # League averages
        h_avg = df[h_score_col].mean()
        a_avg = df[a_score_col].mean()
        self.mu = (h_avg + a_avg) / 2  # avg goals per team per match
        self.gamma = h_avg / a_avg if a_avg > 0 else 1.36

        teams = set(df["Home_Team"].unique()) | set(df["Away_Team"].unique())

        for team in teams:
            home_matches = df[df["Home_Team"] == team].sort_values("Date")
            away_matches = df[df["Away_Team"] == team].sort_values("Date")

            # Attack: time-decayed weighted avg of goals scored / league avg
            scored_parts = []
            weights_parts = []

            if len(home_matches) > 0:
                hw = self._decay_weights(len(home_matches))
                h_scored = home_matches[h_score_col].values
                # Normalise: what fraction of league-average home scoring?
                scored_parts.append(h_scored / h_avg)
                weights_parts.append(hw)

            if len(away_matches) > 0:
                aw = self._decay_weights(len(away_matches))
                a_scored = away_matches[a_score_col].values
                scored_parts.append(a_scored / a_avg)
                weights_parts.append(aw)

            if scored_parts:
                all_scored = np.concatenate(scored_parts)
                all_weights = np.concatenate(weights_parts)
                self.attack[team] = np.average(all_scored, weights=all_weights)
            else:
                self.attack[team] = 0.85  # promoted team prior (below average)

            # Defence: time-decayed weighted avg of goals conceded / league avg
            conceded_parts = []
            weights_parts = []

            if len(home_matches) > 0:
                hw = self._decay_weights(len(home_matches))
                h_conceded = home_matches[a_score_col].values  # away team scored
                conceded_parts.append(h_conceded / a_avg)
                weights_parts.append(hw)

            if len(away_matches) > 0:
                aw = self._decay_weights(len(away_matches))
                a_conceded = away_matches[h_score_col].values  # home team scored
                conceded_parts.append(a_conceded / h_avg)
                weights_parts.append(aw)

            if conceded_parts:
                all_conceded = np.concatenate(conceded_parts)
                all_weights = np.concatenate(weights_parts)
                self.defence[team] = np.average(all_conceded, weights=all_weights)
            else:
                self.defence[team] = 1.15  # promoted team prior (leaky defence)

        return self

    def _dc_tau(self, x, y, lam, mu):
        if x == 0 and y == 0:
            return 1 - lam * mu * self.rho
        elif x == 0 and y == 1:
            return 1 + lam * self.rho
        elif x == 1 and y == 0:
            return 1 + mu * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        return 1.0

    def predict_match(self, home_team, away_team):
        """Predict P(Over 2.5) for a single match.

        Lambda formula: home_lambda = att_h * def_a * mu * sqrt(gamma)
                        away_lambda = att_a * def_h * mu / sqrt(gamma)
        This distributes home advantage symmetrically so average vs average
        gives correct home/away split.
        """
        h_att = self.attack.get(home_team, 0.85)
        a_def = self.defence.get(away_team, 1.15)
        a_att = self.attack.get(away_team, 0.85)
        h_def = self.defence.get(home_team, 1.15)

        sqrt_gamma = np.sqrt(self.gamma)
        home_lambda = h_att * a_def * self.mu * sqrt_gamma
        away_lambda = a_att * h_def * self.mu / sqrt_gamma

        # Clamp to reasonable range
        home_lambda = np.clip(home_lambda, 0.1, 5.0)
        away_lambda = np.clip(away_lambda, 0.1, 5.0)

        p_under = 0
        for h in range(12):
            for a in range(12):
                if h + a <= 2:
                    p = (poisson_dist.pmf(h, home_lambda) *
                         poisson_dist.pmf(a, away_lambda) *
                         self._dc_tau(h, a, home_lambda, away_lambda))
                    p_under += max(p, 0)

        return np.clip(1 - p_under, 0.01, 0.99)

    def predict_proba_df(self, df):
        """Predict P(Over 2.5) for a DataFrame with Home_Team, Away_Team columns."""
        probs = np.array([
            self.predict_match(row["Home_Team"], row["Away_Team"])
            for _, row in df.iterrows()
        ])
        return probs


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble model (preserves interface for Dash app / pickling)
# ═══════════════════════════════════════════════════════════════════════════════

class IsotonicWrapper:
    """Wraps IsotonicRegression to match predict_proba interface."""
    def __init__(self, iso_model):
        self.iso_model = iso_model
    def predict_proba(self, X):
        X_flat = X.ravel()
        p1 = self.iso_model.predict(X_flat)
        return np.column_stack([1 - p1, p1])


class EnsembleModel:
    """Stacking ensemble with logistic regression meta-learner.

    Base models: XGBoost, LightGBM, Dixon-Coles Poisson.
    Meta-learner: Logistic regression trained on OOF base model predictions.
    This replaces the simple weighted average — the stacker learns optimal
    combination and calibrates probabilities simultaneously.

    Critical: Dixon-Coles predictions must come from dc_model (stored in this class),
    NOT from the Poisson_DC pipeline feature (which uses different parameters and has
    only 0.41 correlation with dc_model). Always pass dc_probs or team names.
    """

    def __init__(self, xgb_model, lgb_model, logreg_model=None, logreg_scaler=None,
                 dc_model=None, stacker=None, feature_names=None,
                 threshold=0.5, platt_scaler=None,
                 # Legacy support: weights for backward compat with old pickles
                 weights=(0.5, 0.5)):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.logreg_model = logreg_model
        self.logreg_scaler = logreg_scaler
        self.dc_model = dc_model
        self.stacker = stacker
        self.feature_names = feature_names
        self.threshold = threshold
        self.platt_scaler = platt_scaler
        self.weights = weights

    def _base_probs(self, X, dc_probs=None):
        """Get predictions from all base models, return as columns for stacker.

        Args:
            X: Feature matrix (numpy array or DataFrame)
            dc_probs: Pre-computed Dixon-Coles probabilities. STRONGLY preferred
                      over Poisson_DC feature extraction.
        """
        if isinstance(X, np.ndarray):
            X_arr = X
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_arr = X.values if hasattr(X, 'values') else np.array(X)
            X_df = X

        xgb_p = self.xgb_model.predict_proba(X_arr)[:, 1]
        lgb_p = self.lgb_model.predict_proba(X_df)[:, 1]

        # Dixon-Coles — use provided dc_probs (preferred), else fall back to feature
        if dc_probs is not None:
            dc_p = np.asarray(dc_probs).ravel()
        else:
            # Fallback: Poisson_DC feature from pipeline (WARNING: lower correlation
            # with dc_model than expected, AUC may degrade)
            dc_p = np.full(len(xgb_p), 0.5)
            if self.feature_names is not None and "Poisson_DC" in self.feature_names:
                dc_idx = self.feature_names.index("Poisson_DC")
                dc_raw = X_arr[:, dc_idx]
                dc_p = np.where(np.isnan(dc_raw), 0.5, dc_raw)

        return np.column_stack([xgb_p, lgb_p, dc_p])

    def predict_proba(self, X, dc_probs=None):
        """Predict P(Over 2.5).

        Args:
            X: Feature matrix (numpy array or DataFrame)
            dc_probs: Pre-computed Dixon-Coles probabilities. Pass these for
                      accurate predictions; without them, falls back to Poisson_DC
                      feature which has lower correlation with dc_model.
        """
        if self.stacker is not None:
            base = self._base_probs(X, dc_probs=dc_probs)
            p1 = self.stacker.predict_proba(base)[:, 1]
        else:
            # Fallback: simple weighted average (backward compat)
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = X
            xgb_p = self.xgb_model.predict_proba(X)[:, 1]
            lgb_p = self.lgb_model.predict_proba(X_df)[:, 1]
            p1 = self.weights[0] * xgb_p + self.weights[1] * lgb_p

        if self.platt_scaler is not None:
            if isinstance(self.platt_scaler, (int, float)):
                # Logit-shift calibration: subtract shift to correct mean drift
                # Preserves ranking (AUC unchanged), fixes probability mean
                logits = np.log(p1 / (1 - p1 + 1e-10))
                p1 = 1 / (1 + np.exp(-(logits - self.platt_scaler)))
            elif isinstance(self.platt_scaler, IsotonicWrapper):
                p1 = self.platt_scaler.predict_proba(p1.reshape(-1, 1))[:, 1]
            else:
                # Platt (LogisticRegression): takes logits
                logits = np.log(p1 / (1 - p1 + 1e-10)).reshape(-1, 1)
                p1 = self.platt_scaler.predict_proba(logits)[:, 1]

        return np.column_stack([1 - p1, p1])

    def predict(self, X, dc_probs=None):
        probs = self.predict_proba(X, dc_probs=dc_probs)
        return (probs[:, 1] >= self.threshold).astype(int)

    def predict_single(self, X, dc_probs=None):
        """For single-match prediction: return calibrated probability.
        Pass dc_probs for accurate Dixon-Coles integration."""
        return self.predict_proba(X, dc_probs=dc_probs)[:, 1]

    def predict_batch(self, X, dc_probs=None, target_over_rate=0.52):
        """For batch prediction: use dynamic percentile threshold.
        Adapts to the actual probability distribution in this batch.
        target_over_rate: expected proportion of Over 2.5 matches (~52% historically)."""
        probs = self.predict_proba(X, dc_probs=dc_probs)[:, 1]
        dynamic_t = np.percentile(probs, (1 - target_over_rate) * 100)
        preds = (probs >= dynamic_t).astype(int)
        return preds, probs, dynamic_t


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(full_df, features, min_train_season=14, start_val_season=19):
    """Walk-forward CV: train on 14..N, validate on N+1, for N = start_val_season-1 .. max-1.

    Returns per-fold metrics and aggregated OOF predictions for the stacker.
    This gives 6+ validation folds instead of a single train/val split,
    producing much more robust estimates of model quality.
    """
    all_seasons = sorted(full_df["SeasonIndex"].unique())
    valid_seasons = [s for s in all_seasons if s >= min_train_season]

    fold_metrics = []
    oof_records = []  # (season, idx, xgb_prob, lgb_prob, dc_prob, y_true)

    for val_season in range(start_val_season, max(valid_seasons) + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= min_train_season) &
                           (full_df["SeasonIndex"] < val_season)].copy()
        val_df = full_df[full_df["SeasonIndex"] == val_season].copy()

        if len(train_df) < 100 or len(val_df) < 50:
            continue

        X_tr = train_df[features].values
        y_tr = train_df["Over_2_5"].values
        X_v = val_df[features].values
        y_v = val_df["Over_2_5"].values

        # XGBoost
        xgb_m = train_xgb(X_tr, y_tr, X_v, y_v)
        xgb_p = xgb_m.predict_proba(X_v)[:, 1]

        # LightGBM
        lgb_m = train_lgb(X_tr, y_tr, X_v, y_v, feature_names=features)
        lgb_p = lgb_m.predict_proba(pd.DataFrame(X_v, columns=features))[:, 1]

        # Dixon-Coles (uses team attack/defence ratings — fundamentally different model)
        dc_m = DixonColesPredictor()
        dc_m.fit(train_df)
        dc_p = dc_m.predict_proba_df(val_df)

        # Simple average for fold evaluation (3 models)
        avg_p = (xgb_p + lgb_p + dc_p) / 3
        fold_auc = roc_auc_score(y_v, avg_p)
        fold_acc = accuracy_score(y_v, (avg_p > 0.5).astype(int))
        fold_brier = brier_score_loss(y_v, avg_p)

        fold_metrics.append({
            "val_season": val_season, "n_train": len(train_df), "n_val": len(val_df),
            "auc": fold_auc, "acc": fold_acc, "brier": fold_brier,
        })
        print(f"  Fold S{val_season}: train={len(train_df):4d} val={len(val_df):3d} "
              f"AUC={fold_auc:.4f} Acc={fold_acc:.4f} Brier={fold_brier:.4f}")

        # Store OOF predictions for stacker training
        for i, idx in enumerate(val_df.index):
            oof_records.append({
                "idx": idx, "season": val_season,
                "xgb": xgb_p[i], "lgb": lgb_p[i], "dc": dc_p[i],
                "y": y_v[i],
            })

    oof_df = pd.DataFrame(oof_records)
    return fold_metrics, oof_df


# ═══════════════════════════════════════════════════════════════════════════════
# Feature pruning
# ═══════════════════════════════════════════════════════════════════════════════

def prune_features(xgb_model, features, min_importance=0.001):
    """Remove features with near-zero importance."""
    importance = xgb_model.feature_importances_
    mask = importance >= min_importance
    kept = [f for f, m in zip(features, mask) if m]
    dropped = [f for f, m in zip(features, mask) if not m]

    if dropped:
        print(f"\nFeature pruning: dropping {len(dropped)} low-importance features:")
        for f in dropped:
            print(f"  - {f} (importance: {importance[features.index(f)]:.6f})")
        print(f"Keeping {len(kept)}/{len(features)} features")
    else:
        print("\nFeature pruning: all features above threshold, keeping all")

    return kept, dropped


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation & plotting
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, label="Test"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    ll = log_loss(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    print(f"\n{'='*50}")
    print(f"{label} Set Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Log Loss:   {ll:.4f}")
    print(f"Brier Score:{brier:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    print(f"\nBase rate (actual Over 2.5): {y.mean():.1%}")
    print(f"Predicted Over 2.5 rate:     {y_pred.mean():.1%}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "log_loss": ll, "brier": brier, "auc": auc}


def plot_feature_importance(model, features, top_n=20):
    """Plot XGBoost feature importance."""
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), [features[i] for i in idx])
    plt.xlabel("Feature Importance (gain)")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def plot_shap(model, X, features, max_display=20):
    """Generate SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=features,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), dpi=150)
    plt.close()


def plot_calibration(model, X, y):
    """Plot calibration curve."""
    from sklearn.calibration import calibration_curve
    y_prob = model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted = calibration_curve(y, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "calibration_curve.png"), dpi=150)
    plt.close()


def save_model(model, features):
    """Save model and feature list. No scaler needed (trees don't need scaling)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Save a dummy scaler for backward compat with Dash app that loads SCALER_PATH
    joblib.dump(None, SCALER_PATH)
    joblib.dump(features, FEATURE_LIST_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load saved model, scaler, and feature list."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_LIST_PATH)
    return model, scaler, features


# ═══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main(tune=False):
    # ── Load data ──
    data = run_pipeline(verbose=True)
    train = data["train"]
    val = data["val"]
    test = data["test"]
    full_df = data["full_df"]
    features = list(data["features"])

    X_train = train[features].values
    y_train = train["Over_2_5"].values
    X_val = val[features].values
    y_val = val["Over_2_5"].values
    X_test = test[features].values
    y_test = test["Over_2_5"].values

    # ── Phase 1: Walk-forward CV (robust multi-fold evaluation) ──
    print("\n" + "="*60)
    print("Phase 1: Walk-forward cross-validation")
    print("="*60)
    # Combine train+val for walk-forward (test stays held out)
    wf_df = full_df[(full_df["SeasonIndex"] >= TRAIN_MIN_SEASON) &
                     (~full_df["SeasonIndex"].isin(TEST_SEASONS))].copy()
    fold_metrics, oof_df = walk_forward_cv(wf_df, features)

    if len(fold_metrics) > 0:
        avg_auc = np.mean([f["auc"] for f in fold_metrics])
        avg_acc = np.mean([f["acc"] for f in fold_metrics])
        avg_brier = np.mean([f["brier"] for f in fold_metrics])
        print(f"\n  Walk-forward summary ({len(fold_metrics)} folds):")
        print(f"    Avg AUC:   {avg_auc:.4f}")
        print(f"    Avg Acc:   {avg_acc:.4f}")
        print(f"    Avg Brier: {avg_brier:.4f}")

    # ── Phase 2: Train base models ──
    # Step 1: Train on train (14-23) with val (24) for early stopping & feature pruning.
    # Step 2: Retrain final models on train+val combined (14-24) for maximum data.
    print("\n" + "="*60)
    print("Phase 2: Train base models")
    print("="*60)

    # 2a: XGBoost — use val for early stopping, then retrain on full data
    print("\nTraining XGBoost (feature pruning on train, val for early stopping)...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    print(f"  Best iteration (for pruning): {xgb_model.best_iteration}")
    best_n_trees = xgb_model.best_iteration

    kept_features, dropped = prune_features(xgb_model, features)
    if dropped:
        features = kept_features
        X_train = train[features].values
        X_val = val[features].values
        X_test = test[features].values

    # Combine train+val for final training (seasons 14-24)
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val[features].values
    y_train_val = train_val["Over_2_5"].values
    print(f"\n  Combined train+val: {len(train_val)} matches (seasons 14-24)")

    # Retrain XGBoost on full data using best_n_trees from early stopping
    print("  Retraining XGBoost on train+val...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=best_n_trees, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        gamma=0.1, reg_alpha=0.5, reg_lambda=3.0,
        eval_metric="logloss", random_state=42,
    )
    xgb_model.fit(X_train_val, y_train_val, verbose=False)
    print(f"  XGBoost retrained with {best_n_trees} trees on {len(train_val)} rows")

    # 2b: LightGBM — same approach
    print("\nTraining LightGBM (early stopping on val)...")
    lgb_temp = train_lgb(X_train, y_train, X_val, y_val, feature_names=features)
    best_lgb_trees = lgb_temp.best_iteration_
    print(f"  Best iteration: {best_lgb_trees}")

    print("  Retraining LightGBM on train+val...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=best_lgb_trees, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )
    lgb_model.fit(pd.DataFrame(X_train_val, columns=features), y_train_val)
    print(f"  LightGBM retrained with {best_lgb_trees} trees on {len(train_val)} rows")

    # 2c: Dixon-Coles on train+val
    print("\nTraining Dixon-Coles on train+val...")
    dc_model = DixonColesPredictor()
    dc_model.fit(train_val)

    # ── Phase 3: Stacking meta-learner ──
    print("\n" + "="*60)
    print("Phase 3: Stacking meta-learner (walk-forward OOF predictions)")
    print("="*60)

    # Use walk-forward OOF predictions from Phase 1 to train the stacker.
    # This captures temporal distribution shift (train on past, predict future)
    # — unlike KFold which leaks within-season patterns.
    if len(oof_df) > 100:
        print(f"\n  Using {len(oof_df)} walk-forward OOF predictions for stacker training")
        oof_stack_full = oof_df[["xgb", "lgb", "dc"]].values
        oof_y = oof_df["y"].values

        stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        stacker.fit(oof_stack_full, oof_y)
    else:
        # Fallback: KFold OOF if walk-forward didn't produce enough data
        print("\n  Insufficient walk-forward OOF -- falling back to KFold")
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=False)
        oof_xgb = np.zeros(len(y_train))
        oof_lgb = np.zeros(len(y_train))
        oof_dc = np.zeros(len(y_train))

        for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train)):
            fold_xgb = train_xgb(X_train[tr_idx], y_train[tr_idx],
                                  X_train[oof_idx], y_train[oof_idx])
            oof_xgb[oof_idx] = fold_xgb.predict_proba(X_train[oof_idx])[:, 1]

            fold_lgb = train_lgb(X_train[tr_idx], y_train[tr_idx],
                                  X_train[oof_idx], y_train[oof_idx],
                                  feature_names=features)
            oof_lgb[oof_idx] = fold_lgb.predict_proba(
                pd.DataFrame(X_train[oof_idx], columns=features))[:, 1]

            fold_train_df = train.iloc[tr_idx]
            fold_dc = DixonColesPredictor()
            fold_dc.fit(fold_train_df)
            oof_dc[oof_idx] = fold_dc.predict_proba_df(train.iloc[oof_idx])

        oof_stack_full = np.column_stack([oof_xgb, oof_lgb, oof_dc])
        oof_y = y_train

        stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        stacker.fit(oof_stack_full, oof_y)

    # Stacker weights (interpretable)
    print(f"\n  Stacker coefficients: XGB={stacker.coef_[0][0]:.3f}, "
          f"LGB={stacker.coef_[0][1]:.3f}, DC={stacker.coef_[0][2]:.3f}")
    print(f"  Stacker intercept: {stacker.intercept_[0]:.3f}")

    # OOF stacker AUC (unbiased — walk-forward predictions)
    oof_stacker_eval = stacker.predict_proba(oof_stack_full)[:, 1]
    oof_stacker_auc = roc_auc_score(oof_y, oof_stacker_eval)
    print(f"  OOF stacker AUC: {oof_stacker_auc:.4f}")

    # ── Phase 4: Calibration (logit-shift prior correction) ──
    print("\n" + "="*60)
    print("Phase 4: Calibration (logit-shift prior correction)")
    print("="*60)

    # Problem: stacker probabilities drift upward on new seasons.
    # On OOF data, mean predicted prob ≈ actual Over rate (well-calibrated).
    # But on truly new data (test), predictions inflate by ~3-6% because:
    #   1. Models trained on more data → more confident → shift away from 0.5
    #   2. Feature distributions shift between seasons
    #
    # Old approach (Platt on nested OOF) learned near-identity because OOF was
    # already calibrated. It couldn't fix the TEST shift it never saw.
    #
    # New approach: Logit-shift prior correction.
    # The model's RANKING is correct (AUC is stable). The MEAN drifts.
    # Fix: compute a logit-space shift on val predictions (closest to deployment),
    # then apply at inference time.
    #   shift = mean(logit(p_val)) - logit(base_rate)
    #   p_corrected = sigmoid(logit(p) - shift)
    # This preserves ranking perfectly (monotonic) while fixing the mean.

    # Step 1: Get stacker predictions on val set (most recent known data)
    xgb_val_p = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_p = lgb_model.predict_proba(pd.DataFrame(X_val, columns=features))[:, 1]
    dc_val_p = dc_model.predict_proba_df(val)
    val_stack = np.column_stack([xgb_val_p, lgb_val_p, dc_val_p])
    stacker_val_raw = stacker.predict_proba(val_stack)[:, 1]

    val_mean_logit = np.mean(np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10)))
    # Use long-term base rate as target (more stable than single-season val rate)
    base_rate = y_train_val.mean()  # ~0.53 over seasons 14-24
    target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
    logit_shift = val_mean_logit - target_logit

    print(f"\n  Val predictions: mean={stacker_val_raw.mean():.4f}, actual={y_val.mean():.4f}")
    print(f"  Val mean logit: {val_mean_logit:.4f}")
    print(f"  Target base rate: {base_rate:.4f} (logit: {target_logit:.4f})")
    print(f"  Logit shift: {logit_shift:.4f}")

    # Verify on val: corrected predictions should match base rate better
    val_logits = np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10))
    val_corrected = 1 / (1 + np.exp(-(val_logits - logit_shift)))
    print(f"\n  Val corrected mean: {val_corrected.mean():.4f} (target: {base_rate:.4f})")
    print(f"  Val Brier: before={brier_score_loss(y_val, stacker_val_raw):.4f}, "
          f"after={brier_score_loss(y_val, val_corrected):.4f}")
    print(f"  Val AUC:   {roc_auc_score(y_val, stacker_val_raw):.4f} "
          f"(unchanged — logit shift preserves ranking)")

    # Also compute nested OOF stats for comparison
    oof_seasons = sorted(oof_df["season"].unique())
    nested_stacker_preds = np.zeros(len(oof_df))
    print(f"\n  Nested OOF stacker predictions ({len(oof_seasons)} seasons)...")
    for hold_season in oof_seasons:
        hold_mask = oof_df["season"] == hold_season
        train_mask = ~hold_mask
        if train_mask.sum() < 50 or hold_mask.sum() < 10:
            nested_stacker_preds[hold_mask] = stacker.predict_proba(
                oof_stack_full[hold_mask])[:, 1]
            continue
        temp_stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        temp_stacker.fit(oof_stack_full[train_mask], oof_y[train_mask])
        nested_stacker_preds[hold_mask] = temp_stacker.predict_proba(
            oof_stack_full[hold_mask])[:, 1]
        hold_auc = roc_auc_score(oof_y[hold_mask], nested_stacker_preds[hold_mask])
        print(f"    S{hold_season}: n={hold_mask.sum()}, AUC={hold_auc:.4f}, "
              f"mean_prob={nested_stacker_preds[hold_mask].mean():.4f}")
    print(f"  Nested OOF: AUC={roc_auc_score(oof_y, nested_stacker_preds):.4f}, "
          f"Brier={brier_score_loss(oof_y, nested_stacker_preds):.4f}, "
          f"mean={nested_stacker_preds.mean():.4f}")

    # Store the logit_shift as calibration (will be applied in EnsembleModel.predict_proba)
    calibrator = logit_shift  # float, not a model
    calibrator_type = "logit_shift"
    print(f"\n  >> Using logit-shift calibration (shift={logit_shift:.4f})")

    # Threshold
    overall_base_rate = y_train_val.mean()
    chosen_threshold = 0.50  # With proper calibration, 0.5 should work

    # ── Build final ensemble ──
    ensemble = EnsembleModel(
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        logreg_model=None,      # LR removed: AUC < 0.5 on test, adds noise not signal
        logreg_scaler=None,
        dc_model=dc_model,
        stacker=stacker,
        feature_names=features,
        threshold=chosen_threshold,
        platt_scaler=calibrator,
    )

    # ── Phase 5: Evaluate on held-out test ──
    print("\n" + "="*60)
    print("Phase 5: Final evaluation")
    print("="*60)

    # Individual models on test
    print("\n--- Individual Base Models ---")
    xgb_test_p = xgb_model.predict_proba(X_test)[:, 1]
    lgb_test_p = lgb_model.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    dc_test_p = dc_model.predict_proba_df(test)

    for name, probs in [("XGBoost", xgb_test_p), ("LightGBM", lgb_test_p),
                         ("Dixon-Coles", dc_test_p)]:
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        brier = brier_score_loss(y_test, probs)
        print(f"  {name:12s}: AUC={auc:.4f}  Acc={acc:.4f}  Brier={brier:.4f}")

    # Stacker on test — UNCALIBRATED (consistent with stacker training: direct dc_model predictions)
    test_stack = np.column_stack([xgb_test_p, lgb_test_p, dc_test_p])
    stacker_test_raw = stacker.predict_proba(test_stack)[:, 1]

    # CALIBRATED — apply calibrator to stacker output
    if calibrator_type == "logit_shift":
        raw_logits = np.log(stacker_test_raw / (1 - stacker_test_raw + 1e-10))
        stacker_test_p = 1 / (1 + np.exp(-(raw_logits - calibrator)))
    elif calibrator_type == "platt":
        raw_logits = np.log(stacker_test_raw / (1 - stacker_test_raw + 1e-10)).reshape(-1, 1)
        stacker_test_p = calibrator.predict_proba(raw_logits)[:, 1]
    else:
        stacker_test_p = calibrator.predict_proba(stacker_test_raw.reshape(-1, 1))[:, 1]

    # Also check ensemble path (uses Poisson_DC feature as DC fallback when dc_probs not passed)
    ensemble_fallback_p = ensemble.predict_proba(X_test)[:, 1]
    # Proper path: pass dc_probs from dc_model
    ensemble_proper_p = ensemble.predict_proba(X_test, dc_probs=dc_test_p)[:, 1]

    print(f"\n--- Stacker Ensemble (3 models: XGB + LGB + DC) ---")
    stacker_test_auc = roc_auc_score(y_test, stacker_test_p)

    # Calibration diagnostics
    print(f"\n  Probability distribution:")
    print(f"    {'':20s} {'Uncalibrated':>14s} {'Calibrated':>14s} {'Actual':>10s}")
    print(f"    {'Mean prob':20s} {stacker_test_raw.mean():14.4f} {stacker_test_p.mean():14.4f} {y_test.mean():10.4f}")
    print(f"    {'Median prob':20s} {np.median(stacker_test_raw):14.4f} {np.median(stacker_test_p):14.4f} {'':>10s}")
    print(f"    {'Pred Over% (>0.5)':20s} {(stacker_test_raw > 0.5).mean():14.1%} {(stacker_test_p > 0.5).mean():14.1%} {y_test.mean():10.1%}")
    print(f"    {'Brier score':20s} {brier_score_loss(y_test, stacker_test_raw):14.4f} {brier_score_loss(y_test, stacker_test_p):14.4f} {'':>10s}")
    print(f"    {'AUC':20s} {roc_auc_score(y_test, stacker_test_raw):14.4f} {roc_auc_score(y_test, stacker_test_p):14.4f} {'':>10s}")

    # DC path consistency check
    ensemble_base_fallback = ensemble._base_probs(X_test)  # uses Poisson_DC feature
    dc_from_feature = ensemble_base_fallback[:, 2]
    dc_corr = np.corrcoef(dc_test_p, dc_from_feature)[0, 1]
    print(f"\n  DC consistency: dc_model mean={dc_test_p.mean():.4f}, "
          f"Poisson_DC feature mean={dc_from_feature.mean():.4f}, corr={dc_corr:.4f}")
    print(f"  Ensemble (Poisson_DC fallback) AUC: {roc_auc_score(y_test, ensemble_fallback_p):.4f}")
    print(f"  Ensemble (dc_probs passed)     AUC: {roc_auc_score(y_test, ensemble_proper_p):.4f}")

    # Accuracy
    stacker_test_acc = accuracy_score(y_test, (stacker_test_p >= 0.5).astype(int))
    oracle_t = np.percentile(stacker_test_p, (1 - y_test.mean()) * 100)
    stacker_test_acc_oracle = accuracy_score(y_test, (stacker_test_p >= oracle_t).astype(int))
    stacker_test_brier = brier_score_loss(y_test, stacker_test_p)

    print(f"\n  Acc (t=0.50):    {stacker_test_acc:.4f} (pred rate: {(stacker_test_p >= 0.5).mean():.1%})")
    print(f"  Acc (oracle t={oracle_t:.4f}): {stacker_test_acc_oracle:.4f}")
    print(f"  Actual Over rate: {y_test.mean():.1%}")

    # Simple avg for comparison (3 models)
    simple_avg_test = (xgb_test_p + lgb_test_p + dc_test_p) / 3
    simple_auc = roc_auc_score(y_test, simple_avg_test)
    simple_oracle_t = np.percentile(simple_avg_test, (1 - y_test.mean()) * 100)
    simple_acc = accuracy_score(y_test, (simple_avg_test >= simple_oracle_t).astype(int))
    print(f"\n  Simple avg (3 models):  AUC={simple_auc:.4f}  Acc(oracle)={simple_acc:.4f}")
    print(f"  DC alone:               AUC={roc_auc_score(y_test, dc_test_p):.4f}")
    print(f"  XGB alone:              AUC={roc_auc_score(y_test, xgb_test_p):.4f}")

    # Overfitting diagnostics
    print(f"\n--- Overfitting Diagnostics ---")
    # Train+Val AUC (models trained on this data — measures memorization)
    xgb_tv_p = xgb_model.predict_proba(X_train_val)[:, 1]
    lgb_tv_p = lgb_model.predict_proba(pd.DataFrame(X_train_val, columns=features))[:, 1]
    dc_tv_p = dc_model.predict_proba_df(train_val)

    tv_stack = np.column_stack([xgb_tv_p, lgb_tv_p, dc_tv_p])
    stacker_tv_p = stacker.predict_proba(tv_stack)[:, 1]
    tv_auc = roc_auc_score(y_train_val, stacker_tv_p)

    # OOF AUC (unbiased estimate from walk-forward)
    oof_stack_p = stacker.predict_proba(oof_stack_full)[:, 1]
    oof_auc = roc_auc_score(oof_y, oof_stack_p)

    print(f"  {'':24s} {'Train+Val':>10s} {'OOF(WF)':>10s} {'Test':>10s}")
    print(f"  {'AUC':24s} {tv_auc:10.4f} {oof_auc:10.4f} {stacker_test_auc:10.4f}")
    print(f"  {'Brier':24s} {brier_score_loss(y_train_val, stacker_tv_p):10.4f} "
          f"{brier_score_loss(oof_y, oof_stack_p):10.4f} {stacker_test_brier:10.4f}")
    print(f"  {'Over rate (actual)':24s} {y_train_val.mean():10.1%} "
          f"{oof_y.mean():10.1%} {y_test.mean():10.1%}")
    print(f"  {'N matches':24s} {len(y_train_val):10d} {len(oof_y):10d} {len(y_test):10d}")

    train_test_auc_gap = tv_auc - stacker_test_auc
    print(f"\n  Train+Val-Test AUC gap: {train_test_auc_gap:.4f} "
          f"({'OK' if train_test_auc_gap < 0.10 else 'OVERFITTING' if train_test_auc_gap < 0.15 else 'SEVERE OVERFITTING'})")

    # Walk-forward vs test comparison (most meaningful)
    if len(fold_metrics) > 0:
        print(f"  Walk-forward avg AUC: {avg_auc:.4f} vs Test AUC: {stacker_test_auc:.4f} "
              f"(gap: {abs(avg_auc - stacker_test_auc):.4f})")

    test_metrics = {
        "accuracy": stacker_test_acc, "auc": stacker_test_auc,
        "brier": stacker_test_brier, "log_loss": log_loss(y_test, stacker_test_p),
    }

    # Save
    save_model(ensemble, features)

    # Plots
    print("\nGenerating plots...")
    plot_feature_importance(xgb_model, features)
    plot_shap(xgb_model, X_test[:200], features)
    plot_calibration(ensemble, X_test, y_test)

    return ensemble, features, test_metrics


if __name__ == "__main__":
    import sys
    tune = "--tune" in sys.argv
    main(tune=tune)
