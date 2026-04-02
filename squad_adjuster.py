"""
Squad Strength Adjustment Model.

A lightweight logistic regression that learns how squad composition
affects Over/Under 2.5 probability, trained on the 2 seasons with
player data (2024-25 and 2025-26).

The base XGBoost model produces P(over 2.5) from team-level features.
This adjuster takes that probability + squad features and produces
an adjusted probability.

Usage:
    python squad_adjuster.py          # Train and save the adjuster
    python squad_adjuster.py --eval   # Train and show detailed evaluation
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import MODEL_DIR, SQUAD_FEATURES
from pipeline import run_pipeline

ADJUSTER_PATH = os.path.join(MODEL_DIR, "squad_adjuster.pkl")


def build_adjustment_dataset():
    """
    Build training data for the squad adjuster.
    Uses the base model's predictions + squad features as inputs,
    and Over_2_5 as target.
    """
    # Load base model
    from model import EnsembleModel, IsotonicWrapper, DixonColesPredictor
    sys.modules["__main__"].EnsembleModel = EnsembleModel
    sys.modules["__main__"].IsotonicWrapper = IsotonicWrapper
    sys.modules["__main__"].DixonColesPredictor = DixonColesPredictor
    model = joblib.load(os.path.join(MODEL_DIR, "over_under_model.pkl"))
    _ = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))  # backward compat
    features = joblib.load(os.path.join(MODEL_DIR, "feature_list.pkl"))

    # Run pipeline
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]

    # Get rows with squad features (seasons 24-25)
    squad_mask = full_df["Home_AvailableXG"].notna()
    squad_df = full_df[squad_mask].copy()
    print(f"Squad adjustment dataset: {len(squad_df)} rows with squad features")

    if len(squad_df) < 50:
        print("Not enough data for squad adjustment model.")
        return None, None, None, None

    # Get base model predictions for these rows
    # Compute DC predictions from dc_model for accurate ensemble predictions
    X_base = squad_df[features].values
    dc_probs = None
    if hasattr(model, 'dc_model') and model.dc_model is not None:
        dc_probs = model.dc_model.predict_proba_df(squad_df)
    base_probs = model.predict_proba(X_base, dc_probs=dc_probs)[:, 1]

    # Build adjustment features: base_prob + squad features
    adj_features = ["base_prob"] + [f for f in SQUAD_FEATURES if f in squad_df.columns]
    X_adj = pd.DataFrame(index=squad_df.index)
    X_adj["base_prob"] = base_probs
    for f in SQUAD_FEATURES:
        if f in squad_df.columns:
            X_adj[f] = squad_df[f].values

    y = squad_df["Over_2_5"].values

    # Drop rows with NaN in adjustment features
    valid_mask = X_adj.notna().all(axis=1)
    X_adj = X_adj[valid_mask]
    y = y[valid_mask.values]
    squad_df_valid = squad_df[valid_mask]

    print(f"Valid rows (no NaN): {len(X_adj)}")

    # Split: season 24 = train, season 25 = test
    train_mask = squad_df_valid["SeasonIndex"] == 24
    test_mask = squad_df_valid["SeasonIndex"] == 25

    X_train, y_train = X_adj[train_mask], y[train_mask.values]
    X_test, y_test = X_adj[test_mask], y[test_mask.values]

    print(f"Train: {len(X_train)} (season 24) | Test: {len(X_test)} (season 25)")

    return X_train, y_train, X_test, y_test


def train_adjuster(X_train, y_train):
    """Train a logistic regression adjuster."""
    # Use regularized logistic regression to prevent overfitting on small dataset
    adj = LogisticRegression(
        C=0.1,  # Strong regularization
        max_iter=1000,
        solver="lbfgs",
    )
    adj.fit(X_train, y_train)
    return adj


def evaluate(adj, X_train, y_train, X_test, y_test):
    """Compare base model vs adjusted predictions."""
    print("\n=== Base Model (no squad adjustment) ===")
    # Base model predictions are in the first column
    base_train = X_train["base_prob"].values
    base_test = X_test["base_prob"].values

    base_preds_train = (base_train > 0.5).astype(int)
    base_preds_test = (base_test > 0.5).astype(int)

    print(f"Train accuracy: {accuracy_score(y_train, base_preds_train):.1%}")
    print(f"Test accuracy:  {accuracy_score(y_test, base_preds_test):.1%}")
    print(f"Test log-loss:  {log_loss(y_test, base_test):.4f}")
    print(f"Test Brier:     {brier_score_loss(y_test, base_test):.4f}")

    print("\n=== Squad-Adjusted Model ===")
    adj_probs_train = adj.predict_proba(X_train)[:, 1]
    adj_probs_test = adj.predict_proba(X_test)[:, 1]

    adj_preds_train = (adj_probs_train > 0.5).astype(int)
    adj_preds_test = (adj_probs_test > 0.5).astype(int)

    print(f"Train accuracy: {accuracy_score(y_train, adj_preds_train):.1%}")
    print(f"Test accuracy:  {accuracy_score(y_test, adj_preds_test):.1%}")
    print(f"Test log-loss:  {log_loss(y_test, adj_probs_test):.4f}")
    print(f"Test Brier:     {brier_score_loss(y_test, adj_probs_test):.4f}")

    # Show coefficient analysis
    feature_names = X_train.columns.tolist()
    coefs = adj.coef_[0]
    print("\n=== Feature Coefficients ===")
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        direction = "-> more over" if coef > 0 else "-> more under"
        print(f"  {name:30s} {coef:+.4f}  {direction}")

    # Show examples of largest adjustments
    print("\n=== Largest Adjustments (test set) ===")
    diffs = adj_probs_test - base_test
    sorted_idx = np.argsort(np.abs(diffs))[::-1][:10]
    for i in sorted_idx:
        print(f"  Base: {base_test[i]:.1%} -> Adj: {adj_probs_test[i]:.1%} "
              f"(delta: {diffs[i]:+.1%})")


def main():
    X_train, y_train, X_test, y_test = build_adjustment_dataset()
    if X_train is None:
        return

    adj = train_adjuster(X_train, y_train)

    if "--eval" in sys.argv:
        evaluate(adj, X_train, y_train, X_test, y_test)

    # Save
    joblib.dump(adj, ADJUSTER_PATH)
    print(f"\nSquad adjuster saved to {ADJUSTER_PATH}")


if __name__ == "__main__":
    main()
