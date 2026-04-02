import pandas as pd
import numpy as np
import pytest
from gsmodel import load_model, DixonColesPredictor
from gspipeline import run_pipeline

# 1. Prediction Integrity (Already had this, keeping it for completeness)
def test_model_still_predicts():
    """Guards against shape mismatches: Model must still accept input and output probs."""
    model, _, features = load_model()
    dummy_x = np.random.rand(1, len(features))
    probs = model.predict_proba(dummy_x)
    assert probs.shape == (1, 2)
    assert 0 <= probs[0, 1] <= 1

# 2. Feature Alignment Test
def test_feature_names_consistency():
    """Guards against naming errors: The model's expected features must exist in the pipeline output."""
    model, _, model_features = load_model()
    # Mock a small run of the pipeline
    result = run_pipeline(verbose=False)
    pipeline_features = result["features"]

    # Every feature the model expects must be produced by the pipeline
    for f in model_features:
        assert f in pipeline_features, f"Model expects {f} but pipeline did not produce it!"

# 3. Dixon-Coles Mathematical Stability
def test_dixon_coles_logic():
    """Guards against math regressions: Dixon-Coles must produce valid probabilities."""
    dc = DixonColesPredictor()
    # Mock venue-specific ratings
    dc.attack_home = {"TeamA": 1.3, "TeamB": 0.85}
    dc.attack_away = {"TeamA": 1.1, "TeamB": 0.75}
    dc.defence_home = {"TeamA": 0.85, "TeamB": 1.05}
    dc.defence_away = {"TeamA": 0.95, "TeamB": 1.15}
    dc.mu = 1.35

    prob = dc.predict_match("TeamA", "TeamB")
    # P(Over 2.5) should be a float between 0 and 1
    assert isinstance(prob, float)
    assert 0 < prob < 1

# 4. Leakage Prevention Test
def test_no_target_leakage():
    """Guards against cheating: Match outcomes must NOT be in the feature list."""
    _, _, features = load_model()
    forbidden = ["TG", "Home_Goals", "Away_Goals", "FTR", "Over_2_5"]

    for feat in features:
        assert feat not in forbidden, f"CRITICAL: Leakage detected! {feat} is in the feature list."

# 5. Pipeline End-to-End Test
def test_pipeline_output_structure():
    """Guards against structural changes: Ensure the pipeline returns the expected dictionary keys."""
    result = run_pipeline(verbose=False)
    expected_keys = ["train", "val", "test", "features", "train_medians"]
    for key in expected_keys:
        assert key in result, f"Pipeline output missing key: {key}"
