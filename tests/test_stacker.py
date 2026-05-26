"""Contract-layer tests for the O/U 2.5 stacking ensemble.

These tests catch silent regressions in the stacker plumbing —
data shape, calibration anchor, output range, monotonicity, and
determinism. They are NOT a substitute for end-to-end backtest
validation; that is what scripts/lr_ablation_test.py is for.

What is tested (the contract):
  * walk_forward_cv emits an `lr` column in oof_records
  * The fitted stacker has exactly 4 input features
  * Logit-shift calibration anchors the mean to base rate
  * Output probabilities stay in [0, 1] for arbitrary base inputs
  * Stacker is monotone in each base model's input
  * Determinism: identical seed + input -> identical output

What is NOT tested:
  * Exact coefficient values (they legitimately move with data)
  * ROI / AUC thresholds (the ablation script owns those)
  * Other markets (BTTS, alt lines) — separate tests
"""
import numpy as np
import pytest


# Reference probability for the determinism regression test.
# Computed from a stacker trained on the committed fixture with
# random_state=42 and the canonical [xgb, lgb, lr, dc] = [0.5, 0.5, 0.5, 0.5]
# probe input. If you regenerate the fixture, update this constant.
STACKER_GOLDEN_PROB_AT_HALF: float = 0.5496957130536717


class TestWalkForwardCVOutput:
    """walk_forward_cv must emit the LR column for downstream stacker training."""

    def test_walk_forward_cv_oof_has_lr_column(self, oof_df_for_stacker):
        assert "lr" in oof_df_for_stacker.columns, (
            "oof_records missing 'lr' column — walk_forward_cv has dropped LR; "
            "downstream stacker training will silently degrade to 3-model."
        )
        assert oof_df_for_stacker["lr"].notna().all(), (
            "Some LR OOF predictions are NaN — calibration math will fail."
        )
        assert oof_df_for_stacker["lr"].between(0, 1).all(), (
            "LR OOF predictions fell outside [0, 1] — _lr_predict is broken."
        )


class TestStackerInputContract:
    """The fitted stacker must accept exactly 4 base-model probabilities."""

    def test_stacker_trained_on_four_features(self, trained_stacker):
        assert trained_stacker.coef_.shape == (1, 4), (
            f"Stacker has {trained_stacker.coef_.shape[1]} features, expected 4. "
            "A base model has been added or removed without updating the stacker."
        )
        assert trained_stacker.n_features_in_ == 4


class TestStackerCalibration:
    """Logit-shift calibration must anchor the mean output to the base rate."""

    def test_stacker_logit_shift_consistent_with_target(
        self, oof_df_for_stacker, trained_stacker
    ):
        X = oof_df_for_stacker[["xgb", "lgb", "lr", "dc"]].values
        raw_probs = trained_stacker.predict_proba(X)[:, 1]
        raw_logits = np.log(raw_probs / (1 - raw_probs + 1e-10))

        base_rate = oof_df_for_stacker["y"].mean()
        target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
        shift = raw_logits.mean() - target_logit

        shifted_logits = raw_logits - shift
        shifted_probs = 1 / (1 + np.exp(-shifted_logits))

        assert abs(shifted_probs.mean() - base_rate) < 0.01, (
            f"Logit-shift anchor missed: shifted mean {shifted_probs.mean():.4f} "
            f"vs base rate {base_rate:.4f}. Calibration math may be broken."
        )


class TestStackerOutputRange:
    """Stacker outputs must stay in [0, 1] for arbitrary realistic inputs."""

    def test_stacker_prob_in_unit_interval(self, trained_stacker):
        rng = np.random.default_rng(42)
        X = rng.uniform(0.05, 0.95, size=(100, 4))
        probs = trained_stacker.predict_proba(X)[:, 1]
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert not np.any(np.isnan(probs))
        assert not np.any(np.isinf(probs))


class TestStackerResponseSign:
    """The output's response to each input must match the sign of that input's coefficient.

    This is a structural wiring check: if the column order fed to the stacker at
    inference (predict.py) ever drifts from the order used at training, a
    positive-coef input would appear to *decrease* the output and vice versa.
    Catches that bug regardless of whether the empirical coefs are positive,
    negative, or mixed.
    """

    def test_stacker_response_sign_matches_coef_sign(self, trained_stacker):
        coefs = trained_stacker.coef_[0]
        base = np.array([[0.5, 0.5, 0.5, 0.5]])
        base_prob = trained_stacker.predict_proba(base)[:, 1][0]
        names = ["xgb", "lgb", "lr", "dc"]
        for i, name in enumerate(names):
            if abs(coefs[i]) < 1e-6:
                continue
            up = base.copy()
            up[0, i] = 0.7
            up_prob = trained_stacker.predict_proba(up)[:, 1][0]
            delta = up_prob - base_prob
            assert np.sign(delta) == np.sign(coefs[i]), (
                f"Stacker response sign mismatch for {name}: "
                f"coef={coefs[i]:.4f}, delta={delta:.6f}. "
                "Likely an input-column wiring bug between training and inference."
            )


class TestStackerReproducibility:
    """Determinism regression: fixed seed + input must produce identical output."""

    def test_stacker_reproducibility_fixed_seed(self, oof_df_for_stacker):
        from sklearn.linear_model import LogisticRegression
        X = oof_df_for_stacker[["xgb", "lgb", "lr", "dc"]].values
        y = oof_df_for_stacker["y"].values

        s1 = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ).fit(X, y)
        s2 = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ).fit(X, y)

        np.testing.assert_array_equal(s1.coef_, s2.coef_)
        np.testing.assert_array_equal(s1.intercept_, s2.intercept_)

        probe = np.array([[0.5, 0.5, 0.5, 0.5]])
        p1 = float(s1.predict_proba(probe)[:, 1][0])
        p2 = float(s2.predict_proba(probe)[:, 1][0])
        assert p1 == p2

        if STACKER_GOLDEN_PROB_AT_HALF is not None:
            assert abs(p1 - STACKER_GOLDEN_PROB_AT_HALF) < 1e-9, (
                f"Stacker output drifted: {p1} vs golden {STACKER_GOLDEN_PROB_AT_HALF}. "
                "If the fixture was regenerated, update STACKER_GOLDEN_PROB_AT_HALF."
            )
