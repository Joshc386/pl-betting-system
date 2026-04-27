"""Tests for edge_analytics.py — edge bucket, calibration, and side analysis.

Covers:
  - Edge bucket analysis (binning, win rate, ROI)
  - Calibration curve (predicted vs actual, minimum sample gating)
  - Brier score computation
  - Per-model accuracy
  - Confidence level validation
  - Side analysis
  - Season trends
  - run_full_analytics orchestration
  - Empty DataFrame handling for all functions
"""
import numpy as np
import pandas as pd
import pytest

from edge_analytics import (
    EDGE_BUCKETS,
    brier_score,
    calibration_curve,
    confidence_validation,
    edge_bucket_analysis,
    per_model_accuracy,
    run_full_analytics,
    season_trends,
    side_analysis,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

def _make_bets_df(
    n: int = 100,
    seed: int = 42,
    edge_range: tuple[float, float] = (0.01, 0.15),
) -> pd.DataFrame:
    """Create a synthetic bets DataFrame for testing."""
    rng = np.random.default_rng(seed)
    edges = rng.uniform(*edge_range, size=n)
    model_probs = rng.uniform(0.45, 0.85, size=n)
    odds = 1.0 / model_probs + rng.uniform(-0.1, 0.1, size=n)
    odds = np.clip(odds, 1.1, 10.0)
    won = (rng.random(n) < model_probs).astype(int)
    stake_pct = np.full(n, 0.02)
    profit_pct = np.where(won, stake_pct * (odds - 1), -stake_pct)

    return pd.DataFrame({
        "edge": edges,
        "model_prob": model_probs,
        "odds": odds,
        "won": won,
        "stake_pct": stake_pct,
        "profit_pct": profit_pct,
        "side": rng.choice(["Over", "Under"], size=n),
        "confidence": rng.choice(["high", "medium", "low"], size=n),
        "season": rng.choice([22, 23, 24], size=n),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Bucket Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeBucketAnalysis:
    """Test edge_bucket_analysis function."""

    def test_returns_expected_columns(self) -> None:
        df = _make_bets_df()
        result = edge_bucket_analysis(df)
        expected_cols = {"bucket", "n_bets", "win_rate", "roi",
                         "avg_edge", "avg_odds", "profit"}
        assert expected_cols == set(result.columns)

    def test_buckets_cover_all_bets(self) -> None:
        df = _make_bets_df()
        result = edge_bucket_analysis(df)
        assert result["n_bets"].sum() == len(df)

    def test_win_rate_between_0_and_1(self) -> None:
        df = _make_bets_df()
        result = edge_bucket_analysis(df)
        assert (result["win_rate"] >= 0).all()
        assert (result["win_rate"] <= 1).all()

    def test_empty_df_returns_empty(self) -> None:
        result = edge_bucket_analysis(pd.DataFrame())
        assert result.empty

    def test_single_bucket_all_wins(self) -> None:
        """All bets in same bucket, all winning."""
        df = pd.DataFrame({
            "edge": [0.03, 0.035, 0.031],
            "won": [1, 1, 1],
            "odds": [2.0, 2.1, 1.9],
            "stake_pct": [0.02, 0.02, 0.02],
            "profit_pct": [0.02, 0.022, 0.018],
        })
        result = edge_bucket_analysis(df)
        assert len(result) == 1
        assert result.iloc[0]["bucket"] == "2-4%"
        assert result.iloc[0]["win_rate"] == 1.0

    def test_roi_positive_for_all_winners(self) -> None:
        """If all bets win, ROI should be positive."""
        df = pd.DataFrame({
            "edge": [0.05, 0.06, 0.055],
            "won": [1, 1, 1],
            "odds": [2.0, 2.5, 2.2],
            "stake_pct": [0.02, 0.02, 0.02],
            "profit_pct": [0.02, 0.03, 0.024],
        })
        result = edge_bucket_analysis(df)
        assert (result["roi"] > 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration Curve
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrationCurve:
    """Test calibration_curve function."""

    def test_returns_expected_columns(self) -> None:
        df = _make_bets_df(n=200)
        result = calibration_curve(df)
        expected = {"bin_label", "bin_mid", "predicted", "actual",
                    "n_bets", "gap"}
        assert expected == set(result.columns)

    def test_gap_is_actual_minus_predicted(self) -> None:
        df = _make_bets_df(n=200)
        result = calibration_curve(df)
        if not result.empty:
            np.testing.assert_allclose(
                result["gap"].values,
                (result["actual"] - result["predicted"]).values,
                atol=1e-10,
            )

    def test_minimum_sample_size_enforced(self) -> None:
        """Bins with fewer than 3 bets are excluded."""
        df = pd.DataFrame({
            "model_prob": [0.50, 0.51],  # Only 2 bets in 50-55% bin
            "won": [1, 0],
        })
        result = calibration_curve(df)
        assert result.empty

    def test_empty_df_returns_empty(self) -> None:
        result = calibration_curve(pd.DataFrame())
        assert result.empty

    def test_missing_prob_col_returns_empty(self) -> None:
        df = pd.DataFrame({"won": [1, 0, 1]})
        result = calibration_curve(df, prob_col="model_prob")
        assert result.empty

    def test_perfect_calibration(self) -> None:
        """When all predictions in a bin match outcomes, gap ≈ 0."""
        # 10 bets all at ~62.5% probability, 6 winning = 60% actual
        df = pd.DataFrame({
            "model_prob": [0.61] * 10,
            "won": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        })
        result = calibration_curve(df)
        assert len(result) == 1
        assert abs(result.iloc[0]["gap"]) < 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Brier Score
# ═══════════════════════════════════════════════════════════════════════════════

class TestBrierScore:
    """Test brier_score function."""

    def test_perfect_predictions(self) -> None:
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(probs, outcomes) == 0.0

    def test_worst_predictions(self) -> None:
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1, 0, 1, 0])
        assert brier_score(probs, outcomes) == 1.0

    def test_uniform_predictions(self) -> None:
        """All 0.5 predictions yield Brier score 0.25."""
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        outcomes = np.array([1, 0, 1, 0])
        assert abs(brier_score(probs, outcomes) - 0.25) < 1e-10

    def test_returns_float(self) -> None:
        result = brier_score(np.array([0.7]), np.array([1]))
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Model Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerModelAccuracy:
    """Test per_model_accuracy function."""

    def test_returns_one_row_per_model(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "xgb_prob": rng.uniform(0.3, 0.8, n),
            "lgb_prob": rng.uniform(0.3, 0.8, n),
            "ensemble_prob": rng.uniform(0.3, 0.8, n),
            "actual": rng.choice([0, 1], n),
        })
        result = per_model_accuracy(df)
        assert len(result) == 3
        assert set(result["model"]) == {"XGBoost", "LightGBM", "Ensemble"}

    def test_empty_df_returns_empty(self) -> None:
        result = per_model_accuracy(pd.DataFrame())
        assert result.empty

    def test_missing_actual_returns_empty(self) -> None:
        df = pd.DataFrame({"xgb_prob": [0.5, 0.6]})
        result = per_model_accuracy(df)
        assert result.empty

    def test_brier_score_in_valid_range(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "xgb_prob": rng.uniform(0.3, 0.8, n),
            "actual": rng.choice([0, 1], n),
        })
        result = per_model_accuracy(df)
        assert (result["brier_score"] >= 0).all()
        assert (result["brier_score"] <= 1).all()


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceValidation:
    """Test confidence_validation function."""

    def test_returns_rows_per_level(self) -> None:
        df = _make_bets_df()
        result = confidence_validation(df)
        assert len(result) <= 3
        assert set(result["confidence"]).issubset({"high", "medium", "low"})

    def test_empty_df_returns_empty(self) -> None:
        result = confidence_validation(pd.DataFrame())
        assert result.empty

    def test_missing_confidence_col_returns_empty(self) -> None:
        df = pd.DataFrame({"won": [1, 0], "edge": [0.05, 0.03]})
        result = confidence_validation(df)
        assert result.empty

    def test_single_confidence_level(self) -> None:
        df = pd.DataFrame({
            "confidence": ["high"] * 5,
            "won": [1, 1, 0, 1, 0],
            "edge": [0.05] * 5,
            "odds": [2.0] * 5,
            "stake_pct": [0.02] * 5,
            "profit_pct": [0.02, 0.02, -0.02, 0.02, -0.02],
        })
        result = confidence_validation(df)
        assert len(result) == 1
        assert result.iloc[0]["win_rate"] == 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# Side Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestSideAnalysis:
    """Test side_analysis function."""

    def test_returns_rows_per_side(self) -> None:
        df = _make_bets_df()
        result = side_analysis(df)
        assert set(result["side"]) == {"Over", "Under"}

    def test_empty_df_returns_empty(self) -> None:
        result = side_analysis(pd.DataFrame())
        assert result.empty

    def test_missing_side_col_returns_empty(self) -> None:
        df = pd.DataFrame({"won": [1, 0]})
        result = side_analysis(df)
        assert result.empty

    def test_total_bets_match(self) -> None:
        df = _make_bets_df()
        result = side_analysis(df)
        assert result["n_bets"].sum() == len(df)


# ═══════════════════════════════════════════════════════════════════════════════
# Season Trends
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeasonTrends:
    """Test season_trends function."""

    def test_returns_row_per_season(self) -> None:
        df = _make_bets_df()
        result = season_trends(df)
        assert len(result) == 3  # seasons 22, 23, 24

    def test_bankroll_starts_positive(self) -> None:
        df = _make_bets_df()
        result = season_trends(df)
        assert (result["bankroll"] > 0).all()

    def test_empty_df_returns_empty(self) -> None:
        result = season_trends(pd.DataFrame())
        assert result.empty

    def test_missing_season_col_returns_empty(self) -> None:
        df = pd.DataFrame({"won": [1, 0]})
        result = season_trends(df)
        assert result.empty

    def test_year_format_correct(self) -> None:
        df = pd.DataFrame({
            "season": [23, 24],
            "won": [1, 0],
            "profit_pct": [0.02, -0.02],
            "stake_pct": [0.02, 0.02],
            "edge": [0.05, 0.03],
            "odds": [2.0, 1.9],
            "model_prob": [0.6, 0.55],
        })
        result = season_trends(df)
        assert result.iloc[0]["year"] == "2023/24"
        assert result.iloc[1]["year"] == "2024/25"

    def test_brier_computed_when_model_prob_present(self) -> None:
        df = _make_bets_df()
        result = season_trends(df)
        assert "brier" in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# Run Full Analytics
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunFullAnalytics:
    """Test run_full_analytics orchestration."""

    def test_returns_all_keys(self) -> None:
        df = _make_bets_df(n=200)
        result = run_full_analytics(df, verbose=False)
        expected_keys = {"edge_buckets", "calibration", "confidence",
                         "sides", "seasons"}
        assert expected_keys == set(result.keys())

    def test_all_values_are_dataframes(self) -> None:
        df = _make_bets_df(n=200)
        result = run_full_analytics(df, verbose=False)
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame), f"{key} is not a DataFrame"

    def test_empty_bets_returns_empty_results(self) -> None:
        result = run_full_analytics(pd.DataFrame(), verbose=False)
        for key, val in result.items():
            assert val.empty, f"{key} should be empty"
