"""Tests for alt_lines.py — goal distribution, probability, and settlement logic."""
import pytest
import numpy as np

from alt_lines import (
    build_goal_matrix,
    total_goals_distribution,
    prob_over_line,
    settle_asian,
)


class TestBuildGoalMatrix:
    """Tests for bivariate Poisson goal matrix."""

    def test_matrix_sums_to_one(self) -> None:
        mat = build_goal_matrix(1.5, 1.2)
        assert abs(mat.sum() - 1.0) < 1e-10

    def test_matrix_shape(self) -> None:
        mat = build_goal_matrix(1.5, 1.2, max_goals=8)
        assert mat.shape == (9, 9)

    def test_all_probabilities_non_negative(self) -> None:
        mat = build_goal_matrix(1.5, 1.2, rho=-0.13)
        assert (mat >= 0).all()

    def test_rho_zero_is_independent_poisson(self) -> None:
        mat = build_goal_matrix(1.5, 1.2, rho=0.0)
        mat_rho = build_goal_matrix(1.5, 1.2, rho=-0.13)
        # With rho=0, matrix should differ from rho=-0.13
        assert not np.allclose(mat, mat_rho)

    def test_dc_correction_affects_low_scores(self) -> None:
        """DC tau correction changes P(0,0), P(0,1), P(1,0), P(1,1)."""
        mat_ind = build_goal_matrix(1.5, 1.2, rho=0.0)
        mat_dc = build_goal_matrix(1.5, 1.2, rho=-0.13)
        # Low score cells should differ
        assert mat_ind[0][0] != pytest.approx(mat_dc[0][0])
        assert mat_ind[0][1] != pytest.approx(mat_dc[0][1])
        # Higher score cells should be approximately the same (tau=1)
        assert mat_ind[3][3] == pytest.approx(mat_dc[3][3], rel=0.01)

    def test_tiny_lambda_handled(self) -> None:
        """Lambda near zero shouldn't crash or produce NaN."""
        mat = build_goal_matrix(0.001, 0.001)
        assert mat.sum() == pytest.approx(1.0, abs=1e-6)
        assert not np.isnan(mat).any()


class TestTotalGoalsDistribution:
    """Tests for converting goal matrix to total goals distribution."""

    def test_distribution_sums_to_one(self) -> None:
        mat = build_goal_matrix(1.5, 1.2)
        dist = total_goals_distribution(mat)
        assert abs(sum(dist.values()) - 1.0) < 1e-10

    def test_distribution_keys(self) -> None:
        mat = build_goal_matrix(1.5, 1.2, max_goals=5)
        dist = total_goals_distribution(mat)
        assert 0 in dist
        assert 10 in dist  # 2 * max_goals


class TestProbOverLine:
    """Tests for P(Over) calculations across line types."""

    def setup_method(self) -> None:
        mat = build_goal_matrix(1.5, 1.2)
        self.dist = total_goals_distribution(mat)

    def test_half_line_no_push(self) -> None:
        result = prob_over_line(self.dist, 2.5)
        assert result["type"] == "standard"
        assert result["p_push"] == 0.0
        assert result["p_over"] + result["p_under"] == pytest.approx(1.0)

    def test_half_line_probs_in_range(self) -> None:
        result = prob_over_line(self.dist, 2.5)
        assert 0 < result["p_over"] < 1
        assert 0 < result["p_under"] < 1

    def test_whole_line_has_push(self) -> None:
        result = prob_over_line(self.dist, 2.0)
        assert result["type"] == "whole"
        assert result["p_push"] > 0
        assert (result["p_over"] + result["p_under"] + result["p_push"]
                == pytest.approx(1.0))

    def test_quarter_line_is_asian(self) -> None:
        result = prob_over_line(self.dist, 2.25)
        assert result["type"] == "asian_quarter"
        assert "lower_leg" in result
        assert "upper_leg" in result

    def test_higher_line_less_likely_over(self) -> None:
        r15 = prob_over_line(self.dist, 1.5)
        r25 = prob_over_line(self.dist, 2.5)
        r35 = prob_over_line(self.dist, 3.5)
        assert r15["p_over"] > r25["p_over"] > r35["p_over"]

    def test_over_05_nearly_certain(self) -> None:
        """With lambda 1.5+1.2, P(Over 0.5) should be very high."""
        result = prob_over_line(self.dist, 0.5)
        assert result["p_over"] > 0.9


class TestSettleAsian:
    """Tests for Asian handicap settlement."""

    def test_half_line_over_win(self) -> None:
        result = settle_asian("over", 2.5, 3, 1.90)
        assert result["outcome"] == "win"
        assert result["profit_loss"] == pytest.approx(0.90)

    def test_half_line_over_loss(self) -> None:
        result = settle_asian("over", 2.5, 2, 1.90)
        assert result["outcome"] == "loss"
        assert result["profit_loss"] == pytest.approx(-1.0)

    def test_half_line_under_win(self) -> None:
        result = settle_asian("under", 2.5, 2, 2.10)
        assert result["outcome"] == "win"
        assert result["profit_loss"] == pytest.approx(1.10)

    def test_whole_line_push(self) -> None:
        result = settle_asian("over", 3.0, 3, 1.90)
        assert result["outcome"] == "push"
        assert result["profit_loss"] == 0.0

    def test_whole_line_win(self) -> None:
        result = settle_asian("over", 3.0, 4, 1.90)
        assert result["outcome"] == "win"

    def test_whole_line_loss(self) -> None:
        result = settle_asian("over", 3.0, 2, 1.90)
        assert result["outcome"] == "loss"

    def test_quarter_line_half_win(self) -> None:
        """2.25 = half on 2.0 + half on 2.5. If 3 goals:
        - Leg 1 (2.0): over wins -> +0.45
        - Leg 2 (2.5): over wins -> +0.45
        -> Full win
        """
        result = settle_asian("over", 2.25, 3, 1.90)
        assert result["outcome"] == "win"

    def test_quarter_line_225_over_with_2_goals(self) -> None:
        """2.25 = half on 2.0 + half on 2.5. If 2 goals:
        - Leg 1 (2.0): push -> 0
        - Leg 2 (2.5): loss -> -0.50
        -> half_loss
        """
        result = settle_asian("over", 2.25, 2, 1.90)
        assert result["outcome"] == "half_loss"
        assert result["profit_loss"] == pytest.approx(-0.50)

    def test_quarter_line_275_over_with_3_goals(self) -> None:
        """2.75 = half on 2.5 + half on 3.0. If 3 goals:
        - Leg 1 (2.5): over wins -> +0.45
        - Leg 2 (3.0): push -> 0
        -> half_win
        """
        result = settle_asian("over", 2.75, 3, 1.90)
        assert result["outcome"] == "half_win"
        assert result["profit_loss"] == pytest.approx(0.45)

    def test_custom_stake(self) -> None:
        result = settle_asian("over", 2.5, 3, 2.00, stake=10.0)
        assert result["profit_loss"] == pytest.approx(10.0)
