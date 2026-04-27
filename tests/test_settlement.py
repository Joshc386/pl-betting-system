"""Tests for settlement._determine_outcome() — the most critical function in the system.

A bug here means wrong P/L calculations, so we test all market types and edge cases.
"""
import pytest

from settlement import _determine_outcome


class TestDetermineOutcomeOU25:
    """Standard O/U 2.5 market."""

    def test_over_25_with_3_goals(self) -> None:
        won, result = _determine_outcome("ou25", "over", 2, 1)
        assert won is True
        assert "3 goals" in result
        assert "over 2.5" in result

    def test_under_25_with_2_goals(self) -> None:
        won, result = _determine_outcome("ou25", "under", 1, 1)
        assert won is True
        assert "2 goals" in result
        assert "under 2.5" in result

    def test_over_25_with_2_goals_loses(self) -> None:
        won, result = _determine_outcome("ou25", "over", 1, 1)
        assert won is False

    def test_under_25_with_4_goals_loses(self) -> None:
        won, result = _determine_outcome("ou25", "under", 3, 1)
        assert won is False

    def test_zero_goals(self) -> None:
        won, result = _determine_outcome("ou25", "under", 0, 0)
        assert won is True
        assert "0 goals" in result


class TestDetermineOutcomeAltLines:
    """Alternative O/U lines (1.5, 3.5, 4.5)."""

    def test_over_15_with_2_goals(self) -> None:
        won, _ = _determine_outcome("ou15", "over", 1, 1)
        assert won is True

    def test_over_15_with_1_goal_loses(self) -> None:
        won, _ = _determine_outcome("ou15", "over", 1, 0)
        assert won is False

    def test_under_15_with_1_goal(self) -> None:
        won, _ = _determine_outcome("ou15", "under", 0, 1)
        assert won is True

    def test_over_35_with_4_goals(self) -> None:
        won, result = _determine_outcome("ou35", "over", 2, 2)
        assert won is True
        assert "3.5" in result

    def test_over_35_with_3_goals_loses(self) -> None:
        won, _ = _determine_outcome("ou35", "over", 2, 1)
        assert won is False

    def test_under_35_with_3_goals(self) -> None:
        won, _ = _determine_outcome("ou35", "under", 2, 1)
        assert won is True

    def test_over_45_with_5_goals(self) -> None:
        won, result = _determine_outcome("ou45", "over", 3, 2)
        assert won is True
        assert "4.5" in result

    def test_over_45_with_4_goals_loses(self) -> None:
        won, _ = _determine_outcome("ou45", "over", 2, 2)
        assert won is False

    def test_line_parsing_in_result_string(self) -> None:
        """Result string should show the correct line value."""
        _, result = _determine_outcome("ou15", "over", 2, 0)
        assert "1.5" in result
        _, result = _determine_outcome("ou35", "over", 3, 1)
        assert "3.5" in result
        _, result = _determine_outcome("ou45", "under", 1, 0)
        assert "4.5" in result


class TestDetermineOutcomeBTTS:
    """Both Teams To Score market."""

    def test_btts_yes_both_score(self) -> None:
        won, result = _determine_outcome("btts", "yes", 1, 1)
        assert won is True
        assert "BTTS Yes" in result

    def test_btts_no_one_blanks(self) -> None:
        won, result = _determine_outcome("btts", "no", 2, 0)
        assert won is True
        assert "BTTS No" in result

    def test_btts_yes_when_one_blanks_loses(self) -> None:
        won, _ = _determine_outcome("btts", "yes", 3, 0)
        assert won is False

    def test_btts_no_when_both_score_loses(self) -> None:
        won, _ = _determine_outcome("btts", "no", 1, 2)
        assert won is False

    def test_btts_high_scoring(self) -> None:
        won, _ = _determine_outcome("btts", "yes", 4, 3)
        assert won is True


class TestDetermineOutcomeEdgeCases:
    """Edge cases and error handling."""

    def test_unknown_market_returns_false(self) -> None:
        won, result = _determine_outcome("corners", "over", 1, 1)
        assert won is False
        assert "unknown market" in result

    def test_malformed_ou_market_defaults_to_25(self) -> None:
        """If market code can't be parsed, defaults to 2.5."""
        won, result = _determine_outcome("ouXYZ", "over", 2, 1)
        assert won is True  # 3 goals > 2.5
        assert "2.5" in result

    def test_result_string_shows_score(self) -> None:
        _, result = _determine_outcome("ou25", "over", 3, 2)
        assert "3-2" in result
