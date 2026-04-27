"""Tests for per-bookmaker odds extraction and Championship team resolution.

Covers:
  - _extract_bookmaker_odds for O/U and BTTS markets
  - _resolve_champ_team for newly promoted teams
"""
import pytest

from predict import _extract_bookmaker_odds


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Bookmaker Odds Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractBookmakerOdds:
    """Test _extract_bookmaker_odds function."""

    def _make_match(self) -> dict:
        """Create a realistic match dict from the odds API."""
        return {
            "bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "over": 1.85,
                    "under": 2.05,
                    "all_lines": {
                        2.5: {"over": 1.85, "under": 2.05},
                        1.5: {"over": 1.25, "under": 3.60},
                    },
                },
                "williamhill": {
                    "title": "William Hill",
                    "over": 1.90,
                    "under": 2.00,
                    "all_lines": {
                        2.5: {"over": 1.90, "under": 2.00},
                    },
                },
                "pinnacle": {
                    "title": "Pinnacle",
                    "over": 1.88,
                    "under": 2.02,
                    "all_lines": {
                        2.5: {"over": 1.88, "under": 2.02},
                        1.5: {"over": 1.28, "under": 3.50},
                    },
                },
            },
            "btts_bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "yes": 1.75,
                    "no": 2.10,
                },
                "pinnacle": {
                    "title": "Pinnacle",
                    "yes": 1.80,
                    "no": 2.05,
                },
            },
        }

    def test_ou25_over_extracts_all_books(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou25", "over")
        assert "Bet365" in result
        assert "William Hill" in result
        assert "Pinnacle" in result
        assert result["Bet365"] == 1.85

    def test_ou25_under_extracts_all_books(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou25", "under")
        assert result["Bet365"] == 2.05
        assert result["William Hill"] == 2.00

    def test_ou15_extracts_from_all_lines(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou15", "over")
        # Only bet365 and pinnacle have 1.5 line
        assert "Bet365" in result
        assert "Pinnacle" in result
        assert "William Hill" not in result
        assert result["Bet365"] == 1.25

    def test_btts_yes_extracts(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "btts", "yes")
        assert result["Bet365"] == 1.75
        assert result["Pinnacle"] == 1.80

    def test_btts_no_extracts(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "btts", "no")
        assert result["Bet365"] == 2.10
        assert result["Pinnacle"] == 2.05

    def test_empty_match_returns_empty(self) -> None:
        result = _extract_bookmaker_odds({}, "ou25", "over")
        assert result == {}

    def test_string_line_keys_handled(self) -> None:
        """all_lines with string keys (e.g. '2.5') should be found."""
        match = {
            "bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "over": 1.85,
                    "under": 2.05,
                    "all_lines": {
                        "2.5": {"over": 1.85, "under": 2.05},
                    },
                },
            },
            "btts_bookmakers": {},
        }
        result = _extract_bookmaker_odds(match, "ou25", "over")
        assert result["Bet365"] == 1.85


# ═══════════════════════════════════════════════════════════════════════════════
# Championship Team Resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestChampTeamResolution:
    """Test _resolve_champ_team for newly promoted teams."""

    def test_known_team_in_dataset(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn", "Stoke", "Derby"}
        assert _resolve_champ_team("Blackburn Rovers", our_teams) == "Blackburn"

    def test_newly_promoted_team_not_in_dataset(self) -> None:
        """Newly promoted teams should still resolve even if not in our_teams."""
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn", "Stoke", "Derby"}
        # These teams have explicit mappings but aren't in our_teams
        assert _resolve_champ_team("Southampton", our_teams) == "Southampton"
        assert _resolve_champ_team("Leicester City", our_teams) == "Leicester"
        assert _resolve_champ_team("Ipswich Town", our_teams) == "Ipswich"
        assert _resolve_champ_team("Birmingham City", our_teams) == "Birmingham"
        assert _resolve_champ_team("Charlton Athletic", our_teams) == "Charlton"
        assert _resolve_champ_team("Wrexham AFC", our_teams) == "Wrexham"

    def test_direct_match(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Middlesbrough", "Portsmouth"}
        assert _resolve_champ_team("Middlesbrough", our_teams) == "Middlesbrough"

    def test_unknown_team_returns_none(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn"}
        assert _resolve_champ_team("Tottenham Hotspur", our_teams) is None
