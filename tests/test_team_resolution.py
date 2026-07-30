"""Odds-feed team names must resolve to the right club, or to nothing.

The fuzzy fallback matched on any single shared word, and "City" is shared by
Manchester City, Leicester City, Hull City, Coventry City, Norwich City and
Stoke City. With the 2026/27 feed live, "Coventry City" resolved to
"Manchester City FC" — so `Arsenal v Coventry City` would have been priced,
staked and recommended as `Arsenal v Manchester City`.

That is worse than no match at all. A missing fixture is visible; a confident
prediction for the wrong fixture is not. These tests hold the line that
resolution is either correct or absent.
"""
from __future__ import annotations

import pytest

from api.odds_api import match_to_our_teams

# The 2025/26 Premier League — what `our_teams` holds until season 26 is
# built into the canonical.
PL_2025 = {
    "AFC Bournemouth", "Arsenal FC", "Aston Villa FC", "Brentford FC",
    "Brighton & Hove Albion FC", "Burnley FC", "Chelsea FC",
    "Crystal Palace FC", "Everton FC", "Fulham FC", "Leeds United FC",
    "Liverpool FC", "Manchester City FC", "Manchester United FC",
    "Newcastle United FC", "Nottingham Forest FC", "Sunderland AFC",
    "Tottenham Hotspur FC", "West Ham United FC",
    "Wolverhampton Wanderers FC",
}


def _resolve(name: str, teams: set[str] | None = None) -> str | None:
    """Resolve a single odds-feed name via the public matcher."""
    home, _ = match_to_our_teams(
        {"home_team": name, "away_team": "Arsenal"}, teams or PL_2025)
    return home


@pytest.mark.parametrize("promoted", ["Coventry City", "Hull City"])
def test_promoted_side_never_resolves_to_another_club(promoted: str) -> None:
    """The 2026/27 arrivals must not be mistaken for Manchester City."""
    got = _resolve(promoted)
    assert got != "Manchester City FC", (
        f"{promoted!r} resolved to {got!r} — a fixture would be priced and "
        f"staked as the wrong match entirely.")


@pytest.mark.parametrize("api_name,expected", [
    ("Coventry City", "Coventry City FC"),
    ("Hull City", "Hull City AFC"),
    ("Ipswich Town", "Ipswich Town FC"),
])
def test_explicit_mapping_wins_before_the_our_teams_check(
        api_name: str, expected: str) -> None:
    """A promoted side maps to its canonical name though it is not yet known.

    Downstream logs "no recent data" and skips, so the fixture is passed over
    until season 26 is built — and then starts working with no code change.
    """
    assert _resolve(api_name) == expected


def test_generic_word_alone_is_not_a_match() -> None:
    """A name sharing only a generic word resolves to nothing."""
    assert _resolve("Some City", {"Manchester City FC"}) is None
    assert _resolve("Random United", {"Manchester United FC"}) is None


def test_generic_words_still_break_ties_once_qualified() -> None:
    """"manchester" qualifies both Manchester clubs; "city" then decides."""
    assert _resolve("Manchester City", PL_2025) == "Manchester City FC"
    assert _resolve("Manchester United", PL_2025) == "Manchester United FC"


def test_ambiguous_match_returns_nothing() -> None:
    """Two candidates tied on overlap is not a resolution."""
    teams = {"Sheffield United FC", "Sheffield Wednesday FC"}
    assert _resolve("Sheffield", teams) is None


def test_every_current_pl_club_still_resolves() -> None:
    """Regression: the fix must not cost a single working resolution."""
    from api.odds_api import _ODDS_API_TO_DATASET

    unresolved = {
        api_name: _resolve(api_name)
        for api_name, canonical in _ODDS_API_TO_DATASET.items()
        if canonical in PL_2025 and _resolve(api_name) != canonical
    }
    assert not unresolved, f"regressed: {unresolved}"


# ─────────────────────────────────────────────────────────────────────────────
# OddsPapi — the same feed problem through a second resolver
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("api_name,expected", [
    ("Coventry City", "Coventry City FC"),
    ("Hull City", "Hull City AFC"),
    # OddsPapi drops the suffix, so the "Ipswich Town FC" key never matched.
    ("Ipswich Town", "Ipswich Town FC"),
])
def test_oddspapi_resolves_the_promoted_sides(
        api_name: str, expected: str) -> None:
    from api.oddspapi import map_team
    assert map_team(api_name, PL_2025) == expected


def test_oddspapi_does_not_guess_on_a_generic_word() -> None:
    from api.oddspapi import map_team
    assert map_team("Some City", {"Manchester City FC"}) is None


def test_oddspapi_still_resolves_the_current_clubs() -> None:
    from api.oddspapi import map_team, _ODDSPAPI_TO_DATASET

    unresolved = {
        api_name: map_team(api_name, PL_2025)
        for api_name, canonical in _ODDSPAPI_TO_DATASET.items()
        if canonical in PL_2025 and map_team(api_name, PL_2025) != canonical
    }
    assert not unresolved, f"regressed: {unresolved}"


def test_both_resolvers_share_one_generic_word_list() -> None:
    """Two copies would be two implementations of one contract (ADR 0007)."""
    import api.odds_api as odds_api
    import api.oddspapi as oddspapi

    assert oddspapi._resolve_by_overlap is odds_api._resolve_by_overlap
