"""Settlement matches bets on the team names ESPN gives back.

settlement.py builds a (home_team, away_team) -> result lookup from ESPN and
looks up each open bet by the same key. A name that resolves to the wrong
form does not raise — the key simply never matches, and the bet sits
unsettled with nothing logged.

The Championship half had a real hole: an EFL club missing from
_ESPN_TO_CHAMP fell through to normalize(), which returns Premier League
canonical names ("Blackburn Rovers FC") while the EFL database stores
football-data.co.uk short forms ("Blackburn"). Every such club would have
been unsettleable.
"""
from __future__ import annotations

import pytest

from api.espn_scores import _ESPN_TO_CHAMP, _ESPN_TO_PL, _resolve_team


@pytest.mark.parametrize("espn_name,expected", [
    ("Wrexham", "Wrexham"),
    ("Blackburn Rovers", "Blackburn"),
    ("Sheffield Wednesday", "Sheffield Weds"),
    ("Queens Park Rangers", "QPR"),
    ("West Bromwich Albion", "West Brom"),
    # 2026/27 arrivals: a club with no canonical history, and one that is
    # also in _ESPN_TO_PL — the ELC path must answer in the short form.
    ("Lincoln City", "Lincoln"),
    ("West Ham United", "West Ham"),
])
def test_a_mapped_efl_club_resolves_to_its_short_form(
        espn_name: str, expected: str) -> None:
    """The EFL database keys on football-data.co.uk short forms."""
    assert _resolve_team(espn_name, "ELC") == expected


@pytest.mark.parametrize("espn_name", [
    "Plymouth Argyle",
    "Rotherham United",
    "Luton Town",
])
def test_an_unmapped_efl_club_resolves_to_nothing(espn_name: str) -> None:
    """Not a Premier League long name — those cannot match the EFL database.

    normalize() would return "Plymouth Argyle FC" here. The EFL database
    holds "Plymouth", so the bet would never settle and nothing would say so.
    Returning None lets the caller skip the match and log it.
    """
    assert espn_name not in _ESPN_TO_CHAMP, (
        f"{espn_name!r} is mapped now — pick an unmapped club for this test")
    assert _resolve_team(espn_name, "ELC") is None


@pytest.mark.parametrize("espn_name,expected", [
    ("Arsenal", "Arsenal FC"),
    ("Manchester City", "Manchester City FC"),
    ("Nottingham Forest", "Nottingham Forest FC"),
])
def test_a_pl_club_resolves_to_its_canonical_name(
        espn_name: str, expected: str) -> None:
    """The PL database uses the long canonical form."""
    assert _resolve_team(espn_name, "PL") == expected


@pytest.mark.parametrize("espn_name", [
    "Bristol Rovers",
    "Manchester",
    "Real Madrid",
])
def test_an_unrecognised_name_resolves_to_nothing(espn_name: str) -> None:
    """A cup or friendly opponent is not a fixture we hold bets on."""
    assert _resolve_team(espn_name, "PL") is None
    assert _resolve_team(espn_name, "ELC") is None


def test_every_mapped_club_resolves_in_its_own_competition() -> None:
    """Regression: the guard must not cost a working resolution."""
    broken = {
        name: _resolve_team(name, comp)
        for mapping, comp in ((_ESPN_TO_PL, "PL"), (_ESPN_TO_CHAMP, "ELC"))
        for name, expected in mapping.items()
        if _resolve_team(name, comp) != expected
    }
    assert not broken, f"regressed: {broken}"
