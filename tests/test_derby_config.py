"""One derby list per league, in one place, matched exactly (ADR 0007 dec. 9).

There were four copies: two in the builder, one in `league_config` read by no
code at all, and one in `add_season.py`. The `league_config` copy was the
dangerous kind — right file, right name, plausible content, and **wrong**: 18
of its 29 EFL clubs use long forms ("Sheffield Wednesday") that do not exist in
a canonical holding football-data.co.uk short forms ("Sheffield Weds").

Wiring that copy up would have looked like fixing the duplication while
silently zeroing most EFL derby flags, and the cross-league guard would not
have caught it — a lower derby rate in one league looks like the legitimate
structural difference it already tolerates.

So the guard here is not a distribution test. It checks the thing that was
actually wrong: that every configured pair names a club the canonical knows.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from league_config import LEAGUES

LIST_KEYS = ("derbies_local", "derbies_historical")

# Clubs deliberately configured though they have not appeared in the division
# within the canonical's range. These are real pairings held ready, not typos:
# the distinction matters because a *wrongly formed* name ("Nottm Forest" for
# "Nott'm Forest") also never matches, and looks identical from here.
NOT_YET_IN_DIVISION: dict[str, set[str]] = {
    "EFL": {"Port Vale"},  # Potteries derby; Port Vale has stayed below E1
    "PL": set(),
}


def _canonical_names(league: str) -> set[str]:
    path = LEAGUES[league]["csv_path"]
    if not os.path.exists(path):
        pytest.skip(f"{league} canonical not present")
    df = pd.read_csv(path, usecols=["Home_Team", "Away_Team"], low_memory=False)
    return set(df["Home_Team"]) | set(df["Away_Team"])


@pytest.mark.parametrize("league", sorted(LEAGUES))
@pytest.mark.parametrize("key", LIST_KEYS)
def test_every_configured_club_exists_in_the_canonical(
        league: str, key: str) -> None:
    """A pair naming a club the canonical never uses can never match.

    With exact matching that is a silent no-op, not an error — which is why it
    survived unread for so long.
    """
    names = _canonical_names(league)
    configured = {club for pair in LEAGUES[league][key] for club in pair}
    unknown = sorted(configured - names - NOT_YET_IN_DIVISION[league])
    assert not unknown, (
        f"{league} {key} names clubs absent from the canonical: {unknown}. "
        f"With exact matching these pairs match nothing, silently. If the "
        f"club genuinely has not played in this division, add it to "
        f"NOT_YET_IN_DIVISION with a reason; otherwise the name is wrong.")


@pytest.mark.parametrize("league", sorted(LEAGUES))
def test_not_yet_in_division_entries_are_still_absent(league: str) -> None:
    """An allowance must not outlive what it allowed.

    Once a club appears in the division, its entry here is stale and hides the
    typo class this file exists to catch.
    """
    names = _canonical_names(league)
    arrived = sorted(NOT_YET_IN_DIVISION[league] & names)
    assert not arrived, (
        f"{league}: {arrived} now appear in the canonical — remove them from "
        f"NOT_YET_IN_DIVISION so the name check covers them again.")


@pytest.mark.parametrize("league", sorted(LEAGUES))
def test_derby_lists_are_pairs(league: str) -> None:
    """Each entry is a two-club tuple — order is irrelevant when matching."""
    for key in LIST_KEYS:
        for pair in LEAGUES[league][key]:
            assert isinstance(pair, tuple) and len(pair) == 2, (
                f"{league} {key} has a malformed entry: {pair!r}")
            assert pair[0] != pair[1], f"{league} {key} pairs a club with itself"


@pytest.mark.parametrize("league", sorted(LEAGUES))
def test_a_club_is_not_in_both_lists_for_the_same_opponent(
        league: str) -> None:
    """A fixture is local or historical, never both — the builder's
    `Historical Derby` column already suppresses one, so an overlap means the
    configuration disagrees with itself."""
    def _norm(pairs):
        return {tuple(sorted(p)) for p in pairs}

    overlap = _norm(LEAGUES[league]["derbies_local"]) & _norm(
        LEAGUES[league]["derbies_historical"])
    assert not overlap, f"{league} lists {sorted(overlap)} as both"


def test_builder_matching_is_exact_not_fuzzy() -> None:
    """A club whose name merely contains a configured one is not a derby.

    The old matcher accepted substring overlap in either direction, the same
    silent-failure mode as the odds-feed resolver that mapped Coventry City to
    Manchester City.
    """
    from data.build_canonical_dataset import _is_derby

    pairs = {("Arsenal FC", "Tottenham Hotspur FC")}
    assert _is_derby("Arsenal FC", "Tottenham Hotspur FC", pairs) is True
    assert _is_derby("Tottenham Hotspur FC", "Arsenal FC", pairs) is True, \
        "order must not matter"
    assert _is_derby("Arsenal", "Tottenham", pairs) is False, \
        "substring names must not match"
    assert _is_derby("Arsenal FC", "Chelsea FC", pairs) is False
