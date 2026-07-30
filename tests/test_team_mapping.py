"""A team name must stay a team name.

normalize() used to fall back to substring matching over its alias table, and
the table held three-letter codes. "Bristol Rovers" contains "sto", so it
normalised to Stockport County; "Manchester" contains "che", so it normalised
to Chelsea. Splitting a club's name into fragments and matching a different
club on one of them is never a correct answer.

These tests hold two lines: a name the table knows resolves to its canonical
form, and a name the table does not know comes back untouched.
"""
from __future__ import annotations

import pytest

from api.team_mapping import _ALIASES, normalize


# ─────────────────────────────────────────────────────────────────────────────
# An unknown name is not an invitation to guess
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("unknown", [
    "Bristol Rovers",      # was Stockport County FC, via "bri-STO-l"
    "Manchester",          # was Chelsea FC, via "man-CHE-ster"
    "Forest Green Rovers",
    "Accrington Stanley",
    "Sheffield",           # a city, not a club
    "Wednesday",
    "City",
    "United",
])
def test_an_unrecognised_name_is_returned_unchanged(unknown: str) -> None:
    """No fragment of a name may resolve it to some other club."""
    assert normalize(unknown) == unknown


def test_no_name_resolves_to_a_club_it_does_not_name() -> None:
    """The two cases that made this worth fixing."""
    assert normalize("Bristol Rovers") != "Stockport County FC"
    assert normalize("Manchester") != "Chelsea FC"


# ─────────────────────────────────────────────────────────────────────────────
# A genuine alias is one a human would recognise
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alias,expected", [
    ("Man City", "Manchester City FC"),
    ("Man United", "Manchester United FC"),
    ("Man Utd", "Manchester United FC"),
    ("Spurs", "Tottenham Hotspur FC"),
    ("Tottenham", "Tottenham Hotspur FC"),
    ("Wolves", "Wolverhampton Wanderers FC"),
    ("West Brom", "West Bromwich Albion FC"),
    ("West Ham", "West Ham United FC"),
    ("Nott'm Forest", "Nottingham Forest FC"),
    ("Sheffield Weds", "Sheffield Wednesday FC"),
    ("Sheff Wed", "Sheffield Wednesday FC"),
    ("QPR", "Queens Park Rangers FC"),
    ("Peterboro", "Peterborough United FC"),
    ("Brighton", "Brighton & Hove Albion FC"),
    ("MK Dons", "MK Dons FC"),
])
def test_a_genuine_alias_resolves(alias: str, expected: str) -> None:
    """Real short names that real data sources actually send."""
    assert normalize(alias) == expected


@pytest.mark.parametrize("variant,expected", [
    ("Arsenal", "Arsenal FC"),
    ("Arsenal FC", "Arsenal FC"),
    ("Bournemouth", "AFC Bournemouth"),
    ("AFC Bournemouth", "AFC Bournemouth"),
    ("Bournemouth AFC", "AFC Bournemouth"),
    ("Sunderland", "Sunderland AFC"),
    ("Sunderland FC", "Sunderland AFC"),
    ("Swansea City", "Swansea City AFC"),
    ("Chelsea CF", "Chelsea FC"),
    ("Huddersfield Town", "Huddersfield Town AFC"),
])
def test_the_club_suffix_is_not_part_of_the_name(
        variant: str, expected: str) -> None:
    """FC, AFC and CF are decoration — present or absent, it is one club."""
    assert normalize(variant) == expected


def test_a_promoted_team_asterisk_is_still_stripped() -> None:
    """Some source files mark promoted sides with a trailing asterisk."""
    assert normalize("Chelsea *") == "Chelsea FC"
    assert normalize("  Arsenal  ") == "Arsenal FC"


# ─────────────────────────────────────────────────────────────────────────────
# The table itself
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("ARS", "Arsenal FC"),
    ("MCI", "Manchester City FC"),
    ("MUN", "Manchester United FC"),
    ("CHE", "Chelsea FC"),
    ("BUR", "Burnley FC"),
    ("STO", "Stockport County FC"),
    ("MIL", "Millwall FC"),
    ("QPR", "Queens Park Rangers FC"),
])
def test_a_short_code_names_its_own_club(code: str, expected: str) -> None:
    """A code is a whole name for one club and resolves to that club."""
    assert normalize(code) == expected


@pytest.mark.parametrize("long_name,code_inside", [
    ("Bristol Rovers", "STO"),        # bri-STO-l
    ("Manchester", "CHE"),            # man-CHE-ster
    ("Blackburn Rovers", "BUR"),      # black-BUR-n
    ("Milton Keynes Dons", "MIL"),
    ("Huddersfield", "DER"),
])
def test_a_code_inside_a_longer_name_is_not_a_match(
        long_name: str, code_inside: str) -> None:
    """A club's name is not a bag of fragments to search for codes in.

    Every short code sits inside some real club's name. Matching one there is
    what turned "Bristol Rovers" into Stockport County — the code is a name
    for its own club, never a piece of another's.
    """
    other_club = normalize(code_inside)
    assert normalize(long_name) != other_club, (
        f"{long_name!r} was resolved to {other_club!r} by finding "
        f"{code_inside!r} inside it.")


def test_every_club_keeps_a_code_where_one_is_free() -> None:
    """The codes are part of the table, not something to quietly drop."""
    coded = {
        canonical for canonical, aliases in _ALIASES.items()
        if any(a.isupper() and len(a) <= 4 for a in aliases)
    }
    # Southend's code (SOU) belongs to Southampton; it is the only club
    # without one, and that is recorded in the table.
    missing = sorted(set(_ALIASES) - coded)
    assert missing == ["Southend United FC"], f"clubs missing a code: {missing}"


def test_every_canonical_name_resolves_to_itself() -> None:
    """The canonical form is a fixed point."""
    wrong = {c: normalize(c) for c in _ALIASES if normalize(c) != c}
    assert not wrong, f"canonical names that do not round-trip: {wrong}"


def test_every_alias_resolves_to_its_canonical() -> None:
    """Nothing in the table may point at the wrong club."""
    wrong = {
        alias: (normalize(alias), canonical)
        for canonical, aliases in _ALIASES.items()
        for alias in aliases
        if normalize(alias) != canonical
    }
    assert not wrong, f"aliases resolving elsewhere: {wrong}"


def test_no_two_clubs_share_a_lookup_key() -> None:
    """An ambiguous table would reintroduce the guess by another route."""
    from api.team_mapping import _lookup_key

    seen: dict[str, str] = {}
    clashes = []
    for canonical, aliases in _ALIASES.items():
        for name in (canonical, *aliases):
            key = _lookup_key(name)
            if seen.setdefault(key, canonical) != canonical:
                clashes.append((key, seen[key], canonical))
    assert not clashes, f"one key, two clubs: {clashes}"
