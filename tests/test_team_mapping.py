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


@pytest.mark.parametrize("name,expected", [
    ("Wrexham", "Wrexham AFC"),        # Championship from 2025/26
    ("Wrexham AFC", "Wrexham AFC"),    # how the odds feeds spell it
    ("Tranmere", "Tranmere Rovers FC"),
])
def test_clubs_the_table_had_never_been_taught(
        name: str, expected: str) -> None:
    """Both sit in the canonical datasets and neither used to resolve.

    Wrexham is the live one: in the EFL canonical for season 25 and in the
    Championship odds mapping, while `normalize()` handed its name straight
    back. `efl_alt_lines_data` keys Betfair fixtures on `normalize(short_name)`,
    so Wrexham's fixtures were dropping out of that merge unnoticed.
    """
    assert normalize(name) == expected


def test_every_team_in_the_canonical_datasets_resolves() -> None:
    """The check that would have caught Wrexham.

    A club present in a canonical dataset but absent from this table is the
    gap that promotion produces every season.
    """
    import os

    import pandas as pd

    from league_config import get_league_config

    paths = [get_league_config(lg)["csv_path"] for lg in ("PL", "EFL")]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("canonical datasets not present in this checkout")

    names: set[str] = set()
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in ("Home_Team", "Away_Team"),
                         low_memory=False)
        for col in df.columns:
            names |= set(df[col].dropna().astype(str))

    unknown = sorted(n for n in names if normalize(n) not in _ALIASES)
    assert not unknown, (
        f"in a canonical dataset but not in the team table: {unknown}")


# ─────────────────────────────────────────────────────────────────────────────
# Saying "I don't know this name"
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,known", [
    ("Arsenal", True),
    ("Arsenal FC", True),
    ("Man Utd", True),
    ("MCI", True),
    ("Wrexham", True),
    ("Manchester", False),      # a city; no club is called this
    ("Bristol Rovers", False),  # a real club this table has never held
    ("Sheffield", False),
    ("", False),
])
def test_is_known_team_reports_whether_a_name_resolves(
        name: str, known: bool) -> None:
    """Callers need to distinguish a resolved name from an unrecognised one.

    normalize() returns the input unchanged when it cannot resolve it, so the
    return value alone cannot answer this: "Manchester" comes back looking
    exactly like a club's canonical name.
    """
    from api.team_mapping import is_known_team

    assert is_known_team(name) is known


def test_an_unrecognised_name_is_logged(caplog) -> None:
    """An unknown name is a gap in the table and must not pass silently.

    It reaches the canonical datasets, the pipeline merge keys and settlement,
    where it reads as a team name and quietly matches nothing.
    """
    import logging

    from api import team_mapping

    team_mapping._reset_unknown_name_log()
    with caplog.at_level(logging.WARNING, logger="api.team_mapping"):
        team_mapping.normalize("Bristol Rovers")

    assert any("Bristol Rovers" in r.message for r in caplog.records), (
        "an unrecognised name produced no warning")


def test_a_recognised_name_is_not_logged(caplog) -> None:
    """Normal traffic must not fill the log."""
    import logging

    from api import team_mapping

    team_mapping._reset_unknown_name_log()
    with caplog.at_level(logging.WARNING, logger="api.team_mapping"):
        for name in ("Arsenal", "Man Utd", "Wrexham AFC", "MCI"):
            team_mapping.normalize(name)

    assert not caplog.records, f"unexpected warnings: {caplog.records}"


def test_an_unrecognised_name_is_logged_once(caplog) -> None:
    """pipeline.py normalises whole columns, so one warning per name, not row."""
    import logging

    from api import team_mapping

    team_mapping._reset_unknown_name_log()
    with caplog.at_level(logging.WARNING, logger="api.team_mapping"):
        for _ in range(50):
            team_mapping.normalize("Bristol Rovers")

    assert len(caplog.records) == 1, (
        f"{len(caplog.records)} warnings for one repeated name")


# ─────────────────────────────────────────────────────────────────────────────
# The boundaries that must refuse an unknown name
# ─────────────────────────────────────────────────────────────────────────────

def test_assert_known_teams_passes_for_real_clubs() -> None:
    from api.team_mapping import assert_known_teams

    assert_known_teams(["Arsenal", "Man Utd", "Wrexham AFC"], "a test")


def test_assert_known_teams_names_what_it_could_not_resolve() -> None:
    """A build that would write an unknown name must stop, and say which."""
    from api.team_mapping import assert_known_teams

    with pytest.raises(ValueError) as exc:
        assert_known_teams(
            ["Arsenal", "Bristol Rovers", "Manchester"], "the PL canonical")

    message = str(exc.value)
    assert "Bristol Rovers" in message and "Manchester" in message
    assert "the PL canonical" in message
    assert "Arsenal" not in message, "resolved names should not be reported"


def test_the_canonical_builder_refuses_an_unknown_team() -> None:
    """A name written into the canonical is there until someone rebuilds it.

    The canonical is the training data. A club the table cannot resolve would
    be stored as its raw feed spelling and treated as a separate team from
    every other appearance of that club.
    """
    import pandas as pd

    from data.build_canonical_dataset import _map_columns

    raw = pd.DataFrame({
        "Date": ["10/08/24", "11/08/24"],
        "HomeTeam": ["Arsenal", "Bristol Rovers"],
        "AwayTeam": ["Chelsea", "Man City"],
        "FTHG": [1, 2], "FTAG": [0, 2], "FTR": ["H", "D"],
    })

    # Known names map without complaint.
    ok = _map_columns(raw.iloc[[0]].copy(), 25, normalize_names=True)
    assert ok["Home_Team"].tolist() == ["Arsenal FC"]

    with pytest.raises(ValueError, match="Bristol Rovers"):
        _map_columns(raw, 25, normalize_names=True)


def test_the_canonical_builder_tolerates_a_blank_row(caplog) -> None:
    """A trailing blank row is not an unresolved club.

    Several football-data.co.uk season files carry them (E0_1415.csv does).
    The column goes through astype(str) on the way in, which renders a blank
    as the string "nan" — checking the mapped column instead of the source
    would fail every rebuild on a file that has always been fine.
    """
    import numpy as np
    import pandas as pd

    from data.build_canonical_dataset import _map_columns

    raw = pd.DataFrame({
        "Date": ["10/08/24", np.nan],
        "HomeTeam": ["Arsenal", np.nan],
        "AwayTeam": ["Chelsea", np.nan],
        "FTHG": [1, np.nan], "FTAG": [0, np.nan], "FTR": ["H", np.nan],
    })

    import logging

    from api import team_mapping
    team_mapping._reset_unknown_name_log()
    with caplog.at_level(logging.WARNING, logger="api.team_mapping"):
        out = _map_columns(raw, 25, normalize_names=True)

    assert out["Home_Team"].tolist()[0] == "Arsenal FC"
    # And it must not warn about a team called "nan" on every rebuild.
    assert not caplog.records, f"spurious warnings: {caplog.records}"


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
