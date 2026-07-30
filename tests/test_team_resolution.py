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


# ─────────────────────────────────────────────────────────────────────────────
# A shared place name is not a shared club
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("api_name,teams", [
    # Two clubs, one city, nothing else in common.
    ("Bristol Rovers", {"Bristol City"}),
    ("Manchester United", {"Manchester City FC"}),
    ("Manchester City", {"Manchester United FC"}),
    ("Sheffield Wednesday", {"Sheffield United"}),
    ("Sheffield United", {"Sheffield Weds"}),
])
def test_a_conflicting_surname_is_a_different_club(
        api_name: str, teams: set[str]) -> None:
    """The place name is shared; the surname says they are separate clubs.

    Bristol Rovers and Bristol City share a city and nothing else. Neither
    does Manchester United belong to Manchester City. Resolving on the place
    alone prices a fixture as the wrong club.
    """
    from api.team_resolver import resolve_feed_team

    got = resolve_feed_team(api_name, teams, {})
    assert got is None, (
        f"{api_name!r} resolved to {got!r} — a different club that happens "
        f"to share a city.")


@pytest.mark.parametrize("api_name,expected", [
    # The EFL canonical keeps football-data.co.uk short forms, which carry no
    # surname at all — so there is nothing to conflict with and the place name
    # is the whole name.
    ("Stoke City", "Stoke"),
    ("Hull City", "Hull"),
    ("Luton Town", "Luton"),
    ("Oxford United", "Oxford"),
    ("Cardiff City", "Cardiff"),
])
def test_a_bare_place_name_still_resolves(api_name: str, expected: str) -> None:
    """A short-form canonical name has no surname to disagree with."""
    from api.team_resolver import resolve_feed_team

    teams = {"Stoke", "Hull", "Luton", "Oxford", "Cardiff", "Bristol City"}
    assert resolve_feed_team(api_name, teams, {}) == expected


def test_the_matching_surname_still_decides() -> None:
    """Both Manchester clubs are candidates; the surname picks the right one."""
    from api.team_resolver import resolve_feed_team

    assert resolve_feed_team("Manchester City", PL_2025, {}) == "Manchester City FC"
    assert resolve_feed_team("Manchester United", PL_2025, {}) == "Manchester United FC"


# ─────────────────────────────────────────────────────────────────────────────
# The EFL feed — same contract, a canonical in a different name format
# ─────────────────────────────────────────────────────────────────────────────

# What `our_teams` holds for the Championship: football-data.co.uk short forms,
# not the long names the PL canonical uses.
EFL_TEAMS = {
    "Leeds", "Burnley", "Sheffield Weds", "QPR", "West Brom", "Nott'm Forest",
    "Hull", "Blackburn", "Cardiff", "Luton", "Plymouth", "Preston", "Norwich",
    "Derby", "Stoke", "Coventry", "Middlesbrough", "Oxford", "Swansea",
    "Watford", "Sunderland", "Millwall", "Sheffield United", "Portsmouth",
}


def test_efl_never_guesses_between_the_two_sheffield_clubs() -> None:
    """An unmapped "Sheffield Utd" must not become Sheffield Wednesday.

    The old fallback took the first candidate reaching the best score while
    iterating a set, so which Sheffield club came back depended on hash order.
    """
    from championship_predict import _resolve_champ_team

    assert _resolve_champ_team("Sheffield Utd", EFL_TEAMS) is None
    assert _resolve_champ_team("Sheffield", EFL_TEAMS) is None


def test_efl_explicit_mapping_wins_when_the_club_is_absent() -> None:
    """A promoted side maps to its short form though it is not yet known."""
    from championship_predict import _resolve_champ_team

    thin = {"Blackburn", "Stoke", "Derby"}
    assert _resolve_champ_team("Leicester City", thin) == "Leicester"
    assert _resolve_champ_team("Wrexham AFC", thin) == "Wrexham"


def test_every_mapped_efl_club_still_resolves() -> None:
    """Regression: consolidation must not cost a single working resolution."""
    from championship_predict import _ODDS_API_TO_CHAMP, _resolve_champ_team

    unresolved = {
        api_name: _resolve_champ_team(api_name, EFL_TEAMS)
        for api_name, short in _ODDS_API_TO_CHAMP.items()
        if _resolve_champ_team(api_name, EFL_TEAMS) != short
    }
    assert not unresolved, f"regressed: {unresolved}"


def test_the_resolver_is_name_format_agnostic() -> None:
    """One resolver, two canonicals — long PL names and short EFL forms.

    Nothing in the resolver may assume a name format. The format lives in the
    per-feed mapping dicts and in `our_teams`, never in the matching rule.
    """
    from api.team_resolver import resolve_feed_team

    # The same feed name, against two canonicals that spell clubs differently.
    assert resolve_feed_team("Blackburn Rovers", {"Blackburn Rovers FC"}, {}) \
        == "Blackburn Rovers FC"
    assert resolve_feed_team("Blackburn Rovers", {"Blackburn"}, {}) == "Blackburn"

    assert resolve_feed_team("Nottingham Forest", PL_2025, {}) == "Nottingham Forest FC"
    # The EFL canonical abbreviates this one, so it is the mapping dict's job.
    # Guessing that "Nott'm" means "Nottingham" is not the resolver's business.
    assert resolve_feed_team("Nottingham Forest", EFL_TEAMS, {}) is None
    assert resolve_feed_team(
        "Nottingham Forest", EFL_TEAMS,
        {"Nottingham Forest": "Nott'm Forest"}) == "Nott'm Forest"


# ─────────────────────────────────────────────────────────────────────────────
# One implementation of one contract
# ─────────────────────────────────────────────────────────────────────────────

def test_all_three_feeds_share_one_resolver() -> None:
    """Three copies would be three implementations of one contract (ADR 0007).

    This is the same failure ADR 0007 documents for canonical features: one
    name, two build scripts, nothing forcing them to agree. Three odds-feed
    resolvers drifted the same way — the wrong-club bug was fixed in two of
    them and left live in the third.
    """
    import api.odds_api as odds_api
    import api.oddspapi as oddspapi
    import championship_predict
    from api.team_resolver import resolve_feed_team

    assert odds_api.resolve_feed_team is resolve_feed_team
    assert oddspapi.resolve_feed_team is resolve_feed_team
    assert championship_predict.resolve_feed_team is resolve_feed_team


def test_no_feed_module_defines_its_own_word_list() -> None:
    """The generic-word list lives in one module and is imported, not copied."""
    import api.odds_api as odds_api
    import api.oddspapi as oddspapi
    import championship_predict
    from api import team_resolver

    for module in (odds_api, oddspapi, championship_predict):
        own = module.__dict__.get("_GENERIC_TEAM_WORDS")
        assert own is None or own is team_resolver._GENERIC_TEAM_WORDS, (
            f"{module.__name__} carries its own copy of the word list")
