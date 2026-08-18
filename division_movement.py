"""The Division Movement Seed — one definition, two callers (ADR 0011).

What does a side look like in a division it has not played in yet? The
pipeline asks when it builds training rows; the predictor asks when it builds
a feature row for an unplayed fixture. They must get the same answer, and
before ADR 0011 they did not.

A side's own history is deliberately never consulted. Movement is a cycle —
the only route to the PL is promotion out of the EFL, the only route to
League One is relegation out of it — so a returning side's most recent rows
are always its *exit* season, biased in a direction the route predicts.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from api.team_mapping import normalize

RELEGATED = "relegated"
PROMOTED = "promoted"

# Cohorts, by final league position of the prior season. A side arriving
# from below is one of the division's weakest; a side dropping in from above
# is one of its stronger ones.
_BOTTOM_N = 5
_MIDTABLE_BAND = (8, 16)
_MIDTABLE_FALLBACK_BAND = (6, 18)
_MIN_COHORT = 3

# Arrival events required before measured priors are trusted. Five seasons'
# worth, at six a season. Below this an average is noise with a decimal
# point, and it would price bets — so callers fall back to what they had.
# A sample-size guard, not a tuned parameter: it governs whether the
# measurement is used, never what the measurement is.
_MIN_EVENTS = 30

# Matches an arrival's seed is judged against — the window the seed governs,
# and the window the rolling features it fills are computed over.
_SEED_WINDOW = 5


@dataclass(frozen=True)
class SeedParams:
    """Constants measured from historical arrivals, never hand-picked.

    ADR 0011 also proposed blending a relegated side's own PL form into its
    seed, weighted by a fitted ``w``. The measurement returned 0.317 with a
    95% interval of [-0.22, 0.85] over all 75 relegation events — the
    pre-committed criterion for dropping it. The seed is therefore the
    cohort alone, and `scripts/measure_seed_weight.py` keeps that result
    reproducible as further events accrue.

    Attributes:
        priors: Route -> venue-aware Dixon-Coles priors.
        n_events: Arrival events the measurement was based on.
    """

    priors: dict[str, dict[str, float]]
    n_events: int


def arrivals(
    ef_df: pd.DataFrame,
    pl_df: pd.DataFrame,
    season: int,
) -> dict[str, str]:
    """Sides new to the division this season, mapped to their route.

    Args:
        ef_df: The EFL frame.
        pl_df: The PL frame — the only thing that separates a side dropping
            in from above from one coming up from below.
        season: The season being seeded.

    Returns:
        Team name -> ``RELEGATED`` or ``PROMOTED``. Empty when the prior
        season is absent, since arrival is a difference between two seasons.
    """
    current = _season_teams(ef_df, season)
    previous = _season_teams(ef_df, season - 1)
    if not previous:
        return {}
    return {t: arrival_route(pl_df, t, season) for t in current - previous}


def fit_seed_params(
    ef_df: pd.DataFrame,
    pl_df: pd.DataFrame,
    *,
    through_season: int,
) -> SeedParams:
    """Measure the seed's constants from arrivals in seasons *before* one.

    Walk-forward by construction: nothing at or after ``through_season``
    is read, so a season's seed can never be informed by its own outcome.

    Args:
        ef_df: The EFL frame.
        pl_df: The PL frame.
        through_season: The season being seeded. Events are drawn from
            seasons strictly below it.

    Returns:
        The measured constants. Priors are empty until ``_MIN_EVENTS``
        arrivals are available, which is the caller's signal to fall back.
    """
    events = _arrival_events(ef_df, pl_df, through_season)
    if len(events) < _MIN_EVENTS:
        return SeedParams(priors={}, n_events=len(events))
    return SeedParams(
        priors=_measure_priors(ef_df, events),
        n_events=len(events),
    )


def _measure_priors(
    ef_df: pd.DataFrame,
    events: list[tuple[int, str, str]],
) -> dict[str, dict[str, float]]:
    """Venue-aware Dixon-Coles priors, averaged over arrivals of each route.

    Built in the same units Dixon-Coles estimates its own ratings in — goals
    relative to the league's venue average, so 1.0 is an average side — and
    over the same window the feature seed governs. That is what makes them a
    drop-in replacement for the hand-picked ``PRIORS`` bucket rather than a
    second scale nobody can compare against.
    """
    gathered: dict[str, dict[str, list[float]]] = {
        RELEGATED: {}, PROMOTED: {}}
    for season, team, route in events:
        for name, value in _arrival_ratings(ef_df, team, season).items():
            gathered[route].setdefault(name, []).append(value)

    return {
        route: {name: sum(vals) / len(vals) for name, vals in ratings.items()}
        for route, ratings in gathered.items() if ratings
    }


def _arrival_ratings(
    ef_df: pd.DataFrame,
    team: str,
    season: int,
) -> dict[str, float]:
    """What one arriving side actually did, in Dixon-Coles rating units.

    Attack is goals scored over the league average *at that venue*; defence
    is goals conceded over the average at the venue the opponent occupied —
    so above 1.0 means leaky, matching the sign convention DC already uses.
    """
    rows = ef_df[ef_df["SeasonIndex"] == season]
    if rows.empty:
        return {}
    home_avg = float(rows["Home_Goals"].mean())
    away_avg = float(rows["Away_Goals"].mean())
    if not home_avg or not away_avg:
        return {}

    played = rows[(rows["Home_Team"] == team) | (rows["Away_Team"] == team)]
    played = played.sort_values("Date").head(_SEED_WINDOW)
    at_home = played[played["Home_Team"] == team]
    at_away = played[played["Away_Team"] == team]

    ratings: dict[str, float] = {}
    if not at_home.empty:
        ratings["attack_home"] = at_home["Home_Goals"].mean() / home_avg
        ratings["defence_home"] = at_home["Away_Goals"].mean() / away_avg
    if not at_away.empty:
        ratings["attack_away"] = at_away["Away_Goals"].mean() / away_avg
        ratings["defence_away"] = at_away["Home_Goals"].mean() / home_avg
    return ratings


def _arrival_events(
    ef_df: pd.DataFrame,
    pl_df: pd.DataFrame,
    through_season: int,
) -> list[tuple[int, str, str]]:
    """Every (season, team, route) arrival strictly before *through_season*."""
    seasons = sorted(
        s for s in ef_df["SeasonIndex"].dropna().unique() if s < through_season)
    events = []
    for season in seasons:
        for team, route in arrivals(ef_df, pl_df, int(season)).items():
            events.append((int(season), team, route))
    return events


def seed_features(
    ef_df: pd.DataFrame,
    pl_df: pd.DataFrame,
    team: str,
    season: int,
    features: list[str],
    params: SeedParams,
) -> dict[str, float]:
    """The seed values an arriving side carries into its first matches.

    Args:
        ef_df: The EFL frame.
        pl_df: The PL frame, used to tell a relegated arrival from a
            promoted one.
        team: The arriving side.
        season: The season it arrives in.
        features: Feature names to seed.
        params: Measured constants.

    Returns:
        Feature name -> seed value.
    """
    route = arrival_route(pl_df, team, season)
    cohort = _cohort_teams(ef_df, season - 1, route)
    return _cohort(ef_df, season - 1, cohort, features)


def _season_teams(df: pd.DataFrame, season: int) -> set[str]:
    """Every side appearing in a season, on either side of a fixture."""
    rows = df[df["SeasonIndex"] == season]
    return set(rows["Home_Team"]) | set(rows["Away_Team"])


def _final_positions(df: pd.DataFrame, season: int) -> dict[str, float]:
    """Each side's league position at its last home fixture of a season."""
    rows = df[df["SeasonIndex"] == season].sort_values("Date")
    last = rows.drop_duplicates("Home_Team", keep="last")
    return dict(zip(last["Home_Team"], last["Home_LeaguePosition"]))


def arrival_route(pl_df: pd.DataFrame, team: str, season: int) -> str:
    """Which direction a side arrived from.

    A side that played in the division *above* last season dropped in;
    anything else came up from below. There is no League One frame to check
    against, so "came up" is what remains once "came down" is excluded.
    """
    above = _season_teams(pl_df, season - 1)
    return RELEGATED if normalize(team) in {
        normalize(t) for t in above} else PROMOTED


def _cohort_teams(df: pd.DataFrame, season: int, route: str) -> list[str]:
    """The reference sides whose form an arrival of this route inherits.

    A side dropping in from above is one of this division's *stronger*
    teams, so it inherits mid-table; a side coming up from below is one of
    its weakest, so it inherits the bottom of the table. Collapsing the two
    into one cohort is the defect ADR 0011 removes.
    """
    positions = _final_positions(df, season)
    if route != RELEGATED:
        return sorted(positions, key=lambda t: -positions[t])[:_BOTTOM_N]

    low, high = _MIDTABLE_BAND
    band = [t for t, p in positions.items() if low <= p <= high]
    if len(band) >= _MIN_COHORT:
        return band
    # A short season, or one whose table is incomplete, can leave the band
    # nearly empty — an average of one or two sides is not a cohort. Widen
    # rather than fall through to something that is not mid-table at all.
    low, high = _MIDTABLE_FALLBACK_BAND
    return [t for t, p in positions.items() if low <= p <= high]


def _cohort(
    df: pd.DataFrame,
    season: int,
    cohort: list[str],
    features: list[str],
) -> dict[str, float]:
    """Mean feature values across a cohort of sides.

    Each side contributes its final observed value, matching how the
    pipeline has always computed its promoted-team reference.
    """
    rows = df[df["SeasonIndex"] == season].sort_values("Date")

    seeded: dict[str, float] = {}
    for feature in features:
        # Callers pass whole column blocks, so the list arrives carrying
        # team names and dates alongside the rolling features. Only numbers
        # have a cohort average; everything else the caller keeps as it is.
        if feature not in rows.columns:
            continue
        if not pd.api.types.is_numeric_dtype(rows[feature]):
            continue
        side = "Home_Team" if feature.startswith("Home_") else "Away_Team"
        values = []
        for team in cohort:
            team_rows = rows[rows[side] == team]
            if team_rows.empty:
                continue
            value = team_rows.iloc[-1][feature]
            if pd.notna(value):
                values.append(value)
        if values:
            seeded[feature] = sum(values) / len(values)
    return seeded
