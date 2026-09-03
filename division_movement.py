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

# How much of the seed survives at each match, counted per venue. Both
# pipelines blend on exactly these weights and stop after the fifth
# (``pipeline.initialize_promoted_features``,
# ``championship_pipeline.initialize_promoted_features``), so a live row must
# use them too or a feature means one quantity in training and another at
# kick-off.
#
# Arrival selects the *route* a side is seeded from. It never selects the
# *duration*: ``arrivals_for`` is season-long membership by design, so gating
# the substitution on it alone left the seed in place all season. That is the
# defect these weights exist to close — see tests/test_seed_blend_parity.py.
SEED_BLEND_WEIGHTS: dict[int, float] = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}


def seed_weight(played: int) -> float:
    """Seed weight for a side that has played ``played`` matches at a venue.

    The fixture being priced is its ``played + 1``-th, so a side with five
    behind it is on match six and carries none of the seed.
    """
    return SEED_BLEND_WEIGHTS.get(played + 1, 0.0)


def seed_venues(
    season_rows: pd.DataFrame,
    teams,
) -> dict[str, set[str]]:
    """Which venues each side is still inside the seed window at.

    The window is per venue, and the whole of this module's difficulty comes
    from callers forgetting that. ``Home_Past5Goals`` and ``Away_Past5Goals``
    are different quantities, and Dixon-Coles' four ratings split on exactly
    the same line, so a side's home seed and away seed retire independently.

    Both predictors previously counted a side's *total* appearances, which
    retires everything at five and takes the away half with it — leaving a
    side that has never played away rated on the exit season the seed exists
    to displace.

    Args:
        season_rows: The rows of the season in play.
        teams: The sides to judge.

    Returns:
        Team -> the subset of ``{"home", "away"}`` still seeded. A side past
        the window at both venues is absent, which is the caller's signal to
        stop treating it as seeded at all.
    """
    home = season_rows["Home_Team"].value_counts()
    away = season_rows["Away_Team"].value_counts()

    within: dict[str, set[str]] = {}
    for team in teams:
        at = set()
        if seed_weight(int(home.get(team, 0))) > 0:
            at.add("home")
        if seed_weight(int(away.get(team, 0))) > 0:
            at.add("away")
        if at:
            within[team] = at
    return within


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


def recompute_two_sided(row: pd.Series) -> None:
    """Repair derived features that are a pure function of seeded inputs.

    A derivation computed from both clubs cannot be seeded from one side's
    cohort, so most of them stay approximate for an arriving side. These two
    are plain differences of columns the seed has already corrected, so
    leaving them describing a fixture that is not being played would be a
    choice rather than a limitation.

    Mutates *row* in place. Silent when a column is absent: the two leagues
    carry different feature sets.
    """
    for target, home, away in (
        ("Elo_Diff", "Home_Elo", "Away_Elo"),
        ("LeaguePosition_Diff", "Home_LeaguePosition", "Away_LeaguePosition"),
    ):
        if not {target, home, away} <= set(row.index):
            continue
        if pd.notna(row[home]) and pd.notna(row[away]):
            row[target] = row[home] - row[away]


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


def arrivals_for(
    ef_df: pd.DataFrame,
    pl_df: pd.DataFrame,
    season: int,
    *,
    fixture_teams: set[str] | None = None,
) -> dict[str, str]:
    """Arrivals for a season that may not be fully in the canonical yet.

    :func:`arrivals` compares two seasons the canonical holds. A live caller
    needs the same answer during two windows it does not cover: before the
    season's first results land, when the canonical knows nothing about it,
    and during its opening rounds, when it knows only the sides that happen
    to have played. The fixture list closes both gaps — it names every side
    in the division from the day prices appear.

    Args:
        ef_df: The EFL frame.
        pl_df: The PL frame, which separates a side dropping in from one
            coming up.
        season: The season being seeded — see :func:`season_in_play`.
        fixture_teams: Sides in this season's fixture list, if known.

    Returns:
        Team name -> ``RELEGATED`` or ``PROMOTED``, empty when the prior
        season is absent. Arrival stays what CONTEXT.md defines it to be —
        present in season N, absent in N-1 — never "has no rows yet", which
        is a statement about the calendar rather than about the side.
    """
    previous = _season_teams(ef_df, season - 1)
    if not previous:
        return {}
    current = _season_teams(ef_df, season) | (fixture_teams or set())
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


def season_in_play(df: pd.DataFrame) -> int:
    """The season currently being played, whether or not it has rows yet.

    Judged on match count, not dates. A season already under way is short of
    a full one; a completed season is not. Dates were the obvious alternative
    and are the wrong tool for the same reason
    [ADR 0005](docs/adr/0005-freshness-gate.md) rejected them — an
    international break and a dead ingestion look identical to a
    "most recent match was N days ago" test.

    Args:
        df: A league's Canonical Dataset.

    Returns:
        The latest season when it is still in progress, otherwise the one
        after it — the pre-season window, where the canonical holds no rows
        for the season whose fixtures are already being priced.
    """
    counts = df.groupby("SeasonIndex").size()
    latest = int(counts.index.max())
    prior = counts[counts.index < latest]
    if prior.empty:
        return latest
    # 0.9 admits a season that lost a handful of rows upstream while still
    # separating "under way" from "complete" by a wide margin: one round is
    # ~2% of a season, and the gap only narrows in its final fortnight.
    return latest if counts[latest] < prior.median() * 0.9 else latest + 1


def _season_teams(df: pd.DataFrame, season: int) -> set[str]:
    """Every side appearing in a season, on either side of a fixture."""
    rows = df[df["SeasonIndex"] == season]
    return set(rows["Home_Team"]) | set(rows["Away_Team"])


def _final_positions(df: pd.DataFrame, season: int) -> dict[str, float]:
    """Each side's league position at its last home fixture of a season."""
    rows = df[df["SeasonIndex"] == season].sort_values("Date")
    last = rows.drop_duplicates("Home_Team", keep="last")
    return dict(zip(last["Home_Team"], last["Home_LeaguePosition"]))


def arrival_route(
    above_df: pd.DataFrame | None,
    team: str,
    season: int,
) -> str:
    """Which direction a side arrived from.

    A side that played in the division *above* last season dropped in;
    anything else came up from below. There is no League One frame to check
    against, so "came up" is what remains once "came down" is excluded.

    Args:
        above_df: The division above, or ``None`` when there is none. The
            Premier League is the top division, so every side arriving in it
            came up — one direction, not two. That is not a PL special case
            written anywhere; it is what this function returns when nothing
            sits above, and the single-bucket prior
            ([ADR 0012](docs/adr/0012-division-movement-seed-for-the-premier-league.md))
            falls out of it without further code.
        team: The arriving side.
        season: The season it arrives in.

    Returns:
        ``RELEGATED`` or ``PROMOTED``.
    """
    if above_df is None or above_df.empty:
        return PROMOTED
    above = _season_teams(above_df, season - 1)
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
