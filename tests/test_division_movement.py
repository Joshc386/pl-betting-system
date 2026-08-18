"""The Division Movement Seed: what a side looks like before it has history here.

Two implementations used to answer that question and they disagreed — the
pipeline seeded training rows from a route cohort while the predictor built
the live row from the **league median**, a gap of 16 percentage points on
``Over25_5`` with the live path optimistic every time. [ADR 0011](../docs/adr/0011-one-division-movement-seed-per-arrival.md)
collapses them into one seed with two callers; these tests are what stops
them drifting apart again.

Frames here are synthetic and built so the answer is unambiguous: the
bottom-5 cohort and the league median are separated far enough that a test
cannot pass by accident.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from championship_pipeline import (
    _detect_new_teams,
    initialize_promoted_features,
    load_championship_data,
)
from model import DixonColesPredictor

from division_movement import (
    PROMOTED,
    RELEGATED,
    SeedParams,
    arrivals,
    fit_seed_params,
    seed_features,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFL_CANONICAL = os.path.join(PROJECT_DIR, "CompleteDSChamp_CSV.csv")
PL_CANONICAL = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")

# Three bands, chosen so the bottom-5 cohort, the mid-table cohort and the
# league median are all different numbers. A league's median normally *is*
# roughly its mid-table, which would make "cohort" and "median" impossible to
# tell apart — so mid-table is deliberately given the extreme value here. The
# frame is not meant to be physically plausible; it is meant to make a wrong
# answer impossible to mistake for a right one.
_BOTTOM5 = 2.0    # positions 20-24
_MIDTABLE = 20.0  # positions 8-16
_OTHER = 10.0     # positions 1-7 and 17-19 — and the league median


def _value_for(position: int) -> float:
    if position >= 20:
        return _BOTTOM5
    if 8 <= position <= 16:
        return _MIDTABLE
    return _OTHER


def _efl_frame() -> pd.DataFrame:
    """Three seasons: a prior season, a reference season, and the arrivals.

    Season 0 contains RETURNER, season 1 does not, season 2 has it back —
    a one-season gap, the shortest absence there is. Season 1 is the
    reference season every season-2 seed is drawn from.
    """
    rows = []
    # Season 0 — RETURNER's only history in the division.
    for i in range(1, 25):
        home = "RETURNER" if i == 24 else f"T{i:02d}"
        away_pos = (i % 24) + 1
        rows.append({
            "SeasonIndex": 0,
            "Date": f"2019-01-{i:02d}",
            "Home_Team": home,
            "Away_Team": f"T{away_pos:02d}",
            "Home_LeaguePosition": i,
            "Away_LeaguePosition": away_pos,
            # Deliberately extreme: if a side's own history ever reached its
            # seed, this value would be impossible to miss.
            "Home_Past5Goals": 999.0,
            "Away_Past5Goals": 999.0,
        })

    for i in range(1, 25):
        away_pos = (i % 24) + 1
        rows.append({
            "SeasonIndex": 1,
            "Date": f"2020-01-{i:02d}",
            "Home_Team": f"T{i:02d}",
            "Away_Team": f"T{away_pos:02d}",
            "Home_LeaguePosition": i,
            "Away_LeaguePosition": away_pos,
            "Home_Past5Goals": _value_for(i),
            "Away_Past5Goals": _value_for(away_pos),
        })

    # Season 2 arrivals. Neither has a season-1 row, so both are new to the
    # division. NEWCO never appears in the PL frame, so it came up from
    # below; DROPPER played in the PL last season, so it came down.
    for i, arrival in enumerate(("NEWCO", "DROPPER", "RETURNER"), start=1):
        rows.append({
            "SeasonIndex": 2,
            "Date": f"2021-01-{i:02d}",
            "Home_Team": arrival,
            "Away_Team": f"T{i:02d}",
            "Home_LeaguePosition": 24,
            "Away_LeaguePosition": i,
            "Home_Past5Goals": float("nan"),
            "Away_Past5Goals": _OTHER,
        })
    return pd.DataFrame(rows)


def _pl_frame() -> pd.DataFrame:
    """A PL canonical whose season 1 contains DROPPER but not NEWCO."""
    return pd.DataFrame([{
        "SeasonIndex": 1,
        "Date": "2020-01-01",
        "Home_Team": "DROPPER",
        "Away_Team": "Some PL Side FC",
        "Home_LeaguePosition": 20,
        "Away_LeaguePosition": 1,
        "Home_Past5Goals": 99.0,
        "Away_Past5Goals": 99.0,
    }])


def test_league_one_arrival_is_seeded_from_the_bottom_five_cohort():
    """An arrival from below takes the bottom-5 cohort, never the median.

    This is the live defect ADR 0011 was opened on: ``_synthesize_promoted_fixture``
    handed every arriving side a league-median row, which for a side that is
    by construction one of the division's weakest is optimistic in the
    direction that costs money.
    """
    params = SeedParams(priors={}, n_events=0)

    seeded = seed_features(
        _efl_frame(), _pl_frame(),
        team="NEWCO", season=2,
        features=["Home_Past5Goals"],
        params=params,
    )

    assert seeded["Home_Past5Goals"] == _BOTTOM5, (
        "arrival seeded from the league median rather than the bottom-5 "
        "cohort — the exact defect ADR 0011 exists to remove"
    )


def test_relegated_arrival_is_exactly_the_midtable_cohort():
    """A relegated side's seed is the mid-table cohort and nothing else.

    ADR 0011 proposed blending the side's own PL form in on top, weighted by
    a fitted w. The measurement returned 0.317 with a 95% interval of
    [-0.22, 0.85] across all 75 relegation events, which was the
    pre-committed trigger for dropping it — so the seed is the cohort
    exactly, with no residual and no drift toward the median.
    """
    params = SeedParams(priors={}, n_events=0)

    seeded = seed_features(
        _efl_frame(), _pl_frame(),
        team="DROPPER", season=2,
        features=["Home_Past5Goals"],
        params=params,
    )

    assert seeded["Home_Past5Goals"] == _MIDTABLE, (
        "a side relegated from the PL is one of this division's stronger "
        "teams and must take the mid-table cohort, not the bottom-5 one "
        "reserved for arrivals from below"
    )


def test_a_returning_side_is_an_arrival_however_short_the_gap():
    """One season away makes a side an arrival, exactly as eight seasons does.

    Movement is a cycle, so a returning side's most recent rows are always
    its *exit* season — promotion form for a side coming back down, relegation
    form for one coming back up. That contamination does not fade with a
    shorter absence: Burnley's one-season gap carries its promotion season
    just as Wolves' eight-season gap carries its title win. Keying on the gap
    length would be a threshold to tune; keying on Division Movement is not.
    """
    found = arrivals(_efl_frame(), _pl_frame(), season=2)

    assert found == {
        "NEWCO": PROMOTED,
        "DROPPER": RELEGATED,
        "RETURNER": PROMOTED,
    }, "arrival detection must key on absence in season N-1, not gap length"


def test_a_sides_own_history_never_reaches_its_seed():
    """RETURNER has history here and is seeded as if it had none.

    Season 0 gives RETURNER a Past5Goals of 999 — a value nothing else in
    the frame comes near. If own history reached the seed by any route, that
    number would surface. It must not: RETURNER's seed has to be identical
    to NEWCO's, which has never played in this division at all.
    """
    ef, pl = _efl_frame(), _pl_frame()
    params = SeedParams(priors={}, n_events=0)

    returner = seed_features(
        ef, pl, team="RETURNER", season=2,
        features=["Home_Past5Goals"], params=params)
    newco = seed_features(
        ef, pl, team="NEWCO", season=2,
        features=["Home_Past5Goals"], params=params)

    assert returner == newco, (
        "a side with history in this division was seeded differently from "
        "one without — own history has leaked into the seed"
    )
    assert returner["Home_Past5Goals"] == _BOTTOM5


def test_too_few_events_yields_no_measured_priors():
    """Below the minimum event count the measurement refuses to answer.

    Early walk-forward folds have almost no arrivals. A prior averaged over
    three events is not an estimate, it is noise with a decimal point — and
    it would price bets. Returning nothing is the caller's signal to fall
    back, and it is what production runs until enough seasons accrue.
    """
    params = fit_seed_params(_efl_frame(), _pl_frame(), through_season=2)

    assert params.priors == {}, (
        "priors were measured from too few events")
    assert params.n_events < 30


# ── Fits against the real canonicals ──────────────────────────────────────
# The synthetic frame cannot reach the minimum event count, and the fitted
# weight that ships is the one measured from real relegations — so these
# exercise the production data rather than a miniature of it.

@pytest.fixture(scope="module")
def canonicals() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (os.path.exists(EFL_CANONICAL) and os.path.exists(PL_CANONICAL)):
        pytest.skip("canonicals not present")
    return (pd.read_csv(EFL_CANONICAL, low_memory=False),
            pd.read_csv(PL_CANONICAL, low_memory=False))


def test_every_season_contributes_its_arrivals(canonicals):
    """All 150 arrivals across 25 seasons are found — 3 down, 3 up, each year."""
    ef, pl = canonicals
    params = fit_seed_params(ef, pl, through_season=26)

    assert params.n_events == 150, (
        "expected 6 arrivals a season across 25 seasons")


def test_the_fit_reads_nothing_at_or_after_the_season_it_seeds(canonicals):
    """Walk-forward: truncating the frame at the boundary changes nothing.

    If a season's weight were informed by its own outcome, the seed would
    carry information the model could not have had at kick-off, and every
    backtest built on it would be optimistic. Deleting everything from the
    boundary onward is the strongest available check — a fit that reads past
    it must move when that data disappears.
    """
    ef, pl = canonicals
    boundary = 20

    full = fit_seed_params(ef, pl, through_season=boundary)
    truncated = fit_seed_params(
        ef[ef.SeasonIndex < boundary], pl[pl.SeasonIndex < boundary],
        through_season=boundary)

    assert full.n_events == truncated.n_events, (
        "the measurement read data at or beyond the boundary")
    assert full.priors == truncated.priors


def test_route_priors_are_measured_and_split_the_two_routes(canonicals):
    """Each route gets its own venue-aware prior, measured from arrivals.

    Dixon-Coles shipped a single hand-picked bucket for every arrival,
    commented "promoted teams decent at home" — the same
    one-bucket-for-both-routes defect the feature seed had. A side dropping
    in from the PL is one of this division's stronger teams and a side
    coming up from below is one of its weakest, so their priors must not
    only differ, they must differ in that direction.
    """
    ef, pl = canonicals
    params = fit_seed_params(ef, pl, through_season=26)

    assert set(params.priors) == {RELEGATED, PROMOTED}
    for route, prior in params.priors.items():
        assert set(prior) == {
            "attack_home", "attack_away", "defence_home", "defence_away"
        }, f"{route} prior is not venue-aware on both attack and defence"

    down, up = params.priors[RELEGATED], params.priors[PROMOTED]
    assert down["attack_home"] > up["attack_home"], (
        "sides relegated from the PL must carry a stronger attacking prior "
        "than sides promoted from League One")
    assert down["defence_home"] < up["defence_home"], (
        "defence is leakiness: the stronger side must concede less")


def test_dixon_coles_rates_an_arrival_from_its_route_not_its_exit_season(
        canonicals):
    """Wolves must not be priced on their 2017/18 title win.

    ``_decay_weights`` decays by position in a side's own match sequence, not
    by date, so an eight-season absence is invisible: Wolves' last EFL
    matches were their title-winning run and they carry full weight. Worse,
    ``_shrink_to_league`` keys on match count, so 582 rows buy near-total
    confidence in that rating. An arrival must instead take its route's
    measured prior.
    """
    ef, pl = canonicals
    params = fit_seed_params(ef, pl, through_season=26)
    incoming = {"Wolves": RELEGATED, "Lincoln": PROMOTED}

    stale = DixonColesPredictor(half_life=10)
    stale.fit(ef)
    seeded = DixonColesPredictor(half_life=10)
    seeded.fit(ef, arrivals=incoming, route_priors=params.priors)

    assert stale.attack_home["Wolves"] > 1.2, (
        "precondition: the unseeded fit carries the stale title-winning "
        "rating this test exists to displace")
    assert seeded.attack_home["Wolves"] == params.priors[RELEGATED][
        "attack_home"]
    assert seeded.defence_home["Wolves"] == params.priors[RELEGATED][
        "defence_home"]
    assert seeded.attack_home["Lincoln"] == params.priors[PROMOTED][
        "attack_home"]


def test_seeding_an_arrival_leaves_every_other_side_untouched(canonicals):
    """The change reaches arrivals and nothing else.

    ADR 0011 rejected making ``_decay_weights`` calendar-aware precisely
    because it would move every side's rating. This is the check that the
    narrower fix stayed narrow.
    """
    ef, pl = canonicals
    params = fit_seed_params(ef, pl, through_season=26)
    incoming = {"Wolves": RELEGATED, "Lincoln": PROMOTED}

    before = DixonColesPredictor(half_life=10)
    before.fit(ef)
    after = DixonColesPredictor(half_life=10)
    after.fit(ef, arrivals=incoming, route_priors=params.priors)

    for team in before.attack_home:
        if team in incoming:
            continue
        for ratings in ("attack_home", "attack_away",
                        "defence_home", "defence_away"):
            assert getattr(before, ratings)[team] == getattr(
                after, ratings)[team], f"{team}'s {ratings} moved"


def test_both_callers_get_the_same_seed():
    """The pipeline and the predictor must not answer this differently.

    This is the property ADR 0011 exists to hold. The two used to disagree
    by 16 percentage points on Over25_5, and nothing detected it because a
    feature that means one thing in training and another at kick-off still
    trains, still predicts, and still looks plausible in isolation. Only the
    comparison shows it, so the comparison runs every time.
    """
    ef, pl = _efl_frame(), _pl_frame()
    params = SeedParams(priors={}, n_events=0)
    latest = ef[ef["SeasonIndex"] == 1]

    from championship_predict import ChampionshipPredictor

    predictor = ChampionshipPredictor(verbose=False)
    predictor._full_df = ef
    predictor._pl_df = pl
    predictor._seed_params_cache = params

    live = predictor._synthesize_promoted_fixture(
        "NEWCO", "T01", latest,
        home_missing=True, away_missing=False,
        home_rows=latest[latest["Home_Team"] == "NEWCO"],
        away_rows=latest[latest["Away_Team"] == "T01"],
    )
    training = seed_features(
        ef, pl, "NEWCO", 2, ["Home_Past5Goals"], params)

    assert live["Home_Past5Goals"] == training["Home_Past5Goals"], (
        "the predictor and the pipeline disagree about what this side looks "
        "like — the divergence ADR 0011 removes has reopened")
    assert live["Home_Past5Goals"] == _BOTTOM5
    assert live["Home_Promoted"] == 1


def test_known_output_is_pinned(canonicals):
    """Fixed input, fixed output — the guard against silent drift.

    Every number below was measured, not chosen. Pinning them means a
    refactor that quietly changes what a side is seeded with fails here
    rather than in a week's recommendations. If one of these moves, the
    question to answer is *why*, before deciding whether to re-pin it.

    The shipped Dixon-Coles priors were 0.90 / 0.75 / 1.10 / 1.20 for every
    arrival regardless of route. Measured over 150 events, a relegated side
    is 1.156 attacking and 0.797 defending — above average on both, where
    the hand-picked bucket had it well below.
    """
    ef, pl = canonicals
    params = fit_seed_params(ef, pl, through_season=26)

    assert params.n_events == 150
    assert params.priors[RELEGATED] == pytest.approx({
        "attack_home": 1.156295, "attack_away": 0.992871,
        "defence_home": 0.796532, "defence_away": 0.908643,
    }, abs=1e-6)
    assert params.priors[PROMOTED] == pytest.approx({
        "attack_home": 1.003748, "attack_away": 0.920673,
        "defence_home": 1.047691, "defence_away": 1.093013,
    }, abs=1e-6)

    seeded = seed_features(
        ef, pl, "Birmingham", 25,
        ["Home_Past5Goals", "Home_Past5Conceded", "Away_Past5Goals"],
        SeedParams(priors={}, n_events=0))
    assert seeded == pytest.approx({
        "Home_Past5Goals": 5.0,
        "Home_Past5Conceded": 5.4,
        "Away_Past5Goals": 5.0,
    }, abs=1e-9)


def test_the_seed_reaches_arrivals_and_nothing_else(canonicals):
    """Only an arrival's first five matches may move.

    ADR 0011 rejected the calendar-aware decay fix because it would touch
    every side's rating. This is the check that the narrow fix stayed
    narrow — the other 94.7% of the canonical must come through untouched.
    """
    ef, _ = canonicals
    prepared = load_championship_data()
    seeded, _ = initialize_promoted_features(prepared.copy())

    numeric = [c for c in prepared.columns
               if pd.api.types.is_numeric_dtype(prepared[c])]
    differs = ~((prepared[numeric] == seeded[numeric])
                | (prepared[numeric].isna() & seeded[numeric].isna())).all(axis=1)

    # The window is per venue, not per season: the features being seeded are
    # venue-specific (Home_Past5Goals is a different quantity from
    # Away_Past5Goals), so a side's first five *home* matches and first five
    # *away* matches each get seeded — up to ten rows, not five.
    allowed = set()
    for season, teams in _detect_new_teams(prepared).items():
        in_season = prepared[prepared["SeasonIndex"] == season]
        for team in teams:
            for side in ("Home_Team", "Away_Team"):
                rows = in_season[in_season[side] == team]
                allowed.update(rows.sort_values("Date").head(5).index)

    assert set(prepared.index[differs]) <= allowed, (
        "the seed changed rows outside an arrival's first five matches "
        "at each venue")


def test_the_predictor_seeds_dixon_coles_from_the_fixture_list():
    """Arrivals are seeded once there is a fixture list, not at fit time.

    Who is arriving is usually unknowable when the models are fitted: before
    a season's first results land the canonical holds no rows for it, so
    ``arrivals()`` sees nothing. It is always knowable once the odds feed
    names the teams, which is the moment this runs — and it must reach every
    market's Dixon-Coles, not just the one that happens to be checked.
    """
    from championship_predict import ChampionshipPredictor

    priors = {
        RELEGATED: {"attack_home": 1.16, "attack_away": 0.99,
                    "defence_home": 0.80, "defence_away": 0.91},
        PROMOTED: {"attack_home": 1.00, "attack_away": 0.92,
                   "defence_home": 1.05, "defence_away": 1.09},
    }
    ef = _efl_frame()

    predictor = ChampionshipPredictor(verbose=False)
    predictor._full_df = ef[ef["SeasonIndex"] <= 1]
    predictor._pl_df = _pl_frame()
    predictor._seed_params_cache = SeedParams(priors=priors, n_events=150)

    models = {}
    for market in ("_ou_models", "_ou15_models", "_btts_models"):
        dc = DixonColesPredictor(half_life=10)
        # A stale rating of exactly the kind ADR 0011 exists to displace.
        dc.attack_home["DROPPER"] = 1.9
        models[market] = dc
        setattr(predictor, market, {"dc": dc})

    applied = predictor._seed_dixon_coles({"NEWCO", "DROPPER", "T01", "T02"})

    assert applied == {"NEWCO": PROMOTED, "DROPPER": RELEGATED}, (
        "sides already in the division must not be reseeded")
    for market, dc in models.items():
        assert dc.attack_home["DROPPER"] == 1.16, f"{market} kept a stale rating"
        assert dc.attack_home["NEWCO"] == 1.00, f"{market} not seeded"
        assert "T01" not in dc.attack_home, f"{market} seeded a continuing side"
