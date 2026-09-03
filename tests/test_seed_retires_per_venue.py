"""When the Division Movement Seed retires, counted per venue.

[ADR 0011](../docs/adr/0011-one-division-movement-seed-per-arrival.md) is
explicit that the seed window is per *venue*: "`Home_Past5Goals` and
`Away_Past5Goals` are different quantities, so a side's first five home
matches and first five away matches are each seeded". `SEED_BLEND_WEIGHTS`
says the same — "counted per venue".

Dixon-Coles holds four ratings on exactly that split (`attack_home`,
`attack_away`, `defence_home`, `defence_away`), so the same reasoning applies
to it unchanged. Both predictors nevertheless gated the whole side on its
*total* appearances, which retires the seed roughly twice as fast as the
feature row does and — in the case this module is built around — discards the
away seed of a side that has never played away.

The comment above both call sites claimed the rating and the feature row
"can never disagree about when the seed stops applying". They could, and did.
Parametrised across both leagues, because a single-league test is how two
earlier mutations went uncaught.
"""
from __future__ import annotations

import pytest

from division_movement import PROMOTED, RELEGATED
from model import DixonColesPredictor

_PRIOR = {
    PROMOTED: {"attack_home": 1.02, "attack_away": 0.94,
               "defence_home": 1.06, "defence_away": 1.11},
    RELEGATED: {"attack_home": 1.16, "attack_away": 0.99,
                "defence_home": 0.80, "defence_away": 0.91},
}
_FITTED = 1.77  # what Dixon-Coles estimated; must survive where seed retires


def _dc() -> DixonColesPredictor:
    dc = DixonColesPredictor(half_life=10)
    for name in ("attack_home", "attack_away", "defence_home", "defence_away"):
        getattr(dc, name)["NEWCO"] = _FITTED
    return dc


def test_seeding_one_venue_leaves_the_other_venues_ratings_alone():
    """The primitive: a venue-scoped seed touches only that venue.

    A side five home matches into its first season has a home record worth
    rating and no away record at all. Seeding both, or neither, are each
    wrong in one half.
    """
    dc = _dc()

    dc.seed_arrivals({"NEWCO": PROMOTED}, _PRIOR, venues={"NEWCO": {"away"}})

    assert dc.attack_away["NEWCO"] == _PRIOR[PROMOTED]["attack_away"], (
        "the away half was not seeded")
    assert dc.defence_away["NEWCO"] == _PRIOR[PROMOTED]["defence_away"], (
        "the away half was not seeded in defence")
    assert dc.attack_home["NEWCO"] == _FITTED, (
        "the home half has five matches behind it and must keep the rating "
        "Dixon-Coles fitted for it")
    assert dc.defence_home["NEWCO"] == _FITTED, (
        "the home half must keep its fitted defence rating")


def test_an_unscoped_seed_still_covers_both_venues():
    """Backward compatibility: no venue scoping means the whole side.

    `model.fit` passes no arrivals at all today, and `scripts/validate_seed.py`
    seeds whole sides. Neither should change meaning.
    """
    dc = _dc()

    dc.seed_arrivals({"NEWCO": PROMOTED}, _PRIOR)

    for venue in ("home", "away"):
        assert dc.attack_home["NEWCO"] == _PRIOR[PROMOTED]["attack_home"]
        assert dc.attack_away["NEWCO"] == _PRIOR[PROMOTED]["attack_away"]


# ─────────────────────────────────────────────────────────────────────────────
# The gate each predictor computes, across both leagues.
#
# The case both got wrong: a side five home matches into its first season and
# yet to travel. Counting total appearances retires the whole seed at five,
# taking the away half with it — so a side that has never played away is rated
# on whatever the fit produced for it, which is the stale exit-season value the
# seed exists to displace.
# ─────────────────────────────────────────────────────────────────────────────

def _complete(season: int, teams: int, matches: int) -> list[dict]:
    sides = [f"T{i:02d}" for i in range(1, teams + 1)]
    return [{
        "SeasonIndex": season,
        "Date": f"20{20 + season:02d}-{(i % 9) + 1:02d}-{(i % 28) + 1:02d}",
        "Home_Team": sides[i % teams],
        "Away_Team": sides[(i + 1) % teams],
        "Home_Goals": 1, "Away_Goals": 1,
        "Home_LeaguePosition": (i % teams) + 1,
        "Away_LeaguePosition": ((i + 1) % teams) + 1,
    } for i in range(matches)]


def _five_home_none_away(season: int) -> list[dict]:
    """NEWCO's first five matches, every one of them at home."""
    return [{
        "SeasonIndex": season,
        "Date": f"20{20 + season:02d}-08-{i + 1:02d}",
        "Home_Team": "NEWCO", "Away_Team": f"T{i + 1:02d}",
        "Home_Goals": 1, "Away_Goals": 1,
        "Home_LeaguePosition": 10, "Away_LeaguePosition": 11,
    } for i in range(5)]


def _pl_predictor():
    import pandas as pd
    from division_movement import SeedParams
    from predict import LivePredictor

    df = pd.DataFrame(
        _complete(24, 20, 380) + _complete(25, 20, 380)
        + _five_home_none_away(26))
    p = LivePredictor.__new__(LivePredictor)
    p._full_df = df
    p._seed_params_cache = SeedParams(priors=_PRIOR, n_events=75)
    p.verbose = False
    dc = _dc()
    p._ou_models = {"dc": dc}
    p._btts_models = {"dc": dc}
    return p, dc


def _efl_predictor():
    import pandas as pd
    from championship_predict import ChampionshipPredictor
    from division_movement import SeedParams

    df = pd.DataFrame(
        _complete(24, 24, 552) + _complete(25, 24, 552)
        + _five_home_none_away(26))
    p = ChampionshipPredictor.__new__(ChampionshipPredictor)
    p._full_df = df
    p._pl_df = pd.DataFrame(_complete(25, 20, 380))
    p._seed_params_cache = SeedParams(priors=_PRIOR, n_events=150)
    p.verbose = False
    dc = _dc()
    p._ou_models = {"dc": dc}
    p._ou15_models = {"dc": dc}
    p._btts_models = {"dc": dc}
    return p, dc


@pytest.mark.parametrize("build", [_pl_predictor, _efl_predictor],
                         ids=["PL", "EFL"])
def test_the_away_seed_survives_a_side_that_has_only_played_at_home(build):
    """Five home matches retire the home seed and nothing else.

    Counting the side's *total* appearances gives five, retires everything,
    and leaves the away ratings on a fitted value drawn from an exit season
    the side played years ago — precisely the state the seed displaces.
    """
    predictor, dc = build()

    predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

    assert dc.attack_away["NEWCO"] == _PRIOR[PROMOTED]["attack_away"], (
        "NEWCO has never played away and must still carry the away seed")
    assert dc.defence_away["NEWCO"] == _PRIOR[PROMOTED]["defence_away"], (
        "NEWCO has never played away and must still carry the away seed")
    assert dc.attack_home["NEWCO"] == _FITTED, (
        "five home matches is a record worth rating — the home seed retires")


# ─────────────────────────────────────────────────────────────────────────────
# One declaration of the blend, not three.
#
# Both pipelines held the weights as an inline local, and the only thing
# keeping them in step with `SEED_BLEND_WEIGHTS` was a test matching a literal
# line of source. That test passes against a dict that is correct today and
# says nothing about whether the pipeline still reads it.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module", ["pipeline", "championship_pipeline"])
def test_the_pipeline_blend_follows_the_shared_weights(module, monkeypatch):
    """Change the shared weights and the pipeline's output must change.

    Behavioural, where the previous guard was textual. If a pipeline ever goes
    back to its own copy of the numbers this fails, whereas a source match
    would keep passing for as long as the copy happened to agree.
    """
    import importlib

    import division_movement
    import pandas as pd

    mod = importlib.import_module(module)
    teams = 24 if module == "championship_pipeline" else 20

    def _season(season, sides):
        """A season over `sides`, the weakest five carrying the cohort value.

        The cohort and the arrival's own form must differ, or every weight
        blends the same number into the same number and the test passes
        whatever the pipeline does.
        """
        rows = []
        for i in range(len(sides) * 6):
            h, a = sides[i % len(sides)], sides[(i + 1) % len(sides)]
            hp, ap = (i % len(sides)) + 1, ((i + 1) % len(sides)) + 1
            cut = len(sides) - 5
            rows.append({
                "SeasonIndex": season,
                "Date": f"20{20 + season:02d}-0{(i % 9) + 1}-{(i % 28) + 1:02d}",
                "Home_Team": h, "Away_Team": a,
                "Home_Promoted": int(h == "NEWCO"),
                "Away_Promoted": int(a == "NEWCO"),
                "Home_LeaguePosition": hp, "Away_LeaguePosition": ap,
                "Home_Past5Goals": 2.0 if hp > cut else 9.0,
                "Away_Past5Goals": 2.0 if ap > cut else 9.0,
            })
        return rows

    established = [f"T{i:02d}" for i in range(1, teams + 1)]
    # NEWCO must be absent in season 24 and present in 25: the EFL detects
    # arrivals by that difference, where the PL reads the Promoted flag. A
    # frame that satisfies only one of them tests only one pipeline.
    df = pd.DataFrame(
        _season(24, established)
        + _season(25, ["NEWCO"] + established[1:]))

    baseline = mod.initialize_promoted_features(df.copy())
    baseline = baseline[0] if isinstance(baseline, tuple) else baseline

    # A weight the production dict does not hold, at match 1, where the seed
    # is otherwise total. If the pipeline reads the shared definition the
    # first match stops being pure seed; if it holds its own copy, nothing
    # moves.
    monkeypatch.setitem(division_movement.SEED_BLEND_WEIGHTS, 1, 0.5)
    changed = mod.initialize_promoted_features(df.copy())
    changed = changed[0] if isinstance(changed, tuple) else changed

    first = baseline[(baseline["SeasonIndex"] == 25)
                     & (baseline["Home_Team"] == "NEWCO")].sort_values("Date")
    after = changed[(changed["SeasonIndex"] == 25)
                    & (changed["Home_Team"] == "NEWCO")].sort_values("Date")

    assert not first.empty, "the arrival has no rows; the frame is wrong"
    assert first.iloc[0]["Home_Past5Goals"] != after.iloc[0]["Home_Past5Goals"], (
        f"{module} did not follow SEED_BLEND_WEIGHTS — it is holding its own "
        f"copy of the blend, which is the divergence the shared constant "
        f"exists to prevent")
