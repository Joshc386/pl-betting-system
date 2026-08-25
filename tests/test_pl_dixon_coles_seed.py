"""Dixon-Coles ratings for sides arriving in the Premier League.

[ADR 0012](../docs/adr/0012-division-movement-seed-for-the-premier-league.md)
— the half [ADR 0011](../docs/adr/0011-one-division-movement-seed-per-arrival.md)
scoped out. The PL's other three ensemble members are fed a seeded feature
row; Dixon-Coles consumes *team identity* and looks up venue-specific ratings,
so the row never reaches it. Left alone it rates an arriving side on that
side's exit season, however long ago that was.

Measured against the live pickle on 2026-08-23: Coventry (last PL season index
0) was unrated and fell through to the hand-picked ``PRIORS`` bucket, Hull
carried 2016/17, and Ipswich carried 2024/25 with ``defence_home`` 1.681. On
Ipswich v Liverpool that is 0.7603 against 0.6431 — 11.7 points of P(Over 2.5)
on the Dixon-Coles component.

Unlike the EFL, the PL has **one arrival direction**: nobody is relegated into
it. That is not special-cased anywhere — it is what the shared machinery
produces when there is no division above, which is the whole point of
generalising ``arrival_route`` rather than copying it.

Frames here are synthetic and built so a seeded rating and a stale one can
never be confused.
"""
from __future__ import annotations

import pandas as pd

from division_movement import PROMOTED, SeedParams
from model import DixonColesPredictor

# A stale rating of exactly the kind ADR 0012 exists to displace: the shape of
# a side that was relegated out of this division years ago.
_STALE_ATTACK = 1.9

# The measured prior a PL arrival should receive instead. Deliberately nowhere
# near the stale value, nor near the hand-picked PRIORS bucket, so a test
# cannot pass by landing on either by accident.
_PRIOR = {
    "attack_home": 1.02, "attack_away": 0.94,
    "defence_home": 1.06, "defence_away": 1.11,
}


def _pl_season(season: int, teams: int = 20) -> list[dict]:
    """One complete PL season, every side appearing at both venues."""
    sides = [f"T{i:02d}" for i in range(1, teams + 1)]
    rows = []
    for i in range(380):
        home = sides[i % teams]
        away = sides[(i + 1) % teams]
        rows.append({
            "SeasonIndex": season,
            "Date": f"20{20 + season:02d}-{(i % 9) + 1:02d}-{(i % 28) + 1:02d}",
            "Home_Team": home,
            "Away_Team": away,
            "Home_Goals": 1,
            "Away_Goals": 1,
            "Home_LeaguePosition": (i % teams) + 1,
            "Away_LeaguePosition": ((i + 1) % teams) + 1,
        })
    return rows


def _predictor(full_df: pd.DataFrame, params: SeedParams):
    """A predictor holding only the state the seeding path reads.

    Built with __new__ on purpose: the real constructor loads and trains
    models, none of which this behaviour depends on.
    """
    from predict import LivePredictor

    p = LivePredictor.__new__(LivePredictor)
    p._full_df = full_df
    p._seed_params_cache = params
    p.verbose = False
    return p


def test_an_arriving_side_is_rated_from_the_measured_prior_in_every_market():
    """The tracer bullet: a stale rating is displaced, in both PL markets.

    The PL holds two Dixon-Coles instances — O/U 2.5 and BTTS — where the EFL
    holds three. A seed that reached only the market someone happened to check
    would leave the other pricing on a relegation season, so both are asserted.
    """
    df = pd.DataFrame(_pl_season(24) + _pl_season(25))
    predictor = _predictor(df, SeedParams(priors={PROMOTED: _PRIOR},
                                          n_events=75))

    markets = {}
    for attr in ("_ou_models", "_btts_models"):
        dc = DixonColesPredictor(half_life=10)
        dc.attack_home["NEWCO"] = _STALE_ATTACK
        markets[attr] = dc
        setattr(predictor, attr, {"dc": dc})

    applied = predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

    assert applied == {"NEWCO": PROMOTED}, (
        "only the arriving side is seeded, and its direction is PROMOTED — "
        "nobody is relegated into the Premier League")
    for attr, dc in markets.items():
        assert dc.attack_home["NEWCO"] == _PRIOR["attack_home"], (
            f"{attr} kept a stale rating")
        assert dc.defence_home["NEWCO"] == _PRIOR["defence_home"], (
            f"{attr} was not seeded in defence")
        assert "T01" not in dc.attack_home, (
            f"{attr} seeded a continuing side")


def _season_partly_played(season: int, newco_matches: int) -> list[dict]:
    """A season under way, with NEWCO having played `newco_matches` times.

    Short of a full season on purpose, so `season_in_play` reports it as the
    season being played rather than a completed one.
    """
    rows = []
    for i in range(newco_matches):
        rows.append({
            "SeasonIndex": season,
            "Date": f"20{20 + season:02d}-08-{(i % 28) + 1:02d}",
            "Home_Team": "NEWCO" if i % 2 == 0 else f"T{(i % 18) + 1:02d}",
            "Away_Team": f"T{(i % 18) + 1:02d}" if i % 2 == 0 else "NEWCO",
            "Home_Goals": 1, "Away_Goals": 1,
            "Home_LeaguePosition": 10, "Away_LeaguePosition": 11,
        })
    return rows


def test_the_seed_retires_as_the_arrival_accumulates_a_record():
    """Arrival selects the route. It never selects the *duration*.

    ``arrivals_for`` is season-long membership by design — a side promoted in
    August is still an arrival in May. Gating the seed on arrival alone
    therefore leaves a route prior in place all season, discarding every
    result the side actually produced, including the estimate the weekly
    retrain had just fitted. Duration is `seed_weight`'s question and only
    `seed_weight`'s.

    This exact confusion shipped as two P0s at two call sites last session.
    The PL is the third call site.
    """
    df = pd.DataFrame(
        _pl_season(24) + _pl_season(25) + _season_partly_played(26, 12))
    predictor = _predictor(df, SeedParams(priors={PROMOTED: _PRIOR},
                                          n_events=75))
    dc = DixonColesPredictor(half_life=10)
    dc.attack_home["NEWCO"] = _STALE_ATTACK
    predictor._ou_models = {"dc": dc}
    predictor._btts_models = {"dc": dc}

    applied = predictor._seed_dixon_coles({"NEWCO", "T01"})

    assert applied == {}, (
        "NEWCO has six matches at each venue behind it and is past the seed "
        "window, but is still an arrival by division movement — the seed "
        "must retire on matches played, not on arrival")
    assert dc.attack_home["NEWCO"] == _STALE_ATTACK, (
        "a side past the seed window must keep whatever Dixon-Coles fitted "
        "for it, not be overwritten by the route prior")


# ─────────────────────────────────────────────────────────────────────────────
# The measured constants must reach predict time, or every arrival silently
# falls back to the hand-picked bucket and nothing reports it.
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasuredSeedParamsSurviveThePickle:
    """Round-trip through the file the scheduler actually writes."""

    @staticmethod
    def _bare():
        from predict import LivePredictor

        p = LivePredictor.__new__(LivePredictor)
        p.verbose = False
        for attr in ("_ou_models", "_btts_models", "_ou_features",
                     "_btts_features", "_ou_base_rate", "_btts_base_rate",
                     "_dc_kwargs", "_btts_dc_kwargs", "_train_medians",
                     "_our_teams"):
            setattr(p, attr, {})
        p._ou_stacker = None
        p._ou_logit_shift = 0.0
        p._ou_val_mean_logit = 0.0
        p._btts_cal_shifts = None
        return p

    def test_priors_and_sample_size_both_come_back(self, tmp_path):
        path = str(tmp_path / "state.pkl")

        saver = self._bare()
        saver._seed_params_cache = SeedParams(priors={PROMOTED: _PRIOR},
                                              n_events=75)
        saver.save_trained_state(path)

        loader = self._bare()
        assert loader.load_trained_state(path) is True
        assert loader._seed_params().priors == {PROMOTED: _PRIOR}, (
            "the measured prior did not survive the pickle — every PL arrival "
            "would silently fall back to the hand-picked bucket, which is the "
            "state ADR 0012 exists to leave behind")
        assert loader._seed_params().n_events == 75

    def test_a_pre_adr_pickle_falls_back_rather_than_crashing(self, tmp_path):
        """Every PL pickle written before today has no seed_params key."""
        import pickle

        path = tmp_path / "legacy.pkl"
        saver = self._bare()
        saver._seed_params_cache = None
        saver.save_trained_state(str(path))

        with open(path, "rb") as fh:
            state = pickle.load(fh)
        del state["seed_params"]
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

        loader = self._bare()
        assert loader.load_trained_state(str(path)) is True
        assert loader._seed_params().priors == {}
        assert loader._seed_params().n_events == 0
