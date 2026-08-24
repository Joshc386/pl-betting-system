"""A seeded feature must mean the same thing at kick-off as it did in training.

Both pipelines blend a Division Movement Seed into an arriving side's *first
five matches at a venue* and then stop:

    match 1: 100% seed
    match 2:  80% seed + 20% actual
    match 3:  60/40      match 4: 40/60      match 5: 20/80
    match 6+: 100% actual

`pipeline.initialize_promoted_features` and
`championship_pipeline.initialize_promoted_features` both implement exactly
that, over an explicit rolling-feature list, counting match number separately
per venue.

Serving did not. Both `_fixture_feature_row` implementations gated the
substitution on `home_arriving` — season-long membership, which
`division_movement.arrivals_for` documents as "never 'has no rows yet'". So an
arriving side's real form was computed and then discarded on every fixture for
the whole season, and from match 6 the live row and the trained rows meant
different quantities. Six of 24 EFL sides and three of 20 PL sides were priced
that way from the 2026/27 rollover onward.

These tests pin the contract from the training side first, so the expected
numbers cannot drift away from what the pipeline actually does, then require
serving to produce the same blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from division_movement import _SEED_WINDOW

# The subset asserted on. The production lists are longer; these four are
# enough to catch a wrong weight, and keeping the frame small keeps the
# failure message readable.
_FEATS = [
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
]

_COHORT_VALUE = 1.0    # what the weakest five averaged last season
_ACTUAL_VALUE = 3.0    # what the arriving side has actually been doing

# Deliberately distinct so any blend of them is identifiable on sight:
#   pure seed 1.0 | 60/40 -> 1.8 | 40/60 -> 2.2 | pure actual 3.0
_BLEND_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}


def _expected(match_num: int) -> float:
    """The value training writes for an arriving side's ``match_num``-th match."""
    weight = _BLEND_WEIGHTS.get(match_num, 0.0)
    return weight * _COHORT_VALUE + (1.0 - weight) * _ACTUAL_VALUE


def _row(season, date, home, away, *, home_promoted=0, away_promoted=0,
         position=10, value=_COHORT_VALUE):
    row = {
        "SeasonIndex": season,
        "Date": date,
        "Home_Team": home,
        "Away_Team": away,
        "Home_Promoted": home_promoted,
        "Away_Promoted": away_promoted,
        "Home_LeaguePosition": position,
        "Away_LeaguePosition": position,
    }
    for feat in _FEATS:
        row[feat] = value
    return row


def _prior_season(season: int, teams: list[str]) -> list[dict]:
    """A complete prior season, every side carrying the cohort value.

    Position is the side's index, so `bottom5_cohort` picks the last five
    deterministically and they all average to ``_COHORT_VALUE``.
    """
    rows, day = [], 0
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if home == away:
                continue
            day += 1
            rows.append(_row(season, f"20{20 + season}-01-{day:02d}",
                             home, away, position=i + 1))
    return rows


def _arrival_season(season: int, arrival: str, others: list[str],
                    home_matches: int,
                    away_matches: int | None = None) -> list[dict]:
    """Played fixtures for ``arrival``, at each venue, all real form.

    ``OPP`` is deliberately never hosted by, and never hosts, the arrival:
    the fixture under test has to be one that has not been played, or
    ``_fixture_feature_row`` returns the played row from its exact-fixture
    shortcut and no seeding code runs at all. Two of these tests passed that
    way on the first draft, against the very defect they were written for.

    ``away_matches`` defaults to none played, which is what the per-venue
    tests below need in order to show an away seed surviving a home record.
    The *duration* tests pass it explicitly instead: a real fixture list
    alternates venues, so a side fourteen matches into a season has played
    roughly seven at each, never fourteen at one and none at the other. A
    fixture that only ever plays at home cannot tell "past the window" from
    "has not travelled yet".
    """
    away_matches = 0 if away_matches is None else away_matches
    rows = []
    for n in range(home_matches):
        rows.append(_row(season, f"20{20 + season}-02-{n + 1:02d}",
                         arrival, others[n % len(others)],
                         home_promoted=1, value=_ACTUAL_VALUE))
    for n in range(away_matches):
        rows.append(_row(season, f"20{20 + season}-03-{n + 1:02d}",
                         others[n % len(others)], arrival,
                         away_promoted=1, value=_ACTUAL_VALUE))
    # Established fixtures, so the season is not just the arrival and both
    # OPP and others[0] carry rows at each venue.
    rows.append(_row(season, f"20{20 + season}-02-20", others[0], "OPP"))
    rows.append(_row(season, f"20{20 + season}-02-21", "OPP", others[0]))
    rows.append(_row(season, f"20{20 + season}-02-22", others[0], others[1]))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# The oracle: what training actually does. Everything below matches this.
# ─────────────────────────────────────────────────────────────────────────────

class TestTheTrainingContract:
    """Pin the blend from the training side, so serving has a fixed target.

    If the pipeline's weights or cutoff ever change, this fails first and the
    serving tests below stop being a claim about a number nobody maintains.
    """

    @staticmethod
    def _trained(home_matches: int) -> pd.DataFrame:
        from pipeline import initialize_promoted_features

        others = ["T01", "T02", "T03", "T04", "T05"]
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, home_matches)
        )
        out = initialize_promoted_features(df)
        return out[0] if isinstance(out, tuple) else out

    @pytest.mark.parametrize("match_num", [1, 2, 3, 4, 5])
    def test_the_first_five_matches_blend_seed_into_actual(self, match_num):
        df = self._trained(home_matches=6)
        newco = df[(df["SeasonIndex"] == 25)
                   & (df["Home_Team"] == "NEWCO")].sort_values("Date")

        assert newco.iloc[match_num - 1]["Home_Past5Goals"] == pytest.approx(
            _expected(match_num)), (
            f"training's match {match_num} weight is not "
            f"{_BLEND_WEIGHTS[match_num]}")

    def test_the_sixth_match_is_entirely_actual(self):
        """The cutoff. This is the half serving got wrong."""
        df = self._trained(home_matches=6)
        newco = df[(df["SeasonIndex"] == 25)
                   & (df["Home_Team"] == "NEWCO")].sort_values("Date")

        assert newco.iloc[5]["Home_Past5Goals"] == pytest.approx(_ACTUAL_VALUE)


# ─────────────────────────────────────────────────────────────────────────────
# Serving must produce the same number for the same team-match.
# ─────────────────────────────────────────────────────────────────────────────

class TestPLServingMatchesTraining:
    """`predict.LivePredictor._fixture_feature_row`."""

    @staticmethod
    def _live(home_matches: int) -> pd.Series:
        from predict import LivePredictor

        others = ["T01", "T02", "T03", "T04", "T05"]
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, home_matches)
        )
        predictor = LivePredictor.__new__(LivePredictor)
        predictor.verbose = False
        predictor._full_df = df
        return predictor._fixture_feature_row(
            "NEWCO", "OPP", df, season=25, arrivals={"NEWCO"})

    def test_the_fixture_under_test_has_not_been_played(self):
        """Guards every assertion below.

        If the fixture exists in the frame, `_fixture_feature_row` returns it
        directly and none of the seeding code runs, so every test in this
        class would pass while proving nothing.
        """
        from predict import LivePredictor

        others = ["T01", "T02", "T03", "T04", "T05"]
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, 12)
        )
        played = df[(df["Home_Team"] == "NEWCO") & (df["Away_Team"] == "OPP")]

        assert played.empty, (
            "NEWCO v OPP has been played, so the exact-fixture shortcut will "
            "short-circuit the seed and these tests assert nothing")
        assert not df[df["Away_Team"] == "OPP"].empty
        assert isinstance(LivePredictor, type)

    def test_with_no_history_the_seed_is_the_whole_row(self):
        """Preserved behaviour: this is the case the seed was written for."""
        row = self._live(home_matches=0)

        assert row is not None
        assert row["Home_Past5Goals"] == pytest.approx(_COHORT_VALUE)

    @pytest.mark.parametrize("played,next_match", [(1, 2), (2, 3), (4, 5)])
    def test_partway_through_the_window_it_blends(self, played, next_match):
        row = self._live(home_matches=played)

        assert row["Home_Past5Goals"] == pytest.approx(_expected(next_match)), (
            f"after {played} home matches the next is match {next_match}, "
            f"which training blends at weight {_BLEND_WEIGHTS[next_match]}")

    def test_past_the_window_the_seed_is_gone(self):
        """The defect: an arriving side kept the seed all season."""
        row = self._live(home_matches=5)

        assert row["Home_Past5Goals"] == pytest.approx(_ACTUAL_VALUE), (
            "match 6 must be 100% actual - the side's own form was computed "
            "and then overwritten with a static cohort average")

    def test_deep_into_the_season_it_is_still_gone(self):
        row = self._live(home_matches=12)

        assert row["Home_Past5Goals"] == pytest.approx(_ACTUAL_VALUE)

    def test_the_arrival_flag_still_survives_the_blend(self):
        """Route classification is unaffected; only the duration changes."""
        row = self._live(home_matches=12)

        assert row["Home_Team"] == "NEWCO"
        assert row["Home_Promoted"] == 1


class TestTheSeedWindowIsCountedPerVenue:
    """Training counts home and away matches separately; serving must too."""

    def test_home_matches_do_not_retire_the_away_seed(self):
        from predict import LivePredictor

        others = ["T01", "T02", "T03", "T04", "T05"]
        # Six home matches, none away: the away half is still at match 1.
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, home_matches=6)
        )
        predictor = LivePredictor.__new__(LivePredictor)
        predictor.verbose = False
        predictor._full_df = df

        row = predictor._fixture_feature_row(
            "OPP", "NEWCO", df, season=25, arrivals={"NEWCO"})

        assert row is not None
        assert row["Away_Past5Goals"] == pytest.approx(_COHORT_VALUE), (
            "the away half has no history yet and must still be seeded, "
            "however many home matches the side has played")


class TestBothPipelinesStillUseTheseWeights:
    """The serving blend is only correct while training agrees with it.

    Both pipelines declare the weights inline as a local. Matched line-wise
    rather than by substring, because a substring check here would pass
    against a dict that merely contains these pairs among others.
    """

    @pytest.mark.parametrize("module,func", [
        ("pipeline", "initialize_promoted_features"),
        ("championship_pipeline", "initialize_promoted_features"),
    ])
    def test_the_inline_weights_match_the_shared_constant(self, module, func):
        import importlib
        import inspect
        from division_movement import SEED_BLEND_WEIGHTS

        source = inspect.getsource(
            getattr(importlib.import_module(module), func))
        expected = ("blend_weights = "
                    "{1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}")

        assert any(line.strip() == expected for line in source.splitlines()), (
            f"{module}.{func} no longer blends on SEED_BLEND_WEIGHTS "
            f"({SEED_BLEND_WEIGHTS}); serving and training have diverged")

    @pytest.mark.parametrize("module,func", [
        ("pipeline", "initialize_promoted_features"),
        ("championship_pipeline", "initialize_promoted_features"),
    ])
    def test_the_cutoff_is_still_five(self, module, func):
        import importlib
        import inspect

        source = inspect.getsource(
            getattr(importlib.import_module(module), func))

        assert any(line.strip() == "if match_num > 5:"
                   for line in source.splitlines()), (
            f"{module}.{func} no longer stops at match 5")


class TestEFLServingMatchesTraining:
    """`championship_predict.ChampionshipPredictor._fixture_feature_row`."""

    @staticmethod
    def _live(home_matches: int, *, home=True) -> pd.Series:
        from championship_predict import ChampionshipPredictor
        from division_movement import PROMOTED, SeedParams

        others = ["T01", "T02", "T03", "T04", "T05"]
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, home_matches)
        )
        predictor = ChampionshipPredictor.__new__(ChampionshipPredictor)
        predictor.verbose = False
        predictor._full_df = df
        # A PL frame with the columns but no rows: NEWCO is absent from the
        # division above, so every arrival routes to PROMOTED, which is the
        # cohort this fixture's prior season describes. The columns have to
        # exist even when empty — `_season_teams` indexes SeasonIndex.
        predictor._pl_df = pd.DataFrame(
            columns=["SeasonIndex", "Date", "Home_Team", "Away_Team",
                     "Home_LeaguePosition", "Away_LeaguePosition"])
        predictor._seed_params_cache = SeedParams(priors={}, n_events=0)

        if home:
            return predictor._fixture_feature_row(
                "NEWCO", "OPP", df, season=25, arrivals={"NEWCO": PROMOTED})
        return predictor._fixture_feature_row(
            "OPP", "NEWCO", df, season=25, arrivals={"NEWCO": PROMOTED})

    def test_with_no_history_the_seed_is_the_whole_row(self):
        row = self._live(home_matches=0)

        assert row is not None
        assert row["Home_Past5Goals"] == pytest.approx(_COHORT_VALUE)

    @pytest.mark.parametrize("played,next_match", [(1, 2), (2, 3), (4, 5)])
    def test_partway_through_the_window_it_blends(self, played, next_match):
        row = self._live(home_matches=played)

        assert row["Home_Past5Goals"] == pytest.approx(_expected(next_match)), (
            f"after {played} home matches the next is match {next_match}, "
            f"which training blends at weight {_BLEND_WEIGHTS[next_match]}")

    def test_past_the_window_the_seed_is_gone(self):
        """Six arriving sides in a 24-team division were priced this way."""
        row = self._live(home_matches=5)

        assert row["Home_Past5Goals"] == pytest.approx(_ACTUAL_VALUE), (
            "match 6 must be 100% actual - the side's own form was computed "
            "and then overwritten with a static cohort average")

    def test_deep_into_the_season_it_is_still_gone(self):
        row = self._live(home_matches=12)

        assert row["Home_Past5Goals"] == pytest.approx(_ACTUAL_VALUE)

    def test_the_arrival_flag_still_survives_the_blend(self):
        row = self._live(home_matches=12)

        assert row["Home_Team"] == "NEWCO"
        assert row["Home_Promoted"] == 1

    def test_home_matches_do_not_retire_the_away_seed(self):
        """Training counts the venues separately; serving must too."""
        row = self._live(home_matches=6, home=False)

        assert row is not None
        assert row["Away_Past5Goals"] == pytest.approx(_COHORT_VALUE), (
            "the away half has no history yet and must still be seeded, "
            "however many home matches the side has played")


class TestTheDixonColesSeedRetires:
    """`_seed_dixon_coles` overwrote an arrival's rating on every scan.

    The seed exists for the window before a side has results in the division:
    its own docstring says the pre-season window is "exactly when a returning
    side is most dangerous, because Dixon-Coles still carries the rating from
    its *exit* season". But it was driven by `arrivals_for`, which is
    season-long membership, so the overwrite never stopped. The weekly retrain
    re-estimated the rating from real results and the next scan replaced it
    with the route prior again, all season, for one of the three EFL ensemble
    members.
    """

    class _StubDC:
        """Records what the seed was asked to overwrite."""

        def __init__(self):
            self.seeded: dict[str, str] = {}

        def seed_arrivals(self, incoming, priors, *, venues=None):
            self.seeded.update(incoming)
            self.venues = venues

    @staticmethod
    def _predictor(matches_played: int):
        from championship_predict import ChampionshipPredictor
        from division_movement import SeedParams

        others = ["T01", "T02", "T03", "T04", "T05"]
        # Played at *both* venues. The seed window is per venue, so a side
        # that had only ever played at home would keep its away seed however
        # long the season ran — correctly, but that is the other tests' case,
        # not this one. Duration is what these tests are about.
        df = pd.DataFrame(
            _prior_season(24, others + ["T06"])
            + _arrival_season(25, "NEWCO", others, matches_played,
                              away_matches=matches_played)
        )
        predictor = ChampionshipPredictor.__new__(ChampionshipPredictor)
        predictor.verbose = False
        predictor._full_df = df
        predictor._pl_df = pd.DataFrame(
            columns=["SeasonIndex", "Date", "Home_Team", "Away_Team",
                     "Home_LeaguePosition", "Away_LeaguePosition"])
        predictor._seed_params_cache = SeedParams(priors={}, n_events=0)
        stub = TestTheDixonColesSeedRetires._StubDC()
        predictor._ou_models = {"dc": stub}
        predictor._ou15_models = None
        predictor._btts_models = None
        return predictor, stub

    def test_before_a_ball_is_kicked_the_arrival_is_seeded(self):
        """The case the seed was written for, and it must keep working."""
        predictor, stub = self._predictor(matches_played=0)

        seeded = predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

        assert "NEWCO" in seeded
        assert "NEWCO" in stub.seeded

    def test_inside_the_window_the_arrival_is_still_seeded(self):
        predictor, stub = self._predictor(matches_played=2)

        seeded = predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

        assert "NEWCO" in seeded, (
            "two matches is not enough to rate a side on; the route prior is "
            "still the better estimate")

    def test_past_the_window_the_arrival_keeps_its_fitted_rating(self):
        """The defect: the retrain's estimate was overwritten every scan."""
        predictor, stub = self._predictor(matches_played=_SEED_WINDOW)

        seeded = predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

        assert "NEWCO" not in stub.seeded, (
            f"after {_SEED_WINDOW} matches the side has a real record and "
            "Dixon-Coles has fitted it; overwriting with the route prior "
            "discards every result it has played")
        assert "NEWCO" not in seeded

    def test_deep_into_the_season_it_is_still_not_reseeded(self):
        predictor, stub = self._predictor(matches_played=14)

        predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

        assert "NEWCO" not in stub.seeded

    def test_an_established_side_is_never_seeded(self):
        """Guards the test above: absence must mean the gate, not a typo."""
        predictor, stub = self._predictor(matches_played=0)

        predictor._seed_dixon_coles({"NEWCO", "T01", "T02"})

        assert "T01" not in stub.seeded
