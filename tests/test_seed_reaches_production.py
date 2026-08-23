"""The seed is only worth anything if the live path reaches it.

`tests/test_division_movement.py` and `tests/test_seed_scope.py` prove the
seeding helpers are correct. Neither proves they are *called*. That
distinction is load-bearing in this repo: `keep_system_awake` was thoroughly
tested and invoked by one of three jobs, and the fix for that shipped on this
same branch.

Two links in the chain had no coverage at all:

1. The measured `SeedParams` reach predict time through a pickle. If `train()`
   stopped populating them, or the "seed_params" key were dropped or renamed,
   `load_trained_state`'s fallback would quietly send every arrival back to
   the hand-picked prior bucket — the pre-ADR-0011 behaviour — and the whole
   suite would stay green. The failure is a log line, not an exception.

2. `generate_recommendations` calls the seeding helpers at all.

The second is checked at source level rather than by driving the real method,
which fetches live odds and runs the Freshness Gate. Matched line-wise rather
than by substring: `_seed_dixon_coles(` also occurs inside
`def _seed_dixon_coles(`, so a plain `in` check passes against a function that
only defines itself — the exact vacuous assertion this branch hit twice.
"""
from __future__ import annotations

import inspect

import pytest


class TestMeasuredSeedParamsSurviveThePickle:
    """Round-trip through the file the scheduler actually writes."""

    @staticmethod
    def _predictor():
        from championship_predict import ChampionshipPredictor

        p = ChampionshipPredictor.__new__(ChampionshipPredictor)
        p.verbose = False
        for attr in ("_ou_models", "_ou15_models", "_btts_models",
                     "_ou_features", "_ou15_features", "_btts_features",
                     "_ou_base_rate", "_ou15_base_rate", "_btts_base_rate",
                     "_dc_kwargs", "_ou15_dc_kwargs", "_btts_dc_kwargs",
                     "_our_teams"):
            setattr(p, attr, {})
        p._cal_shifts = {}
        p._val_mean_logits = {}
        return p

    def test_priors_and_sample_size_both_come_back(self, tmp_path):
        from division_movement import PROMOTED, RELEGATED, SeedParams

        priors = {
            RELEGATED: {"attack_home": 1.16, "attack_away": 0.99,
                        "defence_home": 0.80, "defence_away": 0.91},
            PROMOTED: {"attack_home": 1.00, "attack_away": 0.92,
                       "defence_home": 1.05, "defence_away": 1.09},
        }
        path = str(tmp_path / "state.pkl")

        saver = self._predictor()
        saver._seed_params_cache = SeedParams(priors=priors, n_events=150)
        saver.save_trained_state(path)

        loader = self._predictor()
        assert loader.load_trained_state(path) is True
        assert loader._seed_params().priors == priors, (
            "the measured priors did not survive the pickle - every arrival "
            "would silently fall back to the hand-picked bucket")
        assert loader._seed_params().n_events == 150

    def test_a_pre_adr_pickle_falls_back_rather_than_crashing(self, tmp_path):
        """An older state file has no seed_params key at all."""
        import pickle

        path = tmp_path / "legacy.pkl"
        saver = self._predictor()
        saver._seed_params_cache = None
        saver.save_trained_state(str(path))

        with open(path, "rb") as fh:
            state = pickle.load(fh)
        del state["seed_params"]
        with open(path, "wb") as fh:
            pickle.dump(state, fh)

        loader = self._predictor()
        assert loader.load_trained_state(str(path)) is True
        assert loader._seed_params().priors == {}
        assert loader._seed_params().n_events == 0


class TestGenerateRecommendationsReachesTheSeed:
    """The helpers being correct is not the same as their being called.

    Both predictors split the public method from a private body, the same way
    `job_weekly_retrain` wraps `_weekly_retrain`, so both links are asserted:
    either alone can be true while the seed goes unreached.
    """

    @pytest.mark.parametrize("module,cls", [
        ("championship_predict", "ChampionshipPredictor"),
        ("predict", "LivePredictor"),
    ])
    def test_the_public_method_runs_the_body(self, module, cls):
        import importlib

        predictor = getattr(importlib.import_module(module), cls)
        source = inspect.getsource(predictor.generate_recommendations)

        assert any(
            line.strip().startswith("return self._generate_recommendations_body(")
            or line.strip() == "self._generate_recommendations_body()"
            or "self._generate_recommendations_body(" in line
            for line in source.splitlines()), (
            f"{cls}.generate_recommendations no longer runs the body that "
            f"builds seeded rows")

    @pytest.mark.parametrize("module,cls", [
        ("championship_predict", "ChampionshipPredictor"),
        ("predict", "LivePredictor"),
    ])
    def test_the_body_builds_rows_through_the_seeded_builder(self, module, cls):
        import importlib

        predictor = getattr(importlib.import_module(module), cls)
        source = inspect.getsource(predictor._generate_recommendations_body)

        assert any("self._fixture_feature_row(" in line
                   for line in source.splitlines()), (
            f"{cls} no longer builds rows through _fixture_feature_row, so "
            f"arrivals are skipped again")

    def test_the_efl_body_seeds_dixon_coles(self):
        """PL has no equivalent yet - ADR 0011's unbuilt half."""
        from championship_predict import ChampionshipPredictor

        source = inspect.getsource(
            ChampionshipPredictor._generate_recommendations_body)

        assert any("self._seed_dixon_coles(" in line
                   for line in source.splitlines()), (
            "the EFL body no longer seeds Dixon-Coles, so an arrival is "
            "priced on whatever rating it carried out of its previous "
            "division")

    def test_the_seed_is_not_merely_defined(self):
        """Guards the tests above against matching a definition line.

        `_seed_dixon_coles(` occurs inside `def _seed_dixon_coles(`. A
        substring check over the whole class would pass against a helper
        nothing calls, which is how two assertions on this branch came to
        prove nothing.
        """
        from championship_predict import ChampionshipPredictor

        definition = inspect.getsource(
            ChampionshipPredictor._seed_dixon_coles)
        caller = inspect.getsource(
            ChampionshipPredictor._generate_recommendations_body)

        assert "self._seed_dixon_coles(" not in definition
        assert "self._seed_dixon_coles(" in caller
