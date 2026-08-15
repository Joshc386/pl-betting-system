"""Season boundaries must account for every season they can see (ADR 0009).

`temporal_split` partitions by *allowlist* — `isin(TRAIN)/isin(VAL)/isin(TEST)`
— so a season named in none of the three is dropped from all three, silently.
`model.py`'s walk-forward selection is a *denylist* (`~isin(TEST_SEASONS)`) and
includes that same season. Both run in the same job on the same data, so the
season vanishes from the final fit while appearing in a CV fold whose training
window has a hole where the test season should be.

Nothing errored. That is the defect: the partition has to *assert* it is
exhaustive, and name the season it cannot place.

Seasons below TRAIN_MIN_SEASON are a different case — they are excluded on
purpose (pre-xG era) and must stay silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pipeline


def _frame(seasons: list[int], per_season: int = 2) -> pd.DataFrame:
    """A frame carrying `per_season` rows for each season listed."""
    rows = [
        {"SeasonIndex": s, "Over_2_5": i % 2}
        for s in seasons
        for i in range(per_season)
    ]
    return pd.DataFrame(rows)


class TestExhaustivePartition:
    def test_unallocated_season_raises_and_names_it(self, monkeypatch):
        """Season 26 is in the data but in none of TRAIN/VAL/TEST."""
        monkeypatch.setattr(pipeline, "TRAIN_SEASONS", list(range(0, 24)))
        monkeypatch.setattr(pipeline, "VAL_SEASONS", [24])
        monkeypatch.setattr(pipeline, "TEST_SEASONS", [25])

        df = _frame([23, 24, 25, 26])

        with pytest.raises(pipeline.SeasonPartitionError) as exc:
            pipeline.temporal_split(df)

        assert "26" in str(exc.value)

    def test_configured_season_absent_from_data_does_not_raise(self, monkeypatch):
        """TEST_SEASONS=[26] with no season-26 rows yet is the normal state.

        This is what lets the 2026/27 boundaries land before football-data.co.uk
        publishes the season. An empty split is not an unallocated season.
        """
        monkeypatch.setattr(pipeline, "TRAIN_SEASONS", list(range(0, 25)))
        monkeypatch.setattr(pipeline, "VAL_SEASONS", [25])
        monkeypatch.setattr(pipeline, "TEST_SEASONS", [26])

        train, val, test = pipeline.temporal_split(_frame([23, 24, 25]))

        assert len(test) == 0
        assert set(val["SeasonIndex"]) == {25}

    def test_seasons_below_train_min_are_excluded_silently(self, monkeypatch):
        """Pre-xG seasons are dropped on purpose, not by accident.

        TRAIN_SEASONS starts at 0 but the train filter also requires
        >= TRAIN_MIN_SEASON, so seasons 0-13 are deliberately unused. They must
        not be reported as unallocated.
        """
        monkeypatch.setattr(pipeline, "TRAIN_SEASONS", list(range(0, 25)))
        monkeypatch.setattr(pipeline, "VAL_SEASONS", [25])
        monkeypatch.setattr(pipeline, "TEST_SEASONS", [26])

        train, _, _ = pipeline.temporal_split(_frame([0, 5, 13, 14, 24]))

        assert min(train["SeasonIndex"]) == 14

    def test_every_unallocated_season_is_named(self, monkeypatch):
        """A bare count would not tell you which boundary to move."""
        monkeypatch.setattr(pipeline, "TRAIN_SEASONS", list(range(0, 24)))
        monkeypatch.setattr(pipeline, "VAL_SEASONS", [24])
        monkeypatch.setattr(pipeline, "TEST_SEASONS", [25])

        with pytest.raises(pipeline.SeasonPartitionError) as exc:
            pipeline.temporal_split(_frame([24, 26, 27]))

        assert "26" in str(exc.value) and "27" in str(exc.value)


class TestEarlyStoppingSeason:
    """The Production Path picks its Early-Stopping Season from the data.

    `train_seasons[-1]` makes a newly started season the sole validation set on
    its *first* ingested fixture — a dozen August matches deciding XGBoost's
    tree count. A season must be big enough to judge a model on before it can
    judge one.
    """

    def test_thin_current_season_falls_back_to_previous(self):
        """Mid-August: season 26 has 12 fixtures, season 25 is complete."""
        from predictor_utils import seasons_for_validation

        seasons = pd.Series([24] * 380 + [25] * 380 + [26] * 12)

        eligible = seasons_for_validation(seasons)

        assert eligible[-1] == 25

    @pytest.mark.parametrize("count,expected", [(49, 25), (50, 26)])
    def test_threshold_boundary(self, count, expected):
        """50 fixtures is the bar walk_forward_cv already uses to skip a fold."""
        from predictor_utils import seasons_for_validation

        seasons = pd.Series([25] * 380 + [26] * count)

        assert seasons_for_validation(seasons)[-1] == expected

    def test_unstarted_season_is_not_selected(self):
        """A season with no fixtures yet cannot be the Early-Stopping Season."""
        from predictor_utils import seasons_for_validation

        seasons = pd.Series([24] * 380 + [25] * 380)

        assert seasons_for_validation(seasons)[-1] == 25

    def test_falls_back_past_consecutive_thin_seasons(self):
        """Fallback walks down the list, it does not stop at the first miss."""
        from predictor_utils import seasons_for_validation

        seasons = pd.Series([23] * 380 + [24] * 10 + [25] * 5 + [26] * 2)

        assert seasons_for_validation(seasons)[-1] == 23

    def test_base_rate_window_excludes_thin_seasons(self):
        """The Base Rate anchors on complete seasons (CONTEXT.md).

        Without this the window slides to [25, 26] the moment season 26 starts,
        halving the sample from 760 matches to 392 and dropping a whole season,
        silently. Regime detection is what tracks in-season drift.
        """
        from predictor_utils import seasons_for_validation

        seasons = pd.Series([24] * 380 + [25] * 380 + [26] * 12)

        assert seasons_for_validation(seasons)[-2:] == [24, 25]

    def test_degenerate_dataset_falls_back_and_warns(self):
        """No season clears the bar: keep training, but say so."""
        from predictor_utils import seasons_for_validation

        messages: list[str] = []
        seasons = pd.Series([22] * 10 + [23] * 20)

        eligible = seasons_for_validation(seasons, log=messages.append)

        assert eligible[-1] == 23
        assert any("WARNING" in m for m in messages)


class TestRefitAtBestIteration:
    """XGB/LGB were early-stopped and then kept, never refit (ADR 0009).

    So they alone never saw the Early-Stopping Season, while LogReg and
    Dixon-Coles — fitted on the full frame two lines below — did. Guarding the
    season choice without refitting would have made that staleness permanent
    rather than seasonal.

    `championship_model.py:493-511` already does this on the Research Path.
    """

    @staticmethod
    def _data(n=400, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 4))
        y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
        return X, y

    def test_xgb_refit_uses_best_iteration_and_keeps_hyperparameters(self):
        from model import train_xgb
        from predictor_utils import refit_at_best_iteration

        X, y = self._data()
        temp = train_xgb(X[:300], y[:300], X[300:], y[300:])

        refit = refit_at_best_iteration(temp, X, y)

        assert refit.n_estimators == temp.best_iteration
        assert refit.max_depth == temp.max_depth
        assert refit.learning_rate == temp.learning_rate
        assert refit.reg_lambda == temp.reg_lambda

    def test_xgb_refit_carries_no_early_stopping(self):
        """It is refit without an eval set, so early stopping must be off."""
        from model import train_xgb
        from predictor_utils import refit_at_best_iteration

        X, y = self._data()
        temp = train_xgb(X[:300], y[:300], X[300:], y[300:])

        refit = refit_at_best_iteration(temp, X, y)

        assert not refit.get_params().get("early_stopping_rounds")

    def test_model_that_never_early_stopped_is_still_refit(self):
        """No early stop is not a reason to keep the partially-fitted model.

        Returning the original here would silently restore the staleness this
        decision removes — the same shape of failure, one layer down.
        """
        import xgboost as xgb
        from predictor_utils import refit_at_best_iteration

        X, y = self._data()
        plain = xgb.XGBClassifier(n_estimators=17, max_depth=3,
                                  random_state=42, eval_metric="logloss")
        plain.fit(X[:300], y[:300])

        refit = refit_at_best_iteration(plain, X, y)

        assert refit is not plain
        assert refit.n_estimators == 17

    def test_lgb_refit_uses_best_iteration(self):
        from model import train_lgb
        from predictor_utils import refit_at_best_iteration

        X, y = self._data()
        names = [f"f{i}" for i in range(4)]
        temp = train_lgb(X[:300], y[:300], X[300:], y[300:], feature_names=names)

        refit = refit_at_best_iteration(temp, X, y, feature_names=names)

        assert refit.n_estimators == temp.best_iteration_
        assert refit.max_depth == temp.max_depth
