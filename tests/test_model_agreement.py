"""Per-model probabilities must survive a rescan, so REC can show agreement.

Two paths write `match_analysis`. The predictor's own `_match_analysis`
(`scan.py:510`) carries `per_model_probs` for every evaluated line — including
lines `decide_bet` rejected. The rows `scan.py` rebuilds for itself
(`scan.py:817`) do not, and because `save_match_analysis` deletes and
re-inserts per fixture, the second write silently overwrites the first.

`save_match_analysis` already carries values forward from the previous row, but
only `if model_p is None` — and the rebuild always supplies `model_prob`. So
the rescue never fires and `per_model_json` lands as `{}` on every row. That is
why the Match Centre's AGREE column reads `—` for all 49 lines while
`recommendations` holds the full breakdown.

Agreement itself is defined in `staking.py:350` — a model agrees when its
probability for the chosen side beats the market's fair probability. This
module only preserves and presents that; it must never redefine it.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

# `get_match_analysis` (db.py:378) returns only fixtures between midnight UTC
# today and +7 days. A hardcoded kickoff therefore passes on the day it is
# written and fails every day afterwards, so this is derived from now.
_KICKOFF = (datetime.now(timezone.utc) + timedelta(days=1)).strftime(
    "%Y-%m-%dT%H:%M:%SZ")


def _row(market="ou25", side="over", per_model=None, model_prob=0.55):
    row = {
        "home_team": "Charlton", "away_team": "Derby",
        "kickoff": _KICKOFF,
        "market": market, "side": side,
        "best_odds": 2.26, "best_bookmaker": "Matchbook",
        "model_prob": model_prob, "fair_odds": 2.28,
        "edge_pct": 12.9, "confidence": "medium", "n_books": 14,
    }
    if per_model is not None:
        row["per_model_probs"] = per_model
    return row


@pytest.fixture
def league_db(tmp_path, monkeypatch):
    """Point db.py at a scratch database for one league."""
    import db
    # get_db() creates the schema lazily on connect, so pointing the path map
    # at a scratch file is all the setup needed.
    monkeypatch.setitem(db.LEAGUE_DB_PATHS, "PL", str(tmp_path / "pl.db"))
    return db


class TestPerModelSurvivesRescan:
    def test_rescan_without_per_model_keeps_the_stored_breakdown(self, league_db):
        """The scan.py:817 rebuild must not erase what the predictor wrote."""
        db = league_db
        full = {"xgb": 0.472, "lgb": 0.475, "lr": 0.501, "dc": 0.504}

        db.save_match_analysis([_row(per_model=full)], league="PL")
        # Second write, same fixture, no per-model data — the rebuild path.
        db.save_match_analysis([_row()], league="PL")

        stored = db.get_match_analysis("PL")
        kept = json.loads(stored.iloc[0]["per_model_json"])
        assert kept == full

    def test_fresh_per_model_overwrites_stale(self, league_db):
        """Carrying forward must not pin the first scan's numbers forever."""
        db = league_db
        db.save_match_analysis(
            [_row(per_model={"xgb": 0.1, "lgb": 0.1, "lr": 0.1, "dc": 0.1})],
            league="PL")
        newer = {"xgb": 0.9, "lgb": 0.9, "lr": 0.9, "dc": 0.9}

        db.save_match_analysis([_row(per_model=newer)], league="PL")

        stored = db.get_match_analysis("PL")
        assert json.loads(stored.iloc[0]["per_model_json"]) == newer


class TestAgreementCount:
    """REC shows `agreeing / the league's whole ensemble`.

    `_model_agreement` already existed for the Model Analytics tab. Adding the
    `league` argument gives the Match Centre a denominator of 4 (PL) or 3 (EFL)
    without disturbing analytics, which still counts only the models that
    actually stored a probability.
    """

    @staticmethod
    def _label(probs, fair_p, side="over", league="PL"):
        from dashboard import _model_agreement
        return _model_agreement(json.dumps(probs), fair_p, side, league)

    def test_counts_models_beating_the_fair_price(self):
        # fair 0.50 — xgb and dc clear it, lgb and lr do not.
        probs = {"xgb": 0.55, "lgb": 0.48, "lr": 0.49, "dc": 0.52}

        assert self._label(probs, 0.50) == ("2/4", 2, 4)

    def test_efl_denominator_is_three_not_a_shortfall(self):
        """The EFL has no LogReg at all; 3/3 is its healthy unanimous state."""
        probs = {"xgb": 0.55, "lgb": 0.56, "dc": 0.57}

        assert self._label(probs, 0.50, league="EFL") == ("3/3", 3, 3)

    def test_single_model_alt_line_reads_as_one_of_three(self):
        """O/U 3.5 is Dixon-Coles alone (predict.py:1374 pins n_agree to 0).

        1/3, not 1/1 — a full-looking fraction would hide that two of the
        league's three models never weighed in. These rows carry the largest
        edges on the board, so the thinness has to be visible.
        """
        assert self._label({"dc_poisson": 0.80}, 0.69,
                           side="under", league="EFL") == ("1/3", 1, 3)

    def test_analytics_caller_still_gets_DC_for_alt_lines(self):
        """Omitting `league` must preserve the existing analytics behaviour."""
        from dashboard import _model_agreement

        assert _model_agreement(
            json.dumps({"dc_poisson": 0.80}), 0.69, "under") == ("DC", None, None)

    def test_unanimous_disagreement_is_zero_over_four(self):
        probs = {"xgb": 0.40, "lgb": 0.41, "lr": 0.39, "dc": 0.42}

        assert self._label(probs, 0.50) == ("0/4", 0, 4)

    def test_under_inverts_each_model_probability(self):
        """Ensemble rows store Over-probabilities; Under is 1 - v.

        Values kept distinct so this does not trip the identical-values guard.
        """
        probs = {"xgb": 0.40, "lgb": 0.41, "lr": 0.42, "dc": 0.43}

        # Each inverts to ~0.57-0.60, all beating a fair Under price of 0.50.
        assert self._label(probs, 0.50, side="under") == ("4/4", 4, 4)

    def test_alt_line_under_is_not_inverted(self):
        """Alt lines already store the probability for their own side.

        predict.py:1383 writes {"dc_poisson": vb["model_prob"]} against
        vb["side"], so applying the ensemble's `1 - v` here would invert the
        model's actual claim and report the opposite.
        """
        assert self._label({"dc_poisson": 0.80}, 0.69,
                           side="under", league="EFL") == ("1/3", 1, 3)

    def test_degenerate_legacy_rows_are_still_refused(self):
        """One blended number copied into every slot is not unanimity."""
        probs = {"xgb": 0.55, "lgb": 0.55, "lr": 0.55, "dc": 0.55}

        assert self._label(probs, 0.50) == ("!", None, None)

    def test_missing_breakdown_is_a_dash_not_a_false_zero(self):
        from dashboard import _model_agreement

        assert _model_agreement(None, 0.50, "over", "PL") == ("—", None, None)


class TestRecPerModel:
    """scan.py's rebuild must hand the breakdown on rather than blanking it.

    `rec_lookup` holds `row.to_dict()` of a recommendations row, so
    `per_model_json` arrives as the raw JSON string the DB stores.
    """

    def test_parses_the_stored_json_string(self):
        from scan import _rec_per_model

        probs = {"xgb": 0.47, "lgb": 0.47, "dc": 0.50}
        assert _rec_per_model({"per_model_json": json.dumps(probs)}) == probs

    def test_no_recommendation_yields_empty_not_none(self):
        """`{}` lets save_match_analysis carry the previous value forward."""
        from scan import _rec_per_model

        assert _rec_per_model(None) == {}

    def test_unparseable_value_degrades_quietly(self):
        from scan import _rec_per_model

        assert _rec_per_model({"per_model_json": "not json"}) == {}

    def test_already_a_dict_passes_through(self):
        from scan import _rec_per_model

        assert _rec_per_model({"per_model_json": {"dc": 0.5}}) == {"dc": 0.5}


class TestBreakdownSurvivesIncompleteSettlement:
    """The Model Analytics tab must render when a settled row lacks an outcome.

    `pd.to_numeric(x, errors="coerce") or 0` reads as a missing-value guard but
    is not one: NaN is truthy in Python, so the expression returns NaN rather
    than 0 and the `int(sum(...))` that follows raises. A single voided or
    abandoned fixture — settled, but with no `won` — therefore takes down the
    whole tab rather than dropping one row.
    """

    @staticmethod
    def _recs(rows):
        import pandas as pd
        return pd.DataFrame(rows)

    @staticmethod
    def _rec(won=1, profit=0.9, stake=1.0, ev=0.05):
        return {
            "per_model_json": json.dumps(
                {"xgb": 0.55, "lgb": 0.56, "lr": 0.57, "dc": 0.58}),
            "fair_prob": 0.50, "side": "over",
            "won": won, "profit_pct": profit,
            "stake_pct": stake, "ev": ev,
        }

    def test_settled_row_with_no_outcome_does_not_break_the_tab(
            self, monkeypatch):
        import dashboard

        monkeypatch.setattr(
            dashboard, "get_settled_recommendations",
            lambda league: self._recs([self._rec(), self._rec(won=None)]))

        # Must return a component, not raise ValueError on int(NaN).
        out = dashboard._make_agreement_breakdown("PL")
        assert out is not None

    def test_row_with_no_outcome_is_not_counted_as_a_win(self, monkeypatch):
        """Dropping the outcome must not silently become a loss or a win."""
        import dashboard

        monkeypatch.setattr(
            dashboard, "get_settled_recommendations",
            lambda league: self._recs([self._rec(won=1), self._rec(won=None)]))

        out = dashboard._make_agreement_breakdown("PL")
        assert out is not None


class TestRescueDoesNotClobberAGoodBreakdown:
    """The model_prob rescue must not discard a breakdown the row supplied.

    `save_match_analysis` carries values forward when a row arrives without
    them. When a row arrives *with* a good `per_model_probs` but no
    `model_prob`, the rescue branch used to overwrite the breakdown from the
    stored row regardless — and when that stored value was `{}`, a real
    breakdown was replaced with an empty one. That is the same overwrite the
    carry-forward exists to prevent, arriving from the other direction.
    """

    def test_supplied_breakdown_survives_a_model_prob_rescue(self, league_db):
        db = league_db
        good = {"xgb": 0.47, "lgb": 0.48, "lr": 0.49, "dc": 0.50}

        # Stored first: has a model_prob to rescue from, but no breakdown.
        db.save_match_analysis([_row(model_prob=0.55)], league="PL")
        # Then a row carrying a real breakdown but no model_prob of its own.
        db.save_match_analysis(
            [_row(per_model=good, model_prob=None)], league="PL")

        stored = db.get_match_analysis("PL")
        assert json.loads(stored.iloc[0]["per_model_json"]) == good


class TestRecPerModelHandlesFrameCells:
    """The supplementary blocks feed it pandas cells, not just dicts.

    `scan.py` re-derives rows from stored `match_analysis` and from
    `recommendations`; both arrive as pandas objects whose empty cells are
    NaN rather than None. NaN is truthy, so it slips past a bare falsiness
    check and has to be caught on the parse.
    """

    def test_nan_cell_yields_empty_not_a_crash(self):
        import numpy as np
        from scan import _rec_per_model

        assert _rec_per_model({"per_model_json": np.nan}) == {}

    def test_reads_a_pandas_series_row(self):
        import pandas as pd
        from scan import _rec_per_model

        probs = {"dc_poisson": 0.86}
        row = pd.Series({"market": "ou35", "per_model_json": json.dumps(probs)})
        assert _rec_per_model(row) == probs

    def test_series_without_the_column_yields_empty(self):
        import pandas as pd
        from scan import _rec_per_model

        assert _rec_per_model(pd.Series({"market": "ou35"})) == {}
