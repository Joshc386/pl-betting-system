"""Tests for dashboard.py — market label formatting, DB operations, prediction tracking."""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch

from dashboard import _format_market
from db import (
    save_recommendations, get_active_recommendations, get_db, LEAGUE_DB_PATHS,
    log_predictions, get_predictions, toggle_prediction_taken,
)


class TestFormatMarket:
    """Tests for the _format_market helper."""

    def test_ou25(self) -> None:
        assert _format_market("ou25") == "O/U 2.5"

    def test_ou15(self) -> None:
        assert _format_market("ou15") == "O/U 1.5"

    def test_ou35(self) -> None:
        assert _format_market("ou35") == "O/U 3.5"

    def test_ou45(self) -> None:
        assert _format_market("ou45") == "O/U 4.5"

    def test_btts(self) -> None:
        assert _format_market("btts") == "BTTS"

    def test_none_returns_empty(self) -> None:
        assert _format_market(None) == ""

    def test_unknown_code_passthrough(self) -> None:
        assert _format_market("corners") == "corners"

    def test_malformed_ou_passthrough(self) -> None:
        assert _format_market("ouXYZ") == "ouXYZ"


class TestSaveRecommendations:
    """Tests for saving and deduplicating recommendations."""

    @pytest.fixture(autouse=True)
    def _setup_temp_db(self, tmp_path):
        """Use a temporary database for each test."""
        self.db_path = str(tmp_path / "test.db")
        self._patcher = patch.dict("db.LEAGUE_DB_PATHS", {"PL": self.db_path})
        self._patcher.start()
        yield
        self._patcher.stop()

    def _make_rec(self, **overrides) -> dict:
        base = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": "2026-04-05T15:00:00",
            "market": "ou25",
            "side": "over",
            "model_prob": 0.6,
            "blended_prob": 0.58,
            "fair_prob": 0.55,
            "odds": 1.95,
            "edge": 0.05,
            "ev": 0.03,
            "stake_pct": 0.02,
            "confidence": "medium",
            "best_bookmaker": "Pinnacle",
            "n_books": 5,
            "n_agree": 3,
            "per_model_probs": {"xgb": 0.6, "lgb": 0.55},
        }
        base.update(overrides)
        return base

    def test_save_single_recommendation(self) -> None:
        n = save_recommendations([self._make_rec()], league="PL")
        assert n == 1

    def test_deduplication(self) -> None:
        rec = self._make_rec()
        save_recommendations([rec], league="PL")
        n = save_recommendations([rec], league="PL")
        assert n == 0  # duplicate, not saved again

    def test_rescan_updates_existing_recommendation(self) -> None:
        """A KO-1h re-scan must update an existing unsettled row's odds and
        edge in place rather than skip it, so late-scan price moves are not
        lost. The return value still counts only NEW inserts (0 here)."""
        save_recommendations([self._make_rec(odds=1.95, edge=0.05)], league="PL")
        n = save_recommendations(
            [self._make_rec(odds=2.10, edge=0.08)], league="PL")
        assert n == 0  # update, not a new insert
        active = get_active_recommendations(league="PL")
        assert len(active) == 1  # row updated in place, not duplicated
        assert active.iloc[0]["odds"] == 2.10  # odds refreshed
        assert active.iloc[0]["edge"] == 0.08  # edge refreshed

    def test_different_markets_not_duplicated(self) -> None:
        save_recommendations([self._make_rec(market="ou25")], league="PL")
        n = save_recommendations([self._make_rec(market="ou15")], league="PL")
        assert n == 1  # different market, should save

    def test_different_fixtures_not_duplicated(self) -> None:
        save_recommendations([self._make_rec()], league="PL")
        n = save_recommendations([self._make_rec(home_team="Liverpool")], league="PL")
        assert n == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictionTracking:
    """Tests for log_predictions, get_predictions, toggle_prediction_taken."""

    @pytest.fixture(autouse=True)
    def _setup_temp_db(self, tmp_path):
        self.db_path = str(tmp_path / "test.db")
        self._patcher = patch.dict("db.LEAGUE_DB_PATHS", {"PL": self.db_path})
        self._patcher.start()
        yield
        self._patcher.stop()

    def _make_pred(self, **overrides) -> dict:
        base = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "kickoff": "2026-04-12T15:00:00",
            "market": "ou25",
            "side": "over",
            "model_prob": 0.62,
            "fair_odds": 1.85,
            "edge_pct": 3.5,
            "best_odds": 1.95,
            "best_bookmaker": "Pinnacle",
            "confidence": "medium",
            "bookmaker_odds": {"Pinnacle": 1.95, "Bet365": 1.90},
        }
        base.update(overrides)
        return base

    def test_log_positive_edge_prediction(self) -> None:
        n = log_predictions([self._make_pred()], league="PL")
        assert n == 1

    def test_negative_edge_not_logged(self) -> None:
        n = log_predictions([self._make_pred(edge_pct=-1.5)], league="PL")
        assert n == 0

    def test_zero_edge_not_logged(self) -> None:
        n = log_predictions([self._make_pred(edge_pct=0)], league="PL")
        assert n == 0

    def test_deduplication_on_rescan(self) -> None:
        log_predictions([self._make_pred()], league="PL")
        n = log_predictions([self._make_pred()], league="PL")
        assert n == 0  # Same fixture+market+side → skip

    def test_different_market_not_deduplicated(self) -> None:
        log_predictions([self._make_pred(market="ou25")], league="PL")
        n = log_predictions([self._make_pred(market="btts", side="yes")], league="PL")
        assert n == 1

    def test_get_predictions_returns_dataframe(self) -> None:
        log_predictions([self._make_pred()], league="PL")
        df = get_predictions(league="PL")
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Arsenal"
        assert df.iloc[0]["market"] == "ou25"

    def test_get_predictions_settled_only(self) -> None:
        log_predictions([self._make_pred()], league="PL")
        df = get_predictions(league="PL", settled_only=True)
        assert len(df) == 0  # Nothing settled yet

    def test_toggle_taken(self) -> None:
        log_predictions([self._make_pred()], league="PL")
        df = get_predictions(league="PL")
        pred_id = int(df.iloc[0]["id"])

        # Initially not taken
        assert int(df.iloc[0]["taken"]) == 0

        # Mark as taken
        toggle_prediction_taken(pred_id, True, league="PL")
        df = get_predictions(league="PL")
        assert int(df.iloc[0]["taken"]) == 1

        # Mark as not taken
        toggle_prediction_taken(pred_id, False, league="PL")
        df = get_predictions(league="PL")
        assert int(df.iloc[0]["taken"]) == 0

    def test_bookmaker_odds_json_stored(self) -> None:
        log_predictions([self._make_pred()], league="PL")
        df = get_predictions(league="PL")
        stored = json.loads(df.iloc[0]["bookmaker_odds_json"])
        assert stored["Pinnacle"] == 1.95
        assert stored["Bet365"] == 1.90

    def test_multiple_predictions_logged(self) -> None:
        preds = [
            self._make_pred(),
            self._make_pred(home_team="Liverpool", away_team="Man Utd"),
            self._make_pred(market="btts", side="yes"),
        ]
        n = log_predictions(preds, league="PL")
        assert n == 3
        df = get_predictions(league="PL")
        assert len(df) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Match Centre — REC column and conditional-style precedence
# ═══════════════════════════════════════════════════════════════════════════════

def _find_datatable(component):
    """Walk a Dash component tree and return the first DataTable."""
    from dash import dash_table
    if isinstance(component, dash_table.DataTable):
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        found = _find_datatable(child)
        if found is not None:
            return found
    return None


class TestConditionalStyleOrder:
    """Dash folds every matching conditional style with ``Object.assign`` in
    list order, so the LAST match wins. Row-level rules must therefore be
    declared before column-level ones.

    This is a regression guard, not a style preference. With the row rules
    last, the odd-row stripe overwrote the high-confidence tint (so the same
    bet was green or not depending purely on where it sorted) and both row
    rules wiped the Stake % tier backgrounds on every odd row.
    """

    def _styles(self):
        from dashboard import _build_match_centre
        table = _find_datatable(_build_match_centre("PL", show_all=True))
        assert table is not None, "Match Centre rendered no DataTable"
        return table.style_data_conditional

    @staticmethod
    def _is_row_level(rule) -> bool:
        return "column_id" not in rule.get("if", {})

    def test_row_rules_precede_column_rules(self) -> None:
        styles = self._styles()
        row_idx = [i for i, s in enumerate(styles) if self._is_row_level(s)]
        col_idx = [i for i, s in enumerate(styles) if not self._is_row_level(s)]
        assert row_idx and col_idx, "expected both row- and column-level rules"
        assert max(row_idx) < min(col_idx), (
            "a row-level rule is declared after a column-level one; it will "
            "overwrite that column's background because Dash lets the last "
            "match win"
        )

    def test_high_confidence_is_styled_for_both_parities(self) -> None:
        """Confidence must not depend on row parity.

        A single unqualified high-confidence rule is what let the stripe
        swallow it; the fix is one rule per parity, so both must be present.
        """
        styles = self._styles()
        conf_rules = [s for s in styles
                      if '{confidence} = "high"' in s.get("if", {})
                      .get("filter_query", "")]
        parities = {s["if"].get("row_index") for s in conf_rules}
        assert parities == {"odd", "even"}, (
            f"high-confidence styled for parities {parities}; both required "
            "or the zebra stripe reintroduces position-dependent colour"
        )

    def test_stake_tier_backgrounds_survive_row_rules(self) -> None:
        """Stake % tiers set their own background and must outrank rows."""
        styles = self._styles()
        stake_idx = [i for i, s in enumerate(styles)
                     if s.get("if", {}).get("column_id") == "stake_pct"
                     and "backgroundColor" in s]
        row_bg_idx = [i for i, s in enumerate(styles)
                      if self._is_row_level(s) and "backgroundColor" in s]
        assert stake_idx, "expected Stake % tier backgrounds"
        assert min(stake_idx) > max(row_bg_idx)


class TestRecColumn:
    """REC must distinguish which filter a recommendation actually cleared.

    Ensemble bets pass ``staking.decide_bet``, which rejects n_agree < 2.
    Alt lines never call it — ``predict.py`` appends them with n_agree=0,
    so they face neither the agreement gate nor edge shrinkage. Both used
    to render an identical tick.
    """

    @pytest.fixture(autouse=True)
    def _setup_temp_db(self, tmp_path):
        self.db_path = str(tmp_path / "rec.db")
        self._patcher = patch.dict("db.LEAGUE_DB_PATHS", {"PL": self.db_path})
        self._patcher.start()
        yield
        self._patcher.stop()

    def _seed(self, market: str, n_agree: int, per_model: dict) -> None:
        """Write one analysis row and a matching recommendation for it.

        The kickoff is two days out on purpose: ``get_match_analysis`` drops
        fixtures from previous days and anything more than 7 days ahead, so a
        fixed far-future date would be filtered away before it reached REC.
        """
        from datetime import datetime, timedelta
        from db import save_match_analysis
        kickoff = (datetime.now() + timedelta(days=2)).strftime(
            "%Y-%m-%dT15:00:00")
        save_match_analysis([{
            "home_team": "Arsenal", "away_team": "Chelsea",
            "kickoff": kickoff, "matchday": 1,
            "market": market, "side": "over",
            "best_odds": 2.00, "best_bookmaker": "Pinnacle",
            "model_prob": 0.60, "fair_odds": 2.20, "edge_pct": 5.5,
            "confidence": "high", "n_books": 5,
            "per_model_json": json.dumps(per_model),
            "bookmaker_odds_json": json.dumps({"Pinnacle": 2.00}),
            "edge_source": "pinnacle",
        }], league="PL")
        save_recommendations([{
            "home_team": "Arsenal", "away_team": "Chelsea",
            "kickoff": kickoff, "market": market, "side": "over",
            "model_prob": 0.60, "blended_prob": 0.50, "fair_prob": 0.455,
            "odds": 2.00, "edge": 0.045, "ev": 0.02, "stake_pct": 0.02,
            "confidence": "high", "best_bookmaker": "Pinnacle", "n_books": 5,
            "n_agree": n_agree, "per_model_probs": per_model,
        }], league="PL")

    def _rec_row(self):
        from dashboard import _build_match_centre
        table = _find_datatable(_build_match_centre("PL", show_all=True))
        rows = [r for r in table.data if r.get("rec")]
        assert len(rows) == 1, f"expected exactly one REC row, got {len(rows)}"
        return rows[0]

    def test_alt_line_rec_is_marked_dc(self) -> None:
        """n_agree=0 can only come from the DC-only alt-line path."""
        self._seed("ou35", n_agree=0, per_model={"dc_poisson": 0.60})
        row = self._rec_row()
        assert row["rec"] == "\u2713 DC"
        assert row["_rec_alt"] == 1

    def test_ensemble_rec_is_a_plain_tick(self) -> None:
        self._seed("ou25", n_agree=3,
                   per_model={"xgb": 0.60, "lgb": 0.58, "lr": 0.57, "dc": 0.59})
        row = self._rec_row()
        assert row["rec"] == "\u2713"
        assert row["_rec_alt"] == 0

    def test_missing_n_agree_does_not_claim_dc(self) -> None:
        """A legacy row with no n_agree must not be libelled as DC-only.

        A plain tick understates an alt line; a false "DC" misreports a
        genuine ensemble bet. The conservative direction is the plain tick.
        """
        self._seed("ou25", n_agree=None,
                   per_model={"xgb": 0.60, "lgb": 0.58, "dc": 0.59})
        row = self._rec_row()
        assert row["rec"] == "\u2713"
        assert row["_rec_alt"] == 0
