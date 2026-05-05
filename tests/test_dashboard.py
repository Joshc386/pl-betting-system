"""Tests for dashboard.py — market label formatting, DB operations, prediction tracking."""
import json
import os
import sqlite3
import tempfile
import pytest
from unittest.mock import patch

from dashboard import _format_market
from db import (
    save_recommendations, get_db, LEAGUE_DB_PATHS,
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
        patched_paths = {**LEAGUE_DB_PATHS, "PL": self.db_path}
        self._patcher = patch("dashboard.LEAGUE_DB_PATHS", patched_paths)
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
        patched_paths = {**LEAGUE_DB_PATHS, "PL": self.db_path}
        self._patcher = patch("dashboard.LEAGUE_DB_PATHS", patched_paths)
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
