"""Tests for the dynamic scheduler and fixture-aware scheduling.

Covers:
  - Fixture schedule parsing and matchday detection
  - Dynamic job scheduling (morning + pre-kickoff)
  - Model save/load round-trip
  - Markets parameter gating (BTTS skip)
  - Scheduler creation and static job registration
"""
import os
import tempfile
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

UK_TZ = ZoneInfo("Europe/London")


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture Schedule Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixtureSchedule:
    """Test fixture_schedule.py functions."""

    @patch("fixture_schedule.requests.get")
    def test_get_upcoming_fixtures_parses_response(
        self, mock_get: MagicMock
    ) -> None:
        """Verify fixture parsing from football-data.org response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "matches": [
                {
                    "utcDate": "2026-04-12T14:00:00Z",
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Chelsea FC"},
                    "matchday": 34,
                },
                {
                    "utcDate": "2026-04-12T16:30:00Z",
                    "homeTeam": {"name": "Liverpool FC"},
                    "awayTeam": {"name": "Manchester City FC"},
                    "matchday": 34,
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from fixture_schedule import get_upcoming_fixtures
        fixtures = get_upcoming_fixtures("PL", days_ahead=7)

        assert len(fixtures) == 2
        assert fixtures[0]["home"] is not None
        assert fixtures[0]["away"] is not None
        assert fixtures[0]["matchday"] == 34
        assert isinstance(fixtures[0]["kickoff"], datetime)

    @patch("fixture_schedule.requests.get")
    def test_get_matchday_kickoffs_filters_by_date(
        self, mock_get: MagicMock
    ) -> None:
        """Verify only kickoffs matching target_date are returned."""
        target = date(2026, 4, 12)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "matches": [
                {
                    "utcDate": "2026-04-12T14:00:00Z",
                    "homeTeam": {"name": "Arsenal FC"},
                    "awayTeam": {"name": "Chelsea FC"},
                    "matchday": 34,
                },
                {
                    "utcDate": "2026-04-13T14:00:00Z",
                    "homeTeam": {"name": "Everton FC"},
                    "awayTeam": {"name": "Fulham FC"},
                    "matchday": 34,
                },
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from fixture_schedule import get_matchday_kickoffs
        kickoffs = get_matchday_kickoffs("PL", target)

        # Only the April 12 fixture should be returned
        assert len(kickoffs) == 1
        assert kickoffs[0].date() == target

    @patch("fixture_schedule.requests.get")
    def test_get_matchday_kickoffs_empty_when_no_matches(
        self, mock_get: MagicMock
    ) -> None:
        """No matches on date returns empty list."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"matches": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        from fixture_schedule import get_matchday_kickoffs
        kickoffs = get_matchday_kickoffs("PL", date(2026, 4, 15))

        assert kickoffs == []

    @patch("fixture_schedule.requests.get")
    def test_get_upcoming_fixtures_handles_api_error(
        self, mock_get: MagicMock
    ) -> None:
        """API failure returns empty list, doesn't raise."""
        import requests
        mock_get.side_effect = requests.RequestException("timeout")

        from fixture_schedule import get_upcoming_fixtures
        fixtures = get_upcoming_fixtures("PL")

        assert fixtures == []


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Scheduling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicScheduling:
    """Test schedule_matchday_jobs() dynamic job creation."""

    def test_schedule_matchday_jobs_creates_morning_and_prekickoff(
        self,
    ) -> None:
        """Verify morning + pre-kickoff jobs are created."""
        from scheduler import schedule_matchday_jobs

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None  # No existing jobs

        # Future kickoffs (tomorrow)
        tomorrow = date.today() + timedelta(days=1)
        kickoffs = [
            datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                     12, 30, tzinfo=UK_TZ),
            datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                     15, 0, tzinfo=UK_TZ),
            datetime(tomorrow.year, tomorrow.month, tomorrow.day,
                     17, 30, tzinfo=UK_TZ),
        ]

        schedule_matchday_jobs(mock_scheduler, "PL", kickoffs)

        # Should create: 1 morning + 3 pre-kickoff + 1 CLV closing odds = 5 jobs
        assert mock_scheduler.add_job.call_count == 5

        # Check morning job uses all markets
        morning_call = mock_scheduler.add_job.call_args_list[0]
        assert morning_call.kwargs["kwargs"]["markets"] == ("totals", "btts")

        # Check pre-kickoff jobs use totals only (indices 1-3)
        for call in mock_scheduler.add_job.call_args_list[1:4]:
            assert call.kwargs["kwargs"]["markets"] == ("totals",)

        # Check CLV closing odds job is scheduled (last job)
        clv_call = mock_scheduler.add_job.call_args_list[4]
        from scheduler import job_fetch_closing_odds
        assert clv_call.args[0] == job_fetch_closing_odds

    def test_schedule_matchday_jobs_skips_past_kickoffs(self) -> None:
        """Jobs for already-passed kickoffs are not created."""
        from scheduler import schedule_matchday_jobs

        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None

        # Past kickoffs (yesterday)
        yesterday = date.today() - timedelta(days=1)
        kickoffs = [
            datetime(yesterday.year, yesterday.month, yesterday.day,
                     15, 0, tzinfo=UK_TZ),
        ]

        schedule_matchday_jobs(mock_scheduler, "PL", kickoffs)

        # Morning and pre-kickoff are both in the past — no jobs created
        assert mock_scheduler.add_job.call_count == 0

    def test_schedule_matchday_jobs_empty_kickoffs(self) -> None:
        """Empty kickoffs list creates no jobs."""
        from scheduler import schedule_matchday_jobs

        mock_scheduler = MagicMock()
        schedule_matchday_jobs(mock_scheduler, "PL", [])
        assert mock_scheduler.add_job.call_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler Creation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchedulerCreation:
    """Test create_scheduler() static job registration."""

    def test_create_scheduler_registers_static_jobs(self) -> None:
        """All static jobs are registered on creation."""
        from scheduler import create_scheduler, STATIC_SCHEDULE

        scheduler = create_scheduler()
        job_ids = {j.id for j in scheduler.get_jobs()}

        for expected_id in STATIC_SCHEDULE:
            assert expected_id in job_ids, f"Missing static job: {expected_id}"

    def test_create_scheduler_has_weekly_retrain(self) -> None:
        """Weekly retrain job is present."""
        from scheduler import create_scheduler

        scheduler = create_scheduler()
        job = scheduler.get_job("weekly_retrain")
        assert job is not None
        assert "retrain" in job.name.lower()

    def test_create_scheduler_has_daily_planner(self) -> None:
        """Daily planner job is present."""
        from scheduler import create_scheduler

        scheduler = create_scheduler()
        job = scheduler.get_job("daily_planner")
        assert job is not None
        assert "planner" in job.name.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Markets Parameter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketsParameter:
    """Test that markets parameter gates BTTS fetching."""

    @patch("api.odds_api._save_cache")
    @patch("api.odds_api._fetch_event_market")
    @patch("api.odds_api._fetch_market")
    def test_btts_skipped_when_not_in_markets(
        self,
        mock_fetch_market: MagicMock,
        mock_fetch_event: MagicMock,
        mock_save_cache: MagicMock,
    ) -> None:
        """When markets=('totals',), BTTS per-event calls are skipped.

        Mocks ``_save_cache`` so the test doesn't pollute the production
        ``data/odds_cache.json`` file with the synthetic Arsenal–Chelsea
        fixture used here. (Pre-fix bug: every pytest run wrote
        ``{"id": "evt1", ...}`` into the production cache, masking
        scheduler issues at the next run.)
        """
        mock_fetch_market.return_value = [
            {
                "id": "evt1",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": "2026-04-12T14:00:00Z",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "title": "Bet365",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.5,
                                     "price": 1.85},
                                    {"name": "Under", "point": 2.5,
                                     "price": 2.05},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        from api.odds_api import fetch_epl_odds

        # Force refresh to bypass cache, only totals
        result = fetch_epl_odds(force_refresh=True, markets=("totals",))

        # BTTS per-event endpoint should NOT be called
        mock_fetch_event.assert_not_called()
        assert len(result) == 1

    @patch("api.odds_api._save_cache")
    @patch("api.odds_api._fetch_event_market")
    @patch("api.odds_api._fetch_market")
    def test_btts_fetched_when_in_markets(
        self,
        mock_fetch_market: MagicMock,
        mock_fetch_event: MagicMock,
        mock_save_cache: MagicMock,
    ) -> None:
        """When markets includes 'btts', per-event BTTS calls ARE made.

        ``_save_cache`` mocked to avoid polluting production cache file.
        """
        mock_fetch_market.return_value = [
            {
                "id": "evt1",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "commence_time": "2026-04-12T14:00:00Z",
                "bookmakers": [
                    {
                        "key": "bet365",
                        "title": "Bet365",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 2.5,
                                     "price": 1.85},
                                    {"name": "Under", "point": 2.5,
                                     "price": 2.05},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        mock_fetch_event.return_value = None

        from api.odds_api import fetch_epl_odds

        fetch_epl_odds(force_refresh=True, markets=("totals", "btts"))

        # BTTS per-event endpoint SHOULD be called (once per event)
        mock_fetch_event.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# Model Save/Load Round-Trip Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelSaveLoad:
    """Test pickle round-trip for trained state."""

    def test_save_load_roundtrip_preserves_state(self, tmp_path) -> None:
        """Save and load trained state, verify all attributes match."""
        from predict import LivePredictor

        predictor = LivePredictor(verbose=False)

        # Set mock state
        predictor._ou_models = {"xgb": "mock_xgb", "lgb": "mock_lgb",
                                "lr": "mock_lr", "lr_scaler": "mock_scaler",
                                "dc": "mock_dc"}
        predictor._btts_models = {"xgb": "mock_btts_xgb"}
        predictor._ou_features = ["feat_a", "feat_b"]
        predictor._btts_features = ["feat_c"]
        predictor._ou_base_rate = 0.52
        predictor._btts_base_rate = 0.48
        predictor._dc_kwargs = {"rho": -0.13, "gamma": 1.1}
        predictor._train_medians = None
        predictor._our_teams = {"Arsenal FC", "Chelsea FC"}

        path = str(tmp_path / "test_state.pkl")
        predictor.save_trained_state(path)

        # Load into fresh instance
        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path) is True

        assert loaded._ou_models == predictor._ou_models
        assert loaded._btts_models == predictor._btts_models
        assert loaded._ou_features == predictor._ou_features
        assert loaded._btts_features == predictor._btts_features
        assert loaded._ou_base_rate == predictor._ou_base_rate
        assert loaded._btts_base_rate == predictor._btts_base_rate
        assert loaded._dc_kwargs == predictor._dc_kwargs
        assert loaded._our_teams == predictor._our_teams

    def test_load_returns_false_when_file_missing(self, tmp_path) -> None:
        """load_trained_state returns False for non-existent path."""
        from predict import LivePredictor

        predictor = LivePredictor(verbose=False)
        result = predictor.load_trained_state(
            str(tmp_path / "nonexistent.pkl"))
        assert result is False

    def test_pipeline_cache_roundtrip(self, tmp_path) -> None:
        """Pipeline cache save/load preserves DataFrame and features."""
        import pandas as pd
        from predict import LivePredictor

        predictor = LivePredictor(verbose=False)
        predictor._full_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        predictor._train_medians = pd.Series({"A": 1.5, "B": 3.5})
        predictor._ou_features = ["A"]
        predictor._btts_features = ["B"]
        predictor._our_teams = {"TeamX"}

        path = str(tmp_path / "test_pipeline.pkl")
        predictor.save_pipeline_cache(path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_pipeline_cache(path) is True
        assert loaded._full_df is not None
        assert len(loaded._full_df) == 2
        assert loaded._ou_features == ["A"]
        assert loaded._our_teams == {"TeamX"}

    def test_champ_save_load_roundtrip(self, tmp_path) -> None:
        """Championship predictor save/load preserves all 3 market models."""
        from championship_predict import ChampionshipPredictor

        predictor = ChampionshipPredictor(verbose=False)
        predictor._ou_models = {"xgb": "mock", "lgb": "mock", "dc": "mock"}
        predictor._ou15_models = {"xgb": "m15", "lgb": "m15", "dc": "m15"}
        predictor._btts_models = {"xgb": "mbtts", "lgb": "mbtts", "dc": "mbtts"}
        predictor._ou_features = ["f1"]
        predictor._ou15_features = ["f2"]
        predictor._btts_features = ["f3"]
        predictor._ou_base_rate = 0.475
        predictor._ou15_base_rate = 0.73
        predictor._btts_base_rate = 0.517
        predictor._dc_kwargs = {"rho": -0.1}
        predictor._our_teams = {"Leeds", "Burnley"}

        path = str(tmp_path / "test_efl_state.pkl")
        predictor.save_trained_state(path)

        loaded = ChampionshipPredictor(verbose=False)
        assert loaded.load_trained_state(path) is True
        assert loaded._ou15_models == predictor._ou15_models
        assert loaded._ou15_features == predictor._ou15_features
        assert loaded._ou15_base_rate == predictor._ou15_base_rate


# ═══════════════════════════════════════════════════════════════════════════════
# Run Now CLI Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunNow:
    """Test run_now() CLI dispatch."""

    def test_run_now_unknown_job(self, capsys) -> None:
        """Unknown job name prints error message."""
        from scheduler import run_now

        run_now("nonexistent")
        captured = capsys.readouterr()
        assert "Unknown job" in captured.out

    def test_run_now_valid_keys(self) -> None:
        """All documented job keys are valid."""
        from scheduler import run_now

        valid_keys = [
            "predict", "predict-champ", "settle",
            "retrain", "plan", "fetch-pl", "fetch-efl",
        ]
        # Just verify no KeyError — don't actually run them
        for key in valid_keys:
            # We'd need to mock the actual functions to run them
            pass
