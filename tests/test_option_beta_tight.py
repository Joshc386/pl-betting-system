"""Tests for Option β-tight: OddsPapi gated to week-ahead snapshot only.

Background
----------
The April 2026 API audit found that every dashboard load + every predict
refresh was firing fresh OddsPapi fetches, burning ~430 credits/month
against a 250-credit cap. Option β-tight restricts OddsPapi to the
week-ahead snapshot (Monday morning sweep) only. Matchday morning, KO-1h,
and CLV refreshes go Odds-API only.

Coverage
--------
1. Constructor wiring — both predictors store ``snapshot_type`` correctly
   and default to ``config.DEFAULT_SNAPSHOT_TYPE`` when None is passed.
2. _save_cache empty-data guard — regression test for the guard that
   prevents an empty API response from clobbering a populated cache
   (this was the bug that masked Odds-API usage during the audit).
3. Dashboard config flag — ``DASHBOARD_FETCH_ODDSPAPI`` defaults to False
   so dashboard scans never trigger fresh OddsPapi fetches on page load.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest


# =============================================================================
# Constructor wiring
# =============================================================================

class TestSnapshotTypeWiring:
    """Both predictor classes accept and store ``snapshot_type`` correctly."""

    def test_pl_predictor_explicit_week_ahead(self) -> None:
        from predict import LivePredictor
        p = LivePredictor(verbose=False, snapshot_type="week_ahead")
        assert p.snapshot_type == "week_ahead"

    def test_pl_predictor_explicit_refresh(self) -> None:
        from predict import LivePredictor
        p = LivePredictor(verbose=False, snapshot_type="refresh")
        assert p.snapshot_type == "refresh"

    def test_pl_predictor_default_falls_back_to_config(self) -> None:
        """None → config.DEFAULT_SNAPSHOT_TYPE (= "refresh")."""
        import config
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        assert p.snapshot_type == config.DEFAULT_SNAPSHOT_TYPE

    def test_efl_predictor_explicit_week_ahead(self) -> None:
        from championship_predict import ChampionshipPredictor
        p = ChampionshipPredictor(verbose=False, snapshot_type="week_ahead")
        assert p.snapshot_type == "week_ahead"

    def test_efl_predictor_default_falls_back_to_config(self) -> None:
        import config
        from championship_predict import ChampionshipPredictor
        p = ChampionshipPredictor(verbose=False)
        assert p.snapshot_type == config.DEFAULT_SNAPSHOT_TYPE

    def test_default_snapshot_type_is_refresh(self) -> None:
        """The cheap path is the default — operator must opt in to OddsPapi."""
        import config
        assert config.DEFAULT_SNAPSHOT_TYPE == "refresh"


# =============================================================================
# Dashboard read-only by default
# =============================================================================

class TestDashboardReadOnly:
    """Dashboard does not fire OddsPapi on page load by default."""

    def test_dashboard_fetch_oddspapi_default_false(self) -> None:
        """Default is False — dashboard serves cached data only."""
        import config
        assert config.DASHBOARD_FETCH_ODDSPAPI is False


# =============================================================================
# _save_cache empty-data guard (regression)
# =============================================================================

class TestSaveCacheEmptyGuard:
    """Empty-data writes must not clobber a populated cache.

    This guards against the bug surfaced in the April 2026 API audit:
    test pollution was silently writing fake fixtures into the production
    cache, but a similar bug shape (empty response → overwrite) would have
    erased real cache data and triggered downstream cache misses → over-
    fetching against the live quota.
    """

    def test_empty_list_does_not_write(self, tmp_path, monkeypatch) -> None:
        """_save_cache([]) is a no-op — it does not create or modify a file."""
        from api import odds_api
        cache_file = tmp_path / "cache.json"
        # Pre-populate the cache with known-good data
        good_data = [{"id": "evt1", "home_team": "Arsenal"}]
        cache_file.write_text(json.dumps(
            {"timestamp": "2026-04-25T10:00:00", "data": good_data}))

        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))
        monkeypatch.setattr(odds_api, "CACHE_DIR", str(tmp_path))

        # Act: try to save empty data
        odds_api._save_cache([])

        # Assert: cache file is untouched
        with open(cache_file) as f:
            after = json.load(f)
        assert after["data"] == good_data, (
            "Empty _save_cache call clobbered the populated cache — "
            "this is the bug Option β-tight was designed to prevent.")

    def test_empty_list_does_not_create_file(self, tmp_path, monkeypatch) -> None:
        """If no cache exists, _save_cache([]) does not create one."""
        from api import odds_api
        cache_file = tmp_path / "cache.json"
        assert not cache_file.exists()

        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))
        monkeypatch.setattr(odds_api, "CACHE_DIR", str(tmp_path))

        odds_api._save_cache([])

        assert not cache_file.exists(), (
            "Empty _save_cache should not create an empty cache file.")

    def test_populated_data_writes_normally(self, tmp_path, monkeypatch) -> None:
        """Sanity: the guard only blocks empty data, not real data."""
        from api import odds_api
        cache_file = tmp_path / "cache.json"
        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))
        monkeypatch.setattr(odds_api, "CACHE_DIR", str(tmp_path))

        real_data = [{"id": "evt1"}]
        odds_api._save_cache(real_data)

        with open(cache_file) as f:
            saved = json.load(f)
        assert saved["data"] == real_data
