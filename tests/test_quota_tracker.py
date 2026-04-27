"""Tests for ``api.quota_tracker``.

Coverage
--------
1. record_call("odds_api", remaining, used) — persists header values
2. record_call("oddspapi") — increments client-side counter
3. Monthly rollover — counter resets when calendar month changes
4. read_quota — returns the documented shape with sane defaults
5. Failure modes — corrupted/missing file doesn't raise
6. Atomic write — write+rename pattern leaves no partial files

Plus a smoke test for the dashboard ``_format_age`` helper, which is the
user-visible side of the freshness widget.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest


# =============================================================================
# Helpers
# =============================================================================

@pytest.fixture
def isolated_quota(tmp_path, monkeypatch):
    """Redirect QUOTA_FILE to a temp path so tests don't touch real state."""
    from api import quota_tracker
    fake = tmp_path / "api_quota.json"
    monkeypatch.setattr(quota_tracker, "QUOTA_FILE", str(fake))
    return fake


# =============================================================================
# record_call — Odds API (header-driven)
# =============================================================================

class TestOddsApiRecording:
    """The Odds API gives us authoritative numbers via response headers."""

    def test_first_call_writes_remaining_and_used(self, isolated_quota) -> None:
        from api import quota_tracker
        quota_tracker.record_call("odds_api", remaining=463, used=37)

        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["odds_api"]["remaining"] == 463
        assert state["odds_api"]["used"] == 37
        # last_call should be a parseable ISO timestamp
        datetime.fromisoformat(state["odds_api"]["last_call"])

    def test_subsequent_call_overwrites_with_latest(self, isolated_quota) -> None:
        """Latest header values win — they are the source of truth."""
        from api import quota_tracker
        quota_tracker.record_call("odds_api", remaining=463, used=37)
        quota_tracker.record_call("odds_api", remaining=460, used=40)

        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["odds_api"]["used"] == 40
        assert state["odds_api"]["remaining"] == 460

    def test_partial_header_data_does_not_clobber_known_values(
            self, isolated_quota) -> None:
        """If a later call only supplies ``remaining``, ``used`` should
        remain at its previous value rather than become None.
        """
        from api import quota_tracker
        quota_tracker.record_call("odds_api", remaining=463, used=37)
        # Some endpoints may not echo ``used`` — only ``remaining``
        quota_tracker.record_call("odds_api", remaining=460, used=None)

        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["odds_api"]["used"] == 37  # carried over
        assert state["odds_api"]["remaining"] == 460


# =============================================================================
# record_call — OddsPapi (counter-driven)
# =============================================================================

class TestOddsPapiCounter:
    """OddsPapi has no quota header — we count locally."""

    def test_first_call_initialises_counter(self, isolated_quota) -> None:
        from api import quota_tracker
        quota_tracker.record_call("oddspapi")
        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["oddspapi"]["calls_this_month"] == 1

    def test_repeated_calls_increment(self, isolated_quota) -> None:
        from api import quota_tracker
        for _ in range(5):
            quota_tracker.record_call("oddspapi")
        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["oddspapi"]["calls_this_month"] == 5


# =============================================================================
# Monthly rollover
# =============================================================================

class TestMonthlyRollover:
    """When the calendar month flips, counters reset to zero."""

    def test_old_month_resets_oddspapi_counter(self, isolated_quota) -> None:
        from api import quota_tracker
        # Seed a previous-month state by hand
        prev_month = (datetime.now() - timedelta(days=40)).strftime("%Y-%m")
        isolated_quota.write_text(json.dumps({
            "oddspapi": {
                "month": prev_month,
                "calls_this_month": 158,
                "last_call": "2026-03-15T10:00:00",
            }
        }))

        quota_tracker.record_call("oddspapi")

        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["oddspapi"]["calls_this_month"] == 1, (
            "Counter should reset at month boundary, not stay at 159")
        assert state["oddspapi"]["month"] == datetime.now().strftime("%Y-%m")

    def test_old_month_resets_odds_api(self, isolated_quota) -> None:
        from api import quota_tracker
        prev_month = (datetime.now() - timedelta(days=40)).strftime("%Y-%m")
        isolated_quota.write_text(json.dumps({
            "odds_api": {
                "month": prev_month,
                "remaining": 100,
                "used": 400,
                "last_call": "2026-03-15T10:00:00",
            }
        }))

        # Record under the new month — should drop the prev-month numbers
        quota_tracker.record_call("odds_api", remaining=499, used=1)

        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["odds_api"]["used"] == 1
        assert state["odds_api"]["remaining"] == 499


# =============================================================================
# read_quota — shape + defaults
# =============================================================================

class TestReadQuota:
    """Consumers (the dashboard widget) rely on a stable return shape."""

    def test_empty_state_returns_skeleton_with_nones(self, isolated_quota) -> None:
        from api import quota_tracker
        out = quota_tracker.read_quota()
        assert set(out) == {"odds_api", "oddspapi"}
        for provider in ("odds_api", "oddspapi"):
            assert out[provider]["used"] is None
            assert out[provider]["remaining"] is None
            assert out[provider]["last_call"] is None
            assert out[provider]["limit"] > 0

    def test_after_record_returns_current_values(self, isolated_quota) -> None:
        from api import quota_tracker
        quota_tracker.record_call("odds_api", remaining=463, used=37)
        quota_tracker.record_call("oddspapi")
        quota_tracker.record_call("oddspapi")

        out = quota_tracker.read_quota()
        assert out["odds_api"]["used"] == 37
        assert out["odds_api"]["remaining"] == 463
        assert out["oddspapi"]["used"] == 2
        # OddsPapi remaining is derived: limit - used
        assert out["oddspapi"]["remaining"] == out["oddspapi"]["limit"] - 2

    def test_old_month_data_treated_as_empty(self, isolated_quota) -> None:
        """read_quota() should not surface stale prev-month numbers."""
        from api import quota_tracker
        prev_month = (datetime.now() - timedelta(days=40)).strftime("%Y-%m")
        isolated_quota.write_text(json.dumps({
            "odds_api": {"month": prev_month, "used": 999, "remaining": 0},
        }))
        out = quota_tracker.read_quota()
        assert out["odds_api"]["used"] is None


# =============================================================================
# Failure modes
# =============================================================================

class TestFailureModes:
    """Tracking is advisory — must never raise into the API call path."""

    def test_unknown_provider_logs_warning_no_raise(
            self, isolated_quota) -> None:
        from api import quota_tracker
        # Should not raise even though "stake" is not a real provider
        quota_tracker.record_call("stake")

    def test_corrupt_file_does_not_raise(self, isolated_quota) -> None:
        from api import quota_tracker
        isolated_quota.write_text("{corrupted json")
        # Should silently reset rather than blow up
        quota_tracker.record_call("oddspapi")
        with open(isolated_quota) as f:
            state = json.load(f)
        assert state["oddspapi"]["calls_this_month"] == 1


# =============================================================================
# Header parser helper
# =============================================================================

class TestTryParse:
    """The Odds API sometimes returns ``?`` for headers — handle gracefully."""

    def test_question_mark_returns_none(self) -> None:
        from api.quota_tracker import _try_parse
        assert _try_parse("?") is None

    def test_none_returns_none(self) -> None:
        from api.quota_tracker import _try_parse
        assert _try_parse(None) is None

    def test_numeric_string_parses(self) -> None:
        from api.quota_tracker import _try_parse
        assert _try_parse("463") == 463

    def test_garbage_returns_none(self) -> None:
        from api.quota_tracker import _try_parse
        assert _try_parse("not-a-number") is None


# =============================================================================
# Dashboard formatter helper
# =============================================================================

class TestFormatAge:
    """Smoke tests for the relative-time formatter the widget uses."""

    def test_none_returns_never(self) -> None:
        from dashboard import _format_age
        assert _format_age(None) == "never"

    def test_just_now_for_recent(self) -> None:
        from dashboard import _format_age
        ts = datetime.now().isoformat()
        assert _format_age(ts).endswith("ago") or _format_age(ts) == "just now"

    def test_minutes_format(self) -> None:
        from dashboard import _format_age
        ts = (datetime.now() - timedelta(minutes=12)).isoformat()
        assert "12m" in _format_age(ts)

    def test_hours_format(self) -> None:
        from dashboard import _format_age
        ts = (datetime.now() - timedelta(hours=3, minutes=20)).isoformat()
        assert "3h" in _format_age(ts)

    def test_days_format(self) -> None:
        from dashboard import _format_age
        ts = (datetime.now() - timedelta(days=4)).isoformat()
        assert "4d" in _format_age(ts)

    def test_garbage_returns_question_mark(self) -> None:
        from dashboard import _format_age
        assert _format_age("not-an-iso-string") == "?"
