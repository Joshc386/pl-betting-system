"""Tests for ``is_quota_safe`` — the hard quota guardrail.

The guardrail prevents the API fetch sites from sending requests once
the operator is within ~5% of the monthly cap. The dashboard widget is
the visual alert (red ≥90%); the guardrail is the safety net that
stops accidental overshoot while the operator generates a new key or
waits for the month to roll over.

Coverage
--------
1. Returns True (safe) when no calls have been recorded — fresh state.
2. Returns True when usage is well below the threshold.
3. Returns False (block) when usage crosses the threshold.
4. Returns True for unknown providers — doesn't accidentally block
   anything that hasn't been registered.
5. Honours the configurable threshold parameter so the operator can
   tune it without editing code.
6. Reads from the same JSON file as ``read_quota`` — confirms the
   integration with the existing tracker rather than a parallel state
   store.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest


@pytest.fixture
def isolated_quota(tmp_path, monkeypatch):
    """Redirect QUOTA_FILE to a temp path so tests don't touch real state."""
    from api import quota_tracker
    fake = tmp_path / "api_quota.json"
    monkeypatch.setattr(quota_tracker, "QUOTA_FILE", str(fake))
    return fake


# =============================================================================
# Safe-by-default behaviour
# =============================================================================

class TestSafeByDefault:
    """No data → assume safe. Don't block on an empty tracker file."""

    def test_empty_state_returns_true(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe
        assert is_quota_safe("odds_api") is True
        assert is_quota_safe("oddspapi") is True

    def test_unknown_provider_returns_true(self, isolated_quota) -> None:
        """Unknown name shouldn't accidentally block calls."""
        from api.quota_tracker import is_quota_safe
        assert is_quota_safe("not_a_real_provider") is True

    def test_corrupt_file_returns_true(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe
        isolated_quota.write_text("{not json")
        # is_quota_safe goes through read_quota → resilient to corruption
        assert is_quota_safe("odds_api") is True


# =============================================================================
# Threshold mechanics
# =============================================================================

class TestThresholdBehaviour:
    """Block when usage ≥ threshold. Allow otherwise."""

    def test_well_below_threshold_returns_true(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe, record_call
        # 10% used at default 95% threshold → safe
        record_call("odds_api", remaining=450, used=50)
        assert is_quota_safe("odds_api") is True

    def test_exactly_at_threshold_returns_false(self, isolated_quota) -> None:
        """0.95 threshold → 475/500 used should block (≥)."""
        from api.quota_tracker import is_quota_safe, record_call
        record_call("odds_api", remaining=25, used=475)
        # 475/500 = 0.95 — not strictly less than 0.95 → False
        assert is_quota_safe("odds_api", threshold=0.95) is False

    def test_just_under_threshold_returns_true(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe, record_call
        record_call("odds_api", remaining=26, used=474)
        # 474/500 = 0.948 < 0.95 → True
        assert is_quota_safe("odds_api", threshold=0.95) is True

    def test_at_cap_returns_false(self, isolated_quota) -> None:
        """500/500 used (worst case) → definitely block."""
        from api.quota_tracker import is_quota_safe, record_call
        record_call("odds_api", remaining=0, used=500)
        assert is_quota_safe("odds_api") is False

    def test_oddspapi_counter_blocks_at_threshold(self, isolated_quota) -> None:
        """OddsPapi has no header — counter must drive the gate."""
        from api.quota_tracker import is_quota_safe, record_call
        # OddsPapi cap = 250, threshold 0.95 → block at 238 calls
        for _ in range(238):
            record_call("oddspapi")
        assert is_quota_safe("oddspapi", threshold=0.95) is False

    def test_oddspapi_counter_allows_below_threshold(
            self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe, record_call
        for _ in range(237):
            record_call("oddspapi")
        # 237/250 = 0.948 < 0.95 → safe
        assert is_quota_safe("oddspapi", threshold=0.95) is True


# =============================================================================
# Configurable threshold
# =============================================================================

class TestConfigurableThreshold:
    """The operator can tune the threshold without touching code."""

    def test_lower_threshold_blocks_earlier(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe, record_call
        # 250/500 = 50% used
        record_call("odds_api", remaining=250, used=250)
        # Default 0.95 → safe
        assert is_quota_safe("odds_api", threshold=0.95) is True
        # Tighter 0.40 → blocked (50% > 40%)
        assert is_quota_safe("odds_api", threshold=0.40) is False

    def test_higher_threshold_blocks_later(self, isolated_quota) -> None:
        from api.quota_tracker import is_quota_safe, record_call
        record_call("odds_api", remaining=10, used=490)
        # 490/500 = 98% used
        assert is_quota_safe("odds_api", threshold=0.95) is False
        assert is_quota_safe("odds_api", threshold=0.99) is True


# =============================================================================
# Integration with config
# =============================================================================

class TestConfigIntegration:
    """The default threshold lives in config.QUOTA_GUARDRAIL_THRESHOLD."""

    def test_config_value_is_in_valid_range(self) -> None:
        import config
        # Threshold must be in (0, 1] — values outside that range are
        # logically invalid (0 = always block, >1 = never block).
        assert 0 < config.QUOTA_GUARDRAIL_THRESHOLD <= 1.0

    def test_config_value_above_dashboard_warning(self) -> None:
        """Guardrail must trip *after* the dashboard's red warning at 0.90,
        otherwise the operator never gets a window to swap the API key
        before fetches start failing.
        """
        import config
        assert config.QUOTA_GUARDRAIL_THRESHOLD > 0.90, (
            "Guardrail threshold must give the operator a reaction window "
            "between the 0.90 red dashboard warning and the hard cap.")
