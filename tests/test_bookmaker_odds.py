"""Tests for per-bookmaker odds extraction and Championship team resolution.

Covers:
  - _extract_bookmaker_odds for O/U and BTTS markets
  - _resolve_champ_team for newly promoted teams
  - _load_cache stale-age bounding (item #4A)
  - _fetch_market transient-error retry (item #4B)
"""
import json
from datetime import datetime, timedelta

import pytest

from predict import _extract_bookmaker_odds


def _write_cache(path, *, age_minutes, data=None, omit_timestamp=False) -> None:
    """Write a cache file aged ``age_minutes`` minutes in the past.

    Mirrors the on-disk shape produced by ``api.odds_api._save_cache``.
    """
    payload = {"data": data if data is not None else [{"id": "evt1"}]}
    if not omit_timestamp:
        ts = datetime.now() - timedelta(minutes=age_minutes)
        payload["timestamp"] = ts.isoformat()
    with open(path, "w") as f:
        json.dump(payload, f)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Bookmaker Odds Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractBookmakerOdds:
    """Test _extract_bookmaker_odds function."""

    def _make_match(self) -> dict:
        """Create a realistic match dict from the odds API."""
        return {
            "bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "over": 1.85,
                    "under": 2.05,
                    "all_lines": {
                        2.5: {"over": 1.85, "under": 2.05},
                        1.5: {"over": 1.25, "under": 3.60},
                    },
                },
                "williamhill": {
                    "title": "William Hill",
                    "over": 1.90,
                    "under": 2.00,
                    "all_lines": {
                        2.5: {"over": 1.90, "under": 2.00},
                    },
                },
                "pinnacle": {
                    "title": "Pinnacle",
                    "over": 1.88,
                    "under": 2.02,
                    "all_lines": {
                        2.5: {"over": 1.88, "under": 2.02},
                        1.5: {"over": 1.28, "under": 3.50},
                    },
                },
            },
            "btts_bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "yes": 1.75,
                    "no": 2.10,
                },
                "pinnacle": {
                    "title": "Pinnacle",
                    "yes": 1.80,
                    "no": 2.05,
                },
            },
        }

    def test_ou25_over_extracts_all_books(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou25", "over")
        assert "Bet365" in result
        assert "William Hill" in result
        assert "Pinnacle" in result
        assert result["Bet365"] == 1.85

    def test_ou25_under_extracts_all_books(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou25", "under")
        assert result["Bet365"] == 2.05
        assert result["William Hill"] == 2.00

    def test_ou15_extracts_from_all_lines(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "ou15", "over")
        # Only bet365 and pinnacle have 1.5 line
        assert "Bet365" in result
        assert "Pinnacle" in result
        assert "William Hill" not in result
        assert result["Bet365"] == 1.25

    def test_btts_yes_extracts(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "btts", "yes")
        assert result["Bet365"] == 1.75
        assert result["Pinnacle"] == 1.80

    def test_btts_no_extracts(self) -> None:
        match = self._make_match()
        result = _extract_bookmaker_odds(match, "btts", "no")
        assert result["Bet365"] == 2.10
        assert result["Pinnacle"] == 2.05

    def test_empty_match_returns_empty(self) -> None:
        result = _extract_bookmaker_odds({}, "ou25", "over")
        assert result == {}

    def test_string_line_keys_handled(self) -> None:
        """all_lines with string keys (e.g. '2.5') should be found."""
        match = {
            "bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "over": 1.85,
                    "under": 2.05,
                    "all_lines": {
                        "2.5": {"over": 1.85, "under": 2.05},
                    },
                },
            },
            "btts_bookmakers": {},
        }
        result = _extract_bookmaker_odds(match, "ou25", "over")
        assert result["Bet365"] == 1.85


# ═══════════════════════════════════════════════════════════════════════════════
# Best-odds consensus (zero-price robustness)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetBestOdds:
    """get_best_odds must survive bookmakers that report 0 for the 2.5 line."""

    def test_zero_priced_book_does_not_crash_consensus(self) -> None:
        """An alt-only bookmaker reporting over=under=0 (no 2.5 price) must
        be excluded from the consensus, not trigger a ZeroDivisionError."""
        from api.odds_api import get_best_odds
        match = {
            "bookmakers": {
                "pinnacle": {"over": 1.95, "under": 1.95, "title": "Pinnacle"},
                "altonly": {"over": 0, "under": 0, "title": "AltOnly"},
            }
        }
        result = get_best_odds(match)  # must not raise
        assert result is not None
        assert result["best_over"] == 1.95
        assert result["best_under"] == 1.95
        # Consensus is normalised and computed only from the valid book.
        assert result["consensus_over"] + result["consensus_under"] == pytest.approx(1.0)

    def test_all_books_zero_priced_returns_none(self) -> None:
        """If no bookmaker priced the 2.5 line, there is no usable market."""
        from api.odds_api import get_best_odds
        match = {
            "bookmakers": {
                "altonly": {"over": 0, "under": 0, "title": "AltOnly"},
            }
        }
        assert get_best_odds(match) is None


class TestGetBestBttsOdds:
    """get_best_btts_odds must survive bookmakers that report 0 for BTTS."""

    def test_zero_priced_book_does_not_crash_consensus(self) -> None:
        from api.odds_api import get_best_btts_odds
        match = {
            "btts_bookmakers": {
                "pinnacle": {"yes": 1.90, "no": 1.90, "title": "Pinnacle"},
                "altonly": {"yes": 0, "no": 0, "title": "AltOnly"},
            }
        }
        result = get_best_btts_odds(match)  # must not raise
        assert result is not None
        assert result["best_yes"] == 1.90
        assert result["best_no"] == 1.90
        assert result["consensus_yes"] + result["consensus_no"] == pytest.approx(1.0)

    def test_all_books_zero_priced_returns_none(self) -> None:
        from api.odds_api import get_best_btts_odds
        match = {
            "btts_bookmakers": {
                "altonly": {"yes": 0, "no": 0, "title": "AltOnly"},
            }
        }
        assert get_best_btts_odds(match) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Stale-cache age bounding (item #4A)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadCacheStaleBound:
    """_load_cache must never serve odds older than the bounded stale TTL."""

    def test_beyond_stale_ttl_rejected_even_when_allow_stale(
        self, tmp_path, monkeypatch
    ) -> None:
        """A cache older than 90 min must not be served, even on the
        API-failure fallback path. Betting on ancient odds is the failure
        mode this guard exists to prevent."""
        from api import odds_api
        cache_file = tmp_path / "odds_cache.json"
        _write_cache(cache_file, age_minutes=120)
        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))

        assert odds_api._load_cache(allow_stale=True) is None

    def test_fresh_cache_served_on_normal_call(
        self, tmp_path, monkeypatch
    ) -> None:
        """Cache younger than 30 min is served on a normal (non-stale) call."""
        from api import odds_api
        cache_file = tmp_path / "odds_cache.json"
        _write_cache(cache_file, age_minutes=10, data=[{"id": "fresh"}])
        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))

        assert odds_api._load_cache() == [{"id": "fresh"}]

    def test_mid_age_rejected_normal_but_served_when_stale(
        self, tmp_path, monkeypatch
    ) -> None:
        """A 60-min cache is too old for a normal call but is the exact
        case the fallback path exists to serve."""
        from api import odds_api
        cache_file = tmp_path / "odds_cache.json"
        _write_cache(cache_file, age_minutes=60, data=[{"id": "mid"}])
        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))

        assert odds_api._load_cache() is None
        assert odds_api._load_cache(allow_stale=True) == [{"id": "mid"}]

    def test_missing_timestamp_rejected(
        self, tmp_path, monkeypatch
    ) -> None:
        """A cache with no timestamp has unknown age — never served."""
        from api import odds_api
        cache_file = tmp_path / "odds_cache.json"
        _write_cache(cache_file, age_minutes=0, omit_timestamp=True)
        monkeypatch.setattr(odds_api, "CACHE_FILE", str(cache_file))

        assert odds_api._load_cache() is None
        assert odds_api._load_cache(allow_stale=True) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Bulk totals fetch retry (item #4B)
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self.headers = {"x-requests-remaining": "100", "x-requests-used": "5"}
        self.text = "error body"
        self._json = json_data if json_data is not None else []

    def json(self):
        return self._json


class TestFetchMarketRetry:
    """_fetch_market retries transient 5xx/timeouts but never 429/4xx."""

    @pytest.fixture
    def patched(self, monkeypatch):
        """Quota guardrail open, record_call/sleep neutralised, sleep spied."""
        import requests
        from api import odds_api, quota_tracker

        monkeypatch.setattr(quota_tracker, "is_quota_safe", lambda *a, **k: True)
        monkeypatch.setattr(quota_tracker, "record_call", lambda *a, **k: None)

        sleeps: list[float] = []
        monkeypatch.setattr(odds_api.time, "sleep", lambda s: sleeps.append(s))

        def install(*responses):
            """Queue successive requests.get results (responses or exceptions)."""
            queue = list(responses)

            def fake_get(*args, **kwargs):
                item = queue.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

            monkeypatch.setattr(odds_api.requests, "get", fake_get)
            return sleeps

        install.requests = requests  # expose for Timeout construction
        return install

    def test_500_then_200_retries_and_returns_data(self, patched) -> None:
        sleeps = patched(
            _FakeResp(500),
            _FakeResp(200, json_data=[{"id": "evt1"}]),
        )
        from api.odds_api import _fetch_market

        assert _fetch_market("totals") == [{"id": "evt1"}]
        assert sleeps == [1]  # one backoff of 2**0 == 1s

    def test_timeout_then_200_retries_and_returns_data(self, patched) -> None:
        requests = patched.requests
        sleeps = patched(
            requests.Timeout("slow"),
            _FakeResp(200, json_data=[{"id": "evt2"}]),
        )
        from api.odds_api import _fetch_market

        assert _fetch_market("totals") == [{"id": "evt2"}]
        assert sleeps == [1]

    def test_persistent_500_exhausts_retries_returns_empty(self, patched) -> None:
        sleeps = patched(_FakeResp(500), _FakeResp(500), _FakeResp(500))
        from api.odds_api import _fetch_market

        assert _fetch_market("totals") == []
        assert sleeps == [1, 2]  # 2**0 then 2**1, then no more retries

    def test_429_returns_empty_without_retry(self, patched) -> None:
        """A quota error is deterministic — retrying just burns a credit."""
        sleeps = patched(_FakeResp(429))
        from api.odds_api import _fetch_market

        assert _fetch_market("totals") == []
        assert sleeps == []

    def test_4xx_returns_empty_without_retry(self, patched) -> None:
        """4xx (e.g. 422 unsupported market) is not transient — no retry."""
        sleeps = patched(_FakeResp(422))
        from api.odds_api import _fetch_market

        assert _fetch_market("totals") == []
        assert sleeps == []


# ═══════════════════════════════════════════════════════════════════════════════
# Championship Team Resolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestChampTeamResolution:
    """Test _resolve_champ_team for newly promoted teams."""

    def test_known_team_in_dataset(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn", "Stoke", "Derby"}
        assert _resolve_champ_team("Blackburn Rovers", our_teams) == "Blackburn"

    def test_newly_promoted_team_not_in_dataset(self) -> None:
        """Newly promoted teams should still resolve even if not in our_teams."""
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn", "Stoke", "Derby"}
        # These teams have explicit mappings but aren't in our_teams
        assert _resolve_champ_team("Southampton", our_teams) == "Southampton"
        assert _resolve_champ_team("Leicester City", our_teams) == "Leicester"
        assert _resolve_champ_team("Ipswich Town", our_teams) == "Ipswich"
        assert _resolve_champ_team("Birmingham City", our_teams) == "Birmingham"
        assert _resolve_champ_team("Charlton Athletic", our_teams) == "Charlton"
        assert _resolve_champ_team("Wrexham AFC", our_teams) == "Wrexham"

    def test_direct_match(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Middlesbrough", "Portsmouth"}
        assert _resolve_champ_team("Middlesbrough", our_teams) == "Middlesbrough"

    def test_unknown_team_returns_none(self) -> None:
        from championship_predict import _resolve_champ_team
        our_teams = {"Blackburn"}
        assert _resolve_champ_team("Tottenham Hotspur", our_teams) is None
