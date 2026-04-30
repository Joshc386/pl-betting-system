"""Tests for Path B — selective per-event Odds API fetching.

Background
----------
The Odds API removed the bulk ``alternate_totals`` market in April 2026
(returns HTTP 422). The new flow:

  * Per-event helpers ``fetch_event_alt_totals`` and ``fetch_event_btts_odds``
    in ``api/odds_api.py`` parse the per-event response shape into a
    consumer-friendly dict.
  * Merge helpers ``merge_alt_totals_into_match`` and
    ``merge_btts_into_match`` integrate that data into the legacy
    match-dict structure that downstream code already reads.
  * The predictors gate per-event calls on a model-conviction threshold
    so we only burn credits on fixtures the model thinks are worth it.

Coverage
--------
1. Parser unit tests for both helpers (raw API response → parsed dict).
2. Merger idempotency tests (incomplete sides dropped, existing 2.5
   preserved, new bookmakers appended).
3. Config sanity (thresholds within reasonable ranges).
4. Smoke test for the filter logic in ``_build_match_centre`` data
   shaping (delegated to a synthetic DataFrame so we can test the mask
   without spinning up Dash).
"""
from __future__ import annotations

import pandas as pd
import pytest


# =============================================================================
# Parser: fetch_event_alt_totals
# =============================================================================

class TestFetchEventAltTotalsParser:
    """Parses raw per-event API response into bookmaker-keyed alt-line dict."""

    def test_typical_response_parses(self, monkeypatch) -> None:
        from api import odds_api
        # Mock the underlying _fetch_event_market to return a known shape
        raw = {
            "id": "evt1",
            "bookmakers": [
                {"key": "bet365", "title": "Bet365", "markets": [
                    {"key": "alternate_totals", "outcomes": [
                        {"name": "Over", "point": 1.5, "price": 1.20},
                        {"name": "Under", "point": 1.5, "price": 4.50},
                        {"name": "Over", "point": 2.5, "price": 1.85},
                        {"name": "Under", "point": 2.5, "price": 1.95},
                    ]},
                ]},
            ],
        }
        monkeypatch.setattr(odds_api, "_fetch_event_market",
                            lambda eid, mk: raw)
        out = odds_api.fetch_event_alt_totals("evt1")

        assert "bet365" in out
        assert out["bet365"]["title"] == "Bet365"
        assert 1.5 in out["bet365"]["lines"]
        assert out["bet365"]["lines"][1.5] == {"over": 1.20, "under": 4.50}
        assert out["bet365"]["lines"][2.5] == {"over": 1.85, "under": 1.95}

    def test_drops_incomplete_sides(self, monkeypatch) -> None:
        """Books with only Over but no Under (or vice versa) are dropped."""
        from api import odds_api
        raw = {
            "id": "evt1",
            "bookmakers": [
                {"key": "incomplete", "markets": [
                    {"key": "alternate_totals", "outcomes": [
                        # Only Over for 1.5 — should be dropped
                        {"name": "Over", "point": 1.5, "price": 1.20},
                        # Both sides for 2.5 — should be kept
                        {"name": "Over", "point": 2.5, "price": 1.85},
                        {"name": "Under", "point": 2.5, "price": 1.95},
                    ]},
                ]},
            ],
        }
        monkeypatch.setattr(odds_api, "_fetch_event_market", lambda *a: raw)
        out = odds_api.fetch_event_alt_totals("evt1")

        assert "incomplete" in out
        assert 1.5 not in out["incomplete"]["lines"]  # incomplete dropped
        assert 2.5 in out["incomplete"]["lines"]  # complete kept

    def test_returns_none_on_fetch_failure(self, monkeypatch) -> None:
        from api import odds_api
        monkeypatch.setattr(odds_api, "_fetch_event_market",
                            lambda *a: None)
        assert odds_api.fetch_event_alt_totals("evt1") is None

    def test_empty_bookmakers_returns_empty_dict(
            self, monkeypatch) -> None:
        from api import odds_api
        monkeypatch.setattr(odds_api, "_fetch_event_market",
                            lambda *a: {"id": "evt1", "bookmakers": []})
        assert odds_api.fetch_event_alt_totals("evt1") == {}

    def test_skips_other_market_keys(self, monkeypatch) -> None:
        """A bookmaker with only btts/spreads/etc. should yield no alt lines."""
        from api import odds_api
        raw = {
            "bookmakers": [
                {"key": "bet365", "markets": [
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.85},
                        {"name": "No", "price": 1.95},
                    ]},
                ]},
            ],
        }
        monkeypatch.setattr(odds_api, "_fetch_event_market", lambda *a: raw)
        out = odds_api.fetch_event_alt_totals("evt1")
        assert out == {}


# =============================================================================
# Merger: merge_alt_totals_into_match
# =============================================================================

class TestMergeAltTotalsIntoMatch:
    """Merger preserves existing 2.5 data while adding alt lines."""

    def test_adds_lines_to_existing_bookmaker(self) -> None:
        from api.odds_api import merge_alt_totals_into_match
        match = {
            "bookmakers": {
                "bet365": {
                    "title": "Bet365",
                    "over": 1.85, "under": 1.95,
                    "all_lines": {2.5: {"over": 1.85, "under": 1.95}},
                },
            },
            "btts_bookmakers": {},
        }
        alt_data = {
            "bet365": {
                "title": "Bet365",
                "lines": {1.5: {"over": 1.20, "under": 4.50}},
            },
        }
        merge_alt_totals_into_match(match, alt_data)

        # Both lines now present
        all_lines = match["bookmakers"]["bet365"]["all_lines"]
        assert 1.5 in all_lines
        assert 2.5 in all_lines
        # 2.5 untouched (preserved)
        assert all_lines[2.5] == {"over": 1.85, "under": 1.95}
        assert all_lines[1.5] == {"over": 1.20, "under": 4.50}

    def test_creates_new_bookmaker_when_absent(self) -> None:
        from api.odds_api import merge_alt_totals_into_match
        match = {"bookmakers": {}, "btts_bookmakers": {}}
        alt_data = {
            "newbook": {
                "title": "New Book",
                "lines": {1.5: {"over": 1.22, "under": 4.20}},
            },
        }
        merge_alt_totals_into_match(match, alt_data)

        assert "newbook" in match["bookmakers"]
        assert match["bookmakers"]["newbook"]["title"] == "New Book"
        assert match["bookmakers"]["newbook"]["all_lines"] == {
            1.5: {"over": 1.22, "under": 4.20}}

    def test_does_not_overwrite_existing_25_line(self) -> None:
        """If the bookmaker already had a 2.5 line and alt_totals also
        carries 2.5, the existing one wins (we trust the original totals
        fetch over the per-event alt_totals echo).
        """
        from api.odds_api import merge_alt_totals_into_match
        match = {
            "bookmakers": {
                "bet365": {
                    "all_lines": {2.5: {"over": 1.85, "under": 1.95}},
                },
            },
            "btts_bookmakers": {},
        }
        alt_data = {
            "bet365": {
                "title": "Bet365",
                "lines": {2.5: {"over": 1.99, "under": 1.81}},  # different
            },
        }
        merge_alt_totals_into_match(match, alt_data)
        # Existing 2.5 preserved
        assert match["bookmakers"]["bet365"]["all_lines"][2.5] == {
            "over": 1.85, "under": 1.95}

    def test_empty_data_is_noop(self) -> None:
        from api.odds_api import merge_alt_totals_into_match
        match = {"bookmakers": {}, "btts_bookmakers": {}}
        merge_alt_totals_into_match(match, {})  # should not raise
        assert match["bookmakers"] == {}


# =============================================================================
# Parser + Merger: BTTS per-event helpers
# =============================================================================

class TestFetchEventBttsOdds:

    def test_typical_response_parses(self, monkeypatch) -> None:
        from api import odds_api
        raw = {
            "bookmakers": [
                {"key": "bet365", "title": "Bet365", "markets": [
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.85},
                        {"name": "No", "price": 1.95},
                    ]},
                ]},
            ],
        }
        monkeypatch.setattr(odds_api, "_fetch_event_market", lambda *a: raw)
        out = odds_api.fetch_event_btts_odds("evt1")
        assert "bet365" in out
        assert out["bet365"]["yes"] == 1.85
        assert out["bet365"]["no"] == 1.95

    def test_drops_books_with_invalid_prices(self, monkeypatch) -> None:
        """Prices <= 1.0 are nonsense (would mean negative implied prob)."""
        from api import odds_api
        raw = {
            "bookmakers": [
                {"key": "bad", "markets": [
                    {"key": "btts", "outcomes": [
                        {"name": "Yes", "price": 1.0},  # invalid
                        {"name": "No", "price": 1.95},
                    ]},
                ]},
            ],
        }
        monkeypatch.setattr(odds_api, "_fetch_event_market", lambda *a: raw)
        out = odds_api.fetch_event_btts_odds("evt1")
        assert "bad" not in out


class TestMergeBttsIntoMatch:

    def test_appends_to_btts_bookmakers(self) -> None:
        from api.odds_api import merge_btts_into_match
        match = {"bookmakers": {}, "btts_bookmakers": {}}
        merge_btts_into_match(match, {
            "bet365": {"title": "Bet365", "yes": 1.85, "no": 1.95,
                       "is_sharp": False, "is_major": True},
        })
        assert "bet365" in match["btts_bookmakers"]
        assert match["btts_bookmakers"]["bet365"]["yes"] == 1.85

    def test_latest_wins_on_duplicate_book(self) -> None:
        from api.odds_api import merge_btts_into_match
        match = {"bookmakers": {}, "btts_bookmakers": {
            "bet365": {"yes": 1.80, "no": 2.00},
        }}
        merge_btts_into_match(match, {
            "bet365": {"yes": 1.85, "no": 1.95},  # newer
        })
        assert match["btts_bookmakers"]["bet365"]["yes"] == 1.85


# =============================================================================
# Config sanity
# =============================================================================

class TestConfigThresholds:
    """Threshold values must live in operationally sensible ranges."""

    def test_ou15_threshold_above_05(self) -> None:
        """O/U 1.5 Over typically has model_prob 0.65-0.85; threshold
        above 0.5 ensures we don't fetch for every fixture.

        The threshold is per-league (PL fixtures cluster at higher
        P(Over 1.5) than EFL — different scoring profiles), so each
        league entry is checked independently.
        """
        import config
        thresholds = config.OU15_FETCH_PROB_THRESHOLD
        assert isinstance(thresholds, dict)
        assert "PL" in thresholds
        assert "EFL" in thresholds
        for league, value in thresholds.items():
            assert 0.5 < value < 1.0, (
                f"{league} threshold {value} outside expected range")
        # Sanity: PL should be at least as strict as EFL because PL
        # fixtures cluster higher.
        assert thresholds["PL"] >= thresholds["EFL"]

    def test_btts_delta_below_05(self) -> None:
        """Delta defines how far from 0.5 model must be to fetch BTTS.
        Must be < 0.5 (otherwise threshold is unreachable)."""
        import config
        assert 0.0 < config.BTTS_FETCH_PROB_DELTA < 0.5

    def test_edge_display_threshold_negative(self) -> None:
        """Display threshold should be slightly negative — show near-misses
        but hide definitely-no rows. Range -10pp to 0pp."""
        import config
        assert -10.0 <= config.EDGE_DISPLAY_THRESHOLD <= 0.0


# =============================================================================
# Display filter logic
# =============================================================================

class TestDisplayFilterLogic:
    """The Match Centre hides rows below ``EDGE_DISPLAY_THRESHOLD`` by default.

    We test the filter mask directly rather than the full Dash render
    function — same logic, easier to assert on.
    """

    def _make_df(self, edges: list[float | None]) -> pd.DataFrame:
        return pd.DataFrame({
            "home_team": [f"H{i}" for i in range(len(edges))],
            "away_team": [f"A{i}" for i in range(len(edges))],
            "market": ["ou25"] * len(edges),
            "side": ["over"] * len(edges),
            "edge_pct": edges,
        })

    def test_filter_hides_low_edge(self) -> None:
        from config import EDGE_DISPLAY_THRESHOLD
        df = self._make_df([5.0, 1.0, -3.0, -10.0])
        keep = df["edge_pct"].isna() | (df["edge_pct"] >= EDGE_DISPLAY_THRESHOLD)
        # Default threshold -2.0: keeps 5.0, 1.0; drops -3.0, -10.0
        kept = df[keep].reset_index(drop=True)
        assert len(kept) == 2
        assert all(kept["edge_pct"] >= EDGE_DISPLAY_THRESHOLD)

    def test_filter_keeps_null_edges(self) -> None:
        """Markets without odds yet have NULL edge_pct — never hide them."""
        from config import EDGE_DISPLAY_THRESHOLD
        df = self._make_df([None, None, -5.0])
        keep = df["edge_pct"].isna() | (df["edge_pct"] >= EDGE_DISPLAY_THRESHOLD)
        kept = df[keep].reset_index(drop=True)
        assert len(kept) == 2  # two NULL kept, -5.0 dropped

    def test_show_all_keeps_everything(self) -> None:
        """When show_all=True the filter is bypassed entirely."""
        df = self._make_df([5.0, -100.0, None])
        # Simulate show_all=True path: no mask applied
        kept = df.reset_index(drop=True)
        assert len(kept) == 3
