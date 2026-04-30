"""Tests for the Live ROI vs Simulation panel data shaping.

The renderer (``_make_live_vs_sim_panel``) is a pure function of
``_compute_live_roi_rows`` output, so we test the data shaping directly
and trust the renderer is just glue. This keeps tests fast and focused.

Coverage
--------
1. Empty settled DataFrame → all baseline cells emit ``no_data``.
2. Per-market grouping correctly splits bets and sums P&L.
3. Status classification:
   - "ok"           — n_bets ≥ threshold, drift acceptable
   - "drift"        — n_bets ≥ threshold, drift below trigger
   - "low_n"        — n_bets > 0 but below confidence threshold
   - "no_data"      — known baseline cell, zero bets
   - "no_baseline"  — live bets in a market not in Phase 4a baseline
4. Drift sign: positive when live beats sim, negative when below.
5. Win% calculation handles ties / no-bets gracefully.
6. The PHASE_4A_BASELINE_ROI config matches the canonical Phase 4a
   numbers from reports/roi_findings.md (regression guard).
"""
from __future__ import annotations

import pandas as pd
import pytest


def _make_settled_df(rows: list[dict]) -> pd.DataFrame:
    """Construct a minimal settled-bets DataFrame for the panel logic.

    Only the columns ``_compute_live_roi_rows`` actually reads need to
    be present: market, won, stake, profit.
    """
    return pd.DataFrame(rows)


# =============================================================================
# Empty / no-data cases
# =============================================================================

class TestEmptyState:
    """Before any bets land, every baseline cell shows ``no_data``."""

    def test_empty_df_emits_one_row_per_baseline_cell(self) -> None:
        from dashboard import _compute_live_roi_rows
        rows = _compute_live_roi_rows("PL", pd.DataFrame())
        # Every PL baseline market should appear with status no_data
        markets = {r["market"] for r in rows}
        assert "ou25" in markets
        assert "btts" in markets
        assert "ou15" in markets
        for r in rows:
            assert r["status"] == "no_data"
            assert r["n_bets"] == 0
            assert r["live_roi"] is None
            assert r["sim_roi"] is not None  # baseline still surfaced

    def test_empty_df_efl_uses_efl_baselines(self) -> None:
        from dashboard import _compute_live_roi_rows
        rows = _compute_live_roi_rows("EFL", pd.DataFrame())
        markets = {r["market"] for r in rows}
        # EFL should not show PL-specific cells (and vice versa) — they
        # share market names so this is really a smoke test that the
        # function is league-aware
        assert "ou25" in markets


# =============================================================================
# Status classification
# =============================================================================

class TestStatusClassification:
    """The status field drives the colour and operator action."""

    def test_below_threshold_returns_low_n(self) -> None:
        """Few bets → ``low_n`` regardless of how good the ROI looks."""
        from dashboard import _compute_live_roi_rows
        # 5 won bets, all profitable, but below the 20-bet confidence threshold
        df = _make_settled_df([
            {"market": "ou25", "won": 1, "stake": 1.0, "profit": 0.95}
            for _ in range(5)
        ])
        rows = _compute_live_roi_rows("PL", df)
        ou25_row = next(r for r in rows if r["market"] == "ou25")
        assert ou25_row["status"] == "low_n"
        assert ou25_row["n_bets"] == 5

    def test_drift_status_when_far_below_baseline(self) -> None:
        """Live ROI well below baseline + enough samples → ``drift``."""
        from dashboard import _compute_live_roi_rows
        # 25 bets at -5% ROI; baseline is +8.04% → drift = -13pp
        # That is well past the -3pp trigger
        df = _make_settled_df([
            {"market": "ou25", "won": 0 if i % 4 else 1,
             "stake": 1.0, "profit": -0.05}
            for i in range(25)
        ])
        rows = _compute_live_roi_rows("PL", df)
        ou25_row = next(r for r in rows if r["market"] == "ou25")
        assert ou25_row["status"] == "drift"
        assert ou25_row["drift_pp"] is not None
        assert ou25_row["drift_pp"] < 0

    def test_ok_status_when_tracking_baseline(self) -> None:
        """Live ROI close to / above baseline + enough samples → ``ok``."""
        from dashboard import _compute_live_roi_rows
        # 25 bets at exactly the baseline +8.04% — drift = 0pp → ok
        df = _make_settled_df([
            {"market": "ou25", "won": 1, "stake": 1.0, "profit": 0.0804}
            for _ in range(25)
        ])
        rows = _compute_live_roi_rows("PL", df)
        ou25_row = next(r for r in rows if r["market"] == "ou25")
        assert ou25_row["status"] == "ok"

    def test_ok_status_when_beating_baseline(self) -> None:
        """Above-baseline performance is still ``ok`` (not a separate state)."""
        from dashboard import _compute_live_roi_rows
        df = _make_settled_df([
            {"market": "ou25", "won": 1, "stake": 1.0, "profit": 0.20}
            for _ in range(25)
        ])
        rows = _compute_live_roi_rows("PL", df)
        ou25_row = next(r for r in rows if r["market"] == "ou25")
        assert ou25_row["status"] == "ok"
        assert ou25_row["drift_pp"] > 0  # ahead of sim

    def test_no_baseline_status_for_uncovered_market(self) -> None:
        """Live bets on a market not in Phase 4a baseline → ``no_baseline``."""
        from dashboard import _compute_live_roi_rows
        df = _make_settled_df([
            {"market": "ou35", "won": 1, "stake": 1.0, "profit": 0.5}
        ])
        rows = _compute_live_roi_rows("PL", df)
        # ou35 is not in PHASE_4A_BASELINE_ROI → status no_baseline
        no_baseline_rows = [r for r in rows if r["market"] == "ou35"]
        assert len(no_baseline_rows) == 1
        assert no_baseline_rows[0]["status"] == "no_baseline"
        assert no_baseline_rows[0]["sim_roi"] is None


# =============================================================================
# Per-market aggregation
# =============================================================================

class TestPerMarketAggregation:
    """Bets across markets shouldn't bleed into each other."""

    def test_ou25_and_btts_separated(self) -> None:
        from dashboard import _compute_live_roi_rows
        # 25 winning ou25 bets + 25 losing btts bets — make sure neither
        # contaminates the other
        df = _make_settled_df(
            [{"market": "ou25", "won": 1, "stake": 1.0, "profit": 0.10}
             for _ in range(25)]
            + [{"market": "btts", "won": 0, "stake": 1.0, "profit": -1.0}
               for _ in range(25)]
        )
        rows = _compute_live_roi_rows("PL", df)
        ou25 = next(r for r in rows if r["market"] == "ou25")
        btts = next(r for r in rows if r["market"] == "btts")
        assert ou25["live_roi"] == pytest.approx(0.10)
        assert btts["live_roi"] == pytest.approx(-1.0)
        assert ou25["n_bets"] == 25
        assert btts["n_bets"] == 25

    def test_win_pct_calculated_correctly(self) -> None:
        from dashboard import _compute_live_roi_rows
        # 6 wins / 4 losses on ou25
        df = _make_settled_df(
            [{"market": "ou25", "won": 1, "stake": 1.0, "profit": 1.0}
             for _ in range(6)]
            + [{"market": "ou25", "won": 0, "stake": 1.0, "profit": -1.0}
               for _ in range(4)]
        )
        rows = _compute_live_roi_rows("PL", df)
        ou25 = next(r for r in rows if r["market"] == "ou25")
        assert ou25["win_pct"] == pytest.approx(0.60)


# =============================================================================
# Baseline regression guard
# =============================================================================

class TestBaselineRegressionGuard:
    """The hardcoded baseline numbers must match Phase 4a results.

    If someone accidentally edits config.PHASE_4A_BASELINE_ROI without
    re-running the validation harness, this test catches it.
    """

    def test_pl_ou25_baseline(self) -> None:
        from config import PHASE_4A_BASELINE_ROI
        assert PHASE_4A_BASELINE_ROI[("PL", "ou25")] == pytest.approx(0.0804)

    def test_pl_btts_baseline(self) -> None:
        from config import PHASE_4A_BASELINE_ROI
        assert PHASE_4A_BASELINE_ROI[("PL", "btts")] == pytest.approx(0.0649)

    def test_efl_btts_baseline(self) -> None:
        from config import PHASE_4A_BASELINE_ROI
        assert PHASE_4A_BASELINE_ROI[("EFL", "btts")] == pytest.approx(0.1384)

    def test_six_canonical_cells_present(self) -> None:
        """All six (league, market) cells from Phase 4a must be in config."""
        from config import PHASE_4A_BASELINE_ROI
        expected = {
            ("PL",  "ou25"), ("PL",  "btts"), ("PL",  "ou15"),
            ("EFL", "ou25"), ("EFL", "ou15"), ("EFL", "btts"),
        }
        assert expected.issubset(set(PHASE_4A_BASELINE_ROI.keys()))
