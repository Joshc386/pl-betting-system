"""Tests for Option 5: Staking Refinement.

Covers:
  Step 0 — edge-magnitude scaling bug fix (the >0.06 branch reachable)
  Step 1 — matchday portfolio Kelly cap
  Step 2c — same-match correlation discount
  Step 3 — Bayesian edge uncertainty shrinkage
"""
from __future__ import annotations

import pytest


# =============================================================================
# Step 0 — bug fix: edge-magnitude scaling branches resolve correctly
# =============================================================================

class TestEdgeMagnitudeScalingFix:
    """Before Step 0 the `>0.06` branch was unreachable. After the fix,
    crossing that threshold produces a 1.25x multiplier (vs 1.15x at
    >0.04 and 1.0x at <=0.04).
    """

    def test_large_edge_uses_125x_branch(self) -> None:
        """Edge of 0.07 produces a stake ~8.7% larger than at edge 0.04.

        1.25x vs 1.15x = 8.7% relative increase. If the branch were still
        broken, we'd see 0% difference (both at 1.15x).
        """
        from backtest import refined_kelly
        s_medium = refined_kelly(
            blended_prob=0.55, odds=2.10, n_agree=4, edge=0.04)
        s_large = refined_kelly(
            blended_prob=0.55, odds=2.10, n_agree=4, edge=0.07)
        ratio = s_large / s_medium
        # Expected ratio ~1.087. Anything above 1.05 proves the fix worked.
        assert ratio > 1.05, f"Expected >5% ratio, got {ratio:.4f}"

    def test_small_edge_no_multiplier(self) -> None:
        """Edge at or below 0.04 gets no boost."""
        from backtest import refined_kelly
        s_none = refined_kelly(
            blended_prob=0.55, odds=2.10, n_agree=4, edge=0.02)
        s_small = refined_kelly(
            blended_prob=0.55, odds=2.10, n_agree=4, edge=0.04)
        # No boost applies for <=0.04 in the threshold check — they should match
        assert s_none == pytest.approx(s_small, abs=1e-6)


# =============================================================================
# Step 1 + 2c — portfolio constraints
# =============================================================================

class TestPortfolioConstraints:
    """apply_portfolio_constraints() enforces same-match discount + cap."""

    def test_no_op_single_fixture_single_market(self) -> None:
        """A single bet on one fixture gets no adjustment."""
        from backtest import apply_portfolio_constraints
        recs = [{
            "home_team": "A", "away_team": "B",
            "kickoff": "2026-04-22T15:00:00Z",
            "market": "ou25", "stake_pct": 0.04,
        }]
        apply_portfolio_constraints(recs)
        assert recs[0]["stake_pct"] == pytest.approx(0.04)

    def test_no_op_below_cap(self) -> None:
        """Bets totalling < 10% and no same-match pairs stay unchanged."""
        from backtest import apply_portfolio_constraints
        recs = [
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "ou25", "stake_pct": 0.03},
            {"home_team": "C", "away_team": "D",
             "kickoff": "2026-04-22T17:00:00Z",
             "market": "ou25", "stake_pct": 0.03},
        ]
        apply_portfolio_constraints(recs)
        assert recs[0]["stake_pct"] == pytest.approx(0.03)
        assert recs[1]["stake_pct"] == pytest.approx(0.03)

    def test_same_match_no_longer_discounted(self) -> None:
        """Post-Phase-5 cleanup: same-match discount was removed because
        Phase 4a showed it fired zero times across all validated cells.
        Multiple markets on the same fixture now pass through unchanged
        (unless the aggregate breaches MAX_MATCHDAY_STAKE_PCT).

        See reports/roi_findings.md action A1 for the rationale.
        """
        from backtest import apply_portfolio_constraints
        recs = [
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "ou25", "stake_pct": 0.04},
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "btts", "stake_pct": 0.03},
        ]
        apply_portfolio_constraints(recs)
        # Total 0.07 < 0.10 cap → no scaling of any kind
        assert recs[0]["stake_pct"] == pytest.approx(0.04, abs=1e-6)
        assert recs[1]["stake_pct"] == pytest.approx(0.03, abs=1e-6)

    def test_portfolio_cap_pro_rates(self) -> None:
        """When matchday sum > 10%, every stake gets scaled proportionally."""
        from backtest import apply_portfolio_constraints
        recs = [
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T12:00:00Z",
             "market": "ou25", "stake_pct": 0.04},
            {"home_team": "C", "away_team": "D",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "ou25", "stake_pct": 0.05},
            {"home_team": "E", "away_team": "F",
             "kickoff": "2026-04-22T17:30:00Z",
             "market": "ou25", "stake_pct": 0.06},
        ]
        # Total 0.15, cap at 0.10, scale = 0.10/0.15 = 0.667
        apply_portfolio_constraints(recs)
        total = sum(r["stake_pct"] for r in recs)
        assert total == pytest.approx(0.10, abs=1e-4)
        # Each stake scaled by ~0.667
        assert recs[0]["stake_pct"] == pytest.approx(0.04 * 2/3, abs=1e-4)
        assert recs[2]["stake_pct"] == pytest.approx(0.06 * 2/3, abs=1e-4)

    def test_combined_same_match_and_cap(self) -> None:
        """Post-Phase-5: same-match discount removed, only matchday cap
        remains. Aggregate stake above cap gets pro-rated down regardless
        of whether bets overlap on a fixture.
        """
        from backtest import apply_portfolio_constraints
        recs = [
            # Same fixture, both markets — no longer discounted
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "ou25", "stake_pct": 0.05},
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "btts", "stake_pct": 0.04},
            # Different fixture, same day
            {"home_team": "C", "away_team": "D",
             "kickoff": "2026-04-22T17:30:00Z",
             "market": "ou25", "stake_pct": 0.05},
        ]
        apply_portfolio_constraints(recs)
        # Sum = 0.14, cap 0.10 → scale = 0.10/0.14 = 0.7143
        total = sum(r["stake_pct"] for r in recs)
        assert total == pytest.approx(0.10, abs=1e-4)
        assert recs[0]["stake_pct"] == pytest.approx(0.05 * 10/14, abs=1e-4)
        assert recs[1]["stake_pct"] == pytest.approx(0.04 * 10/14, abs=1e-4)
        assert recs[2]["stake_pct"] == pytest.approx(0.05 * 10/14, abs=1e-4)

    def test_under_cap_no_scaling(self) -> None:
        """When aggregate matchday stake is under the cap, no scaling
        happens. (Also incidentally confirms the same-match discount is
        gone — two different fixtures pass through unchanged.)
        """
        from backtest import apply_portfolio_constraints
        recs = [
            {"home_team": "A", "away_team": "B",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "ou25", "stake_pct": 0.02},
            {"home_team": "C", "away_team": "D",
             "kickoff": "2026-04-22T15:00:00Z",
             "market": "btts", "stake_pct": 0.02},
        ]
        # Total 0.04 < 0.10 cap — no scaling
        apply_portfolio_constraints(recs)
        assert recs[0]["stake_pct"] == pytest.approx(0.02)
        assert recs[1]["stake_pct"] == pytest.approx(0.02)

    def test_empty_list_ok(self) -> None:
        """No crash when called with empty list."""
        from backtest import apply_portfolio_constraints
        result = apply_portfolio_constraints([])
        assert result == []


# =============================================================================
# Step 3 — edge shrinkage
# =============================================================================

class TestEdgeShrinkage:
    """shrink_edge() pulls edges toward zero, weighted by model agreement."""

    def test_higher_agreement_less_shrinkage(self) -> None:
        """4/4 agreement shrinks less than 2/4."""
        from backtest import shrink_edge
        shrunk_4 = shrink_edge(0.05, n_agree=4)
        shrunk_2 = shrink_edge(0.05, n_agree=2)
        # 4/4 keeps 10/12 = 0.833; 2/4 keeps 3/5 = 0.60
        assert shrunk_4 > shrunk_2
        assert shrunk_4 == pytest.approx(0.05 * (10 / 12), abs=1e-5)
        assert shrunk_2 == pytest.approx(0.05 * (3 / 5), abs=1e-5)

    def test_preserves_sign(self) -> None:
        """Negative edges stay negative after shrinkage."""
        from backtest import shrink_edge
        assert shrink_edge(-0.03, n_agree=4) < 0
        assert shrink_edge(0.03, n_agree=4) > 0

    def test_zero_edge_stays_zero(self) -> None:
        """A zero raw edge remains zero."""
        from backtest import shrink_edge
        assert shrink_edge(0.0, n_agree=4) == 0.0
        assert shrink_edge(0.0, n_agree=2) == 0.0

    def test_magnitude_reduced(self) -> None:
        """|shrunk_edge| < |raw_edge| for all positive agreement values."""
        from backtest import shrink_edge
        for edge in [0.01, 0.03, 0.05, 0.10]:
            for n_agree in range(5):
                shrunk = shrink_edge(edge, n_agree)
                assert abs(shrunk) <= abs(edge) + 1e-9

    def test_unknown_agreement_defaults_low(self) -> None:
        """Out-of-range n_agree falls back to n=1 (heavy shrinkage)."""
        from backtest import shrink_edge
        shrunk = shrink_edge(0.05, n_agree=99)
        # Should use default n=1, so kept = 1/3
        assert shrunk == pytest.approx(0.05 * (1 / 3), abs=1e-5)

    def test_can_be_disabled_via_config(self, monkeypatch) -> None:
        """When USE_EDGE_SHRINKAGE=False, edge passes through unchanged."""
        import config
        monkeypatch.setattr(config, "USE_EDGE_SHRINKAGE", False)
        from backtest import shrink_edge
        assert shrink_edge(0.05, n_agree=2) == pytest.approx(0.05)
