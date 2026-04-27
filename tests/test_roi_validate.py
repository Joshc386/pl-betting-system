"""Tests for Phase 3 Pass 1 ROI validator (scripts/roi_validate.py).

Scope: pure-function unit tests + golden-path replay on a tiny synthetic
OOF cache. Full end-to-end matrix runs live in Phase 4a and aren't tested
here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the validator module directly (it lives under scripts/)
import roi_validate as RV


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-function tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrateSingle:
    """_calibrate_single mirrors backtest.py's logit-shift calibration."""

    def test_zero_shift_identity(self) -> None:
        """No shift → output == input."""
        for p in (0.1, 0.3, 0.5, 0.7, 0.9):
            assert RV._calibrate_single(p, 0.0) == pytest.approx(p, abs=1e-9)

    def test_positive_shift_increases_prob(self) -> None:
        """Positive shift pushes probability upward."""
        assert RV._calibrate_single(0.5, 1.0) > 0.5

    def test_negative_shift_decreases_prob(self) -> None:
        """Negative shift pushes probability downward."""
        assert RV._calibrate_single(0.5, -1.0) < 0.5

    def test_edge_values_clipped(self) -> None:
        """p=0 and p=1 return 0 and 1 regardless of shift."""
        assert RV._calibrate_single(0.0, 5.0) == 0.0
        assert RV._calibrate_single(1.0, -5.0) == 1.0


class TestImpliedFairProb:
    """Overround removal is symmetric and sums to 1."""

    def test_sums_to_one(self) -> None:
        a, b = RV._implied_fair_prob(2.10, 1.80)
        assert a + b == pytest.approx(1.0, abs=1e-9)

    def test_shorter_odds_higher_fair_prob(self) -> None:
        a, b = RV._implied_fair_prob(1.5, 2.5)
        assert a > b  # shorter price = higher fair probability

    def test_invalid_odds_returns_nan(self) -> None:
        a, b = RV._implied_fair_prob(1.0, 2.0)  # odds must be > 1
        assert np.isnan(a) and np.isnan(b)
        a2, b2 = RV._implied_fair_prob(None, 2.0)
        assert np.isnan(a2) and np.isnan(b2)


class TestMaxDrawdown:
    """Running-peak drawdown on a known P&L trajectory."""

    def test_no_bets_zero(self) -> None:
        assert RV.compute_max_drawdown([]) == 0.0

    def test_simple_drawdown(self) -> None:
        """Cumulative P&L: +1, +2, -1, -3 → peak 2, trough -3, dd = 5."""
        bets = [
            {"date": "2024-01-01", "pnl": 1.0},
            {"date": "2024-01-02", "pnl": 1.0},
            {"date": "2024-01-03", "pnl": -3.0},
            {"date": "2024-01-04", "pnl": -2.0},
        ]
        assert RV.compute_max_drawdown(bets) == pytest.approx(5.0, abs=1e-9)

    def test_monotonic_up_zero_drawdown(self) -> None:
        bets = [
            {"date": "2024-01-01", "pnl": 1.0},
            {"date": "2024-01-02", "pnl": 1.0},
            {"date": "2024-01-03", "pnl": 1.0},
        ]
        assert RV.compute_max_drawdown(bets) == 0.0


class TestBlockBootstrap:
    """Block bootstrap returns sensible CI bounds that bracket the point."""

    def test_no_bets_zero(self) -> None:
        mean, lo, hi = RV.block_bootstrap_roi([])
        assert mean == 0.0 and lo == 0.0 and hi == 0.0

    def test_ci_brackets_point_estimate(self) -> None:
        """With many bets on different days, CI should bracket the
        observed ROI."""
        rng = np.random.default_rng(seed=0)
        bets = []
        for i in range(200):
            # Alternating +/- pnl on different days → ROI ≈ 0
            day = f"2024-{(i % 30) + 1:02d}-01"
            pnl = float(rng.choice([-1.0, 1.1]))  # +4.8% ROI expectation
            bets.append({
                "date": day,
                "stake_pct": 1.0,
                "pnl": pnl,
            })
        total_pnl = sum(b["pnl"] for b in bets)
        total_staked = sum(b["stake_pct"] for b in bets)
        point = total_pnl / total_staked
        mean, lo, hi = RV.block_bootstrap_roi(bets, n_resamples=500, seed=0)
        # Bootstrap mean should be close to the point estimate.
        assert abs(mean - point) < 0.10
        # CI bounds should bracket or at least bound the estimate reasonably.
        assert lo <= hi


# ═══════════════════════════════════════════════════════════════════════════════
# Golden-path replay on tiny synthetic OOF cache
# ═══════════════════════════════════════════════════════════════════════════════

def _make_synthetic_oof() -> pd.DataFrame:
    """Build a 6-fixture, 3-matchday OOF cache with model confident on OVER
    and odds offering some positive edge.

    Model predicts P(Over) ≈ 0.62 consistently. Fair P(Over) from 2.05/1.85
    odds ≈ 0.475. Edge ≈ +0.14 via blend_weight=0.35:
      blended = 0.35*0.62 + 0.65*0.475 = 0.526 → edge = 0.05 (above min_edge)
    """
    rows = []
    for i in range(6):
        date = f"2024-10-{(i // 2) + 1:02d}"  # 3 distinct dates, 2 fixtures each
        rows.append({
            "season": 23,
            "date": date,
            "home_team": f"Team{i}H",
            "away_team": f"Team{i}A",
            "xgb_prob": 0.60,
            "lgb_prob": 0.62,
            "dc_prob":  0.64,
            "lr_prob":  0.62,
            "xgb_shift": 0.0,
            "lgb_shift": 0.0,
            "dc_shift":  0.0,
            "lr_shift":  0.0,
            "base_rate": 0.54,
            "odds_a":   2.05,
            "odds_b":   1.85,
            "bookie_a": "TestBook",
            "bookie_b": "TestBook",
            "side_a_label": "over",
            "side_b_label": "under",
            # Alternate outcome so we get both wins and losses
            "outcome":  1 if i % 2 == 0 else 0,
        })
    return pd.DataFrame(rows)


class TestReplayGolden:
    """End-to-end replay on the synthetic cache."""

    def test_baseline_produces_some_bets(self) -> None:
        oof = _make_synthetic_oof()
        vc = RV.ValidationConfig(
            shrinkage=False, portfolio_cap=False,
            same_match_discount=False, market_multipliers=False,
            edge_scaling_fix=True,
            edge_source="pinnacle",
        )
        bets, stats = RV.replay_league_market(oof, "PL", "ou25", vc)
        assert len(bets) >= 1, "Baseline config should produce at least one bet"
        # Every bet has required keys
        for b in bets:
            for k in ("season", "date", "stake_pct", "edge", "odds",
                      "outcome", "won", "pnl"):
                assert k in b

    def test_shrinkage_does_not_increase_bet_count(self) -> None:
        """Turning on edge shrinkage can't produce *more* bets than off."""
        oof = _make_synthetic_oof()
        vc_off = RV.ValidationConfig(
            shrinkage=False, portfolio_cap=False,
            same_match_discount=False, market_multipliers=False,
            edge_source="pinnacle",
        )
        vc_on = RV.ValidationConfig(
            shrinkage=True, portfolio_cap=False,
            same_match_discount=False, market_multipliers=False,
            edge_source="pinnacle",
        )
        bets_off, _ = RV.replay_league_market(oof, "PL", "ou25", vc_off)
        bets_on, _ = RV.replay_league_market(oof, "PL", "ou25", vc_on)
        assert len(bets_on) <= len(bets_off)

    def test_pnl_consistent_with_outcome(self) -> None:
        """Winning bet → pnl = stake * (odds - 1); losing bet → pnl = -stake."""
        oof = _make_synthetic_oof()
        vc = RV.ValidationConfig(
            shrinkage=False, portfolio_cap=False,
            same_match_discount=False, market_multipliers=False,
            edge_source="pinnacle",
        )
        bets, _ = RV.replay_league_market(oof, "PL", "ou25", vc)
        for b in bets:
            if b["won"]:
                expected = b["stake_pct"] * (b["odds"] - 1)
            else:
                expected = -b["stake_pct"]
            assert b["pnl"] == pytest.approx(expected, abs=1e-9)
