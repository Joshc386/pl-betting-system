"""Tests for the vectorised P(BTTS) calculation in DixonColesPredictor.

Background
----------
The original ``predict_match_btts`` looped over a 12x12 score grid
calling ``scipy.stats.poisson.pmf`` 288 times per fixture. Under load
this intermittently hung for hours during DC tuning (the BTTS DC tune
needs ~5M scipy calls per CV grid sweep).

The replacement uses a closed-form decomposition:

  P(BTTS) = 1 - P(home=0) - P(away=0) + P(home=0, away=0)

Because the DC tau correction is non-trivial only at the four
low-score cells (0,0)/(0,1)/(1,0)/(1,1), the marginals decompose to
expressions involving just pmf(0, lam) and pmf(1, lam) — both
computable directly from the Poisson formula without scipy.

These tests verify the new implementation is **numerically equivalent**
to the legacy scipy-based loop within float tolerance, covering:

  1. Equal-strength teams (lam_h == lam_a) across rho regimes
  2. Heavy mismatch (e.g. Man City vs Sunderland) where one lambda
     is 3-4x the other
  3. The four edge values where the rho clamp matters
  4. Extreme but valid lambdas at the [0.1, 5.0] clip boundaries
  5. predict_proba_btts_df shape + integration with map-based lookup
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson


def _slow_reference_btts(home_lambda: float, away_lambda: float,
                         rho: float) -> float:
    """The pre-vectorisation implementation, kept here as the test oracle.

    Identical to what was in model.py before the rewrite — 12x12 grid
    of scipy.poisson.pmf calls with DC tau correction, then
    inclusion-exclusion to extract P(BTTS).
    """
    def tau(x, y, lam, mu):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        if x == 0 and y == 1:
            return 1 + lam * rho
        if x == 1 and y == 0:
            return 1 + mu * rho
        if x == 1 and y == 1:
            return 1 - rho
        return 1.0

    p_home_zero = 0.0
    p_away_zero = 0.0
    p_both_zero = 0.0

    for h in range(12):
        for a in range(12):
            p = (poisson.pmf(h, home_lambda) *
                 poisson.pmf(a, away_lambda) *
                 tau(h, a, home_lambda, away_lambda))
            p = max(p, 0)
            if h == 0:
                p_home_zero += p
            if a == 0:
                p_away_zero += p
            if h == 0 and a == 0:
                p_both_zero += p

    p_btts = 1 - p_home_zero - p_away_zero + p_both_zero
    return float(np.clip(p_btts, 0.01, 0.99))


def _build_dc_with_teams(rho: float = -0.10, mu: float = 1.35,
                         gamma: float = 1.30):
    """Construct a minimal DC predictor with handpicked team ratings."""
    from model import DixonColesPredictor
    dc = DixonColesPredictor(half_life=20, rho=rho)
    dc.mu = mu
    dc.gamma = gamma
    # Teams with deliberately varied attack/defence to hit different
    # lambda regimes
    dc.attack_home = {"Strong": 1.40, "Average": 1.00, "Weak": 0.60}
    dc.attack_away = {"Strong": 1.30, "Average": 1.00, "Weak": 0.55}
    dc.defence_home = {"Strong": 0.65, "Average": 1.00, "Weak": 1.40}
    dc.defence_away = {"Strong": 0.70, "Average": 1.00, "Weak": 1.45}
    return dc


# =============================================================================
# Single-match equivalence
# =============================================================================

class TestNumericalEquivalence:
    """New vectorised implementation matches the old slow loop."""

    # Tolerance note: the slow reference truncates the Poisson tail at
    # score=11 in each dimension; the vectorised version uses the
    # closed-form (infinite-sum) decomposition, so it's slightly more
    # accurate. At high lambdas the truncation error reaches ~7e-5;
    # at typical match lambdas (1-2.5) it's < 1e-7. A 1e-3 tolerance
    # is well below any threshold that affects bet selection (model
    # probabilities are blended with bookmaker odds at coarser
    # precision than this).
    @pytest.mark.parametrize("rho", [-0.20, -0.10, 0.0, 0.10, 0.20])
    @pytest.mark.parametrize("home,away", [
        ("Strong", "Weak"),
        ("Weak", "Strong"),
        ("Strong", "Strong"),
        ("Average", "Average"),
        ("Strong", "Average"),
    ])
    def test_predict_match_btts_matches_reference(
            self, rho, home, away):
        from scipy.stats import poisson  # used by reference
        dc = _build_dc_with_teams(rho=rho)
        new_p = dc.predict_match_btts(home, away)

        # Reproduce the lambda computation exactly so the reference
        # uses the same inputs
        h_att = dc.attack_home[home]
        a_def = dc.defence_away[away]
        a_att = dc.attack_away[away]
        h_def = dc.defence_home[home]
        sg = float(np.sqrt(dc.gamma))
        h_lam = float(np.clip(h_att * a_def * dc.mu * sg, 0.1, 5.0))
        a_lam = float(np.clip(a_att * h_def * dc.mu / sg, 0.1, 5.0))

        old_p = _slow_reference_btts(h_lam, a_lam, rho)
        assert new_p == pytest.approx(old_p, abs=1e-3), (
            f"BTTS mismatch for {home} vs {away}, rho={rho}: "
            f"new={new_p:.6f}, old={old_p:.6f}")


# =============================================================================
# Lambda boundary cases
# =============================================================================

class TestLambdaBoundaries:
    """Vectorised version handles the [0.1, 5.0] lambda clip cleanly."""

    def test_extreme_low_lambdas(self):
        """Both teams expected to score very few goals — P(BTTS) should
        be small (matching reference). Verifies no floating point
        catastrophe at the lower clip."""
        dc = _build_dc_with_teams(rho=-0.10)
        # Manually call the vectorised core with extreme inputs
        out = dc._predict_btts_from_lambdas(
            np.array([0.1]), np.array([0.1]))
        ref = _slow_reference_btts(0.1, 0.1, -0.10)
        assert float(out[0]) == pytest.approx(ref, abs=1e-3)

    def test_extreme_high_lambdas(self):
        """Both teams expected to score lots — P(BTTS) very high."""
        dc = _build_dc_with_teams(rho=-0.10)
        out = dc._predict_btts_from_lambdas(
            np.array([5.0]), np.array([5.0]))
        ref = _slow_reference_btts(5.0, 5.0, -0.10)
        assert float(out[0]) == pytest.approx(ref, abs=1e-3)

    def test_asymmetric_lambdas(self):
        """One team very strong, one very weak — common in real fixtures."""
        dc = _build_dc_with_teams(rho=-0.15)
        out = dc._predict_btts_from_lambdas(
            np.array([3.5]), np.array([0.4]))
        ref = _slow_reference_btts(3.5, 0.4, -0.15)
        assert float(out[0]) == pytest.approx(ref, abs=1e-3)


# =============================================================================
# Vectorised-vs-row equivalence
# =============================================================================

class TestVectorisedDfPath:
    """The DataFrame method gives the same answer as per-row calls."""

    def test_predict_proba_btts_df_matches_per_row(self):
        dc = _build_dc_with_teams(rho=-0.10)
        df = pd.DataFrame([
            {"Home_Team": "Strong",  "Away_Team": "Weak"},
            {"Home_Team": "Weak",    "Away_Team": "Strong"},
            {"Home_Team": "Average", "Away_Team": "Average"},
            {"Home_Team": "Strong",  "Away_Team": "Strong"},
        ])
        df_probs = dc.predict_proba_btts_df(df)
        per_row = np.array([
            dc.predict_match_btts(r["Home_Team"], r["Away_Team"])
            for _, r in df.iterrows()
        ])
        np.testing.assert_allclose(df_probs, per_row, atol=1e-5)
        assert df_probs.shape == (4,)

    def test_handles_unseen_teams_via_priors(self):
        """Teams not in attack/defence dicts fall back to PRIORS — the
        DataFrame path needs to handle this without KeyError."""
        dc = _build_dc_with_teams(rho=-0.10)
        df = pd.DataFrame([
            {"Home_Team": "BrandNew", "Away_Team": "AlsoNew"},
        ])
        # Should not raise
        out = dc.predict_proba_btts_df(df)
        assert out.shape == (1,)
        assert 0.01 <= out[0] <= 0.99

    def test_empty_df_returns_empty_array(self):
        dc = _build_dc_with_teams(rho=-0.10)
        out = dc.predict_proba_btts_df(pd.DataFrame(columns=[
            "Home_Team", "Away_Team"]))
        assert out.shape == (0,)


# =============================================================================
# Speed sanity check
# =============================================================================

class TestPerformance:
    """Sanity bound on the speedup so a future regression is loud.

    The slow reference takes ~3-5ms per fixture. The vectorised path
    handles ~10000+ fixtures in the same time. For a 200-fixture run
    we're conservatively asserting < 0.5s wallclock.
    """

    def test_200_fixtures_runs_quickly(self):
        import time
        dc = _build_dc_with_teams(rho=-0.10)
        df = pd.DataFrame(
            [{"Home_Team": "Strong", "Away_Team": "Weak"}] * 200)
        t0 = time.perf_counter()
        out = dc.predict_proba_btts_df(df)
        elapsed = time.perf_counter() - t0
        assert out.shape == (200,)
        assert elapsed < 0.5, (
            f"Vectorised BTTS took {elapsed:.2f}s for 200 fixtures — "
            "regression in vectorisation; should be well under 0.5s")
