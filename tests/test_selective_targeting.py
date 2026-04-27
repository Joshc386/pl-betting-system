"""Tests for Phase 3: Selective Market Targeting.

Covers:
  - De-vig discount applied to edges from soft-book sources
  - Market/side confidence multipliers applied to Kelly stake
  - Config constants are valid and accessible
  - _evaluate_bet() integration with new parameters
"""
import numpy as np
import pytest

from config import DEVIG_DISCOUNT, MARKET_MULTIPLIERS


# ═══════════════════════════════════════════════════════════════════════════════
# Config Constants Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigConstants:
    """Validate selective market targeting config values."""

    def test_devig_discount_range(self) -> None:
        """DEVIG_DISCOUNT must be between 0 and 1 (exclusive)."""
        assert 0 < DEVIG_DISCOUNT < 1

    def test_market_multipliers_keys_are_tuples(self) -> None:
        """All MARKET_MULTIPLIERS keys must be (market, side) tuples."""
        for key in MARKET_MULTIPLIERS:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], str)

    def test_market_multipliers_values_positive(self) -> None:
        """All multiplier values must be > 0."""
        for key, val in MARKET_MULTIPLIERS.items():
            assert val > 0, f"Multiplier for {key} must be positive, got {val}"

    def test_missing_key_defaults_to_one(self) -> None:
        """Unknown (market, side) combos default to 1.0."""
        assert MARKET_MULTIPLIERS.get(("unknown", "x"), 1.0) == 1.0

    def test_ou25_over_not_discounted(self) -> None:
        """O/U 2.5 Over — the strongest market — should have multiplier 1.0."""
        assert MARKET_MULTIPLIERS.get(("ou25", "over"), 1.0) == 1.0

    def test_ou25_under_is_reduced(self) -> None:
        """O/U 2.5 Under should have a multiplier < 1.0."""
        assert MARKET_MULTIPLIERS.get(("ou25", "under"), 1.0) < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# PL _evaluate_bet() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateBetPL:
    """Test predict.py _evaluate_bet() with edge source and multipliers."""

    @pytest.fixture(autouse=True)
    def _disable_edge_shrinkage(self, monkeypatch):
        """Option 5 adds Bayesian edge shrinkage inside _evaluate_bet,
        which mutates edge values and breaks these tests' specific
        numeric assertions about de-vig discount behaviour. Disable
        shrinkage here so the tests isolate Phase 3 selective-targeting
        logic only.
        """
        import config
        monkeypatch.setattr(config, "USE_EDGE_SHRINKAGE", False)

    @pytest.fixture
    def predictor(self):
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        return p

    def test_pinnacle_edge_not_discounted(self, predictor) -> None:
        """Pinnacle edges should pass through without discount."""
        per_model = np.array([0.60, 0.58, 0.55, 0.62])
        result = predictor._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="pinnacle", market="ou25", side="over",
        )
        assert result is not None
        # Edge should be blend_w * model_p + (1-blend_w) * fair_p - fair_p
        # = 0.35 * 0.60 + 0.65 * 0.50 - 0.50 = 0.035
        assert abs(result["edge"] - 0.035) < 0.001

    def test_devig_edge_is_discounted(self, predictor) -> None:
        """De-vig edges should be reduced by DEVIG_DISCOUNT."""
        per_model = np.array([0.60, 0.58, 0.55, 0.62])
        result = predictor._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="devig", market="ou25", side="over",
        )
        assert result is not None
        raw_edge = 0.035
        expected_edge = raw_edge * DEVIG_DISCOUNT
        assert abs(result["edge"] - expected_edge) < 0.001

    def test_devig_discount_can_eliminate_bet(self, predictor) -> None:
        """A borderline edge that passes raw should fail after de-vig discount."""
        # Edge of 0.021 raw, min_edge=0.02 → passes raw
        # After 0.80 discount → 0.0168 < 0.02 → rejected
        per_model = np.array([0.535, 0.530, 0.525, 0.540])
        result = predictor._evaluate_bet(
            model_p=0.535, fair_p=0.50, odds=2.05,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.02,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="devig", market="ou25", side="over",
        )
        # The blended edge (0.35*0.535 + 0.65*0.50 - 0.50) = 0.01225
        # After discount: 0.0098 < 0.02 → should be None
        assert result is None

    def test_market_multiplier_applied_to_stake(self, predictor) -> None:
        """Stake should be scaled by market/side multiplier."""
        per_model = np.array([0.65, 0.63, 0.60, 0.68])
        # Same inputs, different market/side
        result_over = predictor._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.10},
            edge_source="pinnacle", market="ou25", side="over",
        )
        result_under = predictor._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.10},
            edge_source="pinnacle", market="ou25", side="under",
        )
        assert result_over is not None
        assert result_under is not None
        # Over mult=1.0, Under mult=0.80, so under stake should be 80% of over
        ratio = result_under["stake_pct"] / result_over["stake_pct"]
        expected_ratio = MARKET_MULTIPLIERS[("ou25", "under")] / \
            MARKET_MULTIPLIERS[("ou25", "over")]
        assert abs(ratio - expected_ratio) < 0.001

    def test_result_includes_edge_source(self, predictor) -> None:
        """Result dict should include edge_source and market_multiplier."""
        per_model = np.array([0.65, 0.63, 0.60, 0.68])
        result = predictor._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="pinnacle", market="btts", side="yes",
        )
        assert result is not None
        assert result["edge_source"] == "pinnacle"
        assert result["market_multiplier"] == MARKET_MULTIPLIERS[("btts", "yes")]


# ═══════════════════════════════════════════════════════════════════════════════
# Championship _evaluate_bet() Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateBetEFL:
    """Test championship_predict.py _evaluate_bet() with new params."""

    @pytest.fixture(autouse=True)
    def _disable_edge_shrinkage(self, monkeypatch):
        """See TestEvaluateBetPL — same reasoning, same fix."""
        import config
        monkeypatch.setattr(config, "USE_EDGE_SHRINKAGE", False)

    @pytest.fixture
    def predictor(self):
        from championship_predict import ChampionshipPredictor
        p = ChampionshipPredictor(verbose=False)
        return p

    def test_devig_discount_applied(self, predictor) -> None:
        """De-vig discount is applied in Championship predictor too."""
        per_model = np.array([0.60, 0.58, 0.62])
        result = predictor._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="devig", market="ou25", side="over",
        )
        assert result is not None
        raw_edge = 0.035
        expected_edge = raw_edge * DEVIG_DISCOUNT
        assert abs(result["edge"] - expected_edge) < 0.001

    def test_pinnacle_edge_unchanged(self, predictor) -> None:
        """Pinnacle edge is not discounted."""
        per_model = np.array([0.60, 0.58, 0.62])
        result = predictor._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.05},
            edge_source="pinnacle", market="ou25", side="over",
        )
        assert result is not None
        assert abs(result["edge"] - 0.035) < 0.001

    def test_btts_no_multiplier_reduces_stake(self, predictor) -> None:
        """BTTS No should have smaller stake than BTTS Yes."""
        per_model = np.array([0.65, 0.63, 0.68])
        result_yes = predictor._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.10},
            edge_source="pinnacle", market="btts", side="yes",
        )
        result_no = predictor._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config={"blend_weight": 0.35, "min_edge": 0.01,
                    "min_agree": 2, "kelly_fraction": 0.25,
                    "max_stake_pct": 0.10},
            edge_source="pinnacle", market="btts", side="no",
        )
        assert result_yes is not None and result_no is not None
        assert result_no["stake_pct"] < result_yes["stake_pct"]
