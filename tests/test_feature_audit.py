"""Tests for Option 3 Step 1 / Step 3 infrastructure.

Covers:
  - get_active_features() helper (config.py)
  - use_sparse_features flag plumbing in LivePredictor and
    ChampionshipPredictor constructors.
  - Light audit_features() sanity: output schema + noise-baseline presence.

Deliberately does NOT run a full audit against real pipeline data - that's
covered by scripts/run_feature_audit.py end-to-end.
"""
from __future__ import annotations

import pytest
import pandas as pd


# =============================================================================
# get_active_features()
# =============================================================================

class TestGetActiveFeatures:
    """Pure-logic tests for the config helper."""

    def test_returns_input_unchanged_when_use_sparse_true(self) -> None:
        """use_sparse=True never filters anything."""
        from config import get_active_features
        base = ["a", "b", "c", "d"]
        assert get_active_features(base, use_sparse=True) == base

    def test_returns_input_when_sparse_groups_empty(self) -> None:
        """Empty SPARSE_FEATURE_GROUPS means use_sparse=False is a no-op."""
        from config import get_active_features
        import config
        original_groups = config.SPARSE_FEATURE_GROUPS
        try:
            config.SPARSE_FEATURE_GROUPS = {}
            base = ["a", "b", "c"]
            assert get_active_features(base, use_sparse=False) == base
        finally:
            config.SPARSE_FEATURE_GROUPS = original_groups

    def test_excludes_features_in_flagged_groups(self) -> None:
        """use_sparse=False drops features that appear in any flagged group."""
        from config import get_active_features
        import config
        original_groups = config.SPARSE_FEATURE_GROUPS
        try:
            config.SPARSE_FEATURE_GROUPS = {
                "GROUP_A": ["a", "b"],
                "GROUP_C": ["e"],
            }
            base = ["a", "b", "c", "d", "e", "f"]
            result = get_active_features(base, use_sparse=False)
            # Should drop a, b (from GROUP_A) and e (from GROUP_C); keep c, d, f
            assert result == ["c", "d", "f"]
        finally:
            config.SPARSE_FEATURE_GROUPS = original_groups

    def test_preserves_order(self) -> None:
        """Filter must preserve the input order of the base list."""
        from config import get_active_features
        import config
        original_groups = config.SPARSE_FEATURE_GROUPS
        try:
            config.SPARSE_FEATURE_GROUPS = {"DROP": ["b"]}
            assert (get_active_features(["a", "b", "c"], use_sparse=False)
                    == ["a", "c"])
            assert (get_active_features(["c", "b", "a"], use_sparse=False)
                    == ["c", "a"])
        finally:
            config.SPARSE_FEATURE_GROUPS = original_groups

    def test_returns_list_not_original_reference(self) -> None:
        """Even when passthrough, return a copy to avoid aliasing."""
        from config import get_active_features
        base = ["a", "b"]
        result = get_active_features(base, use_sparse=True)
        result.append("c")
        assert base == ["a", "b"]  # original unchanged


# =============================================================================
# Predictor constructor flag plumbing
# =============================================================================

class TestUseSparseFeaturesFlag:
    """Verify LivePredictor and ChampionshipPredictor expose the flag
    and honour it when explicit; otherwise fall back to the config default.
    """

    def test_pl_predictor_default_matches_config(self) -> None:
        """LivePredictor with no flag arg picks up USE_SPARSE_FEATURES."""
        from predict import LivePredictor
        from config import USE_SPARSE_FEATURES
        p = LivePredictor(verbose=False)
        assert p.use_sparse_features == USE_SPARSE_FEATURES

    def test_pl_predictor_explicit_override(self) -> None:
        """Explicit constructor arg overrides the global default."""
        from predict import LivePredictor
        p_on = LivePredictor(verbose=False, use_sparse_features=True)
        p_off = LivePredictor(verbose=False, use_sparse_features=False)
        assert p_on.use_sparse_features is True
        assert p_off.use_sparse_features is False

    def test_efl_predictor_default_matches_config(self) -> None:
        """ChampionshipPredictor with no flag arg picks up the global."""
        from championship_predict import ChampionshipPredictor
        from config import USE_SPARSE_FEATURES
        p = ChampionshipPredictor(verbose=False)
        assert p.use_sparse_features == USE_SPARSE_FEATURES

    def test_efl_predictor_explicit_override(self) -> None:
        """Explicit arg overrides the global default for EFL too."""
        from championship_predict import ChampionshipPredictor
        p_on = ChampionshipPredictor(
            verbose=False, use_sparse_features=True)
        p_off = ChampionshipPredictor(
            verbose=False, use_sparse_features=False)
        assert p_on.use_sparse_features is True
        assert p_off.use_sparse_features is False


# =============================================================================
# audit_features schema
# =============================================================================

class TestAuditFeaturesSchema:
    """Light schema tests on audit_features output. Uses a minimal
    synthetic DataFrame so we don't depend on the full pipeline.
    """

    @pytest.fixture
    def synthetic_df(self) -> pd.DataFrame:
        """Build a tiny DataFrame that has the minimum columns audit needs."""
        import numpy as np
        from datetime import datetime, timedelta

        rng = np.random.default_rng(7)
        rows = []
        base = datetime(2019, 8, 1)
        teams = [f"T{i}" for i in range(6)]
        # Seasons 18, 19, 20 — need >=14 for audit to even consider
        for season in (18, 19, 20):
            # Each season gets 60 matches
            for i in range(60):
                home = teams[i % 6]
                away = teams[(i + 1) % 6]
                hg = rng.poisson(1.3)
                ag = rng.poisson(1.0)
                rows.append({
                    "SeasonIndex": season,
                    "Date": base + timedelta(days=season * 300 + i),
                    "Home_Team": home,
                    "Away_Team": away,
                    "Home_Goals": hg,
                    "Away_Goals": ag,
                    "Over_2_5": int((hg + ag) > 2),
                    # Include at least one feature from ALL_FEATURES so the
                    # audit has something to work with
                    "Home_ScoringRate_10": rng.normal(0, 1),
                    "Away_ScoringRate_10": rng.normal(0, 1),
                })
        return pd.DataFrame(rows)

    def test_schema_has_expected_columns(self, synthetic_df) -> None:
        """Output DataFrame has the expected column names."""
        from model import audit_features
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            df = audit_features(
                league="PL", market="ou25",
                full_df=synthetic_df, output_dir=tmp,
                verbose=False, n_permutations=2,
            )
        expected = {"feature", "group", "xgb_gain", "lgb_gain",
                    "perm_auc_drop", "perm_auc_drop_std",
                    "nan_rate", "pruning_candidate"}
        assert expected.issubset(set(df.columns))

    def test_noise_baseline_row_present(self, synthetic_df) -> None:
        """Output includes a row for the Gaussian noise baseline feature."""
        from model import audit_features
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            df = audit_features(
                league="PL", market="ou25",
                full_df=synthetic_df, output_dir=tmp,
                verbose=False, n_permutations=2,
            )
        assert (df["feature"] == "__noise_baseline__").any()
        noise_row = df[df["feature"] == "__noise_baseline__"].iloc[0]
        assert noise_row["group"] == "NOISE_BASELINE"
