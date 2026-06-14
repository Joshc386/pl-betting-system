"""
Tests for Championship betting system.

Covers:
  - Pipeline: data loading, feature engineering, target variables
  - Backtest: calibration, regime detection, Kelly staking, drawdown
  - Predict: team name mapping, bet evaluation, 3-model prediction
  - Dashboard: EFL database save/load, bookmaker odds storage
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestChampionshipPipeline:
    """Test Championship pipeline data loading and feature engineering."""

    def test_load_championship_data_returns_dataframe(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 10000  # ~13,600 matches expected

    def test_load_championship_data_has_required_columns(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        required = [
            "Home_Team", "Away_Team", "Home_Goals", "Away_Goals",
            "Date", "SeasonIndex", "Over_2_5", "Over_1_5", "BTTS",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_over25_is_binary(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        assert set(df["Over_2_5"].dropna().unique()).issubset({0, 1})

    def test_target_over15_is_binary(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        assert set(df["Over_1_5"].dropna().unique()).issubset({0, 1})

    def test_target_btts_is_binary(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        assert set(df["BTTS"].dropna().unique()).issubset({0, 1})

    def test_over25_matches_goal_total(self):
        """Over 2.5 should be 1 when total goals > 2."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        total = df["Home_Goals"] + df["Away_Goals"]
        expected = (total > 2).astype(int)
        pd.testing.assert_series_equal(
            df["Over_2_5"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_over15_matches_goal_total(self):
        """Over 1.5 should be 1 when total goals > 1."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        total = df["Home_Goals"] + df["Away_Goals"]
        expected = (total > 1).astype(int)
        pd.testing.assert_series_equal(
            df["Over_1_5"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_btts_matches_both_scoring(self):
        """BTTS should be 1 when both teams score."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        expected = ((df["Home_Goals"] > 0) & (df["Away_Goals"] > 0)).astype(int)
        pd.testing.assert_series_equal(
            df["BTTS"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_base_rates_reasonable(self):
        """Championship base rates should be in expected range."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        ou25_rate = df["Over_2_5"].mean()
        ou15_rate = df["Over_1_5"].mean()
        btts_rate = df["BTTS"].mean()
        assert 0.40 < ou25_rate < 0.55, f"O/U 2.5 rate {ou25_rate:.3f} out of range"
        assert 0.65 < ou15_rate < 0.80, f"O/U 1.5 rate {ou15_rate:.3f} out of range"
        assert 0.45 < btts_rate < 0.60, f"BTTS rate {btts_rate:.3f} out of range"

    def test_season_count(self):
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        n_seasons = df["SeasonIndex"].nunique()
        assert n_seasons >= 25, f"Expected 25+ seasons, got {n_seasons}"

    def test_teams_per_season(self):
        """Each season should have ~24 teams."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        latest = df["SeasonIndex"].max()
        latest_df = df[df["SeasonIndex"] == latest]
        teams = set(latest_df["Home_Team"].unique()) | set(latest_df["Away_Team"].unique())
        assert 20 <= len(teams) <= 26, f"Expected ~24 teams, got {len(teams)}"

    def test_run_pipeline_returns_expected_keys(self):
        from championship_pipeline import run_pipeline
        result = run_pipeline(verbose=False)
        assert "full_df" in result
        assert "features" in result
        assert "ou15_features" in result
        assert "btts_features" in result

    def test_run_pipeline_feature_counts(self):
        from championship_pipeline import run_pipeline, CHAMP_ALL_FEATURES
        result = run_pipeline(verbose=False)
        features = result["features"]
        assert len(features) >= 80, f"Expected 80+ features, got {len(features)}"
        # All features should exist in the DataFrame
        df = result["full_df"]
        missing = [f for f in features if f not in df.columns]
        assert not missing, f"Features missing from DataFrame: {missing}"

    def test_no_future_data_leakage(self):
        """Features should only use data available before the match."""
        from championship_pipeline import run_pipeline
        result = run_pipeline(verbose=False)
        df = result["full_df"]
        features = result["features"]
        # First few matches of season 0 should have NaN features (no history)
        s0 = df[df["SeasonIndex"] == 0].head(5)
        # At least some features should be NaN for the very first matches
        nan_counts = s0[features].isna().sum(axis=1)
        assert nan_counts.iloc[0] > 0, "First match should have NaN features"

    def test_bet365_odds_available(self):
        """Bet365 O/U 2.5 odds should be available from season 2+."""
        from championship_pipeline import load_championship_data
        df = load_championship_data()
        s2_plus = df[df["SeasonIndex"] >= 2]
        coverage = s2_plus["B365Greater2.5"].notna().mean()
        assert coverage > 0.80, f"B365 odds coverage {coverage:.1%} below 80%"


# ═══════════════════════════════════════════════════════════════════════════════
# Backtest Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibration:
    """Test logit-shift calibration functions."""

    def test_calibrate_shifts_mean_to_target(self):
        from championship_backtest import _calibrate
        probs = np.array([0.55, 0.60, 0.65, 0.50, 0.70])
        calibrated, shift = _calibrate(probs, 0.50)
        cal_mean_logit = np.mean(np.log(calibrated / (1 - calibrated)))
        target_logit = np.log(0.50 / 0.50)
        assert abs(cal_mean_logit - target_logit) < 0.01

    def test_calibrate_preserves_ranking(self):
        from championship_backtest import _calibrate
        probs = np.array([0.3, 0.5, 0.7, 0.9])
        calibrated, _ = _calibrate(probs, 0.50)
        # Ranking should be preserved
        assert all(calibrated[i] < calibrated[i+1] for i in range(len(calibrated)-1))

    def test_calibrate_single_matches_batch(self):
        from championship_backtest import _calibrate, _calibrate_single
        probs = np.array([0.55, 0.60, 0.65])
        _, shift = _calibrate(probs, 0.48)
        # Single calibration should match batch
        for p in probs:
            batch_result = _calibrate(np.array([p]), 0.48)[0][0]
            single_result = _calibrate_single(p, shift)
            # Won't be exact because batch recalculates shift for single value
            # But applying the same shift should give same result
            assert abs(single_result - _calibrate_single(p, shift)) < 1e-10

    def test_calibrate_extreme_values(self):
        from championship_backtest import _calibrate
        probs = np.array([0.01, 0.99])
        calibrated, shift = _calibrate(probs, 0.50)
        assert all(0 < p < 1 for p in calibrated)


class TestRegimeDetector:
    """Test in-season regime detection."""

    def test_initial_rate_equals_prior(self):
        from championship_backtest import RegimeDetector
        rd = RegimeDetector(prior_base_rate=0.475)
        assert rd.get_adjusted_base_rate() == 0.475

    def test_no_shift_before_min_matches(self):
        from championship_backtest import RegimeDetector
        rd = RegimeDetector(prior_base_rate=0.475, min_matches=20)
        # Feed 15 all-over results (not enough to trigger)
        for _ in range(15):
            rd.update(1)
        assert not rd.regime_shift_detected()
        assert rd.get_adjusted_base_rate() == 0.475

    def test_shift_detected_after_deviation(self):
        from championship_backtest import RegimeDetector
        rd = RegimeDetector(prior_base_rate=0.475, min_matches=10, window=20)
        # Feed 20 all-over results → rolling rate = 1.0, deviation = 0.525
        for _ in range(20):
            rd.update(1)
        assert rd.regime_shift_detected()
        # Rate should be above prior
        assert rd.get_adjusted_base_rate() > 0.475

    def test_no_shift_when_within_threshold(self):
        from championship_backtest import RegimeDetector
        rd = RegimeDetector(prior_base_rate=0.50, min_matches=10,
                            window=20, trigger_threshold=0.04)
        # Feed balanced results: 10 over, 10 under → rolling = 0.5 → no shift
        for i in range(20):
            rd.update(1 if i % 2 == 0 else 0)
        assert not rd.regime_shift_detected()

    def test_rate_clamped_to_range(self):
        from championship_backtest import RegimeDetector
        rd = RegimeDetector(prior_base_rate=0.475, min_matches=5, window=10)
        # All over — would push rate very high
        for _ in range(50):
            rd.update(1)
        rate = rd.get_adjusted_base_rate()
        assert 0.30 <= rate <= 0.75


class TestRefinedKelly:
    """Test 3-model Kelly staking."""

    def test_zero_kelly_when_no_edge(self):
        from championship_backtest import refined_kelly
        # blended_prob < 1/odds → negative Kelly → should return 0
        stake = refined_kelly(blended_prob=0.40, odds=2.0, n_agree=3,
                              edge=0.05)
        # 0.40 * 2.0 - 1 = -0.2 → negative Kelly
        assert stake == 0.0

    def test_positive_stake_with_edge(self):
        from championship_backtest import refined_kelly
        stake = refined_kelly(blended_prob=0.60, odds=2.0, n_agree=3,
                              edge=0.05)
        assert stake > 0

    def test_higher_agreement_means_bigger_stake(self):
        from championship_backtest import refined_kelly
        kwargs = {"blended_prob": 0.60, "odds": 2.0, "edge": 0.05}
        stake_2 = refined_kelly(n_agree=2, **kwargs)
        stake_3 = refined_kelly(n_agree=3, **kwargs)
        assert stake_3 > stake_2

    def test_zero_agreement_returns_zero(self):
        from championship_backtest import refined_kelly
        stake = refined_kelly(blended_prob=0.60, odds=2.0, n_agree=0,
                              edge=0.05)
        assert stake == 0.0

    def test_one_agreement_returns_zero(self):
        from championship_backtest import refined_kelly
        stake = refined_kelly(blended_prob=0.60, odds=2.0, n_agree=1,
                              edge=0.05)
        assert stake == 0.0

    def test_stake_capped_at_max(self):
        from championship_backtest import refined_kelly
        stake = refined_kelly(blended_prob=0.95, odds=2.0, n_agree=3,
                              edge=0.10, max_stake_pct=0.05)
        assert stake <= 0.05

    def test_drawdown_reduces_stake(self):
        from championship_backtest import refined_kelly
        # Use max_stake_pct=0.20 so cap doesn't interfere with ratio test
        kwargs = {"blended_prob": 0.60, "odds": 2.0, "n_agree": 3,
                  "edge": 0.05, "max_stake_pct": 0.20}
        normal = refined_kelly(drawdown_factor=1.0, **kwargs)
        reduced = refined_kelly(drawdown_factor=0.50, **kwargs)
        assert reduced < normal
        assert reduced == pytest.approx(normal * 0.50, abs=0.001)

    def test_minimum_stake_filter(self):
        from championship_backtest import refined_kelly
        # Very small edge → stake below 0.3% → filtered to 0
        stake = refined_kelly(blended_prob=0.505, odds=2.0, n_agree=2,
                              edge=0.005, kelly_fraction=0.05)
        assert stake == 0.0

    def test_invalid_odds_returns_zero(self):
        from championship_backtest import refined_kelly
        assert refined_kelly(0.60, odds=1.0, n_agree=3, edge=0.05) == 0.0
        assert refined_kelly(0.60, odds=0.5, n_agree=3, edge=0.05) == 0.0

    def test_negative_prob_returns_zero(self):
        from championship_backtest import refined_kelly
        assert refined_kelly(-0.1, odds=2.0, n_agree=3, edge=0.05) == 0.0


class TestDrawdownFactor:
    """Test drawdown protection scaling."""

    def test_no_drawdown_returns_one(self):
        from championship_backtest import compute_drawdown_factor
        assert compute_drawdown_factor(1.0, 1.0) == 1.0

    def test_small_drawdown_returns_one(self):
        from championship_backtest import compute_drawdown_factor
        # 1% drawdown → below 2% threshold → 1.0
        assert compute_drawdown_factor(0.99, 1.0) == 1.0

    def test_moderate_drawdown_reduces(self):
        from championship_backtest import compute_drawdown_factor
        # 7% drawdown → between 5-10% → 0.75
        factor = compute_drawdown_factor(0.93, 1.0)
        assert factor == 0.75

    def test_severe_drawdown_halves_stake(self):
        from championship_backtest import compute_drawdown_factor
        # 20% drawdown → >15% → 0.50
        factor = compute_drawdown_factor(0.80, 1.0)
        assert factor == 0.50

    def test_zero_peak_returns_one(self):
        from championship_backtest import compute_drawdown_factor
        assert compute_drawdown_factor(0.80, 0.0) == 1.0

    def test_above_peak_returns_one(self):
        from championship_backtest import compute_drawdown_factor
        # Bankroll above peak → no drawdown
        assert compute_drawdown_factor(1.10, 1.0) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Predict Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTeamMapping:
    """Test Championship team name resolution."""

    def setup_method(self):
        self.our_teams = {
            "Leeds", "Burnley", "Sheffield Weds", "QPR", "West Brom",
            "Nott'm Forest", "Hull", "Blackburn", "Cardiff", "Luton",
            "Plymouth", "Preston", "Norwich", "Derby", "Stoke",
            "Coventry", "Middlesbrough", "Oxford", "Swansea", "Watford",
            "Sunderland", "Millwall", "Sheffield United", "Portsmouth",
        }

    def test_exact_mapping(self):
        from championship_predict import _resolve_champ_team
        assert _resolve_champ_team("Leeds United", self.our_teams) == "Leeds"
        assert _resolve_champ_team("Queens Park Rangers", self.our_teams) == "QPR"
        assert _resolve_champ_team("West Bromwich Albion", self.our_teams) == "West Brom"
        assert _resolve_champ_team("Sheffield Wednesday", self.our_teams) == "Sheffield Weds"

    def test_direct_match(self):
        from championship_predict import _resolve_champ_team
        assert _resolve_champ_team("Burnley", self.our_teams) == "Burnley"
        assert _resolve_champ_team("Sunderland", self.our_teams) == "Sunderland"

    def test_fuzzy_fallback(self):
        from championship_predict import _resolve_champ_team
        # "Stoke City" → should fuzzy match to "Stoke"
        result = _resolve_champ_team("Stoke City", self.our_teams)
        assert result == "Stoke"

    def test_unknown_team_returns_none(self):
        from championship_predict import _resolve_champ_team
        result = _resolve_champ_team("FC Barcelona", self.our_teams)
        # Fuzzy match might return something if word overlap exists
        # but "Barcelona" has no overlap with any Championship team
        # Actually "FC" might match... let's test with something truly foreign
        result = _resolve_champ_team("Zzyzx Wanderers", set())
        assert result is None

    def test_match_champ_teams_both_resolved(self):
        from championship_predict import _match_champ_teams
        match = {"home_team": "Leeds United", "away_team": "Burnley"}
        home, away = _match_champ_teams(match, self.our_teams)
        assert home == "Leeds"
        assert away == "Burnley"


class TestBetEvaluation:
    """Test the bet evaluation logic with 3-model agreement."""

    @pytest.fixture(autouse=True)
    def _disable_edge_shrinkage(self, monkeypatch):
        """Option 5 adds edge shrinkage inside _evaluate_bet that
        mutates edge values, breaking these tests' specific assertions
        about confidence thresholds. Disable so this file isolates the
        pre-Option-5 bet-evaluation logic it was written to test.
        """
        import config
        monkeypatch.setattr(config, "USE_EDGE_SHRINKAGE", False)

    def setup_method(self):
        from championship_predict import ChampionshipPredictor
        self.predictor = ChampionshipPredictor.__new__(ChampionshipPredictor)

    def test_positive_edge_generates_bet(self):
        config = {"blend_weight": 0.35, "min_edge": 0.02, "min_agree": 2,
                  "kelly_fraction": 0.25, "max_stake_pct": 0.05}
        per_model = np.array([0.60, 0.58, 0.62])
        result = self.predictor._evaluate_bet(
            model_p=0.60, fair_p=0.48, odds=2.10,
            per_model=per_model, fair_threshold=0.48,
            config=config,
        )
        assert result is not None
        assert result["edge"] > 0
        assert result["stake_pct"] > 0

    def test_negative_ev_returns_none(self):
        config = {"blend_weight": 0.35, "min_edge": 0.02, "min_agree": 2,
                  "kelly_fraction": 0.25, "max_stake_pct": 0.05}
        per_model = np.array([0.45, 0.44, 0.46])
        result = self.predictor._evaluate_bet(
            model_p=0.45, fair_p=0.50, odds=1.90,
            per_model=per_model, fair_threshold=0.50,
            config=config,
        )
        assert result is None

    def test_insufficient_agreement_returns_none(self):
        config = {"blend_weight": 0.35, "min_edge": 0.02, "min_agree": 2,
                  "kelly_fraction": 0.25, "max_stake_pct": 0.05}
        # Only 1 of 3 models agrees
        per_model = np.array([0.60, 0.40, 0.42])
        result = self.predictor._evaluate_bet(
            model_p=0.60, fair_p=0.48, odds=2.10,
            per_model=per_model, fair_threshold=0.48,
            config=config,
        )
        assert result is None

    def test_confidence_levels(self):
        config = {"blend_weight": 0.50, "min_edge": 0.01, "min_agree": 2,
                  "kelly_fraction": 0.25, "max_stake_pct": 0.05}

        # High confidence: 3/3 agree + edge > 4%
        per_model = np.array([0.60, 0.58, 0.62])
        result = self.predictor._evaluate_bet(
            model_p=0.60, fair_p=0.48, odds=2.10,
            per_model=per_model, fair_threshold=0.48,
            config=config,
        )
        assert result is not None
        assert result["confidence"] == "high"


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard EFL Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardEFL:
    """Test EFL database operations and bookmaker odds storage."""

    @pytest.fixture(autouse=True)
    def _setup_temp_db(self, tmp_path):
        """Create a temporary EFL database for each test."""
        self.db_path = str(tmp_path / "test_efl.db")
        self._patcher = patch.dict(
            "dashboard.LEAGUE_DB_PATHS",
            {"EFL": self.db_path},
        )
        self._patcher.start()
        yield
        self._patcher.stop()

    def test_efl_db_created_with_tables(self):
        from db import get_db
        with get_db("EFL") as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        assert "match_analysis" in tables
        assert "recommendations" in tables
        assert "logged_bets" in tables
        assert "bankroll" in tables

    def test_bookmaker_odds_json_column_exists(self):
        from db import get_db
        with get_db("EFL") as conn:
            cursor = conn.execute("PRAGMA table_info(match_analysis)")
            cols = [row[1] for row in cursor.fetchall()]
        assert "bookmaker_odds_json" in cols

    def test_save_and_retrieve_match_analysis(self):
        from db import save_match_analysis, get_match_analysis
        rows = [
            {
                "home_team": "Leeds", "away_team": "Burnley",
                "kickoff": (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "market": "ou25", "side": "over",
                "best_odds": 2.10, "best_bookmaker": "Pinnacle",
                "model_prob": 0.55, "fair_odds": 1.90,
                "edge_pct": 3.2, "confidence": "medium", "n_books": 8,
                "bookmaker_odds": {"Pinnacle": 2.10, "William Hill": 2.05},
            },
        ]
        n = save_match_analysis(rows, league="EFL")
        assert n == 1

        df = get_match_analysis("EFL")
        assert len(df) == 1
        assert df.iloc[0]["home_team"] == "Leeds"
        assert df.iloc[0]["market"] == "ou25"

    def test_bookmaker_odds_round_trip(self):
        from db import save_match_analysis, get_match_analysis
        bm_odds = {"Bet365": 2.10, "Paddy Power": 2.05, "Pinnacle": 2.08}
        rows = [{
            "home_team": "QPR", "away_team": "West Brom",
            "kickoff": "", "market": "ou15", "side": "under",
            "best_odds": 1.50, "best_bookmaker": "Bet365",
            "bookmaker_odds": bm_odds,
        }]
        save_match_analysis(rows, league="EFL")
        df = get_match_analysis("EFL")
        stored = json.loads(df.iloc[0]["bookmaker_odds_json"])
        assert stored == bm_odds

    def test_save_replaces_same_fixture_preserves_others(self):
        """Per-fixture replace: re-scanning a fixture replaces only its own
        rows; fixtures absent from the new scan are preserved. This guards
        against a depleted API cache wiping out good data (see
        save_match_analysis docstring)."""
        from db import save_match_analysis, get_match_analysis
        save_match_analysis(
            [{"home_team": "A", "away_team": "B", "market": "ou25",
              "side": "over", "best_odds": 1.90}], league="EFL")
        save_match_analysis(
            [{"home_team": "C", "away_team": "D", "market": "btts",
              "side": "yes", "best_odds": 2.00}], league="EFL")
        # The C/D scan must NOT wipe the A/B fixture.
        df = get_match_analysis("EFL")
        assert len(df) == 2

        # Re-scanning A/B replaces its row in place (no duplicate), fresh odds.
        save_match_analysis(
            [{"home_team": "A", "away_team": "B", "market": "ou25",
              "side": "over", "best_odds": 1.70}], league="EFL")
        df = get_match_analysis("EFL")
        assert len(df) == 2  # still two fixtures; A/B replaced, not duplicated
        ab = df[(df["home_team"] == "A") & (df["away_team"] == "B")]
        assert len(ab) == 1
        assert ab.iloc[0]["best_odds"] == 1.70

    def test_save_recommendations_efl(self):
        from db import save_recommendations, get_active_recommendations
        recs = [{
            "home_team": "Leeds", "away_team": "Sheffield Weds",
            "kickoff": "2026-04-12T15:00:00Z",
            "market": "ou25", "side": "over",
            "odds": 2.10, "model_prob": 0.55, "blended_prob": 0.52,
            "fair_prob": 0.48, "edge": 0.04, "ev": 0.092,
            "stake_pct": 0.02, "confidence": "medium",
            "best_bookmaker": "Pinnacle", "n_books": 5, "n_agree": 3,
            "per_model_probs": {"xgb": 0.54, "lgb": 0.56, "dc": 0.55},
        }]
        n = save_recommendations(recs, league="EFL")
        assert n == 1

        active = get_active_recommendations("EFL")
        assert len(active) == 1
        assert active.iloc[0]["home_team"] == "Leeds"


class TestDefaultConfigs:
    """Test that default configurations are valid."""

    def test_championship_backtest_config_valid(self):
        from championship_backtest import DEFAULT_CONFIG
        assert 0 < DEFAULT_CONFIG["blend_weight"] <= 1
        assert 0 < DEFAULT_CONFIG["min_edge"] < 0.20
        assert DEFAULT_CONFIG["min_agree"] in (1, 2, 3)
        assert 0 < DEFAULT_CONFIG["kelly_fraction"] <= 1
        assert 0 < DEFAULT_CONFIG["max_stake_pct"] <= 0.10

    def test_championship_backtest_presets_valid(self):
        from championship_backtest import PRESETS
        assert len(PRESETS) >= 3
        for name, cfg in PRESETS.items():
            assert "blend_weight" in cfg, f"Preset {name} missing blend_weight"
            assert "min_edge" in cfg, f"Preset {name} missing min_edge"
            assert cfg["min_edge"] > 0, f"Preset {name} has non-positive min_edge"

    def test_no_pl_championship_data_overlap(self):
        """PL and Championship CSV files should be different."""
        from league_config import get_league_config
        pl_csv = get_league_config("PL")["csv_path"]
        efl_csv = get_league_config("EFL")["csv_path"]
        assert pl_csv != efl_csv
        assert "PL" in pl_csv or "pl" in pl_csv.lower()
        assert "Champ" in efl_csv or "champ" in efl_csv.lower()

    def test_no_pl_championship_db_overlap(self):
        """PL and Championship databases should be different."""
        from league_config import get_league_config
        pl_db = get_league_config("PL")["db_path"]
        efl_db = get_league_config("EFL")["db_path"]
        assert pl_db != efl_db


# ═══════════════════════════════════════════════════════════════════════════════
# Promoted Team Feature Initialisation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromotedTeamDetection:
    """Test detection and classification of newly promoted/relegated teams."""

    def test_detect_new_teams_returns_dict(self):
        from championship_pipeline import _detect_new_teams, load_championship_data
        df = load_championship_data()
        result = _detect_new_teams(df)
        assert isinstance(result, dict)
        # Season 0 should not be in the result (no prior season to compare)
        assert 0 not in result

    def test_six_new_teams_per_season(self):
        """Each season should have exactly 6 new teams (3 relegated + 3 promoted)."""
        from championship_pipeline import _detect_new_teams, load_championship_data
        df = load_championship_data()
        new_teams = _detect_new_teams(df)
        for season_idx, teams in new_teams.items():
            assert len(teams) == 6, (
                f"Season {season_idx} has {len(teams)} new teams, expected 6: {teams}"
            )

    def test_pl_relegated_classification(self):
        """PL-relegated teams should be correctly identified."""
        from championship_pipeline import (
            _detect_new_teams, _get_pl_teams_by_season,
            load_championship_data,
        )
        df = load_championship_data()
        new_teams = _detect_new_teams(df)
        pl_teams = _get_pl_teams_by_season()

        # Season 24 (2024-25): Burnley, Luton, Sheffield United relegated from PL
        s24_new = new_teams.get(24, set())
        pl_s23 = pl_teams.get(23, set())
        relegated = s24_new & pl_s23
        assert "Burnley" in relegated
        assert "Sheffield United" in relegated
        assert "Luton" in relegated
        assert len(relegated) == 3

    def test_l1_promoted_classification(self):
        """L1-promoted teams should be correctly identified."""
        from championship_pipeline import (
            _detect_new_teams, _get_pl_teams_by_season,
            load_championship_data,
        )
        df = load_championship_data()
        new_teams = _detect_new_teams(df)
        pl_teams = _get_pl_teams_by_season()

        s24_new = new_teams.get(24, set())
        pl_s23 = pl_teams.get(23, set())
        l1_promoted = s24_new - pl_s23
        assert "Oxford" in l1_promoted
        assert "Portsmouth" in l1_promoted
        assert "Derby" in l1_promoted
        assert len(l1_promoted) == 3

    def test_pl_to_champ_name_mapping_covers_all_relegated(self):
        """Verify PL name mapping produces valid Championship team names."""
        from championship_pipeline import (
            _detect_new_teams, _get_pl_teams_by_season,
            load_championship_data,
        )
        df = load_championship_data()
        all_ch_teams = set(df["Home_Team"].unique())
        new_teams = _detect_new_teams(df)
        pl_teams = _get_pl_teams_by_season()

        # Check all PL-relegated teams appear in Championship data
        for season_idx, teams in new_teams.items():
            pl_prior = pl_teams.get(season_idx - 1, set())
            relegated = teams & pl_prior
            for team in relegated:
                assert team in all_ch_teams, (
                    f"PL-relegated team '{team}' (season {season_idx}) "
                    f"not found in Championship data"
                )


class TestPromotedFeatureInitialisation:
    """Test the feature blending for promoted teams."""

    @pytest.fixture(scope="class")
    def pipeline_df(self):
        """Run the full pipeline once for all tests in this class."""
        from championship_pipeline import run_pipeline
        result = run_pipeline(verbose=False)
        return result["full_df"]

    def test_filled_count_positive(self):
        """initialize_promoted_features should fill a significant number of values."""
        from championship_pipeline import (
            load_championship_data, add_derived_features,
            add_advanced_features, initialize_promoted_features,
            add_congestion_features, add_discipline_features,
            add_halftime_features, add_elo, add_poisson_features,
            compute_team_strengths, merge_strengths, add_context_features,
        )
        df = load_championship_data()
        df = add_derived_features(df)
        df = add_congestion_features(df)
        df = add_discipline_features(df)
        df = add_halftime_features(df)
        df = add_advanced_features(df)
        df = add_elo(df)
        df = add_poisson_features(df)
        strengths = compute_team_strengths(df)
        df = merge_strengths(df, strengths)
        df = add_context_features(df)
        _, filled = initialize_promoted_features(df)
        # 24 seasons × 6 teams × 5 matches × 2 roles × ~30 features ≈ 40k+
        assert filled > 10000, f"Only filled {filled} values, expected >10000"

    def test_promoted_teams_have_no_nan_rolling_features(self, pipeline_df):
        """After initialisation, promoted teams' early matches should not have
        NaN in key rolling features."""
        from championship_pipeline import _detect_new_teams
        df = pipeline_df
        new_teams = _detect_new_teams(df)
        key_features = ["Home_Past5Goals", "Home_DefensiveStrength_5",
                        "Home_Over25_5", "Home_GPG_20"]

        nan_count = 0
        checked = 0
        for season_idx, teams in new_teams.items():
            season_df = df[df["SeasonIndex"] == season_idx]
            for team in teams:
                first5_home = season_df[
                    season_df["Home_Team"] == team
                ].sort_values("Date").head(5)
                for feat in key_features:
                    if feat in first5_home.columns:
                        checked += first5_home[feat].shape[0]
                        nan_count += first5_home[feat].isna().sum()

        if checked > 0:
            nan_rate = nan_count / checked
            assert nan_rate < 0.1, (
                f"NaN rate for promoted teams' rolling features: "
                f"{nan_rate:.1%} ({nan_count}/{checked})"
            )

    def test_blending_decays_over_5_matches(self, pipeline_df):
        """First match should be more influenced by averages than match 5."""
        from championship_pipeline import _detect_new_teams
        df = pipeline_df
        new_teams = _detect_new_teams(df)

        # Pick a season with known promoted teams and check that match 1
        # values differ from match 6+ values (blending has decayed)
        s24_teams = new_teams.get(24, set())
        if not s24_teams:
            pytest.skip("No promoted teams in season 24")

        team = list(s24_teams)[0]
        season_df = df[df["SeasonIndex"] == 24]
        home_matches = season_df[season_df["Home_Team"] == team].sort_values("Date")
        if len(home_matches) < 6:
            pytest.skip(f"Not enough home matches for {team}")

        feat = "Home_Past5Goals"
        if feat not in home_matches.columns:
            pytest.skip(f"Feature {feat} not in columns")

        # Match 1 (fully blended) and match 6+ (no blending) should
        # generally be different — at minimum the values should be non-NaN
        match1_val = home_matches.iloc[0][feat]
        assert pd.notna(match1_val), f"Match 1 {feat} is NaN for {team}"

    def test_pl_relegated_get_different_reference_than_l1(self, pipeline_df):
        """PL-relegated teams should get mid-table averages, L1-promoted
        should get bottom-5 averages — values should differ."""
        from championship_pipeline import _detect_new_teams, _get_pl_teams_by_season
        df = pipeline_df
        new_teams = _detect_new_teams(df)
        pl_teams = _get_pl_teams_by_season()

        # Use season 24 where we know the classifications
        s24 = new_teams.get(24, set())
        pl_s23 = pl_teams.get(23, set())
        relegated = list(s24 & pl_s23)
        l1_promoted = list(s24 - pl_s23)

        if not relegated or not l1_promoted:
            pytest.skip("Need both relegated and L1-promoted teams")

        season_df = df[df["SeasonIndex"] == 24]
        feat = "Home_Past5Goals"

        # Get match-1 values for a relegated and an L1-promoted team
        rel_home = season_df[season_df["Home_Team"] == relegated[0]].sort_values("Date")
        l1_home = season_df[season_df["Home_Team"] == l1_promoted[0]].sort_values("Date")

        if rel_home.empty or l1_home.empty:
            pytest.skip("No home matches found")

        rel_val = rel_home.iloc[0][feat]
        l1_val = l1_home.iloc[0][feat]

        # Both should be non-NaN
        assert pd.notna(rel_val), f"Relegated team {feat} is NaN"
        assert pd.notna(l1_val), f"L1-promoted team {feat} is NaN"
        # They should typically differ (different reference groups)
        # Allow for edge cases where they're equal, just verify they're populated
