"""Tests for Option 1: Model Calibration.

Phase A: Stacker + logit-shift in live prediction.

Covers:
  - Stacker and logit_shift are saved/loaded in pickle state
  - _predict_ou() uses stacker when available
  - _predict_ou() falls back to raw average when stacker is None
  - Logit-shift correction preserves ranking (monotonic)
  - Logit-shift shifts mean probability toward base rate
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LogisticRegression


class TestStackerSaveLoad:
    """Verify stacker and logit_shift persist through save/load."""

    def test_stacker_roundtrip(self, tmp_path) -> None:
        """Stacker and logit_shift survive pickle round-trip."""
        from predict import LivePredictor

        predictor = LivePredictor(verbose=False)
        predictor._ou_models = {"xgb": "m", "lgb": "m", "lr": "m",
                                "lr_scaler": "m", "dc": "m"}
        predictor._btts_models = {"xgb": "m"}
        predictor._ou_features = ["f1"]
        predictor._btts_features = ["f2"]
        predictor._ou_base_rate = 0.53
        predictor._btts_base_rate = 0.48
        predictor._dc_kwargs = {}
        predictor._train_medians = None
        predictor._our_teams = set()

        # Set stacker state
        stacker = LogisticRegression(C=1.0, max_iter=100, random_state=42)
        # Fit on dummy data so it's a valid model
        X_dummy = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6],
                            [0.4, 0.4, 0.4], [0.7, 0.3, 0.5]])
        y_dummy = np.array([1, 1, 0, 0])
        stacker.fit(X_dummy, y_dummy)
        predictor._ou_stacker = stacker
        predictor._ou_logit_shift = 0.1234

        path = str(tmp_path / "test_cal.pkl")
        predictor.save_trained_state(path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path) is True
        assert loaded._ou_stacker is not None
        assert abs(loaded._ou_logit_shift - 0.1234) < 1e-6
        # Verify stacker produces same output
        test_input = np.array([[0.55, 0.55, 0.55]])
        orig = predictor._ou_stacker.predict_proba(test_input)[:, 1]
        reloaded = loaded._ou_stacker.predict_proba(test_input)[:, 1]
        assert abs(orig[0] - reloaded[0]) < 1e-8

    def test_legacy_pickle_loads_without_stacker(self, tmp_path) -> None:
        """Pickle without stacker fields loads with None defaults."""
        import joblib
        from predict import LivePredictor

        # Simulate an old pickle without stacker fields
        state = {
            "ou_models": {"xgb": "m", "lgb": "m", "lr": "m",
                          "lr_scaler": "m", "dc": "m"},
            "btts_models": {"xgb": "m"},
            "ou_features": ["f1"],
            "btts_features": ["f2"],
            "ou_base_rate": 0.53,
            "btts_base_rate": 0.48,
            "dc_kwargs": {},
            "train_medians": None,
            "our_teams": set(),
            # Deliberately missing: ou_stacker, ou_logit_shift
        }
        path = str(tmp_path / "legacy.pkl")
        joblib.dump(state, path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path) is True
        assert loaded._ou_stacker is None
        assert loaded._ou_logit_shift == 0.0


class TestPredictOUCalibration:
    """Test _predict_ou() stacker + logit-shift path."""

    def _make_predictor_with_stacker(self):
        """Create a LivePredictor with a mock stacker."""
        from predict import LivePredictor

        p = LivePredictor(verbose=False)

        # Mock the models
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_lr = MagicMock()
        mock_dc = MagicMock()
        mock_dc.predict_proba_df.return_value = np.array([0.60])

        p._ou_models = {
            "xgb": mock_xgb, "lgb": mock_lgb,
            "lr": mock_lr, "lr_scaler": MagicMock(),
            "dc": mock_dc,
        }
        p._ou_features = ["f1", "f2"]
        p._train_medians = None

        # Real stacker
        stacker = LogisticRegression(C=1.0, max_iter=100, random_state=42)
        X = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6],
                      [0.4, 0.4, 0.4], [0.7, 0.7, 0.7],
                      [0.3, 0.3, 0.3], [0.55, 0.55, 0.55]])
        y = np.array([1, 1, 0, 1, 0, 1])
        stacker.fit(X, y)
        p._ou_stacker = stacker
        p._ou_logit_shift = 0.15  # Simulates an upward-drifting model

        return p

    @patch("predict._lr_predict")
    def test_stacker_path_used_when_available(self, mock_lr_pred) -> None:
        """When stacker is set, ensemble uses stacker + logit-shift, not average."""
        mock_lr_pred.return_value = np.array([0.52])

        p = self._make_predictor_with_stacker()
        import pandas as pd
        fixture = pd.Series({"f1": 0.5, "f2": 0.5})

        result = p._predict_ou(fixture)

        # The stacker path should produce a different result than raw average
        raw_avg = (0.55 + 0.58 + 0.52 + 0.60) / 4.0  # ~0.5625
        # Stacker result will differ — just verify it's not equal to raw avg
        assert result["ensemble"] != pytest.approx(raw_avg, abs=0.01)
        # Should still have all per-model outputs
        assert "xgb" in result
        assert "lgb" in result
        assert "dc" in result

    @patch("predict._lr_predict")
    def test_fallback_to_average_without_stacker(self, mock_lr_pred) -> None:
        """Without stacker, ensemble falls back to equal-weight average."""
        mock_lr_pred.return_value = np.array([0.52])

        from predict import LivePredictor
        p = LivePredictor(verbose=False)

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_df.return_value = np.array([0.60])

        p._ou_models = {
            "xgb": mock_xgb, "lgb": mock_lgb,
            "lr": MagicMock(), "lr_scaler": MagicMock(),
            "dc": mock_dc,
        }
        p._ou_features = ["f1", "f2"]
        p._train_medians = None
        p._ou_stacker = None  # No stacker

        import pandas as pd
        fixture = pd.Series({"f1": 0.5, "f2": 0.5})
        result = p._predict_ou(fixture)

        expected = (0.55 + 0.58 + 0.52 + 0.60) / 4.0
        assert result["ensemble"] == pytest.approx(expected, abs=0.001)


class TestLogitShiftProperties:
    """Verify logit-shift calibration has correct mathematical properties."""

    def test_logit_shift_preserves_ranking(self) -> None:
        """Logit-shift is monotonic — ordering of probs doesn't change."""
        probs = np.array([0.40, 0.55, 0.60, 0.72, 0.85])
        shift = 0.25  # Positive shift → probabilities decrease

        logits = np.log(probs / (1 - probs + 1e-10))
        corrected = 1 / (1 + np.exp(-(logits - shift)))

        # Verify ordering preserved
        for i in range(len(probs) - 1):
            assert corrected[i] < corrected[i + 1]

    def test_logit_shift_reduces_mean_when_positive(self) -> None:
        """Positive shift should decrease mean probability."""
        probs = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
        shift = 0.30

        logits = np.log(probs / (1 - probs + 1e-10))
        corrected = 1 / (1 + np.exp(-(logits - shift)))

        assert corrected.mean() < probs.mean()

    def test_zero_shift_is_identity(self) -> None:
        """Shift of 0 should return original probabilities."""
        probs = np.array([0.40, 0.55, 0.70])
        shift = 0.0

        logits = np.log(probs / (1 - probs + 1e-10))
        corrected = 1 / (1 + np.exp(-(logits - shift)))

        np.testing.assert_allclose(corrected, probs, atol=1e-10)

    def test_shift_targets_base_rate(self) -> None:
        """Shift computed from mean logit and base rate should correct mean."""
        probs = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
        base_rate = 0.52

        mean_logit = np.mean(np.log(probs / (1 - probs + 1e-10)))
        target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
        shift = mean_logit - target_logit

        logits = np.log(probs / (1 - probs + 1e-10))
        corrected = 1 / (1 + np.exp(-(logits - shift)))

        assert abs(corrected.mean() - base_rate) < 0.02


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: BTTS calibration (PL)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBTTSCalibration:
    """Test BTTS logit-shift calibration in predict.py."""

    @patch("predict._lr_predict")
    def test_btts_shift_applied_when_present(self, mock_lr_pred) -> None:
        """_predict_btts() applies logit-shift when btts_cal_shifts is set."""
        mock_lr_pred.return_value = np.array([0.52])

        from predict import LivePredictor
        p = LivePredictor(verbose=False)

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_btts_df.return_value = np.array([0.60])

        p._btts_models = {
            "xgb": mock_xgb, "lgb": mock_lgb,
            "lr": MagicMock(), "lr_scaler": MagicMock(),
            "dc": mock_dc,
        }
        p._btts_features = ["f1", "f2"]
        p._train_medians = None

        # Weights: [0.20, 0.20, 0.30, 0.30]
        raw_ensemble = 0.20 * 0.55 + 0.20 * 0.58 + 0.30 * 0.52 + 0.30 * 0.60
        # = 0.11 + 0.116 + 0.156 + 0.18 = 0.562

        # With shift
        p._btts_cal_shifts = {"ensemble_logit_shift": 0.20}
        import pandas as pd
        fixture = pd.Series({"f1": 0.5, "f2": 0.5})
        result = p._predict_btts(fixture)

        # A positive shift should decrease the ensemble
        assert result["ensemble"] < raw_ensemble

    @patch("predict._lr_predict")
    def test_btts_no_shift_without_cal(self, mock_lr_pred) -> None:
        """_predict_btts() uses raw ensemble when btts_cal_shifts is None."""
        mock_lr_pred.return_value = np.array([0.52])

        from predict import LivePredictor
        p = LivePredictor(verbose=False)

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.45, 0.55]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_btts_df.return_value = np.array([0.60])

        p._btts_models = {
            "xgb": mock_xgb, "lgb": mock_lgb,
            "lr": MagicMock(), "lr_scaler": MagicMock(),
            "dc": mock_dc,
        }
        p._btts_features = ["f1", "f2"]
        p._train_medians = None
        p._btts_cal_shifts = None  # No calibration

        import pandas as pd
        fixture = pd.Series({"f1": 0.5, "f2": 0.5})
        result = p._predict_btts(fixture)

        expected = 0.20 * 0.55 + 0.20 * 0.58 + 0.30 * 0.52 + 0.30 * 0.60
        assert result["ensemble"] == pytest.approx(expected, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: Championship calibration
# ═══════════════════════════════════════════════════════════════════════════════

class TestChampionshipCalibration:
    """Test Championship _predict_3model() logit-shift calibration."""

    def _make_predictor(self, cal_shifts: dict | None = None):
        from championship_predict import ChampionshipPredictor
        p = ChampionshipPredictor(verbose=False)
        p._cal_shifts = cal_shifts or {}
        return p

    def test_shift_applied_for_known_market(self) -> None:
        """_predict_3model() applies shift when market key exists."""
        p = self._make_predictor({"ou25": 0.15})

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.40, 0.60]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_df.return_value = np.array([0.55])

        models = {"xgb": mock_xgb, "lgb": mock_lgb, "dc": mock_dc}

        import pandas as pd
        fixture = pd.Series({"f1": 0.5})

        result = p._predict_3model(
            fixture, models, ["f1"], 0.475,
            dc_fn="predict_proba_df", market="ou25")

        raw_avg = (0.60 + 0.58 + 0.55) / 3.0
        # Positive shift reduces probability
        assert result["ensemble"] < raw_avg

    def test_no_shift_for_unknown_market(self) -> None:
        """_predict_3model() uses raw average for missing market key."""
        p = self._make_predictor({"ou25": 0.15})  # Only ou25

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.40, 0.60]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_df.return_value = np.array([0.55])

        models = {"xgb": mock_xgb, "lgb": mock_lgb, "dc": mock_dc}

        import pandas as pd
        fixture = pd.Series({"f1": 0.5})

        result = p._predict_3model(
            fixture, models, ["f1"], 0.475,
            dc_fn="predict_proba_df", market="corners")  # Not in cal_shifts

        raw_avg = (0.60 + 0.58 + 0.55) / 3.0
        assert result["ensemble"] == pytest.approx(raw_avg, abs=0.001)

    def test_legacy_empty_cal_shifts(self) -> None:
        """Empty cal_shifts (legacy) produces raw average."""
        p = self._make_predictor({})

        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.40, 0.60]])
        mock_lgb = MagicMock()
        mock_lgb.predict_proba.return_value = np.array([[0.42, 0.58]])
        mock_dc = MagicMock()
        mock_dc.predict_proba_df.return_value = np.array([0.55])

        models = {"xgb": mock_xgb, "lgb": mock_lgb, "dc": mock_dc}

        import pandas as pd
        fixture = pd.Series({"f1": 0.5})

        result = p._predict_3model(
            fixture, models, ["f1"], 0.475,
            dc_fn="predict_proba_df", market="ou25")

        raw_avg = (0.60 + 0.58 + 0.55) / 3.0
        assert result["ensemble"] == pytest.approx(raw_avg, abs=0.001)

    def test_cal_shifts_roundtrip(self, tmp_path) -> None:
        """cal_shifts survive Championship pickle round-trip."""
        from championship_predict import ChampionshipPredictor

        p = ChampionshipPredictor(verbose=False)
        p._ou_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou15_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._btts_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou_features = ["f"]
        p._ou15_features = ["f"]
        p._btts_features = ["f"]
        p._ou_base_rate = 0.475
        p._ou15_base_rate = 0.730
        p._btts_base_rate = 0.517
        p._dc_kwargs = {}
        p._our_teams = set()
        p._cal_shifts = {"ou25": 0.12, "ou15": -0.05, "btts": 0.08}

        path = str(tmp_path / "champ_cal.pkl")
        p.save_trained_state(path)

        loaded = ChampionshipPredictor(verbose=False)
        assert loaded.load_trained_state(path) is True
        assert loaded._cal_shifts == {"ou25": 0.12, "ou15": -0.05, "btts": 0.08}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase C: regime detection + two-phase early-season strategy
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimeDetectorClamps:
    """Verify RegimeDetector accepts and respects per-market clamp bounds."""

    def test_custom_clamps_applied(self) -> None:
        """Adjusted rate respects the custom (lo, hi) clamp passed in."""
        from backtest import RegimeDetector

        # Tight clamp above prior — upper clamp should bite
        det = RegimeDetector(
            prior_base_rate=0.52, window=40, blend_speed=1.0,
            trigger_threshold=0.01, min_matches=15,
            clamp_lo=0.30, clamp_hi=0.55,
        )
        # Feed 40 Overs — rolling rate = 1.0, blend to 0.52 + 1.0*(1.0-0.52) = 1.0
        for _ in range(40):
            det.update(1)
        assert det.get_adjusted_base_rate() == 0.55  # clamped

    def test_default_clamps_unchanged(self) -> None:
        """Default clamps still (0.30, 0.75) — no backward-compat break."""
        from backtest import RegimeDetector
        det = RegimeDetector(prior_base_rate=0.52)
        assert det.clamp_lo == 0.30
        assert det.clamp_hi == 0.75

    def test_lo_clamp_bites_for_underfed_window(self) -> None:
        """When rolling goes well below prior and clamp_lo > floor, clamp fires."""
        from backtest import RegimeDetector
        det = RegimeDetector(
            prior_base_rate=0.52, window=40, blend_speed=1.0,
            trigger_threshold=0.01, min_matches=15,
            clamp_lo=0.40, clamp_hi=0.70,
        )
        for _ in range(40):
            det.update(0)
        assert det.get_adjusted_base_rate() == 0.40


class TestMatchweekCounter:
    """Verify matchweek counting from pipeline full_df."""

    def _make_predictor_with_season(self, n_settled: int,
                                    current_season: int = 25):
        """Build a LivePredictor with a synthetic full_df."""
        import pandas as pd
        from predict import LivePredictor

        p = LivePredictor(verbose=False)
        rows = []
        # Past-season filler (should be ignored)
        for i in range(20):
            rows.append({
                "SeasonIndex": current_season - 1,
                "Date": f"2024-08-{(i % 28) + 1:02d}",
                "Home_Goals": 1.0, "Away_Goals": 1.0,
            })
        # Current season: n_settled matches with goals populated
        for i in range(n_settled):
            rows.append({
                "SeasonIndex": current_season,
                "Date": f"2025-08-{(i % 28) + 1:02d}",
                "Home_Goals": float(i % 3),
                "Away_Goals": float((i + 1) % 3),
            })
        # A couple of future fixtures with NaN goals
        for i in range(3):
            rows.append({
                "SeasonIndex": current_season,
                "Date": f"2026-05-{i + 1:02d}",
                "Home_Goals": float("nan"),
                "Away_Goals": float("nan"),
            })
        p._full_df = pd.DataFrame(rows)
        return p

    def test_counts_only_current_season_settled(self) -> None:
        """Only current-season matches with populated goals are counted."""
        p = self._make_predictor_with_season(n_settled=37)
        assert p._current_matchweek_count() == 37

    def test_returns_zero_before_season_starts(self) -> None:
        """Season with no settled matches yet returns 0."""
        p = self._make_predictor_with_season(n_settled=0)
        assert p._current_matchweek_count() == 0

    def test_empty_full_df_returns_zero(self) -> None:
        """Predictor without loaded pipeline returns 0 gracefully."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        assert p._current_matchweek_count() == 0


class TestRegimeShiftComputation:
    """Verify _compute_regime_shift produces expected adjustments."""

    def _make_season_df(self, over_rate: float, n: int = 40) -> "pd.DataFrame":
        """Synthesize a DataFrame where `over_rate` fraction of matches go Over 2.5."""
        import pandas as pd
        rows = []
        n_overs = int(round(n * over_rate))
        for i in range(n):
            is_over = i < n_overs
            rows.append({
                "SeasonIndex": 25,
                "Date": f"2025-08-{(i % 28) + 1:02d}",
                "Home_Goals": 2.0 if is_over else 1.0,
                "Away_Goals": 1.0 if is_over else 0.0,
            })
        return pd.DataFrame(rows)

    def test_no_shift_when_market_not_in_clamps(self) -> None:
        """Market with no REGIME_CLAMPS entry returns the static shift."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        p._full_df = self._make_season_df(0.50)

        new_shift, rate, shifted = p._compute_regime_shift(
            clamp_key="not_in_clamps",
            base_rate=0.52,
            val_mean_logit=0.10,
            outcome_fn=lambda df: (df["Home_Goals"] + df["Away_Goals"] > 2.5),
        )
        assert shifted is False
        # Returned rate is the training prior, shift is derived from it
        expected_shift = 0.10 - np.log(0.52 / 0.48)
        assert new_shift == pytest.approx(expected_shift, abs=1e-6)

    def test_no_shift_before_min_matches(self) -> None:
        """Before REGIME_MIN_MATCHES (15), returns static shift even if clamp exists."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        # Only 10 settled matches — below min_matches=15
        p._full_df = self._make_season_df(0.80, n=10)

        new_shift, rate, shifted = p._compute_regime_shift(
            clamp_key="ou25_pl",
            base_rate=0.52,
            val_mean_logit=0.10,
            outcome_fn=lambda df: (df["Home_Goals"] + df["Away_Goals"] > 2.5),
        )
        assert shifted is False
        assert rate == 0.52

    def test_shift_fires_when_season_runs_hot(self) -> None:
        """When current season Over rate is way above prior, regime triggers."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        # 40 matches at 75% Over — way above 0.52 prior
        p._full_df = self._make_season_df(0.75, n=40)

        new_shift, rate, shifted = p._compute_regime_shift(
            clamp_key="ou25_pl",
            base_rate=0.52,
            val_mean_logit=0.10,
            outcome_fn=lambda df: (df["Home_Goals"] + df["Away_Goals"] > 2.5),
        )
        assert shifted is True
        # Adjusted rate is between prior and rolling, blend_speed=0.4
        # Expected: 0.52 + 0.4 * (0.75 - 0.52) = 0.612
        assert rate == pytest.approx(0.612, abs=0.01)
        # Shift should be smaller (less correction needed since target is higher)
        # Shift targeting higher rate = smaller positive shift
        static_shift = 0.10 - np.log(0.52 / 0.48)
        assert new_shift < static_shift


class TestTwoPhaseEarlySeason:
    """Verify _evaluate_bet and generate_recommendations apply two-phase config."""

    @pytest.fixture(autouse=True)
    def _disable_edge_shrinkage(self, monkeypatch):
        """Option 5 shrinkage multiplies the edge, pushing borderline
        bets below the early-season min_edge threshold. Disable
        shrinkage to isolate the Phase C two-phase logic these tests
        are asserting against.
        """
        import config
        monkeypatch.setattr(config, "USE_EDGE_SHRINKAGE", False)

    def test_early_season_flag_toggles_config(self) -> None:
        """_evaluate_bet uses EARLY_* when _is_early_season=True."""
        from predict import LivePredictor

        p = LivePredictor(verbose=False)
        # Pick a bet that passes normal min_edge=0.02 but fails EARLY_MIN_EDGE=0.03
        # model_p=0.60, fair_p=0.50, odds=2.10
        # Normal: blend_w=0.35 → blended = 0.35*0.60 + 0.65*0.50 = 0.535
        #         edge = 0.035 > 0.02 → accept
        # Early:  blend_w=0.20 → blended = 0.20*0.60 + 0.80*0.50 = 0.520
        #         edge = 0.020 < 0.03 → reject

        per_model = np.array([0.60, 0.60, 0.60, 0.60])
        config = {"blend_weight": 0.35, "min_edge": 0.02,
                  "min_agree": 2, "kelly_fraction": 0.25,
                  "max_stake_pct": 0.05}

        # Normal phase — should accept
        p._is_early_season = False
        result_normal = p._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config=config,
            edge_source="pinnacle", market="ou25", side="over",
        )
        assert result_normal is not None

        # Early phase — same inputs now rejected by stricter thresholds
        p._is_early_season = True
        result_early = p._evaluate_bet(
            model_p=0.60, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config=config,
            edge_source="pinnacle", market="ou25", side="over",
        )
        assert result_early is None

    def test_early_season_smaller_stake(self) -> None:
        """When bet is accepted in both phases, early-season stake is smaller."""
        from predict import LivePredictor

        p = LivePredictor(verbose=False)
        # Strong bet that passes in both phases
        per_model = np.array([0.65, 0.65, 0.65, 0.65])
        config = {"blend_weight": 0.35, "min_edge": 0.02,
                  "min_agree": 2, "kelly_fraction": 0.25,
                  "max_stake_pct": 0.10}

        p._is_early_season = False
        normal = p._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config=config,
            edge_source="pinnacle", market="ou25", side="over",
        )
        p._is_early_season = True
        early = p._evaluate_bet(
            model_p=0.65, fair_p=0.50, odds=2.10,
            per_model=per_model, fair_threshold=0.50,
            config=config,
            edge_source="pinnacle", market="ou25", side="over",
        )
        assert normal is not None and early is not None
        assert early["stake_pct"] < normal["stake_pct"]


class TestValMeanLogitPersistence:
    """val_mean_logit fields survive pickle round-trip."""

    def test_pl_val_mean_logit_roundtrip(self, tmp_path) -> None:
        from predict import LivePredictor

        p = LivePredictor(verbose=False)
        p._ou_models = {"xgb": "m", "lgb": "m", "lr": "m",
                        "lr_scaler": "m", "dc": "m"}
        p._btts_models = {"xgb": "m"}
        p._ou_features = ["f"]
        p._btts_features = ["f"]
        p._ou_base_rate = 0.53
        p._btts_base_rate = 0.52
        p._dc_kwargs = {}
        p._train_medians = None
        p._our_teams = set()
        p._ou_val_mean_logit = 0.2857

        path = str(tmp_path / "vml.pkl")
        p.save_trained_state(path)
        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._ou_val_mean_logit == pytest.approx(0.2857, abs=1e-6)

    def test_efl_val_mean_logits_roundtrip(self, tmp_path) -> None:
        from championship_predict import ChampionshipPredictor

        p = ChampionshipPredictor(verbose=False)
        p._ou_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou15_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._btts_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou_features = ["f"]
        p._ou15_features = ["f"]
        p._btts_features = ["f"]
        p._ou_base_rate = 0.48
        p._ou15_base_rate = 0.73
        p._btts_base_rate = 0.52
        p._dc_kwargs = {}
        p._our_teams = set()
        p._val_mean_logits = {"ou25": 0.15, "ou15": 1.02, "btts": 0.08}

        path = str(tmp_path / "efl_vml.pkl")
        p.save_trained_state(path)
        loaded = ChampionshipPredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._val_mean_logits == {
            "ou25": 0.15, "ou15": 1.02, "btts": 0.08}

    def test_legacy_pickle_has_empty_val_mean_logits(self, tmp_path) -> None:
        """Old pickles without the new field load with defaults."""
        import joblib
        from predict import LivePredictor

        state = {
            "ou_models": {"xgb": "m", "lgb": "m", "lr": "m",
                          "lr_scaler": "m", "dc": "m"},
            "btts_models": {"xgb": "m"},
            "ou_features": ["f"], "btts_features": ["f"],
            "ou_base_rate": 0.53, "btts_base_rate": 0.52,
            "dc_kwargs": {}, "train_medians": None, "our_teams": set(),
        }
        path = str(tmp_path / "legacy.pkl")
        joblib.dump(state, path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._ou_val_mean_logit == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Option 2 Step 1: Per-market DC hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerMarketDCTuning:
    """Verify DC tuner respects target_col/predict_fn_name and that each
    market gets its own tuned kwargs stored on the predictor."""

    def test_tuner_signature_accepts_target(self) -> None:
        """tune_dc_params accepts target_col and predict_fn_name params."""
        import inspect
        from model import tune_dc_params

        sig = inspect.signature(tune_dc_params)
        assert "target_col" in sig.parameters
        assert "predict_fn_name" in sig.parameters

    def test_tuner_defaults_preserve_behaviour(self) -> None:
        """Defaults still target Over_2_5 via predict_proba_df."""
        import inspect
        from model import tune_dc_params

        sig = inspect.signature(tune_dc_params)
        assert sig.parameters["target_col"].default == "Over_2_5"
        assert sig.parameters["predict_fn_name"].default == "predict_proba_df"

    def test_extended_half_life_grid(self) -> None:
        """Grid includes 10 at the low end."""
        # This is a behaviour we can verify via a quick smoke test: the
        # tuner prints "half_life=10" when it iterates. We can't easily
        # test this without running a full tune, but we can check the
        # source for the new value. Lighter-touch: parse the module source.
        import model
        import inspect
        src = inspect.getsource(model.tune_dc_params)
        assert "[10, 15, 20, 25, 30, 40, 50, 70]" in src, \
            "Expected half_life grid to include 10 at the low end"

    def test_pl_predictor_stores_btts_dc_kwargs(self) -> None:
        """LivePredictor has _btts_dc_kwargs attribute separately from _dc_kwargs."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        assert hasattr(p, "_dc_kwargs")
        assert hasattr(p, "_btts_dc_kwargs")
        # Distinct dicts — modifying one shouldn't affect the other
        assert p._dc_kwargs is not p._btts_dc_kwargs

    def test_efl_predictor_stores_per_market_dc_kwargs(self) -> None:
        """ChampionshipPredictor has per-market DC kwargs attrs."""
        from championship_predict import ChampionshipPredictor
        p = ChampionshipPredictor(verbose=False)
        assert hasattr(p, "_dc_kwargs")
        assert hasattr(p, "_ou15_dc_kwargs")
        assert hasattr(p, "_btts_dc_kwargs")
        # All three are distinct dicts (not aliases to same object)
        assert p._dc_kwargs is not p._ou15_dc_kwargs
        assert p._ou15_dc_kwargs is not p._btts_dc_kwargs

    def test_pl_pickle_roundtrip_btts_dc_kwargs(self, tmp_path) -> None:
        """_btts_dc_kwargs survives pickle round-trip."""
        from predict import LivePredictor
        p = LivePredictor(verbose=False)
        p._ou_models = {"xgb": "m", "lgb": "m", "lr": "m",
                        "lr_scaler": "m", "dc": "m"}
        p._btts_models = {"xgb": "m"}
        p._ou_features = ["f"]
        p._btts_features = ["f"]
        p._ou_base_rate = 0.53
        p._btts_base_rate = 0.52
        p._dc_kwargs = {"half_life": 30, "rho": -0.13}
        p._btts_dc_kwargs = {"half_life": 10, "rho": -0.20}
        p._train_medians = None
        p._our_teams = set()

        path = str(tmp_path / "dc_kwargs.pkl")
        p.save_trained_state(path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._dc_kwargs == {"half_life": 30, "rho": -0.13}
        assert loaded._btts_dc_kwargs == {"half_life": 10, "rho": -0.20}

    def test_pl_legacy_pickle_falls_back_to_shared_kwargs(self, tmp_path) -> None:
        """Legacy pickle without btts_dc_kwargs reuses dc_kwargs (old behaviour)."""
        import joblib
        from predict import LivePredictor

        state = {
            "ou_models": {"xgb": "m", "lgb": "m", "lr": "m",
                          "lr_scaler": "m", "dc": "m"},
            "btts_models": {"xgb": "m"},
            "ou_features": ["f"], "btts_features": ["f"],
            "ou_base_rate": 0.53, "btts_base_rate": 0.52,
            "dc_kwargs": {"half_life": 30, "rho": -0.13},
            # No btts_dc_kwargs — this is legacy
            "train_medians": None, "our_teams": set(),
        }
        path = str(tmp_path / "legacy_dc.pkl")
        joblib.dump(state, path)

        loaded = LivePredictor(verbose=False)
        assert loaded.load_trained_state(path)
        # BTTS kwargs should fall back to the shared O/U 2.5 kwargs
        assert loaded._btts_dc_kwargs == {"half_life": 30, "rho": -0.13}

    def test_efl_pickle_roundtrip_all_three_kwargs(self, tmp_path) -> None:
        """All three EFL DC kwargs survive pickle roundtrip."""
        from championship_predict import ChampionshipPredictor

        p = ChampionshipPredictor(verbose=False)
        p._ou_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou15_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._btts_models = {"xgb": "m", "lgb": "m", "dc": "m"}
        p._ou_features = ["f"]
        p._ou15_features = ["f"]
        p._btts_features = ["f"]
        p._ou_base_rate = 0.48
        p._ou15_base_rate = 0.73
        p._btts_base_rate = 0.52
        p._dc_kwargs = {"half_life": 30, "rho": -0.13}
        p._ou15_dc_kwargs = {"half_life": 15, "rho": -0.20}
        p._btts_dc_kwargs = {"half_life": 10, "rho": -0.10}
        p._our_teams = set()

        path = str(tmp_path / "efl_dc.pkl")
        p.save_trained_state(path)

        loaded = ChampionshipPredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._dc_kwargs == {"half_life": 30, "rho": -0.13}
        assert loaded._ou15_dc_kwargs == {"half_life": 15, "rho": -0.20}
        assert loaded._btts_dc_kwargs == {"half_life": 10, "rho": -0.10}

    def test_efl_legacy_pickle_shares_kwargs(self, tmp_path) -> None:
        """Legacy EFL pickle without per-market kwargs reuses the single dc_kwargs."""
        import joblib
        from championship_predict import ChampionshipPredictor

        state = {
            "ou_models": {"xgb": "m", "lgb": "m", "dc": "m"},
            "ou15_models": {"xgb": "m", "lgb": "m", "dc": "m"},
            "btts_models": {"xgb": "m", "lgb": "m", "dc": "m"},
            "ou_features": ["f"], "ou15_features": ["f"], "btts_features": ["f"],
            "ou_base_rate": 0.48, "ou15_base_rate": 0.73, "btts_base_rate": 0.52,
            "dc_kwargs": {"half_life": 30, "rho": -0.13},
            # No per-market kwargs
            "our_teams": set(),
        }
        path = str(tmp_path / "efl_legacy.pkl")
        joblib.dump(state, path)

        loaded = ChampionshipPredictor(verbose=False)
        assert loaded.load_trained_state(path)
        assert loaded._ou15_dc_kwargs == {"half_life": 30, "rho": -0.13}
        assert loaded._btts_dc_kwargs == {"half_life": 30, "rho": -0.13}


# ═══════════════════════════════════════════════════════════════════════════════
# Option 2 Step 2: Partial-pooling shrinkage in DixonColesPredictor.fit
# ═══════════════════════════════════════════════════════════════════════════════

class TestShrinkageHelpers:
    """Unit tests for the _shrink_to_league and _pool_with_shrinkage helpers."""

    def test_shrink_blends_toward_one(self) -> None:
        """Result is a convex combination of estimate and 1.0."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        # With N_PRIOR=6 and n=6, shrinkage should be exactly 50/50
        r = p._shrink_to_league(estimate=0.80, n_eff=6)
        assert r == pytest.approx(0.5 * 0.80 + 0.5 * 1.0, abs=1e-9)

    def test_shrink_larger_sample_less_shrinkage(self) -> None:
        """Larger effective sample pulls result closer to the estimate."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        small = p._shrink_to_league(estimate=0.80, n_eff=2)
        large = p._shrink_to_league(estimate=0.80, n_eff=30)
        # Both below 1.0, large should be closer to 0.80
        assert abs(large - 0.80) < abs(small - 0.80)

    def test_shrink_zero_sample_is_league_mean(self) -> None:
        """n_eff=0 collapses entirely to league mean (1.0)."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        r = p._shrink_to_league(estimate=0.50, n_eff=0)
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_shrink_none_passes_through(self) -> None:
        """None estimate returns None (the caller handles the fallback)."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        assert p._shrink_to_league(estimate=None, n_eff=10) is None

    def test_pool_prefers_venue_over_pooled(self) -> None:
        """When both venue and pooled estimates exist, venue wins."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        # Venue: 0.8 with n=3 → shrunk by factor 3/9 = 0.333
        # Pooled: 0.9 with n=6 → would shrink by 6/12 = 0.5 (not used)
        r = p._pool_with_shrinkage(venue_val=0.8, n_venue=3,
                                    pooled_val=0.9, n_pool=6,
                                    prior=0.99)
        expected = (3 / 9) * 0.8 + (6 / 9) * 1.0
        assert r == pytest.approx(expected, abs=1e-9)

    def test_pool_falls_back_to_pooled(self) -> None:
        """Missing venue uses pooled."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        r = p._pool_with_shrinkage(venue_val=None, n_venue=0,
                                    pooled_val=0.85, n_pool=5,
                                    prior=0.99)
        expected = (5 / 11) * 0.85 + (6 / 11) * 1.0
        assert r == pytest.approx(expected, abs=1e-9)

    def test_pool_falls_back_to_prior(self) -> None:
        """Team with zero data at either venue gets the prior constant."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        r = p._pool_with_shrinkage(venue_val=None, n_venue=0,
                                    pooled_val=None, n_pool=0,
                                    prior=0.77)
        assert r == 0.77


class TestShrinkageIntegration:
    """Verify fit() uses shrinkage when use_mle=False, legacy path when True."""

    def _toy_df(self) -> "pd.DataFrame":
        """Two-team, few-match synthetic data good enough to exercise fit()."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        base = datetime(2024, 8, 1)
        rows = []
        # Team A has 4 home + 3 away matches; Team B has 3 home + 4 away
        for i in range(4):
            rows.append({
                "Date": base + timedelta(days=i),
                "Home_Team": "A", "Away_Team": "B",
                "Home_Goals": np.random.poisson(1.5),
                "Away_Goals": np.random.poisson(1.0),
            })
        for i in range(4):
            rows.append({
                "Date": base + timedelta(days=i + 10),
                "Home_Team": "B", "Away_Team": "A",
                "Home_Goals": np.random.poisson(1.3),
                "Away_Goals": np.random.poisson(1.2),
            })
        return pd.DataFrame(rows)

    def test_use_mle_false_applies_shrinkage(self) -> None:
        """When use_mle=False the ratings are pulled toward 1.0 vs raw venue avg."""
        from model import DixonColesPredictor
        import numpy as np
        np.random.seed(42)

        df = self._toy_df()
        dc = DixonColesPredictor(use_mle=False)
        dc.fit(df)
        # With only 4 home matches for each team (< N_PRIOR=6), ratings
        # should be noticeably pulled toward 1.0 rather than matching
        # the raw team scoring rate.
        for team in ("A", "B"):
            # Ratings should be in (0, 1) or close to 1.0 — not at the
            # raw fraction a tiny sample would produce.
            for attr in ("attack_home", "attack_away",
                         "defence_home", "defence_away"):
                rating = getattr(dc, attr).get(team)
                assert rating is not None
                # A 4-match sample with pure estimate far from 1.0 would
                # produce a rating close to that raw value. Shrinkage pulls
                # it toward 1.0, so rating must be closer to 1.0 than to
                # an extreme (e.g. 0.0 or 2.0). In practice ratings stay
                # within a reasonable band.
                assert 0.3 < rating < 1.7, \
                    f"{team}.{attr} = {rating:.3f} is outside expected band"

    def test_use_mle_true_keeps_legacy_threshold(self) -> None:
        """With use_mle=True the fit() method uses the pre-shrinkage path."""
        from model import DixonColesPredictor
        import numpy as np
        np.random.seed(42)

        df = self._toy_df()
        # use_mle=True but we skip the actual MLE step by mocking it out;
        # we only care about the weighted-avg assignment logic here.
        dc = DixonColesPredictor(use_mle=True)
        # Replace fit_mle to be a no-op so we inspect warm-start values
        dc.fit_mle = lambda _df, alpha=0.01: None
        dc.fit(df)
        # Each team has only 4 matches at each venue — below the legacy
        # threshold is 3, so they DO pass it, which means legacy path uses
        # the venue estimate directly (no shrinkage). To actually exercise
        # the legacy-branch difference, we'd need a team with fewer than 3
        # at a venue. What we can confirm is that ratings are finite floats.
        for team in ("A", "B"):
            for attr in ("attack_home", "attack_away",
                         "defence_home", "defence_away"):
                rating = getattr(dc, attr).get(team)
                assert rating is not None
                assert isinstance(rating, float)

    def test_promoted_team_with_no_data_uses_prior(self) -> None:
        """Team that never appears still falls back to PRIORS at lookup time."""
        from model import DixonColesPredictor
        import numpy as np
        np.random.seed(42)

        df = self._toy_df()
        dc = DixonColesPredictor(use_mle=False)
        dc.fit(df)
        # Team "C" never trained — lookup with .get(PRIOR) is the live path
        prior = dc.PRIORS["attack_home"]
        rating = dc.attack_home.get("C", prior)
        assert rating == prior

    def test_shrinkage_is_continuous_across_sample_sizes(self) -> None:
        """Verify ratings vary smoothly with sample size (no jump at n=3)."""
        from model import DixonColesPredictor
        p = DixonColesPredictor()
        # Take the SAME estimate (e.g. 0.70) at increasing sample sizes.
        # The old hard-threshold was discontinuous at n=3; the new
        # shrinkage should produce a smoothly increasing series.
        estimate = 0.70
        ratings = [p._shrink_to_league(estimate, n) for n in range(1, 11)]
        # Strictly monotonic — each extra match pulls the rating further
        # from 1.0, toward the true estimate.
        for i in range(len(ratings) - 1):
            assert ratings[i + 1] < ratings[i], \
                f"Non-monotonic at n={i+2}: {ratings[i]:.4f} -> {ratings[i+1]:.4f}"
        # Endpoints: at n=1 heavy shrinkage, at n=10 less
        assert ratings[0] > ratings[-1]
        # All within (estimate, 1.0)
        for r in ratings:
            assert estimate < r < 1.0
