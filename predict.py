"""
Live prediction engine for O/U goals and BTTS markets.

Loads trained models, generates predictions for upcoming fixtures,
blends with live odds, and outputs bet recommendations.

Markets evaluated:
  - O/U 2.5 goals (4-model ensemble: XGB + LGB + LR + DC)
  - BTTS (4-model ensemble)
  - Alternative O/U lines (1.5, 3.5, 4.5, etc.) via DC Poisson + market blend
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Optional

from pipeline import run_pipeline
from config import (
    ALL_FEATURES, BTTS_ALL_FEATURES,
    ALLOWED_ALT_LINES, ALT_LINES_OVER_ONLY,
    DC_LAMBDA_MIN, DC_LAMBDA_MAX,
    MODEL_DIR,
    EARLY_SEASON_MATCHES, EARLY_BLEND_WEIGHT,
    EARLY_MIN_EDGE, EARLY_KELLY_FRACTION,
)
from predictor_utils import (
    save_pickle, load_pickle, compute_regime_shift, seasons_for_validation,
    refit_at_best_iteration,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from model import (
    train_xgb, train_lgb, train_logreg, train_xgb_btts, train_lgb_btts,
    DixonColesPredictor, tune_dc_params, _fill_nan_median,
    walk_forward_cv,
)
from backtest import (
    _calibrate, _calibrate_single, _lr_predict, DEFAULT_CONFIG,
)
from staking import decide_bet, apply_portfolio_constraints, PL_AGREE_SCALE
from btts_backtest import BTTS_DEFAULT_CONFIG
from api.odds_api import (
    fetch_epl_odds, get_best_odds, get_best_btts_odds, get_best_odds_all_lines,
    match_to_our_teams,
)
from api.oddspapi import (
    fetch_epl_all_odds as fetch_oddspapi_odds,
    map_team as oddspapi_map_team,
)
from alt_lines import scan_all_lines, get_value_bets

logger = logging.getLogger(__name__)


def _extract_bookmaker_odds(
    match: dict, market: str, side: str,
) -> dict[str, float]:
    """Extract per-bookmaker odds for a specific market+side from raw API data.

    Args:
        match: Raw match dict from odds API (has 'bookmakers', 'btts_bookmakers').
        market: Market code ('ou25', 'ou15', 'ou35', 'btts').
        side: Bet side ('over', 'under', 'yes', 'no').

    Returns:
        Dict mapping bookmaker display name → odds value.
    """
    book_odds: dict[str, float] = {}

    if market == "btts":
        side_key = "yes" if side == "yes" else "no"
        for bk, bm in match.get("btts_bookmakers", {}).items():
            title = bm.get("title", bk)
            val = bm.get(side_key)
            if val and val > 1:
                book_odds[title] = round(val, 2)
    else:
        line = {"ou15": 1.5, "ou35": 3.5, "ou45": 4.5}.get(market, 2.5)
        side_key = "over" if side == "over" else "under"
        for bk, bm in match.get("bookmakers", {}).items():
            title = bm.get("title", bk)
            all_lines = bm.get("all_lines", {})
            line_data = all_lines.get(line) or all_lines.get(str(line))
            if line_data:
                val = line_data.get(side_key)
                if val and val > 1:
                    book_odds[title] = round(val, 2)
            elif line == 2.5:
                val = bm.get(side_key)
                if val and val > 1:
                    book_odds[title] = round(val, 2)

    return book_odds


class LivePredictor:
    """Generates bet recommendations for upcoming fixtures using trained models.

    Trains on all available historical data, calibrates probabilities,
    and compares against live bookmaker odds.
    """

    def __init__(
        self,
        ou_config: Optional[dict] = None,
        btts_config: Optional[dict] = None,
        verbose: bool = True,
        use_sparse_features: Optional[bool] = None,
        snapshot_type: Optional[str] = None,
    ) -> None:
        self.ou_config = ou_config or DEFAULT_CONFIG.copy()
        self.btts_config = btts_config or BTTS_DEFAULT_CONFIG.copy()
        self.verbose = verbose
        # Option 3 Step 3: optional flag to exclude features in groups listed
        # in config.SPARSE_FEATURE_GROUPS. None = use global default
        # (USE_SPARSE_FEATURES from config.py). Exposed as constructor
        # parameter so diagnose_option3.py can A/B compare variants without
        # touching the global.
        from config import USE_SPARSE_FEATURES, DEFAULT_SNAPSHOT_TYPE
        self.use_sparse_features = (
            USE_SPARSE_FEATURES if use_sparse_features is None
            else use_sparse_features)
        # Option β-tight: gates OddsPapi fetch. "week_ahead" = full sweep,
        # "refresh" = Odds-API only. See config.DEFAULT_SNAPSHOT_TYPE notes.
        self.snapshot_type = (
            DEFAULT_SNAPSHOT_TYPE if snapshot_type is None
            else snapshot_type)

        self._pipeline_data: Optional[dict] = None
        self._full_df: Optional[pd.DataFrame] = None
        self._ou_features: list[str] = []
        self._btts_features: list[str] = []

        # Trained models (set by train())
        self._ou_models: Optional[dict] = None
        self._btts_models: Optional[dict] = None
        self._ou_base_rate: float = 0.5
        self._btts_base_rate: float = 0.5
        # _dc_kwargs = O/U 2.5 DC hyperparameters (primary, backward-compat name)
        # _btts_dc_kwargs = BTTS-specific DC hyperparameters (Option 2 Step 1)
        self._dc_kwargs: dict = {}
        self._btts_dc_kwargs: dict = {}
        self._train_medians: Optional[pd.Series] = None
        self._our_teams: set[str] = set()

        # Calibration (set by train())
        self._ou_stacker: Optional[LogisticRegression] = None
        self._ou_logit_shift: float = 0.0
        self._ou_val_mean_logit: float = 0.0  # Phase C: for regime recompute
        self._btts_cal_shifts: Optional[dict] = None  # Phase B

        # Phase C: two-phase early-season flag (set by generate_recommendations)
        self._is_early_season: bool = False

    # Default paths for trained state and pipeline cache
    _STATE_PATH = os.path.join(MODEL_DIR, "pl_trained_state.pkl")
    _PIPELINE_CACHE_PATH = os.path.join(MODEL_DIR, "pl_pipeline_cache.pkl")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")
        logger.info(msg)

    # ── Save / Load Infrastructure ──

    def save_trained_state(self, path: Optional[str] = None) -> None:
        """Save all trained models, features, calibration params to disk.

        Args:
            path: Override path for the pickle file.
        """
        state = {
            "ou_models": self._ou_models,
            "btts_models": self._btts_models,
            "ou_features": self._ou_features,
            "btts_features": self._btts_features,
            "ou_base_rate": self._ou_base_rate,
            "btts_base_rate": self._btts_base_rate,
            "dc_kwargs": self._dc_kwargs,
            "btts_dc_kwargs": self._btts_dc_kwargs,
            "train_medians": self._train_medians,
            "our_teams": self._our_teams,
            "ou_stacker": self._ou_stacker,
            "ou_logit_shift": self._ou_logit_shift,
            "ou_val_mean_logit": self._ou_val_mean_logit,
            "btts_cal_shifts": self._btts_cal_shifts,
        }
        save_pickle(state, path, self._STATE_PATH, self._log,
                     label="Trained state")

    def load_trained_state(self, path: Optional[str] = None) -> bool:
        """Load pre-trained models from disk. Skips full retraining.

        Args:
            path: Override path for the pickle file.

        Returns:
            True if loaded successfully, False if file missing.
        """
        state = load_pickle(path, self._STATE_PATH, self._log,
                            label="Trained state")
        if state is None:
            return False
        self._ou_models = state["ou_models"]
        self._btts_models = state["btts_models"]
        self._ou_features = state["ou_features"]
        self._btts_features = state["btts_features"]
        self._ou_base_rate = state["ou_base_rate"]
        self._btts_base_rate = state["btts_base_rate"]
        self._dc_kwargs = state["dc_kwargs"]
        # Backward compat: legacy pickles have no separate BTTS kwargs
        # — fall back to the shared dc_kwargs (prior behaviour).
        self._btts_dc_kwargs = state.get("btts_dc_kwargs", state["dc_kwargs"])
        self._train_medians = state["train_medians"]
        self._our_teams = state["our_teams"]
        # Calibration — backward compat: older pickles lack these
        self._ou_stacker = state.get("ou_stacker")
        self._ou_logit_shift = state.get("ou_logit_shift", 0.0)
        self._ou_val_mean_logit = state.get("ou_val_mean_logit", 0.0)
        self._btts_cal_shifts = state.get("btts_cal_shifts")
        if self._ou_stacker is not None:
            self._log(f"  Stacker loaded (logit_shift={self._ou_logit_shift:.4f})")
        else:
            self._log("  No stacker — using raw model average (legacy pickle)")
        return True

    def save_pipeline_cache(self, path: Optional[str] = None) -> None:
        """Cache the pipeline DataFrame to skip feature engineering on reload.

        Args:
            path: Override path for the pickle file.
        """
        save_pickle(
            {
                "full_df": self._full_df,
                "train_medians": self._train_medians,
                "ou_features": self._ou_features,
                "btts_features": self._btts_features,
                "our_teams": self._our_teams,
            },
            path, self._PIPELINE_CACHE_PATH, self._log,
            label="Pipeline cache",
        )

    def load_pipeline_cache(self, path: Optional[str] = None) -> bool:
        """Load cached pipeline DataFrame. Skips 60-120s feature engineering.

        Args:
            path: Override path for the pickle file.

        Returns:
            True if loaded successfully, False if file missing.
        """
        cache = load_pickle(path, self._PIPELINE_CACHE_PATH, self._log,
                            label="Pipeline cache")
        if cache is None:
            return False
        self._full_df = cache["full_df"]
        self._train_medians = cache["train_medians"]
        self._ou_features = cache["ou_features"]
        self._btts_features = cache["btts_features"]
        self._our_teams = cache["our_teams"]
        return True

    def light_refresh(self, markets: tuple[str, ...] = ("totals", "btts")) -> list[dict]:
        """Lightweight prediction: load pickles + fetch odds + recalculate edges.

        Skips training entirely. Falls back to full train if no pickle exists.

        Args:
            markets: Which odds markets to fetch ("totals", "btts").

        Returns:
            List of recommendation dicts.
        """
        if not self.load_trained_state():
            self._log("No trained state found — falling back to full train")
            if not self.load_pipeline_cache():
                self.load_data()
                self.save_pipeline_cache()
            self.train()
            self.save_trained_state()

        # Need pipeline data for fixture features even in light mode
        if self._full_df is None:
            if not self.load_pipeline_cache():
                self.load_data()
                self.save_pipeline_cache()

        return self.generate_recommendations(markets=markets)

    def load_data(self) -> None:
        """Load pipeline data and identify available features."""
        self._log("Loading pipeline data...")
        self._pipeline_data = run_pipeline(verbose=self.verbose)
        self._full_df = self._pipeline_data["full_df"]
        self._train_medians = self._pipeline_data.get("train_medians")

        # Option 3 Step 3: apply sparse-feature flag before filtering to
        # columns actually present in the DataFrame.
        from config import get_active_features
        active_ou = get_active_features(ALL_FEATURES, self.use_sparse_features)
        active_btts = get_active_features(
            BTTS_ALL_FEATURES, self.use_sparse_features)
        self._ou_features = [f for f in active_ou if f in self._full_df.columns]
        self._btts_features = [
            f for f in active_btts if f in self._full_df.columns]
        if not self.use_sparse_features:
            self._log(f"  Sparse-feature filter ACTIVE: "
                       f"O/U features reduced from {len(ALL_FEATURES)} "
                       f"to {len(self._ou_features)}")

        # Get current team names
        latest_season = self._full_df["SeasonIndex"].max()
        latest = self._full_df[self._full_df["SeasonIndex"] == latest_season]
        self._our_teams = set(latest["Home_Team"].unique()) | set(latest["Away_Team"].unique())

        self._log(f"O/U features: {len(self._ou_features)}")
        self._log(f"BTTS features: {len(self._btts_features)}")
        self._log(f"Current teams: {len(self._our_teams)}")

    def train(self) -> None:
        """Train all models on full historical data."""
        if self._full_df is None:
            self.load_data()

        df = self._full_df
        train_df = df[df["SeasonIndex"] >= 14].copy()

        # Materialise BTTS target column for the BTTS DC tuner.
        # pipeline.py computes BTTS in long-format only; the wide df
        # recomputes on the fly where needed.
        if "BTTS" not in train_df.columns:
            train_df["BTTS"] = (
                (train_df["Home_Goals"] > 0)
                & (train_df["Away_Goals"] > 0)
            ).astype(int)

        # ── Per-market Dixon-Coles tuning (Option 2 Step 1) ──
        # Each live market gets its own best (half_life, rho, MLE) combo
        # because goals-vs-BTTS-vs-low-score-tails respond to different
        # time horizons and low-score corrections.
        self._log("Tuning Dixon-Coles (O/U 2.5)...")
        self._dc_kwargs = tune_dc_params(
            train_df,
            target_col="Over_2_5",
            predict_fn_name="predict_proba_df",
        )
        self._log(f"  O/U 2.5 DC params: {self._dc_kwargs}")

        self._log("Tuning Dixon-Coles (BTTS)...")
        self._btts_dc_kwargs = tune_dc_params(
            train_df,
            target_col="BTTS",
            predict_fn_name="predict_proba_btts_df",
        )
        self._log(f"  BTTS DC params: {self._btts_dc_kwargs}")

        # Early-Stopping Season + Base Rate window (ADR 0009). Only seasons with
        # enough fixtures to judge a model on: previously this was every season
        # present, so a newly started season became the sole early-stopping set
        # on its first ingested fixture.
        train_seasons = seasons_for_validation(
            train_df["SeasonIndex"], log=self._log)
        last_season = train_seasons[-1]
        newest = int(train_df["SeasonIndex"].max())
        if last_season != newest:
            self._log(f"  Early-Stopping Season: S{last_season} "
                      f"(S{newest} has too few fixtures)")
        es_val_mask = train_df["SeasonIndex"] == last_season
        es_train_mask = ~es_val_mask

        # ── O/U 2.5 Models ──
        self._log("Training O/U 2.5 models...")
        X_train_ou = train_df[self._ou_features].values
        y_train_ou = train_df["Over_2_5"].values

        X_es_train = train_df.loc[es_train_mask, self._ou_features].values
        y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
        X_es_val = train_df.loc[es_val_mask, self._ou_features].values
        y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

        # Early-stop to choose the tree count, then refit on the full frame so
        # XGB/LGB see the Early-Stopping Season too — LogReg and DC below
        # already do (ADR 0009).
        ou_xgb = refit_at_best_iteration(
            train_xgb(X_es_train, y_es_train, X_es_val, y_es_val),
            X_train_ou, y_train_ou)
        ou_lgb = refit_at_best_iteration(
            train_lgb(X_es_train, y_es_train, X_es_val, y_es_val,
                      feature_names=self._ou_features),
            X_train_ou, y_train_ou, feature_names=self._ou_features)
        ou_lr, ou_lr_scaler = train_logreg(X_train_ou, y_train_ou)
        ou_dc = DixonColesPredictor(**self._dc_kwargs)
        ou_dc.fit(train_df)

        # Base rate from last 2 seasons
        recent_mask = train_df["SeasonIndex"].isin(train_seasons[-2:])
        self._ou_base_rate = train_df.loc[recent_mask, "Over_2_5"].mean()

        self._ou_models = {
            "xgb": ou_xgb, "lgb": ou_lgb,
            "lr": ou_lr, "lr_scaler": ou_lr_scaler,
            "dc": ou_dc,
        }
        self._log(f"O/U base rate: {self._ou_base_rate:.3f}")

        # ── O/U 2.5 Stacker + Calibration ──
        self._log("Training O/U 2.5 stacker (walk-forward OOF)...")
        _, oof_df = walk_forward_cv(
            train_df, self._ou_features,
            min_train_season=14, start_val_season=19,
            dc_kwargs=self._dc_kwargs,
        )

        if len(oof_df) > 100:
            oof_stack = oof_df[["xgb", "lgb", "lr", "dc"]].values
            oof_y = oof_df["y"].values

            self._ou_stacker = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42)
            self._ou_stacker.fit(oof_stack, oof_y)

            oof_auc = roc_auc_score(oof_y,
                                    self._ou_stacker.predict_proba(oof_stack)[:, 1])
            self._log(f"  Stacker OOF AUC: {oof_auc:.4f}")
            self._log(f"  Stacker coefs: XGB={self._ou_stacker.coef_[0][0]:.3f}, "
                       f"LGB={self._ou_stacker.coef_[0][1]:.3f}, "
                       f"LR={self._ou_stacker.coef_[0][2]:.3f}, "
                       f"DC={self._ou_stacker.coef_[0][3]:.3f}")

            # Logit-shift calibration: compute on validation set (last season)
            val_df = train_df[es_val_mask]
            xgb_val_p = ou_xgb.predict_proba(
                val_df[self._ou_features].values)[:, 1]
            lgb_val_p = ou_lgb.predict_proba(
                pd.DataFrame(val_df[self._ou_features].values,
                             columns=self._ou_features))[:, 1]
            dc_val_p = ou_dc.predict_proba_df(val_df)
            lr_val_p = _lr_predict(ou_lr, ou_lr_scaler,
                                   val_df[self._ou_features].values)
            val_stack = np.column_stack([xgb_val_p, lgb_val_p, lr_val_p, dc_val_p])
            stacker_val_raw = self._ou_stacker.predict_proba(val_stack)[:, 1]

            val_mean_logit = np.mean(
                np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10)))
            target_logit = np.log(
                self._ou_base_rate / (1 - self._ou_base_rate + 1e-10))
            self._ou_logit_shift = val_mean_logit - target_logit
            # Persist val_mean_logit so regime (Phase C) can derive a new
            # shift against a regime-adjusted target without rerunning
            # models on the validation set.
            self._ou_val_mean_logit = float(val_mean_logit)

            # Verify calibration effect
            val_corrected = 1 / (1 + np.exp(
                -(np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10))
                  - self._ou_logit_shift)))
            self._log(f"  Logit shift: {self._ou_logit_shift:.4f}")
            self._log(f"  Val mean: raw={stacker_val_raw.mean():.4f}, "
                       f"corrected={val_corrected.mean():.4f}, "
                       f"actual={val_df['Over_2_5'].mean():.4f}")
            self._log(f"  Val Brier: raw={brier_score_loss(val_df['Over_2_5'], stacker_val_raw):.4f}, "
                       f"corrected={brier_score_loss(val_df['Over_2_5'], val_corrected):.4f}")
        else:
            self._log("  Insufficient OOF data for stacker — using raw average")
            self._ou_stacker = None
            self._ou_logit_shift = 0.0

        # ── BTTS Models ──
        self._log("Training BTTS models...")
        X_train_btts = train_df[self._btts_features].values
        y_train_btts = ((train_df["Home_Goals"] > 0) &
                        (train_df["Away_Goals"] > 0)).astype(int).values

        X_es_train_b = train_df.loc[es_train_mask, self._btts_features].values
        y_es_train_b = ((train_df.loc[es_train_mask, "Home_Goals"] > 0) &
                        (train_df.loc[es_train_mask, "Away_Goals"] > 0)).astype(int).values
        X_es_val_b = train_df.loc[es_val_mask, self._btts_features].values
        y_es_val_b = ((train_df.loc[es_val_mask, "Home_Goals"] > 0) &
                      (train_df.loc[es_val_mask, "Away_Goals"] > 0)).astype(int).values

        btts_xgb = refit_at_best_iteration(
            train_xgb_btts(X_es_train_b, y_es_train_b, X_es_val_b, y_es_val_b),
            X_train_btts, y_train_btts)
        btts_lgb = refit_at_best_iteration(
            train_lgb_btts(X_es_train_b, y_es_train_b, X_es_val_b, y_es_val_b,
                           feature_names=self._btts_features),
            X_train_btts, y_train_btts, feature_names=self._btts_features)
        btts_lr, btts_lr_scaler = train_logreg(X_train_btts, y_train_btts)
        # Use BTTS-tuned DC kwargs (Option 2 Step 1) rather than the
        # O/U 2.5 ones — different market, different optimal half-life/rho.
        btts_dc = DixonColesPredictor(**self._btts_dc_kwargs)
        btts_dc.fit(train_df)

        recent_btts = ((train_df.loc[recent_mask, "Home_Goals"] > 0) &
                       (train_df.loc[recent_mask, "Away_Goals"] > 0)).mean()
        self._btts_base_rate = recent_btts

        self._btts_models = {
            "xgb": btts_xgb, "lgb": btts_lgb,
            "lr": btts_lr, "lr_scaler": btts_lr_scaler,
            "dc": btts_dc,
        }
        self._log(f"BTTS base rate: {self._btts_base_rate:.3f}")

        # ── BTTS Calibration (ensemble logit-shift) ──
        self._log("Computing BTTS calibration shift...")
        val_btts_df = train_df[es_val_mask]
        btts_val_xgb = btts_xgb.predict_proba(X_es_val_b)[:, 1]
        btts_val_lgb = btts_lgb.predict_proba(
            pd.DataFrame(X_es_val_b, columns=self._btts_features))[:, 1]
        btts_val_lr = _lr_predict(btts_lr, btts_lr_scaler, X_es_val_b)
        btts_val_dc = btts_dc.predict_proba_btts_df(val_btts_df)

        btts_weights = np.array([0.20, 0.20, 0.30, 0.30])
        btts_val_ensemble = (btts_weights[0] * btts_val_xgb +
                             btts_weights[1] * btts_val_lgb +
                             btts_weights[2] * btts_val_lr +
                             btts_weights[3] * btts_val_dc)

        btts_val_mean_logit = np.mean(
            np.log(btts_val_ensemble / (1 - btts_val_ensemble + 1e-10)))
        btts_target_logit = np.log(
            self._btts_base_rate / (1 - self._btts_base_rate + 1e-10))
        btts_logit_shift = btts_val_mean_logit - btts_target_logit

        # Verify
        btts_val_logits = np.log(
            btts_val_ensemble / (1 - btts_val_ensemble + 1e-10))
        btts_val_corrected = 1 / (1 + np.exp(-(btts_val_logits - btts_logit_shift)))
        self._btts_cal_shifts = {"ensemble_logit_shift": btts_logit_shift}
        self._log(f"  BTTS logit shift: {btts_logit_shift:.4f}")
        self._log(f"  BTTS val mean: raw={btts_val_ensemble.mean():.4f}, "
                   f"corrected={btts_val_corrected.mean():.4f}, "
                   f"actual={y_es_val_b.mean():.4f}")

        self._log("All models trained.")

    def _current_season_matches(self) -> pd.DataFrame:
        """Return settled matches from the current season in chronological order.

        Used for both the two-phase matchweek counter and the regime
        detector's rolling-rate input. Returns an empty DataFrame before
        the season has any settled matches.
        """
        if self._full_df is None or self._full_df.empty:
            return pd.DataFrame()
        current_season = self._full_df["SeasonIndex"].max()
        season_df = self._full_df[
            self._full_df["SeasonIndex"] == current_season]
        # Settled = has goal counts populated
        settled = season_df[season_df["Home_Goals"].notna()
                            & season_df["Away_Goals"].notna()]
        if "Date" in settled.columns:
            settled = settled.sort_values("Date")
        return settled

    def _current_matchweek_count(self) -> int:
        """Number of settled matches in the current season.

        Drives the two-phase early-season config switch (< 60 = early).
        Cheap to compute — called once per generate_recommendations() run.
        """
        return len(self._current_season_matches())

    def _compute_regime_shift(
        self,
        clamp_key: str,
        base_rate: float,
        val_mean_logit: float,
        outcome_fn,
    ) -> tuple[float, float, bool]:
        """Compute a regime-adjusted logit shift for a single market.

        Delegates to the shared ``compute_regime_shift`` in
        ``predictor_utils``.  See that function for full documentation.
        """
        return compute_regime_shift(
            self._current_season_matches(),
            clamp_key, base_rate, val_mean_logit, outcome_fn,
        )

    def _predict_ou(self, fixture_row: pd.Series) -> dict:
        """Generate O/U 2.5 probability from all models for a single fixture.

        Uses stacker + logit-shift calibration when available (trained in
        train()). Falls back to equal-weight average for legacy pickles.
        """
        m = self._ou_models
        feats = np.array([fixture_row[self._ou_features].values], dtype=float)

        # Fill NaN with training medians
        if self._train_medians is not None:
            for i, f in enumerate(self._ou_features):
                if np.isnan(feats[0, i]) and f in self._train_medians.index:
                    feats[0, i] = self._train_medians[f]

        xgb_raw = float(m["xgb"].predict_proba(feats)[:, 1][0])
        lgb_raw = float(m["lgb"].predict_proba(
            pd.DataFrame(feats, columns=self._ou_features))[:, 1][0])
        lr_raw = float(_lr_predict(m["lr"], m["lr_scaler"], feats)[0])
        dc_raw = float(m["dc"].predict_proba_df(
            fixture_row.to_frame().T)[0])

        if self._ou_stacker is not None:
            base = np.array([[xgb_raw, lgb_raw, lr_raw, dc_raw]])
            stacker_p = float(
                self._ou_stacker.predict_proba(base)[:, 1][0])
            # Apply logit-shift calibration
            logit = np.log(stacker_p / (1 - stacker_p + 1e-10))
            ensemble = float(
                1 / (1 + np.exp(-(logit - self._ou_logit_shift))))
        else:
            # Legacy fallback: equal-weight average (no calibration)
            ensemble = (xgb_raw + lgb_raw + lr_raw + dc_raw) / 4.0

        return {
            "xgb": xgb_raw, "lgb": lgb_raw, "lr": lr_raw, "dc": dc_raw,
            "ensemble": ensemble,
            "per_model": np.array([xgb_raw, lgb_raw, lr_raw, dc_raw]),
        }

    def _get_dc_lambdas(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Extract home/away goal lambdas from the DC model for alt-line evaluation.

        Args:
            home_team: Home team name (our dataset format).
            away_team: Away team name (our dataset format).

        Returns:
            Tuple of (home_lambda, away_lambda).
        """
        dc = self._ou_models["dc"]
        h_att = dc.attack_home.get(home_team, dc.PRIORS["attack_home"])
        a_def = dc.defence_away.get(away_team, dc.PRIORS["defence_away"])
        a_att = dc.attack_away.get(away_team, dc.PRIORS["attack_away"])
        h_def = dc.defence_home.get(home_team, dc.PRIORS["defence_home"])

        sqrt_gamma = np.sqrt(dc.gamma)
        home_lambda = np.clip(h_att * a_def * dc.mu * sqrt_gamma, DC_LAMBDA_MIN, DC_LAMBDA_MAX)
        away_lambda = np.clip(a_att * h_def * dc.mu / sqrt_gamma, DC_LAMBDA_MIN, DC_LAMBDA_MAX)
        return float(home_lambda), float(away_lambda)

    def _fetch_selective_alt_totals(self, matches: list[dict]) -> None:
        """Fetch per-event alt_totals only for fixtures with high O/U 1.5 conviction.

        Path B (April 2026): the Odds API removed alternate_totals from its
        bulk endpoint, leaving per-event as the only path. Each per-event
        call costs 1 credit, so we gate on a quick DC-Poisson estimate of
        P(total goals > 1.5) — only fetch when the model thinks the Over
        is highly likely (above ``OU15_FETCH_PROB_THRESHOLD``).

        The function MUTATES the ``matches`` list in place by enriching each
        candidate match's ``bookmakers`` dict with alt-line entries. The
        downstream ``get_best_odds_all_lines`` reads from that structure
        unchanged.

        Args:
            matches: The-Odds-API match dicts (with totals already merged).
        """
        from config import OU15_FETCH_PROB_THRESHOLD
        from api.odds_api import (
            fetch_alt_totals_for_events,
            merge_alt_totals_into_match,
        )
        from scipy.stats import poisson as _poisson

        if not self._ou_models or "dc" not in self._ou_models:
            return  # No DC model loaded — can't gate

        # Per-league threshold lookup. Falls back to a sane default if the
        # config dict is missing the key (e.g. a third league added later).
        threshold = (
            OU15_FETCH_PROB_THRESHOLD.get("PL", 0.82)
            if isinstance(OU15_FETCH_PROB_THRESHOLD, dict)
            else float(OU15_FETCH_PROB_THRESHOLD)  # backward-compat
        )

        candidate_lookup: dict[str, dict] = {}
        for match in matches:
            home_mapped, away_mapped = match_to_our_teams(match, self._our_teams)
            if home_mapped is None or away_mapped is None:
                continue
            try:
                home_lam, away_lam = self._get_dc_lambdas(home_mapped, away_mapped)
            except Exception:
                continue
            # P(total goals <= 1) = P(0,0) + P(1,0) + P(0,1) under independent
            # Poissons. Ignoring DC's tau correction (small-score correlation)
            # is fine for a gate — we want a rough conviction signal, not the
            # final edge calc.
            p_le_1 = (
                _poisson.pmf(0, home_lam) * _poisson.pmf(0, away_lam)
                + _poisson.pmf(1, home_lam) * _poisson.pmf(0, away_lam)
                + _poisson.pmf(0, home_lam) * _poisson.pmf(1, away_lam)
            )
            p_over_15 = 1.0 - float(p_le_1)
            if p_over_15 > threshold:
                eid = match.get("id", "")
                if eid:
                    candidate_lookup[eid] = match

        if not candidate_lookup:
            self._log(
                f"OU 1.5 selective fetch: 0/{len(matches)} fixtures cleared "
                f"the {threshold:.2f} conviction threshold")
            return

        self._log(
            f"OU 1.5 selective fetch: {len(candidate_lookup)}/{len(matches)} "
            f"fixtures (threshold {threshold:.2f}) -- "
            f"fetching per-event alt_totals")
        try:
            alt_data = fetch_alt_totals_for_events(list(candidate_lookup.keys()))
        except Exception as e:
            logger.warning("Selective alt_totals fetch failed: %s", e)
            return

        for eid, data in alt_data.items():
            if eid in candidate_lookup:
                merge_alt_totals_into_match(candidate_lookup[eid], data)
        self._log(
            f"OU 1.5 selective fetch: enriched {len(alt_data)} matches "
            f"with alt-line data")

    def _predict_btts(self, fixture_row: pd.Series) -> dict:
        """Generate BTTS probability from all 4 models for a single fixture.

        Applies logit-shift calibration to ensemble when available.
        """
        m = self._btts_models
        feats = np.array([fixture_row[self._btts_features].values], dtype=float)

        if self._train_medians is not None:
            for i, f in enumerate(self._btts_features):
                if np.isnan(feats[0, i]) and f in self._train_medians.index:
                    feats[0, i] = self._train_medians[f]

        xgb_raw = float(m["xgb"].predict_proba(feats)[:, 1][0])
        lgb_raw = float(m["lgb"].predict_proba(
            pd.DataFrame(feats, columns=self._btts_features))[:, 1][0])
        lr_raw = float(_lr_predict(m["lr"], m["lr_scaler"], feats)[0])
        dc_raw = float(m["dc"].predict_proba_btts_df(
            fixture_row.to_frame().T)[0])

        # Weighted ensemble for BTTS (LR + DC get more weight)
        weights = np.array([0.20, 0.20, 0.30, 0.30])
        ensemble = float(np.dot([xgb_raw, lgb_raw, lr_raw, dc_raw], weights))

        # Apply logit-shift calibration to ensemble
        if (self._btts_cal_shifts is not None
                and "ensemble_logit_shift" in self._btts_cal_shifts):
            shift = self._btts_cal_shifts["ensemble_logit_shift"]
            logit = np.log(ensemble / (1 - ensemble + 1e-10))
            ensemble = float(1 / (1 + np.exp(-(logit - shift))))

        return {
            "xgb": xgb_raw, "lgb": lgb_raw, "lr": lr_raw, "dc": dc_raw,
            "ensemble": ensemble,
            "per_model": np.array([xgb_raw, lgb_raw, lr_raw, dc_raw]),
        }

    def _evaluate_bet(
        self,
        model_p: float,
        fair_p: float,
        odds: float,
        per_model: np.ndarray,
        fair_threshold: float,
        config: dict,
        edge_source: str = "devig",
        market: str = "",
        side: str = "",
        cumulative_bankroll: float = 1.0,
        peak_bankroll: float = 1.0,
    ) -> Optional[dict]:
        """Thin wrapper around ``staking.decide_bet`` for PL.

        Defers all bet-evaluation logic (blending, de-vig discount,
        shrinkage, Kelly sizing, market multiplier) to the shared
        ``decide_bet`` function so that the live prediction path and the
        ROI validator (``scripts/roi_validate.py``) make identical
        decisions given identical inputs.

        Args:
            edge_source: "pinnacle" or "devig" — devig edges are discounted.
            market: Market label (e.g. "ou25", "btts") for confidence multiplier.
            side: Side label (e.g. "over", "under", "yes", "no").
            cumulative_bankroll / peak_bankroll: feed drawdown protection.
        """
        return decide_bet(
            model_p, fair_p, odds, per_model, fair_threshold,
            config, edge_source, market, side,
            agree_scale=PL_AGREE_SCALE,
            cumulative_bankroll=cumulative_bankroll,
            peak_bankroll=peak_bankroll,
            early_season=getattr(self, "_is_early_season", False),
        )

    def generate_recommendations(
        self,
        markets: tuple[str, ...] = ("totals", "btts"),
        prefetched_matches: list[dict] | None = None,
    ) -> list[dict]:
        """Fetch live odds, generate predictions, return bet recommendations.

        Args:
            markets: Which odds markets to fetch. ("totals", "btts") for full,
                     ("totals",) to skip BTTS and save API calls.
            prefetched_matches: Optional pre-fetched match list in The-Odds-API
                format. When provided, skips the fetch_epl_odds() call. Used by
                the dashboard scan when The-Odds-API is down but OddsPapi data
                is available.

        Returns list of recommendation dicts, each containing:
          - fixture info (home, away, kickoff)
          - market (ou25 or btts)
          - side (over/under or yes/no)
          - model_prob, fair_prob, blended_prob, edge, ev
          - stake_pct, confidence, best_bookmaker
        """
        if self._ou_models is None or self._btts_models is None:
            raise RuntimeError("Models not trained. Call train() first.")

        # ── Freshness Gate (ADR 0005) ──
        # Checked here rather than at each caller: this one place covers every
        # recommendation path — scheduler, dashboard scan and both CLIs — so
        # there is no call site left to forget it.
        from freshness import assert_fresh
        assert_fresh("PL")

        # ── Phase C: regime detection + two-phase early-season setup ──
        # Both happen once per run and drive subsequent calibration/betting.
        matchweek_count = self._current_matchweek_count()
        self._is_early_season = matchweek_count < EARLY_SEASON_MATCHES
        if self._is_early_season:
            self._log(
                f"Early-season phase ACTIVE (matchweek count={matchweek_count} "
                f"< {EARLY_SEASON_MATCHES}): stricter edge, smaller stakes.")

        # Save original shifts so finally-block restores them after the run.
        # Regime adjustment is recomputed every run from live pipeline data,
        # so we never want to persist the regime-shifted value.
        _orig_ou_shift = self._ou_logit_shift

        # O/U 2.5 regime adjustment (PL)
        if self._ou_stacker is not None:
            new_shift, adj_rate, is_shifted = self._compute_regime_shift(
                clamp_key="ou25_pl",
                base_rate=self._ou_base_rate,
                val_mean_logit=self._ou_val_mean_logit,
                outcome_fn=lambda df: (
                    df["Home_Goals"] + df["Away_Goals"] > 2.5),
            )
            self._ou_logit_shift = new_shift
            if is_shifted:
                self._log(
                    f"O/U 2.5 regime SHIFTED: training_rate="
                    f"{self._ou_base_rate:.3f} → adjusted={adj_rate:.3f}, "
                    f"new_shift={new_shift:.4f}")

        try:
            return self._generate_recommendations_body(markets, prefetched_matches)
        finally:
            # Restore pristine calibration state for next run / pickling
            self._ou_logit_shift = _orig_ou_shift

    def _generate_recommendations_body(
        self,
        markets: tuple[str, ...],
        prefetched_matches: list[dict] | None,
    ) -> list[dict]:
        """The real body of generate_recommendations.

        Split out so the outer method can wrap it in a try/finally that
        restores any regime-adjusted calibration state.
        """
        if prefetched_matches is not None:
            matches = prefetched_matches
            self._log(f"Using {len(matches)} pre-fetched fixtures")
        else:
            self._log("Fetching live odds...")
            matches = fetch_epl_odds(force_refresh=False, markets=markets)
            self._log(f"Found {len(matches)} upcoming fixtures (pre-filter)")

        if not matches:
            return []

        # Filter to fixtures within 7-day lookahead window
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        _cutoff = _dt.now(_tz.utc) + _td(days=7)
        _filtered = []
        for m in matches:
            ct = m.get("commence_time", "")
            if ct:
                try:
                    ko = _dt.fromisoformat(ct.replace("Z", "+00:00"))
                    if ko > _cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            _filtered.append(m)
        self._log(f"After 7-day filter: {len(_filtered)} fixtures")
        matches = _filtered

        # Fetch OddsPapi for alt lines + AH (supplements The-Odds-API).
        # Option β-tight: only fire on the week-ahead snapshot. Matchday
        # morning, KO-1h, and CLV refreshes use Odds-API only — saves
        # ~215 OddsPapi credits/month. The merge code downstream is
        # null-safe (every op_data lookup tolerates an empty dict).
        oddspapi_data: dict[str, dict] = {}
        if self.snapshot_type == "week_ahead":
            try:
                oddspapi_fixtures = fetch_oddspapi_odds(force_refresh=False)

                # If API returned nothing (quota exhausted, timeout, etc.),
                # load stale cache directly — old odds are better than none
                if not oddspapi_fixtures:
                    from api.oddspapi import CACHE_FILE as _op_cache_file
                    try:
                        with open(_op_cache_file, "r") as _f:
                            import json as _json
                            _raw = _json.load(_f)
                        oddspapi_fixtures = _raw.get("data", [])
                        if oddspapi_fixtures:
                            logger.info(
                                "Loaded stale OddsPapi cache: %d fixtures "
                                "(cached %s)", len(oddspapi_fixtures),
                                _raw.get("timestamp", "unknown"),
                            )
                    except (FileNotFoundError, ValueError):
                        pass

                for fx in oddspapi_fixtures:
                    home = oddspapi_map_team(fx["home_team"], self._our_teams)
                    away = oddspapi_map_team(fx["away_team"], self._our_teams)
                    if home and away:
                        oddspapi_data[(home, away)] = fx
                self._log(f"OddsPapi: {len(oddspapi_data)} fixtures matched")
            except Exception as e:
                logger.warning(
                    "OddsPapi fetch failed (using The-Odds-API only): %s", e)
        else:
            self._log(
                f"OddsPapi skipped (snapshot_type={self.snapshot_type!r}, "
                "Option β-tight)")

        # ── Path B: selective per-event alt_totals fetch ──
        # The Odds API removed alternate_totals from its bulk endpoint
        # (April 2026); the only working path is per-event, which costs
        # 1 credit per fixture. So we fetch only for fixtures where the
        # DC model is highly confident on O/U 1.5 Over — books price
        # short prices tightly so without high conviction there's no
        # edge to find anyway.
        self._fetch_selective_alt_totals(matches)

        # Build fixture features from historical data
        df = self._full_df
        latest_season = df["SeasonIndex"].max()
        latest_df = df[df["SeasonIndex"] == latest_season].copy()

        recommendations = []
        self._match_analysis: list[dict] = []  # ALL fixture-market-side rows

        for match in matches:
            home_mapped, away_mapped = match_to_our_teams(match, self._our_teams)
            if home_mapped is None or away_mapped is None:
                self._log(f"Skipping {match['home_team']} vs {match['away_team']} "
                          f"(team mapping failed)")
                continue

            # Find most recent row for this fixture to get features
            fixture_mask = (
                (latest_df["Home_Team"] == home_mapped) &
                (latest_df["Away_Team"] == away_mapped)
            )
            if fixture_mask.sum() == 0:
                # Try to build features from most recent home/away rows
                home_rows = latest_df[latest_df["Home_Team"] == home_mapped]
                away_rows = latest_df[latest_df["Away_Team"] == away_mapped]
                if len(home_rows) == 0 or len(away_rows) == 0:
                    self._log(f"Skipping {home_mapped} vs {away_mapped} "
                              f"(no recent data)")
                    continue
                fixture_row = home_rows.iloc[-1].copy()
                # Override away-specific features from away team's latest row
                away_row = away_rows.iloc[-1]
                for col in fixture_row.index:
                    if col.startswith("Away_"):
                        fixture_row[col] = away_row[col]
            else:
                fixture_row = latest_df[fixture_mask].iloc[-1]

            kickoff = match.get("commence_time", "")

            # OddsPapi data for this fixture (used across all markets)
            op_data = oddspapi_data.get((home_mapped, away_mapped))

            # ── O/U 2.5 ──
            best_ou = get_best_odds(match)
            # Merge OddsPapi O/U 2.5 for wider bookmaker coverage
            if op_data and op_data.get("ou_lines"):
                op_ou25 = op_data["ou_lines"].get(2.5)
                if op_ou25:
                    if best_ou is None:
                        # No The-Odds-API data — use OddsPapi entirely
                        best_ou = {
                            "best_over": op_ou25["best_over"],
                            "best_under": op_ou25["best_under"],
                            "best_over_book": op_ou25["best_over_book"],
                            "best_under_book": op_ou25["best_under_book"],
                            "sharp_fair_over": op_ou25.get("sharp_fair_over"),
                            "sharp_fair_under": op_ou25.get("sharp_fair_under"),
                            "n_books": op_ou25.get("n_books", 0),
                            "consensus_over": 0, "consensus_under": 0,
                            "median_over": 0, "median_under": 0,
                        }
                    else:
                        # Merge: keep best price across both sources
                        if op_ou25["best_over"] > best_ou.get("best_over", 0):
                            best_ou["best_over"] = op_ou25["best_over"]
                            best_ou["best_over_book"] = op_ou25["best_over_book"]
                        if op_ou25["best_under"] > best_ou.get("best_under", 0):
                            best_ou["best_under"] = op_ou25["best_under"]
                            best_ou["best_under_book"] = op_ou25["best_under_book"]
                        # Prefer Pinnacle sharp from OddsPapi if available
                        if op_ou25.get("sharp_fair_over") is not None:
                            best_ou["sharp_fair_over"] = op_ou25["sharp_fair_over"]
                            best_ou["sharp_fair_under"] = op_ou25["sharp_fair_under"]
                        best_ou["n_books"] = max(
                            best_ou.get("n_books", 0),
                            op_ou25.get("n_books", 0))

            if best_ou:
                ou_pred = self._predict_ou(fixture_row)

                # Fair probabilities — prefer Pinnacle sharp line when
                # available (true market consensus), fall back to
                # de-vigging the best available book price
                sharp_over = best_ou.get("sharp_fair_over")
                sharp_under = best_ou.get("sharp_fair_under")
                raw_over_imp = 1.0 / best_ou["best_over"]
                raw_under_imp = 1.0 / best_ou["best_under"]
                overround = raw_over_imp + raw_under_imp
                devig_over = raw_over_imp / overround
                devig_under = raw_under_imp / overround

                if sharp_over is not None and sharp_under is not None:
                    fair_over = sharp_over
                    fair_under = sharp_under
                    edge_source = "pinnacle"
                else:
                    fair_over = devig_over
                    fair_under = devig_under
                    edge_source = "devig"

                for side, model_p, fair_p, odds, book in [
                    ("over", ou_pred["ensemble"], fair_over,
                     best_ou["best_over"], best_ou["best_over_book"]),
                    ("under", 1 - ou_pred["ensemble"], fair_under,
                     best_ou["best_under"], best_ou["best_under_book"]),
                ]:
                    edge_pct = (model_p - fair_p) * 100
                    fair_odds_val = 1 / fair_p if fair_p > 0 else None
                    self._match_analysis.append({
                        "home_team": home_mapped, "away_team": away_mapped,
                        "kickoff": kickoff, "market": "ou25", "side": side,
                        "best_odds": odds, "best_bookmaker": book,
                        "model_prob": model_p, "fair_odds": fair_odds_val,
                        "edge_pct": edge_pct,
                        "edge_source": edge_source,
                        "confidence": (
                            "high" if edge_pct > 4 else
                            "medium" if edge_pct > 2.5 else
                            "low" if edge_pct > 0 else None
                        ),
                        "n_books": best_ou.get("n_books"),
                        "per_model_probs": {
                            "xgb": ou_pred["xgb"], "lgb": ou_pred["lgb"],
                            "lr": ou_pred["lr"], "dc": ou_pred["dc"],
                        },
                        "bookmaker_odds": _extract_bookmaker_odds(
                            match, "ou25", side),
                    })

                    if side == "over":
                        fair_threshold = fair_over
                        per_model = ou_pred["per_model"]
                    else:
                        fair_threshold = fair_under
                        per_model = 1 - ou_pred["per_model"]

                    bet = self._evaluate_bet(
                        model_p, fair_p, odds, per_model, fair_threshold,
                        self.ou_config,
                        edge_source=edge_source,
                        market="ou25",
                        side=side,
                    )
                    if bet:
                        bet.update({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": "ou25",
                            "side": side,
                            "best_bookmaker": book,
                            "n_books": best_ou["n_books"],
                            "per_model_probs": {
                                "xgb": ou_pred["xgb"],
                                "lgb": ou_pred["lgb"],
                                "lr": ou_pred["lr"],
                                "dc": ou_pred["dc"],
                            },
                        })
                        recommendations.append(bet)

            # ── BTTS ── (skip if not in requested markets)
            best_btts = get_best_btts_odds(match) if "btts" in markets else None
            # Merge OddsPapi BTTS for wider bookmaker coverage
            if op_data and op_data.get("btts"):
                op_btts = op_data["btts"]
                if best_btts is None:
                    # No The-Odds-API data — use OddsPapi entirely
                    best_btts = {
                        "best_yes": op_btts["best_yes"],
                        "best_no": op_btts["best_no"],
                        "best_yes_book": op_btts["best_yes_book"],
                        "best_no_book": op_btts["best_no_book"],
                        "sharp_fair_yes": op_btts.get("sharp_fair_yes"),
                        "sharp_fair_no": op_btts.get("sharp_fair_no"),
                        "n_books": op_btts.get("n_books", 0),
                        "consensus_yes": 0, "consensus_no": 0,
                        "median_yes": 0, "median_no": 0,
                    }
                else:
                    # Merge: keep best price across both sources
                    if op_btts["best_yes"] > best_btts.get("best_yes", 0):
                        best_btts["best_yes"] = op_btts["best_yes"]
                        best_btts["best_yes_book"] = op_btts["best_yes_book"]
                    if op_btts["best_no"] > best_btts.get("best_no", 0):
                        best_btts["best_no"] = op_btts["best_no"]
                        best_btts["best_no_book"] = op_btts["best_no_book"]
                    # Prefer Pinnacle sharp from OddsPapi if available
                    if op_btts.get("sharp_fair_yes") is not None:
                        best_btts["sharp_fair_yes"] = op_btts["sharp_fair_yes"]
                        best_btts["sharp_fair_no"] = op_btts["sharp_fair_no"]
                    best_btts["n_books"] = max(
                        best_btts.get("n_books", 0),
                        op_btts.get("n_books", 0))

            if best_btts:
                btts_pred = self._predict_btts(fixture_row)

                # Fair probabilities — prefer Pinnacle sharp line
                sharp_yes = best_btts.get("sharp_fair_yes")
                sharp_no = best_btts.get("sharp_fair_no")
                raw_yes_imp = 1.0 / best_btts["best_yes"]
                raw_no_imp = 1.0 / best_btts["best_no"]
                overround = raw_yes_imp + raw_no_imp
                devig_yes = raw_yes_imp / overround
                devig_no = raw_no_imp / overround

                if sharp_yes is not None and sharp_no is not None:
                    fair_yes = sharp_yes
                    fair_no = sharp_no
                    btts_edge_source = "pinnacle"
                else:
                    fair_yes = devig_yes
                    fair_no = devig_no
                    btts_edge_source = "devig"

                btts_weights = np.array([0.20, 0.20, 0.30, 0.30])

                for side, model_p, fair_p, odds, book in [
                    ("yes", btts_pred["ensemble"], fair_yes,
                     best_btts["best_yes"], best_btts["best_yes_book"]),
                    ("no", 1 - btts_pred["ensemble"], fair_no,
                     best_btts["best_no"], best_btts["best_no_book"]),
                ]:
                    edge_pct = (model_p - fair_p) * 100
                    fair_odds_val = 1 / fair_p if fair_p > 0 else None
                    self._match_analysis.append({
                        "home_team": home_mapped, "away_team": away_mapped,
                        "kickoff": kickoff, "market": "btts", "side": side,
                        "best_odds": odds, "best_bookmaker": book,
                        "model_prob": model_p, "fair_odds": fair_odds_val,
                        "edge_pct": edge_pct,
                        "edge_source": btts_edge_source,
                        "confidence": (
                            "high" if edge_pct > 4 else
                            "medium" if edge_pct > 2.5 else
                            "low" if edge_pct > 0 else None
                        ),
                        "n_books": best_btts.get("n_books"),
                        "per_model_probs": {
                            "xgb": btts_pred["xgb"], "lgb": btts_pred["lgb"],
                            "lr": btts_pred["lr"], "dc": btts_pred["dc"],
                        },
                        "bookmaker_odds": _extract_bookmaker_odds(
                            match, "btts", side),
                    })

                    if side == "yes":
                        fair_threshold = fair_yes
                        per_model = btts_pred["per_model"]
                    else:
                        fair_threshold = fair_no
                        per_model = 1 - btts_pred["per_model"]

                    bet = self._evaluate_bet(
                        model_p, fair_p, odds, per_model, fair_threshold,
                        self.btts_config,
                        edge_source=btts_edge_source,
                        market="btts",
                        side=side,
                    )
                    if bet:
                        bet.update({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": "btts",
                            "side": side,
                            "best_bookmaker": book,
                            "n_books": best_btts["n_books"],
                            "per_model_probs": {
                                "xgb": btts_pred["xgb"],
                                "lgb": btts_pred["lgb"],
                                "lr": btts_pred["lr"],
                                "dc": btts_pred["dc"],
                            },
                        })
                        recommendations.append(bet)

            # ── Alternative O/U Lines ──
            # Merge odds from The-Odds-API and OddsPapi for best coverage
            all_line_odds = get_best_odds_all_lines(match)

            # Overlay OddsPapi data if available (more bookmakers, more lines)
            if op_data and op_data.get("ou_lines"):
                if all_line_odds is None:
                    all_line_odds = {}
                for line, op_ld in op_data["ou_lines"].items():
                    if line not in all_line_odds:
                        # New line only from OddsPapi
                        all_line_odds[line] = op_ld
                    else:
                        # Merge: keep best price across both sources
                        existing = all_line_odds[line]
                        if op_ld["best_over"] > existing.get("best_over", 0):
                            existing["best_over"] = op_ld["best_over"]
                            existing["best_over_book"] = op_ld["best_over_book"]
                        if op_ld["best_under"] > existing.get("best_under", 0):
                            existing["best_under"] = op_ld["best_under"]
                            existing["best_under_book"] = op_ld["best_under_book"]
                        # Prefer Pinnacle sharp from OddsPapi
                        if op_ld.get("sharp_fair_over") is not None:
                            existing["sharp_fair_over"] = op_ld["sharp_fair_over"]
                            existing["sharp_fair_under"] = op_ld["sharp_fair_under"]
                        existing["n_books"] = max(
                            existing.get("n_books", 0), op_ld.get("n_books", 0))

            if all_line_odds:
                alt_line_odds = {k: v for k, v in all_line_odds.items()
                                 if float(k) in ALLOWED_ALT_LINES}

                if alt_line_odds:
                    home_lam, away_lam = self._get_dc_lambdas(
                        home_mapped, away_mapped)
                    dc = self._ou_models["dc"]

                    # Phase C: two-phase override for alt lines path
                    _alt_blend_w = (EARLY_BLEND_WEIGHT
                                     if self._is_early_season
                                     else self.ou_config.get("blend_weight", 0.35))
                    _alt_min_edge = (EARLY_MIN_EDGE
                                      if self._is_early_season
                                      else self.ou_config.get("min_edge", 0.02))

                    opps = scan_all_lines(
                        home_lambda=home_lam,
                        away_lambda=away_lam,
                        odds_by_line=alt_line_odds,
                        rho=dc.rho,
                        blend_weight=_alt_blend_w,
                        min_edge=_alt_min_edge,
                    )

                    # Record ALL alt line opportunities in match analysis
                    for opp in opps:
                        line_label = f"ou{opp['line']:.1f}".replace(".", "")
                        fair_odds_val = (1 / opp["fair_prob"]
                                         if opp["fair_prob"] > 0 else None)
                        edge_pct = opp["edge"] * 100
                        # Check if this line had Pinnacle data
                        _line_data = alt_line_odds.get(opp["line"], {})
                        _alt_edge_src = ("pinnacle"
                                         if _line_data.get(f"sharp_fair_{opp['side']}")
                                         is not None else "devig")
                        self._match_analysis.append({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": line_label,
                            "side": opp["side"],
                            "best_odds": opp["odds"],
                            "best_bookmaker": opp["book"],
                            "model_prob": opp["model_prob"],
                            "fair_odds": fair_odds_val,
                            "edge_pct": edge_pct,
                            "edge_source": _alt_edge_src,
                            "confidence": (
                                "high" if edge_pct > 4 else
                                "medium" if edge_pct > 2.5 else
                                "low" if edge_pct > 0 else None
                            ),
                            "n_books": _line_data.get("n_books", 0),
                            "per_model_probs": {
                                "dc_poisson": opp["model_prob"],
                            },
                            "bookmaker_odds": _extract_bookmaker_odds(
                                match, line_label, opp["side"]),
                        })

                    value_bets = get_value_bets(
                        opps,
                        min_edge=_alt_min_edge,
                        min_ev=0.0,
                    )

                    if ALT_LINES_OVER_ONLY:
                        value_bets = [vb for vb in value_bets
                                      if vb["side"] == "over"]

                    for vb in value_bets:
                        _vb_market = f"ou{vb['line']:.1f}".replace(".", "")
                        _vb_line_data = alt_line_odds.get(vb["line"], {})
                        _vb_edge_src = (
                            "pinnacle"
                            if _vb_line_data.get(f"sharp_fair_{vb['side']}")
                            is not None else "devig"
                        )
                        # Apply de-vig discount to edge
                        _vb_edge = vb["edge"]
                        if _vb_edge_src == "devig":
                            from config import DEVIG_DISCOUNT
                            _vb_edge *= DEVIG_DISCOUNT
                        # Apply market/side confidence multiplier to Kelly
                        from config import MARKET_MULTIPLIERS
                        _vb_mult = MARKET_MULTIPLIERS.get(
                            (_vb_market, vb["side"]), 1.0)
                        _vb_stake = vb["kelly"] * _vb_mult
                        # Phase C: early-season Kelly reduction. scan_all_lines
                        # already applies quarter-Kelly (0.25); rescale to the
                        # early-season fraction (0.15) = 60% of normal stake.
                        if self._is_early_season:
                            _vb_stake *= (EARLY_KELLY_FRACTION / 0.25)

                        recommendations.append({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": _vb_market,
                            "side": vb["side"],
                            "model_prob": vb["model_prob"],
                            "blended_prob": vb["blended_prob"],
                            "fair_prob": vb["fair_prob"],
                            "odds": vb["odds"],
                            "edge": _vb_edge,
                            "ev": vb["ev"],
                            "n_agree": 0,  # alt lines use DC only
                            "stake_pct": _vb_stake,
                            "edge_source": _vb_edge_src,
                            "market_multiplier": _vb_mult,
                            "confidence": "high" if _vb_edge > 0.04 else
                                          "medium" if _vb_edge > 0.025 else
                                          "low",
                            "best_bookmaker": vb["book"],
                            "n_books": _vb_line_data.get("n_books", 0),
                            "per_model_probs": {"dc_poisson": vb["model_prob"]},
                            "line": vb["line"],
                            "line_type": vb["line_type"],
                        })

        # Option 5 Step 1 + 2c: apply same-match correlation discount and
        # matchday portfolio cap. Done after all bets are accumulated so
        # the cap can see the full daily load. (``apply_portfolio_constraints``
        # is imported at module top from ``staking``.)
        pre_stakes = sum(r.get("stake_pct", 0.0) for r in recommendations)
        apply_portfolio_constraints(recommendations)
        post_stakes = sum(r.get("stake_pct", 0.0) for r in recommendations)
        if pre_stakes > 0 and abs(pre_stakes - post_stakes) > 1e-6:
            self._log(
                f"Portfolio constraints applied: total stake "
                f"{pre_stakes:.3f} -> {post_stakes:.3f}")

        # Sort by edge descending
        recommendations.sort(key=lambda x: x["edge"], reverse=True)
        self._log(f"Generated {len(recommendations)} recommendations")
        return recommendations


def run_predictions(verbose: bool = True) -> list[dict]:
    """Convenience function: load, train, predict in one call."""
    predictor = LivePredictor(verbose=verbose)
    predictor.load_data()
    predictor.train()
    return predictor.generate_recommendations()


if __name__ == "__main__":
    predictor = LivePredictor(verbose=True)
    predictor.load_data()
    predictor.train()
    recs = predictor.generate_recommendations()

    if not recs:
        print("\nNo recommendations (no upcoming fixtures or no edges found)")
    else:
        print(f"\n{'='*90}")
        print(f"{'MATCH':<35} {'MKT':<8} {'SIDE':<6} {'EDGE':>6} "
              f"{'ODDS':>5} {'STAKE':>6} {'CONF':<6} {'BOOK':<15}")
        print(f"{'='*90}")
        for r in recs:
            fixture = f"{r['home_team']} v {r['away_team']}"
            if len(fixture) > 34:
                fixture = fixture[:31] + "..."
            mkt = r['market']
            print(f"{fixture:<35} {mkt:<8} {r['side']:<6} "
                  f"{r['edge']:>+5.1%} {r['odds']:>5.2f} "
                  f"{r['stake_pct']:>5.1%} {r['confidence']:<6} "
                  f"{r['best_bookmaker']:<15}")

    # Save to dashboard database
    from db import save_recommendations, save_match_analysis
    n_saved = save_recommendations(recs)
    print(f"\nSaved {n_saved} new recommendations to dashboard DB")

    # Save full match analysis (ALL fixture-market-side rows, not just edges)
    analysis = getattr(predictor, "_match_analysis", [])
    if analysis:
        n_analysis = save_match_analysis(analysis)
        print(f"Saved {n_analysis} match analysis rows to dashboard DB")
