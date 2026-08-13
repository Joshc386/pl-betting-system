"""
Championship live prediction engine.

Loads trained Championship models (or trains on-the-fly), generates
predictions for upcoming EFL Championship fixtures, blends with live
odds from The-Odds-API and OddsPapi, and outputs bet recommendations.

Markets evaluated:
  - O/U 2.5 goals (3-model ensemble: XGB + LGB + Dixon-Coles)
  - O/U 1.5 goals (3-model ensemble)
  - BTTS (3-model ensemble)

Key differences from PL predict.py:
  - 3 models (no Logistic Regression — removed for same reasons as PL)
  - Championship-specific team name mapping (CSV uses short names)
  - Odds-API sport key: soccer_efl_champ
  - OddsPapi tournament ID: 18 (Championship)
  - Alt O/U lines via DC Poisson (O/U 3.5 only — dedup guard skips 1.5/2.5)
"""
from __future__ import annotations

import json
import logging
import os
import numpy as np
import pandas as pd
from typing import Optional

from championship_pipeline import (
    run_pipeline,
    CHAMP_ALL_FEATURES,
    CHAMP_OU15_FEATURES,
    CHAMP_BTTS_FEATURES,
)
from championship_model import (
    train_xgb_champ, train_lgb_champ,
    train_xgb_btts_champ, train_lgb_btts_champ,
    train_xgb_ou15_champ, train_lgb_ou15_champ,
    tune_dc_params_champ,
    MIN_TRAIN_SEASON,
    MODEL_DIR as CHAMP_MODEL_DIR,
)
from championship_backtest import (
    _calibrate, _calibrate_single,
    DEFAULT_CONFIG,
)
from staking import decide_bet, apply_portfolio_constraints, EFL_AGREE_SCALE
from config import (
    EARLY_SEASON_MATCHES, EARLY_BLEND_WEIGHT,
    EARLY_MIN_EDGE, EARLY_KELLY_FRACTION,
    EFL_ALLOWED_ALT_LINES, EFL_ALT_LINES_OVER_ONLY,
    MARKET_MULTIPLIERS, DEVIG_DISCOUNT,
    DC_LAMBDA_MIN, DC_LAMBDA_MAX,
)
from predictor_utils import save_pickle, load_pickle, compute_regime_shift
from model import DixonColesPredictor
from league_config import get_league_config
from api.odds_api import (
    fetch_epl_odds, get_best_odds, get_best_btts_odds,
    get_best_odds_all_lines,
    match_to_our_teams,
)
from alt_lines import scan_all_lines, get_value_bets
from predict import _extract_bookmaker_odds
from api.oddspapi import (
    fetch_epl_all_odds as fetch_oddspapi_odds,
    map_team as oddspapi_map_team,
)
from api.team_resolver import resolve_feed_team
import api.odds_api as odds_api_module
import api.oddspapi as oddspapi_module

logger = logging.getLogger(__name__)

LEAGUE_CFG = get_league_config("EFL")

# BTTS default config (mirrors O/U but slightly more conservative)
BTTS_DEFAULT_CONFIG: dict = {
    "blend_weight": 0.35,
    "min_edge": 0.02,
    "min_agree": 2,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.05,
}

# O/U 1.5 default config (high base rate means lean on market)
OU15_DEFAULT_CONFIG: dict = {
    "blend_weight": 0.30,
    "min_edge": 0.03,
    "min_agree": 2,
    "kelly_fraction": 0.20,
    "max_stake_pct": 0.05,
}

# ═══════════════════════════════════════════════════════════════════════════════
# Championship team name mapping (Odds-API → CSV short names)
# ═══════════════════════════════════════════════════════════════════════════════

_ODDS_API_TO_CHAMP: dict[str, str] = {
    # Current 2024/25 Championship
    "Blackburn Rovers": "Blackburn",
    "Bristol City": "Bristol City",
    "Burnley": "Burnley",
    "Cardiff City": "Cardiff",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Hull City": "Hull",
    "Leeds United": "Leeds",
    "Luton Town": "Luton",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Norwich City": "Norwich",
    "Oxford United": "Oxford",
    "Plymouth Argyle": "Plymouth",
    "Portsmouth": "Portsmouth",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "QPR": "QPR",
    "Sheffield United": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Weds",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea City": "Swansea",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "West Brom": "West Brom",
    # 2026/27 arrivals — relegated from the PL or promoted from League One.
    # Lincoln have no canonical history, so the fuzzy path cannot back this
    # entry up: without it their fixtures resolve to None and vanish.
    "Bolton Wanderers": "Bolton",
    "Lincoln City": "Lincoln",
    "West Ham United": "West Ham",
    # Recent Championship teams (for when squads change)
    "Birmingham City": "Birmingham",
    "Charlton Athletic": "Charlton",
    "Huddersfield Town": "Huddersfield",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Nottingham Forest": "Nott'm Forest",
    "Rotherham United": "Rotherham",
    "Southampton": "Southampton",
    "Wrexham AFC": "Wrexham",
    "Wolverhampton Wanderers": "Wolves",
    "Wigan Athletic": "Wigan",
    "Reading": "Reading",
    "Barnsley": "Barnsley",
    "Peterborough United": "Peterboro",
    # OddsPapi "FC" suffix variants (OddsPapi appends FC to many names)
    "Southampton FC": "Southampton",
    "Middlesbrough FC": "Middlesbrough",
    "Millwall FC": "Millwall",
    "Portsmouth FC": "Portsmouth",
    "Watford FC": "Watford",
    "Sunderland FC": "Sunderland",
    "Burnley FC": "Burnley",
    "Lincoln City FC": "Lincoln",
}


def _resolve_champ_team(api_name: str, our_teams: set[str]) -> str | None:
    """Map Odds-API team name to Championship CSV name.

    The matching rule is shared with both PL feeds — see api.team_resolver.
    Only the mapping dict is Championship-specific, because only the name
    format is: this canonical keeps football-data.co.uk short forms.

    If the team isn't in the historical dataset (e.g. newly promoted),
    still returns the mapped short name so fixtures appear in the dashboard.

    Args:
        api_name: Team name from The-Odds-API or OddsPapi.
        our_teams: Set of team names from the Championship CSV.

    Returns:
        Matched team name (may not be in our_teams if newly promoted).
        None only if no mapping exists at all.
    """
    return resolve_feed_team(api_name, our_teams, _ODDS_API_TO_CHAMP)


def _match_champ_teams(odds_match: dict,
                       our_teams: set[str]) -> tuple[str | None, str | None]:
    """Map both home and away from Odds-API to Championship CSV names."""
    home = _resolve_champ_team(odds_match["home_team"], our_teams)
    away = _resolve_champ_team(odds_match["away_team"], our_teams)
    return home, away


# ═══════════════════════════════════════════════════════════════════════════════
# Odds fetching with Championship sport key override
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_champ_odds(
    force_refresh: bool = True,
    markets: tuple[str, ...] = ("totals", "btts"),
) -> list[dict]:
    """Fetch Championship odds by temporarily overriding the sport key.

    The-Odds-API uses 'soccer_efl_champ' for Championship fixtures.
    We swap the module-level SPORT variable, fetch, then restore it.

    Args:
        force_refresh: Bypass odds cache.
        markets: Which markets to fetch ("totals", "btts").

    Returns:
        List of match dicts with bookmaker odds.
    """
    original_sport = odds_api_module.SPORT
    original_cache = odds_api_module.CACHE_FILE

    try:
        odds_api_module.SPORT = LEAGUE_CFG["odds_api_sport"]
        odds_api_module.CACHE_FILE = os.path.join(
            odds_api_module.CACHE_DIR, "odds_cache_efl.json")
        return fetch_epl_odds(force_refresh=force_refresh, markets=markets)
    finally:
        odds_api_module.SPORT = original_sport
        odds_api_module.CACHE_FILE = original_cache


def _fetch_champ_oddspapi(
    force_refresh: bool = True,
    our_teams: set[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """Fetch Championship odds from OddsPapi.

    Overrides the tournament ID to Championship, fetches, then restores.

    Args:
        force_refresh: Bypass cache.
        our_teams: Set of CSV team names for mapping.

    Returns:
        Dict keyed by (home, away) tuple of CSV team names.
    """
    original_tid = oddspapi_module.EPL_TOURNAMENT_ID
    original_cache = getattr(oddspapi_module, "CACHE_FILE", None)

    try:
        oddspapi_module.EPL_TOURNAMENT_ID = (
            oddspapi_module.CHAMPIONSHIP_TOURNAMENT_ID)
        if original_cache:
            oddspapi_module.CACHE_FILE = original_cache.replace(
                "oddspapi_cache", "oddspapi_cache_efl")

        fixtures = fetch_oddspapi_odds(force_refresh=force_refresh)

        # If API returned nothing (quota exhausted, timeout, etc.),
        # load stale cache directly — old odds are better than none
        if not fixtures:
            efl_cache = oddspapi_module.CACHE_FILE
            try:
                with open(efl_cache, "r") as _f:
                    _raw = json.load(_f)
                fixtures = _raw.get("data", [])
                if fixtures:
                    logger.info(
                        "Loaded stale OddsPapi EFL cache: %d fixtures "
                        "(cached %s)", len(fixtures),
                        _raw.get("timestamp", "unknown"),
                    )
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        result: dict[tuple[str, str], dict] = {}

        if our_teams:
            for fx in fixtures:
                home = _resolve_champ_team(fx["home_team"], our_teams)
                away = _resolve_champ_team(fx["away_team"], our_teams)
                if home and away:
                    result[(home, away)] = fx

        return result
    except Exception as e:
        logger.warning("OddsPapi Championship fetch failed: %s", e)
        return {}
    finally:
        oddspapi_module.EPL_TOURNAMENT_ID = original_tid
        if original_cache:
            oddspapi_module.CACHE_FILE = original_cache


# ═══════════════════════════════════════════════════════════════════════════════
# Championship Live Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class ChampionshipPredictor:
    """Generates bet recommendations for Championship fixtures.

    Trains 3-model ensembles (XGB + LGB + Dixon-Coles) for O/U 2.5,
    O/U 1.5, and BTTS markets. Compares calibrated model probabilities
    against live bookmaker odds with model-market blend and Kelly staking.
    """

    def __init__(
        self,
        ou_config: dict | None = None,
        ou15_config: dict | None = None,
        btts_config: dict | None = None,
        verbose: bool = True,
        use_sparse_features: bool | None = None,
        snapshot_type: str | None = None,
    ) -> None:
        self.ou_config = ou_config or DEFAULT_CONFIG.copy()
        self.ou15_config = ou15_config or OU15_DEFAULT_CONFIG.copy()
        self.btts_config = btts_config or BTTS_DEFAULT_CONFIG.copy()
        self.verbose = verbose
        # Option 3 Step 3: mirrors LivePredictor — None = use global default.
        from config import USE_SPARSE_FEATURES, DEFAULT_SNAPSHOT_TYPE
        self.use_sparse_features = (
            USE_SPARSE_FEATURES if use_sparse_features is None
            else use_sparse_features)
        # Option β-tight: gates OddsPapi fetch. "week_ahead" = full sweep,
        # "refresh" = Odds-API only.
        self.snapshot_type = (
            DEFAULT_SNAPSHOT_TYPE if snapshot_type is None
            else snapshot_type)

        self._pipeline_data: dict | None = None
        self._full_df: pd.DataFrame | None = None
        self._ou_features: list[str] = []
        self._ou15_features: list[str] = []
        self._btts_features: list[str] = []

        # Trained models (set by train())
        self._ou_models: dict | None = None
        self._ou15_models: dict | None = None
        self._btts_models: dict | None = None
        self._ou_base_rate: float = 0.475
        self._ou15_base_rate: float = 0.730
        self._btts_base_rate: float = 0.517
        # Per-market DC kwargs (Option 2 Step 1).
        # _dc_kwargs retains O/U 2.5 values for backward compat; the other
        # two slots hold market-specific tuned values.
        self._dc_kwargs: dict = {}
        self._ou15_dc_kwargs: dict = {}
        self._btts_dc_kwargs: dict = {}
        self._our_teams: set[str] = set()

        # Calibration: per-market logit-shifts (set by train())
        self._cal_shifts: dict[str, float] = {}
        # Phase C: per-market val-set mean logits — cached so regime can
        # rederive shifts against adjusted base rates without rerunning
        # models on the validation set.
        self._val_mean_logits: dict[str, float] = {}
        # Phase C: two-phase early-season flag (set by generate_recommendations)
        self._is_early_season: bool = False

        # Match analysis (all fixture-market-side rows for dashboard)
        self._match_analysis: list[dict] = []

    # Default paths for trained state and pipeline cache
    _STATE_PATH = os.path.join(CHAMP_MODEL_DIR, "efl_trained_state.pkl")
    _PIPELINE_CACHE_PATH = os.path.join(CHAMP_MODEL_DIR, "efl_pipeline_cache.pkl")

    def _log(self, msg: str) -> None:
        """Log and optionally print."""
        if self.verbose:
            print(f"  {msg}")
        logger.info(msg)

    # ── Save / Load Infrastructure ──

    def save_trained_state(self, path: str | None = None) -> None:
        """Save all trained models, features, calibration params to disk.

        Args:
            path: Override path for the pickle file.
        """
        state = {
            "ou_models": self._ou_models,
            "ou15_models": self._ou15_models,
            "btts_models": self._btts_models,
            "ou_features": self._ou_features,
            "ou15_features": self._ou15_features,
            "btts_features": self._btts_features,
            "ou_base_rate": self._ou_base_rate,
            "ou15_base_rate": self._ou15_base_rate,
            "btts_base_rate": self._btts_base_rate,
            "dc_kwargs": self._dc_kwargs,
            "ou15_dc_kwargs": self._ou15_dc_kwargs,
            "btts_dc_kwargs": self._btts_dc_kwargs,
            "our_teams": self._our_teams,
            "cal_shifts": self._cal_shifts,
            "val_mean_logits": self._val_mean_logits,
        }
        save_pickle(state, path, self._STATE_PATH, self._log,
                     label="Trained state")

    def load_trained_state(self, path: str | None = None) -> bool:
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
        self._ou15_models = state["ou15_models"]
        self._btts_models = state["btts_models"]
        self._ou_features = state["ou_features"]
        self._ou15_features = state["ou15_features"]
        self._btts_features = state["btts_features"]
        self._ou_base_rate = state["ou_base_rate"]
        self._ou15_base_rate = state["ou15_base_rate"]
        self._btts_base_rate = state["btts_base_rate"]
        self._dc_kwargs = state["dc_kwargs"]
        # Backward compat: legacy pickles only have dc_kwargs — fall back
        # to reusing it, matching pre-Option-2 behaviour.
        self._ou15_dc_kwargs = state.get("ou15_dc_kwargs", state["dc_kwargs"])
        self._btts_dc_kwargs = state.get("btts_dc_kwargs", state["dc_kwargs"])
        self._our_teams = state["our_teams"]
        self._cal_shifts = state.get("cal_shifts", {})
        self._val_mean_logits = state.get("val_mean_logits", {})
        if self._cal_shifts:
            self._log(f"  Calibration shifts: {self._cal_shifts}")
        else:
            self._log("  No calibration shifts (legacy pickle)")
        return True

    def save_pipeline_cache(self, path: str | None = None) -> None:
        """Cache the pipeline DataFrame to skip feature engineering on reload.

        Args:
            path: Override path for the pickle file.
        """
        save_pickle(
            {
                "full_df": self._full_df,
                "ou_features": self._ou_features,
                "ou15_features": self._ou15_features,
                "btts_features": self._btts_features,
                "our_teams": self._our_teams,
            },
            path, self._PIPELINE_CACHE_PATH, self._log,
            label="Pipeline cache",
        )

    def load_pipeline_cache(self, path: str | None = None) -> bool:
        """Load cached pipeline DataFrame. Skips feature engineering.

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
        self._ou_features = cache["ou_features"]
        self._ou15_features = cache["ou15_features"]
        self._btts_features = cache["btts_features"]
        self._our_teams = cache["our_teams"]
        return True

    def light_refresh(
        self,
        markets: tuple[str, ...] = ("totals", "btts"),
    ) -> list[dict]:
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
        """Load Championship pipeline data."""
        self._log("Loading Championship pipeline data...")
        self._pipeline_data = run_pipeline(verbose=self.verbose)
        self._full_df = self._pipeline_data["full_df"]

        # Option 3 Step 3: apply sparse-feature flag before filtering to
        # columns actually present in the DataFrame.
        from config import get_active_features
        active_ou = get_active_features(
            CHAMP_ALL_FEATURES, self.use_sparse_features)
        active_ou15 = get_active_features(
            CHAMP_OU15_FEATURES, self.use_sparse_features)
        active_btts = get_active_features(
            CHAMP_BTTS_FEATURES, self.use_sparse_features)
        self._ou_features = [
            f for f in active_ou if f in self._full_df.columns]
        self._ou15_features = [
            f for f in active_ou15 if f in self._full_df.columns]
        self._btts_features = [
            f for f in active_btts if f in self._full_df.columns]
        if not self.use_sparse_features:
            self._log(f"  Sparse-feature filter ACTIVE: "
                       f"O/U 2.5 reduced from {len(CHAMP_ALL_FEATURES)} "
                       f"to {len(self._ou_features)}")

        latest_season = self._full_df["SeasonIndex"].max()
        latest = self._full_df[
            self._full_df["SeasonIndex"] == latest_season]
        self._our_teams = (set(latest["Home_Team"].unique())
                           | set(latest["Away_Team"].unique()))

        self._log(f"O/U 2.5 features: {len(self._ou_features)}")
        self._log(f"O/U 1.5 features: {len(self._ou15_features)}")
        self._log(f"BTTS features: {len(self._btts_features)}")
        self._log(f"Current teams: {len(self._our_teams)}")

    def train(self) -> None:
        """Train all 3-model ensembles on full historical data."""
        if self._full_df is None:
            self.load_data()

        df = self._full_df
        train_df = df[df["SeasonIndex"] >= MIN_TRAIN_SEASON].copy()

        # ── Per-market Dixon-Coles tuning (Option 2 Step 1) ──
        # Tune DC separately for each EFL market. The tune_dc_params_champ
        # signature already supports per-target tuning.
        self._log("Tuning Dixon-Coles (EFL O/U 2.5)...")
        self._dc_kwargs = tune_dc_params_champ(
            train_df, target="Over_2_5", predict_fn="predict_proba_df")
        self._log(f"  O/U 2.5 DC params: {self._dc_kwargs}")

        self._log("Tuning Dixon-Coles (EFL O/U 1.5)...")
        self._ou15_dc_kwargs = tune_dc_params_champ(
            train_df, target="Over_1_5", predict_fn="predict_proba_ou15_df")
        self._log(f"  O/U 1.5 DC params: {self._ou15_dc_kwargs}")

        self._log("Tuning Dixon-Coles (EFL BTTS)...")
        self._btts_dc_kwargs = tune_dc_params_champ(
            train_df, target="BTTS", predict_fn="predict_proba_btts_df")
        self._log(f"  BTTS DC params: {self._btts_dc_kwargs}")

        # Early stopping split
        train_seasons = sorted(train_df["SeasonIndex"].unique())
        last_season = train_seasons[-1]
        es_val_mask = train_df["SeasonIndex"] == last_season
        es_train_mask = ~es_val_mask

        # Base rates from recent 2 seasons
        recent_mask = train_df["SeasonIndex"].isin(train_seasons[-2:])

        # ── O/U 2.5 ──
        self._log("Training O/U 2.5 models...")
        X_tr = train_df.loc[es_train_mask, self._ou_features].values
        y_tr = train_df.loc[es_train_mask, "Over_2_5"].values
        X_val = train_df.loc[es_val_mask, self._ou_features].values
        y_val = train_df.loc[es_val_mask, "Over_2_5"].values

        ou_xgb = train_xgb_champ(X_tr, y_tr, X_val, y_val)
        ou_lgb = train_lgb_champ(X_tr, y_tr, X_val, y_val,
                                  feature_names=self._ou_features)
        ou_dc = DixonColesPredictor(**self._dc_kwargs)
        ou_dc.fit(train_df)

        self._ou_base_rate = train_df.loc[
            recent_mask, "Over_2_5"].mean()
        self._ou_models = {"xgb": ou_xgb, "lgb": ou_lgb, "dc": ou_dc}
        self._log(f"O/U 2.5 base rate: {self._ou_base_rate:.3f}")

        # ── O/U 1.5 ──
        self._log("Training O/U 1.5 models...")
        X_tr_15 = train_df.loc[es_train_mask, self._ou15_features].values
        y_tr_15 = train_df.loc[es_train_mask, "Over_1_5"].values
        X_val_15 = train_df.loc[es_val_mask, self._ou15_features].values
        y_val_15 = train_df.loc[es_val_mask, "Over_1_5"].values

        ou15_xgb = train_xgb_ou15_champ(X_tr_15, y_tr_15,
                                          X_val_15, y_val_15)
        ou15_lgb = train_lgb_ou15_champ(X_tr_15, y_tr_15,
                                          X_val_15, y_val_15,
                                          feature_names=self._ou15_features)
        # Option 2 Step 1: use O/U 1.5-specific DC kwargs rather than the
        # O/U 2.5 ones. Low-score markets prefer smaller half_life + stronger
        # rho.
        dc_15 = DixonColesPredictor(**self._ou15_dc_kwargs)
        dc_15.fit(train_df)

        self._ou15_base_rate = train_df.loc[
            recent_mask, "Over_1_5"].mean()
        self._ou15_models = {"xgb": ou15_xgb, "lgb": ou15_lgb,
                             "dc": dc_15}
        self._log(f"O/U 1.5 base rate: {self._ou15_base_rate:.3f}")

        # ── BTTS ──
        self._log("Training BTTS models...")
        X_tr_b = train_df.loc[es_train_mask, self._btts_features].values
        y_tr_b = train_df.loc[es_train_mask, "BTTS"].values
        X_val_b = train_df.loc[es_val_mask, self._btts_features].values
        y_val_b = train_df.loc[es_val_mask, "BTTS"].values

        btts_xgb = train_xgb_btts_champ(X_tr_b, y_tr_b, X_val_b, y_val_b)
        btts_lgb = train_lgb_btts_champ(X_tr_b, y_tr_b, X_val_b, y_val_b,
                                          feature_names=self._btts_features)
        # Option 2 Step 1: use BTTS-specific DC kwargs
        btts_dc = DixonColesPredictor(**self._btts_dc_kwargs)
        btts_dc.fit(train_df)

        self._btts_base_rate = train_df.loc[recent_mask, "BTTS"].mean()
        self._btts_models = {"xgb": btts_xgb, "lgb": btts_lgb,
                             "dc": btts_dc}
        self._log(f"BTTS base rate: {self._btts_base_rate:.3f}")

        # ── Per-Market Calibration (logit-shift on validation ensemble) ──
        self._log("Computing calibration shifts...")
        self._cal_shifts = {}
        self._val_mean_logits = {}
        val_df = train_df[es_val_mask]

        for mkt_name, models, features, y_col, base_rate, dc_fn in [
            ("ou25", self._ou_models, self._ou_features,
             "Over_2_5", self._ou_base_rate, "predict_proba_df"),
            ("ou15", self._ou15_models, self._ou15_features,
             "Over_1_5", self._ou15_base_rate, "predict_proba_ou15_df"),
            ("btts", self._btts_models, self._btts_features,
             "BTTS", self._btts_base_rate, "predict_proba_btts_df"),
        ]:
            X_v = val_df[features].values
            y_v = val_df[y_col].values
            xgb_v = models["xgb"].predict_proba(X_v)[:, 1]
            lgb_v = models["lgb"].predict_proba(
                pd.DataFrame(X_v, columns=features))[:, 1]
            dc_v = getattr(models["dc"], dc_fn)(val_df)
            ens_v = (xgb_v + lgb_v + dc_v) / 3.0

            mean_logit = np.mean(
                np.log(ens_v / (1 - ens_v + 1e-10)))
            target_logit = np.log(
                base_rate / (1 - base_rate + 1e-10))
            shift = mean_logit - target_logit
            self._cal_shifts[mkt_name] = shift
            self._val_mean_logits[mkt_name] = float(mean_logit)

            # Verify
            corrected = 1 / (1 + np.exp(
                -(np.log(ens_v / (1 - ens_v + 1e-10)) - shift)))
            self._log(f"  {mkt_name}: shift={shift:.4f}, "
                       f"raw_mean={ens_v.mean():.4f}, "
                       f"corrected={corrected.mean():.4f}, "
                       f"actual={y_v.mean():.4f}")

        self._log("All Championship models trained.")

    def _current_season_matches(self) -> pd.DataFrame:
        """Return settled matches from the current EFL season in chronological
        order. Used for matchweek count + regime rolling rates.
        """
        if self._full_df is None or self._full_df.empty:
            return pd.DataFrame()
        current_season = self._full_df["SeasonIndex"].max()
        season_df = self._full_df[
            self._full_df["SeasonIndex"] == current_season]
        settled = season_df[season_df["Home_Goals"].notna()
                            & season_df["Away_Goals"].notna()]
        if "Date" in settled.columns:
            settled = settled.sort_values("Date")
        return settled

    def _current_matchweek_count(self) -> int:
        """Number of settled matches in the current EFL season."""
        return len(self._current_season_matches())

    def _compute_regime_shift(
        self,
        clamp_key: str,
        base_rate: float,
        val_mean_logit: float,
        outcome_fn,
    ) -> tuple[float, float, bool]:
        """Compute a regime-adjusted logit shift for one EFL market.

        Delegates to the shared ``compute_regime_shift`` in
        ``predictor_utils``.  See that function for full documentation.
        """
        return compute_regime_shift(
            self._current_season_matches(),
            clamp_key, base_rate, val_mean_logit, outcome_fn,
        )

    def _get_dc_lambdas(
        self, home_team: str, away_team: str,
    ) -> tuple[float, float]:
        """Extract home/away goal lambdas from the DC model for alt-line evaluation.

        Uses the O/U 2.5 DC model (same as PL predict.py).

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
        home_lambda = np.clip(
            h_att * a_def * dc.mu * sqrt_gamma,
            DC_LAMBDA_MIN, DC_LAMBDA_MAX,
        )
        away_lambda = np.clip(
            a_att * h_def * dc.mu / sqrt_gamma,
            DC_LAMBDA_MIN, DC_LAMBDA_MAX,
        )
        return float(home_lambda), float(away_lambda)

    def _fetch_selective_alt_totals(self, matches: list[dict]) -> None:
        """Fetch per-event alt_totals only for fixtures where the EFL DC
        model has high O/U 1.5 Over conviction.

        Same rationale as the PL counterpart in predict.py — Odds API
        removed bulk alternate_totals (April 2026), per-event is the only
        path and costs 1 credit per call. We gate on a quick DC-Poisson
        estimate of P(total goals > 1.5).

        Mutates ``matches`` in place by enriching candidate fixtures'
        ``bookmakers`` dicts with alt-line entries.
        """
        from config import OU15_FETCH_PROB_THRESHOLD
        from api.odds_api import (
            fetch_alt_totals_for_events,
            merge_alt_totals_into_match,
        )
        from scipy.stats import poisson as _poisson

        if not self._ou15_models or "dc" not in self._ou15_models:
            return  # No OU1.5 DC model loaded — can't gate

        # EFL-specific threshold (lower than PL because EFL fixtures cluster
        # at a lower P(Over 1.5) baseline — see config notes).
        threshold = (
            OU15_FETCH_PROB_THRESHOLD.get("EFL", 0.70)
            if isinstance(OU15_FETCH_PROB_THRESHOLD, dict)
            else float(OU15_FETCH_PROB_THRESHOLD)
        )

        dc = self._ou15_models["dc"]
        sqrt_gamma = float(np.sqrt(dc.gamma))

        candidate_lookup: dict[str, dict] = {}
        for match in matches:
            home, away = _match_champ_teams(match, self._our_teams)
            if home is None or away is None:
                continue
            try:
                h_att = dc.attack_home.get(home, dc.PRIORS["attack_home"])
                a_def = dc.defence_away.get(away, dc.PRIORS["defence_away"])
                a_att = dc.attack_away.get(away, dc.PRIORS["attack_away"])
                h_def = dc.defence_home.get(home, dc.PRIORS["defence_home"])
                home_lam = float(np.clip(h_att * a_def * dc.mu * sqrt_gamma,
                                         DC_LAMBDA_MIN, DC_LAMBDA_MAX))
                away_lam = float(np.clip(a_att * h_def * dc.mu / sqrt_gamma,
                                         DC_LAMBDA_MIN, DC_LAMBDA_MAX))
            except Exception:
                continue
            # P(total <= 1) under independent Poisson — same approximation
            # as PL helper. Tau correction skipped for the gate decision.
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

        # Critical: the per-event alt_totals call uses the module-level
        # SPORT constant, which defaults to "soccer_epl". For EFL we must
        # temporarily swap to "soccer_efl_champ" so the request hits the
        # right endpoint — without this, EFL event IDs are submitted to
        # the PL sport and the API returns no usable data.
        original_sport = odds_api_module.SPORT
        try:
            odds_api_module.SPORT = LEAGUE_CFG["odds_api_sport"]
            alt_data = fetch_alt_totals_for_events(
                list(candidate_lookup.keys()))
        except Exception as e:
            logger.warning("Selective alt_totals fetch failed: %s", e)
            return
        finally:
            odds_api_module.SPORT = original_sport

        for eid, data in alt_data.items():
            if eid in candidate_lookup:
                merge_alt_totals_into_match(candidate_lookup[eid], data)
        self._log(
            f"OU 1.5 selective fetch: enriched {len(alt_data)} EFL matches "
            f"with alt-line data")

    def _predict_3model(
        self,
        fixture_row: pd.Series,
        models: dict,
        features: list[str],
        base_rate: float,
        dc_fn: str = "predict_proba_df",
        market: str = "",
    ) -> dict:
        """Generate calibrated probability from 3 models for one fixture.

        Args:
            fixture_row: Single row from the pipeline DataFrame.
            models: Dict with 'xgb', 'lgb', 'dc' model objects.
            features: Feature column names for GBDTs.
            base_rate: Target base rate for logit-shift calibration.
            dc_fn: DixonColesPredictor method name.
            market: Market key (e.g. "ou25") for looking up calibration shift.

        Returns:
            Dict with per-model probs, ensemble, and per_model array.
        """
        feats = np.array(
            [fixture_row[features].values], dtype=float)

        xgb_raw = float(
            models["xgb"].predict_proba(feats)[:, 1][0])
        lgb_raw = float(
            models["lgb"].predict_proba(
                pd.DataFrame(feats, columns=features))[:, 1][0])
        dc_raw = float(
            getattr(models["dc"], dc_fn)(
                fixture_row.to_frame().T)[0])

        ensemble = (xgb_raw + lgb_raw + dc_raw) / 3.0

        # Apply logit-shift calibration if available for this market
        shift = self._cal_shifts.get(market, 0.0)
        if shift != 0.0:
            logit = np.log(ensemble / (1 - ensemble + 1e-10))
            ensemble = float(1 / (1 + np.exp(-(logit - shift))))

        per_model = np.array([xgb_raw, lgb_raw, dc_raw])

        return {
            "xgb": xgb_raw, "lgb": lgb_raw, "dc": dc_raw,
            "ensemble": ensemble,
            "per_model": per_model,
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
    ) -> dict | None:
        """Thin wrapper around ``staking.decide_bet`` for EFL.

        Defers all bet-evaluation logic to the shared ``decide_bet``
        function. EFL uses the 3-model agreement scale and does not
        currently apply drawdown protection (see Phase 5 action item on
        unifying EFL drawdown with PL).

        Args:
            edge_source: "pinnacle" or "devig" — devig edges are discounted.
            market: Market label (e.g. "ou25", "btts") for confidence multiplier.
            side: Side label (e.g. "over", "under", "yes", "no").

        Returns:
            Bet recommendation dict, or None if no edge.
        """
        return decide_bet(
            model_p, fair_p, odds, per_model, fair_threshold,
            config, edge_source, market, side,
            agree_scale=EFL_AGREE_SCALE,
            # EFL does not apply drawdown protection (defaults = no-op)
            early_season=getattr(self, "_is_early_season", False),
        )

    def _synthesize_promoted_fixture(
        self,
        home: str,
        away: str,
        latest_df: pd.DataFrame,
        *,
        home_missing: bool,
        away_missing: bool,
        home_rows: pd.DataFrame,
        away_rows: pd.DataFrame,
    ) -> pd.Series | None:
        """Synthesise a feature row for a fixture involving promoted team(s).

        For teams with no Championship history in the current season,
        uses league-average feature values as a stand-in.  If only one
        team is missing, the known team's actual features are used for
        their half.

        Args:
            home: Home team name.
            away: Away team name.
            latest_df: Current season's DataFrame.
            home_missing: True if home team has no rows.
            away_missing: True if away team has no rows.
            home_rows: Home team rows (may be empty).
            away_rows: Away team rows (may be empty).

        Returns:
            Synthesised feature row as pd.Series, or None if synthesis
            is not possible (e.g. empty latest_df).
        """
        if latest_df.empty:
            return None

        # Compute league-average features from the current season
        # Use median to be robust against outliers
        league_medians = latest_df.median(numeric_only=True)

        # Start from a template row (any existing row as structure)
        template = latest_df.iloc[-1].copy()

        if home_missing and away_missing:
            # Both promoted — use league medians for everything
            for col in template.index:
                if col in league_medians.index:
                    template[col] = league_medians[col]
            template["Home_Team"] = home
            template["Away_Team"] = away
            template["Home_Promoted"] = 1
            template["Away_Promoted"] = 1
        elif home_missing:
            # Only home team is promoted — use away team's actual data
            fixture_row = away_rows.iloc[-1].copy()
            # Fill Home_ features with league medians
            for col in fixture_row.index:
                if col.startswith("Home_") and col in league_medians.index:
                    fixture_row[col] = league_medians[col]
            fixture_row["Home_Team"] = home
            fixture_row["Home_Promoted"] = 1
            template = fixture_row
        else:
            # Only away team is promoted — use home team's actual data
            fixture_row = home_rows.iloc[-1].copy()
            # Fill Away_ features with league medians
            for col in fixture_row.index:
                if col.startswith("Away_") and col in league_medians.index:
                    fixture_row[col] = league_medians[col]
            fixture_row["Away_Team"] = away
            fixture_row["Away_Promoted"] = 1
            template = fixture_row

        return template

    def generate_recommendations(
        self,
        markets: tuple[str, ...] = ("totals", "btts"),
    ) -> list[dict]:
        """Fetch live Championship odds, generate predictions, return recs.

        Args:
            markets: Which odds markets to fetch ("totals", "btts").

        Returns:
            List of recommendation dicts sorted by edge descending.
        """
        if self._ou_models is None:
            raise RuntimeError("Models not trained. Call train() first.")

        # ── Freshness Gate (ADR 0005) ──
        # League-wide and independent: a stale PL canonical must not block EFL
        # recommendations, and vice versa.
        from freshness import assert_fresh
        assert_fresh("EFL")

        # ── Phase C: regime + two-phase early-season setup ──
        matchweek_count = self._current_matchweek_count()
        self._is_early_season = matchweek_count < EARLY_SEASON_MATCHES
        if self._is_early_season:
            self._log(
                f"Early-season phase ACTIVE (matchweek count={matchweek_count} "
                f"< {EARLY_SEASON_MATCHES}): stricter edge, smaller stakes.")

        # Snapshot original shifts for try/finally restore
        _orig_shifts = dict(self._cal_shifts)

        # Apply regime to ou25 and ou15 (BTTS deliberately excluded)
        for mkt, clamp_key, base_rate, outcome in [
            ("ou25", "ou25_efl", self._ou_base_rate,
             lambda df: (df["Home_Goals"] + df["Away_Goals"] > 2.5)),
            ("ou15", "ou15_efl", self._ou15_base_rate,
             lambda df: (df["Home_Goals"] + df["Away_Goals"] > 1.5)),
        ]:
            val_ml = self._val_mean_logits.get(mkt)
            if val_ml is None:
                # Legacy pickle — keep existing static shift
                continue
            new_shift, adj_rate, is_shifted = self._compute_regime_shift(
                clamp_key=clamp_key, base_rate=base_rate,
                val_mean_logit=val_ml, outcome_fn=outcome,
            )
            self._cal_shifts[mkt] = new_shift
            if is_shifted:
                self._log(
                    f"EFL {mkt} regime SHIFTED: training_rate="
                    f"{base_rate:.3f} → adjusted={adj_rate:.3f}, "
                    f"new_shift={new_shift:.4f}")

        try:
            return self._generate_recommendations_body(markets)
        finally:
            self._cal_shifts = _orig_shifts

    def _generate_recommendations_body(
        self,
        markets: tuple[str, ...],
    ) -> list[dict]:
        """Real body of generate_recommendations, wrapped for try/finally."""
        self._match_analysis = []
        recommendations: list[dict] = []

        # Fetch odds — use cache if available, refresh if possible
        self._log("Fetching Championship odds (The-Odds-API)...")
        matches = _fetch_champ_odds(force_refresh=False, markets=markets)
        self._log(f"Found {len(matches)} upcoming Championship fixtures (pre-filter)")

        if not matches:
            self._log("No Championship fixtures found")
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

        # OddsPapi for supplementary odds.
        # Option β-tight: only fire on the week-ahead snapshot. Matchday
        # morning, KO-1h, and CLV refreshes use Odds-API only — saves
        # ~215 OddsPapi credits/month. The merge code downstream is
        # null-safe when oddspapi_data is empty.
        oddspapi_data: dict[tuple[str, str], dict] = {}
        if self.snapshot_type == "week_ahead":
            try:
                oddspapi_data = _fetch_champ_oddspapi(
                    force_refresh=False, our_teams=self._our_teams)
                self._log(f"OddsPapi: {len(oddspapi_data)} fixtures matched")
            except Exception as e:
                logger.warning("OddsPapi Championship failed: %s", e)
        else:
            self._log(
                f"OddsPapi skipped (snapshot_type={self.snapshot_type!r}, "
                "Option β-tight)")

        # ── Path B: selective per-event alt_totals fetch ──
        # Same rationale as the PL counterpart — Odds API removed bulk
        # alternate_totals; we fetch per-event only when the EFL DC model
        # is highly confident on O/U 1.5 Over.
        self._fetch_selective_alt_totals(matches)

        # Build fixture features
        df = self._full_df
        latest_season = df["SeasonIndex"].max()
        latest_df = df[df["SeasonIndex"] == latest_season].copy()

        for match in matches:
            home, away = _match_champ_teams(match, self._our_teams)
            if home is None or away is None:
                self._log(
                    f"Skipping {match['home_team']} vs "
                    f"{match['away_team']} (team mapping failed)")
                continue

            # Find feature row for this fixture
            fixture_mask = (
                (latest_df["Home_Team"] == home)
                & (latest_df["Away_Team"] == away)
            )
            if fixture_mask.sum() == 0:
                # Synthesise from most recent home/away rows
                home_rows = latest_df[latest_df["Home_Team"] == home]
                away_rows = latest_df[latest_df["Away_Team"] == away]
                home_missing = len(home_rows) == 0
                away_missing = len(away_rows) == 0
                if home_missing or away_missing:
                    # Newly promoted team(s) — synthesise from league averages
                    fixture_row = self._synthesize_promoted_fixture(
                        home, away, latest_df,
                        home_missing=home_missing,
                        away_missing=away_missing,
                        home_rows=home_rows,
                        away_rows=away_rows,
                    )
                    if fixture_row is None:
                        self._log(f"Skipping {home} vs {away} (no data)")
                        continue
                    self._log(
                        f"Synthesised features for {home} vs {away} "
                        f"(promoted team{'s' if home_missing and away_missing else ''})"
                    )
                else:
                    fixture_row = home_rows.iloc[-1].copy()
                    away_row = away_rows.iloc[-1]
                    for col in fixture_row.index:
                        if col.startswith("Away_"):
                            fixture_row[col] = away_row[col]
            else:
                fixture_row = latest_df[fixture_mask].iloc[-1]

            kickoff = match.get("commence_time", "")
            op_data = oddspapi_data.get((home, away))

            # ── O/U 2.5 ──
            best_ou = get_best_odds(match)
            best_ou = self._merge_oddspapi_ou(best_ou, op_data, 2.5)

            if best_ou:
                ou_pred = self._predict_3model(
                    fixture_row, self._ou_models,
                    self._ou_features, self._ou_base_rate,
                    dc_fn="predict_proba_df", market="ou25",
                )
                self._process_ou_market(
                    home, away, kickoff, "ou25", ou_pred, best_ou,
                    self.ou_config, recommendations, match,
                )

            # ── O/U 1.5 ──
            # Check if OddsPapi has O/U 1.5 line
            best_ou15 = self._get_ou15_odds(match, op_data)
            if best_ou15:
                ou15_pred = self._predict_3model(
                    fixture_row, self._ou15_models,
                    self._ou15_features, self._ou15_base_rate,
                    dc_fn="predict_proba_ou15_df", market="ou15",
                )
                self._process_ou_market(
                    home, away, kickoff, "ou15", ou15_pred, best_ou15,
                    self.ou15_config, recommendations, match,
                )

            # ── BTTS ── (skip if not in requested markets)
            best_btts = (get_best_btts_odds(match)
                         if "btts" in markets else None)
            best_btts = self._merge_oddspapi_btts(best_btts, op_data)

            if best_btts:
                btts_pred = self._predict_3model(
                    fixture_row, self._btts_models,
                    self._btts_features, self._btts_base_rate,
                    dc_fn="predict_proba_btts_df", market="btts",
                )
                self._process_btts_market(
                    home, away, kickoff, btts_pred, best_btts,
                    self.btts_config, recommendations, match,
                )

            # ── Alt O/U Lines (DC Poisson only) ──
            # Dedup guard: EFL_ALLOWED_ALT_LINES = {3.5} excludes 1.5 and 2.5
            # which are already handled by dedicated 3-model ensemble paths above.
            all_line_odds = get_best_odds_all_lines(match)

            # Overlay OddsPapi data if available (more bookmakers, more lines)
            if op_data and op_data.get("ou_lines"):
                if all_line_odds is None:
                    all_line_odds = {}
                for line, op_ld in op_data["ou_lines"].items():
                    if line not in all_line_odds:
                        all_line_odds[line] = op_ld
                    else:
                        existing = all_line_odds[line]
                        if op_ld["best_over"] > existing.get("best_over", 0):
                            existing["best_over"] = op_ld["best_over"]
                            existing["best_over_book"] = op_ld["best_over_book"]
                        if op_ld["best_under"] > existing.get("best_under", 0):
                            existing["best_under"] = op_ld["best_under"]
                            existing["best_under_book"] = op_ld["best_under_book"]
                        if op_ld.get("sharp_fair_over") is not None:
                            existing["sharp_fair_over"] = op_ld["sharp_fair_over"]
                            existing["sharp_fair_under"] = op_ld["sharp_fair_under"]
                        existing["n_books"] = max(
                            existing.get("n_books", 0),
                            op_ld.get("n_books", 0),
                        )

            if all_line_odds:
                # Dedup: only lines in EFL_ALLOWED_ALT_LINES (currently {3.5})
                alt_line_odds = {
                    k: v for k, v in all_line_odds.items()
                    if float(k) in EFL_ALLOWED_ALT_LINES
                }

                if alt_line_odds:
                    home_lam, away_lam = self._get_dc_lambdas(home, away)
                    dc = self._ou_models["dc"]

                    # Early-season overrides (mirror PL Phase C pattern)
                    _alt_blend_w = (
                        EARLY_BLEND_WEIGHT if self._is_early_season
                        else self.ou_config.get("blend_weight", 0.35)
                    )
                    _alt_min_edge = (
                        EARLY_MIN_EDGE if self._is_early_season
                        else self.ou_config.get("min_edge", 0.02)
                    )

                    opps = scan_all_lines(
                        home_lambda=home_lam,
                        away_lambda=away_lam,
                        odds_by_line=alt_line_odds,
                        rho=dc.rho,
                        blend_weight=_alt_blend_w,
                        min_edge=_alt_min_edge,
                    )

                    # Record ALL alt-line opportunities in match analysis
                    for opp in opps:
                        line_label = f"ou{opp['line']:.1f}".replace(".", "")
                        fair_odds_val = (
                            1 / opp["fair_prob"]
                            if opp["fair_prob"] > 0 else None
                        )
                        edge_pct = opp["edge"] * 100
                        _line_data = alt_line_odds.get(opp["line"], {})
                        _alt_edge_src = (
                            "pinnacle"
                            if _line_data.get(f"sharp_fair_{opp['side']}")
                            is not None else "devig"
                        )
                        self._match_analysis.append({
                            "home_team": home,
                            "away_team": away,
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
                            "league": "EFL",
                            "bookmaker_odds": _extract_bookmaker_odds(
                                match, line_label, opp["side"],
                            ) if match else {},
                        })

                    value_bets = get_value_bets(
                        opps, min_edge=_alt_min_edge, min_ev=0.0,
                    )

                    if EFL_ALT_LINES_OVER_ONLY:
                        value_bets = [
                            vb for vb in value_bets
                            if vb["side"] == "over"
                        ]

                    for vb in value_bets:
                        _vb_market = (
                            f"ou{vb['line']:.1f}".replace(".", "")
                        )
                        _vb_line_data = alt_line_odds.get(
                            vb["line"], {},
                        )
                        _vb_edge_src = (
                            "pinnacle"
                            if _vb_line_data.get(
                                f"sharp_fair_{vb['side']}")
                            is not None else "devig"
                        )
                        # Apply de-vig discount to edge
                        _vb_edge = vb["edge"]
                        if _vb_edge_src == "devig":
                            _vb_edge *= DEVIG_DISCOUNT
                        # Market/side confidence multiplier
                        _vb_mult = MARKET_MULTIPLIERS.get(
                            (_vb_market, vb["side"]), 1.0,
                        )
                        _vb_stake = vb["kelly"] * _vb_mult
                        # Early-season Kelly reduction
                        if self._is_early_season:
                            _vb_stake *= (EARLY_KELLY_FRACTION / 0.25)

                        recommendations.append({
                            "home_team": home,
                            "away_team": away,
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
                            "confidence": (
                                "high" if _vb_edge > 0.04 else
                                "medium" if _vb_edge > 0.025 else
                                "low"
                            ),
                            "best_bookmaker": vb["book"],
                            "n_books": _vb_line_data.get("n_books", 0),
                            "per_model_probs": {
                                "dc_poisson": vb["model_prob"],
                            },
                            "league": "EFL",
                            "line": vb["line"],
                            "line_type": vb["line_type"],
                        })

        # Option 5 Step 1 + 2c: apply same-match correlation discount and
        # matchday portfolio cap. (``apply_portfolio_constraints`` is
        # imported at module top from ``staking``.)
        pre_stakes = sum(r.get("stake_pct", 0.0) for r in recommendations)
        apply_portfolio_constraints(recommendations)
        post_stakes = sum(r.get("stake_pct", 0.0) for r in recommendations)
        if pre_stakes > 0 and abs(pre_stakes - post_stakes) > 1e-6:
            self._log(
                f"Portfolio constraints applied: total stake "
                f"{pre_stakes:.3f} -> {post_stakes:.3f}")

        recommendations.sort(key=lambda x: x["edge"], reverse=True)
        self._log(
            f"Generated {len(recommendations)} Championship recommendations")
        return recommendations

    def _process_ou_market(
        self,
        home: str, away: str, kickoff: str,
        market: str, pred: dict, best_ou: dict,
        config: dict, recommendations: list[dict],
        match: dict | None = None,
    ) -> None:
        """Evaluate Over/Under market and append recs if edge found."""
        # Fair probabilities — prefer Pinnacle sharp line
        sharp_o = best_ou.get("sharp_fair_over")
        sharp_u = best_ou.get("sharp_fair_under")
        raw_o = 1.0 / best_ou["best_over"]
        raw_u = 1.0 / best_ou["best_under"]
        overround = raw_o + raw_u
        devig_over = raw_o / overround
        devig_under = raw_u / overround

        if sharp_o is not None and sharp_u is not None:
            fair_over = sharp_o
            fair_under = sharp_u
            edge_source = "pinnacle"
        else:
            fair_over = devig_over
            fair_under = devig_under
            edge_source = "devig"

        for side, model_p, fair_p, odds, book in [
            ("over", pred["ensemble"], fair_over,
             best_ou["best_over"], best_ou["best_over_book"]),
            ("under", 1 - pred["ensemble"], fair_under,
             best_ou["best_under"], best_ou["best_under_book"]),
        ]:
            edge_pct = (model_p - fair_p) * 100
            self._match_analysis.append({
                "home_team": home, "away_team": away,
                "kickoff": kickoff, "market": market, "side": side,
                "best_odds": odds, "best_bookmaker": book,
                "model_prob": model_p,
                "fair_odds": 1 / fair_p if fair_p > 0 else None,
                "edge_pct": edge_pct,
                "edge_source": edge_source,
                "confidence": (
                    "high" if edge_pct > 4 else
                    "medium" if edge_pct > 2.5 else
                    "low" if edge_pct > 0 else None
                ),
                "n_books": best_ou.get("n_books"),
                "per_model_probs": {
                    "xgb": pred["xgb"], "lgb": pred["lgb"],
                    "dc": pred["dc"],
                },
                "league": "EFL",
                "bookmaker_odds": _extract_bookmaker_odds(
                    match, market, side) if match else {},
            })

            if side == "over":
                per_model = pred["per_model"]
                fair_threshold = fair_over
            else:
                per_model = 1 - pred["per_model"]
                fair_threshold = fair_under

            bet = self._evaluate_bet(
                model_p, fair_p, odds, per_model, fair_threshold, config,
                edge_source=edge_source, market=market, side=side)
            if bet:
                bet.update({
                    "home_team": home, "away_team": away,
                    "kickoff": kickoff, "market": market,
                    "side": side, "best_bookmaker": book,
                    "n_books": best_ou.get("n_books"),
                    "league": "EFL",
                    "per_model_probs": {
                        "xgb": pred["xgb"], "lgb": pred["lgb"],
                        "dc": pred["dc"],
                    },
                })
                recommendations.append(bet)

    def _process_btts_market(
        self,
        home: str, away: str, kickoff: str,
        pred: dict, best_btts: dict,
        config: dict, recommendations: list[dict],
        match: dict | None = None,
    ) -> None:
        """Evaluate BTTS market and append recs if edge found."""
        # Fair probabilities — prefer Pinnacle sharp line
        sharp_y = best_btts.get("sharp_fair_yes")
        sharp_n = best_btts.get("sharp_fair_no")
        raw_y = 1.0 / best_btts["best_yes"]
        raw_n = 1.0 / best_btts["best_no"]
        overround = raw_y + raw_n
        devig_yes = raw_y / overround
        devig_no = raw_n / overround

        if sharp_y is not None and sharp_n is not None:
            fair_yes = sharp_y
            fair_no = sharp_n
            edge_source = "pinnacle"
        else:
            fair_yes = devig_yes
            fair_no = devig_no
            edge_source = "devig"

        for side, model_p, fair_p, odds, book in [
            ("yes", pred["ensemble"], fair_yes,
             best_btts["best_yes"], best_btts["best_yes_book"]),
            ("no", 1 - pred["ensemble"], fair_no,
             best_btts["best_no"], best_btts["best_no_book"]),
        ]:
            edge_pct = (model_p - fair_p) * 100
            self._match_analysis.append({
                "home_team": home, "away_team": away,
                "kickoff": kickoff, "market": "btts", "side": side,
                "best_odds": odds, "best_bookmaker": book,
                "model_prob": model_p,
                "fair_odds": 1 / fair_p if fair_p > 0 else None,
                "edge_pct": edge_pct,
                "edge_source": edge_source,
                "confidence": (
                    "high" if edge_pct > 4 else
                    "medium" if edge_pct > 2.5 else
                    "low" if edge_pct > 0 else None
                ),
                "n_books": best_btts.get("n_books"),
                "per_model_probs": {
                    "xgb": pred["xgb"], "lgb": pred["lgb"],
                    "dc": pred["dc"],
                },
                "league": "EFL",
                "bookmaker_odds": _extract_bookmaker_odds(
                    match, "btts", side) if match else {},
            })

            if side == "yes":
                per_model = pred["per_model"]
                fair_threshold = fair_yes
            else:
                per_model = 1 - pred["per_model"]
                fair_threshold = fair_no

            bet = self._evaluate_bet(
                model_p, fair_p, odds, per_model, fair_threshold, config,
                edge_source=edge_source, market="btts", side=side)
            if bet:
                bet.update({
                    "home_team": home, "away_team": away,
                    "kickoff": kickoff, "market": "btts",
                    "side": side, "best_bookmaker": book,
                    "n_books": best_btts.get("n_books"),
                    "league": "EFL",
                    "per_model_probs": {
                        "xgb": pred["xgb"], "lgb": pred["lgb"],
                        "dc": pred["dc"],
                    },
                })
                recommendations.append(bet)

    @staticmethod
    def _merge_oddspapi_ou(best_ou: dict | None,
                           op_data: dict | None,
                           line: float) -> dict | None:
        """Merge OddsPapi O/U odds into best_ou dict."""
        if not op_data or not op_data.get("ou_lines"):
            return best_ou

        op_line = op_data["ou_lines"].get(line)
        if not op_line:
            return best_ou

        if best_ou is None:
            return {
                "best_over": op_line["best_over"],
                "best_under": op_line["best_under"],
                "best_over_book": op_line["best_over_book"],
                "best_under_book": op_line["best_under_book"],
                "sharp_fair_over": op_line.get("sharp_fair_over"),
                "sharp_fair_under": op_line.get("sharp_fair_under"),
                "n_books": op_line.get("n_books", 0),
                "consensus_over": 0, "consensus_under": 0,
                "median_over": 0, "median_under": 0,
            }

        # Merge: keep best price across both
        if op_line["best_over"] > best_ou.get("best_over", 0):
            best_ou["best_over"] = op_line["best_over"]
            best_ou["best_over_book"] = op_line["best_over_book"]
        if op_line["best_under"] > best_ou.get("best_under", 0):
            best_ou["best_under"] = op_line["best_under"]
            best_ou["best_under_book"] = op_line["best_under_book"]
        if op_line.get("sharp_fair_over") is not None:
            best_ou["sharp_fair_over"] = op_line["sharp_fair_over"]
            best_ou["sharp_fair_under"] = op_line["sharp_fair_under"]
        best_ou["n_books"] = max(
            best_ou.get("n_books", 0), op_line.get("n_books", 0))

        return best_ou

    @staticmethod
    def _get_ou15_odds(match: dict,
                       op_data: dict | None) -> dict | None:
        """Extract O/U 1.5 odds from The-Odds-API all-lines or OddsPapi.

        O/U 1.5 isn't always on the bulk endpoint — check alt lines
        in the match dict and OddsPapi.
        """
        result = None

        # Check The-Odds-API bookmakers for 1.5 line
        for bm_data in match.get("bookmakers", {}).values():
            lines = bm_data.get("all_lines", {})
            line_data = lines.get(1.5) or lines.get("1.5")
            if line_data:
                over = line_data.get("over", 0)
                under = line_data.get("under", 0)
                if over > 0 and under > 0:
                    if result is None:
                        result = {
                            "best_over": over, "best_under": under,
                            "best_over_book": bm_data.get("title", ""),
                            "best_under_book": bm_data.get("title", ""),
                            "n_books": 1,
                        }
                    else:
                        if over > result["best_over"]:
                            result["best_over"] = over
                            result["best_over_book"] = bm_data.get(
                                "title", "")
                        if under > result["best_under"]:
                            result["best_under"] = under
                            result["best_under_book"] = bm_data.get(
                                "title", "")
                        result["n_books"] += 1

        # Merge OddsPapi O/U 1.5
        if op_data and op_data.get("ou_lines"):
            op_15 = (op_data["ou_lines"].get("1.5")
                     or op_data["ou_lines"].get(1.5))
            if op_15:
                if result is None:
                    result = {
                        "best_over": op_15["best_over"],
                        "best_under": op_15["best_under"],
                        "best_over_book": op_15["best_over_book"],
                        "best_under_book": op_15["best_under_book"],
                        "sharp_fair_over": op_15.get("sharp_fair_over"),
                        "sharp_fair_under": op_15.get("sharp_fair_under"),
                        "n_books": op_15.get("n_books", 0),
                    }
                else:
                    if op_15["best_over"] > result["best_over"]:
                        result["best_over"] = op_15["best_over"]
                        result["best_over_book"] = op_15["best_over_book"]
                    if op_15["best_under"] > result["best_under"]:
                        result["best_under"] = op_15["best_under"]
                        result["best_under_book"] = op_15["best_under_book"]
                    if op_15.get("sharp_fair_over") is not None:
                        result["sharp_fair_over"] = op_15["sharp_fair_over"]
                        result["sharp_fair_under"] = op_15["sharp_fair_under"]

        return result

    @staticmethod
    def _merge_oddspapi_btts(best_btts: dict | None,
                             op_data: dict | None) -> dict | None:
        """Merge OddsPapi BTTS odds into best_btts dict."""
        if not op_data or not op_data.get("btts"):
            return best_btts

        op_btts = op_data["btts"]
        if best_btts is None:
            return {
                "best_yes": op_btts["best_yes"],
                "best_no": op_btts["best_no"],
                "best_yes_book": op_btts["best_yes_book"],
                "best_no_book": op_btts["best_no_book"],
                "sharp_fair_yes": op_btts.get("sharp_fair_yes"),
                "sharp_fair_no": op_btts.get("sharp_fair_no"),
                "n_books": op_btts.get("n_books", 0),
            }

        if op_btts["best_yes"] > best_btts.get("best_yes", 0):
            best_btts["best_yes"] = op_btts["best_yes"]
            best_btts["best_yes_book"] = op_btts["best_yes_book"]
        if op_btts["best_no"] > best_btts.get("best_no", 0):
            best_btts["best_no"] = op_btts["best_no"]
            best_btts["best_no_book"] = op_btts["best_no_book"]
        if op_btts.get("sharp_fair_yes") is not None:
            best_btts["sharp_fair_yes"] = op_btts["sharp_fair_yes"]
            best_btts["sharp_fair_no"] = op_btts["sharp_fair_no"]
        best_btts["n_books"] = max(
            best_btts.get("n_books", 0), op_btts.get("n_books", 0))

        return best_btts


def run_predictions(verbose: bool = True) -> list[dict]:
    """Convenience function: load, train, predict in one call."""
    predictor = ChampionshipPredictor(verbose=verbose)
    predictor.load_data()
    predictor.train()
    return predictor.generate_recommendations()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    predictor = ChampionshipPredictor(verbose=True)
    predictor.load_data()
    predictor.train()
    recs = predictor.generate_recommendations()

    if not recs:
        print("\nNo Championship recommendations "
              "(no fixtures or no edges found)")
    else:
        print(f"\n{'='*95}")
        print(f"CHAMPIONSHIP RECOMMENDATIONS")
        print(f"{'='*95}")
        print(f"{'MATCH':<30} {'MKT':<6} {'SIDE':<6} {'EDGE':>6} "
              f"{'ODDS':>5} {'STAKE':>6} {'CONF':<6} {'AGREE':>5} "
              f"{'BOOK':<15}")
        print(f"{'-'*95}")
        for r in recs:
            fixture = f"{r['home_team']} v {r['away_team']}"
            if len(fixture) > 29:
                fixture = fixture[:26] + "..."
            print(
                f"{fixture:<30} {r['market']:<6} {r['side']:<6} "
                f"{r['edge']:>+5.1%} {r['odds']:>5.2f} "
                f"{r['stake_pct']:>5.1%} {r['confidence']:<6} "
                f"{r['n_agree']:>3}/3  {r['best_bookmaker']:<15}"
            )

    # Summary
    if recs:
        total_stake = sum(r["stake_pct"] for r in recs)
        avg_edge = np.mean([r["edge"] for r in recs])
        print(f"\nTotal stake: {total_stake:.1%} of bankroll")
        print(f"Average edge: {avg_edge:+.2%}")
        print(f"Bets: {len(recs)} across "
              f"{len(set(r['market'] for r in recs))} markets")
