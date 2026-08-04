"""
Championship Backtest: Walk-forward evaluation against Bet365 O/U 2.5 odds.

Architecture mirrors backtest.py but adapted for Championship:
  - 3 base models: XGBoost + LightGBM + Dixon-Coles (no LR)
  - Per-model logit-shift calibration (anchored to recent 2-season base rate)
  - Model-market blend: final_prob = w*model + (1-w)*market
  - Model agreement gating: only bet when N+ models agree on direction
  - Kelly-fraction sizing with configurable cap
  - Regime detection for in-season base rate shifts

Key differences from PL backtest:
  - 3 models instead of 4 (agreement scale: {2: 0.75, 3: 1.10})
  - ~552 matches/season (more bets available per season)
  - Bet365 O/U 2.5 odds available from season 2 (2002/03), 89.6% coverage
  - No BTTS odds in CSV (only available via live APIs)
  - Walk-forward starts from season 4 (odds available from season 2)
  - Early-season window = 80 matches (~4 matchweeks of 12 games each)
  - Championship-tuned hyperparameters from championship_model.py
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from championship_pipeline import run_pipeline, CHAMP_ALL_FEATURES
from championship_model import (
    train_xgb_champ, train_lgb_champ,
    tune_dc_params_champ,
    MIN_TRAIN_SEASON, START_VAL_SEASON,
)
from model import DixonColesPredictor


# ═══════════════════════════════════════════════════════════════════════════════
# Default blend configuration
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG: dict = {
    "blend_weight": 0.35,        # 35% model / 65% market
    "min_edge": 0.02,            # 2% minimum edge to bet
    "min_agree": 2,              # 2+ of 3 models must agree
    "kelly_fraction": 0.25,      # quarter-Kelly sizing
    "max_stake_pct": 0.05,       # max 5% bankroll per bet
    "regime_detection": True,    # in-season regime adaptation
    "refined_staking": True,     # confidence-scaled Kelly + drawdown protection
    # Two-phase early-season strategy
    "early_season": True,
    "early_season_matches": 80,  # ~4 matchweeks (Championship has 12 games/mw)
    "early_blend_weight": 0.20,  # lean harder on market early
    "early_min_edge": 0.03,      # require higher edge early
    "early_kelly_fraction": 0.15,
}

# Preset configurations for comparison
PRESETS: dict = {
    "conservative": {
        "blend_weight": 0.30, "min_edge": 0.03, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "balanced": {
        "blend_weight": 0.35, "min_edge": 0.02, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "aggressive": {
        "blend_weight": 0.40, "min_edge": 0.02, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "high_conviction": {
        "blend_weight": 0.40, "min_edge": 0.05, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "pure_model": {
        "blend_weight": 1.0, "min_edge": 0.05, "min_agree": 1,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _calibrate(probs: np.ndarray, base_rate: float
               ) -> tuple[np.ndarray, float]:
    """Logit-shift calibration: shift mean logit to match base_rate."""
    logits = np.log(probs / (1 - probs + 1e-10))
    target = np.log(base_rate / (1 - base_rate + 1e-10))
    shift = np.mean(logits) - target
    return 1 / (1 + np.exp(-(logits - shift))), shift


def _calibrate_single(prob: float, shift: float) -> float:
    """Apply a pre-computed logit shift to a single probability."""
    logit = np.log(prob / (1 - prob + 1e-10))
    return 1 / (1 + np.exp(-(logit - shift)))


# ═══════════════════════════════════════════════════════════════════════════════
# Regime Detection: Rolling base-rate monitor
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Detect in-season regime shifts by tracking rolling Over 2.5 rate.

    Championship has more matches per season (552 vs 380), so we use a
    larger window (50) and higher min_matches (20) for stability.
    """

    def __init__(self, prior_base_rate: float, window: int = 50,
                 blend_speed: float = 0.4, trigger_threshold: float = 0.04,
                 min_matches: int = 20) -> None:
        self.prior = prior_base_rate
        self.window = window
        self.blend_speed = blend_speed
        self.trigger = trigger_threshold
        self.min_matches = min_matches
        self.results: list[int] = []
        self.current_rate = prior_base_rate

    def update(self, over_result: int) -> float:
        """Add a match result (1=Over, 0=Under) and update regime."""
        self.results.append(over_result)

        if len(self.results) < self.min_matches:
            self.current_rate = self.prior
            return self.current_rate

        recent = self.results[-self.window:]
        rolling = np.mean(recent)
        deviation = rolling - self.prior

        if abs(deviation) > self.trigger:
            self.current_rate = self.prior + self.blend_speed * deviation
        else:
            self.current_rate = self.prior

        self.current_rate = np.clip(self.current_rate, 0.30, 0.75)
        return self.current_rate

    def get_adjusted_base_rate(self) -> float:
        """Get current regime-adjusted base rate."""
        return self.current_rate

    def regime_shift_detected(self) -> bool:
        """Check if we're currently in a detected shift."""
        if len(self.results) < self.min_matches:
            return False
        recent = self.results[-self.window:]
        deviation = abs(np.mean(recent) - self.prior)
        return deviation > self.trigger


# ═══════════════════════════════════════════════════════════════════════════════
# Staking: Confidence-scaled Kelly with drawdown protection
# ═══════════════════════════════════════════════════════════════════════════════

def refined_kelly(blended_prob: float, odds: float, n_agree: int,
                  edge: float, kelly_fraction: float = 0.25,
                  max_stake_pct: float = 0.05,
                  drawdown_factor: float = 1.0) -> float:
    """Confidence-scaled Kelly with 3-model agreement weighting.

    Championship uses 3 models (XGB, LGB, DC), so agreement scale is:
      2/3 agree = 0.75x, 3/3 agree = 1.10x

    Args:
        blended_prob: blended P(win) for this bet side.
        odds: decimal odds.
        n_agree: number of models agreeing (out of 3).
        edge: blended edge over fair market.
        kelly_fraction: base Kelly fraction.
        max_stake_pct: hard cap on stake as fraction of bankroll.
        drawdown_factor: 1.0 = normal, <1.0 = in drawdown.

    Returns:
        Stake as fraction of bankroll (0 if no bet).
    """
    if odds <= 1 or blended_prob <= 0:
        return 0.0

    kelly = (blended_prob * odds - 1) / (odds - 1)
    if kelly <= 0:
        return 0.0

    # 1. Base fraction
    stake = kelly * kelly_fraction

    # 2. Agreement scaling (3-model system)
    agree_scale = {0: 0.0, 1: 0.0, 2: 0.75, 3: 1.10}
    stake *= agree_scale.get(n_agree, 0.75)

    # 3. Edge confidence: scale up for larger edges
    if edge > 0.06:
        stake *= 1.25
    elif edge > 0.04:
        stake *= 1.15

    # 4. Drawdown protection
    stake *= drawdown_factor

    # 5. Apply caps
    stake = min(stake, max_stake_pct)

    # 6. Minimum stake filter
    if stake < 0.003:
        return 0.0

    return stake


def compute_drawdown_factor(cumulative_bankroll: float,
                            peak_bankroll: float) -> float:
    """Reduce stakes when in drawdown.

    At 5% drawdown: 90% of normal stakes
    At 10% drawdown: 75% of normal stakes
    At 15%+ drawdown: 60% of normal stakes
    """
    if peak_bankroll <= 0:
        return 1.0
    dd = 1.0 - (cumulative_bankroll / peak_bankroll)
    if dd <= 0.02:
        return 1.0
    elif dd <= 0.05:
        return 0.90
    elif dd <= 0.10:
        return 0.75
    elif dd <= 0.15:
        return 0.60
    else:
        return 0.50


# ═══════════════════════════════════════════════════════════════════════════════
# EFL OOF prediction cache — Phase 3 ROI validator input
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_season_efl(train_df: pd.DataFrame, test_df: pd.DataFrame,
                          features: list[str],
                          dc_kwargs: dict | None = None) -> dict:
    """Train EFL O/U 2.5 models once, return raw predictions + metadata.

    Mirrors ``backtest.precompute_season`` shape so the OOF generator is
    uniform. EFL uses 3 base models (XGB + LGB + DC) — no LR — so the
    resulting dict has ``lr_raw`` / ``lr_shift`` absent from its keys
    (the generator fills NaN for those columns).

    Args:
        train_df: Training data (all prior seasons).
        test_df:  Test season with Odds_Over_2.5 / Odds_Under_2.5 merged.
        features: Champ-specific feature column names.
        dc_kwargs: Dixon-Coles constructor kwargs.

    Returns:
        dict with xgb_raw / lgb_raw / dc_raw arrays, per-model calibration
        shifts, base_rate, per-fixture match_data, season.
    """
    if dc_kwargs is None:
        dc_kwargs = {}

    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    # Early-stopping split: use last training season as val set
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask
    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values
    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(552, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    # --- Train 3 base models (EFL-tuned) ---
    xgb_m = train_xgb_champ(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb_champ(X_es_train, y_es_train, X_es_val, y_es_val,
                            feature_names=features)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Raw predictions on test set ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    dc_raw = dc_m.predict_proba_df(test_df)

    # --- Calibration (recent-2-seasons base rate) ---
    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    base_rate = (train_df.loc[recent_mask, "Over_2_5"].mean()
                 if recent_mask.sum() >= 100 else y_train.mean())
    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    # --- Per-fixture metadata (odds columns already in test_df) ---
    y_test = test_df["Over_2_5"].values
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = (test_df.reset_index(drop=True)
                        .sort_values("Date").index.tolist())
    match_data = []
    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        match_data.append({
            "pred_idx": pred_idx,
            "actual": int(y_test[pred_idx]),
            "odds_over": row.get("Odds_Over_2.5", np.nan),
            "odds_under": row.get("Odds_Under_2.5", np.nan),
            "season": row.get("SeasonIndex", 0),
            "home": row.get("Home_Team", ""),
            "away": row.get("Away_Team", ""),
            "date": row.get("Date", ""),
        })

    return {
        "xgb_raw": xgb_raw, "lgb_raw": lgb_raw, "dc_raw": dc_raw,
        "xgb_shift": xgb_shift, "lgb_shift": lgb_shift, "dc_shift": dc_shift,
        "base_rate": base_rate, "match_data": match_data,
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EFL BTTS OOF prediction cache — Phase 3 Pass 5 (new walk-forward harness)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_btts_season_efl(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    features: list[str], dc_kwargs: dict | None = None,
) -> dict:
    """Train EFL BTTS models once, return raw predictions + metadata.

    No existing EFL BTTS backtest harness existed — this is new code added
    in Phase 3 Pass 5 to support ROI validation. Mirrors the shape of the
    other precompute_* functions for OOF cache uniformity.

    3-model EFL ensemble (XGB + LGB + DC) trained on the BTTS target.
    Uses ``train_xgb_btts_champ`` / ``train_lgb_btts_champ`` from
    ``championship_model`` and ``DixonColesPredictor.predict_proba_btts_df``.

    Args:
        train_df: Training set (all strictly prior seasons).
        test_df:  Test season. Betfair-sourced yes/no odds must be merged
            upstream (by the OOF generator).
        features: BTTS feature column names (``data["btts_features"]``
            from ``championship_pipeline.run_pipeline``).
        dc_kwargs: Dixon-Coles constructor kwargs.

    Returns:
        dict with xgb_raw / lgb_raw / dc_raw / per-model shifts /
        base_rate / match_data / season.
    """
    from championship_model import (
        train_xgb_btts_champ, train_lgb_btts_champ,
    )

    if dc_kwargs is None:
        dc_kwargs = {}

    X_train = train_df[features].values
    y_train = train_df["BTTS"].values

    # Early-stopping split
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask
    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "BTTS"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "BTTS"].values
    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(552, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    # --- Train 3 base models on BTTS target ---
    xgb_m = train_xgb_btts_champ(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb_btts_champ(X_es_train, y_es_train, X_es_val, y_es_val,
                                 feature_names=features)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Raw predictions ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(
        pd.DataFrame(X_test, columns=features))[:, 1]
    dc_raw = dc_m.predict_proba_btts_df(test_df)

    # --- Calibration ---
    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    if recent_mask.sum() >= 100:
        base_rate = train_df.loc[recent_mask, "BTTS"].mean()
    else:
        base_rate = y_train.mean()
    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    # --- Per-fixture metadata (BTTS odds merged upstream by generator) ---
    y_test = test_df["BTTS"].values
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = (test_df.reset_index(drop=True)
                        .sort_values("Date").index.tolist())
    match_data = []
    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        match_data.append({
            "pred_idx": pred_idx,
            "actual": int(y_test[pred_idx]),
            "odds_yes": row.get("BTTSY", np.nan),
            "odds_no": row.get("BTTSN", np.nan),
            "season": row.get("SeasonIndex", 0),
            "home": row.get("Home_Team", ""),
            "away": row.get("Away_Team", ""),
            "date": row.get("Date", ""),
        })

    return {
        "xgb_raw": xgb_raw, "lgb_raw": lgb_raw, "dc_raw": dc_raw,
        "xgb_shift": xgb_shift, "lgb_shift": lgb_shift, "dc_shift": dc_shift,
        "base_rate": base_rate, "match_data": match_data,
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_season(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    features: list[str], config: dict | None = None,
                    dc_kwargs: dict | None = None,
                    cumulative_bankroll: float = 1.0,
                    peak_bankroll: float = 1.0
                    ) -> tuple[pd.DataFrame, dict, float, float]:
    """Backtest one Championship season with 3-model blend + regime detection.

    Trains XGBoost, LightGBM, and Dixon-Coles on training data, generates
    calibrated predictions on the test season, then simulates match-by-match
    betting with model-market blend, agreement gating, and Kelly staking.

    Args:
        train_df: Training data (all prior seasons).
        test_df: Test season data with the Odds_Over/Under_2.5 columns.
        features: Feature column names for GBDTs.
        config: Blend/staking configuration dict.
        dc_kwargs: Dixon-Coles constructor kwargs.
        cumulative_bankroll: Current bankroll level.
        peak_bankroll: Peak bankroll for drawdown calc.

    Returns:
        Tuple of (bets_df, metrics, cumulative_bankroll, peak_bankroll).
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    if dc_kwargs is None:
        dc_kwargs = {}

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)
    use_regime = config.get("regime_detection", True)
    use_refined = config.get("refined_staking", True)

    # Two-phase early-season settings
    use_early = config.get("early_season", True)
    early_matches = config.get("early_season_matches", 80)
    early_blend_w = config.get("early_blend_weight", 0.20)
    early_min_edge = config.get("early_min_edge", 0.03)
    early_kelly = config.get("early_kelly_fraction", 0.15)

    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    # Early stopping split: use last training season as validation
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(552, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    # --- Train 3 base models ---
    xgb_m = train_xgb_champ(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb_champ(X_es_train, y_es_train, X_es_val, y_es_val,
                             feature_names=features)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Raw predictions on test set ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(
        pd.DataFrame(X_test, columns=features))[:, 1]
    dc_raw = dc_m.predict_proba_df(test_df)

    # --- Initial calibration (recent 2-season base rate) ---
    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    base_rate = (train_df.loc[recent_mask, "Over_2_5"].mean()
                 if recent_mask.sum() >= 100 else y_train.mean())

    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    xgb_cal = np.array([_calibrate_single(p, xgb_shift) for p in xgb_raw])
    lgb_cal = np.array([_calibrate_single(p, lgb_shift) for p in lgb_raw])
    dc_cal = np.array([_calibrate_single(p, dc_shift) for p in dc_raw])

    ensemble_p = (xgb_cal + lgb_cal + dc_cal) / 3.0

    # AUC
    y_test = test_df["Over_2_5"].values
    try:
        auc = roc_auc_score(y_test, ensemble_p)
    except ValueError:
        auc = 0.5

    # --- Regime detector ---
    regime = RegimeDetector(
        prior_base_rate=base_rate,
        window=config.get("regime_window", 50),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.04),
        min_matches=config.get("regime_min_matches", 20),
    )
    regime_shifts = 0

    # --- Bet simulation: match-by-match (chronological) ---
    bets: list[dict] = []
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = (test_df.reset_index(drop=True)
                        .sort_values("Date").index.tolist())

    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        actual = y_test[pred_idx]

        # Update regime detector with actual result
        regime.update(actual)

        # If regime shifted, recalibrate for this match
        if use_regime and regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            _, xgb_s = _calibrate(xgb_raw, adj_rate)
            _, lgb_s = _calibrate(lgb_raw, adj_rate)
            _, dc_s = _calibrate(dc_raw, adj_rate)
            xgb_p = _calibrate_single(xgb_raw[pred_idx], xgb_s)
            lgb_p = _calibrate_single(lgb_raw[pred_idx], lgb_s)
            dc_p = _calibrate_single(dc_raw[pred_idx], dc_s)
            regime_shifts += 1
        else:
            xgb_p = xgb_cal[pred_idx]
            lgb_p = lgb_cal[pred_idx]
            dc_p = dc_cal[pred_idx]

        per_model = np.array([xgb_p, lgb_p, dc_p])
        model_over = per_model.mean()
        model_under = 1 - model_over

        odds_over = row.get("Odds_Over_2.5", np.nan)
        odds_under = row.get("Odds_Under_2.5", np.nan)

        if pd.isna(odds_over) or pd.isna(odds_under):
            continue
        if odds_over <= 1.0 or odds_under <= 1.0:
            continue

        # Fair market probabilities (overround removed)
        raw_o = 1.0 / odds_over
        raw_u = 1.0 / odds_under
        overround = raw_o + raw_u
        fair_over = raw_o / overround
        fair_under = raw_u / overround

        # Drawdown factor
        dd_factor = (compute_drawdown_factor(cumulative_bankroll,
                                             peak_bankroll)
                     if use_refined else 1.0)

        # Two-phase: early-season params if in cold-start period
        is_early = use_early and match_num < early_matches
        active_blend_w = early_blend_w if is_early else blend_w
        active_min_edge = early_min_edge if is_early else min_edge
        active_kelly = early_kelly if is_early else kelly_fraction

        for side, model_p, fair_p, odds in [
            ("over", model_over, fair_over, odds_over),
            ("under", model_under, fair_under, odds_under),
        ]:
            blended_p = (active_blend_w * model_p
                         + (1 - active_blend_w) * fair_p)
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            # Model agreement: how many of 3 models agree?
            if side == "over":
                n_agree = int(np.sum(per_model > fair_over))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_under))

            if ev <= 0 or edge < active_min_edge or n_agree < min_agree:
                continue

            # Staking
            if use_refined:
                stake = refined_kelly(
                    blended_p, odds, n_agree, edge,
                    kelly_fraction=active_kelly,
                    max_stake_pct=max_stake_pct,
                    drawdown_factor=dd_factor,
                )
            else:
                kelly = ((blended_p * odds - 1) / (odds - 1)
                         if odds > 1 else 0)
                stake = min(max_stake_pct, max(0, kelly * active_kelly))

            if stake <= 0:
                continue

            won = ((side == "over" and actual == 1)
                   or (side == "under" and actual == 0))
            profit = stake * (odds - 1) if won else -stake

            # Update bankroll
            actual_stake = stake * cumulative_bankroll
            if won:
                cumulative_bankroll += actual_stake * (odds - 1)
            else:
                cumulative_bankroll -= actual_stake
            peak_bankroll = max(peak_bankroll, cumulative_bankroll)

            bets.append({
                "season": row.get("SeasonIndex", 0),
                "home": row.get("Home_Team", ""),
                "away": row.get("Away_Team", ""),
                "date": row.get("Date", ""),
                "side": side,
                "model_prob": model_p,
                "blended_prob": blended_p,
                "fair_prob": fair_p,
                "odds": odds,
                "edge": edge,
                "ev": ev,
                "n_agree": n_agree,
                "kelly": ((blended_p * odds - 1) / (odds - 1)
                          if odds > 1 else 0),
                "stake_pct": stake,
                "won": won,
                "profit_pct": profit,
                "actual_over": actual,
                "regime_rate": regime.get_adjusted_base_rate(),
                "dd_factor": dd_factor,
                "phase": "early" if is_early else "regime",
            })

    bets_df = pd.DataFrame(bets)

    metrics: dict = {
        "season": (test_df["SeasonIndex"].iloc[0]
                   if len(test_df) > 0 else 0),
        "n_matches": len(test_df),
        "n_bets": len(bets_df),
        "auc": auc,
        "base_rate": base_rate,
        "actual_over_rate": float(y_test.mean()),
        "model_mean": float(ensemble_p.mean()),
        "regime_shifts": regime_shifts,
        "final_regime_rate": regime.get_adjusted_base_rate(),
        "cumulative_bankroll": cumulative_bankroll,
        "peak_bankroll": peak_bankroll,
    }

    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["avg_edge"] = bets_df["edge"].mean()
        metrics["avg_odds"] = bets_df["odds"].mean()
        metrics["avg_stake"] = bets_df["stake_pct"].mean()
        metrics["roi"] = (bets_df["profit_pct"].sum()
                          / bets_df["stake_pct"].sum())
        for side in ["over", "under"]:
            sb = bets_df[bets_df["side"] == side]
            metrics[f"n_{side}"] = len(sb)
            if len(sb) > 0:
                metrics[f"{side}_roi"] = (sb["profit_pct"].sum()
                                          / sb["stake_pct"].sum())
                metrics[f"{side}_win_rate"] = sb["won"].mean()
            else:
                metrics[f"{side}_roi"] = 0
                metrics[f"{side}_win_rate"] = 0
    else:
        metrics.update({
            "total_profit_pct": 0, "win_rate": 0, "avg_edge": 0,
            "avg_odds": 0, "avg_stake": 0, "roi": 0,
            "n_over": 0, "n_under": 0,
            "over_roi": 0, "under_roi": 0,
            "over_win_rate": 0, "under_win_rate": 0,
        })

    return bets_df, metrics, cumulative_bankroll, peak_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(config: dict | None = None, start_season: int = 4,
                 end_season: int = 24, verbose: bool = True
                 ) -> tuple[pd.DataFrame | None, list[dict], float]:
    """Walk-forward backtest across Championship seasons.

    Bet365 O/U 2.5 odds available from season 2 (2002/03). We start from
    season 4 to have at least 4 seasons (~2200 matches) of training data.

    Args:
        config: blend/staking configuration dict.
        start_season: first test season (4 = 2004/05).
        end_season: last test season (24 = 2024/25).
        verbose: print per-season results.

    Returns:
        Tuple of (all_bets_df, season_metrics, final_bankroll).
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if verbose:
        print("Loading Championship pipeline data...")
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])

    if verbose:
        print("Tuning Dixon-Coles hyperparameters...")
    dc_kwargs = tune_dc_params_champ(full_df)

    bw = config.get("blend_weight", 0.35)
    if verbose:
        print(f"\nChampionship Backtest settings:")
        print(f"  Blend: {bw:.0%} model / {1-bw:.0%} market")
        print(f"  Min edge: {config.get('min_edge', 0.02):.1%}")
        print(f"  Min agreement: {config.get('min_agree', 2)}/3 models")
        print(f"  Kelly fraction: {config.get('kelly_fraction', 0.25)}")
        print(f"  Max stake: {config.get('max_stake_pct', 0.05):.1%}")
        print(f"  Seasons: S{start_season} - S{end_season}")

        extras = []
        if config.get("regime_detection", True):
            extras.append("regime detection ON")
        if config.get("refined_staking", True):
            extras.append("refined staking ON")
        if extras:
            print(f"  Enhancements: {', '.join(extras)}")

    all_bets: list[pd.DataFrame] = []
    all_metrics: list[dict] = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    if verbose:
        print(f"\n{'Season':>8s} {'Year':>8s} {'AUC':>6s} {'Bets':>5s} "
              f"{'O/U':>8s} {'WinR':>6s} {'ROI':>7s} {'O_ROI':>7s} "
              f"{'U_ROI':>7s} {'Regime':>8s} {'Bankroll':>10s}")
        print("-" * 100)

    for season in range(start_season, end_season + 1):
        train_df = full_df[
            (full_df["SeasonIndex"] >= MIN_TRAIN_SEASON)
            & (full_df["SeasonIndex"] < season)
        ].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_odds = test_df["Odds_Over_2.5"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            if verbose:
                print(f"  S{season:>5d}  skipped (odds={has_odds}, "
                      f"train={len(train_df)})")
            continue

        bets_df, metrics, cumulative_bankroll, peak_bankroll = (
            backtest_season(
                train_df, test_df, features, config=config,
                dc_kwargs=dc_kwargs,
                cumulative_bankroll=cumulative_bankroll,
                peak_bankroll=peak_bankroll,
            )
        )

        all_bets.append(bets_df)
        all_metrics.append(metrics)

        if verbose:
            m = metrics
            year = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
            n_o = m.get("n_over", 0)
            n_u = m.get("n_under", 0)
            o_roi = m.get("over_roi", 0)
            u_roi = m.get("under_roi", 0)
            regime_str = (
                f"{m['regime_shifts']:>2d}x>{m['final_regime_rate']:.0%}"
                if m["regime_shifts"] > 0
                else "  stable"
            )
            print(
                f"S{m['season']:>5d}  {year:>8s} {m['auc']:>5.3f} "
                f"{m['n_bets']:>5d} {n_o:>3d}O/{n_u:<3d}U "
                f"{m.get('win_rate', 0):>5.1%} "
                f"{m.get('roi', 0):>+6.1%} "
                f"{o_roi:>+6.1%} {u_roi:>+6.1%} {regime_str:>8s} "
                f"{cumulative_bankroll:>9.4f}"
            )

    # Summary
    if all_bets and any(len(b) > 0 for b in all_bets):
        total_bets = pd.concat(all_bets, ignore_index=True)

        if verbose and len(total_bets) > 0:
            total_staked = total_bets["stake_pct"].sum()
            total_profit = total_bets["profit_pct"].sum()
            overall_roi = (total_profit / total_staked
                           if total_staked > 0 else 0)

            print(f"\n{'='*85}")
            print(f"CHAMPIONSHIP BACKTEST SUMMARY "
                  f"({bw:.0%} model / {1-bw:.0%} market blend)")
            print(f"{'='*85}")
            print(f"  Seasons: S{start_season}-S{end_season} "
                  f"({len(all_metrics)} tested)")
            print(f"  Total bets: {len(total_bets)} "
                  f"({len(total_bets)/max(len(all_metrics),1):.0f}"
                  f"/season)")
            print(f"  Win rate: {total_bets['won'].mean():.1%}")
            print(f"  Avg edge: {total_bets['edge'].mean():.2%}")
            print(f"  Avg odds: {total_bets['odds'].mean():.2f}")
            print(f"  Overall ROI: {overall_roi:+.1%}")
            print(f"  Final bankroll: {cumulative_bankroll:.4f} "
                  f"({(cumulative_bankroll - 1)*100:+.1f}%)")

            for side in ["over", "under"]:
                sb = total_bets[total_bets["side"] == side]
                if len(sb) > 0:
                    sroi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"\n  {side.upper():>5s}: {len(sb)} bets, "
                          f"win {sb['won'].mean():.1%}, ROI {sroi:+.1%}")

            profitable = sum(1 for m in all_metrics
                             if m.get("total_profit_pct", 0) > 0)
            print(f"\n  Profitable seasons: "
                  f"{profitable}/{len(all_metrics)}")

            # Calibration & regime diagnostics
            print(f"\n  Season diagnostics:")
            for m in all_metrics:
                yr = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
                regime_info = ""
                if m.get("regime_shifts", 0) > 0:
                    regime_info = (
                        f" REGIME SHIFT: {m['regime_shifts']}x, "
                        f"adapted to {m['final_regime_rate']:.1%}")
                print(f"    {yr}: base={m['base_rate']:.1%}, "
                      f"actual={m['actual_over_rate']:.1%}, "
                      f"model={m['model_mean']:.1%}{regime_info}")

            # Max drawdown
            if "profit_pct" in total_bets.columns:
                cum_pl = total_bets["profit_pct"].cumsum()
                running_max = cum_pl.cummax()
                drawdowns = cum_pl - running_max
                max_dd = drawdowns.min()
                print(f"\n  Max drawdown: {max_dd*100:.1f}% of bankroll")

        return total_bets, all_metrics, cumulative_bankroll

    return None, all_metrics, cumulative_bankroll


def compare_presets(start_season: int = 4, end_season: int = 24) -> None:
    """Run all presets side by side for comparison."""
    print("=" * 90)
    print("CHAMPIONSHIP PRESET COMPARISON: Walk-Forward "
          f"S{start_season}-S{end_season}")
    print("=" * 90)

    for name, cfg in PRESETS.items():
        print(f"\n\n{'#'*90}")
        print(f"# Preset: {name.upper()}")
        print(f"{'#'*90}")
        run_backtest(config=cfg, start_season=start_season,
                     end_season=end_season)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_presets()

    elif "--preset" in [a.split("=")[0] for a in sys.argv[1:]]:
        preset_name = [a.split("=")[1] for a in sys.argv[1:]
                       if a.startswith("--preset=")][0]
        if preset_name in PRESETS:
            run_backtest(config=PRESETS[preset_name])
        else:
            print(f"Unknown preset: {preset_name}")
            print(f"Available: {', '.join(PRESETS.keys())}")

    else:
        run_backtest()
