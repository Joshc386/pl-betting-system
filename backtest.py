"""
Backtest: Walk-forward evaluation of model profitability against Bet365 odds.

Architecture:
  - 4 base models: XGBoost + LightGBM + Logistic Regression + Dixon-Coles
  - Per-model logit-shift calibration (anchored to recent 2-season base rate)
  - Model-market blend: final_prob = w*model + (1-w)*market (shrinkage toward bookmaker)
  - Model agreement gating: only bet when N+ models agree on direction vs market
  - Kelly-fraction sizing with configurable cap

Default configuration (optimised via grid search across S19-S25):
  - blend_weight=0.35 (35% model / 65% market)
  - min_edge=0.02 (2% edge threshold)
  - min_agree=2 (2+ of 4 models must agree on bet direction)
  - kelly_fraction=0.25, max_stake=5%
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import run_pipeline
from model import (train_xgb, train_lgb, train_logreg, _fill_nan_median,
                    _clip_scaled, DixonColesPredictor, tune_dc_params)


# ═══════════════════════════════════════════════════════════════════════════════
# Default blend configuration (from grid search)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "blend_weight": 0.35,      # 35% model / 65% market
    "min_edge": 0.02,          # 2% minimum edge to bet
    "min_agree": 2,            # 2+ models must agree on direction
    "kelly_fraction": 0.25,    # quarter-Kelly sizing
    "max_stake_pct": 0.05,     # max 5% bankroll per bet
    "regime_detection": True,  # in-season regime adaptation
    "refined_staking": True,   # confidence-scaled Kelly + drawdown protection
    # Two-phase early-season strategy
    "early_season": True,              # enable early-season phase
    "early_season_matches": 60,        # ~6 matchweeks before regime takes over
    "early_blend_weight": 0.20,        # lean harder on market early (20% model / 80% market)
    "early_min_edge": 0.03,            # require higher edge early (3% vs 2%)
    "early_kelly_fraction": 0.15,      # smaller stakes early (1/6 Kelly vs 1/4)
}

# Preset configurations for comparison
PRESETS = {
    "conservative": {  # fewer bets, higher ROI
        "blend_weight": 0.30, "min_edge": 0.03, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "balanced": {  # good volume + solid ROI
        "blend_weight": 0.35, "min_edge": 0.02, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "aggressive": {  # max volume, lower ROI per bet
        "blend_weight": 0.40, "min_edge": 0.02, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "high_conviction": {  # fewest bets, highest ROI
        "blend_weight": 0.40, "min_edge": 0.05, "min_agree": 2,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
    "pure_model": {  # no blend (V1 baseline for comparison)
        "blend_weight": 1.0, "min_edge": 0.05, "min_agree": 1,
        "kelly_fraction": 0.25, "max_stake_pct": 0.05,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _lr_predict(lr_model, lr_scaler, X):
    """Get LR predictions, handling NaN fill + scaling + clipping."""
    X_filled, _ = _fill_nan_median(X, medians=lr_model._col_medians)
    X_scaled = _clip_scaled(lr_scaler.transform(X_filled))
    return lr_model.predict_proba(X_scaled)[:, 1]


def _calibrate(probs, base_rate):
    """Logit-shift calibration: shift mean logit to match base_rate."""
    logits = np.log(probs / (1 - probs + 1e-10))
    target = np.log(base_rate / (1 - base_rate + 1e-10))
    shift = np.mean(logits) - target
    return 1 / (1 + np.exp(-(logits - shift))), shift


def _calibrate_single(prob, shift):
    """Apply a pre-computed logit shift to a single probability."""
    logit = np.log(prob / (1 - prob + 1e-10))
    return 1 / (1 + np.exp(-(logit - shift)))


# ═══════════════════════════════════════════════════════════════════════════════
# Regime Detection: Rolling base-rate monitor
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """Detect in-season regime shifts by tracking rolling Over 2.5 rate.

    When the rolling rate deviates significantly from the prior, it adjusts
    the calibration anchor to reduce lag. This prevents the S23-style blowup
    where the Over rate spiked to 64.7% but the model still calibrated to ~52%.

    Strategy:
      1. Start season with prior = training base rate
      2. After each matchweek, update rolling Over rate (last N league matches)
      3. If rolling deviates > threshold from prior, blend toward rolling rate
      4. Recalibrate model probabilities using adjusted base rate
    """

    def __init__(self, prior_base_rate, window=40, blend_speed=0.4,
                 trigger_threshold=0.04, min_matches=15,
                 clamp_lo=0.30, clamp_hi=0.75):
        """
        Args:
            prior_base_rate: base rate from training data (e.g. 0.52)
            window: rolling window in league matches (not just bets)
            blend_speed: how quickly to adapt (0=ignore, 1=fully reactive)
            trigger_threshold: deviation from prior needed to activate (e.g. 0.04 = 4pp)
            min_matches: minimum matches before regime detection activates
            clamp_lo: lower bound on adjusted rate (safety rail against
                extreme short-sample deviations)
            clamp_hi: upper bound on adjusted rate
        """
        self.prior = prior_base_rate
        self.window = window
        self.blend_speed = blend_speed
        self.trigger = trigger_threshold
        self.min_matches = min_matches
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.results = []  # list of Over 2.5 outcomes (1 or 0)
        self.current_rate = prior_base_rate

    def update(self, over_result):
        """Add a match result (1=Over, 0=Under) and update regime estimate."""
        self.results.append(over_result)

        if len(self.results) < self.min_matches:
            self.current_rate = self.prior
            return self.current_rate

        # Rolling rate over last N matches
        recent = self.results[-self.window:]
        rolling = np.mean(recent)

        # How far has the environment shifted?
        deviation = rolling - self.prior
        if abs(deviation) > self.trigger:
            # Blend toward rolling rate proportionally to deviation
            self.current_rate = self.prior + self.blend_speed * deviation
        else:
            # Within normal range — stay with prior
            self.current_rate = self.prior

        # Clamp to reasonable range (per-market bounds)
        self.current_rate = np.clip(
            self.current_rate, self.clamp_lo, self.clamp_hi)
        return self.current_rate

    def get_adjusted_base_rate(self):
        """Get current regime-adjusted base rate."""
        return self.current_rate

    def regime_shift_detected(self):
        """Check if we're currently in a detected shift."""
        if len(self.results) < self.min_matches:
            return False
        recent = self.results[-self.window:]
        deviation = abs(np.mean(recent) - self.prior)
        return deviation > self.trigger


# ═══════════════════════════════════════════════════════════════════════════════
# Staking Refinement: confidence-scaled Kelly with drawdown protection.
#
# Canonical definitions live in ``staking.py``. ``shrink_edge``,
# ``apply_portfolio_constraints``, and ``compute_drawdown_factor`` are
# re-exported unchanged (same signatures). ``refined_kelly`` keeps a thin
# backward-compat wrapper that injects ``PL_AGREE_SCALE`` so peripheral
# callers (corners, alt lines, btts grid search) continue to import from
# ``backtest`` without edits.
# ═══════════════════════════════════════════════════════════════════════════════

from staking import (  # noqa: E402 — re-exports for backward compatibility
    shrink_edge,
    apply_portfolio_constraints,
    compute_drawdown_factor,
    refined_kelly as _staking_refined_kelly,
    PL_AGREE_SCALE,
)


def refined_kelly(blended_prob, odds, n_agree, edge,
                  kelly_fraction=0.25, max_stake_pct=0.05,
                  drawdown_factor=1.0):
    """Backward-compat wrapper: PL-scale refined Kelly.

    Injects ``PL_AGREE_SCALE`` automatically. Callers that want the EFL
    scale (or any other) should import ``refined_kelly`` directly from
    ``staking`` and pass ``agree_scale=`` explicitly.
    """
    return _staking_refined_kelly(
        blended_prob, odds, n_agree, edge,
        agree_scale=PL_AGREE_SCALE,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        drawdown_factor=drawdown_factor,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Core backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_season(train_df, test_df, features, config=None, dc_kwargs=None,
                    cumulative_bankroll=1.0, peak_bankroll=1.0):
    """Backtest one season with model-market blend + regime detection + refined staking.

    Returns (bets_df, metrics) tuple.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)
    use_regime = config.get("regime_detection", True)
    use_refined_staking = config.get("refined_staking", True)

    # Two-phase early-season settings
    use_early = config.get("early_season", True)
    early_matches = config.get("early_season_matches", 60)
    early_blend_w = config.get("early_blend_weight", 0.20)
    early_min_edge = config.get("early_min_edge", 0.03)
    early_kelly = config.get("early_kelly_fraction", 0.15)

    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    # Early stopping: use last training season as validation
    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    if dc_kwargs is None:
        dc_kwargs = {}

    # --- Train 4 base models ---
    xgb_m = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val,
                       feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    # --- Raw predictions on test set ---
    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
    dc_raw = dc_m.predict_proba_df(test_df)

    # --- Initial calibration (pre-season base rate) ---
    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    base_rate = (train_df.loc[recent_mask, "Over_2_5"].mean()
                 if recent_mask.sum() >= 100 else y_train.mean())

    # Compute calibration shifts (we'll re-apply these per-match if regime adjusts)
    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, lr_shift = _calibrate(lr_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    # Initial calibrated predictions
    xgb_cal = np.array([_calibrate_single(p, xgb_shift) for p in xgb_raw])
    lgb_cal = np.array([_calibrate_single(p, lgb_shift) for p in lgb_raw])
    lr_cal = np.array([_calibrate_single(p, lr_shift) for p in lr_raw])
    dc_cal = np.array([_calibrate_single(p, dc_shift) for p in dc_raw])

    ensemble_p = (xgb_cal + lgb_cal + lr_cal + dc_cal) / 4.0

    # AUC (using initial calibration — before regime adjustments)
    y_test = test_df["Over_2_5"].values
    try:
        auc = roc_auc_score(y_test, ensemble_p)
    except ValueError:
        auc = 0.5

    # --- Regime detector (params from config, with defaults) ---
    regime = RegimeDetector(
        prior_base_rate=base_rate,
        window=config.get("regime_window", 40),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.04),
        min_matches=config.get("regime_min_matches", 15),
    )
    regime_shifts = 0

    # --- Bet simulation: match-by-match (chronological) ---
    bets = []
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    all_models_over = np.column_stack([xgb_cal, lgb_cal, lr_cal, dc_cal])

    # Map sorted index back to original prediction index
    original_indices = test_df.sort_values("Date").index
    idx_map = {new_i: orig_i for new_i, orig_i in
               enumerate(range(len(test_df)))}
    # Actually need to map via position in the sorted df
    sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        actual = y_test[pred_idx]

        # Update regime detector with this match's actual result
        regime.update(actual)

        # If regime has shifted, recalibrate for this match
        if use_regime and regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            # Recompute shifts with adjusted base rate
            _, xgb_s_adj = _calibrate(xgb_raw, adj_rate)
            _, lgb_s_adj = _calibrate(lgb_raw, adj_rate)
            _, lr_s_adj = _calibrate(lr_raw, adj_rate)
            _, dc_s_adj = _calibrate(dc_raw, adj_rate)

            # Recalibrate this match's predictions
            xgb_p = _calibrate_single(xgb_raw[pred_idx], xgb_s_adj)
            lgb_p = _calibrate_single(lgb_raw[pred_idx], lgb_s_adj)
            lr_p = _calibrate_single(lr_raw[pred_idx], lr_s_adj)
            dc_p = _calibrate_single(dc_raw[pred_idx], dc_s_adj)
            regime_shifts += 1
        else:
            xgb_p = xgb_cal[pred_idx]
            lgb_p = lgb_cal[pred_idx]
            lr_p = lr_cal[pred_idx]
            dc_p = dc_cal[pred_idx]

        per_model = np.array([xgb_p, lgb_p, lr_p, dc_p])
        model_over = per_model.mean()
        model_under = 1 - model_over

        odds_over = row.get("B365Greater2.5", np.nan)
        odds_under = row.get("B365LessThan2.5", np.nan)

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

        # Drawdown factor for refined staking
        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll) if use_refined_staking else 1.0

        # Two-phase: use early-season params if in cold-start period
        is_early = use_early and match_num < early_matches
        active_blend_w = early_blend_w if is_early else blend_w
        active_min_edge = early_min_edge if is_early else min_edge
        active_kelly = early_kelly if is_early else kelly_fraction

        for side, model_p, fair_p, odds in [
            ("over", model_over, fair_over, odds_over),
            ("under", model_under, fair_under, odds_under),
        ]:
            # Blend model with market (early: lean harder on market)
            blended_p = active_blend_w * model_p + (1 - active_blend_w) * fair_p
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            # Model agreement: how many models think this side has edge vs market?
            if side == "over":
                n_agree = int(np.sum(per_model > fair_over))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_under))

            if ev <= 0 or edge < active_min_edge or n_agree < min_agree:
                continue

            # Staking (early: smaller Kelly fraction)
            if use_refined_staking:
                stake = refined_kelly(
                    blended_p, odds, n_agree, edge,
                    kelly_fraction=active_kelly,
                    max_stake_pct=max_stake_pct,
                    drawdown_factor=dd_factor,
                )
            else:
                kelly = (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0
                stake = min(max_stake_pct, max(0, kelly * active_kelly))

            if stake <= 0:
                continue

            won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
            profit = stake * (odds - 1) if won else -stake

            # Update bankroll tracking for drawdown protection
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
                "kelly": (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0,
                "stake_pct": stake,
                "won": won,
                "profit_pct": profit,
                "actual_over": actual,
                "regime_rate": regime.get_adjusted_base_rate(),
                "dd_factor": dd_factor,
                "phase": "early" if is_early else "regime",
            })

    bets_df = pd.DataFrame(bets)

    metrics = {
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
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
        metrics["roi"] = bets_df["profit_pct"].sum() / bets_df["stake_pct"].sum()
        # Per-side breakdown
        for side in ["over", "under"]:
            sb = bets_df[bets_df["side"] == side]
            metrics[f"n_{side}"] = len(sb)
            if len(sb) > 0:
                metrics[f"{side}_roi"] = sb["profit_pct"].sum() / sb["stake_pct"].sum()
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
# Fast replay: re-run betting logic on pre-computed predictions
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_season(train_df, test_df, features, dc_kwargs=None):
    """Train models once and return raw predictions + metadata for replay.

    Returns dict with everything needed by replay_season().
    """
    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    if dc_kwargs is None:
        dc_kwargs = {}

    xgb_m = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val,
                       feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
    dc_raw = dc_m.predict_proba_df(test_df)

    recent_s = sorted(train_seasons)[-2:]
    recent_mask = train_df["SeasonIndex"].isin(recent_s)
    base_rate = (train_df.loc[recent_mask, "Over_2_5"].mean()
                 if recent_mask.sum() >= 100 else y_train.mean())

    _, xgb_shift = _calibrate(xgb_raw, base_rate)
    _, lgb_shift = _calibrate(lgb_raw, base_rate)
    _, lr_shift = _calibrate(lr_raw, base_rate)
    _, dc_shift = _calibrate(dc_raw, base_rate)

    y_test = test_df["Over_2_5"].values
    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

    # Build per-match data for fast replay
    match_data = []
    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        match_data.append({
            "pred_idx": pred_idx,
            "actual": int(y_test[pred_idx]),
            "odds_over": row.get("B365Greater2.5", np.nan),
            "odds_under": row.get("B365LessThan2.5", np.nan),
            "season": row.get("SeasonIndex", 0),
            "home": row.get("Home_Team", ""),
            "away": row.get("Away_Team", ""),
            "date": row.get("Date", ""),
        })

    return {
        "xgb_raw": xgb_raw, "lgb_raw": lgb_raw, "lr_raw": lr_raw, "dc_raw": dc_raw,
        "xgb_shift": xgb_shift, "lgb_shift": lgb_shift, "lr_shift": lr_shift,
        "dc_shift": dc_shift,
        "base_rate": base_rate, "match_data": match_data,
        "season": test_df["SeasonIndex"].iloc[0] if len(test_df) > 0 else 0,
    }


def replay_season(cached, config=None, cumulative_bankroll=1.0, peak_bankroll=1.0,
                   regime_update_fn=None):
    """Replay betting logic on cached predictions. ~1000x faster than backtest_season.

    Args:
        cached: dict from precompute_season()
        config: betting config dict
        cumulative_bankroll, peak_bankroll: bankroll state
        regime_update_fn: optional override for RegimeDetector.update (for permutation test).
                          Signature: fn(detector, over_result) -> adjusted_rate

    Returns: (bets_df, metrics, cumulative_bankroll, peak_bankroll)
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)
    use_regime = config.get("regime_detection", True)
    use_refined_staking = config.get("refined_staking", True)

    xgb_raw = cached["xgb_raw"]
    lgb_raw = cached["lgb_raw"]
    lr_raw = cached["lr_raw"]
    dc_raw = cached["dc_raw"]

    # Initial calibrated predictions
    xgb_cal = np.array([_calibrate_single(p, cached["xgb_shift"]) for p in xgb_raw])
    lgb_cal = np.array([_calibrate_single(p, cached["lgb_shift"]) for p in lgb_raw])
    lr_cal = np.array([_calibrate_single(p, cached["lr_shift"]) for p in lr_raw])
    dc_cal = np.array([_calibrate_single(p, cached["dc_shift"]) for p in dc_raw])

    regime = RegimeDetector(
        prior_base_rate=cached["base_rate"],
        window=config.get("regime_window", 40),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.04),
        min_matches=config.get("regime_min_matches", 15),
    )
    regime_shifts = 0

    bets = []
    for md in cached["match_data"]:
        pred_idx = md["pred_idx"]
        actual = md["actual"]

        # Update regime detector
        if regime_update_fn is not None:
            regime_update_fn(regime, actual)
        else:
            regime.update(actual)

        if use_regime and regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            _, xgb_s_adj = _calibrate(xgb_raw, adj_rate)
            _, lgb_s_adj = _calibrate(lgb_raw, adj_rate)
            _, lr_s_adj = _calibrate(lr_raw, adj_rate)
            _, dc_s_adj = _calibrate(dc_raw, adj_rate)

            xgb_p = _calibrate_single(xgb_raw[pred_idx], xgb_s_adj)
            lgb_p = _calibrate_single(lgb_raw[pred_idx], lgb_s_adj)
            lr_p = _calibrate_single(lr_raw[pred_idx], lr_s_adj)
            dc_p = _calibrate_single(dc_raw[pred_idx], dc_s_adj)
            regime_shifts += 1
        else:
            xgb_p = xgb_cal[pred_idx]
            lgb_p = lgb_cal[pred_idx]
            lr_p = lr_cal[pred_idx]
            dc_p = dc_cal[pred_idx]

        per_model = np.array([xgb_p, lgb_p, lr_p, dc_p])
        model_over = per_model.mean()
        model_under = 1 - model_over

        odds_over = md["odds_over"]
        odds_under = md["odds_under"]
        if pd.isna(odds_over) or pd.isna(odds_under):
            continue
        if odds_over <= 1.0 or odds_under <= 1.0:
            continue

        raw_o = 1.0 / odds_over
        raw_u = 1.0 / odds_under
        overround = raw_o + raw_u
        fair_over = raw_o / overround
        fair_under = raw_u / overround

        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll) if use_refined_staking else 1.0

        for side, model_p, fair_p, odds in [
            ("over", model_over, fair_over, odds_over),
            ("under", model_under, fair_under, odds_under),
        ]:
            blended_p = blend_w * model_p + (1 - blend_w) * fair_p
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            if side == "over":
                n_agree = int(np.sum(per_model > fair_over))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_under))

            if ev <= 0 or edge < min_edge or n_agree < min_agree:
                continue

            if use_refined_staking:
                stake = refined_kelly(blended_p, odds, n_agree, edge,
                                      kelly_fraction=kelly_fraction,
                                      max_stake_pct=max_stake_pct,
                                      drawdown_factor=dd_factor)
            else:
                kelly = (blended_p * odds - 1) / (odds - 1) if odds > 1 else 0
                stake = min(max_stake_pct, max(0, kelly * kelly_fraction))

            if stake <= 0:
                continue

            won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
            profit = stake * (odds - 1) if won else -stake

            actual_stake = stake * cumulative_bankroll
            if won:
                cumulative_bankroll += actual_stake * (odds - 1)
            else:
                cumulative_bankroll -= actual_stake
            peak_bankroll = max(peak_bankroll, cumulative_bankroll)

            bets.append({
                "season": md["season"], "side": side,
                "blended_prob": blended_p, "fair_prob": fair_p,
                "odds": odds, "edge": edge, "ev": ev, "n_agree": n_agree,
                "stake_pct": stake, "won": won, "profit_pct": profit,
            })

    bets_df = pd.DataFrame(bets)
    metrics = {
        "season": cached["season"],
        "n_bets": len(bets_df),
        "regime_shifts": regime_shifts,
        "cumulative_bankroll": cumulative_bankroll,
        "peak_bankroll": peak_bankroll,
    }
    if len(bets_df) > 0:
        metrics["total_profit_pct"] = bets_df["profit_pct"].sum()
        metrics["win_rate"] = bets_df["won"].mean()
        metrics["roi"] = bets_df["profit_pct"].sum() / bets_df["stake_pct"].sum()
    else:
        metrics.update({"total_profit_pct": 0, "win_rate": 0, "roi": 0})

    return bets_df, metrics, cumulative_bankroll, peak_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest(config=None, start_season=19, end_season=25, verbose=True):
    """Walk-forward backtest across multiple seasons.

    Args:
        config: dict with blend_weight, min_edge, min_agree, kelly_fraction,
                max_stake_pct. Defaults to DEFAULT_CONFIG.
        start_season: first test season (19 = 2019/20)
        end_season: last test season (25 = 2025/26)
        verbose: print per-season results
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if verbose:
        print("Loading pipeline data...")
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])

    if verbose:
        print("Tuning Dixon-Coles hyperparameters...")
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)

    bw = config["blend_weight"]
    if verbose:
        print(f"\nBacktest settings:")
        print(f"  Blend: {bw:.0%} model / {1-bw:.0%} market")
        print(f"  Min edge: {config['min_edge']:.1%}")
        print(f"  Min agreement: {config['min_agree']}/4 models")
        print(f"  Kelly fraction: {config['kelly_fraction']}")
        print(f"  Max stake: {config['max_stake_pct']:.1%}")
        print(f"  Seasons: S{start_season} - S{end_season}")

    all_bets = []
    all_metrics = []
    cumulative_bankroll = 1.0
    peak_bankroll = 1.0

    use_regime = config.get("regime_detection", True)
    use_refined = config.get("refined_staking", True)
    if verbose:
        extras = []
        if use_regime:
            extras.append("regime detection ON")
        if use_refined:
            extras.append("refined staking ON")
        if extras:
            print(f"  Enhancements: {', '.join(extras)}")

    if verbose:
        print(f"\n{'Season':>8s} {'Year':>8s} {'AUC':>6s} {'Bets':>5s} "
              f"{'O/U':>8s} {'WinR':>6s} {'ROI':>7s} {'O_ROI':>7s} "
              f"{'U_ROI':>7s} {'Regime':>8s} {'Bankroll':>10s}")
        print("-" * 100)

    for season in range(start_season, end_season + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                           (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()

        has_odds = test_df["B365Greater2.5"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            continue

        bets_df, metrics, cumulative_bankroll, peak_bankroll = backtest_season(
            train_df, test_df, features, config=config, dc_kwargs=dc_kwargs,
            cumulative_bankroll=cumulative_bankroll, peak_bankroll=peak_bankroll,
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
            regime_str = f"{m['regime_shifts']:>2d}x>{m['final_regime_rate']:.0%}"
            print(f"S{m['season']:>5d}  {year:>8s} {m['auc']:>5.3f} {m['n_bets']:>5d} "
                  f"{n_o:>3d}O/{n_u:<3d}U {m['win_rate']:>5.1%} {m['roi']:>+6.1%} "
                  f"{o_roi:>+6.1%} {u_roi:>+6.1%} {regime_str:>8s} "
                  f"{cumulative_bankroll:>9.4f}")

    # Summary
    if all_bets and any(len(b) > 0 for b in all_bets):
        total_bets = pd.concat(all_bets, ignore_index=True)

        if verbose and len(total_bets) > 0:
            total_staked = total_bets["stake_pct"].sum()
            total_profit = total_bets["profit_pct"].sum()
            overall_roi = total_profit / total_staked if total_staked > 0 else 0

            print(f"\n{'='*85}")
            print(f"BACKTEST SUMMARY ({bw:.0%} model / {1-bw:.0%} market blend)")
            print(f"{'='*85}")
            print(f"  Seasons: S{start_season}-S{end_season} ({len(all_metrics)} tested)")
            print(f"  Total bets: {len(total_bets)} ({len(total_bets)/len(all_metrics):.0f}/season)")
            print(f"  Win rate: {total_bets['won'].mean():.1%}")
            print(f"  Avg edge: {total_bets['edge'].mean():.2%}")
            print(f"  Avg odds: {total_bets['odds'].mean():.2f}")
            print(f"  Overall ROI: {overall_roi:+.1%}")
            print(f"  Final bankroll: {cumulative_bankroll:.4f} ({(cumulative_bankroll - 1)*100:+.1f}%)")

            for side in ["over", "under"]:
                sb = total_bets[total_bets["side"] == side]
                if len(sb) > 0:
                    sroi = sb["profit_pct"].sum() / sb["stake_pct"].sum()
                    print(f"\n  {side.upper():>5s}: {len(sb)} bets, "
                          f"win {sb['won'].mean():.1%}, ROI {sroi:+.1%}")

            profitable = sum(1 for m in all_metrics
                             if m.get("total_profit_pct", 0) > 0)
            print(f"\n  Profitable seasons: {profitable}/{len(all_metrics)}")

            # Calibration & regime diagnostics
            print(f"\n  Season diagnostics:")
            for m in all_metrics:
                yr = f"{2000+m['season']}/{str(2001+m['season'])[-2:]}"
                regime_info = ""
                if m.get("regime_shifts", 0) > 0:
                    regime_info = (f" REGIME SHIFT: {m['regime_shifts']}x adjustments, "
                                   f"adapted to {m['final_regime_rate']:.1%}")
                print(f"    {yr}: base_rate={m['base_rate']:.1%}, "
                      f"actual_over={m['actual_over_rate']:.1%}, "
                      f"model_mean={m['model_mean']:.1%}{regime_info}")

            # Max drawdown
            if "profit_pct" in total_bets.columns:
                cum_pl = total_bets["profit_pct"].cumsum()
                running_max = cum_pl.cummax()
                drawdowns = cum_pl - running_max
                max_dd = drawdowns.min()
                print(f"\n  Max drawdown: {max_dd*100:.1f}% of bankroll")

        return total_bets, all_metrics, cumulative_bankroll

    return None, all_metrics, cumulative_bankroll


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        # Compare all presets side by side
        print("=" * 90)
        print("PRESET COMPARISON: Walk-Forward S19-S25")
        print("=" * 90)

        for name, cfg in PRESETS.items():
            print(f"\n\n{'#'*90}")
            print(f"# Preset: {name.upper()}")
            print(f"{'#'*90}")
            run_backtest(config=cfg)

    elif "--preset" in [a.split("=")[0] for a in sys.argv[1:]]:
        # Run specific preset
        preset_name = [a.split("=")[1] for a in sys.argv[1:]
                       if a.startswith("--preset=")][0]
        if preset_name in PRESETS:
            run_backtest(config=PRESETS[preset_name])
        else:
            print(f"Unknown preset: {preset_name}")
            print(f"Available: {', '.join(PRESETS.keys())}")

    elif "--validate" in sys.argv:
        # Overfit validation suite: ablation + sensitivity + permutation
        # Load data ONCE and reuse for all tests
        print("=" * 100)
        print("OVERFIT VALIDATION SUITE")
        print("=" * 100)

        print("\nLoading pipeline data (once for all tests)...")
        data = run_pipeline(verbose=False)
        full_df = data["full_df"]
        features = list(data["features"])
        print("Tuning Dixon-Coles (once for all tests)...")
        tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
        dc_kwargs = tune_dc_params(tune_df)

        def _quick_backtest(cfg, full_df=full_df, features=features,
                            dc_kwargs=dc_kwargs):
            """Run walk-forward backtest using pre-loaded data. Returns (bets_df, metrics_list, bankroll)."""
            all_bets = []
            all_metrics = []
            cum_bank = 1.0
            peak_bank = 1.0
            for season in range(19, 26):
                train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                                   (full_df["SeasonIndex"] < season)].copy()
                test_df = full_df[full_df["SeasonIndex"] == season].copy()
                if test_df["B365Greater2.5"].notna().sum() < 50 or len(train_df) < 500:
                    continue
                bets_df, metrics, cum_bank, peak_bank = backtest_season(
                    train_df, test_df, features, config=cfg, dc_kwargs=dc_kwargs,
                    cumulative_bankroll=cum_bank, peak_bankroll=peak_bank,
                )
                all_bets.append(bets_df)
                all_metrics.append(metrics)
            if all_bets and any(len(b) > 0 for b in all_bets):
                total = pd.concat(all_bets, ignore_index=True)
                return total, all_metrics, cum_bank
            return None, all_metrics, cum_bank

        def _summarise(label, result):
            """Extract summary dict from backtest result tuple."""
            total_bets, metrics_list, bankroll = result
            if total_bets is None or len(total_bets) == 0:
                return None
            staked = total_bets["stake_pct"].sum()
            profit = total_bets["profit_pct"].sum()
            return {
                "n_bets": len(total_bets),
                "roi": profit / staked if staked > 0 else 0,
                "bankroll": bankroll,
                "win_rate": total_bets["won"].mean(),
                "profitable_seasons": sum(1 for m in metrics_list
                                          if m.get("total_profit_pct", 0) > 0),
            }

        # ── 1. Ablation: isolate each improvement ──
        print("\n\n" + "#" * 100)
        print("# TEST 1: ABLATION (which improvement actually helps?)")
        print("#" * 100)

        ablation_configs = {
            "A: baseline (no regime, no refined staking)": {
                **DEFAULT_CONFIG, "regime_detection": False, "refined_staking": False,
            },
            "B: regime detection ONLY": {
                **DEFAULT_CONFIG, "regime_detection": True, "refined_staking": False,
            },
            "C: refined staking ONLY": {
                **DEFAULT_CONFIG, "regime_detection": False, "refined_staking": True,
            },
            "D: both (current default)": {
                **DEFAULT_CONFIG, "regime_detection": True, "refined_staking": True,
            },
        }

        ablation_results = {}
        for name, cfg in ablation_configs.items():
            print(f"\n  Running: {name}...")
            result = _quick_backtest(cfg)
            summary = _summarise(name, result)
            if summary:
                ablation_results[name] = summary
                print(f"    {summary['n_bets']} bets, ROI {summary['roi']:+.1%}, "
                      f"bankroll {summary['bankroll']:.4f}")

        print("\n\n" + "=" * 100)
        print("ABLATION SUMMARY")
        print("=" * 100)
        print(f"{'Config':<50s} {'Bets':>5s} {'WinR':>6s} {'ROI':>8s} {'Bankroll':>10s} {'ProfS':>6s}")
        print("-" * 90)
        for name, r in ablation_results.items():
            print(f"{name:<50s} {r['n_bets']:>5d} {r['win_rate']:>5.1%} "
                  f"{r['roi']:>+7.1%} {r['bankroll']:>9.4f} "
                  f"{r['profitable_seasons']:>4d}/7")

        # ── 2. Parameter Sensitivity: vary regime params ──
        print("\n\n" + "#" * 100)
        print("# TEST 2: REGIME PARAMETER SENSITIVITY")
        print("# (do results collapse if we change regime params?)")
        print("#" * 100)

        sensitivity_results = {}
        param_combos = [
            ("window=20, speed=0.2, trig=0.03", 20, 0.2, 0.03),
            ("window=30, speed=0.3, trig=0.04", 30, 0.3, 0.04),
            ("window=40, speed=0.4, trig=0.04 (default)", 40, 0.4, 0.04),
            ("window=50, speed=0.5, trig=0.04", 50, 0.5, 0.04),
            ("window=60, speed=0.6, trig=0.05", 60, 0.6, 0.05),
            ("window=40, speed=0.2, trig=0.06", 40, 0.2, 0.06),
            ("window=40, speed=0.7, trig=0.03", 40, 0.7, 0.03),
        ]

        for label, window, speed, trigger in param_combos:
            cfg = DEFAULT_CONFIG.copy()
            cfg["regime_window"] = window
            cfg["regime_blend_speed"] = speed
            cfg["regime_trigger"] = trigger
            result = _quick_backtest(cfg)
            summary = _summarise(label, result)
            if summary:
                sensitivity_results[label] = summary
            else:
                sensitivity_results[label] = {
                    "n_bets": 0, "roi": 0, "bankroll": 1.0,
                    "win_rate": 0, "profitable_seasons": 0,
                }
            print(f"  {label}: {sensitivity_results[label]['n_bets']} bets, "
                  f"ROI {sensitivity_results[label]['roi']:+.1%}, "
                  f"bankroll {sensitivity_results[label]['bankroll']:.4f}")

        # ── 3. Permutation Test: regime signal vs random noise ──
        print("\n\n" + "#" * 100)
        print("# TEST 3: PERMUTATION TEST (regime signal)")
        print("# Does feeding the regime detector RANDOM data produce similar results?")
        print("# If real regime signal >> random noise, the detection is real.")
        print("# Using fast replay (models trained once, only regime logic varies)")
        print("#" * 100)

        # Real result (with regime) and baseline (no regime) — reuse from ablation
        real_roi = ablation_results.get(
            "D: both (current default)", {}).get("roi", 0)
        baseline_roi = ablation_results.get(
            "C: refined staking ONLY", {}).get("roi", 0)
        real_lift = real_roi - baseline_roi

        print(f"\n  Real ROI (with regime): {real_roi:+.1%}")
        print(f"  Baseline ROI (no regime): {baseline_roi:+.1%}")
        print(f"  Regime lift: {real_lift:+.1%}")

        # Pre-compute model predictions once per season (the expensive part)
        print(f"\n  Pre-computing model predictions per season (train once)...")
        cached_seasons = []
        for season in range(19, 26):
            train_df = full_df[(full_df["SeasonIndex"] >= 14) &
                               (full_df["SeasonIndex"] < season)].copy()
            test_df = full_df[full_df["SeasonIndex"] == season].copy()
            if test_df["B365Greater2.5"].notna().sum() < 50 or len(train_df) < 500:
                continue
            cached = precompute_season(train_df, test_df, features, dc_kwargs=dc_kwargs)
            cached_seasons.append(cached)
            print(f"    S{season}: {len(cached['match_data'])} matches cached")

        def _replay_all(cfg, regime_update_fn=None):
            """Replay all seasons with cached predictions. Returns (total_bets, bankroll)."""
            all_bets = []
            cum_bank = 1.0
            peak_bank = 1.0
            prof_seasons = 0
            for cached in cached_seasons:
                bets_df, metrics, cum_bank, peak_bank = replay_season(
                    cached, config=cfg, cumulative_bankroll=cum_bank,
                    peak_bankroll=peak_bank, regime_update_fn=regime_update_fn,
                )
                all_bets.append(bets_df)
                if metrics.get("total_profit_pct", 0) > 0:
                    prof_seasons += 1
            if all_bets and any(len(b) > 0 for b in all_bets):
                total = pd.concat(all_bets, ignore_index=True)
                staked = total["stake_pct"].sum()
                return total["profit_pct"].sum() / staked if staked > 0 else 0, cum_bank
            return 0.0, cum_bank

        # Verify replay matches: real regime result
        verify_roi, _ = _replay_all({**DEFAULT_CONFIG, "regime_detection": True,
                                      "refined_staking": True})
        print(f"\n  Replay verification: ROI = {verify_roi:+.1%} (should be ~{real_roi:+.1%})")

        # Permutation: feed regime detector random noise instead of real results
        n_perms = 500
        print(f"\n  Running {n_perms} permutations with shuffled regime signal (fast replay)...")

        perm_rois = []
        perm_lifts = []

        for perm_i in range(n_perms):
            if (perm_i + 1) % 100 == 0:
                print(f"    ...permutation {perm_i + 1}/{n_perms}")

            rng = np.random.RandomState(perm_i + 42)

            def make_random_update(rng_local):
                def random_update(detector, over_result):
                    fake_result = int(rng_local.random() < detector.prior)
                    detector.update(fake_result)
                return random_update

            update_fn = make_random_update(rng)

            cfg = {**DEFAULT_CONFIG, "regime_detection": True,
                   "refined_staking": True}
            perm_roi, _ = _replay_all(cfg, regime_update_fn=update_fn)
            perm_rois.append(perm_roi)
            perm_lifts.append(perm_roi - baseline_roi)

        perm_rois = np.array(perm_rois)
        perm_lifts = np.array(perm_lifts)
        better_count = int(np.sum(perm_lifts >= real_lift))
        p_value = better_count / n_perms

        print(f"\n  {'='*60}")
        print(f"  PERMUTATION RESULTS ({n_perms} shuffles)")
        print(f"  {'='*60}")
        print(f"  Real regime lift:   {real_lift:+.1%}")
        print(f"  Random noise mean lift: {perm_lifts.mean():+.1%}")
        print(f"  Random noise std:   {perm_lifts.std():.1%}")
        print(f"  Times random >= real: {better_count}/{n_perms}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  95th pctile of random lift: {np.percentile(perm_lifts, 95):+.1%}")
        print(f"  99th pctile of random lift: {np.percentile(perm_lifts, 99):+.1%}")

        if p_value < 0.05:
            print(f"\n  RESULT: Regime signal is REAL (p={p_value:.4f})")
            print(f"  The regime detector exploits genuine in-season pattern shifts.")
        elif p_value < 0.10:
            print(f"\n  RESULT: Marginally significant (p={p_value:.4f})")
            print(f"  Some evidence of a real signal, but not conclusive.")
        else:
            print(f"\n  WARNING: NOT significant (p={p_value:.4f})")
            print(f"  The regime lift could be noise -- detector may be overfitting.")

        # Sensitivity summary table
        print("\n\n" + "=" * 100)
        print("SENSITIVITY SUMMARY")
        print("=" * 100)
        print(f"{'Params':<50s} {'Bets':>5s} {'WinR':>6s} {'ROI':>8s} {'Bankroll':>10s} {'ProfS':>6s}")
        print("-" * 90)
        for label, r in sensitivity_results.items():
            print(f"{label:<50s} {r['n_bets']:>5d} {r['win_rate']:>5.1%} "
                  f"{r['roi']:>+7.1%} {r['bankroll']:>9.4f} "
                  f"{r['profitable_seasons']:>4d}/7")

        # Sensitivity stability check
        rois = [r["roi"] for r in sensitivity_results.values()]
        if len(rois) > 1:
            roi_std = np.std(rois)
            roi_range = max(rois) - min(rois)
            print(f"\n  ROI range across params: {roi_range*100:.1f}pp")
            print(f"  ROI std: {roi_std*100:.1f}pp")
            if roi_range < 0.08:
                print(f"  ROBUST: Results stable across parameter choices")
            elif roi_range < 0.15:
                print(f"  MODERATE: Some sensitivity to parameters")
            else:
                print(f"  FRAGILE: Results highly parameter-dependent (overfit risk)")

        print("\n\n" + "=" * 100)
        print("VALIDATION COMPLETE")
        print("=" * 100)
        print("\nKey questions answered:")
        print("  1. Ablation: Which feature(s) actually contribute?")
        print("  2. Sensitivity: Do results survive parameter changes?")
        print("  3. Permutation: Does the regime signal beat random noise?")

    else:
        # Default: run balanced preset
        print("=" * 85)
        print("WALK-FORWARD BACKTEST: Model-Market Blend")
        print("=" * 85)
        run_backtest()
