"""Shared mechanical utilities for PL and EFL predictor classes.

Extracts the identical boilerplate that both ``predict.py`` (PLPredictor)
and ``championship_predict.py`` (EFLPredictor) duplicate:

- **Pickle save/load** — path handling, makedirs, joblib dump/load,
  logging.  Each predictor still defines *what* goes into the state dict;
  this module handles *how* it gets to/from disk.
- **Regime shift computation** — the RegimeDetector algorithm is
  league-agnostic; only the clamp keys and outcome functions differ.
- **Validation season selection** — which seasons are big enough to judge a
  model on (ADR 0009).

The first two are pure infrastructure with zero impact on model output.
``seasons_for_validation`` is **not**: it decides the Early-Stopping Season and
the Base Rate window, so it changes what the models learn. It lives here
because both predictors need identical behaviour, not because it is incidental.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from backtest import RegimeDetector
from config import (
    REGIME_CLAMPS,
    REGIME_WINDOW,
    REGIME_BLEND_SPEED,
    REGIME_TRIGGER_THRESHOLD,
    REGIME_MIN_MATCHES,
)


# ── Validation Season Selection (ADR 0009) ─────────────────────────

# A season needs this many fixtures before a model can be judged on it.
# Not a new threshold: walk_forward_cv already skips any fold whose validation
# season holds fewer (model.py:1280, `len(val_df) < 50`). Named here because
# that one is a bare literal and this is the second place needing it.
MIN_VALIDATION_FIXTURES = 50


def seasons_for_validation(
    season_index: pd.Series,
    minimum: int = MIN_VALIDATION_FIXTURES,
    log: Callable[[str], None] | None = None,
) -> list[int]:
    """Seasons holding enough fixtures to validate against, ascending.

    The caller takes ``[-1]`` for the Early-Stopping Season and ``[-2:]`` for
    the Base Rate window. Both previously used every season present, so a
    newly-started season became the sole early-stopping set on its first
    ingested fixture and simultaneously halved the Base Rate's sample.
    """
    counts = season_index.value_counts()
    eligible = sorted(int(s) for s in counts[counts >= minimum].index)

    if eligible:
        return eligible

    # Only reachable on tiny or synthetic datasets — production would need both
    # canonicals to be broken, which earlier checks catch first.
    fallback = sorted(int(s) for s in counts.index)
    if log:
        log(f"  WARNING: no season has >= {minimum} fixtures "
            f"(largest: {int(counts.max()) if len(counts) else 0}). "
            f"Falling back to all {len(fallback)} season(s).")
    return fallback


def refit_at_best_iteration(model, X_all, y_all, feature_names=None):
    """Refit an early-stopped GBDT on the full frame at its chosen tree count.

    Early stopping holds a season back to decide *how many trees*, then leaves
    the model fitted on everything except that season. LogReg and Dixon-Coles
    are fitted on the full frame, so without this step XGB and LGB alone stay a
    season behind their own ensemble (ADR 0009).

    Mirrors `championship_model.py:493-511`, but copies the full parameter set
    via ``get_params()`` rather than re-listing nine hyperparameters by hand —
    a hand-written list silently drops any parameter later added to the
    trainer, which is the failure mode this ADR exists to remove.
    """
    params = dict(model.get_params())

    best = getattr(model, "best_iteration", None)
    if best is None:
        best = getattr(model, "best_iteration_", None)
    # Early stopping may never trigger (the model used every tree it was given).
    # Refit at the full count rather than returning the partially-fitted model —
    # keeping it would silently restore the staleness this function removes.
    params["n_estimators"] = int(best) if best else int(params["n_estimators"])
    if "early_stopping_rounds" in params:
        params["early_stopping_rounds"] = None  # no eval set on the refit

    refit = type(model)(**params)
    X = pd.DataFrame(X_all, columns=feature_names) if feature_names else X_all
    refit.fit(X, y_all)
    return refit


# ── Pickle Save / Load ─────────────────────────────────────────────

def save_pickle(
    data: dict[str, Any],
    path: str | None,
    default_path: str,
    log_fn: Callable[[str], None],
    *,
    label: str = "State",
) -> None:
    """Serialize a dict to disk via joblib.

    Handles path defaulting, parent-directory creation, and logging.
    The caller decides *what* to save; this function handles *how*.

    Args:
        data: Dict to serialize (model weights, cache, etc.).
        path: Explicit override path, or None to use *default_path*.
        default_path: Fallback path when *path* is None.
        log_fn: Logging callable (e.g. ``self._log``).
        label: Human-readable label for the log message
               (e.g. "Trained state", "Pipeline cache").
    """
    dest = path or default_path
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    joblib.dump(data, dest)
    log_fn(f"{label} saved to {dest}")


def load_pickle(
    path: str | None,
    default_path: str,
    log_fn: Callable[[str], None],
    *,
    label: str = "State",
) -> dict[str, Any] | None:
    """Deserialize a joblib pickle from disk.

    Args:
        path: Explicit override path, or None to use *default_path*.
        default_path: Fallback path when *path* is None.
        log_fn: Logging callable (e.g. ``self._log``).
        label: Human-readable label for log messages.

    Returns:
        The loaded dict, or ``None`` if the file does not exist.
    """
    src = path or default_path
    if not os.path.exists(src):
        log_fn(f"No {label.lower()} at {src}")
        return None
    data = joblib.load(src)
    log_fn(f"{label} loaded from {src}")
    return data


# ── Regime Shift ────────────────────────────────────────────────────

def compute_regime_shift(
    settled: pd.DataFrame,
    clamp_key: str,
    base_rate: float,
    val_mean_logit: float,
    outcome_fn: Callable[[pd.DataFrame], pd.Series],
) -> tuple[float, float, bool]:
    """Compute a regime-adjusted logit shift for a single market.

    This is the league-agnostic algorithm shared by both PL and EFL
    predictors.  It feeds current-season settled results into a
    ``RegimeDetector`` and returns an adjusted calibration shift.

    Args:
        settled: Current-season settled matches DataFrame (from
            ``_current_season_matches()``).
        clamp_key: Key into ``REGIME_CLAMPS`` (e.g. "ou25_pl",
            "ou25_efl").  If not present, regime is disabled for
            this market and the raw shift is returned.
        base_rate: Training-set base rate (calibration prior).
        val_mean_logit: Mean logit of validation-set predictions,
            stored during ``train()``.
        outcome_fn: Callable taking the settled DataFrame and
            returning a binary Series of market outcomes
            (1 = Over/Yes, 0 = Under/No).

    Returns:
        ``(adjusted_shift, adjusted_rate, is_shifted)``.
        When regime is disabled or insufficient data exists,
        returns ``(original_shift, base_rate, False)``.
    """
    if clamp_key not in REGIME_CLAMPS:
        target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
        return val_mean_logit - target_logit, base_rate, False

    if len(settled) < REGIME_MIN_MATCHES:
        target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
        return val_mean_logit - target_logit, base_rate, False

    clamp_lo, clamp_hi = REGIME_CLAMPS[clamp_key]
    detector = RegimeDetector(
        prior_base_rate=base_rate,
        window=REGIME_WINDOW,
        blend_speed=REGIME_BLEND_SPEED,
        trigger_threshold=REGIME_TRIGGER_THRESHOLD,
        min_matches=REGIME_MIN_MATCHES,
        clamp_lo=clamp_lo,
        clamp_hi=clamp_hi,
    )
    for o in outcome_fn(settled).astype(int).tolist():
        detector.update(o)

    adjusted_rate = float(detector.get_adjusted_base_rate())
    is_shifted = bool(detector.regime_shift_detected())
    target_logit = np.log(adjusted_rate / (1 - adjusted_rate + 1e-10))
    return float(val_mean_logit - target_logit), adjusted_rate, is_shifted
