"""Shared mechanical utilities for PL and EFL predictor classes.

Extracts the identical boilerplate that both ``predict.py`` (PLPredictor)
and ``championship_predict.py`` (EFLPredictor) duplicate:

- **Pickle save/load** — path handling, makedirs, joblib dump/load,
  logging.  Each predictor still defines *what* goes into the state dict;
  this module handles *how* it gets to/from disk.
- **Regime shift computation** — the RegimeDetector algorithm is
  league-agnostic; only the clamp keys and outcome functions differ.

Zero impact on model output — these are pure infrastructure helpers.
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
