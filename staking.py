"""Shared bet-selection, Kelly staking, and portfolio-constraint logic.

Canonical home for the functions that decide whether a bet happens and
at what stake. Called by both the live prediction path (``predict.py``,
``championship_predict.py``) and the historical ROI validator
(``scripts/roi_validate.py``).

Design decisions:

* ``decide_bet`` is a pure function — no model state, no DB, no logging
  sinks. Every input is explicit. This is the guarantee that the live
  path and the backtest path make the same decision given the same
  inputs.
* ``agree_scale`` is keyword-only and required on ``refined_kelly`` and
  ``decide_bet``. Callers must state whether they are using the 4-model
  PL ensemble or the 3-model EFL ensemble. This makes the silent
  cross-league drift from the pre-extraction code impossible to
  reintroduce.
* ``backtest.py`` keeps a thin backward-compat wrapper around
  ``refined_kelly`` that injects ``PL_AGREE_SCALE`` so peripheral
  modules (corners, alt lines, btts grid search) continue to work
  without edits.
* ``championship_backtest.py`` retains its own local copies of
  ``refined_kelly`` and ``compute_drawdown_factor``. That is an
  intentional preservation decision; the EFL backtest and live paths
  will remain equivalent as long as nobody touches one without the
  other. Flagged for Phase 5 unification review.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Agreement scales (callers pick the one matching their ensemble size)
# ─────────────────────────────────────────────────────────────────────────────

PL_AGREE_SCALE: dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0.70, 3: 0.90, 4: 1.10,
}
"""Agreement-scaling factors for the PL 4-model ensemble (XGB+LGB+DC+LR).

Used by ``refined_kelly`` to damp or amplify the Kelly stake based on how
many of the base models agree the bet is positive-EV.
"""

EFL_AGREE_SCALE: dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0.75, 3: 1.10,
}
"""Agreement-scaling factors for the EFL 3-model ensemble (XGB+LGB+DC).

Slightly more aggressive at 2/3 than PL's 2/4 because there's one fewer
dissenter, and the top-end 3/3 matches PL's 4/4 at 1.10x.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Drawdown protection
# ─────────────────────────────────────────────────────────────────────────────

def compute_drawdown_factor(
    cumulative_bankroll: float, peak_bankroll: float,
) -> float:
    """Reduce stakes when in drawdown.

    Piecewise step function:
      * ≤ 2% drawdown → 1.0   (normal)
      * ≤ 5%         → 0.90
      * ≤ 10%        → 0.75
      * ≤ 15%        → 0.60
      * > 15%        → 0.50   (severe — half stakes)

    Args:
        cumulative_bankroll: Current bankroll level.
        peak_bankroll: Running peak bankroll since the simulation started.

    Returns:
        Multiplier in [0.5, 1.0].
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


# ─────────────────────────────────────────────────────────────────────────────
# Edge shrinkage (Option 5 Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def shrink_edge(raw_edge: float, n_agree: int) -> float:
    """Bayesian edge shrinkage weighted by model agreement.

    **Phase 4a verdict:** load-bearing. Contributes +0.9% to +10.7% DD
    reduction across every cell and is the dominant source of the EFL
    BTTS ROI lift (+6.91pp alone). See ``reports/roi_findings.md`` §
    "Phase 4a — per-toggle contribution matrix" for the per-cell
    numbers.

    Formula: ``shrunk = raw * (n / (n + N_PRIOR))``, where ``n`` is looked
    up from ``EDGE_SHRINKAGE_N_BY_AGREE`` (higher agreement → higher n →
    less shrinkage) and ``N_PRIOR`` represents virtual observations of a
    zero-edge world. Rationale: ensemble AUC sits ~0.55-0.58, so each
    estimated edge carries real noise; Kelly assumes known edge, so
    pulling estimates toward zero proportional to uncertainty prevents
    overbetting.

    Gated by ``config.USE_EDGE_SHRINKAGE``; when False, passes through
    unchanged.

    Args:
        raw_edge: Estimated edge (blended_p - fair_p), can be negative.
        n_agree: Number of base models agreeing on direction.

    Returns:
        Shrunken edge, same sign as raw_edge.
    """
    from config import (
        USE_EDGE_SHRINKAGE, EDGE_SHRINKAGE_N_PRIOR, EDGE_SHRINKAGE_N_BY_AGREE,
    )
    if not USE_EDGE_SHRINKAGE:
        return raw_edge
    n = EDGE_SHRINKAGE_N_BY_AGREE.get(int(n_agree), 1)
    return raw_edge * (n / (n + EDGE_SHRINKAGE_N_PRIOR))


# ─────────────────────────────────────────────────────────────────────────────
# Refined Kelly staking
# ─────────────────────────────────────────────────────────────────────────────

def refined_kelly(
    blended_prob: float,
    odds: float,
    n_agree: int,
    edge: float,
    *,
    agree_scale: dict[int, float],
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.05,
    drawdown_factor: float = 1.0,
) -> float:
    """Confidence-scaled Kelly stake with agreement + edge magnitude + drawdown.

    Five-step pipeline:
      1. Raw Kelly = ``(p * odds - 1) / (odds - 1)``.
      2. Base fraction = ``kelly * kelly_fraction``.
      3. Agreement scaling via ``agree_scale`` (caller-provided — pick
         ``PL_AGREE_SCALE`` or ``EFL_AGREE_SCALE`` to match your ensemble).
      4. Edge-magnitude scaling: > 0.06 → 1.25x, > 0.04 → 1.15x, else 1.0x.
         (Order matters — Option 5 Step 0 fixed a bug where the > 0.04
         branch came first and made > 0.06 unreachable.)
      5. Drawdown scaling and max-stake cap, then a minimum-stake filter
         that zeros out anything below 0.3% (not worth the effort/variance).

    Args:
        blended_prob: Blended P(win) for this bet side.
        odds: Decimal odds.
        n_agree: Number of models agreeing on direction.
        edge: Blended edge over fair-market probability.
        agree_scale: Required — either ``PL_AGREE_SCALE`` (4-model) or
            ``EFL_AGREE_SCALE`` (3-model). Keyword-only so callers have
            to be explicit.
        kelly_fraction: Base Kelly fraction (default 0.25 = quarter-Kelly).
        max_stake_pct: Hard cap on stake as fraction of bankroll.
        drawdown_factor: 1.0 = normal, < 1.0 = in drawdown.

    Returns:
        Stake as fraction of bankroll, or 0.0 if no bet.
    """
    if odds <= 1 or blended_prob <= 0:
        return 0.0

    kelly = (blended_prob * odds - 1) / (odds - 1)
    if kelly <= 0:
        return 0.0

    # 1. Base fraction
    stake = kelly * kelly_fraction

    # 2. Agreement scaling (caller-supplied scale)
    stake *= agree_scale.get(int(n_agree), next(iter(agree_scale.values())))

    # 3. Edge-magnitude scaling (ordered so > 0.06 is reachable)
    if edge > 0.06:
        stake *= 1.25
    elif edge > 0.04:
        stake *= 1.15

    # 4. Drawdown protection
    stake *= drawdown_factor

    # 5. Apply cap + minimum-stake filter
    stake = min(stake, max_stake_pct)
    if stake < 0.003:
        return 0.0
    return stake


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio constraints (Option 5 Step 1 — matchday cap)
# Step 2c (same-match discount) removed post-Phase-5 — empirically dormant.
# See reports/roi_findings.md action A1 for rationale.
# ─────────────────────────────────────────────────────────────────────────────

def apply_portfolio_constraints(
    recommendations: list[dict],
) -> list[dict]:
    """Matchday portfolio cap post-processing over a recommendation list.

    For each distinct matchday (kickoff date), if the aggregate stake
    exceeds ``MAX_MATCHDAY_STAKE_PCT``, every stake is pro-rated down so
    the aggregate equals the cap. Addresses Kelly's independence
    assumption being violated on correlated matchdays.

    Option 5 originally included a same-match correlation discount pass
    here. Phase 4a ROI validation (see ``reports/roi_findings.md``)
    showed it fired zero times across 6 cells × 6 seasons — different
    markets rarely identified positive EV on the same fixture
    simultaneously. Removed in the post-Phase-5 cleanup for simplicity.
    ``config.SAME_MATCH_DISCOUNT`` and ``scripts/measure_correlation.py``
    are retained as historical record; see ``roi_findings.md`` action
    A1 for rationale.

    Args:
        recommendations: list of bet dicts with ``home_team``,
            ``away_team``, ``kickoff`` (ISO string), ``market``, and
            ``stake_pct`` keys.

    Returns:
        The same list (for chaining), with ``stake_pct`` values updated
        in place.
    """
    from config import MAX_MATCHDAY_STAKE_PCT

    if not recommendations:
        return recommendations

    # ── Matchday portfolio cap ──
    day_groups: dict[str, list[dict]] = defaultdict(list)
    for rec in recommendations:
        kickoff = rec.get("kickoff", "") or ""
        day_key = kickoff[:10] if len(kickoff) >= 10 else kickoff
        day_groups[day_key].append(rec)

    for _day, recs in day_groups.items():
        total = sum(float(r.get("stake_pct", 0.0)) for r in recs)
        if total > MAX_MATCHDAY_STAKE_PCT and total > 0:
            scale = MAX_MATCHDAY_STAKE_PCT / total
            for r in recs:
                r["stake_pct"] = float(r.get("stake_pct", 0.0)) * scale

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Model agreement — read side
# ─────────────────────────────────────────────────────────────────────────────

# How many models each league's ensemble runs. The EFL has no LogReg (too few
# matches to fit one), so three is its full complement, not a shortfall.
ENSEMBLE_SIZE: dict[str, int] = {"PL": 4, "EFL": 3}


# ─────────────────────────────────────────────────────────────────────────────
# decide_bet — the shared bet-evaluation core
# ─────────────────────────────────────────────────────────────────────────────

def decide_bet(
    model_p: float,
    fair_p: float,
    odds: float,
    per_model: np.ndarray,
    fair_threshold: float,
    config: dict,
    edge_source: str = "devig",
    market: str = "",
    side: str = "",
    *,
    agree_scale: dict[int, float],
    cumulative_bankroll: float = 1.0,
    peak_bankroll: float = 1.0,
    early_season: bool = False,
) -> Optional[dict]:
    """Pure bet-evaluation function — return bet dict or None.

    Extracted from the ``_evaluate_bet`` methods on ``LivePredictor`` and
    ``ChampionshipPredictor`` so that both the live path and the ROI
    validator exercise identical selection logic.

    Pipeline (in order):
      1. Optional Phase C early-season override (stricter blend + edge + Kelly).
      2. Blend model_p with fair_p; edge = blended - fair.
      3. De-vig discount on edge if edge_source == "devig".
      4. EV and agreement count.
      5. Bayesian edge shrinkage (gated by config.USE_EDGE_SHRINKAGE).
      6. Reject if ev ≤ 0 or edge < min_edge or n_agree < min_agree.
      7. Compute drawdown factor from bankroll numbers.
      8. Refined Kelly sizing with caller-supplied agree_scale.
      9. Apply market/side confidence multiplier.
     10. Reject if final stake ≤ 0.

    Args:
        model_p: Model probability for this side.
        fair_p: Fair market probability (overround-removed).
        odds: Best available decimal odds.
        per_model: Array of per-model probabilities for this side.
        fair_threshold: Probability threshold for agreement count.
        config: Blend/staking dict with keys blend_weight, min_edge,
            min_agree, kelly_fraction, max_stake_pct.
        edge_source: ``"pinnacle"`` or ``"devig"`` — devig edges are
            discounted by ``config.DEVIG_DISCOUNT``.
        market: Market label (e.g. "ou25", "btts") for confidence
            multiplier lookup.
        side: Side label (e.g. "over", "under", "yes", "no").
        agree_scale: Required keyword-only. Use ``PL_AGREE_SCALE`` (4
            models) or ``EFL_AGREE_SCALE`` (3 models).
        cumulative_bankroll: Current bankroll for drawdown calc.
        peak_bankroll: Peak bankroll for drawdown calc.
        early_season: True → use Phase C early-season overrides.

    Returns:
        Bet dict with standard keys, or None if the bet is rejected.
    """
    from config import DEVIG_DISCOUNT, MARKET_MULTIPLIERS
    from config import (
        EARLY_BLEND_WEIGHT, EARLY_MIN_EDGE, EARLY_KELLY_FRACTION,
    )

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)

    # Phase C early-season override (stricter settings)
    if early_season:
        blend_w = EARLY_BLEND_WEIGHT
        min_edge = EARLY_MIN_EDGE
        kelly_fraction = EARLY_KELLY_FRACTION

    blended_p = blend_w * model_p + (1 - blend_w) * fair_p
    edge = blended_p - fair_p

    # De-vig discount for soft-book edges
    if edge_source == "devig":
        edge *= DEVIG_DISCOUNT

    ev = blended_p * odds - 1
    n_agree = int(np.sum(per_model > fair_threshold))

    # Option 5 Step 3: Bayesian edge shrinkage based on model agreement
    edge = shrink_edge(edge, n_agree)

    if ev <= 0 or edge < min_edge or n_agree < min_agree:
        return None

    dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)
    stake = refined_kelly(
        blended_p, odds, n_agree, edge,
        agree_scale=agree_scale,
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        drawdown_factor=dd_factor,
    )

    # Apply market/side confidence multiplier.
    # Phase 4a verdict: load-bearing. Biggest single DD reducer on
    # average (+3.4% to +14.8% per cell, largest on alt-lines). See
    # reports/roi_findings.md.
    mkt_mult = MARKET_MULTIPLIERS.get((market, side), 1.0)
    stake *= mkt_mult

    if stake <= 0:
        return None

    return {
        "model_prob": model_p,
        "blended_prob": blended_p,
        "fair_prob": fair_p,
        "odds": odds,
        "edge": edge,
        "ev": ev,
        "n_agree": n_agree,
        "stake_pct": stake,
        "edge_source": edge_source,
        "market_multiplier": mkt_mult,
        "confidence": (
            "high" if n_agree >= 3 and edge > 0.04
            else "medium" if n_agree >= 2 and edge > 0.025
            else "low"
        ),
    }
