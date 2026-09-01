"""Does the ensemble's goal expectation run systematically low, across seasons?

Live tracking over 2025/26 showed the model understating Over/Yes by 6.8
percentage points and overstating Under/No by 11.9, on 327 settled
predictions. That is one finding, not two: `roi_validate.replay_league_market`
computes ``model_b = 1.0 - model_a``, so the two sides are exact complements
and "Under/No underperforms" is arithmetically the same statement as
"Over/Yes outperforms". This therefore tests one quantity, one-sided.

The question is whether that gap is normal for this model or new to 2025/26.
The OOF caches cover seasons 19-24 and stop there, so this cannot confirm the
live gap directly. It establishes the baseline the live gap is judged against:

    no gap in 19-24            -> 2025/26 is new; regime or noise, not bias
    same gap, same direction   -> persistent bias; live data confirms it
    smaller gap                -> drift, and the difference is the finding

**Criteria are pre-committed and this file is committed before it is run**,
so the sequence is checkable in the history rather than asserted afterwards.

    persistent : same sign in >= 5 of 6 seasons AND pooled CI excludes zero
    regime     : pooled CI includes zero (so 19-24 is clean)
    noise      : sign flips across seasons with no trend, pooled CI spans zero

No threshold, weight or strategy changes follow from any outcome. This is a
measurement. If it reports persistent bias, that is an input to a separate
decision taken deliberately, not something this script acts on.

Run:
    python scripts/validate_calibration_drift.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One definition of "the model's probability". Reimplementing the calibration
# here is the defect this codebase keeps paying for.
from scripts.roi_validate import _calibrate_single  # noqa: E402

CACHE_DIR = "reports/roi_validate/oof_cache"
MODELS = ("xgb", "lgb", "dc", "lr")
BOOTSTRAP = 5000
SEED = 42


def model_prob(row) -> float:
    """Calibrated ensemble P(side_a), exactly as roi_validate builds it."""
    per_model = []
    for mdl in MODELS:
        raw, shift = row.get(f"{mdl}_prob"), row.get(f"{mdl}_shift")
        if raw is None or pd.isna(raw):
            continue
        per_model.append(_calibrate_single(float(raw), float(shift)))
    return float(np.mean(per_model)) if per_model else np.nan


def gap_ci(model: np.ndarray, actual: np.ndarray) -> tuple[float, float, float]:
    """Mean (model - actual) in points, with a bootstrap 95% interval."""
    rng = np.random.default_rng(SEED)
    point = float(model.mean() - actual.mean())
    draws = [
        float(model[p].mean() - actual[p].mean())
        for p in (rng.integers(0, len(model), len(model))
                  for _ in range(BOOTSTRAP))
    ]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point * 100, lo * 100, hi * 100


def main() -> None:
    frames = []
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, "*.parquet"))):
        league, market = os.path.basename(path).replace(".parquet", "").split("_")
        df = pd.read_parquet(path)
        df["league"], df["market"] = league.upper(), market
        df["model_p"] = df.apply(model_prob, axis=1)
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.model_p.notna() & d.outcome.notna()]
    d["actual"] = d.outcome.astype(float)

    print("Calibration gap = mean model P(side_a) - realised rate of side_a")
    print("Positive = model OVERSTATES side_a (over/yes). "
          "Negative = model UNDERSTATES it.\n")

    print("=== PER SEASON, POOLED ACROSS LEAGUE AND MARKET ===")
    signs = []
    for season, g in d.groupby("season"):
        pt, lo, hi = gap_ci(g.model_p.to_numpy(), g.actual.to_numpy())
        signs.append(np.sign(pt))
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  season {int(season)}  n={len(g):5d}  gap {pt:+6.2f} pp   "
              f"95% CI [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    print("\n=== PER LEAGUE AND MARKET, POOLED ACROSS SEASONS ===")
    for (lg, mkt), g in d.groupby(["league", "market"]):
        pt, lo, hi = gap_ci(g.model_p.to_numpy(), g.actual.to_numpy())
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  {lg:3s} {mkt:5s}  n={len(g):5d}  gap {pt:+6.2f} pp   "
              f"95% CI [{lo:+6.2f}, {hi:+6.2f}]{flag}")

    pt, lo, hi = gap_ci(d.model_p.to_numpy(), d.actual.to_numpy())
    same = int(max((np.array(signs) > 0).sum(), (np.array(signs) < 0).sum()))
    print(f"\n=== POOLED, SEASONS 19-24 ===")
    print(f"  n={len(d)}  gap {pt:+.2f} pp   95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"  seasons sharing the majority sign: {same} of {len(signs)}")

    excludes_zero = not (lo <= 0 <= hi)
    if same >= 5 and excludes_zero:
        verdict = "PERSISTENT BIAS — the 2025/26 gap is not new"
    elif not excludes_zero:
        verdict = ("REGIME OR NOISE — 19-24 is clean, so the 2025/26 gap is "
                   "new to that window")
    else:
        verdict = "INCONSISTENT — pooled gap excludes zero but the sign is not stable"
    print(f"\n  verdict: {verdict}")
    print("\n  Live 2025/26 for comparison: Over/Yes understated by 6.76 pp")
    print("  (mean model 0.638 vs actual 0.706, n=163 settled predictions).")
    print("  Sign convention matches: a NEGATIVE gap here is the same "
          "direction as that observation.")


if __name__ == "__main__":
    main()


def selection_effect() -> None:
    """How much of an apparent edge is the winner's curse?

    `db.log_predictions` stores a prediction only when `edge_pct > 0`, so the
    live predictions table is a *selected* sample: it records the model only
    where the model already disagreed with the market. Conditional on that,
    any model looks over-confident, because the subset is selected for its
    largest errors as much as for its largest insights.

    Measuring the same walk-forward rows under each selection separates the
    model's actual calibration from the artefact of how it is sampled.
    """
    import glob
    import os

    from scripts.roi_validate import _implied_fair_prob

    frames = []
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, "*.parquet"))):
        league, market = os.path.basename(path).replace(".parquet", "").split("_")
        df = pd.read_parquet(path)
        df["league"], df["market"] = league.upper(), market
        df["model_p"] = df.apply(model_prob, axis=1)
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    d = d[d.model_p.notna() & d.outcome.notna()
          & d.odds_a.notna() & d.odds_b.notna()]
    fair = d.apply(lambda r: _implied_fair_prob(r.odds_a, r.odds_b), axis=1)
    d["fair_a"] = [f[0] for f in fair]
    d["fair_b"] = [f[1] for f in fair]
    d = d[d.fair_a.notna()]
    actual_a = d.outcome.astype(float)

    # Both sides, each selectable on its own edge — as log_predictions does.
    long = pd.concat([
        pd.DataFrame({"model": d.model_p, "fair": d.fair_a, "actual": actual_a}),
        pd.DataFrame({"model": 1 - d.model_p, "fair": d.fair_b,
                      "actual": 1 - actual_a}),
    ], ignore_index=True)
    long["edge"] = long.model - long.fair

    print("\n=== SELECTION EFFECT (same rows, different sampling) ===")
    print("Gap = mean model P(side) - realised rate. Positive = model TOO HIGH.")
    for label, sub in (
        ("every game, both sides", long),
        ("positive edge only (what predictions logs)", long[long.edge > 0]),
        ("edge >= 2% (the gate minimum)", long[long.edge >= 0.02]),
        ("negative edge (never logged)", long[long.edge <= 0]),
    ):
        pt, lo, hi = gap_ci(sub.model.to_numpy(), sub.actual.to_numpy())
        star = "" if lo <= 0 <= hi else "  *"
        print(f"  {label:44s} n={len(sub):6d}  gap {pt:+6.2f} pp  "
              f"[{lo:+6.2f},{hi:+6.2f}]{star}")
