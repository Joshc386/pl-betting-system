"""Measure the squad-decay weight for a relegated side's PL form (ADR 0011).

This is the evidence behind a *negative* result, kept reproducible rather
than kept in production. ADR 0011 proposed blending a relegated side's own
PL form into its Division Movement Seed:

    seed = w * (PL form, rebased onto the EFL scale) + (1 - w) * cohort

and pre-committed to dropping the blend if ``w``'s confidence interval
included zero. It did. The blend is not in the seed; the measurement lives
here so a future session can re-run it as events accumulate — three a
season — instead of re-deriving the proposal from scratch.

Run:
    python scripts/measure_seed_weight.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from division_movement import (  # noqa: E402
    RELEGATED,
    _arrival_events,
    _cohort_teams,
)
from api.team_mapping import normalize  # noqa: E402

BOOTSTRAP_RESAMPLES = 5000
SEED = 42
SEED_WINDOW = 5


def _total_goals_rate(df, team, season, limit=None):
    """A side's mean total goals per match, optionally over its first few."""
    key = normalize(team)
    rows = df[(df["SeasonIndex"] == season)
              & ((df["Home_Team"].map(normalize) == key)
                 | (df["Away_Team"].map(normalize) == key))]
    if rows.empty:
        return None
    rows = rows.sort_values("Date")
    if limit is not None:
        rows = rows.head(limit)
    return float((rows["Home_Goals"] + rows["Away_Goals"]).mean())


def _league_rate(df, season):
    rows = df[df["SeasonIndex"] == season]
    if rows.empty:
        return None
    return float((rows["Home_Goals"] + rows["Away_Goals"]).mean())


def _cohort_rate(ef_df, season):
    cohort = set(_cohort_teams(ef_df, season, RELEGATED))
    rows = ef_df[(ef_df["SeasonIndex"] == season)
                 & (ef_df["Home_Team"].isin(cohort)
                    | ef_df["Away_Team"].isin(cohort))]
    if rows.empty:
        return None
    return float((rows["Home_Goals"] + rows["Away_Goals"]).mean())


def anchors(ef_df, pl_df, team, season):
    """(x, r, y) — rebased PL rate, cohort rate, what the side actually did."""
    pl_rate = _total_goals_rate(pl_df, team, season - 1)
    pl_avg = _league_rate(pl_df, season - 1)
    ef_avg = _league_rate(ef_df, season - 1)
    r = _cohort_rate(ef_df, season - 1)
    y = _total_goals_rate(ef_df, team, season, limit=SEED_WINDOW)
    if None in (pl_rate, pl_avg, ef_avg, r, y) or not pl_avg:
        return None
    return pl_rate / pl_avg * ef_avg, r, y


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ef = pd.read_csv(os.path.join(root, "CompleteDSChamp_CSV.csv"),
                     low_memory=False)
    pl = pd.read_csv(os.path.join(root, "CompleteDSPL_CSV.csv"),
                     low_memory=False)

    events = [e for e in _arrival_events(ef, pl, 26) if e[2] == RELEGATED]
    triples = [a for a in (anchors(ef, pl, t, s) for s, t, _ in events) if a]
    x = np.array([t[0] for t in triples])
    r = np.array([t[1] for t in triples])
    y = np.array([t[2] for t in triples])

    num, den = (x - r) * (y - r), (x - r) ** 2
    w = num.sum() / den.sum()

    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, len(triples), len(triples))
        if den[pick].sum() > 0:
            draws.append(num[pick].sum() / den[pick].sum())
    low, high = np.percentile(draws, [2.5, 97.5])

    print(f"relegation events      : {len(triples)} of {len(events)}")
    print(f"mean rebased PL rate X : {x.mean():.3f}")
    print(f"mean cohort rate R     : {r.mean():.3f}")
    print(f"mean actual rate Y     : {y.mean():.3f}")
    print(f"corr(X-R, Y-R)         : {np.corrcoef(x - r, y - r)[0, 1]:.4f}")
    print()
    print(f"point estimate w       : {w:.4f}")
    print(f"bootstrap 95% CI       : [{low:.4f}, {high:.4f}] "
          f"({BOOTSTRAP_RESAMPLES} resamples, seed {SEED})")
    print()
    for label, pred in (("cohort alone", r), ("rebased PL alone", x),
                        (f"blend at w={w:.3f}", w * x + (1 - w) * r)):
        print(f"  RMSE {label:<22}: {np.sqrt(((pred - y) ** 2).mean()):.4f}")
    print()
    verdict = "INCLUDES" if low <= 0 <= high else "EXCLUDES"
    print(f"ADR 0011 criterion 4: CI {verdict} zero -> "
          f"{'drop' if verdict == 'INCLUDES' else 'ship'} the PL transfer")


if __name__ == "__main__":
    main()
