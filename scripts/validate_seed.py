"""Criterion 1 for ADR 0011: does the seed improve the seed slice?

``backtest_promoted.py`` measures whether seeding training rows at all beats
not seeding them. That is not the question here — the training path's cohort
logic is substantively unchanged by ADR 0011, only relocated. What changed is
what an *arrival* is priced from: Dixon-Coles used to rate it on its exit
season, and now rates it from its route.

So this walks forward season by season and scores every fixture in the seed
slice twice — once on the ratings Dixon-Coles estimates unaided, once after
the seed is applied — against what actually happened. Priors for season N are
measured from seasons below N only, so nothing here sees its own outcome.

Run:
    python scripts/validate_seed.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from division_movement import arrivals, fit_seed_params  # noqa: E402
from model import DixonColesPredictor  # noqa: E402

# Per league: the frame being seeded, the division above it (None where there
# is none), and the seasons worth walking. Generalised rather than copied —
# a second harness would drift from this one exactly as a second seed
# implementation would.
LEAGUES = {
    "EFL": {"csv": "CompleteDSChamp_CSV.csv", "above": "CompleteDSPL_CSV.csv",
            "first": 10, "last": 25},
    "PL": {"csv": "CompleteDSPL_CSV.csv", "above": None,
           "first": 10, "last": 25},
}

FIRST_SEASON = 10   # earlier seasons cannot clear the minimum event count
LAST_SEASON = 25
SEED_WINDOW = 5
EPS = 1e-15


def seed_slice(ef: pd.DataFrame, season: int, incoming: dict) -> pd.Index:
    """Rows the seed governs: each arrival's first five at each venue."""
    in_season = ef[ef["SeasonIndex"] == season]
    rows = set()
    for team in incoming:
        for side in ("Home_Team", "Away_Team"):
            played = in_season[in_season[side] == team]
            rows.update(played.sort_values("Date").head(SEED_WINDOW).index)
    return pd.Index(sorted(rows))


def log_loss(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probabilities, EPS, 1 - EPS)
    return float(-(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)).mean())


def main(league: str = "EFL") -> None:
    cfg = LEAGUES[league]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ef = pd.read_csv(os.path.join(root, cfg["csv"]), low_memory=False)
    pl = (pd.read_csv(os.path.join(root, cfg["above"]), low_memory=False)
          if cfg["above"] else None)
    if "TG" not in ef.columns:
        ef["TG"] = ef["Home_Goals"] + ef["Away_Goals"]
    ef["Over_2_5"] = (ef["TG"] > 2.5).astype(int)
    print(f"League: {league}")

    unaided, seeded, outcomes = [], [], []
    per_season = []

    for season in range(cfg["first"], cfg["last"] + 1):
        incoming = arrivals(ef, pl, season)
        if not incoming:
            continue
        rows = seed_slice(ef, season, incoming)
        if rows.empty:
            continue

        history = ef[ef["SeasonIndex"] < season]
        params = fit_seed_params(ef, pl, through_season=season)

        base = DixonColesPredictor(half_life=10, rho=-0.2)
        base.fit(history)
        with_seed = DixonColesPredictor(half_life=10, rho=-0.2)
        with_seed.fit(history)
        with_seed.seed_arrivals(incoming, params.priors)

        fixtures = ef.loc[rows]
        p_base = base.predict_proba_df(fixtures)
        p_seed = with_seed.predict_proba_df(fixtures)
        y = fixtures["Over_2_5"].to_numpy()

        unaided.extend(p_base)
        seeded.extend(p_seed)
        outcomes.extend(y)
        per_season.append((season, len(rows), log_loss(np.array(p_base), y),
                           log_loss(np.array(p_seed), y)))

    unaided = np.array(unaided)
    seeded = np.array(seeded)
    outcomes = np.array(outcomes)

    print(f"{'season':>7} {'rows':>6} {'unaided':>10} {'seeded':>10} {'delta':>9}")
    for season, n, a, b in per_season:
        print(f"{season:>7} {n:>6} {a:>10.4f} {b:>10.4f} {b - a:>+9.4f}")

    base_ll = log_loss(unaided, outcomes)
    seed_ll = log_loss(seeded, outcomes)
    print()
    print(f"seed slice rows        : {len(outcomes)}")
    print(f"base rate (Over 2.5)   : {outcomes.mean():.4f}")
    print(f"log-loss unaided       : {base_ll:.4f}")
    print(f"log-loss seeded        : {seed_ll:.4f}")
    print(f"improvement            : {base_ll - seed_ll:+.4f} "
          f"({100 * (base_ll - seed_ll) / base_ll:+.2f}%)")

    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(5000):
        pick = rng.integers(0, len(outcomes), len(outcomes))
        deltas.append(log_loss(unaided[pick], outcomes[pick])
                      - log_loss(seeded[pick], outcomes[pick]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"bootstrap 95% CI       : [{low:+.4f}, {high:+.4f}]")
    print()
    verdict = "PASSES" if low > 0 else (
        "FAILS" if high < 0 else "INCONCLUSIVE (CI spans zero)")
    print(f"criterion: log-loss on the seed slice -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EFL")
