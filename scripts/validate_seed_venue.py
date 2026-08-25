"""Arm 4: does counting the seed window per venue do no harm?

Both predictors gated the Dixon-Coles seed on a side's *total* appearances
while the feature row counted per venue. This scores the rows where the two
schemes disagree — the only rows the change can move — under each, against
what actually happened.

Scoring the whole seed slice would bury the effect: the two agree everywhere
outside a narrow band, so most of the slice contributes identical numbers to
both arms and only shrinks the difference toward zero.

Run:
    python scripts/validate_seed_venue.py EFL
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from division_movement import arrivals, fit_seed_params, seed_weight  # noqa: E402
from model import DixonColesPredictor  # noqa: E402
from scripts.validate_seed import LEAGUES, log_loss  # noqa: E402


def main(league: str = "EFL") -> None:
    cfg = LEAGUES[league]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, cfg["csv"]), low_memory=False)
    above = (pd.read_csv(os.path.join(root, cfg["above"]), low_memory=False)
             if cfg["above"] else None)
    if "TG" not in df.columns:
        df["TG"] = df["Home_Goals"] + df["Away_Goals"]
    df["Over_2_5"] = (df["TG"] > 2.5).astype(int)

    total_arm, venue_arm, outcomes = [], [], []

    for season in range(cfg["first"], cfg["last"] + 1):
        incoming = arrivals(df, above, season)
        if not incoming:
            continue
        history = df[df["SeasonIndex"] < season]
        params = fit_seed_params(df, above, through_season=season)
        if not params.priors:
            continue

        base = DixonColesPredictor(half_life=10, rho=-0.2).fit(history)
        in_season = df[df["SeasonIndex"] == season].sort_values("Date")

        for team, route in incoming.items():
            prior = params.priors[route]
            seen = {"home": 0, "away": 0}
            played = in_season[(in_season["Home_Team"] == team)
                               | (in_season["Away_Team"] == team)]
            for idx, row in played.iterrows():
                at = "home" if row["Home_Team"] == team else "away"
                # State *before* this fixture, which is what a live scan sees.
                by_total = seed_weight(seen["home"] + seen["away"]) > 0
                by_venue = seed_weight(seen[at]) > 0
                seen[at] += 1
                if by_total == by_venue:
                    continue  # the two agree; this row cannot separate them

                fixture = df.loc[[idx]]
                for arm, applies in ((total_arm, by_total),
                                     (venue_arm, by_venue)):
                    dc = DixonColesPredictor(half_life=10, rho=-0.2)
                    dc.__dict__.update(base.__dict__)
                    if applies:
                        dc.seed_arrivals({team: route}, params.priors,
                                         venues={team: {at}})
                    arm.extend(dc.predict_proba_df(fixture))
                outcomes.extend(fixture["Over_2_5"].to_numpy())

    if not outcomes:
        print("no rows where the two gates disagree")
        return

    total_arm = np.array(total_arm)
    venue_arm = np.array(venue_arm)
    outcomes = np.array(outcomes)

    ll_total = log_loss(total_arm, outcomes)
    ll_venue = log_loss(venue_arm, outcomes)
    print(f"League: {league}")
    print(f"rows where the gates disagree : {len(outcomes)}")
    print(f"base rate (Over 2.5)          : {outcomes.mean():.4f}")
    print(f"log-loss counting totals      : {ll_total:.4f}")
    print(f"log-loss counting per venue   : {ll_venue:.4f}")
    print(f"improvement                   : {ll_total - ll_venue:+.4f} "
          f"({100 * (ll_total - ll_venue) / ll_total:+.2f}%)")

    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(5000):
        pick = rng.integers(0, len(outcomes), len(outcomes))
        deltas.append(log_loss(total_arm[pick], outcomes[pick])
                      - log_loss(venue_arm[pick], outcomes[pick]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"bootstrap 95% CI              : [{low:+.4f}, {high:+.4f}]")
    print()
    # No-harm on ADR 0012 criterion C3's tolerance: worse only counts as harm
    # when it is worse by more than the interval. A point-estimate reading
    # would fail this arm on the PL at -0.0011, a difference its own interval
    # cannot distinguish from zero — and C3 was agreed as interval tolerance
    # before any of these numbers existed.
    harmed = (ll_venue > ll_total) and high < 0
    print("criterion is NO-HARM at C3's bootstrap-interval tolerance.")
    print(f"verdict: {'FAILS' if harmed else 'PASSES'}")
    if ll_venue > ll_total and not harmed:
        print(f"  (worse by {ll_venue - ll_total:.4f}, inside "
              f"[{low:+.4f}, {high:+.4f}] — not separable from zero)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EFL")
