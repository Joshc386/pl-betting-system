"""Arm 3: does widening the PL blend to eight more features do no harm?

C1 already holds as a hard invariant — widening touches only seed-slice rows.
This is the statistical half: holding the model class fixed, does correcting
those eight on an arrival's rows predict better or worse than leaving them
stale?

Walk-forward. For each season the frame is initialised twice, once with the
blend list as it was and once as it is, a model is fitted on everything below
that season, and that season's seed slice is scored against what happened.
Only the seed slice is scored: every other row is identical between the two
frames by construction, so including them would dilute the difference toward
zero and prove nothing either way.

The model is a single fixed-seed XGBoost rather than the production ensemble.
The question is whether the corrected values carry more signal than the stale
ones, and that is a property of the rows; putting four models and a stacker in
front of it would add variance without addressing it.

Run:
    python scripts/validate_seed_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline  # noqa: E402
from pipeline import initialize_promoted_features  # noqa: E402
from scripts.validate_seed import log_loss, seed_slice  # noqa: E402
from division_movement import arrivals  # noqa: E402

CLOSED = [
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",
    "Home_CR_20", "Away_CR_20",
    "Home_SOT_CR_5", "Away_SOT_CR_5",
    "Home_SOT_CR_20", "Away_SOT_CR_20",
]
FIRST, LAST = 10, 25
PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, random_state=42,
              eval_metric="logloss", verbosity=0)


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(root, "CompleteDSPL_enriched.csv"),
                     low_memory=False)
    df["TG"] = df["Home_Goals"] + df["Away_Goals"]
    df["Over_2_5"] = (df["TG"] > 2.5).astype(int)

    import config
    feats = [f for f in config.ALL_FEATURES if f in df.columns]
    print(f"model features: {len(feats)} (of {len(config.ALL_FEATURES)})")

    wide = list(pipeline.PROMOTED_ROLLING_FEATURES)
    narrow = [f for f in wide if f not in CLOSED]

    frames = {}
    for name, blend in (("narrow", narrow), ("wide", wide)):
        original = pipeline.PROMOTED_ROLLING_FEATURES
        try:
            pipeline.PROMOTED_ROLLING_FEATURES = blend
            frames[name] = initialize_promoted_features(df.copy())[0]
        finally:
            pipeline.PROMOTED_ROLLING_FEATURES = original

    arms = {"narrow": [], "wide": []}
    outcomes = []

    for season in range(FIRST, LAST + 1):
        incoming = arrivals(df, None, season)
        if not incoming:
            continue
        rows = seed_slice(df, season, incoming)
        if rows.empty:
            continue

        for name, frame in frames.items():
            train = frame[frame["SeasonIndex"] < season]
            model = xgb.XGBClassifier(**PARAMS)
            model.fit(train[feats], train["Over_2_5"])
            arms[name].extend(
                model.predict_proba(frame.loc[rows, feats])[:, 1])
        outcomes.extend(df.loc[rows, "Over_2_5"].to_numpy())

    a = np.array(arms["narrow"])
    b = np.array(arms["wide"])
    y = np.array(outcomes)

    ll_a, ll_b = log_loss(a, y), log_loss(b, y)
    print()
    print(f"seed slice rows          : {len(y)}")
    print(f"base rate (Over 2.5)     : {y.mean():.4f}")
    print(f"log-loss eleven pairs    : {ll_a:.4f}")
    print(f"log-loss nineteen pairs  : {ll_b:.4f}")
    print(f"improvement              : {ll_a - ll_b:+.4f} "
          f"({100 * (ll_a - ll_b) / ll_a:+.2f}%)")

    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(5000):
        pick = rng.integers(0, len(y), len(y))
        deltas.append(log_loss(a[pick], y[pick]) - log_loss(b[pick], y[pick]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    print(f"bootstrap 95% CI         : [{low:+.4f}, {high:+.4f}]")
    print()
    harmed = ll_b > ll_a and high < 0
    print("criterion is NO-HARM at C3's bootstrap-interval tolerance.")
    print(f"verdict: {'FAILS' if harmed else 'PASSES'}")
    if ll_b > ll_a and not harmed:
        print(f"  (worse by {ll_b - ll_a:.4f}, inside the interval)")


if __name__ == "__main__":
    main()
