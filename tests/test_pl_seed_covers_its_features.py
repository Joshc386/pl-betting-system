"""The PL seeds every rolling feature it holds, or training and serving differ.

`pipeline.initialize_promoted_features` fills *and* blends
`PROMOTED_ROLLING_FEATURES`. Since `fd64bd5`, `predict._fixture_feature_row`
fills every numeric `Home_`/`Away_` column and blends that same list. Any
feature outside the list is therefore filled at serve time and not at train
time — one name, two values, which is the defect
[ADR 0011](../docs/adr/0011-one-division-movement-seed-per-arrival.md) opened on.

Eight entries were in that gap: `Past5CornersConceded`, `CR_20`, `SOT_CR_5`
and `SOT_CR_20`, Home and Away each. All are ~99.8% populated in the PL
canonical, all are in `config.ALL_FEATURES`, six are in `BTTS_ALL_FEATURES`.

The value training kept was not merely different, it was stale in the same way
the Dixon-Coles rating was: rolling features are computed with
`groupby("team")` and no gap awareness, so Ipswich's `CR_20` for their first
match of season 24 was a rolling mean over their 2000/01 and 2001/02 matches.
See [ADR 0012](../docs/adr/0012-division-movement-seed-for-the-premier-league.md).
"""
from __future__ import annotations

import pandas as pd
import pytest

_CLOSED = [
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",
    "Home_CR_20", "Away_CR_20",
    "Home_SOT_CR_5", "Away_SOT_CR_5",
    "Home_SOT_CR_20", "Away_SOT_CR_20",
]


@pytest.mark.parametrize("feature", _CLOSED)
def test_the_eight_are_blended_by_training(feature):
    """Training must blend them, or serving's fill has nothing to agree with."""
    from pipeline import PROMOTED_ROLLING_FEATURES

    assert feature in PROMOTED_ROLLING_FEATURES, (
        f"{feature} is filled at serve time and not at train time, so it "
        f"means one quantity in training and another at kick-off")


@pytest.mark.parametrize("feature", _CLOSED)
def test_the_eight_are_real_columns_carrying_real_values(feature):
    """Guards the test above from being satisfied by adding names.

    Scoped to these eight rather than the whole list on purpose. The list also
    holds features `run_pipeline` derives — `GoalDiff_5`, `ShotSuppression_5`
    and the other defensive components — which exist in the pipeline's output
    and not in the canonical on disk. Asserting the whole list against the CSV
    fails on those, and it fails for the reason ADR 0012 records as a lesson:
    the canonical is pre-pipeline, so a seeding question asked of it gets the
    wrong answer. These eight are canonical columns, so the CSV can answer
    for them.
    """
    df = pd.read_csv("CompleteDSPL_enriched.csv", low_memory=False,
                     usecols=[feature])

    assert df[feature].notna().mean() > 0.9, (
        f"{feature} is seeded but barely populated — blending it would "
        f"spread a cohort average over nothing")


def test_the_pl_seeds_every_rolling_feature_the_efl_does_and_holds():
    """No feature the PL holds is seeded by the EFL and not by the PL.

    The two lists were never compared — `PROMOTED_ROLLING_FEATURES` was
    hoisted unchanged and its last membership change renamed three entries.
    This is what makes the gap visible if it reopens.
    """
    from championship_pipeline import SEEDED_ROLLING_FEATURES
    from pipeline import PROMOTED_ROLLING_FEATURES

    columns = set(pd.read_csv("CompleteDSPL_enriched.csv",
                              low_memory=False, nrows=2).columns)
    gap = [f for f in SEEDED_ROLLING_FEATURES
           if f in columns and f not in PROMOTED_ROLLING_FEATURES]

    assert not gap, (
        f"the EFL seeds these and the PL holds them but does not: {gap}")


def test_widening_the_blend_touches_only_the_seed_slice():
    """ADR 0012 criterion C1, for the feature change.

    A hard invariant, not a statistical test: the seed governs an arrival's
    first five matches at each venue and nothing else. A row outside that
    slice changing means the change is not what it claims to be, whatever the
    log-loss says.
    """
    import numpy as np
    from pipeline import PROMOTED_ROLLING_FEATURES, initialize_promoted_features

    df = pd.read_csv("CompleteDSPL_enriched.csv", low_memory=False)
    df = df[df["SeasonIndex"].isin([23, 24])].copy()

    narrow = [f for f in PROMOTED_ROLLING_FEATURES if f not in _CLOSED]
    import pipeline
    original = pipeline.PROMOTED_ROLLING_FEATURES
    try:
        pipeline.PROMOTED_ROLLING_FEATURES = narrow
        before = initialize_promoted_features(df.copy())[0]
        pipeline.PROMOTED_ROLLING_FEATURES = original
        after = initialize_promoted_features(df.copy())[0]
    finally:
        pipeline.PROMOTED_ROLLING_FEATURES = original

    # Which rows the seed is allowed to touch: each arrival's first five at
    # each venue, in the season it arrives.
    allowed = set()
    for season in (24,):
        cur = df[df["SeasonIndex"] == season]
        prev = df[df["SeasonIndex"] == season - 1]
        known = set(prev["Home_Team"]) | set(prev["Away_Team"])
        for team in (set(cur["Home_Team"]) | set(cur["Away_Team"])) - known:
            for side in ("Home_Team", "Away_Team"):
                played = cur[cur[side] == team].sort_values("Date")
                allowed.update(played.head(5).index)

    changed = set()
    for col in _CLOSED:
        a, b = before[col], after[col]
        differs = ~((a.isna() & b.isna()) | np.isclose(
            a.fillna(-9e9), b.fillna(-9e9)))
        changed.update(before.index[differs])

    assert changed, (
        "widening the blend changed nothing at all — the eight are not "
        "reaching the arrival's rows and the repair is inert")
    assert changed <= allowed, (
        f"{len(changed - allowed)} rows outside the seed slice changed; the "
        f"seed must govern an arrival's first five at each venue and no more")
