"""Guard: one feature contract per name, across leagues (ADR 0007).

``EXISTING_FEATURES`` are read straight from a Canonical Dataset — ``pipeline.py``
recomputes none of them — so each is frozen at whatever the build script wrote.
Two build scripts wrote them, and nothing forced the two implementations to
agree. 15 of the 39 diverged: ``ShotRatio_5`` was SOT÷shots in one league and
shots÷SOT in the other, ``DefensiveStrength_5`` was not a defensive metric at
all in the PL, ``H2H`` counted 5 meetings in one league and all 33 in the other.

None of it surfaced, because a feature that means something different in each
league still trains, still predicts, and still looks plausible in isolation.
Only the comparison shows it. This test is that comparison, run every time.

Each entry in ``_KNOWN_DIVERGENCES`` is xfail(strict=True): the instant a fix
lands, the test XPASSes and fails the suite, forcing the exemption out rather
than letting it linger as a stale allowance.
"""
from __future__ import annotations

import functools
import os

import pandas as pd
import pytest

from config import EXISTING_FEATURES
from league_config import LEAGUES

# Seasons where both leagues have full coverage. Earlier PL seasons predate
# the match-stat columns; 24-25 are the only seasons the hand-maintained
# promoted dicts populate, so including them would mask the dead feature.
_FIRST_SEASON, _LAST_SEASON = 5, 23

# A feature whose league means sit outside this band is not the same
# quantity twice. The band is deliberately wide — this catches units and
# formula errors (6x, 24x), not distributional nuance.
_TOL_LOW, _TOL_HIGH = 0.8, 1.25

# Differences that are real, structural, and correct. Each needs a reason;
# "the leagues are different" is not one.
_EXEMPT: dict[str, str] = {
    "Home_LeaguePosition": "20-team division vs 24 — mid-table is 10.5 vs 12.5",
    "Away_LeaguePosition": "20-team division vs 24 — mid-table is 10.5 vs 12.5",
    # One list and one exact matcher now serve both leagues (ADR 0007 decision
    # 9), so the contract holds; the rates still differ because the divisions
    # contain different clubs. A mean-ratio test can never pass here, and left
    # as xfail it would be a permanent allowance that never resolves — exactly
    # what the strict mechanism exists to prevent. The real contract is checked
    # by tests/test_derby_config.py instead.
    "Local Derby": "different clubs per division — rate is not comparable",
    "Historical Derby": "different clubs per division — rate is not comparable",
}

# The 15 divergences ADR 0007 accepted but has not yet fixed, each against
# the decision that retires it. Delete an entry when its decision lands.
_KNOWN_DIVERGENCES: dict[str, str] = {
    "Home Factor": "decision 7 — rolling-10 mean vs ratio-to-league-average",
    "Away Factor": "decision 7 — rolling-10 mean vs ratio-to-league-average",
    "Home_ShotRatio_5": "decision 8 — shots÷SOT (PL) vs SOT÷shots (EFL)",
    "Away_ShotRatio_5": "decision 8 — shots÷SOT (PL) vs SOT÷shots (EFL)",
    "Home_DefensiveStrength_5": "decision 4 — 1÷shots-conceded is not a rate",
    "Away_DefensiveStrength_5": "decision 4 — 1÷shots-conceded is not a rate",
    "Home_DefensiveStrength_SOT": "decision 4 — 1÷SOT-conceded is not a rate",
    "Away_DefensiveStrength_SOT": "decision 4 — 1÷SOT-conceded is not a rate",
    "Home_Promoted": "decision 1 — hand-maintained dicts, dead in both leagues",
    "Away_Promoted": "decision 1 — hand-maintained dicts, dead in both leagues",
    "H2H_HomeWins": "decision 6 — last 5 meetings vs all meetings; being dropped",
    "H2H_AwayWins": "decision 6 — last 5 meetings vs all meetings; being dropped",
    "H2H_Draws": "decision 6 — last 5 meetings vs all meetings; being dropped",
}

_NEAR_ZERO = 1e-9


@functools.lru_cache(maxsize=None)
def _canonical(league: str) -> pd.DataFrame | None:
    """Load a league's canonical, restricted to the comparison window."""
    path = LEAGUES[league]["csv_path"]
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    return df[df["SeasonIndex"].between(_FIRST_SEASON, _LAST_SEASON)]


def _mean(league: str, feature: str) -> float | None:
    """Mean of a feature over the comparison window, or None if unavailable."""
    df = _canonical(league)
    if df is None or feature not in df.columns:
        return None
    values = pd.to_numeric(df[feature], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _feature_params() -> list:
    """One param per shared feature, xfailed where ADR 0007 knows it diverges."""
    params = []
    for feature in EXISTING_FEATURES:
        if feature in _EXEMPT:
            continue
        marks = []
        if feature in _KNOWN_DIVERGENCES:
            marks.append(pytest.mark.xfail(
                strict=True,
                reason=f"ADR 0007 {_KNOWN_DIVERGENCES[feature]}",
            ))
        params.append(pytest.param(feature, marks=marks, id=feature))
    return params


@pytest.mark.parametrize("feature", _feature_params())
def test_feature_means_agree_across_leagues(feature: str) -> None:
    """The same feature name must denote the same quantity in both leagues."""
    pl_mean = _mean("PL", feature)
    efl_mean = _mean("EFL", feature)

    if pl_mean is None or efl_mean is None:
        pytest.skip(f"{feature} unavailable in one or both canonicals")

    both_zero = abs(pl_mean) < _NEAR_ZERO and abs(efl_mean) < _NEAR_ZERO
    if both_zero:
        return

    assert abs(pl_mean) > _NEAR_ZERO and abs(efl_mean) > _NEAR_ZERO, (
        f"{feature}: constant zero in one league only "
        f"(PL {pl_mean:.4g}, EFL {efl_mean:.4g}) — the feature is dead on "
        f"one side, so the model learned it from the other alone."
    )

    ratio = pl_mean / efl_mean
    assert _TOL_LOW <= ratio <= _TOL_HIGH, (
        f"{feature}: PL mean {pl_mean:.4g} vs EFL mean {efl_mean:.4g} "
        f"(ratio {ratio:.3g}, tolerated {_TOL_LOW}–{_TOL_HIGH}) over seasons "
        f"{_FIRST_SEASON}–{_LAST_SEASON}. One feature name, two computations "
        f"— see docs/adr/0007-one-feature-contract-per-name.md."
    )


def test_every_known_divergence_is_still_a_feature() -> None:
    """A divergence must not be exempted by quietly disappearing.

    Renaming or dropping a feature is a legitimate fix, but it has to be a
    deliberate one: the entry comes out of ``_KNOWN_DIVERGENCES`` in the same
    change. Otherwise the exemption outlives the thing it exempted.
    """
    stale = sorted(set(_KNOWN_DIVERGENCES) - set(EXISTING_FEATURES))
    assert not stale, (
        f"{stale} are exempted in _KNOWN_DIVERGENCES but no longer appear in "
        f"EXISTING_FEATURES. Remove the exemptions."
    )


def test_exemptions_are_documented() -> None:
    """Every structural exemption carries a reason, and still exists."""
    stale = sorted(set(_EXEMPT) - set(EXISTING_FEATURES))
    assert not stale, f"{stale} are exempt but no longer features"
    for feature, reason in _EXEMPT.items():
        assert len(reason) > 20, f"{feature} exemption needs a real reason"
