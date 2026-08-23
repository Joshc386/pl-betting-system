"""What a side promoted into the PL looks like before it has history here.

The PL had the defect [ADR 0011](../docs/adr/0011-one-division-movement-seed-per-arrival.md)
named for the EFL, in a harsher form. The training path fills an arriving
side's rolling features from the **bottom-5 cohort** of the prior season
(`initialize_promoted_features`); the live path built no row at all and
skipped the fixture, so every market on Arsenal v Coventry, Hull v Man United
and Ipswich v Sunderland was missing on 2026-08-21.

The EFL's live path at least produced a row, wrong by 16 percentage points on
``Over25_5``. The PL produced nothing, which is safe — it forgoes a bet rather
than pricing one wrong — but it forgoes every bet on a promoted side's
fixtures all season.

Frames here are synthetic and built so the bottom-5 cohort value cannot be
confused with anything else in the frame.
"""
from __future__ import annotations

import pandas as pd

from division_movement import season_in_play

# Two bands, far enough apart that a seeded row and a real one are never
# mistakable. The cohort is what a promoted side should receive; nothing else
# in the frame carries this number.
_COHORT = 2.0
_ESTABLISHED = 9.0

_ROLLING = ["Home_Past5Goals", "Away_Past5Goals"]


def _pl_season(season: int, matches: int = 380, teams: int = 20,
               names: list[str] | None = None) -> list[dict]:
    """One PL season, every side appearing at both venues."""
    sides = names or [f"T{i:02d}" for i in range(1, teams + 1)]
    rows = []
    for i in range(matches):
        home = sides[i % teams]
        away = sides[(i + 1) % teams]
        home_pos = (i % teams) + 1
        away_pos = ((i + 1) % teams) + 1
        rows.append({
            "SeasonIndex": season,
            "Date": f"20{20 + season:02d}-{(i % 9) + 1:02d}-{(i % 28) + 1:02d}",
            "Home_Team": home,
            "Away_Team": away,
            "Home_LeaguePosition": home_pos,
            "Away_LeaguePosition": away_pos,
            # Bottom five by position carry the cohort value.
            "Home_Past5Goals": _COHORT if home_pos >= 16 else _ESTABLISHED,
            "Away_Past5Goals": _COHORT if away_pos >= 16 else _ESTABLISHED,
            "Home_Promoted": 0,
            "Away_Promoted": 0,
        })
    return rows


def _predictor(full_df: pd.DataFrame):
    """A predictor holding only the state the row builder reads.

    Built with __new__ on purpose: the real constructor loads and trains
    models, none of which this behaviour depends on.
    """
    from predict import LivePredictor

    p = LivePredictor.__new__(LivePredictor)
    p._full_df = full_df
    p.verbose = False
    return p


def test_a_promoted_side_gets_a_feature_row_instead_of_being_skipped():
    """The reported bug: every market on a promoted side's fixture was empty.

    `latest_df` is scoped to the current season, a promoted side has no rows
    in it, and the fixture was dropped outright — so the model never saw it.
    """
    df = pd.DataFrame(_pl_season(24) + _pl_season(25))
    predictor = _predictor(df)

    row = predictor._fixture_feature_row(
        "T01", "NEWCO", df, season=26, arrivals={"NEWCO"})

    assert row is not None
    assert row["Away_Team"] == "NEWCO"


def test_live_seed_matches_the_training_seed():
    """The property this fix exists to hold.

    The pipeline seeds a promoted side's training rows from the bottom-five
    cohort; the predictor seeds its live row. If those two ever differ, the
    model is scored on a row unlike the ones it learned from — and nothing
    detects it, because a feature that means one thing in training and
    another at kick-off still trains, still predicts and still looks
    plausible on its own. Only the comparison shows it.
    """
    from pipeline import PROMOTED_ROLLING_FEATURES, bottom5_cohort

    df = pd.DataFrame(_pl_season(24) + _pl_season(25))
    predictor = _predictor(df)

    row = predictor._fixture_feature_row(
        "T01", "NEWCO", df, season=26, arrivals={"NEWCO"})
    training = bottom5_cohort(df, 25, PROMOTED_ROLLING_FEATURES)

    assert row["Away_Past5Goals"] == training["Away_Past5Goals"] == _COHORT


def test_the_seed_is_the_cohort_and_never_the_established_value():
    """A wrong seed would most likely be a league average. The frame keeps
    those two numbers far apart so it cannot pass by accident."""
    df = pd.DataFrame(_pl_season(24) + _pl_season(25))
    predictor = _predictor(df)

    row = predictor._fixture_feature_row(
        "T01", "NEWCO", df, season=26, arrivals={"NEWCO"})

    assert row["Away_Past5Goals"] == _COHORT
    assert row["Away_Past5Goals"] != _ESTABLISHED
    assert row["Away_Promoted"] == 1
    # The established side's own half is untouched by the seed.
    assert row["Home_Team"] == "T01"


def _venue_split_frame() -> pd.DataFrame:
    """Season 25 complete; season 26 one round in, T01 having played away only.

    The shape the PL takes the moment upstream publishes E0 2026/27 — which
    is the state that seeded six established EFL sides as arrivals.
    """
    rows = _pl_season(24) + _pl_season(25)
    sides = [f"T{i:02d}" for i in range(1, 21)]
    for i in range(10):
        rows.append({
            "SeasonIndex": 26,
            "Date": "2026-08-22",
            "Home_Team": sides[i + 10],
            "Away_Team": sides[i],
            "Home_LeaguePosition": i + 11,
            "Away_LeaguePosition": i + 1,
            "Home_Past5Goals": _ESTABLISHED,
            "Away_Past5Goals": _ESTABLISHED,
            "Home_Promoted": 0,
            "Away_Promoted": 0,
        })
    return pd.DataFrame(rows)


def test_established_side_without_a_home_row_is_not_seeded():
    """T01 has played only away this season. It is not new to the division,
    so its own history builds the row — never the promoted-side cohort."""
    df = _venue_split_frame()
    predictor = _predictor(df)

    row = predictor._fixture_feature_row(
        "T01", "T02", df, season=26, arrivals=set())

    assert row is not None
    assert row["Home_Promoted"] == 0
    assert row["Home_Past5Goals"] != _COHORT


def test_arrival_is_decided_by_division_movement_not_row_availability():
    """Being absent from season N-1 is what makes a side an arrival. Having
    no rows yet is a statement about the calendar, not about the side."""
    df = _venue_split_frame()
    predictor = _predictor(df)

    established = predictor._fixture_feature_row(
        "T01", "T02", df, season=26, arrivals=set())
    arriving = predictor._fixture_feature_row(
        "T01", "NEWCO", df, season=26, arrivals={"NEWCO"})

    assert established["Home_Promoted"] == 0
    assert arriving["Away_Promoted"] == 1


def test_a_side_with_no_history_and_no_arrival_is_skipped():
    """Neither new to the division nor known in it: there is nothing honest
    to build a row from, and inventing one would price a bet on a guess."""
    df = _venue_split_frame()
    predictor = _predictor(df)

    assert predictor._fixture_feature_row(
        "GHOST", "T02", df, season=26, arrivals=set()) is None


def test_season_in_play_drives_the_pl_the_same_way():
    """The PL uses the same season determination as the EFL — one rule, so
    the two leagues cannot disagree about which season is being played."""
    complete = pd.DataFrame(_pl_season(24) + _pl_season(25))
    in_progress = _venue_split_frame()

    assert season_in_play(complete) == 26     # pre-season window
    assert season_in_play(in_progress) == 26  # one round played
