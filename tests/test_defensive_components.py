"""Defensive Strength is three named components, never one number (ADR 0007
decisions 4 and 5, tier 1).

- ``ShotSuppression_5`` — shots conceded relative to what that opponent
  usually generates (their pre-match rolling-5 shot volume). Opponent-
  adjusted by shot *generation*, not Elo: process-based per the Wheatcroft
  principle, and self-normalising across leagues and eras.
- ``ChanceQualityAllowed_5`` — SOT conceded ÷ shots conceded (the old EFL
  ``DefensiveStrength_5`` semantic, correctly named).
- ``ConversionAllowed_5`` — goals conceded ÷ SOT conceded (the old EFL
  ``DefensiveStrength_SOT`` semantic, correctly named).

All are means of per-match ratios over ``shift(1).rolling(5, min_periods=1)``
per team — the same convention as the canonical builder's rolling features —
with zero denominators becoming NaN, never inf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.common import add_defensive_components


def _fixture() -> pd.DataFrame:
    """Three meetings of A and B with hand-computable stats.

    Before match 3:
      A conceded SOT/shots of 2/8 then 6/12  -> ChanceQuality (.25+.5)/2
      A conceded goals/SOT of 1/2 then 2/6   -> Conversion (.5+1/3)/2
      A's suppression: match 1 skipped (B had no history), match 2 ratio
      12 shots conceded ÷ B's prior volume of 8 -> 1.5
    """
    rows = [
        # date, home, away, h_shots, a_shots, h_sot, a_sot, h_goals, a_goals
        ("2024-08-01", "A", "B", 10, 8, 4, 2, 2, 1),
        ("2024-08-08", "B", "A", 12, 6, 6, 3, 2, 1),
        ("2024-08-15", "A", "B", 8, 10, 2, 5, 0, 1),
    ]
    df = pd.DataFrame(rows, columns=[
        "Date", "Home_Team", "Away_Team", "Home_Shots", "Away_Shots",
        "Home_Shots_Target", "Away_Shots_Target", "Home_Goals", "Away_Goals",
    ])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def test_chance_quality_allowed_is_sot_over_shots_conceded():
    out = add_defensive_components(_fixture())
    last = out.iloc[-1]
    assert last["Home_ChanceQualityAllowed_5"] == pytest.approx((0.25 + 0.5) / 2)
    assert last["Away_ChanceQualityAllowed_5"] == pytest.approx((0.4 + 0.5) / 2)


def test_conversion_allowed_is_goals_over_sot_conceded():
    out = add_defensive_components(_fixture())
    last = out.iloc[-1]
    assert last["Home_ConversionAllowed_5"] == pytest.approx((0.5 + 1 / 3) / 2)
    assert last["Away_ConversionAllowed_5"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_shot_suppression_is_relative_to_opponent_volume():
    """A conceded 12 against a B who usually produces 8 — suppression 1.5
    (worse than neutral). B conceded 6 against an A who usually produces
    10 — suppression 0.6. Match-1 ratios are NaN (no opponent history) and
    are skipped, not zero-filled."""
    out = add_defensive_components(_fixture())
    last = out.iloc[-1]
    assert last["Home_ShotSuppression_5"] == pytest.approx(12 / 8)
    assert last["Away_ShotSuppression_5"] == pytest.approx(6 / 10)


def test_first_match_has_no_defensive_features():
    """shift(1) everywhere: nothing about a match leaks into its own features."""
    out = add_defensive_components(_fixture())
    first = out.iloc[0]
    for col in ("Home_ShotSuppression_5", "Home_ChanceQualityAllowed_5",
                "Home_ConversionAllowed_5", "Away_ShotSuppression_5",
                "Away_ChanceQualityAllowed_5", "Away_ConversionAllowed_5"):
        assert pd.isna(first[col]), f"{col} present on a first match"


def test_zero_denominators_become_nan_not_inf():
    """A shotless opponent must not divide anything by zero."""
    df = _fixture()
    df.loc[0, "Away_Shots"] = 0          # B took no shots in match 1
    df.loc[0, "Away_Shots_Target"] = 0
    out = add_defensive_components(df)
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()
    # A's chance-quality window now holds only the match-2 ratio
    assert out.iloc[-1]["Home_ChanceQualityAllowed_5"] == pytest.approx(0.5)


def test_missing_shot_columns_leave_df_unchanged():
    """No shot columns at all -> unchanged, same convention as the discipline
    features. (Eras where the columns exist but hold NaN get NaN values —
    never a substitute formula, per the feature-contract rule.)"""
    df = _fixture().drop(columns=["Home_Shots_Target", "Away_Shots_Target"])
    out = add_defensive_components(df)
    assert "Home_ShotSuppression_5" not in out.columns
    assert "Home_ChanceQualityAllowed_5" not in out.columns
