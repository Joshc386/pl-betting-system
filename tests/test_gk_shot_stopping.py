"""Tier 3 of the Defensive Strength decomposition (ADR 0007 decision 5).

``GKShotStopping_5`` is the playing goalkeeper's rolling-5 goals_prevented
per 90 from FPL-Core-Insights — the one signal that separates keeper
shot-stopping from defensive quality, the confound baked into Conversion
Allowed. PL seasons 24-25 only (~7% of training rows); everywhere else NaN.

The tier is PROVISIONAL: it ships only if the walk-forward AUC/Brier gate
passes. These tests pin the computation either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from api.player_features import compute_squad_features_historical


def _squad(season="2024-2025", match_id=1, gk_rolling_gp90=0.3):
    """A full two-team match frame: 11 players a side, one keeper each.

    The home keeper carries ``gk_rolling_gp90``; the away keeper has no
    rolling history (NaN) — a debutant.
    """
    rows = []
    for is_home in (True, False):
        for i in range(11):
            is_gk = i == 0
            rows.append({
                "season": season, "match_id": match_id, "gameweek": 10,
                "home_team_name": "Arsenal", "away_team_name": "Chelsea",
                "kickoff_time": "2024-10-01T15:00:00Z",
                "is_home": is_home,
                "player_id": (0 if is_home else 100) + i,
                "position": "Goalkeeper" if is_gk else "Midfielder",
                "minutes_played": 90,
                "start_min": 0,
                "rolling_xg_5": 0.2, "rolling_xa_5": 0.1,
                "rolling_def_5": 5.0, "cum_minutes": 900.0,
                "rolling_gp90_5": (
                    (gk_rolling_gp90 if is_home else np.nan)
                    if is_gk else np.nan),
            })
    return pd.DataFrame(rows)


def test_gk_shot_stopping_is_the_playing_keepers_rolling_value():
    out = compute_squad_features_historical(_squad(gk_rolling_gp90=0.3))
    home = out[out["side"] == "home"].iloc[0]
    assert home["GKShotStopping_5"] == pytest.approx(0.3)


def test_a_keeper_without_history_yields_nan_not_zero():
    """A debutant keeper is unknown, not average — NaN, never 0."""
    out = compute_squad_features_historical(_squad())
    away = out[out["side"] == "away"].iloc[0]
    assert pd.isna(away["GKShotStopping_5"])


def test_no_keeper_on_the_pitch_yields_nan():
    """An outfield-only frame (keeper under 45 minutes) has no GK signal."""
    df = _squad(gk_rolling_gp90=0.3)
    df.loc[df["position"] == "Goalkeeper", "minutes_played"] = 30
    out = compute_squad_features_historical(df)
    home = out[out["side"] == "home"].iloc[0]
    assert pd.isna(home["GKShotStopping_5"])
