"""Promoted / Relegated are derived from the Canonical Datasets (ADR 0007).

They used to come from three hand-maintained dicts, and the result was a dead
feature: constant zero across 24 of 26 PL seasons and 21 of 26 EFL seasons,
because nobody kept the lists current. Whatever the model learned about
promotion, it learned from two seasons while the other 24 asserted that
promotion never happens.

A team in season N but not in N-1 is new to the division. For the PL that can
only mean promotion. For the EFL it is ambiguous — a side down from the PL is
one of the *stronger* teams in the division, a side up from League One one of
the weakest — and only the PL canonical separates them.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.build_canonical_dataset import _add_promotion_flags

PL_TEAMS = [f"Team{i:02d} FC" for i in range(20)]
EFL_TEAMS = [f"Club{i:02d}" for i in range(24)]


def _season(season: int, teams: list[str]) -> pd.DataFrame:
    """One matchday pairing every team once, so the season's team set is whole."""
    return pd.DataFrame([
        {
            "SeasonIndex": season,
            "Date": pd.Timestamp(f"20{20 + season:02d}-08-15"),
            "Home_Team": teams[i],
            "Away_Team": teams[i + 1],
            "Home_Goals": 1,
            "Away_Goals": 0,
        }
        for i in range(0, len(teams), 2)
    ])


def _flags(out: pd.DataFrame, team: str, col: str) -> set:
    """Every value of *col* recorded for *team*, whichever side it played."""
    home = out.loc[out["Home_Team"] == team, f"Home_{col}"]
    away = out.loc[out["Away_Team"] == team, f"Away_{col}"]
    return set(pd.concat([home, away]).dropna().tolist())


def test_pl_arrivals_are_flagged_promoted():
    """A side in the PL this season but not last was promoted into it."""
    arrivals = ["Up01 FC", "Up02 FC", "Up03 FC"]
    df = pd.concat([
        _season(0, PL_TEAMS),
        _season(1, PL_TEAMS[:-3] + arrivals),
    ], ignore_index=True)

    out = _add_promotion_flags(df, "PL", None)

    for team in arrivals:
        assert _flags(out, team, "Promoted") == {1}, f"{team} should be promoted"
    for team in PL_TEAMS[:-3]:
        assert _flags(out, team, "Promoted") == {0}, f"{team} was already up"


def test_efl_arrivals_split_into_relegated_and_promoted(tmp_path):
    """The PL canonical separates sides coming down from sides coming up.

    The three real short forms are bridged through ``normalize()`` — the EFL
    canonical stores "Burnley", the PL canonical "Burnley FC".
    """
    came_down = ["Burnley", "Leeds", "Sunderland"]
    came_up = ["Wrexham", "Charlton", "Barnsley"]

    sibling = tmp_path / "pl.csv"
    _season(0, ["Burnley FC", "Leeds United FC", "Sunderland AFC"]
            + [f"Stayer{i:02d} FC" for i in range(17)]).to_csv(
        sibling, index=False)

    df = pd.concat([
        _season(0, EFL_TEAMS),
        _season(1, EFL_TEAMS[:-6] + came_down + came_up),
    ], ignore_index=True)

    out = _add_promotion_flags(df, "EFL", str(sibling))

    for team in came_down:
        assert _flags(out, team, "Relegated") == {1}, f"{team} came down"
        assert _flags(out, team, "Promoted") == {0}
    for team in came_up:
        assert _flags(out, team, "Promoted") == {1}, f"{team} came up"
        assert _flags(out, team, "Relegated") == {0}


def test_incomplete_season_is_null_not_guessed(capsys):
    """Mid-rollover, a part-loaded season cannot say who is new.

    Season 26 spends the start of August in exactly this state. Guessing would
    assert something false about the arrivals; ADR 0007 says the feature is
    null where the data does not support it.
    """
    df = pd.concat([
        _season(0, PL_TEAMS),
        _season(1, PL_TEAMS[:-3] + [f"Up{i} FC" for i in range(3)]),
        _season(2, PL_TEAMS[:6]),  # only 6 of 20 teams have played
    ], ignore_index=True)

    out = _add_promotion_flags(df, "PL", None)

    partial = out[out["SeasonIndex"] == 2]
    assert partial["Home_Promoted"].isna().all()
    assert partial["Home_Relegated"].isna().all()
    # A season that *can* be derived is unaffected by the incomplete one.
    assert out[out["SeasonIndex"] == 1]["Home_Promoted"].notna().all()
    assert "incomplete" in capsys.readouterr().out.lower()


def test_first_season_is_null_not_zero():
    """With no prior season there is nothing to difference against.

    Three teams *were* promoted into 2000/01; the canonical simply cannot say
    which. Recording 0 would assert they were not — the same falsehood the
    hand-maintained dicts told for 24 seasons, just smaller.
    """
    df = pd.concat([
        _season(0, PL_TEAMS),
        _season(1, PL_TEAMS[:-3] + [f"Up{i} FC" for i in range(3)]),
    ], ignore_index=True)

    out = _add_promotion_flags(df, "PL", None)

    first = out[out["SeasonIndex"] == 0]
    assert first["Home_Promoted"].isna().all()
    assert first["Away_Promoted"].isna().all()
    assert first["Home_Relegated"].isna().all()


def test_unexpected_arrival_count_is_an_error():
    """A complete season with the wrong number of arrivals is a real fault.

    The invariant held for all 25 season transitions in both leagues, so a
    break means a name-bridge failure or corrupt data. Either is something to
    stop for, not to train on.
    """
    df = pd.concat([
        _season(0, PL_TEAMS),
        _season(1, PL_TEAMS[:-4] + [f"Up{i} FC" for i in range(4)]),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="4 arrivals"):
        _add_promotion_flags(df, "PL", None)


def test_efl_split_must_be_three_down_and_three_up(tmp_path):
    """Six arrivals that do not split 3/3 means the name bridge failed."""
    sibling = tmp_path / "pl.csv"
    # Only one of the three ex-PL sides is present, so the bridge finds 1 not 3.
    _season(0, ["Burnley FC"] + [f"Stayer{i:02d} FC" for i in range(19)]).to_csv(
        sibling, index=False)

    df = pd.concat([
        _season(0, EFL_TEAMS),
        _season(1, EFL_TEAMS[:-6] + ["Burnley", "Leeds", "Sunderland",
                                     "Wrexham", "Charlton", "Barnsley"]),
    ], ignore_index=True)

    with pytest.raises(ValueError, match="1 relegated"):
        _add_promotion_flags(df, "EFL", str(sibling))


def test_efl_without_the_pl_canonical_is_null(tmp_path, capsys):
    """Without the sibling canonical the split is unknowable, so it is null.

    Bootstrapping a fresh checkout must not silently label three relegated
    sides as promoted — they are the strongest teams in the division, not the
    weakest, so the flag would be exactly inverted for them.
    """
    df = pd.concat([
        _season(0, EFL_TEAMS),
        _season(1, EFL_TEAMS[:-6] + [f"New{i:02d}" for i in range(6)]),
    ], ignore_index=True)

    out = _add_promotion_flags(df, "EFL", str(tmp_path / "absent.csv"))

    assert out["Home_Promoted"].isna().all()
    assert out["Home_Relegated"].isna().all()
    assert "no pl canonical" in capsys.readouterr().out.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Against the real canonicals
# ─────────────────────────────────────────────────────────────────────────────

# The hand-maintained dicts this derivation replaces. Seasons 24 and 25 are the
# only ones they populated correctly, so they are the only ones that can serve
# as an independent check — which is itself the argument for deriving.
_HAND_WRITTEN = {
    "PL": {24: {"Ipswich", "Leicester", "Southampton"},
           25: {"Leeds", "Burnley", "Sunderland"}},
    "EFL": {24: {"Derby", "Portsmouth", "Oxford"},
            25: {"Birmingham", "Wrexham", "Charlton"}},
}


@pytest.fixture(scope="module")
def real_flags() -> dict:
    """Derive flags over both real canonicals."""
    import os
    from league_config import LEAGUES

    out = {}
    for league in ("PL", "EFL"):
        path = LEAGUES[league]["csv_path"]
        if not os.path.exists(path):
            pytest.skip(f"{league} canonical not present")
        df = pd.read_csv(path, low_memory=False)
        sibling = LEAGUES["PL"]["csv_path"] if league == "EFL" else None
        out[league] = _add_promotion_flags(df, league, sibling)
    return out


@pytest.mark.parametrize("league", ["PL", "EFL"])
def test_derivation_reproduces_the_hand_written_seasons(real_flags, league):
    """Seasons 24-25 must come out exactly as the old dicts had them."""
    df = real_flags[league]
    for season, expected in _HAND_WRITTEN[league].items():
        rows = df[(df["SeasonIndex"] == season) & (df["Home_Promoted"] == 1)]
        got = set(rows["Home_Team"])
        # The dicts stored short forms; match on substring in either direction.
        for want in expected:
            assert any(want.lower() in g.lower() or g.lower() in want.lower()
                       for g in got), (
                f"{league} s{season}: {want!r} not among derived promoted "
                f"{sorted(got)}")
        assert len(got) == 3, f"{league} s{season}: got {sorted(got)}"


def test_promoted_is_no_longer_a_dead_feature(real_flags):
    """The whole point: the flag must vary across nearly every season.

    It was constant zero in 24 of 26 PL seasons and 21 of 26 EFL seasons.
    """
    for league, df in real_flags.items():
        varying = {
            int(s) for s, grp in df.groupby("SeasonIndex")
            if grp["Home_Promoted"].fillna(0).sum() > 0
        }
        assert len(varying) >= 25, (
            f"{league}: only {len(varying)} seasons have any promoted team")


def test_pl_relegated_is_always_zero(real_flags):
    """Nothing is relegated *into* the top flight — the column exists so both
    leagues keep one schema (ADR 0007 decision 2)."""
    df = real_flags["PL"]
    assert (df["Home_Relegated"].fillna(0) == 0).all()
    assert (df["Away_Relegated"].fillna(0) == 0).all()


def test_efl_relegated_teams_are_flagged(real_flags):
    """Sides down from the PL are flagged relegated, not promoted."""
    df = real_flags["EFL"]
    s25 = df[df["SeasonIndex"] == 25]
    relegated = set(s25[s25["Home_Relegated"] == 1]["Home_Team"])
    # 2024/25 PL relegation: Leicester, Ipswich, Southampton went down.
    for want in ("Leicester", "Ipswich", "Southampton"):
        assert any(want.lower() in t.lower() for t in relegated), (
            f"{want} should be flagged relegated in EFL s25, got {sorted(relegated)}")
