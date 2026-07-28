"""
Build a league's Canonical Dataset from football-data.co.uk season CSVs.

Downloads raw season CSVs (division E0 for PL, E1 for EFL), maps columns to
the canonical schema, and computes all rolling features expected by the
pipeline. Both leagues emit the identical 72-column schema.

football-data.co.uk is the sole authority for Facts in both leagues — see
docs/adr/0004-canonical-composition-and-facts-provenance.md.

Usage:
    python data/build_canonical_dataset.py --league EFL
    python data/build_canonical_dataset.py --league PL --output /tmp/pl_dry_run.csv
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from api.team_mapping import normalize  # noqa: E402
from league_config import get_league_config  # noqa: E402

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"

# ── EFL Championship derbies — names as they appear in football-data.co.uk ──
# (EFL keeps source names verbatim; see _LEAGUES[...]["normalize_names"].)
_EFL_DERBIES_LOCAL: set[tuple[str, str]] = {
    ("Sheffield Weds", "Sheffield United"),
    ("Nottm Forest", "Derby"),
    ("Leeds", "Sheffield United"),
    ("Sunderland", "Middlesbrough"),
    ("Bristol City", "Cardiff"),
    ("Norwich", "Ipswich"),
    ("Burnley", "Blackburn"),
    ("QPR", "Millwall"),
    ("Watford", "Luton"),
}

_EFL_DERBIES_HISTORICAL: set[tuple[str, str]] = {
    ("West Brom", "Wolves"),
    ("Leeds", "Sheffield Weds"),
    ("Preston", "Blackpool"),
    ("Stoke", "Port Vale"),
    ("Coventry", "Leicester"),
    ("Hull", "Sheffield United"),
    ("Derby", "Leicester"),
    ("Sunderland", "Newcastle"),
    ("Huddersfield", "Leeds"),
    ("Birmingham", "West Brom"),
    ("Charlton", "Millwall"),
}

# ── Premier League derbies ──
# In *canonical* form ("Arsenal FC"), because PL normalises team names during
# column mapping and derby detection runs afterwards. Derived from the derby
# flags already present in the PL canonical so a rebuild reproduces them.
_PL_DERBIES_LOCAL: set[tuple[str, str]] = {
    ("Arsenal FC", "Tottenham Hotspur FC"),
    ("Aston Villa FC", "Birmingham City FC"),
    ("Brighton & Hove Albion FC", "Crystal Palace FC"),
    ("Chelsea FC", "Fulham FC"),
    ("Everton FC", "Liverpool FC"),
    ("Manchester City FC", "Manchester United FC"),
    ("Newcastle United FC", "Sunderland AFC"),
    ("West Bromwich Albion FC", "Wolverhampton Wanderers FC"),
}

_PL_DERBIES_HISTORICAL: set[tuple[str, str]] = {
    ("Arsenal FC", "Chelsea FC"),
    ("Arsenal FC", "Manchester United FC"),
    ("Chelsea FC", "Liverpool FC"),
    ("Chelsea FC", "Manchester United FC"),
    ("Liverpool FC", "Manchester City FC"),
    ("Liverpool FC", "Manchester United FC"),
}

# ── Per-league build settings ──
# Only what is genuinely league-specific lives here; division code, output
# path and season range come from league_config so there is one source of truth.
_LEAGUES: dict[str, dict[str, Any]] = {
    "PL": {
        "raw_dir": os.path.join(PROJECT_DIR, "data", "pl_raw"),
        "derbies_local": _PL_DERBIES_LOCAL,
        "derbies_historical": _PL_DERBIES_HISTORICAL,
        # PL canonical uses long-form names ("Arsenal FC"), so source names
        # must be normalised on the way in.
        "normalize_names": True,
    },
    "EFL": {
        "raw_dir": os.path.join(PROJECT_DIR, "data", "championship_raw"),
        "derbies_local": _EFL_DERBIES_LOCAL,
        "derbies_historical": _EFL_DERBIES_HISTORICAL,
        # EFL canonical uses football-data.co.uk's own short forms
        # ("Blackburn"). Normalising would rewrite 23 of 24 names and break
        # the ESPN/odds mappings and the Betfair League Split allowlists.
        "normalize_names": False,
    },
}


def _settings(league: str, output: str | None = None) -> dict[str, Any]:
    """Resolve the build settings for a league.

    Args:
        league: "PL" or "EFL".
        output: Override the output CSV path (used for dry runs).

    Returns:
        Merged settings dict.

    Raises:
        ValueError: If the league is unknown.
    """
    if league not in _LEAGUES:
        raise ValueError(
            f"Unknown league {league!r}; expected one of {sorted(_LEAGUES)}")
    cfg = get_league_config(league)
    settings = dict(_LEAGUES[league])
    settings["league"] = league
    settings["div"] = cfg["football_data_co_uk_div"]
    settings["output_path"] = output or cfg["csv_path"]
    # The live canonical is the preservation source and is tracked separately
    # from output_path: a dry run writes elsewhere but must still read the
    # real canonical, or it would "reproduce" the build by dropping the very
    # rows preservation exists to keep.
    settings["canonical_path"] = cfg["csv_path"]
    # The EFL split needs the division above: an arrival that was in the PL
    # last season came *down*, and is one of the division's stronger teams
    # rather than one of its weakest (ADR 0007 decisions 1-2). The PL needs no
    # sibling — every arrival there came up.
    settings["sibling_canonical_path"] = (
        get_league_config("PL")["csv_path"] if league == "EFL" else None)
    settings["first_season"] = cfg["first_season_idx"]
    settings["last_season"] = cfg["current_season_idx"]
    return settings


def _season_code(season_idx: int) -> str:
    """Convert season index to football-data.co.uk URL code.

    Season 0 = 2000/01 -> '0001', Season 24 = 2024/25 -> '2425'.
    """
    start = season_idx % 100
    end = (start + 1) % 100
    return f"{start:02d}{end:02d}"


def download_season(
    season_idx: int,
    div: str,
    raw_dir: str,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Download a single season CSV from football-data.co.uk.

    Args:
        season_idx: Season index (0 = 2000/01).
        div: football-data.co.uk division code ("E0" PL, "E1" EFL).
        raw_dir: Directory for the raw-CSV cache.
        use_cache: Reuse the cached raw if present. Finished seasons are
            immutable so caching them is always safe; the *current* season
            gains fixtures every matchday, so its cache must be bypassed or
            the canonical silently freezes at whenever the cache was written.

    Returns:
        DataFrame with raw match data, or None if download fails.
    """
    code = _season_code(season_idx)
    url = BASE_URL.format(code=code, div=div)

    os.makedirs(raw_dir, exist_ok=True)
    cache_path = os.path.join(raw_dir, f"{div}_{code}.csv")

    # Use cached version if available
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, encoding="utf-8", on_bad_lines="skip")
            if len(df) > 0:
                # Apply the same malformed-row filter as the download path,
                # otherwise a junk row in a cached raw (e.g. E1_1415 has one
                # all-NaN row) leaks into the canonical.
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"],
                               how="any")
                print(f"  Season {code}: loaded from cache ({len(df)} matches)")
                return df
        except Exception:
            pass  # Re-download on cache corruption

    print(f"  Season {code}: downloading from football-data.co.uk...")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  Season {code}: download failed — {e}")
        return None

    # Save raw CSV
    with open(cache_path, "wb") as f:
        f.write(resp.content)

    try:
        df = pd.read_csv(io.StringIO(resp.text), encoding="utf-8", on_bad_lines="skip")
    except Exception as e:
        print(f"  Season {code}: parse failed — {e}")
        return None

    # Drop fully-empty rows (football-data.co.uk sometimes has trailing empties)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"], how="any")
    print(f"  Season {code}: downloaded ({len(df)} matches)")
    time.sleep(1.5)  # Be polite to the server
    return df


def _map_columns(
    df: pd.DataFrame,
    season_idx: int,
    normalize_names: bool = False,
) -> pd.DataFrame:
    """Map football-data.co.uk columns to the canonical schema.

    Args:
        df: Raw football-data.co.uk season frame.
        season_idx: Season index to stamp on every row.
        normalize_names: Map source team names to canonical form (PL only).
    """
    out = pd.DataFrame()

    # Parse date — format varies by season (DD/MM/YY or DD/MM/YYYY)
    out["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    out["Home_Team"] = df["HomeTeam"].astype(str).str.strip()
    out["Away_Team"] = df["AwayTeam"].astype(str).str.strip()
    if normalize_names:
        out["Home_Team"] = out["Home_Team"].map(normalize)
        out["Away_Team"] = out["Away_Team"].map(normalize)
    out["Home_Goals"] = pd.to_numeric(df["FTHG"], errors="coerce")
    out["Away_Goals"] = pd.to_numeric(df["FTAG"], errors="coerce")
    out["TG"] = out["Home_Goals"] + out["Away_Goals"]
    out["FTR"] = df["FTR"]

    # Half-time
    out["HTHG"] = pd.to_numeric(df.get("HTHG"), errors="coerce")
    out["HTAG"] = pd.to_numeric(df.get("HTAG"), errors="coerce")
    out["HTR"] = df.get("HTR")

    # Match stats
    out["Home_Shots"] = pd.to_numeric(df.get("HS"), errors="coerce")
    out["Away_Shots"] = pd.to_numeric(df.get("AS"), errors="coerce")
    out["Home_Shots_Target"] = pd.to_numeric(df.get("HST"), errors="coerce")
    out["Away_Shots_Target"] = pd.to_numeric(df.get("AST"), errors="coerce")
    out["HF"] = pd.to_numeric(df.get("HF"), errors="coerce")
    out["AF"] = pd.to_numeric(df.get("AF"), errors="coerce")
    out["Home_Corners"] = pd.to_numeric(df.get("HC"), errors="coerce")
    out["Away_Corners"] = pd.to_numeric(df.get("AC"), errors="coerce")
    out["HY"] = pd.to_numeric(df.get("HY"), errors="coerce")
    out["AY"] = pd.to_numeric(df.get("AY"), errors="coerce")
    out["HR"] = pd.to_numeric(df.get("HR"), errors="coerce")
    out["AR"] = pd.to_numeric(df.get("AR"), errors="coerce")

    # Betting odds (B365 match result)
    out["B365H"] = pd.to_numeric(df.get("B365H"), errors="coerce")
    out["B365D"] = pd.to_numeric(df.get("B365D"), errors="coerce")
    out["B365A"] = pd.to_numeric(df.get("B365A"), errors="coerce")

    # O/U 2.5 odds — column names vary by era
    if "B365>2.5" in df.columns:
        out["B365Greater2.5"] = pd.to_numeric(df["B365>2.5"], errors="coerce")
        out["B365LessThan2.5"] = pd.to_numeric(df["B365<2.5"], errors="coerce")
    elif "BbAv>2.5" in df.columns:
        # Use Betbrain average as fallback for older seasons
        out["B365Greater2.5"] = pd.to_numeric(df["BbAv>2.5"], errors="coerce")
        out["B365LessThan2.5"] = pd.to_numeric(df["BbAv<2.5"], errors="coerce")
    else:
        out["B365Greater2.5"] = np.nan
        out["B365LessThan2.5"] = np.nan

    out["SeasonIndex"] = season_idx
    out["DateOnly"] = out["Date"].dt.strftime("%Y-%m-%d")

    return out


def _is_derby(home: str, away: str, derby_set: set[tuple[str, str]]) -> bool:
    """Check if a fixture is a derby (order-independent)."""
    pair = (home, away)
    rev = (away, home)
    for d in derby_set:
        if pair == d or rev == d:
            return True
        # Fuzzy: check if either team name is contained in the derby tuple
        h_low, a_low = home.lower(), away.lower()
        d0_low, d1_low = d[0].lower(), d[1].lower()
        if ((d0_low in h_low or h_low in d0_low) and
                (d1_low in a_low or a_low in d1_low)):
            return True
        if ((d0_low in a_low or a_low in d0_low) and
                (d1_low in h_low or h_low in d1_low)):
            return True
    return False


def _season_teams(df: pd.DataFrame, season: int) -> set[str]:
    """Every team that appears in a season, on either side of a fixture."""
    rows = df[df["SeasonIndex"] == season]
    return set(rows["Home_Team"]) | set(rows["Away_Team"])


def _down_slots(league: str) -> int:
    """How many teams drop out of a division each season.

    Equally, how many arrive from the division below — the two are the same
    number, which is what makes the arrival count checkable.
    """
    cfg = get_league_config(league)
    return cfg["teams_count"] - cfg["relegation_pos"] + 1


def _add_promotion_flags(
    df: pd.DataFrame,
    league: str,
    sibling_canonical_path: str | None,
) -> pd.DataFrame:
    """Derive Home_/Away_Promoted and Home_/Away_Relegated from the canonicals.

    A team in season N but not in N-1 is new to the division. In the PL that
    can only mean promotion. In the EFL it is ambiguous, and only the PL
    canonical separates a side relegated from above (one of the division's
    *stronger* teams) from one promoted from League One (one of its weakest).

    Args:
        df: The frame being built, with SeasonIndex and team columns.
        league: "PL" or "EFL".
        sibling_canonical_path: The PL canonical, required for the EFL split.

    Returns:
        *df* with the four flag columns added.
    """
    df = df.copy()
    df["Home_Promoted"] = 0
    df["Away_Promoted"] = 0
    df["Home_Relegated"] = 0
    df["Away_Relegated"] = 0

    sibling = None
    if league == "EFL":
        if sibling_canonical_path and os.path.exists(sibling_canonical_path):
            sibling = pd.read_csv(sibling_canonical_path, low_memory=False)
        else:
            print("  WARNING: no PL canonical available — cannot tell a side "
                  "relegated into the EFL from one promoted into it. Both "
                  "flags left null for every season.")
            df["Home_Promoted"] = np.nan
            df["Away_Promoted"] = np.nan
            df["Home_Relegated"] = np.nan
            df["Away_Relegated"] = np.nan
            return df

    expected_teams = get_league_config(league)["teams_count"]
    seasons = sorted(df["SeasonIndex"].dropna().unique())

    def _complete(season: int) -> bool:
        return len(_season_teams(df, season)) == expected_teams

    # The earliest season has nothing to difference against. Three teams were
    # promoted into it too; the canonical simply cannot say which, and 0 would
    # assert they were not.
    for col in ("Home_Promoted", "Away_Promoted",
                "Home_Relegated", "Away_Relegated"):
        df.loc[df["SeasonIndex"] == seasons[0], col] = np.nan

    for season in seasons[1:]:
        # Arrivals are a difference between two seasons, so both must be whole.
        # A part-loaded season — the state of season 26 through early August —
        # cannot say who is new, and guessing would assert something false.
        if not (_complete(season) and _complete(season - 1)):
            print(f"  WARNING: season {int(season)} incomplete "
                  f"({len(_season_teams(df, season))} of {expected_teams} "
                  f"teams) — promotion flags left null")
            for col in ("Home_Promoted", "Away_Promoted",
                        "Home_Relegated", "Away_Relegated"):
                df.loc[df["SeasonIndex"] == season, col] = np.nan
            continue

        arrivals = _season_teams(df, season) - _season_teams(df, season - 1)

        relegated: set[str] = set()
        if sibling is not None:
            above = _season_teams(sibling, season - 1)
            relegated = {t for t in arrivals if normalize(t) in above}
        promoted = arrivals - relegated

        if league == "PL":
            if len(arrivals) != _down_slots("PL"):
                raise ValueError(
                    f"PL season {int(season)}: {len(arrivals)} arrivals, "
                    f"expected {_down_slots('PL')} — {sorted(arrivals)}. "
                    f"Either the canonical is corrupt or team names drifted "
                    f"between seasons.")
        else:
            if len(relegated) != _down_slots("PL") or \
                    len(promoted) != _down_slots("EFL"):
                raise ValueError(
                    f"EFL season {int(season)}: {len(relegated)} relegated + "
                    f"{len(promoted)} promoted, expected "
                    f"{_down_slots('PL')} + {_down_slots('EFL')}. "
                    f"Arrivals {sorted(arrivals)}; matched as down from the PL: "
                    f"{sorted(relegated)}. A shortfall means normalize() failed "
                    f"to bridge an EFL short form to its PL canonical name.")

        mask = df["SeasonIndex"] == season
        for team in promoted:
            df.loc[mask & (df["Home_Team"] == team), "Home_Promoted"] = 1
            df.loc[mask & (df["Away_Team"] == team), "Away_Promoted"] = 1
        for team in relegated:
            df.loc[mask & (df["Home_Team"] == team), "Home_Relegated"] = 1
            df.loc[mask & (df["Away_Team"] == team), "Away_Relegated"] = 1

    return df


def _add_derby_flags(
    df: pd.DataFrame,
    derbies_local: set[tuple[str, str]],
    derbies_historical: set[tuple[str, str]],
) -> pd.DataFrame:
    """Add Local Derby, Historical Derby, and Not a Derby columns."""
    local = []
    historical = []
    for _, row in df.iterrows():
        h, a = row["Home_Team"], row["Away_Team"]
        is_local = _is_derby(h, a, derbies_local)
        is_hist = _is_derby(h, a, derbies_historical)
        local.append(1 if is_local else 0)
        historical.append(1 if (is_hist and not is_local) else 0)
    df["Local Derby"] = local
    df["Historical Derby"] = historical
    df["Not a Derby"] = ((df["Local Derby"] == 0) & (df["Historical Derby"] == 0)).astype(int)
    return df


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling mean with min_periods=1, shifted by 1 (no lookahead)."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling sum with min_periods=1, shifted by 1."""
    return series.shift(1).rolling(window, min_periods=1).sum()


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling features per team, matching the PL CSV schema.

    Features are computed from each team's perspective across all their
    matches (home and away), then mapped back to Home_/Away_ columns.
    """
    # Build per-team match history
    teams = sorted(set(df["Home_Team"].unique()) | set(df["Away_Team"].unique()))

    # Create team-level match log
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        date = row["Date"]
        si = row["SeasonIndex"]
        hg = row["Home_Goals"]
        ag = row["Away_Goals"]

        # Home team record
        records.append({
            "team": row["Home_Team"], "date": date, "season": si,
            "goals_scored": hg, "goals_conceded": ag,
            "shots": row.get("Home_Shots", np.nan),
            "shots_target": row.get("Home_Shots_Target", np.nan),
            "shots_against": row.get("Away_Shots", np.nan),
            "shots_target_against": row.get("Away_Shots_Target", np.nan),
            "corners": row.get("Home_Corners", np.nan),
            "corners_conceded": row.get("Away_Corners", np.nan),
            "is_home": True,
            "match_idx": row.name,
        })
        # Away team record
        records.append({
            "team": row["Away_Team"], "date": date, "season": si,
            "goals_scored": ag, "goals_conceded": hg,
            "shots": row.get("Away_Shots", np.nan),
            "shots_target": row.get("Away_Shots_Target", np.nan),
            "shots_against": row.get("Home_Shots", np.nan),
            "shots_target_against": row.get("Home_Shots_Target", np.nan),
            "corners": row.get("Away_Corners", np.nan),
            "corners_conceded": row.get("Home_Corners", np.nan),
            "is_home": False,
            "match_idx": row.name,
        })

    team_log = pd.DataFrame(records)
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute rolling features per team
    feat_map: dict[int, dict[str, float]] = {}  # match_idx -> {feature: value}

    for team, grp in team_log.groupby("team"):
        g = grp.copy()

        # Shots
        g["avg_shots_5"] = _rolling_mean(g["shots"], 5)
        g["avg_sot_5"] = _rolling_mean(g["shots_target"], 5)
        shots_nz = g["shots"].replace(0, np.nan)
        sot_ratio = g["shots_target"] / shots_nz
        g["shot_ratio_5"] = _rolling_mean(sot_ratio, 5)

        # Shots per goal
        gs_nz = g["goals_scored"].replace(0, np.nan)
        g["shots_per_goal_5"] = _rolling_mean(g["shots"] / gs_nz, 5)

        # Conversion rates
        shots_shifted = g["shots"].shift(1)
        sot_shifted = g["shots_target"].shift(1)
        gs_shifted = g["goals_scored"].shift(1)

        g["cr_5"] = _rolling_mean(g["goals_scored"] / shots_nz, 5)
        g["cr_20"] = _rolling_mean(g["goals_scored"] / shots_nz, 20)

        sot_nz = g["shots_target"].replace(0, np.nan)
        g["sot_cr_5"] = _rolling_mean(g["goals_scored"] / sot_nz, 5)
        g["sot_cr_20"] = _rolling_mean(g["goals_scored"] / sot_nz, 20)

        # Defensive strength (shots conceded / shots against)
        sa_nz = g["shots_against"].replace(0, np.nan)
        g["def_strength_5"] = _rolling_mean(g["shots_target_against"] / sa_nz, 5)

        sta_nz = g["shots_target_against"].replace(0, np.nan)
        g["def_strength_sot"] = _rolling_mean(
            g["goals_conceded"] / sta_nz, 5
        )

        # Goals
        g["past5_goals"] = _rolling_sum(g["goals_scored"], 5)
        g["past5_conceded"] = _rolling_sum(g["goals_conceded"], 5)

        # Corners
        g["past5_corners"] = _rolling_sum(g["corners"], 5)
        g["past5_corners_conceded"] = _rolling_sum(g["corners_conceded"], 5)

        # Store features keyed by match index
        for _, r in g.iterrows():
            midx = r["match_idx"]
            prefix = "home" if r["is_home"] else "away"
            if midx not in feat_map:
                feat_map[midx] = {}
            feat_map[midx][f"{prefix}_avg_shots_5"] = r["avg_shots_5"]
            feat_map[midx][f"{prefix}_avg_sot_5"] = r["avg_sot_5"]
            feat_map[midx][f"{prefix}_shot_ratio_5"] = r["shot_ratio_5"]
            feat_map[midx][f"{prefix}_shots_per_goal_5"] = r["shots_per_goal_5"]
            feat_map[midx][f"{prefix}_cr_5"] = r["cr_5"]
            feat_map[midx][f"{prefix}_cr_20"] = r["cr_20"]
            feat_map[midx][f"{prefix}_sot_cr_5"] = r["sot_cr_5"]
            feat_map[midx][f"{prefix}_sot_cr_20"] = r["sot_cr_20"]
            feat_map[midx][f"{prefix}_def_5"] = r["def_strength_5"]
            feat_map[midx][f"{prefix}_def_sot"] = r["def_strength_sot"]
            feat_map[midx][f"{prefix}_past5_goals"] = r["past5_goals"]
            feat_map[midx][f"{prefix}_past5_conceded"] = r["past5_conceded"]
            feat_map[midx][f"{prefix}_past5_corners"] = r["past5_corners"]
            feat_map[midx][f"{prefix}_past5_corners_conc"] = r["past5_corners_conceded"]

    # Map computed features back to main DataFrame
    feat_df = pd.DataFrame.from_dict(feat_map, orient="index")
    feat_df.index.name = "match_idx"

    col_map = {
        "home_avg_shots_5": "Home_AvgShots_5",
        "home_avg_sot_5": "Home_AvgShotsOnTarget_5",
        "away_avg_shots_5": "Away_AvgShots_5",
        "away_avg_sot_5": "Away_AvgShotsOnTarget_5",
        "home_shot_ratio_5": "Home_ShotRatio_5",
        "away_shot_ratio_5": "Away_ShotRatio_5",
        "home_shots_per_goal_5": "Home_ShotsPerGoal_5",
        "away_shots_per_goal_5": "Away_ShotsPerGoal_5",
        "home_cr_5": "Home_CR_5",
        "home_cr_20": "Home_CR_20",
        "away_cr_5": "Away_CR_5",
        "away_cr_20": "Away_CR_20",
        "home_sot_cr_5": "Home_SOT_CR_5",
        "home_sot_cr_20": "Home_SOT_CR_20",
        "away_sot_cr_5": "Away_SOT_CR_5",
        "away_sot_cr_20": "Away_SOT_CR_20",
        "home_def_5": "Home_DefensiveStrength_5",
        "away_def_5": "Away_DefensiveStrength_5",
        "home_def_sot": "Home_DefensiveStrength_SOT",
        "away_def_sot": "Away_DefensiveStrength_SOT",
        "home_past5_goals": "Home_Past5Goals",
        "away_past5_goals": "Away_Past5Goals",
        "home_past5_conceded": "Home_Past5Conceded",
        "away_past5_conceded": "Away_Past5Conceded",
        "home_past5_corners": "Home_Past5Corners",
        "away_past5_corners": "Away_Past5Corners",
        "home_past5_corners_conc": "Home_Past5CornersConceded",
        "away_past5_corners_conc": "Away_Past5CornersConceded",
    }
    feat_df = feat_df.rename(columns=col_map)
    df = df.join(feat_df, how="left")
    return df


def _add_league_position(df: pd.DataFrame) -> pd.DataFrame:
    """Compute running league position for each team within each season."""
    positions: list[dict[str, int | float]] = []

    for _, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        points: dict[str, int] = {}
        gd: dict[str, int] = {}

        for idx, row in season_matches.iterrows():
            home, away = row["Home_Team"], row["Away_Team"]
            for t in [home, away]:
                if t not in points:
                    points[t] = 0
                    gd[t] = 0

            # Position BEFORE this match
            standings = sorted(
                points.keys(),
                key=lambda t: (-points[t], -gd[t], t),
            )
            pos_map = {t: i + 1 for i, t in enumerate(standings)}
            positions.append({
                "idx": idx,
                "Home_LeaguePosition": pos_map.get(home, 12),
                "Away_LeaguePosition": pos_map.get(away, 12),
            })

            # Update standings with this match result
            hg = row["Home_Goals"]
            ag = row["Away_Goals"]
            if pd.notna(hg) and pd.notna(ag):
                hg, ag = int(hg), int(ag)
                gd[home] += (hg - ag)
                gd[away] += (ag - hg)
                ftr = row.get("FTR", "")
                if ftr == "H":
                    points[home] += 3
                elif ftr == "A":
                    points[away] += 3
                elif ftr == "D":
                    points[home] += 1
                    points[away] += 1

    pos_df = pd.DataFrame(positions).set_index("idx")
    df["Home_LeaguePosition"] = pos_df["Home_LeaguePosition"]
    df["Away_LeaguePosition"] = pos_df["Away_LeaguePosition"]
    return df


def _add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head stats for each fixture."""
    df["H2H_HomeWins"] = 0
    df["H2H_AwayWins"] = 0
    df["H2H_Draws"] = 0
    df["H2H_AvgGoals_5"] = np.nan
    df["H2HAvgGoals"] = np.nan
    df["H2H_MatchCount"] = 0

    # Build H2H history
    h2h_results: dict[tuple[str, str], list[dict]] = {}

    for idx, row in df.sort_values("Date").iterrows():
        home, away = row["Home_Team"], row["Away_Team"]
        key = tuple(sorted([home, away]))

        history = h2h_results.get(key, [])

        if len(history) > 0:
            # Compute H2H stats from PRIOR meetings
            hw = sum(1 for m in history if m["winner"] == home)
            aw = sum(1 for m in history if m["winner"] == away)
            draws = sum(1 for m in history if m["winner"] is None)
            all_goals = [m["tg"] for m in history]
            recent_goals = all_goals[-5:] if len(all_goals) >= 5 else all_goals

            df.at[idx, "H2H_HomeWins"] = hw
            df.at[idx, "H2H_AwayWins"] = aw
            df.at[idx, "H2H_Draws"] = draws
            df.at[idx, "H2H_AvgGoals_5"] = np.mean(recent_goals) if recent_goals else np.nan
            df.at[idx, "H2HAvgGoals"] = np.mean(all_goals) if all_goals else np.nan
            df.at[idx, "H2H_MatchCount"] = len(history)

        # Add this match to history
        hg = row["Home_Goals"]
        ag = row["Away_Goals"]
        if pd.notna(hg) and pd.notna(ag):
            ftr = row.get("FTR", "")
            winner = home if ftr == "H" else (away if ftr == "A" else None)
            history.append({"tg": hg + ag, "winner": winner})
            h2h_results[key] = history

    return df


def _add_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Home/Away Factor (rolling goal ratio vs league average)."""
    df["Home Factor"] = np.nan
    df["Away Factor"] = np.nan

    for _, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        team_home_goals: dict[str, list[float]] = {}
        team_away_goals: dict[str, list[float]] = {}

        for idx, row in season_matches.iterrows():
            home, away = row["Home_Team"], row["Away_Team"]
            hg = row["Home_Goals"]
            ag = row["Away_Goals"]

            # Factor BEFORE this match
            h_hist = team_home_goals.get(home, [])
            a_hist = team_away_goals.get(away, [])

            if len(h_hist) >= 3:
                league_avg_home = np.mean(
                    [g for goals in team_home_goals.values() for g in goals]
                )
                if league_avg_home > 0:
                    df.at[idx, "Home Factor"] = np.mean(h_hist[-10:]) / league_avg_home

            if len(a_hist) >= 3:
                league_avg_away = np.mean(
                    [g for goals in team_away_goals.values() for g in goals]
                )
                if league_avg_away > 0:
                    df.at[idx, "Away Factor"] = np.mean(a_hist[-10:]) / league_avg_away

            # Record this match
            if pd.notna(hg):
                team_home_goals.setdefault(home, []).append(float(hg))
            if pd.notna(ag):
                team_away_goals.setdefault(away, []).append(float(ag))

    return df


def _preserve_canonical_only_rows(
    df: pd.DataFrame,
    canonical_path: str,
    first_season: int,
    last_season: int,
) -> pd.DataFrame:
    """Re-add matches the canonical holds that the upstream source no longer serves.

    football-data.co.uk does not always serve a complete season forever: `E0`
    seasons 3 and 4 return 335 rows against the 380 that were actually played,
    verified against `Content-Length` — an upstream gap, not a truncated
    download. A rebuild that trusts the source therefore *destroys* 90 real
    Premier League fixtures, which ADR 0001 forbids.

    This must run *before* feature computation, not after. Rolling windows are
    computed over whatever rows exist at that moment, so restoring fixtures
    afterwards would leave correct Facts carrying features derived from a
    335-match season — the same silent corruption in a subtler form.

    Matching is on (SeasonIndex, Home_Team, Away_Team), which is unique within
    a double round-robin and, unlike Date, survives a rescheduled fixture.

    Args:
        df: The freshly mapped upstream frame.
        canonical_path: The live canonical to preserve from.
        first_season: First season index the build covers.
        last_season: Last season index the build covers.

    Returns:
        *df* plus any canonical-only rows, re-sorted by Date.
    """
    if not os.path.exists(canonical_path):
        print("  No existing canonical — nothing to preserve "
              f"({canonical_path})")
        return df

    canonical = pd.read_csv(canonical_path, low_memory=False)
    canonical = canonical[
        canonical["SeasonIndex"].between(first_season, last_season)]

    key = ["SeasonIndex", "Home_Team", "Away_Team"]
    built_keys = set(map(tuple, df[key].itertuples(index=False, name=None)))
    missing_mask = ~canonical[key].apply(tuple, axis=1).isin(built_keys)
    missing = canonical[missing_mask]

    if missing.empty:
        print("  Upstream serves every canonical row — nothing to preserve")
        return df

    # Carry only the columns the upstream mapping produces; every computed
    # feature is recomputed downstream over the completed frame.
    cols = [c for c in df.columns if c in missing.columns]
    restored = missing[cols].copy()
    restored["Date"] = pd.to_datetime(restored["Date"], format="mixed",
                                      dayfirst=True)

    per_season = restored.groupby("SeasonIndex").size().to_dict()
    print(f"  Preserved {len(restored)} canonical row(s) absent upstream: "
          f"{ {int(k): int(v) for k, v in per_season.items()} }")

    out = pd.concat([df, restored], ignore_index=True)
    return out.sort_values("Date").reset_index(drop=True)


def _backfill_canonical_values(
    df: pd.DataFrame,
    canonical_path: str,
    first_season: int,
    last_season: int,
) -> pd.DataFrame:
    """Fill cells the source leaves empty but the canonical has a value for.

    Preserving rows is not sufficient. football-data.co.uk carries no B365
    columns at all for `E0` seasons 0-1, yet the canonical holds 380 match
    prices for each and 374 O/U prices for season 1 — inherited from the
    dataset that predates the builder. Those rows *are* served, so row
    preservation never looks at them, and a rebuild silently blanks the very
    odds the backtests price against.

    Only gaps are filled. Where the source supplies a value it wins, so a
    genuine upstream correction is never reverted to a stale canonical one.

    Args:
        df: The frame being built, after row preservation.
        canonical_path: The live canonical to backfill from.
        first_season: First season index the build covers.
        last_season: Last season index the build covers.

    Returns:
        *df* with source-column gaps filled where the canonical can fill them.
    """
    if not os.path.exists(canonical_path):
        return df

    canonical = pd.read_csv(canonical_path, low_memory=False)
    canonical = canonical[
        canonical["SeasonIndex"].between(first_season, last_season)]
    if canonical.empty:
        return df

    key = ["SeasonIndex", "Home_Team", "Away_Team"]
    fillable = [c for c in df.columns
                if c in canonical.columns and c not in key and c != "Date"]

    indexed = canonical.set_index(key)
    indexed = indexed[~indexed.index.duplicated(keep="first")]
    target = df.set_index(key)

    filled: dict[str, int] = {}
    for col in fillable:
        gaps = target[col].isna()
        if not gaps.any():
            continue
        source = indexed[col].reindex(target.index)
        n = int((gaps & source.notna()).sum())
        if n:
            target.loc[gaps, col] = source[gaps]
            filled[col] = n

    if filled:
        print(f"  Backfilled {sum(filled.values())} cell(s) the source omits: "
              f"{filled}")
    else:
        print("  No canonical values to backfill")

    return target.reset_index()[df.columns.tolist()]


def build(
    league: str = "EFL",
    output: str | None = None,
    refresh_current_season: bool = True,
) -> pd.DataFrame:
    """Download, merge, and feature-engineer a league's Canonical Dataset.

    A full rebuild is cheap (~17s for 26 EFL seasons off cached raws), so
    there is deliberately no incremental-append path: an append is a second
    implementation of the same contract and drifts from the rebuild unless
    something forces them to agree. See ADR 0004.

    Args:
        league: "PL" or "EFL".
        output: Override output path. Use for dry runs so the live
            canonical is not overwritten.
        refresh_current_season: Re-download the current season instead of
            trusting its cache. Finished seasons are immutable and always
            cached. Set False for hermetic (offline) test runs.

    Returns:
        The built DataFrame.

    Raises:
        RuntimeError: If no season data could be downloaded.
    """
    s = _settings(league, output)
    print(f"=== Building {league} Canonical Dataset "
          f"(division {s['div']}) ===\n")

    # Step 1: Download all seasons
    season_dfs: list[pd.DataFrame] = []
    for si in range(s["first_season"], s["last_season"] + 1):
        # The current season gains fixtures every matchday, so its cached raw
        # goes stale the moment a game is played. Older seasons never change.
        is_current = si == s["last_season"]
        use_cache = not (is_current and refresh_current_season)
        raw = download_season(si, s["div"], s["raw_dir"], use_cache=use_cache)
        if raw is not None and len(raw) > 0:
            mapped = _map_columns(raw, si, s["normalize_names"])
            season_dfs.append(mapped)

    if not season_dfs:
        raise RuntimeError(f"No {league} data downloaded.")

    df = pd.concat(season_dfs, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"\nTotal matches from source: {len(df)}")

    # Step 1b: Restore matches the source has stopped serving. Must precede
    # feature computation — see _preserve_canonical_only_rows.
    print("\nChecking for canonical rows absent upstream...")
    df = _preserve_canonical_only_rows(
        df, s["canonical_path"], s["first_season"], s["last_season"])
    df = _backfill_canonical_values(
        df, s["canonical_path"], s["first_season"], s["last_season"])
    print(f"Total matches: {len(df)}")

    # Step 2: Add rolling features
    print("\nComputing rolling features...")
    df = _add_rolling_features(df)

    # Step 3: League position
    print("Computing league positions...")
    df = _add_league_position(df)

    # Step 4: H2H
    print("Computing H2H stats...")
    df = _add_h2h(df)

    # Step 5: Promotion / relegation flags, derived from the canonicals
    print("Deriving promotion flags...")
    df = _add_promotion_flags(df, league, s["sibling_canonical_path"])

    # Step 6: Derby flags
    print("Adding derby flags...")
    df = _add_derby_flags(df, s["derbies_local"], s["derbies_historical"])

    # Step 7: Home/Away Factor
    print("Computing home/away factor...")
    df = _add_factor(df)

    # Save
    df.to_csv(s["output_path"], index=False)
    print(f"\nSaved to {s['output_path']}")
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['SeasonIndex'].min()} - {df['SeasonIndex'].max()}")
    print(f"Unique teams: {df['Home_Team'].nunique()}")
    print(f"O/U odds available: {df['B365Greater2.5'].notna().sum()} / {len(df)} matches")

    return df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build a league's Canonical Dataset from football-data.co.uk")
    parser.add_argument("--league", choices=sorted(_LEAGUES), default="EFL",
                        help="League to build (default: EFL)")
    parser.add_argument("--output", default=None,
                        help="Override output path — use for dry runs so the "
                             "live canonical is not overwritten")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Trust the cached current-season raw instead of "
                             "re-downloading it (offline/testing)")
    args = parser.parse_args()
    build(league=args.league, output=args.output,
          refresh_current_season=not args.no_refresh)


if __name__ == "__main__":
    main()
