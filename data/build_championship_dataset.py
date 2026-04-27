"""
Build CompleteDSChamp_CSV.csv from football-data.co.uk Championship CSVs.

Downloads raw season CSVs (division E1, 2000/01-2024/25), maps columns to
the PL schema, and computes all rolling features expected by the pipeline.

Usage:
    python data/build_championship_dataset.py
"""
from __future__ import annotations

import io
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "championship_raw")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "CompleteDSChamp_CSV.csv")

# football-data.co.uk season codes: "0001" = 2000/01, "2425" = 2024/25
FIRST_SEASON = 0   # 2000/01
LAST_SEASON = 24    # 2024/25
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E1.csv"

# Championship derbies — team names as they appear in football-data.co.uk
DERBIES_LOCAL: set[tuple[str, str]] = {
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

DERBIES_HISTORICAL: set[tuple[str, str]] = {
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

# Promoted teams by season index (from League One into Championship)
# Key = season index, Value = set of team names that were promoted INTO that season.
# Sources: historical records.  Empty seasons = no data or not yet mapped.
PROMOTED_TEAMS: dict[int, set[str]] = {
    1:  {"Millwall", "Rotherham", "Blackpool"},
    2:  {"Wigan", "Crewe", "Cardiff"},
    3:  {"Wolves", "Plymouth", "West Ham"},
    4:  {"Luton", "Hull", "Doncaster"},
    5:  {"Southend", "Colchester", "Barnsley"},
    6:  {"Scunthorpe", "Bristol City", "Blackpool"},
    7:  {"Swansea", "Nott'm Forest", "Doncaster"},
    8:  {"Leicester", "Peterboro", "Scunthorpe"},
    9:  {"Norwich", "Leeds", "Millwall"},
    10: {"Brighton", "Southampton", "Peterboro"},
    11: {"Charlton", "Sheff Utd", "Huddersfield"},
    12: {"Doncaster", "Bournemouth", "Yeovil"},
    13: {"Wolves", "Brentford", "Rotherham"},
    14: {"Bristol City", "MK Dons", "Preston"},
    15: {"Wigan", "Burton", "Barnsley"},
    16: {"Sheffield Weds", "Bolton", "Millwall"},  # approximate
    17: {"Wigan", "Blackburn", "Rotherham"},  # approximate
    18: {"Luton", "Charlton", "Barnsley"},  # approximate
    19: {"Coventry", "Rotherham", "Wycombe"},
    20: {"Hull", "Peterboro", "Blackpool"},
    21: {"Wigan", "Rotherham", "Sunderland"},
    22: {"Sheff Wed", "Ipswich", "Plymouth"},
    23: {"Leyton Orient", "Stevenage", "Carlisle"},  # approximate; adjust
    24: {"Derby", "Portsmouth", "Oxford"},  # 2024/25 promoted from L1
}


def _season_code(season_idx: int) -> str:
    """Convert season index to football-data.co.uk URL code.

    Season 0 = 2000/01 -> '0001', Season 24 = 2024/25 -> '2425'.
    """
    start = season_idx % 100
    end = (start + 1) % 100
    return f"{start:02d}{end:02d}"


def download_season(season_idx: int) -> pd.DataFrame | None:
    """Download a single season CSV from football-data.co.uk.

    Returns:
        DataFrame with raw match data, or None if download fails.
    """
    code = _season_code(season_idx)
    url = BASE_URL.format(code=code)

    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, f"E1_{code}.csv")

    # Use cached version if available
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, encoding="utf-8", on_bad_lines="skip")
            if len(df) > 0:
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


def _map_columns(df: pd.DataFrame, season_idx: int) -> pd.DataFrame:
    """Map football-data.co.uk columns to the PL CSV schema."""
    out = pd.DataFrame()

    # Parse date — format varies by season (DD/MM/YY or DD/MM/YYYY)
    out["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    out["Home_Team"] = df["HomeTeam"].astype(str).str.strip()
    out["Away_Team"] = df["AwayTeam"].astype(str).str.strip()
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


def _add_promoted_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Home_Promoted and Away_Promoted columns."""
    df["Home_Promoted"] = 0
    df["Away_Promoted"] = 0

    for season_idx, teams in PROMOTED_TEAMS.items():
        mask = df["SeasonIndex"] == season_idx
        for team in teams:
            team_low = team.lower()
            for idx in df[mask].index:
                home_low = df.at[idx, "Home_Team"].lower()
                away_low = df.at[idx, "Away_Team"].lower()
                if team_low in home_low or home_low in team_low:
                    df.at[idx, "Home_Promoted"] = 1
                if team_low in away_low or away_low in team_low:
                    df.at[idx, "Away_Promoted"] = 1
    return df


def _add_derby_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add Local Derby, Historical Derby, and Not a Derby columns."""
    local = []
    historical = []
    for _, row in df.iterrows():
        h, a = row["Home_Team"], row["Away_Team"]
        is_local = _is_derby(h, a, DERBIES_LOCAL)
        is_hist = _is_derby(h, a, DERBIES_HISTORICAL)
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


def build() -> pd.DataFrame:
    """Download, merge, and feature-engineer the full Championship dataset."""
    print("=== Building Championship Dataset ===\n")

    # Step 1: Download all seasons
    season_dfs: list[pd.DataFrame] = []
    for si in range(FIRST_SEASON, LAST_SEASON + 1):
        raw = download_season(si)
        if raw is not None and len(raw) > 0:
            mapped = _map_columns(raw, si)
            season_dfs.append(mapped)

    if not season_dfs:
        raise RuntimeError("No Championship data downloaded.")

    df = pd.concat(season_dfs, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"\nTotal matches: {len(df)}")

    # Step 2: Add rolling features
    print("\nComputing rolling features...")
    df = _add_rolling_features(df)

    # Step 3: League position
    print("Computing league positions...")
    df = _add_league_position(df)

    # Step 4: H2H
    print("Computing H2H stats...")
    df = _add_h2h(df)

    # Step 5: Promoted flags
    print("Adding promoted flags...")
    df = _add_promoted_flags(df)

    # Step 6: Derby flags
    print("Adding derby flags...")
    df = _add_derby_flags(df)

    # Step 7: Home/Away Factor
    print("Computing home/away factor...")
    df = _add_factor(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Seasons: {df['SeasonIndex'].min()} - {df['SeasonIndex'].max()}")
    print(f"Unique teams: {df['Home_Team'].nunique()}")
    print(f"O/U odds available: {df['B365Greater2.5'].notna().sum()} / {len(df)} matches")

    return df


if __name__ == "__main__":
    build()
