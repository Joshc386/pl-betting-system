"""
Download historical FPL player data from vaastav/Fantasy-Premier-League GitHub repo.
Provides per-gameweek player data (injuries, form, xG, minutes) from 2016/17 onwards.
"""
import os
import time
import requests
import pandas as pd
from api.team_mapping import normalize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fpl_historical")

# GitHub raw URL pattern for merged gameweek files
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Seasons available with merged_gw.csv
SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]

# Map FPL team IDs to names (varies by season, fetched from teams.csv)
def _get_team_map(season: str) -> dict:
    """Download teams.csv for a season and return id->canonical_name mapping."""
    url = f"{BASE_URL}/{season}/teams.csv"
    try:
        df = pd.read_csv(url)
        # Column names vary: 'id'/'code' and 'name'/'short_name'
        if "id" in df.columns and "name" in df.columns:
            return {row["id"]: normalize(row["name"]) for _, row in df.iterrows()}
    except Exception:
        pass
    return {}


def download_season(season: str, delay: float = 1.0) -> pd.DataFrame:
    """
    Download merged gameweek data for a season.

    Returns DataFrame with player-level per-gameweek data including:
    name, team, GW, minutes, total_points, was_home, opponent_team,
    status (i=injured, d=doubtful, s=suspended, u=unavailable),
    chance_of_playing_next_round, expected_goals, expected_assists, form
    """
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    time.sleep(delay)

    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(url, encoding="latin-1")
        except Exception as e:
            print(f"  Failed to download {season}: {e}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  Failed to download {season}: {e}")
        return pd.DataFrame()

    # Normalize column names (they vary slightly across seasons)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower == "gw":
            col_map[col] = "GW"
        elif lower == "name":
            col_map[col] = "name"
        elif lower in ("team", "team_x"):
            col_map[col] = "team_id"
        elif lower == "minutes":
            col_map[col] = "minutes"
        elif lower == "total_points":
            col_map[col] = "total_points"
        elif lower == "was_home":
            col_map[col] = "was_home"
        elif lower == "opponent_team":
            col_map[col] = "opponent_team"
        elif lower in ("expected_goals", "xg"):
            col_map[col] = "expected_goals"
        elif lower in ("expected_assists", "xa"):
            col_map[col] = "expected_assists"
        elif lower == "form" and "form" not in col_map.values():
            col_map[col] = "form"
        elif lower == "chance_of_playing_next_round":
            col_map[col] = "chance_of_playing"
        elif lower == "kickoff_time":
            col_map[col] = "kickoff_time"

    df = df.rename(columns=col_map)

    # Get team name mapping
    team_map = _get_team_map(season)
    if "team_id" in df.columns:
        # team_id might be numeric (needs mapping) or already a string name
        if df["team_id"].dtype in ("int64", "float64") and team_map:
            df["team"] = df["team_id"].map(team_map)
        else:
            # team_id is already a string team name
            df["team"] = df["team_id"].apply(lambda x: normalize(str(x)) if pd.notna(x) else x)
    elif "team" in df.columns:
        df["team"] = df["team"].apply(lambda x: normalize(str(x)) if pd.notna(x) else x)

    df["season"] = season
    return df


def download_all_seasons(seasons=None, delay=1.0) -> pd.DataFrame:
    """Download all available seasons of FPL data."""
    if seasons is None:
        seasons = SEASONS

    all_dfs = []
    for season in seasons:
        print(f"  Downloading {season}...")
        df = download_season(season, delay=delay)
        if not df.empty:
            all_dfs.append(df)
            print(f"    Got {len(df)} player-gameweek rows")
        else:
            print(f"    No data")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def compute_team_injury_burden(fpl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player data to team-level injury/availability metrics per gameweek.

    Uses minutes-based approach: players who played in the previous GW but got 0 minutes
    in the current GW are considered unavailable (injured/dropped/suspended).

    Returns DataFrame with: season, GW, team, injury_burden, key_absences, squad_depth
    """
    if fpl_df.empty:
        return pd.DataFrame()

    needed = ["season", "GW", "team", "total_points", "minutes"]
    if not all(c in fpl_df.columns for c in needed):
        print(f"  Warning: missing columns for injury computation.")
        return pd.DataFrame()

    df = fpl_df.copy()
    df = df.sort_values(["season", "team", "name", "GW"])

    # Use player value (price in 0.1m) as importance proxy
    if "value" not in df.columns:
        print("  Warning: no 'value' column available")
        return pd.DataFrame()

    # For each player, check if they played in the previous GW
    df["prev_minutes"] = df.groupby(["season", "team", "name"])["minutes"].shift(1)

    rows = []
    for (season, gw, team), group in df.groupby(["season", "GW", "team"]):
        # Players who played last GW (>0 min) but got 0 minutes this GW
        previously_active = group[group["prev_minutes"] > 0]
        now_absent = previously_active[previously_active["minutes"] == 0]

        # Injury burden = sum of absent players' values (higher value = bigger loss)
        injury_burden = now_absent["value"].sum() if len(now_absent) > 0 else 0

        # Key absences = count of absent players worth more than team median value
        if len(previously_active) > 0:
            median_val = previously_active["value"].median()
            key_absences = (now_absent["value"] > median_val).sum() if len(now_absent) > 0 else 0
        else:
            key_absences = 0

        # Squad depth: how many players got minutes
        players_used = (group["minutes"] > 0).sum()

        rows.append({
            "season": season,
            "GW": gw,
            "team": team,
            "injury_burden": injury_burden,
            "key_absences": key_absences,
            "squad_depth": players_used,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "fpl_all_seasons.csv")

    print("Downloading historical FPL data...")
    df = download_all_seasons(delay=0.5)

    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} rows to {output_path}")
        print(f"Seasons: {df['season'].unique()}")
        print(f"Columns: {list(df.columns[:15])}")

        # Compute injury aggregation
        injury_df = compute_team_injury_burden(df)
        if not injury_df.empty:
            injury_path = os.path.join(DATA_DIR, "team_injury_burden.csv")
            injury_df.to_csv(injury_path, index=False)
            print(f"Injury burden data saved to {injury_path} ({len(injury_df)} rows)")
    else:
        print("No data downloaded.")
