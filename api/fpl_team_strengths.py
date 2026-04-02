"""
Download and cache historical FPL team strength ratings from vaastav GitHub repo.
Available from 2019-20 onwards (matching our backtest range S19-S25).

These are PRE-SEASON ratings set by FPL before GW1 — perfect cold-start features.
"""
import os
import pandas as pd
from api.team_mapping import normalize

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fpl_historical")
CACHE_PATH = os.path.join(DATA_DIR, "team_strengths.csv")

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Season string -> SeasonIndex mapping
SEASON_MAP = {
    "2019-20": 19, "2020-21": 20, "2021-22": 21,
    "2022-23": 22, "2023-24": 23, "2024-25": 24,
}


def download_team_strengths(seasons=None, use_cache=True):
    """Download team strength ratings for all available seasons.

    Returns DataFrame with columns:
        season_index, team, strength_overall,
        strength_attack_home, strength_attack_away,
        strength_defence_home, strength_defence_away
    """
    if use_cache and os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH)
        if len(df) > 0:
            return df

    if seasons is None:
        seasons = list(SEASON_MAP.keys())

    all_rows = []
    for season_str in seasons:
        season_idx = SEASON_MAP.get(season_str)
        if season_idx is None:
            continue

        url = f"{BASE_URL}/{season_str}/teams.csv"
        try:
            df = pd.read_csv(url)
            if "strength_attack_home" not in df.columns:
                print(f"  {season_str}: no strength columns, skipping")
                continue

            for _, row in df.iterrows():
                all_rows.append({
                    "season_index": season_idx,
                    "team": normalize(row["name"]),
                    "strength_overall": row.get("strength", 3),
                    "strength_attack_home": row["strength_attack_home"],
                    "strength_attack_away": row["strength_attack_away"],
                    "strength_defence_home": row["strength_defence_home"],
                    "strength_defence_away": row["strength_defence_away"],
                })
            print(f"  {season_str} (S{season_idx}): {len(df)} teams")
        except Exception as e:
            print(f"  {season_str}: failed - {e}")

    result = pd.DataFrame(all_rows)

    if len(result) > 0:
        os.makedirs(DATA_DIR, exist_ok=True)
        result.to_csv(CACHE_PATH, index=False)
        print(f"  Cached {len(result)} rows to {CACHE_PATH}")

    return result


def get_match_strength_features(strengths_df, home_team, away_team, season_index):
    """Compute match-level FPL strength features for a single match.

    Returns dict with features or empty dict if data unavailable.
    """
    season_data = strengths_df[strengths_df["season_index"] == season_index]
    if season_data.empty:
        return {}

    home = season_data[season_data["team"] == home_team]
    away = season_data[season_data["team"] == away_team]

    if home.empty or away.empty:
        return {}

    h = home.iloc[0]
    a = away.iloc[0]

    # Raw strength ratings (typically 1000-1400 range)
    h_att = h["strength_attack_home"]
    h_def = h["strength_defence_home"]
    a_att = a["strength_attack_away"]
    a_def = a["strength_defence_away"]

    # Normalize to 0-1 scale (min ~1000, max ~1400)
    def norm(val):
        return (val - 1000) / 400

    return {
        "Home_FPL_Attack": norm(h_att),
        "Away_FPL_Attack": norm(a_att),
        "Home_FPL_Defence": norm(h_def),
        "Away_FPL_Defence": norm(a_def),
        # Match-level derived features
        "FPL_Openness": norm(h_att) + norm(a_att) - norm(h_def) - norm(a_def),
        "FPL_HomeDominance": (norm(h_att) + norm(h_def)) - (norm(a_att) + norm(a_def)),
    }


if __name__ == "__main__":
    print("Downloading FPL team strength ratings...")
    df = download_team_strengths(use_cache=False)
    if not df.empty:
        print(f"\nTotal: {len(df)} team-season entries")
        print(f"Seasons: {sorted(df['season_index'].unique())}")
        print(f"\nSample (S24):")
        s24 = df[df["season_index"] == 24].sort_values("strength_attack_home", ascending=False)
        print(s24[["team", "strength_attack_home", "strength_defence_home"]].head(10).to_string(index=False))
