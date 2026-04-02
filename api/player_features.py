"""
Player-level feature engineering from FPL-Core-Insights data.
Computes rolling per-player stats and aggregates to team-level squad features.

Data source: https://github.com/olbauday/FPL-Core-Insights
Covers 2024-25 and 2025-26 seasons.
"""
import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_DATA_DIR = os.path.join(PROJECT_DIR, "data", "player_data")

# Map FPL-Core-Insights team codes -> short names (from teams.csv)
# Built dynamically from teams.csv, but we keep a static fallback
_TEAM_CODE_CACHE = {}

SEASONS = ["2024-2025", "2025-2026"]
SEASON_TO_INDEX = {"2024-2025": 24, "2025-2026": 25}

# Rolling window for player stats
ROLLING_WINDOW = 5

# Defensive action columns to sum for a composite metric
DEF_ACTION_COLS = ["tackles_won", "interceptions", "blocks", "clearances", "recoveries"]


def _load_season_data(season):
    """Load all CSVs for a season and join them."""
    base = os.path.join(PLAYER_DATA_DIR, season)

    pms = pd.read_csv(os.path.join(base, "playermatchstats.csv"))
    players = pd.read_csv(os.path.join(base, "players.csv"))
    matches = pd.read_csv(os.path.join(base, "matches.csv"))
    teams = pd.read_csv(os.path.join(base, "teams.csv"))

    # Build team code -> name mapping
    code_to_name = dict(zip(teams["code"], teams["name"]))
    _TEAM_CODE_CACHE[season] = code_to_name

    # Add player info
    player_cols = ["player_id", "first_name", "second_name", "web_name", "team_code", "position"]
    player_cols = [c for c in player_cols if c in players.columns]
    pms = pms.merge(players[player_cols], on="player_id", how="left")

    # Add match info (gameweek, teams, date)
    match_cols = ["match_id", "gameweek", "kickoff_time", "home_team", "away_team",
                  "home_score", "away_score", "finished"]
    match_cols = [c for c in match_cols if c in matches.columns]
    pms = pms.merge(matches[match_cols], on="match_id", how="left")

    # Only finished matches
    if "finished" in pms.columns:
        pms = pms[pms["finished"] == True].copy()

    # Determine which team each player was on for this match
    # Player's team_code tells us their team; compare with home/away team codes
    pms["team_name"] = pms["team_code"].map(code_to_name)
    pms["home_team_name"] = pms["home_team"].map(code_to_name)
    pms["away_team_name"] = pms["away_team"].map(code_to_name)
    pms["is_home"] = pms["team_code"] == pms["home_team"]

    # Parse kickoff time
    if "kickoff_time" in pms.columns:
        pms["kickoff_time"] = pd.to_datetime(pms["kickoff_time"], errors="coerce")

    pms["season"] = season

    # Ensure numeric columns
    for col in ["xg", "xa", "minutes_played"] + DEF_ACTION_COLS:
        if col in pms.columns:
            pms[col] = pd.to_numeric(pms[col], errors="coerce").fillna(0)

    # Compute per-90 stats
    mins = pms["minutes_played"].clip(lower=1)
    pms["xg_p90"] = pms["xg"] / mins * 90
    pms["xa_p90"] = pms["xa"] / mins * 90
    pms["def_actions"] = sum(pms[c] for c in DEF_ACTION_COLS if c in pms.columns)
    pms["def_actions_p90"] = pms["def_actions"] / mins * 90

    return pms


def compute_player_rolling_stats():
    """
    Compute rolling 5-match per-player stats across all seasons.
    Returns DataFrame with one row per (season, gameweek, player_id, match_id).
    Rolling stats use only prior matches (shifted by 1) to avoid leakage.
    """
    all_data = []
    for season in SEASONS:
        path = os.path.join(PLAYER_DATA_DIR, season, "playermatchstats.csv")
        if not os.path.exists(path):
            continue
        pms = _load_season_data(season)
        all_data.append(pms)

    if not all_data:
        print("No player data found.")
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)

    # Sort by player, then chronologically
    df = df.sort_values(["player_id", "season", "gameweek", "kickoff_time"]).reset_index(drop=True)

    # Compute rolling averages per player (shift 1 to avoid leakage)
    rolling_cols = {
        "rolling_xg_5": "xg_p90",
        "rolling_xa_5": "xa_p90",
        "rolling_def_5": "def_actions_p90",
        "rolling_mins_5": "minutes_played",
    }

    for new_col, src_col in rolling_cols.items():
        df[new_col] = (
            df.groupby("player_id")[src_col]
            .transform(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    # Cumulative season stats (for identifying best XI)
    df["cum_xg"] = df.groupby(["player_id", "season"])["xg"].cumsum().shift(1)
    df["cum_minutes"] = df.groupby(["player_id", "season"])["minutes_played"].cumsum().shift(1)
    df["cum_matches"] = df.groupby(["player_id", "season"]).cumcount()

    return df


def compute_squad_features_historical(player_df=None):
    """
    Compute squad-level features for each (season, gameweek, match, team).
    Uses rolling player stats to determine best XI vs available XI.

    Returns DataFrame ready to merge with main match dataset.
    """
    if player_df is None:
        player_df = compute_player_rolling_stats()

    if player_df.empty:
        return pd.DataFrame()

    results = []

    # Group by match
    for (season, match_id, gw), match_df in player_df.groupby(["season", "match_id", "gameweek"]):
        home_team_name = match_df["home_team_name"].iloc[0]
        away_team_name = match_df["away_team_name"].iloc[0]
        kickoff = match_df["kickoff_time"].iloc[0] if "kickoff_time" in match_df.columns else None

        for side, team_name, is_home_flag in [("home", home_team_name, True), ("away", away_team_name, False)]:
            team_players = match_df[match_df["is_home"] == is_home_flag].copy()

            if len(team_players) < 5:
                continue

            # Who actually played (started and got significant minutes)
            available = team_players[team_players["minutes_played"] >= 45]
            starters = team_players[team_players["start_min"] == 0] if "start_min" in team_players.columns else available

            # Best XI: top 11 by cumulative minutes (first-choice players)
            best_xi = team_players.nlargest(11, "cum_minutes") if "cum_minutes" in team_players.columns else team_players.nlargest(11, "minutes_played")

            # Ensure we have rolling stats
            has_rolling = team_players["rolling_xg_5"].notna().sum() > 3

            if not has_rolling:
                # Not enough history for rolling stats yet (early GWs)
                results.append({
                    "season": season, "match_id": match_id, "gameweek": gw,
                    "team": team_name, "side": side, "kickoff_time": kickoff,
                    "AvailableXG": np.nan, "AvailableXA": np.nan,
                    "AttackMissing": np.nan, "DefenceMissing": np.nan,
                    "StarAvailable": np.nan, "FormXG5": np.nan,
                    "SquadDepth": np.nan,
                })
                continue

            # === Feature computation ===

            # Best XI xG sum (from rolling stats)
            best_xg = best_xi["rolling_xg_5"].fillna(0).sum()
            avail_xg = available["rolling_xg_5"].fillna(0).sum()

            # AvailableXG: ratio of available xG to best XI xG
            available_xg_ratio = avail_xg / best_xg if best_xg > 0 else 1.0

            # AvailableXA
            best_xa = best_xi["rolling_xa_5"].fillna(0).sum()
            avail_xa = available["rolling_xa_5"].fillna(0).sum()
            available_xa_ratio = avail_xa / best_xa if best_xa > 0 else 1.0

            # AttackMissing: focus on FWD + MID
            attack_positions = ["Forward", "Midfielder"]
            best_attackers = best_xi[best_xi["position"].isin(attack_positions)] if "position" in best_xi.columns else best_xi
            avail_attackers = available[available["position"].isin(attack_positions)] if "position" in available.columns else available
            best_att_xg = best_attackers["rolling_xg_5"].fillna(0).sum()
            avail_att_xg = avail_attackers["rolling_xg_5"].fillna(0).sum()
            attack_missing = 1.0 - (avail_att_xg / best_att_xg) if best_att_xg > 0 else 0.0

            # DefenceMissing: focus on DEF + GK
            def_positions = ["Defender", "Goalkeeper"]
            best_defs = best_xi[best_xi["position"].isin(def_positions)] if "position" in best_xi.columns else best_xi
            avail_defs = available[available["position"].isin(def_positions)] if "position" in available.columns else available
            best_def_score = best_defs["rolling_def_5"].fillna(0).sum()
            avail_def_score = avail_defs["rolling_def_5"].fillna(0).sum()
            defence_missing = 1.0 - (avail_def_score / best_def_score) if best_def_score > 0 else 0.0

            # StarAvailable: is the top xG contributor on the pitch?
            star = best_xi.nlargest(1, "rolling_xg_5")
            if len(star) > 0 and best_xg > 0:
                star_id = star.iloc[0]["player_id"]
                star_share = star.iloc[0]["rolling_xg_5"] / best_xg
                star_played = star_id in available["player_id"].values
                star_available = star_share if star_played else 0.0
            else:
                star_available = 1.0

            # FormXG5: average per-90 xG of available players over last 5
            form_xg = available["rolling_xg_5"].dropna().mean() if len(available) > 0 else np.nan

            # SquadDepth: unique players used recently (from cum_matches)
            recent_players = team_players[team_players["minutes_played"] > 0]
            squad_depth = len(recent_players) / 20.0

            results.append({
                "season": season, "match_id": match_id, "gameweek": gw,
                "team": team_name, "side": side, "kickoff_time": kickoff,
                "AvailableXG": np.clip(available_xg_ratio, 0, 2),
                "AvailableXA": np.clip(available_xa_ratio, 0, 2),
                "AttackMissing": np.clip(attack_missing, 0, 1),
                "DefenceMissing": np.clip(defence_missing, 0, 1),
                "StarAvailable": np.clip(star_available, 0, 1),
                "FormXG5": form_xg,
                "SquadDepth": squad_depth,
            })

    return pd.DataFrame(results)


def build_squad_features_csv():
    """Run the full pipeline and save to CSV."""
    print("Computing player rolling stats...")
    player_df = compute_player_rolling_stats()
    print(f"  {len(player_df)} player-match rows")

    print("Computing squad features...")
    squad_df = compute_squad_features_historical(player_df)
    print(f"  {len(squad_df)} team-match rows")

    out_path = os.path.join(PLAYER_DATA_DIR, "squad_features.csv")
    squad_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Summary
    for season in squad_df["season"].unique():
        s = squad_df[squad_df["season"] == season]
        print(f"\n{season}:")
        print(f"  Matches: {s['match_id'].nunique()}")
        for col in ["AvailableXG", "AttackMissing", "DefenceMissing", "StarAvailable", "FormXG5"]:
            vals = s[col].dropna()
            if len(vals) > 0:
                print(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}")

    return squad_df


def download_latest_player_data():
    """Download latest data from FPL-Core-Insights GitHub repo."""
    import urllib.request

    base_url = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"

    for season in SEASONS:
        season_dir = os.path.join(PLAYER_DATA_DIR, season)
        os.makedirs(season_dir, exist_ok=True)

        if season == "2024-2025":
            # Consolidated files at top level
            files = {
                "playermatchstats.csv": f"{base_url}/{season}/playermatchstats/playermatchstats.csv",
                "players.csv": f"{base_url}/{season}/players/players.csv",
                "teams.csv": f"{base_url}/{season}/teams/teams.csv",
                "matches.csv": f"{base_url}/{season}/matches/matches.csv",
            }
            for fname, url in files.items():
                dest = os.path.join(season_dir, fname)
                print(f"  Downloading {season}/{fname}...")
                urllib.request.urlretrieve(url, dest)

        elif season == "2025-2026":
            # Players and teams at top level
            for fname in ["players.csv", "teams.csv"]:
                url = f"{base_url}/{season}/{fname}"
                dest = os.path.join(season_dir, fname)
                print(f"  Downloading {season}/{fname}...")
                urllib.request.urlretrieve(url, dest)

            # playermatchstats and matches: concatenate from per-GW files
            for csv_name in ["playermatchstats.csv", "matches.csv"]:
                print(f"  Downloading {season}/{csv_name} (per-GW)...")
                all_rows = []
                header = None
                for gw in range(1, 39):
                    url = f"{base_url}/{season}/By%20Tournament/Premier%20League/GW{gw}/{csv_name}"
                    try:
                        response = urllib.request.urlopen(url)
                        text = response.read().decode("utf-8")
                        lines = text.strip().split("\n")
                        if not lines or "player_id" not in lines[0] and "gameweek" not in lines[0]:
                            continue
                        if header is None:
                            header = lines[0]
                            all_rows.append(header)
                        all_rows.extend(lines[1:])
                    except Exception:
                        break  # No more gameweeks

                if all_rows:
                    dest = os.path.join(season_dir, csv_name)
                    with open(dest, "w", encoding="utf-8") as f:
                        f.write("\n".join(all_rows))
                    print(f"    {len(all_rows) - 1} rows")

    print("Player data download complete.")


# ── Live prediction functions ──

def get_current_player_stats(team_name):
    """
    Get current player stats for a team from the latest season data.
    Returns DataFrame with per-player rolling stats.
    """
    from api.team_mapping import normalize

    season = "2025-2026"
    base = os.path.join(PLAYER_DATA_DIR, season)

    if not os.path.exists(os.path.join(base, "playermatchstats.csv")):
        return pd.DataFrame()

    pms = _load_season_data(season)

    # Normalize team name
    canonical = normalize(team_name)

    # Find the FPL-Core-Insights team name matching canonical
    teams_csv = pd.read_csv(os.path.join(base, "teams.csv"))
    fpl_name = None
    for _, row in teams_csv.iterrows():
        if normalize(row["name"]) == canonical:
            fpl_name = row["name"]
            break

    if fpl_name is None:
        return pd.DataFrame()

    team_pms = pms[pms["team_name"] == fpl_name].copy()
    if team_pms.empty:
        return pd.DataFrame()

    # Sort and compute rolling stats
    team_pms = team_pms.sort_values(["player_id", "gameweek"]).reset_index(drop=True)

    for new_col, src_col in [("rolling_xg_5", "xg_p90"), ("rolling_xa_5", "xa_p90"),
                              ("rolling_def_5", "def_actions_p90"), ("rolling_mins_5", "minutes_played")]:
        team_pms[new_col] = (
            team_pms.groupby("player_id")[src_col]
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=2).mean())
        )

    # Get latest stats per player
    latest = team_pms.sort_values("gameweek").groupby("player_id").tail(1)
    latest["cum_minutes"] = team_pms.groupby("player_id")["minutes_played"].transform("sum")

    return latest


def compute_live_squad_features(team_name, lineup=None):
    """
    Compute squad features for a team at prediction time.

    Args:
        team_name: Team name (any format, will be normalized)
        lineup: Optional list of player names (web_name) for confirmed lineup.
                If None, uses FPL injury data to estimate availability.

    Returns:
        dict with feature values: AvailableXG, AvailableXA, AttackMissing,
        DefenceMissing, StarAvailable, FormXG5, SquadDepth
    """
    default = {
        "AvailableXG": np.nan, "AvailableXA": np.nan,
        "AttackMissing": np.nan, "DefenceMissing": np.nan,
        "StarAvailable": np.nan, "FormXG5": np.nan,
        "SquadDepth": np.nan,
    }

    player_stats = get_current_player_stats(team_name)
    if player_stats.empty:
        return default

    # Best XI: top 11 by cumulative minutes this season
    best_xi = player_stats.nlargest(11, "cum_minutes")

    if lineup is not None:
        # Lineup confirmed: match by web_name
        lineup_lower = {n.lower().strip() for n in lineup}
        available = player_stats[player_stats["web_name"].str.lower().str.strip().isin(lineup_lower)]
    else:
        # Pre-match: use FPL injury data to estimate availability
        try:
            from api.fpl import get_injury_report
            injuries = get_injury_report()
            from api.team_mapping import normalize
            canonical = normalize(team_name)

            # Get injured/doubtful/suspended player names
            team_injuries = injuries[injuries["team"].apply(lambda t: normalize(t) == canonical)]
            unavailable_names = set()
            for _, row in team_injuries.iterrows():
                status = row.get("status", "")
                chance = row.get("chance_of_playing", 100)
                if status in ("i", "s", "u") or (chance is not None and chance < 50):
                    unavailable_names.add(row.get("web_name", "").lower().strip())

            # Available = all players NOT in the unavailable set
            available = player_stats[~player_stats["web_name"].str.lower().str.strip().isin(unavailable_names)]
            # Limit to likely starters (top by cumulative minutes among available)
            available = available.nlargest(11, "cum_minutes")
        except Exception:
            # Fallback: assume best XI is available
            available = best_xi

    if len(available) < 3 or len(best_xi) < 3:
        return default

    has_rolling = available["rolling_xg_5"].notna().sum() > 2

    if not has_rolling:
        return default

    # Compute features (same logic as historical)
    best_xg = best_xi["rolling_xg_5"].fillna(0).sum()
    avail_xg = available["rolling_xg_5"].fillna(0).sum()
    available_xg_ratio = avail_xg / best_xg if best_xg > 0 else 1.0

    best_xa = best_xi["rolling_xa_5"].fillna(0).sum()
    avail_xa = available["rolling_xa_5"].fillna(0).sum()
    available_xa_ratio = avail_xa / best_xa if best_xa > 0 else 1.0

    attack_positions = ["Forward", "Midfielder"]
    best_att = best_xi[best_xi["position"].isin(attack_positions)] if "position" in best_xi.columns else best_xi
    avail_att = available[available["position"].isin(attack_positions)] if "position" in available.columns else available
    best_att_xg = best_att["rolling_xg_5"].fillna(0).sum()
    avail_att_xg = avail_att["rolling_xg_5"].fillna(0).sum()
    attack_missing = 1.0 - (avail_att_xg / best_att_xg) if best_att_xg > 0 else 0.0

    def_positions = ["Defender", "Goalkeeper"]
    best_def = best_xi[best_xi["position"].isin(def_positions)] if "position" in best_xi.columns else best_xi
    avail_def = available[available["position"].isin(def_positions)] if "position" in available.columns else available
    best_def_score = best_def["rolling_def_5"].fillna(0).sum()
    avail_def_score = avail_def["rolling_def_5"].fillna(0).sum()
    defence_missing = 1.0 - (avail_def_score / best_def_score) if best_def_score > 0 else 0.0

    star = best_xi.nlargest(1, "rolling_xg_5")
    if len(star) > 0 and best_xg > 0:
        star_id = star.iloc[0]["player_id"]
        star_share = star.iloc[0]["rolling_xg_5"] / best_xg
        star_played = star_id in available["player_id"].values
        star_available = star_share if star_played else 0.0
    else:
        star_available = 1.0

    form_xg = available["rolling_xg_5"].dropna().mean() if len(available) > 0 else np.nan
    squad_depth = player_stats[player_stats["minutes_played"] > 0]["player_id"].nunique() / 20.0

    return {
        "AvailableXG": np.clip(available_xg_ratio, 0, 2),
        "AvailableXA": np.clip(available_xa_ratio, 0, 2),
        "AttackMissing": np.clip(attack_missing, 0, 1),
        "DefenceMissing": np.clip(defence_missing, 0, 1),
        "StarAvailable": np.clip(star_available, 0, 1),
        "FormXG5": form_xg,
        "SquadDepth": squad_depth,
    }


def get_player_summary(team_name):
    """
    Get a summary of all players for a team for dashboard display.
    Returns list of dicts with player info, stats, and availability.
    """
    player_stats = get_current_player_stats(team_name)
    if player_stats.empty:
        return []

    # Try to get injury info
    injury_map = {}
    try:
        from api.fpl import get_injury_report
        from api.team_mapping import normalize
        injuries = get_injury_report()
        canonical = normalize(team_name)
        team_injuries = injuries[injuries["team"].apply(lambda t: normalize(t) == canonical)]
        for _, row in team_injuries.iterrows():
            name = row.get("web_name", "").strip()
            injury_map[name.lower()] = {
                "status": row.get("status", "a"),
                "chance": row.get("chance_of_playing", None),
                "news": row.get("news", ""),
            }
    except Exception:
        pass

    summary = []
    for _, p in player_stats.sort_values("cum_minutes", ascending=False).iterrows():
        web_name = str(p.get("web_name", "Unknown"))
        inj = injury_map.get(web_name.lower(), {"status": "a", "chance": None, "news": ""})

        summary.append({
            "name": web_name,
            "position": p.get("position", "Unknown"),
            "minutes": int(p.get("cum_minutes", 0)),
            "xg_p90": round(float(p.get("rolling_xg_5", 0) or 0), 3),
            "xa_p90": round(float(p.get("rolling_xa_5", 0) or 0), 3),
            "def_p90": round(float(p.get("rolling_def_5", 0) or 0), 1),
            "status": inj["status"],
            "chance": inj["chance"],
            "news": inj["news"],
        })

    return summary


if __name__ == "__main__":
    squad_df = build_squad_features_csv()
