"""
Understat xG scraper using understatapi package.
Fetches match-level xG data from understat.com for Premier League seasons.
Available from 2014/15 onwards.
"""
import os
import time
import re
import json
import codecs
import pandas as pd
import requests
from understatapi import UnderstatClient
from api.team_mapping import normalize

# Understat season codes: "2014" means 2014/15, "2023" means 2023/24
AVAILABLE_SEASONS = list(range(2014, 2025))  # 2014/15 through 2024/25

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def scrape_season(client, season_year: int) -> pd.DataFrame:
    """
    Fetch all match results + xG for a PL season from Understat.

    Args:
        client: UnderstatClient instance
        season_year: e.g. 2023 for the 2023/24 season

    Returns:
        DataFrame with match-level xG data
    """
    data = client.league(league="EPL").get_match_data(season=str(season_year))
    if not data:
        print(f"  Warning: No match data found for {season_year}/{season_year+1}")
        return pd.DataFrame()

    rows = []
    for m in data:
        if not m.get("isResult", False):
            continue
        rows.append({
            "date": m["datetime"],
            "home_team": normalize(m["h"]["title"]),
            "away_team": normalize(m["a"]["title"]),
            "home_goals": int(m["goals"]["h"]),
            "away_goals": int(m["goals"]["a"]),
            "home_xg": float(m["xG"]["h"]),
            "away_xg": float(m["xG"]["a"]),
            "understat_match_id": m["id"],
            "season": f"{season_year}/{str(season_year+1)[-2:]}",
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["total_xg"] = df["home_xg"] + df["away_xg"]
    df["xg_diff"] = df["total_xg"] - df["total_goals"]
    return df


def scrape_all_seasons(seasons=None, delay=1.5) -> pd.DataFrame:
    """
    Scrape multiple seasons of xG data.

    Args:
        seasons: list of season years (default: all available 2014-2024)
        delay: seconds between requests
    """
    if seasons is None:
        seasons = AVAILABLE_SEASONS

    all_dfs = []
    with UnderstatClient() as client:
        for season in seasons:
            print(f"  Scraping {season}/{season+1}...")
            df = scrape_season(client, season)
            if not df.empty:
                all_dfs.append(df)
                print(f"    Got {len(df)} matches")
            else:
                print(f"    No data found")
            time.sleep(delay)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def add_rolling_xg_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling xG features per team from Understat match data.

    Returns DataFrame with: team, date, rolling_xg_for, rolling_xg_against,
    rolling_xg_overperformance (goals - xG), rolling_total_xg
    """
    home = df[["date", "home_team", "home_xg", "away_xg", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]

    away = df[["date", "away_team", "away_xg", "home_xg", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]

    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling features (exclude current match with shift)
    long["rolling_xg_for"] = (
        long.groupby("team")["xg_for"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    long["rolling_xg_against"] = (
        long.groupby("team")["xg_against"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    long["xg_overperformance"] = long["goals_for"] - long["xg_for"]
    long["rolling_xg_overperf"] = (
        long.groupby("team")["xg_overperformance"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    long["match_total_xg"] = long["xg_for"] + long["xg_against"]
    long["rolling_total_xg"] = (
        long.groupby("team")["match_total_xg"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    return long[["date", "team", "rolling_xg_for", "rolling_xg_against",
                  "rolling_xg_overperf", "rolling_total_xg"]]


SHOTS_CACHE_PATH = os.path.join(DATA_DIR, "understat_shots.csv")
MATCH_INFO_CACHE_PATH = os.path.join(DATA_DIR, "understat_match_info.csv")


def scrape_match_shots(client, match_id: str) -> list:
    """Fetch all shots for a single match from Understat.

    Returns list of dicts with per-shot fields:
    id, minute, result, X, Y, xG, player, situation, shotType, lastAction, h_a
    """
    data = client.match(match=str(match_id)).get_shot_data()
    shots = []
    for side in ["h", "a"]:
        for s in data.get(side, []):
            shots.append({
                "match_id": str(match_id),
                "side": side,
                "minute": int(s.get("minute", 0)),
                "result": s.get("result", ""),
                "X": float(s.get("X", 0)),
                "Y": float(s.get("Y", 0)),
                "xG": float(s.get("xG", 0)),
                "player": s.get("player", ""),
                "situation": s.get("situation", ""),
                "shotType": s.get("shotType", ""),
                "lastAction": s.get("lastAction", ""),
            })
    return shots


def build_shots_cache(match_ids, delay=1.5):
    """Fetch shot data for all match IDs, caching results incrementally.

    Args:
        match_ids: list of Understat match IDs (ints or strings)
        delay: seconds between API requests
    """
    # Load existing cache
    if os.path.exists(SHOTS_CACHE_PATH):
        existing = pd.read_csv(SHOTS_CACHE_PATH)
        cached_ids = set(existing["match_id"].astype(str))
        print(f"  Existing shots cache: {len(cached_ids)} matches")
    else:
        existing = pd.DataFrame()
        cached_ids = set()

    needed = [str(mid) for mid in match_ids if str(mid) not in cached_ids]
    if not needed:
        print("  Shots cache is up to date.")
        return existing

    print(f"  Need shots for {len(needed)} matches (est. {len(needed) * delay / 60:.0f} min)...")

    all_new_shots = []
    with UnderstatClient() as client:
        for i, mid in enumerate(needed):
            try:
                shots = scrape_match_shots(client, mid)
                all_new_shots.extend(shots)
            except Exception as e:
                print(f"    Error match {mid}: {e}")

            if (i + 1) % 50 == 0:
                print(f"    Fetched {i+1}/{len(needed)} matches...")

            # Save incrementally every 200 matches
            if (i + 1) % 200 == 0 and all_new_shots:
                new_df = pd.DataFrame(all_new_shots)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_csv(SHOTS_CACHE_PATH, index=False)
                existing = combined
                cached_ids = set(existing["match_id"].astype(str))
                all_new_shots = []

            time.sleep(delay)

    # Final save
    if all_new_shots:
        new_df = pd.DataFrame(all_new_shots)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(SHOTS_CACHE_PATH, index=False)
        print(f"  Shots cache updated: {combined['match_id'].nunique()} matches, {len(combined)} shots")
        return combined

    return existing


def load_shots_cache():
    """Load shots cache from disk."""
    if os.path.exists(SHOTS_CACHE_PATH):
        return pd.read_csv(SHOTS_CACHE_PATH)
    return pd.DataFrame()


def scrape_match_info(match_id) -> dict:
    """Fetch tactical match info (PPDA, deep completions, shots on target) from Understat.

    Scrapes the match page HTML directly since understatapi doesn't expose
    the match_info variable. Returns a flat dict of home/away tactical stats.

    Args:
        match_id: Understat match ID (int or string)

    Returns:
        Dict with keys: match_id, h_ppda, a_ppda, h_deep, a_deep,
                        h_shot, a_shot, h_shotOnTarget, a_shotOnTarget
    """
    url = f"https://understat.com/match/{match_id}"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    # Extract match_info JSON from the page script tag
    pattern = r"var\s+match_info\s*=\s*JSON\.parse\('(.+?)'\)"
    m = re.search(pattern, resp.text)
    if not m:
        raise ValueError(f"Could not find match_info in page for match {match_id}")

    raw = m.group(1)
    decoded = codecs.decode(raw, "unicode_escape")
    info = json.loads(decoded)

    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def safe_int(val, default=0):
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    # PPDA may be a simple float string or a dict with 'att' and 'def' keys
    h_ppda_raw = info.get("h_ppda", "0")
    a_ppda_raw = info.get("a_ppda", "0")

    if isinstance(h_ppda_raw, dict):
        h_att = safe_float(h_ppda_raw.get("att", 0))
        h_def = safe_float(h_ppda_raw.get("def", 1))
        h_ppda = h_att / h_def if h_def > 0 else 0.0
    else:
        h_ppda = safe_float(h_ppda_raw)

    if isinstance(a_ppda_raw, dict):
        a_att = safe_float(a_ppda_raw.get("att", 0))
        a_def = safe_float(a_ppda_raw.get("def", 1))
        a_ppda = a_att / a_def if a_def > 0 else 0.0
    else:
        a_ppda = safe_float(a_ppda_raw)

    return {
        "match_id": str(match_id),
        "h_ppda": round(h_ppda, 3),
        "a_ppda": round(a_ppda, 3),
        "h_deep": safe_int(info.get("h_deep")),
        "a_deep": safe_int(info.get("a_deep")),
        "h_shot": safe_int(info.get("h_shot")),
        "a_shot": safe_int(info.get("a_shot")),
        "h_shotOnTarget": safe_int(info.get("h_shotOnTarget")),
        "a_shotOnTarget": safe_int(info.get("a_shotOnTarget")),
    }


def build_match_info_cache(match_ids, delay=6.0):
    """Fetch tactical match info for all match IDs, caching results incrementally.

    Args:
        match_ids: list of Understat match IDs (ints or strings)
        delay: seconds between requests (conservative for HTML scraping)
    """
    # Load existing cache
    if os.path.exists(MATCH_INFO_CACHE_PATH):
        existing = pd.read_csv(MATCH_INFO_CACHE_PATH)
        cached_ids = set(existing["match_id"].astype(str))
        print(f"  Existing match_info cache: {len(cached_ids)} matches")
    else:
        existing = pd.DataFrame()
        cached_ids = set()

    needed = [str(mid) for mid in match_ids if str(mid) not in cached_ids]
    if not needed:
        print("  Match info cache is up to date.")
        return existing

    print(f"  Need match_info for {len(needed)} matches (est. {len(needed) * delay / 60:.0f} min)...")

    all_new_rows = []
    for i, mid in enumerate(needed):
        try:
            row = scrape_match_info(mid)
            all_new_rows.append(row)
        except Exception as e:
            print(f"    Error match {mid}: {e}")

        if (i + 1) % 50 == 0:
            print(f"    Fetched {i+1}/{len(needed)} matches...")

        # Save incrementally every 100 matches
        if (i + 1) % 100 == 0 and all_new_rows:
            new_df = pd.DataFrame(all_new_rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(MATCH_INFO_CACHE_PATH, index=False)
            existing = combined
            cached_ids = set(existing["match_id"].astype(str))
            all_new_rows = []

        time.sleep(delay)

    # Final save
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(MATCH_INFO_CACHE_PATH, index=False)
        print(f"  Match info cache updated: {len(combined)} matches")
        return combined

    return existing


def load_match_info_cache():
    """Load match info cache from disk."""
    if os.path.exists(MATCH_INFO_CACHE_PATH):
        return pd.read_csv(MATCH_INFO_CACHE_PATH)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Match roster data (xGChain, xGBuildup, key_passes, xA per team)
# ═══════════════════════════════════════════════════════════════════════════════

ROSTER_CACHE_PATH = os.path.join(DATA_DIR, "understat_rosters.csv")


def scrape_match_roster(client, match_id: str) -> list:
    """Fetch player roster data for a match, aggregate to team totals.

    Returns list of 2 dicts (home, away) with team-level totals of:
    xGChain, xGBuildup, key_passes, xA, shots, goals.
    """
    data = client.match(match=str(match_id)).get_roster_data()
    rows = []
    for side in ["h", "a"]:
        players = data.get(side, {})
        if not players:
            continue

        total_xgchain = sum(float(p.get("xGChain", 0)) for p in players.values())
        total_xgbuildup = sum(float(p.get("xGBuildup", 0)) for p in players.values())
        total_kp = sum(int(p.get("key_passes", 0)) for p in players.values())
        total_xa = sum(float(p.get("xA", 0)) for p in players.values())
        total_xg = sum(float(p.get("xG", 0)) for p in players.values())
        total_shots = sum(int(p.get("shots", 0)) for p in players.values())
        n_players = len(players)

        rows.append({
            "match_id": str(match_id),
            "side": side,
            "xGChain": round(total_xgchain, 4),
            "xGBuildup": round(total_xgbuildup, 4),
            "key_passes": total_kp,
            "xA": round(total_xa, 4),
            "roster_xG": round(total_xg, 4),
            "roster_shots": total_shots,
            "n_players": n_players,
        })
    return rows


def build_roster_cache(match_ids, delay=1.5):
    """Fetch roster data for all match IDs, caching results incrementally.

    Uses understatapi client (not HTML scraping), so faster rate is safe.
    """
    if os.path.exists(ROSTER_CACHE_PATH):
        existing = pd.read_csv(ROSTER_CACHE_PATH)
        cached_ids = set(existing["match_id"].astype(str))
        print(f"  Existing roster cache: {len(cached_ids)} matches")
    else:
        existing = pd.DataFrame()
        cached_ids = set()

    needed = [str(mid) for mid in match_ids if str(mid) not in cached_ids]
    if not needed:
        print("  Roster cache is up to date.")
        return existing

    print(f"  Need rosters for {len(needed)} matches (est. {len(needed) * delay / 60:.0f} min)...")

    all_new_rows = []
    with UnderstatClient() as client:
        for i, mid in enumerate(needed):
            try:
                rows = scrape_match_roster(client, mid)
                all_new_rows.extend(rows)
            except Exception as e:
                print(f"    Error match {mid}: {e}")

            if (i + 1) % 50 == 0:
                print(f"    Fetched {i+1}/{len(needed)} matches...")

            # Save incrementally every 200 matches
            if (i + 1) % 200 == 0 and all_new_rows:
                new_df = pd.DataFrame(all_new_rows)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_csv(ROSTER_CACHE_PATH, index=False)
                existing = combined
                cached_ids = set(existing["match_id"].astype(str).unique())
                all_new_rows = []

            time.sleep(delay)

    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(ROSTER_CACHE_PATH, index=False)
        print(f"  Roster cache updated: {combined['match_id'].nunique()} matches, {len(combined)} rows")
        return combined

    return existing


def load_roster_cache():
    """Load roster cache from disk."""
    if os.path.exists(ROSTER_CACHE_PATH):
        return pd.read_csv(ROSTER_CACHE_PATH)
    return pd.DataFrame()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "understat_xg.csv")

    print("Scraping Understat xG data for EPL...")
    df = scrape_all_seasons(delay=1.5)

    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} matches to {output_path}")
        print(f"Seasons: {df['season'].unique()}")
        print(f"\nSample:")
        print(df.head(5).to_string(index=False))

        rolling = add_rolling_xg_features(df)
        rolling_path = os.path.join(DATA_DIR, "understat_xg_rolling.csv")
        rolling.to_csv(rolling_path, index=False)
        print(f"\nRolling xG features saved to {rolling_path}")
    else:
        print("No data scraped.")
