"""
football-data.org API client.
Free tier: 10 requests/minute, current season.
Sign up at https://www.football-data.org/ for a free API key.
"""
import requests
import pandas as pd
from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL
from api.team_mapping import normalize

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}


def _get(endpoint):
    url = f"{FOOTBALL_DATA_BASE_URL}{endpoint}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_standings():
    """
    Get current PL standings.
    Returns DataFrame with: position, team_name, points, goal_difference,
    games_played, wins, draws, losses
    """
    data = _get("/competitions/PL/standings")
    table = data["standings"][0]["table"]
    rows = []
    for entry in table:
        rows.append({
            "position": entry["position"],
            "team_name": normalize(entry["team"]["name"]),
            "points": entry["points"],
            "goal_difference": entry["goalDifference"],
            "games_played": entry["playedGames"],
            "wins": entry["won"],
            "draws": entry["draw"],
            "losses": entry["lost"],
            "goals_for": entry["goalsFor"],
            "goals_against": entry["goalsAgainst"],
        })
    return pd.DataFrame(rows)


def get_scheduled_matches():
    """
    Get scheduled (upcoming) PL matches.
    Returns DataFrame with: home_team, away_team, match_date, matchday
    """
    data = _get("/competitions/PL/matches?status=SCHEDULED")
    matches = data.get("matches", [])
    rows = []
    for m in matches:
        rows.append({
            "home_team": normalize(m["homeTeam"]["name"]),
            "away_team": normalize(m["awayTeam"]["name"]),
            "match_date": m["utcDate"],
            "matchday": m["matchday"],
        })
    return pd.DataFrame(rows)


def get_team_recent_form(team_name, n=5):
    """
    Get a team's last N completed matches and compute form stats.
    Returns dict with recent match results and computed metrics.
    """
    data = _get("/competitions/PL/matches?status=FINISHED")
    matches = data.get("matches", [])

    team_matches = []
    for m in matches:
        home = normalize(m["homeTeam"]["name"])
        away = normalize(m["awayTeam"]["name"])
        if home == team_name or away == team_name:
            team_matches.append({
                "date": m["utcDate"],
                "home_team": home,
                "away_team": away,
                "home_goals": m["score"]["fullTime"]["home"],
                "away_goals": m["score"]["fullTime"]["away"],
            })

    # Get last N
    team_matches.sort(key=lambda x: x["date"], reverse=True)
    recent = team_matches[:n]

    if not recent:
        return None

    # Compute form
    goals_scored = 0
    goals_conceded = 0
    points = 0
    total_goals = 0

    for m in recent:
        tg = m["home_goals"] + m["away_goals"]
        total_goals += tg
        if m["home_team"] == team_name:
            goals_scored += m["home_goals"]
            goals_conceded += m["away_goals"]
            if m["home_goals"] > m["away_goals"]:
                points += 3
            elif m["home_goals"] == m["away_goals"]:
                points += 1
        else:
            goals_scored += m["away_goals"]
            goals_conceded += m["home_goals"]
            if m["away_goals"] > m["home_goals"]:
                points += 3
            elif m["away_goals"] == m["home_goals"]:
                points += 1

    return {
        "team": team_name,
        "matches_analysed": len(recent),
        "points_last_n": points,
        "goals_scored": goals_scored,
        "goals_conceded": goals_conceded,
        "avg_total_goals": total_goals / len(recent),
        "over_2_5_rate": sum(1 for m in recent if m["home_goals"] + m["away_goals"] > 2) / len(recent),
    }


if __name__ == "__main__":
    if not FOOTBALL_DATA_API_KEY:
        print("No API key set. Set FOOTBALL_DATA_API_KEY environment variable.")
        print("Sign up free at https://www.football-data.org/")
    else:
        print("=== Current Standings ===")
        standings = get_standings()
        print(standings.to_string(index=False))

        print("\n=== Upcoming Matches ===")
        matches = get_scheduled_matches()
        print(matches.head(10).to_string(index=False))
