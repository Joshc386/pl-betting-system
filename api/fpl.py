"""
FPL (Fantasy Premier League) API client.
No API key required.
Provides: team strength ratings, player injuries, team form.
"""
import requests
import pandas as pd
from api.team_mapping import normalize

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def _fetch(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_bootstrap():
    """Fetch the main FPL bootstrap data (teams, players, events)."""
    data = _fetch(FPL_BOOTSTRAP_URL)
    return {
        "players": data["elements"],
        "teams": data["teams"],
        "events": data["events"],
        "element_types": data["element_types"],
    }


def get_team_strengths():
    """
    Returns a DataFrame of FPL team strength ratings.
    Columns: team_name, strength_attack_home, strength_attack_away,
             strength_defence_home, strength_defence_away, strength_overall
    """
    data = get_bootstrap()
    teams = data["teams"]
    rows = []
    for t in teams:
        rows.append({
            "fpl_id": t["id"],
            "team_name": normalize(t["name"]),
            "short_name": t["short_name"],
            "strength_overall": t["strength"],
            "strength_attack_home": t["strength_attack_home"],
            "strength_attack_away": t["strength_attack_away"],
            "strength_defence_home": t["strength_defence_home"],
            "strength_defence_away": t["strength_defence_away"],
        })
    return pd.DataFrame(rows)


def get_injury_report():
    """
    Returns a DataFrame of injured/doubtful players with their team and impact.
    Columns: player_name, team_name, status, chance_of_playing,
             total_points, position, news
    """
    data = get_bootstrap()
    players = data["players"]
    teams = {t["id"]: normalize(t["name"]) for t in data["teams"]}
    positions = {et["id"]: et["singular_name_short"] for et in data["element_types"]}

    injured = []
    for p in players:
        if p["status"] in ("i", "d", "s", "u"):  # injured, doubtful, suspended, unavailable
            injured.append({
                "player_name": f"{p['first_name']} {p['second_name']}",
                "team_name": teams.get(p["team"], "Unknown"),
                "position": positions.get(p["element_type"], "?"),
                "status": p["status"],
                "chance_of_playing": p.get("chance_of_playing_next_round"),
                "total_points": p["total_points"],
                "news": p.get("news", ""),
            })
    return pd.DataFrame(injured)


def get_team_injury_burden():
    """
    Aggregate injury impact per team.
    Returns dict: {team_name: injury_burden_score}
    Burden = sum of total_points for unavailable players (higher = more impactful losses).
    """
    injuries = get_injury_report()
    if injuries.empty:
        return {}
    burden = injuries.groupby("team_name")["total_points"].sum()
    return burden.to_dict()


def get_upcoming_fixtures():
    """
    Returns upcoming unfinished PL fixtures.
    Columns: home_team, away_team, gameweek, kickoff_time
    """
    data = get_bootstrap()
    teams = {t["id"]: normalize(t["name"]) for t in data["teams"]}
    fixtures = _fetch(FPL_FIXTURES_URL)

    upcoming = []
    for f in fixtures:
        if not f["finished"] and not f["finished_provisional"]:
            upcoming.append({
                "home_team": teams.get(f["team_h"], "Unknown"),
                "away_team": teams.get(f["team_a"], "Unknown"),
                "gameweek": f["event"],
                "kickoff_time": f["kickoff_time"],
            })
    return pd.DataFrame(upcoming)


if __name__ == "__main__":
    print("=== Team Strengths ===")
    strengths = get_team_strengths()
    print(strengths.to_string(index=False))

    print("\n=== Injury Report (top 10 by impact) ===")
    injuries = get_injury_report()
    if not injuries.empty:
        print(injuries.sort_values("total_points", ascending=False).head(10).to_string(index=False))

    print("\n=== Injury Burden by Team ===")
    burden = get_team_injury_burden()
    for team, score in sorted(burden.items(), key=lambda x: -x[1])[:10]:
        print(f"  {team:35s} {score}")

    print("\n=== Upcoming Fixtures (next 5) ===")
    upcoming = get_upcoming_fixtures()
    if not upcoming.empty:
        print(upcoming.head(5).to_string(index=False))
