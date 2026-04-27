"""
Fixture-aware scheduling: detects upcoming matches and their kickoff times.

Uses football-data.org API (free tier, 10 req/min) to fetch scheduled fixtures.
This is separate from The-Odds-API quota — football-data.org fixtures are free.

Used by the dynamic scheduler to create matchday-specific jobs (morning fetch,
pre-kickoff odds refresh) instead of polling every 30 minutes.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL
from api.team_mapping import normalize

logger = logging.getLogger(__name__)

HEADERS = {"X-Auth-Token": FOOTBALL_DATA_API_KEY}
UK_TZ = ZoneInfo("Europe/London")

# Competition codes for football-data.org
LEAGUE_COMPETITION_CODES: dict[str, str] = {
    "PL": "PL",
    "EFL": "ELC",  # English League Championship
}


def _get(endpoint: str) -> dict:
    """Make a GET request to football-data.org.

    Args:
        endpoint: API endpoint path (e.g. /competitions/PL/matches).

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: On non-2xx response.
    """
    url = f"{FOOTBALL_DATA_BASE_URL}{endpoint}"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_upcoming_fixtures(
    league: str = "PL",
    days_ahead: int = 7,
) -> list[dict]:
    """Get upcoming fixtures with kickoff times.

    Args:
        league: League key ("PL" or "EFL").
        days_ahead: How many days ahead to look.

    Returns:
        List of fixture dicts with home, away, kickoff (UK time), matchday.
    """
    comp = LEAGUE_COMPETITION_CODES.get(league, "PL")
    today = date.today()
    date_from = today.isoformat()
    date_to = (today + timedelta(days=days_ahead)).isoformat()

    try:
        data = _get(
            f"/competitions/{comp}/matches"
            f"?status=SCHEDULED&dateFrom={date_from}&dateTo={date_to}"
        )
    except requests.RequestException as e:
        logger.error("Failed to fetch %s fixtures: %s", league, e)
        return []

    fixtures = []
    for m in data.get("matches", []):
        utc_str = m.get("utcDate", "")
        if not utc_str:
            continue
        kickoff_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        kickoff_uk = kickoff_utc.astimezone(UK_TZ)

        fixtures.append({
            "home": normalize(m["homeTeam"]["name"]),
            "away": normalize(m["awayTeam"]["name"]),
            "kickoff": kickoff_uk,
            "matchday": m.get("matchday"),
        })

    fixtures.sort(key=lambda f: f["kickoff"])
    return fixtures


def get_matchday_kickoffs(
    league: str = "PL",
    target_date: date | None = None,
) -> list[datetime]:
    """Get distinct kickoff times for a specific date.

    Args:
        league: League key ("PL" or "EFL").
        target_date: Date to check (defaults to today).

    Returns:
        Sorted list of distinct kickoff datetimes (UK time).
        Empty list if no matches on that date.
    """
    target_date = target_date or date.today()
    fixtures = get_upcoming_fixtures(league, days_ahead=1)

    kickoffs = set()
    for fx in fixtures:
        if fx["kickoff"].date() == target_date:
            kickoffs.add(fx["kickoff"])

    return sorted(kickoffs)


def get_next_matchday(
    league: str = "PL",
) -> tuple[date | None, list[datetime]]:
    """Get the next date with matches and its kickoff times.

    Args:
        league: League key ("PL" or "EFL").

    Returns:
        Tuple of (date, kickoff_times). (None, []) if no upcoming matches.
    """
    fixtures = get_upcoming_fixtures(league, days_ahead=14)
    if not fixtures:
        return None, []

    # Group by date, return first date
    first_date = fixtures[0]["kickoff"].date()
    kickoffs = sorted({
        fx["kickoff"] for fx in fixtures
        if fx["kickoff"].date() == first_date
    })
    return first_date, kickoffs


def get_todays_fixtures_summary() -> dict[str, list[datetime]]:
    """Get today's kickoff times for all leagues.

    Returns:
        Dict mapping league key to list of kickoff datetimes.
        Only includes leagues with matches today.
    """
    today = date.today()
    result: dict[str, list[datetime]] = {}

    for league in LEAGUE_COMPETITION_CODES:
        kickoffs = get_matchday_kickoffs(league, today)
        if kickoffs:
            result[league] = kickoffs

    return result


if __name__ == "__main__":
    if not FOOTBALL_DATA_API_KEY:
        print("No API key. Set FOOTBALL_DATA_API_KEY env var.")
    else:
        print("=== Today's Fixtures ===")
        summary = get_todays_fixtures_summary()
        if not summary:
            print("  No matches today.")
        for league, kickoffs in summary.items():
            print(f"\n  {league}:")
            for ko in kickoffs:
                print(f"    {ko:%H:%M}")

        print("\n=== Upcoming PL Fixtures ===")
        for fx in get_upcoming_fixtures("PL"):
            print(f"  {fx['kickoff']:%a %d %b %H:%M}  "
                  f"{fx['home']} vs {fx['away']}")

        print("\n=== Next PL Matchday ===")
        next_date, kickoffs = get_next_matchday("PL")
        if next_date:
            print(f"  {next_date:%A %d %B}")
            for ko in kickoffs:
                print(f"    Kickoff: {ko:%H:%M}")
