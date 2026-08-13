"""
ESPN scores fetcher for match settlement.

Uses ESPN's public scoreboard API (no API key required) to fetch
completed match results for Premier League and Championship.

Replaces football-data.org which required an API key and had
frequent 403 errors.
"""
import logging
from datetime import datetime, timedelta

import requests

from api.team_mapping import is_known_team, normalize

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Never set a custom User-Agent on these requests. ESPN's Akamai edge 403s
# unrecognised "name/version" strings; this module works because `requests`
# sends its own default, which is on the allowed side of that rule.


class UnresolvedTeamError(RuntimeError):
    """A finished fixture named a club that maps to no Canonical Dataset name."""

# ESPN league identifiers
LEAGUE_IDS = {
    "PL": "eng.1",
    "EFL": "eng.2",
    "ELC": "eng.2",  # alias used by settlement.py
}

# ESPN name → DB name (PL). DB uses "FC" suffix format.
# normalize() from team_mapping handles most PL names, but ESPN
# drops "FC" so we map explicitly where normalize() would fail.
_ESPN_TO_PL: dict[str, str] = {
    "AFC Bournemouth": "AFC Bournemouth",
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Brentford": "Brentford FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Burnley": "Burnley FC",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Ipswich Town": "Ipswich Town FC",
    "Leeds United": "Leeds United FC",
    "Leicester City": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Newcastle United": "Newcastle United FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Southampton": "Southampton FC",
    "Sunderland": "Sunderland AFC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "West Ham United": "West Ham United FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
}

# ESPN name → DB name (Championship). DB uses short forms.
_ESPN_TO_CHAMP: dict[str, str] = {
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Burnley": "Burnley",
    "Cardiff City": "Cardiff",
    "Lincoln City": "Lincoln",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Bristol City": "Bristol City",
    "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leicester City": "Leicester",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Norwich City": "Norwich",
    "Oxford United": "Oxford",
    "Portsmouth": "Portsmouth",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "Sheffield United": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Weds",
    "Southampton": "Southampton",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Watford": "Watford",
    "West Bromwich Albion": "West Brom",
    "Wrexham": "Wrexham",
    "Wrexham AFC": "Wrexham",
}


def _resolve_team(espn_name: str, competition: str) -> str | None:
    """Map ESPN team name to the name used in the dashboard database.

    Settlement keys bets on (home_team, away_team). A name in the wrong format
    does not raise — the key simply never matches, and the bet stays open with
    nothing said. So an unresolved name returns None and the caller skips the
    match, which is visible.

    Args:
        espn_name: Team name from ESPN API.
        competition: Competition code ('PL' or 'EFL'/'ELC').

    Returns:
        Database-compatible team name, or None if the name does not resolve.
    """
    if competition in ("EFL", "ELC"):
        # The EFL database holds football-data.co.uk short forms ("Blackburn").
        # normalize() answers in Premier League canonical names ("Blackburn
        # Rovers FC"), which cannot match it, so there is no fallback here —
        # an unmapped Championship club is unresolved, not nearly resolved.
        return _ESPN_TO_CHAMP.get(espn_name)

    mapped = _ESPN_TO_PL.get(espn_name)
    if mapped:
        return mapped

    # The PL database uses canonical long names, which is what normalize()
    # returns — but only where it recognises the name.
    if is_known_team(espn_name):
        return normalize(espn_name)
    return None


def fetch_finished_window(
    competition: str,
    start: "datetime.date",
    end: "datetime.date",
) -> list[tuple]:
    """Finished fixtures in one date window, as ``(date, home, away)`` tuples.

    Built for the ADR 0005 Freshness Gate, and deliberately *not* built on
    ``fetch_completed_matches``, which loops one request per day and
    ``continue``s past a failure. For settlement that is benign — a missed day
    leaves a bet open, which is visible. For a gate it inverts the meaning:
    fixtures never checked look like fixtures that never happened, so the gate
    would pass on incomplete evidence.

    This makes **one** request for the whole window and lets failures raise, so
    the caller can tell "none were played" from "I could not find out".

    Teams come back already in the league's Canonical Dataset format, via the
    same ``_resolve_team`` the settlement path uses.

    Raises:
        requests.RequestException: On any HTTP or network failure.
        UnresolvedTeamError: If a finished fixture names a club that does not
            resolve. Skipping it would silently shrink the evidence the gate
            reconciles against, which is the failure this function exists to
            avoid.
    """
    espn_league = LEAGUE_IDS.get(competition)
    if not espn_league:
        raise ValueError(f"Unknown competition code: {competition}")

    resp = requests.get(
        f"{BASE_URL}/{espn_league}/scoreboard",
        params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 400},
        timeout=20,
    )
    resp.raise_for_status()

    finished: list[tuple] = []
    for event in resp.json().get("events", []):
        match_comp = event.get("competitions", [{}])[0]
        if not match_comp.get("status", {}).get("type", {}).get("completed"):
            continue

        competitors = match_comp.get("competitors", [])
        home_raw = next(
            (c["team"]["displayName"] for c in competitors
             if c.get("homeAway") == "home"), None)
        away_raw = next(
            (c["team"]["displayName"] for c in competitors
             if c.get("homeAway") == "away"), None)
        if not home_raw or not away_raw:
            raise UnresolvedTeamError(
                f"ESPN [{competition}]: finished event {event.get('id')} is "
                f"missing a home or away competitor"
            )

        home = _resolve_team(home_raw, competition)
        away = _resolve_team(away_raw, competition)
        if not home or not away:
            unresolved = home_raw if not home else away_raw
            raise UnresolvedTeamError(
                f"ESPN [{competition}]: cannot resolve {unresolved!r} to a "
                f"Canonical Dataset name. Add it to the map in "
                f"api/espn_scores.py — the freshness gate cannot verify a "
                f"league whose fixture list it cannot read."
            )

        finished.append((
            datetime.strptime(event["date"][:10], "%Y-%m-%d").date(),
            home,
            away,
        ))

    return finished


def fetch_completed_matches(
    days_back: int = 7,
    competitions: list[str] | None = None,
) -> list[dict]:
    """Fetch recently completed matches from ESPN.

    Queries each date in the lookback window for each competition.
    No API key required.

    Args:
        days_back: How many days back to look for finished matches.
        competitions: Competition codes to fetch. Defaults to ['PL', 'ELC'].

    Returns:
        List of dicts with home_team, away_team, home_goals, away_goals,
        total_goals, date, competition.
    """
    if competitions is None:
        competitions = ["PL", "ELC"]

    results: list[dict] = []
    seen: set[tuple] = set()  # deduplicate

    for comp in competitions:
        espn_league = LEAGUE_IDS.get(comp)
        if not espn_league:
            logger.warning("Unknown competition code: %s", comp)
            continue

        for day_offset in range(days_back + 1):
            date = datetime.utcnow() - timedelta(days=day_offset)
            date_str = date.strftime("%Y%m%d")

            try:
                url = f"{BASE_URL}/{espn_league}/scoreboard"
                resp = requests.get(
                    url, params={"dates": date_str}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.warning(
                    "ESPN fetch failed for %s on %s: %s",
                    comp, date_str, e,
                )
                continue

            for event in data.get("events", []):
                match_comp = event.get("competitions", [{}])[0]
                status = match_comp.get("status", {}).get("type", {})

                if not status.get("completed", False):
                    continue

                competitors = match_comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                # ESPN: competitors[0] is always home
                home_data = competitors[0]
                away_data = competitors[1]

                home_name = home_data["team"]["displayName"]
                away_name = away_data["team"]["displayName"]

                home_goals = int(home_data.get("score", 0))
                away_goals = int(away_data.get("score", 0))

                # Resolve to DB names
                home_db = _resolve_team(home_name, comp)
                away_db = _resolve_team(away_name, comp)

                if home_db is None or away_db is None:
                    # Settling this would mean matching bets on a name the
                    # database does not use, which silently matches nothing.
                    # Skipping is the same outcome, said out loud.
                    logger.warning(
                        "ESPN [%s]: cannot resolve %s v %s to database names "
                        "(%s unresolved) — match skipped, any bets on it stay "
                        "open. Add it to the map in api/espn_scores.py.",
                        comp, home_name, away_name,
                        "home" if home_db is None else "away",
                    )
                    continue

                # Deduplicate (same match can appear on adjacent date queries)
                dedup_key = (home_db, away_db, date_str)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                match_date = event.get("date", "")

                results.append({
                    "home_team": home_db,
                    "away_team": away_db,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "total_goals": home_goals + away_goals,
                    "date": match_date,
                    "competition": comp,
                })

    logger.info(
        "ESPN: fetched %d completed matches (last %d days, %s)",
        len(results), days_back, ", ".join(competitions),
    )
    return results
