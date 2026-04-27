"""
OddsPapi API integration for multi-market odds aggregation.

Provides odds from 388+ bookmakers for:
  - Over/Under goal lines (0.5 to 8.5 in 0.25 steps)
  - Asian Handicap (-6 to +1 in 0.25 steps)
  - BTTS (Both Teams To Score)
  - First Half O/U, team totals, correct score

Used alongside The-Odds-API:
  - OddsPapi: alt lines, Asian Handicap, Pinnacle sharp line
  - The-Odds-API: bulk O/U 2.5 + BTTS polling (higher free quota)

Free tier: 250 requests/month. One call per fixture returns all markets.
"""
import os
import json
import logging
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from config import ODDSPAPI_CACHE_TTL_MINUTES, ODDSPAPI_SHARP_BOOKS, ODDSPAPI_TIER1_BOOKS

logger = logging.getLogger(__name__)

# ── Configuration ──
API_KEY = os.environ.get("ODDSPAPI_KEY", "")
BASE_URL = "https://api.oddspapi.io/v4"
# Tournament IDs — EPL default, Championship via env var
EPL_TOURNAMENT_ID = int(os.environ.get("ODDSPAPI_TOURNAMENT_ID", "17"))
CHAMPIONSHIP_TOURNAMENT_ID = 18  # OddsPapi Championship ID (verify via API)
SOCCER_SPORT_ID = 10

# Cache to avoid burning quota
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "oddspapi_cache.json")
CACHE_TTL_MINUTES = ODDSPAPI_CACHE_TTL_MINUTES

# Bookmaker priority for sharp lines
SHARP_BOOKS = ODDSPAPI_SHARP_BOOKS
TIER1_BOOKS = ODDSPAPI_TIER1_BOOKS

# ── Market ID Mapping ──
# O/U Full Time: handicap field gives the line
OU_MARKET_IDS = {
    106: 0.5, 108: 1.5, 1010: 2.5, 1012: 3.5, 1014: 4.5,
    1016: 5.5, 1018: 6.5, 1020: 7.5, 1022: 8.5,
    # Quarter/half lines
    10158: 0.25, 10160: 0.75, 10162: 1.0, 10164: 1.25,
    10166: 1.75, 10168: 2.0, 10170: 2.25, 10172: 2.75,
    10174: 3.0, 10176: 3.25, 10178: 3.75,
    10180: 4.0, 10182: 4.25, 10184: 4.75,
    10186: 5.0, 10188: 5.25, 10190: 5.75,
    10192: 6.0, 10194: 6.25, 10196: 6.75,
    10198: 7.0, 10200: 7.25, 10202: 7.75,
    10204: 8.0, 10206: 8.25,
}

# Asian Handicap Full Time (negative = home gives, positive = home receives)
AH_MARKET_IDS = {
    1024: -6.0, 1026: -5.75, 1028: -5.5, 1030: -5.25, 1032: -5.0,
    1034: -4.75, 1036: -4.5, 1038: -4.25, 1040: -4.0,
    1042: -3.75, 1044: -3.5, 1046: -3.25, 1048: -3.0,
    1050: -2.75, 1052: -2.5, 1054: -2.25, 1056: -2.0,
    1058: -1.75, 1060: -1.5, 1062: -1.25, 1064: -1.0,
    1066: -0.75, 1068: -0.5, 1070: -0.25, 1072: 0.0,
    1074: 0.25, 1076: 0.5, 1078: 0.75, 1080: 1.0,
}

BTTS_MARKET_ID = 104
MATCH_RESULT_MARKET_ID = 101


def _load_cache(cache_file: str = CACHE_FILE) -> dict | None:
    """Load cached odds if still fresh.

    Args:
        cache_file: Path to cache JSON file.

    Returns:
        Cached data list, or None if cache is missing/expired.
    """
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
        cached_at = datetime.fromisoformat(cache.get("timestamp", "2000-01-01"))
        if datetime.now() - cached_at < timedelta(minutes=CACHE_TTL_MINUTES):
            return cache.get("data", None)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _save_cache(data: dict, cache_file: str = CACHE_FILE) -> None:
    """Save odds data to cache.

    Args:
        data: Odds data to cache.
        cache_file: Path to cache JSON file.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {"timestamp": datetime.now().isoformat(), "data": data}
    with open(cache_file, "w") as f:
        json.dump(cache, f)


def _api_get(endpoint: str, params: dict | None = None, max_retries: int = 2) -> dict | list | None:
    """Make a GET request to OddsPapi API with retry logic.

    Args:
        endpoint: API endpoint path (e.g. '/fixtures').
        params: Additional query parameters.
        max_retries: Number of retries for transient errors (timeouts, 5xx).

    Returns:
        Parsed JSON response, or None on failure.
    """
    if not API_KEY:
        logger.error("ODDSPAPI_KEY not set in environment")
        return None

    url = f"{BASE_URL}{endpoint}"
    all_params = {"apiKey": API_KEY}
    if params:
        all_params.update(params)

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=all_params, timeout=20)
            if resp.status_code == 429:
                logger.warning("OddsPapi quota exceeded!")
                return None
            if resp.status_code >= 500 and attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("OddsPapi %s returned %d, retrying in %ds...",
                               endpoint, resp.status_code, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.error("OddsPapi %s: %d - %s",
                             endpoint, resp.status_code, resp.text[:200])
                return None
            # Successful call → one credit consumed. OddsPapi has no quota
            # header so we increment a client-side counter (resets monthly).
            from api.quota_tracker import record_call
            record_call("oddspapi")
            return resp.json()
        except requests.Timeout:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("OddsPapi %s timed out, retrying in %ds...",
                               endpoint, wait)
                time.sleep(wait)
                continue
            logger.error("OddsPapi %s timed out after %d attempts",
                         endpoint, max_retries + 1)
            return None
        except requests.RequestException as e:
            logger.error("OddsPapi request failed: %s", e)
            return None
    return None


def get_fixtures(
    tournament_id: int = EPL_TOURNAMENT_ID,
    upcoming_only: bool = True,
) -> list[dict]:
    """Fetch fixtures from OddsPapi for any tournament.

    Args:
        tournament_id: OddsPapi tournament ID (17=EPL, 18=Championship).
        upcoming_only: If True, only return fixtures with odds (upcoming).

    Returns:
        List of fixture dicts with fixtureId, team names, startTime.
    """
    data = _api_get("/fixtures", {"tournamentId": tournament_id})
    if not data or not isinstance(data, list):
        return []

    fixtures = []
    for f in data:
        if upcoming_only and not f.get("hasOdds"):
            continue
        fixtures.append({
            "fixture_id": f["fixtureId"],
            "home_team": f.get("participant1Name", ""),
            "away_team": f.get("participant2Name", ""),
            "start_time": f.get("startTime", ""),
            "has_odds": f.get("hasOdds", False),
        })

    return fixtures


def get_epl_fixtures(upcoming_only: bool = True) -> list[dict]:
    """Fetch EPL fixtures from OddsPapi. Backwards-compatible wrapper.

    Args:
        upcoming_only: If True, only return fixtures with odds (upcoming).

    Returns:
        List of fixture dicts with fixtureId, team names, startTime.
    """
    return get_fixtures(EPL_TOURNAMENT_ID, upcoming_only)


def _parse_ou_odds(
    bookmaker_odds: dict,
    market_ids: dict[int, float],
) -> dict[float, dict]:
    """Parse Over/Under odds from all bookmakers for given market IDs.

    Returns:
        Dict keyed by goal line: {
            line: {
                "best_over": float, "best_under": float,
                "best_over_book": str, "best_under_book": str,
                "sharp_fair_over": float|None, "sharp_fair_under": float|None,
                "n_books": int,
            }
        }
    """
    # Collect prices per line
    line_data: dict[float, dict] = {}

    for bm_key, bm_data in bookmaker_odds.items():
        markets = bm_data.get("markets", {})
        is_sharp = bm_key in SHARP_BOOKS

        for mid_str, market in markets.items():
            mid = int(mid_str)
            if mid not in market_ids:
                continue

            line = market_ids[mid]
            outcomes = market.get("outcomes", {})

            # Even outcome ID = Over, odd = Under (based on API structure)
            over_price = None
            under_price = None
            for oid_str, odata in outcomes.items():
                oid = int(oid_str)
                for pid, pdata in odata.get("players", {}).items():
                    if not pdata.get("active", True):
                        continue
                    price = pdata.get("price")
                    if price is None or price <= 1.0:
                        continue
                    # Over outcome has same ID as market, Under = market + 1
                    if oid == mid:
                        over_price = price
                    elif oid == mid + 1:
                        under_price = price

            if over_price is None and under_price is None:
                continue

            if line not in line_data:
                line_data[line] = {
                    "best_over": 0, "best_under": 0,
                    "best_over_book": "", "best_under_book": "",
                    "pin_over": None, "pin_under": None,
                    "n_books": 0,
                }

            ld = line_data[line]
            if over_price and over_price > ld["best_over"]:
                ld["best_over"] = over_price
                ld["best_over_book"] = bm_key
            if under_price and under_price > ld["best_under"]:
                ld["best_under"] = under_price
                ld["best_under_book"] = bm_key
            if is_sharp:
                if over_price:
                    ld["pin_over"] = over_price
                if under_price:
                    ld["pin_under"] = under_price
            ld["n_books"] += 1

    # Compute sharp fair probabilities
    result = {}
    for line, ld in sorted(line_data.items()):
        if ld["best_over"] <= 1 or ld["best_under"] <= 1:
            continue

        sharp_over = None
        sharp_under = None
        if ld["pin_over"] and ld["pin_under"]:
            pin_total = 1.0 / ld["pin_over"] + 1.0 / ld["pin_under"]
            sharp_over = (1.0 / ld["pin_over"]) / pin_total
            sharp_under = (1.0 / ld["pin_under"]) / pin_total

        result[line] = {
            "best_over": ld["best_over"],
            "best_under": ld["best_under"],
            "best_over_book": ld["best_over_book"],
            "best_under_book": ld["best_under_book"],
            "sharp_fair_over": sharp_over,
            "sharp_fair_under": sharp_under,
            "n_books": ld["n_books"],
        }

    return result


def _parse_ah_odds(bookmaker_odds: dict) -> dict[float, dict]:
    """Parse Asian Handicap odds from all bookmakers.

    Returns:
        Dict keyed by handicap line: {
            line: {
                "best_home": float, "best_away": float,
                "best_home_book": str, "best_away_book": str,
                "sharp_fair_home": float|None, "sharp_fair_away": float|None,
                "n_books": int,
            }
        }
    """
    line_data: dict[float, dict] = {}

    for bm_key, bm_data in bookmaker_odds.items():
        markets = bm_data.get("markets", {})
        is_sharp = bm_key in SHARP_BOOKS

        for mid_str, market in markets.items():
            mid = int(mid_str)
            if mid not in AH_MARKET_IDS:
                continue

            handicap = AH_MARKET_IDS[mid]
            outcomes = market.get("outcomes", {})

            home_price = None
            away_price = None
            for oid_str, odata in outcomes.items():
                oid = int(oid_str)
                for pid, pdata in odata.get("players", {}).items():
                    if not pdata.get("active", True):
                        continue
                    price = pdata.get("price")
                    if price is None or price <= 1.0:
                        continue
                    # First outcome = Home (Team 1), second = Away (Team 2)
                    if oid == mid:
                        home_price = price
                    elif oid == mid + 1:
                        away_price = price

            if home_price is None and away_price is None:
                continue

            if handicap not in line_data:
                line_data[handicap] = {
                    "best_home": 0, "best_away": 0,
                    "best_home_book": "", "best_away_book": "",
                    "pin_home": None, "pin_away": None,
                    "n_books": 0,
                }

            ld = line_data[handicap]
            if home_price and home_price > ld["best_home"]:
                ld["best_home"] = home_price
                ld["best_home_book"] = bm_key
            if away_price and away_price > ld["best_away"]:
                ld["best_away"] = away_price
                ld["best_away_book"] = bm_key
            if is_sharp:
                if home_price:
                    ld["pin_home"] = home_price
                if away_price:
                    ld["pin_away"] = away_price
            ld["n_books"] += 1

    result = {}
    for handicap, ld in sorted(line_data.items()):
        if ld["best_home"] <= 1 or ld["best_away"] <= 1:
            continue

        sharp_home = None
        sharp_away = None
        if ld["pin_home"] and ld["pin_away"]:
            pin_total = 1.0 / ld["pin_home"] + 1.0 / ld["pin_away"]
            sharp_home = (1.0 / ld["pin_home"]) / pin_total
            sharp_away = (1.0 / ld["pin_away"]) / pin_total

        result[handicap] = {
            "best_home": ld["best_home"],
            "best_away": ld["best_away"],
            "best_home_book": ld["best_home_book"],
            "best_away_book": ld["best_away_book"],
            "sharp_fair_home": sharp_home,
            "sharp_fair_away": sharp_away,
            "n_books": ld["n_books"],
        }

    return result


def _parse_btts_odds(bookmaker_odds: dict) -> dict | None:
    """Parse BTTS odds from all bookmakers.

    Returns:
        Dict with best_yes, best_no, sharp fair probs, or None.
    """
    best_yes = 0
    best_no = 0
    best_yes_book = ""
    best_no_book = ""
    pin_yes = None
    pin_no = None
    n_books = 0

    mid = BTTS_MARKET_ID
    mid_str = str(mid)

    for bm_key, bm_data in bookmaker_odds.items():
        markets = bm_data.get("markets", {})
        if mid_str not in markets:
            continue

        market = markets[mid_str]
        outcomes = market.get("outcomes", {})
        is_sharp = bm_key in SHARP_BOOKS

        yes_price = None
        no_price = None
        for oid_str, odata in outcomes.items():
            oid = int(oid_str)
            for pid, pdata in odata.get("players", {}).items():
                if not pdata.get("active", True):
                    continue
                price = pdata.get("price")
                if price is None or price <= 1.0:
                    continue
                if oid == mid:      # 104 = Yes
                    yes_price = price
                elif oid == mid + 1:  # 105 = No
                    no_price = price

        if yes_price is None and no_price is None:
            continue

        n_books += 1
        if yes_price and yes_price > best_yes:
            best_yes = yes_price
            best_yes_book = bm_key
        if no_price and no_price > best_no:
            best_no = no_price
            best_no_book = bm_key
        if is_sharp:
            if yes_price:
                pin_yes = yes_price
            if no_price:
                pin_no = no_price

    if best_yes <= 1 or best_no <= 1:
        return None

    sharp_yes = None
    sharp_no = None
    if pin_yes and pin_no:
        pin_total = 1.0 / pin_yes + 1.0 / pin_no
        sharp_yes = (1.0 / pin_yes) / pin_total
        sharp_no = (1.0 / pin_no) / pin_total

    return {
        "best_yes": best_yes,
        "best_no": best_no,
        "best_yes_book": best_yes_book,
        "best_no_book": best_no_book,
        "sharp_fair_yes": sharp_yes,
        "sharp_fair_no": sharp_no,
        "n_books": n_books,
    }


def fetch_fixture_odds(fixture_id: str) -> dict | None:
    """Fetch all odds for a single fixture. Returns parsed multi-market dict.

    Args:
        fixture_id: OddsPapi fixture ID.

    Returns:
        Dict with ou_lines, ah_lines, btts keys, or None on failure.
    """
    data = _api_get("/odds", {"fixtureId": fixture_id})
    if not data or not isinstance(data, dict):
        return None

    bm_odds = data.get("bookmakerOdds", {})
    if not bm_odds:
        return None

    return {
        "fixture_id": fixture_id,
        "ou_lines": _parse_ou_odds(bm_odds, OU_MARKET_IDS),
        "ah_lines": _parse_ah_odds(bm_odds),
        "btts": _parse_btts_odds(bm_odds),
        "n_bookmakers": len(bm_odds),
    }


def fetch_all_odds(
    tournament_id: int = EPL_TOURNAMENT_ID,
    cache_file: str = CACHE_FILE,
    force_refresh: bool = False,
    max_fixtures: int | None = None,
) -> list[dict]:
    """Fetch odds for all upcoming fixtures in a tournament.

    Generalised fetcher that works for any OddsPapi tournament.
    Caches results to avoid quota burn. Each fixture costs 1 API request.

    Args:
        tournament_id: OddsPapi tournament ID (17=EPL, 18=Championship).
        cache_file: Path to the cache JSON file for this tournament.
        force_refresh: Bypass cache.
        max_fixtures: Limit number of fixtures to fetch (quota management).

    Returns:
        List of dicts, each with fixture info + parsed odds for all markets.
    """
    if not force_refresh:
        cached = _load_cache(cache_file)
        if cached is not None:
            logger.info(
                "Using cached OddsPapi data (%d fixtures, tournament %d)",
                len(cached), tournament_id,
            )
            return cached

    # First get fixtures (1 request)
    fixtures = get_fixtures(tournament_id, upcoming_only=True)
    if not fixtures:
        logger.warning(
            "No upcoming fixtures found (tournament %d)", tournament_id)
        return _load_cache(cache_file) or []

    if max_fixtures:
        fixtures = fixtures[:max_fixtures]

    logger.info(
        "OddsPapi: fetching odds for %d fixtures (tournament %d)...",
        len(fixtures), tournament_id,
    )
    results = []

    for i, fixture in enumerate(fixtures):
        odds = fetch_fixture_odds(fixture["fixture_id"])
        if odds is None:
            logger.warning("No odds for %s vs %s",
                           fixture["home_team"], fixture["away_team"])
            continue

        fixture.update(odds)
        results.append(fixture)

        if i < len(fixtures) - 1:
            time.sleep(0.5)

    n_ou = sum(1 for r in results if r.get("ou_lines"))
    n_ah = sum(1 for r in results if r.get("ah_lines"))
    n_btts = sum(1 for r in results if r.get("btts"))

    logger.info("OddsPapi: %d fixtures loaded (O/U: %d, AH: %d, BTTS: %d)",
                len(results), n_ou, n_ah, n_btts)
    logger.info("API requests used: %d (1 fixtures + %d odds)",
                len(fixtures) + 1, len(fixtures))

    _save_cache(results, cache_file)
    return results


# ── Cache file paths per league ──
CHAMPIONSHIP_CACHE_FILE = os.path.join(CACHE_DIR, "oddspapi_cache_efl.json")


def fetch_epl_all_odds(
    force_refresh: bool = False,
    max_fixtures: int | None = None,
) -> list[dict]:
    """Fetch odds for all upcoming EPL fixtures. Backwards-compatible wrapper.

    Args:
        force_refresh: Bypass cache.
        max_fixtures: Limit number of fixtures to fetch (quota management).

    Returns:
        List of dicts, each with fixture info + parsed odds for all markets.
    """
    return fetch_all_odds(
        EPL_TOURNAMENT_ID, CACHE_FILE, force_refresh, max_fixtures,
    )


def fetch_championship_all_odds(
    force_refresh: bool = False,
    max_fixtures: int | None = None,
) -> list[dict]:
    """Fetch odds for all upcoming Championship fixtures.

    Uses tournament ID 18 and a separate cache file to avoid
    collisions with EPL data.

    Args:
        force_refresh: Bypass cache.
        max_fixtures: Limit number of fixtures to fetch (quota management).

    Returns:
        List of dicts, each with fixture info + parsed odds for all markets.
    """
    return fetch_all_odds(
        CHAMPIONSHIP_TOURNAMENT_ID, CHAMPIONSHIP_CACHE_FILE,
        force_refresh, max_fixtures,
    )


# ── Team Name Mapping ──

_ODDSPAPI_TO_DATASET = {
    "Arsenal FC": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Brentford FC": "Brentford FC",
    "Brighton & Hove Albion": "Brighton & Hove Albion FC",
    "Burnley FC": "Burnley FC",
    "Chelsea FC": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton FC": "Everton FC",
    "Fulham FC": "Fulham FC",
    "Ipswich Town FC": "Ipswich Town FC",
    "Leeds United": "Leeds United FC",
    "Leicester City": "Leicester City FC",
    "Liverpool FC": "Liverpool FC",
    "Manchester City": "Manchester City FC",
    "Manchester United": "Manchester United FC",
    "Newcastle United": "Newcastle United FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Sheffield United": "Sheffield United FC",
    "Southampton FC": "Southampton FC",
    "Sunderland AFC": "Sunderland AFC",
    "Tottenham Hotspur": "Tottenham Hotspur FC",
    "West Ham United": "West Ham United FC",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers FC",
    "Wolverhampton": "Wolverhampton Wanderers FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Watford FC": "Watford FC",
    "Luton Town": "Luton Town FC",
}


def map_team(api_name: str, our_teams: set[str]) -> str | None:
    """Map OddsPapi team name to our dataset team name.

    Args:
        api_name: Team name from OddsPapi.
        our_teams: Set of team names in our dataset.

    Returns:
        Matched team name, or None if no match found.
    """
    # Direct lookup
    if api_name in _ODDSPAPI_TO_DATASET:
        mapped = _ODDSPAPI_TO_DATASET[api_name]
        if mapped in our_teams:
            return mapped

    # Exact match
    if api_name in our_teams:
        return api_name

    # Fuzzy: word overlap
    api_words = set(api_name.lower().replace("fc", "").replace("afc", "").split())
    best_score = 0
    best_match = None
    for team in our_teams:
        team_words = set(team.lower().replace("fc", "").replace("afc", "").split())
        overlap = len(api_words & team_words)
        if overlap > best_score:
            best_score = overlap
            best_match = team
    return best_match if best_score >= 1 else None


if __name__ == "__main__":
    print("=" * 60)
    print("  OddsPapi — EPL Multi-Market Odds")
    print("=" * 60)

    fixtures = get_epl_fixtures(upcoming_only=True)
    print(f"\nUpcoming fixtures with odds: {len(fixtures)}")

    if not fixtures:
        print("No fixtures found.")
        exit()

    # Fetch odds for first 3 fixtures (conserve quota)
    print("\nFetching odds for first 3 fixtures...\n")
    for fixture in fixtures[:3]:
        print(f"{fixture['home_team']} vs {fixture['away_team']}")
        print(f"  Kickoff: {fixture['start_time']}")

        odds = fetch_fixture_odds(fixture["fixture_id"])
        if odds is None:
            print("  No odds available\n")
            continue

        print(f"  Bookmakers: {odds['n_bookmakers']}")

        # O/U lines
        ou = odds.get("ou_lines", {})
        if ou:
            print(f"  O/U lines ({len(ou)}):")
            for line in [1.5, 2.5, 3.5, 4.5]:
                if line in ou:
                    ld = ou[line]
                    print(f"    {line}: Over {ld['best_over']:.2f} "
                          f"({ld['best_over_book']}) / "
                          f"Under {ld['best_under']:.2f} "
                          f"({ld['best_under_book']}) "
                          f"[{ld['n_books']} books]")

        # Asian Handicap
        ah = odds.get("ah_lines", {})
        if ah:
            print(f"  AH lines ({len(ah)}):")
            for hc in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
                if hc in ah:
                    ld = ah[hc]
                    print(f"    {hc:+.1f}: Home {ld['best_home']:.2f} "
                          f"({ld['best_home_book']}) / "
                          f"Away {ld['best_away']:.2f} "
                          f"({ld['best_away_book']}) "
                          f"[{ld['n_books']} books]")

        # BTTS
        btts = odds.get("btts")
        if btts:
            print(f"  BTTS: Yes {btts['best_yes']:.2f} "
                  f"({btts['best_yes_book']}) / "
                  f"No {btts['best_no']:.2f} "
                  f"({btts['best_no_book']}) "
                  f"[{btts['n_books']} books]")

        print()
