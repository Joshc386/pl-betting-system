"""
The-Odds-API integration for real-time bookmaker odds.

Fetches Over/Under 2.5 goals and BTTS odds from multiple bookmakers for EPL matches.
Key features:
  - Best price line shopping (find highest odds across all books)
  - Pinnacle consensus probability (sharpest market for blend anchor)
  - Caching to minimize API quota usage (500 free requests/month)
  - Multi-market support: totals (O/U 2.5) + btts
"""
import os
import json
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from api.team_resolver import resolve_feed_team

# ── Configuration ──
API_KEY = os.environ.get("ODDS_API_KEY", "")
BASE_URL = "https://api.the-odds-api.com/v4"
# Default sport — overridden per league via league_config
SPORT = os.environ.get("ODDS_API_SPORT", "soccer_epl")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "odds_cache.json")
CACHE_TTL_MINUTES = 30  # refresh every 30 mins (odds move)
# Hard upper bound on cache age served on the API-failure fallback path.
# Odds move constantly; serving prices older than this risks betting into
# lines that no longer exist. Beyond this age the cache is treated as if it
# did not exist (caller gets [] and places no bets).
STALE_TTL_MINUTES = 90

# Bookmaker tiers (for weighting/display)
SHARP_BOOKS = {"pinnacle"}  # sharpest lines
MAJOR_BOOKS = {"betfair", "bet365", "williamhill", "unibet", "betsson"}


def _load_cache(allow_stale: bool = False, cache_file: str | None = None):
    """Load cached odds if within the applicable age bound.

    Two age bounds apply, never unbounded: normal calls serve only cache
    younger than ``CACHE_TTL_MINUTES`` (30); the API-failure fallback path
    (``allow_stale=True``) serves cache younger than ``STALE_TTL_MINUTES``
    (90). A cache with no timestamp has unknown age and is never served.

    Args:
        allow_stale: If True, apply the wider ``STALE_TTL_MINUTES`` bound
            instead of the normal freshness bound.
        cache_file: Which cache to read. Defaults to the module path, which
            is the Premier League's. Pass it explicitly for any other league
            — see the note on ``fetch_epl_odds``.
    """
    path = cache_file or CACHE_FILE
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            cache = json.load(f)
        data = cache.get("data", None)
        # Treat empty data as no cache
        if not data:
            return None
        ts = cache.get("timestamp", "")
        # Unknown age — never serve, regardless of allow_stale. We cannot
        # confirm these odds are recent enough to bet on.
        if not ts:
            return None
        cached_at = datetime.fromisoformat(ts)
        ttl = STALE_TTL_MINUTES if allow_stale else CACHE_TTL_MINUTES
        if datetime.now() - cached_at < timedelta(minutes=ttl):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _save_cache(data, cache_file: str | None = None):
    """Save odds data to cache. Never overwrites good data with empty data."""
    if not data:
        return
    path = cache_file or CACHE_FILE
    os.makedirs(os.path.dirname(path) or CACHE_DIR, exist_ok=True)
    cache = {"timestamp": datetime.now().isoformat(), "data": data}
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _fetch_market(market_key: str, sport: str | None = None) -> list[dict]:
    """Fetch a single market from The-Odds-API bulk endpoint.

    Args:
        market_key: API market key (e.g. "totals").

    Returns:
        Raw JSON list of events, or empty list on failure.
    """
    url = f"{BASE_URL}/sports/{sport or SPORT}/odds/"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": market_key,
        "oddsFormat": "decimal",
    }

    # Hard quota guardrail: refuse to fire if we're inside the threshold
    # window. Returns empty list — the caller's existing stale-cache fallback
    # path takes over and the dashboard's red quota widget tells the
    # operator to swap the API key.
    from api.quota_tracker import is_quota_safe
    from config import QUOTA_GUARDRAIL_THRESHOLD
    if not is_quota_safe("odds_api", QUOTA_GUARDRAIL_THRESHOLD):
        print(f"  Odds API [{market_key}]: SKIPPED — quota guardrail "
              f"({QUOTA_GUARDRAIL_THRESHOLD:.0%}) tripped. Swap API key "
              "or wait for monthly rollover.")
        return []

    # Retry transient failures (5xx, timeouts) with exponential backoff.
    # Never retry 429 (quota) or other 4xx — those are deterministic and a
    # retry just burns another credit. Mirrors api/oddspapi._api_get.
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=15)
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"  Odds API [{market_key}]: {resp.status_code}, "
                  f"quota remaining: {remaining}, used: {used}")
            # Persist the quota snapshot for the dashboard widget. Best-effort:
            # any failure inside record_call is swallowed by the tracker itself.
            from api.quota_tracker import record_call, _try_parse
            record_call("odds_api",
                        remaining=_try_parse(remaining),
                        used=_try_parse(used))

            if resp.status_code == 429:
                print("  WARNING: Odds API quota exceeded!")
                return []

            if resp.status_code >= 500 and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Odds API [{market_key}] returned "
                      f"{resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"  Odds API [{market_key}] error: {resp.status_code} - "
                      f"{resp.text[:200]}")
                return []

            return resp.json()
        except requests.Timeout:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Odds API [{market_key}] timed out, "
                      f"retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  Odds API [{market_key}] timed out after "
                  f"{max_retries + 1} attempts")
            return []
        except requests.RequestException as e:
            print(f"  Odds API [{market_key}] request failed: {e}")
            return []
    return []


def _fetch_event_market(event_id: str, market_key: str) -> dict | None:
    """Fetch a market for a single event via the event-level endpoint.

    The btts market is not available on the bulk /odds endpoint but works
    on the per-event /events/{id}/odds endpoint.

    Args:
        event_id: The-Odds-API event ID.
        market_key: API market key (e.g. "btts").

    Returns:
        Raw JSON event dict with bookmakers, or None on failure.
    """
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "eu",
        "markets": market_key,
        "oddsFormat": "decimal",
    }

    # Hard quota guardrail (per-event endpoint is the heavy consumer —
    # one credit per fixture). See _fetch_market for rationale.
    from api.quota_tracker import is_quota_safe
    from config import QUOTA_GUARDRAIL_THRESHOLD
    if not is_quota_safe("odds_api", QUOTA_GUARDRAIL_THRESHOLD):
        return None

    try:
        resp = requests.get(url, params=params, timeout=15)
        # Per-event endpoint also returns quota headers — track them so
        # the BTTS fetch loop's heavy credit consumption is visible.
        from api.quota_tracker import record_call, _try_parse
        record_call(
            "odds_api",
            remaining=_try_parse(resp.headers.get("x-requests-remaining")),
            used=_try_parse(resp.headers.get("x-requests-used")),
        )
        if resp.status_code == 429:
            print("  WARNING: Odds API quota exceeded!")
            return None
        if resp.status_code != 200:
            return None
        return resp.json()
    except requests.RequestException:
        return None


def fetch_event_alt_totals(event_id: str) -> dict | None:
    """Fetch alternate_totals odds for a single event via per-event endpoint.

    Replaces the broken bulk alternate_totals call on /sports/{sport}/odds/.
    Costs 1 API credit per call — the predictor decides which fixtures are
    worth fetching based on model conviction (see config.OU15_FETCH_PROB
    _THRESHOLD).

    Args:
        event_id: The-Odds-API event ID.

    Returns:
        Dict shaped::

            {
              bookmaker_key: {
                "title": "...",
                "lines": {1.5: {"over": 1.20, "under": 4.50}, ...},
              },
              ...
            }

        Returns None on fetch failure. Bookmakers offering only one side
        (Over without Under or vice versa) are silently dropped — we only
        retain "complete" lines.
    """
    raw = _fetch_event_market(event_id, "alternate_totals")
    if not raw:
        return None

    out: dict[str, dict] = {}
    for bm in raw.get("bookmakers", []):
        bm_key = bm.get("key", "")
        if not bm_key:
            continue
        bm_lines: dict[float, dict] = {}
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "alternate_totals":
                continue
            for outcome in mkt.get("outcomes", []):
                pt = outcome.get("point")
                if pt is None:
                    continue
                pt = float(pt)
                if pt not in bm_lines:
                    bm_lines[pt] = {}
                name = (outcome.get("name") or "").lower()
                price = outcome.get("price")
                if name == "over" and price:
                    bm_lines[pt]["over"] = price
                elif name == "under" and price:
                    bm_lines[pt]["under"] = price
        # Keep only lines with both sides — incomplete entries can't be priced
        complete = {pt: sides for pt, sides in bm_lines.items()
                    if "over" in sides and "under" in sides}
        if complete:
            out[bm_key] = {
                "title": bm.get("title", bm_key),
                "lines": complete,
            }
    return out


def merge_alt_totals_into_match(match: dict, alt_data: dict) -> None:
    """Integrate per-event alt_totals data into an existing match dict.

    The downstream consumer (``get_best_odds_all_lines``) reads from
    ``match["bookmakers"][bm_key]["all_lines"]``. This function preserves
    that structure so callers don't need to know about the new fetch path.

    For bookmakers already present in match (from the totals fetch), the
    new alt-totals points are merged into their ``all_lines`` dict
    without overwriting existing 2.5 entries. Bookmakers appearing only
    in alt_totals (no main 2.5 line) get a fresh entry with whatever
    2.5 they offer (or zeros if they only carry alt lines).
    """
    if not alt_data:
        return
    for bm_key, bm_data in alt_data.items():
        new_lines = bm_data.get("lines", {})
        if not new_lines:
            continue
        if bm_key in match["bookmakers"]:
            existing = match["bookmakers"][bm_key].get("all_lines", {}) or {}
            for pt, sides in new_lines.items():
                if pt not in existing:
                    existing[pt] = sides
            match["bookmakers"][bm_key]["all_lines"] = existing
        else:
            over_25 = new_lines.get(2.5, {}).get("over")
            under_25 = new_lines.get(2.5, {}).get("under")
            match["bookmakers"][bm_key] = {
                "title": bm_data.get("title", bm_key),
                "over": over_25 or 0,
                "under": under_25 or 0,
                "is_sharp": bm_key in SHARP_BOOKS,
                "is_major": bm_key in MAJOR_BOOKS,
                "all_lines": new_lines,
            }


def fetch_alt_totals_for_events(
    event_ids: list[str],
    sleep_between: float = 0.0,
) -> dict[str, dict]:
    """Batch helper: fetch alt_totals for a list of event IDs.

    Loops over event IDs and calls ``fetch_event_alt_totals`` for each.
    Each call costs 1 API credit, so callers should pre-filter the list
    aggressively (typically just fixtures where the model has conviction
    on O/U 1.5 Over).

    Args:
        event_ids: List of The-Odds-API event IDs to fetch.
        sleep_between: Optional delay (seconds) between calls — defaults
            to zero. Pass a small value (e.g. 0.2) if you want to be
            extra polite to the API on large batches.

    Returns:
        Dict ``{event_id: {bookmaker_key: {"title", "lines"}}}``.
        Events that returned no usable data are omitted from the result.
    """
    out: dict[str, dict] = {}
    for eid in event_ids:
        data = fetch_event_alt_totals(eid)
        if data:
            out[eid] = data
        if sleep_between > 0:
            time.sleep(sleep_between)
    return out


def fetch_event_btts_odds(event_id: str) -> dict | None:
    """Fetch BTTS odds for a single event via per-event endpoint.

    BTTS has always been per-event-only (unlike totals which is bulk).
    The legacy ``fetch_epl_odds`` calls this for every fixture; we now
    expose it directly so the predictor can pre-filter and skip
    coin-flip fixtures (where edge is statistically unlikely).

    Returns:
        Dict shaped::

            {
              bookmaker_key: {
                "title": "...",
                "yes": 1.85,
                "no": 1.95,
                "is_sharp": bool,
                "is_major": bool,
              },
              ...
            }

        Returns None on fetch failure, or {} if no books offered usable
        BTTS markets.
    """
    raw = _fetch_event_market(event_id, "btts")
    if not raw:
        return None

    out: dict[str, dict] = {}
    for bm in raw.get("bookmakers", []):
        bm_key = bm.get("key", "")
        if not bm_key:
            continue
        for mkt in bm.get("markets", []):
            if mkt.get("key") != "btts":
                continue
            yes_price = None
            no_price = None
            for outcome in mkt.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if name == "Yes":
                    yes_price = price
                elif name == "No":
                    no_price = price
            if yes_price and no_price and yes_price > 1 and no_price > 1:
                out[bm_key] = {
                    "title": bm.get("title", bm_key),
                    "yes": yes_price,
                    "no": no_price,
                    "is_sharp": bm_key in SHARP_BOOKS,
                    "is_major": bm_key in MAJOR_BOOKS,
                }
    return out


def merge_btts_into_match(match: dict, btts_data: dict) -> None:
    """Integrate per-event BTTS data into an existing match dict's
    ``btts_bookmakers`` field. Idempotent — incoming entries with the
    same bookmaker key replace any pre-existing value (latest wins).
    """
    if not btts_data:
        return
    for bm_key, bm_entry in btts_data.items():
        match.setdefault("btts_bookmakers", {})[bm_key] = bm_entry


def fetch_btts_for_events(
    event_ids: list[str],
    sleep_between: float = 0.0,
) -> dict[str, dict]:
    """Batch helper: fetch BTTS for a list of event IDs.

    Each call costs 1 API credit. Pre-filter the list to only fixtures
    where the model has meaningful conviction (model_prob far from 0.5),
    since coin-flip range rarely yields edge.

    Args:
        event_ids: List of event IDs.
        sleep_between: Optional inter-call delay.

    Returns:
        ``{event_id: {bookmaker_key: btts_entry}}``. Events with no usable
        data are omitted.
    """
    out: dict[str, dict] = {}
    for eid in event_ids:
        data = fetch_event_btts_odds(eid)
        if data:
            out[eid] = data
        if sleep_between > 0:
            time.sleep(sleep_between)
    return out


def fetch_epl_odds(
    force_refresh: bool = False,
    markets: tuple[str, ...] = ("totals", "btts"),
    sport: str | None = None,
    cache_file: str | None = None,
) -> list[dict]:
    """Fetch Over/Under 2.5 and BTTS odds for all upcoming EPL matches.

    Makes separate API calls for totals and btts markets, then merges
    results by event ID. Uses cache to avoid burning API quota.

    Args:
        force_refresh: Bypass cache TTL.
        markets: Which markets to fetch. ("totals", "btts") for full,
                 ("totals",) to skip BTTS and save API calls.
        sport: Odds-API sport key. Defaults to the module's, which is the
            Premier League's.
        cache_file: Where to read and write this league's cache. Defaults to
            the module's, which is the Premier League's.

    Note:
        Both are parameters rather than module state on purpose. They used to
        be globals that callers swapped in place and restored in a `finally`,
        which is not thread-safe — and Dash serves on Flask with
        `threaded=True`, so two league tabs scan concurrently. On 2026-09-01
        that put 24 Championship fixtures in the Premier League's cache and 20
        Premier League fixtures in the Championship's, silently, for a week.
        Pass them; do not reintroduce a swap.

    Returns:
        List of match dicts with odds from all bookmakers.
    """
    if not force_refresh:
        cached = _load_cache(cache_file=cache_file)
        if cached is not None:
            return cached

    # Fetch totals via bulk endpoint
    totals_raw = _fetch_market("totals", sport=sport)

    if not totals_raw:
        # API failed (quota exhausted, network error) — use stale cache
        return _load_cache(allow_stale=True, cache_file=cache_file) or []

    # Build match dict keyed by event ID
    matches_by_id: dict[str, dict] = {}

    # Parse totals
    for event in totals_raw:
        eid = event.get("id", "")
        if eid not in matches_by_id:
            matches_by_id[eid] = {
                "id": eid,
                "home_team": event.get("home_team", ""),
                "away_team": event.get("away_team", ""),
                "commence_time": event.get("commence_time", ""),
                "bookmakers": {},
                "btts_bookmakers": {},
            }

        match = matches_by_id[eid]
        for bm in event.get("bookmakers", []):
            bm_key = bm.get("key", "")
            bm_title = bm.get("title", bm_key)

            for market in bm.get("markets", []):
                if market.get("key") != "totals":
                    continue

                lines = {}
                for outcome in market.get("outcomes", []):
                    point = outcome.get("point", 0)
                    if point is None:
                        continue
                    if point not in lines:
                        lines[point] = {}
                    if outcome.get("name") == "Over":
                        lines[point]["over"] = outcome.get("price")
                    elif outcome.get("name") == "Under":
                        lines[point]["under"] = outcome.get("price")

                over_odds = lines.get(2.5, {}).get("over")
                under_odds = lines.get(2.5, {}).get("under")
                complete_lines = {pt: lo for pt, lo in lines.items()
                                  if "over" in lo and "under" in lo}

                if over_odds and under_odds:
                    match["bookmakers"][bm_key] = {
                        "title": bm_title,
                        "over": over_odds,
                        "under": under_odds,
                        "is_sharp": bm_key in SHARP_BOOKS,
                        "is_major": bm_key in MAJOR_BOOKS,
                        "all_lines": complete_lines,
                    }

    # NOTE: bulk alternate_totals removed (April 2026) — Odds API returns
    # HTTP 422 "Markets not supported by this endpoint: alternate_totals"
    # on the bulk /sports/{sport}/odds/ endpoint. Alt totals are now only
    # available on the per-event /events/{id}/odds endpoint.
    #
    # Use ``fetch_event_alt_totals(event_id)`` (defined below) selectively —
    # the predictor calls it only for fixtures where the model probability
    # for O/U 1.5 Over crosses the conviction threshold, since pre-event
    # calls cost 1 credit each.

    # Fetch BTTS per event (not available on bulk endpoint)
    event_ids = list(matches_by_id.keys())
    btts_fetched = 0
    if "btts" not in markets:
        print(f"  Skipping BTTS fetch ({len(event_ids)} events) "
              f"— not in requested markets")
    else:
        print(f"  Fetching BTTS for {len(event_ids)} events...")
        for eid in event_ids:
            btts_event = _fetch_event_market(eid, "btts")
            if btts_event is None:
                continue

            match = matches_by_id[eid]
            for bm in btts_event.get("bookmakers", []):
                bm_key = bm.get("key", "")
                bm_title = bm.get("title", bm_key)

                for market in bm.get("markets", []):
                    if market.get("key") != "btts":
                        continue

                    btts_yes = None
                    btts_no = None
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Yes":
                            btts_yes = outcome.get("price")
                        elif outcome.get("name") == "No":
                            btts_no = outcome.get("price")

                    if btts_yes and btts_no and btts_yes > 1 and btts_no > 1:
                        match["btts_bookmakers"][bm_key] = {
                            "title": bm_title,
                            "yes": btts_yes,
                            "no": btts_no,
                            "is_sharp": bm_key in SHARP_BOOKS,
                            "is_major": bm_key in MAJOR_BOOKS,
                        }

            if match["btts_bookmakers"]:
                btts_fetched += 1

    remaining = "?"
    try:
        # Check quota with a lightweight call
        resp = requests.get(f"{BASE_URL}/sports/", params={"apiKey": API_KEY},
                            timeout=10)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        from api.quota_tracker import record_call, _try_parse
        record_call("odds_api",
                    remaining=_try_parse(remaining),
                    used=_try_parse(used))
    except requests.RequestException:
        pass
    print(f"  BTTS: {btts_fetched}/{len(event_ids)} events have odds "
          f"(quota remaining: {remaining})")

    # Filter to matches with at least one market
    matches = [m for m in matches_by_id.values()
               if m["bookmakers"] or m["btts_bookmakers"]]

    _save_cache(matches, cache_file=cache_file)
    return matches


def get_best_odds(match):
    """Find best Over and Under odds across all bookmakers for a match.

    Returns dict with best_over, best_under, best_over_book, best_under_book,
    pinnacle odds (if available), and market consensus.
    """
    books = match.get("bookmakers", {})
    if not books:
        return None

    best_over = 0
    best_under = 0
    best_over_book = ""
    best_under_book = ""
    pinnacle_over = None
    pinnacle_under = None

    all_over = []
    all_under = []

    for key, bm in books.items():
        over = bm["over"]
        under = bm["under"]
        all_over.append(over)
        all_under.append(under)

        if over > best_over:
            best_over = over
            best_over_book = bm["title"]
        if under > best_under:
            best_under = under
            best_under_book = bm["title"]

        if key in SHARP_BOOKS:
            pinnacle_over = over
            pinnacle_under = under

    # Market consensus: median implied probability across books that actually
    # priced this line. Alt-only books can report 0 for a line they do not
    # offer; including them would divide by zero and distort the median, so
    # exclude any non-positive odds (valid decimal odds are always > 1).
    valid_over = [o for o in all_over if o > 1]
    valid_under = [u for u in all_under if u > 1]
    if not valid_over or not valid_under:
        return None
    over_implied = [1.0 / o for o in valid_over]
    under_implied = [1.0 / u for u in valid_under]
    consensus_over = np.median(over_implied)
    consensus_under = np.median(under_implied)
    # Normalise to remove overround
    total = consensus_over + consensus_under
    consensus_over /= total
    consensus_under /= total

    # Sharp (Pinnacle) fair probability
    sharp_over = None
    sharp_under = None
    if pinnacle_over and pinnacle_under:
        pin_total = 1.0/pinnacle_over + 1.0/pinnacle_under
        sharp_over = (1.0/pinnacle_over) / pin_total
        sharp_under = (1.0/pinnacle_under) / pin_total

    return {
        "best_over": best_over,
        "best_under": best_under,
        "best_over_book": best_over_book,
        "best_under_book": best_under_book,
        "pinnacle_over": pinnacle_over,
        "pinnacle_under": pinnacle_under,
        "sharp_fair_over": sharp_over,
        "sharp_fair_under": sharp_under,
        "consensus_over": consensus_over,
        "consensus_under": consensus_under,
        "n_books": len(books),
        "median_over": float(np.median(valid_over)),
        "median_under": float(np.median(valid_under)),
    }


def get_best_btts_odds(match):
    """Find best BTTS Yes and No odds across all bookmakers for a match.

    Returns dict with best_yes, best_no, best_yes_book, best_no_book,
    pinnacle odds (if available), and fair probabilities.
    """
    books = match.get("btts_bookmakers", {})
    if not books:
        return None

    best_yes = 0
    best_no = 0
    best_yes_book = ""
    best_no_book = ""
    pinnacle_yes = None
    pinnacle_no = None

    all_yes = []
    all_no = []

    for key, bm in books.items():
        yes_odds = bm["yes"]
        no_odds = bm["no"]
        all_yes.append(yes_odds)
        all_no.append(no_odds)

        if yes_odds > best_yes:
            best_yes = yes_odds
            best_yes_book = bm["title"]
        if no_odds > best_no:
            best_no = no_odds
            best_no_book = bm["title"]

        if key in SHARP_BOOKS:
            pinnacle_yes = yes_odds
            pinnacle_no = no_odds

    # Market consensus: median implied probability across books that priced
    # this market. A book reporting 0 for yes/no would divide by zero and
    # distort the median, so exclude any non-positive odds.
    valid_yes = [y for y in all_yes if y > 1]
    valid_no = [n for n in all_no if n > 1]
    if not valid_yes or not valid_no:
        return None
    yes_implied = [1.0 / y for y in valid_yes]
    no_implied = [1.0 / n for n in valid_no]
    consensus_yes = np.median(yes_implied)
    consensus_no = np.median(no_implied)
    total = consensus_yes + consensus_no
    consensus_yes /= total
    consensus_no /= total

    # Sharp (Pinnacle) fair probability
    sharp_yes = None
    sharp_no = None
    if pinnacle_yes and pinnacle_no:
        pin_total = 1.0 / pinnacle_yes + 1.0 / pinnacle_no
        sharp_yes = (1.0 / pinnacle_yes) / pin_total
        sharp_no = (1.0 / pinnacle_no) / pin_total

    return {
        "best_yes": best_yes,
        "best_no": best_no,
        "best_yes_book": best_yes_book,
        "best_no_book": best_no_book,
        "pinnacle_yes": pinnacle_yes,
        "pinnacle_no": pinnacle_no,
        "sharp_fair_yes": sharp_yes,
        "sharp_fair_no": sharp_no,
        "consensus_yes": consensus_yes,
        "consensus_no": consensus_no,
        "n_books": len(books),
        "median_yes": float(np.median(valid_yes)),
        "median_no": float(np.median(valid_no)),
    }


def get_best_odds_all_lines(match):
    """Find best Over/Under odds across all bookmakers for ALL total lines.

    Returns dict: {line: {"best_over", "best_under", "best_over_book",
                          "best_under_book", "pinnacle_over", "pinnacle_under",
                          "sharp_fair_over", "sharp_fair_under", "n_books"}}
    """
    books = match.get("bookmakers", {})
    if not books:
        return {}

    # Gather all available lines across all bookmakers
    all_lines = set()
    for bm_data in books.values():
        for pt in bm_data.get("all_lines", {}).keys():
            all_lines.add(float(pt))

    result = {}
    for line in sorted(all_lines):
        best_over = 0
        best_under = 0
        best_over_book = ""
        best_under_book = ""
        pin_over = None
        pin_under = None
        n_books = 0

        for key, bm in books.items():
            bm_lines = bm.get("all_lines", {})
            # Try both float and string keys (JSON may stringify)
            lo = bm_lines.get(line) or bm_lines.get(str(line))
            if lo is None:
                continue
            over = lo.get("over")
            under = lo.get("under")
            if not over or not under:
                continue

            n_books += 1
            if over > best_over:
                best_over = over
                best_over_book = bm["title"]
            if under > best_under:
                best_under = under
                best_under_book = bm["title"]

            if key in SHARP_BOOKS:
                pin_over = over
                pin_under = under

        if best_over <= 1 or best_under <= 1:
            continue

        # Sharp fair probability
        sharp_over = None
        sharp_under = None
        if pin_over and pin_under:
            pin_total = 1.0/pin_over + 1.0/pin_under
            sharp_over = (1.0/pin_over) / pin_total
            sharp_under = (1.0/pin_under) / pin_total

        result[line] = {
            "best_over": best_over,
            "best_under": best_under,
            "best_over_book": best_over_book,
            "best_under_book": best_under_book,
            "pinnacle_over": pin_over,
            "pinnacle_under": pin_under,
            "sharp_fair_over": sharp_over,
            "sharp_fair_under": sharp_under,
            "n_books": n_books,
        }

    return result


_ODDS_API_TO_DATASET = {
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "AFC Bournemouth": "AFC Bournemouth",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton and Hove Albion": "Brighton & Hove Albion FC",
    "Brighton": "Brighton & Hove Albion FC",
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
    "Wolves": "Wolverhampton Wanderers FC",
    "Watford": "Watford FC",
    "Luton Town": "Luton Town FC",
    "Sheffield United": "Sheffield United FC",
    # Promoted for 2026/27 (names as the canonical will hold them once
    # season 26 is built — see api.team_mapping.normalize).
    "Coventry City": "Coventry City FC",
    "Hull City": "Hull City AFC",
}

def match_to_our_teams(odds_match, our_teams):
    """Map Odds-API team names to our dataset team names.

    Returns None for a name that cannot be resolved unambiguously. Callers
    skip those fixtures, which is the intended outcome — pricing the wrong
    fixture is far worse than pricing none.
    """
    def resolve(api_name):
        return resolve_feed_team(api_name, our_teams, _ODDS_API_TO_DATASET)

    return resolve(odds_match["home_team"]), resolve(odds_match["away_team"])


def get_upcoming_with_odds(our_teams=None):
    """Get upcoming EPL fixtures with best odds for all markets, mapped to our team names.

    Returns list of dicts with O/U 2.5 and BTTS odds ready for dashboard display.
    """
    matches = fetch_epl_odds()
    results = []

    for m in matches:
        best_ou = get_best_odds(m)
        best_btts = get_best_btts_odds(m)

        entry = {
            "api_home": m["home_team"],
            "api_away": m["away_team"],
            "commence_time": m["commence_time"],
            "all_bookmakers": m["bookmakers"],
            "btts_bookmakers": m.get("btts_bookmakers", {}),
        }

        # O/U 2.5 odds
        if best_ou:
            entry.update({
                "best_over": best_ou["best_over"],
                "best_under": best_ou["best_under"],
                "best_over_book": best_ou["best_over_book"],
                "best_under_book": best_ou["best_under_book"],
                "pinnacle_over": best_ou["pinnacle_over"],
                "pinnacle_under": best_ou["pinnacle_under"],
                "sharp_fair_over": best_ou["sharp_fair_over"],
                "sharp_fair_under": best_ou["sharp_fair_under"],
                "consensus_over": best_ou["consensus_over"],
                "consensus_under": best_ou["consensus_under"],
                "ou_n_books": best_ou["n_books"],
            })

        # BTTS odds
        if best_btts:
            entry.update({
                "best_btts_yes": best_btts["best_yes"],
                "best_btts_no": best_btts["best_no"],
                "best_btts_yes_book": best_btts["best_yes_book"],
                "best_btts_no_book": best_btts["best_no_book"],
                "pinnacle_btts_yes": best_btts["pinnacle_yes"],
                "pinnacle_btts_no": best_btts["pinnacle_no"],
                "sharp_fair_btts_yes": best_btts["sharp_fair_yes"],
                "sharp_fair_btts_no": best_btts["sharp_fair_no"],
                "consensus_btts_yes": best_btts["consensus_yes"],
                "consensus_btts_no": best_btts["consensus_no"],
                "btts_n_books": best_btts["n_books"],
            })

        # Map to our team names if provided
        if our_teams:
            home_mapped, away_mapped = match_to_our_teams(m, our_teams)
            entry["home_team"] = home_mapped
            entry["away_team"] = away_mapped
        else:
            entry["home_team"] = m["home_team"]
            entry["away_team"] = m["away_team"]

        results.append(entry)

    return results


if __name__ == "__main__":
    print("Fetching EPL odds...\n")
    matches = fetch_epl_odds(force_refresh=True)
    print(f"\n{len(matches)} matches found\n")

    for m in matches:
        best = get_best_odds(m)
        if best is None:
            continue

        print(f"{m['home_team']} vs {m['away_team']}")
        print(f"  Kickoff: {m['commence_time']}")
        print(f"  Books: {best['n_books']}")
        print(f"  Best Over 2.5:  {best['best_over']:.2f} ({best['best_over_book']})")
        print(f"  Best Under 2.5: {best['best_under']:.2f} ({best['best_under_book']})")
        if best["pinnacle_over"]:
            print(f"  Pinnacle:       O {best['pinnacle_over']:.2f} / U {best['pinnacle_under']:.2f}")
            print(f"  Sharp fair:     O {best['sharp_fair_over']:.1%} / U {best['sharp_fair_under']:.1%}")
        print(f"  Consensus fair: O {best['consensus_over']:.1%} / U {best['consensus_under']:.1%}")
        print()
