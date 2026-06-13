---
name: api-budget
description: "API rate limits, quota budgets, call costs per endpoint, cache behaviour, and fallback strategies for all external data sources."
user-invocable: false
---

## API Budget Reference

### The-Odds-API (`api/odds_api.py`)

**Quota**: 500 requests/month (free tier)
**Budget target**: ~150 calls/month across both leagues

**Call costs per endpoint**:
| Endpoint | Cost | What it fetches |
|----------|------|-----------------|
| `GET /sports/{sport}/odds/` (totals) | 1 request | All fixtures, O/U 2.5 odds, 14+ bookmakers |
| `GET /sports/{sport}/odds/` (alternate_totals) | 1 request | All fixtures, O/U 1.5/3.5/4.5 etc. |
| `GET /sports/{sport}/events/{id}/odds` (btts) | 1 request per fixture | BTTS odds for a single fixture |

**Typical matchday cost**:
- Morning fetch (all markets): 1 totals + 1 alt_totals + N btts = ~14 calls (10-12 PL fixtures)
- Pre-kickoff fetch (totals only): 1 totals + 1 alt_totals = 2 calls
- Championship morning fetch: ~16 calls (12-14 fixtures)

**Monthly estimate** (dynamic scheduler):
| Scenario | Calls |
|----------|-------|
| PL matchday mornings (~10/month) | ~130 |
| PL pre-kickoff (~25 kickoffs/month) | ~50 |
| EFL matchday mornings (~12/month) | ~156 |
| EFL pre-kickoff (~30 kickoffs/month) | ~60 |
| **Total** | **~150** |

**Sport keys**: `soccer_epl` (PL), `soccer_efl_champ` (Championship)

**Cache**: `data/odds_cache.json` (PL), `data/odds_cache_efl.json` (Championship)
- TTL: 30 minutes (`CACHE_TTL_MINUTES = 30`)
- Stale cache loaded when API fails (`allow_stale=True` fallback)
- Cache never overwritten with empty data (`_save_cache` guards against this)

**Quota tracking**: Response headers `x-requests-remaining` and `x-requests-used` logged on every call. A lightweight `/sports/` call checks remaining quota after BTTS fetch.

**Rate limit response**: HTTP 429. Logged as warning, returns empty list, falls back to stale cache.

**`markets` parameter**: `fetch_epl_odds(markets=("totals",))` skips BTTS to save N calls per fetch. Used for pre-kickoff refreshes where only totals odds are changing.

### OddsPapi (`api/oddspapi.py`)

**Quota**: 250 requests/month (free tier)
**Budget target**: ~50-80 calls/month

**Call costs**:
| Endpoint | Cost | What it fetches |
|----------|------|-----------------|
| `GET /fixtures` | 1 request | All fixtures for a tournament |
| `GET /odds` | 1 request per fixture | All markets for one fixture (O/U, AH, BTTS, match result) |

**Typical cost per league refresh**: 1 fixtures + N odds = ~13-15 calls (12-14 fixtures)

**Tournament IDs**: 17 (EPL), 18 (Championship)

**Cache**: `data/oddspapi_cache.json` (PL), `data/oddspapi_cache_efl.json` (Championship)
- TTL: Configurable via `ODDSPAPI_CACHE_TTL_MINUTES` in `config.py`

**Retry logic**: 3 attempts with exponential backoff (2^attempt seconds) for 5xx errors and timeouts. 429 returns immediately with None.

**Rate limiting**: 0.5s sleep between fixture odds fetches (`time.sleep(0.5)` in `fetch_all_odds()`).

**`max_fixtures` parameter**: Caps how many fixtures to fetch odds for. Use for quota management during testing.

### ESPN Public API (`api/espn_scores.py`)

**Quota**: Unlimited (no API key required)
**Cost**: 0 against any paid quota

**Endpoint**: `GET /apis/site/v2/sports/soccer/{league}/scoreboard?dates=YYYYMMDD`
- One request per date per competition
- Default lookback: 7 days x 2 competitions = ~14 requests per settlement run

**No cache**: Results fetched fresh each time (cheap and always up to date)

**Timeout**: 10 seconds per request

### Football-Data.org (fixture detection only)

**Quota**: 10 requests/minute (free tier, token required)
**Used by**: `fixture_schedule.py` for matchday detection (not odds or settlement)
**Cost**: 1-2 calls per daily planner run at 07:00

### OddsPapi + Odds-API Merge Strategy

OddsPapi is merged into The-Odds-API data via `_merge_oddspapi_into_matches()` in `dashboard.py`:

1. **Primary**: The-Odds-API provides bulk O/U 2.5 and per-event BTTS
2. **Merge**: OddsPapi fills alt lines (O/U 1.5, 3.5, etc.) and BTTS gaps without overwriting existing data
3. **Fallback**: When The-Odds-API quota is exhausted, OddsPapi serves as full replacement

**Matching**: `_normalise_team_for_merge()` strips FC/AFC suffixes and lowercases for cross-API fixture matching.

### Budget Alerts

When reviewing or modifying code, flag any change that would:
- Add new API calls inside a loop (e.g., per-fixture BTTS fetch for a new market)
- Remove or bypass cache TTL checks
- Add a new `force_refresh=True` call path
- Reduce `time.sleep()` between OddsPapi requests
- Add a new scheduled job that makes API calls
- Change `CACHE_TTL_MINUTES` to a shorter value

### Scheduler API Usage

The dynamic scheduler (`scheduler.py`) controls when API calls happen:

| Job | When | API calls |
|-----|------|-----------|
| Weekly retrain | Sunday 23:30 | 0 (uses cached CSV data) |
| Daily planner | 07:00 | 1-2 (football-data.org) |
| Morning fetch | 09:00 on matchday | ~26 (PL+EFL, all markets) |
| Pre-kickoff | KO - 60min | ~8 per league (totals only) |
| Settlement | 09:00 + 23:00 | 0 (ESPN, unlimited) |
