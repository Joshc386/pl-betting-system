---
name: data-schema
description: "Data source documentation, schema definitions, and API contracts for the betting bot's data pipeline."
user-invocable: false
---

## Data Pipeline Architecture

### Source Overview

The system ingests from multiple data sources across APIs, a web scraper, and historical CSVs. When validating or modifying any pipeline component, always verify against this reference.

### 1. Odds Data (Two Sources, Merged)

**The-Odds-API** (`api/odds_api.py`):
- Sport keys: `soccer_epl` (PL), `soccer_efl_champ` (Championship)
- Bulk endpoint: O/U 2.5 totals (14+ bookmakers)
- Alternate totals endpoint: O/U 1.5, 3.5, 4.5 etc.
- Per-event endpoint: BTTS (one API call per fixture)
- Cache: `data/odds_cache.json` (PL), `data/odds_cache_efl.json` (Championship)

**OddsPapi** (`api/oddspapi.py`):
- Tournament IDs: 17 (EPL), 18 (Championship)
- 300+ bookmakers, all O/U lines from 0.5 to 7.5, BTTS, Asian Handicap
- Cache: `data/oddspapi_cache.json` (PL), `data/oddspapi_cache_efl.json` (Championship)
- Merged into The-Odds-API data via `_merge_oddspapi_into_matches()` in `dashboard.py`
- Falls back to full replacement when The-Odds-API is exhausted

**Merged match dict format:**
```python
{
    "id": "event_id",
    "home_team": "Team Name",
    "away_team": "Team Name",
    "commence_time": "2026-04-18T14:00:00Z",
    "bookmakers": {
        "bookmaker_key": {
            "title": "Display Name",
            "over": 1.95,  # O/U 2.5 over odds
            "under": 1.85,
            "is_sharp": bool,
            "is_major": bool,
            "all_lines": {
                2.5: {"over": 1.95, "under": 1.85},
                1.5: {"over": 1.30, "under": 3.50},
            }
        }
    },
    "btts_bookmakers": {
        "bookmaker_key": {
            "title": "Display Name",
            "yes": 1.85, "no": 2.00,
        }
    }
}
```

### 2. Feature Data (Training Pipeline)

**Historical CSVs** (`data/` directory):
- Football-Data.org match results with betting odds columns
- Loaded in `pipeline.py` (PL) and `championship_pipeline.py` (Championship)

**FPL API** (`api/fpl.py`, `api/fpl_historical.py`, `api/fpl_team_strengths.py`):
- Team strengths, player availability, form metrics
- PL only (FPL doesn't cover Championship)

**Understat** (`api/understat_scraper.py`):
- xG, shots, tactical data per match
- PL only (Understat covers top 5 leagues only)

**Player Features** (`api/player_features.py`):
- Squad availability from FPL-Core-Insights GitHub repo
- `PLAYER_FEATURES` (4) **are** in the main model — live in both leagues
- `SQUAD_FEATURES` (16) are **not**, and now feed nothing: 2024-25 onwards only, so they are NaN for every training season and cannot enter `ALL_FEATURES`. The squad adjuster that consumed them was deleted 2026-08-14 (dead at both ends — see CONTEXT.md). `pipeline.py` still computes the columns; no model reads them.

**Weather** (`api/weather.py`):
- Open-Meteo API for match-day weather conditions

### 3. Result Verification (ESPN API)

**ESPN Public Scoreboard** (`api/espn_scores.py`):
- No API key required
- Endpoints: `site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard` (PL), `eng.2` (Championship)
- Query parameter: `dates=YYYYMMDD`
- Returns completed matches with scores, status, team names
- Team name mapping in `_ESPN_TO_PL` and `_ESPN_TO_CHAMP` dicts

### SQLite Database Schema

Two databases: `data/dashboard.db` (PL), `data/dashboard_efl.db` (Championship). Identical schema:

**`match_analysis`** — scan results (one row per fixture-market-side):
```sql
id, scanned_at, home_team, away_team, kickoff, matchday,
market, side, best_odds, best_bookmaker, model_prob, fair_odds,
edge_pct, confidence, n_books, per_model_json, bookmaker_odds_json
```

**`recommendations`** — positive-EV bets passing strict filters:
```sql
id, created_at, home_team, away_team, kickoff,
market, side, model_prob, blended_prob, fair_prob, odds,
edge, ev, stake_pct, confidence, best_bookmaker,
n_books, n_agree, per_model_json,
settled, won, profit_pct, actual_result, settled_at
```

**`predictions`** — all positive-edge predictions:
```sql
id, created_at, home_team, away_team, kickoff,
market, side, model_prob, fair_odds, edge_pct,
best_odds, best_bookmaker, confidence, bookmaker_odds_json,
taken, settled, won, actual_result, settled_at
```

**`logged_bets`** — user-tracked bets
**`bankroll`** — bankroll snapshots

### Team Name Standardisation

This is the single most common source of data issues. Multiple mapping layers exist:

| Context | Module | Format Example |
|---------|--------|---------------|
| PL CSV/DB | `api/team_mapping.py` `normalize()` | "Arsenal FC", "Manchester United FC" |
| Championship DB | `championship_predict.py` `_ODDS_API_TO_CHAMP` | "West Brom", "QPR", "Sheffield Weds" |
| The-Odds-API | Raw API response | "Southampton", "Millwall" (no FC) |
| OddsPapi | Raw API response | "Southampton FC", "Millwall FC" (with FC) |
| ESPN | `api/espn_scores.py` mappings | "Southampton", "Queens Park Rangers" |
| Merge matching | `dashboard.py` `_normalise_team_for_merge()` | Strips FC/AFC, lowercases |

When reviewing any pipeline change, verify that new sources use the appropriate mapping layer.
