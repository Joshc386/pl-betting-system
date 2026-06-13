---
name: settlement-pipeline
description: "Settlement flow reference covering ESPN data fetching, team name resolution, outcome determination, and database updates for both recommendations and prediction tracking."
user-invocable: false
---

## Settlement Pipeline Reference

### Overview

Settlement runs twice daily (09:00 and 23:00 UK time) via `scheduler.py`. It pulls completed match results from ESPN's public API, matches them to unsettled bets and predictions in SQLite, determines win/loss, and updates the database. No API key required.

### Data Flow

```
ESPN Public API (no key)
    -> fetch_completed_matches() [api/espn_scores.py]
        -> _resolve_team() maps ESPN names to DB names
    -> get_finished_matches() [settlement.py]
        -> adds btts field (home_goals > 0 and away_goals > 0)
    -> settle_bets() [settlement.py]
        -> matches (home_team, away_team) against unsettled recommendations
        -> _determine_outcome() determines win/loss
        -> updates recommendations table: settled=1, won, profit_pct, actual_result, settled_at
        -> updates bankroll table with new balance
    -> settle_predictions() [settlement.py]
        -> same matching logic against predictions table (all positive-edge, not just recommended)
        -> updates predictions table: settled=1, won, actual_result, settled_at
```

### ESPN API Details

**Endpoint**: `https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard`

**League IDs** (in `api/espn_scores.py`):
- `eng.1` — Premier League (code: `PL`)
- `eng.2` — Championship (codes: `EFL`, `ELC`)

**Query parameter**: `dates=YYYYMMDD` (one date per request)

**Lookback window**: 7 days by default (`days_back=7`). Each date is queried separately for each competition.

**Match completion check**: `event.competitions[0].status.type.completed == True`

**Score extraction**: `competitors[0]` is always home, `competitors[1]` is always away. Score in `competitor.score`.

**Deduplication**: Uses `(home_db, away_db, date_str)` tuple to prevent double-counting when the same match appears in adjacent date queries.

### Team Name Mapping (ESPN -> DB)

ESPN names must be resolved to the format used in the dashboard database. Two mapping dicts handle this:

**`_ESPN_TO_PL`** — ESPN name to PL DB name (with FC suffix):
| ESPN Name | DB Name |
|-----------|---------|
| Arsenal | Arsenal FC |
| Manchester United | Manchester United FC |
| AFC Bournemouth | AFC Bournemouth |
| Sunderland | Sunderland AFC |
| *(22 teams total)* | |

**`_ESPN_TO_CHAMP`** — ESPN name to Championship DB name (short form):
| ESPN Name | DB Name |
|-----------|---------|
| Queens Park Rangers | QPR |
| Sheffield Wednesday | Sheffield Weds |
| West Bromwich Albion | West Brom |
| Birmingham City | Birmingham |
| *(24 teams total)* | |

**Fallback**: If no mapping found, calls `normalize()` from `api/team_mapping.py`.

### Outcome Determination

`_determine_outcome(market, side, home_goals, away_goals)` in `settlement.py`:

**Over/Under markets** (`ou15`, `ou25`, `ou35`, etc.):
- Parses goal line from market code: `ou25` -> 2.5, `ou15` -> 1.5
- `actual = "over"` if `total_goals > line`, else `"under"`
- Won if `side == actual`

**BTTS market**:
- `btts = home_goals > 0 and away_goals > 0`
- `actual = "yes"` if btts, else `"no"`
- Won if `side == actual`

### Database Updates

**Recommendations table** (settled bets with stakes):
```sql
UPDATE recommendations
SET settled=1, won=?, profit_pct=?, actual_result=?, settled_at=?
WHERE id=?
```
- `profit_pct = stake_pct * (odds - 1)` if won, else `-stake_pct`
- Bankroll table gets a new row with updated balance

**Predictions table** (all positive-edge, for model accuracy tracking):
```sql
UPDATE predictions
SET settled=1, won=?, actual_result=?, settled_at=?
WHERE id=?
```
- No profit/stake calculation — purely for hit rate tracking

### Settlement Scope

- `settle_bets()`: Operates on a single league DB (`ACTIVE_LEAGUE` from config)
- `settle_predictions()`: Iterates ALL league DBs in `LEAGUE_DB_PATHS` (both PL and EFL)

### Known Edge Cases

1. **Team not in mapping**: Falls back to `normalize()`, which may not produce correct DB name. Symptom: match settles for one league but not the other. Fix: add the missing team to `_ESPN_TO_PL` or `_ESPN_TO_CHAMP`.

2. **Postponed/abandoned matches**: ESPN marks these as `completed=False`, so they're automatically skipped. No manual intervention needed.

3. **Walkovers/forfeits**: ESPN may report unusual scores (e.g., 3-0 awarded). Settlement treats them as normal results. This is correct for betting settlement.

4. **`edge_pct: None` in predictions**: Some predictions have null edge_pct. The verbose output handles this with a None check (`edge_str` variable).

5. **Cross-league team names**: Teams like Southampton and Ipswich can appear in both PL and Championship mappings. The competition parameter in `_resolve_team()` ensures the correct mapping is used.

6. **Stale unsettled bets**: If settlement misses a match (API downtime, mapping gap), it will be picked up on the next run within the 7-day lookback window.

### Key Files

- `api/espn_scores.py` — ESPN API client, team name mappings, `fetch_completed_matches()`
- `settlement.py` — `settle_bets()`, `settle_predictions()`, `_determine_outcome()`
- `scheduler.py` — `job_settle_bets()` called at 09:00 and 23:00 daily
- `dashboard.py` — `LEAGUE_DB_PATHS` dict mapping league codes to SQLite paths
