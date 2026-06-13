---
description: "Quick sanity check before a dashboard session — verify environment, databases, caches, and team name coverage."
---

## Pre-Scan Health Check

Run the following checks and report a pass/fail summary. Stop at the first critical failure and flag it.

### 1. Environment Variables
Verify these keys are loaded and non-empty:
- `ODDS_API_KEY` (The-Odds-API)
- `ODDSPAPI_KEY` (OddsPapi)
- `FOOTBALL_DATA_API_KEY` (football-data.org, used for fixture detection)

Check: `config.py` calls `load_dotenv()` before any `os.environ.get()`. If any key is empty string or None, flag as **CRITICAL**.

### 2. SQLite Database Integrity
For both `data/dashboard.db` (PL) and `data/dashboard_efl.db` (Championship):
- Confirm file exists
- Confirm these tables exist: `match_analysis`, `recommendations`, `predictions`, `logged_bets`, `bankroll`
- Confirm column names match the schema in `.claude/skills/data-schema/SKILL.md`
- Check for unsettled recommendations older than 7 days (may indicate settlement failure)

### 3. Odds Cache Freshness
Check these cache files:
- `data/odds_cache.json` (PL, The-Odds-API)
- `data/odds_cache_efl.json` (Championship, The-Odds-API)
- `data/oddspapi_cache.json` (PL, OddsPapi)
- `data/oddspapi_cache_efl.json` (Championship, OddsPapi)

For each: report age (hours since last update). Flag as **WARNING** if older than 48 hours on a matchday.

### 4. Market Coverage
Load the most recent The-Odds-API cache and check:
- Every fixture has at least one O/U 2.5 bookmaker
- Every fixture has at least one BTTS bookmaker (or OddsPapi fills the gap)
- O/U 1.5 lines are present (from alternate_totals or OddsPapi merge)

Flag as **WARNING** if any fixture is missing a market entirely.

### 5. Team Name Mapping Coverage
Cross-reference teams in the odds caches against:
- `_ODDS_API_TO_DATASET` in `api/odds_api.py`
- `_ODDSPAPI_TO_DATASET` in `api/oddspapi.py`
- `_ESPN_TO_PL` and `_ESPN_TO_CHAMP` in `api/espn_scores.py`
- `_ODDS_API_TO_CHAMP` in `championship_predict.py`

Flag as **WARNING** any team that appears in API data but has no explicit mapping (relying on fuzzy fallback).

### 6. Model Pickle State
Check `models/` directory for:
- `pl_trained_state.pkl` — PL trained models
- `efl_trained_state.pkl` — Championship trained models
- `pl_pipeline_cache.pkl` — PL pipeline DataFrame cache
- `efl_pipeline_cache.pkl` — Championship pipeline DataFrame cache

Report age of each file. Flag as **WARNING** if older than 7 days (should be refreshed weekly by Sunday retrain).

### Output Format

```
PRE-SCAN RESULTS
================
[PASS] Environment variables loaded
[PASS] SQLite databases intact (PL: 5 tables, EFL: 5 tables)
[WARN] Odds cache stale — PL last updated 52 hours ago
[PASS] Market coverage complete (10 PL fixtures, 12 EFL fixtures)
[WARN] Unmapped team: "Wrexham AFC" in ESPN Championship data
[PASS] Model pickles fresh (PL: 2 days, EFL: 2 days)

Summary: 4 passed, 2 warnings, 0 critical
```
