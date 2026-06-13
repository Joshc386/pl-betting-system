---
name: dashboard-specs
description: "Dashboard specification reference covering market definitions, edge formulas, display rules, recommendation criteria, and model analytics tracking."
user-invocable: false
---

## Dashboard Specification

### Fixtures Display (Match Centre Tab)

The main view shows all upcoming Premier League and Championship fixtures within a 7-day rolling window from the current date. Data is stored in SQLite databases: `data/dashboard.db` (PL) and `data/dashboard_efl.db` (Championship).

**Per fixture, three markets with two sides each (6 rows per fixture):**

| Market | Sides | Model Source |
|--------|-------|-------------|
| O/U 1.5 | Over / Under | Dixon-Coles Poisson goal matrix |
| O/U 2.5 | Over / Under | 4-model ensemble (XGB + LGB + DC stacker) |
| BTTS | Yes / No | 4-model ensemble |

**Columns in the DataTable:**
Fixture, Kickoff, Market, Side, Odds, Bookmaker, Model %, Fair Odds, Edge %, Conf, Books, Rec, Taken

### Edge Calculation (Display Rules)

```
implied_probability = 1 / decimal_odds
overround = sum(implied_probabilities_for_both_sides)
fair_probability = implied_probability / overround
edge = model_probability - fair_probability
```

Note: In `_evaluate_bet()`, edge uses `blended_p` (35% model + 65% market), but the display shows `model_prob - fair_prob` for clarity.

**Display formatting:**
- Edge values as percentages with 1 decimal place (e.g., +4.2%, -1.8%)
- Edge > 4%: cyan bold text
- Edge 0-4%: green text
- Edge < 0: red text
- High confidence rows: dark green background tint

### Recommendation Criteria ("Rec" Column)

A prediction gets the "Rec" checkmark when it passes ALL of these filters in `_evaluate_bet()`:
- `edge >= 0.02` (2% minimum)
- `n_agree >= 2` (at least 2 base models agree the bet has positive edge)
- `EV > 0` (positive expected value: `blended_prob * odds - 1 > 0`)
- `kelly_stake > 0` (refined Kelly criterion returns a non-zero stake)

Recommendations are stored in the `recommendations` table. The "Rec" column in both the match centre and analytics tabs cross-references this table.

### Odds Sources

**Primary**: The-Odds-API (`api/odds_api.py`)
- Bulk endpoint: O/U 2.5 with 14+ bookmakers
- Alternate totals: O/U 1.5, 3.5, etc.
- Per-event: BTTS (separate API call per fixture)

**Secondary (merged in)**: OddsPapi (`api/oddspapi.py`)
- 300+ bookmakers per line
- All O/U lines from 0.5 to 7.5
- BTTS, Asian Handicap
- Fills gaps where The-Odds-API lacks alt lines or BTTS

When The-Odds-API is exhausted, OddsPapi serves as full fallback. Stale cache is loaded if both APIs are unavailable.

### Model Analytics Tab

**Purpose**: Track model prediction accuracy over time.

**Tracking scope**: All positive-edge predictions are tracked in the `predictions` table. The "Rec'd" split shows which of these were formally recommended.

**Settlement process (23:00 and 09:00 daily via scheduler.py):**
1. Fetch completed match results from ESPN public API (`api/espn_scores.py`)
2. Match results to unsettled recommendations and predictions by team name
3. Determine win/loss using `_determine_outcome()` in `settlement.py`
4. Update settled status, won flag, profit_pct, and actual_result

**Analytics display:**
- Stat cards: Predictions, Settled, Pending, Model Hit Rate, Rec'd hit rate, Not Rec'd hit rate
- Edge bucket breakdown: 0-2%, 2-4%, 4-6%, 6%+ with Recommended hit rate overlay
- Market breakdown: per-market hit rates with Rec'd split
- Fixture detail table: date, fixture, market, side, edge, odds, result, outcome, Rec flag

**Data integrity rules:**
- Predictions are append-only — never overwrite historical entries
- Settlement only processes matches with status "completed" from ESPN
- If odds were unavailable at prediction time, that market is excluded from tracking
