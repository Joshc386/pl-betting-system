# Premier League Over/Under Betting Bot

## Project Overview

A statistical betting system for Premier League football markets. Uses a 4-model stacking ensemble (XGBoost + LightGBM + LogReg + Dixon-Coles Poisson) to generate probability estimates, compares them against bookmaker odds from multiple sources (The-Odds-API, OddsPapi), and produces bet recommendations with Kelly staking. Active markets: O/U 2.5 goals, alternative O/U lines (1.5, 3.5, etc.), and BTTS.

## Tech Stack

- **Language:** Python 3.13
- **Core Models:** XGBoost, LightGBM, Logistic Regression, Dixon-Coles Poisson (4-model stacking ensemble)
- **Data Sources:** Football-Data.org, FPL API, Understat, The-Odds-API, OddsPapi, Betfair, historical CSVs
- **Dashboard:** Dash + Plotly + SQLite
- **Scheduling:** APScheduler
- **Testing:** pytest
- **Dependencies:** See `requirements.txt`

## Project Status Notes

- Dixon-Coles model is fully implemented and working — do not refactor or restructure it without explicit approval
- BTTS and O/U strategies are finalised — do not modify strategy logic, thresholds, blend weights, or recommendation parameters without explicit approval
- Betfair data integration is on hold (website issues)
- Before modifying any existing file, check whether it is actively imported/used elsewhere in the project

## Active File Structure

### Core Pipeline
- `config.py` — Centralised configuration (paths, features, seasons, API keys, alt line settings)
- `pipeline.py` — Feature engineering (150+ features from CSV, xG, FPL, Understat, weather)
- `model.py` — 4-model ensemble (XGBoost + LightGBM + LogReg + Dixon-Coles) with walk-forward CV

### Prediction & Betting
- `predict.py` — Live prediction engine (O/U 2.5, BTTS, alt O/U lines)
- `alt_lines.py` — Alternative O/U line evaluation (Poisson goal distribution, Asian settlement)
- `backtest.py` — Walk-forward backtesting for O/U 2.5
- `btts_backtest.py` — BTTS backtesting
- `alt_lines_backtest.py` — Alternative O/U lines backtesting

### Data Loading
- `btts_data.py` — BTTS odds from footiqo CSVs
- `corners_data.py` — Corners O/U odds from Betfair
- `alt_lines_data.py` — Betfair goal O/U odds merger

### Database & Dashboard
- `db.py` — Database layer: SQLite connection management, schema creation, all CRUD for recommendations, predictions, match analysis, logged bets, bankroll
- `dashboard.py` — Dash web UI (port 8050) with active picks, history, performance (imports DB layer from db.py)
- `settlement.py` — Post-match bet settlement via football-data.org API

### Scheduling & Entry Points
- `run.py` — Main entry point (dashboard + scheduler + CLI actions)
- `scheduler.py` — APScheduler: daily data refresh, fixture-aware matchday fetching, weekly retrain, settlement

### API Integrations (`api/`)
- `odds_api.py` — The-Odds-API (bulk O/U 2.5 + BTTS)
- `oddspapi.py` — OddsPapi (alt lines, Asian Handicap, Pinnacle sharp lines)
- `football_data.py` — football-data.org (match results)
- `fpl.py`, `fpl_historical.py`, `fpl_team_strengths.py` — FPL data
- `player_features.py` — Squad availability from FPL-Core-Insights
- `understat_scraper.py` — Understat xG/shots/tactical data
- `team_mapping.py` — Team name normalisation
- `weather.py` — Open-Meteo weather features

### Testing (`tests/`)
- `test_settlement.py` — Settlement outcome determination
- `test_alt_lines.py` — Goal distribution, probability, Asian settlement
- `test_dashboard.py` — Market label formatting, DB operations

---

## Project-Specific Agents

### Statistical Modelling Agent

**Role:** Owns the mathematical and statistical core of the project.

**Responsibilities:**
- Maintain and improve the existing Dixon-Coles model implementation
- Build new statistical models when expanding to additional markets (e.g. Asian handicaps, both teams to score, correct score, match result)
- Implement Poisson regression, bivariate Poisson, and any other probability models needed
- Ensure all models output calibrated probabilities — not just predictions
- Validate model accuracy using proper backtesting methodology (walk-forward validation, not in-sample testing)
- Implement model diagnostics: calibration plots, Brier scores, log-loss tracking

**Constraints:**
- Never modify the working Dixon-Coles implementation without explicit approval
- All new models must include a validation step before they can be used for live recommendations
- Statistical assumptions must be documented (e.g. independence assumptions, time decay choices)
- Use scipy and numpy for statistical computations, not custom implementations of standard algorithms

**Key files/areas:** Model logic, probability calculations, statistical validation

---

### Data Pipeline Agent

**Role:** Manages all data ingestion, cleaning, transformation, and storage.

**Responsibilities:**
- Build and maintain API connectors for all football data sources (match results, team stats, player stats)
- Handle CSV data ingestion and parsing for historical datasets
- Clean and normalise data across different sources (team name standardisation, date formats, league identifiers)
- Build feature engineering pipelines that transform raw data into model-ready inputs (attack/defence ratings, form metrics, home/away splits, head-to-head records)
- Implement data validation checks — flag missing matches, duplicate records, and suspicious values
- Cache API responses to avoid redundant calls and respect rate limits
- Maintain a clear separation between raw data, processed data, and model-ready features

**Constraints:**
- All API calls must include error handling, retries, and rate limit compliance
- Never overwrite raw data — always store transformations separately
- Team and player names must be standardised across all data sources using a consistent mapping
- Document every data source: what it provides, its update frequency, and any known limitations

**Key files/areas:** API connectors, data cleaning scripts, feature engineering, data storage

---

### Odds Engine Agent

**Role:** Handles bookmaker odds integration and bet recommendation logic.

**Responsibilities:**
- Build and maintain connectors to odds APIs for retrieving current bookmaker prices
- Implement odds comparison logic across multiple bookmakers to find the best available price
- Convert between odds formats (decimal, fractional, American, implied probability) as needed
- Calculate expected value (EV) by comparing model probabilities against bookmaker implied probabilities
- Implement the bet recommendation system: output a clear yes/no with confidence level and reasoning
- Build Kelly Criterion or fractional Kelly position sizing for bankroll management
- Track historical recommendations and actual outcomes for model performance monitoring
- Flag suspicious odds movements or missing markets

**Constraints:**
- Never recommend a bet with negative expected value
- Confidence levels must be clearly defined and documented (e.g. what makes something "high confidence" vs "low confidence")
- All odds must include the bookmaker source and timestamp of when they were retrieved
- Position sizing recommendations must account for the full bankroll, not individual bets in isolation

**Key files/areas:** Odds API connectors, EV calculations, recommendation engine, bet tracking

---

### Market Expansion Agent

**Role:** Handles the expansion of the system into new betting markets beyond over/under goals.

**Responsibilities:**
- Research and scope new markets for expansion (Asian handicaps, both teams to score, correct score, match result, first half markets)
- Assess which existing model components can be reused vs what needs to be built fresh for each new market
- Build market-specific probability calculations that feed from the core model outputs
- Ensure new markets integrate cleanly with the existing odds comparison and recommendation pipeline
- Prioritise markets by potential edge — focus on markets where the model is likely to have an advantage

**Constraints:**
- New markets must not break or interfere with existing over/under functionality
- Each new market must go through a paper trading / backtesting phase before going live
- Document the mathematical relationship between the core model and each new market's probability derivation
- New markets should reuse the existing data pipeline wherever possible rather than building parallel pipelines

**Key files/areas:** New market modules, probability derivations, integration with existing pipeline

---

## Agent Maintenance Rule

After any session that modifies data schemas, adds or removes data sources, changes model parameters or architecture, adds new markets, or changes the dashboard structure:

1. **Review `.claude/agents/` and `.claude/skills/`** for drift against the current codebase
2. **Update any references** that are now stale — table schemas, file paths, API endpoints, column names, model details, team name mappings, market definitions
3. **Update `.claude/commands/full-review.md`** if the review checklist no longer matches what exists
4. **Do not create new agents** unless a genuinely distinct domain of responsibility has emerged that cannot be covered by an existing agent or skill. Adding a new data source does not justify a new agent — update the existing data-qa agent instead

This is a maintenance task, not a feature. Do it at the end of the session after all code changes are complete, and summarise what was updated.

## Pre-Scan Rule

At the start of any session that involves the data pipeline, scheduling, settlement, dashboard display, or API integrations, run `/pre-scan` before making changes. This catches stale caches, missing env vars, broken DB schemas, and team name mapping gaps before they cause silent failures downstream. Skip this for sessions that only touch model logic, tests, or documentation.

---

## Agent Team Patterns For This Project

### Pattern 1: Building a new market
Spin up: Market Expansion Agent + Statistical Modelling Agent + QA Agent (global)
- Market Expansion scopes the requirements
- Statistical Modelling builds the probability model
- QA writes tests as they go

### Pattern 2: Data source integration
Spin up: Data Pipeline Agent + Security Agent (global) + Documentation Agent (global)
- Data Pipeline builds the connector
- Security reviews API key handling and error resilience
- Documentation updates the data source registry

### Pattern 3: Full feature build
Spin up: Data Pipeline Agent + Statistical Modelling Agent + Odds Engine Agent + Code Review Agent (global)
- Each agent owns their layer
- Code Review acts as final gate

### Pattern 4: Project cleanup
Spin up: Data Pipeline Agent + Documentation Agent (global) + Code Review Agent (global)
- Audit all files and identify what is actively used vs dead code
- Document the current architecture
- Propose a cleanup plan for approval before deleting anything

---

## Agent skills

### Issue tracker

GitHub Issues at `Joshc386/pl-betting-system` via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical defaults — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout — `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
