# Premier League Over/Under Betting Bot

## Project Overview

A statistical betting system for Premier League football markets. Uses a Dixon-Coles model (fully implemented) to generate probability estimates for match outcomes, compares them against bookmaker odds across multiple sources, and produces bet recommendations with confidence levels. Currently focused on over/under goals markets with plans to expand into additional football betting markets.

## Tech Stack

- **Language:** Python
- **Core Model:** Dixon-Coles (implemented and working)
- **Data Sources:** Football APIs (Football-Data.org, FBref/StatsBomb), odds APIs, historical CSV datasets
- **Testing:** pytest

## Project Status Notes

- Dixon-Coles model is fully implemented and working — do not refactor or restructure it without explicit approval
- There are miscellaneous unused files in the project that need cleanup — do not assume every file is active
- Before modifying any existing file, check whether it is actively imported/used elsewhere in the project

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
