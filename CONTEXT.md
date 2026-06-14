# PL / EFL Betting System

A statistical betting system that generates probability estimates for English football matches, identifies positive-edge opportunities against bookmaker odds, and outputs prioritised staking advice.

## Language

### Betting lifecycle

**Bet**:
Any opportunity to stake money on a market outcome. The most general form — every fixture × market × side is a potential bet.
_Avoid_: Wager, punt

**Pick**:
A bet where the model identifies a positive expected value (edge > 0). The system's opinion that a market is mispriced, but not necessarily worth staking on at full conviction.
_Avoid_: Prediction (ambiguous — also used for raw model output)

**Recommendation**:
The highest-priority picks — the bets the model would place if it were autonomous. Passes all filters: minimum edge, model agreement, and staking thresholds. This is what appears in the "Active Picks" dashboard tab.
_Avoid_: Tip, signal

### Markets & structure

**Market**:
A binary question about a match outcome that you can stake money on. Has exactly two **Sides**.
_Avoid_: Bet type

**Side**:
One of the two possible answers to a Market's binary question. Over/Under for goals markets; Yes/No for BTTS.
_Avoid_: Direction, selection

**Line**:
The specific goal threshold for an Over/Under goals market (e.g. 0.5, 1.5, 2.5, 3.5). Each line is a distinct market.
_Avoid_: Total, handicap

**Alt Line**:
Any Over/Under goals line that is not 2.5. Called "alternative" because O/U 2.5 is the standard line offered by every bookmaker. Alt lines have thinner liquidity, fewer bookmaker prices, and are modelled by Dixon-Coles Poisson alone (not the full ensemble).
_Avoid_: Exotic, secondary

## Relationships

- A **Bet** is the superset; all **Picks** are Bets, all **Recommendations** are Picks
- A **Pick** has a positive edge but may not pass agreement or staking filters
- A **Recommendation** is a Pick that passes all filters and has a computed stake size
- A **Market** has exactly two **Sides** (binary outcome)
- Each **Line** defines a distinct Over/Under goals **Market** (O/U 1.5, O/U 2.5, O/U 3.5 are three separate markets)
- **Alt Lines** are modelled differently from the standard 2.5 line (DC Poisson only vs full ensemble)

### Model architecture

**Ensemble**:
The full prediction system: multiple base models whose outputs are combined to produce a single calibrated probability. PL uses a 4-model ensemble (XGBoost + LightGBM + Logistic Regression + Dixon-Coles). EFL uses a 3-model ensemble (XGBoost + LightGBM + Dixon-Coles — LR excluded due to insufficient data).
_Avoid_: Model (too vague — could mean any single component)

**Base Model**:
One of the individual models within the ensemble. Each makes independent predictions that are later combined. XGBoost and LightGBM learn feature interactions; Dixon-Coles encodes Poisson goal-scoring structure; Logistic Regression (PL only) provides a linear baseline.
_Avoid_: Learner, estimator

**Stacker**:
A logistic regression meta-learner trained on out-of-fold predictions from the base models. Learns optimal combination weights. The stacker IS part of the ensemble — not separate from it.
_Avoid_: Meta-model, blender (conflicts with the model-market Blend)

**Squad Adjuster**:
A second-stage overlay that takes the ensemble's base probability and adjusts it using real-time player availability data from the FPL API. A separate LogReg model that accounts for injuries, suspensions, and key absences. Only available for recent seasons (24+) due to FPL data limitations. Planned for expansion with richer player data.
_Avoid_: Player model, availability model

**Dixon-Coles (DC)**:
A bivariate Poisson model that estimates expected goals (lambdas) for each team, with a tau correction for low-score dependency. The strongest single base model by AUC. Also used standalone for alt line pricing.
_Avoid_: Poisson model (ambiguous — there are multiple Poisson variants in features)

**Lambda**:
The expected number of goals a team will score in a specific match, as estimated by Dixon-Coles. Each fixture produces a home lambda and an away lambda. These two values define the bivariate Poisson distribution from which all goal-line probabilities are derived (O/U 1.5, 2.5, 3.5, BTTS, correct score, etc.).
_Avoid_: Expected goals (ambiguous — also refers to xG from StatsBomb/Understat, which is a different thing)

**Tau (τ)**:
The Dixon-Coles correction factor applied to the four low-score cells (0-0, 0-1, 1-0, 1-1) to account for the dependency between home and away goals at low scores. Without it, independent Poisson underestimates 0-0 draws by ~15%.
_Avoid_: Correction factor (too generic)

### Data pipeline

**Normalize (team name)**:
Map any variant spelling of a team name to a single canonical form used throughout the system. Critical because different data sources use different names for the same team (e.g. "Manchester United" / "Man United" / "Man Utd"). The canonical name is source-agnostic — the same form is used regardless of league or API.
_Avoid_: Standardise, map (too generic)

**Canonical Dataset**:
The single source-of-truth match-results CSV for a league, built from football-data.co.uk raw season files plus all computed rolling features (PL: `CompleteDS_CSV.csv`; EFL: `CompleteDSChamp_CSV.csv`). The pipeline and all model training read from it. Its columns split into two kinds: **Facts** (Date, teams, goals, FTR, HT scores — copied verbatim from the raw source, immutable) and **computed features** (rolling form, promoted/derby flags — regenerated by the build script, may legitimately change when the build logic improves). A gitignored build artefact: regenerable, never hand-edited.
_Avoid_: Master file, raw data (the raw `E1_*.csv` season files are a distinct, upstream thing)

**Data Refresh**:
Re-running the build → pipeline-cache → retrain chain so the models learn from newer match data, **without changing any strategy logic** (blend weights, agreement thresholds, Kelly fraction, DC parameters, model architecture all stay byte-for-byte identical). A data refresh re-fits learned parameters only; it is explicitly *not* a strategy change and does not require the same approval gate. The weekly Sunday retrain is an automated data refresh.
_Avoid_: Retrain (ambiguous — could imply architecture changes), update

### Fixtures

**Fixture**:
A single scheduled match between two teams. The atomic unit that the system processes — each fixture generates model probabilities, edge calculations, and potential Recommendations across all active markets.
_Avoid_: Match (acceptable synonym but "fixture" is the codebase term), game

**Matchday**:
A calendar day on which one or more fixtures take place. A single gameweek typically spans multiple matchdays (e.g. Friday, Saturday, Sunday, Monday).
_Avoid_: Game day

**Gameweek**:
The period in which every team in a league plays once. Contains 10 fixtures for PL, 12 for EFL Championship. Typically spans 3-4 matchdays across a weekend. The scheduler triggers scans relative to individual fixture kickoff times, not gameweek boundaries.
_Avoid_: Round, matchweek

### Operations

**Scan**:
A single pass through all upcoming fixtures: fetches live bookmaker odds from APIs, compares against model probabilities, and produces/updates Recommendations. Multiple scans per matchday (KO-3h and KO-1h) bracket the lineup announcement window — the early scan captures pre-lineup odds, the late scan captures post-lineup adjusted odds.
_Avoid_: Refresh, update (too generic)

**Settlement**:
The system's internal process of matching a Recommendation to a final match result and recording whether it won or lost. Independent of any bookmaker — purely internal record-keeping based on the actual score.
_Avoid_: Payout, resolution

**CLV (Closing Line Value)**:
A performance metric comparing the odds captured at bet time against the final pre-kickoff closing odds. Consistently beating the closing line is the strongest evidence of a real edge. Not used in edge detection or staking — purely retrospective validation.
_Avoid_: Line movement, odds drift

### Validation

**Walk-Forward (CV)**:
The cross-validation methodology used throughout this system. Train on all seasons up to N, validate on season N+1, slide forward and repeat. Prevents temporal leakage (seeing future data when predicting past matches). Produces out-of-fold predictions for each validation season which become training data for the stacker.
_Avoid_: K-Fold (explicitly rejected — shuffles time and leaks), backtesting (overlapping but different concept)

**OOF (Out-of-Fold)**:
Predictions generated by a base model on data it was NOT trained on (its walk-forward validation season). These are collected across all folds and used to train the stacker — ensuring the stacker never sees predictions on data the base models memorised.
_Avoid_: Holdout predictions, test predictions

### Feature philosophy

**Wheatcroft Principle**:
The idea that process-driven indicators from high-frequency match events (shots, shots on target, corners) are more accurate and consistent measures of team strength than raw goals, which are subject to chance. Derived from Wheatcroft's published research (LSE). Core feature engineering philosophy for this system — explains why shot and corner features dominate SHAP importance over goal-based features.
_Avoid_: N/A (unique term)

**League Position (seeding)**:
A team's rank in the league table at the point a fixture is played, used as a model feature (`Home_LeaguePosition`, `Away_LeaguePosition`, `LeaguePosition_Diff`). Mid-season it is the live points table (points, then goal difference, then goals scored). At **matchday 1, before any games are played, position is seeded by the previous season's outcome**, not left undefined: returning teams take their finishing position from last season; promoted teams are seeded at the bottom by promotion route — division-below **champions** at 3rd-from-bottom (22nd EFL / 18th PL), **runners-up** at 2nd-from-bottom (23rd EFL / 19th PL), **play-off winners** at the very bottom (24th EFL / 20th PL). The seed is also the natural tie-breaker once games begin (equal points fall back to seed order). See [docs/adr/0002-league-position-previous-season-seeding.md](docs/adr/0002-league-position-previous-season-seeding.md).
_Avoid_: Table position (ambiguous about timing — must specify "before this match"), form

### Calibration

**Regime**:
A sustained shift in the goal-scoring environment within a season — e.g. a "high-scoring regime" where the Over rate is 65% vs the 52% historical prior. The system detects regimes via rolling in-season outcomes and adjusts model calibration mid-season. Only applied to Over/Under markets where base rates genuinely drift — BTTS is excluded because its rate is stable across seasons.
_Avoid_: Trend, form (too vague — "form" usually refers to team-level performance)

**Base Rate**:
The historical frequency of a market outcome (e.g. ~52% of PL matches go Over 2.5). Used as the calibration anchor for model probabilities. Regime detection adjusts this mid-season when the environment deviates.
_Avoid_: Prior (overloaded — also used for Bayesian team priors in DC)

### Model consensus

**Agreement**:
The count of how many base models independently exceed the fair probability threshold on the same side. A simple headcount — the margin by which each model agrees doesn't affect the count, but the count scales stake size (2 agree = 0.7× stake, 4 agree = 1.1× stake). Minimum agreement of 2 required for any Recommendation.
_Avoid_: Consensus (too vague), conviction (used informally but not precise)

### Staking

**Kelly Criterion**:
The mathematically optimal stake size given your edge and odds. Full Kelly maximises long-term growth but produces stakes that are too large in practice because the model's edge estimates are uncertain. This system uses fractional Kelly (currently quarter-Kelly, `kelly_fraction=0.25`) as a risk management layer. The fraction may be adjusted as live performance data accumulates.
_Avoid_: Optimal f, growth-optimal sizing

**Drawdown**:
The peak-to-trough decline in bankroll. The system has a piecewise drawdown protection function that automatically reduces stake sizes during losing runs (e.g. halve stakes at 15%+ drawdown). Prevents ruin during inevitable cold streaks.
_Avoid_: Loss streak (related but not the same — drawdown is cumulative)

**Market Multiplier**:
A per-market/side scaling factor applied to Kelly stakes, reflecting how much edge each market historically provides. O/U 2.5 Over gets 1.0× (strongest), O/U 1.5 Under gets 0.5× (weakest). Derived from backtest evidence.
_Avoid_: Confidence weight

### Edge & value

**Edge**:
The raw probability gap between the model's estimate and the market's fair probability. Computed as `blended_prob - fair_prob`. A positive edge means the model thinks the market is underpricing that outcome.
_Avoid_: Advantage, overlay

**EV (Expected Value)**:
The monetary expression of edge — what you'd expect to profit per unit staked. Computed as `blended_prob * odds - 1`. Positive EV means the bet is profitable in expectation.
_Avoid_: Expected return, profit margin

**Blend**:
The weighted combination of model probability (35%) and market fair probability (65%). Acts as built-in conservatism — the model only overrides the market at the margin, not wholesale.
_Avoid_: Shrinkage (overloaded — see flagged ambiguities)

**Fair Probability**:
The market's best estimate of true probability with the bookmaker's margin (overround) removed. Pinnacle is the preferred source (sharpest book, closest to efficient). When Pinnacle is unavailable, derived via proportional de-vig of soft bookmaker odds — but edges from this fallback are discounted (80%) because the margin removal is less reliable.
_Avoid_: True probability, implied probability (the latter still includes margin)

**De-vig**:
The process of removing a bookmaker's built-in margin (overround) from their odds to recover the fair probability. Proportional de-vig divides each side's implied probability by the total overround.
_Avoid_: De-juice, margin stripping

**Sharp (book)**:
A bookmaker whose odds sum close to 100% implied probability (low overround), meaning they're closest to efficient/true pricing. Pinnacle is the reference sharp book in this system — they accept professional bettors and don't limit winners, so their lines reflect informed money. Contrast with soft books whose overround is higher and pricing less efficient.
_Avoid_: Efficient (technically correct but less specific)

**Soft (book)**:
A bookmaker with higher overround and less efficient pricing. Soft books limit or ban winning bettors, so their odds reflect recreational money rather than informed pricing. Edges derived from de-vigging soft books are discounted (80%) because the fair probability estimate is less reliable.
_Avoid_: Recreational book

## Relationships

- A **Bet** is the superset; all **Picks** are Bets, all **Recommendations** are Picks
- A **Pick** has a positive edge but may not pass agreement or staking filters
- A **Recommendation** is a Pick that passes all filters and has a computed stake size
- A **Market** has exactly two **Sides** (binary outcome)
- Each **Line** defines a distinct Over/Under goals **Market** (O/U 1.5, O/U 2.5, O/U 3.5 are three separate markets)
- **Alt Lines** are modelled differently from the standard 2.5 line (DC Poisson only vs full ensemble)
- The **Ensemble** produces a model probability per **Fixture** × **Market**; the **Squad Adjuster** optionally modifies it
- **Edge** is computed from the **Blend** minus the **Fair Probability**
- **EV** is the monetary translation of **Edge** at the available odds
- **Agreement** gates whether a Pick becomes a **Recommendation**; it also scales **Kelly** stake size
- A **Scan** processes all upcoming **Fixtures**, producing **Picks** and **Recommendations**
- **Settlement** closes a **Recommendation** by matching it to a final score
- **CLV** retrospectively validates whether the **Scan** captured good odds
- A **Gameweek** contains multiple **Matchdays**; each **Matchday** contains one or more **Fixtures**
- **Walk-Forward CV** produces **OOF** predictions that train the **Stacker**
- **Regime** detection adjusts the **Base Rate** which shifts the ensemble's calibration
- **Lambdas** from **Dixon-Coles** feed into the Poisson distribution to price any **Line**

## Example dialogue

> **Dev:** "The KO-1h **Scan** found a new **Pick** on Arsenal vs Chelsea O/U 2.5 Over — the **Edge** jumped to 4.2% after lineups dropped."
> **Domain expert:** "How many models in **Agreement**?"
> **Dev:** "Three — XGB, LGB, and DC all above **Fair Probability**. LR is below."
> **Domain expert:** "That's enough for a **Recommendation**. What does **Kelly** say for the stake?"
> **Dev:** "2.8% of bankroll after the **Market Multiplier** and **Drawdown** factor."
> **Domain expert:** "Good. We'll check **CLV** after kickoff to see if we beat the closing line."

## Flagged ambiguities

- "Pick" vs "Recommendation" — resolved: Picks have positive edge; Recommendations are the subset the model actively advises staking on.
- "Prediction" — ambiguous; could mean raw model probability output OR a Pick. Prefer "model probability" for the raw number, "Pick" for the edge-positive opportunity.
- "Bayesian edge shrinkage" — documented in Obsidian decisions doc with code snippets and impact numbers, but the function does NOT exist in the codebase. The *effect* it describes (pulling edge toward zero based on model agreement) is partially achieved by the blend weight + agreement scaling in `refined_kelly()`. Needs resolution: either build it or correct the docs.
