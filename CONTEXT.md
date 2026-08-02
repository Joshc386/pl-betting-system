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

**Monitored Market**:
A Market the Ensemble still prices and whose outcomes are settlement-tracked for calibration (to observe whether the model over/under-estimates it), but which **never produces staked Recommendations**. A market is moved to Monitored status when its proven edge is too thin to justify the opportunity cost of tying up capital. The first Monitored Market is **EFL O/U 2.5** (~+1.2% gross ROI, likely negative after exchange commission): kept in the model for diagnostics and possible future re-activation, not bet. Contrast with an **Active (staked) Market**, which can produce Recommendations.
_Avoid_: Paper market (acceptable synonym), dead market (it is still priced and tracked, not dead)

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
The single source-of-truth match-results CSV for a league (PL: `CompleteDSPL_CSV.csv`; EFL: `CompleteDSChamp_CSV.csv`). **Exactly one artefact per league** — the pipeline and all model training read from it, and no rival file may take precedence. Its columns split into three kinds:

- **Facts** (Date, teams, goals, FTR, HT scores, shots, corners, cards, closing B365 odds) — copied verbatim from the raw source, immutable, byte-identical across a regeneration. **football-data.co.uk is the sole authority for Facts in both leagues** (`E0` for PL, `E1` for EFL), for *both* the full rebuild and in-season incremental appends. An incremental append is only sound when it carries the same columns as a full rebuild; appending from a thinner source silently produces structurally poorer rows that no schema records. Understat supplies **xG enrichment only** — it is not a Facts source.
- **Computed features** (rolling form, promoted/derby flags) — derived from the Facts by the build script; may legitimately change when build logic improves.
- **Enrichment columns** (xG from Understat, injury burden from FPL) — joined from a *separate external source* rather than derived from the Facts. Legitimately sparse: they only exist for seasons the upstream source covers, so partial coverage is expected, not a defect.

**Facts always win.** Facts and computed features are the critical path and are always written fresh. The enrichment join is *best-effort*: if the external source fails, the previous run's enrichment columns are carried forward and the file records an **enrichment as-of** date per source. A flaky secondary source must never block ingestion of match results, and carried-forward enrichment must be visible as a date rather than silently indistinguishable from fresh.

A gitignored build artefact: regenerable, never hand-edited.
_Avoid_: Master file, raw data (the raw `E1_*.csv` season files are a distinct, upstream thing), **Enriched dataset** (superseded 2026-07-25 — enrichment is a *column class within* the canonical, not a separate file; the old `CompleteDSPL_enriched.csv` was an undocumented rival artefact that silently took precedence over the canonical and went two months stale)

**Data Refresh**:
Re-running the build → pipeline-cache → retrain chain so the models learn from newer match data, **without changing any strategy logic** (blend weights, agreement thresholds, Kelly fraction, DC parameters, model architecture all stay byte-for-byte identical). A data refresh re-fits learned parameters only; it is explicitly *not* a strategy change and does not require the same approval gate. The weekly Sunday retrain is an automated data refresh.
_Avoid_: Retrain (ambiguous — could imply architecture changes), update

**Freshness Gate**:
A hard precondition on both producing **Recommendations** and running a **Data Refresh** for a league: every fixture football-data.org reports as `FINISHED` within a rolling window must be present in that league's **Canonical Dataset**. Reconciles against an *authoritative fixture list* rather than a date heuristic, so "no fixtures were played" (international break, off-season) is distinguishable from "ingestion silently broke" — an ambiguity that previously required a manual off-season flag and allowed two months of staleness to pass unnoticed. Falls back to a date heuristic only if football-data.org is unreachable.

On failure the gate blocks that league's Data Refresh *and* its Recommendation output; the other league is unaffected (the leagues have independent canonicals, pipelines and databases). **Settlement** and dashboard display continue, since neither reads the Canonical Dataset. The gate is **league-wide, never per-fixture**: rolling-form features have a per-team blast radius, but league-table features (`LeaguePosition_Diff`) mean one missing result perturbs the ranking of teams that never played in it, so gate granularity must match the widest feature's blast radius.
_Avoid_: Freshness check, staleness check (too generic — this is a hard gate with defined consequences, not a warning)

**Betfair GB Feed**:
The country-wide GB Over/Under + BTTS odds extracted from Betfair **historical** data (Basic plan — last-traded-price only, **no liquidity/volume**). Stored as master CSVs (`betfair_goal_ou.csv`, `betfair_btts.csv`) spanning all GB football, refreshed monthly by an automated Task Scheduler job, then narrowed to PL/EFL via League Split. Distinct from live **exchange** odds. See `docs/betfair_ingestion_scope.md`.
_Avoid_: Betfair data (ambiguous — could mean the live exchange feed)

**League Split**:
The derived step that filters the Betfair GB Feed down to one league's fixtures by mapping Betfair team names to the Canonical Dataset and joining on (home, away) + ±1 day. Produces `betfair_pl_*` / `betfair_efl_*` files. Unmapped names (women's, reserves, youth, non-league, overseas) are dropped silently-but-safely — a missing senior team surfaces as reduced per-season coverage, not an error.

**Ordering rule:** the split is downstream of *two* inputs — the Betfair GB Feed **and** the Canonical Dataset — so it must be re-derived whenever *either* changes, not only when Betfair data arrives. A split regenerated only on the monthly Betfair cadence will silently lag a canonical updated out-of-band.

The two upstream feeds run on deliberately **decoupled cadences**: Facts ingest per-matchday (football-data.co.uk publishes same evening), while the Betfair GB Feed stays monthly because it is a *historical archive* that lags by design — not a tuning choice. Per-fixture exchange prices would require the live **Execution Venue** feed, which is a separate pipeline.
_Avoid_: Filter, league filter (too generic)

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

**Scoring Rate / Scoring Index**:
Two related but distinct measures of how much a team scores, kept as **separate features** because they answer different questions:

- **Scoring Rate** (`Home_ScoringRate_10` / `Away_ScoringRate_10`) — rolling mean goals scored over the team's last 10 matches. The absolute level.
- **Scoring Index** (`Home_ScoringIndex_10` / `Away_ScoringIndex_10`) — that same rate divided by the **league average for that season**. Centred on 1.0, so it reads as "this team scores 20% above the division's norm".

They correlate at ~0.99, so the index is not carrying much *extra* information within a season — its value is **cross-era comparability**. The league scoring environment drifts substantially over the dataset: PL ranges from 1.225 goals per team per game (season 6) to 1.637 (season 23), a **33.6% swing**, and the model trains across all 26 seasons at once. 1.4 goals a game is an above-average attack in one era and a below-average one in another; the Rate cannot express that and the Index can. This is the cross-season counterpart to **Regime**, which describes the same drift *within* a season. See [ADR 0007](docs/adr/0007-one-feature-contract-per-name.md).

Retires the name **Factor**, which said nothing about what it measured and — worse — denoted the Rate in the PL canonical and the Index in the EFL one. Same column, two quantities, no way to tell from the name.
_Avoid_: Factor, Home Factor, Away Factor (superseded — ambiguous, and historically overloaded across the two leagues)

**H2H (Head-to-Head)**:
The goal-scoring history of *this specific pairing* before the current fixture, as **two features only**: `H2H_AvgGoals_5` (mean total goals over the last 5 meetings) and `H2HAvgGoals` (mean over all prior meetings). Both are uncapped in the sense that matters — the 5 is the window, not a hard limit on history retained.

Deliberately **goal averages, not results**. `H2H_HomeWins` / `H2H_AwayWins` / `H2H_Draws` were dropped: measured against total goals they sit at ±0.01–0.045 correlation, and `H2H_HomeWins` has *opposite signs* in the two leagues (+0.045 PL, −0.012 EFL) — the signature of noise plus a definitional split. They also encode **match result**, which no market this system bets actually asks about.

The goal averages survive the same test: `H2HAvgGoals` correlates +0.051 (PL) and +0.022 (EFL) with total goals, and **retains +0.047 / +0.021 after team form is partialled out** — so it is not a restatement of "these two teams score a lot".

Both leagues must use the same window. Previously PL capped all H2H at 5 meetings while EFL was uncapped, which made `H2H_AvgGoals_5` and `H2HAvgGoals` *the same column* in PL — a duplicate feature nobody noticed because the two implementations were never compared.
_Avoid_: Rivalry record, historical record (both suggest results matter, which is what the evidence rejects)

**Defensive Strength**:
How well a team prevents the opposition scoring — deliberately **three separate components**, never one number, because they are driven by different things and a single score hides which one moved:

- **Shot Suppression** — how few shots the team allows (volume), adjusted for the attacking strength of the opponent faced: per-match shots conceded ÷ the opponent's own pre-match rolling-5 shot volume, averaged over the team's last 5. 1.0 = the opponent got exactly its usual volume; below 1 = suppression. The adjuster is the opponent's shot *generation*, not Elo — process beats results (Wheatcroft), and a ratio of like quantities is self-normalising across leagues and eras, which is what the feature contract wants. The most Wheatcroft-aligned of the three.
- **Chance Quality Allowed** — SOT conceded ÷ shots conceded. How dangerous the chances allowed were; a Facts-only proxy for xG per shot.
- **Conversion Allowed** — goals conceded ÷ SOT conceded. How many of those chances were finished — keeper quality plus finishing variance, and the noisiest of the three.

The first is computed from Facts; the second and third are what the EFL builder already computed, correctly, under the misleading name `DefensiveStrength_5` / `DefensiveStrength_SOT`. The PL formula (`1 ÷ Σ shots conceded`) was never a defensive metric at all — the reciprocal of a *count* scales with how many matches are in the window, not with how well the team defends.

**Computed in the pipeline, not the Canonical Dataset.** Opponent-adjustment needs each side's rolling shot volume at match time (originally sketched as Elo; implemented as the shot-volume ratio above), and the xG/lineup variants need enrichment — neither belongs at build time; putting them in the canonical would violate the Facts-only rule for computed columns (ADR 0004). One implementation in `features/common.py` serves both leagues.

Defensive data arrives in **three tiers of decreasing coverage**, each a *separately named* set of features rather than one name whose formula varies:

| Tier | Source | Seasons | Leagues |
|---|---|---|---|
| 1 — Facts | Canonical Dataset | 0–25 | PL + EFL |
| 2 — team xG conceded | Understat | 14–25 | PL only |
| 3 — player & goalkeeper level | FPL-Core-Insights (`xgot_faced`, `goals_prevented`, minutes, positions) | 24–25 | PL only |

A league or era simply has NaN on the tiers it cannot reach — XGBoost handles NaN natively, so coverage degrades gracefully. **The EFL's "different approach" is that it runs on tier 1 alone; it must never mean the same column name computed a different way**, which is the exact defect this vocabulary exists to prevent.

Tier 3 reaches only ~7% of training rows, so it was **provisional**: it ships only if a walk-forward comparison on the final fold shows it improves AUC/Brier. Tier 3's `goals_prevented` is what separates keeper shot-stopping from defensive quality — the confound baked into Conversion Allowed. **The gate ran 2026-08-02 and failed** — `GKShotStopping_5` moved final-fold AUC by nothing (home) and by less than its own standard deviation (away) — so tier 3 is computed and tested but **not shipped**; re-evaluate when coverage grows past two seasons.
_Avoid_: Defence rating, defensive score (both imply a single number, which is exactly what this is not)

**Wheatcroft Principle**:
The idea that process-driven indicators from high-frequency match events (shots, shots on target, corners) are more accurate and consistent measures of team strength than raw goals, which are subject to chance. Derived from Wheatcroft's published research (LSE). Core feature engineering philosophy for this system — explains why shot and corner features dominate SHAP importance over goal-based features.
_Avoid_: N/A (unique term)

**League Position (seeding)**:
A team's rank in the league table at the point a fixture is played, used as a model feature (`Home_LeaguePosition`, `Away_LeaguePosition`, `LeaguePosition_Diff`). Mid-season it is the live points table (points, then goal difference, then goals scored). At **matchday 1, before any games are played, position is seeded by the previous season's outcome**, not left undefined: returning teams take their finishing position from last season; promoted teams are seeded at the bottom by promotion route — division-below **champions** at 3rd-from-bottom (22nd EFL / 18th PL), **runners-up** at 2nd-from-bottom (23rd EFL / 19th PL), **play-off winners** at the very bottom (24th EFL / 20th PL). The seed is also the natural tie-breaker once games begin (equal points fall back to seed order). See [docs/adr/0002-league-position-previous-season-seeding.md](docs/adr/0002-league-position-previous-season-seeding.md).
_Avoid_: Table position (ambiguous about timing — must specify "before this match"), form

**Division Movement (Promoted / Relegated)**:
Whether a team is **new to this division** for the current season, and from which direction. Two independent flags per side (`Home_Promoted`/`Away_Promoted`, `Home_Relegated`/`Away_Relegated`), because the two directions carry **opposite** strength signals: a side relegated from the Premier League is one of the *stronger* teams in the Championship, while a side promoted from League One is one of the *weakest*. Collapsing them into a single "new team" flag — or leaving relegated sides unflagged, so they read as ordinary returning teams — throws that away.

**Derived from the Canonical Datasets, never hand-maintained.** A team present in season N but absent in season N−1 is new to the division; if it appears in the division *above* in season N−1 it was relegated, otherwise it was promoted. Both leagues confirm the expected arrival counts every season (PL +3, EFL +6 = 3 down + 3 up), and the derivation reproduces the previously hand-listed seasons exactly. The hardcoded promoted-team lists are deleted: they went stale precisely because someone had to remember to extend them, and after season 25 nobody did — leaving the flag constant-zero across 24 of 26 PL seasons.

For the PL, `Relegated` is always 0 (there is no division above), so the two leagues keep an identical schema.
_Avoid_: New team, newcomer (both lose the direction, which is the whole signal)

**Promotion Route**:
*How* a promoted team came up — division-below **champion**, **runner-up**, or **play-off winner** — used by [ADR 0002](docs/adr/0002-league-position-previous-season-seeding.md) to seed matchday-1 League Position, since route is a real if weak strength signal (champions > runners-up > play-off winners on average).

Availability is asymmetric, and this is a **data-availability fallback, not a per-league formula**: PL's route is read off the EFL Canonical Dataset's final table for season N−1 (verified for all 25 seasons), whereas EFL's promoted teams come from League One, which this system does not hold. EFL therefore takes ADR 0002's neutral promoted seed. One code path, one documented fallback — deliberately not a second implementation, which is the failure mode this vocabulary exists to prevent.
_Avoid_: Promotion type, promotion method

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
The market's best estimate of true probability with the margin (overround) removed — always an estimate of *true price*, never the raw price you execute at. Source hierarchy: **(1) Pinnacle** (preferred — sharpest book, closest to efficient); **(2) the Exchange** de-vigged back/lay midpoint (Betfair/Matchbook) when Pinnacle is unavailable or stale — sharp, but note this is also the Execution Venue, so edge measured against it is structurally limited to what you can post above the midpoint (a circularity we accept only as a fallback); **(3) proportional de-vig of soft bookmaker odds** as a last resort, with edges discounted (80%) because the margin removal is less reliable.
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

**Execution Venue**:
Where a bet is actually placed and matched — kept conceptually distinct from the Fair Probability reference. Exchanges (Betfair, Matchbook) are the primary Execution Venue for scaling, because they do not limit winning bettors (they profit on commission regardless of who wins) and allow earlier market access. The exchange's price feeds `odds` → EV → Kelly, **net of commission** — it is *not* the Fair Probability source (except as the rank-2 fallback above), because grading a bet against the price you execute into is circular. Betfair has the deepest liquidity (matters for EFL/alt lines); Matchbook has lower commission but thinner liquidity. **Execution is advisory/manual**: the system computes Minimum Odds + Post Target and surfaces them on each Recommendation; the human places and manages the order on the Exchange. Automated order placement is explicitly out of scope — deferred behind the concurrency hardening (settlement punch-list item #5) and a future security review.
_Avoid_: Book (a Sharp/Soft book is a price reference; an Exchange is where you transact)

**Commission**:
The fee an Exchange charges on a winning bet — the price of un-limited execution. Modelled as a first-class, per-venue config constant applied to *net winnings* (reduces effective payout odds, not stake). Working rates: **Betfair 5%** (new account, no discount earned yet) and **Matchbook 2%**. Commission surfaces in exactly two places: **(1) pre-bet** it sets the per-venue **Minimum Odds** shown on the dashboard for *both* exchanges; **(2) post-bet** it is deducted from winnings when a **logged bet** is settled, using the rate of the venue actually chosen, with that rate snapshotted onto the settled row (immutable financial record). The advisory/paper track (`recommendations`, `predictions`) is *not* commission-netted — `logged_bets` is the net-of-fees source of truth for real P&L.
_Avoid_: Fee, vig/juice (those name the bookmaker overround, a different thing)

**Minimum Odds (break-even)**:
The commission-aware decimal odds at which a Recommendation's *net* EV equals zero, computed from the **blended probability** and the venue Commission: `O_min = 1 + (1 − p_blended) / (p_blended × (1 − c))`. A hard floor — never accept a price below it. **Computed per venue** (Betfair @ 5%, Matchbook @ 2%) and displayed for *both*, since you choose the venue offering the best price at bet time — the same bet has a higher floor on Betfair than Matchbook. It is an **execution floor for already-qualified Recommendations**, not a selection filter: the Edge and Agreement gates still decide whether a Pick becomes a Recommendation; Minimum Odds only governs the price at which you execute one.
_Avoid_: Break-even price (acceptable synonym), fair odds (those ignore commission and use fair_prob, not blended)

**Post Target (odds)**:
The odds at which a limit order is actually posted on the Exchange — strictly *above* the Minimum Odds, chosen to preserve the required edge net of commission after allowing for fill risk and adverse selection. You post at the Post Target; the Minimum Odds is only the line you never let a chase or partial fill drag you under.
_Avoid_: Ask price, lay price (lay is the opposite side of the exchange)

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
