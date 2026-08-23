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

**Squad Adjuster** (removed 2026-08-14):
A second-stage overlay that took the ensemble's base probability and adjusted it using player availability from the FPL API — a separate LogReg for injuries, suspensions and key absences, meaningful only for seasons 24+ because FPL data does not exist before that. **Deleted, because it was dead at both ends.** Its output `squad_adjuster.pkl` was rewritten every Sunday and loaded by nothing; it trained against `over_under_model.pkl` / `scaler.pkl` / `feature_list.pkl` dated 2026-05-04 — a legacy *single* model, not the current **Ensemble** — and had been failing outright since the 2026-08-03 publish retired columns that stale feature list still named.

The 16 `SQUAD_FEATURES` it consumed are **still computed by `pipeline.py` and still reach no model**: they are absent from `ALL_FEATURES` because they are NaN for every training season before 24, which is the exact constraint the adjuster existed to work around. Squad availability therefore has **no route into a Recommendation** today, and giving it one again would be a strategy change rather than a repair.

**This does orphan `api/player_features.py` entirely** — corrected 2026-08-14, an earlier revision of this entry claimed otherwise. That module produces *only* the 16 dead `SQUAD_FEATURES`, and its `compute_live_squad_features()` has had no callers since at least 2026-07-26. The four live `PLAYER_FEATURES` (`Home_InjuryBurden`, `Away_InjuryBurden`, `Home_KeyAbsences`, `Away_KeyAbsences`) come from a **different chain**: `api/fpl_historical.py` → `data/build_enriched_dataset.py` → the **Enriched Dataset**'s `home_injury_burden` columns → renamed by `pipeline.add_player_features`.

**And they are PL-only, not both leagues.** `CompleteDSChamp_enriched.csv` is configured in `league_config.py` but **does not exist on disk**, so the EFL takes the `else DATA_PATH` branch of `pipeline.py:32` and reads its canonical directly. Verified 2026-08-14: PL cache carries `Home_InjuryBurden` at 22.3% coverage; the EFL cache does not carry the column at all, and **zero** player features appear in its 127 `ou_features`. The two leagues therefore take *different branches of the same line of code* — which is why claims about "the pipeline" must always name the league.
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

**The division is verified on arrival, never assumed.** football-data.co.uk does not 404 a season it has not yet published — Apache's `mod_speling` redirects to the nearest filename it can find, so `E0.csv` for an unpublished season answers `301 → EC.csv` and serves the **National League** under a 200. Those rows are well-formed (real teams, real scores, every column the malformed-row filter checks), so nothing structural rejects them; only the `Div` column disagrees. Every download *and* every cached raw is therefore checked against the division that was requested, and a mismatch is discarded rather than cached — the season stays **absent**, which the **Freshness Gate** reports, instead of becoming quietly **wrong**, which nothing would. Confirmed live on 2026-08-14, when both 2026/27 files were still unpublished. Implemented as `_serves_division` in the shared builder.

**A new season must arrive in the same shape as the seasons already there, and only half of that is currently enforced.** `_map_columns` reads the raw feed two ways. Date, teams, goals and FTR are indexed directly (`df["FTHG"]`), so a rename or removal raises `KeyError` and the build stops — the loud, correct outcome. Every match stat and odds column is read with `df.get("HS")`, which returns `None` when absent; `pd.to_numeric` turns that into an **all-NaN column** and the build completes normally. A column football-data.co.uk renames between seasons therefore enters the canonical as silent nulls, and the models absorb them into confident-looking probabilities — the same "failure that looks like success" the **Freshness Gate** was written for, one layer earlier.

**So a season addition is only safe once its column coverage is compared against the seasons already present**: any column populated in recent complete seasons that arrives near-empty is a schema change, not a quiet season. The builder already verifies the division, the team count, and the promotion/relegation slots; coverage is the missing check, and it is the one that matters when the upstream format drifts. Verify it whenever a new season lands — see [[season-rollover-2026-27]].

A gitignored build artefact: regenerable, never hand-edited.
_Avoid_: Master file, raw data (the raw `E1_*.csv` season files are a distinct, upstream thing)

**Enriched Dataset**:
The **derived superset the models are actually trained on** — `CompleteDSPL_enriched.csv` — equal to the **Canonical Dataset** plus 8 enrichment columns (`home_xg`/`away_xg`, `home_injury_burden`/`away_injury_burden`, `home_key_absences`/`away_key_absences`, `home_squad_depth`/`away_squad_depth`). Built by `data/build_enriched_dataset.py` immediately after the canonical, from `api/fpl_historical.py`'s `team_injury_burden.csv`.

**`load_data()` prefers it, and this is intended** — `pipeline.py:32` reads `ENRICHED_DATA_PATH if os.path.exists(...) else DATA_PATH`. So "the pipeline reads the canonical" is *not* literally true, and reasoning about training data must name this file.

Verified 2026-08-14: 9,880 rows both, **75 canonical cols ⊂ 83 enriched cols, zero canonical columns dropped**, regenerated three minutes after the canonical in the same build (08:24:03 → 08:27:30). It is a strict superset kept in sync, **not** the stale rival artefact an earlier revision of this document described. That earlier claim — that the file had been superseded on 2026-07-25 and enrichment now lived as a column class inside the canonical — was **wrong about the code**: the fallback at `pipeline.py:32` was never removed and the file was never retired. Corrected here rather than in code, because the artefact is doing its job.

**Its rollover consequence is the part that bites.** A new season must land in **both** artefacts. If the canonical gains season 26 and the enriched rebuild does not run, `load_data()` returns a frame with **zero season-26 rows** — every temporal split boundary would then be dividing data that does not contain the season at all, and training would complete normally and write valid pickles. Same "failure that looks like success" family as the `301 → EC.csv` substitution.

Still a gitignored, regenerable build artefact: never hand-edited, and never a place to add a column that belongs in the canonical.
_Avoid_: Enriched dataset as *a synonym for* the Canonical Dataset (they are different files with different column counts), master file

**Data Refresh**:
Re-running the build → pipeline-cache → retrain chain so the models learn from newer match data, **without changing any strategy logic** (blend weights, agreement thresholds, Kelly fraction, DC parameters, model architecture all stay byte-for-byte identical). A data refresh re-fits learned parameters only; it is explicitly *not* a strategy change and does not require the same approval gate. The weekly Sunday retrain is an automated data refresh.
_Avoid_: Retrain (ambiguous — could imply architecture changes), update

**Freshness Gate**:
A hard precondition on both producing **Recommendations** and running a **Data Refresh** for a league: every fixture an **authoritative fixture list** reports as finished within a rolling window must be present in that league's **Canonical Dataset**. Reconciles against that list rather than a date heuristic, so "no fixtures were played" (international break, off-season) is distinguishable from "ingestion silently broke" — an ambiguity that previously required a manual off-season flag and allowed two months of staleness to pass unnoticed.

**Two ordered authorities, never a vote.** ESPN is asked first (it answers in each canonical's own name format, so the gate carries no team resolver); football-data.org is the fallback when ESPN cannot answer. Strictly ordered rather than cross-checked: a disagreement between two sources would need its own semantics, and both agree exactly where measured (2026-05-11→24: PL 22, EFL 3 finished, identical from both). ESPN is first despite being the feed **Settlement** also uses — the accepted cost of that shared dependency is that one CDN policy change can blind both at once, which is why the fallback is a *different* provider rather than a retry.

**The fetch must report three states, not two.** Fixtures-found, none-played, and *could-not-determine* are all different, and only the first two are answers. Collapsing the third into "none-played" turns an outage into the gate's own pass condition. This is not hypothetical: `fixture_schedule.py:74` and `api/espn_scores.py:170` both return an empty list on a network failure today, and the same substitution took down two sibling-project jobs on 2026-08-05 (see [[espn-user-agent-403]]).

**The window ends two days short of today — a publish grace.** The gate judges only fixtures the daily ingest has certainly had a chance to collect. ESPN flips a fixture to finished at full time, but the Canonical Dataset is rebuilt once a day at 06:00 ([ADR 0006](docs/adr/0006-task-scheduler-for-data-critical-jobs.md)), so between a Saturday evening kickoff finishing and the next morning's ingest a fixture is legitimately finished-and-absent. Judging it would block betting on every matchday evening — measured against the EFL final day, a gate run that evening would have demanded **12 fixtures that could not yet exist** in the canonical. Two days rather than one because the gate reasons in dates, not hours: a dashboard scan at 03:00 runs before that morning's ingest. The judged span stays 14 days; the grace shifts the window back rather than shortening it.

**The window is 14 days.** Sized to the *gate's own run cadence*, not to the fixture calendar — the window never needs to span a break, because zero finished fixtures in it is an unambiguous pass. Its only job is to keep the evidence of a failed ingest visible until the gate next runs. The binding cadence is the weekly Sunday Data Refresh, so 14 days is 2× margin: one missed Sunday still leaves a missing fixture in view, where a 7-day window would forgive it permanently. It also covers the modal 13-day international break (measured across PL/EFL seasons 23–25: typical largest in-season gap 12–13 days, worst 19), so a fixture played just before a break is still checked after it. Deliberately not wider: an unfixable upstream gap blocks betting for the window's length, and 30 days would quadruple that exposure for no detection benefit.

**Failure raises; it never returns empty.** A blocked gate raises `FreshnessError` — an empty recommendation list is indistinguishable from a quiet Tuesday, which is the same "couldn't determine rendered as nothing here" substitution found in three places on 2026-08-06 (`fixture_schedule.py:74`, `api/espn_scores.py:170`, and the sibling project's ingest). A result object callers must inspect is rejected for [ADR 0008](docs/adr/0008-one-team-resolution-contract-per-feed.md)'s reason: a contract every call site must remember is one that drifts, and the site that forgets is the site that bets.

Checked at **two boundaries, one implementation**: inside `generate_recommendations()` on both predictors (which covers every recommendation path at once), and before *every* `train()` — because the inner check fires after training, by which point stale data is already in the pickles. Note there are **three** train sites, not one: both leagues in `job_weekly_retrain`, *and* `scan.py:474`/`scan.py:487`, where a scan retrains inline whenever `load_trained_state()` fails. A matchday scan is therefore a Data Refresh in this sense, which ADR 0005 did not anticipate.

**A fixture is present or it is not — matched exactly on `(Date, Home_Team, Away_Team)`.** No date tolerance. ESPN reports kickoff in UTC and the canonical holds the UK local match date, and for English domestic football those dates cannot differ: kickoffs top out near 20:15 local against a UTC or UTC+1 offset, so crossing midnight would need a kickoff after 01:00 BST. Measured at 132/132 exact across a month of each league. Note this deliberately differs from the **League Split**, which *does* join `±1 day` — that tolerance is a property of the Betfair feed's divergent event dates, not a house style, and importing it here would mask a fixture ingested under the wrong date, which is a defect the strict key surfaces.

**No bypass, by design.** There is no override flag and no known-missing ledger: across seasons 16–25 both canonicals are complete (380 PL / 552 EFL per season, zero missing scores), so an unfixable upstream gap has no precedent to justify a mechanism. A bypass boolean is specifically rejected — a disabled safety gate is a thing that stays disabled, flipped during one bad Saturday and still set in November. Should a permanent gap ever appear, the established pattern to copy is a ledger strict in *both* directions (`PENDING_NEW_COLUMNS`, `_KNOWN_DIVERGENCES`), so a fixture that later arrives upstream forces its entry's removal. In exchange, the gate **must name every missing fixture** — date and both teams, never a bare count — because with no bypass the message is the only route to action.

**The off-season retrain flags stay.** `PL_RETRAIN_ENABLED` / `EFL_RETRAIN_ENABLED` are *not* retired by this gate, contrary to [ADR 0005](docs/adr/0005-freshness-gate.md)'s original consequence. The two ask different questions: the gate asks "is anything **missing**?" and passes in the off-season (nothing finished, nothing to miss); the flags ask "is there anything **new**?" and skip. Deleting them would not automate the pause, it would remove it, starting a full two-league retrain every off-season Sunday on data that has not moved. Their `config.py` rationale is half-true — training is seeded throughout (`random_state=42`), so an identical-input retrain is deterministic and drift is *not* a risk; wasted CPU on a job that rewrites the live pickles is the real cost. The check that would genuinely retire them is "did the Canonical Dataset gain rows since the last retrain?", which needs no calendar knowledge — deliberately left as separate future work rather than shipped alongside the gate.

**Could-not-determine blocks. There is no date-heuristic fallback.** When neither authority can answer, the gate is `UNKNOWN` and `UNKNOWN` fails closed — deliberately *not* falling back to `max(Date) >= today − N` on the canonical. Such a heuristic cannot answer the gate's question (an old `max(Date)` is precisely the off-season-versus-dead-ingest ambiguity the gate exists to resolve), and `max(Date)` is not even trustworthy on these files: `betfair_goal_ou.csv` carries 8 Betfair sandbox rows dated **2030**, which would make a max-date check pass forever. A second, weaker definition of freshness is a liability, not a safety net. Accepted cost: a simultaneous outage at two independent providers stops betting for that session.

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

**Training Path (Production vs Research)**:
Two different code paths train models in this repo, they **derive their season boundaries differently**, and confusing them has already produced three wrong documents. Any statement about "what the model trains on" must say which path it means.

| | **Production Path** | **Research Path** |
|---|---|---|
| PL | `predict.py:320 train()` | `model.py:1749 main()` |
| EFL | `championship_predict.py:533 train()` | `championship_model.py:664 main()` |
| Invoked by | `scheduler.py:217` weekly retrain; `scan.py:474/487` inline retrain | a human typing `python model.py` |
| Season boundaries | **derived from the data** — PL `SeasonIndex >= 14`, EFL `>= 0`; early-stopping validation season is `max(SeasonIndex)` present | **hardcoded constants** — `config.py:37-39` `TRAIN`/`VAL`/`TEST_SEASONS` (PL), `championship_model.py:79 TEST_SEASON` (EFL) |
| Held-out test season | **none** | yes |
| Writes | `pl_trained_state.pkl` / `efl_trained_state.pkl` — **the pickles that price live bets** | `over_under_model.pkl` and console metrics |

**The Production Path never reads the split constants.** `TRAIN_SEASONS`, `VAL_SEASONS`, `TEST_SEASONS` and `TEST_SEASON` govern the Research Path only. A change to them alters backtest and evaluation output; it does **not** change what the live models learn. See [ADR 0009](docs/adr/0009-one-season-boundary-contract-per-training-path.md).

**The Research Path's allowlist/denylist mismatch** (verified 2026-08-14): `pipeline.py:1554` partitions by `isin(TRAIN)/isin(VAL)/isin(TEST)` — an *allowlist*, so a season named in none is dropped — while `model.py:1769` selects `>= TRAIN_MIN_SEASON & ~isin(TEST_SEASONS)` — a *denylist*, which includes it. Both run in the same job on the same data. With data past `TEST_SEASONS`, `walk_forward_cv` (whose folds come from `max(SeasonIndex)`, not from config) then trains a fold whose window has the test season punched out of the middle. `championship_model.py` avoids the *mismatch* by partitioning one scalar with `<` and `==`, but not the *staleness* — any season above `TEST_SEASON` still falls outside both.

**Early-Stopping Season**:
The single season the Production Path holds back to decide how many trees XGBoost/LightGBM should add. **The most recent season holding at least 50 fixtures** — not simply the latest season present. The same eligible list, taking its last two, sets the **Base Rate** window.

The threshold exists because the season is derived from the data, not configured: before it, `train_seasons[-1]` made a newly-started season the Early-Stopping Season **on its first ingested fixture**, so a dozen August matches would decide the tree count, and the Base Rate would slide from two complete seasons (760 PL matches) to one-plus-a-fragment (392) — halving its sample and dropping a whole season, silently, on the first weekly retrain after the new season's rows landed. No config change or deploy was needed to trigger it.

50 is not a new number: `walk_forward_cv` already skips any fold whose validation season holds fewer (`model.py:1280`). Implemented as `predictor_utils.seasons_for_validation`, shared by both leagues. Where no season clears the bar — reachable only on tiny or synthetic datasets — it falls back to every season present and logs a warning rather than raising, so small-data experiments still train.

**The Base Rate follows the same threshold, deliberately.** This document defines the Base Rate as the stable historical calibration anchor and **Regime** as the mechanism that tracks in-season drift. Letting a fragment of the current season into the anchor does Regime's job badly and the anchor's job worse.

**Early stopping is followed by a refit.** XGB and LGB are refit on the full training frame at the tree count early stopping chose (`predictor_utils.refit_at_best_iteration`). Without it, guarding the season choice would have made staleness permanent: XGB and LGB were previously *kept* as fitted on everything-except-the-Early-Stopping-Season, while LogReg and Dixon-Coles — fitted two lines below on the full frame — were not. See [ADR 0009](docs/adr/0009-one-season-boundary-contract-per-training-path.md).
_Avoid_: Validation season (ambiguous — the Research Path's `VAL_SEASONS` is a different, configured thing), holdout

**Walk-Forward (CV)**:
The cross-validation methodology used throughout this system. Train on all seasons up to N, validate on season N+1, slide forward and repeat. Prevents temporal leakage (seeing future data when predicting past matches). Produces out-of-fold predictions for each validation season which become training data for the stacker.
_Avoid_: K-Fold (explicitly rejected — shuffles time and leaks), backtesting (overlapping but different concept)

**OOF (Out-of-Fold)**:
Predictions generated by a base model on data it was NOT trained on (its walk-forward validation season). These are collected across all folds and used to train the stacker — ensuring the stacker never sees predictions on data the base models memorised.
_Avoid_: Holdout predictions, test predictions

**OOF Cache**:
A stored table of OOF predictions for one (league, **Market**) cell, one row per **Fixture**, holding each **Base Model**'s probability, both **Sides**' odds with the book named, and the outcome. Six exist, covering PL and EFL × O/U 2.5, O/U 1.5 and BTTS. Written by `scripts/generate_oof_cache.py`, read by `scripts/roi_validate.py`.

**Its defining property is that rows are pre-gate.** The runners in `backtest.py` and its siblings emit only bets that already passed minimum edge, positive EV and `min_agree`, so the agreement levels the gate *rejects* leave no trace. The OOF Cache stores every fixture the model priced, so the gate can be applied after the fact and moved. That makes it the substrate for any question about *where a threshold should sit* rather than *how the current threshold performed* — see [ADR 0010](docs/adr/0010-agreement-evidence-from-the-oof-cache.md).

**It does not go stale on a Data Refresh.** The walk-forward path fits its own models per season and never loads the production pickles, so a retrain leaves the cache's meaning unchanged. What invalidates it is a **Canonical Dataset** change, a change to staked config, or a code change.
_Avoid_: OOF parquet (implementation), prediction cache (ambiguous — the pipeline cache is a different artefact)

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
A team's rank in the league table at the point a fixture is played, used as a model feature (`Home_LeaguePosition`, `Away_LeaguePosition`, `LeaguePosition_Diff`). Mid-season it is the live points table (points, then goal difference, then goals scored). At **matchday 1, before any games are played, position is seeded by the previous season's outcome**, not left undefined: returning teams take their finishing position from last season; promoted teams are seeded at the bottom by promotion route — division-below **champions** at 3rd-from-bottom (22nd EFL / 18th PL), **runners-up** at 2nd-from-bottom (23rd EFL / 19th PL), **play-off winners** at the very bottom (24th EFL / 20th PL). Sides **relegated into the EFL seed 1, 2, 3 in their PL finishing order** — the division's strongest priors — and arrivals whose route is unknowable (League One promotions; this system holds no L1 table) take the **neutral 2nd-from-bottom seed**. Seeds may collide (a returning 3rd-place play-off loser and the third relegated side both seed 3); collisions break alphabetically, as does the whole table in the dataset's first season, where there is nothing to seed from. The seed is also the tie-breaker once games begin (equal points, GD and goals scored fall back to seed order, replacing the alphabet). Implemented in `_add_league_position` / `_matchday1_seeds` in the shared builder, 2026-08-03. See [docs/adr/0002-league-position-previous-season-seeding.md](docs/adr/0002-league-position-previous-season-seeding.md).
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

**Division Movement Seed**:
The rolling-feature values a side carries in this division **before it has history here** — the answer to "what does this team look like when it has not played in this division yet?". Distinct from **League Position (seeding)**, which seeds one feature (rank); this seeds the *form* features (`Past5Goals`, `GoalDiff_5`, `Over25_5`, the shot and corner rollings) that have no value at all for a side new to the division.

**A side's own history in this division is never the seed.** Movement is a cycle: the only route to the PL is promotion *out of* the EFL, and the only route to League One is relegation *out of* it. So a returning side's last EFL season is always its **exit** season, and the exit direction is always the opposite of the return direction — every PL-relegated side left in positions 1–3, every L1-promoted side left in 23–24. Own-history is biased by construction, and stale on top: Cardiff's last EFL season ended 24th with an unobserved League One season since, Wolves' promotion form is eight seasons old. The reference cohort is both less biased and lower variance (five teams, not one).

**The seed is the cohort for that side's route.** League One arrivals take the **bottom-5 cohort** of the prior EFL season; sides relegated from the PL take the **mid-table (8–16) cohort**. The two directions carry opposite strength signals, so one cohort for both — which is what the live path's league median amounted to — is the defect, not the simplification.

A relegated side's **own PL form** was expected to improve on its cohort, since we hold a complete current PL season for every such side and they are demonstrably not interchangeable. It was measured and it does not, at least not detectably: rebased onto the EFL scale via the **Scoring Index** mechanism and blended at a fitted weight, `w` estimates 0.317 with a 95% interval of [−0.22, 0.85] across all 75 relegation events. The signal is real but weak (correlation 0.145, RMSE 0.7341 against the cohort's 0.7404) and 75 events cannot separate a 0.8% gain from nothing — the rebased rate and the cohort rate average 2.596 against 2.561, so there is barely any spread to fit against. **The blend is not in the seed.** `scripts/measure_seed_weight.py` keeps the result rerunnable as events accrue, three a season.

**The seed governs Dixon-Coles too, in its own parameter space.** XGB and LightGBM consume a feature row, so the seed reaches them by filling that row. DC consumes *team identity* and looks up venue-specific attack/defence ratings, so the row never reaches it — and left alone it rates a returning side on that side's **exit season**, because `_decay_weights` decays by position in the team's own match sequence, not by calendar date. An eight-season absence is invisible to the weighting, and `_shrink_to_league` keys on match *count*, so a long-absent side is rated with near-total confidence: Wolves carry a title-winning 2017/18 rating (`attack_home` 1.247, `defence_home` 0.704) against an actual current side's 0.618 / 0.907.

A side is therefore treated as **unrated** whenever Division Movement says it is new to the division — absent in season N−1 — regardless of how long the gap is. The cut is not a staleness threshold to tune: because movement is a cycle, *any* absence means the newest data is exit data, so Burnley's one-season gap is contaminated in exactly the way Wolves' eight-season gap is, only less obviously. An unrated side falls back to the venue-aware prior for **its route** — the single `PRIORS` bucket, calibrated for weak arrivals, is the same one-bucket-for-both-routes defect this seed exists to remove. **Both route priors are measured, not chosen**, from what arriving sides of that route actually did in their first five matches across the 150 historical arrival events, walk-forward as with `w`. This retires four hand-picked constants for the same reason ADR 0002 decision 10 deleted the hand-maintained promoted-team lists: a constant nobody re-derives is a constant that silently goes stale.

**One concept, one implementation, two callers.** The seed is consumed at two moments — when the pipeline builds training rows, and when the predictor builds a feature row for an unplayed fixture — and those must be the *same* seed. They are two callers of one definition, never two definitions. This is the Promotion Route principle applied one layer down, and ADR 0007's contract applied to a feature's *value* rather than its formula: `Home_Past5Goals` cannot mean one quantity in training and another at predict time.

**The seed is per league, and the two leagues split on different axes.** The EFL splits arrivals by **Arrival Direction** — relegated or promoted — because both directions exist there and carry opposite strength signals. Nobody is relegated into the PL, so that axis collapses to a single bucket. The finer **Promotion Route** axis (champion / runner-up / play-off winner) *is* available for the PL and not for the EFL, the exact inverse: PL route is read off the sibling EFL final table, while EFL arrivals come from a League One table this system does not hold. **Availability is inverted, so the seed's shape is not symmetric between the leagues** — and the PL still takes one bucket, because 75 events split three ways is 19 / 30 / 26 against a 30-event guard, and walk-forward leaves early folds in single digits. Recorded rather than assumed, so it is re-decidable as events accrue.

**"Route" now names two different axes** — Arrival Direction (ADR 0011) and Promotion Route (ADR 0002). They are not interchangeable, and the collision is live in the code, where `division_movement.arrival_route` means the former. Disambiguation is deferred, not resolved: renaming would touch shipped EFL code.

See [ADR 0011](docs/adr/0011-one-division-movement-seed-per-arrival.md) for the EFL and [ADR 0012](docs/adr/0012-division-movement-seed-for-the-premier-league.md) for the PL.
_Avoid_: Synthesis, synthesised row (describes the mechanism at one call site, not the concept — and the name hid that two call sites disagreed), promoted-team defaults; "route" unqualified, now that it names two axes

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

**What Agreement is worth is measured against Realised Edge, it lives entirely at unanimity, and it exists in only two of six cells.** Measured on the April 2026 **OOF Caches**; `PL ou25` was regenerated on current models on 2026-08-15 and moved 11bp (+4.33% → +4.22%), so the April figures describe today's models adequately.

| Cell | Unanimity rows | Realised Edge (pre-gate) |
|---|---|---|
| PL O/U 2.5 | 726 | **+4.33%** (CI +0.64% to +7.68%) |
| PL BTTS | 616 | **+5.72%** (CI +1.85% to +9.63%) |
| PL O/U 1.5 | 2280 | +0.49% (CI −1.18% to +2.21%) |
| EFL O/U 2.5 | 1687 | +0.43% (CI −1.98% to +2.77%) |
| EFL BTTS | 1634 | +0.62% (CI −1.91% to +2.93%) |
| EFL O/U 1.5 | 3301 | +0.49% (CI −1.04% to +1.97%) |

**The four nulls are well-powered, not thin.** EFL O/U 1.5 is the largest sample in the study at 3,301 unanimous rows and returns +0.49%; PL O/U 1.5 is the largest PL sample at 2,280 and returns the same. Where Agreement does not predict, it is not because we cannot see — it is because there is nothing there. Below unanimity there is nothing anywhere: PL 3/4 realises **−1.61%**.

**Those figures count unanimity on either Side, and pooling the Sides conceals that the effect is one-directional.** Split by Side — which is what the pre-gate table does, keeping one Side per **Fixture** — the picture changes materially. Three bins have intervals excluding zero, and **every one is a bin-0: unanimous opposition, all negative.**

| Cell | Bin | Rows | Realised Edge | 95% CI |
|---|---|---|---|---|
| PL O/U 2.5 | 0 (none back Over) | 392 | **−5.85%** | −10.86% to −1.02% |
| PL BTTS | 0 | 481 | **−6.18%** | −10.56% to −1.76% |
| EFL O/U 1.5 | 0 | 1150 | **−2.54%** | −5.18% to −0.03% |
| PL O/U 2.5 | 4 (all back Over) | 334 | +2.56% | −2.63% to +8.04% |
| PL BTTS | 4 | 135 | +4.10% | −3.64% to +12.35% |

**No unanimous-*support* bin clears zero.** The pooled +4.33% for PL O/U 2.5 was 334 rows of *all back Over* (+2.56%, not significant) averaged with 392 rows of *all back Under* (+5.85%, significant) — a real signal and a non-signal reported as their mean. EFL O/U 1.5 has the same shape: backing Over on unanimous support realises **−0.61%**, while the opposite direction realises +2.54%.

**So the ensemble is dependable about what not to back, and not yet demonstrably dependable about what to back.** Any presentation that pools Sides destroys this distinction by construction, because it forces the `0` and `N` bins to be exact negatives.

**Never pool cells to compute this.** Realised Edge is base-rate neutral so pooling is arithmetically legal, but PL's three markets pooled give +2.15% — two real signals diluted by one null — which reads as a weak system-wide effect instead of a strong two-market one.

**Claimed Edge exceeded Realised Edge in every gated bin, by more as agreement fell.** The bins claiming the most were not the bins earning the most: the signature of upward-miscalibrated ensemble confidence rather than of a working threshold.

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

**Historical Odds (canonical)**:
The price a *backtest* settles against, held in both Canonical Datasets as `Odds_Over_{line}` / `Odds_Under_{line}` for the 1.5 and 2.5 goal lines, with `Odds_Source_{line}` naming its origin per row — `betfair`, `b365`, or null.

**Betfair first, Bet365 only as a stand-in.** The exchange is the Execution Venue (see [ADR 0003](docs/adr/0003-exchange-execution-and-commission.md)), so its price is what a backtest should measure against. But Betfair's history begins **2016-08-01** while the canonicals begin 2000/01, so roughly 55% of rows carry a soft-book price instead. Those are *different kinds of number* — a soft book's margin is wider and its price is not one you would execute at — which is exactly why the source column exists rather than a single silently-mixed column. O/U **1.5 has no fallback**: football-data.co.uk serves no such line, so it is the exchange price or nothing.

**Always the first traded price, never the last.** Betfair's `over_ltp` is the *last* traded price and is contaminated by in-play trading — across 66k settled O/U 2.5 markets its median is 1.53 when the over won and 15.00 when it lost. A pre-match price cannot know the result; only `*_ltp_first` is safe. Women's, youth and reserve fixtures are excluded explicitly: Betfair carries them under near-identical names on the same day.

Note this is **not** the Fair Probability source — that stays independent (see below), because grading a bet against the venue you trade on erases edge by construction.
_Avoid_: "B365 odds" (the columns are no longer Bet365-only), closing price

**Edge**:
The raw probability gap between the model's estimate and the market's fair probability. Computed as `blended_prob - fair_prob`. A positive edge means the model thinks the market is underpricing that outcome.
_Avoid_: Advantage, overlay

**EV (Expected Value)**:
The monetary expression of edge — what you'd expect to profit per unit staked. Computed as `blended_prob * odds - 1`. Positive EV means the bet is profitable in expectation.
_Avoid_: Expected return, profit margin

**Realised Edge**:
What the edge turned out to be worth, over a *set* of bets: `mean(won) − mean(fair_prob)`. **Edge** (above) is the claim the model makes before the match; Realised Edge is the claim graded against outcomes. Use them as a contrasting pair — quoting one without the other is how a market that only ever *claims* edge passes for one that earns it.

**Its purpose is comparison across markets, which hit rate cannot do.** Hit rate is dominated by the market's **Base Rate**: PL O/U 1.5 Over wins ~75% of the time and EFL O/U 3.5 ~35%, so a 75% hit rate on the former is the coin landing as expected, not skill. Subtracting the Fair Probability removes exactly that — an unskilled bet scores ~0 in any market, so bets on different Lines, Markets and leagues become comparable. It is also not an ROI estimand, so it is unaffected by the historical-`_first`-price vs best-of-14-books mismatch that makes backtest and live ROI unpoolable.

**Defined only over a set, never a single bet.** For one bet `won − fair_prob` is just a noisy Bernoulli residual. It means something once averaged.
_Avoid_: Actual edge, true edge (implies the model's estimate was wrong rather than untested), alpha

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
- The **Ensemble** produces a model probability per **Fixture** × **Market** (nothing modifies it downstream — the **Squad Adjuster** that once could was removed 2026-08-14)
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
- **"The Canonical Dataset is for training" — the intent, but not what the code does.** The **Canonical Dataset** is meant to serve training only, with live fixtures ingested from the APIs (The-Odds-API for the fixtures being priced, ESPN for finished results and settlement). In practice it is also a **live input to every Recommendation**: `predict.py:296` and `championship_predict.py:499` call `run_pipeline()` inside `load_data()`, so rolling form, league position and the rest of the feature set are recomputed from the canonical at prediction time. That is why the **Freshness Gate** blocks Recommendations and not merely a **Data Refresh**, and why a canonical lagging upstream publication stops live betting. **Do not narrow the gate to the retrain path on the strength of the intent** — the coupling is real until live feature computation is fed from the live results feed instead, which is unbuilt and would need its own ADR.
- **"The EFL O/U 2.5 ROI" — resolved, but only by naming the method.** Three numbers describe this one market and all three are real: **+1.20%** (`config.py:764`, `PHASE_4A_BASELINE_ROI`, simulated from the historical OOF cache by `scripts/run_phase4a_matrix.py`, validated April 2026) — this is the figure ADR 0003 cites and the basis of the **Monitored Market** demotion; **+5.6% on 874 bets** (`config.py:498`, Dixon-Coles alone via `efl_alt_lines_backtest.py`, exploratory config); and a walk-forward figure from the **EFL Ensemble** (`championship_backtest.py`) that is not recorded anywhere. Separately, **EFL O/U 3.5 is also +1.2% (857 bets)** on the DC runner — a coincidence of value between two different markets that makes misquotation easy. **Always name the method when quoting a per-market ROI: OOF simulation, walk-forward ensemble, or DC alone.** They are not interchangeable and they disagree by up to 4.4pp on the same market.
