# System Overview — The Whole Project in Plain English

_A complete, simple-terms walkthrough of everything this project does: where the
data comes from, how it becomes predictions, the maths behind the models, how a
prediction becomes a bet recommendation, and how the system runs and grades
itself. Written 2026-07-04. For precise vocabulary, see `CONTEXT.md`; for
decisions and their reasons, see `docs/adr/`._

---

## 1. What this system is, in one paragraph

This is a statistical betting system for English football (Premier League and
EFL Championship). It builds its own estimate of how likely certain match
outcomes are — will there be over 2.5 goals? will both teams score? — using ten
years of historical data and four different statistical models whose opinions
are combined into one. It then compares its probability against the
probabilities implied by bookmaker odds. When the model believes an outcome is
meaningfully **more likely than the market's price says**, that's an "edge" —
and the system recommends a bet, sized by a formula (Kelly) that balances
growth against risk. Every recommendation is recorded, settled against the real
result, and fed into performance analytics, so the system continuously answers
the only question that matters: *is the edge real?*

**The core idea in one sentence:** find the gaps between what the model thinks
and what the bookmakers charge, and only bet into those gaps.

---

## 2. The whole journey, end to end

```
 DATA IN                 LEARNING                 DECIDING                RUNNING
┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌─────────────────┐
│ Match results │   │ Feature pipeline  │   │ Live odds fetched   │   │ Scheduler fires │
│ xG / shots    │──▶│ (150+ features)   │──▶│ Model prob vs       │──▶│ scans on match- │
│ Player data   │   │        │          │   │ market fair prob    │   │ days, retrains  │
│ Weather       │   │        ▼          │   │        │            │   │ weekly, settles │
│ Betfair hist. │   │ 4-model ensemble  │   │        ▼            │   │ bets daily      │
│ Bookmaker odds│   │ → one calibrated  │   │ Edge → filters →    │   │        │        │
└──────────────┘   │   probability     │   │ Kelly stake →       │   │        ▼        │
                    └──────────────────┘   │ RECOMMENDATION      │   │ Dashboard shows │
                                           └────────────────────┘   │ picks & results │
                                                                     └─────────────────┘
```

Six stages: **ingest data → engineer features → model probabilities → compare
to odds & recommend → operate automatically → settle and analyse.** Each is a
section below.

---

## 3. Stage 1 — Data ingestion (what comes in, from where)

The system pulls from many sources because no single source has everything.
Each source has one job:

| Source | What it provides | Used for |
|---|---|---|
| **football-data.co.uk CSVs** | Historical match results (2000→now): scores, shots, corners, cards | The training backbone — the "Canonical Dataset" |
| **football-data.org API** | Recent match results | Keeping results current between CSV updates |
| **Understat (scraper)** | Expected goals (xG), shot quality, tactical stats | Richer PL features (PL only) |
| **FPL API** | Team strengths, player availability, injuries | PL features + live squad news |
| **FPL-Core-Insights (GitHub)** | Per-player match stats (xG, xA, tackles…) | The Squad Adjuster (2024-25 onwards only) |
| **Open-Meteo** | Match-day weather | Minor features (wind/rain suppress goals slightly) |
| **The-Odds-API** | Live bookmaker odds: O/U 2.5 + alt lines + BTTS from 14+ books | The "what does the market charge?" side |
| **OddsPapi** | Live odds from 100+ books incl. Pinnacle (the sharpest) and Betfair exchange | Sharp reference prices, alt lines, Asian handicap |
| **ESPN scoreboard API** | Final scores (free, no key) | Settling bets after matches finish |
| **Betfair Historical Data** | Exchange closing/opening prices back to 2016 | Backtesting against real execution prices (see `docs/betfair_ingestion_scope.md`) |

Two important disciplines apply everywhere:

- **Team-name normalisation.** Every source spells teams differently ("Man Utd"
  / "Manchester United" / "Manchester United FC"). Explicit mapping tables
  convert everything to one canonical form. This is historically the #1 source
  of silent bugs, so mappings are explicit allowlists, not fuzzy guesses.
- **Quota care.** The odds APIs have small free-tier quotas (The-Odds-API
  500/month, OddsPapi 250/month). Every response is cached (30-min TTL),
  stale caches are served if an API fails, a client-side counter tracks
  OddsPapi usage (it sends no quota header), and a guardrail refuses calls
  near the cap. The scheduler is fixture-aware so it only spends calls on
  actual matchdays (~150–200 calls/month total).

---

## 4. Stage 2 — Feature engineering (turning results into model food)

Models can't eat raw scorelines; they need numbers that describe *how good each
team is right now*. The pipelines (`pipeline.py` for PL — 150+ features;
`championship_pipeline.py` for EFL — 80+) compute, for every historical match,
things like:

- **Rolling form**: goals, shots, shots-on-target over the last 5/10/20 games
- **Attack/defence strength**: how a team scores/concedes relative to league average
- **Conversion rates**: goals per shot, per shot-on-target (5- and 20-game windows)
- **xG-based measures**: expected goals for/against (PL, from Understat)
- **Congestion & discipline**: days of rest, fixture pile-ups, card rates
- **Context flags**: promoted team, derby match, league position (seeded from
  last season's finish at matchday 1 — see ADR 0002), Elo ratings (EFL)
- **Weather**: wind and rain on the day

**The guiding philosophy (the "Wheatcroft Principle"):** shots and corners are
*better* predictors than goals themselves. Goals are rare and lucky; a team
creating 15 chances per game is genuinely strong even if it lost 1-0 last week.
High-frequency process stats carry more signal than low-frequency outcomes —
and indeed shot/corner features dominate the model's importance rankings.

**One deliberate exclusion:** bookmaker odds are **never** model features. The
whole design is "model vs market" — if the market's opinion leaked into the
model, comparing the two would be circular.

Everything trains from one file per league (the **Canonical Dataset**:
`CompleteDS_CSV.csv` / `CompleteDSChamp_CSV.csv`) — facts copied verbatim from
source, features regenerated by the build script, never hand-edited.

---

## 5. Stage 3 — The models (the maths, in simple terms)

### 5a. Dixon-Coles: the football-specific statistician

Goals in football are well described by a **Poisson distribution** — the maths
of "rare events at a steady rate." If a team is expected to score 1.5 goals
(its **lambda**, λ), Poisson tells you the exact probability it scores 0, 1, 2,
3… So the model's real job is estimating **two lambdas per fixture** (home and
away), from attack strength × opponent's defence weakness × home advantage,
with recent games weighted more heavily (time decay).

With both lambdas you can price *any* goals market by adding up score
probabilities: P(over 2.5) is the sum of every scoreline with 3+ goals;
P(BTTS) is the sum of every scoreline where both sides score.

**The Dixon-Coles twist (tau, τ):** real football has slightly *more* 0-0 and
1-1 draws than independent Poisson predicts (teams shut up shop at low scores
— the two teams' goals aren't quite independent). τ nudges the four low-score
cells (0-0, 1-0, 0-1, 1-1) to fix this — without it, 0-0s are underestimated
by ~15%.

Dixon-Coles is the **strongest single model** here (test AUC ~0.60, beating
the machine-learning models individually) and it alone prices the alternative
lines (O/U 1.5, 3.5) since it produces a full scoreline distribution.

### 5b. XGBoost & LightGBM: the pattern hunters

Two gradient-boosted decision tree models. Intuition: build a small decision
tree that predicts over/under crudely; look at what it got wrong; build a
second tree that corrects those mistakes; repeat hundreds of times. The final
prediction is the sum of all corrections. These models excel at **interactions**
a formula would miss — e.g. "high shot volume matters *more* when the opponent
defends deep *and* it's not raining." They eat all 150+ features. Individually
mediocre (AUC ~0.53–0.55); valuable because their errors differ from DC's.

### 5c. Logistic Regression: the straight-line baseline

The simplest model: a weighted sum of features pushed through an S-curve to
output a probability. It can't learn interactions, which makes it a stable,
hard-to-fool baseline. PL only — the Championship has too little rich data for
it to add value, so the EFL ensemble is 3-model. (A guard clips its scaled
inputs to ±5 standard deviations, after one feature once drifted to +257 σ and
produced nonsense.)

### 5d. Stacking: combining four opinions into one

Rather than hand-picking weights, a **stacker** (a small logistic regression)
*learns* how much to trust each model. Crucial detail — it's trained on
**out-of-fold (OOF)** predictions: each base model predicting seasons it was
NOT trained on. Otherwise the stacker would reward whichever model memorised
the training data best. Learned weights: XGB ≈ 2.2, LGB ≈ 1.2, DC ≈ 1.1.

### 5e. Walk-forward validation: never peek at the future

All validation slides forward in time: train on seasons up to N, test on
season N+1, slide, repeat. Ordinary shuffled cross-validation would let the
model "see" 2024 while predicting 2019 — everything would look better than it
really is. Football also drifts across eras (tactics, rules, xG revolution),
and walk-forward measures performance under that drift honestly.

### 5f. Calibration: making 60% mean 60%

A model can *rank* matches well yet output probabilities that are too hot or
too cold. Two corrections:

- **Platt scaling** — a learned squash/stretch of the stacker's outputs so
  that, historically, matches given 60% actually went over ~60% of the time.
- **Regime detection** — if the current season is running unusually
  high/low-scoring vs the ~52% historical over-2.5 base rate, calibration
  shifts mid-season. (O/U only; BTTS rates are stable across seasons.)

Known honest caveat: test-set probabilities still drift ~3–6% hot. Rankings
are reliable; edges are computed *relatively*, so edge detection survives this
— but it's why conservatism layers (below) exist.

### 5g. The Squad Adjuster: injury news, bolted on

Player-level data only exists from 2024-25, so it can't train inside the main
model (it would be blank for 90% of history). Instead a small second-stage
model takes the ensemble's probability + 16 squad-availability features (key
absences, missing attack/defence strength from FPL data) and nudges the
probability. Worth ~+1.7pp accuracy; will strengthen as seasons of player data
accumulate.

**Bottom line of Stage 3:** for each fixture and market, one calibrated
probability — e.g. *"Arsenal v Chelsea: 58.3% over 2.5 goals."*

---

## 6. Stage 4 — From probability to bet (the betting maths)

### 6a. What the market thinks: implied probability and the de-vig

Decimal odds of 2.00 imply 1/2.00 = 50%. But bookmakers over-charge: a
two-sided market's implied probabilities sum to ~104–108%, not 100% — the
excess is their margin (the "vig"). **De-vigging** strips it (proportionally
rescale so the sides sum to 100%) to recover the market's true opinion, the
**fair probability**.

Whose odds count as "the market's opinion"? A strict hierarchy:

1. **Pinnacle** — the sharp book. Low margin, welcomes winners, so its line
   reflects informed money. Best available proxy for the true probability.
2. **The exchange** (Betfair) de-vigged midpoint — sharp, but it's also where
   we'd execute, so grading against it is slightly circular; fallback only.
3. **De-vigged soft books** — least reliable; edges found this way are
   **discounted by 20%** (`DEVIG_DISCOUNT = 0.80`) as a penalty for the noise.

### 6b. Blend: institutionalised humility

The system does **not** bet its raw model output. It blends:

> **blended probability = 35% × model + 65% × market fair probability**

Why trust the market more than your own model? Because the market is very
efficient and the model is imperfect — this blend means the model only ever
disagrees with the market *at the margin*. It's the single biggest conservatism
device in the system. (First ~4 gameweeks of a season: even more conservative —
20% model weight, smaller stakes, higher edge bar — because early-season form
data is thin.)

### 6c. Edge and EV

- **Edge** = blended probability − fair probability. *"I think it's 55%, the
  market prices 52% → edge = +3%."*
- **EV (expected value)** = blended prob × decimal odds − 1. The profit per £1
  staked *if the blended probability is right*. Negative-EV bets are never
  recommended, full stop.

### 6d. The filters: Pick → Recommendation

Positive edge alone isn't enough (that's just a **Pick**). To become a
**Recommendation** (a bet the system actually advises):

- **Minimum edge** must be cleared (per-market thresholds; ~2%+),
- **Agreement**: at least **2 of the base models** must independently land on
  the same side of the market's fair price. A simple headcount that guards
  against one model going rogue — and it also scales the stake (2 agree →
  0.7×, all 4 agree → 1.1×).

### 6e. Kelly staking: how much to bet

The **Kelly Criterion** computes the stake fraction that maximises long-run
bankroll growth given your edge and odds — bigger edge and shorter odds →
bigger stake. Full Kelly is famously too aggressive when your edge estimate is
uncertain (and every model's is), so the system stakes **quarter-Kelly**
(0.25×), further shaped by:

- **Market multiplier** — markets with more proven historical edge get fuller
  stakes (O/U 2.5 Over 1.0×; weaker markets down to 0.5×),
- **Agreement scaling** (above),
- **Drawdown protection** — if the bankroll falls into a losing streak, stakes
  shrink automatically (e.g. halved beyond 15% drawdown). You can't go broke
  during the cold streak every strategy eventually hits,
- **A hard per-bet cap** on stake percentage.

Typical output: *"stake 2.8% of bankroll."*

### 6f. Which markets, exactly

| Market | Priced by | Status |
|---|---|---|
| PL O/U 2.5 | Full 4-model ensemble | **Active** — the flagship |
| PL BTTS | Full ensemble (BTTS-trained) | **Active** |
| PL O/U 1.5 (alt line) | Dixon-Coles alone | **Active — Over bets only** (backtest: Under has no edge) |
| Other PL alt lines (3.5, 4.5…) | DC (evaluated) | **Not bet** — no proven edge; market too efficient there |
| EFL O/U 1.5, BTTS | EFL 3-model ensemble / DC | **Active** |
| EFL O/U 2.5 | EFL ensemble | **Monitored** — priced and tracked but never staked (~+1.2% gross ROI won't survive commission) |
| EFL O/U 3.5 | DC | Alt-line path (both sides allowed in EFL) |

The principle: **only bet where backtests prove edge; keep everything else
priced-but-unbet** so calibration keeps being measured (a "Monitored Market")
and the market can be re-activated if evidence improves.

### 6g. The exchange layer (designed, parked)

ADR 0003 specs executing on betting exchanges (Betfair/Matchbook) — they don't
ban winners, but charge **commission** on winnings (Betfair 5%, Matchbook 2%).
That demands commission-aware **Minimum Odds** — the price below which a bet
stops being +EV after fees: `O_min = 1 + (1 − p)/(p × (1 − c))` — plus a **Post
Target** (the price you actually ask for). Execution is deliberately
**advisory/manual**: the system computes the numbers, the human places the
order. A 2026-06-25 probe found exchange liquidity in goals markets is
near-zero until ~a day before kickoff, so this whole layer is **parked** as a
separate problem (see `memory/oddspapi_exchange_coverage.md`).

---

## 7. Stage 5 — Operations (how it runs by itself)

Two layers of automation:

**In-process scheduler (APScheduler, while `run.py all` is running):**

| When | Job |
|---|---|
| 06:30 daily | Data refresh — new results, player data |
| 07:00 daily | Fixture planner — checks today's kickoffs, schedules matchday jobs |
| Matchday: morning + KO−3h + KO−1h | **Scans** — fetch live odds, run models, compare, write Picks/Recommendations. The KO−3h/KO−1h pair brackets team-news: early scan catches pre-lineup prices, late scan reacts to confirmed lineups |
| Near kickoff | Closing-odds capture (for CLV measurement) |
| 09:00 & 23:00 daily | Settlement (below) |
| Sunday 23:30 | Weekly retrain — full **Data Refresh**: rebuild features, retrain all models on newest results. New learned parameters, zero strategy changes. Also refreshes the Betfair league splits |

**Windows Task Scheduler (independent of the dashboard being open):**

| When | Job |
|---|---|
| 5th monthly, 10:00 | Betfair historical download — previous month's O/U 1.5, O/U 2.5, BTTS closing prices appended to the master CSVs (skips June/July off-season) |
| Weekly | Retrain backup task |
| Daily | Settlement backup task |

Guardrails throughout: jobs can't stack on themselves (`max_instances=1`),
missed runs collapse into one catch-up (`coalesce`), settlement runs are
file-locked against concurrent execution, quota trackers are lock-protected,
and odds caches older than 90 minutes are refused on matchdays.

---

## 8. Stage 6 — Settlement and the feedback loop

After matches end, settlement (09:00 and 23:00) fetches final scores from
ESPN's free API and grades every open Recommendation: over 2.5 with a 2-1
final → won. Rows are marked settled exactly once (idempotent — a re-run can't
double-settle), and profit/loss is recorded. For **logged bets** (bets you
actually placed), the chosen venue's commission rate is deducted from winnings
and snapshotted onto the row, making `logged_bets` the true net-P&L record.

This closes the loop: predictions → bets → outcomes → analytics → (weekly)
retraining on the newly grown dataset.

---

## 9. The dashboard (Dash web app, port 8050)

Four tabs:

1. **Match Centre** — today's/upcoming fixtures: model probabilities per
   market, best odds and which bookmaker, edge %, confidence, Kelly stake %
   (colour-binned; amber ≥5% flags an outlier worth sanity-checking), model
   agreement, and squad-availability panels (with lineup-confirmed re-runs).
2. **Bet Tracker** — log real bets; tracked to settlement with commission-aware
   net P&L.
3. **Performance** — history of logged bets: win rate, ROI, P&L over time.
4. **Model Analytics** — the self-honesty tab:
   - **Prediction tracking** with **Wilson 95% confidence intervals** (honest
     uncertainty on small samples — 10 wins from 15 bets is a wide interval,
     not "67% win rate!"),
   - **Calibration plot** — "when the model says 60%, does it happen 60% of
     the time?" (bins with n<5 dropped, error bars on the rest),
   - **Strategy counterfactuals** — three alternate histories: recommended
     bets with Kelly sizing vs the same bets flat-staked vs *every*
     positive-edge pick flat-staked. Separates "does the filter add value?"
     from "does Kelly sizing add value?",
   - **Cumulative P/L curve** with drawdown annotation and the counterfactual
     overlaid,
   - **CLV tracking** — did the odds you took beat the closing line? The
     literature's strongest early evidence of real edge, since closing lines
     are the market's most-informed price,
   - **Per-market breakdown** with 🟢/🟡/🔴 sample-adequacy badges (bootstrap
     CI width), so thin-sample markets can't masquerade as proven.

---

## 10. Where everything is stored

| Store | Contents |
|---|---|
| `data/dashboard.db` (PL) & `data/dashboard_efl.db` (EFL) — SQLite | 5 tables each: `match_analysis` (every scan result), `recommendations` (filtered bets + settlement), `predictions` (all positive-edge picks, for counterfactuals), `logged_bets` (real money, commission-netted), `bankroll` (snapshots) |
| `CompleteDS_CSV.csv` / `CompleteDSChamp_CSV.csv` | Canonical training datasets (facts + computed features) |
| `models/*.pkl` (PL) & `models/championship/*.pkl` (EFL) | Trained ensembles, pipeline caches, squad adjuster |
| `data/betfair_*.csv` | Historical exchange prices (masters + per-league splits) |
| `data/*_cache.json` | Live-odds API caches |
| `logs/` | Job logs (rolling + timestamped) |

---

## 11. Honest limitations (know these when explaining the system)

1. **The edge is small and the market is good.** Best single-model AUC ~0.60;
   theoretical ceiling for O/U 2.5 is ~75–78%. This is a thin-margins game won
   by discipline, not a crystal ball.
2. **Calibration drifts ~3–6% hot** on test data. Relative edges survive;
   absolute probabilities are slightly inflated — hence blend, quarter-Kelly,
   agreement gates.
3. **Backtest ROI ≠ future ROI.** Historical +9.7% ROI (10% edge threshold)
   was measured against historical prices; live markets adapt, and soft books
   limit winners (the motivation for the parked exchange plan).
4. **No liquidity data.** Historical Betfair data is last-traded-price only;
   live probes show goals-market liquidity arrives only ~a day before kickoff.
   Early prices exist; early *fillable size* mostly doesn't.
5. **Small live samples.** Wilson CIs and adequacy badges exist precisely
   because a season of recommendations is statistically few bets.
6. **EFL is thinner** — less data, fewer sources (no Understat/FPL), 3-model
   ensemble, and its O/U 2.5 is Monitored, not staked.

---

## 12. The one-minute explanation (for telling a friend)

> "I built a system that estimates the probability of football outcomes —
> mainly 'will there be over 2.5 goals' and 'will both teams score' — using
> four statistical models trained on a decade of match data, combined and
> calibrated so their probabilities are honest. It converts bookmaker odds
> into the market's implied probability, strips out their margin, and looks
> for gaps where my number is meaningfully higher. It only bets those gaps,
> stakes a quarter of the mathematically optimal amount, requires multiple
> models to agree, cuts stakes during losing streaks, and grades every
> prediction against real results — including checking whether I beat the
> closing line, which is the gold-standard test of real edge. It runs itself:
> fetches odds on matchdays, retrains weekly, settles bets daily, and shows
> everything on a dashboard."

---

*Companion docs: `CONTEXT.md` (vocabulary), `docs/adr/` (decisions),
`docs/betfair_ingestion_scope.md` (Betfair pipeline detail),
`.claude/skills/data-schema/` (schemas), `.claude/skills/api-budget/` (quotas).*
