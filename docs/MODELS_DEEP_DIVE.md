# Stage 3 Deep Dive — The Models: What They Are, Exactly How They Work, and Why Each Was Chosen

_Companion to `docs/SYSTEM_OVERVIEW.md` §5, at full technical depth. Every
formula and constant here was verified against `model.py` (2026-07-04). Design
rationale comes from code docstrings, `docs/adr/`, and validated backtest
results. Written 2026-07-04; §7 rewritten 2026-08-14 after the Squad Adjuster
was deleted, with feature-membership counts re-verified against `config.py`
and the live pickles on that date._

---

## 0. The design problem this stage solves

The betting market is *nearly* efficient — Pinnacle's closing line is a very
good probability estimate. To beat it after margin, a model doesn't need to be
"good" in the usual ML sense; it needs to be **differently wrong** from the
market in a way that's right slightly more often than chance. Two consequences
drive every choice below:

1. **Error diversity beats raw accuracy.** Three mediocre models that fail in
   *different* ways combine into something stronger than one good model — but
   only if their errors are genuinely uncorrelated. That's why the ensemble
   mixes a hand-built statistical model (Dixon-Coles), two machine-learning
   pattern hunters (XGBoost, LightGBM), and a linear baseline (LogReg): four
   different *kinds* of mistake.
2. **Honest validation is worth more than a better model.** Every component is
   validated walk-forward (train on the past, test on the unseen future),
   because a model that only looks good with hindsight is worse than useless —
   it bets real money on an illusion.

The stage's contract: fixture in → **one calibrated probability per market**
out.

---

## 1. Dixon-Coles — the structural model

### 1.1 Why Poisson at all

Goals are rare events arriving more or less independently through 90 minutes
at some underlying rate. That is *precisely* the process the *Poisson
distribution describes: if a team's scoring rate over a match is λ ("lambda"),
then

> P(score exactly k goals) = e^(−λ) · λ^k ⁄ k!

Empirically this fits football astonishingly well — decades of literature
(and this dataset) confirm goal counts are near-Poisson. This is a **structural
assumption**: rather than learning "what does 2.7 shots-per-game imply?" from
data like the GBDTs must, DC *builds in* the known physics of scoring and only
has to learn each team's rates. Structure is cheap accuracy: it's why DC is
the strongest single model here despite having ~80 parameters instead of
thousands of tree splits.

So the entire model reduces to one question: **what are the two λs for this
fixture?** Everything else is arithmetic.

### 1.2 The lambda formula — exactly

Each team carries **four ratings**, all centred on 1.0 = league average:

| Rating | Meaning |
|---|---|
| `attack_home[T]` | How many goals T scores **at home**, relative to the league's home-scoring average |
| `attack_away[T]` | Same, away |
| `defence_home[T]` | How many goals T **concedes at home**, relative to what an average away side scores (>1 = leaky) |
| `defence_away[T]` | Same, away |

Plus two league-level parameters:

- **μ (mu)** — league average goals per team per match (~1.35 in the PL)
- **γ (gamma)** — the home-advantage factor: the ratio of home goals to away
  goals across the league (~1.36, i.e. home teams score ~36% more)

For a fixture Home vs Away (`model.py:predict_match`):

```
λ_home = attack_home[Home] × defence_away[Away] × μ × √γ
λ_away = attack_away[Away] × defence_home[Home] × μ ÷ √γ
```

then clamped to [0.1, 5.0] as a sanity bound.

**Read it left to right:** start from the league-average scoring rate μ; scale
up/down by how good this team's attack is *at this venue*; scale again by how
leaky the opponent's defence is *at their venue*; finally apply home advantage.

**Why √γ on both lines rather than γ on one?** This is the "no double-counting"
fix noted in the class docstring. γ is defined as the *ratio* home/away. If
you multiplied λ_home by the full γ and left λ_away alone, total goals (λ_home
+ λ_away) would inflate above the league mean for every fixture — you'd have
applied home advantage *and* kept the away rate at the venue-neutral level.
Splitting it — home gets ×√γ, away gets ÷√γ — applies the full *relative*
advantage (their ratio is exactly γ) while keeping the *product* of the two
adjustments equal to 1, so the league's total-goals level is preserved. For an
average-vs-average fixture: λ_home = 1.35×√1.36 ≈ 1.57, λ_away = 1.35/√1.36 ≈
1.16 — ratio 1.36 ✓, total 2.73 ≈ 2×μ ✓.

**Worked example** (illustrative numbers): Arsenal at home to Coventry.
Arsenal `attack_home` = 1.30 (scores 30% more at home than league average);
Coventry `defence_away` = 1.15 (concedes 15% more away than average);
Coventry `attack_away` = 0.80; Arsenal `defence_home` = 0.85. With μ = 1.35,
γ = 1.36 (√γ ≈ 1.166):

```
λ_home = 1.30 × 1.15 × 1.35 × 1.166 ≈ 2.35   (Arsenal expected goals)
λ_away = 0.80 × 0.85 × 1.35 ÷ 1.166 ≈ 0.79   (Coventry expected goals)
```

Those two numbers now define the full probability distribution over every
scoreline — which is what makes DC able to price *any* goals market (§1.6).

**Why venue-specific ratings (4 per team, not 2)?** Because home/away splits
are real and large — some teams are genuinely different sides away from home
(the promoted-team priors below encode exactly this: decent at home 0.90,
struggling away 0.75). A single attack number would average away a signal the
market itself prices.

### 1.3 How the ratings are estimated — Path 1: weighted averages with shrinkage (the default)

The intuitive estimator, verified from `fit()`:

**Step 1 — raw venue ratio.** For team T's home attack: take every home match
in the training window, compute `goals scored ÷ league home-scoring average`
per match, and average them. A team that scores 2.0 at home when the league
home average is 1.55 gets ≈ 1.29.

**Step 2 — time decay.** That average is *weighted*, recent matches counting
more: match k games back gets weight

> w = 0.5^(k / half_life)

With the default `half_life = 30`, a match 30 games ago counts half as much as
last week's; 60 games ago a quarter. **Why:** squads, managers, and tactics
drift — last season's Arsenal only partially describes this season's. The
half-life is a **tuned hyperparameter**, grid-searched per market over
{10, 15, 20, 25, 30, 40, 50, 70} via walk-forward CV — and notably the
fast-moving markets (BTTS, O/U 1.5) prefer *short* half-lives (10–15): recent
scoring/defensive streaks matter more there than long-run class.

**Step 3 — partial pooling (Bayesian shrinkage).** Small samples lie. A
promoted team 3 games into the season with ratio 1.8 is *not* a 1.8-attack
side — it's had a lucky fortnight. Each estimate is therefore blended toward
the league mean (1.0), weighted by sample size:

> rating = (n/(n+6)) × own_estimate + (6/(n+6)) × 1.0

(`N_PRIOR = 6`.) At n=2 matches you're 25% your own data, 75% league average;
by n=20 you're ~77% your own data. **Why this design:** it replaced a hard
threshold ("use venue estimate only if ≥3 matches") that created a
discontinuity — a team's rating could jump discretely on its 3rd home game.
Shrinkage makes the transition continuous and is the standard Bayesian answer
to small-sample noise.

**Step 4 — the fallback ladder.** If a team has *no* matches at a venue, fall
back to its cross-venue pooled estimate; if it has no matches at all (a newly
promoted side's first-ever fixture), fall back to **venue-aware promoted-team
priors**: attack 0.90 home / 0.75 away, defence 1.10 home / 1.20 away. **Why
these numbers:** they encode the well-established empirical pattern that
promoted teams compete respectably at home but travel badly — a uniform prior
(all 1.0) would systematically overrate them.

The same four steps produce defence ratings, using goals *conceded* divided by
the *opponent-side* league average (conceding at home is measured against what
away teams typically score, and vice versa).

μ and γ come directly from the data: μ = (home avg + away avg)/2, γ = home avg
÷ away avg.

### 1.4 How the ratings are estimated — Path 2: full maximum likelihood (Dixon & Coles 1997)

The weighted-average path is fast and robust but estimates each rating
*independently*. It has a subtle blind spot: if a team happened to face six
leaky defences in a row, its raw scoring ratio overstates its attack — the
ratio method can't tell "good attack" apart from "easy schedule."

**Maximum Likelihood Estimation (MLE)** fixes exactly that, by answering:
*which single set of all parameters — every team's four ratings, plus μ, γ,
and ρ jointly — makes the actual observed scorelines most probable?* Because
every rating is estimated *in the context of who it was achieved against*,
strength-of-schedule is handled automatically.

Verified implementation details (`fit_mle()`), each with its reason:

- **Objective**: the time-decay-weighted Poisson log-likelihood of every
  historical scoreline, *including the τ low-score correction* (so ρ is
  estimated jointly, not bolted on).
- **Log-space parameters**: the optimiser works on log(ratings), so ratings
  are guaranteed positive without awkward constraints, and multiplicative
  structure becomes additive: λ_home = exp(log_att_H + log_def_A + log_μ +
  log_γ/2) — the same formula as §1.2, in logs.
- **Sum-to-zero constraint** on each log-rating block (enforced as a heavy
  ×100 penalty). **Why:** the model has a built-in ambiguity — double every
  attack and halve μ and *nothing observable changes*. Pinning each rating
  family's log-mean to zero resolves this "identifiability" problem: the scale
  lives in μ, the ratings are pure relative strengths.
- **Ridge penalty** (α = 0.01) pulling log-ratings toward 0 (= average team).
  The MLE analogue of Step-3 shrinkage: keeps small-sample teams from getting
  extreme ratings. (Shrinkage is *disabled* on this path to avoid
  double-regularising.)
- **Warm start**: the optimiser starts from Path-1's estimates rather than
  from scratch — faster convergence to a sensible optimum.
- **Bounds**: log-ratings ∈ [−2, 2] (ratings 0.14–7.4), μ ∈ [0.5, 3.0],
  γ ∈ [0.8, 2.0], ρ ∈ [−0.30, 0.05] — generous but rules out pathological fits.
- **Optimiser**: L-BFGS-B (the standard choice for smooth bounded problems of
  this size; ~87 parameters for a 20-team league).

**Why two paths at all, and who uses which?** The tuning harness
(`tune_dc_params`) grid-searches both and picks whichever wins walk-forward
validation for each league/market. One structural constraint decides the
Championship: its training pool spans **64 distinct teams** (relegations,
promotions), meaning 4×64+3 = **259 parameters — the MLE fails to converge**,
so the EFL always uses the weighted-average path. It's a genuine trade-off,
not a shortcut: Path 1 is more robust, Path 2 is more statistically principled
when it converges.

### 1.5 The tau (τ) correction — fixing Poisson's one big lie

Independent Poisson assumes the two teams' goal counts don't influence each
other. Mostly true — except at low scores, where football is visibly
*strategic*: at 0-0 both teams often settle; at 1-1 late, both may shut up
shop. Real data shows more 0-0 and 1-1 draws than independence predicts
(~15% more 0-0s).

Dixon & Coles' fix multiplies exactly four cells of the scoreline grid by a
correction τ, controlled by one parameter ρ (rho):

| Scoreline | τ multiplier |
|---|---|
| 0-0 | 1 − λ_home·λ_away·ρ |
| 0-1 | 1 + λ_home·ρ |
| 1-0 | 1 + λ_away·ρ |
| 1-1 | 1 − ρ |
| anything else | 1 (untouched) |

With ρ negative (default **−0.13** here; tuned over [−0.20, 0.0]; MLE bounds
allow [−0.30, +0.05]), the 0-0 and 1-1 cells get *boosted* and the 0-1/1-0
cells *reduced* — matching observed reality. The corrections are constructed
so probability mass is redistributed among the low-score cells; the
distribution still sums to 1.

**Why this matters for the bottom line:** the profitable markets here are
low-line goals markets (O/U 1.5, O/U 2.5, BTTS). Their probabilities are
dominated by exactly the cells τ corrects. An uncorrected Poisson would
systematically underprice Under 1.5 and BTTS-No — a direct, market-relevant
bias.

### 1.6 From two lambdas to any market price

With λ_home, λ_away, and τ, build the full scoreline grid — P(h goals, a
goals) for h, a ∈ 0..11 (a 12×12 matrix; beyond 11 goals the mass is
negligible):

> P(h, a) = Poisson(h; λ_home) × Poisson(a; λ_away) × τ(h, a)

Then any goals market is a sum over cells:

- **P(Over 2.5)** = 1 − Σ P(h,a) for all h+a ≤ 2 (i.e. 1 − [P(0,0) + P(1,0) +
  P(0,1) + P(2,0) + P(1,1) + P(0,2)])
- **P(Over 1.5)** = 1 − P(0,0) − P(1,0) − P(0,1)
- **P(BTTS Yes)** = 1 − P(home blanks) − P(away blanks) + P(0-0), by
  inclusion–exclusion. (Implemented in closed form using only pmf(0,·) and
  pmf(1,·) — a performance fix after the naive 288-scipy-calls-per-fixture
  version intermittently hung tuning runs for hours.)

**This is why DC alone prices the alt lines**: one fitted model yields a whole
scoreline distribution, so every line (0.5 through 8.5, BTTS, correct score if
ever wanted) comes from the *same* two lambdas, mutually consistent by
construction. The GBDTs would need a separately trained model per market.

### 1.7 Odds and ends

- **xG option** (`use_xg=True`): ratings can be fit on expected goals instead
  of actual goals where Understat data exists. Rationale: xG is a
  lower-variance measure of chance creation (a 3-0 with 0.9 xG was lucky, and
  goals-based ratings would be fooled). Whether it's used is decided by the
  same walk-forward tuning as everything else.
- **Clamps everywhere**: λ ∈ [0.1, 5.0] at prediction ([0.05, 8.0] during MLE),
  final probability ∈ [0.01, 0.99]. Cheap insurance against a degenerate
  rating producing a "certain" price.
- **Why DC wins here** (test AUC **0.597** vs XGB 0.545, LGB 0.527): its
  structure encodes true facts about football (Poisson arrivals, venue
  effects, low-score dependency), so its ~80 parameters are all spent on
  learning *team strengths* rather than rediscovering the physics. Fewer
  parameters + correct structure = less overfitting on ~8k matches.

---

## 2. XGBoost — the interaction hunter

**What it is.** Gradient-boosted decision trees: build a shallow tree that
crudely predicts over/under from the features; compute its errors; build a
second tree to predict *those errors*; repeat hundreds of times; the final
score is the running sum of corrections, squashed to a probability.

**Why it's here.** DC deliberately ignores almost everything except goals.
But the feature pipeline computes 150+ signals — shot volumes, conversion
rates, congestion, discipline, weather, league position — and the **Wheatcroft
principle** (shots/corners out-predict goals) says much of the real signal
lives there. GBDTs are the state of the art for exactly this: medium-sized
tabular data with unknown non-linear interactions ("high shot volume matters
more when the opponent defends deep *and* it's not raining"). No other model
class in the ensemble can discover such rules.

**Why XGBoost specifically:** the reference implementation of the family —
regularised (depth limits, subsampling), handles missing values natively (no
imputation needed for the tree path), fast enough for walk-forward retraining
every fold. Trained with early stopping on the validation season (stop adding
trees when validation stops improving — the standard overfitting brake), and
followed by **feature pruning**: features with zero importance after an
initial fit are dropped and the model refit, trimming noise dimensions.

**Its role in the blend:** the stacker gives XGB the largest coefficient in
both variants (live 1.306; research ≈2.16) despite DC's better standalone AUC
— evidence its errors are the most *complementary* to the other inputs (it
brings information they lack, even if noisier on its own). In the live
4-input stacker DC is a close second at 1.247 (§5).

---

## 3. LightGBM — the second opinion, grown differently

**What it is.** The same gradient-boosting idea, from a different library with
a genuinely different growth strategy: XGBoost grows trees **level-by-level**
(balanced), LightGBM grows **leaf-by-leaf** (always splitting wherever the
gain is highest, producing deep lopsided trees).

**Why include a second GBDT at all?** Not for accuracy — for **decorrelation**.
Different growth policy + different sampling/binning quirks = different
mistakes on the same data. The whole value of averaging/stacking models comes
from their errors *not* lining up; a cheap second GBDT buys real error
diversity for free. Its stacker weight confirms it adds signal beyond XGB
rather than echoing it — though it is the *smallest* of the four in the live
stacker (0.820, below LR's 1.048), which is worth revisiting the next time the
ensemble's composition is on the table.

---

## 4. Logistic Regression — the straight-line baseline (and its curious role)

**What it is.** A weighted sum of features through a sigmoid. No interactions,
no thresholds — the simplest probabilistic model that exists.

**Why it's here.** Three reasons:
1. **A floor and a sanity check.** If trees can't beat the linear model,
   the interactions they "found" are noise.
2. **A fourth, differently-shaped opinion** for the Agreement gate (below).
3. **Stability**: heavily regularised (C = 0.01 — strong penalty pulling
   coefficients toward zero), it barely moves week to week.

**Implementation guards (each earned by an incident):** features are
median-imputed (LR can't take NaNs, unlike trees), standard-scaled, and the
scaled values **clipped to ±5 standard deviations** — after a real failure
where a drifted feature (`Home_DefensiveStrength_5`) reached **+257σ** on test
data and produced garbage predictions. The clip converts "feature drift
catastrophe" into "feature saturates harmlessly."

**The nuance — where LR does and doesn't count** (verified in code, and the
docs' most commonly mis-stated fact):

- LR **is trained** in every walk-forward fold, and its predictions **do count
  toward the Agreement gate** (the 2-of-N headcount that gates
  Recommendations and scales stakes) and appear in the per-model breakdown on
  the dashboard.
- **LR *is* an input to the live stacker — corrected 2026-08-14.** Earlier
  revisions of this section said it had been dropped for adding noise. That is
  false of the model that prices bets. Read directly from
  `models/pl_trained_state.pkl`: `n_features_in_ = 4`, coefficients
  **XGB 1.306, LGB 0.820, LR 1.048, DC 1.247** (intercept −2.197). LR outweighs
  LightGBM.

  The error came from citing `model.py:1856` (`oof_df[["xgb","lgb","dc"]]`) as
  though it were production. It is the **Research Path** — a script run by hand,
  not what `scheduler.py:217` retrains. The Production Path is
  `predict.py:400` (`oof_df[["xgb","lgb","lr","dc"]]`); `backtest.py:309` and
  `tests/conftest.py:45` agree with it. The two paths have genuinely diverged,
  and `scripts/lr_ablation_test.py` exists to compare 3-stack against 4-stack
  but its conclusion was never recorded anywhere. See **Training Path** in
  `CONTEXT.md`.
- **PL only.** The Championship's thinner feature set gives a linear model too
  little to work with; the EFL ensemble is 3-model throughout — so
  `championship_model.py:546`'s three-column stack is correct for the EFL.

---

## 5. The stacker — learning how much to trust each model

**The problem:** you have several probabilities per fixture. A simple average
treats them as equally reliable, which they demonstrably aren't. Hand-picked
weights would be guesses.

**The solution:** a tiny logistic regression (the **stacker**) whose *inputs
are the base probabilities* and whose output is the final probability. It
learns the optimal combination from data — effectively regressing "when XGB
says X, LGB says Y, DC says Z, how often did the match actually go over?" As a
bonus, a logistic meta-learner partially *re-calibrates* while it combines.

**How many inputs depends on which path you mean** (§4, and **Training Path**
in `CONTEXT.md`) — verified 2026-08-14:

| Path | Inputs | Coefficients |
|---|---|---|
| **PL live** (`predict.py:400`) | 4 — XGB, LGB, **LR**, DC | 1.306 / 0.820 / **1.048** / 1.247, intercept −2.197 |
| PL research (`model.py:1856`) | 3 — XGB, LGB, DC | ≈2.16 / 1.18 / 1.10 |
| EFL live (`championship_model.py:546`) | 3 — XGB, LGB, DC | (no LR in the EFL ensemble) |

The often-quoted 2.16/1.18/1.10 figures are the **research** row. Quote the
first row when describing what prices a bet.

**The rule that makes stacking honest — out-of-fold (OOF) training.** The
stacker must never see a base model's prediction on data that model was
trained on. GBDTs partially memorise their training data, so their in-sample
predictions look artificially brilliant; a stacker trained on those would
learn "trust XGB 95%" and fail live. So the stacker trains **only on
walk-forward predictions**: each fold's models are trained on seasons up to N
and predict season N+1 — every OOF prediction is a genuine
past-predicts-future forecast, exactly matching live conditions. (Six folds,
validation seasons 19–24. A shuffled-KFold fallback exists only for the
degenerate case of insufficient OOF rows.)

**The operational gotcha (worth knowing):** the ensemble must be *given* the
DC probability explicitly (`predict_proba(X, dc_probs=...)`). There's a
pipeline feature also called `Poisson_DC`, computed with different parameters
— it correlates only **0.41** with the real DC model. Relying on the fallback
silently degrades the ensemble; every production call passes `dc_probs`.

---

## 6. Calibration — making the number mean what it says

A model can rank matches well while its probabilities run systematically hot
or cold. Betting punishes this brutally: stakes (Kelly) and edges are computed
*from the probability itself*, not the ranking.

Two layers, verified in `predict_proba`:

1. **Platt-family scaling** on the stacker output — fit on the OOF
   predictions via nested temporal CV (never on data the stacker trained on).
   The deployed variant is a **logit-shift**: a single constant subtracted in
   log-odds space, correcting the mean level while *provably preserving the
   ranking* (AUC unchanged). The code also supports full Platt (a fitted
   logistic curve) and isotonic regression (a monotone lookup) — logit-shift
   won on walk-forward evidence and is the least prone to overfitting its own
   calibration set.
2. **Regime detection** — if the current season's Over rate drifts from the
   ~52% historical base rate (a "high-scoring regime"), calibration shifts
   mid-season. Applied to O/U only; BTTS rates are stable across seasons, so
   adjusting them would add variance without signal.

**Known, accepted imperfection:** test-set probabilities still run ~3–6% hot
(mean 0.586 vs actual 0.528 over-rate). Ranking is intact. This is tolerated
because edge = blended − fair is *relative* (both sides shift together), and
the drift is larger than de-vig method differences (~0.1–0.3pp) — which is
also why swapping de-vig formulas is considered below the noise floor. The
35/65 blend and quarter-Kelly exist precisely to absorb residual
miscalibration.

---

## 7. The Squad Adjuster — built, then deleted (2026-08-14)

**This stage no longer exists.** It is documented here because the constraint
that produced it is still live, and because *how* it died is instructive.

**The original design.** Player-availability data (injuries, suspensions, key
absences) only exists from season 2024-25. Put those features in the main
models and they'd be NaN for ~90% of training history — trees would learn
"missing means old match," a date proxy, not a squad signal. So availability
lived in a **separate second stage**: a small logistic regression taking
(ensemble probability + 16 squad features) → adjusted probability, trained
only on the two seasons where the data exists. Measured gain at the time:
52.3% → 54.0% test accuracy.

**Why it was deleted** (`eba6dcc`) — it was dead at both ends:

- *Output end.* `squad_adjuster.pkl` was rewritten by the weekly retrain and
  loaded by **nothing**. It appeared in exactly two places repo-wide:
  `scheduler.py`, which wrote it, and `squad_adjuster.py`, which named the
  path. Never in `predict.py`, `scan.py`, `dashboard.py` or `run.py`. The
  measured +1.7pp never reached a single Recommendation.
- *Input end.* It trained against `over_under_model.pkl`, `scaler.pkl` and
  `feature_list.pkl` — all dated 2026-05-04, a legacy **single** model, not
  the current ensemble. The 2026-08-03 republish retired columns that stale
  feature list still named, so it raised "not in index" on **every retrain for
  eleven days**. Artefact mtimes tell the story: both trained states rebuilt
  10 Aug, `squad_adjuster.pkl` untouched since 3 Aug.

**Why it wasn't caught sooner.** It failed as *a job that ran and wrote a
file*. Nothing asserted the file was newer than its inputs, or that anything
read it — the same "failure that looks like success" shape as the `301 → EC.csv`
substitution and the missed Task Scheduler run.

**The constraint that outlived it.** The 16 `SQUAD_FEATURES` are still computed
by `pipeline.py` on every run, and still absent from `ALL_FEATURES` — verified
**0/16** in `ou_features` and `btts_features` for both leagues. For contrast,
every other enrichment family is live in the PL O/U model: XG 8/8, PLAYER 4/4,
WEATHER 3/3, ROSTER 12/12, TACTICAL 8/8, SHOT_LEVEL 36/36. They are kept
because the blocker is temporal: once 3–4 seasons of player data accumulate,
they can graduate into the main models directly — the design the adjuster was
always a workaround for.

Note the naming trap: **`PLAYER_FEATURES` ≠ `SQUAD_FEATURES`**. The former
(`Home_InjuryBurden`, `Away_InjuryBurden`, `Home_KeyAbsences`,
`Away_KeyAbsences`) come from the FPL API via `api/fpl.py` and are **4/4 live**.
The dead 16 come from FPL-Core-Insights via `api/player_features.py`. Injury
signal reaches the models; availability signal does not.

---

## 8. Validation — the methodology all of the above answers to

- **Walk-forward only.** Train ≤ season N, test N+1, slide. Chosen over
  standard KFold because shuffling time lets every model peek at the future
  (a 2019 prediction informed by 2024 patterns) — football drifts (tactics,
  xG-era pressing, rule changes), and the *only* deployment condition is
  past→future.
- **Metrics:** AUC (pure ranking quality — the primary gate), Brier score
  (probability accuracy — punishes miscalibration), accuracy (reported, not
  optimised). Backtested ROI at realistic edge thresholds is the final,
  binding criterion — a model that ranks well but can't clear the market's
  margin doesn't ship.
- **The honest scale of the numbers:** stacker test AUC **0.5799**, walk-forward
  mean **0.583**, DC standalone **0.597**. A coin flip is 0.500; the estimated
  information ceiling for O/U 2.5 is ~0.75–0.78. This is a small, real edge
  — which is exactly why Stage 4 (blend, agreement, fractional Kelly,
  drawdown guards) is engineered around *protecting* it rather than betting
  it aggressively.

---

## 9. Why-this-not-that (the roads not taken)

| Alternative | Why rejected |
|---|---|
| One big model instead of an ensemble | A single family = a single error style; the market forgives nothing. Diversity is the edge. |
| Neural networks | ~8k training matches is tiny by deep-learning standards; GBDTs dominate tabular data at this scale, and NNs would overfit and be uninspectable. |
| Bookmaker odds as features | Would leak the market's opinion into the model, collapsing the entire "model vs market" comparison into circularity. Odds are used **only** in the Stage-4 comparison layer. |
| Full bivariate Poisson (shared-λ correlation term) | τ captures the empirically important dependency (the four low-score cells) with **one** parameter; the full bivariate model adds parameters and fitting fragility for negligible gain on this data. |
| KFold CV | Leaks future→past; overstates every model's skill; rejected on principle. |
| Feature scaling for the trees | Trees split on thresholds — scaling is a no-op for them. Only LR needs (and has) its own scaler. Simpler pipeline, one less thing to drift. |
| Hard sample-size thresholds for DC ratings | Replaced by continuous shrinkage (n/(n+6)) — the old threshold caused rating jumps at match 3. |
| MLE for the Championship | 64 teams → 259 parameters → optimiser fails to converge. Weighted-average path is the deliberate fallback. |

---

## 10. One-paragraph summary (for explaining Stage 3 aloud)

> "Four models look at every fixture. Dixon-Coles is a purpose-built
> statistical model: it estimates each team's attack and defence strength at
> each venue — recent games weighted more, small samples shrunk toward
> average, optionally refined by maximum likelihood so strength-of-schedule
> is handled — multiplies them with the league scoring rate and a
> split-in-half home-advantage factor to get each side's expected goals, and
> turns those two numbers into a probability for every possible scoreline,
> with a correction for football's extra 0-0s and 1-1s. XGBoost and LightGBM
> are machine-learning models that mine 150+ engineered features for
> interaction patterns the statistical model can't see, and a heavily
> regularised logistic regression provides a stable linear baseline. A small
> meta-model — trained only on honest, walk-forward, out-of-sample
> predictions — learns how much to trust each of the four and merges them
> into one probability, which is then calibrated
> so that '60%' historically means 60%, with a mid-season adjustment when the
> league's scoring environment shifts. All four models' opinions are then
> counted in the agreement check that gates whether a bet is ever
> recommended."

---

*Sources: `model.py` (DixonColesPredictor, EnsembleModel, walk-forward +
stacker training — all formulas verified 2026-07-04), `championship_model.py`
(EFL tuning constraints), `.claude/skills/dixon-coles-methodology/`,
`docs/adr/`, `CONTEXT.md` (vocabulary). Academic anchors: Dixon & Coles
(1997); Rue & Salvesen (2000); Wheatcroft (LSE) on shots/corners vs goals.*
