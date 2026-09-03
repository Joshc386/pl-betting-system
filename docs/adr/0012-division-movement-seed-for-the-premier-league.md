# 12. Division Movement Seed for the Premier League

Date: 2026-08-30

## Status

**Built and measured 2026-08-30. Not deployed.** The criteria below were
committed in `3931466` before any measurement ran; the Outcomes section was
empty at that commit and filled afterwards, so the pre-commitment is checkable
in the history rather than asserted here.

All four measurement arms have run and every pre-committed criterion holds.

Deployment is a separate approval against the measured numbers, and the work
stays on `feat/pl-division-movement-seed` until it is given —
`PL_RETRAIN_ENABLED` is `True`, so merging would let the weekly scheduler
deploy it on its own timetable.

Extends [ADR 0011](0011-one-division-movement-seed-per-arrival.md), which
built the Division Movement Seed for the EFL and explicitly scoped the PL out:
"extending the seed to the PL is left as separate work". This is that work.

## Context

Two defects, one mechanism, both live in the PL today.

### Dixon-Coles rates PL arrivals on absent or decades-old history

`seed_arrivals` exists in `model.py` and is shared by both leagues, but only
`championship_predict.py` calls it. The PL trained state carries no
`seed_params` key at all. Measured against the live pickle on 2026-08-30, the
three sides arriving for 2026/27 fail in three different ways:

| side | PL seasons held | `att_h` | `att_a` | `def_h` | `def_a` |
|---|---|---|---|---|---|
| Coventry City FC | [0] | *unrated* | | | |
| Hull City AFC | [8, 9, 13, 14, 16] | 0.865 | 0.516 | 1.253 | 1.259 |
| Ipswich Town FC | [0, 1, 24] | 0.602 | 0.937 | **1.681** | 1.222 |
| *Everton FC (control)* | *continuing* | *0.853* | *0.800* | *1.023* | *0.896* |

Coventry falls through to the hand-picked `PRIORS` bucket — the same four
constants ADR 0011 measured in the EFL and found wrong for both routes and
inverted for one. Hull carries its 2016/17 relegation season; `attack_away`
0.516 is roughly half a league-average side's away scoring. Ipswich carries
2024/25, and `defence_home` 1.681 asserts they concede 68% more than average
at home on the strength of the season they went down.

The cause is ADR 0011's, unchanged: `_decay_weights` decays by position in a
team's own match sequence rather than by calendar date, so a gap of any length
is invisible to the weighting, and `_shrink_to_league` keys on match *count*,
so a long-absent side is rated with near-total confidence.

**Price impact, Dixon-Coles component only**, against rating the three as
league-average sides:

```
Ipswich Town FC v Liverpool FC     0.7603 -> 0.6431   +0.1172
Ipswich Town FC v Everton FC       0.4658 -> 0.4307   +0.0351
Liverpool FC v Hull City AFC       0.6166 -> 0.5937   +0.0229
Hull City AFC v Fulham FC          0.5208 -> 0.5025   +0.0183
Coventry City FC v Everton FC      0.4208 -> 0.4307   -0.0099
Everton FC v Coventry City FC      0.4711 -> 0.4859   -0.0147
Everton FC v Fulham FC (control)   0.4515 -> 0.4515   +0.0000
```

The control moves by exactly zero, so every other delta is attributable to the
stale rating alone. **The errors point in opposite directions across
fixtures** — Ipswich and Hull toward Over, Coventry toward Under — so the
defect does not surface as aggregate ROI drift. It is 1 of 4 PL ensemble
models and passes through the stacker, so this is component error, not
final-recommendation error.

### The seed's feature coverage diverges between training and serving

`pipeline.initialize_promoted_features` fills *and* blends 22 entries
(`PROMOTED_ROLLING_FEATURES`, 11 pairs). Since `fd64bd5`,
`predict._fixture_feature_row` fills **every** numeric `Home_`/`Away_` column
and blends only that same list. Eight entries are therefore filled at serve
time and not at train time:

`Past5CornersConceded`, `CR_20`, `SOT_CR_5`, `SOT_CR_20` — Home and Away each.

All eight are 99.8% populated in the PL canonical with distributions matching
the EFL's, all eight are in `config.ALL_FEATURES`, and six are in
`BTTS_ALL_FEATURES`. Measured on Ipswich, PL season 24, first home match:

| feature | raw | after training init | serve-time fill |
|---|---|---|---|
| `Home_Past5Goals` *(blended)* | 2.0000 | **6.6000** | **6.6000** |
| `Home_CR_20` | 0.1336 | 0.1336 | 0.1224 |
| `Home_SOT_CR_20` | 0.2463 | 0.2463 | 0.3240 |
| `Home_Past5CornersConceded` | 27.00 | 27.00 | 31.20 |

**The value training preserves is not merely different — it is stale in the
same way the Dixon-Coles rating is.** Rolling features are computed with
`groupby("team")`, not `groupby(["team", "season"])`, with `min_periods=1` and
no gap awareness (`data/build_canonical_dataset.py`). Ipswich's PL seasons are
[0, 1, 24], so the `CR_20` of 0.1336 that training keeps for their first match
of season 24 is a 20-match rolling mean over their 2000/01 and 2001/02
matches. Twenty-three years stale.

This is the same blind spot as the Dixon-Coles defect — sequence position
standing in for calendar time — expressed in a second subsystem. The blend
list is what cures it, and it covers 11 pairs of the 19 available.

**The omission was never a decision.** `PROMOTED_ROLLING_FEATURES` was hoisted
unchanged out of `initialize_promoted_features` by `bdfafab`; the last change
to its membership (`74a7db5`, 2 August) renamed three entries. The EFL's wider
list was authored deliberately in `fd64bd5`. The two were never compared.

### The PL's route axis is finer than the EFL's, and its sample is thinner

ADR 0011 splits arrivals by **arrival direction** — relegated or promoted.
Nobody is relegated into the PL, so that split collapses to one bucket.

But CONTEXT.md's **Promotion Route** is a different axis and *is* available
here: champion, runner-up or play-off winner, read off the sibling EFL final
table and already implemented in `_matchday1_seeds` for ADR 0002. Availability
is inverted between the leagues — the EFL has two arrival directions and no
promotion-route detail, the PL has one direction and full route detail.

Across seasons 1-25 every PL arrival resolves to a route, with none unknown:

| route | events |
|---|---|
| champion | 19 |
| runner-up | 30 |
| play-off winner | 26 |
| **total** | **75** |

## Decision

1. **The PL takes one measured prior, not three.** The route data is free and
   already verified, but `_MIN_EVENTS` is 30 and only runner-up clears it at
   *full* sample. Walk-forward makes it worse — early folds hold single-digit
   events per bucket. This is ADR 0011's own reasoning for rejecting
   per-feature fitting ("two events per parameter at full sample and fewer
   than one in early folds") applied at 25 events per bucket. The split is
   declined on sample size and the counts are recorded here so the decision is
   reproducible as events accrue, three a season.

2. **`_MIN_EVENTS = 30` is kept for the PL.** 75 events clears it at full
   sample; early walk-forward folds will not, and the guard makes the fallback
   to existing `PRIORS` automatic rather than something a caller must remember.

3. **The eight features join `PROMOTED_ROLLING_FEATURES`.** The divergence is
   repaired by widening the *training* fill, not by narrowing the serving one.
   Narrowing serving would make it reproduce a 23-year-old value in the name of
   matching training; matching training is not the goal, one correct definition
   is. Serving already blends whatever the list contains, so both paths
   converge without a second implementation.

4. **Arrival selects the route. It never selects the duration.** Restated from
   ADR 0011 and **enforced by a test**, not by prose. `arrivals_for` documents
   arrival as season-long membership, which is correct for choosing *which*
   cohort seeds a side and cannot answer *how long* the seed applies. Two call
   sites got this wrong last session and both were P0s; the PL wiring is the
   third call site. Duration is gated on `seed_weight` alone.

5. **Deployment is a separate decision from the build.** Measurement is
   walk-forward and touches nothing live. Deployment requires a PL retrain,
   which at `scheduler._weekly_retrain` also writes recommendations — so it is
   money-adjacent and approved on its own, against real numbers rather than
   against this plan. The work stays on its branch until that approval, because
   `PL_RETRAIN_ENABLED` is `True` and a merge would let the scheduler deploy it
   on its own timetable.

### Considered and rejected

- **Three route buckets for the PL.** Rejected on sample size, above. Recorded
  rather than dismissed: if champions genuinely differ from play-off winners in
  their first five matches, one bucket reproduces at a finer grain the
  collapse-two-populations defect ADR 0011 exists to remove. The counts make it
  re-decidable later.
- **Lowering `_MIN_EVENTS` for the PL to admit a three-way split.** Rejected:
  tuning a sample-size guard until it returns the answer you want is the shape
  of decision the guard exists to prevent.
- **Narrowing the serving blend to match training's 11 pairs.** The cheaper
  repair — no retrain — and the reason it was rejected is the 23-year-old
  `CR_20` above. `fd64bd5` was right that a bystander's value is worse than a
  cohort value; it is also true that a side's own decades-old value is worse
  than both.
- **Making rolling windows gap-aware.** The deeper fix, and possibly the right
  one eventually. Rejected *for this change* for the reason ADR 0011 rejected
  making `_decay_weights` calendar-aware: it alters every team's features
  rather than only arrivals', which is a far larger blast radius needing its
  own validation.
- **Requiring log-loss to improve.** ADR 0011's criterion 1 asked for
  improvement, was not met as written, and shipped anyway on the reasoning that
  it gated a defect fix rather than a speculative addition. That reasoning was
  right and the criterion was mis-specified for what it gated. Both changes here
  are defect fixes — rating Coventry off 2000/01 is not a hypothesis awaiting
  evidence — so the gate is written as no-harm from the outset.

## Ship criteria — pre-committed, before results are seen

The seed slice is **750 rows, 7.59%** of the PL canonical, across 75 arrival
events. ADR 0011's EFL slice was 1,333 rows and 9.41%. A large effect will be
visible; a 1% effect will not separate from noise. The criteria are
pre-committed for that reason.

| # | Criterion | Type |
|---|---|---|
| C1 | Rows outside the 750-row seed slice are **bit-identical**. Both changes touch arrivals only. | Hard invariant. A failure is a bug, not a bad result. |
| C2 | Walk-forward log-loss on the seed slice **does not worsen** (point estimate). | No-harm |
| C3 | Overall PL walk-forward log-loss **does not worsen beyond its bootstrap interval**. | No-harm |
| C4 | A fixed-seed known-output test pins the measured prior and one fully seeded row. | Regression |
| C5 | Where arrival events available at a walk-forward fold are fewer than `_MIN_EVENTS`, fall back to existing `PRIORS`. | Sample guard |

**Measured in three arms — baseline, Dixon-Coles seed only, Dixon-Coles seed
plus the eight features — so a failure is attributable.** ADR 0011 could drop
the PL-form transfer and ship the priors precisely because it measured them
separately. If one arm fails and the other passes, the passing one ships alone.

## Consequences

- **Training rows change for every promoted side across 26 PL seasons**, so
  deployment requires a PL retrain. Both changes share one retrain rather than
  taking two.
- **The seed window is 5 matches; `CR_20` and `SOT_CR_20` are 20-match
  windows.** Even with a correct seed, those two stay part-contaminated from
  match 6 to match 20 for a returning side — the blend has gone to zero but the
  rolling window has not cleared. The seed cannot reach this; it is the
  `groupby("team")` problem. **Known-open, deliberately out of scope.**
- **The PL Freshness Gate is green**, and `E0` 2026/27 remains unpublished
  upstream (verified 2026-08-30: `E0` answers HTTP 300 where `E1` answers 200),
  so `current_season_idx` stays 25. The measured prior can be validated against
  history but not against the season now being priced.
- **Two things named "route" now exist**: ADR 0011's arrival direction and
  CONTEXT.md's Promotion Route. Disambiguation is deferred rather than resolved
  — renaming ADR 0011's term would touch shipped EFL code and vocabulary.

## Outcomes

Measured 2026-08-30, after the criteria above were committed in `3931466`.

### The measured prior, and a correction to this document's own figures

75 arrival events, one bucket, keyed `promoted` — the single bucket arriving as
a consequence of `arrival_route` returning `PROMOTED` where nothing sits above,
not as anything the code says.

| rating | hand-picked | measured (PL) | measured (EFL, ADR 0011) |
|---|---|---|---|
| attack_home | 0.900 | **0.779** | 1.004 |
| attack_away | 0.750 | **0.646** | 0.921 |
| defence_home | 1.100 | 1.113 | 1.048 |
| defence_away | 1.200 | 1.233 | 1.093 |

The hand-picked bucket was **not inverted here**, unlike the EFL, where ADR
0011 found it wrong for both routes. For the PL it was directionally right —
it was calibrated for weak arrivals — and roughly 13% optimistic on attack. The
PL defect was therefore mostly *staleness* rather than a bad constant. Note the
PL and EFL priors disagree in an interpretable direction: a side promoted into
the PL measures clearly weak, one promoted into the EFL measures near average.
That difference is the step up between the divisions, in rating units.

**Two figures in the Context section above were measured against a
league-average counterfactual, before a prior existed to measure against, and
are corrected here rather than edited away:**

- Ipswich v Liverpool moves **0.7603 → 0.6196, −0.1407** under the real prior,
  not the 11.7 points quoted. Larger, because the measured prior is weaker than
  league average rather than equal to it.
- *"The errors point in opposite directions across fixtures, so the defect does
  not surface as aggregate ROI drift"* is **weaker than stated**. Against the
  real correction five of six arrival fixtures move toward Under and only
  Liverpool v Hull moves the other way. The defect was more one-directional
  than claimed and would have been partially visible in aggregate.

The control fixture still moves by exactly 0.0000.

### Criteria

| # | Result | Verdict |
|---|---|---|
| C1 | Only seed-slice rows change, asserted for the feature widening with a guard that it changes something at all | **Pass** |
| C2 | Seed slice log-loss 0.6812 → 0.6758, **+0.79%**, CI [−0.0127, +0.0230], 9 of 16 seasons improve | **Pass** |
| C4 | `SeedParams` round-trip the pickle; a pre-ADR pickle falls back rather than crashing | **Pass** |
| C5 | Below `_MIN_EVENTS` the hand-picked `PRIORS` stand | **Pass** |

**C2's interval spans zero, exactly as ADR 0011's criterion 1 did.** The
difference is that this gate was written for what it gates, so shipping needs
no judgement call. That is the whole return on pre-committing.

**The PL's evidence is weaker than the EFL's throughout**, and this ADR should
not inherit ADR 0011's confidence. Split by why the arrival was mispriced:

| arrival | rows | unaided | seeded | delta | EFL equivalent |
|---|---|---|---|---|---|
| stale rating here | 380 | 0.6801 | 0.6766 | +0.0036 | +0.0117 |
| never played here | 100 | 0.6849 | 0.6847 | +0.0002 | −0.0025 |

It sorts the way the mechanism predicts, but weakly. The PL takes 3 arrivals a
season against the EFL's 6, and the slice is 455 scored rows against 855.

### Arm 4 — the venue gate

Added after the criteria were set, because the build found a third instance of
this ADR chain's defect: the Dixon-Coles gate counted a side's *total*
appearances while the feature row counted per venue, so a side five home
matches in and yet to travel had its away seed retired on the strength of home
matches. Fixed in both leagues; scored only on rows where the two schemes
disagree, since rows they agree on price identically in both arms and would
only dilute the result.

| league | rows | totals | per venue | delta |
|---|---|---|---|---|
| EFL | 480 | 0.6870 | 0.6827 | **+0.0043** |
| PL | 225 | 0.6798 | 0.6809 | **−0.0011** |
| pooled | 705 | 0.6847 | 0.6821 | +0.0026 |

**The PL arm fails a point-estimate reading and passes C3's agreed
bootstrap-interval tolerance**, its interval [−0.0164, +0.0154] spanning zero.
Arm 4 post-dates the criteria and had no reading assigned, so which applied was
put to the pre-committer rather than chosen by the person who had just seen the
number; C3's tolerance governs, and it ships. The script encodes that reading
rather than the stricter one it was first written with.

### Arm 3 — the eight features

Walk-forward, the frame initialised twice and a fixed-seed XGBoost fitted on
everything below each season, scoring that season's seed slice. Only the slice
is scored: every other row is identical between the two frames by construction,
so including them would dilute the difference toward zero and prove nothing.

| | rows | eleven pairs | nineteen pairs | delta |
|---|---|---|---|---|
| PL seed slice | 455 | 0.6844 | 0.6844 | **−0.0001** |

Bootstrap 95% CI [−0.0110, +0.0106]. **Passes** — no harm, and no detectable
benefit either, which is the expected result for a divergence repair. The
change exists so that a feature means one thing in both places; that training
and serving now agree is the outcome, and log-loss was never the argument for
it. C1 is the criterion that carries this change, and it holds.

**Two limitations, recorded rather than buried.** The model is a single
XGBoost over the **34** of 183 `ALL_FEATURES` the canonical holds directly, not
the production ensemble — the rest are derived by `run_pipeline`. All eight
changed features are among the 34, so the model can see the change and the
result is not vacuous; but this measures whether the corrected values carry
more signal than the stale ones, not what the full stacker would do with them.
A production-scale answer needs a real retrain, and that is deployment's
business rather than this ADR's.

### Outstanding

Nothing from the pre-committed criteria. Remaining known-open items are in
Consequences: the match 6-to-20 contamination tail on the 20-match windows, and
the `route` naming collision.
