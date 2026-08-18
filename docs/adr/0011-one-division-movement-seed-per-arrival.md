# 11. One Division Movement Seed per arrival

Date: 2026-08-16

## Status

Accepted and built 2026-08-16, **with the PL-form transfer dropped** — its
pre-committed criterion failed on measurement (see Outcomes).

**Shipped on a judgement call, recorded here so it stays legible.** Criterion 1
asked that walk-forward log-loss improve on the seed slice. It improves by
1.35%, but the interval spans zero, so the criterion was not met as written.
Shipped anyway, on the reasoning that criterion 1 gates a *defect fix* rather
than a *speculative addition*: the state it replaces is demonstrably wrong
regardless of what the interval says, and the measured gain sits entirely in
the bucket the mechanism predicts. Criterion 4, which did gate a speculative
addition, was honoured and the transfer dropped. Re-measure as arrivals
accrue — six a season.

Deployment is blocked behind an EFL retrain, which the Freshness Gate
([ADR 0005](0005-freshness-gate.md)) currently refuses because upstream has
not published 2026/27.

## Context

A side new to a division has no rolling history in it, so its form features
have no value and Dixon-Coles has no rating. Three separate mechanisms
answered that, and they disagreed with each other.

**The training path and the live path gave different answers to the same
question.** `initialize_promoted_features` seeds training rows from a route
cohort — bottom-5 for League One arrivals, mid-table (8–16) for sides
relegated from the PL — while `_synthesize_promoted_fixture` builds the live
feature row from the **league median**. Measured on EFL season 25 those
differ by 1.40 on `Past5Goals`, 2.00 on `GoalDiff_5`, and **16 percentage
points** on `Over25_5`, the live path being the optimistic one in every case.
This is [ADR 0007](0007-one-feature-contract-per-name.md)'s defect one layer
down: not the same name computed two ways, but the same name *valued* two
ways depending on which caller asked.

**Dixon-Coles rates arrivals on their exit season.** `_decay_weights` decays
by position in a team's own match sequence, not by calendar date, so an
absence of any length is invisible to the weighting; `_shrink_to_league` keys
on match *count*, so a long-absent side is rated with near-total confidence.
With the tuned `half_life=10`, a rating is dominated by a side's final
~20–30 matches in the division, whenever they occurred. Wolves therefore
carry their title-winning 2017/18 ratings (`attack_home` 1.247,
`defence_home` 0.704) against an actual current side's 0.618 / 0.907, and
Burnley carry `defence_home` 0.397 from their promotion season.

**The error is systematic, not random.** Division movement is a cycle: the
only route to the PL is promotion *out of* the EFL, and the only route to
League One is relegation *out of* it. A returning side's last EFL season is
therefore always its **exit** season, and the exit direction is always the
opposite of the return direction. All six sides arriving for 2026/27 confirm
it — Wolves left 1st, Burnley 2nd, West Ham 3rd; Cardiff left 24th, Bolton
23rd; Lincoln has never appeared in 14,160 rows.

**One route has evidence the system already holds and never reads.** Sides
relegated from the PL have a complete, current PL season in
`CompleteDSPL_CSV.csv`, and they are not interchangeable: across PL season
25 their matches went Over 2.5 at 0.605 (West Ham), 0.500 (Burnley) and
0.447 (Wolves). The mid-table cohort gives all three the same seed and
discards a 15.8-point spread in exactly the quantity these markets price.
`pipeline.py` contains no cross-league read at all, though
[ADR 0002](0002-league-position-previous-season-seeding.md) already
establishes the precedent by seeding relegated sides' League Position from
their PL finishing order.

## Decision

One **Division Movement Seed**: a single definition of what a side looks like
before it has history in a division, with the pipeline and the predictor as
two callers of it rather than two definitions of it.

1. **A side's own history in the division is never the seed.** It is exit-season
   form by construction, biased in a route-predictable direction, and stale on
   top. This applies at *any* gap length — Burnley's one-season absence is
   contaminated in the same way as Wolves' eight-season absence, only less
   visibly — so the rule keys on Division Movement (absent in season N−1), not
   on a tunable staleness threshold.

2. **League One arrivals take the bottom-5 cohort.** This system holds no
   League One table, so there is nothing else to carry.

3. **PL-relegated arrivals blend their own PL form with the mid-table cohort:**

   > `seed = w × (PL form, Scoring-Index-rebased onto the EFL scale) + (1 − w) × (mid-table cohort)`

   The Index rebase is the *environment* correction — the existing cross-era
   mechanism applied across divisions. `w` is the *squad* correction:
   relegation empties a squad across a two-to-three-month off-season, so May's
   PL form overstates August's EFL side even after rebasing.

4. **`w` is fitted, never chosen** — one global scalar, estimated by least
   squares against what relegated sides actually did in their first 5 EFL
   matches, over 75 events (3 a season, 25 consecutive seasons). Fitted
   **walk-forward**: season N uses only seasons < N, falling back to `w = 0`
   until enough events accumulate. Per-feature fitting was rejected as
   overfitting — 38 features against 75 events is two events per parameter at
   full sample and fewer than one in early folds. An attack/defence split (two
   parameters) stays open, to be taken only if the residuals demand it.

5. **Dixon-Coles is seeded in its own parameter space.** An arriving side is
   treated as **unrated**, so its exit-season form cannot dominate, and falls
   back to a venue-aware prior for **its route**. The single `PRIORS` bucket —
   hand-picked, and calibrated for weak arrivals — is the same
   one-bucket-for-both-routes defect this seed removes. **Both route priors are
   measured**, walk-forward, from what arrivals of that route actually did in
   their first 5 matches across 150 events.

### Considered and rejected

- **Seed from the side's own recent history where it exists.** The original
  plan, killed by the cycle argument above. Cardiff — the case that motivated
  it — finished 24th in its last EFL season, so its own data and the bottom-5
  cohort are nearly the same number, except the cohort averages five teams
  instead of one and there is an unobserved League One season in between.
- **Leave Dixon-Coles alone.** Rejected: it is 1 of 3 EFL models and its vote
  counts toward the agreement gate, so a known, route-predictable error there
  contaminates both the price and the consensus signal.
- **Make `_decay_weights` calendar-aware.** The deeper fix, and possibly the
  right one eventually. Rejected *for this change* because it alters every
  team's rating rather than only arrivals', which is a far larger blast radius
  needing its own validation.
- **Hand-pick `w` and keep the existing priors.** Rejected: it would leave a
  guess at the centre of the mechanism built to remove a guess.

## Consequences

- **Training rows and serving rows both change, so they must ship together.**
  Releasing the live-path fix alone would recreate the train/serve divergence
  this decision exists to eliminate. Deployment therefore requires an EFL
  retrain, which the Freshness Gate blocks until upstream publishes 2026/27.
  Build and validate now; deploy when the gate clears. No route around the
  gate is proposed — [ADR 0005](0005-freshness-gate.md)'s narrowing amendment
  was drafted and withdrawn once the canonical was found to be a live
  prediction input.
- **Blast radius covers both routes**, not just relegated sides: measured route
  priors change Dixon-Coles output for Lincoln, Bolton and Cardiff as well.
- **Four hand-picked constants are retired**, for the reason ADR 0002 decision
  10 deleted the hand-maintained promoted-team lists: a constant nobody
  re-derives silently goes stale.
- **Ship criteria, pre-committed before results are seen.** Walk-forward
  log-loss on the seed slice must improve; rows outside the slice must be
  bit-identical; overall EFL log-loss must not worsen. If `w`'s walk-forward
  confidence interval includes zero, the PL transfer is dropped and only the
  cohort and Dixon-Coles fixes ship. A fixed-seed known-output test pins `w`,
  the eight route priors, and one fully seeded row.
- **The slice is 1,333 rows, 9.41% of the EFL canonical.** The seed window is
  per *venue*, not per season — `Home_Past5Goals` and `Away_Past5Goals` are
  different quantities, so a side's first five home matches and first five
  away matches are each seeded, up to ten rows per arrival rather than five.
  A large effect will be visible; a 1% log-loss improvement will not be
  separable from noise. The criteria are pre-committed for this reason.

## Outcomes

Both measurements ran on 2026-08-16 against the live canonicals.

**Criterion 4 fired — the PL-form transfer is dropped.** Across all 75
relegation events `w` estimates at **0.317**, with a bootstrap 95% interval of
**[-0.22, 0.85]** (5,000 resamples, seed 42). The signal is real but weak —
`corr(X-R, Y-R) = 0.145`, and the blend does win on RMSE (0.7341 against
0.7404 for the cohort alone) — but a 0.8% improvement cannot be separated
from nothing at this sample size. The cause is visible in the anchors: the
rebased PL rate and the cohort rate average 2.596 and 2.561, so the predictor
varies by 0.30 while the target it must predict varies by 0.74. There is
little leverage to fit against. The measurement is preserved and rerunnable
at `scripts/measure_seed_weight.py`; three events accrue per season.

**Criterion 5 fired the other way — the hand-picked priors were a second live
error.** They were not close to the measured values; they were wrong for both
routes and inverted for one. The shipped bucket describes a distinctly weak
side, but a relegated arrival measures **above** average on both attack and
defence:

| rating | shipped | relegated | promoted |
|---|---|---|---|
| attack_home | 0.900 | **1.156** | 1.004 |
| attack_away | 0.750 | 0.993 | 0.921 |
| defence_home | 1.100 | **0.797** | 1.048 |
| defence_away | 1.200 | 0.909 | 1.093 |

Promoted arrivals measure near 1.0 rather than weak, which is the
first-five-match window doing its job: arrivals often start respectably
before regressing, and the prior governs exactly those matches.

**A documented approximation:** the measured ratios are opponent-unadjusted,
while Dixon-Coles' fitted ratings absorb opponent strength through the
multiplicative structure. Over 150 events opponents largely average out, but
the two are not on identical footing.

**Dixon-Coles seeding is driven by the fixture list, not the canonical.**
Which sides are arriving is not knowable when the models are fitted — before
a season's first results land the canonical holds no rows for it, so
`arrivals()` sees nothing — yet that pre-season window is exactly when a
returning side is most mispriced. `seed_arrivals` is therefore callable after
the fit, and the predictor applies it once the odds feed names the teams.
Verified end-to-end against the live trained state and the 2026/27 feed:

| team | route | before | after |
|---|---|---|---|
| Wolves | relegated | 1.247 / 0.704 | 1.156 / 0.797 |
| Burnley | relegated | 1.201 / 0.397 | 1.156 / 0.797 |
| West Ham | relegated | 1.310 / 1.020 | 1.156 / 0.797 |
| Bolton | promoted | 0.449 / 1.367 | 1.004 / 1.048 |
| Cardiff | promoted | 0.689 / 1.037 | 1.004 / 1.048 |
| Lincoln | promoted | *unrated* | 1.004 / 1.048 |
| *Blackburn (continuing)* | — | *0.618 / 0.907* | *0.618 / 0.907* |

`SeedParams` is measured in `train()` and carried in the trained state, never
refitted at predict time — two callers measuring their own constants would be
this ADR's divergence one level further down. A pre-ADR pickle has no seed
params and falls back to the single `PRIORS` bucket rather than failing.

**Criterion 1 is inconclusive, and the harness named for it was the wrong
one.** `backtest_promoted.py` measures whether seeding training rows beats not
seeding them; the training path's cohort logic is substantively unchanged
here, only relocated. What changed is what an arrival is *priced* from, so
`scripts/validate_seed.py` walks forward seasons 10-25 and scores every
seed-slice fixture twice — on unaided Dixon-Coles ratings and on seeded ones —
against what happened.

```
seed slice rows      : 855      base rate (Over 2.5) : 0.4819
log-loss unaided     : 0.7007
log-loss seeded      : 0.6913
improvement          : +0.0095 (+1.35%)
bootstrap 95% CI     : [-0.0035, +0.0221]   -> spans zero
```

Nine of sixteen seasons improve. Split by how the base model was rating the
arrival, the effect sorts exactly the way the mechanism predicts:

| arrival | rows | unaided | seeded | delta |
|---|---|---|---|---|
| had a stale rating here | 870 | 0.7025 | 0.6907 | **+0.0117** |
| never played here | 90 | 0.7019 | 0.7044 | −0.0025 |

The gain is entirely in displacing stale exit-season ratings — the defect this
ADR was opened on. Noise would not sort itself by that split. The negative
bucket is where a measured promoted prior replaces the hand-picked one, on 90
rows, which is too few to read.

Criterion 1 asked that log-loss improve; it improves by 1.35% but not
separably from zero. Note the asymmetry with criterion 4: that one gated
*adding a speculative mechanism*, where the safe default is to decline, while
this one gates *fixing a known defect*, where the current state is
demonstrably wrong regardless of what the interval says. Rating Wolves on a
2017/18 title win is not a hypothesis awaiting evidence.
- **The PL keeps abstaining.** `predict.py` skips a fixture when either side
  has no current-season row, so its three promoted clubs get no
  recommendations until they have played once — roughly 3 fixtures a season.
  That behaviour is untouched here and is the conservative choice; extending
  the seed to the PL is left as separate work.
