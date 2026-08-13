# 5. Freshness Gate on Data Refresh and Recommendations

Date: 2026-07-25

## Status

Accepted. **Specification sharpened 2026-08-06 via `/grill-with-docs`, still
unimplemented.** Seven decisions were pinned down before code; three of them
*correct* what this ADR originally said, so the Decision and Consequences
sections below are annotated rather than left to mislead:

1. **The authority is ESPN, with football-data.org as an ordered fallback** — not
   football-data.org alone. ESPN answers in each canonical's own name format, so
   the gate carries no team resolver at all. Both sources agree exactly where
   measured (2026-05-11→24: PL 22, EFL 3 finished, identical). Strictly ordered,
   never cross-checked: a disagreement would need semantics of its own, and
   there is no evidence of one.
2. **No date-heuristic fallback exists.** When neither authority answers, the
   verdict is `UNKNOWN` and `UNKNOWN` blocks. See the corrected bullet below.
3. **The window is 14 days**, sized to the gate's own run cadence rather than to
   the fixture calendar.
4. **The match key is exact `(Date, Home_Team, Away_Team)`.** No date tolerance.
5. **Failure raises `FreshnessError`; it never returns an empty list.**
6. **No bypass flag and no known-missing ledger**, on the evidence that seasons
   16–25 are complete in both canonicals (380 PL / 552 EFL, zero missing scores).
7. **`PL_RETRAIN_ENABLED` / `EFL_RETRAIN_ENABLED` are NOT retired.** See the
   corrected consequence below.

Full vocabulary in `CONTEXT.md`, "Freshness Gate".

## Context

The system had no assertion that its model inputs were current. A stale canonical
(see [0004](0004-canonical-composition-and-facts-provenance.md)) trained the PL models
for roughly two months without producing a single error, warning or dashboard signal.

Every available staleness signal is individually unreliable:

- **Date heuristics cannot distinguish "no fixtures were played" from "ingestion
  broke."** An international break and a dead ingestion job look identical to a
  `max(Date) >= today - N` check. This ambiguity is precisely why the off-season
  retrain pause had to be a *manual* boolean (`PL_RETRAIN_ENABLED`) rather than
  something the system could infer.
- **Missing enrichment degrades silently.** `pipeline.py`'s `add_xg_features` returns
  early when the `home_xg` column is absent, and downstream lambda computation uses
  `df.get(..., pd.Series(dtype=float))`. No exception is raised; XGBoost absorbs the
  resulting `NaN` and returns a confident-looking probability.
- **`max(date)` is not even trustworthy on the files themselves.** `betfair_goal_ou.csv`
  contains 8 Betfair sandbox rows dated 2030 ("Test T087 v Test T088"), which would make
  a max-date freshness check pass forever.

The blast radius of a single missing fixture is wider than it first appears. Rolling
form features are contaminated per-team, but league-table features
(`LeaguePosition_Diff`) shift the ranking of teams that never played in the missing
fixture. Gate granularity must therefore match the *widest* feature's blast radius.

## Decision

A **Freshness Gate** is a hard precondition on both producing **Recommendations** and
running a **Data Refresh** for a league.

- **Measurement is authoritative, not heuristic.** The gate asks an authoritative
  fixture list for the finished fixtures in a rolling window and asserts every one is
  present in that league's Canonical Dataset. Zero finished fixtures is an unambiguous
  pass; a missing finished fixture is an unambiguous defect.

  *Corrected 2026-08-06 — this bullet originally named football-data.org as the sole
  authority and specified a date heuristic as its fallback. Both were changed.* The
  authority is **ESPN first, football-data.org second**: ESPN returns names already in
  each canonical's format, whereas football-data.org returns `'Hull City AFC'` against
  an EFL canonical holding `'Hull'`, so it would need a resolver inside the gate — and
  an unmapped promoted club returns `None`, forcing the gate to either skip the fixture
  (fails open) or flag it missing (fails closed on a naming bug rather than a data one).
  Neither is acceptable in a gate. Note ESPN is also Settlement's feed; the accepted
  cost is that one CDN policy change can blind both, which is why the fallback is a
  different provider rather than a retry.

  **The date-heuristic fallback is removed entirely.** It cannot answer the gate's
  question — an old `max(Date)` *is* the off-season-versus-dead-ingest ambiguity this
  ADR exists to resolve — and this document's own Context section records that
  `max(date)` is untrustworthy on these files (`betfair_goal_ou.csv` carries 8 sandbox
  rows dated 2030). Specifying it anyway was an error. `UNKNOWN` now blocks, which is
  cheap because it requires simultaneous failure at two independent providers, by which
  point the odds fetch has also failed and there is nothing to bet on.

- **The window is 14 days**, sized to the gate's run cadence, not the fixture calendar
  — the window never needs to span a break, since zero finished fixtures passes by
  construction. Its only job is keeping a failed ingest visible until the gate next
  runs, and the binding cadence is the weekly Sunday Data Refresh, so 14 days gives 2×
  margin: one missed Sunday still leaves the fixture in view where 7 days would forgive
  it permanently. Not wider, because an unfixable upstream gap blocks betting for the
  window's length.

- **A fixture matches exactly on `(Date, Home_Team, Away_Team)`.** ESPN reports kickoff
  in UTC and the canonical holds the UK local date; for English domestic football those
  cannot differ (kickoffs top out near 20:15 local against a UTC/UTC+1 offset), measured
  132/132 exact. Deliberately unlike the Betfair League Split's `±1 day`, whose
  tolerance is a property of that feed and would here mask a fixture ingested under the
  wrong date.

- **A blocked gate raises `FreshnessError` and names every missing fixture** — date and
  both teams, never a bare count. It must never return an empty recommendation list,
  which is indistinguishable from a quiet Tuesday; that exact substitution of "could not
  determine" for "nothing here" was found in three live code paths on 2026-08-06
  (`fixture_schedule.py:74`, `api/espn_scores.py:170`, and a sibling project's ingest).
  With no bypass flag, the message is the only route to action.

- **There are three `train()` sites, not one.** Both leagues in `job_weekly_retrain`
  *and* `scan.py:474` / `scan.py:487`, where a matchday scan retrains inline whenever
  `load_trained_state()` fails. This ADR did not anticipate that a scan is also a Data
  Refresh. The gate is checked inside `generate_recommendations()` — one place covering
  all six recommendation call sites — *and* before every one of those three trains,
  because the inner check fires after training has already baked in the staleness.

- **On failure, that league's Data Refresh and Recommendation output are both blocked.**
  Blocking Recommendations alone would still bake staleness into a retrained model and
  persist the damage after the gate goes green.

- **Settlement and dashboard display continue.** Neither reads the Canonical Dataset.

- **The gate is league-wide, never per-fixture**, per the league-table blast radius above.
  The other league is unaffected — the leagues have independent canonicals, pipelines
  and databases.

## Consequences

- The gate can stop you betting. This is accepted: for live capital, failing closed on
  unverifiable inputs is the correct direction, and the alternative (warn-and-proceed)
  is exactly what failed silently for two months.
- ~~Off-season handling becomes automatic. "No finished fixtures" passes cleanly, so the
  manual `PL_RETRAIN_ENABLED` / `EFL_RETRAIN_ENABLED` flags become redundant and should
  be retired once the gate is live — removing a step that must be remembered twice a
  year.~~

  **Withdrawn 2026-08-06. This consequence does not hold.** The gate and the flags
  answer different questions: the gate asks *"is anything missing?"* and in the
  off-season passes — nothing finished, so nothing to miss — while the flags ask *"is
  there anything new?"* and skip. Retiring the flags would not automate the pause, it
  would delete it, starting a full two-league retrain every off-season Sunday on data
  that has not moved. The flags therefore **stay**, and the twice-a-year step this ADR
  promised to remove remains.

  Half of their stated rationale is also wrong: `config.py:501` cites *"model drift on
  stale features"*, but `model.py` seeds base models, stacker and permutation RNG at
  `random_state=42`, so an identical-input retrain is deterministic. Wasted CPU on a job
  that rewrites the live pickles is the real cost.

  The check that *would* genuinely retire them is **"did the Canonical Dataset gain rows
  since the last retrain?"** — no calendar knowledge required. Left as separate future
  work rather than shipped alongside the gate, so that when something blocks a retrain it
  is unambiguous which mechanism did it.

- A new runtime dependency on the ESPN scoreboard sits on the path to producing
  Recommendations, with football-data.org as an ordered fallback. Not mitigated by any
  heuristic — see the corrected Decision bullet. ESPN is already Settlement's feed, and
  it CDN-blocked a sibling project on 2026-08-05, so this dependency is known-fragile
  and the fallback provider is the mitigation.
- The gate needs somewhere reliable to run; see [0006](0006-task-scheduler-for-data-critical-jobs.md).
