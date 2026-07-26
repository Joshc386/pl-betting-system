# 5. Freshness Gate on Data Refresh and Recommendations

Date: 2026-07-25

## Status

Accepted

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

- **Measurement is authoritative, not heuristic.** The gate queries football-data.org
  for `status=FINISHED` fixtures in a rolling window (both `PL` and `ELC`; free tier,
  already integrated, separate from the odds quota) and asserts every one is present in
  that league's Canonical Dataset. Zero finished fixtures is an unambiguous pass; a
  missing finished fixture is an unambiguous defect. A date heuristic is used **only**
  as a fallback when football-data.org is unreachable.

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
- Off-season handling becomes automatic. "No finished fixtures" passes cleanly, so the
  manual `PL_RETRAIN_ENABLED` / `EFL_RETRAIN_ENABLED` flags become redundant and should
  be retired once the gate is live — removing a step that must be remembered twice a year.
- A new runtime dependency on football-data.org sits on the path to producing
  Recommendations, mitigated by the heuristic fallback.
- The gate needs somewhere reliable to run; see [0006](0006-task-scheduler-for-data-critical-jobs.md).
