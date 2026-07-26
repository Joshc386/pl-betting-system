# 4. Single canonical artefact; football-data.co.uk as sole Facts source

Date: 2026-07-25

## Status

Accepted

Amends [0001](0001-canonical-dataset-regeneration-criteria.md) (extends the two-class
column model to three).

## Context

Two defects surfaced while auditing model inputs ahead of the 2026-27 restart.

**A rival artefact silently outranked the canonical.** `pipeline.py` preferred
`CompleteDSPL_enriched.csv` over `CompleteDSPL_CSV.csv` whenever the former existed.
That enriched file was rebuilt only by a manual script wired into no automation, so it
drifted to 2026-03-16 while the canonical tracked to 2026-05-24 — roughly ten
gameweeks. All PL training read the stale file. The artefact appeared nowhere in
`CONTEXT.md` and was not covered by ADR 0001's guarantees. EFL never had this problem:
`championship_pipeline.py` reads `csv_path` directly, and `league_config`'s
`enriched_csv_path` key is dead.

**The PL canonical had three provenance regimes.** Seasons 0-23 were bulk-downloaded via
the `soccerdata` package (whose `MatchHistory` reader itself sources football-data.co.uk,
which is why those seasons reconcile almost exactly against `E0` — verified 2026-07-25:
380/380 scores agree for seasons 10 and 20, 379/380 for season 23); season 24 via
`add_season.py --fotmob`; season 25 via `live_updater.py` (Understat). Understat supplies goals and xG only, so its append path hardcodes `NaN`
for shots, shots-on-target, corners, half-time scores, cards and closing B365 odds.
Measured impact on season 25 — the most recent season and `TEST_SEASONS`: of 39 active
features sourced from the canonical, **13 are entirely empty and 13 partially empty**,
including `Away_Past5Goals`, `Away_Past5Corners` and `Away_SOT_CR_20`. This directly
undercuts the Wheatcroft Principle, the documented rationale for those features
existing at all. Nothing errored, because XGBoost and LightGBM absorb `NaN` natively.

No football-data.co.uk builder exists for PL — only `build_championship_dataset.py`
(division `E1`). ADR 0002 refers to "the PL equivalent" build script; that script does
not exist.

## Decision

1. **One canonical artefact per league.** No rival file may take precedence.
   `CompleteDSPL_enriched.csv` and the `pipeline.py` preference are removed.

2. **Three column classes**, extending ADR 0001's two:
   - *Facts* — verbatim from the raw source, byte-identical across regeneration.
   - *Computed features* — derived from Facts; may drift when build logic improves.
   - *Enrichment columns* — joined from a **separate external source** (xG, injury).
     Legitimately sparse; partial coverage is expected, not a defect.

3. **football-data.co.uk is the sole authority for Facts in both leagues** (`E0` for PL,
   `E1` for EFL), for the full rebuild *and* incremental in-season appends. An
   incremental append is sound only when it carries the same columns as a full rebuild.

4. **Understat is demoted to xG enrichment only.** It is not a Facts source.

5. **Facts always win.** Facts and computed features are always written fresh. The
   enrichment join is best-effort: on upstream failure the previous run's enrichment
   columns are carried forward and an **enrichment as-of** date is recorded per source.
   A flaky secondary source must never block ingestion of match results.

6. `build_championship_dataset.py` is generalised on the division code so one builder
   serves both leagues, and PL is rebuilt to backfill seasons 24-25.

## Consequences

- The PL rebuild **will trip ADR 0001's Facts hard-stop by design**: ~26 columns that
  are currently `NaN` for season 25 will gain values. This is an intended improvement,
  not cache corruption. Future readers hitting that alarm must distinguish this case
  from genuine corruption — the discriminator is whether the diff *adds* data to
  previously-null columns or *changes* existing values. Only the latter is a real stop.
- A backup of the canonical and model pickles is required before the rebuild, per
  ADR 0001 — the artefacts are gitignored and git offers no rollback.
- The Understat path in `live_updater.py` is retired for Facts; its xG scrape is kept.
- Both leagues converge on identical provenance and a single build path.
- ADR 0002's reference to a PL build script should be corrected when next touched.
- Carried-forward enrichment is visible as a date rather than being indistinguishable
  from fresh data — a precondition for the Freshness Gate ([0005](0005-freshness-gate.md)).
