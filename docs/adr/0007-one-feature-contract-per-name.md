# 7. One feature contract per name, across leagues

Date: 2026-07-26

## Status

Accepted. Implementation in progress — decisions **1 and 2 landed 2026-07-28**
(`Promoted`/`Relegated` derived from the Canonical Datasets; both builder dicts
deleted). Decisions 3–10 remain pending.

The derived flags reach the models only once the canonicals are rebuilt and
republished, which happens with the retrain. Until then the published
canonicals still carry the dead hand-maintained flag, and
`tests/test_cross_league_features.py` correctly still reports `Promoted` as a
divergence — it reads the published artefact, not the builder.

`PROMOTED_TEAMS` in `data/add_season.py` is deliberately still present: it is
consumed by that file's own feature code, which decision 10 deletes as a unit.
Removing the dict alone would leave the module broken for no gain.

The pre-change baseline — artefact fingerprints, and which models train on
which diverging features — is recorded in
[adr0007_baseline.md](../adr0007_baseline.md).

Amends [0002](0002-league-position-previous-season-seeding.md) (replaces the
hand-maintained `PROMOTED_TEAMS` with derivation, and settles the neutral-seed
fallback). Follows from [0004](0004-canonical-composition-and-facts-provenance.md),
which converged both leagues on one builder.

## Context

`ALL_FEATURES` has 182 entries, but only the 39 in `EXISTING_FEATURES` are read
straight from a Canonical Dataset. The other 143 are computed in `pipeline.py` at
run time and are reproducible by construction. `pipeline.py` recomputes none of the
39 — its only touch is `LeaguePosition_Diff` (`pipeline.py:303`). Those 39 are
therefore frozen at whatever the build script wrote.

Two build scripts wrote them. `data/add_season.py` produced the PL canonical;
`data/build_canonical_dataset.py` produced the EFL one. Same column names, two
independent implementations, nothing forcing them to agree. **15 of the 39 diverge**,
measured over seasons 5–23 where both leagues have full coverage:

| Feature | PL | EFL |
|---|---|---|
| `ShotRatio_5` | Σshots ÷ ΣSOT — **2.561** | ΣSOT ÷ Σshots — **0.411** |
| `DefensiveStrength_5` | 1 ÷ Σshots conceded — **0.017** | SOT-ag ÷ shots-ag — **0.410** |
| `DefensiveStrength_SOT` | 1 ÷ ΣSOT conceded — **0.043** | goals-ag ÷ SOT-ag — **0.293** |
| `Home/Away Factor` | rolling-10 mean goals — **1.355** | goals ÷ league avg — **0.999** |
| `H2H_HomeWins/AwayWins/Draws` | last **5** meetings | **all** meetings (max 33) |
| `Home/Away_Promoted` | — | — |
| `Local/Historical Derby` | exact tuple match | fuzzy substring match |

Three findings make this more than cosmetic drift.

**`Promoted` is a dead feature.** It is constant zero across **24 of 26 PL seasons**
and **21 of 26 EFL seasons**. `PROMOTED_TEAMS` (`add_season.py:22`) and the builder's
`_PL_PROMOTED_TEAMS` contain entries for seasons 24 and 25 only; EFL's list covers
more seasons but stores football-data.co.uk short forms matched by substring, so
nearly all miss. The only seasons where the feature varies are the two most recent,
so any effect the model learned was fit on 2 seasons while the other 24 asserted that
promotion never happens.

**`ShotRatio_5` never matched its own name in the PL.** The comment at
`add_season.py:161` reads "total shots / shots on target (original formula from
Functions.py)". `Functions.py` no longer exists in the repo.

**PL carried a duplicate column undetected.** Because PL capped all H2H at 5 meetings,
`H2H_AvgGoals_5` and `H2HAvgGoals` computed identically there (2.6373 vs 2.6339). Two
implementations were never compared, so nobody noticed.

The `league_config.py` derby sets (`:58`, `:111`) are read by **no code anywhere** — a
third list that reads like the authoritative one.

## Decision

**A feature name is a contract. The same name means the same computation in both
leagues.** Where a league lacks the data, the feature is null — never a substitute
formula under the same name.

1. **`Promoted` and `Relegated` are derived from the Canonical Datasets, never
   hand-maintained.** A team in season N but not N−1 is new to the division;
   if it was in the division above in N−1 it was relegated, else promoted. Both
   hardcoded dicts are deleted. Verified: PL gains exactly 3 teams every season
   s1–s25, EFL exactly 6 (3 down, 3 up), and seasons 24–25 reproduce the hand-written
   lists exactly in both leagues.

2. **`Relegated` is added as a separate flag.** A side relegated from the PL is one of
   the *stronger* teams in the Championship; a side promoted from League One is one of
   the weakest. Leaving relegated sides unflagged made them indistinguishable from
   ordinary returning teams. For the PL, `Relegated` is always 0, so both leagues keep
   an identical schema.

3. **Promotion Route is derived for the PL and falls back to a neutral seed for the
   EFL.** PL's route reads off the EFL canonical's final table for season N−1 —
   1st = champion, 2nd = runner-up, remaining promoted side = play-off winner —
   verified against all 25 seasons. EFL's promoted teams come from League One, which
   this system does not hold. League One (`E2`) *is* available from
   football-data.co.uk, and was deliberately not adopted: it would pull ~40 unmapped
   clubs through `normalize()`, whose silent substring fallback is the same failure
   mode that hid the Bradford City gap, to derive a signal affecting three teams'
   matchday-1 seed. This is one code path with a documented data-availability
   fallback, **not** a second implementation.

4. **`DefensiveStrength` becomes three named components, computed in `pipeline.py`.**
   *Shot Suppression* (volume allowed, opponent-adjusted), *Chance Quality Allowed*
   (SOT ÷ shots against), *Conversion Allowed* (goals ÷ SOT against). The EFL builder
   already computed the latter two correctly under a misleading name; the PL formula
   was never a defensive metric at all, since the reciprocal of a count scales with
   window length rather than defensive quality. It moves out of the canonical because
   opponent-adjustment needs Elo and the richer variants need enrichment — neither
   exists at build time, and building them in would violate ADR 0004's Facts-only rule
   for computed columns.

5. **Defensive data is tiered, with separately named features per tier.**

   | Tier | Source | Seasons | Leagues |
   |---|---|---|---|
   | 1 — Facts | canonical | 0–25 | PL + EFL |
   | 2 — team xG conceded | Understat | 14–25 | PL only |
   | 3 — player & GK (`xgot_faced`, `goals_prevented`, minutes, positions) | FPL-Core-Insights | 24–25 | PL only |

   The EFL runs on tier 1 alone. Tier 3 reaches ~7% of training rows and is
   **provisional**: it ships only if a walk-forward comparison on the final fold
   improves AUC/Brier.

6. **H2H keeps the goal averages and drops the win counts.**
   `H2H_HomeWins`/`AwayWins`/`Draws` are removed: correlation with total goals sits at
   ±0.01–0.045, and `H2H_HomeWins` has *opposite signs* across the leagues (+0.045 PL,
   −0.012 EFL). They also encode match result, which no market this system bets asks
   about. `H2HAvgGoals` is kept — it correlates +0.051 (PL) / +0.022 (EFL) with total
   goals and **retains +0.047 / +0.021 once team form is partialled out**, so it is not
   a restatement of recent form. Both leagues use the same window.

7. **`Factor` is retired and replaced by two features.** `ScoringRate_10` (rolling-10
   mean goals) and `ScoringIndex_10` (that rate ÷ the season's league average). They
   correlate at 0.987 (PL) / 0.990 (EFL), so the Index adds little *within* a season —
   its value is cross-era comparability. League scoring drifts **33.6%** across PL
   seasons (1.225 goals per team per game in season 6, 1.637 in season 23) and the
   model trains across all 26 at once, so the same raw rate means different things in
   different eras. "Factor" is retired because it described nothing and denoted the
   Rate in one league and the Index in the other.

8. **`ShotRatio_5` is SOT ÷ shots** (EFL's formula). **`ShotsPerGoal_5` adopts the
   builder's formula**, which reproduces both stored canonicals (8.42 EFL, 8.48 PL)
   where `add_season.py`'s current code produces 11.42 — evidence that `add_season.py`
   is not what wrote the stored PL values.

9. **One derby list, one matcher.** The authoritative pairs live in `league_config.py`,
   read by the shared builder with exact matching against canonical names. Fuzzy
   substring matching is dropped: it is the same silent-failure mode as `normalize()`.

10. **`add_season.py`'s feature code is deleted.** Both leagues build through
    `data/build_canonical_dataset.py`. This is the decision that prevents recurrence;
    the other nine are consequences of having had two implementations.

## Consequences

- `EXISTING_FEATURES` goes from 39 to 38: three H2H win counts out, two `Relegated`
  in, one `Factor` pair becomes two, `DefensiveStrength`'s two columns leave the
  canonical for three-plus in the pipeline.
- **Both leagues need a rebuild and a retrain**, not just PL. The EFL's `Factor`,
  `H2H_*` and derby columns all change meaning. This is a feature-schema change, not a
  Data Refresh as `CONTEXT.md` defines it, and does not qualify for that approval path.
- Renaming `"Home Factor"` / `"Away Factor"` (note the space) touches `config.py:45`,
  `championship_pipeline.py:693`, both builders, and `tests/test_feature_audit.py`.
- The cross-league comparison that produced these findings becomes a **permanent
  test**: for every shared feature, PL and EFL distributions must agree within a
  tolerance, so a future divergence fails CI instead of sitting undetected. All 15
  defects here would have been caught on day one by that test.
- `scripts/run_feature_audit.py` already runs an end-to-end audit with a noise
  baseline; it is the existing tool for the tier-3 gate in decision 5 and for the
  post-change validation, so no new validation infrastructure is needed.
- ADR 0002 becomes implementable: it required `PROMOTED_TEAMS` to encode route per
  team per season, which decisions 1 and 3 supply for the PL and explicitly waive for
  the EFL.
- These changes are independent of the Facts merge that repairs five wrong season-24
  scores and the empty season 25. That merge should land **first**, on current
  definitions, so the corrupt Facts stop propagating while the feature work proceeds
  as a separate change with its own retrain and its own rollback point.
- The five features whose PL values came from the vanished `Functions.py` are now
  defined by code in the repo, so the canonical becomes fully reproducible for the
  first time.
