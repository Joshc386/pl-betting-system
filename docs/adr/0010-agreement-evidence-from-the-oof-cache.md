# 10. Model-agreement evidence comes from the OOF cache, not the backtest runners

Date: 2026-08-15

## Status

Accepted

## Context

The `min_agree = 2` gate has never been tested. It is applied on every
Recommendation in both leagues, and the two samples that speak to it disagree:
the walk-forward backtest showed hit rate rising with agreement (61.5% at 4/4
vs 50.0% at 2/4, 98 bets), while live settled Recommendations showed the
reverse (34.5% at 4/4, 75.0% on the DC-only alt-line row, 8 bets). Every live
bin is 🔴 Noise, so neither sample settles anything.

The planned answer was a **backtest bet harvest**: call the five walk-forward
runners with shared pipeline data and shared Dixon-Coles tunes, concatenate
their bets to CSV, and bin by `n_agree`. Roughly 50 minutes per run, dominated
by three `tune_dc_params` calls.

Two facts made that the wrong substrate.

**The runners emit post-gate bets only.** `if ev <= 0 or edge < active_min_edge
or n_agree < min_agree: continue` runs before `bets.append`, so bins `0/N` and
`1/N` cannot exist in the output. A harvest of runner bets can ask whether the
gate should be *higher*; it structurally cannot ask whether it should be
*lower*, which is half the question.

**The evidence already existed.** `reports/roi_validate/oof_cache/` holds six
parquet caches — one per (league, market) cell — each carrying `xgb_prob`,
`lgb_prob`, `dc_prob`, `lr_prob` per fixture, both sides' odds with the book
named, and the outcome. These are *pre-gate* rows. `lr_prob` is null for all
three EFL cells and populated for all three PL cells, matching the 3-model and
4-model ensembles. `scripts/roi_validate.py` already calibrates them
(`_calibrate_single` with the per-season `*_shift`), de-vigs
(`_implied_fair_prob`) and applies `decide_bet`.

## Decision

**Agreement evidence comes from the OOF cache.** The dashboard's historical
agreement section reads the six caches, applies the gate itself, and reports
both a pre-gate view (all agreement levels) and a post-gate view (the bins the
live gate actually bets). The backtest-runner harvest is not built.

**The headline statistic is Realised Edge**, `mean(won) − mean(fair_prob)`, not
hit rate and not ROI. Hit rate is dominated by the market's Base Rate — PL O/U
1.5 Over wins ~75% of the time — so it cannot be compared across the markets
this table necessarily spans. Realised Edge scores an unskilled bet at ~0 in
any market. It is also not an ROI estimand, so the historical-`_first`-price vs
best-of-14-books mismatch does not apply. Claimed **Edge** is shown beside it;
the pair is the finding.

**Confidence intervals are fixture-clustered bootstrap.** One fixture yields
several bets across markets, and O/U 2.5 Over and O/U 1.5 Over on one match are
logically nested — if the total beat 2.5 it has already beaten 1.5. Resampling
fixtures rather than rows keeps the interval honest under that correlation.
`wilson_ci` and `edge_analytics._bootstrap_roi_ci` both assume independent
trials and are not used here.

**The bet universe is pinned to live staked config**, not to runner defaults —
`ALLOWED_ALT_LINES` and `EFL_ALLOWED_ALT_LINES`. This is recorded because the
runner defaults diverge sharply: `efl_alt_lines_backtest` defaults to
`EXPLORATORY_CONFIG`, which evaluates lines 0.5/1.5/2.5/3.5/4.5 both sides with
`best_line_only=False`, against a live EFL alt-line set of `{3.5}`.

## Consequences

- **The pre-gate view is antisymmetric and must be presented one side per
  fixture.** Evaluating both sides forces `n_agree(a) + n_agree(b) = n_models`,
  so the `0` and `N` bins are the same fixtures seen from opposite sides and
  their Realised Edges are exact negatives. PL's middle `2/4` bin is
  self-mirroring and returns precisely `+0.00%` with a zero-width interval.
  That is arithmetic, not evidence, and it will read as a finding to anyone who
  does not know.

- **Keeping one side per fixture is not merely hygiene — it changed the
  finding.** Pooled over both sides, PL O/U 2.5 unanimity read +4.33% on 726
  rows with an interval excluding zero. Split, that bin is 334 rows of *all
  four back Over* at **+2.56% (not significant)** averaged with 392 rows of
  *all four back Under* at **+5.85% (significant)** — a real signal and a
  non-signal reported as their mean.

  Across all six cells, **three bins have intervals excluding zero and every
  one is a bin-0 — unanimous opposition, all negative**: PL O/U 2.5 −5.85%,
  PL BTTS −6.18%, EFL O/U 1.5 −2.54%. **No unanimous-support bin clears zero.**
  The defensible claim is therefore that the ensemble is dependable about what
  *not* to back and not yet demonstrably dependable about what to back — which
  the two-sided form cannot express, since it forces the `0` and `N` bins to be
  exact negatives. Pinned by `tests/test_agreement_analysis.py`.

- **The table is per-cell, never pooled.** Realised Edge is base-rate neutral,
  so pooling markets is arithmetically legal — but it destroys the finding.
  Agreement pays in PL O/U 2.5 (+4.33%) and PL BTTS (+5.72%) and in no other
  cell; pooling PL's three markets averages those against O/U 1.5's null and
  reports +2.15%, which reads as a weak system-wide effect rather than a strong
  two-market one. The four null cells are well-powered, not thin: EFL O/U 1.5
  returns +0.49% on 3,301 unanimous rows.

- **The caches carry their own staleness, but less than feared.** They are dated
  23–24 April 2026 and therefore predate ADR 0007's retrain (3 Aug) and ADR
  0009's refit (14 Aug). `PL ou25` was regenerated on 2026-08-15 to test whether
  that mattered: pre-gate unanimity moved from **+4.33%** to **+4.22%**, 11
  basis points, with near-identical intervals. The remaining five cells were
  therefore left un-regenerated. Baseline preserved at
  `reports/roi_validate/oof_cache_bak-20260815-april-baseline/`, probe at
  `oof_cache_probe/`.
  A cache is invalidated by a Canonical Dataset change, a staked-config change,
  or a code change — **not** by a **Data Refresh**, because the walk-forward
  path fits its own models per season and never loads the production pickles.
  This is *not* a **Freshness Gate**: nothing is blocked, the section is
  stamped.

- **Regeneration is expensive, and more expensive than the route rejected.**
  `scripts/generate_oof_cache.py` pays `run_pipeline()` and `tune_dc_params()`
  per cell, so all six cost roughly 90 minutes against the harvest's 50. The
  pivot is justified by what the substrate can answer, not by cost.

- **The fair-price basis differs from the runners'.** The caches carry
  `bookie_a` per row — Bet365 for both `ou25` cells, Betfair for `efl_btts`,
  footiqo for `pl_ou15` — while the runners de-vig the canonical
  `Odds_Over_{line}` column. Figures from the two are not interchangeable, which
  is a specific instance of the rule in `CONTEXT.md` that a per-market ROI must
  always name its method.

- **The uncommitted `data=` / `dc_kwargs=` injection parameters on the four
  backtest runners are not orphaned by this decision**, but they lose their
  original justification. Their remaining value is fixing the double full
  backtest at `edge_analytics.py:601`, which re-runs because a comment claims
  `run_backtest` "returns None but prints" — it does not; it returns
  `(total_bets, all_metrics, cumulative_bankroll)`. They should be justified on
  that basis or dropped.

- **"Phase 4b" is now overloaded.** `config.py:758` already defines it as a
  re-spin of the Phase 4a matrix, triggered when a `PHASE_4A_BASELINE_ROI` cell
  drifts >3pp below baseline for two consecutive months. That is a different
  piece of work from the agreement analysis this ADR describes.
