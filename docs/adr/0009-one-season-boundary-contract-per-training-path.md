# 9. One season-boundary contract per training path

Date: 2026-08-14

## Status

Accepted 2026-08-14, following a `/grill-with-docs` session the same day, ahead
of the 2026/27 season rollover. This document was written before any code
changed.

All six decisions implemented the same day via `/tdd`, test-first:
`tests/test_season_boundaries.py`, 15 tests. Full suite 756 passed / 1 skipped /
0 failed, against a 741-passed baseline. Verification measured across all four
affected markets — see Consequences. Uncommitted at time of writing.

Decision 5 is **not yet exercised by real data**: with seasons 0-25 complete it
selects the same season the previous rule did. Its first live exercise is the
first weekly retrain after 2026/27 rows land.

Applies to *temporal* boundaries the principle
[0007](0007-one-feature-contract-per-name.md) establishes for feature names and
[0008](0008-one-team-resolution-contract-per-feed.md) for team resolution: one
name, one contract, and where two mechanisms legitimately differ, say so
explicitly rather than letting a reader assume they agree.

## Context

Two code paths train models in this repo. They derive their season boundaries by
completely different mechanisms, nothing says so, and every document in the repo
that described "what the model trains on" was describing the wrong one.

| | Production Path | Research Path |
|---|---|---|
| PL | `predict.py:320 train()` | `model.py:1749 main()` |
| EFL | `championship_predict.py:533 train()` | `championship_model.py:664 main()` |
| Invoked by | `scheduler.py:217`; `scan.py:474`/`487` inline retrain | a human typing `python model.py` |
| Boundaries | **derived from the data** — PL `SeasonIndex >= 14`, EFL `>= 0`; early-stopping season is `max(SeasonIndex)` | **hardcoded** — `config.py:37-39`; `championship_model.py:79` |
| Held-out test | none | yes |
| Writes | `pl_trained_state.pkl` / `efl_trained_state.pkl` — the pickles that price bets | `over_under_model.pkl`, console metrics |

Three separate defects followed from the confusion, all verified 2026-08-14.

**1. The Research Path drops seasons silently.** `pipeline.py:1554` partitions by
`isin(TRAIN)/isin(VAL)/isin(TEST)` — an *allowlist*, so a season named in none is
dropped from all three. `model.py:1769` selects
`>= TRAIN_MIN_SEASON & ~isin(TEST_SEASONS)` — a *denylist*, which includes it.
Both run in the same job on the same data. With `TEST_SEASONS = [25]` and season
26 present, `wf_df` becomes seasons 14–24 **and** 26 with 25 punched out of the
middle; `walk_forward_cv` — whose folds come from `max(SeasonIndex)`, not from
config — then trains a fold at `val_season = 26` whose training window is missing
the most recent complete season, and those OOF rows train the stacker. The
`len(val_df) < 50` guard at `model.py:1280` makes it **time-delayed**: skipped
while season 26 is young, activating silently around the 50th fixture.

`championship_model.py` avoids the *mismatch* by partitioning one scalar with `<`
and `==`, but not the *staleness*: any season above `TEST_SEASON` still falls
outside both. At `TEST_SEASON = 24` with data through 25, season 25 is in neither
`wf_df` nor `test_df`.

**2. The Production Path early-stops on whatever season is newest.**
`predict.py:358-360` sets `last_season = train_seasons[-1]` and uses that season
alone as the early-stopping validation set; `train_seasons[-2:]` sets the Base
Rate. Both are derived from `max(SeasonIndex)`, so **a newly started season
becomes the early-stopping set on its first ingested fixture** — potentially a
dozen August matches deciding XGBoost's tree count, and a Base Rate averaged over
one complete season plus a fragment. This requires no config change and no
deploy. It fires on the first weekly retrain after the new season's rows land.

**3. XGBoost and LightGBM are never refit after early stopping.**
`predict.py:373-374` keeps the model fit on `es_train` — everything *except* the
latest season — while `train_logreg` and `DixonColesPredictor.fit` at lines
376-378 receive the full frame. So the live XGB and LGB have never seen 2025/26
while LR and DC have. `championship_model.py:493-511` already does the right
thing on the Research Path (early-stop for `best_iteration`, then refit on all
non-test data); the Production Path simply never adopted it.

The confusion also produced documentation that was wrong about the live model:
`docs/MODELS_DEEP_DIVE.md` §4 stated LogReg had been dropped from the stacker for
adding noise, citing `model.py:1856` (`oof_df[["xgb","lgb","dc"]]`). The live PL
stacker read from `pl_trained_state.pkl` has `n_features_in_ = 4` with
coefficients **XGB 1.306, LGB 0.820, LR 1.048, DC 1.247** — LogReg outweighs
LightGBM. `predict.py:400`, `backtest.py:309` and `tests/conftest.py:45` all use
four inputs.

## Decision

**1. Name the two paths, and treat "which path" as part of every claim.**
"The model trains on X" is not a well-formed statement in this repo. `CONTEXT.md`
carries **Training Path** and **Early-Stopping Season** as glossary terms.

**2. The config constants belong to the Research Path, and say so.**
`config.py:37-39` and `championship_model.py:79` are Research Path settings.
They are not renamed or deleted — the Research Path is a real path that produces
the backtest numbers decisions are made on — but they carry a comment stating
they do not affect live models.

**3. Roll both leagues' Research Path boundaries forward for 2026/27.**
PL: `TRAIN_SEASONS = 0..24`, `VAL_SEASONS = [25]`, `TEST_SEASONS = [26]`.
EFL: `TEST_SEASON = 24 → 26`, which by derivation makes season 25 the
early-stopping season and folds 24 back into training.

**4. Assert the Research Path partition is exhaustive.** `temporal_split` fails
loudly if any season `>= TRAIN_MIN_SEASON` present in the data falls into none of
`TRAIN`/`VAL`/`TEST`. Three lines, and it converts the silent drop into a message
naming the season. This is the assertion the defect always lacked.

**5. The Early-Stopping Season requires at least 50 fixtures.** Below that, fall
back to the most recent season that clears the threshold. 50 is not a new number
— it is the existing `len(val_df) < 50` fold-skip threshold in
`walk_forward_cv` (`model.py:1280`) and the sibling of
`championship_model.py:613`'s `len(test_df) >= 20`. Reusing it keeps one notion
of "enough fixtures to judge on".

**6. Refit XGBoost and LightGBM on the full training frame** at the
`best_iteration` early stopping discovered, mirroring
`championship_model.py:493-511`. Without this, decision 5 would fix the noisy
stopping signal by paying for it in staleness — XGB and LGB would stay pinned to
the previous season for the first month or two of every season.

Decisions 5 and 6 apply to **both** Production Paths (`predict.py` and
`championship_predict.py`).

### Rejected

- **Deriving the Research Path boundaries from `current_season_idx`.** It removes
  the hand-maintained literal but not the failure: `championship_model.py`
  already derives its early-stopping and pruning seasons from one scalar and is
  *still* stale, because the scalar itself is hand-maintained. Deriving does not
  save you; advancing does. The assertion in decision 4 is what actually
  prevents recurrence, and it is far smaller.
- **Converging the PL onto the EFL's single-scalar partition.** Structurally
  tidier and it makes the allowlist/denylist mismatch impossible rather than
  merely detected — but it rewrites `temporal_split` and `wf_df` on the PL data
  path for a defect confined to scripts a human runs by hand. Revisit if the
  Research Path ever becomes automated.
- **Guarding without refitting (decision 5 alone).** Smallest change to live
  behaviour, and rejected because it makes staleness permanent rather than
  seasonal.
- **Refitting without the threshold (decision 6 alone).** A bad `best_iteration`
  then costs tree count only, not training data — but a dozen August fixtures
  choosing the tree count is still a real way to under- or over-fit.
- **Documenting only.** Rejected: defect 2 fires automatically on the first
  retrain after the season's rows land.

## Consequences

**What changes.** Decision 6 changes what the live PL and EFL models are fit on —
XGB and LGB gain the most recent season. This is a **model behaviour change, not
a Data Refresh**. No strategy parameter is touched: no threshold, blend weight,
Kelly fraction, agreement rule or market multiplier changes.

### Verification (measured 2026-08-14)

Walk-forward, season N held out of *both* training sets so the refit cannot be
flattered by scoring on data it gained. Three folds per market
(N = 23, 24, 25) × two model types = 6 comparisons each. Every market the refit
now touches was measured.

| Market | AUC before → after | Δ | Folds improved | Brier Δ |
|---|---|---|---|---|
| PL O/U 2.5 (staked) | 0.5137 → 0.5283 | **+0.0146** (1.47 SE) | 5/6 | **−0.00140** |
| EFL O/U 1.5 (staked) | 0.5270 → 0.5338 | **+0.0068** (1.89 SE) | 5/6 | −0.00013 |
| EFL BTTS (staked) | 0.5388 → 0.5353 | −0.0035 (0.43 SE) | 4/6 | +0.00025 |
| EFL O/U 2.5 (**Monitored**) | 0.5223 → 0.5208 | −0.0015 (0.32 SE) | 1/6 | +0.00003 |

**Two markets improve, two are flat inside noise, none degrades materially.**
The two positives are the two with the largest measured effects and the most
consistent fold agreement; the two negatives are both under half a standard
error from zero with Brier unchanged to four decimal places.

Worth noting which market came out worst: **EFL O/U 2.5, the one that never
stakes.** It is this repo's only **Monitored Market** — priced and
settlement-tracked for calibration, never producing a Recommendation — so its
1/6 fold agreement carries no capital. Of the three *staked* markets, two
improve and one is noise. An earlier revision of this section flagged the
unmeasured staked EFL markets as the weakest link in the evidence; measuring
them resolved it in the change's favour rather than against it.

A plausible mechanism for the spread is pool size. PL trains from season 14
(~4,200 rows), so one extra season is a large proportional gain; the EFL trains
from season 0 (~14,000 rows), where one more season adds proportionally little.
If that reading is right the refit's benefit shrinks as a league's history
grows — which would make PL, the smaller pool, the market that most needed it.
Not established by four markets; recorded as the hypothesis to test if the
question returns.

Decision 5 is a **no-op against today's data** and could not be measured: with
seasons 0-25 all complete, the Early-Stopping Season is 25 under both the old and
new rules and the Base Rate window is [24, 25] either way. It first takes effect
when season 26 begins, which is the situation it exists for. Tests are therefore
its only evidence until then.

Decision 5 is a **no-op against today's data** and could not be measured: with
seasons 0-25 all complete, the Early-Stopping Season is 25 under both the old and
new rules and the Base Rate window is [24, 25] either way. It first takes effect
when season 26 begins, which is the situation it exists for.

**The rollover now has five sites, not four.** `league_config.py:81` and `:156`
(`current_season_idx`), `api/player_features.py:19` (`SEASONS`,
`SEASON_TO_INDEX`), `config.py:37-39`, and `championship_model.py:79`. Six
previous handoff blocks listed at most three.

**A new season must land in two artefacts, not one.** The PL Production Path
reads the **Enriched Dataset** (`pipeline.py:32` prefers it when present), not the
Canonical Dataset. If the canonical gains season 26 and
`data/build_enriched_dataset.py` does not run, `load_data()` returns a frame with
zero season-26 rows and every boundary above divides data that does not contain
the season — while training completes and writes valid pickles. The EFL is
unaffected only because `CompleteDSChamp_enriched.csv` does not exist, so it
takes the `else DATA_PATH` branch. This asymmetry is recorded in `CONTEXT.md`
under **Enriched Dataset**; closing it is deliberately left as separate work.

**The 50-fixture threshold delays nothing in practice.** PL reaches 50 fixtures
after 5 gameweeks, EFL after ~4. Before that point both leagues early-stop
against the previous complete season, which is strictly better evidence than the
fragment they would otherwise use.

**Accepted cost: the two paths still differ.** After this ADR the Research Path
has a held-out test season and the Production Path does not. That is not
reconciled here, and it means Research Path metrics are not measuring the model
that prices bets. Naming the paths makes the gap visible; closing it is a larger
question about whether production should hold out a test season at all.

**Deliberately not addressed.** The four-versus-three stacker divergence between
`predict.py:400` and `model.py:1856` is documented but not resolved.
`scripts/lr_ablation_test.py` exists to compare the two and its conclusion was
never recorded. Whether the Research Path should adopt the live four-input
stacker — or the live path drop to three — is a strategy question requiring its
own evidence, and is out of scope here.
