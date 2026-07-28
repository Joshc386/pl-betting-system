# ADR 0007 — pre-change baseline

Recorded 2026-07-28, before any feature-contract change. This is the "before"
side of the comparison that gates the ADR 0007 retrain.

## Artefacts pinned

| Artefact | Fingerprint |
|---|---|
| `CompleteDSPL_CSV.csv` | 9,880 rows × 72 cols — sha256 `edcd10c59223d604…` |
| `CompleteDSChamp_CSV.csv` | 14,160 rows × 72 cols — sha256 `10ffa345660c6c32…` |
| `models/pl_trained_state.pkl` | trained 2026-07-27, O/U 174 features, BTTS 117 |
| `models/championship/efl_trained_state.pkl` | trained 2026-07-27, O/U 126, O/U 1.5 126, BTTS 83 |

Base rates at training time — PL O/U 0.5579, BTTS 0.5671; EFL O/U 0.4755,
O/U 1.5 0.7364, BTTS 0.5353.

## Blast radius, measured

How many of the 15 diverging features each live model actually trains on:

| Model | Diverging features in use |
|---|---|
| PL O/U 2.5 | **15 / 15** |
| PL BTTS | 10 / 15 |
| EFL O/U 2.5 | **15 / 15** |
| EFL O/U 1.5 | **15 / 15** |
| EFL BTTS | 6 / 15 |

No model is unaffected. The three O/U models carry every divergence, which is
why ADR 0007 requires both leagues to be rebuilt and retrained rather than PL
alone.

## AUC / Brier — deliberately not run here

No stored walk-forward metrics artefact exists, and generating one costs a full
CV run that the current phase does not otherwise need. It is not lost: the
canonicals and the code that produced them are both committed, so the
pre-change metrics are reproducible at any point by building from this commit.

Reproduce with `scripts/run_feature_audit.py`, which already runs an end-to-end
audit with a noise baseline. Do this as the first step of the retrain phase, so
the "before" and "after" numbers come off the same code path.

## Guard in place

`tests/test_cross_league_features.py` compares every shared canonical feature
across leagues and currently reports **24 passed, 15 xfailed** — the 15 being
exactly the divergences above. Each is `xfail(strict=True)`, so a fix turns into
a suite failure until its exemption is removed. Verified by temporarily
exempting an already-agreeing feature (`Home_CR_5`), which failed as
`XPASS(strict)` as intended.
