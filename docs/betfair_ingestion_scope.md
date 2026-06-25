# Betfair Historical Ingestion — Pipeline Scope

_Scoping note from 2026-06-25 session. Describes the historical Betfair odds
ingestion pipeline, its current state, and known issues. Not a decision record
(see `docs/adr/` for those); this is an architectural map + findings._

## Architecture: two tracks + a split layer

```
TRACK A — one-time bulk backfill (manual)
  download_betfair_goals.py ──▶ raw .tar streams  ──▶ extract_goal_odds.py  ──▶ data/betfair_goal_ou.csv
   (betfairlightweight, certs)   BetFairData_OU/        extract_btts_odds.py ──▶ data/betfair_btts.csv
   downloads Basic-plan bz2       (GB, OU 0.5–4.5)       parse ltp ticks → CSV

TRACK B — monthly incremental (automated: Task Scheduler "Betting Bot Monthly
          Betfair Update", 5th @ 10:00, catch-up enabled)
  monthly_betfair_update.bat ──▶ betfair_monthly_update.py
     ├─ logs in (cert-based), then RAW HTTP to historicdata.betfair.com
     │   (bypasses betfairlightweight.get_file_list(), which sends snake_case
     │    strings instead of camelCase arrays → 400s)
     ├─ downloads PREVIOUS month: OVER_UNDER_15, OVER_UNDER_25, BOTH_TEAMS_TO_SCORE (GB)
     ├─ appends (deduped) to betfair_goal_ou.csv + betfair_btts.csv
     ├─ re-runs the SPLIT LAYER below
     └─ skips June/July (off-season); propagates split failures as non-zero exit

SPLIT LAYER (derived per-league files; idempotent regenerators, no download)
  extract_btts_by_league.py    ──▶ betfair_pl_btts.csv + betfair_efl_btts.csv
  extract_efl_ou15_betfair.py  ──▶ betfair_efl_ou15.csv
     join: master CSV  →  Canonical Dataset (CompleteDSPL_CSV / CompleteDSChamp_CSV)
           on (canonical home, canonical away) + ±1 day date window, SeasonIndex ≥ 16
           team names mapped via per-league allowlists; unmapped names dropped (silent-but-safe)
```

Key property: **Track B does not read Track A's tar archives** — it pulls each
month fresh from Betfair's API. The two tracks meet only at the shared master
CSVs (`betfair_goal_ou.csv`, `betfair_btts.csv`).

## Data captured

Betfair **Basic plan** stream files (`fileType=M`) carry **last-traded-price
(`ltp`) ticks only** — extracted as `*_ltp_first` (first trade), `*_ltp_pre`
(last pre-in-play = closing line), `*_ltp` (last trade overall). They do **NOT**
carry traded volume (`tv`) or the back/lay ladder (`exchangeMeta`) — those are
Pro-plan only. So there is **no liquidity data** in the historical archive.

## Current state (2026-06-25)

| Artifact | Latest data | Status |
|---|---|---|
| `betfair_goal_ou.csv` (GB O/U 0.5–4.5) | 2026-05-29 | current |
| `betfair_btts.csv` (GB BTTS) | 2026-05-29 | current |
| `betfair_pl_btts.csv` | 2026-05-24 | current |
| `betfair_efl_btts.csv` | 2025-05-03 | **a season stale** |
| `betfair_efl_ou15.csv` | 2025-05-03 | **a season stale** |

## Issues + resolutions

- **P1 — EFL split CSVs a season stale (root cause: derived artifact not
  regenerated).** Both join inputs are already current on disk: raw Betfair OU1.5
  for 2025-26 exists (6,975 GB rows), and `CompleteDSChamp_CSV.csv` now runs to
  2026-05-02. The split scripts simply haven't been re-run since the EFL Canonical
  Dataset was refreshed. **Dry-run (read-only) confirms re-running yields S25
  (2025-26) coverage of 552/552 = 100%** — the team-name allowlist already covers
  every 2025-26 Championship side. Fix = re-run `extract_efl_ou15_betfair.py` +
  `extract_btts_by_league.py`. No download. _(Pending user go-ahead to write.)_

- **P2 — `extract_goal_odds.py` TAR_DIR path drift. FIXED.** It pointed at the
  now-renamed `BetFairData/`; repointed to `BetFairData_OU/`. (Latent — only bit
  on a bulk re-extract; Track B unaffected.)

- **P3 — Betfair "Test" markets leaked into the GB feed. FIXED.** 8 dummy rows
  (e.g. "Test T087 v Test T088", all OVER_UNDER_35, future-dated 2030) were in
  `betfair_goal_ou.csv`. Added an `event_name`-prefix guard (`test` / `team `) to
  the goal + BTTS parsers in `extract_goal_odds.py` and `betfair_monthly_update.py`
  to prevent recurrence. The 8 existing rows are inert (no consumer reads OU3.5)
  and will be purged on the next bulk re-extract.

- **P4 — no liquidity in the data (info, out of scope).** Basic plan = `ltp`
  only. Relevant only to the parked exchange-execution work; would require a paid
  Betfair plan tier. See `memory/oddspapi_exchange_coverage.md`.

- **P5 — League Splits silently lagged the canonical (root cause of P1). FIXED.**
  The splits are a join of two inputs (Betfair master CSV + Canonical Dataset),
  but only the monthly Betfair *download* re-ran them. When the EFL canonical was
  refreshed out-of-band (2026-06-13, eight days after the 2026-06-05 monthly run),
  the splits stayed pinned to the old canonical until manually re-run. **Fix:**
  `scheduler.job_weekly_retrain()` now calls `_refresh_betfair_splits()` every
  Sunday, *ungated* by the off-season retrain flags, so the splits can never lag
  the canonical by more than a week. Non-critical (failures logged, don't block
  the retrain).

## Health note

The monthly job's engineering is robust: timestamped + rolling logs, off-season
skip, Task Scheduler catch-up, and non-zero exit propagation on split-script
failure (no silent successes). Not the weak link.
