# 6. Windows Task Scheduler for data-critical jobs

Date: 2026-07-25

## Status

Accepted. **Counterexample observed 2026-08-14 — the catch-up this decision
rests on failed once, silently. See the corrected consequence below.**

The decision itself stands: Task Scheduler is still strictly better than
APScheduler's `misfire_grace_time` for a laptop that sleeps. What this ADR got
wrong is the strength of the guarantee — it treated `StartWhenAvailable` as
reliable rather than as usually-reliable, and therefore never asked what would
notice if a catch-up did not happen. Nothing did.

## Context

The system has two scheduling mechanisms and only one of them actually runs.

`scheduler.py` defines an APScheduler tier inside `run.py`: daily data refresh (06:30),
fixture planner (07:00), matchday fetch (09:00), pre-kickoff odds refresh (KO−60min),
CLV capture (KO−5min), and evening/morning settlement. All of it requires a long-lived
process. **No such process was running, and no task starts one.** Windows Task Scheduler
holds exactly three entries for this project — weekly retrain, daily settlement, monthly
Betfair update — all `.bat` wrappers, all last exited 0.

The two schedulers encode opposite assumptions about missed work, and the evidence on
this machine is decisive:

- The weekly retrain is scheduled **Sunday 23:30** (`StartBoundary 2026-05-03T23:30`,
  `DaysOfWeek=1`). It last ran **Monday 2026-07-20 at 09:10** — about ten hours late, as
  a catch-up after the laptop woke. This works because the task sets
  `StartWhenAvailable: True`, with `RestartCount: 3` at 15-minute intervals.
- APScheduler sets `misfire_grace_time=3600`. A job missed by more than an hour is
  **discarded, not caught up**. Even had the process survived an overnight sleep, the
  06:30 refresh would have been silently dropped rather than run late.

APScheduler assumes a server that is always up, so a missed job is probably stale and
best skipped. Task Scheduler assumes a personal machine that sleeps, so a missed job is
probably still wanted and best run late. For a laptop-hosted betting system the second
assumption is plainly correct.

The one thing static cron cannot express is "60 minutes before each kickoff", since
kickoff times vary per matchday — the reason APScheduler's `DateTrigger` was used.

## Decision

- **The data-critical tier moves to Windows Task Scheduler `.bat` entries** with
  `StartWhenAvailable` enabled: match-results ingestion, the League Split re-derivation,
  the Freshness Gate ([0005](0005-freshness-gate.md)), and the Data Refresh. These follow
  the pattern of the three tasks already proven on this machine.

- **The dynamic per-kickoff odds tier stays in APScheduler** and is explicitly out of
  scope for this decision.

The split is drawn on precision requirements, not convenience: **per-kickoff timing
matters for odds, which move; it does not matter for data freshness, which a morning
refresh satisfies.** Nothing in the data-currency goal needs sub-hour granularity.

## Consequences

- Two scheduling mechanisms coexist by design. This ADR exists primarily so that a
  future reader asking "why are there two schedulers?" finds the answer rather than
  assuming it is an accident and unifying them.
- ~~The data tier gains catch-up semantics and survives the laptop's sleep cycle.~~
  **Corrected 2026-08-14 — catch-up is reliable but not guaranteed, and its
  failure is silent.** `Betting Bot Daily Ingest` (06:00 daily,
  `StartWhenAvailable: True`, `WakeToRun: False`) caught up correctly on 11, 12
  and 13 August, each time logging event `114` — *"could not launch as
  scheduled … started now as required by the configuration option to start the
  task when available"* — at 07:49, 07:57 and 07:49, shortly after each morning
  wake. On **14 August** the laptop slept at 21:49 the previous evening and
  resumed at **07:52**, the same window, and the task emitted **no events of any
  kind**. `NumberOfMissedRuns` incremented to 1 and `NextRunTime` advanced past
  the day to 15 August 06:00. Confirmed by two independent event-log queries
  (XPath on `TaskName`, and a message-text scan of the preceding 48 hours).
  Root cause undetermined; the configuration is correct and unchanged.

  **The operationally important part is that this is invisible.**
  `LastTaskResult` still reads `0`, because the last run that *happened*
  succeeded — so the one field an operator would check reports health on a day
  the job never ran. That is the same silent-discard failure this ADR criticised
  APScheduler for; moving tiers reduced its frequency without removing it. A
  missed ingest is only detectable today by the **Freshness Gate**
  ([0005](0005-freshness-gate.md)) noticing the *consequence* two days later,
  which is a downstream symptom rather than an alarm on the job itself. Nothing
  currently asserts "the ingest ran today".
- The odds tier still depends on `run.py` being alive, which it generally is not. This
  remains a **known open gap** — per-kickoff odds refresh and CLV capture at KO−5min are
  not currently firing, and neither is the settlement pair inside APScheduler (daily
  settlement is separately covered by its own Task Scheduler entry).
- `misfire_grace_time` in `scheduler.py` should be revisited if the odds tier is ever
  made persistent, since its current value would discard exactly the jobs most worth
  catching up.
