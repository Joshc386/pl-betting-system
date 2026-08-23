"""Did the scheduled jobs actually run? (ADR 0006's counterexample.)

Windows Task Scheduler catch-up is reliable but not guaranteed, and when it
fails it fails silently — `LastTaskResult` keeps reporting the last run that
*happened*, so a day the job never ran still reads `0`. Nothing asserted "the
ingest ran today", and on 2026-08-14 nothing noticed that it hadn't.

This module reads each job's own rolling log: its mtime for when the run
finished, its `exit=` line for whether it succeeded. That is deliberately not
the same question as the **Freshness Gate** (`freshness.py`, ADR 0005), which
asks whether the *data* is current, fetches from the network, and blocks
Recommendations. This one is local, read-only, and never blocks — it only
reports.

The distinction matters right now: football-data.co.uk has not published
2026/27, so a correct ingest finds nothing new every single day. Judged on data
alone that is indistinguishable from an ingest that never ran. Judged on its
log, it is obvious.
"""
from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# How far back a scheduled settle actually looks. Not settlement.py's own
# default of 7 — scheduler.py:376 and :384 pass days_back=3 explicitly, and the
# call site wins. A row older than this is unreachable by any scheduled run,
# however many times settlement succeeds afterwards.
SETTLE_HORIZON_DAYS = 3

_BACKLOG_TABLES = ("predictions", "recommendations", "logged_bets")

# Each job's log glob, and the ages at which it stops being reassuring.
# Thresholds are 1.5x and 3x the job's own cadence: enough slack that a late
# catch-up after a weekend away is not an alarm, tight enough that a genuinely
# missed run surfaces on the next dashboard load.
#
# Retrain reads its dated archives rather than `weekly_retrain.log`, because
# the .bat writes that rolling copy on its *last line* — interrupt the batch
# and the copy never happens, so the rolling log keeps reporting the previous
# week. That is ADR 0006's failure reproduced one layer below Task Scheduler,
# and it hid the interrupted 2026-08-17 run for twelve days. An archive is
# written as the run goes, so it exists whatever happens afterwards.
#
# The pattern is deliberately `weekly_retrain_*` and not `*retrain*`:
# `retrain_manual_*.log` answers a different question (are the models
# current) from this one (did the scheduled job fire).
JOBS: dict[str, tuple[str, float, float]] = {
    #  label       log glob                 amber(h)  red(h)
    "Ingest":  ("daily_ingest.log",           36,      72),
    "Settle":  ("daily_settle.log",           36,      72),
    "Retrain": ("weekly_retrain_*.log",      9 * 24,  14 * 24),
}

_EXIT_RE = re.compile(r"exit=(-?\d+)")


class JobState(Enum):
    OK = "ok"
    STALE = "stale"
    INCOMPLETE = "incomplete"  # ran, but never stamped an exit code
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True)
class JobStatus:
    name: str
    last_run: datetime | None
    age_hours: float | None
    exit_code: int | None
    state: JobState


def unsettled_backlog(
    db_path: str | Path,
    now: datetime | None = None,
    horizon_days: int = SETTLE_HORIZON_DAYS,
) -> int:
    """Rows past the settle horizon that are still open. Never raises.

    A non-zero count means settlement can succeed forever without these ever
    being picked up — they need a manual run with a wider ``days_back``.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0

    cutoff = ((now or datetime.now()) - timedelta(days=horizon_days)).isoformat()
    total = 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        for table in _BACKLOG_TABLES:
            try:
                total += conn.execute(
                    f"SELECT COUNT(*) FROM {table} "
                    f"WHERE settled = 0 AND kickoff < ?", (cutoff,)
                ).fetchone()[0]
            except sqlite3.Error:
                continue  # table absent in this league's schema
    finally:
        conn.close()
    return total


def unpriced_fixtures(
    db_path: str | Path,
    now: datetime | None = None,
) -> int:
    """Fixtures still ahead of kickoff that the predictor priced nothing for.

    ``scan.py`` computes the set of scanned fixtures with no ``model_prob``,
    logs the count, runs the predictor — and never re-checks whether the gap
    closed. A predictor that ran and fixed it and one that ran and could not
    produce identical output.

    On 2026-08-18 that line read ``Missing model data for 1/12 fixtures``. The
    one was Arsenal v Coventry, a promoted side with no cohort seed. It
    reached its 21 August kickoff with all six markets NULL, yielding no
    prediction and no recommendation, and left no trace but an absence.

    A fixture counts only while something can still be done about it, so this
    returns to zero on its own once the slate has kicked off. One priced
    market clears a fixture: partial coverage is a real but different signal,
    and this one is for "the predictor produced nothing at all".

    Kickoffs are stored UTC and ``now`` is local, matching
    :func:`unsettled_backlog`. That skews the boundary by the UK offset, which
    on a fixture-level alarm costs at most an hour of notice on a fixture
    already about to start. Never raises.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return 0

    cutoff = (now or datetime.now()).isoformat()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return 0
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT home_team, away_team, kickoff FROM match_analysis"
            "  WHERE kickoff > ?"
            "  GROUP BY home_team, away_team, kickoff"
            "  HAVING SUM(model_prob IS NOT NULL) = 0)",
            (cutoff,),
        ).fetchone()[0]
    except sqlite3.Error:
        return 0  # table absent in this league's schema
    finally:
        conn.close()


def data_coverage(csv_path: str | Path) -> int | None:
    """Highest SeasonIndex in a canonical/enriched CSV, or None. Never raises.

    Reads one column so this stays cheap enough for a per-callback refresh.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        col = pd.read_csv(csv_path, usecols=["SeasonIndex"])["SeasonIndex"]
        return int(col.max()) if len(col) else None
    except Exception:
        return None


def _parse_exit_code(path: Path) -> int | None:
    """The exit code the .bat wrapper stamped, or None if it never got there."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = _EXIT_RE.findall(text)
    return int(matches[-1]) if matches else None


def read_job_status(
    log_dir: str | Path = "logs",
    now: datetime | None = None,
) -> list[JobStatus]:
    """Health of each scheduled job, from its own log. Never raises."""
    log_dir = Path(log_dir)
    now = now or datetime.now()

    out: list[JobStatus] = []
    for name, (pattern, amber, red) in JOBS.items():
        # Newest match wins. For the daily jobs the glob is a literal name and
        # matches at most one file; for Retrain it spans the dated archives.
        candidates = sorted(log_dir.glob(pattern), key=os.path.getmtime)
        if not candidates:
            out.append(JobStatus(name, None, None, None, JobState.MISSING))
            continue
        path = candidates[-1]

        last_run = datetime.fromtimestamp(os.path.getmtime(path))
        age = (now - last_run).total_seconds() / 3600
        exit_code = _parse_exit_code(path)

        # Ordering, worst first. A non-zero exit outranks age: a job that ran
        # an hour ago and failed is worse news than one that is merely late.
        # A long silence outranks *how* the last run ended — four days dead
        # matters more than its footer. Below that, a missing `exit=` means
        # the run was killed mid-flight, which is the more useful of the two
        # things a merely-amber age could be telling you.
        #
        # One known false positive: a retrain still running has no footer yet,
        # so it reads INCOMPLETE for its ~25-70 minutes. It clears itself.
        if exit_code:
            state = JobState.FAILED
        elif age >= red:
            state = JobState.FAILED
        elif exit_code is None:
            state = JobState.INCOMPLETE
        elif age >= amber:
            state = JobState.STALE
        else:
            state = JobState.OK

        out.append(JobStatus(name, last_run, age, exit_code, state))

    return out
