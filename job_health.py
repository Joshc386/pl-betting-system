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

# Each job's rolling log, and the ages at which it stops being reassuring.
# Thresholds are 1.5x and 3x the job's own cadence: enough slack that a late
# catch-up after a weekend away is not an alarm, tight enough that a genuinely
# missed run surfaces on the next dashboard load.
JOBS: dict[str, tuple[str, float, float]] = {
    #  label       log filename            amber(h)  red(h)
    "Ingest":  ("daily_ingest.log",           36,      72),
    "Settle":  ("daily_settle.log",           36,      72),
    "Retrain": ("weekly_retrain.log",        9 * 24,  14 * 24),
}

_EXIT_RE = re.compile(r"exit=(-?\d+)")


class JobState(Enum):
    OK = "ok"
    STALE = "stale"
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
    for name, (filename, amber, red) in JOBS.items():
        path = log_dir / filename
        if not path.exists():
            out.append(JobStatus(name, None, None, None, JobState.MISSING))
            continue

        last_run = datetime.fromtimestamp(os.path.getmtime(path))
        age = (now - last_run).total_seconds() / 3600
        exit_code = _parse_exit_code(path)

        # A non-zero exit outranks age: a job that ran an hour ago and failed
        # is worse news than one that is merely late.
        if exit_code:
            state = JobState.FAILED
        elif age >= red:
            state = JobState.FAILED
        elif age >= amber:
            state = JobState.STALE
        else:
            state = JobState.OK

        out.append(JobStatus(name, last_run, age, exit_code, state))

    return out
