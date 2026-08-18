"""A scheduled job that never ran must not look like one that ran clean.

On 2026-08-14 `Bettig Bot Daily Settlement` did not fire. Its catch-up
(`StartWhenAvailable`) silently failed, and every field an operator would check
still read healthy: `LastTaskResult` 0, `NumberOfMissedRuns` 0. The last run
that *happened* had succeeded, so Windows reported success on a day the job
never ran. See ADR 0006's counterexample.

The only local evidence that a run happened is its own log. A run that finds
nothing still writes one; a run that never happened does not. So job health is
read from the log's timestamp and its `exit=` line, never from the data — data
alone cannot distinguish "ingest ran and upstream had nothing new" from
"ingest never ran", and right now the first of those is the correct state
every day until football-data.co.uk publishes 2026/27.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import pytest

from job_health import JobState, read_job_status


def _write_log(log_dir, name: str, exit_code: int | None, age_hours: float):
    """A rolling job log of the shape the .bat wrappers produce."""
    p = log_dir / name
    body = "============================================================\n"
    if exit_code is not None:
        body += f" Daily job finished: 15/08/2026  9:56:55.35 (exit={exit_code})\n"
    p.write_text(body, encoding="utf-8")
    when = time.time() - age_hours * 3600
    os.utime(p, (when, when))
    return p


class TestJobStatus:
    def test_recent_clean_run_is_ok(self, tmp_path):
        _write_log(tmp_path, "daily_settle.log", exit_code=0, age_hours=2)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].state is JobState.OK
        assert status["Settle"].exit_code == 0

    def test_job_that_never_ran_is_not_ok(self, tmp_path):
        """The 14 August case: no log at all, and nothing else would say so."""
        _write_log(tmp_path, "daily_ingest.log", exit_code=0, age_hours=1)
        # daily_settle.log deliberately absent

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].state is JobState.MISSING
        assert status["Settle"].last_run is None
        # The healthy sibling must not mask it.
        assert status["Ingest"].state is JobState.OK

    def test_stale_run_is_flagged_before_it_is_critical(self, tmp_path):
        """A daily job 40h old is late but not yet alarming."""
        _write_log(tmp_path, "daily_settle.log", exit_code=0, age_hours=40)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].state is JobState.STALE

    def test_long_silence_escalates_to_failed(self, tmp_path):
        """Three days without a daily job is not merely late."""
        _write_log(tmp_path, "daily_settle.log", exit_code=0, age_hours=80)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].state is JobState.FAILED

    def test_nonzero_exit_outranks_recency(self, tmp_path):
        """A job that failed an hour ago is worse than one merely late."""
        _write_log(tmp_path, "daily_settle.log", exit_code=1, age_hours=1)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].state is JobState.FAILED
        assert status["Settle"].exit_code == 1

    def test_retrain_uses_a_weekly_cadence_not_a_daily_one(self, tmp_path):
        """5 days is healthy for the Sunday retrain, dead for a daily job."""
        _write_log(tmp_path, "weekly_retrain.log", exit_code=0, age_hours=5 * 24)
        _write_log(tmp_path, "daily_settle.log", exit_code=0, age_hours=5 * 24)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Retrain"].state is JobState.OK
        assert status["Settle"].state is JobState.FAILED

    def test_log_without_an_exit_line_is_not_read_as_success(self, tmp_path):
        """A run killed mid-flight never stamps `exit=`; absence is not zero."""
        _write_log(tmp_path, "daily_settle.log", exit_code=None, age_hours=1)

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].exit_code is None
        assert status["Settle"].state is JobState.OK  # recent; age is the signal

    def test_last_exit_line_wins_when_a_log_holds_several(self, tmp_path):
        """The rolling log is overwritten per run, but be explicit about it."""
        p = tmp_path / "daily_settle.log"
        p.write_text("finished (exit=0)\nfinished (exit=3)\n", encoding="utf-8")

        status = {s.name: s for s in read_job_status(tmp_path)}

        assert status["Settle"].exit_code == 3
        assert status["Settle"].state is JobState.FAILED


class TestUnsettledBacklog:
    """Rows the scheduled settle can no longer reach.

    `scheduler.py:376` and `:384` call settlement with `days_back=3` — the
    hardcoded call site wins over `settlement.py`'s own default of 7. So a row
    whose kickoff is more than three days old will never be settled by any
    scheduled run again, however many times settlement succeeds afterwards.
    Three PL predictions from 12 April proved it: settlement ran clean roughly
    120 times while they sat there, because it never looked back far enough to
    see them.
    """

    @staticmethod
    def _db(tmp_path, rows):
        """A dashboard.db holding just what the backlog query reads."""
        import sqlite3
        p = tmp_path / "dashboard.db"
        c = sqlite3.connect(p)
        for t in ("predictions", "recommendations", "logged_bets"):
            c.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY, "
                      f"kickoff TEXT, settled INTEGER)")
        for table, kickoff, settled in rows:
            c.execute(f"INSERT INTO {table} (kickoff, settled) VALUES (?,?)",
                      (kickoff, settled))
        c.commit()
        c.close()
        return p

    def test_row_past_the_horizon_counts(self, tmp_path):
        from job_health import unsettled_backlog

        now = datetime(2026, 8, 15, 12, 0)
        db = self._db(tmp_path, [
            ("predictions", (now - timedelta(days=10)).isoformat(), 0),
        ])

        assert unsettled_backlog(db, now=now) == 1

    def test_row_inside_the_horizon_does_not(self, tmp_path):
        """Last night's fixture is not a backlog — settlement will get it."""
        from job_health import unsettled_backlog

        now = datetime(2026, 8, 15, 12, 0)
        db = self._db(tmp_path, [
            ("predictions", (now - timedelta(hours=14)).isoformat(), 0),
        ])

        assert unsettled_backlog(db, now=now) == 0

    def test_settled_rows_never_count_however_old(self, tmp_path):
        from job_health import unsettled_backlog

        now = datetime(2026, 8, 15, 12, 0)
        db = self._db(tmp_path, [
            ("predictions", (now - timedelta(days=400)).isoformat(), 1),
        ])

        assert unsettled_backlog(db, now=now) == 0

    def test_counts_across_all_three_tables(self, tmp_path):
        from job_health import unsettled_backlog

        now = datetime(2026, 8, 15, 12, 0)
        old = (now - timedelta(days=9)).isoformat()
        db = self._db(tmp_path, [
            ("predictions", old, 0),
            ("recommendations", old, 0),
            ("logged_bets", old, 0),
        ])

        assert unsettled_backlog(db, now=now) == 3

    def test_missing_database_is_zero_not_a_crash(self, tmp_path):
        """The dashboard must render even if a league's DB is absent."""
        from job_health import unsettled_backlog

        assert unsettled_backlog(tmp_path / "nope.db") == 0


class TestDataCoverage:
    """Which season the training data actually reaches.

    Distinct from job health: the daily ingest can run perfectly and still add
    nothing, because football-data.co.uk has not published 2026/27 for either
    league. That is the correct state today, and it must not read as a failure
    — but the day upstream does publish, the number here is what changes.
    """

    def test_reports_the_highest_season_present(self, tmp_path):
        from job_health import data_coverage

        csv = tmp_path / "canonical.csv"
        csv.write_text("SeasonIndex,Home_Team\n24,A\n25,B\n25,C\n",
                       encoding="utf-8")

        assert data_coverage(csv) == 25

    def test_missing_file_is_none_not_a_crash(self, tmp_path):
        from job_health import data_coverage

        assert data_coverage(tmp_path / "absent.csv") is None
