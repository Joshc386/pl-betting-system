"""Concurrency hardening for settlement (pre-live punch-list item #5).

The DB is WAL mode, so two settle processes can both SELECT settled=0 before
either commits, then both settle the same bets and both credit the bankroll —
double-counting real P&L. The `WHERE settled=0` guard on the *SELECT* does not
help: both processes read before either writes.

We reproduce the race deterministically (no threads — the GIL serialises them
unreliably) by simulating a competing settle that commits *inside the window
between our SELECT and our UPDATE*. `settle_bets` calls `_determine_outcome`
once per bet in exactly that window, so it is the injection seam.

All DB access is monkeypatched onto a temp file — never the production DBs.
"""
import sqlite3
from datetime import datetime

from filelock import FileLock

import db
import scheduler
import settlement
from db import get_db


def _isolate_db(tmp_path, monkeypatch) -> str:
    """Point both db and settlement at a single temp PL database."""
    dbfile = tmp_path / "pl_test.db"
    paths = {"PL": str(dbfile)}
    monkeypatch.setattr(db, "LEAGUE_DB_PATHS", paths)
    monkeypatch.setattr(settlement, "LEAGUE_DB_PATHS", paths)
    return str(dbfile)


def _seed_one_unsettled_bet(now: str) -> None:
    """Insert one unsettled O/U 2.5 Over bet + an opening bankroll row."""
    with get_db("PL") as conn:
        conn.execute(
            """INSERT INTO recommendations
               (created_at, home_team, away_team, market, side, odds,
                stake_pct, settled)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
            (now, "Arsenal", "Chelsea", "ou25", "over", 2.0, 0.02),
        )
        conn.execute(
            "INSERT INTO bankroll (timestamp, balance, event) VALUES (?, ?, ?)",
            (now, 1.0, "init"),
        )
        conn.commit()


_FINISHED = [{
    "home_team": "Arsenal", "away_team": "Chelsea",
    "home_goals": 2, "away_goals": 1, "total_goals": 3, "btts": True,
}]


def test_settle_is_idempotent_against_concurrent_committer(
    tmp_path, monkeypatch
) -> None:
    """If a competing process settles the same bet in the window between our
    SELECT and our UPDATE, our settle must NOT re-settle it or credit the
    bankroll a second time."""
    dbfile = _isolate_db(tmp_path, monkeypatch)
    now = datetime.now().isoformat()
    _seed_one_unsettled_bet(now)
    monkeypatch.setattr(
        settlement, "get_finished_matches", lambda days_back=7, **k: _FINISHED
    )

    real_determine = settlement._determine_outcome
    fired = {"done": False}

    def racing_determine(market, side, home_goals, away_goals):
        # First call lands between our SELECT (settled=0 already read) and our
        # UPDATE. A competing settle commits the whole thing right here.
        if not fired["done"]:
            fired["done"] = True
            with sqlite3.connect(dbfile) as c2:
                c2.execute(
                    """UPDATE recommendations
                       SET settled=1, won=1, profit_pct=0.02,
                           actual_result='competing', settled_at=?
                       WHERE settled=0""",
                    (now,),
                )
                c2.execute(
                    "INSERT INTO bankroll (timestamp, balance, event) "
                    "VALUES (?, ?, ?)",
                    (now, 1.02, "Settled 1 bets (1W/0L)"),
                )
                c2.commit()
        return real_determine(market, side, home_goals, away_goals)

    monkeypatch.setattr(settlement, "_determine_outcome", racing_determine)

    settlement.settle_bets(verbose=False)

    with get_db("PL") as conn:
        settlement_rows = conn.execute(
            "SELECT COUNT(*) AS c FROM bankroll WHERE event LIKE 'Settled%'"
        ).fetchone()["c"]
        bet = conn.execute(
            "SELECT settled FROM recommendations"
        ).fetchone()

    assert bet["settled"] == 1
    assert settlement_rows == 1, (
        f"bankroll credited {settlement_rows}x for one bet — double-count race"
    )


def test_settle_skipped_when_settle_lock_held(tmp_path, monkeypatch) -> None:
    """If another settle process holds the settle lock, job_settle_bets must
    skip rather than run a concurrent (double-counting) settlement."""
    lockpath = str(tmp_path / "settle.lock")
    monkeypatch.setattr(scheduler, "_SETTLE_LOCK_PATH", lockpath, raising=False)

    calls: list[int] = []

    def spy_settle(**kwargs):
        calls.append(1)
        return {"settled": 0, "won": 0, "lost": 0, "profit": 0.0}

    monkeypatch.setattr(settlement, "settle_bets", spy_settle)
    monkeypatch.setattr(settlement, "settle_predictions", lambda **k: {})

    held = FileLock(lockpath)
    held.acquire()
    try:
        scheduler.job_settle_bets()  # lock held → must skip
        assert calls == [], "settle ran while another settle held the lock"
    finally:
        held.release()

    scheduler.job_settle_bets()  # lock free → runs once
    assert calls == [1]
