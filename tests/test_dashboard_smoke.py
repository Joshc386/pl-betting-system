"""
Render-time smoke tests for the Dash dashboard.

Asserts every (tab, league) combination imports and renders without raising,
and that section counts are within the expected range given the current
state of each league's SQLite database.

Live-DB: reads row counts from `data/dashboard.db` and `data/dashboard_efl.db`
to compute expected section minimums. Skips gracefully when DBs are missing
(e.g. fresh clone before first scan).

Catches two regression classes that calculation tests miss:
- Render-time crashes from library API drift (e.g. dbc.Table `dark` kwarg
  removed in dbc 2.0 — caused the Performance tab to crash silently
  until the dashboard-tester subagent surfaced it on 2026-05-02)
- Silent-empty regressions where conditional sections drop because of
  upstream data-state bugs (e.g. EFL settlement scope bug fixed in
  commit 969f366 caused Model Analytics to drop from 19 to 8 sections)

Companion to `.claude/agents/dashboard-tester.md` — same invariants, faster
feedback loop (every commit via the Stop hook), no prose narrative.
"""
import sqlite3
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
_LEAGUES = ["PL", "EFL"]
_DB_PATHS = {
    "PL": _PROJECT_ROOT / "data" / "dashboard.db",
    "EFL": _PROJECT_ROOT / "data" / "dashboard_efl.db",
}


def _db_snapshot(league: str) -> dict:
    """Row counts that drive expected section minimums.

    Returns empty dict if the DB doesn't exist; callers use that as a
    skip signal so a fresh clone doesn't fail the whole suite.
    """
    path = _DB_PATHS[league]
    if not path.exists():
        return {}
    conn = sqlite3.connect(str(path))
    try:
        def c(table: str, where: str = "") -> int:
            try:
                q = f"SELECT COUNT(*) FROM {table}"
                if where:
                    q += f" WHERE {where}"
                return conn.execute(q).fetchone()[0]
            except sqlite3.Error:
                return 0
        return {
            "pred_total": c("predictions"),
            "pred_settled": c("predictions", "settled=1"),
            "rec_total": c("recommendations"),
            "rec_settled": c("recommendations", "settled=1"),
            "logged": c("logged_bets"),
        }
    finally:
        conn.close()


def _section_count(div) -> int:
    """Count top-level children in a Dash Div result."""
    if not hasattr(div, "children"):
        return 1
    ch = div.children
    return len(ch) if isinstance(ch, list) else 1


# ─────────────────────────────────────────────────────────────────────
# Import smoke
# ─────────────────────────────────────────────────────────────────────

def test_dashboard_imports_clean():
    """dashboard.py imports without raising — catches module-load issues
    (syntax, missing deps, top-level API drift)."""
    import dashboard  # noqa: F401
    for fn in ("_build_match_centre", "_build_bet_tracker",
               "_build_performance", "_build_analytics"):
        assert hasattr(dashboard, fn), f"dashboard.{fn} missing after import"


# ─────────────────────────────────────────────────────────────────────
# Render no-crash — every (tab, league) combination
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("league", _LEAGUES)
def test_match_centre_renders(league):
    if not _DB_PATHS[league].exists():
        pytest.skip(f"{league} DB not present")
    from dashboard import _build_match_centre
    result = _build_match_centre(league, show_all=False)
    assert _section_count(result) >= 1


@pytest.mark.parametrize("league", _LEAGUES)
def test_bet_tracker_renders(league):
    if not _DB_PATHS[league].exists():
        pytest.skip(f"{league} DB not present")
    from dashboard import _build_bet_tracker
    result = _build_bet_tracker(league)
    assert _section_count(result) >= 1


@pytest.mark.parametrize("league", _LEAGUES)
def test_performance_renders(league):
    """Performance always renders >= 6 sections regardless of data state.
    A count below 6 means a section silently dropped — likely an unintended
    early-return or a swallowed render crash. The dbc.Table(dark=True) bug
    from commit 969f366 made this count 0; this test would have caught it."""
    if not _DB_PATHS[league].exists():
        pytest.skip(f"{league} DB not present")
    from dashboard import _build_performance
    result = _build_performance(league)
    n = _section_count(result)
    assert n >= 6, (
        f"{league} Performance rendered {n} sections; expected >= 6 "
        "(bankroll + live-vs-sim + market + monthly + side + edge-source "
        "breakdowns). A count below 6 is a regression."
    )


@pytest.mark.parametrize("league", _LEAGUES)
def test_analytics_renders(league):
    """Model Analytics renders without raising. Empty data state is fine —
    the empty-state stub still renders. Section count bounds are checked
    by `test_analytics_full_when_data_complete` below."""
    if not _DB_PATHS[league].exists():
        pytest.skip(f"{league} DB not present")
    from dashboard import _build_analytics
    result = _build_analytics(league)
    assert _section_count(result) >= 1


# ─────────────────────────────────────────────────────────────────────
# Silent-empty regression test
# Targets the EFL bug class: when the DB has substantial settled data,
# Model Analytics should render its full layout, not get cut short by
# an early-return on `if settled_recs.empty`.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("league", _LEAGUES)
def test_analytics_full_when_data_complete(league):
    """When DB has settled recs >= 5 and settled preds >= 5, the Model
    Analytics tab should render its full layout (>= 15 top-level sections).
    Below that with this much data implies a silent-empty regression like
    the EFL settlement-scope bug (settle_bets only iterating ACTIVE_LEAGUE)
    fixed in commit 969f366."""
    if not _DB_PATHS[league].exists():
        pytest.skip(f"{league} DB not present")
    snap = _db_snapshot(league)
    if snap.get("rec_settled", 0) < 5 or snap.get("pred_settled", 0) < 5:
        pytest.skip(
            f"{league} has insufficient settled data for full-render check "
            f"(rec_settled={snap.get('rec_settled')}, "
            f"pred_settled={snap.get('pred_settled')}); needs >= 5 each"
        )
    from dashboard import _build_analytics
    result = _build_analytics(league)
    n = _section_count(result)
    assert n >= 15, (
        f"{league} Model Analytics rendered only {n} sections despite "
        f"sufficient data ({snap}). Expected >= 15. Likely an early-return "
        "regression — the EFL bug fixed in 969f366 was exactly this class."
    )
