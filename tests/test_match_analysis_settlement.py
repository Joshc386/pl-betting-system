"""Outcomes for *every* evaluated market, not just the ones we liked.

The system could not measure its own calibration. The only tables carrying
outcomes were selected ones — `predictions` stores a row only when
`edge_pct > 0` (`db.log_predictions`), and `recommendations` only when the bet
cleared the gate. Measured walk-forward on the OOF caches, conditioning on the
model's own positive edge inflates its apparent confidence by **+6.3 points**
regardless of the model's real calibration, which is the winner's curse rather
than a fault. A monitor fed from those tables would sit at +6 points forever
and report nothing.

`match_analysis` already holds every evaluated fixture-market-side and had no
outcome at all. Settling it supplies the unbiased sample, and leaves
`predictions` meaning what it has always meant.

Outcome determination is `settlement._determine_outcome`, reused rather than
reimplemented; a second definition of "did this bet win" is the defect this
codebase keeps paying for.
"""
from __future__ import annotations

import sqlite3

import pytest


def _rows(**over):
    base = {
        "home_team": "Arsenal FC", "away_team": "Everton FC",
        "kickoff": "2026-09-01T14:00:00Z", "matchday": "2026-09-01",
        "market": "ou25", "side": "over", "best_odds": 2.0,
        "best_bookmaker": "bet365", "model_prob": 0.55, "fair_odds": 1.9,
        "edge_pct": 3.0, "confidence": "high", "n_books": 12,
    }
    base.update(over)
    return [base]


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """A real database at the current schema, isolated per test.

    Patches ``LEAGUE_DB_PATHS`` the way the existing dashboard tests do;
    ``get_db`` creates the tables on first connection, so opening it once is
    the migration.
    """
    import db as dbmod

    path = tmp_path / "dash.db"
    monkeypatch.setitem(dbmod.LEAGUE_DB_PATHS, "PL", str(path))
    with dbmod.get_db("PL"):
        pass
    return dbmod, str(path)


class TestTheMigrationIsAdditiveAndIdempotent:
    """V3 and V4: adding columns must not disturb what is already there."""

    def test_the_settlement_columns_exist(self, db):
        dbmod, path = db
        cols = {r[1] for r in sqlite3.connect(path)
                .execute("PRAGMA table_info(match_analysis)")}

        assert {"settled", "won", "actual_result", "settled_at"} <= cols, (
            "match_analysis has no outcome columns, so every calibration "
            "measurement is stuck on a selected sample")

    def test_reopening_the_database_is_a_no_op(self, db):
        dbmod, path = db
        dbmod.save_match_analysis(_rows(), league="PL")
        before = sqlite3.connect(path).execute(
            "SELECT COUNT(*) FROM match_analysis").fetchone()[0]

        with dbmod.get_db("PL"):
            pass

        after = sqlite3.connect(path).execute(
            "SELECT COUNT(*) FROM match_analysis").fetchone()[0]
        assert after == before == 1, (
            "re-running the migration changed the row count; it must be "
            "additive only")


class TestARescanDoesNotUnsettleAFixture:
    """V2 — the silent failure this change is most likely to introduce.

    `save_match_analysis` upserts, replacing rows for fixtures the scan
    covers. Nothing stops it overwriting a settled row with a fresh unsettled
    one, and when it does there is no error: the calibration series simply
    gets shorter, and the shortening looks like a quiet week.

    The same function already carries `model_prob` forward for exactly this
    reason. Settlement gets the same treatment.
    """

    def test_settled_outcome_survives_a_rescan(self, db):
        dbmod, path = db
        dbmod.save_match_analysis(_rows(), league="PL")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE match_analysis SET settled=1, won=1, "
                "actual_result='3-1', settled_at='2026-09-01T18:00:00Z'")

        # The same fixture comes round again on a later scan.
        dbmod.save_match_analysis(_rows(model_prob=0.61), league="PL")

        row = sqlite3.connect(path).execute(
            "SELECT settled, won, actual_result, model_prob "
            "FROM match_analysis").fetchone()
        assert row[0] == 1, "the rescan unsettled a settled fixture"
        assert row[1] == 1, "the rescan discarded the outcome"
        assert row[2] == "3-1", "the rescan discarded the actual result"
        assert row[3] == pytest.approx(0.61), (
            "the rescan should still refresh the model probability")


class TestSettlementUsesTheOneOutcomeDefinition:
    """V1 — every evaluated market gets an outcome, from the shared function.

    `settlement._determine_outcome` already decides whether a bet won, for
    both the ou* family and btts. Writing a second version here would be the
    defect this codebase keeps finding: two definitions agreeing on the day
    they are written and drifting the first time either is touched.
    """

    def test_both_sides_of_a_market_are_settled_and_disagree(self, db):
        """Over and Under on the same fixture cannot both win.

        The point of settling this table is an *unbiased* sample, so the
        losing side has to be recorded as carefully as the winning one — it
        is the half `predictions` throws away.
        """
        from settlement import settle_match_analysis

        dbmod, path = db
        rows = _rows() + _rows(side="under") + _rows(market="btts", side="yes")
        dbmod.save_match_analysis(rows, league="PL")

        # Arsenal 3-1 Everton: 4 goals, both scored.
        settle_match_analysis(
            finished={("Arsenal FC", "Everton FC"): (3, 1)}, league="PL")

        got = {
            (r[0], r[1]): (r[2], r[3])
            for r in sqlite3.connect(path).execute(
                "SELECT market, side, settled, won FROM match_analysis")
        }
        assert got[("ou25", "over")] == (1, 1), "over 2.5 won on 4 goals"
        assert got[("ou25", "under")] == (1, 0), (
            "under 2.5 lost on 4 goals, and must be recorded as such — "
            "dropping losers is what makes the sample biased")
        assert got[("btts", "yes")] == (1, 1), "both teams scored"

    def test_an_unfinished_fixture_is_left_alone(self, db):
        from settlement import settle_match_analysis

        dbmod, path = db
        dbmod.save_match_analysis(_rows(), league="PL")

        settle_match_analysis(finished={}, league="PL")

        row = sqlite3.connect(path).execute(
            "SELECT settled, won FROM match_analysis").fetchone()
        assert row == (0, None), (
            "a fixture with no result must stay unsettled rather than be "
            "recorded as a loss")

    def test_it_agrees_with_the_shared_outcome_function(self, db):
        """No second definition of winning."""
        from settlement import _determine_outcome, settle_match_analysis

        dbmod, path = db
        dbmod.save_match_analysis(
            _rows(market="btts", side="no"), league="PL")
        settle_match_analysis(
            finished={("Arsenal FC", "Everton FC"): (2, 0)}, league="PL")

        expected_won, expected_result = _determine_outcome("btts", "no", 2, 0)
        row = sqlite3.connect(path).execute(
            "SELECT won, actual_result FROM match_analysis").fetchone()
        assert bool(row[0]) is expected_won
        assert row[1] == expected_result
