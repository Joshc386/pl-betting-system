"""Tests for the ADR 0005 Freshness Gate.

The gate is a hard precondition on producing Recommendations and on running a
Data Refresh: every fixture an authoritative fixture list reports as finished in
a rolling 14-day window must be present in that league's Canonical Dataset.

The reconciliation itself is pure — it takes the finished fixtures and the
canonical's keys and returns what is absent — so the gate's actual logic is
tested without a clock, a network, or a CSV. Same split as the skip verdict in
the sibling project's nightly job.
"""
from datetime import date

import pytest

import freshness
from freshness import (
    WINDOW_DAYS,
    FreshnessError,
    Verdict,
    assert_fresh,
    check_freshness,
    reconcile,
    window_bounds,
)


class TestReconcile:
    """The pure core: which finished fixtures are absent from the canonical."""

    def test_a_finished_fixture_absent_from_the_canonical_is_reported(self) -> None:
        """The gate's whole purpose, in one assertion.

        Reported by identity — date and both teams — because with no bypass flag
        the message is the operator's only route to action, so a bare count
        would be useless.
        """
        played = (date(2026, 8, 22), "Hull City AFC", "Manchester United FC")

        assert reconcile([played], canonical_keys=set()) == [played]

    def test_a_fixture_present_on_the_wrong_date_is_reported_missing(self) -> None:
        """The key is exact — no date tolerance.

        ESPN reports kickoff in UTC and the canonical holds the UK local date,
        which for English domestic football cannot differ. So a date mismatch is
        not a timezone artefact to be forgiven; it is a fixture ingested under
        the wrong date, and tolerating +/-1 day would hide it. Deliberately
        unlike the Betfair League Split, whose tolerance is a property of that
        feed.
        """
        played = (date(2026, 8, 22), "Hull City AFC", "Manchester United FC")
        off_by_one = (date(2026, 8, 21), "Hull City AFC", "Manchester United FC")

        assert reconcile([played], canonical_keys={off_by_one}) == [played]


class TestVerdict:
    """FRESH, STALE, UNKNOWN — three states, because two is what caused the bug."""

    def test_no_finished_fixtures_is_fresh(self) -> None:
        """The decision that makes the off-season need no manual flag.

        An international break and a closed season both produce zero finished
        fixtures, and zero-missing-out-of-zero is a clean pass. This is what
        replaces the ambiguity that forced PL_RETRAIN_ENABLED to be a boolean
        somebody had to remember to flip.
        """
        result = check_freshness(
            "PL", finished=[], canonical_keys=set(),
        )

        assert result.verdict is Verdict.FRESH
        assert result.missing == []

    def test_both_authorities_unreachable_is_unknown_never_fresh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The anti-regression test for the bug class this gate exists to end.

        `fixture_schedule.py:74`, `api/espn_scores.py:170` and a sibling
        project's ingest all turned "I could not find out" into "there is
        nothing here". On a gate whose pass condition IS "nothing finished",
        that substitution is fatal: an outage would silently authorise betting
        on a stale canonical. UNKNOWN must be its own verdict.
        """
        def dead(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr(freshness, "_fetch_espn", dead)
        monkeypatch.setattr(freshness, "_fetch_football_data", dead)

        result = check_freshness("PL", canonical_keys=set())

        assert result.verdict is Verdict.UNKNOWN
        assert result.verdict is not Verdict.FRESH


class TestAssertFresh:
    """The boundary call. Raises, never returns an empty list."""

    PLAYED = (date(2026, 8, 22), "Hull City AFC", "Manchester United FC")

    def test_stale_raises_and_names_the_missing_fixture(self) -> None:
        """A bare count would be useless — there is no bypass flag, so this
        message is the operator's only route to action."""
        with pytest.raises(FreshnessError) as excinfo:
            assert_fresh("PL", finished=[self.PLAYED], canonical_keys=set())

        message = str(excinfo.value)
        assert "2026-08-22" in message
        assert "Hull City AFC" in message
        assert "Manchester United FC" in message

    def test_unknown_also_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """UNKNOWN fails closed. There is no date-heuristic second opinion:
        for live capital, refusing to act on unverifiable inputs is correct."""
        def dead(*args, **kwargs):
            raise OSError("network down")

        monkeypatch.setattr(freshness, "_fetch_espn", dead)
        monkeypatch.setattr(freshness, "_fetch_football_data", dead)

        with pytest.raises(FreshnessError):
            assert_fresh("PL", canonical_keys=set())

    def test_fresh_returns_quietly(self) -> None:
        assert assert_fresh("PL", finished=[], canonical_keys=set()) is None


class TestOrderedAuthorities:
    """ESPN first, football-data.org second. Strictly ordered, never a vote."""

    PLAYED = (date(2026, 8, 22), "Hull City AFC", "Manchester United FC")

    def test_football_data_answers_when_espn_cannot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ESPN is also Settlement's feed and it CDN-blocked a sibling project
        on 2026-08-05, so the fallback is a *different provider*, not a retry."""
        def dead(*args, **kwargs):
            raise OSError("ESPN unreachable")

        monkeypatch.setattr(freshness, "_fetch_espn", dead)
        monkeypatch.setattr(
            freshness, "_fetch_football_data",
            lambda league, window_days, now: [self.PLAYED],
        )

        result = check_freshness("PL", canonical_keys={self.PLAYED})

        assert result.verdict is Verdict.FRESH

    def test_espn_is_preferred_and_football_data_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordered, not cross-checked — a disagreement between two sources
        would need semantics of its own, and there is no evidence of one."""
        def must_not_run(*args, **kwargs):
            raise AssertionError("fallback used while ESPN was answering")

        monkeypatch.setattr(
            freshness, "_fetch_espn",
            lambda league, window_days, now: [self.PLAYED],
        )
        monkeypatch.setattr(freshness, "_fetch_football_data", must_not_run)

        assert check_freshness(
            "PL", canonical_keys={self.PLAYED}).verdict is Verdict.FRESH


class TestLeagueIsolation:
    """The leagues have independent canonicals, pipelines and databases."""

    def test_one_league_being_stale_does_not_block_the_other(self) -> None:
        """ADR 0005: the gate is league-wide, never system-wide. A missing EFL
        result must not stop PL recommendations."""
        efl_played = (date(2026, 8, 15), "Bristol City", "Millwall")

        stale = check_freshness("EFL", finished=[efl_played], canonical_keys=set())
        fresh = check_freshness("PL", finished=[], canonical_keys=set())

        assert stale.verdict is Verdict.STALE
        assert fresh.verdict is Verdict.FRESH
        assert assert_fresh("PL", finished=[], canonical_keys=set()) is None


class TestWindow:
    """14 days, inclusive of today, never reaching into the future."""

    def test_window_spans_exactly_fourteen_days_back_from_today(self) -> None:
        """2x the weekly Sunday Data Refresh cadence, so one missed Sunday
        still leaves a missing fixture in view. 7 days would forgive it."""
        start, end = window_bounds(now=date(2026, 9, 20), window_days=WINDOW_DAYS)

        assert end == date(2026, 9, 20)
        assert start == date(2026, 9, 6)
        assert (end - start).days == 14

    def test_the_window_never_extends_past_today(self) -> None:
        """A fixture that has not kicked off cannot be missing from the
        canonical, so asking about the future would manufacture failures."""
        _, end = window_bounds(now=date(2026, 9, 20), window_days=WINDOW_DAYS)

        assert end <= date(2026, 9, 20)
