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


@pytest.fixture(autouse=True)
def _no_cached_fetches() -> None:
    """The fetch cache is module-level, so without this a FRESH fetch cached by
    one test answers for the next and the suite passes on stale evidence."""
    freshness.clear_cache()


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


class TestBoundariesAreWired:
    """The gate must actually be reachable from the paths it guards.

    Built as a separate step from the gate itself so a wiring bug could not be
    mistaken for a gate bug. These assert the wiring exists at all — a gate
    nothing calls is the failure mode that matters here.
    """

    def test_both_predictors_gate_their_recommendations(self) -> None:
        """One check inside generate_recommendations() covers all six
        recommendation call sites, so none is left to forget it."""
        import inspect

        import championship_predict
        import predict

        pl = inspect.getsource(predict.LivePredictor.generate_recommendations)
        efl = inspect.getsource(
            championship_predict.ChampionshipPredictor.generate_recommendations)

        assert 'assert_fresh("PL")' in pl
        assert 'assert_fresh("EFL")' in efl

    def test_every_train_site_is_gated(self) -> None:
        """Three train sites, not one. Blocking recommendations alone would
        still bake staleness into the pickles, where it survives the gate
        going green again."""
        import inspect

        import scan
        import scheduler

        # The scheduled entry point is a thin wrapper that holds the sleep
        # request open (keep_system_awake, added after Modern Standby killed a
        # retrain mid-run); the retrain body — and so the gate — moved into
        # _weekly_retrain. Both halves are asserted because either alone is
        # blind: reading the wrapper misses the gate entirely, and reading the
        # body would still pass if the wrapper stopped calling it.
        # Matched line-wise, not as a substring: "_weekly_retrain()" also
        # occurs inside "def job_weekly_retrain()", so a plain `in` check
        # passes even when the wrapper's body is empty.
        entry = inspect.getsource(scheduler.job_weekly_retrain)
        assert any(line.strip() == "_weekly_retrain()"
                   for line in entry.splitlines()), (
            "job_weekly_retrain no longer runs _weekly_retrain, so the gate "
            "asserted below is unreachable from the scheduled job"
        )

        retrain = inspect.getsource(scheduler._weekly_retrain)
        assert 'assert_fresh("PL")' in retrain
        assert 'assert_fresh("EFL")' in retrain

        # scan.py retrains inline when load_trained_state() fails — the site
        # ADR 0005 did not anticipate.
        scan_src = inspect.getsource(scan.run_scan)
        assert "assert_fresh(league)" in scan_src
        assert scan_src.index("assert_fresh(league)") < scan_src.index(
            "_predictor.train()"
        ), "the gate must precede the inline retrain, not follow it"

    def test_scan_reports_the_gate_by_name(self) -> None:
        """A blocked scan must say so. Falling through to the generic handler
        would log 'Predictor run during scan failed', burying which fixtures
        are missing and that the odds themselves are fine."""
        import inspect

        import scan

        source = inspect.getsource(scan.run_scan)
        assert "except FreshnessError" in source

        # Compared against the generic handler *in the same try block*, not the
        # first `except Exception` in the function — run_scan has an earlier,
        # unrelated one in the OddsPapi block.
        assert source.index("except FreshnessError") < source.index(
            "Predictor run during scan failed"
        ), "the specific handler must precede the generic one or it is dead code"


class TestCache:
    """A per-process cache, deliberately asymmetric: only FRESH is cached."""

    PLAYED = (date(2026, 8, 22), "Hull City AFC", "Manchester United FC")

    def test_a_fresh_verdict_is_not_refetched_within_the_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The saving. A Sunday retrain checks twice per league — once before
        train(), once inside generate_recommendations() — seconds apart."""
        freshness.clear_cache()
        calls = []

        def counting(league, window_days, now):
            calls.append(league)
            return []

        monkeypatch.setattr(freshness, "_fetch_espn", counting)

        check_freshness("PL", canonical_keys=set())
        check_freshness("PL", canonical_keys=set())

        assert len(calls) == 1

    def test_a_repaired_canonical_unblocks_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safety property, and the reason the verdict is not what's cached.

        A blocked gate is a thing the operator is fixing right now — re-running
        the ingest, adding the missing row. The fix must take effect on the next
        call, not after a TTL and not after restarting the dashboard, which is
        exactly the ritual a safety gate must not create. Caching only the fetch
        keeps the canonical re-read every time, so this holds without the cache
        being cleared.
        """
        monkeypatch.setattr(
            freshness, "_fetch_espn",
            lambda league, window_days, now: [self.PLAYED],
        )

        blocked = check_freshness("PL", canonical_keys=set())
        assert blocked.verdict is Verdict.STALE

        # Operator re-runs the ingest; the fixture is now in the canonical.
        repaired = check_freshness("PL", canonical_keys={self.PLAYED})

        assert repaired.verdict is Verdict.FRESH

    def test_an_unknown_verdict_is_never_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An outage that recovers must be picked up at once, not after a TTL."""
        freshness.clear_cache()
        calls = []

        def dead(league, window_days, now):
            calls.append(league)
            raise OSError("network down")

        monkeypatch.setattr(freshness, "_fetch_espn", dead)
        monkeypatch.setattr(freshness, "_fetch_football_data", dead)

        check_freshness("PL", canonical_keys=set())
        check_freshness("PL", canonical_keys=set())

        assert len(calls) == 4  # both authorities tried, both times

    def test_the_cache_is_per_league(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cached PL pass must not answer for EFL."""
        freshness.clear_cache()
        calls = []

        def counting(league, window_days, now):
            calls.append(league)
            return []

        monkeypatch.setattr(freshness, "_fetch_espn", counting)

        check_freshness("PL", canonical_keys=set())
        check_freshness("EFL", canonical_keys=set())

        assert calls == ["PL", "EFL"]


class TestWindow:
    """14 judged days, ending before the most recent ingest could have run."""

    def test_window_spans_fourteen_days(self) -> None:
        """2x the weekly Sunday Data Refresh cadence, so one missed Sunday
        still leaves a missing fixture in view. 7 days would forgive it."""
        start, end = window_bounds(now=date(2026, 9, 20), window_days=WINDOW_DAYS)

        assert (end - start).days == 14

    def test_fixtures_the_daily_ingest_has_not_seen_are_not_judged(self) -> None:
        """The false positive that would have fired on the 14 August opener.

        ESPN flips a fixture to completed at full time, but the canonical is
        only rebuilt by scripts/daily_ingest.py at 06:00 (ADR 0006). Between a
        Saturday 17:00 kickoff finishing and the next morning's ingest, the
        fixture is legitimately finished-and-absent — and the KO-1h scan for
        that evening's later fixtures falls inside that gap. Judging it would
        block betting on every matchday evening.

        Same shape as the sibling project's PUBLISH_GRACE: do not judge a
        source before it has had its chance to publish.
        """
        _, end = window_bounds(now=date(2026, 9, 20), window_days=WINDOW_DAYS)

        assert end < date(2026, 9, 20), (
            "today's finished fixtures cannot be in a canonical rebuilt at 06:00"
        )
        assert end == date(2026, 9, 18)

    def test_the_grace_covers_a_gate_run_at_any_hour(self) -> None:
        """Two days, not one, because the gate has no clock — only a date.

        A dashboard scan at 03:00 Monday runs *before* that morning's 06:00
        ingest, so Sunday's fixtures are still uncollected. A one-day grace
        would judge them and block. Detection cost is bounded: a missing
        fixture still surfaces well inside both the 14-day window and the
        weekly retrain cadence.
        """
        _, end = window_bounds(now=date(2026, 9, 20), window_days=WINDOW_DAYS)

        assert (date(2026, 9, 20) - end).days == 2
