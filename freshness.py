"""The Freshness Gate — ADR 0005.

A hard precondition on producing Recommendations and on running a Data Refresh:
every fixture an authoritative fixture list reports as finished within a rolling
window must be present in that league's Canonical Dataset.

See `CONTEXT.md`, "Freshness Gate", for the vocabulary and
`docs/adr/0005-freshness-gate.md` for the decisions and their reasoning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

# One finished fixture, keyed the way the Canonical Dataset keys it.
Fixture = tuple[date, str, str]

# Sized to the gate's own run cadence, not the fixture calendar — the window
# never needs to span a break, since zero finished fixtures passes by
# construction. The binding cadence is the weekly Sunday Data Refresh, so 14
# days is 2x margin: one missed Sunday still leaves a missing fixture in view,
# where 7 days would forgive it permanently.
WINDOW_DAYS = 14

# Do not judge a fixture the daily ingest has not had a chance to collect.
# ESPN flips a fixture to completed at full time, but the canonical is only
# rebuilt by scripts/daily_ingest.py at 06:00 (ADR 0006), so between a Saturday
# evening kickoff finishing and the next morning's ingest a fixture is
# legitimately finished-and-absent. Judging it would block betting on every
# matchday evening — the KO-1h scan for a 20:00 fixture falls squarely in that
# gap. Two days rather than one because the gate reasons in dates, not hours: a
# dashboard scan at 03:00 runs before that morning's ingest, so yesterday is
# not yet safe to judge either. Same principle as the sibling project's
# PUBLISH_GRACE — never judge a source before its publishing window closes.
PUBLISH_GRACE_DAYS = 2


class Verdict(Enum):
    """Three states, because two is what allowed the original bug.

    ``UNKNOWN`` is not a flavour of ``FRESH``. Collapsing "could not determine"
    into "nothing was missing" turns an outage into the gate's own pass
    condition — the substitution found in three live code paths on 2026-08-06.
    """

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GateResult:
    """One league's verdict, plus the evidence behind it."""

    league: str
    verdict: Verdict
    missing: list[Fixture]


def window_bounds(
    *,
    now: date,
    window_days: int = WINDOW_DAYS,
    grace_days: int = PUBLISH_GRACE_DAYS,
) -> tuple[date, date]:
    """The rolling window's inclusive ``(start, end)``.

    Pure, so the boundary is testable without a clock. ``end`` stops
    ``grace_days`` short of today so the gate only judges fixtures the daily
    ingest has certainly had a chance to collect — see ``PUBLISH_GRACE_DAYS``.
    The judged span stays ``window_days`` wide; the grace shifts it back rather
    than shortening it.
    """
    end = now - timedelta(days=grace_days)
    return end - timedelta(days=window_days), end


class FreshnessError(RuntimeError):
    """The gate blocked this league.

    Raised rather than signalled by an empty recommendation list, which is
    indistinguishable from a quiet Tuesday.
    """


class FreshnessUndetermined(RuntimeError):
    """No authority could answer, so freshness is unknown rather than fine.

    Deliberately an exception and not an empty list: a caller writing
    ``if not finished`` cannot accidentally coerce "could not determine" into
    "none were played", which is exactly how this class of bug propagates.
    """


def _fetch_espn(league: str, window_days: int, now: date) -> list[Fixture]:
    """Finished fixtures from ESPN — the first authority.

    Chosen first because it answers in each canonical's own name format, so no
    team resolver sits inside the gate.
    """
    from api.espn_scores import fetch_finished_window

    start, end = window_bounds(now=now, window_days=window_days)
    return fetch_finished_window(league, start, end)


# football-data.org returns long forms ("Queens Park Rangers FC") while the EFL
# canonical holds football-data.co.uk short forms. `resolve_feed_team` handles
# 21 of the 24 unaided; these three are the abbreviations ADR 0008 predicted
# would need explicit entries. PL needs none — all 20 resolve unaided.
_FOOTBALL_DATA_TO_CHAMP: dict[str, str] = {
    "Queens Park Rangers FC": "QPR",
    "Sheffield Wednesday FC": "Sheffield Weds",
    "West Bromwich Albion FC": "West Brom",
}


def _fetch_football_data(league: str, window_days: int, now: date) -> list[Fixture]:
    """Finished fixtures from football-data.org — the ordered fallback.

    Only reached when ESPN cannot answer. Unlike the ESPN path this one *does*
    carry a resolver, because the feed's names are not the canonical's.
    """
    import requests

    from api.team_resolver import resolve_feed_team
    from config import FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL
    from fixture_schedule import LEAGUE_COMPETITION_CODES

    start, end = window_bounds(now=now, window_days=window_days)
    comp = LEAGUE_COMPETITION_CODES[league]

    resp = requests.get(
        f"{FOOTBALL_DATA_BASE_URL}/competitions/{comp}/matches",
        params={"status": "FINISHED", "dateFrom": str(start), "dateTo": str(end)},
        headers={"X-Auth-Token": FOOTBALL_DATA_API_KEY},
        timeout=20,
    )
    resp.raise_for_status()

    our_teams = {home for _, home, _ in _canonical_keys(league)}
    finished: list[Fixture] = []
    for match in resp.json().get("matches", []):
        home = resolve_feed_team(
            match["homeTeam"]["name"], our_teams, _FOOTBALL_DATA_TO_CHAMP)
        away = resolve_feed_team(
            match["awayTeam"]["name"], our_teams, _FOOTBALL_DATA_TO_CHAMP)
        if not home or not away:
            unresolved = (match["homeTeam"]["name"] if not home
                          else match["awayTeam"]["name"])
            raise ValueError(
                f"football-data.org [{league}]: cannot resolve {unresolved!r} "
                f"to a Canonical Dataset name. Skipping it would shrink the "
                f"evidence the gate reconciles against."
            )
        finished.append((
            date.fromisoformat(match["utcDate"][:10]), home, away))

    return finished


def _canonical_keys(league: str) -> set[Fixture]:
    """Every ``(Date, Home_Team, Away_Team)`` in a league's Canonical Dataset.

    Reads the three key columns only — the canonical is ~8 MB and 69 columns
    wide, and the gate needs none of the rest.
    """
    import pandas as pd

    from league_config import get_league_config

    frame = pd.read_csv(
        get_league_config(league)["csv_path"],
        usecols=["Date", "Home_Team", "Away_Team"],
    )
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y-%m-%d").dt.date
    return set(
        zip(frame["Date"], frame["Home_Team"], frame["Away_Team"])
    )


def fetch_finished(
    league: str,
    *,
    window_days: int = WINDOW_DAYS,
    now: date | None = None,
) -> list[Fixture]:
    """Finished fixtures in the window, from the first authority that answers.

    ESPN first (it answers in each canonical's own name format, so no team
    resolver is needed), football-data.org second. Strictly ordered, never a
    vote — a disagreement between two sources would need semantics of its own.

    Raises:
        FreshnessUndetermined: If neither authority answered. Never returns an
            empty list to mean this; empty means "none were played".
    """
    now = now or date.today()
    failures: list[str] = []
    for fetch in (_fetch_espn, _fetch_football_data):
        try:
            return fetch(league, window_days, now)
        except Exception as e:  # any failure falls through to the next authority
            failures.append(f"{fetch.__name__}: {type(e).__name__}: {e}")
    raise FreshnessUndetermined(
        f"no authority could report {league} finished fixtures — "
        + "; ".join(failures)
    )


# The *authority fetch* is cached, never the verdict. The saving being bought is
# small and specific: a Sunday retrain checks twice per league, seconds apart —
# once before train(), once inside generate_recommendations() — and each check
# is an HTTP round trip on the path to producing recommendations.
#
# Caching the verdict instead would be actively unsafe. A blocked gate is a
# thing the operator is fixing right now, by re-running the ingest or adding the
# missing row; a cached STALE would keep betting blocked after the fix landed and
# make "restart the dashboard" the remedy, which is exactly the ritual a safety
# gate must not create. Caching only the fetch keeps the canonical re-read on
# every call, so a fix is picked up immediately. A failed fetch raises and is
# never cached, so an outage that recovers is picked up immediately too.
_CACHE_TTL_SECONDS = 600
_fetch_cache: dict[str, tuple[float, list[Fixture]]] = {}


def clear_cache() -> None:
    """Drop every cached fixture list. For tests and for an explicit re-check."""
    _fetch_cache.clear()


def check_freshness(
    league: str,
    *,
    window_days: int = WINDOW_DAYS,
    now: date | None = None,
    finished: list[Fixture] | None = None,
    canonical_keys: set[Fixture] | None = None,
) -> GateResult:
    """Reconcile a league's finished fixtures against its Canonical Dataset.

    ``finished`` and ``canonical_keys`` are fetched when not supplied; passing
    them makes the verdict testable without a network, the same shape as
    ``generate_recommendations(prefetched_matches=...)``.

    The authority fetch is cached per league for ``_CACHE_TTL_SECONDS``; the
    canonical is re-read and reconciled every call, so a fixed canonical is
    picked up at once — see the note above the cache.
    """
    if canonical_keys is None:
        canonical_keys = _canonical_keys(league)

    if finished is None:
        cached = _fetch_cache.get(league)
        if cached is not None and (
            time.monotonic() - cached[0] < _CACHE_TTL_SECONDS
        ):
            finished = cached[1]
        else:
            try:
                finished = fetch_finished(
                    league, window_days=window_days, now=now)
            except FreshnessUndetermined:
                return GateResult(
                    league=league, verdict=Verdict.UNKNOWN, missing=[])
            _fetch_cache[league] = (time.monotonic(), finished)

    missing = reconcile(finished, canonical_keys)
    verdict = Verdict.STALE if missing else Verdict.FRESH
    return GateResult(league=league, verdict=verdict, missing=missing)


def assert_fresh(league: str, **kwargs) -> None:
    """Raise unless this league's canonical is verifiably current.

    The boundary call — used before every ``train()`` and inside
    ``generate_recommendations()``. Mirrors ``assert_known_teams()`` from
    ADR 0008: one function asks, one insists.

    Raises:
        FreshnessError: On STALE, naming every missing fixture, or on UNKNOWN.
    """
    result = check_freshness(league, **kwargs)

    if result.verdict is Verdict.FRESH:
        return

    if result.verdict is Verdict.UNKNOWN:
        raise FreshnessError(
            f"{league} freshness could not be determined — no authority "
            f"answered. Blocking rather than assuming: for live capital, "
            f"refusing to act on unverifiable inputs is the correct direction."
        )

    named = "; ".join(
        f"{fixture_date:%Y-%m-%d} {home} v {away}"
        for fixture_date, home, away in result.missing
    )
    raise FreshnessError(
        f"{league} Canonical Dataset is missing {len(result.missing)} finished "
        f"fixture(s) — {named}. Re-run scripts/daily_ingest.py for this league; "
        f"recommendations and retraining are blocked until it is present."
    )


def reconcile(
    finished: list[Fixture],
    canonical_keys: set[Fixture],
) -> list[Fixture]:
    """Which finished fixtures are absent from the Canonical Dataset.

    Pure — no clock, no network, no CSV — so the gate's actual logic is testable
    directly. Returns the absent fixtures themselves rather than a count,
    because the gate has no bypass flag and the operator needs to know *which*
    fixture to chase.
    """
    return [fixture for fixture in finished if fixture not in canonical_keys]
