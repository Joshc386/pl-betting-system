"""Lightweight API quota tracker.

Surfaces credit consumption for both odds providers in a single JSON file
so the dashboard can show ``Odds API: 37/500 · OddsPapi: 5/250`` at a
glance and the operator can spot quota drift before hitting a hard cap.

Two write modes
---------------
* The Odds API exposes ``x-requests-remaining`` / ``x-requests-used``
  response headers — we just persist the most recent values.
* OddsPapi has no such header on the free tier — we increment a
  client-side counter on every successful call.

Both providers reset on the 1st of each month; the tracker auto-resets
the OddsPapi counter when ``last_call`` falls into a previous calendar
month.

Storage
-------
``data/api_quota.json``::

    {
      "odds_api":  {"remaining": 463, "used": 37,
                    "last_call": "2026-04-27T10:30:00",
                    "month": "2026-04"},
      "oddspapi":  {"calls_this_month": 5,
                    "last_call": "2026-04-27T10:30:00",
                    "month": "2026-04"}
    }

The file is intentionally simple JSON (not SQLite) — quota tracking is
advisory; if the file is corrupted or missing, callers fall back to
``unknown`` rather than failing the underlying API call.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Resolve relative to project root regardless of cwd
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUOTA_FILE = os.path.join(_PROJECT_DIR, "data", "api_quota.json")

# Documented monthly caps (free tier). Surfaced for the dashboard widget.
QUOTA_LIMITS = {
    "odds_api": 500,
    "oddspapi": 250,
}


def _now_month() -> str:
    """Return the current calendar month as ``YYYY-MM``."""
    return datetime.now().strftime("%Y-%m")


def _load() -> dict:
    """Load the quota file. Returns an empty skeleton on any failure."""
    if not os.path.exists(QUOTA_FILE):
        return {}
    try:
        with open(QUOTA_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning("quota_tracker: failed to load %s (%s) — resetting",
                       QUOTA_FILE, e)
        return {}


def _save(state: dict) -> None:
    """Write atomically (write+rename) so a crash mid-write can't corrupt."""
    os.makedirs(os.path.dirname(QUOTA_FILE), exist_ok=True)
    tmp = QUOTA_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, QUOTA_FILE)


def record_call(
    provider: str,
    remaining: Optional[int] = None,
    used: Optional[int] = None,
) -> None:
    """Record one API call.

    For The Odds API, pass the values parsed from response headers.
    For OddsPapi, call with no extra args — the tracker increments
    ``calls_this_month`` itself.

    Errors are swallowed (with a warning) so an upstream call never fails
    just because tracking has a problem.

    Args:
        provider: ``"odds_api"`` or ``"oddspapi"``.
        remaining: Quota remaining as reported by the API, if available.
        used: Quota used as reported by the API, if available.
    """
    if provider not in QUOTA_LIMITS:
        logger.warning("quota_tracker: unknown provider %r", provider)
        return

    try:
        state = _load()
        month = _now_month()
        now_iso = datetime.now().isoformat(timespec="seconds")
        entry = state.get(provider, {})

        # Rollover: clear counter when a new calendar month starts
        if entry.get("month") != month:
            entry = {"month": month}

        if provider == "odds_api":
            if remaining is not None:
                entry["remaining"] = int(remaining)
            if used is not None:
                entry["used"] = int(used)
        else:  # oddspapi — no header, count locally
            entry["calls_this_month"] = int(entry.get("calls_this_month", 0)) + 1

        entry["last_call"] = now_iso
        state[provider] = entry
        _save(state)
    except Exception as e:  # broad catch — tracking is advisory
        logger.warning("quota_tracker: record_call(%s) failed: %s", provider, e)


def read_quota() -> dict:
    """Return current quota snapshot for both providers.

    Always returns the same shape, even when no calls have been recorded
    yet (consumers don't have to handle missing keys). Each provider
    block contains ``used``, ``remaining`` (best-known), ``limit``, and
    ``last_call`` (ISO string or None).
    """
    state = _load()
    month = _now_month()
    out: dict[str, dict] = {}

    for provider, limit in QUOTA_LIMITS.items():
        entry = state.get(provider, {})
        # If the entry's month doesn't match the current month, treat it
        # as freshly rolled over (stale numbers aren't useful)
        if entry.get("month") != month:
            entry = {}

        if provider == "odds_api":
            used = entry.get("used")
            remaining = entry.get("remaining")
        else:
            used = entry.get("calls_this_month")
            remaining = (limit - used) if used is not None else None

        out[provider] = {
            "used": used,
            "remaining": remaining,
            "limit": limit,
            "last_call": entry.get("last_call"),
            "month": month,
        }
    return out


def is_quota_safe(provider: str, threshold: float = 0.95) -> bool:
    """Return False when usage has crossed the threshold for this provider.

    Used by API fetch sites as a hard guardrail: if we've hit, say, 95%
    of the monthly cap, refuse to fire any more requests until the
    operator swaps the key (Odds API) or the month rolls over (OddsPapi).

    Failure-mode posture: when we don't have data (no calls recorded
    this month, corrupt file, unknown provider), return ``True`` —
    "safe by default" — so an empty tracker file never blocks legitimate
    fetches. The dashboard's quota widget is the primary alert; this
    function only kicks in once we *know* we're near the cap.

    Args:
        provider: ``"odds_api"`` or ``"oddspapi"``.
        threshold: Block fraction. 0.95 = block at 95% used. Range (0, 1].

    Returns:
        True if it's safe to call the API; False if quota is too close
        to the cap.
    """
    if provider not in QUOTA_LIMITS:
        return True  # unknown provider — don't block
    limit = QUOTA_LIMITS[provider]
    snap = read_quota().get(provider, {})
    used = snap.get("used")
    if used is None:
        return True  # no data yet — assume fresh
    return (used / limit) < threshold


def _try_parse(header_value) -> Optional[int]:
    """Parse the Odds API quota header values (which may be '?' or missing).

    Helper for callers that pass raw header strings directly rather than
    converting to int themselves.
    """
    if header_value is None or header_value == "?":
        return None
    try:
        return int(header_value)
    except (TypeError, ValueError):
        return None
