"""Daily canonical-dataset ingestion.

Rebuilds each league's Canonical Dataset from football-data.co.uk (re-downloading
the current season, which gains fixtures every matchday), then re-derives the
Betfair League Splits.

Two design points worth knowing:

* **Full rebuild, no incremental append.** A rebuild costs ~17s for 26 EFL
  seasons off cached raws. An incremental append would be a second
  implementation of the same contract and would drift from the rebuild — the
  exact failure that left PL season 25 without shots/corners. See ADR 0004.

* **Splits must follow the canonical.** The League Split joins the Betfair GB
  Feed *against the Canonical Dataset*, so it is stale whenever the canonical
  moves — not only when new Betfair data arrives. Re-deriving here preserves
  that ordering rule (see CONTEXT.md, "League Split").

Scheduled by the "Betting Bot Daily Ingest" Task Scheduler entry.

Usage:
    python scripts/daily_ingest.py                 # default leagues
    python scripts/daily_ingest.py --leagues EFL PL
    python scripts/daily_ingest.py --dry-run       # build to temp, don't publish
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.build_canonical_dataset import build, _settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily_ingest")

# PL is absent because this job REPLACES the canonical, and for PL a replace
# is lossy. football-data.co.uk serves only 335 rows for seasons 3 and 4
# (2003/04, 2004/05) — verified against Content-Length, so it is an upstream
# gap, not a truncated download — where the PL canonical holds the full 380.
# It also lacks B365 odds for ~760 early-season fixtures the canonical has.
# A rebuild therefore drops 90 real fixtures, which is why the corrected
# Facts were merged onto the canonical rather than replacing it.
#
# UPDATE 2026-07-28: the builder now preserves canonical rows absent upstream
# and backfills cells the source omits, so a PL rebuild is verifiably lossless
# — same 9,880 rows, seasons 3-4 back to 380, no column with less coverage.
# The technical blocker is gone; adding "PL" here is now a deliberate decision
# rather than a bug waiting to happen, and it needs the PL canonical
# republished through the new builder first.
#
# Until that happens, PL still has NO in-season results ingestion (the
# Understat path was retired by ADR 0004). That is safe only while PL is
# between seasons — season 25 ended 2026-05-24 and 2026/27 starts 2026-08-21.
DEFAULT_LEAGUES = ["EFL"]

SPLIT_SCRIPTS = ("extract_btts_by_league.py", "extract_efl_ou15_betfair.py")

# How many pre-ingest backups to retain per league. The canonical is ~8 MB and
# this runs daily, so unbounded retention costs ~2.9 GB/year per league.
KEEP_BACKUPS = 5

# Facts are immutable across a regeneration (ADR 0001). If they move, the raw
# source or the column mapping changed and the result must not be published.
FACT_COLUMNS = ["Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals",
                "TG", "FTR", "SeasonIndex"]


def _facts_regressed(live_path: str, candidate: pd.DataFrame) -> str | None:
    """Compare a freshly built canonical against the live one.

    Returns:
        A description of the problem, or None if the rebuild is safe to publish.
    """
    if not os.path.exists(live_path):
        return None  # first build — nothing to compare against

    live = pd.read_csv(live_path)
    if len(candidate) < len(live):
        return (f"row count shrank: {len(live)} -> {len(candidate)}; "
                f"refusing to publish a canonical that lost matches")

    # Only historical seasons must be stable — the current season legitimately
    # gains rows as fixtures are played.
    current = candidate["SeasonIndex"].max()
    live_hist = live[live["SeasonIndex"] < current]
    cand_hist = candidate[candidate["SeasonIndex"] < current]

    if len(live_hist) != len(cand_hist):
        return (f"historical row count changed: {len(live_hist)} -> "
                f"{len(cand_hist)}")

    for col in FACT_COLUMNS:
        if col not in live_hist.columns or col not in cand_hist.columns:
            continue
        a = live_hist[col].astype(str).reset_index(drop=True)
        b = cand_hist[col].astype(str).reset_index(drop=True)
        if not a.equals(b):
            n = int((a != b).sum())
            return f"Fact column {col!r} changed in {n} historical row(s)"
    return None


def _schema_changed(live_path: str, candidate: pd.DataFrame) -> str | None:
    """Compare the candidate's column set against the live canonical's.

    Adding or removing a feature column changes what the models train on, so
    it belongs with a deliberate retrain rather than arriving overnight by
    cron. This job published a 74-column EFL canonical on 2026-07-29 the
    morning after ADR 0007 decision 2 landed — correct work, but unattended,
    against a model trained on the previous schema.

    Column *order* is not a change; only the set is compared.

    Returns:
        A description of the difference, or None if the schema is unchanged.
    """
    if not os.path.exists(live_path):
        return None  # first build — nothing to compare against

    live_columns = set(pd.read_csv(live_path, nrows=0).columns)
    candidate_columns = set(candidate.columns)

    added = sorted(candidate_columns - live_columns)
    removed = sorted(live_columns - candidate_columns)
    if not added and not removed:
        return None

    parts = []
    if added:
        parts.append(f"added {added}")
    if removed:
        parts.append(f"removed {removed}")
    return (f"schema changed ({'; '.join(parts)}); a feature-schema change "
            f"belongs with a retrain — re-run with --allow-schema-change once "
            f"that is intended")


def _prune_backups(live_path: str, keep: int = KEEP_BACKUPS) -> int:
    """Delete all but the newest ``keep`` pre-ingest backups.

    The canonical is ~8 MB and this job runs daily, so unpruned backups cost
    ~2.9 GB/year per league — and the repo lives in OneDrive, so every one of
    those would also sync to the cloud.

    Returns:
        Number of backups deleted.
    """
    pattern = f"{live_path}.pre-ingest.*.bak"
    backups = sorted(glob.glob(pattern))  # timestamp suffix sorts chronologically
    stale = backups[:-keep] if len(backups) > keep else []
    for path in stale:
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("could not remove old backup %s: %s", path, exc)
    if stale:
        logger.info("pruned %d old backup(s), kept newest %d",
                    len(stale), min(len(backups), keep))
    return len(stale)


def ingest_league(league: str, dry_run: bool = False,
                  allow_schema_change: bool = False) -> bool:
    """Rebuild one league's canonical and publish it if it is safe to do so.

    Args:
        league: League key to ingest.
        dry_run: Build to a temp path and report, without publishing.
        allow_schema_change: Permit publishing a canonical whose column set
            differs from the live one. Off by default: a feature-schema change
            belongs with a retrain, not with an overnight cron run.

    Returns:
        True on success, False on failure.
    """
    settings = _settings(league)
    live_path = settings["output_path"]

    tmp_dir = tempfile.mkdtemp(prefix=f"ingest_{league}_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(live_path))
    try:
        df = build(league=league, output=tmp_path)
    except Exception as exc:
        logger.error("%s build failed: %s", league, exc)
        return False

    problem = _facts_regressed(live_path, df)
    if problem is not None:
        logger.error("%s REFUSED - %s", league, problem)
        logger.error("  candidate left at %s for inspection", tmp_path)
        return False

    if not allow_schema_change:
        problem = _schema_changed(live_path, df)
        if problem is not None:
            logger.error("%s REFUSED - %s", league, problem)
            logger.error("  candidate left at %s for inspection", tmp_path)
            return False

    latest = pd.to_datetime(df["Date"], format="mixed", dayfirst=True,
                            errors="coerce").max()
    logger.info("%s built: %d rows, latest fixture %s",
                league, len(df), latest.date() if pd.notna(latest) else "?")

    if dry_run:
        logger.info("%s DRY RUN - not publishing (candidate at %s)",
                    league, tmp_path)
        return True

    # Back up before overwriting: the canonical is gitignored, so git offers
    # no rollback (ADR 0001).
    if os.path.exists(live_path):
        backup = f"{live_path}.pre-ingest.{datetime.now():%Y%m%d_%H%M%S}.bak"
        shutil.copy2(live_path, backup)
        _prune_backups(live_path)

    shutil.copy2(tmp_path, live_path)
    logger.info("%s published -> %s", league, live_path)
    return True


def refresh_splits() -> bool:
    """Re-derive the Betfair League Splits against the updated canonical."""
    ok = True
    for name in SPLIT_SCRIPTS:
        path = os.path.join(PROJECT_ROOT, "scripts", name)
        try:
            subprocess.run(
                [sys.executable, path], cwd=PROJECT_ROOT, check=True,
                capture_output=True, text=True, timeout=300,
            )
            logger.info("League Split refreshed: %s", name)
        except subprocess.CalledProcessError as exc:
            logger.error("League Split %s failed (exit %s): %s",
                         name, exc.returncode, (exc.stderr or "")[:300])
            ok = False
        except subprocess.TimeoutExpired:
            logger.error("League Split %s timed out", name)
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES,
                        help=f"Leagues to ingest (default: {DEFAULT_LEAGUES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build and validate but do not publish")
    parser.add_argument("--skip-splits", action="store_true",
                        help="Do not re-derive the Betfair League Splits")
    parser.add_argument("--allow-schema-change", action="store_true",
                        help="Publish even if the column set differs from the "
                             "live canonical. Use when a feature-schema change "
                             "is intended and a retrain is following it")
    args = parser.parse_args()

    logger.info("=== Daily ingest: %s ===", ", ".join(args.leagues))
    failed = [lg for lg in args.leagues
              if not ingest_league(lg, args.dry_run, args.allow_schema_change)]

    if not args.skip_splits and not args.dry_run:
        if not refresh_splits():
            failed.append("splits")

    if failed:
        logger.error("Daily ingest FAILED: %s", ", ".join(failed))
        return 1
    logger.info("Daily ingest complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
