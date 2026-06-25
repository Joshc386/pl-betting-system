"""Monthly Betfair historical data download and extraction.

Downloads the previous month's bz2 market files from Betfair's Historical
Data API for O/U 1.5, O/U 2.5, and BTTS markets (GB only), parses them,
appends to the existing CSVs, and re-runs the league-split scripts.

Uses raw HTTP requests to the Betfair Historic Data API because
betfairlightweight's ``get_file_list()`` sends malformed parameters
(snake_case strings instead of camelCase arrays) causing 400 errors.

Credentials loaded from .env (never hardcoded).

Usage:
    python scripts/betfair_monthly_update.py                # Previous month
    python scripts/betfair_monthly_update.py --year 2025 --month 4   # Specific month
    python scripts/betfair_monthly_update.py --dry-run       # List files, don't download
"""
from __future__ import annotations

import argparse
import bz2
import calendar
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import betfairlightweight
import requests
from dotenv import load_dotenv

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DATA_DIR = PROJECT_ROOT / "data"
CERTS_DIR = PROJECT_ROOT / "certs"

load_dotenv(PROJECT_ROOT / ".env")

# -- Credentials --
USERNAME = os.environ.get("BETFAIR_USERNAME", "")
PASSWORD = os.environ.get("BETFAIR_PASSWORD", "")
APP_KEY = os.environ.get("BETFAIR_APP_KEY", "")

# -- API endpoints --
FILE_LIST_URL = "https://historicdata.betfair.com/api/DownloadListOfFiles"
FILE_DOWNLOAD_URL = "https://historicdata.betfair.com/api/DownloadFile"

# -- Output CSVs --
GOAL_OU_CSV = DATA_DIR / "betfair_goal_ou.csv"
BTTS_CSV = DATA_DIR / "betfair_btts.csv"

# -- Off-season skip months --
# No PL or EFL Championship football in June/July.  The task still fires
# (Task Scheduler has no month-level exclusion) but exits early with a
# log message rather than hitting the API for nothing.
SKIP_MONTHS: set[int] = {6, 7}

# -- Market definitions --
GOAL_MARKETS: dict[str, float] = {
    "OVER_UNDER_15": 1.5,
    "OVER_UNDER_25": 2.5,
}

BTTS_MARKET = "BOTH_TEAMS_TO_SCORE"

# -- CSV schemas --
GOAL_OU_FIELDS = [
    "event_name", "market_type", "goal_line", "market_time",
    "settled_time", "over_ltp", "under_ltp", "over_ltp_first",
    "under_ltp_first", "over_ltp_pre", "under_ltp_pre",
    "winner", "country_code",
]

BTTS_FIELDS = [
    "event_name", "market_time", "settled_time",
    "yes_ltp", "no_ltp", "yes_ltp_first", "no_ltp_first",
    "yes_ltp_pre", "no_ltp_pre",
    "winner", "country_code",
]


# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------

def create_client() -> betfairlightweight.APIClient:
    """Create and authenticate Betfair API client.

    Uses cert-based (non-interactive/bot) login. The session token
    from the authenticated client is then used for raw HTTP requests
    to the Historical Data API.

    Returns:
        Authenticated APIClient whose .session_token we need.

    Raises:
        SystemExit: If credentials are missing or login fails.
    """
    if not USERNAME or not PASSWORD or not APP_KEY:
        print("ERROR: Betfair credentials not found in .env")
        print("Required: BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY")
        sys.exit(1)

    if not CERTS_DIR.exists():
        print(f"ERROR: Certs directory not found: {CERTS_DIR}")
        print("Generate certs per Betfair non-interactive login docs.")
        sys.exit(1)

    client = betfairlightweight.APIClient(
        username=USERNAME,
        password=PASSWORD,
        app_key=APP_KEY,
        certs=str(CERTS_DIR),
    )

    print("Logging in to Betfair (cert-based)...")
    try:
        client.login()
        print("  Login successful.")
    except Exception as e:
        print(f"  Login failed: {e}")
        sys.exit(1)

    return client


# -----------------------------------------------------------------------------
# Raw API calls (bypassing broken betfairlightweight wrapper)
# -----------------------------------------------------------------------------

def get_file_list(
    session_token: str,
    market_types: list[str],
    from_year: int,
    from_month: int,
    from_day: int,
    to_year: int,
    to_month: int,
    to_day: int,
) -> list[str]:
    """Fetch list of available bz2 file paths from the Betfair Historical API.

    Uses raw HTTP POST because betfairlightweight's get_file_list() sends
    snake_case string params instead of camelCase array params, causing 400s.

    Args:
        session_token: SSOID from authenticated client.
        market_types: e.g. ["OVER_UNDER_25"] or ["BOTH_TEAMS_TO_SCORE"].
        from_year: Start year.
        from_month: Start month (1-12).
        from_day: Start day (1-31).
        to_year: End year.
        to_month: End month (1-12).
        to_day: End day (1-31).

    Returns:
        List of remote file paths (strings).
    """
    headers = {
        "ssoid": session_token,
        "Content-Type": "application/json",
    }
    payload = {
        "sport": "Soccer",
        "plan": "Basic Plan",
        "fromDay": from_day,
        "fromMonth": from_month,
        "fromYear": from_year,
        "toDay": to_day,
        "toMonth": to_month,
        "toYear": to_year,
        "marketTypesCollection": market_types,
        "countriesCollection": ["GB"],
        "fileTypeCollection": ["M"],
    }

    resp = requests.post(FILE_LIST_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    # API returns a JSON list of file path strings
    file_list = resp.json()
    if isinstance(file_list, list):
        return file_list
    return []


def download_file(session_token: str, file_path: str) -> bytes:
    """Download a single bz2 file from the Betfair Historical API.

    Args:
        session_token: SSOID from authenticated client.
        file_path: Remote path returned by get_file_list().

    Returns:
        Raw bz2-compressed bytes.
    """
    headers = {"ssoid": session_token}
    resp = requests.get(
        FILE_DOWNLOAD_URL,
        headers=headers,
        params={"filePath": file_path},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


# -----------------------------------------------------------------------------
# bz2 parsing — mirrors extract_goal_odds.py and extract_btts_odds.py
# -----------------------------------------------------------------------------

def parse_goal_ou_bz2(raw_compressed: bytes) -> dict[str, Any] | None:
    """Parse a bz2-compressed Betfair goal O/U market file.

    Identical logic to extract_goal_odds._parse_bz2_bytes but accepts
    any GOAL_MARKETS key (not the full 0.5-4.5 range).

    Args:
        raw_compressed: Raw bz2 bytes from download.

    Returns:
        Dict matching GOAL_OU_FIELDS schema, or None if filtered out.
    """
    try:
        data = bz2.decompress(raw_compressed)
        lines = data.decode("utf-8").strip().split("\n")
    except Exception:
        return None

    market_type: str | None = None
    country_code: str | None = None
    event_name: str | None = None
    market_time: str | None = None
    settled_time: str | None = None

    over_id: int | None = None
    under_id: int | None = None
    over_ltp: float | None = None
    under_ltp: float | None = None
    over_ltp_first: float | None = None
    under_ltp_first: float | None = None
    over_ltp_pre: float | None = None
    under_ltp_pre: float | None = None
    in_play: bool = False
    winner: str | None = None

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "mc" not in obj:
            continue

        for mc in obj["mc"]:
            md = mc.get("marketDefinition", {})

            if md.get("marketType"):
                market_type = md["marketType"]
                country_code = md.get("countryCode", "")
                event_name = md.get("eventName", "")
                market_time = md.get("marketTime", "")

                if market_type not in GOAL_MARKETS:
                    return None
                if country_code != "GB":
                    return None
                if event_name and event_name.strip().lower().startswith(
                    ("test", "team ")
                ):
                    return None  # Betfair test/placeholder market

                for r in md.get("runners", []):
                    rid = r["id"]
                    rname = r.get("name", "").lower()
                    if "over" in rname:
                        over_id = rid
                    elif "under" in rname:
                        under_id = rid

            if "inPlay" in md:
                in_play = bool(md["inPlay"])

            if md.get("status") == "CLOSED" and md.get("settledTime"):
                settled_time = md["settledTime"]
                for r in md.get("runners", []):
                    rid = r["id"]
                    if r.get("status") == "WINNER":
                        if rid == over_id:
                            winner = "over"
                        elif rid == under_id:
                            winner = "under"

            if "rc" in mc:
                for rc in mc["rc"]:
                    rid = rc.get("id")
                    ltp = rc.get("ltp")
                    if ltp is None:
                        continue
                    if rid == over_id:
                        if over_ltp_first is None:
                            over_ltp_first = ltp
                        if not in_play:
                            over_ltp_pre = ltp
                        over_ltp = ltp
                    elif rid == under_id:
                        if under_ltp_first is None:
                            under_ltp_first = ltp
                        if not in_play:
                            under_ltp_pre = ltp
                        under_ltp = ltp

    if market_type not in GOAL_MARKETS:
        return None
    if country_code != "GB":
        return None
    if over_ltp is None and under_ltp is None:
        return None

    return {
        "event_name": event_name or "",
        "market_type": market_type,
        "goal_line": GOAL_MARKETS[market_type],
        "market_time": market_time or "",
        "settled_time": settled_time or "",
        "over_ltp": over_ltp,
        "under_ltp": under_ltp,
        "over_ltp_first": over_ltp_first,
        "under_ltp_first": under_ltp_first,
        "over_ltp_pre": over_ltp_pre,
        "under_ltp_pre": under_ltp_pre,
        "winner": winner or "",
        "country_code": "GB",
    }


def parse_btts_bz2(raw_compressed: bytes) -> dict[str, Any] | None:
    """Parse a bz2-compressed Betfair BTTS market file.

    Identical logic to extract_btts_odds._parse_file but works
    on raw bytes instead of a file path.

    Args:
        raw_compressed: Raw bz2 bytes from download.

    Returns:
        Dict matching BTTS_FIELDS schema, or None if filtered out.
    """
    try:
        data = bz2.decompress(raw_compressed)
        lines = data.decode("utf-8").strip().split("\n")
    except Exception:
        return None

    market_type: str | None = None
    country_code: str | None = None
    event_name: str | None = None
    market_time: str | None = None
    settled_time: str | None = None

    yes_id: int | None = None
    no_id: int | None = None
    yes_ltp: float | None = None
    no_ltp: float | None = None
    yes_ltp_first: float | None = None
    no_ltp_first: float | None = None
    yes_ltp_pre: float | None = None
    no_ltp_pre: float | None = None
    in_play: bool = False
    winner: str | None = None

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "mc" not in obj:
            continue

        for mc in obj["mc"]:
            md = mc.get("marketDefinition", {})

            if md.get("marketType"):
                market_type = md["marketType"]
                country_code = md.get("countryCode", "")
                event_name = md.get("eventName", "")
                market_time = md.get("marketTime", "")

                if market_type != BTTS_MARKET:
                    return None
                if country_code != "GB":
                    return None
                if event_name and event_name.strip().lower().startswith(
                    ("test", "team ")
                ):
                    return None  # Betfair test/placeholder market

                for r in md.get("runners", []):
                    rid = r["id"]
                    rname = r.get("name", "").lower().strip()
                    if rname == "yes":
                        yes_id = rid
                    elif rname == "no":
                        no_id = rid

            if "inPlay" in md:
                in_play = bool(md["inPlay"])

            if md.get("status") == "CLOSED" and md.get("settledTime"):
                settled_time = md["settledTime"]
                for r in md.get("runners", []):
                    rid = r["id"]
                    if r.get("status") == "WINNER":
                        if rid == yes_id:
                            winner = "yes"
                        elif rid == no_id:
                            winner = "no"

            if "rc" in mc:
                for rc in mc["rc"]:
                    rid = rc.get("id")
                    ltp = rc.get("ltp")
                    if ltp is None:
                        continue
                    if rid == yes_id:
                        if yes_ltp_first is None:
                            yes_ltp_first = ltp
                        if not in_play:
                            yes_ltp_pre = ltp
                        yes_ltp = ltp
                    elif rid == no_id:
                        if no_ltp_first is None:
                            no_ltp_first = ltp
                        if not in_play:
                            no_ltp_pre = ltp
                        no_ltp = ltp

    if market_type != BTTS_MARKET:
        return None
    if country_code != "GB":
        return None
    if yes_ltp is None and no_ltp is None:
        return None

    return {
        "event_name": event_name or "",
        "market_time": market_time or "",
        "settled_time": settled_time or "",
        "yes_ltp": yes_ltp,
        "no_ltp": no_ltp,
        "yes_ltp_first": yes_ltp_first,
        "no_ltp_first": no_ltp_first,
        "yes_ltp_pre": yes_ltp_pre,
        "no_ltp_pre": no_ltp_pre,
        "winner": winner or "",
        "country_code": "GB",
    }


# -----------------------------------------------------------------------------
# CSV helpers
# -----------------------------------------------------------------------------

def load_existing_keys(csv_path: Path, key_cols: list[str]) -> set[tuple[str, ...]]:
    """Load dedup keys from an existing CSV to avoid re-inserting rows.

    Args:
        csv_path: Path to existing CSV.
        key_cols: Column names that form the dedup key.

    Returns:
        Set of tuples, one per existing row.
    """
    if not csv_path.exists():
        return set()

    keys: set[tuple[str, ...]] = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = tuple(row.get(c, "") for c in key_cols)
            keys.add(key)
    return keys


def append_rows(
    csv_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    dedup_keys: list[str],
) -> int:
    """Append new rows to an existing CSV, skipping duplicates.

    If the CSV doesn't exist, creates it with headers.
    If the CSV exists but has fewer columns than fieldnames (e.g. missing
    _pre columns from an older extraction), the new rows are appended
    with the existing header — new columns are silently dropped so the
    file stays internally consistent. A future full re-extraction will
    add the _pre columns to all rows.

    Args:
        csv_path: Path to the CSV file.
        rows: New rows to append.
        fieldnames: Full schema (used for new files).
        dedup_keys: Column names for deduplication.

    Returns:
        Number of new rows actually appended.
    """
    existing_keys = load_existing_keys(csv_path, dedup_keys)

    # Filter out duplicates
    new_rows = []
    for row in rows:
        key = tuple(str(row.get(c, "")) for c in dedup_keys)
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)

    if not new_rows:
        return 0

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    if file_exists:
        # Read the existing header to stay consistent
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader)
        write_fields = existing_header
    else:
        write_fields = fieldnames

    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=write_fields, extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    return len(new_rows)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def get_previous_month(ref_date: date | None = None) -> tuple[int, int]:
    """Return (year, month) for the calendar month before ref_date.

    Args:
        ref_date: Reference date. Defaults to today.

    Returns:
        (year, month) tuple.
    """
    if ref_date is None:
        ref_date = date.today()
    if ref_date.month == 1:
        return ref_date.year - 1, 12
    return ref_date.year, ref_date.month - 1


def _download_one(
    session_token: str,
    file_path: str,
    parse_fn: Any,
) -> dict[str, Any] | None:
    """Download and parse a single bz2 file. Thread-safe.

    Args:
        session_token: Betfair SSOID.
        file_path: Remote file path.
        parse_fn: Parser function (parse_goal_ou_bz2 or parse_btts_bz2).

    Returns:
        Parsed dict or None.
    """
    try:
        raw = download_file(session_token, file_path)
        return parse_fn(raw)
    except Exception:
        return "ERROR"  # Sentinel to count errors


def _download_and_parse_batch(
    session_token: str,
    file_paths: list[str],
    parse_fn: Any,
    label: str,
    workers: int = 8,
) -> tuple[list[dict[str, Any]], int]:
    """Download and parse files concurrently using a thread pool.

    Args:
        session_token: Betfair SSOID.
        file_paths: List of remote file paths.
        parse_fn: Parser function.
        label: Display label for progress messages.
        workers: Number of concurrent download threads.

    Returns:
        (results, error_count) tuple.
    """
    if not file_paths:
        return [], 0

    print(f"\n{'-' * 40}")
    print(f"  Downloading & parsing {len(file_paths)} {label} files "
          f"({workers} threads)")
    print(f"{'-' * 40}", flush=True)

    results: list[dict[str, Any]] = []
    errors = 0
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, session_token, fp, parse_fn): fp
            for fp in file_paths
        }
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result == "ERROR":
                errors += 1
            elif result is not None:
                results.append(result)

            if done % 200 == 0 or done == len(file_paths):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(file_paths)}] "
                      f"{len(results)} kept, {errors} errors "
                      f"({rate:.1f} files/s)", flush=True)

    elapsed = time.time() - t0
    print(f"  {label}: {len(results)} markets from {len(file_paths)} files "
          f"in {elapsed:.0f}s ({errors} errors)", flush=True)
    return results, errors


def run_monthly_update(
    year: int,
    month: int,
    dry_run: bool = False,
    max_files: int | None = None,
) -> list[str]:
    """Download, parse, and append one month of Betfair historical data.

    Args:
        year: Calendar year to download.
        month: Calendar month (1-12) to download.
        dry_run: If True, list available files but don't download.
        max_files: If set, cap each market's file list (for testing).

    Returns:
        List of subprocess script names that failed during the
        league-split regeneration step. Empty list means full success.
        Used by main() to propagate a non-zero exit code so Task
        Scheduler surfaces silent post-processing failures.
    """
    last_day = calendar.monthrange(year, month)[1]
    month_label = f"{year}-{month:02d}"

    # Off-season gate: no PL/EFL football in June or July.
    # The previous month (which we'd be downloading) would be May or June
    # respectively, but the data month itself is what matters — if the
    # TARGET month is in SKIP_MONTHS, there's nothing useful to fetch.
    if month in SKIP_MONTHS:
        print("=" * 60)
        print(f"  Betfair Monthly Update: {month_label}")
        print("=" * 60)
        print(f"  SKIPPED -- month {month} is in the off-season "
              f"(SKIP_MONTHS = {sorted(SKIP_MONTHS)})")
        print(f"  No PL or EFL Championship football this month.")
        print("=" * 60)
        return []

    print("=" * 60)
    print(f"  Betfair Monthly Update: {month_label}")
    print("=" * 60)
    print(f"  Date range: {year}-{month:02d}-01 to {year}-{month:02d}-{last_day}")
    print(f"  Markets: O/U 1.5, O/U 2.5, BTTS")
    print(f"  Country: GB")
    print(f"  Goal O/U CSV: {GOAL_OU_CSV}")
    print(f"  BTTS CSV: {BTTS_CSV}")
    print()

    # -- Login --
    client = create_client()
    token = client.session_token
    print()

    # -- Discover files for each market type --
    all_goal_files: list[str] = []
    btts_files: list[str] = []

    for mtype in GOAL_MARKETS:
        print(f"  Listing {mtype} files...", end=" ", flush=True)
        try:
            files = get_file_list(
                token, [mtype],
                from_year=year, from_month=month, from_day=1,
                to_year=year, to_month=month, to_day=last_day,
            )
            print(f"{len(files)} files")
            all_goal_files.extend(files)
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"  Listing BTTS files...", end=" ", flush=True)
    try:
        files = get_file_list(
            token, [BTTS_MARKET],
            from_year=year, from_month=month, from_day=1,
            to_year=year, to_month=month, to_day=last_day,
        )
        print(f"{len(files)} files")
        btts_files.extend(files)
    except Exception as e:
        print(f"ERROR: {e}")

    # Cap file lists for testing
    if max_files is not None:
        all_goal_files = all_goal_files[:max_files]
        btts_files = btts_files[:max_files]
        print(f"\n  [TEST MODE] Capped to {max_files} files per market")

    total_files = len(all_goal_files) + len(btts_files)
    print(f"\n  Total files to process: {total_files}")
    print(f"    Goal O/U: {len(all_goal_files)}")
    print(f"    BTTS:     {len(btts_files)}")

    if dry_run:
        print("\n  [DRY RUN] Listing first 10 file paths per market:")
        for label, flist in [("Goal O/U", all_goal_files), ("BTTS", btts_files)]:
            print(f"\n  {label}:")
            for fp in flist[:10]:
                print(f"    {fp}")
            if len(flist) > 10:
                print(f"    ... and {len(flist) - 10} more")
        return []

    if total_files == 0:
        print("\n  No files available for this month. "
              "Data may not be published yet.")
        return []

    # -- Download and parse using concurrent threads --
    goal_results, goal_errors = _download_and_parse_batch(
        token, all_goal_files, parse_goal_ou_bz2, "goal O/U", workers=8,
    )
    btts_results, btts_errors = _download_and_parse_batch(
        token, btts_files, parse_btts_bz2, "BTTS", workers=8,
    )

    # -- Append to CSVs --
    print(f"\n{'-' * 40}")
    print(f"  Updating CSVs")
    print(f"{'-' * 40}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    goal_dedup_keys = ["event_name", "market_type", "market_time"]
    n_goal = append_rows(GOAL_OU_CSV, goal_results, GOAL_OU_FIELDS, goal_dedup_keys)
    print(f"  betfair_goal_ou.csv: +{n_goal} new rows "
          f"({len(goal_results) - n_goal} duplicates skipped)")

    btts_dedup_keys = ["event_name", "market_time"]
    n_btts = append_rows(BTTS_CSV, btts_results, BTTS_FIELDS, btts_dedup_keys)
    print(f"  betfair_btts.csv:    +{n_btts} new rows "
          f"({len(btts_results) - n_btts} duplicates skipped)")

    # -- Re-run league-split scripts --
    # Track any subprocess failures so we can (a) print accurate status
    # in the summary instead of falsely claiming "regenerated", and
    # (b) propagate a non-zero exit code through main() so Task Scheduler
    # surfaces the failure rather than logging LastTaskResult=0.
    failed_scripts: list[str] = []

    def _run_split(script_name: str) -> None:
        """Run one split script. Records the script name in failed_scripts
        if it returns non-zero or times out. Full stderr is printed to
        stdout so the .bat wrapper captures it in the rolling log.
        """
        print(f"  Running {script_name} ...", flush=True)
        try:
            subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / script_name)],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            print("    Done.")
        except subprocess.CalledProcessError as e:
            failed_scripts.append(script_name)
            print(f"    ERROR: {script_name} exited {e.returncode}. "
                  f"Full stderr below:")
            stderr_text = e.stderr or "(no stderr captured)"
            for line in stderr_text.splitlines():
                print(f"      {line}")
        except subprocess.TimeoutExpired:
            failed_scripts.append(script_name)
            print(f"    ERROR: {script_name} timed out after 300s")

    if n_goal > 0 or n_btts > 0:
        print(f"\n{'-' * 40}")
        print(f"  Re-running league-split scripts")
        print(f"{'-' * 40}")

        if n_btts > 0:
            _run_split("extract_btts_by_league.py")
        if n_goal > 0:
            _run_split("extract_efl_ou15_betfair.py")

    # -- Summary --
    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {month_label}")
    print(f"    Goal O/U rows added: {n_goal}")
    print(f"    BTTS rows added:     {n_btts}")
    if n_goal > 0 or n_btts > 0:
        if failed_scripts:
            print(f"    League splits:       FAILED -- "
                  f"{', '.join(failed_scripts)}")
        else:
            print(f"    League splits:       regenerated")
    print(f"{'=' * 60}")
    return failed_scripts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and process monthly Betfair historical data"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Year to download (default: previous month's year)"
    )
    parser.add_argument(
        "--month", type=int, default=None,
        help="Month to download, 1-12 (default: previous month)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List available files without downloading"
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Cap files per market type (for testing)"
    )
    args = parser.parse_args()

    if args.year and args.month:
        year, month = args.year, args.month
    elif args.year or args.month:
        print("ERROR: Specify both --year and --month, or neither.")
        return 1
    else:
        year, month = get_previous_month()

    if not 1 <= month <= 12:
        print(f"ERROR: Invalid month: {month}")
        return 1

    failed = run_monthly_update(year, month, dry_run=args.dry_run,
                                max_files=args.max_files)
    # Non-zero exit if any league-split subprocess failed -- the .bat
    # wrapper passes %ERRORLEVEL% to Task Scheduler so LastTaskResult
    # accurately reflects post-processing failures (no silent successes).
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
