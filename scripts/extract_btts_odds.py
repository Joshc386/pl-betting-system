"""Extract GB BOTH_TEAMS_TO_SCORE market odds from Betfair historical data.

Walks the ``BTTS_data/`` tree directly (files already unpacked — no tar).
Parses every .bz2 file, keeps those with ``marketType == BOTH_TEAMS_TO_SCORE``
and ``countryCode == GB``, extracts Yes/No last-traded prices + first-traded
prices + settlement info, and writes a single CSV.

Downstream: ``scripts/extract_btts_by_league.py`` splits the output into
PL-only and EFL-only files using canonical team-name mapping.

Price columns emitted per runner (yes/no):
  *_ltp_first  — first-ever traded price in the stream (early pre-match)
  *_ltp_pre    — last traded price BEFORE inPlay flipped True (true
                 pre-match closing line; best available pre-match price)
  *_ltp        — last traded price in the entire stream (in-play /
                 settlement — DO NOT use as a pre-match price)

Run: ``python extract_btts_odds.py``
"""
from __future__ import annotations

import bz2
import csv
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BTTS_ROOT = Path(r"C:\Users\joshc\OneDrive\Documents\Project\BTTS_data")
OUTPUT_DIR = Path(r"C:\Users\joshc\OneDrive\Documents\Project\data")
OUTPUT_CSV = OUTPUT_DIR / "betfair_btts.csv"

TARGET_MARKET = "BOTH_TEAMS_TO_SCORE"
TARGET_COUNTRY = "GB"

CSV_FIELDS = [
    "event_name", "market_time", "settled_time",
    "yes_ltp", "no_ltp", "yes_ltp_first", "no_ltp_first",
    "yes_ltp_pre", "no_ltp_pre",
    "winner", "country_code",
]


def _parse_file(path_str: str) -> dict | None:
    """Parse one .bz2 file. Returns BTTS result dict or None if filtered out."""
    try:
        with open(path_str, "rb") as fh:
            raw = fh.read()
        data = bz2.decompress(raw)
    except Exception:
        return None

    try:
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

                # Early exit: wrong market or wrong country
                if market_type != TARGET_MARKET:
                    return None
                if country_code != TARGET_COUNTRY:
                    return None

                for r in md.get("runners", []):
                    rid = r["id"]
                    rname = r.get("name", "").lower().strip()
                    if rname == "yes":
                        yes_id = rid
                    elif rname == "no":
                        no_id = rid

            # Track inPlay transitions — once True, stop updating _pre
            if "inPlay" in md:
                in_play = bool(md["inPlay"])

            # Settlement
            if md.get("status") == "CLOSED" and md.get("settledTime"):
                settled_time = md["settledTime"]
                for r in md.get("runners", []):
                    rid = r["id"]
                    rstatus = r.get("status", "")
                    if rstatus == "WINNER":
                        if rid == yes_id:
                            winner = "yes"
                        elif rid == no_id:
                            winner = "no"

            # Prices
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

    # Validate
    if market_type != TARGET_MARKET:
        return None
    if country_code != TARGET_COUNTRY:
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


def extract_all(workers: int = 8) -> None:
    """Walk BTTS_ROOT and extract all BOTH_TEAMS_TO_SCORE GB markets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Source: {BTTS_ROOT}", flush=True)
    print(f"Output: {OUTPUT_CSV}", flush=True)
    print(f"Target market: {TARGET_MARKET}", flush=True)
    print(f"Target country: {TARGET_COUNTRY}", flush=True)
    print(f"Workers: {workers}\n", flush=True)

    start = time.time()
    print("Discovering .bz2 files ...", flush=True)
    all_files = [str(p) for p in BTTS_ROOT.rglob("*.bz2")]
    print(f"Discovered {len(all_files):,} .bz2 files "
          f"({time.time()-start:.1f}s)", flush=True)

    results: list[dict] = []
    scanned = 0
    batch = 10000
    t0 = time.time()

    # Use pool.map with chunksize so workers iterate lazily rather than
    # materialising 423k futures at once (which exhausts memory).
    # chunksize ~2000 balances IPC overhead vs buffer size.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_parse_file, all_files, chunksize=2000):
            scanned += 1
            if r is not None:
                results.append(r)
            if scanned % batch == 0:
                rate = scanned / (time.time() - t0 + 1e-9)
                pct = 100.0 * scanned / len(all_files)
                print(f"  ... {scanned:,}/{len(all_files):,} "
                      f"({pct:.1f}%) @ {rate:,.0f} files/s — "
                      f"{len(results):,} BTTS markets kept",
                      flush=True)

    elapsed = time.time() - start
    print(f"\nScanned {scanned:,} files in {elapsed:.0f}s "
          f"({scanned/elapsed:,.0f} files/s)", flush=True)
    print(f"Extracted {len(results):,} BTTS market records (pre-dedup)",
          flush=True)

    # Dedup on event + market_time (same market may appear multiple times if
    # the file contains replays of the feed).
    seen: set = set()
    unique: list[dict] = []
    for r in results:
        key = (r.get("event_name", ""), r.get("market_time", ""))
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"After dedup: {len(unique):,} unique BTTS markets")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(unique)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Betfair BTTS (BOTH_TEAMS_TO_SCORE) Odds Extraction")
    print("=" * 60)
    extract_all(workers=min(8, (os.cpu_count() or 4)))
