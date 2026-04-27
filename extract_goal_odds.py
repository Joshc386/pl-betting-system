"""
Extract GB Over/Under goal market odds from Betfair historical data.

Reads directly from .tar archives in the BetFairData directory — no need
to unpack bz2 files to disk.

Handles goal lines: 0.5, 1.5, 2.5, 3.5, 4.5

Outputs a single CSV with all lines, one row per match-line combination.
"""
import bz2
import csv
import json
import time
import tarfile
from pathlib import Path


TAR_DIR = Path(r"C:\Users\joshc\OneDrive\Documents\Project\BetFairData")
OUTPUT_DIR = Path(r"C:\Users\joshc\OneDrive\Documents\Project\data")
OUTPUT_CSV = OUTPUT_DIR / "betfair_goal_ou.csv"

# Market types to extract and their corresponding goal lines
GOAL_MARKETS: dict[str, float] = {
    "OVER_UNDER_05": 0.5,
    "OVER_UNDER_15": 1.5,
    "OVER_UNDER_25": 2.5,
    "OVER_UNDER_35": 3.5,
    "OVER_UNDER_45": 4.5,
}

CSV_FIELDS = [
    "event_name", "market_type", "goal_line", "market_time",
    "settled_time", "over_ltp", "under_ltp", "over_ltp_first",
    "under_ltp_first", "winner", "country_code",
]


def extract_all() -> None:
    """Scan all tar files in TAR_DIR and extract GB goal O/U market odds."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    total_files = 0
    skipped = 0

    start = time.time()
    counts_by_line: dict[str, int] = {k: 0 for k in GOAL_MARKETS}

    tar_files = sorted(TAR_DIR.glob("*.tar"))
    if not tar_files:
        print(f"No .tar files found in {TAR_DIR}")
        return

    print(f"Found {len(tar_files)} tar archives\n")

    for tar_path in tar_files:
        tar_start = time.time()
        tar_count = 0
        tar_extracted = 0

        print(f"Processing {tar_path.name}...", end=" ", flush=True)

        try:
            with tarfile.open(tar_path, "r") as tf:
                for member in tf:
                    if not member.name.endswith(".bz2"):
                        continue
                    tar_count += 1
                    total_files += 1

                    try:
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        raw_compressed = f.read()
                        result = _parse_bz2_bytes(raw_compressed)
                        if result is not None:
                            results.append(result)
                            tar_extracted += 1
                            mt = result["market_type"]
                            counts_by_line[mt] = counts_by_line.get(mt, 0) + 1
                    except Exception:
                        skipped += 1
                        continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        elapsed_tar = time.time() - tar_start
        print(f"{tar_count} files, {tar_extracted} goal markets ({elapsed_tar:.0f}s)")

    elapsed = time.time() - start
    print(f"\nScanned {total_files} bz2 files in {elapsed:.0f}s")
    if skipped:
        print(f"Skipped {skipped} files due to errors")

    # Deduplicate
    before_dedup = len(results)
    results = _dedup(results, ("event_name", "market_type", "market_time"))
    print(f"\nTotal goal O/U markets: {before_dedup} raw, "
          f"{len(results)} after dedup")

    print("\nBreakdown by line:")
    for mt in sorted(counts_by_line.keys()):
        count = counts_by_line[mt]
        if count > 0:
            print(f"  O/U {GOAL_MARKETS[mt]}: {count}")

    # Write CSV
    if results:
        _write_csv(OUTPUT_CSV, results, CSV_FIELDS)
        print(f"\nSaved to {OUTPUT_CSV}")
    else:
        print("\nNo results to save.")


def _dedup(results: list[dict], keys: tuple) -> list[dict]:
    """Remove duplicates based on key tuple."""
    seen: set = set()
    unique: list[dict] = []
    for r in results:
        key = tuple(r.get(k, "") for k in keys)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write list of dicts to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_bz2_bytes(raw_compressed: bytes) -> dict | None:
    """Parse bz2 bytes. Returns goal odds dict if it's a GB goal O/U market."""
    data = bz2.decompress(raw_compressed)
    lines = data.decode("utf-8").strip().split("\n")

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

            # Extract market definition
            if md.get("marketType"):
                market_type = md["marketType"]
                country_code = md.get("countryCode", "")
                event_name = md.get("eventName", "")
                market_time = md.get("marketTime", "")

                # Early exit: not a goal O/U market or not GB
                if market_type not in GOAL_MARKETS:
                    return None
                if country_code != "GB":
                    return None

                for r in md.get("runners", []):
                    rid = r["id"]
                    rname = r.get("name", "").lower()
                    if "over" in rname:
                        over_id = rid
                    elif "under" in rname:
                        under_id = rid

            # Check settlement
            if md.get("status") == "CLOSED" and md.get("settledTime"):
                settled_time = md["settledTime"]
                for r in md.get("runners", []):
                    rid = r["id"]
                    rstatus = r.get("status", "")
                    if rstatus == "WINNER":
                        if rid == over_id:
                            winner = "over"
                        elif rid == under_id:
                            winner = "under"

            # Extract prices
            if "rc" in mc:
                for rc in mc["rc"]:
                    rid = rc.get("id")
                    ltp = rc.get("ltp")
                    if ltp is None:
                        continue

                    if rid == over_id:
                        if over_ltp_first is None:
                            over_ltp_first = ltp
                        over_ltp = ltp
                    elif rid == under_id:
                        if under_ltp_first is None:
                            under_ltp_first = ltp
                        under_ltp = ltp

    # Validate
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
        "winner": winner or "",
        "country_code": "GB",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  Betfair Goal O/U Odds Extraction")
    print("=" * 60)
    print(f"\nSource: {TAR_DIR}")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Markets: {list(GOAL_MARKETS.values())}\n")
    extract_all()
