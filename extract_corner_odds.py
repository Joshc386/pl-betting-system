"""
Extract all GB corner market odds from Betfair historical data.
Handles two market types:
  - OVER_UNDER_105_CORNR: Over/Under 10.5 corners (main betting line)
  - CORNER_ODDS / Corners Number: Band markets (9 or less, 10-12, 13+)
Scans BASIC/{year}/{month}/{day}/{event}/ folders.
Outputs two CSVs: one for O/U 10.5, one for corner bands.
"""
import bz2
import csv
import json
import time
from pathlib import Path


BASE_DIR = Path(r"C:\Users\joshc\Downloads\data (2)\BASIC")
OUTPUT_DIR = Path(r"C:\Users\joshc\OneDrive\Documents\Project\data")
OUTPUT_OU = OUTPUT_DIR / "betfair_corner_ou105.csv"
OUTPUT_BANDS = OUTPUT_DIR / "betfair_corner_bands.csv"


def extract_all() -> None:
    """Scan all bz2 files and extract GB corner market odds."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ou_results: list[dict] = []
    band_results: list[dict] = []
    total_files = 0

    start = time.time()

    for year_dir in sorted(BASE_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        year = year_dir.name
        year_ou = 0
        year_bands = 0

        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for day_dir in sorted(month_dir.iterdir()):
                if not day_dir.is_dir():
                    continue
                for event_dir in sorted(day_dir.iterdir()):
                    if not event_dir.is_dir():
                        continue
                    for bz2_file in event_dir.glob("*.bz2"):
                        total_files += 1
                        try:
                            result = _parse_any_bz2(bz2_file)
                            if result is None:
                                continue
                            if result["_type"] == "ou":
                                ou_results.append(result)
                                year_ou += 1
                            elif result["_type"] == "bands":
                                band_results.append(result)
                                year_bands += 1
                        except Exception:
                            continue

        if year_ou > 0 or year_bands > 0:
            print(f"  {year}: {year_ou} O/U 10.5, {year_bands} band markets")

    elapsed = time.time() - start
    print(f"\nScanned {total_files} bz2 files in {elapsed:.0f}s")

    # Deduplicate
    ou_results = _dedup(ou_results, ("event_name", "market_type", "market_time"))
    band_results = _dedup(band_results, ("event_name", "market_type", "market_time"))

    print(f"O/U 10.5 markets: {len(ou_results)}")
    print(f"Band markets: {len(band_results)}")

    # Write O/U CSV
    if ou_results:
        _write_csv(OUTPUT_OU, ou_results, [
            "event_name", "market_type", "market_time", "settled_time",
            "over_ltp", "under_ltp", "over_ltp_first", "under_ltp_first",
            "winner", "country_code",
        ])
        print(f"Saved O/U 10.5 to {OUTPUT_OU}")

    # Write bands CSV
    if band_results:
        _write_csv(OUTPUT_BANDS, band_results, [
            "event_name", "market_type", "market_time", "settled_time",
            "runners", "country_code",
        ])
        print(f"Saved bands to {OUTPUT_BANDS}")


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


def _parse_any_bz2(filepath: Path) -> dict | None:
    """Parse any bz2 file. Returns corner odds dict if it's a GB corner market."""
    with open(filepath, "rb") as f:
        data = bz2.decompress(f.read())

    lines = data.decode("utf-8").strip().split("\n")

    market_type = None
    country_code = None
    event_name = None
    market_time = None
    settled_time = None

    # O/U tracking
    over_id = None
    under_id = None
    over_ltp = None
    under_ltp = None
    over_ltp_first = None
    under_ltp_first = None
    winner = None

    # Band tracking (for CORNER_ODDS markets)
    runner_info: dict[int, dict] = {}  # id -> {name, ltp, ltp_first, status}

    for line in lines:
        obj = json.loads(line)
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

                for r in md.get("runners", []):
                    rid = r["id"]
                    rname = r.get("name", "")
                    name_lower = rname.lower()

                    if market_type == "OVER_UNDER_105_CORNR":
                        if "over" in name_lower:
                            over_id = rid
                        elif "under" in name_lower:
                            under_id = rid
                    else:
                        # CORNER_ODDS / Corners Number — track all runners
                        if rid not in runner_info:
                            runner_info[rid] = {
                                "name": rname, "ltp": None,
                                "ltp_first": None, "status": "ACTIVE",
                            }
                        runner_info[rid]["name"] = rname

            # Check settlement
            if md.get("status") == "CLOSED" and md.get("settledTime"):
                settled_time = md["settledTime"]
                for r in md.get("runners", []):
                    rid = r["id"]
                    rstatus = r.get("status", "")
                    if market_type == "OVER_UNDER_105_CORNR":
                        if rstatus == "WINNER":
                            if rid == over_id:
                                winner = "over"
                            elif rid == under_id:
                                winner = "under"
                    else:
                        if rid in runner_info:
                            runner_info[rid]["status"] = rstatus

            # Extract prices
            if "rc" in mc:
                for rc in mc["rc"]:
                    rid = rc.get("id")
                    ltp = rc.get("ltp")
                    if ltp is None:
                        continue

                    if market_type == "OVER_UNDER_105_CORNR":
                        if rid == over_id:
                            if over_ltp_first is None:
                                over_ltp_first = ltp
                            over_ltp = ltp
                        elif rid == under_id:
                            if under_ltp_first is None:
                                under_ltp_first = ltp
                            under_ltp = ltp
                    else:
                        if rid not in runner_info:
                            runner_info[rid] = {
                                "name": "", "ltp": None,
                                "ltp_first": None, "status": "ACTIVE",
                            }
                        if runner_info[rid]["ltp_first"] is None:
                            runner_info[rid]["ltp_first"] = ltp
                        runner_info[rid]["ltp"] = ltp

    # Filter: only GB corner markets
    if country_code != "GB":
        return None

    if market_type == "OVER_UNDER_105_CORNR":
        if over_ltp is None and under_ltp is None:
            return None
        return {
            "_type": "ou",
            "event_name": event_name or "",
            "market_type": market_type,
            "market_time": market_time or "",
            "settled_time": settled_time or "",
            "over_ltp": over_ltp,
            "under_ltp": under_ltp,
            "over_ltp_first": over_ltp_first,
            "under_ltp_first": under_ltp_first,
            "winner": winner or "",
            "country_code": "GB",
        }
    elif market_type == "CORNER_ODDS":
        # Serialize runner data as JSON string
        priced_runners = {
            info["name"]: {
                "ltp": info["ltp"],
                "ltp_first": info["ltp_first"],
                "status": info["status"],
            }
            for rid, info in runner_info.items()
            if info["ltp"] is not None
        }
        if not priced_runners:
            return None
        return {
            "_type": "bands",
            "event_name": event_name or "",
            "market_type": market_type,
            "market_time": market_time or "",
            "settled_time": settled_time or "",
            "runners": json.dumps(priced_runners),
            "country_code": "GB",
        }

    return None


if __name__ == "__main__":
    extract_all()
