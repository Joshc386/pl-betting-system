"""Diagnose why O/U 1.5 isn't showing in the dashboard.

Walks the odds caches and the match_analysis DB to find where alt-totals
data is being lost — at the API fetch step, the predictor evaluation
step, or the dashboard rendering step.
"""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def inspect_cache(label: str, path: str) -> None:
    print(f"\n=== {label}: {path} ===")
    if not os.path.exists(path):
        print("  (file does not exist)")
        return
    with open(path) as f:
        d = json.load(f)
    ms = d.get("data", [])
    ts = d.get("timestamp", "?")
    print(f"  cached at: {ts}")
    print(f"  fixtures:  {len(ms)}")

    if not ms:
        print("  (no fixtures in cache)")
        return

    # Per-fixture: count bookmakers with 1.5 lines, with 2.5, with btts
    line_counter: Counter = Counter()
    book_count_with_15 = 0
    book_count_with_25 = 0
    btts_book_count = 0

    for m in ms:
        for bk_key, bk in m.get("bookmakers", {}).items():
            lines = bk.get("all_lines") or {}
            for pt in lines.keys():
                line_counter[float(pt)] += 1
            if 1.5 in [float(p) for p in lines.keys()]:
                book_count_with_15 += 1
            if 2.5 in [float(p) for p in lines.keys()]:
                book_count_with_25 += 1
        btts_book_count += len(m.get("btts_bookmakers") or {})

    print(f"  bookmaker entries with line 1.5: {book_count_with_15}")
    print(f"  bookmaker entries with line 2.5: {book_count_with_25}")
    print(f"  btts bookmaker entries:          {btts_book_count}")
    print(f"  line distribution (point: count): "
          f"{dict(sorted(line_counter.items()))}")

    if ms:
        sample = ms[0]
        print(f"  sample fixture: {sample['home_team']} v {sample['away_team']}")
        print(f"    bookmakers: {len(sample.get('bookmakers', {}))}")
        print(f"    btts_bookmakers: {len(sample.get('btts_bookmakers', {}))}")
        if sample.get("bookmakers"):
            first_bk_key = next(iter(sample["bookmakers"]))
            first_bk = sample["bookmakers"][first_bk_key]
            print(f"    first bookmaker '{first_bk_key}' "
                  f"all_lines keys: {sorted(first_bk.get('all_lines', {}).keys())}")


def inspect_db(league: str, db_path: str) -> None:
    print(f"\n=== match_analysis ({league}): {db_path} ===")
    if not os.path.exists(db_path):
        print("  (DB does not exist)")
        return

    conn = sqlite3.connect(db_path)
    try:
        # Count by market
        rows = conn.execute(
            "SELECT market, COUNT(*) FROM match_analysis GROUP BY market"
        ).fetchall()
        print(f"  Rows by market: {dict(rows)}")

        # Latest scan timestamp
        latest = conn.execute(
            "SELECT MAX(scanned_at) FROM match_analysis"
        ).fetchone()[0]
        print(f"  Latest scanned_at: {latest}")

        # Check ou15 specifically
        ou15_count = conn.execute(
            "SELECT COUNT(*) FROM match_analysis WHERE market='ou15'"
        ).fetchone()[0]
        print(f"  ou15 rows: {ou15_count}")
    finally:
        conn.close()


if __name__ == "__main__":
    inspect_cache("PL Odds API cache",
                  os.path.join(PROJECT, "data/odds_cache.json"))
    inspect_cache("EFL Odds API cache",
                  os.path.join(PROJECT, "data/odds_cache_efl.json"))
    inspect_cache("PL OddsPapi cache",
                  os.path.join(PROJECT, "data/oddspapi_cache.json"))
    inspect_cache("EFL OddsPapi cache",
                  os.path.join(PROJECT, "data/oddspapi_cache_efl.json"))

    inspect_db("PL", os.path.join(PROJECT, "data/dashboard.db"))
    inspect_db("EFL", os.path.join(PROJECT, "data/dashboard_efl.db"))
