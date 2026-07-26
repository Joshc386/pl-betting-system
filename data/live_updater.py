"""Refresh the current season's Understat xG data.

**This module does not write the Canonical Dataset.** It used to: it scraped
Understat and appended goals-only rows to ``CompleteDSPL_CSV.csv`` with all
twenty stat columns hardcoded to ``NaN``. Running daily from the scheduler,
that left PL season 25 with no shots, corners, half-time scores or odds — the
whole Wheatcroft feature set empty for the season closest to live betting.

football-data.co.uk is the sole authority for Facts in both leagues; Understat
supplies **xG enrichment only**. See
``docs/adr/0004-canonical-composition-and-facts-provenance.md``, which retires
this path for Facts and keeps the xG scrape. Match results are ingested by
``data/build_canonical_dataset.py``, driven by ``scripts/daily_ingest.py``.

Usage:
    python -m data.live_updater                  # current season (auto-detect)
    python -m data.live_updater --season 2025    # specific Understat season
"""
import argparse
import os
import sys

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from api.understat_scraper import scrape_season
from understatapi import UnderstatClient

XG_PATH = os.path.join(PROJECT_DIR, "data", "understat_xg.csv")


def get_current_season_year() -> int:
    """Determine current Understat season year (e.g. 2025 for 2025/26)."""
    from datetime import date
    today = date.today()
    # Season starts in August; if before August, it's still the previous season
    if today.month < 8:
        return today.year - 1
    return today.year


def season_year_to_index(year: int) -> int:
    """Convert Understat season year to SeasonIndex. 2000 = index 0."""
    return year - 2000


def refresh_xg(season_year: int) -> int:
    """Scrape Understat for one season and refresh ``data/understat_xg.csv``.

    The season's existing rows are replaced wholesale rather than appended to,
    so a corrected upstream xG value propagates instead of duplicating.

    Args:
        season_year: Understat season year (e.g. 2025 for 2025/26).

    Returns:
        Number of matches scraped for the season (0 if none were found).
    """
    print(f"Fetching xG from Understat for {season_year}/{season_year + 1}...")
    with UnderstatClient() as client:
        xg_df = scrape_season(client, season_year)

    if xg_df.empty:
        print("  No matches found.")
        return 0

    print(f"  Found {len(xg_df)} completed matches")

    if os.path.exists(XG_PATH):
        existing = pd.read_csv(XG_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        season_str = f"{season_year}/{str(season_year + 1)[-2:]}"
        existing = existing[existing["season"] != season_str]
        updated = pd.concat([existing, xg_df], ignore_index=True)
    else:
        updated = xg_df

    updated = updated.sort_values("date")
    updated.to_csv(XG_PATH, index=False)
    print(f"  Updated xG data: {len(updated)} total matches -> {XG_PATH}")
    return len(xg_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None,
                        help="Understat season year (e.g. 2025). "
                             "Auto-detected if omitted.")
    args = parser.parse_args()

    season_year = args.season or get_current_season_year()
    print(f"Understat xG refresh for {season_year}/{season_year + 1} "
          f"(SeasonIndex {season_year_to_index(season_year)})")
    refresh_xg(season_year)
