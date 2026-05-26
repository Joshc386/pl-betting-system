"""
Download Betfair historical goal O/U market data programmatically.

Bypasses the broken Betfair Historic Data website UI by using the
betfairlightweight API directly. Downloads GB Soccer goal O/U markets
for seasons 2016-2026.

Credentials loaded from .env (never hardcoded).

Usage:
    python download_betfair_goals.py           # Download all seasons
    python download_betfair_goals.py --check   # Just check what's available
"""
import os
import sys
import time
import argparse
from pathlib import Path

import betfairlightweight
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Configuration ──
USERNAME = os.environ.get("BETFAIR_USERNAME", "")
PASSWORD = os.environ.get("BETFAIR_PASSWORD", "")
APP_KEY = os.environ.get("BETFAIR_APP_KEY", "")

DOWNLOAD_DIR = Path(os.path.dirname(__file__)) / "data" / "betfair_goals_raw"

# Goal O/U market types to download
GOAL_MARKETS = [
    "OVER_UNDER_05",
    "OVER_UNDER_15",
    "OVER_UNDER_25",
    "OVER_UNDER_35",
    "OVER_UNDER_45",
    "OVER_UNDER_55",
]

# Seasons: Aug year to May year+1
SEASONS = [
    (2016, 2017), (2017, 2018), (2018, 2019), (2019, 2020),
    (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024),
    (2024, 2025), (2025, 2026),
]


def create_client() -> betfairlightweight.APIClient:
    """Create and authenticate Betfair API client."""
    if not USERNAME or not PASSWORD or not APP_KEY:
        print("ERROR: Betfair credentials not found in .env")
        print("Required: BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY")
        sys.exit(1)

    certs_dir = os.path.join(os.path.dirname(__file__), "..", "certs")
    client = betfairlightweight.APIClient(
        username=USERNAME,
        password=PASSWORD,
        app_key=APP_KEY,
        certs=certs_dir,
    )

    print("Logging in to Betfair (cert-based)...")
    try:
        client.login()
        print("  Login successful.\n")
    except Exception as e:
        print(f"  Login failed: {e}")
        print("\n  If using 2FA, you may need to generate a certificate.")
        print("  See: https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Non-Interactive+(bot)+login")
        sys.exit(1)

    return client


def check_availability(client: betfairlightweight.APIClient) -> None:
    """Check what data is available for each season without downloading."""
    print("=" * 60)
    print("  Checking data availability per season")
    print("=" * 60)

    for start_year, end_year in SEASONS:
        print(f"\n  Season {start_year}/{end_year}:")

        try:
            options = client.historic.get_collection_options(
                sport="Soccer",
                plan="Basic Plan",
                from_day="1",
                from_month="8",
                from_year=str(start_year),
                to_day="31",
                to_month="5",
                to_year=str(end_year),
                market_types_collection=",".join(GOAL_MARKETS),
                countries_collection="GB",
                file_type_collection="M",
            )
            print(f"    Collection options: {options}")
        except Exception as e:
            print(f"    Error: {e}")

        try:
            size = client.historic.get_data_size(
                sport="Soccer",
                plan="Basic Plan",
                from_day="1",
                from_month="8",
                from_year=str(start_year),
                to_day="31",
                to_month="5",
                to_year=str(end_year),
                market_types_collection=",".join(GOAL_MARKETS),
                countries_collection="GB",
                file_type_collection="M",
            )
            print(f"    Data size: {size}")
        except Exception as e:
            print(f"    Size error: {e}")

        time.sleep(1)  # Be polite to the API


def download_season(
    client: betfairlightweight.APIClient,
    start_year: int,
    end_year: int,
) -> int:
    """Download goal O/U data for a single season.

    Args:
        client: Authenticated Betfair client.
        start_year: Season start year (e.g. 2016).
        end_year: Season end year (e.g. 2017).

    Returns:
        Number of files downloaded.
    """
    season_dir = DOWNLOAD_DIR / f"{start_year}_{end_year}"
    season_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Getting file list for {start_year}/{end_year}...")

    try:
        file_list = client.historic.get_file_list(
            sport="Soccer",
            plan="Basic Plan",
            from_day="1",
            from_month="8",
            from_year=str(start_year),
            to_day="31",
            to_month="5",
            to_year=str(end_year),
            market_types_collection=",".join(GOAL_MARKETS),
            countries_collection="GB",
            file_type_collection="M",
        )
    except Exception as e:
        print(f"  Error getting file list: {e}")
        return 0

    if not file_list:
        print(f"  No files available for {start_year}/{end_year}")
        return 0

    # file_list should be a list of file paths
    files = file_list if isinstance(file_list, list) else [file_list]
    print(f"  Found {len(files)} files to download")

    downloaded = 0
    for i, file_path in enumerate(files):
        try:
            # Check if it's a dict with a path key or a string
            if isinstance(file_path, dict):
                fp = file_path.get("filePath", file_path.get("file_path", ""))
            else:
                fp = str(file_path)

            if not fp:
                continue

            # Check if already downloaded
            local_name = Path(fp).name
            local_path = season_dir / local_name
            if local_path.exists():
                downloaded += 1
                continue

            print(f"    [{i+1}/{len(files)}] Downloading {local_name}...")
            client.historic.download_file(
                file_path=fp,
                store_directory=str(season_dir),
            )
            downloaded += 1
            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"    Failed: {e}")
            continue

    print(f"  Downloaded {downloaded}/{len(files)} files to {season_dir}")
    return downloaded


def download_all(client: betfairlightweight.APIClient) -> None:
    """Download goal O/U data for all seasons."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Betfair Historical Goal O/U Download")
    print("=" * 60)
    print(f"  Markets: {GOAL_MARKETS}")
    print(f"  Country: GB")
    print(f"  Seasons: {SEASONS[0][0]} to {SEASONS[-1][1]}")
    print(f"  Output: {DOWNLOAD_DIR}")

    total_files = 0
    for start_year, end_year in SEASONS:
        print(f"\n{'─' * 40}")
        print(f"  Season {start_year}/{end_year}")
        print(f"{'─' * 40}")

        n = download_season(client, start_year, end_year)
        total_files += n
        time.sleep(2)  # Pause between seasons

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE: {total_files} files downloaded")
    print(f"  Location: {DOWNLOAD_DIR}")
    print(f"\n  Next step: run extract_goal_odds.py to parse the data")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Betfair historical goal O/U data"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Just check availability, don't download"
    )
    parser.add_argument(
        "--season", type=int, default=None,
        help="Download a single season start year (e.g. 2020 for 2020/21)"
    )
    args = parser.parse_args()

    client = create_client()

    if args.check:
        check_availability(client)
    elif args.season:
        end = args.season + 1
        download_season(client, args.season, end)
    else:
        download_all(client)
