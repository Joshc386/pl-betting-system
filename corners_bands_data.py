"""
Load and merge Corners Band market odds from Betfair historical exchange data.

Data source:
  - data/betfair_corner_bands.csv: extracted by extract_corner_odds.py
    Columns: event_name, market_type, market_time, settled_time, runners (JSON), country_code
    Runners JSON: {"9 or less": {ltp, ltp_first, status}, "10 - 12": {...}, "13 or more": {...}}

Provides Band_9orLess_Odds / Band_10to12_Odds / Band_13plus_Odds + settlement for backtesting.
"""
import json
import os
import numpy as np
import pandas as pd
from api.team_mapping import normalize, get_all_current_teams

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CORNER_BANDS_PATH = os.path.join(PROJECT_DIR, "data", "betfair_corner_bands.csv")

_EPL_TEAMS = set(get_all_current_teams())

# Runner name mapping
_BAND_KEYS = {
    "9 or less": "Band_9orLess",
    "10 - 12": "Band_10to12",
    "13 or more": "Band_13plus",
}


def _parse_runners(runners_json: str) -> dict:
    """Parse runners JSON into structured band odds and winner.

    Returns:
        Dict with keys: Band_9orLess_Odds, Band_10to12_Odds, Band_13plus_Odds,
                        Band_9orLess_Odds_First, Band_10to12_Odds_First, Band_13plus_Odds_First,
                        Band_Winner (canonical band name of winner, or None)
    """
    try:
        parsed = json.loads(runners_json)
    except (json.JSONDecodeError, TypeError):
        return {}

    result: dict = {}
    winner = None

    for runner_name, col_prefix in _BAND_KEYS.items():
        info = parsed.get(runner_name, {})
        result[f"{col_prefix}_Odds"] = info.get("ltp")
        result[f"{col_prefix}_Odds_First"] = info.get("ltp_first")
        if info.get("status") == "WINNER":
            winner = col_prefix

    result["Band_Winner"] = winner
    return result


def load_corner_bands(path: str | None = None) -> pd.DataFrame:
    """Load and normalize Betfair corner band market odds data.

    Args:
        path: Optional override path to the CSV file.

    Returns:
        DataFrame with columns:
            DateOnly, Home_Team, Away_Team, SeasonIndex,
            Band_9orLess_Odds, Band_10to12_Odds, Band_13plus_Odds,
            Band_9orLess_Odds_First, Band_10to12_Odds_First, Band_13plus_Odds_First,
            Band_Winner
    """
    csv_path = path or CORNER_BANDS_PATH
    if not os.path.exists(csv_path):
        print(f"  Warning: corner bands file not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Split event_name "Home v Away" into teams
    parts = df["event_name"].str.split(" v ", n=1, expand=True)
    if parts.shape[1] < 2:
        print("  Warning: could not split event_name into Home/Away")
        return pd.DataFrame()

    df["Home_Team_Raw"] = parts[0].str.strip()
    df["Away_Team_Raw"] = parts[1].str.strip()

    # Normalize team names
    df["Home_Team"] = df["Home_Team_Raw"].apply(normalize)
    df["Away_Team"] = df["Away_Team_Raw"].apply(normalize)

    # EPL filter
    mask = df["Home_Team"].isin(_EPL_TEAMS) & df["Away_Team"].isin(_EPL_TEAMS)
    non_epl = len(df) - mask.sum()
    df = df[mask].copy()
    if non_epl > 0:
        print(f"  Filtered out {non_epl} non-EPL matches")

    # Parse market_time -> DateOnly
    df["market_time_parsed"] = pd.to_datetime(df["market_time"], utc=True)
    df["DateOnly"] = df["market_time_parsed"].dt.strftime("%Y-%m-%d")

    # SeasonIndex
    month = df["market_time_parsed"].dt.month
    year = df["market_time_parsed"].dt.year
    df["SeasonIndex"] = np.where(month >= 8, year - 2000, year - 2001)

    # Parse runners JSON into band columns
    band_data = df["runners"].apply(_parse_runners).apply(pd.Series)
    df = pd.concat([df, band_data], axis=1)

    # Convert odds to numeric
    for col in ["Band_9orLess_Odds", "Band_10to12_Odds", "Band_13plus_Odds",
                "Band_9orLess_Odds_First", "Band_10to12_Odds_First", "Band_13plus_Odds_First"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where all 3 band odds are missing
    odds_cols = ["Band_9orLess_Odds", "Band_10to12_Odds", "Band_13plus_Odds"]
    all_missing = df[odds_cols].isna().all(axis=1)
    df = df[~all_missing].copy()

    # Select final columns
    keep = [
        "DateOnly", "Home_Team", "Away_Team", "SeasonIndex",
        "Band_9orLess_Odds", "Band_10to12_Odds", "Band_13plus_Odds",
        "Band_9orLess_Odds_First", "Band_10to12_Odds_First", "Band_13plus_Odds_First",
        "Band_Winner",
    ]
    result = df[[c for c in keep if c in df.columns]].copy()

    # Remove duplicates
    result = result.drop_duplicates(subset=["DateOnly", "Home_Team", "Away_Team"])

    return result


def merge_corner_bands(pipeline_df: pd.DataFrame,
                       bands_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge corner band odds into pipeline DataFrame.

    Args:
        pipeline_df: Main pipeline DataFrame with Date/DateOnly + Home_Team + Away_Team.
        bands_df: Pre-loaded band odds. If None, loads from CSV.

    Returns:
        pipeline_df with band odds and winner columns added.
    """
    if bands_df is None:
        bands_df = load_corner_bands()

    if bands_df.empty:
        print("  Warning: no corner band data available")
        return pipeline_df

    df = pipeline_df.copy()

    if "DateOnly" in df.columns:
        df["_merge_date"] = df["DateOnly"].astype(str)
    elif "Date" in df.columns:
        df["_merge_date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    else:
        print("  Warning: no date column found for corner bands merge")
        return pipeline_df

    odds = bands_df.copy()
    odds["_merge_date"] = odds["DateOnly"].astype(str)

    merge_cols = ["_merge_date", "Home_Team", "Away_Team"]
    odds_cols = [c for c in bands_df.columns
                 if c.startswith("Band_") and c not in merge_cols and c != "DateOnly"]

    merged = df.merge(
        odds[merge_cols + odds_cols],
        on=merge_cols,
        how="left",
    )
    merged.drop(columns=["_merge_date"], inplace=True)

    n_matched = merged["Band_9orLess_Odds"].notna().sum()
    print(f"  Corner bands merged: {n_matched}/{len(merged)} matches have band odds")

    return merged


if __name__ == "__main__":
    df = load_corner_bands()
    print(f"Loaded {len(df)} EPL matches with corner band odds")
    if not df.empty:
        print(f"Seasons: S{df['SeasonIndex'].min()}-S{df['SeasonIndex'].max()}")
        print(f"\nPer-season counts:")
        print(df.groupby("SeasonIndex").size().to_string())
        print(f"\nBand winner distribution:")
        print(df["Band_Winner"].value_counts().to_string())

        # Liquidity check: how many have all 3 bands priced
        has_all_3 = (df["Band_9orLess_Odds"].notna() &
                     df["Band_10to12_Odds"].notna() &
                     df["Band_13plus_Odds"].notna()).sum()
        print(f"\nMatches with all 3 bands priced: {has_all_3}/{len(df)}")

        print(f"\nOdds ranges:")
        for col in ["Band_9orLess_Odds", "Band_10to12_Odds", "Band_13plus_Odds"]:
            vals = df[col].dropna()
            print(f"  {col}: {vals.min():.2f} - {vals.max():.2f} (median {vals.median():.2f})")

        print(f"\nSample:")
        print(df.head(5).to_string(index=False))
