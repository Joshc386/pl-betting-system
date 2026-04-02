"""
Load and merge Corners Over/Under 10.5 odds from Betfair historical exchange data.

Data source:
  - data/betfair_corner_ou105.csv: extracted by extract_corner_odds.py
    Columns: event_name, market_type, market_time, settled_time,
             over_ltp, under_ltp, over_ltp_first, under_ltp_first, winner, country_code

Provides Corner_Over_Odds / Corner_Under_Odds + settlement for backtesting.
"""
import os
import numpy as np
import pandas as pd
from api.team_mapping import normalize, get_all_current_teams

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Prefer new extraction output; fall back to old single-file extraction
_NEW_PATH = os.path.join(PROJECT_DIR, "data", "betfair_corner_ou105.csv")
_OLD_PATH = os.path.join(PROJECT_DIR, "data", "betfair_corner_odds.csv")
CORNER_OU_PATH = _NEW_PATH if os.path.exists(_NEW_PATH) else _OLD_PATH

# EPL teams: all canonical names from team_mapping
_EPL_TEAMS = set(get_all_current_teams())


def load_corner_odds(path: str | None = None) -> pd.DataFrame:
    """Load and normalize Betfair corner O/U 10.5 odds data.

    Args:
        path: Optional override path to the CSV file.

    Returns:
        DataFrame with columns:
            DateOnly, Home_Team, Away_Team, SeasonIndex,
            Corner_Over_Odds, Corner_Under_Odds, Corner_Winner
    """
    csv_path = path or CORNER_OU_PATH
    if not os.path.exists(csv_path):
        print(f"  Warning: corner odds file not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Split event_name "Home v Away" into teams
    parts = df["event_name"].str.split(" v ", n=1, expand=True)
    if parts.shape[1] < 2:
        print("  Warning: could not split event_name into Home/Away")
        return pd.DataFrame()

    df["Home_Team_Raw"] = parts[0].str.strip()
    df["Away_Team_Raw"] = parts[1].str.strip()

    # Normalize team names to canonical CSV names
    df["Home_Team"] = df["Home_Team_Raw"].apply(normalize)
    df["Away_Team"] = df["Away_Team_Raw"].apply(normalize)

    # EPL filter: keep only matches where BOTH teams are in the EPL whitelist
    mask = df["Home_Team"].isin(_EPL_TEAMS) & df["Away_Team"].isin(_EPL_TEAMS)
    non_epl = len(df) - mask.sum()
    df = df[mask].copy()
    if non_epl > 0:
        print(f"  Filtered out {non_epl} non-EPL matches")

    # Parse market_time -> DateOnly
    df["market_time_parsed"] = pd.to_datetime(df["market_time"], utc=True)
    df["DateOnly"] = df["market_time_parsed"].dt.strftime("%Y-%m-%d")

    # Compute SeasonIndex: Aug-Dec = season starting that year, Jan-Jul = season starting previous year
    month = df["market_time_parsed"].dt.month
    year = df["market_time_parsed"].dt.year
    df["SeasonIndex"] = np.where(month >= 8, year - 2000, year - 2001)

    # Odds columns
    df["Corner_Over_Odds"] = pd.to_numeric(df["over_ltp"], errors="coerce")
    df["Corner_Under_Odds"] = pd.to_numeric(df["under_ltp"], errors="coerce")

    # Derive missing odds from the other side: implied_prob = 1/odds
    # If one side is known: other_odds = 1 / (1 - 1/known_odds)
    # This assumes ~100% book (exchange has no margin, close enough)
    over_missing = df["Corner_Over_Odds"].isna() & df["Corner_Under_Odds"].notna()
    under_missing = df["Corner_Under_Odds"].isna() & df["Corner_Over_Odds"].notna()

    df.loc[over_missing, "Corner_Over_Odds"] = (
        1.0 / (1.0 - 1.0 / df.loc[over_missing, "Corner_Under_Odds"])
    )
    df.loc[under_missing, "Corner_Under_Odds"] = (
        1.0 / (1.0 - 1.0 / df.loc[under_missing, "Corner_Over_Odds"])
    )

    # Drop rows where both odds are still missing
    both_missing = df["Corner_Over_Odds"].isna() & df["Corner_Under_Odds"].isna()
    df = df[~both_missing].copy()

    # Settlement
    df["Corner_Winner"] = df["winner"].str.strip().str.lower()

    # Select final columns
    keep = [
        "DateOnly", "Home_Team", "Away_Team", "SeasonIndex",
        "Corner_Over_Odds", "Corner_Under_Odds", "Corner_Winner",
    ]
    result = df[keep].copy()

    # Remove duplicates (same match, same date)
    result = result.drop_duplicates(subset=["DateOnly", "Home_Team", "Away_Team"])

    return result


def merge_corner_odds(pipeline_df: pd.DataFrame,
                      corner_odds_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Merge corner O/U 10.5 odds into pipeline DataFrame.

    Args:
        pipeline_df: Main pipeline DataFrame with Date/DateOnly + Home_Team + Away_Team.
        corner_odds_df: Pre-loaded corner odds. If None, loads from CSV.

    Returns:
        pipeline_df with Corner_Over_Odds, Corner_Under_Odds, Corner_Winner columns added.
    """
    if corner_odds_df is None:
        corner_odds_df = load_corner_odds()

    if corner_odds_df.empty:
        print("  Warning: no corner odds data available")
        return pipeline_df

    df = pipeline_df.copy()

    # Create string date column for merging
    if "DateOnly" in df.columns:
        df["_merge_date"] = df["DateOnly"].astype(str)
    elif "Date" in df.columns:
        df["_merge_date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    else:
        print("  Warning: no date column found for corner odds merge")
        return pipeline_df

    odds = corner_odds_df.copy()
    odds["_merge_date"] = odds["DateOnly"].astype(str)

    merge_cols = ["_merge_date", "Home_Team", "Away_Team"]
    odds_cols = ["Corner_Over_Odds", "Corner_Under_Odds", "Corner_Winner"]

    merged = df.merge(
        odds[merge_cols + odds_cols],
        on=merge_cols,
        how="left",
    )
    merged.drop(columns=["_merge_date"], inplace=True)

    n_matched = merged["Corner_Over_Odds"].notna().sum()
    print(f"  Corner odds merged: {n_matched}/{len(merged)} matches have odds")

    return merged


if __name__ == "__main__":
    df = load_corner_odds()
    print(f"Loaded {len(df)} EPL matches with corner O/U 10.5 odds")
    if not df.empty:
        print(f"Seasons: S{df['SeasonIndex'].min()}-S{df['SeasonIndex'].max()}")
        print(f"Over odds range: {df['Corner_Over_Odds'].min():.2f} - {df['Corner_Over_Odds'].max():.2f}")
        print(f"Under odds range: {df['Corner_Under_Odds'].min():.2f} - {df['Corner_Under_Odds'].max():.2f}")
        print(f"\nPer-season counts:")
        print(df.groupby("SeasonIndex").size().to_string())
        print(f"\nSettlement: {df['Corner_Winner'].value_counts().to_string()}")
        print(f"\nSample:")
        print(df.head(5).to_string(index=False))
