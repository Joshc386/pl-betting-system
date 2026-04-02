"""
Load and merge BTTS (Both Teams To Score) odds from footiqo CSVs into our pipeline data.

Data sources:
  - OtherSeasonsBTTS.csv: seasons 2015/16 — 2024/25 (S15-S24)
  - BTTS2526.csv: current season 2025/26 (S25)

Provides BTTSY/BTTSN odds + O/U odds at multiple goal lines (0.5-4.5).
"""
import os
import pandas as pd
from api.team_mapping import normalize

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORICAL_PATH = os.path.join(PROJECT_DIR, "OtherSeasonsBTTS.csv")
CURRENT_PATH = os.path.join(PROJECT_DIR, "BTTS2526.csv")


def load_btts_odds(include_current=True):
    """Load and normalize footiqo BTTS odds data.

    Returns DataFrame with columns:
        DateOnly, Home_Team, Away_Team, BTTSY, BTTSN,
        H, D, A, O05-O45, U05-U45
    """
    dfs = []

    if os.path.exists(HISTORICAL_PATH):
        dfs.append(pd.read_csv(HISTORICAL_PATH))
    if include_current and os.path.exists(CURRENT_PATH):
        dfs.append(pd.read_csv(CURRENT_PATH))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # Parse dates: DD-MM-YY HH:MM
    df["parsed"] = pd.to_datetime(df["matchDate"], format="%d-%m-%y %H:%M")
    df["DateOnly"] = df["parsed"].dt.strftime("%Y-%m-%d")

    # Normalize team names to match CompleteDSPL_CSV.csv
    df["Home_Team"] = df["homeTeam"].apply(normalize)
    df["Away_Team"] = df["awayTeam"].apply(normalize)

    # Season index from footiqo "YYYY/YYYY" format
    df["SeasonIndex"] = df["Season"].apply(lambda s: int(s.split("/")[0]) - 2000)

    # Keep only the columns we need
    keep = ["DateOnly", "Home_Team", "Away_Team", "SeasonIndex",
            "BTTSY", "BTTSN", "H", "D", "A",
            "O05", "U05", "O15", "U15", "O25", "U25",
            "O35", "U35", "O45", "U45"]
    return df[keep].copy()


def merge_btts_odds(pipeline_df, btts_odds_df=None):
    """Merge BTTS odds into pipeline DataFrame.

    Merges on date + Home_Team + Away_Team.
    Handles both 'DateOnly' (raw CSV) and 'Date' (pipeline datetime) columns.
    Returns pipeline_df with BTTSY/BTTSN columns added.
    """
    if btts_odds_df is None:
        btts_odds_df = load_btts_odds()

    if btts_odds_df.empty:
        print("  Warning: no BTTS odds data available")
        return pipeline_df

    df = pipeline_df.copy()

    # Create a string date column for merging
    if "DateOnly" in df.columns:
        df["_merge_date"] = df["DateOnly"].astype(str)
    elif "Date" in df.columns:
        df["_merge_date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    else:
        print("  Warning: no date column found for BTTS merge")
        return pipeline_df

    odds = btts_odds_df.copy()
    odds["_merge_date"] = odds["DateOnly"].astype(str)

    merge_cols = ["_merge_date", "Home_Team", "Away_Team"]
    odds_cols = ["BTTSY", "BTTSN"]

    merged = df.merge(
        odds[merge_cols + odds_cols],
        on=merge_cols,
        how="left",
    )
    merged.drop(columns=["_merge_date"], inplace=True)

    n_matched = merged["BTTSY"].notna().sum()
    print(f"  BTTS odds merged: {n_matched}/{len(merged)} matches have odds")

    return merged


if __name__ == "__main__":
    df = load_btts_odds()
    print(f"Loaded {len(df)} matches with BTTS odds")
    print(f"Seasons: S{df['SeasonIndex'].min()}-S{df['SeasonIndex'].max()}")
    print(f"BTTSY range: {df['BTTSY'].min():.2f} - {df['BTTSY'].max():.2f}")
    print(f"BTTSN range: {df['BTTSN'].min():.2f} - {df['BTTSN'].max():.2f}")
    print(f"\nSample:")
    print(df.head(3).to_string(index=False))
