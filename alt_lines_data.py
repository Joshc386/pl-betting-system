"""
Alternative goal O/U lines data loader.

Loads Betfair exchange odds for multiple goal lines (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
and merges with the main pipeline data for backtesting.

Data source: betfair_goal_ou.csv (extracted by extract_goal_odds.py)
"""
import os
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from api.team_mapping import normalize

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "betfair_goal_ou.csv")


def _parse_event_name(event_name: str) -> tuple[str | None, str | None]:
    """Parse 'Home Team v Away Team' from Betfair event name.

    Args:
        event_name: Betfair event name, e.g. 'Arsenal v Chelsea'.

    Returns:
        Tuple of (home_team, away_team) normalized, or (None, None).
    """
    parts = re.split(r"\s+v\s+", event_name, maxsplit=1)
    if len(parts) != 2:
        return None, None
    return normalize(parts[0].strip()), normalize(parts[1].strip())


def load_betfair_goal_odds(path: str | None = None) -> pd.DataFrame:
    """Load and clean Betfair goal O/U odds from extracted CSV.

    Args:
        path: Path to betfair_goal_ou.csv. Uses default if None.

    Returns:
        DataFrame with columns: Home_Team, Away_Team, DateOnly, goal_line,
        Over_Odds, Under_Odds, Winner, market_type.
    """
    if path is None:
        path = DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Betfair goal odds not found at {path}. "
            "Run extract_goal_odds.py first."
        )

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} raw Betfair goal O/U records")

    # Parse teams from event name
    parsed = df["event_name"].apply(_parse_event_name)
    df["Home_Team"] = parsed.apply(lambda x: x[0])
    df["Away_Team"] = parsed.apply(lambda x: x[1])
    df = df.dropna(subset=["Home_Team", "Away_Team"])

    # Parse date
    df["DateOnly"] = pd.to_datetime(
        df["market_time"], format="mixed", utc=True
    ).dt.date

    # Clean odds columns
    for col in ["over_ltp", "under_ltp", "over_ltp_first", "under_ltp_first"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Use closing odds (ltp) as primary, fall back to first traded price
    df["Over_Odds"] = df["over_ltp"].fillna(df["over_ltp_first"])
    df["Under_Odds"] = df["under_ltp"].fillna(df["under_ltp_first"])

    # Derive missing side from the other (Betfair exchange: near-zero margin)
    mask_no_over = df["Over_Odds"].isna() & df["Under_Odds"].notna()
    mask_no_under = df["Under_Odds"].isna() & df["Over_Odds"].notna()
    df.loc[mask_no_over, "Over_Odds"] = (
        df.loc[mask_no_over, "Under_Odds"] / (df.loc[mask_no_over, "Under_Odds"] - 1)
    )
    df.loc[mask_no_under, "Under_Odds"] = (
        df.loc[mask_no_under, "Over_Odds"] / (df.loc[mask_no_under, "Over_Odds"] - 1)
    )

    # Drop rows without odds
    df = df.dropna(subset=["Over_Odds", "Under_Odds"])

    # Filter unreasonable odds
    df = df[(df["Over_Odds"] > 1.01) & (df["Under_Odds"] > 1.01)]
    df = df[(df["Over_Odds"] < 100) & (df["Under_Odds"] < 100)]

    # Rename winner column
    df["Winner"] = df["winner"].str.lower()

    # Select and rename
    df["goal_line"] = df["goal_line"].astype(float)

    result = df[[
        "Home_Team", "Away_Team", "DateOnly", "goal_line", "market_type",
        "Over_Odds", "Under_Odds", "Winner",
        "over_ltp_first", "under_ltp_first",
    ]].copy()

    logger.info(f"Cleaned: {len(result)} records across "
                f"{result['goal_line'].nunique()} lines")
    return result


def _get_epl_teams() -> set[str]:
    """Get set of EPL team names from the main pipeline data."""
    from pipeline import load_data
    df = load_data()
    return set(df["Home_Team"].unique()) | set(df["Away_Team"].unique())


def load_and_merge(pipeline_df: pd.DataFrame,
                   path: str | None = None) -> pd.DataFrame:
    """Load Betfair goal odds and merge with pipeline data.

    Adds Over_Odds_{line} and Under_Odds_{line} columns for each goal line
    available in the Betfair data.

    Args:
        pipeline_df: DataFrame from run_pipeline() with DateOnly column.
        path: Path to betfair_goal_ou.csv.

    Returns:
        Pipeline DataFrame with Betfair odds columns merged.
    """
    odds_df = load_betfair_goal_odds(path)

    # Filter to EPL teams only
    epl_teams = set(pipeline_df["Home_Team"].unique()) | set(
        pipeline_df["Away_Team"].unique()
    )
    odds_df = odds_df[
        odds_df["Home_Team"].isin(epl_teams) &
        odds_df["Away_Team"].isin(epl_teams)
    ].copy()

    logger.info(f"EPL-filtered: {len(odds_df)} records")

    # Ensure DateOnly is comparable
    if "DateOnly" not in pipeline_df.columns:
        pipeline_df = pipeline_df.copy()
        pipeline_df["DateOnly"] = pd.to_datetime(pipeline_df["Date"]).dt.date
    odds_df["DateOnly"] = pd.to_datetime(odds_df["DateOnly"]).dt.date

    # Pivot: one row per match with columns for each line
    merged = pipeline_df.copy()
    lines = sorted(odds_df["goal_line"].unique())

    for line in lines:
        line_df = odds_df[odds_df["goal_line"] == line][[
            "Home_Team", "Away_Team", "DateOnly", "Over_Odds", "Under_Odds",
            "Winner",
        ]].copy()
        line_str = f"{line:.1f}".replace(".", "")  # e.g. "25" for 2.5
        line_df = line_df.rename(columns={
            "Over_Odds": f"BF_Over_{line_str}",
            "Under_Odds": f"BF_Under_{line_str}",
            "Winner": f"BF_Winner_{line_str}",
        })
        # Drop duplicates on merge keys
        line_df = line_df.drop_duplicates(
            subset=["Home_Team", "Away_Team", "DateOnly"], keep="last"
        )

        merged = merged.merge(
            line_df,
            on=["Home_Team", "Away_Team", "DateOnly"],
            how="left",
        )

    # Count available lines per match
    bf_cols = [c for c in merged.columns if c.startswith("BF_Over_")]
    merged["n_betfair_lines"] = merged[bf_cols].notna().sum(axis=1)

    n_with_odds = (merged["n_betfair_lines"] > 0).sum()
    logger.info(f"Merged: {n_with_odds}/{len(merged)} matches have Betfair odds")

    return merged


if __name__ == "__main__":
    print("=" * 60)
    print("  Betfair Goal O/U Data Loader")
    print("=" * 60)

    try:
        df = load_betfair_goal_odds()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Run: python extract_goal_odds.py")
        exit(1)

    print(f"\nTotal records: {len(df)}")
    print(f"Date range: {df['DateOnly'].min()} to {df['DateOnly'].max()}")

    print(f"\nRecords per goal line:")
    for line, count in df.groupby("goal_line").size().items():
        print(f"  O/U {line}: {count}")

    # EPL filtering
    epl_teams = _get_epl_teams()
    epl_df = df[
        df["Home_Team"].isin(epl_teams) & df["Away_Team"].isin(epl_teams)
    ]
    print(f"\nEPL records: {len(epl_df)}")
    print(f"EPL records per goal line:")
    for line, count in epl_df.groupby("goal_line").size().items():
        print(f"  O/U {line}: {count}")

    # Check settlement coverage
    settled = epl_df[epl_df["Winner"].isin(["over", "under"])]
    print(f"\nSettled matches: {len(settled)}/{len(epl_df)}")
