"""
EFL Championship alternative goal O/U lines data loader.

Loads Betfair exchange odds for multiple goal lines (0.5, 1.5, 2.5, 3.5, 4.5)
from the shared betfair_goal_ou.csv and merges with the Championship pipeline
data for backtesting.

Mirrors alt_lines_data.py (PL version) but filters to EFL teams.
"""
import os
import re
import logging

import numpy as np
import pandas as pd

from api.team_mapping import normalize

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "betfair_goal_ou.csv")


def _parse_event_name(event_name: str) -> tuple[str | None, str | None]:
    """Parse 'Home Team v Away Team' from Betfair event name.

    Args:
        event_name: Betfair event name, e.g. 'Burnley v Leeds'.

    Returns:
        Tuple of (home_team, away_team) normalized, or (None, None).
    """
    parts = re.split(r"\s+v\s+", event_name, maxsplit=1)
    if len(parts) != 2:
        return None, None
    return normalize(parts[0].strip()), normalize(parts[1].strip())


def load_betfair_efl_goal_odds(path: str | None = None) -> pd.DataFrame:
    """Load and clean Betfair goal O/U odds for EFL Championship.

    Loads from the shared betfair_goal_ou.csv (all GB football) and
    filters to Championship teams only.

    Args:
        path: Path to betfair_goal_ou.csv. Uses default if None.

    Returns:
        DataFrame with columns: Home_Team, Away_Team, DateOnly, goal_line,
        Over_Odds, Under_Odds, Winner.
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
    price_cols = ["over_ltp", "under_ltp", "over_ltp_first", "under_ltp_first"]
    # _pre columns added by extract_goal_odds.py A5 fix; may be absent in
    # older CSVs so we handle gracefully.
    if "over_ltp_pre" in df.columns:
        price_cols += ["over_ltp_pre", "under_ltp_pre"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price preference: _pre (last pre-kickoff trade) > _first (earliest
    # trade) > _ltp (in-play contaminated, last resort only).
    # See roi_findings.md action A5 for the full analysis.
    if "over_ltp_pre" in df.columns:
        df["Over_Odds"] = (
            df["over_ltp_pre"]
            .fillna(df["over_ltp_first"])
            .fillna(df["over_ltp"])
        )
        df["Under_Odds"] = (
            df["under_ltp_pre"]
            .fillna(df["under_ltp_first"])
            .fillna(df["under_ltp"])
        )
    else:
        df["Over_Odds"] = df["over_ltp_first"].fillna(df["over_ltp"])
        df["Under_Odds"] = df["under_ltp_first"].fillna(df["under_ltp"])

    # Derive missing side from the other (Betfair exchange: near-zero margin)
    mask_no_over = df["Over_Odds"].isna() & df["Under_Odds"].notna()
    mask_no_under = df["Under_Odds"].isna() & df["Over_Odds"].notna()
    df.loc[mask_no_over, "Over_Odds"] = (
        df.loc[mask_no_over, "Under_Odds"]
        / (df.loc[mask_no_over, "Under_Odds"] - 1)
    )
    df.loc[mask_no_under, "Under_Odds"] = (
        df.loc[mask_no_under, "Over_Odds"]
        / (df.loc[mask_no_under, "Over_Odds"] - 1)
    )

    # Drop rows without odds
    df = df.dropna(subset=["Over_Odds", "Under_Odds"])

    # Filter unreasonable odds
    df = df[(df["Over_Odds"] > 1.01) & (df["Under_Odds"] > 1.01)]
    df = df[(df["Over_Odds"] < 100) & (df["Under_Odds"] < 100)]

    # Rename winner column
    df["Winner"] = df["winner"].str.lower()

    # Goal line as float
    df["goal_line"] = df["goal_line"].astype(float)

    result = df[[
        "Home_Team", "Away_Team", "DateOnly", "goal_line",
        "Over_Odds", "Under_Odds", "Winner",
    ]].copy()

    logger.info(
        f"Cleaned: {len(result)} records across "
        f"{result['goal_line'].nunique()} lines"
    )
    return result


def _build_betfair_to_pipeline_map(
    pipeline_df: pd.DataFrame,
) -> dict[str, str]:
    """Build a mapping from Betfair normalised names to pipeline short names.

    The championship pipeline uses short names from Football-Data.org CSVs
    (e.g. "Blackburn", "Cardiff") while Betfair event names normalize to
    long forms (e.g. "Blackburn Rovers FC", "Cardiff City FC").

    This builds the reverse mapping: normalize(short_name) → short_name
    so Betfair names can be converted to pipeline names for the merge.

    Args:
        pipeline_df: DataFrame with Home_Team/Away_Team in pipeline format.

    Returns:
        Dict mapping normalized names to pipeline short names.
    """
    pipe_teams = set(pipeline_df["Home_Team"].unique()) | set(
        pipeline_df["Away_Team"].unique()
    )
    return {normalize(t): t for t in pipe_teams}


def load_and_merge(
    pipeline_df: pd.DataFrame,
    path: str | None = None,
) -> pd.DataFrame:
    """Load Betfair goal odds and merge with Championship pipeline data.

    Handles the name format mismatch: Betfair uses long names
    ("Blackburn Rovers FC") while the pipeline uses short names
    ("Blackburn"). Builds a reverse mapping via ``normalize()`` to
    convert Betfair names to pipeline format before merging.

    Args:
        pipeline_df: DataFrame from championship_pipeline.run_pipeline()
            with DateOnly column.
        path: Path to betfair_goal_ou.csv.

    Returns:
        Pipeline DataFrame with BF_Over_{line}, BF_Under_{line},
        BF_Winner_{line} columns merged for each available goal line.
    """
    odds_df = load_betfair_efl_goal_odds(path)

    # Convert Betfair normalised names → pipeline short names
    name_map = _build_betfair_to_pipeline_map(pipeline_df)
    odds_df["Home_Team"] = odds_df["Home_Team"].map(name_map)
    odds_df["Away_Team"] = odds_df["Away_Team"].map(name_map)
    odds_df = odds_df.dropna(subset=["Home_Team", "Away_Team"])

    # Filter to EFL teams from the pipeline
    efl_teams = set(pipeline_df["Home_Team"].unique()) | set(
        pipeline_df["Away_Team"].unique()
    )
    odds_df = odds_df[
        odds_df["Home_Team"].isin(efl_teams)
        & odds_df["Away_Team"].isin(efl_teams)
    ].copy()

    logger.info(f"EFL-filtered: {len(odds_df)} records")

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
            "Home_Team", "Away_Team", "DateOnly",
            "Over_Odds", "Under_Odds", "Winner",
        ]].copy()
        line_str = f"{line:.1f}".replace(".", "")  # e.g. "25" for 2.5
        line_df = line_df.rename(columns={
            "Over_Odds": f"BF_Over_{line_str}",
            "Under_Odds": f"BF_Under_{line_str}",
            "Winner": f"BF_Winner_{line_str}",
        })
        # Drop duplicates on merge keys
        line_df = line_df.drop_duplicates(
            subset=["Home_Team", "Away_Team", "DateOnly"], keep="last",
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
    print("  EFL Championship Betfair Goal O/U Data Loader")
    print("=" * 60)

    try:
        df = load_betfair_efl_goal_odds()
    except FileNotFoundError as e:
        print(f"\n{e}")
        exit(1)

    print(f"\nTotal records (all GB): {len(df)}")
    print(f"Date range: {df['DateOnly'].min()} to {df['DateOnly'].max()}")

    print(f"\nRecords per goal line:")
    for line, count in df.groupby("goal_line").size().items():
        print(f"  O/U {line}: {count}")
