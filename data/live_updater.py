"""
Live updater for the current season.
Fetches latest match results from Understat and updates the dataset.

Usage:
    python -m data.live_updater                  # Update current season (auto-detect)
    python -m data.live_updater --season 2025    # Specify Understat season year
    python -m data.live_updater --rebuild        # Full rebuild of enriched dataset
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from api.team_mapping import normalize
from api.understat_scraper import scrape_season
from understatapi import UnderstatClient
from data.add_season import (
    compute_rolling_features, compute_league_positions,
    compute_h2h, add_promoted, add_derbies, add_factor,
    PROMOTED_TEAMS,
)


def get_current_season_year():
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


def load_main_csv():
    """Load existing CompleteDSPL_CSV.csv."""
    path = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df


def fetch_latest_matches(season_year: int):
    """Fetch all completed matches from Understat for the given season."""
    print(f"Fetching matches from Understat for {season_year}/{season_year+1}...")
    with UnderstatClient() as client:
        xg_df = scrape_season(client, season_year)

    if xg_df.empty:
        print("  No matches found.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"  Found {len(xg_df)} completed matches")

    # Convert to main CSV base format
    season_index = season_year_to_index(season_year)
    rows = []
    for _, r in xg_df.iterrows():
        hg = int(r["home_goals"])
        ag = int(r["away_goals"])
        tg = hg + ag
        ftr = "H" if hg > ag else ("A" if hg < ag else "D")

        rows.append({
            "Date": r["date"],
            "Home_Team": r["home_team"],
            "Away_Team": r["away_team"],
            "Home_Goals": hg,
            "Away_Goals": ag,
            "TG": tg,
            "FTR": ftr,
            "HTHG": np.nan, "HTAG": np.nan, "HTR": np.nan,
            "Home_Shots": np.nan, "Away_Shots": np.nan,
            "Home_Shots_Target": np.nan, "Away_Shots_Target": np.nan,
            "HF": np.nan, "AF": np.nan,
            "Home_Corners": np.nan, "Away_Corners": np.nan,
            "HY": np.nan, "AY": np.nan,
            "HR": np.nan, "AR": np.nan,
            "B365H": np.nan, "B365D": np.nan, "B365A": np.nan,
            "B365Greater2.5": np.nan, "B365LessThan2.5": np.nan,
            "SeasonIndex": season_index,
        })

    base_df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    return base_df, xg_df


def update_dataset(season_year: int, rebuild_enriched: bool = False):
    """
    Fetch latest matches and update the main CSV.
    Only adds new matches that aren't already in the dataset.
    """
    season_index = season_year_to_index(season_year)
    base_df, xg_df = fetch_latest_matches(season_year)

    if base_df.empty:
        print("No new matches to add.")
        return

    # Load existing data
    existing = load_main_csv()
    existing_season = existing[existing["SeasonIndex"] == season_index]

    # Find new matches not already in dataset
    if not existing_season.empty:
        # Match on date + teams to find already-added matches
        existing_keys = set(
            zip(existing_season["Date"].dt.date,
                existing_season["Home_Team"],
                existing_season["Away_Team"])
        )
        new_mask = ~base_df.apply(
            lambda r: (r["Date"].date(), r["Home_Team"], r["Away_Team"]) in existing_keys,
            axis=1
        )
        new_matches = base_df[new_mask].copy()
        print(f"  {len(existing_season)} matches already in dataset, {len(new_matches)} new")
    else:
        new_matches = base_df
        print(f"  No existing data for season {season_index}, adding all {len(new_matches)} matches")

    if new_matches.empty:
        print("Dataset is already up to date.")
        return

    # Remove existing season data and replace with full updated version
    existing_no_season = existing[existing["SeasonIndex"] != season_index]
    combined = pd.concat([existing_no_season, base_df], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"  Combined: {len(combined)} rows")

    # Recompute features for the updated season
    print("  Computing rolling features...")
    combined = compute_rolling_features(combined)

    # Apply _new columns for the season
    new_mask = combined["SeasonIndex"] == season_index
    rolling_cols = ["Past5Goals", "Past5Conceded", "Past5Corners", "Past5CornersConceded",
                    "AvgShots_5", "AvgShotsOnTarget_5", "ShotRatio_5", "ShotsPerGoal_5",
                    "CR_5", "CR_20", "SOT_CR_5", "SOT_CR_20",
                    "DefensiveStrength_5", "DefensiveStrength_SOT"]

    for prefix in ["Home", "Away"]:
        for col in rolling_cols:
            new_col = f"{prefix}_{col}_new" if f"{prefix}_{col}_new" in combined.columns else f"{prefix}_{col}"
            target = f"{prefix}_{col}"
            if new_col in combined.columns and new_col != target:
                combined.loc[new_mask, target] = combined.loc[new_mask, new_col]

    drop_cols = [c for c in combined.columns if c.endswith("_new") or c.endswith("_away_new")]
    combined = combined.drop(columns=drop_cols, errors="ignore")

    print("  Computing league positions...")
    combined = compute_league_positions(combined)

    print("  Computing H2H for new season...")
    new_season = combined[combined["SeasonIndex"] == season_index].copy()
    new_season = compute_h2h(new_season)
    for col in ["H2H_HomeWins", "H2H_AwayWins", "H2H_Draws", "H2H_AvgGoals_5", "H2HAvgGoals", "H2H_MatchCount"]:
        if col in new_season.columns:
            combined.loc[new_mask, col] = new_season[col].values

    print("  Adding promoted flags...")
    combined = add_promoted(combined)

    print("  Adding derby flags...")
    combined = add_derbies(combined)

    # Factor for new season
    if "Home Factor" not in combined.columns or combined.loc[new_mask, "Home Factor"].isna().all():
        print("  Computing Factor...")
        combined = combined.drop(columns=["Home Factor", "Away Factor"], errors="ignore")
        combined = add_factor(combined)

    # Save
    output_path = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(f"  Season {season_index}: {len(combined[combined['SeasonIndex'] == season_index])} matches")

    # Update xG data file
    if not xg_df.empty:
        xg_path = os.path.join(PROJECT_DIR, "data", "understat_xg.csv")
        if os.path.exists(xg_path):
            existing_xg = pd.read_csv(xg_path)
            existing_xg["date"] = pd.to_datetime(existing_xg["date"])
            # Remove old season data and replace
            season_str = f"{season_year}/{str(season_year+1)[-2:]}"
            existing_xg = existing_xg[existing_xg["season"] != season_str]
            updated_xg = pd.concat([existing_xg, xg_df], ignore_index=True).sort_values("date")
            updated_xg.to_csv(xg_path, index=False)
            print(f"  Updated xG data: {len(updated_xg)} total matches")

    # Optionally rebuild enriched dataset
    if rebuild_enriched:
        print("\nRebuilding enriched dataset...")
        from data.build_enriched_dataset import build_enriched_dataset
        build_enriched_dataset()

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update dataset with latest matches")
    parser.add_argument("--season", type=int, default=None,
                        help="Understat season year (e.g. 2025). Auto-detected if omitted.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Also rebuild the enriched dataset")
    args = parser.parse_args()

    season_year = args.season or get_current_season_year()
    print(f"Live updater for {season_year}/{season_year+1} season")
    print(f"  SeasonIndex: {season_year_to_index(season_year)}")
    update_dataset(season_year, rebuild_enriched=args.rebuild)
