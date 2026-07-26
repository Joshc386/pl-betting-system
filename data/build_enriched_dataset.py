"""
Merge xG data (Understat) and player injury data (FPL) into the main dataset.
Produces CompleteDSPL_enriched.csv with new columns for xG and injury features.
"""
import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")


def load_main_dataset():
    """Load the original CompleteDSPL_CSV.csv."""
    path = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_xg_data():
    """Load scraped Understat xG data."""
    path = os.path.join(DATA_DIR, "understat_xg.csv")
    if not os.path.exists(path):
        print("  Warning: understat_xg.csv not found. Run understat_scraper first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_injury_data():
    """Load FPL injury burden data."""
    path = os.path.join(DATA_DIR, "fpl_historical", "team_injury_burden.csv")
    if not os.path.exists(path):
        print("  Warning: team_injury_burden.csv not found. Run fpl_historical first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def merge_xg(main_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge xG data into main dataset by matching on date + team names.
    Uses ±1 day tolerance for date matching.
    """
    if xg_df.empty:
        main_df["home_xg"] = np.nan
        main_df["away_xg"] = np.nan
        return main_df

    # Build a lookup: (date_str, home_team, away_team) -> (home_xg, away_xg)
    xg_lookup = {}
    for _, row in xg_df.iterrows():
        date = row["date"]
        key = (date.date(), row["home_team"], row["away_team"])
        xg_lookup[key] = (row["home_xg"], row["away_xg"])
        # Also store ±1 day for fuzzy matching
        for offset in [-1, 1]:
            alt_date = (date + pd.Timedelta(days=offset)).date()
            alt_key = (alt_date, row["home_team"], row["away_team"])
            if alt_key not in xg_lookup:
                xg_lookup[alt_key] = (row["home_xg"], row["away_xg"])

    home_xg = []
    away_xg = []
    matched = 0

    for _, row in main_df.iterrows():
        key = (row["Date"].date(), row["Home_Team"], row["Away_Team"])
        if key in xg_lookup:
            hxg, axg = xg_lookup[key]
            home_xg.append(hxg)
            away_xg.append(axg)
            matched += 1
        else:
            home_xg.append(np.nan)
            away_xg.append(np.nan)

    main_df["home_xg"] = home_xg
    main_df["away_xg"] = away_xg

    print(f"  xG merged: {matched}/{len(main_df)} matches ({matched/len(main_df):.1%})")
    return main_df


def merge_injuries(main_df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge injury burden data into main dataset.
    Maps FPL seasons and gameweeks to match dates.
    """
    if injury_df.empty:
        main_df["home_injury_burden"] = np.nan
        main_df["away_injury_burden"] = np.nan
        main_df["home_key_absences"] = np.nan
        main_df["away_key_absences"] = np.nan
        main_df["home_squad_depth"] = np.nan
        main_df["away_squad_depth"] = np.nan
        return main_df

    # Map SeasonIndex to FPL season strings
    # SeasonIndex 0 = 2000/01, so SeasonIndex 16 = 2016/17
    season_map = {
        16: "2016-17", 17: "2017-18", 18: "2018-19", 19: "2019-20",
        20: "2020-21", 21: "2021-22", 22: "2022-23", 23: "2023-24",
        24: "2024-25", 25: "2025-26",
    }

    # Build injury lookup: (fpl_season, gw, team) -> metrics
    injury_lookup = {}
    for _, row in injury_df.iterrows():
        key = (row["season"], int(row["GW"]), row["team"])
        injury_lookup[key] = {
            "injury_burden": row["injury_burden"],
            "key_absences": row["key_absences"],
            "squad_depth": row["squad_depth"],
        }

    # For each match, estimate the GW from the date within the season
    # Group by season to assign approximate gameweek numbers
    home_burden, away_burden = [], []
    home_absences, away_absences = [], []
    home_depth, away_depth = [], []
    matched = 0

    for season_idx, season_group in main_df.groupby("SeasonIndex"):
        fpl_season = season_map.get(season_idx)
        if fpl_season is None:
            for _ in range(len(season_group)):
                home_burden.append(np.nan)
                away_burden.append(np.nan)
                home_absences.append(np.nan)
                away_absences.append(np.nan)
                home_depth.append(np.nan)
                away_depth.append(np.nan)
            continue

        # Assign approximate GW (matches are roughly ordered by date within season)
        dates_sorted = season_group.sort_values("Date")
        unique_dates = sorted(dates_sorted["Date"].unique())
        # Each GW has ~10 matches over 2-3 days; group dates into GWs
        date_to_gw = {}
        gw = 1
        prev_date = None
        for d in unique_dates:
            if prev_date is not None and (d - prev_date).days > 4:
                gw += 1
            date_to_gw[d] = min(gw, 38)
            prev_date = d

        for idx, row in season_group.iterrows():
            gw = date_to_gw.get(row["Date"], 1)

            home_key = (fpl_season, gw, row["Home_Team"])
            away_key = (fpl_season, gw, row["Away_Team"])

            h_data = injury_lookup.get(home_key, {})
            a_data = injury_lookup.get(away_key, {})

            home_burden.append(h_data.get("injury_burden", np.nan))
            away_burden.append(a_data.get("injury_burden", np.nan))
            home_absences.append(h_data.get("key_absences", np.nan))
            away_absences.append(a_data.get("key_absences", np.nan))
            home_depth.append(h_data.get("squad_depth", np.nan))
            away_depth.append(a_data.get("squad_depth", np.nan))

            if h_data or a_data:
                matched += 1

    main_df["home_injury_burden"] = home_burden
    main_df["away_injury_burden"] = away_burden
    main_df["home_key_absences"] = home_absences
    main_df["away_key_absences"] = away_absences
    main_df["home_squad_depth"] = home_depth
    main_df["away_squad_depth"] = away_depth

    print(f"  Injuries merged: {matched}/{len(main_df)} matches ({matched/len(main_df):.1%})")
    return main_df


def build_enriched_dataset():
    """Main function: merge all data sources and save enriched CSV."""
    print("Loading main dataset...")
    main_df = load_main_dataset()
    print(f"  {len(main_df)} rows, seasons {main_df['SeasonIndex'].min()}-{main_df['SeasonIndex'].max()}")

    print("\nLoading xG data...")
    xg_df = load_xg_data()
    if not xg_df.empty:
        print(f"  {len(xg_df)} matches with xG")

    print("\nMerging xG data...")
    main_df = merge_xg(main_df, xg_df)

    print("\nLoading injury data...")
    injury_df = load_injury_data()
    if not injury_df.empty:
        print(f"  {len(injury_df)} team-gameweek rows")

    print("\nMerging injury data...")
    main_df = merge_injuries(main_df, injury_df)

    # Save enriched dataset
    output_path = os.path.join(PROJECT_DIR, "CompleteDSPL_enriched.csv")
    main_df.to_csv(output_path, index=False)
    print(f"\nSaved enriched dataset to {output_path}")
    print(f"  Shape: {main_df.shape}")
    print(f"  New columns: home_xg, away_xg, home_injury_burden, away_injury_burden, "
          f"home_key_absences, away_key_absences, home_squad_depth, away_squad_depth")

    # Summary of xG coverage
    xg_coverage = main_df["home_xg"].notna().sum()
    inj_coverage = main_df["home_injury_burden"].notna().sum()
    print(f"\n  xG data available for {xg_coverage}/{len(main_df)} matches "
          f"(seasons {main_df[main_df['home_xg'].notna()]['SeasonIndex'].min()}-"
          f"{main_df[main_df['home_xg'].notna()]['SeasonIndex'].max()})")
    print(f"  Injury data available for {inj_coverage}/{len(main_df)} matches")

    return main_df


if __name__ == "__main__":
    build_enriched_dataset()
