"""
Add a new season's data to CompleteDSPL_CSV.csv.
Converts FotMob match data to the main CSV format and computes all rolling features.
Can also fetch from Understat if no FotMob CSV is available.

Usage:
    python -m data.add_season --fotmob "New Project Data/matches2425.csv" --season 24
    python -m data.add_season --understat 2024 --season 24
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from api.team_mapping import assert_known_teams, normalize


# Promoted teams per season (extend as needed)
PROMOTED_TEAMS = {
    24: ["Ipswich", "Leicester", "Southampton"],  # 2024/25
    25: ["Leeds", "Burnley", "Sunderland"],  # 2025/26
}


def load_main_csv():
    """Load existing CompleteDSPL_CSV.csv."""
    path = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df


def fotmob_to_base(fotmob_path: str, season_index: int) -> pd.DataFrame:
    """Convert a FotMob matches CSV to the base columns of CompleteDSPL_CSV.csv."""
    fm = pd.read_csv(fotmob_path)

    # These names go into the canonical, which is the training data. One that
    # does not resolve becomes a team of its own rather than the club it names.
    assert_known_teams(
        set(fm["home_team"]) | set(fm["away_team"]),
        f"season {season_index} from {os.path.basename(fotmob_path)}")

    rows = []
    for _, r in fm.iterrows():
        home = normalize(r["home_team"])
        away = normalize(r["away_team"])
        hg = int(r["home_goals"])
        ag = int(r["away_goals"])
        tg = hg + ag

        if hg > ag:
            ftr = "H"
        elif hg < ag:
            ftr = "A"
        else:
            ftr = "D"

        rows.append({
            "Date": pd.to_datetime(r["kickoff_time"], dayfirst=True),
            "Home_Team": home,
            "Away_Team": away,
            "Home_Goals": hg,
            "Away_Goals": ag,
            "TG": tg,
            "FTR": ftr,
            "HTHG": np.nan,  # Half-time not in FotMob
            "HTAG": np.nan,
            "HTR": np.nan,
            "Home_Shots": r.get("home_total_shots", np.nan),
            "Away_Shots": r.get("away_total_shots", np.nan),
            "Home_Shots_Target": r.get("home_shots_on_target", np.nan),
            "Away_Shots_Target": r.get("away_shots_on_target", np.nan),
            "HF": r.get("home_fouls_committed", np.nan),
            "AF": r.get("away_fouls_committed", np.nan),
            "Home_Corners": r.get("home_corners", np.nan),
            "Away_Corners": r.get("away_corners", np.nan),
            "HY": r.get("home_yellow_cards", np.nan),
            "AY": r.get("away_yellow_cards", np.nan),
            "HR": r.get("home_red_cards", np.nan),
            "AR": r.get("away_red_cards", np.nan),
            # No betting odds from FotMob
            "B365H": np.nan, "B365D": np.nan, "B365A": np.nan,
            "B365Greater2.5": np.nan, "B365LessThan2.5": np.nan,
            "SeasonIndex": season_index,
        })

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def understat_to_base(season_year: int, season_index: int) -> pd.DataFrame:
    """Fetch a season from Understat and convert to base columns."""
    from api.understat_scraper import scrape_season
    from understatapi import UnderstatClient

    with UnderstatClient() as client:
        xg_df = scrape_season(client, season_year)

    if xg_df.empty:
        print(f"  No Understat data for {season_year}")
        return pd.DataFrame()

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

    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all rolling features for the ENTIRE dataset.
    This replaces Functions.py for feature computation - vectorized and much faster.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # --- Build long-format team-centric data ---
    home = df[["Date", "Home_Team", "Home_Goals", "Away_Goals",
               "Home_Shots", "Away_Shots", "Home_Shots_Target", "Away_Shots_Target",
               "Home_Corners", "Away_Corners"]].copy()
    home.columns = ["Date", "Team", "GF", "GA", "Shots", "ShotsAgainst",
                    "SOT", "SOTAgainst", "Corners", "CornersAgainst"]

    away = df[["Date", "Away_Team", "Away_Goals", "Home_Goals",
               "Away_Shots", "Home_Shots", "Away_Shots_Target", "Home_Shots_Target",
               "Away_Corners", "Home_Corners"]].copy()
    away.columns = ["Date", "Team", "GF", "GA", "Shots", "ShotsAgainst",
                    "SOT", "SOTAgainst", "Corners", "CornersAgainst"]

    long = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)

    # Rolling 5-game stats (shift to exclude current match)
    g = long.groupby("Team")
    long["Past5Goals"] = g["GF"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["Past5Conceded"] = g["GA"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["Past5Corners"] = g["Corners"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["Past5CornersConceded"] = g["CornersAgainst"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["AvgShots_5"] = g["Shots"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    long["AvgShotsOnTarget_5"] = g["SOT"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Shot ratio = total shots / shots on target (original formula from Functions.py)
    long["RollingShots_5"] = g["Shots"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["RollingSOT_5_sum"] = g["SOT"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["ShotRatio_5"] = long["RollingShots_5"] / long["RollingSOT_5_sum"].replace(0, np.nan)

    # Shots per goal
    long["ShotsPerGoal_5"] = long["AvgShots_5"] / (long["Past5Goals"] / 5).replace(0, np.nan)

    # Conversion rates (5 and 20 game)
    long["CR_5"] = (long["Past5Goals"] / 5) / long["AvgShots_5"].replace(0, np.nan)
    long["TotalGoals_20"] = g["GF"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    long["TotalShots_20"] = g["Shots"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    long["CR_20"] = (long["TotalGoals_20"] / 20) / (long["TotalShots_20"] / 20).replace(0, np.nan)

    # SOT conversion rates
    long["TotalSOT_5"] = g["SOT"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["SOT_CR_5"] = (long["Past5Goals"] / 5) / (long["TotalSOT_5"] / 5).replace(0, np.nan)
    long["TotalSOT_20"] = g["SOT"].transform(lambda x: x.shift(1).rolling(20, min_periods=1).sum())
    long["SOT_CR_20"] = (long["TotalGoals_20"] / 20) / (long["TotalSOT_20"] / 20).replace(0, np.nan)

    # Defensive strength = 1 / rolling sum of shots conceded (higher = stronger defense)
    long["RollingShotsConceded_5"] = g["ShotsAgainst"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["DefensiveStrength_5"] = (1 / long["RollingShotsConceded_5"]).replace([np.inf, -np.inf], np.nan)
    long["RollingSOTConceded_5"] = g["SOTAgainst"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    long["DefensiveStrength_SOT"] = (1 / long["RollingSOTConceded_5"]).replace([np.inf, -np.inf], np.nan)

    # --- Merge back to home/away format ---
    feat_cols = ["Date", "Team", "Past5Goals", "Past5Conceded", "Past5Corners", "Past5CornersConceded",
                 "AvgShots_5", "AvgShotsOnTarget_5", "ShotRatio_5", "ShotsPerGoal_5",
                 "CR_5", "CR_20", "SOT_CR_5", "SOT_CR_20",
                 "DefensiveStrength_5", "DefensiveStrength_SOT"]

    feats = long[feat_cols].copy()

    # Home merge
    home_feats = feats.rename(columns={c: f"Home_{c}" for c in feat_cols if c not in ["Date", "Team"]})
    df = df.merge(home_feats, left_on=["Date", "Home_Team"], right_on=["Date", "Team"],
                  how="left", suffixes=("", "_new")).drop(columns=["Team"], errors="ignore")

    # Away merge
    away_feats = feats.rename(columns={c: f"Away_{c}" for c in feat_cols if c not in ["Date", "Team"]})
    df = df.merge(away_feats, left_on=["Date", "Away_Team"], right_on=["Date", "Team"],
                  how="left", suffixes=("", "_away_new")).drop(columns=["Team"], errors="ignore")

    return df


def compute_league_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute league position at time of each match."""
    df = df.copy()
    home_pos = []
    away_pos = []

    for season_idx in df["SeasonIndex"].unique():
        season_df = df[df["SeasonIndex"] == season_idx].sort_values("Date")
        points = {}
        gd = {}

        for idx, row in season_df.iterrows():
            home, away = row["Home_Team"], row["Away_Team"]
            for t in [home, away]:
                if t not in points:
                    points[t] = 0
                    gd[t] = 0

            # Get current positions before updating
            sorted_teams = sorted(points.keys(), key=lambda t: (-points[t], -gd[t]))
            pos_map = {t: i + 1 for i, t in enumerate(sorted_teams)}
            home_pos.append(pos_map.get(home, 10))
            away_pos.append(pos_map.get(away, 10))

            # Update points
            hg, ag = row["Home_Goals"], row["Away_Goals"]
            if hg > ag:
                points[home] += 3
            elif hg < ag:
                points[away] += 3
            else:
                points[home] += 1
                points[away] += 1
            gd[home] += hg - ag
            gd[away] += ag - hg

    df["Home_LeaguePosition"] = home_pos
    df["Away_LeaguePosition"] = away_pos
    return df


def compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head features."""
    df = df.copy()
    h2h_home_wins = []
    h2h_away_wins = []
    h2h_draws = []
    h2h_avg5 = []
    h2h_avg_all = []
    h2h_count = []

    for idx, row in df.iterrows():
        home, away = row["Home_Team"], row["Away_Team"]
        date = row["Date"]

        prev = df[
            (((df["Home_Team"] == home) & (df["Away_Team"] == away)) |
             ((df["Home_Team"] == away) & (df["Away_Team"] == home))) &
            (df["Date"] < date)
        ].sort_values("Date")

        n = len(prev)
        h2h_count.append(n)

        if n == 0:
            h2h_home_wins.append(0)
            h2h_away_wins.append(0)
            h2h_draws.append(0)
            h2h_avg5.append(np.nan)
            h2h_avg_all.append(np.nan)
            continue

        hw = sum((prev["Home_Team"] == home) & (prev["FTR"] == "H")) + \
             sum((prev["Away_Team"] == home) & (prev["FTR"] == "A"))
        aw = sum((prev["Home_Team"] == away) & (prev["FTR"] == "H")) + \
             sum((prev["Away_Team"] == away) & (prev["FTR"] == "A"))
        d = sum(prev["FTR"] == "D")

        h2h_home_wins.append(hw)
        h2h_away_wins.append(aw)
        h2h_draws.append(d)
        h2h_avg_all.append(prev["TG"].mean())
        h2h_avg5.append(prev.tail(5)["TG"].mean())

    df["H2H_HomeWins"] = h2h_home_wins
    df["H2H_AwayWins"] = h2h_away_wins
    df["H2H_Draws"] = h2h_draws
    df["H2H_AvgGoals_5"] = h2h_avg5
    df["H2HAvgGoals"] = h2h_avg_all
    df["H2H_MatchCount"] = h2h_count
    return df


def add_promoted(df: pd.DataFrame) -> pd.DataFrame:
    """Add promoted team flags."""
    df = df.copy()

    def is_promoted(team, si):
        promoted = PROMOTED_TEAMS.get(si, [])
        return int(any(p.lower() in team.lower() for p in promoted))

    df["Home_Promoted"] = df.apply(lambda r: is_promoted(r["Home_Team"], r["SeasonIndex"]), axis=1)
    df["Away_Promoted"] = df.apply(lambda r: is_promoted(r["Away_Team"], r["SeasonIndex"]), axis=1)
    return df


def add_derbies(df: pd.DataFrame) -> pd.DataFrame:
    """Add derby flags (simplified)."""
    local_derbies = [
        ("Arsenal FC", "Tottenham Hotspur FC"),
        ("Liverpool FC", "Everton FC"),
        ("Manchester United FC", "Manchester City FC"),
        ("Chelsea FC", "Fulham FC"),
        ("Crystal Palace FC", "Brighton & Hove Albion FC"),
        ("Newcastle United FC", "Sunderland AFC"),
        ("Aston Villa FC", "Birmingham City FC"),
        ("Wolverhampton Wanderers FC", "West Bromwich Albion FC"),
    ]
    historical_derbies = [
        ("Arsenal FC", "Manchester United FC"),
        ("Arsenal FC", "Chelsea FC"),
        ("Liverpool FC", "Manchester United FC"),
        ("Liverpool FC", "Chelsea FC"),
        ("Manchester United FC", "Chelsea FC"),
        ("Manchester City FC", "Liverpool FC"),
    ]

    df = df.copy()
    local_set = set((a, b) for a, b in local_derbies) | set((b, a) for a, b in local_derbies)
    hist_set = set((a, b) for a, b in historical_derbies) | set((b, a) for a, b in historical_derbies)

    df["Local Derby"] = df.apply(lambda r: int((r["Home_Team"], r["Away_Team"]) in local_set), axis=1)
    df["Historical Derby"] = df.apply(lambda r: int((r["Home_Team"], r["Away_Team"]) in hist_set), axis=1)
    df["Not a Derby"] = ((df["Local Derby"] == 0) & (df["Historical Derby"] == 0)).astype(int)
    return df


def add_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Add Home/Away Factor columns (placeholder - uses rolling goal avg as proxy)."""
    df = df.copy()
    # Factor is a composite metric from Functions.py. For new seasons, use rolling goal rate as proxy.
    home = df[["Date", "Home_Team", "Home_Goals"]].copy()
    home.columns = ["Date", "Team", "Goals"]
    away = df[["Date", "Away_Team", "Away_Goals"]].copy()
    away.columns = ["Date", "Team", "Goals"]
    long = pd.concat([home, away]).sort_values(["Team", "Date"])

    long["Factor"] = long.groupby("Team")["Goals"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # Merge back
    df = df.merge(long[["Date", "Team", "Factor"]].rename(columns={"Factor": "Home Factor"}),
                  left_on=["Date", "Home_Team"], right_on=["Date", "Team"],
                  how="left").drop(columns=["Team"], errors="ignore")
    df = df.merge(long[["Date", "Team", "Factor"]].rename(columns={"Factor": "Away Factor"}),
                  left_on=["Date", "Away_Team"], right_on=["Date", "Team"],
                  how="left").drop(columns=["Team"], errors="ignore")
    return df


def add_season_to_csv(new_matches: pd.DataFrame, season_index: int):
    """
    Add new season matches to the main CSV with all computed features.
    Recomputes rolling features for the new season using existing data as context.
    """
    print(f"\nLoading existing data...")
    existing = load_main_csv()

    # Remove any existing data for this season (in case of re-run)
    existing = existing[existing["SeasonIndex"] != season_index]
    print(f"  Existing: {len(existing)} rows (seasons 0-{existing['SeasonIndex'].max()})")

    # Combine
    combined = pd.concat([existing, new_matches], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    print(f"  Combined: {len(combined)} rows")

    # Compute all features on the combined dataset
    # Only recompute for the new season, but need full data for context
    print("  Computing rolling features...")
    combined = compute_rolling_features(combined)

    # Use _new columns to fill in the feature columns for the new season
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

    # Drop temporary _new columns
    drop_cols = [c for c in combined.columns if c.endswith("_new") or c.endswith("_away_new")]
    combined = combined.drop(columns=drop_cols, errors="ignore")

    # League positions
    print("  Computing league positions...")
    combined = compute_league_positions(combined)

    # H2H (only for new season - reuse existing for old)
    print("  Computing H2H features for new season...")
    new_season = combined[combined["SeasonIndex"] == season_index].copy()
    new_season = compute_h2h(new_season)
    for col in ["H2H_HomeWins", "H2H_AwayWins", "H2H_Draws", "H2H_AvgGoals_5", "H2HAvgGoals", "H2H_MatchCount"]:
        combined.loc[new_mask, col] = new_season[col].values

    # Promoted teams
    print("  Adding promoted flags...")
    combined = add_promoted(combined)

    # Derbies
    print("  Adding derby flags...")
    combined = add_derbies(combined)

    # Factor (for new season only)
    if "Home Factor" not in combined.columns or combined.loc[new_mask, "Home Factor"].isna().all():
        print("  Computing Factor for new season...")
        # Drop existing Factor cols to avoid _x/_y on merge
        combined = combined.drop(columns=["Home Factor", "Away Factor"], errors="ignore")
        combined = add_factor(combined)

    # Save
    output_path = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
    combined.to_csv(output_path, index=False)
    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(f"  Seasons: {combined['SeasonIndex'].min()}-{combined['SeasonIndex'].max()}")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a season to the main dataset")
    parser.add_argument("--fotmob", type=str, help="Path to FotMob CSV")
    parser.add_argument("--understat", type=int, help="Understat season year (e.g., 2024)")
    parser.add_argument("--season", type=int, required=True, help="SeasonIndex to assign")
    args = parser.parse_args()

    if args.fotmob:
        path = os.path.join(PROJECT_DIR, args.fotmob)
        print(f"Converting FotMob data from {path}...")
        new_matches = fotmob_to_base(path, args.season)
    elif args.understat:
        print(f"Fetching from Understat for {args.understat}/{args.understat+1}...")
        new_matches = understat_to_base(args.understat, args.season)
    else:
        print("Provide either --fotmob or --understat")
        sys.exit(1)

    if new_matches.empty:
        print("No matches to add.")
        sys.exit(1)

    print(f"Got {len(new_matches)} matches for season {args.season}")
    add_season_to_csv(new_matches, args.season)
