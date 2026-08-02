"""Source converters for Premier League match facts.

FotMob and Understat exports become base fact rows in the canonical's
column layout. That is ALL this module does now: ADR 0007 decision 10
deleted its feature code and its direct write to CompleteDSPL_CSV.csv —
for eleven seasons two implementations wrote the same 39 feature columns
and nothing forced them to agree. Both leagues build through
``data/build_canonical_dataset.py``, whose publishes the schema gate in
``scripts/daily_ingest.py`` guards.

Usage (conversion only — no file is written):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a season's source data to base fact rows")
    parser.add_argument("--fotmob", type=str, help="Path to FotMob CSV")
    parser.add_argument("--understat", type=int,
                        help="Understat season year (e.g., 2024)")
    parser.add_argument("--season", type=int, required=True,
                        help="SeasonIndex to assign")
    args = parser.parse_args()

    if args.fotmob:
        path = os.path.join(PROJECT_DIR, args.fotmob)
        print(f"Converting FotMob data from {path}...")
        new_matches = fotmob_to_base(path, args.season)
    elif args.understat:
        print(f"Fetching from Understat for {args.understat}/{args.understat + 1}...")
        new_matches = understat_to_base(args.understat, args.season)
    else:
        print("Provide either --fotmob or --understat")
        sys.exit(1)

    print(f"Converted {len(new_matches)} matches for season {args.season}.")
    print("This module no longer writes the canonical (ADR 0007 decision 10).")
    print("Build and publish through data/build_canonical_dataset.py — the")
    print("schema gate in scripts/daily_ingest.py guards that path.")
