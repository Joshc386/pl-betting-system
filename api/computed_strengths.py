"""
Computed team strength ratings — FPL replacement for non-PL leagues.

Derives pre-season team strength from the previous season's match data:
attack/defence ratings (home/away splits), normalised to [0, 1].

Used as cold-start features for early-season matches where rolling stats
have insufficient history.

Output schema matches FPL strength features:
    Home_FPL_Attack, Away_FPL_Attack, Home_FPL_Defence, Away_FPL_Defence,
    FPL_Openness, FPL_HomeDominance
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_team_strengths(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team strength ratings from previous season's results.

    For each season S, team strengths are derived from season S-1 results.
    Promoted teams (no data in S-1) get the league-average strength from
    the bottom 3 teams as a proxy.

    Args:
        df: Full dataset with Date, Home_Team, Away_Team, Home_Goals,
            Away_Goals, SeasonIndex columns.

    Returns:
        DataFrame with columns: season_index, team,
        attack_home, attack_away, defence_home, defence_away.
        All values normalised to [0, 1] within each season.
    """
    records: list[dict] = []

    seasons = sorted(df["SeasonIndex"].unique())

    for season_idx in seasons:
        # Use prior season data
        prior = df[df["SeasonIndex"] == season_idx - 1]
        if len(prior) < 100:
            # No prior season — skip (first season in dataset)
            continue

        # Compute per-team stats from prior season
        team_stats: dict[str, dict] = {}

        for _, row in prior.iterrows():
            home = row["Home_Team"]
            away = row["Away_Team"]
            hg = row["Home_Goals"]
            ag = row["Away_Goals"]
            if pd.isna(hg) or pd.isna(ag):
                continue
            hg, ag = float(hg), float(ag)

            for team in [home, away]:
                if team not in team_stats:
                    team_stats[team] = {
                        "home_scored": [], "home_conceded": [],
                        "away_scored": [], "away_conceded": [],
                    }

            team_stats[home]["home_scored"].append(hg)
            team_stats[home]["home_conceded"].append(ag)
            team_stats[away]["away_scored"].append(ag)
            team_stats[away]["away_conceded"].append(hg)

        # Compute raw ratings
        raw: dict[str, dict[str, float]] = {}
        for team, stats in team_stats.items():
            if len(stats["home_scored"]) < 5:
                continue
            raw[team] = {
                "attack_home": np.mean(stats["home_scored"]),
                "attack_away": np.mean(stats["away_scored"]),
                "defence_home": np.mean(stats["home_conceded"]),
                "defence_away": np.mean(stats["away_conceded"]),
            }

        if not raw:
            continue

        # Normalise to [0, 1] within the season
        raw_df = pd.DataFrame(raw).T
        for col in raw_df.columns:
            mn, mx = raw_df[col].min(), raw_df[col].max()
            if mx > mn:
                raw_df[col] = (raw_df[col] - mn) / (mx - mn)
            else:
                raw_df[col] = 0.5

        # Defence is inverted: fewer goals conceded = higher strength
        raw_df["defence_home"] = 1.0 - raw_df["defence_home"]
        raw_df["defence_away"] = 1.0 - raw_df["defence_away"]

        # Promoted-team default: bottom-3 average from prior season
        bottom_3 = raw_df.nsmallest(3, "attack_home").mean()
        promoted_default = {
            "attack_home": bottom_3["attack_home"],
            "attack_away": bottom_3["attack_away"],
            "defence_home": bottom_3["defence_home"],
            "defence_away": bottom_3["defence_away"],
        }

        # Get teams appearing in this season
        season_df = df[df["SeasonIndex"] == season_idx]
        season_teams = set(season_df["Home_Team"].unique()) | set(season_df["Away_Team"].unique())

        for team in season_teams:
            if team in raw_df.index:
                r = raw_df.loc[team]
                records.append({
                    "season_index": season_idx,
                    "team": team,
                    "attack_home": r["attack_home"],
                    "attack_away": r["attack_away"],
                    "defence_home": r["defence_home"],
                    "defence_away": r["defence_away"],
                })
            else:
                # Promoted team — use bottom-3 average
                records.append({
                    "season_index": season_idx,
                    "team": team,
                    **promoted_default,
                })

    return pd.DataFrame(records)


def merge_strengths(df: pd.DataFrame, strengths: pd.DataFrame) -> pd.DataFrame:
    """Merge computed strength ratings into the match DataFrame.

    Produces the same feature columns as FPL strength features:
    Home_FPL_Attack, Away_FPL_Attack, Home_FPL_Defence, Away_FPL_Defence,
    FPL_Openness, FPL_HomeDominance.

    Args:
        df: Match DataFrame with Home_Team, Away_Team, SeasonIndex.
        strengths: Output of compute_team_strengths().

    Returns:
        df with strength columns added.
    """
    # Build lookup: (season_index, team) -> strengths
    lookup: dict[tuple[int, str], dict] = {}
    for _, row in strengths.iterrows():
        key = (int(row["season_index"]), row["team"])
        lookup[key] = {
            "attack_home": row["attack_home"],
            "attack_away": row["attack_away"],
            "defence_home": row["defence_home"],
            "defence_away": row["defence_away"],
        }

    h_attack = []
    a_attack = []
    h_defence = []
    a_defence = []

    for _, row in df.iterrows():
        si = int(row["SeasonIndex"])
        home_s = lookup.get((si, row["Home_Team"]))
        away_s = lookup.get((si, row["Away_Team"]))

        if home_s:
            h_attack.append(home_s["attack_home"])
            h_defence.append(home_s["defence_home"])
        else:
            h_attack.append(np.nan)
            h_defence.append(np.nan)

        if away_s:
            a_attack.append(away_s["attack_away"])
            a_defence.append(away_s["defence_away"])
        else:
            a_attack.append(np.nan)
            a_defence.append(np.nan)

    df["Home_FPL_Attack"] = h_attack
    df["Away_FPL_Attack"] = a_attack
    df["Home_FPL_Defence"] = h_defence
    df["Away_FPL_Defence"] = a_defence

    # Derived features (same as PL FPL features)
    df["FPL_Openness"] = (
        (df["Home_FPL_Attack"] + df["Away_FPL_Attack"]) -
        (df["Home_FPL_Defence"] + df["Away_FPL_Defence"])
    )
    df["FPL_HomeDominance"] = (
        (df["Home_FPL_Attack"] + df["Home_FPL_Defence"]) -
        (df["Away_FPL_Attack"] + df["Away_FPL_Defence"])
    )

    return df
