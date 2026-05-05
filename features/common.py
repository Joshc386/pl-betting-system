"""
Shared feature engineering functions used by both PL and EFL pipelines.

These functions operate on DataFrames with the standard home/away match
format (columns: Date, Home_Team, Away_Team, plus raw stats). They are
league-agnostic — the same algorithm applies to Premier League and
Championship data identically.

Both ``pipeline.py`` and ``championship_pipeline.py`` import from here
instead of maintaining duplicate copies.
"""
import numpy as np
import pandas as pd


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fixture congestion features beyond simple rest days.

    Tracks each team's recent match history and derives two features per
    side (home/away):

    - **MatchesLast14Days**: number of matches in the last 14 days
      (high = congested schedule, potential fatigue).
    - **AvgRest3**: average rest days across the last 3 matches
      (low = congested; captures sustained load, not just one short gap).

    Args:
        df: Match DataFrame with Date, Home_Team, Away_Team columns.
            Must be sorted by Date (ascending) for correct lookback.

    Returns:
        DataFrame with Home_MatchesLast14Days, Away_MatchesLast14Days,
        Home_AvgRest3, Away_AvgRest3 columns added.
    """
    home_m14: list[int] = []
    away_m14: list[int] = []
    home_avgrest3: list[float] = []
    away_avgrest3: list[float] = []

    team_dates: dict[str, list] = {}

    for _, row in df.iterrows():
        home = row["Home_Team"]
        away = row["Away_Team"]
        date = row["Date"]

        for team, m14_list, avgrest_list in [
            (home, home_m14, home_avgrest3),
            (away, away_m14, away_avgrest3),
        ]:
            if team not in team_dates:
                team_dates[team] = []

            past = team_dates[team]

            # Matches in last 14 days
            recent = [d for d in past if (date - d).days <= 14]
            m14_list.append(len(recent))

            # Average rest across last 3 matches
            if len(past) >= 3:
                last3 = past[-3:]
                gaps = [(date - last3[-1]).days]
                for i in range(len(last3) - 1, 0, -1):
                    gaps.append((last3[i] - last3[i - 1]).days)
                avgrest_list.append(float(np.mean(gaps)))
            elif len(past) >= 1:
                avgrest_list.append(float((date - past[-1]).days))
            else:
                avgrest_list.append(np.nan)

            team_dates[team].append(date)

    df["Home_MatchesLast14Days"] = home_m14
    df["Away_MatchesLast14Days"] = away_m14
    df["Home_AvgRest3"] = home_avgrest3
    df["Away_AvgRest3"] = away_avgrest3
    return df


def add_discipline_features(
    df: pd.DataFrame, window: int = 5,
) -> pd.DataFrame:
    """Compute rolling discipline/card features from HY, AY, HR, AR, HF, AF.

    Cards and fouls capture match tempo and aggression — high-foul matches
    produce more set pieces which correlate with goals.

    Features (per side):
    - **YellowCards_5**: rolling mean of yellow cards over *window* matches.
    - **RedCards_10**: rolling mean of red cards over 10 matches (rare events
      need a wider window).
    - **Fouls_5**: rolling mean of fouls committed over *window* matches.

    All features are lagged by one match (shift(1)) to avoid data leakage.

    Args:
        df: Match DataFrame with HY, AY, HR, AR, HF, AF columns.
            Returns unchanged if HY is missing.
        window: Rolling window size for yellows and fouls (default 5).

    Returns:
        DataFrame with Home_YellowCards_5, Away_YellowCards_5,
        Home_RedCards_10, Away_RedCards_10, Home_Fouls_5, Away_Fouls_5.
    """
    if "HY" not in df.columns:
        return df

    # Build long format: one row per team per match
    home_long = df[["Date", "Home_Team", "HY", "HR", "HF"]].copy()
    home_long.columns = ["Date", "Team", "YellowCards", "RedCards", "Fouls"]
    away_long = df[["Date", "Away_Team", "AY", "AR", "AF"]].copy()
    away_long.columns = ["Date", "Team", "YellowCards", "RedCards", "Fouls"]
    long = pd.concat([home_long, away_long]).sort_values(
        ["Team", "Date"]
    ).reset_index(drop=True)

    g = long.groupby("Team")
    long["YellowCards_5"] = g["YellowCards"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )
    long["RedCards_10"] = g["RedCards"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    long["Fouls_5"] = g["Fouls"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=2).mean()
    )

    new_cols = ["YellowCards_5", "RedCards_10", "Fouls_5"]
    feat_long = long[["Date", "Team"] + new_cols].drop_duplicates(
        ["Date", "Team"]
    )

    for prefix in ["Home", "Away"]:
        renamed = feat_long.rename(
            columns={c: f"{prefix}_{c}" for c in new_cols}
        )
        df = df.merge(
            renamed,
            left_on=["Date", f"{prefix}_Team"],
            right_on=["Date", "Team"],
            how="left",
        ).drop(columns=["Team"], errors="ignore")

    return df
