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


def add_defensive_components(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the three Defensive Strength components (ADR 0007 decisions
    4 and 5, tier 1 — Facts only).

    Deliberately three named features, never one number:

    - **ShotSuppression_5** — per-match shots conceded ÷ the opponent's own
      pre-match rolling-5 shot volume, averaged over the team's last 5.
      1.0 = the opponent got exactly its usual volume; below 1 = suppression.
      Opponent-adjusted by shot *generation* rather than Elo — process-based
      per the Wheatcroft principle, and self-normalising across leagues.
    - **ChanceQualityAllowed_5** — SOT conceded ÷ shots conceded.
    - **ConversionAllowed_5** — goals conceded ÷ SOT conceded.

    All follow the canonical builder's rolling convention: mean of per-match
    ratios over ``shift(1).rolling(5, min_periods=1)`` per team, zero
    denominators becoming NaN. Eras without shot data yield NaN — never a
    substitute formula under the same name.

    Args:
        df: Match DataFrame with Date, Home/Away_Team, Home/Away_Goals and
            Home/Away_Shots(_Target) columns. Returned unchanged if the shot
            columns are absent entirely.

    Returns:
        DataFrame with Home_/Away_ ShotSuppression_5, ChanceQualityAllowed_5
        and ConversionAllowed_5 columns added.
    """
    required = ("Home_Shots", "Away_Shots",
                "Home_Shots_Target", "Away_Shots_Target")
    if any(c not in df.columns for c in required):
        return df

    records = []
    for idx, row in df.iterrows():
        for side, opp in (("Home", "Away"), ("Away", "Home")):
            records.append({
                "match_idx": idx,
                "side": side,
                "team": row[f"{side}_Team"],
                "opponent": row[f"{opp}_Team"],
                "date": row["Date"],
                "shots": row[f"{side}_Shots"],
                "shots_against": row[f"{opp}_Shots"],
                "sot_against": row[f"{opp}_Shots_Target"],
                "goals_conceded": row[f"{opp}_Goals"],
            })
    long = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(
        drop=True)

    def _lagged_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(5, min_periods=1).mean()

    # Each team's expected shot volume BEFORE each match, then looked up
    # from the other side of the fixture as the opponent's expectation.
    long["volume"] = long.groupby("team")["shots"].transform(_lagged_mean)
    opp_volume = long.set_index(["team", "match_idx"])["volume"].reindex(
        pd.MultiIndex.from_arrays([long["opponent"], long["match_idx"]])
    ).to_numpy()

    long["supp"] = long["shots_against"] / pd.Series(
        opp_volume, index=long.index).replace(0, np.nan)
    long["cq"] = long["sot_against"] / long["shots_against"].replace(0, np.nan)
    long["conv"] = long["goals_conceded"] / long["sot_against"].replace(
        0, np.nan)

    grouped = long.groupby("team")
    component_names = {"supp": "ShotSuppression_5",
                       "cq": "ChanceQualityAllowed_5",
                       "conv": "ConversionAllowed_5"}
    for src, name in component_names.items():
        long[name] = grouped[src].transform(_lagged_mean)

    for side in ("Home", "Away"):
        one_side = long[long["side"] == side].set_index("match_idx")
        for name in component_names.values():
            df[f"{side}_{name}"] = one_side[name].reindex(df.index)

    return df
