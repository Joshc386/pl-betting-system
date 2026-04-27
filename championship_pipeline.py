"""
Championship feature engineering pipeline.

Mirrors pipeline.py but adapted for Championship data availability:
- No xG features (Understat/FBref don't cover Championship)
- No FPL features (replaced with computed_strengths from prior season)
- No detailed match features (no matches2425.csv equivalent)
- All other features computed from goals, shots, corners, cards, positions

Produces a DataFrame with ~80 features ready for the 4-model ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson as poisson_dist

import os

from league_config import get_league_config
from api.computed_strengths import compute_team_strengths, merge_strengths

LEAGUE_CFG = get_league_config("EFL")

# Mapping from PL CSV team names (stripped of FC/AFC) to Championship CSV names.
# Used to detect PL-relegated teams entering the Championship.
_PL_TO_CHAMP_NAME: dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Birmingham City": "Birmingham",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Bradford City": "Bradford",
    "Brighton & Hove Albion": "Brighton",
    "Cardiff City": "Cardiff",
    "Charlton Athletic": "Charlton",
    "Coventry City": "Coventry",
    "Derby County": "Derby",
    "Huddersfield Town": "Huddersfield",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Norwich City": "Norwich",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "Sheffield United": "Sheffield United",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Sunderland": "Sunderland",
    "West Bromwich Albion": "West Brom",
    "West Ham United": "West Ham",
    "Wigan Athletic": "Wigan",
    "Wolverhampton Wanderers": "Wolves",
}


def load_championship_data() -> pd.DataFrame:
    """Load CompleteDSChamp_CSV.csv and prepare base columns."""
    df = pd.read_csv(LEAGUE_CFG["csv_path"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    # Target variables
    df["Over_2_5"] = (df["TG"] > 2.5).astype(int)
    df["Over_1_5"] = (df["TG"] > 1.5).astype(int)
    df["BTTS"] = ((df["Home_Goals"] > 0) & (df["Away_Goals"] > 0)).astype(int)

    # Half-time total goals (for HT features)
    df["HT_TG"] = df["HTHG"].fillna(0) + df["HTAG"].fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Derived features (from CSV columns)
# ═══════════════════════════════════════════════════════════════════════════════

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic derived features from existing CSV columns."""
    df["LeaguePosition_Diff"] = df["Home_LeaguePosition"] - df["Away_LeaguePosition"]
    df["Home_GoalDiff_5"] = df["Home_Past5Goals"] - df["Home_Past5Conceded"]
    df["Away_GoalDiff_5"] = df["Away_Past5Goals"] - df["Away_Past5Conceded"]

    # Rest days (compute from date gaps per team)
    df["Home_RestDays"] = np.nan
    df["Away_RestDays"] = np.nan
    last_match: dict[str, pd.Timestamp] = {}
    for idx, row in df.iterrows():
        home, away = row["Home_Team"], row["Away_Team"]
        dt = row["Date"]
        if home in last_match:
            df.at[idx, "Home_RestDays"] = (dt - last_match[home]).days
        if away in last_match:
            df.at[idx, "Away_RestDays"] = (dt - last_match[away]).days
        last_match[home] = dt
        last_match[away] = dt

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Congestion features
# ═══════════════════════════════════════════════════════════════════════════════

def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fixture congestion features beyond simple rest days.

    Championship has heavy midweek scheduling (46 games + playoffs + cup runs),
    making congestion a stronger signal than in the PL.

    Features:
        - MatchesLast14Days: games in the last 14 days (high = congested)
        - AvgRest3: average rest across last 3 matches (low = congested)
    """
    home_m14: list[int] = []
    away_m14: list[int] = []
    home_avgrest3: list[float] = []
    away_avgrest3: list[float] = []

    team_dates: dict[str, list[pd.Timestamp]] = {}

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


# ═══════════════════════════════════════════════════════════════════════════════
# Discipline features
# ═══════════════════════════════════════════════════════════════════════════════

def add_discipline_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling discipline/card features from HY, AY, HR, AR, HF, AF.

    Cards and fouls capture match tempo and aggression — high-foul matches
    produce more set pieces which correlate with goals. Championship is
    notoriously more physical than the PL.
    """
    if "HY" not in df.columns:
        return df

    # Build long format
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
    feat_long = long[["Date", "Team"] + new_cols].drop_duplicates(["Date", "Team"])

    for prefix in ["Home", "Away"]:
        renamed = feat_long.rename(columns={c: f"{prefix}_{c}" for c in new_cols})
        df = df.merge(
            renamed,
            left_on=["Date", f"{prefix}_Team"],
            right_on=["Date", "Team"],
            how="left",
        ).drop(columns=["Team"], errors="ignore")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Half-time scoring features
# ═══════════════════════════════════════════════════════════════════════════════

def add_halftime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling half-time scoring features from HTHG/HTAG columns.

    Half-time patterns capture team pace/style — teams that score early
    tend to open games up, creating more second-half goals. Championship
    has distinct HT patterns due to higher tempo.
    """
    if "HTHG" not in df.columns:
        return df

    records: list[dict] = []
    for idx, row in df.iterrows():
        hthg = row.get("HTHG", np.nan)
        htag = row.get("HTAG", np.nan)
        ht_tg = row.get("HT_TG", np.nan)

        for team, ht_scored, ht_conceded, is_home in [
            (row["Home_Team"], hthg, htag, True),
            (row["Away_Team"], htag, hthg, False),
        ]:
            records.append({
                "match_idx": idx,
                "team": team,
                "date": row["Date"],
                "is_home": is_home,
                "ht_scored": ht_scored if pd.notna(ht_scored) else np.nan,
                "ht_conceded": ht_conceded if pd.notna(ht_conceded) else np.nan,
                "ht_tg": ht_tg if pd.notna(ht_tg) else np.nan,
                "ht_over05": 1 if pd.notna(ht_tg) and ht_tg > 0.5 else 0,
                "ht_over15": 1 if pd.notna(ht_tg) and ht_tg > 1.5 else 0,
                "ht_btts": 1 if (pd.notna(ht_scored) and pd.notna(ht_conceded)
                                 and ht_scored > 0 and ht_conceded > 0) else 0,
            })

    ht_log = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)
    feat_map: dict[int, dict[str, float]] = {}

    for team, grp in ht_log.groupby("team"):
        g = grp.copy()

        def _rm(col: str, w: int) -> pd.Series:
            return g[col].shift(1).rolling(w, min_periods=1).mean()

        g["ht_scored_5"] = _rm("ht_scored", 5)
        g["ht_conceded_5"] = _rm("ht_conceded", 5)
        g["ht_tg_5"] = _rm("ht_tg", 5)
        g["ht_over05_5"] = _rm("ht_over05", 5)
        g["ht_over15_5"] = _rm("ht_over15", 5)
        g["ht_btts_5"] = _rm("ht_btts", 5)

        for _, r in g.iterrows():
            midx = r["match_idx"]
            prefix = "Home" if r["is_home"] else "Away"
            if midx not in feat_map:
                feat_map[midx] = {}
            feat_map[midx][f"{prefix}_HT_Scored_5"] = r["ht_scored_5"]
            feat_map[midx][f"{prefix}_HT_Conceded_5"] = r["ht_conceded_5"]
            feat_map[midx][f"{prefix}_HT_TG_5"] = r["ht_tg_5"]
            feat_map[midx][f"{prefix}_HT_Over05_5"] = r["ht_over05_5"]
            feat_map[midx][f"{prefix}_HT_Over15_5"] = r["ht_over15_5"]
            feat_map[midx][f"{prefix}_HT_BTTS_5"] = r["ht_btts_5"]

    ht_feat_df = pd.DataFrame.from_dict(feat_map, orient="index")
    df = df.join(ht_feat_df, how="left")

    # Combined HT features
    df["Combined_HT_TG"] = (
        df.get("Home_HT_TG_5", pd.Series(dtype=float)).fillna(0)
        + df.get("Away_HT_TG_5", pd.Series(dtype=float)).fillna(0)
    ) / 2
    df["Combined_HT_Over05"] = (
        df.get("Home_HT_Over05_5", pd.Series(dtype=float)).fillna(0)
        + df.get("Away_HT_Over05_5", pd.Series(dtype=float)).fillna(0)
    ) / 2

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Advanced rolling features
# ═══════════════════════════════════════════════════════════════════════════════

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute advanced features from team-level match history.

    Includes Over 2.5 rates, BTTS rates, clean sheets, EWM variants,
    Elo ratings, multiple Poisson models, and corner features.
    """
    # Build per-team match log
    records: list[dict] = []
    for idx, row in df.iterrows():
        hg = row["Home_Goals"]
        ag = row["Away_Goals"]
        tg = row["TG"]
        for team, scored, conceded, is_home in [
            (row["Home_Team"], hg, ag, True),
            (row["Away_Team"], ag, hg, False),
        ]:
            records.append({
                "match_idx": idx,
                "team": team,
                "date": row["Date"],
                "season": row["SeasonIndex"],
                "is_home": is_home,
                "scored": scored if pd.notna(scored) else np.nan,
                "conceded": conceded if pd.notna(conceded) else np.nan,
                "tg": tg if pd.notna(tg) else np.nan,
                "over25": 1 if pd.notna(tg) and tg > 2.5 else 0,
                "btts": 1 if (pd.notna(scored) and pd.notna(conceded) and
                              scored > 0 and conceded > 0) else 0,
                "cs": 1 if pd.notna(conceded) and conceded == 0 else 0,
                "corners": row.get("Home_Corners", np.nan) if is_home else row.get("Away_Corners", np.nan),
                "corners_conc": row.get("Away_Corners", np.nan) if is_home else row.get("Home_Corners", np.nan),
                "shots": row.get("Home_Shots", np.nan) if is_home else row.get("Away_Shots", np.nan),
                "sot": row.get("Home_Shots_Target", np.nan) if is_home else row.get("Away_Shots_Target", np.nan),
            })

    team_log = pd.DataFrame(records).sort_values(["team", "date"]).reset_index(drop=True)

    feat_map: dict[int, dict[str, float]] = {}

    for team, grp in team_log.groupby("team"):
        g = grp.copy()
        n = len(g)

        # Shifted series (no lookahead)
        def _rm(col: str, w: int) -> pd.Series:
            return g[col].shift(1).rolling(w, min_periods=1).mean()

        def _rs(col: str, w: int) -> pd.Series:
            return g[col].shift(1).rolling(w, min_periods=1).sum()

        def _ewm(col: str, span: int) -> pd.Series:
            return g[col].shift(1).ewm(span=span, min_periods=1).mean()

        # Rolling 5-game
        g["over25_5"] = _rm("over25", 5)
        g["btts_5"] = _rm("btts", 5)
        g["cs_5"] = _rm("cs", 5)
        g["tg_avg_5"] = _rm("tg", 5)
        g["gpg_20"] = _rm("scored", 20)
        g["gapg_20"] = _rm("conceded", 20)

        # Option 3 Step 2c: short-horizon _3 windows
        g["over25_3"] = _rm("over25", 3)
        g["btts_3"] = _rm("btts", 3)
        g["tg_avg_3"] = _rm("tg", 3)
        g["past3_goals"] = _rs("scored", 3)
        g["corners_avg_3"] = _rm("corners", 3)

        # EWM (span=10) — lower variance
        g["over25_ewm10"] = _ewm("over25", 10)
        g["tg_ewm10"] = _ewm("tg", 10)
        g["btts_ewm10"] = _ewm("btts", 10)
        g["gpg_ewm10"] = _ewm("scored", 10)

        # Corners
        g["corners_avg_5"] = _rm("corners", 5)
        g["corners_conc_5"] = _rm("corners_conc", 5)

        # BTTS-specific
        fts = (g["scored"] == 0).astype(float)
        g["fts_5"] = fts.shift(1).rolling(5, min_periods=1).mean()
        g["fts_10"] = fts.shift(1).rolling(10, min_periods=1).mean()
        g["btts_10"] = _rm("btts", 10)
        g["cs_streak"] = 0.0
        streak = 0
        for i in range(n):
            if i > 0:
                g.iloc[i, g.columns.get_loc("cs_streak")] = streak
            if g.iloc[i]["cs"] == 1:
                streak += 1
            else:
                streak = 0
        g["goal_std_10"] = g["scored"].shift(1).rolling(10, min_periods=3).std()

        # Store per match
        for _, r in g.iterrows():
            midx = r["match_idx"]
            prefix = "home" if r["is_home"] else "away"
            if midx not in feat_map:
                feat_map[midx] = {}
            feat_map[midx][f"{prefix}_over25_5"] = r["over25_5"]
            feat_map[midx][f"{prefix}_btts_5"] = r["btts_5"]
            feat_map[midx][f"{prefix}_cs_5"] = r["cs_5"]
            feat_map[midx][f"{prefix}_tg_avg_5"] = r["tg_avg_5"]
            feat_map[midx][f"{prefix}_gpg_20"] = r["gpg_20"]
            feat_map[midx][f"{prefix}_gapg_20"] = r["gapg_20"]
            feat_map[midx][f"{prefix}_over25_ewm10"] = r["over25_ewm10"]
            feat_map[midx][f"{prefix}_tg_ewm10"] = r["tg_ewm10"]
            feat_map[midx][f"{prefix}_btts_ewm10"] = r["btts_ewm10"]
            feat_map[midx][f"{prefix}_gpg_ewm10"] = r["gpg_ewm10"]
            feat_map[midx][f"{prefix}_corners_avg_5"] = r["corners_avg_5"]
            feat_map[midx][f"{prefix}_corners_conc_5"] = r["corners_conc_5"]
            feat_map[midx][f"{prefix}_fts_5"] = r["fts_5"]
            feat_map[midx][f"{prefix}_fts_10"] = r["fts_10"]
            feat_map[midx][f"{prefix}_btts_10"] = r["btts_10"]
            feat_map[midx][f"{prefix}_cs_streak"] = r["cs_streak"]
            feat_map[midx][f"{prefix}_goal_std_10"] = r["goal_std_10"]
            # Option 3 Step 2c: short-horizon _3 windows
            feat_map[midx][f"{prefix}_over25_3"] = r["over25_3"]
            feat_map[midx][f"{prefix}_btts_3"] = r["btts_3"]
            feat_map[midx][f"{prefix}_tg_avg_3"] = r["tg_avg_3"]
            feat_map[midx][f"{prefix}_past3_goals"] = r["past3_goals"]
            feat_map[midx][f"{prefix}_corners_avg_3"] = r["corners_avg_3"]

    feat_df = pd.DataFrame.from_dict(feat_map, orient="index")

    # Map to canonical column names
    col_map = {
        "home_over25_5": "Home_Over25_5", "away_over25_5": "Away_Over25_5",
        "home_btts_5": "Home_BTTS_5", "away_btts_5": "Away_BTTS_5",
        "home_cs_5": "Home_CS_5", "away_cs_5": "Away_CS_5",
        "home_tg_avg_5": "Home_TGAvg_5", "away_tg_avg_5": "Away_TGAvg_5",
        "home_gpg_20": "Home_GPG_20", "away_gpg_20": "Away_GPG_20",
        "home_gapg_20": "Home_GAPG_20", "away_gapg_20": "Away_GAPG_20",
        "home_over25_ewm10": "Home_Over25_EWM10", "away_over25_ewm10": "Away_Over25_EWM10",
        "home_tg_ewm10": "Home_TGAvg_EWM10", "away_tg_ewm10": "Away_TGAvg_EWM10",
        "home_btts_ewm10": "Home_BTTS_EWM10", "away_btts_ewm10": "Away_BTTS_EWM10",
        "home_gpg_ewm10": "Home_GPG_EWM10", "away_gpg_ewm10": "Away_GPG_EWM10",
        "home_corners_avg_5": "Home_CornersAvg_5", "away_corners_avg_5": "Away_CornersAvg_5",
        "home_corners_conc_5": "Home_CornersConcAvg_5", "away_corners_conc_5": "Away_CornersConcAvg_5",
        "home_fts_5": "Home_FTS_5", "away_fts_5": "Away_FTS_5",
        "home_fts_10": "Home_FTS_10", "away_fts_10": "Away_FTS_10",
        "home_btts_10": "Home_BTTS_10", "away_btts_10": "Away_BTTS_10",
        "home_cs_streak": "Home_CSStreak", "away_cs_streak": "Away_CSStreak",
        "home_goal_std_10": "Home_GoalStd_10", "away_goal_std_10": "Away_GoalStd_10",
        # Option 3 Step 2c: _3 rolling windows
        "home_over25_3": "Home_Over25_3", "away_over25_3": "Away_Over25_3",
        "home_btts_3": "Home_BTTS_3", "away_btts_3": "Away_BTTS_3",
        "home_tg_avg_3": "Home_TGAvg_3", "away_tg_avg_3": "Away_TGAvg_3",
        "home_past3_goals": "Home_Past3Goals", "away_past3_goals": "Away_Past3Goals",
        "home_corners_avg_3": "Home_CornersAvg_3", "away_corners_avg_3": "Away_CornersAvg_3",
    }
    feat_df = feat_df.rename(columns=col_map)
    df = df.join(feat_df, how="left")

    # Combined/interaction features
    df["Combined_Over25"] = (df["Home_Over25_5"] + df["Away_Over25_5"]) / 2
    df["Combined_BTTS"] = (df["Home_BTTS_5"] + df["Away_BTTS_5"]) / 2
    df["Attack_Power"] = df["Home_GPG_20"] + df["Away_GPG_20"]
    df["Corner_Dominance"] = (
        (df["Home_CornersAvg_5"] - df["Home_CornersConcAvg_5"]) +
        (df["Away_CornersConcAvg_5"] - df["Away_CornersAvg_5"])
    )

    # Option 3 Step 2a: Corner efficiency = rolling goals / rolling corners.
    # Hypothesis: teams converting set-piece pressure into goals more
    # reliably sustain higher goal expectancy in tight games.
    # Formula: Past5Goals is a 5-match sum; CornersAvg_5 is a per-match
    # mean, so the sum-equivalent is CornersAvg_5 * 5. Ratio gives
    # goals-per-corner over a 5-match window. Guard divide-by-zero.
    _h_corners_5_sum = df["Home_CornersAvg_5"] * 5
    _a_corners_5_sum = df["Away_CornersAvg_5"] * 5
    df["Home_CornerEfficiency_5"] = (
        df["Home_Past5Goals"] / (_h_corners_5_sum + 1e-6))
    df["Away_CornerEfficiency_5"] = (
        df["Away_Past5Goals"] / (_a_corners_5_sum + 1e-6))
    # If a team won fewer than 1 corner in the window, mark NaN
    df.loc[_h_corners_5_sum < 1.0, "Home_CornerEfficiency_5"] = float("nan")
    df.loc[_a_corners_5_sum < 1.0, "Away_CornerEfficiency_5"] = float("nan")

    # BTTS-specific combined
    df["Combined_FTS"] = (df["Home_FTS_5"] + df["Away_FTS_5"]) / 2
    df["Blanking_Risk"] = 1 - (1 - df["Home_FTS_5"]) * (1 - df["Away_FTS_5"])
    df["BTTS_Attack_Power"] = df["Home_GPG_20"] * df["Away_GPG_20"]
    df["CS_Risk"] = (df["Home_CS_5"] + df["Away_CS_5"]) / 2

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Elo ratings
# ═══════════════════════════════════════════════════════════════════════════════

def add_elo(df: pd.DataFrame, k: float = 20.0, home_adv: float = 50.0) -> pd.DataFrame:
    """Compute Elo ratings and Elo_Diff feature."""
    elo: dict[str, float] = {}
    elo_diffs: list[float] = []

    for idx, row in df.iterrows():
        home, away = row["Home_Team"], row["Away_Team"]
        h_elo = elo.get(home, 1500.0)
        a_elo = elo.get(away, 1500.0)

        elo_diffs.append(h_elo - a_elo + home_adv)

        # Update Elo after the match
        ftr = row.get("FTR", "")
        if ftr in ("H", "A", "D"):
            expected_h = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
            actual_h = 1.0 if ftr == "H" else (0.5 if ftr == "D" else 0.0)
            elo[home] = h_elo + k * (actual_h - expected_h)
            elo[away] = a_elo + k * ((1 - actual_h) - (1 - expected_h))

    df["Elo_Diff"] = elo_diffs
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Poisson probability models
# ═══════════════════════════════════════════════════════════════════════════════

def add_poisson_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Poisson P(Over 2.5) and P(Over 1.5) from goals/shots lambdas."""

    def _p_over_n(hl: float, al: float, threshold: float) -> float:
        """P(total goals > threshold) from independent Poisson."""
        if np.isnan(hl) or np.isnan(al):
            return np.nan
        hl = np.clip(hl, 0.1, 5.0)
        al = np.clip(al, 0.1, 5.0)
        n = int(threshold)  # 2 for O/U 2.5, 1 for O/U 1.5
        p_under = sum(
            poisson_dist.pmf(hg, hl) * poisson_dist.pmf(ag, al)
            for hg in range(n + 2) for ag in range(n + 2)
            if hg + ag <= n
        )
        return 1 - p_under

    # Goals-based lambda
    home_goals_lambda = df["Home_GPG_20"]
    away_goals_lambda = df["Away_GPG_20"]

    # Shots-based lambda (Wheatcroft approach)
    sot_cr_home = df.get("Home_SOT_CR_5", pd.Series([np.nan] * len(df)))
    sot_cr_away = df.get("Away_SOT_CR_5", pd.Series([np.nan] * len(df)))
    home_sot = df.get("Home_AvgShotsOnTarget_5", pd.Series([np.nan] * len(df)))
    away_sot = df.get("Away_AvgShotsOnTarget_5", pd.Series([np.nan] * len(df)))
    home_shots_lambda = home_sot * sot_cr_home
    away_shots_lambda = away_sot * sot_cr_away

    # O/U 2.5 Poisson
    poisson_goals_25 = np.array([
        _p_over_n(home_goals_lambda.iloc[i], away_goals_lambda.iloc[i], 2.5)
        for i in range(len(df))
    ])
    poisson_shots_25 = np.array([
        _p_over_n(home_shots_lambda.iloc[i], away_shots_lambda.iloc[i], 2.5)
        for i in range(len(df))
    ])

    df["Poisson_Goals"] = poisson_goals_25
    df["Poisson_Shots"] = poisson_shots_25
    df["Poisson_Consensus"] = np.nanmean(
        np.column_stack([poisson_goals_25, poisson_shots_25]), axis=1
    )

    # O/U 1.5 Poisson
    poisson_goals_15 = np.array([
        _p_over_n(home_goals_lambda.iloc[i], away_goals_lambda.iloc[i], 1.5)
        for i in range(len(df))
    ])
    poisson_shots_15 = np.array([
        _p_over_n(home_shots_lambda.iloc[i], away_shots_lambda.iloc[i], 1.5)
        for i in range(len(df))
    ])

    df["Poisson_Goals_15"] = poisson_goals_15
    df["Poisson_Shots_15"] = poisson_shots_15
    df["Poisson_Consensus_15"] = np.nanmean(
        np.column_stack([poisson_goals_15, poisson_shots_15]), axis=1
    )

    # Expected total goals consensus
    df["Expected_TG_Goals"] = home_goals_lambda + away_goals_lambda
    df["Expected_TG_Shots"] = home_shots_lambda + away_shots_lambda
    df["Expected_TG_Consensus"] = np.nanmean(
        np.column_stack([
            df["Expected_TG_Goals"], df["Expected_TG_Shots"]
        ]), axis=1
    )

    # BTTS Poisson
    def _p_btts(hl: float, al: float) -> float:
        if np.isnan(hl) or np.isnan(al):
            return np.nan
        hl = np.clip(hl, 0.1, 5.0)
        al = np.clip(al, 0.1, 5.0)
        p_home_zero = poisson_dist.pmf(0, hl)
        p_away_zero = poisson_dist.pmf(0, al)
        return (1 - p_home_zero) * (1 - p_away_zero)

    df["Poisson_BTTS"] = np.array([
        _p_btts(home_goals_lambda.iloc[i], away_goals_lambda.iloc[i])
        for i in range(len(df))
    ])
    df["Poisson_BTTS_Consensus"] = df["Poisson_BTTS"]  # Only one variant without xG

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Context features (relegation, title, promotion proximity)
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_prior_season_proximities(df: pd.DataFrame) -> dict:
    """Pre-compute final-season proximity values for each team per season.

    Used to carry over context into the next season's early matches.
    Returns: {season_idx: {team: {relprox, titleprox, promoprox, playoffprox}}}
    """
    games_per_season = LEAGUE_CFG["games_per_season"]
    max_pts = float(LEAGUE_CFG["max_points"])
    rel_pos = LEAGUE_CFG["relegation_pos"]  # 22

    prior_proxim: dict[int, dict] = {}

    for season_idx, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        points: dict[str, int] = {}
        gd_map: dict[str, int] = {}

        for _, row in season_matches.iterrows():
            home, away = row["Home_Team"], row["Away_Team"]
            for t in [home, away]:
                if t not in points:
                    points[t] = 0
                    gd_map[t] = 0

            hg, ag = row["Home_Goals"], row["Away_Goals"]
            if pd.notna(hg) and pd.notna(ag):
                hg, ag = int(hg), int(ag)
                gd_map[home] += (hg - ag)
                gd_map[away] += (ag - hg)
                ftr = row.get("FTR", "")
                if ftr == "H":
                    points[home] += 3
                elif ftr == "A":
                    points[away] += 3
                elif ftr == "D":
                    points[home] += 1
                    points[away] += 1

        # Final standings
        standings = sorted(points.keys(), key=lambda t: (-points[t], -gd_map[t]))
        pts_list = [points[t] for t in standings]
        pts_1st = pts_list[0] if len(pts_list) >= 1 else 0
        pts_2nd = pts_list[1] if len(pts_list) >= 2 else 0
        pts_6th = pts_list[5] if len(pts_list) >= 6 else 0
        pts_rel = pts_list[rel_pos - 1] if len(pts_list) >= rel_pos else 0

        team_proxim: dict[str, dict[str, float]] = {}
        for t in standings:
            team_proxim[t] = {
                "relprox": (points[t] - pts_rel) / max_pts,
                "titleprox": (pts_1st - points[t]) / max_pts,
                "promoprox": (points[t] - pts_2nd) / max_pts,
                "playoffprox": (points[t] - pts_6th) / max_pts,
            }

        # Default for promoted/new teams entering NEXT season
        team_proxim["__promoted_default__"] = {
            "relprox": 0.0,
            "titleprox": (pts_1st - pts_rel) / max_pts,
            "promoprox": (pts_rel - pts_2nd) / max_pts,
            "playoffprox": (pts_rel - pts_6th) / max_pts,
        }

        prior_proxim[season_idx] = team_proxim

    return prior_proxim


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute season progress and standing proximity features with
    prior-season blending.

    For each match, BEFORE applying its result, compute current-season
    proximity values, then blend with prior season's final proximity:
        blended = (1 - Season_Progress) * prior + Season_Progress * current

    This gives early-season matches meaningful context signal based on
    where teams finished last year, transitioning smoothly to current
    standings as real results accumulate.

    Championship-specific: relegation_pos=22, promotion (top 2),
    playoff (3rd-6th). No Euro proximity.
    """
    games_per_season = LEAGUE_CFG["games_per_season"]
    rel_pos = LEAGUE_CFG["relegation_pos"]  # 22

    # Initialise columns
    context_cols = [
        "Season_Progress",
        "Home_RelegationProximity", "Away_RelegationProximity",
        "Home_TitleProximity", "Away_TitleProximity",
        "Home_PromotionProximity", "Away_PromotionProximity",
        "Home_PlayoffProximity", "Away_PlayoffProximity",
    ]
    for col in context_cols:
        df[col] = 0.0

    # Pre-compute prior season final proximities
    prior_proxim = _compute_prior_season_proximities(df)
    no_prior = {"relprox": 0.0, "titleprox": 0.0, "promoprox": 0.0, "playoffprox": 0.0}

    for season_idx, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        indices = season_matches.index.tolist()

        prev_season = prior_proxim.get(season_idx - 1, {})
        promoted_default = prev_season.get("__promoted_default__", no_prior)

        points: dict[str, int] = {}
        gd: dict[str, int] = {}
        played: dict[str, int] = {}

        for idx in indices:
            row = df.loc[idx]
            home, away = row["Home_Team"], row["Away_Team"]

            for t in [home, away]:
                if t not in points:
                    points[t] = 0
                    gd[t] = 0
                    played[t] = 0

            # BEFORE result: compute context from current standings
            standings = sorted(points.keys(), key=lambda t: (-points[t], -gd[t]))
            pts_by_pos = [points[t] for t in standings]

            pts_1st = pts_by_pos[0] if len(pts_by_pos) >= 1 else 0
            pts_2nd = pts_by_pos[1] if len(pts_by_pos) >= 2 else 0
            pts_6th = pts_by_pos[5] if len(pts_by_pos) >= 6 else 0
            pts_rel = pts_by_pos[rel_pos - 1] if len(pts_by_pos) >= rel_pos else 0

            avg_played = (played[home] + played[away]) / 2
            progress = avg_played / float(games_per_season)
            max_remaining = max((games_per_season - avg_played) * 3, 1)

            df.at[idx, "Season_Progress"] = progress

            for team, prefix in [(home, "Home"), (away, "Away")]:
                t_pts = points[team]
                curr_rel = (t_pts - pts_rel) / max_remaining
                curr_title = (pts_1st - t_pts) / max_remaining
                curr_promo = (t_pts - pts_2nd) / max_remaining
                curr_playoff = (t_pts - pts_6th) / max_remaining

                # Blend with prior season (if available)
                if season_idx > 0 and prev_season:
                    prior = prev_season.get(team, promoted_default)
                    w = progress  # 0 at start -> 100% prior; 1 at end -> 100% current
                    df.at[idx, f"{prefix}_RelegationProximity"] = (1 - w) * prior["relprox"] + w * curr_rel
                    df.at[idx, f"{prefix}_TitleProximity"] = (1 - w) * prior["titleprox"] + w * curr_title
                    df.at[idx, f"{prefix}_PromotionProximity"] = (1 - w) * prior["promoprox"] + w * curr_promo
                    df.at[idx, f"{prefix}_PlayoffProximity"] = (1 - w) * prior["playoffprox"] + w * curr_playoff
                else:
                    df.at[idx, f"{prefix}_RelegationProximity"] = curr_rel
                    df.at[idx, f"{prefix}_TitleProximity"] = curr_title
                    df.at[idx, f"{prefix}_PromotionProximity"] = curr_promo
                    df.at[idx, f"{prefix}_PlayoffProximity"] = curr_playoff

            # AFTER: update standings with this match's result
            hg, ag = row["Home_Goals"], row["Away_Goals"]
            if pd.notna(hg) and pd.notna(ag):
                hg, ag = int(hg), int(ag)
                gd[home] += (hg - ag)
                gd[away] += (ag - hg)
                played[home] += 1
                played[away] += 1
                ftr = row.get("FTR", "")
                if ftr == "H":
                    points[home] += 3
                elif ftr == "A":
                    points[away] += 3
                elif ftr == "D":
                    points[home] += 1
                    points[away] += 1

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════════

# All Championship features in canonical order (O/U 2.5 model)
CHAMP_ALL_FEATURES = [
    # From CSV (basic rolling)
    "Home Factor", "Away Factor",
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    "Home_ShotRatio_5", "Away_ShotRatio_5",
    "Home_ShotsPerGoal_5", "Away_ShotsPerGoal_5",
    "Home_CR_5", "Home_CR_20", "Away_CR_5", "Away_CR_20",
    "Home_SOT_CR_5", "Home_SOT_CR_20", "Away_SOT_CR_5", "Away_SOT_CR_20",
    "Home_DefensiveStrength_5", "Away_DefensiveStrength_5",
    "Home_DefensiveStrength_SOT", "Away_DefensiveStrength_SOT",
    "Home_LeaguePosition", "Away_LeaguePosition",
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
    "Home_Past5Corners", "Away_Past5Corners",
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",
    "Home_Promoted", "Away_Promoted",
    "Local Derby", "Historical Derby",
    "H2H_HomeWins", "H2H_AwayWins", "H2H_Draws",
    "H2H_AvgGoals_5", "H2HAvgGoals",
    # Derived
    "LeaguePosition_Diff", "Home_GoalDiff_5", "Away_GoalDiff_5",
    "Home_RestDays", "Away_RestDays",
    # Congestion
    "Home_MatchesLast14Days", "Away_MatchesLast14Days",
    "Home_AvgRest3", "Away_AvgRest3",
    # Discipline
    "Home_YellowCards_5", "Away_YellowCards_5",
    "Home_RedCards_10", "Away_RedCards_10",
    "Home_Fouls_5", "Away_Fouls_5",
    # Half-time
    "Home_HT_Scored_5", "Away_HT_Scored_5",
    "Home_HT_Conceded_5", "Away_HT_Conceded_5",
    "Home_HT_TG_5", "Away_HT_TG_5",
    "Home_HT_Over05_5", "Away_HT_Over05_5",
    "Home_HT_Over15_5", "Away_HT_Over15_5",
    "Combined_HT_TG", "Combined_HT_Over05",
    # Advanced rolling
    "Home_Over25_5", "Away_Over25_5",
    "Home_BTTS_5", "Away_BTTS_5",
    "Home_CS_5", "Away_CS_5",
    "Home_TGAvg_5", "Away_TGAvg_5",
    "Home_GPG_20", "Away_GPG_20",
    "Home_GAPG_20", "Away_GAPG_20",
    "Home_Over25_EWM10", "Away_Over25_EWM10",
    "Home_TGAvg_EWM10", "Away_TGAvg_EWM10",
    "Home_BTTS_EWM10", "Away_BTTS_EWM10",
    "Home_GPG_EWM10", "Away_GPG_EWM10",
    "Home_CornersAvg_5", "Away_CornersAvg_5",
    "Home_CornersConcAvg_5", "Away_CornersConcAvg_5",
    "Corner_Dominance",
    # Option 3 Step 2a: Corner efficiency (goals per corner, rolling 5)
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    # Option 3 Step 2c: short-horizon _3 windows
    "Home_Over25_3", "Away_Over25_3",
    "Home_BTTS_3", "Away_BTTS_3",
    "Home_TGAvg_3", "Away_TGAvg_3",
    "Home_Past3Goals", "Away_Past3Goals",
    "Home_CornersAvg_3", "Away_CornersAvg_3",
    "Combined_Over25", "Combined_BTTS", "Attack_Power",
    # Elo
    "Elo_Diff",
    # Poisson
    "Poisson_Goals", "Poisson_Shots", "Poisson_Consensus",
    "Expected_TG_Consensus",
    # Computed strengths (FPL replacement)
    "Home_FPL_Attack", "Away_FPL_Attack",
    "Home_FPL_Defence", "Away_FPL_Defence",
    "FPL_Openness", "FPL_HomeDominance",
    # Context
    "Season_Progress",
    "Home_RelegationProximity", "Away_RelegationProximity",
    "Home_TitleProximity", "Away_TitleProximity",
    "Home_PromotionProximity", "Away_PromotionProximity",
    "Home_PlayoffProximity", "Away_PlayoffProximity",
]

# O/U 1.5 feature set — same as O/U 2.5 but with 1.5-specific Poisson
CHAMP_OU15_FEATURES = [f for f in CHAMP_ALL_FEATURES]
# Replace O/U 2.5 Poisson with O/U 1.5 variants
for _old, _new in [
    ("Poisson_Goals", "Poisson_Goals_15"),
    ("Poisson_Shots", "Poisson_Shots_15"),
    ("Poisson_Consensus", "Poisson_Consensus_15"),
]:
    if _old in CHAMP_OU15_FEATURES:
        CHAMP_OU15_FEATURES[CHAMP_OU15_FEATURES.index(_old)] = _new

# BTTS-specific feature set for Championship
CHAMP_BTTS_FEATURES = [
    # Option 3 Step 2a: corner efficiency
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    # Option 3 Step 2c: short-horizon _3 BTTS-relevant features
    "Home_Past3Goals", "Away_Past3Goals",
    "Home_BTTS_3", "Away_BTTS_3",
    # Scoring/conceding
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
    "Home_CR_5", "Home_CR_20", "Away_CR_5", "Away_CR_20",
    "Home_SOT_CR_5", "Home_SOT_CR_20", "Away_SOT_CR_5", "Away_SOT_CR_20",
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    "Home_DefensiveStrength_5", "Away_DefensiveStrength_5",
    "Home_LeaguePosition", "Away_LeaguePosition",
    "LeaguePosition_Diff",
    "Home_RestDays", "Away_RestDays",
    # Congestion
    "Home_MatchesLast14Days", "Away_MatchesLast14Days",
    "Home_AvgRest3", "Away_AvgRest3",
    # Discipline
    "Home_YellowCards_5", "Away_YellowCards_5",
    "Home_Fouls_5", "Away_Fouls_5",
    # Half-time
    "Home_HT_Scored_5", "Away_HT_Scored_5",
    "Home_HT_BTTS_5", "Away_HT_BTTS_5",
    "Combined_HT_TG",
    # Basic
    "Home_Promoted", "Away_Promoted",
    "Local Derby", "Historical Derby",
    "Home_BTTS_5", "Away_BTTS_5",
    "Home_CS_5", "Away_CS_5",
    "Combined_BTTS",
    "Home_BTTS_EWM10", "Away_BTTS_EWM10",
    "Home_GPG_20", "Away_GPG_20",
    "Home_GAPG_20", "Away_GAPG_20",
    "Home_GPG_EWM10", "Away_GPG_EWM10",
    "Attack_Power", "Elo_Diff",
    "Home_FPL_Attack", "Away_FPL_Attack",
    "Home_FPL_Defence", "Away_FPL_Defence",
    "FPL_Openness",
    "Season_Progress",
    "Home_RelegationProximity", "Away_RelegationProximity",
    # BTTS-specific
    "Home_FTS_5", "Away_FTS_5",
    "Home_FTS_10", "Away_FTS_10",
    "Home_GoalStd_10", "Away_GoalStd_10",
    "Home_CSStreak", "Away_CSStreak",
    "Home_BTTS_10", "Away_BTTS_10",
    "Combined_FTS", "Blanking_Risk",
    "Poisson_BTTS", "Poisson_BTTS_Consensus",
    "BTTS_Attack_Power", "CS_Risk",
]


def _get_pl_teams_by_season() -> dict[int, set[str]]:
    """Load PL CSV and return Championship-normalised team names per season.

    Returns:
        Dict mapping Championship SeasonIndex -> set of Championship-style
        team names that played in the PL that season.
    """
    pl_cfg = get_league_config("PL")
    pl_path = pl_cfg["csv_path"]
    if not os.path.exists(pl_path):
        return {}

    pl_df = pd.read_csv(pl_path, usecols=["Home_Team", "SeasonIndex"])
    result: dict[int, set[str]] = {}
    for season_idx, grp in pl_df.groupby("SeasonIndex"):
        raw_names = set(grp["Home_Team"].unique())
        champ_names: set[str] = set()
        for raw in raw_names:
            stripped = raw.strip()
            for suffix in (" FC", " AFC"):
                if stripped.endswith(suffix):
                    stripped = stripped[: -len(suffix)].strip()
                    break
            champ_names.add(_PL_TO_CHAMP_NAME.get(stripped, stripped))
        result[int(season_idx)] = champ_names
    return result


def _detect_new_teams(
    df: pd.DataFrame,
) -> dict[int, set[str]]:
    """Detect teams new to the Championship each season via set difference.

    Returns:
        Dict mapping SeasonIndex -> set of team names that are new this season.
    """
    new_teams: dict[int, set[str]] = {}
    for season_idx in sorted(df["SeasonIndex"].unique()):
        if season_idx == 0:
            continue
        curr = (
            set(df[df["SeasonIndex"] == season_idx]["Home_Team"].unique())
            | set(df[df["SeasonIndex"] == season_idx]["Away_Team"].unique())
        )
        prev = (
            set(df[df["SeasonIndex"] == season_idx - 1]["Home_Team"].unique())
            | set(df[df["SeasonIndex"] == season_idx - 1]["Away_Team"].unique())
        )
        entered = curr - prev
        if entered:
            new_teams[int(season_idx)] = entered
    return new_teams


def initialize_promoted_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Initialise rolling features for newly promoted/relegated Championship teams.

    Teams entering the Championship have no rolling history, so their first
    few matches have NaN or unreliable features.  This function detects new
    teams each season, classifies them as **PL-relegated** or **L1-promoted**,
    and seeds their early-match features with reference averages from the
    prior Championship season:

    * **PL-relegated teams** (were in the PL last season): seeded with
      *mid-table* (positions 8-16) averages — these teams have more
      resources and are expected to be competitive.
    * **L1-promoted teams**: seeded with *bottom-5* averages — same
      approach as the PL pipeline uses for newly promoted teams.

    Features are blended over the first 5 matches with a decaying weight:
        Match 1: 100% reference avg
        Match 2:  80% avg + 20% actual
        Match 3:  60% avg + 40% actual
        Match 4:  40% avg + 60% actual
        Match 5:  20% avg + 80% actual
        Match 6+: 100% actual

    Args:
        df: Full Championship DataFrame with all features computed.

    Returns:
        Tuple of (modified DataFrame, count of feature values filled).
    """
    df = df.copy()

    # Rolling features to initialise (all window-based features that are
    # unreliable or NaN for teams with no Championship history)
    rolling_features = [
        # CSV-sourced rolling (5-game)
        "Home_Past5Goals", "Away_Past5Goals",
        "Home_Past5Conceded", "Away_Past5Conceded",
        "Home_Past5Corners", "Away_Past5Corners",
        "Home_Past5CornersConceded", "Away_Past5CornersConceded",
        "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
        "Home_ShotRatio_5", "Away_ShotRatio_5",
        "Home_ShotsPerGoal_5", "Away_ShotsPerGoal_5",
        "Home_CR_5", "Away_CR_5",
        "Home_CR_20", "Away_CR_20",
        "Home_SOT_CR_5", "Away_SOT_CR_5",
        "Home_SOT_CR_20", "Away_SOT_CR_20",
        "Home_DefensiveStrength_5", "Away_DefensiveStrength_5",
        "Home_DefensiveStrength_SOT", "Away_DefensiveStrength_SOT",
        # Derived rolling
        "Home_GoalDiff_5", "Away_GoalDiff_5",
        # Advanced rolling
        "Home_Over25_5", "Away_Over25_5",
        "Home_BTTS_5", "Away_BTTS_5",
        "Home_CS_5", "Away_CS_5",
        "Home_TGAvg_5", "Away_TGAvg_5",
        "Home_GPG_20", "Away_GPG_20",
        "Home_GAPG_20", "Away_GAPG_20",
        # EWM
        "Home_Over25_EWM10", "Away_Over25_EWM10",
        "Home_TGAvg_EWM10", "Away_TGAvg_EWM10",
        "Home_BTTS_EWM10", "Away_BTTS_EWM10",
        "Home_GPG_EWM10", "Away_GPG_EWM10",
        # Corners rolling
        "Home_CornersAvg_5", "Away_CornersAvg_5",
        "Home_CornersConcAvg_5", "Away_CornersConcAvg_5",
        # Half-time
        "Home_HT_Scored_5", "Away_HT_Scored_5",
        "Home_HT_Conceded_5", "Away_HT_Conceded_5",
        "Home_HT_TG_5", "Away_HT_TG_5",
        "Home_HT_Over05_5", "Away_HT_Over05_5",
        "Home_HT_Over15_5", "Away_HT_Over15_5",
        # Discipline
        "Home_YellowCards_5", "Away_YellowCards_5",
        "Home_Fouls_5", "Away_Fouls_5",
        # BTTS-specific
        "Home_FTS_5", "Away_FTS_5",
        "Home_FTS_10", "Away_FTS_10",
        "Home_BTTS_10", "Away_BTTS_10",
        "Home_GoalStd_10", "Away_GoalStd_10",
    ]
    # Only keep features actually present in the DataFrame
    rolling_features = [f for f in rolling_features if f in df.columns]

    blend_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2}
    filled = 0

    # Cross-reference PL data to classify new teams
    pl_teams_by_season = _get_pl_teams_by_season()
    new_teams_by_season = _detect_new_teams(df)

    for season_idx in sorted(df["SeasonIndex"].unique()):
        if season_idx == 0:
            continue

        new_teams = new_teams_by_season.get(season_idx, set())
        if not new_teams:
            continue

        # Classify: PL-relegated vs L1-promoted
        # A team is PL-relegated if it was in the PL the prior season
        pl_prior = pl_teams_by_season.get(season_idx - 1, set())
        pl_relegated = new_teams & pl_prior
        l1_promoted = new_teams - pl_relegated

        # Compute reference averages from prior Championship season
        prev_mask = df["SeasonIndex"] == (season_idx - 1)
        prev_df = df[prev_mask]
        if prev_df.empty:
            continue

        # Get league positions from last match of prior season
        last_matches = prev_df.sort_values("Date").drop_duplicates(
            "Home_Team", keep="last"
        )
        team_positions: dict[str, float] = {}
        for _, row in last_matches.iterrows():
            team_positions[row["Home_Team"]] = row.get(
                "Home_LeaguePosition", 20
            )

        # Bottom-5 teams (worst league positions) for L1-promoted reference
        bottom5_teams = sorted(
            team_positions.keys(),
            key=lambda t: -team_positions.get(t, 20),
        )[:5]

        # Mid-table teams (positions 8-16) for PL-relegated reference
        sorted_teams = sorted(
            team_positions.keys(),
            key=lambda t: team_positions.get(t, 12),
        )
        midtable_teams = [
            t for t in sorted_teams
            if 8 <= team_positions.get(t, 99) <= 16
        ]
        # Fallback: if not enough mid-table teams, use positions 6-18
        if len(midtable_teams) < 3:
            midtable_teams = [
                t for t in sorted_teams
                if 6 <= team_positions.get(t, 99) <= 18
            ]

        def _compute_reference_avgs(
            reference_teams: list[str],
        ) -> dict[str, float]:
            """Average the last-match feature values for a set of reference teams."""
            avgs: dict[str, float] = {}
            for feat in rolling_features:
                prefix = "Home" if feat.startswith("Home") else "Away"
                vals: list[float] = []
                for team in reference_teams:
                    if prefix == "Home":
                        team_rows = prev_df[prev_df["Home_Team"] == team]
                    else:
                        team_rows = prev_df[prev_df["Away_Team"] == team]
                    if not team_rows.empty and feat in team_rows.columns:
                        last_val = team_rows.sort_values("Date").iloc[-1][feat]
                        if pd.notna(last_val):
                            vals.append(last_val)
                avgs[feat] = float(np.mean(vals)) if vals else np.nan
            return avgs

        bottom5_avgs = _compute_reference_avgs(bottom5_teams)
        midtable_avgs = _compute_reference_avgs(midtable_teams)

        # Apply blended features
        season_mask = df["SeasonIndex"] == season_idx

        for team in new_teams:
            # Select the right reference group
            ref_avgs = midtable_avgs if team in pl_relegated else bottom5_avgs

            for prefix in ["Home", "Away"]:
                team_col = f"{prefix}_Team"
                team_matches = df[season_mask & (df[team_col] == team)].sort_values("Date")

                for match_num, (idx, _) in enumerate(team_matches.iterrows(), 1):
                    if match_num > 5:
                        break
                    weight = blend_weights.get(match_num, 0)
                    if weight == 0:
                        continue

                    for feat in rolling_features:
                        if not feat.startswith(prefix):
                            continue
                        avg_val = ref_avgs.get(feat, np.nan)
                        if pd.isna(avg_val):
                            continue

                        actual = df.at[idx, feat]
                        if pd.isna(actual):
                            df.at[idx, feat] = avg_val
                        else:
                            df.at[idx, feat] = weight * avg_val + (1 - weight) * actual
                        filled += 1

    return df, filled


def run_pipeline(verbose: bool = True) -> dict:
    """Run the full Championship feature engineering pipeline.

    Returns:
        dict with keys: full_df, features, ou15_features, btts_features
    """
    if verbose:
        print("=== Championship Pipeline ===\n")

    df = load_championship_data()
    if verbose:
        print(f"Loaded {len(df)} matches, {df['SeasonIndex'].nunique()} seasons")

    if verbose:
        print("Adding derived features...")
    df = add_derived_features(df)

    if verbose:
        print("Adding congestion features...")
    df = add_congestion_features(df)

    if verbose:
        print("Adding discipline features...")
    df = add_discipline_features(df)

    if verbose:
        print("Adding half-time features...")
    df = add_halftime_features(df)

    if verbose:
        print("Computing advanced rolling features...")
    df = add_advanced_features(df)

    if verbose:
        print("Computing Elo ratings...")
    df = add_elo(df)

    if verbose:
        print("Computing Poisson features...")
    df = add_poisson_features(df)

    if verbose:
        print("Computing team strength ratings...")
    strengths = compute_team_strengths(df)
    df = merge_strengths(df, strengths)

    if verbose:
        print("Adding context features...")
    df = add_context_features(df)

    if verbose:
        print("Initialising promoted team features...")
    df, promoted_filled = initialize_promoted_features(df)
    if verbose:
        print(f"  Filled {promoted_filled} feature values for promoted teams")

    # Filter to available features
    available = [f for f in CHAMP_ALL_FEATURES if f in df.columns and df[f].notna().any()]
    ou15_available = [f for f in CHAMP_OU15_FEATURES if f in df.columns and df[f].notna().any()]
    btts_available = [f for f in CHAMP_BTTS_FEATURES if f in df.columns and df[f].notna().any()]

    if verbose:
        print(f"\nFeatures available: {len(available)} / {len(CHAMP_ALL_FEATURES)}")
        print(f"O/U 1.5 features available: {len(ou15_available)} / {len(CHAMP_OU15_FEATURES)}")
        print(f"BTTS features available: {len(btts_available)} / {len(CHAMP_BTTS_FEATURES)}")
        print(f"Over 2.5 rate: {df['Over_2_5'].mean():.3f}")
        print(f"Over 1.5 rate: {df['Over_1_5'].mean():.3f}")
        print(f"BTTS rate: {df['BTTS'].mean():.3f}")

    return {
        "full_df": df,
        "features": available,
        "ou15_features": ou15_available,
        "btts_features": btts_available,
    }


if __name__ == "__main__":
    result = run_pipeline()
    df = result["full_df"]
    feats = result["features"]
    ou15_feats = result["ou15_features"]
    btts_feats = result["btts_features"]
    print(f"\nFinal shape: {df.shape}")
    print(f"\nO/U 2.5 Feature list ({len(feats)}):")
    for f in feats:
        pct = df[f].notna().mean() * 100
        print(f"  {f:40s} {pct:5.1f}% non-null")
    print(f"\nO/U 1.5 Feature list ({len(ou15_feats)}):")
    for f in ou15_feats:
        pct = df[f].notna().mean() * 100
        print(f"  {f:40s} {pct:5.1f}% non-null")
    print(f"\nBTTS Feature list ({len(btts_feats)}):")
    for f in btts_feats:
        pct = df[f].notna().mean() * 100
        print(f"  {f:40s} {pct:5.1f}% non-null")
