"""
Data pipeline for Over/Under 2.5 Goals prediction.
Loads historical data, engineers features, and produces train/val/test splits.
"""
import pandas as pd
import numpy as np
import os
from scipy.stats import poisson
from features.common import (
    add_congestion_features,
    add_defensive_components,
    add_discipline_features,
)
from division_movement import seed_weight
from config import (
    DATA_PATH, ENRICHED_DATA_PATH, EXISTING_FEATURES, DERIVED_FEATURES,
    XG_FEATURES, PLAYER_FEATURES, SQUAD_FEATURES, DEFENSIVE_TIER3_FEATURES,
    ADVANCED_FEATURES,
    SHOT_LEVEL_FEATURES, ROSTER_FEATURES, TACTICAL_FEATURES,
    DETAILED_MATCH_FEATURES, CONTEXT_FEATURES,
    CONGESTION_FEATURES, DISCIPLINE_FEATURES, WEATHER_FEATURES, ALL_FEATURES,
    BTTS_ALL_FEATURES, CORNERS_ALL_FEATURES,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, TRAIN_MIN_SEASON,
    MATCHES_2425_PATH,
    GAMES_PER_SEASON, MAX_POINTS, RELEGATION_POS, EURO_POS,
    HAS_FPL, HAS_UNDERSTAT,
)


def load_data(path=None):
    """Load the dataset (enriched if available, otherwise original) and parse dates."""
    if path is None:
        path = ENRICHED_DATA_PATH if os.path.exists(ENRICHED_DATA_PATH) else DATA_PATH
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_target(df):
    """Create binary target: 1 if total goals > 2, else 0."""
    df["Over_2_5"] = (df["TG"] > 2).astype(int)
    return df


def add_rest_days(df):
    """Compute days since each team's previous match."""
    df = df.copy()
    home_rest = []
    away_rest = []

    # Build a dict of each team's last match date
    team_last_match = {}

    for _, row in df.iterrows():
        home = row["Home_Team"]
        away = row["Away_Team"]
        date = row["Date"]

        if home in team_last_match:
            home_rest.append((date - team_last_match[home]).days)
        else:
            home_rest.append(np.nan)

        if away in team_last_match:
            away_rest.append((date - team_last_match[away]).days)
        else:
            away_rest.append(np.nan)

        team_last_match[home] = date
        team_last_match[away] = date

    df["Home_RestDays"] = home_rest
    df["Away_RestDays"] = away_rest
    return df


def add_weather_features(df):
    """
    Add weather features from Open-Meteo historical data via cached CSV.
    Temperature, precipitation, and wind speed at the home team's stadium.
    """
    from api.weather import build_weather_cache, CACHE_PATH

    # Build/update weather cache
    weather_df = build_weather_cache(df)

    if weather_df.empty:
        for col in WEATHER_FEATURES:
            df[col] = np.nan
        return df

    df = df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date

    # Build lookup
    weather_lookup = {}
    for _, row in weather_df.iterrows():
        key = (row["date"], row["home_team"])
        weather_lookup[key] = {
            "Match_Temperature": row["temperature"],
            "Match_Precipitation": row["precipitation"],
            "Match_WindSpeed": row["wind_speed"],
        }

    # Merge
    temps, precips, winds = [], [], []
    for _, row in df.iterrows():
        match_date = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
        key = (match_date, row["Home_Team"])
        w = weather_lookup.get(key, {})
        temps.append(w.get("Match_Temperature", np.nan))
        precips.append(w.get("Match_Precipitation", np.nan))
        winds.append(w.get("Match_WindSpeed", np.nan))

    df["Match_Temperature"] = temps
    df["Match_Precipitation"] = precips
    df["Match_WindSpeed"] = winds

    matched = sum(1 for t in temps if pd.notna(t))
    print(f"  Weather features merged: {matched}/{len(df)} matches")
    return df


def _compute_prior_season_proximities(df):
    """
    Pre-compute final-season proximity values for each team in each season.
    Used to carry over context into the next season's early matches.
    Returns: {season_idx: {team: {relprox, titleprox, europrox}}}
    """
    DENOM = float(MAX_POINTS)  # games_per_season * 3, keeps scale aligned with max_remaining

    prior_proxim = {}
    for season_idx, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        points = {}
        gd_map = {}

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
        rel_idx = RELEGATION_POS - 1 if RELEGATION_POS else len(pts_list) - 3
        euro_idx = (EURO_POS - 1) if EURO_POS else 6  # default 7th if no euro
        pts_1st = pts_list[0] if len(pts_list) >= 1 else 0
        pts_euro = pts_list[euro_idx] if len(pts_list) > euro_idx else 0
        pts_rel = pts_list[rel_idx] if len(pts_list) > rel_idx else 0

        team_proxim = {}
        for t in standings:
            team_proxim[t] = {
                "relprox": (points[t] - pts_rel) / DENOM,
                "titleprox": (pts_1st - points[t]) / DENOM,
                "europrox": (points[t] - pts_euro) / DENOM,
            }

        # Default for promoted teams entering NEXT season
        team_proxim["__promoted_default__"] = {
            "relprox": 0.0,  # right at the relegation line
            "titleprox": (pts_1st - pts_rel) / DENOM,
            "europrox": (pts_rel - pts_euro) / DENOM,
        }

        prior_proxim[season_idx] = team_proxim

    return prior_proxim


def add_match_context_features(df):
    """
    Compute match context features from running standings tables with
    previous-season carry-over.

    For each match, BEFORE applying its result, compute current-season
    proximity values, then blend with prior season's final proximity:
      blended = (1 - Season_Progress) * prior + Season_Progress * current

    This gives early-season matches meaningful context signal based on
    where teams finished last year, transitioning smoothly to current-season
    standings as real results accumulate.

    Leakage-safe: features use only results from prior matches.
    """
    df = df.copy()

    # Initialize columns
    for col in CONTEXT_FEATURES:
        df[col] = 0.0

    # Pre-compute prior season final proximities
    prior_proxim = _compute_prior_season_proximities(df)
    no_prior = {"relprox": 0.0, "titleprox": 0.0, "europrox": 0.0}

    for season_idx, season_group in df.groupby("SeasonIndex"):
        season_matches = season_group.sort_values("Date")
        indices = season_matches.index.tolist()

        # Get prior season lookup (if exists)
        prev_season = prior_proxim.get(season_idx - 1, {})
        promoted_default = prev_season.get("__promoted_default__", no_prior)

        # Running standings: points, goal_difference, games_played per team
        points = {}
        gd = {}
        played = {}

        for idx in indices:
            row = df.loc[idx]
            home = row["Home_Team"]
            away = row["Away_Team"]

            # Ensure teams exist in standings (first appearance)
            for team in [home, away]:
                if team not in points:
                    points[team] = 0
                    gd[team] = 0
                    played[team] = 0

            # --- BEFORE result: compute context features from current standings ---
            all_teams = list(points.keys())
            standings = sorted(all_teams, key=lambda t: (-points[t], -gd[t]))

            pts_by_pos = [points[t] for t in standings]
            _rel_idx = RELEGATION_POS - 1 if RELEGATION_POS else len(pts_by_pos) - 3
            _euro_idx = (EURO_POS - 1) if EURO_POS else 6
            pts_1st = pts_by_pos[0] if len(pts_by_pos) >= 1 else 0
            pts_euro = pts_by_pos[_euro_idx] if len(pts_by_pos) > _euro_idx else 0
            pts_rel = pts_by_pos[_rel_idx] if len(pts_by_pos) > _rel_idx else 0

            avg_played = (played[home] + played[away]) / 2
            season_progress = avg_played / float(GAMES_PER_SEASON)
            max_remaining = max((GAMES_PER_SEASON - avg_played) * 3, 1)

            df.at[idx, "Season_Progress"] = season_progress

            # Compute current-season proximity for both teams
            for team, prefix in [(home, "Home"), (away, "Away")]:
                t_pts = points[team]
                curr_rel = (t_pts - pts_rel) / max_remaining
                curr_title = (pts_1st - t_pts) / max_remaining
                curr_euro = (t_pts - pts_euro) / max_remaining

                # Blend with prior season (if available)
                if season_idx > 0 and prev_season:
                    prior = prev_season.get(team, promoted_default)
                    w = season_progress  # 0 at start -> 100% prior; 1 at end -> 100% current
                    df.at[idx, f"{prefix}_RelegationProximity"] = (1 - w) * prior["relprox"] + w * curr_rel
                    df.at[idx, f"{prefix}_TitleProximity"] = (1 - w) * prior["titleprox"] + w * curr_title
                    df.at[idx, f"{prefix}_EuroProximity"] = (1 - w) * prior["europrox"] + w * curr_euro
                else:
                    # No prior season (season 0) — use current only
                    df.at[idx, f"{prefix}_RelegationProximity"] = curr_rel
                    df.at[idx, f"{prefix}_TitleProximity"] = curr_title
                    df.at[idx, f"{prefix}_EuroProximity"] = curr_euro

            # --- AFTER: update standings with this match's result ---
            hg = row["Home_Goals"]
            ag = row["Away_Goals"]

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


def add_derived_features(df):
    """Add features derived from existing columns."""
    # Betting odds implied probability (handle NaN for early seasons)
    df["B365_Implied_Over25"] = 1.0 / df["B365Greater2.5"]
    df["B365_Implied_Under25"] = 1.0 / df["B365LessThan2.5"]

    # League position difference (negative = home team higher in table)
    df["LeaguePosition_Diff"] = df["Home_LeaguePosition"] - df["Away_LeaguePosition"]

    # Goal difference over last 5 games
    df["Home_GoalDiff_5"] = df["Home_Past5Goals"] - df["Home_Past5Conceded"]
    df["Away_GoalDiff_5"] = df["Away_Past5Goals"] - df["Away_Past5Conceded"]

    return df


def add_xg_features(df, window=5):
    """
    Compute rolling xG features from the home_xg/away_xg columns.
    Uses long-format groupby + shift(1) + rolling to avoid data leakage.
    """
    if "home_xg" not in df.columns:
        return df

    df = df.copy()

    # Build long-format: one row per team per match
    home = df[["Date", "Home_Team", "home_xg", "away_xg", "Home_Goals", "Away_Goals"]].copy()
    home.columns = ["date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]

    away = df[["Date", "Away_Team", "away_xg", "home_xg", "Away_Goals", "Home_Goals"]].copy()
    away.columns = ["date", "team", "xg_for", "xg_against", "goals_for", "goals_against"]

    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling xG for (created)
    long["rolling_xg_for"] = (
        long.groupby("team")["xg_for"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    # Rolling xG against (conceded)
    long["rolling_xg_against"] = (
        long.groupby("team")["xg_against"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    # Overperformance: goals - xG (positive = scoring more than expected)
    long["xg_overperf"] = long["goals_for"] - long["xg_for"]
    long["rolling_xg_overperf"] = (
        long.groupby("team")["xg_overperf"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    # Total match xG (both teams combined) - key for over/under prediction
    long["match_total_xg"] = long["xg_for"] + long["xg_against"]
    long["rolling_total_xg"] = (
        long.groupby("team")["match_total_xg"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Split back to home/away and merge into main df
    home_feats = long.iloc[:len(df)][["rolling_xg_for", "rolling_xg_against",
                                       "rolling_xg_overperf", "rolling_total_xg"]].values
    away_feats = long.iloc[len(df):][["rolling_xg_for", "rolling_xg_against",
                                       "rolling_xg_overperf", "rolling_total_xg"]].values

    # Can't simply slice since sort order changed - merge on index instead
    home_long = long[long["team"].isin(df["Home_Team"].unique())].copy()
    away_long = long[long["team"].isin(df["Away_Team"].unique())].copy()

    # Safer approach: rebuild from long format with merge keys
    # Home features: match on (date, home_team)
    home_roll = long.groupby(["date", "team"]).first().reset_index()
    home_roll = home_roll.rename(columns={
        "rolling_xg_for": "Home_RollingXG_5",
        "rolling_xg_against": "Home_RollingXGAgainst_5",
        "rolling_xg_overperf": "Home_XGOverperf_5",
        "rolling_total_xg": "Home_RollingTotalXG_5",
    })

    df = df.merge(
        home_roll[["date", "team", "Home_RollingXG_5", "Home_RollingXGAgainst_5",
                    "Home_XGOverperf_5", "Home_RollingTotalXG_5"]],
        left_on=["Date", "Home_Team"],
        right_on=["date", "team"],
        how="left"
    ).drop(columns=["date", "team"], errors="ignore")

    # Away features
    away_roll = home_roll.rename(columns={
        "Home_RollingXG_5": "Away_RollingXG_5",
        "Home_RollingXGAgainst_5": "Away_RollingXGAgainst_5",
        "Home_XGOverperf_5": "Away_XGOverperf_5",
        "Home_RollingTotalXG_5": "Away_RollingTotalXG_5",
    })

    df = df.merge(
        away_roll[["date", "team", "Away_RollingXG_5", "Away_RollingXGAgainst_5",
                    "Away_XGOverperf_5", "Away_RollingTotalXG_5"]],
        left_on=["Date", "Away_Team"],
        right_on=["date", "team"],
        how="left"
    ).drop(columns=["date", "team"], errors="ignore")

    return df


def add_player_features(df):
    """
    Add player availability features from the enriched dataset columns.
    Renames raw columns to feature names expected by config.
    """
    df = df.copy()
    rename_map = {
        "home_injury_burden": "Home_InjuryBurden",
        "away_injury_burden": "Away_InjuryBurden",
        "home_key_absences": "Home_KeyAbsences",
        "away_key_absences": "Away_KeyAbsences",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df[new] = df[old]
    return df


def add_fpl_strength_features(df):
    """Add pre-season FPL team strength ratings as features.

    These are static ratings known before GW1, providing a cold-start signal
    for early-season matches when rolling features are unreliable.
    Available for seasons 19-24 (2019/20 - 2024/25). NaN for earlier seasons.
    """
    from api.fpl_team_strengths import download_team_strengths, get_match_strength_features

    df = df.copy()
    strengths = download_team_strengths(use_cache=True)
    if strengths.empty:
        return df

    # Pre-build lookup for speed
    feature_cols = ["Home_FPL_Attack", "Away_FPL_Attack", "Home_FPL_Defence",
                    "Away_FPL_Defence", "FPL_Openness", "FPL_HomeDominance"]
    for col in feature_cols:
        df[col] = float("nan")

    # Vectorised approach: merge home/away strengths
    home_str = strengths.rename(columns={
        "team": "Home_Team", "season_index": "SeasonIndex",
        "strength_attack_home": "_h_att", "strength_defence_home": "_h_def",
    })[["Home_Team", "SeasonIndex", "_h_att", "_h_def"]]

    away_str = strengths.rename(columns={
        "team": "Away_Team", "season_index": "SeasonIndex",
        "strength_attack_away": "_a_att", "strength_defence_away": "_a_def",
    })[["Away_Team", "SeasonIndex", "_a_att", "_a_def"]]

    df = df.merge(home_str, on=["Home_Team", "SeasonIndex"], how="left")
    df = df.merge(away_str, on=["Away_Team", "SeasonIndex"], how="left")

    # Compute normalised features where data exists
    def norm(val):
        return (val - 1000) / 400

    has_data = df["_h_att"].notna()
    df.loc[has_data, "Home_FPL_Attack"] = norm(df.loc[has_data, "_h_att"])
    df.loc[has_data, "Away_FPL_Attack"] = norm(df.loc[has_data, "_a_att"])
    df.loc[has_data, "Home_FPL_Defence"] = norm(df.loc[has_data, "_h_def"])
    df.loc[has_data, "Away_FPL_Defence"] = norm(df.loc[has_data, "_a_def"])
    df.loc[has_data, "FPL_Openness"] = (
        norm(df.loc[has_data, "_h_att"]) + norm(df.loc[has_data, "_a_att"])
        - norm(df.loc[has_data, "_h_def"]) - norm(df.loc[has_data, "_a_def"])
    )
    df.loc[has_data, "FPL_HomeDominance"] = (
        (norm(df.loc[has_data, "_h_att"]) + norm(df.loc[has_data, "_h_def"]))
        - (norm(df.loc[has_data, "_a_att"]) + norm(df.loc[has_data, "_a_def"]))
    )

    # Drop temp columns
    df = df.drop(columns=["_h_att", "_h_def", "_a_att", "_a_def"], errors="ignore")

    n_with = has_data.sum()
    print(f"  FPL strength features: {n_with}/{len(df)} matches have data")

    return df


def add_squad_features(df):
    """
    Add squad-level features from FPL-Core-Insights player data.
    Merges on date + team name (normalized) with ±1 day tolerance.
    Only covers seasons 24-25 and 25-26; earlier seasons get NaN (XGBoost handles natively).
    """
    from api.team_mapping import normalize as normalize_team

    squad_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "player_data", "squad_features.csv")
    if not os.path.exists(squad_path):
        print("  Warning: squad_features.csv not found. Skipping squad features.")
        for col in SQUAD_FEATURES + DEFENSIVE_TIER3_FEATURES:
            df[col] = np.nan
        return df

    sf = pd.read_csv(squad_path)

    # Normalize team names and parse dates
    sf["team_canonical"] = sf["team"].apply(normalize_team)
    sf["date"] = pd.to_datetime(sf["kickoff_time"], errors="coerce").dt.date

    # Build lookup: (date, canonical_team, side) -> features
    # Also store ±1 day for fuzzy matching
    feature_cols = ["AvailableXG", "AvailableXA", "AttackMissing",
                    "DefenceMissing", "StarAvailable", "FormXG5", "SquadDepth",
                    "GKShotStopping_5"]
    squad_lookup = {}
    for _, row in sf.iterrows():
        if pd.isna(row["date"]):
            continue
        team = row["team_canonical"]
        side = row["side"]
        feats = {c: row[c] for c in feature_cols}
        key = (row["date"], team, side)
        squad_lookup[key] = feats
        # ±1 day tolerance
        for offset in [-1, 1]:
            alt_date = row["date"] + pd.Timedelta(days=offset)
            alt_key = (alt_date.date() if hasattr(alt_date, 'date') else alt_date, team, side)
            if alt_key not in squad_lookup:
                squad_lookup[alt_key] = feats

    df = df.copy()

    # Initialize all squad feature columns
    for col in SQUAD_FEATURES + DEFENSIVE_TIER3_FEATURES:
        df[col] = np.nan

    matched = 0
    for idx, row in df.iterrows():
        if row["SeasonIndex"] not in (24, 25):
            continue

        match_date = row["Date"].date() if hasattr(row["Date"], 'date') else row["Date"]
        home_key = (match_date, row["Home_Team"], "home")
        away_key = (match_date, row["Away_Team"], "away")

        h_data = squad_lookup.get(home_key, {})
        a_data = squad_lookup.get(away_key, {})

        if h_data or a_data:
            matched += 1

        for col in feature_cols:
            if h_data:
                df.at[idx, f"Home_{col}"] = h_data.get(col, np.nan)
            if a_data:
                df.at[idx, f"Away_{col}"] = a_data.get(col, np.nan)

    # Compute interaction features
    mask = df["SeasonIndex"].isin([24, 25])
    df.loc[mask, "AvailableXG_Diff"] = (
        df.loc[mask, "Home_AvailableXG"].fillna(1) - df.loc[mask, "Away_AvailableXG"].fillna(1)
    )
    df.loc[mask, "DefenceMissing_Diff"] = (
        df.loc[mask, "Home_DefenceMissing"].fillna(0) - df.loc[mask, "Away_DefenceMissing"].fillna(0)
    )

    print(f"  Squad features merged: {matched} matches (seasons 24-25)")
    return df


def add_advanced_features(df):
    """
    Add advanced features: over/under rates, BTTS, Elo, Poisson probabilities.
    These features capture match-level goal expectation more directly.
    """
    df = df.copy()

    # --- Long format for rolling stats ---
    home = df[["Date", "Home_Team", "Home_Goals", "Away_Goals", "TG", "FTR",
               "Home_Corners", "Away_Corners",
               "Home_Shots_Target", "Away_Shots_Target",
               "Home_Shots", "Away_Shots"]].copy()
    home.columns = ["Date", "Team", "GF", "GA", "TG", "FTR",
                    "Corners_For", "Corners_Against",
                    "SOT_For", "SOT_Against", "Shots_For", "Shots_Against"]
    away = df[["Date", "Away_Team", "Away_Goals", "Home_Goals", "TG", "FTR",
               "Away_Corners", "Home_Corners",
               "Away_Shots_Target", "Home_Shots_Target",
               "Away_Shots", "Home_Shots"]].copy()
    away.columns = ["Date", "Team", "GF", "GA", "TG", "FTR",
                    "Corners_For", "Corners_Against",
                    "SOT_For", "SOT_Against", "Shots_For", "Shots_Against"]
    long = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)
    g = long.groupby("Team")

    long["Over25_5"] = g["TG"].transform(lambda x: (x.shift(1) > 2).rolling(5, min_periods=2).mean())
    long["BTTS"] = ((long["GF"] > 0) & (long["GA"] > 0)).astype(int)
    long["BTTS_5"] = g["BTTS"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["CS"] = (long["GA"] == 0).astype(int)
    long["CS_5"] = g["CS"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["TGAvg_5"] = g["TG"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["GPG_20"] = g["GF"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    long["GAPG_20"] = g["GA"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())

    # Option 3 Step 2c: short-horizon (_3) rolling windows.
    # Hypothesis: current windows are _5 / _10 / _20 / _EWM10. A _3 window
    # captures micro-trends (injury returns, tactical shifts, hot/cold runs)
    # that a _5 window smooths over. Collinearity with _5 is expected but
    # acceptable — the model can pick whichever it finds useful.
    long["Over25_3"] = g["TG"].transform(
        lambda x: (x.shift(1) > 2).rolling(3, min_periods=1).mean())
    long["BTTS_3"] = g["BTTS"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    long["TGAvg_3"] = g["TG"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    long["Past3Goals"] = g["GF"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum())

    # Corner rolling stats (Wheatcroft: corners outperform goals as inputs)
    long["CornersAvg_5"] = g["Corners_For"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["CornersConcAvg_5"] = g["Corners_Against"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    # Option 3 Step 2c: _3 corner windows
    long["CornersAvg_3"] = g["Corners_For"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())

    # Shots rolling stats for Poisson lambdas (Wheatcroft: shots are lower variance)
    long["SOT_Avg_5"] = g["SOT_For"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["SOT_Against_Avg_5"] = g["SOT_Against"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # EWM (span=10) versions: lower variance than hard 5-game window
    long["Over25_EWM10"] = g["TG"].transform(lambda x: (x.shift(1) > 2).astype(float).ewm(span=10, min_periods=3).mean())
    long["TGAvg_EWM10"] = g["TG"].transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
    long["BTTS_EWM10"] = g["BTTS"].transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
    long["GPG_EWM10"] = g["GF"].transform(lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())

    # Option 3 Step 2a: Corner efficiency = rolling goals / rolling corners.
    # Hypothesis: teams converting set-piece pressure into goals at above-
    # league-average rate sustain higher goal expectancy in tight games.
    # Computed as SUM ratio (rather than MEAN/MEAN) for numerical clarity;
    # guarded against teams with near-zero corners in the window.
    goals_sum_5 = g["GF"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).sum())
    corners_sum_5 = g["Corners_For"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).sum())
    long["CornerEfficiency_5"] = goals_sum_5 / (corners_sum_5 + 1e-6)
    # If a team won fewer than 1 corner in the whole window, mark NaN
    # (otherwise we'd get inflated ratios from a single lucky corner).
    long.loc[corners_sum_5 < 1.0, "CornerEfficiency_5"] = np.nan

    goals_sum_10 = g["GF"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    corners_sum_10 = g["Corners_For"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).sum())
    long["CornerEfficiency_10"] = goals_sum_10 / (corners_sum_10 + 1e-6)
    long.loc[corners_sum_10 < 1.0, "CornerEfficiency_10"] = np.nan

    new_cols = ["Over25_5", "BTTS_5", "CS_5", "TGAvg_5", "GPG_20", "GAPG_20",
                "CornersAvg_5", "CornersConcAvg_5", "SOT_Avg_5", "SOT_Against_Avg_5",
                "Over25_EWM10", "TGAvg_EWM10", "BTTS_EWM10", "GPG_EWM10",
                "CornerEfficiency_5", "CornerEfficiency_10",
                # Option 3 Step 2c: short-horizon _3 windows
                "Over25_3", "BTTS_3", "TGAvg_3", "Past3Goals", "CornersAvg_3"]
    feat_long = long[["Date", "Team"] + new_cols].drop_duplicates(["Date", "Team"])

    for prefix in ["Home", "Away"]:
        renamed = feat_long.rename(columns={c: f"{prefix}_{c}" for c in new_cols})
        df = df.merge(renamed, left_on=["Date", f"{prefix}_Team"], right_on=["Date", "Team"],
                      how="left").drop(columns=["Team"], errors="ignore")

    # --- Elo ratings ---
    K = 20
    HOME_ADV = 50
    teams_elo = {}
    elo_diffs = []
    for _, row in df.sort_values("Date").iterrows():
        h, a = row["Home_Team"], row["Away_Team"]
        he = teams_elo.get(h, 1500)
        ae = teams_elo.get(a, 1500)
        elo_diffs.append(he - ae + HOME_ADV)
        exp_h = 1 / (1 + 10 ** ((ae - he - HOME_ADV) / 400))
        act = 1 if row["FTR"] == "H" else (0 if row["FTR"] == "A" else 0.5)
        teams_elo[h] = he + K * (act - exp_h)
        teams_elo[a] = ae + K * ((1 - act) - (1 - exp_h))
    df["Elo_Diff"] = elo_diffs

    # --- Poisson features ---
    # xG-based lambdas (additive average)
    home_xg = df.get("Home_RollingXG_5", pd.Series(dtype=float))
    away_xg = df.get("Away_RollingXG_5", pd.Series(dtype=float))
    home_xga = df.get("Home_RollingXGAgainst_5", pd.Series(dtype=float))
    away_xga = df.get("Away_RollingXGAgainst_5", pd.Series(dtype=float))
    df["Home_Lambda_xG"] = (home_xg + away_xga) / 2
    df["Away_Lambda_xG"] = (away_xg + home_xga) / 2
    df["Expected_TG_xG"] = df["Home_Lambda_xG"] + df["Away_Lambda_xG"]

    # Goals-based lambdas (additive average)
    df["Home_Lambda_Goals"] = df["Home_Past5Goals"] / 5
    df["Away_Lambda_Goals"] = df["Away_Past5Goals"] / 5
    df["Expected_TG_Goals"] = df["Home_Lambda_Goals"] + df["Away_Lambda_Goals"]

    # Dixon-Coles multiplicative lambdas: attack * opp_defence_weakness * home_factor
    # league_avg_goals = mean goals per team per match across the dataset
    league_avg_gpg = df["Home_Goals"].mean()  # ~1.5 goals per home team
    # attack_strength = team_gpg / league_avg, defence_weakness = team_ga / league_avg
    home_attack = df["Home_GPG_20"] / league_avg_gpg
    away_defence_w = df["Away_GAPG_20"] / league_avg_gpg
    away_attack = df["Away_GPG_20"] / df["Away_Goals"].mean()
    home_defence_w = df["Home_GAPG_20"] / df["Away_Goals"].mean()
    HOME_FACTOR = df["Home_Goals"].mean() / df["Away_Goals"].mean()  # ~1.35
    df["Home_Lambda_DC"] = home_attack * away_defence_w * league_avg_gpg * HOME_FACTOR
    df["Away_Lambda_DC"] = away_attack * home_defence_w * df["Away_Goals"].mean()
    df["Expected_TG_DC"] = df["Home_Lambda_DC"] + df["Away_Lambda_DC"]

    # Dixon-Coles tau correction for low-score dependency (0-0, 1-0, 0-1, 1-1)
    def _dc_tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _poisson_over25(hl, al):
        if pd.isna(hl) or pd.isna(al):
            return np.nan
        p = sum(poisson.pmf(h, max(hl, 0.01)) * poisson.pmf(a, max(al, 0.01))
                for h in range(12) for a in range(12) if h + a <= 2)
        return 1 - p

    def _poisson_over25_dc(hl, al, rho=-0.13):
        """Poisson P(over 2.5) with Dixon-Coles low-score correction."""
        if pd.isna(hl) or pd.isna(al):
            return np.nan
        hl, al = max(hl, 0.01), max(al, 0.01)
        p_under = 0
        for h in range(12):
            for a in range(12):
                if h + a <= 2:
                    p = poisson.pmf(h, hl) * poisson.pmf(a, al) * _dc_tau(h, a, hl, al, rho)
                    p_under += max(p, 0)  # clamp to avoid negative from tau
        return 1 - p_under

    df["Poisson_xG"] = [_poisson_over25(h, a) for h, a in zip(df["Home_Lambda_xG"], df["Away_Lambda_xG"])]
    df["Poisson_Goals"] = [_poisson_over25(h, a) for h, a in zip(df["Home_Lambda_Goals"], df["Away_Lambda_Goals"])]
    df["Poisson_DC"] = [_poisson_over25_dc(h, a) for h, a in zip(df["Home_Lambda_DC"], df["Away_Lambda_DC"])]

    # --- Shots-based Poisson (Wheatcroft: shots on target are lower variance than goals) ---
    # Convert SOT averages to expected goals using league-average conversion rate
    # League average SOT conversion ~0.30 (roughly 30% of shots on target become goals)
    sot_conv = df["Home_Goals"].sum() / df["Home_Shots_Target"].sum() if df["Home_Shots_Target"].sum() > 0 else 0.30
    df["Home_Lambda_Shots"] = df.get("Home_SOT_Avg_5", np.nan) * sot_conv
    df["Away_Lambda_Shots"] = df.get("Away_SOT_Avg_5", np.nan) * sot_conv
    df["Expected_TG_Shots"] = df["Home_Lambda_Shots"] + df["Away_Lambda_Shots"]
    df["Poisson_Shots"] = [_poisson_over25(h, a) for h, a in
                           zip(df["Home_Lambda_Shots"], df["Away_Lambda_Shots"])]

    # Consensus features: average across Poisson/Expected_TG variants to reduce noise
    # Poisson_DC excluded from consensus — real DC signal enters via stacker
    df["Poisson_Consensus"] = df[["Poisson_xG", "Poisson_Goals", "Poisson_Shots"]].mean(axis=1)
    df["Expected_TG_Consensus"] = df[["Expected_TG_xG", "Expected_TG_Goals", "Expected_TG_DC", "Expected_TG_Shots"]].mean(axis=1)

    # Corner dominance feature
    df["Corner_Dominance"] = (
        (df.get("Home_CornersAvg_5", 0) - df.get("Home_CornersConcAvg_5", 0)) +
        (df.get("Away_CornersAvg_5", 0) - df.get("Away_CornersConcAvg_5", 0))
    )

    # Combination features
    df["Combined_Over25"] = (df.get("Home_Over25_5", 0.5) + df.get("Away_Over25_5", 0.5)) / 2
    df["Combined_BTTS"] = (df.get("Home_BTTS_5", 0.5) + df.get("Away_BTTS_5", 0.5)) / 2
    df["Attack_Power"] = df.get("Home_GPG_20", 1.0) + df.get("Away_GPG_20", 1.0)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BTTS-specific features
# ═══════════════════════════════════════════════════════════════════════════════

def _poisson_btts(hl, al):
    """Poisson P(BTTS) via inclusion-exclusion: 1 - P(H=0) - P(A=0) + P(0-0).

    Much faster than the 12x12 loop since under independent Poisson:
      P(Home=0) = e^{-hl}, P(Away=0) = e^{-al}
    """
    if pd.isna(hl) or pd.isna(al):
        return np.nan
    hl, al = max(hl, 0.01), max(al, 0.01)
    p_h0 = poisson.pmf(0, hl)  # P(Home scores 0)
    p_a0 = poisson.pmf(0, al)  # P(Away scores 0)
    return 1 - p_h0 - p_a0 + p_h0 * p_a0


def add_btts_features(df):
    """Add BTTS-specific features: failed-to-score rates, CS streaks,
    half-time scoring, Poisson BTTS, and combination features.

    These features target the question 'will BOTH teams score?' rather than
    'how many total goals?' -- fundamentally different from Over/Under features.
    """
    df = df.copy()

    # --- Long format for per-team rolling stats ---
    # Include half-time goals for first-half scoring features
    has_ht = "HTHG" in df.columns and "HTAG" in df.columns

    home_cols = ["Date", "Home_Team", "Home_Goals", "Away_Goals"]
    away_cols = ["Date", "Away_Team", "Away_Goals", "Home_Goals"]
    home_names = ["Date", "Team", "GF", "GA"]
    away_names = ["Date", "Team", "GF", "GA"]

    if has_ht:
        home_cols += ["HTHG", "HTAG"]
        away_cols += ["HTAG", "HTHG"]
        home_names += ["HT_GF", "HT_GA"]
        away_names += ["HT_GF", "HT_GA"]

    home = df[home_cols].copy()
    home.columns = home_names
    away = df[away_cols].copy()
    away.columns = away_names

    long = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)
    g = long.groupby("Team")

    # Failed to score (blanking rate)
    long["FTS"] = (long["GF"] == 0).astype(int)
    long["FTS_5"] = g["FTS"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["FTS_10"] = g["FTS"].transform(lambda x: x.shift(1).rolling(10, min_periods=4).mean())
    long["ScoredAtLeast1_5"] = 1 - long["FTS_5"]

    # Scoring consistency (std dev of goals scored)
    long["GoalStd_10"] = g["GF"].transform(lambda x: x.shift(1).rolling(10, min_periods=4).std())

    # BTTS rate over 10 games (longer window than existing BTTS_5)
    long["BTTS_flag"] = ((long["GF"] > 0) & (long["GA"] > 0)).astype(int)
    long["BTTS_10"] = g["BTTS_flag"].transform(lambda x: x.shift(1).rolling(10, min_periods=4).mean())

    # Clean sheet streak (consecutive games with GA == 0)
    def _cs_streak(ga_series):
        streaks = []
        current = 0
        for ga in ga_series:
            streaks.append(current)  # streak BEFORE this match (no leakage)
            if ga == 0:
                current += 1
            else:
                current = 0
        return pd.Series(streaks, index=ga_series.index)

    long["CSStreak"] = g["GA"].transform(_cs_streak)

    # Half-time scoring tendency
    if has_ht:
        long["HT_Scored"] = (long["HT_GF"] > 0).astype(float)
        long["HT_Conceded"] = (long["HT_GA"] > 0).astype(float)
        # Handle NaN in half-time data (some older matches may lack it)
        long.loc[long["HT_GF"].isna(), "HT_Scored"] = np.nan
        long.loc[long["HT_GA"].isna(), "HT_Conceded"] = np.nan
        long["HT_Scored_5"] = g["HT_Scored"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        long["HT_Conceded_5"] = g["HT_Conceded"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # --- Merge back to wide format ---
    btts_cols = ["FTS_5", "FTS_10", "ScoredAtLeast1_5", "GoalStd_10",
                 "BTTS_10", "CSStreak"]
    if has_ht:
        btts_cols += ["HT_Scored_5", "HT_Conceded_5"]

    feat_long = long[["Date", "Team"] + btts_cols].drop_duplicates(["Date", "Team"])

    for prefix in ["Home", "Away"]:
        renamed = feat_long.rename(columns={c: f"{prefix}_{c}" for c in btts_cols})
        df = df.merge(renamed, left_on=["Date", f"{prefix}_Team"],
                      right_on=["Date", "Team"], how="left").drop(columns=["Team"], errors="ignore")

    # --- Combination / interaction features (match-level) ---
    df["Combined_FTS"] = (df.get("Home_FTS_5", 0.2) + df.get("Away_FTS_5", 0.2)) / 2
    df["Blanking_Risk"] = 1 - (1 - df.get("Home_FTS_5", 0.2)) * (1 - df.get("Away_FTS_5", 0.2))
    df["CS_Risk"] = (df.get("Home_CS_5", 0.3) + df.get("Away_CS_5", 0.3)) / 2
    df["BTTS_Attack_Power"] = df.get("Home_GPG_20", 1.0) * df.get("Away_GPG_20", 1.0)

    # Poisson BTTS from goal-based lambdas
    if "Home_Lambda_Goals" in df.columns and "Away_Lambda_Goals" in df.columns:
        df["Poisson_BTTS"] = [_poisson_btts(h, a) for h, a in
                              zip(df["Home_Lambda_Goals"], df["Away_Lambda_Goals"])]
    else:
        # Fallback: compute from Past5Goals
        df["Poisson_BTTS"] = [_poisson_btts(h/5, a/5) for h, a in
                              zip(df["Home_Past5Goals"].fillna(7.5),
                                  df["Away_Past5Goals"].fillna(5.0))]

    # Poisson BTTS from xG lambdas
    if "Home_Lambda_xG" in df.columns and "Away_Lambda_xG" in df.columns:
        df["Poisson_BTTS_xG"] = [_poisson_btts(h, a) for h, a in
                                  zip(df["Home_Lambda_xG"], df["Away_Lambda_xG"])]
    else:
        df["Poisson_BTTS_xG"] = np.nan

    # Consensus
    df["Poisson_BTTS_Consensus"] = df[["Poisson_BTTS", "Poisson_BTTS_xG"]].mean(axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Corners-specific features (for Corners O/U 10.5 model)
# ═══════════════════════════════════════════════════════════════════════════════

def _poisson_over105_corners(lambda_h: float, lambda_a: float) -> float:
    """P(total corners > 10.5) assuming independent Poisson for each team's corners.

    Args:
        lambda_h: Expected home team corners.
        lambda_a: Expected away team corners.

    Returns:
        Probability of over 10.5 total corners.
    """
    if pd.isna(lambda_h) or pd.isna(lambda_a):
        return np.nan
    lambda_h = max(lambda_h, 0.01)
    lambda_a = max(lambda_a, 0.01)
    total_lambda = lambda_h + lambda_a
    # P(X <= 10) where X ~ Poisson(total_lambda)
    return 1.0 - poisson.cdf(10, total_lambda)


def add_corner_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add corners-specific features for the Corners O/U 10.5 model.

    Computes per-team rolling corner stats at multiple windows,
    corner volatility, over 10.5 rates, EWM versions, and Poisson estimates.
    """
    df = df.copy()

    # --- Long format for per-team rolling stats ---
    home = df[["Date", "Home_Team", "Home_Corners", "Away_Corners"]].copy()
    home.columns = ["Date", "Team", "Corners_For", "Corners_Against"]
    away = df[["Date", "Away_Team", "Away_Corners", "Home_Corners"]].copy()
    away.columns = ["Date", "Team", "Corners_For", "Corners_Against"]

    long = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)
    long["Total_Corners"] = long["Corners_For"] + long["Corners_Against"]
    long["Over105"] = (long["Total_Corners"] > 10.5).astype(int)
    long["Corner_Diff"] = long["Corners_For"] - long["Corners_Against"]
    g = long.groupby("Team")

    # 10-game rolling windows (more stable for corners — higher variance than goals)
    long["CornersAvg_10"] = g["Corners_For"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=4).mean())
    long["CornersConcAvg_10"] = g["Corners_Against"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=4).mean())
    long["TotalCorners_10"] = g["Total_Corners"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=4).mean())
    long["CornersStd_10"] = g["Corners_For"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=4).std())

    # 20-game total corners (long-term baseline)
    long["TotalCorners_20"] = g["Total_Corners"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=8).mean())

    # Corner differential (5-game)
    long["CornerDiff_5"] = g["Corner_Diff"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())

    # EWM versions (span=10)
    long["CornersEWM_10"] = g["Corners_For"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())
    long["CornersConcEWM_10"] = g["Corners_Against"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=3).mean())

    # Over 10.5 rate at 5 and 10 game windows
    long["Over105_5"] = g["Over105"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    long["Over105_10"] = g["Over105"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=4).mean())

    # --- Merge back to wide format ---
    corner_cols = [
        "CornersAvg_10", "CornersConcAvg_10", "TotalCorners_10", "CornersStd_10",
        "TotalCorners_20", "CornerDiff_5",
        "CornersEWM_10", "CornersConcEWM_10",
        "Over105_5", "Over105_10",
    ]
    feat_long = long[["Date", "Team"] + corner_cols].drop_duplicates(["Date", "Team"])

    for prefix in ["Home", "Away"]:
        renamed = feat_long.rename(columns={c: f"{prefix}_{c}" for c in corner_cols})
        df = df.merge(renamed, left_on=["Date", f"{prefix}_Team"],
                      right_on=["Date", "Team"], how="left").drop(columns=["Team"], errors="ignore")

    # --- Match-level combination features ---
    df["Combined_TotalCorners"] = (
        df.get("Home_TotalCorners_10", 10.8) + df.get("Away_TotalCorners_10", 10.8)
    ) / 2

    # Poisson P(Over 10.5) from rolling corner averages
    # lambda_home = Home_CornersAvg_10 (expected corners won by home team)
    # lambda_away = Away_CornersAvg_10 (expected corners won by away team)
    # total_lambda = lambda_home + lambda_away
    df["Corner_Poisson_Over105"] = [
        _poisson_over105_corners(h, a)
        for h, a in zip(
            df.get("Home_CornersAvg_10", pd.Series([5.4] * len(df))),
            df.get("Away_CornersAvg_10", pd.Series([5.4] * len(df))),
        )
    ]

    n_filled = df["Home_CornersAvg_10"].notna().sum()
    print(f"  Corner features: {n_filled}/{len(df)} matches have data")

    return df


def add_detailed_match_features(df, ewm_span=8):
    """
    Add tactical features from matches2425.csv (possession, big chances,
    touches in box, set piece xG). Only available for 2024-25 season.
    Uses EWM rolling averages to smooth signal.
    """
    if not os.path.exists(MATCHES_2425_PATH):
        print("  matches2425.csv not found — skipping detailed match features")
        return df

    from api.team_mapping import normalize
    m = pd.read_csv(MATCHES_2425_PATH)
    m["kickoff_time"] = pd.to_datetime(m["kickoff_time"], format="mixed", dayfirst=True)

    # Normalize team names to match enriched dataset
    m["home_team"] = m["home_team"].apply(normalize)
    m["away_team"] = m["away_team"].apply(normalize)

    # Build long format: one row per team per match
    home = m[["kickoff_time", "home_team", "home_possession", "home_big_chances",
              "home_touches_in_opposition_box", "home_xg_set_play"]].copy()
    home.columns = ["date", "team", "possession", "big_chances", "touches_in_box", "xg_set_play"]

    away = m[["kickoff_time", "away_team", "away_possession", "away_big_chances",
              "away_touches_in_opposition_box", "away_xg_set_play"]].copy()
    away.columns = ["date", "team", "possession", "big_chances", "touches_in_box", "xg_set_play"]

    long = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
    g = long.groupby("team")

    raw_cols = ["possession", "big_chances", "touches_in_box", "xg_set_play"]
    feat_names = ["PossessionAvg", "BigChancesAvg", "TouchesInBoxAvg", "xGSetPlayAvg"]

    for raw, feat in zip(raw_cols, feat_names):
        long[f"{feat}_8"] = g[raw].transform(
            lambda x: x.shift(1).ewm(span=ewm_span, min_periods=3).mean()
        )

    rolling_cols = [f"{f}_8" for f in feat_names]
    feats = long[["date", "team"] + rolling_cols].drop_duplicates(["date", "team"])

    df = df.copy()
    for prefix in ["Home", "Away"]:
        renamed = feats.rename(columns={c: f"{prefix}_{c}" for c in rolling_cols})
        df = df.merge(
            renamed[["date", "team"] + [f"{prefix}_{c}" for c in rolling_cols]],
            left_on=["Date", f"{prefix}_Team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["date", "team"], errors="ignore")

    n_filled = df["Home_PossessionAvg_8"].notna().sum()
    print(f"  Detailed match features: {n_filled}/{len(df)} matches have data")
    return df


def add_shot_level_features(df, ewm_span=8):
    """
    Add rolling shot-level features from Understat per-shot data.
    Extracts process-level signals from 114k+ shots: shot quality, volume,
    chance creation style, timing patterns, and defensive indicators.

    Joins via understat_match_id (from understat_xg.csv merge).
    Returns df with new features (NaN for seasons without shot data).
    """
    shots_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_shots.csv")
    if not os.path.exists(shots_path):
        print("  Shot data not found — skipping shot-level features")
        return df

    shots = pd.read_csv(shots_path)
    if shots.empty:
        return df

    # Load match mapping: understat_match_id -> (date, home_team, away_team)
    xg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_xg.csv")
    if not os.path.exists(xg_path):
        print("  Understat xG data not found — skipping shot-level features")
        return df

    xg_df = pd.read_csv(xg_path)
    xg_df["date"] = pd.to_datetime(xg_df["date"]).dt.normalize()  # strip time component for merge
    match_map = xg_df.set_index("understat_match_id")[["date", "home_team", "away_team"]].to_dict("index")

    # Compute per-match per-team shot aggregates
    shots["match_id"] = shots["match_id"].astype(str)
    shot_agg_rows = []

    for mid, group in shots.groupby("match_id"):
        mid_int = int(mid) if mid.isdigit() else mid
        if mid_int not in match_map:
            continue

        info = match_map[mid_int]
        match_date = info["date"]

        for side, team_name in [("h", info["home_team"]), ("a", info["away_team"])]:
            team_shots = group[group["side"] == side]
            n_shots = len(team_shots)
            if n_shots == 0:
                continue

            total_xg = team_shots["xG"].sum()
            xg_per_shot = total_xg / n_shots
            big_chance_rate = (team_shots["xG"] > 0.3).mean()

            # --- Shot volume ---
            n_on_target = team_shots["result"].isin(["Goal", "SavedShot", "ShotOnPost"]).sum()
            sot_rate = n_on_target / n_shots  # shots on target %

            # --- Distance to goal centre: goal at X=1.0, Y=0.5 on 0-1 pitch ---
            team_shots_xy = team_shots[["X", "Y"]].copy()
            dist = np.sqrt((1.0 - team_shots_xy["X"])**2 + (0.5 - team_shots_xy["Y"])**2)
            avg_dist = dist.mean()

            # --- Situation breakdown ---
            set_piece_mask = team_shots["situation"].isin(["SetPiece", "FromCorner", "DirectFreekick"])
            open_play_mask = team_shots["situation"] == "OpenPlay"
            corner_mask = team_shots["situation"] == "FromCorner"

            set_piece_xg = team_shots.loc[set_piece_mask, "xG"].sum()
            open_play_xg = team_shots.loc[open_play_mask, "xG"].sum()
            corner_xg = team_shots.loc[corner_mask, "xG"].sum()
            set_piece_shot_share = set_piece_mask.sum() / n_shots  # set piece dependency

            # --- Shot type: headed shot rate (aerial threat / crossing style) ---
            headed_rate = (team_shots["shotType"] == "Head").sum() / n_shots

            # --- lastAction distribution: playing style indicators ---
            n_from_cross = (team_shots["lastAction"] == "Cross").sum()
            n_from_throughball = (team_shots["lastAction"] == "Throughball").sum()
            n_from_takeon = (team_shots["lastAction"] == "TakeOn").sum()
            n_from_rebound = (team_shots["lastAction"] == "Rebound").sum()
            cross_rate = n_from_cross / n_shots          # crossing/wing play
            throughball_rate = n_from_throughball / n_shots  # direct/fast play
            takeon_rate = n_from_takeon / n_shots         # individual skill
            rebound_rate = n_from_rebound / n_shots       # second balls / chaos

            # --- Shot timing: early vs late goal threat ---
            early_shots = team_shots[team_shots["minute"] <= 30]
            late_shots = team_shots[team_shots["minute"] >= 75]
            early_xg = early_shots["xG"].sum() if len(early_shots) > 0 else 0.0
            late_xg = late_shots["xG"].sum() if len(late_shots) > 0 else 0.0
            late_xg_share = late_xg / total_xg if total_xg > 0 else 0.0

            # --- Blocked shot rate: opponent defensive organisation ---
            blocked_rate = (team_shots["result"] == "BlockedShot").sum() / n_shots

            # --- xG variance: consistency of chance creation ---
            xg_std = team_shots["xG"].std() if n_shots > 1 else 0.0

            shot_agg_rows.append({
                "date": match_date,
                "team": team_name,
                # Original features
                "xg_per_shot": xg_per_shot,
                "big_chance_rate": big_chance_rate,
                "set_piece_xg": set_piece_xg,
                "open_play_xg": open_play_xg,
                "avg_shot_dist": avg_dist,
                "corner_xg": corner_xg,
                # New: volume & accuracy
                "shot_volume": n_shots,
                "sot_rate": sot_rate,
                # New: style indicators
                "set_piece_share": set_piece_shot_share,
                "headed_rate": headed_rate,
                "cross_rate": cross_rate,
                "throughball_rate": throughball_rate,
                "takeon_rate": takeon_rate,
                "rebound_rate": rebound_rate,
                # New: timing
                "late_xg_share": late_xg_share,
                # New: defensive / variance
                "blocked_rate": blocked_rate,
                "xg_std": xg_std,
            })

    if not shot_agg_rows:
        print("  No shot aggregates computed — skipping")
        return df

    shot_agg = pd.DataFrame(shot_agg_rows)
    shot_agg = shot_agg.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute rolling EWM features (shift to avoid leakage)
    g = shot_agg.groupby("team")
    raw_cols = [
        "xg_per_shot", "big_chance_rate", "set_piece_xg", "open_play_xg",
        "avg_shot_dist", "corner_xg",
        "shot_volume", "sot_rate",
        "set_piece_share", "headed_rate",
        "cross_rate", "throughball_rate", "takeon_rate", "rebound_rate",
        "late_xg_share",
        "blocked_rate", "xg_std",
    ]
    feat_names = [
        "xGPerShot", "BigChanceRate", "SetPieceXG", "OpenPlayXG",
        "AvgShotDist", "CornerXG",
        "ShotVolume", "SOTRate",
        "SetPieceShare", "HeadedRate",
        "CrossRate", "ThroughballRate", "TakeOnRate", "ReboundRate",
        "LateXGShare",
        "BlockedRate", "xGStd",
    ]

    for raw, feat in zip(raw_cols, feat_names):
        shot_agg[f"{feat}_8"] = g[raw].transform(
            lambda x: x.shift(1).ewm(span=ewm_span, min_periods=3).mean()
        )

    # Deduplicate (one row per date+team)
    rolling_cols = [f"{f}_8" for f in feat_names]
    shot_feats = shot_agg[["date", "team"] + rolling_cols].drop_duplicates(["date", "team"])

    # Merge into main df for home and away
    df = df.copy()
    for prefix in ["Home", "Away"]:
        renamed = shot_feats.rename(columns={c: f"{prefix}_{c}" for c in rolling_cols})
        df = df.merge(
            renamed[["date", "team"] + [f"{prefix}_{c}" for c in rolling_cols]],
            left_on=["Date", f"{prefix}_Team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["date", "team"], errors="ignore")

    n_filled = df[[f"Home_{rolling_cols[0]}"]].notna().sum().iloc[0]
    print(f"  Shot-level features: {n_filled}/{len(df)} matches have data")

    # Option 3 Step 2b: Set-play xG ratio = SetPieceXG_8 / (SetPieceXG_8 + OpenPlayXG_8).
    # Hypothesis: teams relying heavily on set pieces for xG have narrower
    # goal distributions (low open-play creativity floor) — useful discriminator
    # for O/U markets. Guard against division-by-zero; ratio in [0, 1].
    for prefix in ["Home", "Away"]:
        sp_col = f"{prefix}_SetPieceXG_8"
        op_col = f"{prefix}_OpenPlayXG_8"
        ratio_col = f"{prefix}_SetPieceXG_Ratio_8"
        if sp_col in df.columns and op_col in df.columns:
            denom = df[sp_col] + df[op_col]
            df[ratio_col] = df[sp_col] / (denom + 1e-6)
            # Mark NaN when total xG is vanishing — the ratio is meaningless
            df.loc[denom < 0.05, ratio_col] = float("nan")
        else:
            df[ratio_col] = float("nan")

    return df


def add_roster_features(df, ewm_span=8):
    """Add rolling team-level features from Understat match roster data.

    Features capture chance creation quality (xGChain, xGBuildup, key_passes, xA)
    which measure how teams build attacks — fundamentally different from shot outcomes.

    Returns df with new features (NaN for matches without roster data).
    """
    roster_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_rosters.csv")
    if not os.path.exists(roster_path):
        print("  Roster data not found — skipping roster features")
        return df

    rosters = pd.read_csv(roster_path)
    if rosters.empty:
        return df

    # Load match mapping
    xg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_xg.csv")
    if not os.path.exists(xg_path):
        print("  Understat xG data not found — skipping roster features")
        return df

    xg_df = pd.read_csv(xg_path)
    xg_df["date"] = pd.to_datetime(xg_df["date"]).dt.normalize()
    match_map = xg_df.set_index("understat_match_id")[["date", "home_team", "away_team"]].to_dict("index")

    rosters["match_id"] = rosters["match_id"].astype(str)
    agg_rows = []

    for mid, group in rosters.groupby("match_id"):
        mid_int = int(mid) if mid.isdigit() else mid
        if mid_int not in match_map:
            continue
        info = match_map[mid_int]
        match_date = info["date"]

        for side, team_name in [("h", info["home_team"]), ("a", info["away_team"])]:
            row = group[group["side"] == side]
            if row.empty:
                continue
            r = row.iloc[0]
            roster_shots = max(r.get("roster_shots", 1), 1)

            agg_rows.append({
                "date": match_date,
                "team": team_name,
                "xgchain": float(r.get("xGChain", 0)),
                "xgbuildup": float(r.get("xGBuildup", 0)),
                "key_passes": int(r.get("key_passes", 0)),
                "xa": float(r.get("xA", 0)),
                # Derived: buildup quality = xGBuildup / xGChain (how much value from buildup vs final ball)
                "buildup_ratio": float(r.get("xGBuildup", 0)) / max(float(r.get("xGChain", 0.01)), 0.01),
                # Derived: chance creation efficiency = xA / key_passes
                "xa_per_kp": float(r.get("xA", 0)) / max(int(r.get("key_passes", 1)), 1),
            })

    if not agg_rows:
        print("  No roster aggregates computed — skipping")
        return df

    roster_agg = pd.DataFrame(agg_rows)
    roster_agg = roster_agg.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute rolling EWM features
    g = roster_agg.groupby("team")
    raw_cols = ["xgchain", "xgbuildup", "key_passes", "xa", "buildup_ratio", "xa_per_kp"]
    feat_names = ["xGChain", "xGBuildup", "KeyPasses", "xA", "BuildupRatio", "xAPerKP"]

    for raw, feat in zip(raw_cols, feat_names):
        roster_agg[f"{feat}_8"] = g[raw].transform(
            lambda x: x.shift(1).ewm(span=ewm_span, min_periods=3).mean()
        )

    rolling_cols = [f"{f}_8" for f in feat_names]
    roster_feats = roster_agg[["date", "team"] + rolling_cols].drop_duplicates(["date", "team"])

    df = df.copy()
    for prefix in ["Home", "Away"]:
        renamed = roster_feats.rename(columns={c: f"{prefix}_{c}" for c in rolling_cols})
        df = df.merge(
            renamed[["date", "team"] + [f"{prefix}_{c}" for c in rolling_cols]],
            left_on=["Date", f"{prefix}_Team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["date", "team"], errors="ignore")

    n_filled = df[[f"Home_{rolling_cols[0]}"]].notna().sum().iloc[0]
    print(f"  Roster features: {n_filled}/{len(df)} matches have data")
    return df


def add_tactical_features(df, ewm_span=8):
    """Add rolling tactical features from Understat match_info data.

    Features: PPDA (pressing intensity), deep completions (passes near goal).
    PPDA measures how aggressively a team presses — high-press teams create
    more chaotic, higher-scoring games. Deep completions measure attacking
    penetration near the box.

    Returns df with new features (NaN for matches without data).
    """
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_match_info.csv")
    if not os.path.exists(info_path):
        print("  Match info data not found — skipping tactical features")
        return df

    match_info = pd.read_csv(info_path)
    if match_info.empty:
        return df

    xg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "understat_xg.csv")
    if not os.path.exists(xg_path):
        print("  Understat xG data not found — skipping tactical features")
        return df

    xg_df = pd.read_csv(xg_path)
    xg_df["date"] = pd.to_datetime(xg_df["date"]).dt.normalize()
    match_map = xg_df.set_index("understat_match_id")[["date", "home_team", "away_team"]].to_dict("index")

    match_info["match_id"] = match_info["match_id"].astype(str)
    agg_rows = []

    for _, row in match_info.iterrows():
        mid = str(row["match_id"])
        mid_int = int(mid) if mid.isdigit() else mid
        if mid_int not in match_map:
            continue
        info = match_map[mid_int]
        match_date = info["date"]

        # Home team stats
        agg_rows.append({
            "date": match_date,
            "team": info["home_team"],
            "ppda": float(row.get("h_ppda", 0)),             # own pressing intensity
            "ppda_against": float(row.get("a_ppda", 0)),     # opponent's pressing
            "deep": int(row.get("h_deep", 0)),               # deep completions made
            "deep_against": int(row.get("a_deep", 0)),       # deep completions conceded
        })

        # Away team stats
        agg_rows.append({
            "date": match_date,
            "team": info["away_team"],
            "ppda": float(row.get("a_ppda", 0)),
            "ppda_against": float(row.get("h_ppda", 0)),
            "deep": int(row.get("a_deep", 0)),
            "deep_against": int(row.get("h_deep", 0)),
        })

    if not agg_rows:
        print("  No tactical aggregates computed — skipping")
        return df

    tac_agg = pd.DataFrame(agg_rows)
    tac_agg = tac_agg.sort_values(["team", "date"]).reset_index(drop=True)

    g = tac_agg.groupby("team")
    raw_cols = ["ppda", "ppda_against", "deep", "deep_against"]
    feat_names = ["PPDA", "PPDAAgainst", "Deep", "DeepAgainst"]

    for raw, feat in zip(raw_cols, feat_names):
        tac_agg[f"{feat}_8"] = g[raw].transform(
            lambda x: x.shift(1).ewm(span=ewm_span, min_periods=3).mean()
        )

    rolling_cols = [f"{f}_8" for f in feat_names]
    tac_feats = tac_agg[["date", "team"] + rolling_cols].drop_duplicates(["date", "team"])

    df = df.copy()
    for prefix in ["Home", "Away"]:
        renamed = tac_feats.rename(columns={c: f"{prefix}_{c}" for c in rolling_cols})
        df = df.merge(
            renamed[["date", "team"] + [f"{prefix}_{c}" for c in rolling_cols]],
            left_on=["Date", f"{prefix}_Team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["date", "team"], errors="ignore")

    n_filled = df[[f"Home_{rolling_cols[0]}"]].notna().sum().iloc[0]
    print(f"  Tactical features: {n_filled}/{len(df)} matches have data")
    return df


# The rolling features a promoted side has no basis for, and so the ones the
# cohort seeds. One list, because the pipeline fills these in training rows and
# the predictor fills them in the live row — a feature seeded in one place and
# not the other is the divergence ADR 0011 exists to prevent.
PROMOTED_ROLLING_FEATURES = [
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
    "Home_Past5Corners", "Away_Past5Corners",
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    "Home_ShotRatio_5", "Away_ShotRatio_5",
    "Home_ShotsPerGoal_5", "Away_ShotsPerGoal_5",
    "Home_CR_5", "Away_CR_5",
    "Home_ShotSuppression_5", "Away_ShotSuppression_5",
    "Home_ChanceQualityAllowed_5", "Away_ChanceQualityAllowed_5",
    "Home_ConversionAllowed_5", "Away_ConversionAllowed_5",
    "Home_GoalDiff_5", "Away_GoalDiff_5",
    # Added by ADR 0012. Serving has filled these since `fd64bd5` widened the
    # fill to every numeric column of an arriving side; training blended only
    # the list above, so each meant one quantity in training and another at
    # kick-off. What training kept was not a better value either — rolling
    # features are computed with `groupby("team")` and no gap awareness, so
    # Ipswich's `CR_20` for their first match of season 24 was a rolling mean
    # over their 2000/01 and 2001/02 matches.
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",
    "Home_CR_20", "Away_CR_20",
    "Home_SOT_CR_5", "Away_SOT_CR_5",
    "Home_SOT_CR_20", "Away_SOT_CR_20",
]


def bottom5_cohort(df, season, features):
    """What the division's weakest five looked like at the end of a season.

    The seed a side promoted into the PL receives, and the only definition of
    it. Two callers need this answer — the pipeline when it builds training
    rows, the predictor when it builds a feature row for an unplayed fixture
    — and they must not compute it separately. That is
    [ADR 0011](docs/adr/0011-one-division-movement-seed-per-arrival.md)'s
    contract applied to the PL: a feature cannot mean one quantity in
    training and another at kick-off.

    A promoted side's own history is never the seed. The only route into the
    PL is promotion out of the EFL, so a returning side's most recent PL rows
    are its relegation season — biased by construction, and stale besides.
    The bottom-five cohort is both less biased and lower variance.

    Args:
        df: The PL Canonical Dataset.
        season: The season to draw the cohort from — the one *before* the
            side arrives.
        features: Rolling feature names to average, each prefixed ``Home_``
            or ``Away_``; the prefix selects the venue the average is taken
            over, since a side's home and away form differ.

    Returns:
        Feature name -> cohort average, NaN where the season carries no
        usable values for it.
    """
    prev_df = df[df["SeasonIndex"] == season]
    if prev_df.empty:
        return {feat: np.nan for feat in features}

    # Position at each side's last home fixture: the final table.
    last_match = prev_df.sort_values("Date").drop_duplicates(
        "Home_Team", keep="last")
    team_positions = {
        row["Home_Team"]: row.get("Home_LeaguePosition", 20)
        for _, row in last_match.iterrows()
    }
    bottom5_teams = sorted(
        team_positions, key=lambda t: -team_positions.get(t, 20))[:5]

    cohort = {}
    for feat in features:
        prefix = "Home" if feat.startswith("Home") else "Away"
        team_col = f"{prefix}_Team"
        vals = []
        for team in bottom5_teams:
            team_rows = prev_df[prev_df[team_col] == team]
            if not team_rows.empty and feat in team_rows.columns:
                last_val = team_rows.sort_values("Date").iloc[-1][feat]
                if pd.notna(last_val):
                    vals.append(last_val)
        cohort[feat] = np.mean(vals) if vals else np.nan
    return cohort


def initialize_promoted_features(df):
    """
    Replace NaN/unreliable rolling features for promoted teams with
    bottom-5 league averages from the prior season.

    For a promoted team's first N matches (where rolling features are unreliable),
    blend bottom-5 averages with actual stats:
      Match 1: 100% bottom-5 avg
      Match 2: 80% avg + 20% actual
      ...
      Match 5: 20% avg + 80% actual
      Match 6+: 100% actual
    """
    df = df.copy()

    rolling_features = PROMOTED_ROLLING_FEATURES

    filled = 0

    for season_idx in sorted(df["SeasonIndex"].unique()):
        if season_idx == 0:
            continue

        # Find promoted teams this season
        season_mask = df["SeasonIndex"] == season_idx
        season_df = df[season_mask]
        promoted_home = season_df[season_df["Home_Promoted"] == 1]["Home_Team"].unique()
        promoted_away = season_df[season_df["Away_Promoted"] == 1]["Away_Team"].unique()
        promoted_teams = set(list(promoted_home) + list(promoted_away))

        if not promoted_teams:
            continue

        # Compute bottom-5 averages from end of prior season
        if df[df["SeasonIndex"] == (season_idx - 1)].empty:
            continue
        bottom5_avgs = bottom5_cohort(df, season_idx - 1, rolling_features)

        # Apply blended features to promoted teams' early matches
        for team in promoted_teams:
            for prefix in ["Home", "Away"]:
                team_col = f"{prefix}_Team"
                team_matches = df[season_mask & (df[team_col] == team)].sort_values("Date")

                for match_num, (idx, _) in enumerate(team_matches.iterrows(), 1):
                    if match_num > 5:
                        break
                    weight = seed_weight(match_num - 1)
                    if weight == 0:
                        continue

                    for feat in rolling_features:
                        if not feat.startswith(prefix):
                            continue
                        avg_val = bottom5_avgs.get(feat, np.nan)
                        if pd.isna(avg_val):
                            continue

                        actual = df.at[idx, feat]
                        if pd.isna(actual):
                            df.at[idx, feat] = avg_val
                        else:
                            df.at[idx, feat] = weight * avg_val + (1 - weight) * actual
                        filled += 1

    return df, filled


class SeasonPartitionError(RuntimeError):
    """A season in the data belongs to none of TRAIN/VAL/TEST (ADR 0009)."""


def temporal_split(df):
    """Split data by season index (no random shuffling).
    Training restricted to seasons with xG data (14+) for better signal.

    Raises SeasonPartitionError if any season present in `df` and eligible for
    training falls into none of the three sets. See ADR 0009: this partition is
    an allowlist while model.py's walk-forward selection is a denylist, so an
    unallocated season silently disappears here and reappears there.
    """
    eligible = df.loc[df["SeasonIndex"] >= TRAIN_MIN_SEASON, "SeasonIndex"]
    allocated = set(TRAIN_SEASONS) | set(VAL_SEASONS) | set(TEST_SEASONS)
    orphans = sorted(set(eligible.unique()) - allocated)
    if orphans:
        raise SeasonPartitionError(
            f"Season(s) {orphans} are present in the data and >= "
            f"TRAIN_MIN_SEASON ({TRAIN_MIN_SEASON}) but belong to no split. "
            f"TRAIN={min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)}, "
            f"VAL={VAL_SEASONS}, TEST={TEST_SEASONS}. "
            f"Roll the boundaries forward (ADR 0009)."
        )

    train = df[df["SeasonIndex"].isin(TRAIN_SEASONS) & (df["SeasonIndex"] >= TRAIN_MIN_SEASON)].copy()
    val = df[df["SeasonIndex"].isin(VAL_SEASONS)].copy()
    test = df[df["SeasonIndex"].isin(TEST_SEASONS)].copy()
    return train, val, test


def fill_nans(train_df, val_df, test_df, features):
    """
    Fill NaN values using training set medians only (no leakage).
    XG features are left as NaN so XGBoost can handle them natively
    (it learns optimal split direction for missing values).
    Returns the median dict so it can be saved for inference.
    """
    # Don't fill xG/Poisson-xG/injury features - XGBoost handles NaN better than median-filling
    # These features have partial season coverage; native NaN handling learns optimal splits
    native_nan = set(XG_FEATURES) | set(PLAYER_FEATURES) | set(SQUAD_FEATURES) | set(DEFENSIVE_TIER3_FEATURES) | set(WEATHER_FEATURES) | set(SHOT_LEVEL_FEATURES) | set(ROSTER_FEATURES) | set(TACTICAL_FEATURES) | set(DETAILED_MATCH_FEATURES) | {
        "Poisson_xG", "Expected_TG_xG", "Home_Lambda_xG", "Away_Lambda_xG",
        "Poisson_DC", "Expected_TG_DC",
        "Poisson_Shots", "Expected_TG_Shots",
        "Home_Lambda_Shots", "Away_Lambda_Shots",
        "Poisson_Consensus", "Expected_TG_Consensus",
    }
    features_to_fill = [f for f in features if f not in native_nan]

    train_medians = train_df[features].median()

    train_df[features_to_fill] = train_df[features_to_fill].fillna(train_medians[features_to_fill])
    val_df[features_to_fill] = val_df[features_to_fill].fillna(train_medians[features_to_fill])
    test_df[features_to_fill] = test_df[features_to_fill].fillna(train_medians[features_to_fill])

    return train_df, val_df, test_df, train_medians


def run_pipeline(verbose=True):
    """
    Full pipeline: load -> engineer features -> split -> fill NaNs.
    Returns train, val, test DataFrames and metadata.
    """
    if verbose:
        print("Loading data...")
    df = load_data()
    df = add_target(df)

    if verbose:
        print(f"Dataset: {len(df)} rows, {df.shape[1]} columns")
        print(f"Class balance: Over 2.5 = {df['Over_2_5'].mean():.1%}")

    if verbose:
        print("Computing rest days...")
    df = add_rest_days(df)

    if verbose:
        print("Computing fixture congestion features...")
    df = add_congestion_features(df)

    if verbose:
        print("Computing discipline features (cards, fouls)...")
    df = add_discipline_features(df)

    if verbose:
        print("Computing defensive components...")
    df = add_defensive_components(df)

    if verbose:
        print("Fetching weather features...")
    df = add_weather_features(df)

    if verbose:
        print("Computing match context features (relegation/title/euro proximity)...")
    df = add_match_context_features(df)

    if verbose:
        print("Adding derived features...")
    df = add_derived_features(df)

    if verbose:
        print("Computing rolling xG features...")
    df = add_xg_features(df)

    if verbose:
        print("Adding player availability features...")
    df = add_player_features(df)

    if verbose:
        print("Adding FPL pre-season strength features...")
    df = add_fpl_strength_features(df)

    if verbose:
        print("Adding squad strength features...")
    df = add_squad_features(df)

    if verbose:
        print("Computing advanced features (Elo, Poisson, rates)...")
    df = add_advanced_features(df)

    if verbose:
        print("Computing BTTS-specific features...")
    df = add_btts_features(df)

    if verbose:
        print("Computing corners-specific features...")
    df = add_corner_features(df)

    if verbose:
        print("Computing shot-level features...")
    df = add_shot_level_features(df)

    if verbose:
        print("Computing roster features (xGChain, xGBuildup, xA)...")
    df = add_roster_features(df)

    if verbose:
        print("Computing tactical features (PPDA, deep completions)...")
    df = add_tactical_features(df)

    if verbose:
        print("Computing detailed match features (2024-25)...")
    df = add_detailed_match_features(df)

    # Verify all features exist
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")
        features = [f for f in ALL_FEATURES if f in df.columns]
    else:
        features = ALL_FEATURES

    # Verify BTTS features
    btts_features = [f for f in BTTS_ALL_FEATURES if f in df.columns]
    btts_missing = [f for f in BTTS_ALL_FEATURES if f not in df.columns]
    if btts_missing and verbose:
        print(f"BTTS features missing: {btts_missing}")

    # Verify Corners features
    corners_features = [f for f in CORNERS_ALL_FEATURES if f in df.columns]
    corners_missing = [f for f in CORNERS_ALL_FEATURES if f not in df.columns]
    if corners_missing and verbose:
        print(f"Corners features missing: {corners_missing}")

    if verbose:
        print(f"Using {len(features)} O/U features, {len(btts_features)} BTTS features, {len(corners_features)} Corners features")

    # Initialize promoted teams' features with bottom-5 league averages
    if verbose:
        print("Initializing promoted team features...")
    df, promoted_filled = initialize_promoted_features(df)
    if verbose:
        print(f"  Filled {promoted_filled} feature values for promoted teams")

    # Temporal split
    train, val, test = temporal_split(df)
    if verbose:
        print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        print(f"Train seasons: {train['SeasonIndex'].min()}-{train['SeasonIndex'].max()}")
        print(f"Val seasons: {val['SeasonIndex'].min()}-{val['SeasonIndex'].max()}")
        print(f"Test seasons: {test['SeasonIndex'].min()}-{test['SeasonIndex'].max()}")

    # Fill NaNs with training medians
    train, val, test, train_medians = fill_nans(train, val, test, features)

    if verbose:
        remaining_nans = train[features].isnull().sum().sum()
        print(f"Remaining NaNs in train features: {remaining_nans}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "features": features,
        "btts_features": btts_features,
        "train_medians": train_medians,
        "full_df": df,
    }


if __name__ == "__main__":
    result = run_pipeline(verbose=True)
    train = result["train"]
    test = result["test"]
    features = result["features"]

    print(f"\nFeature list ({len(features)}):")
    for f in features:
        print(f"  {f}")

    print(f"\nTrain Over 2.5 rate: {train['Over_2_5'].mean():.1%}")
    print(f"Test Over 2.5 rate:  {test['Over_2_5'].mean():.1%}")
