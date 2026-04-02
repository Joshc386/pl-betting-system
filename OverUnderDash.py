"""
Over/Under 2.5 Goals Prediction Dashboard.
Local Dash app - select home/away teams, get probability + confidence + explanations.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
import webbrowser

from datetime import datetime
from scipy.stats import poisson as poisson_dist
from config import (MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH, DATA_PATH,
                    XG_FEATURES, SQUAD_FEATURES, CONTEXT_FEATURES,
                    CONGESTION_FEATURES, DISCIPLINE_FEATURES, WEATHER_FEATURES)
from model import EnsembleModel, IsotonicWrapper, DixonColesPredictor  # needed for unpickling
from api.team_mapping import normalize
from clv_tracker import log_bet as clv_log_bet, get_all_bets, calculate_clv, get_unsettled_bets
from alt_lines import build_goal_matrix, total_goals_distribution, prob_over_line, scan_all_lines, get_value_bets

# ── Load model and data at startup ──
print("Loading model and data...")
# Register classes in __main__ so unpickling works regardless of entry point
import sys
sys.modules["__main__"].EnsembleModel = EnsembleModel
sys.modules["__main__"].IsotonicWrapper = IsotonicWrapper
sys.modules["__main__"].DixonColesPredictor = DixonColesPredictor
model = joblib.load(MODEL_PATH)
_ = joblib.load(SCALER_PATH)  # backward compat (no longer used)
features = joblib.load(FEATURE_LIST_PATH)

# Load squad adjuster if available
ADJUSTER_PATH = os.path.join(os.path.dirname(MODEL_PATH), "squad_adjuster.pkl")
squad_adjuster = None
if os.path.exists(ADJUSTER_PATH):
    squad_adjuster = joblib.load(ADJUSTER_PATH)
    print("Squad adjuster loaded.")

# Load historical data WITH derived features via pipeline
from pipeline import run_pipeline
_pipeline_data = run_pipeline(verbose=False)
df = _pipeline_data["full_df"]
train_medians = _pipeline_data["train_medians"]

# Get list of current teams (from latest season)
latest_season = df["SeasonIndex"].max()
latest_df = df[df["SeasonIndex"] == latest_season]
current_teams = sorted(
    set(latest_df["Home_Team"].unique()) | set(latest_df["Away_Team"].unique())
)

# Identify promoted teams in the latest season
promoted_teams = set(
    list(latest_df[latest_df["Home_Promoted"] == 1]["Home_Team"].unique()) +
    list(latest_df[latest_df["Away_Promoted"] == 1]["Away_Team"].unique())
) if "Home_Promoted" in latest_df.columns else set()

# Pre-compute prior season final proximities for blending in live predictions
from pipeline import _compute_prior_season_proximities
_all_prior_proxim = _compute_prior_season_proximities(df)
# Get the prior season's final proximity values (for the current/latest season)
prior_proxim_lookup = _all_prior_proxim.get(latest_season - 1, {})
_promoted_default = prior_proxim_lookup.get("__promoted_default__", {"relprox": 0, "titleprox": 0, "europrox": 0})

# Dataset info
latest_match_date = df["Date"].max().strftime("%Y-%m-%d")
total_matches = len(df[df["SeasonIndex"] == latest_season])

# Try to set up SHAP explainer from the base model
try:
    # Handle EnsembleModel, CalibratedClassifierCV, or raw XGBoost
    if hasattr(model, 'xgb_model'):
        base_xgb = model.xgb_model
    elif hasattr(model, 'calibrated_classifiers_'):
        base_xgb = model.calibrated_classifiers_[0].estimator
    else:
        base_xgb = model
    explainer = shap.TreeExplainer(base_xgb)
except Exception:
    explainer = None


def get_team_features(team_name, is_home=True):
    """
    Compute the latest features for a team from historical data.
    Uses the most recent available data for rolling stats.
    """
    prefix = "Home" if is_home else "Away"
    team_col = "Home_Team" if is_home else "Away_Team"

    # Get team's most recent match
    team_matches = df[(df["Home_Team"] == team_name) | (df["Away_Team"] == team_name)]
    if team_matches.empty:
        return {}

    latest = team_matches.iloc[-1]

    feature_vals = {}
    for f in features:
        if f.startswith(prefix + "_") or f.startswith(prefix[:4]):
            feature_vals[f] = latest.get(f, np.nan)

    return feature_vals, latest


def compute_prediction_features(home_team, away_team):
    """
    Build a full feature vector for a home vs away matchup.
    Uses each team's most recent stats from the dataset.
    """
    feature_vec = {}

    # Get most recent row where each team played at home/away
    home_rows = df[df["Home_Team"] == home_team]
    away_rows = df[df["Away_Team"] == away_team]

    if home_rows.empty or away_rows.empty:
        return None

    latest_home = home_rows.iloc[-1]
    latest_away = away_rows.iloc[-1]

    # Fill from each team's most recent home/away appearance
    for f in features:
        if f.startswith("Home"):
            feature_vec[f] = latest_home.get(f, np.nan)
        elif f.startswith("Away"):
            feature_vec[f] = latest_away.get(f, np.nan)
        elif f == "B365_Implied_Over25":
            feature_vec[f] = np.nan
        elif f == "B365_Implied_Under25":
            feature_vec[f] = np.nan
        elif f == "LeaguePosition_Diff":
            hp = latest_home.get("Home_LeaguePosition", 10)
            ap = latest_away.get("Away_LeaguePosition", 10)
            feature_vec[f] = hp - ap
        elif f == "Local Derby" or f == "Historical Derby":
            feature_vec[f] = 0
        elif f.startswith("H2H"):
            h2h = df[
                ((df["Home_Team"] == home_team) & (df["Away_Team"] == away_team)) |
                ((df["Home_Team"] == away_team) & (df["Away_Team"] == home_team))
            ]
            if not h2h.empty:
                feature_vec[f] = h2h.iloc[-1].get(f, np.nan)
            else:
                feature_vec[f] = np.nan
        elif f == "Elo_Diff":
            feature_vec[f] = latest_home.get("Elo_Diff", 50)
        elif f in ("Expected_TG_xG", "Expected_TG_Goals", "Poisson_xG", "Poisson_Goals",
                    "Poisson_DC", "Expected_TG_DC", "Poisson_Shots", "Expected_TG_Shots",
                    "Poisson_Consensus", "Expected_TG_Consensus",
                    "Combined_Over25", "Combined_BTTS", "Attack_Power", "Corner_Dominance"):
            feature_vec[f] = latest_home.get(f, np.nan)
        else:
            feature_vec[f] = np.nan

    # Dixon-Coles tau correction for low-score dependency
    def _dc_tau(x, y, lam, mu, rho=-0.13):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _poisson_over25_dc(hl, al, rho=-0.13):
        hl, al = max(hl, 0.01), max(al, 0.01)
        p_under = 0
        for h in range(12):
            for a in range(12):
                if h + a <= 2:
                    p = poisson_dist.pmf(h, hl) * poisson_dist.pmf(a, al) * _dc_tau(h, a, hl, al, rho)
                    p_under += max(p, 0)
        return 1 - p_under

    # Recompute Poisson and combination features from current team stats
    home_gpg = feature_vec.get("Home_Past5Goals", 5) / 5 if pd.notna(feature_vec.get("Home_Past5Goals")) else 1.0
    away_gpg = feature_vec.get("Away_Past5Goals", 5) / 5 if pd.notna(feature_vec.get("Away_Past5Goals")) else 1.0
    feature_vec["Expected_TG_Goals"] = home_gpg + away_gpg
    # Poisson from goals (naive)
    p_under = sum(poisson_dist.pmf(h, max(home_gpg, 0.01)) * poisson_dist.pmf(a, max(away_gpg, 0.01))
                  for h in range(12) for a in range(12) if h + a <= 2)
    feature_vec["Poisson_Goals"] = 1 - p_under

    # Dixon-Coles multiplicative lambdas
    league_avg_gpg = df["Home_Goals"].mean()
    league_avg_away = df["Away_Goals"].mean()
    h_gpg20 = feature_vec.get("Home_GPG_20", 1.0) or 1.0
    a_gapg20 = feature_vec.get("Away_GAPG_20", 1.0) or 1.0
    a_gpg20 = feature_vec.get("Away_GPG_20", 1.0) or 1.0
    h_gapg20 = feature_vec.get("Home_GAPG_20", 1.0) or 1.0
    home_factor = league_avg_gpg / league_avg_away
    hl_dc = (h_gpg20 / league_avg_gpg) * (a_gapg20 / league_avg_gpg) * league_avg_gpg * home_factor
    al_dc = (a_gpg20 / league_avg_away) * (h_gapg20 / league_avg_away) * league_avg_away
    feature_vec["Expected_TG_DC"] = hl_dc + al_dc
    feature_vec["Poisson_DC"] = _poisson_over25_dc(hl_dc, al_dc)

    # xG-based Poisson
    home_xg_val = feature_vec.get("Home_RollingXG_5", np.nan)
    away_xga_val = feature_vec.get("Away_RollingXGAgainst_5", np.nan)
    away_xg_val = feature_vec.get("Away_RollingXG_5", np.nan)
    home_xga_val = feature_vec.get("Home_RollingXGAgainst_5", np.nan)
    if pd.notna(home_xg_val) and pd.notna(away_xga_val):
        hl = (home_xg_val + away_xga_val) / 2
        al = (away_xg_val + home_xga_val) / 2
        feature_vec["Expected_TG_xG"] = hl + al
        p_under_xg = sum(poisson_dist.pmf(h, max(hl, 0.01)) * poisson_dist.pmf(a, max(al, 0.01))
                         for h in range(12) for a in range(12) if h + a <= 2)
        feature_vec["Poisson_xG"] = 1 - p_under_xg
    # Shots-based Poisson (Wheatcroft: lower variance than goals)
    home_sot = feature_vec.get("Home_SOT_Avg_5", np.nan)
    away_sot = feature_vec.get("Away_SOT_Avg_5", np.nan)
    if pd.notna(home_sot) and pd.notna(away_sot):
        sot_conv = df["Home_Goals"].sum() / df["Home_Shots_Target"].sum() if df["Home_Shots_Target"].sum() > 0 else 0.30
        hl_shots = home_sot * sot_conv
        al_shots = away_sot * sot_conv
        feature_vec["Expected_TG_Shots"] = hl_shots + al_shots
        p_under_shots = sum(poisson_dist.pmf(h, max(hl_shots, 0.01)) * poisson_dist.pmf(a, max(al_shots, 0.01))
                            for h in range(12) for a in range(12) if h + a <= 2)
        feature_vec["Poisson_Shots"] = 1 - p_under_shots

    # Corner dominance
    h_corn = feature_vec.get("Home_CornersAvg_5", np.nan)
    h_corn_c = feature_vec.get("Home_CornersConcAvg_5", np.nan)
    a_corn = feature_vec.get("Away_CornersAvg_5", np.nan)
    a_corn_c = feature_vec.get("Away_CornersConcAvg_5", np.nan)
    if all(pd.notna(v) for v in [h_corn, h_corn_c, a_corn, a_corn_c]):
        feature_vec["Corner_Dominance"] = (h_corn - h_corn_c) + (a_corn - a_corn_c)

    # Consensus features: average of all 4 Poisson/Expected_TG variants
    poisson_vals = [feature_vec.get(k) for k in ["Poisson_xG", "Poisson_Goals", "Poisson_DC", "Poisson_Shots"]]
    poisson_valid = [v for v in poisson_vals if pd.notna(v)]
    feature_vec["Poisson_Consensus"] = np.mean(poisson_valid) if poisson_valid else np.nan
    etg_vals = [feature_vec.get(k) for k in ["Expected_TG_xG", "Expected_TG_Goals", "Expected_TG_DC", "Expected_TG_Shots"]]
    etg_valid = [v for v in etg_vals if pd.notna(v)]
    feature_vec["Expected_TG_Consensus"] = np.mean(etg_valid) if etg_valid else np.nan

    # Combination features
    h_over25 = feature_vec.get("Home_Over25_5", 0.5)
    a_over25 = feature_vec.get("Away_Over25_5", 0.5)
    feature_vec["Combined_Over25"] = (h_over25 + a_over25) / 2 if pd.notna(h_over25) and pd.notna(a_over25) else 0.5
    h_btts = feature_vec.get("Home_BTTS_5", 0.5)
    a_btts = feature_vec.get("Away_BTTS_5", 0.5)
    feature_vec["Combined_BTTS"] = (h_btts + a_btts) / 2 if pd.notna(h_btts) and pd.notna(a_btts) else 0.5
    h_gpg20 = feature_vec.get("Home_GPG_20", 1.0)
    a_gpg20 = feature_vec.get("Away_GPG_20", 1.0)
    feature_vec["Attack_Power"] = (h_gpg20 or 1.0) + (a_gpg20 or 1.0)

    # Live match context features from current standings
    try:
        from api.football_data import get_standings
        standings_df = get_standings()
        if standings_df is not None and not standings_df.empty:
            # Build lookup: team_name -> {points, goal_difference, games_played}
            stnd = {}
            for _, s_row in standings_df.iterrows():
                stnd[s_row["team_name"]] = {
                    "points": s_row["points"],
                    "gd": s_row["goal_difference"],
                    "gp": s_row["games_played"],
                    "position": s_row["position"],
                }

            if home_team in stnd and away_team in stnd:
                h_info = stnd[home_team]
                a_info = stnd[away_team]

                # Points at key positions
                sorted_teams = sorted(stnd.keys(), key=lambda t: (-stnd[t]["points"], -stnd[t]["gd"]))
                pts_list = [stnd[t]["points"] for t in sorted_teams]
                pts_1st = pts_list[0] if len(pts_list) >= 1 else 0
                pts_7th = pts_list[6] if len(pts_list) >= 7 else 0
                pts_18th = pts_list[17] if len(pts_list) >= 18 else 0

                avg_played = (h_info["gp"] + a_info["gp"]) / 2
                season_progress = avg_played / 38.0
                max_remaining = max((38 - avg_played) * 3, 1)

                feature_vec["Season_Progress"] = season_progress

                # Current-season proximity values
                curr_context = {
                    "Home_RelegationProximity": (h_info["points"] - pts_18th) / max_remaining,
                    "Home_TitleProximity": (pts_1st - h_info["points"]) / max_remaining,
                    "Home_EuroProximity": (h_info["points"] - pts_7th) / max_remaining,
                    "Away_RelegationProximity": (a_info["points"] - pts_18th) / max_remaining,
                    "Away_TitleProximity": (pts_1st - a_info["points"]) / max_remaining,
                    "Away_EuroProximity": (a_info["points"] - pts_7th) / max_remaining,
                }

                # Blend with prior season final proximity
                w = season_progress  # 0 at start -> 100% prior; 1 at end -> 100% current
                for team, prefix in [(home_team, "Home"), (away_team, "Away")]:
                    prior = prior_proxim_lookup.get(team, _promoted_default)
                    feature_vec[f"{prefix}_RelegationProximity"] = (1 - w) * prior["relprox"] + w * curr_context[f"{prefix}_RelegationProximity"]
                    feature_vec[f"{prefix}_TitleProximity"] = (1 - w) * prior["titleprox"] + w * curr_context[f"{prefix}_TitleProximity"]
                    feature_vec[f"{prefix}_EuroProximity"] = (1 - w) * prior["europrox"] + w * curr_context[f"{prefix}_EuroProximity"]
    except Exception:
        # Fall back to last known values from dataset
        for f in CONTEXT_FEATURES:
            if f not in feature_vec or pd.isna(feature_vec.get(f)):
                feature_vec[f] = latest_home.get(f, 0) if f.startswith("Home") or f == "Season_Progress" else latest_away.get(f, 0)

    # Congestion features from recent match history
    for team, prefix in [(home_team, "Home"), (away_team, "Away")]:
        team_matches = df[(df["Home_Team"] == team) | (df["Away_Team"] == team)].sort_values("Date")
        if not team_matches.empty:
            last_date = team_matches.iloc[-1]["Date"]
            recent_14d = team_matches[team_matches["Date"] > (last_date - pd.Timedelta(days=14))]
            feature_vec[f"{prefix}_MatchesLast14Days"] = len(recent_14d)
            if len(team_matches) >= 3:
                last3_dates = team_matches.tail(3)["Date"].tolist()
                gaps = [(last3_dates[i] - last3_dates[i-1]).days for i in range(1, len(last3_dates))]
                feature_vec[f"{prefix}_AvgRest3"] = np.mean(gaps) if gaps else np.nan
            else:
                feature_vec[f"{prefix}_AvgRest3"] = latest_home.get(f"{prefix}_AvgRest3", np.nan)

    # Discipline features from latest match data
    for f in DISCIPLINE_FEATURES:
        if f.startswith("Home"):
            feature_vec[f] = latest_home.get(f, np.nan)
        elif f.startswith("Away"):
            feature_vec[f] = latest_away.get(f, np.nan)

    # Weather features from live forecast
    try:
        from api.weather import get_live_weather
        weather = get_live_weather(home_team)
        feature_vec["Match_Temperature"] = weather.get("temperature")
        feature_vec["Match_Precipitation"] = weather.get("precipitation")
        feature_vec["Match_WindSpeed"] = weather.get("wind_speed")
    except Exception:
        pass

    # Fill NaNs with training medians (except features XGBoost handles natively)
    from config import PLAYER_FEATURES, SHOT_LEVEL_FEATURES, DETAILED_MATCH_FEATURES
    native_nan_set = set(XG_FEATURES) | set(PLAYER_FEATURES) | set(WEATHER_FEATURES) | set(SHOT_LEVEL_FEATURES) | set(DETAILED_MATCH_FEATURES) | {
        "Poisson_xG", "Expected_TG_xG", "Poisson_Shots", "Expected_TG_Shots",
        "Poisson_DC", "Expected_TG_DC", "Poisson_Consensus", "Expected_TG_Consensus",
    }
    for f in features:
        if f in feature_vec and (pd.isna(feature_vec[f]) or feature_vec[f] is None):
            if f not in native_nan_set:
                feature_vec[f] = train_medians.get(f, 0)

    return feature_vec


def get_confidence_level(prob):
    """Map probability to confidence level."""
    distance = abs(prob - 0.5)
    if distance > 0.2:
        return "High", "#2ecc71"
    elif distance > 0.1:
        return "Medium", "#f39c12"
    else:
        return "Low", "#e74c3c"


def compute_edge(model_prob, over_odds, under_odds, per_model_probs=None,
                 sharp_fair_over=None, sharp_fair_under=None):
    """
    Compare model probability against bookmaker odds using model-market blend.

    Uses 35% model / 65% market blend (calibrated via backtest grid search).
    Only flags bets where 2+ of 4 models agree on direction vs market.

    Args:
        model_prob: ensemble P(Over 2.5)
        over_odds: best available Over 2.5 odds (for sizing/EV)
        under_odds: best available Under 2.5 odds (for sizing/EV)
        per_model_probs: list of [xgb, lgb, lr, dc] Over probabilities (optional)
        sharp_fair_over: Pinnacle fair Over probability (if available, used as blend anchor)
        sharp_fair_under: Pinnacle fair Under probability (if available)

    Returns dict with edge analysis for both over and under 2.5.
    """
    BLEND_WEIGHT = 0.35   # 35% model / 65% market (from backtest optimisation)
    MIN_AGREE = 2         # minimum models agreeing on direction

    model_under = 1 - model_prob

    # Implied probabilities (raw, includes margin)
    implied_over = 1 / over_odds
    implied_under = 1 / under_odds
    overround = implied_over + implied_under

    # Fair probabilities: prefer Pinnacle (sharpest), else remove overround from best odds
    if sharp_fair_over is not None and sharp_fair_under is not None:
        fair_over = sharp_fair_over
        fair_under = sharp_fair_under
    else:
        fair_over = implied_over / overround
        fair_under = implied_under / overround

    # Model-market blend
    blended_over = BLEND_WEIGHT * model_prob + (1 - BLEND_WEIGHT) * fair_over
    blended_under = 1 - blended_over

    # Edge = blended prob minus fair bookmaker prob
    edge_over = blended_over - fair_over
    edge_under = blended_under - fair_under

    # Expected value per £1 staked (using blended probability)
    ev_over = (blended_over * over_odds) - 1
    ev_under = (blended_under * under_odds) - 1

    # Model agreement: how many of 4 models think this side beats the market?
    n_agree_over = 0
    n_agree_under = 0
    if per_model_probs is not None and len(per_model_probs) >= 2:
        n_agree_over = sum(1 for p in per_model_probs if p > fair_over)
        n_agree_under = sum(1 for p in per_model_probs if (1 - p) > fair_under)
    else:
        # Without per-model data, assume agreement if ensemble has edge
        n_agree_over = 4 if model_prob > fair_over else 0
        n_agree_under = 4 if model_under > fair_under else 0

    # Quarter-Kelly stake sizing (capped at 5%)
    def quarter_kelly(prob, odds):
        k = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
        return min(max(k * 0.25, 0), 0.05)

    kelly_over = quarter_kelly(blended_over, over_odds)
    kelly_under = quarter_kelly(blended_under, under_odds)

    # Recommendations (uses blend + agreement gating)
    def get_recommendation(edge, ev, n_agree):
        if ev > 0 and edge > 0.02 and n_agree >= MIN_AGREE:
            if edge > 0.05:
                return "STRONG VALUE", "#2ecc71"
            else:
                return "VALUE BET", "#27ae60"
        elif ev > 0 and edge > 0.01:
            return "Marginal", "#f39c12"
        else:
            return "No Edge", "#95a5a6"

    rec_over, col_over = get_recommendation(edge_over, ev_over, n_agree_over)
    rec_under, col_under = get_recommendation(edge_under, ev_under, n_agree_under)

    return {
        "overround": (overround - 1) * 100,
        "over": {
            "model_prob": model_prob, "blended_prob": blended_over,
            "fair_implied": fair_over,
            "edge": edge_over, "ev": ev_over, "kelly": kelly_over,
            "n_agree": n_agree_over,
            "recommendation": rec_over, "color": col_over,
        },
        "under": {
            "model_prob": model_under, "blended_prob": blended_under,
            "fair_implied": fair_under,
            "edge": edge_under, "ev": ev_under, "kelly": kelly_under,
            "n_agree": n_agree_under,
            "recommendation": rec_under, "color": col_under,
        },
    }


def _build_edge_panel(prob_over, over_odds, under_odds, per_model_probs=None,
                      sharp_fair_over=None, sharp_fair_under=None):
    """Build the edge detection HTML panel. Returns empty Div if no odds."""
    if not over_odds or not under_odds or over_odds <= 1 or under_odds <= 1:
        return html.Div()

    edge = compute_edge(prob_over, over_odds, under_odds, per_model_probs,
                        sharp_fair_over, sharp_fair_under)

    def side_block(label, data, odds):
        is_value = data["ev"] > 0 and data["edge"] > 0.02 and data.get("n_agree", 0) >= 2
        edge_pct = data["edge"] * 100
        bar_width = min(abs(edge_pct) * 5, 100)  # scale for visual
        bar_color = data["color"] if data["edge"] > 0 else "#e74c3c"

        stake_text = ""
        if data["kelly"] > 0 and data["ev"] > 0 and is_value:
            stake_text = f"Suggested stake: {data['kelly']*100:.1f}% of bankroll"

        agree_text = f"{data.get('n_agree', '?')}/4 models agree"
        agree_color = "#2ecc71" if data.get("n_agree", 0) >= 3 else "#f39c12" if data.get("n_agree", 0) >= 2 else "#e74c3c"

        return html.Div([
            html.P(label, style={"fontWeight": "bold", "fontSize": "16px",
                                 "marginBottom": "8px", "color": "#2c3e50"}),
            html.Div([
                html.Span("Blended: ", style={"color": "#7f8c8d"}),
                html.Span(f"{data['blended_prob']*100:.1f}%",
                          style={"fontWeight": "bold", "fontSize": "15px"}),
                html.Span(f"  (model: {data['model_prob']*100:.1f}%)",
                          style={"color": "#95a5a6", "fontSize": "12px"}),
                html.Span(f"  vs  Market: ", style={"color": "#7f8c8d"}),
                html.Span(f"{data['fair_implied']*100:.1f}%",
                          style={"fontWeight": "bold", "fontSize": "15px"}),
            ], style={"marginBottom": "8px"}),
            # Edge bar
            html.Div([
                html.Div(style={
                    "width": f"{bar_width}%", "height": "8px",
                    "backgroundColor": bar_color, "borderRadius": "4px",
                }),
            ], style={"backgroundColor": "#ecf0f1", "borderRadius": "4px",
                       "height": "8px", "marginBottom": "8px"}),
            html.Div([
                html.Span(f"Edge: {'+' if edge_pct > 0 else ''}{edge_pct:.1f}%",
                          style={"marginRight": "20px"}),
                html.Span(f"EV: {'+' if data['ev'] > 0 else ''}{data['ev']:.3f} per \u00a31",
                          style={"marginRight": "20px"}),
                html.Span(agree_text,
                          style={"marginRight": "20px", "color": agree_color,
                                 "fontWeight": "bold"}),
                html.Span(f"Odds: {odds}",
                          style={"color": "#7f8c8d"}),
            ], style={"fontSize": "13px", "marginBottom": "8px"}),
            # Recommendation badge
            html.Span(data["recommendation"],
                       style={"backgroundColor": data["color"], "color": "white",
                              "padding": "4px 12px", "borderRadius": "12px",
                              "fontWeight": "bold", "fontSize": "13px"}),
            html.Span(f"  {stake_text}",
                       style={"fontSize": "12px", "color": "#7f8c8d"}) if stake_text else html.Span(),
        ], style={"padding": "12px", "backgroundColor": "#fafafa",
                  "borderRadius": "8px", "marginBottom": "10px"})

    return html.Div([
        html.H3("Betting Edge Analysis",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "5px"}),
        html.P(f"Bookmaker overround: {edge['overround']:.1f}%",
               style={"textAlign": "center", "color": "#95a5a6", "fontSize": "12px",
                       "marginBottom": "15px"}),
        side_block("OVER 2.5", edge["over"], over_odds),
        side_block("UNDER 2.5", edge["under"], under_odds),
    ], style={"backgroundColor": "white", "borderRadius": "10px", "padding": "20px",
              "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "marginTop": "15px"})


def get_h2h_history(home_team, away_team, n=10):
    """Get head-to-head match history."""
    h2h = df[
        ((df["Home_Team"] == home_team) & (df["Away_Team"] == away_team)) |
        ((df["Home_Team"] == away_team) & (df["Away_Team"] == home_team))
    ].sort_values("Date", ascending=False).head(n)

    if h2h.empty:
        return pd.DataFrame()

    rows = []
    for _, m in h2h.iterrows():
        hxg = m.get("home_xg", np.nan)
        axg = m.get("away_xg", np.nan)
        xg_str = f"{hxg:.1f}-{axg:.1f}" if pd.notna(hxg) else "-"
        rows.append({
            "Date": m["Date"].strftime("%Y-%m-%d"),
            "Home": m["Home_Team"],
            "Away": m["Away_Team"],
            "Score": f"{int(m['Home_Goals'])}-{int(m['Away_Goals'])}",
            "xG": xg_str,
            "Total Goals": int(m["TG"]),
            "Over 2.5": "Yes" if m["TG"] > 2 else "No",
        })
    return pd.DataFrame(rows)


def get_team_form(team_name, n=5):
    """Get a team's recent form."""
    team_matches = df[
        (df["Home_Team"] == team_name) | (df["Away_Team"] == team_name)
    ].sort_values("Date", ascending=False).head(n)

    rows = []
    for _, m in team_matches.iterrows():
        is_home = m["Home_Team"] == team_name
        goals_for = int(m["Home_Goals"]) if is_home else int(m["Away_Goals"])
        goals_against = int(m["Away_Goals"]) if is_home else int(m["Home_Goals"])
        if goals_for > goals_against:
            result = "W"
        elif goals_for < goals_against:
            result = "L"
        else:
            result = "D"

        # xG data (if available)
        xg_for = m.get("home_xg" if is_home else "away_xg", np.nan)
        xg_against = m.get("away_xg" if is_home else "home_xg", np.nan)
        xg_str = f"{xg_for:.1f}-{xg_against:.1f}" if pd.notna(xg_for) else "-"

        rows.append({
            "Date": m["Date"].strftime("%Y-%m-%d"),
            "Opponent": m["Away_Team"] if is_home else m["Home_Team"],
            "H/A": "H" if is_home else "A",
            "Score": f"{goals_for}-{goals_against}",
            "xG": xg_str,
            "Result": result,
            "TG": int(m["TG"]),
        })
    return pd.DataFrame(rows)


# ── Dash App ──
app = Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Over/Under 2.5 Goals Predictor",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "5px"}),
        html.P("Premier League Match Prediction",
               style={"textAlign": "center", "color": "#7f8c8d", "fontSize": "16px"}),
        html.P(f"Data through: {latest_match_date} | {total_matches} matches this season",
               style={"textAlign": "center", "color": "#95a5a6", "fontSize": "12px",
                       "marginTop": "5px"}),
    ], style={"padding": "20px", "backgroundColor": "#ecf0f1", "borderRadius": "10px",
              "marginBottom": "20px"}),

    # Team Selection
    html.Div([
        html.Div([
            html.Label("Home Team", style={"fontWeight": "bold", "fontSize": "16px"}),
            dcc.Dropdown(
                id="home-team",
                options=[{"label": f"{t} (P)" if t in promoted_teams else t, "value": t}
                         for t in current_teams],
                placeholder="Select home team...",
                style={"fontSize": "14px"},
            ),
        ], style={"width": "40%", "display": "inline-block", "marginRight": "5%"}),

        html.Div([
            html.Label("Away Team", style={"fontWeight": "bold", "fontSize": "16px"}),
            dcc.Dropdown(
                id="away-team",
                options=[{"label": f"{t} (P)" if t in promoted_teams else t, "value": t}
                         for t in current_teams],
                placeholder="Select away team...",
                style={"fontSize": "14px"},
            ),
        ], style={"width": "40%", "display": "inline-block"}),

        # Prediction stage selector
        html.Div([
            html.P("Prediction Mode", style={"fontWeight": "bold", "fontSize": "14px",
                                              "textAlign": "center", "marginTop": "15px"}),
            dcc.RadioItems(
                id="prediction-stage",
                options=[
                    {"label": " Pre-Match (estimated availability)", "value": "pre_match"},
                    {"label": " Lineup Confirmed (enter starting XI)", "value": "lineup"},
                ],
                value="pre_match",
                inline=True,
                style={"textAlign": "center", "fontSize": "13px"},
            ),
        ]),

        # Lineup input (hidden by default, shown when "lineup" mode selected)
        html.Div(id="lineup-section", children=[
            html.Div([
                html.Div([
                    html.Label("Home Starting XI", style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="home-lineup",
                        multi=True,
                        placeholder="Select 11 players...",
                        style={"fontSize": "12px"},
                    ),
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
                html.Div(style={"width": "10%", "display": "inline-block"}),
                html.Div([
                    html.Label("Away Starting XI", style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Dropdown(
                        id="away-lineup",
                        multi=True,
                        placeholder="Select 11 players...",
                        style={"fontSize": "12px"},
                    ),
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top"}),
            ]),
        ], style={"display": "none", "marginTop": "15px"}),

        # Odds inputs (auto-populated from Odds API or manual entry)
        html.Div([
            html.P("Odds (auto-populated from live market or enter manually)",
                   style={"textAlign": "center", "color": "#7f8c8d", "fontSize": "13px",
                          "marginBottom": "10px", "marginTop": "15px"}),
            html.Div([
                html.Div([
                    html.Label("Over 2.5 Odds", style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Input(id="over-odds", type="number", placeholder="e.g. 1.85",
                              min=1.01, step=0.01,
                              style={"width": "100%", "padding": "8px", "fontSize": "14px",
                                     "borderRadius": "4px", "border": "1px solid #bdc3c7"}),
                ], style={"width": "35%", "display": "inline-block", "marginRight": "5%"}),
                html.Div([
                    html.Label("Under 2.5 Odds", style={"fontWeight": "bold", "fontSize": "13px"}),
                    dcc.Input(id="under-odds", type="number", placeholder="e.g. 2.05",
                              min=1.01, step=0.01,
                              style={"width": "100%", "padding": "8px", "fontSize": "14px",
                                     "borderRadius": "4px", "border": "1px solid #bdc3c7"}),
                ], style={"width": "35%", "display": "inline-block"}),
            ], style={"textAlign": "center"}),
            # Live odds info panel (populated by callback)
            html.Div(id="live-odds-info", style={"marginTop": "8px", "textAlign": "center"}),
        ]),

        html.Div([
            html.Button("Predict", id="predict-btn", n_clicks=0,
                       style={"backgroundColor": "#3498db", "color": "white",
                              "border": "none", "padding": "12px 40px",
                              "fontSize": "16px", "borderRadius": "5px",
                              "cursor": "pointer", "marginTop": "20px"}),
        ], style={"textAlign": "center", "marginTop": "15px"}),
    ], style={"padding": "20px", "backgroundColor": "white", "borderRadius": "10px",
              "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "marginBottom": "20px"}),

    # Squad Analysis
    html.Div(id="squad-analysis", style={"marginBottom": "20px"}),

    # Prediction Output
    html.Div(id="prediction-output", style={"marginBottom": "20px"}),

    # Alternative Lines Analysis
    html.Div(id="alt-lines-section", style={"marginBottom": "20px"}),

    # Bet logging controls (hidden, updated by edge panel callback)
    html.Div([
        html.Button(id="log-bet-over", n_clicks=0, style={"display": "none"}),
        html.Button(id="log-bet-under", n_clicks=0, style={"display": "none"}),
        dcc.Store(id="edge-data-store", data={}),
        html.Div(id="log-bet-status", style={"display": "none"}),
    ], style={"display": "none"}),

    # Stats Comparison
    html.Div(id="stats-comparison", style={"marginBottom": "20px"}),

    # H2H History
    html.Div(id="h2h-section", style={"marginBottom": "20px"}),

    # CLV Tracker Section
    html.Div([
        html.Hr(style={"borderColor": "#bdc3c7", "margin": "30px 0"}),
        html.H2("Closing Line Value Tracker",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "5px"}),
        html.P("The best predictor of long-term betting edge",
               style={"textAlign": "center", "color": "#7f8c8d", "fontSize": "13px",
                       "marginBottom": "15px"}),
        html.Div([
            html.Button("Refresh CLV Stats", id="refresh-clv-btn", n_clicks=0,
                        style={"backgroundColor": "#3498db", "color": "white",
                               "border": "none", "padding": "8px 20px",
                               "fontSize": "13px", "borderRadius": "5px",
                               "cursor": "pointer", "marginRight": "10px"}),
            html.Button("Fetch Closing Odds", id="fetch-closing-btn", n_clicks=0,
                        style={"backgroundColor": "#9b59b6", "color": "white",
                               "border": "none", "padding": "8px 20px",
                               "fontSize": "13px", "borderRadius": "5px",
                               "cursor": "pointer", "marginRight": "10px"}),
            html.Button("Auto-Settle from Results", id="settle-bets-btn", n_clicks=0,
                        style={"backgroundColor": "#e67e22", "color": "white",
                               "border": "none", "padding": "8px 20px",
                               "fontSize": "13px", "borderRadius": "5px",
                               "cursor": "pointer"}),
        ], style={"textAlign": "center", "marginBottom": "15px"}),
        html.Div(id="clv-status", style={"textAlign": "center", "fontSize": "13px",
                                          "marginBottom": "10px"}),
        html.Div(id="clv-summary", style={"marginBottom": "15px"}),
        html.Div(id="clv-chart", style={"marginBottom": "15px"}),
        html.Div(id="bet-log-table", style={"marginBottom": "15px"}),
    ], style={"padding": "20px", "backgroundColor": "white", "borderRadius": "10px",
              "boxShadow": "0 2px 4px rgba(0,0,0,0.1)", "marginTop": "20px"}),

], style={"maxWidth": "1000px", "margin": "0 auto", "padding": "20px",
          "fontFamily": "Arial, sans-serif", "backgroundColor": "#f5f6fa"})


# ── Load live odds at startup (cached, only hits API if stale) ──
_live_odds_cache = {}
try:
    from api.odds_api import get_upcoming_with_odds
    _live_odds_raw = get_upcoming_with_odds(our_teams=current_teams)
    for entry in _live_odds_raw:
        if entry.get("home_team") and entry.get("away_team"):
            key = (entry["home_team"], entry["away_team"])
            _live_odds_cache[key] = entry
    print(f"Live odds loaded: {len(_live_odds_cache)} fixtures")
except Exception as e:
    print(f"Live odds not available: {e}")


@app.callback(
    [Output("over-odds", "value"),
     Output("under-odds", "value"),
     Output("live-odds-info", "children")],
    [Input("home-team", "value"),
     Input("away-team", "value")],
)
def auto_populate_odds(home_team, away_team):
    """Auto-populate odds from Odds API when teams are selected."""
    if not home_team or not away_team or home_team == away_team:
        return None, None, ""

    key = (home_team, away_team)
    if key not in _live_odds_cache:
        return None, None, html.Span(
            "No live odds found for this fixture — enter manually",
            style={"color": "#95a5a6", "fontSize": "12px"})

    entry = _live_odds_cache[key]
    best_over = entry.get("best_over")
    best_under = entry.get("best_under")
    n_books = entry.get("n_books", 0)
    over_book = entry.get("best_over_book", "")
    under_book = entry.get("best_under_book", "")

    # Build info text
    info_parts = [
        html.Span(f"Best prices from {n_books} bookmakers",
                   style={"color": "#27ae60", "fontWeight": "bold", "fontSize": "12px"}),
        html.Span(f"  |  Over: {best_over:.2f} ({over_book})"
                   f"  |  Under: {best_under:.2f} ({under_book})",
                   style={"color": "#7f8c8d", "fontSize": "12px"}),
    ]

    # Add Pinnacle info if available
    pin_over = entry.get("pinnacle_over")
    pin_under = entry.get("pinnacle_under")
    if pin_over and pin_under:
        sharp_o = entry.get("sharp_fair_over", 0)
        sharp_u = entry.get("sharp_fair_under", 0)
        info_parts.append(html.Br())
        info_parts.append(html.Span(
            f"Pinnacle (sharp): O {pin_over:.2f} / U {pin_under:.2f}  "
            f"(fair: {sharp_o:.1%} / {sharp_u:.1%})",
            style={"color": "#8e44ad", "fontSize": "11px"}))

    return best_over, best_under, html.Div(info_parts)


@app.callback(
    Output("lineup-section", "style"),
    Input("prediction-stage", "value"),
)
def toggle_lineup_section(stage):
    if stage == "lineup":
        return {"display": "block", "marginTop": "15px"}
    return {"display": "none", "marginTop": "15px"}


@app.callback(
    [Output("home-lineup", "options"),
     Output("away-lineup", "options")],
    [Input("home-team", "value"),
     Input("away-team", "value")],
)
def populate_lineup_dropdowns(home_team, away_team):
    home_opts, away_opts = [], []
    try:
        from api.player_features import get_player_summary
        if home_team:
            players = get_player_summary(home_team)
            home_opts = [{"label": f"{p['name']} ({p['position'][:3]}) - {p['xg_p90']:.2f} xG/90",
                          "value": p["name"]} for p in players if p["minutes"] > 0]
        if away_team:
            players = get_player_summary(away_team)
            away_opts = [{"label": f"{p['name']} ({p['position'][:3]}) - {p['xg_p90']:.2f} xG/90",
                          "value": p["name"]} for p in players if p["minutes"] > 0]
    except Exception:
        pass
    return home_opts, away_opts


@app.callback(
    [Output("prediction-output", "children"),
     Output("alt-lines-section", "children"),
     Output("stats-comparison", "children"),
     Output("h2h-section", "children"),
     Output("squad-analysis", "children")],
    Input("predict-btn", "n_clicks"),
    [State("home-team", "value"),
     State("away-team", "value"),
     State("over-odds", "value"),
     State("under-odds", "value"),
     State("prediction-stage", "value"),
     State("home-lineup", "value"),
     State("away-lineup", "value")],
)
def make_prediction(n_clicks, home_team, away_team, over_odds, under_odds,
                    stage, home_lineup, away_lineup):
    if not n_clicks or not home_team or not away_team:
        return "", "", "", "", ""

    if home_team == away_team:
        return html.P("Please select different teams.", style={"color": "red", "textAlign": "center"}), "", "", "", ""

    # Compute features
    feature_vec = compute_prediction_features(home_team, away_team)
    if feature_vec is None:
        return html.P("Insufficient data for this matchup.", style={"color": "red"}), "", "", ""

    # Make base prediction + extract per-model probabilities for blend
    X = np.array([[feature_vec[f] for f in features]])
    # Compute Dixon-Coles prediction from stored dc_model
    dc_probs = None
    if hasattr(model, 'dc_model') and model.dc_model is not None:
        dc_prob = model.dc_model.predict_match(home_team, away_team)
        dc_probs = np.array([dc_prob])
    prob_over = model.predict_proba(X, dc_probs=dc_probs)[0][1]
    base_prob = prob_over

    # Extract per-model Over probabilities for model agreement check
    per_model_probs = None
    try:
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=features)
        else:
            X_df = X
        pmp = []
        if hasattr(model, 'xgb_model'):
            pmp.append(float(model.xgb_model.predict_proba(X)[:, 1][0]))
        if hasattr(model, 'lgb_model'):
            pmp.append(float(model.lgb_model.predict_proba(X_df)[:, 1][0]))
        if hasattr(model, 'logreg_model') and model.logreg_model is not None:
            from model import _fill_nan_median, _clip_scaled
            X_f, _ = _fill_nan_median(X, medians=model.logreg_model._col_medians)
            X_s = _clip_scaled(model.logreg_scaler.transform(X_f))
            pmp.append(float(model.logreg_model.predict_proba(X_s)[:, 1][0]))
        if dc_probs is not None:
            pmp.append(float(dc_probs[0]))
        if len(pmp) >= 2:
            per_model_probs = pmp
    except Exception:
        per_model_probs = None

    # Squad adjustment
    squad_analysis_panel = html.Div()
    adj_prob = None
    home_squad_feats, away_squad_feats = {}, {}

    try:
        from api.player_features import compute_live_squad_features, get_player_summary

        h_lineup = home_lineup if stage == "lineup" and home_lineup else None
        a_lineup = away_lineup if stage == "lineup" and away_lineup else None

        home_squad_feats = compute_live_squad_features(home_team, lineup=h_lineup)
        away_squad_feats = compute_live_squad_features(away_team, lineup=a_lineup)

        if squad_adjuster is not None and not all(np.isnan(v) for v in home_squad_feats.values() if isinstance(v, float)):
            # Build adjustment feature vector
            adj_vec = {"base_prob": prob_over}
            for feat_name in SQUAD_FEATURES:
                if feat_name.startswith("Home_"):
                    key = feat_name.replace("Home_", "")
                    adj_vec[feat_name] = home_squad_feats.get(key, np.nan)
                elif feat_name.startswith("Away_"):
                    key = feat_name.replace("Away_", "")
                    adj_vec[feat_name] = away_squad_feats.get(key, np.nan)
                elif feat_name == "AvailableXG_Diff":
                    adj_vec[feat_name] = (home_squad_feats.get("AvailableXG", 1) or 1) - (away_squad_feats.get("AvailableXG", 1) or 1)
                elif feat_name == "DefenceMissing_Diff":
                    adj_vec[feat_name] = (home_squad_feats.get("DefenceMissing", 0) or 0) - (away_squad_feats.get("DefenceMissing", 0) or 0)

            adj_X = pd.DataFrame([adj_vec])
            if not adj_X.isnull().all(axis=1).iloc[0]:
                adj_X = adj_X.fillna(adj_X.median())
                adj_prob = squad_adjuster.predict_proba(adj_X)[0][1]
                prob_over = adj_prob  # Use adjusted probability

        # Build squad analysis panel
        def _squad_summary(team_name, feats, lineup_list=None):
            players = get_player_summary(team_name)
            if not players:
                return html.P("No player data available", style={"color": "#7f8c8d"})

            avail_xg = feats.get("AvailableXG", None)
            att_missing = feats.get("AttackMissing", None)
            def_missing = feats.get("DefenceMissing", None)
            star = feats.get("StarAvailable", None)

            # Status summary
            injured = [p for p in players if p["status"] in ("i", "s", "u")]
            doubtful = [p for p in players if p["status"] == "d"]

            items = []
            if avail_xg is not None and not np.isnan(avail_xg):
                pct = avail_xg * 100
                color = "#2ecc71" if pct >= 90 else "#f39c12" if pct >= 75 else "#e74c3c"
                items.append(html.P([
                    html.Span("Squad Strength: ", style={"color": "#7f8c8d"}),
                    html.Span(f"{pct:.0f}%", style={"fontWeight": "bold", "color": color, "fontSize": "18px"}),
                ]))

            if att_missing is not None and not np.isnan(att_missing) and att_missing > 0.05:
                items.append(html.P(f"Attack weakened: {att_missing*100:.0f}% missing",
                                   style={"color": "#e74c3c", "fontSize": "13px"}))

            if def_missing is not None and not np.isnan(def_missing) and def_missing > 0.05:
                items.append(html.P(f"Defence weakened: {def_missing*100:.0f}% missing",
                                   style={"color": "#e74c3c", "fontSize": "13px"}))

            if injured:
                inj_text = ", ".join(f"{p['name']} ({p['position'][:3]})" for p in injured[:5])
                items.append(html.P([
                    html.Span("Out: ", style={"color": "#e74c3c", "fontWeight": "bold"}),
                    html.Span(inj_text, style={"fontSize": "12px"}),
                ]))

            if doubtful:
                dbt_text = ", ".join(f"{p['name']} ({p['chance']}%)" for p in doubtful[:3])
                items.append(html.P([
                    html.Span("Doubtful: ", style={"color": "#f39c12", "fontWeight": "bold"}),
                    html.Span(dbt_text, style={"fontSize": "12px"}),
                ]))

            mode_text = "Lineup confirmed" if lineup_list else "Pre-match estimate"
            items.insert(0, html.P(mode_text, style={"fontSize": "11px", "color": "#95a5a6",
                                                      "fontStyle": "italic"}))

            return html.Div(items)

        squad_analysis_panel = html.Div([
            html.H3("Squad Analysis",
                    style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "15px"}),
            html.Div([
                html.Div([
                    html.H4(home_team, style={"textAlign": "center", "marginBottom": "10px"}),
                    _squad_summary(home_team, home_squad_feats, home_lineup if stage == "lineup" else None),
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top",
                          "padding": "10px"}),
                html.Div([
                    html.Span("vs", style={"fontSize": "20px", "color": "#bdc3c7", "fontWeight": "bold"}),
                ], style={"width": "10%", "display": "inline-block", "textAlign": "center",
                          "verticalAlign": "middle", "paddingTop": "40px"}),
                html.Div([
                    html.H4(away_team, style={"textAlign": "center", "marginBottom": "10px"}),
                    _squad_summary(away_team, away_squad_feats, away_lineup if stage == "lineup" else None),
                ], style={"width": "45%", "display": "inline-block", "verticalAlign": "top",
                          "padding": "10px"}),
            ]),
        ], style={"backgroundColor": "white", "borderRadius": "10px", "padding": "20px",
                  "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})

    except Exception as e:
        squad_analysis_panel = html.Div()

    prob_under = 1 - prob_over
    confidence, conf_color = get_confidence_level(prob_over)

    # SHAP values for explanation
    shap_children = []
    if explainer is not None:
        try:
            sv = explainer.shap_values(X[0:1])
            top_idx = np.argsort(np.abs(sv[0]))[-10:][::-1]
            shap_fig = go.Figure(go.Bar(
                y=[features[i] for i in top_idx],
                x=[sv[0][i] for i in top_idx],
                orientation="h",
                marker_color=["#e74c3c" if sv[0][i] < 0 else "#2ecc71" for i in top_idx],
            ))
            shap_fig.update_layout(
                title="Key Factors (SHAP Values)",
                xaxis_title="Impact on Over 2.5 Prediction",
                height=350, margin=dict(l=200, r=20, t=40, b=40),
            )
            shap_children = [dcc.Graph(figure=shap_fig)]
        except Exception:
            pass

    # Build prediction display
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_over * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#3498db"},
            "steps": [
                {"range": [0, 30], "color": "#e8f8f5"},
                {"range": [30, 40], "color": "#d5f5e3"},
                {"range": [40, 60], "color": "#fef9e7"},
                {"range": [60, 70], "color": "#fdebd0"},
                {"range": [70, 100], "color": "#fdedec"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
        title={"text": f"Over 2.5 Goals Probability<br>{home_team} vs {away_team}"},
    ))
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))

    # Squad adjustment info
    adj_info = []
    if adj_prob is not None:
        delta = adj_prob - base_prob
        delta_color = "#e74c3c" if delta > 0 else "#2ecc71"
        delta_text = f"+{delta*100:.1f}%" if delta > 0 else f"{delta*100:.1f}%"
        adj_info = [
            html.Div([
                html.Span("Base model: ", style={"color": "#7f8c8d", "fontSize": "13px"}),
                html.Span(f"{base_prob*100:.1f}%", style={"fontSize": "13px"}),
                html.Span("  |  Squad adjustment: ", style={"color": "#7f8c8d", "fontSize": "13px"}),
                html.Span(delta_text, style={"fontWeight": "bold", "color": delta_color, "fontSize": "13px"}),
            ], style={"textAlign": "center", "marginTop": "5px"}),
        ]

    # Context badge (late-season match importance)
    context_badges = []
    season_prog = feature_vec.get("Season_Progress", 0) or 0
    if season_prog > 0.65:
        for team, prefix in [(home_team, "Home"), (away_team, "Away")]:
            rel_prox = feature_vec.get(f"{prefix}_RelegationProximity", 1)
            title_prox = feature_vec.get(f"{prefix}_TitleProximity", 1)
            euro_prox = feature_vec.get(f"{prefix}_EuroProximity", 0)

            if rel_prox is not None and rel_prox <= 0:
                context_badges.append(
                    html.Span(f"{team}: Relegation battle",
                              style={"backgroundColor": "#e74c3c", "color": "white",
                                     "padding": "3px 10px", "borderRadius": "10px",
                                     "fontSize": "12px", "marginRight": "8px"}))
            elif title_prox is not None and title_prox <= 0.15:
                context_badges.append(
                    html.Span(f"{team}: Title race",
                              style={"backgroundColor": "#f1c40f", "color": "#2c3e50",
                                     "padding": "3px 10px", "borderRadius": "10px",
                                     "fontSize": "12px", "marginRight": "8px"}))
            elif euro_prox is not None and -0.1 <= euro_prox <= 0.1:
                context_badges.append(
                    html.Span(f"{team}: European race",
                              style={"backgroundColor": "#3498db", "color": "white",
                                     "padding": "3px 10px", "borderRadius": "10px",
                                     "fontSize": "12px", "marginRight": "8px"}))
            elif rel_prox is not None and rel_prox > 0.5 and title_prox is not None and title_prox > 0.5:
                context_badges.append(
                    html.Span(f"{team}: Nothing to play for",
                              style={"backgroundColor": "#95a5a6", "color": "white",
                                     "padding": "3px 10px", "borderRadius": "10px",
                                     "fontSize": "12px", "marginRight": "8px"}))

    context_badge_div = html.Div(context_badges,
                                  style={"textAlign": "center", "marginTop": "8px"}) if context_badges else html.Div()

    # Look up Pinnacle sharp probabilities for better blend anchor
    _sharp_over, _sharp_under = None, None
    _odds_key = (home_team, away_team)
    if _odds_key in _live_odds_cache:
        _sharp_over = _live_odds_cache[_odds_key].get("sharp_fair_over")
        _sharp_under = _live_odds_cache[_odds_key].get("sharp_fair_under")

    prediction_section = html.Div([
        html.Div([
            dcc.Graph(figure=gauge),
            html.Div([
                html.Span(f"Confidence: ", style={"fontSize": "18px"}),
                html.Span(confidence, style={"fontSize": "18px", "fontWeight": "bold",
                                              "color": conf_color}),
            ], style={"textAlign": "center", "marginTop": "5px"}),
            *adj_info,
            context_badge_div,
            html.Div([
                html.Div([
                    html.P("Over 2.5", style={"fontWeight": "bold", "marginBottom": "5px"}),
                    html.P(f"{prob_over*100:.1f}%", style={"fontSize": "24px", "color": "#e74c3c"}),
                ], style={"display": "inline-block", "width": "45%", "textAlign": "center"}),
                html.Div([
                    html.P("Under 2.5", style={"fontWeight": "bold", "marginBottom": "5px"}),
                    html.P(f"{prob_under*100:.1f}%", style={"fontSize": "24px", "color": "#2ecc71"}),
                ], style={"display": "inline-block", "width": "45%", "textAlign": "center"}),
            ], style={"textAlign": "center", "marginTop": "10px"}),
        ], style={"backgroundColor": "white", "borderRadius": "10px", "padding": "20px",
                  "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}),

        # Edge detection panel (only if odds provided)
        _build_edge_panel(prob_over, over_odds, under_odds, per_model_probs,
                          _sharp_over, _sharp_under),

        # SHAP explanation
        html.Div(shap_children,
                 style={"backgroundColor": "white", "borderRadius": "10px",
                        "padding": "10px", "marginTop": "15px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}) if shap_children else html.Div(),
    ])

    # Stats comparison
    home_form = get_team_form(home_team, 5)
    away_form = get_team_form(away_team, 5)

    stats_section = html.Div([
        html.H3("Recent Form", style={"textAlign": "center", "color": "#2c3e50"}),
        html.Div([
            html.Div([
                html.H4(home_team, style={"textAlign": "center"}),
                dash_table.DataTable(
                    data=home_form.to_dict("records") if not home_form.empty else [],
                    columns=[{"name": c, "id": c} for c in home_form.columns] if not home_form.empty else [],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "fontSize": "12px", "padding": "5px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#ecf0f1"},
                    page_size=5,
                ),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
            html.Div(style={"width": "4%", "display": "inline-block"}),
            html.Div([
                html.H4(away_team, style={"textAlign": "center"}),
                dash_table.DataTable(
                    data=away_form.to_dict("records") if not away_form.empty else [],
                    columns=[{"name": c, "id": c} for c in away_form.columns] if not away_form.empty else [],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "fontSize": "12px", "padding": "5px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#ecf0f1"},
                    page_size=5,
                ),
            ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top"}),
        ]),
    ], style={"backgroundColor": "white", "borderRadius": "10px", "padding": "20px",
              "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})

    # H2H History
    h2h_df = get_h2h_history(home_team, away_team)
    h2h_section = html.Div([
        html.H3("Head-to-Head History", style={"textAlign": "center", "color": "#2c3e50"}),
        dash_table.DataTable(
            data=h2h_df.to_dict("records") if not h2h_df.empty else [],
            columns=[{"name": c, "id": c} for c in h2h_df.columns] if not h2h_df.empty else [],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px"},
            style_header={"fontWeight": "bold", "backgroundColor": "#ecf0f1"},
            style_data_conditional=[
                {"if": {"filter_query": '{Over 2.5} = "Yes"'},
                 "backgroundColor": "#fdedec"},
                {"if": {"filter_query": '{Over 2.5} = "No"'},
                 "backgroundColor": "#e8f8f5"},
            ],
            page_size=10,
        ) if not h2h_df.empty else html.P("No head-to-head history found.",
                                           style={"textAlign": "center", "color": "#7f8c8d"}),
    ], style={"backgroundColor": "white", "borderRadius": "10px", "padding": "20px",
              "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})

    # ── Alternative Lines Analysis ──
    alt_lines_panel = html.Div()
    try:
        # Get home/away lambdas from the feature vector
        home_l = feature_vec.get("Expected_TG_DC", 2.5) / 2 if pd.notna(feature_vec.get("Expected_TG_DC")) else None
        away_l = feature_vec.get("Expected_TG_DC", 2.5) / 2 if pd.notna(feature_vec.get("Expected_TG_DC")) else None

        # Better: use xG-based lambdas if available
        home_xg = feature_vec.get("Home_RollingXG_5", np.nan)
        away_xga = feature_vec.get("Away_RollingXGAgainst_5", np.nan)
        away_xg = feature_vec.get("Away_RollingXG_5", np.nan)
        home_xga = feature_vec.get("Home_RollingXGAgainst_5", np.nan)
        if pd.notna(home_xg) and pd.notna(away_xga) and pd.notna(away_xg) and pd.notna(home_xga):
            home_l = (home_xg + away_xga) / 2
            away_l = (away_xg + home_xga) / 2
        elif pd.notna(feature_vec.get("Expected_TG_Goals")):
            # Fallback to goals-based
            home_gpg = feature_vec.get("Home_Past5Goals", 5) / 5
            away_gpg = feature_vec.get("Away_Past5Goals", 5) / 5
            home_l = home_gpg
            away_l = away_gpg

        if home_l is not None and away_l is not None and home_l > 0 and away_l > 0:
            # Get all-lines odds from live cache
            key = (home_team, away_team)
            odds_by_line = {}
            if key in _live_odds_cache and "all_bookmakers" in _live_odds_cache[key]:
                from api.odds_api import get_best_odds_all_lines
                # Reconstruct match dict for the function
                mock_match = {"bookmakers": _live_odds_cache[key]["all_bookmakers"]}
                odds_by_line = get_best_odds_all_lines(mock_match)

            if odds_by_line:
                # Scan all lines for value
                dc_rho = -0.13
                if hasattr(model, 'dc_model') and model.dc_model is not None:
                    dc_rho = getattr(model.dc_model, 'rho', -0.13)

                all_opps = scan_all_lines(home_l, away_l, odds_by_line,
                                          rho=dc_rho, blend_weight=0.35)
                value_bets = get_value_bets(all_opps, min_edge=0.015, min_ev=0.0)

                # Build the goal distribution for display
                mat = build_goal_matrix(home_l, away_l, rho=dc_rho)
                goal_dist = total_goals_distribution(mat)

                # Goal distribution mini-chart
                goals_x = list(range(8))
                goals_y = [goal_dist.get(g, 0) * 100 for g in goals_x]
                dist_fig = go.Figure(go.Bar(
                    x=[str(g) for g in goals_x],
                    y=goals_y,
                    marker_color=["#e74c3c" if g > 2 else "#2ecc71" for g in goals_x],
                    text=[f"{y:.1f}%" for y in goals_y],
                    textposition="outside",
                ))
                dist_fig.update_layout(
                    title=f"Goal Distribution (λ_home={home_l:.2f}, λ_away={away_l:.2f})",
                    xaxis_title="Total Goals", yaxis_title="Probability %",
                    template="plotly_white", height=250,
                    margin=dict(l=40, r=20, t=40, b=30),
                )

                # Line probabilities table
                line_rows = []
                for line in sorted(odds_by_line.keys()):
                    lp = prob_over_line(goal_dist, line)
                    od = odds_by_line[line]
                    # Find if there's a value bet for this line
                    best_opp = None
                    for opp in value_bets:
                        if abs(opp["line"] - line) < 0.001:
                            if best_opp is None or opp["ev"] > best_opp["ev"]:
                                best_opp = opp

                    line_type = "Asian" if lp["type"] == "asian_quarter" else (
                        "Push" if lp["type"] == "whole" else "Std")
                    row = {
                        "Line": f"{line}",
                        "Type": line_type,
                        "P(Over)": f"{lp['p_over']*100:.1f}%",
                        "P(Under)": f"{lp['p_under']*100:.1f}%",
                        "Best Over": f"{od['best_over']:.2f}",
                        "Best Under": f"{od['best_under']:.2f}",
                        "Value": f"{best_opp['side'].upper()} +{best_opp['edge']*100:.1f}% edge" if best_opp else "-",
                    }
                    if lp.get("p_push", 0) > 0.001:
                        row["P(Over)"] += f" (push {lp['p_push']*100:.1f}%)"
                    line_rows.append(row)

                line_df = pd.DataFrame(line_rows)

                # Value bets highlight
                value_cards = []
                for opp in value_bets[:5]:  # Top 5 value bets
                    line_type_label = ""
                    if opp["line_type"] == "asian_quarter":
                        line_type_label = " (Asian)"
                    elif opp["line_type"] == "whole":
                        line_type_label = " (whole)"

                    edge_color = "#2ecc71" if opp["edge"] >= 0.03 else "#f39c12"
                    value_cards.append(html.Div([
                        html.Span(f"{opp['side'].upper()} {opp['line']}{line_type_label}",
                                  style={"fontWeight": "bold", "fontSize": "15px",
                                         "marginRight": "12px"}),
                        html.Span(f"@ {opp['odds']:.2f}",
                                  style={"fontSize": "14px", "marginRight": "12px"}),
                        html.Span(f"Edge: {opp['edge']*100:+.1f}%",
                                  style={"color": edge_color, "fontWeight": "bold",
                                         "marginRight": "12px"}),
                        html.Span(f"EV: {opp['ev']:+.3f}",
                                  style={"marginRight": "12px", "fontSize": "13px"}),
                        html.Span(f"Stake: {opp['kelly']*100:.1f}%",
                                  style={"fontSize": "13px", "color": "#7f8c8d"}),
                        html.Span(f"  ({opp['book']})",
                                  style={"fontSize": "11px", "color": "#95a5a6"}),
                    ], style={"padding": "8px 12px", "backgroundColor": "#f0fff0",
                              "borderRadius": "6px", "marginBottom": "6px",
                              "borderLeft": f"4px solid {edge_color}"}))

                if not value_cards:
                    value_cards = [html.P("No value bets found across alternative lines.",
                                          style={"color": "#7f8c8d", "textAlign": "center"})]

                alt_lines_panel = html.Div([
                    html.H3("Alternative Lines Analysis",
                            style={"textAlign": "center", "color": "#2c3e50",
                                   "marginBottom": "5px"}),
                    html.P(f"Scanning {len(odds_by_line)} lines across {odds_by_line[2.5]['n_books'] if 2.5 in odds_by_line else '?'} bookmakers",
                           style={"textAlign": "center", "color": "#95a5a6",
                                  "fontSize": "12px", "marginBottom": "15px"}),

                    # Value bets
                    html.Div([
                        html.H4("Value Bets Found",
                                style={"color": "#27ae60", "marginBottom": "8px"}),
                        *value_cards,
                    ], style={"marginBottom": "15px"}),

                    # Goal distribution chart
                    dcc.Graph(figure=dist_fig),

                    # All lines table
                    html.H4("All Available Lines",
                            style={"color": "#2c3e50", "marginBottom": "8px",
                                   "marginTop": "10px"}),
                    dash_table.DataTable(
                        data=line_df.to_dict("records"),
                        columns=[{"name": c, "id": c} for c in line_df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "center", "fontSize": "12px",
                                    "padding": "5px"},
                        style_header={"fontWeight": "bold", "backgroundColor": "#ecf0f1"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Value} != "-"'},
                             "backgroundColor": "#e8f8f5", "fontWeight": "bold"},
                        ],
                    ),
                ], style={"backgroundColor": "white", "borderRadius": "10px",
                          "padding": "20px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})
    except Exception as e:
        alt_lines_panel = html.Div(
            html.P(f"Alt lines unavailable: {e}",
                   style={"color": "#95a5a6", "textAlign": "center", "fontSize": "12px"}))

    return prediction_section, alt_lines_panel, stats_section, h2h_section, squad_analysis_panel


# ── CLV Tracker Callbacks ──

@app.callback(
    Output("log-bet-status", "children"),
    [Input("log-bet-over", "n_clicks"),
     Input("log-bet-under", "n_clicks")],
    [State("home-team", "value"),
     State("away-team", "value"),
     State("over-odds", "value"),
     State("under-odds", "value"),
     State("edge-data-store", "data")],
    prevent_initial_call=True,
)
def handle_log_bet(over_clicks, under_clicks, home_team, away_team,
                   over_odds, under_odds, edge_data):
    """Log a bet when user clicks the Log Bet button."""
    ctx = callback_context
    if not ctx.triggered or not home_team or not away_team or not edge_data:
        return ""

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger_id == "log-bet-over":
        side = "over"
        odds = over_odds
    elif trigger_id == "log-bet-under":
        side = "under"
        odds = under_odds
    else:
        return ""

    if not odds or odds <= 1:
        return html.Span("No valid odds to log.", style={"color": "#e74c3c"})

    data = edge_data.get(side, {})

    # Look up bookmaker and Pinnacle from live odds cache
    bookmaker = None
    sharp_fair = None
    commence_time = None
    key = (home_team, away_team)
    if key in _live_odds_cache:
        entry = _live_odds_cache[key]
        bookmaker = entry.get(f"best_{side}_book")
        sharp_fair = entry.get(f"sharp_fair_{side}")
        commence_time = entry.get("commence_time")

    bet_id = clv_log_bet(
        home_team=home_team,
        away_team=away_team,
        bet_side=side,
        odds_at_bet=odds,
        bookmaker=bookmaker,
        model_prob=data.get("model_prob"),
        blended_prob=data.get("blended_prob"),
        fair_implied=data.get("fair_implied"),
        edge=data.get("edge"),
        ev=data.get("ev"),
        kelly_stake=data.get("kelly"),
        n_agree=data.get("n_agree"),
        sharp_fair=sharp_fair,
        commence_time=commence_time,
    )

    return html.Span(
        f"Bet #{bet_id} logged: {side.upper()} {home_team} vs {away_team} @ {odds:.2f}",
        style={"color": "#27ae60", "fontWeight": "bold"})


@app.callback(
    [Output("clv-summary", "children"),
     Output("clv-chart", "children"),
     Output("bet-log-table", "children"),
     Output("clv-status", "children")],
    [Input("refresh-clv-btn", "n_clicks"),
     Input("fetch-closing-btn", "n_clicks"),
     Input("settle-bets-btn", "n_clicks")],
    prevent_initial_call=True,
)
def handle_clv_actions(refresh_clicks, fetch_clicks, settle_clicks):
    """Handle CLV panel actions: refresh stats, fetch closing odds, settle bets."""
    ctx = callback_context
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    status_msg = ""

    # Handle fetch closing odds
    if trigger_id == "fetch-closing-btn":
        from clv_tracker import fetch_and_store_closing_odds
        n = fetch_and_store_closing_odds()
        status_msg = f"Updated closing odds for {n} bets."

    # Handle auto-settle
    if trigger_id == "settle-bets-btn":
        from clv_tracker import _settle_from_results
        _settle_from_results()
        status_msg = "Auto-settle complete. Check results below."

    # Build CLV summary
    stats = calculate_clv()
    all_bets = get_all_bets()

    if stats["n_bets"] == 0 and all_bets.empty:
        empty_msg = html.Div([
            html.P("No bets logged yet.", style={"textAlign": "center", "color": "#7f8c8d",
                                                   "fontSize": "15px", "marginTop": "20px"}),
            html.P("Use the 'Log Bet' buttons on value bets above to start tracking CLV.",
                   style={"textAlign": "center", "color": "#95a5a6", "fontSize": "13px"}),
        ])
        return empty_msg, html.Div(), html.Div(), status_msg

    # Summary cards
    summary_cards = []
    n_total = len(all_bets)
    n_settled = int(all_bets["settled"].sum()) if not all_bets.empty else 0
    n_pending = n_total - n_settled

    summary_cards.append(_clv_card("Total Bets", str(n_total), "#3498db"))
    summary_cards.append(_clv_card("Settled", str(n_settled), "#27ae60"))
    summary_cards.append(_clv_card("Pending", str(n_pending), "#f39c12"))

    if stats["n_bets"] > 0:
        clv_color = "#27ae60" if stats["mean_clv_pct"] > 0 else "#e74c3c"
        summary_cards.append(_clv_card("Mean CLV",
                                        f"{stats['mean_clv_pct']:+.2f}%", clv_color))
        summary_cards.append(_clv_card("Beat Close Rate",
                                        f"{stats['beat_close_rate']:.0f}%",
                                        "#27ae60" if stats["beat_close_rate"] > 50 else "#e74c3c"))
        if stats["actual_roi"] is not None:
            roi_color = "#27ae60" if stats["actual_roi"] > 0 else "#e74c3c"
            summary_cards.append(_clv_card("ROI", f"{stats['actual_roi']:+.1f}%", roi_color))
        if stats["pinnacle_clv_pct"] is not None:
            pin_color = "#27ae60" if stats["pinnacle_clv_pct"] > 0 else "#e74c3c"
            summary_cards.append(_clv_card("Pinnacle CLV",
                                            f"{stats['pinnacle_clv_pct']:+.2f}%", pin_color))

    summary_div = html.Div(summary_cards,
                            style={"display": "flex", "flexWrap": "wrap",
                                   "justifyContent": "center", "gap": "10px"})

    # CLV chart (cumulative CLV over time)
    chart_div = html.Div()
    if stats["n_bets"] > 0 and stats.get("clv_series"):
        series = pd.DataFrame(stats["clv_series"])
        series["timestamp"] = pd.to_datetime(series["timestamp"])
        series = series.sort_values("timestamp")
        series["cumulative_clv"] = series["clv_pct"].cumsum()
        series["cumulative_pl"] = series["profit_loss"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series["timestamp"], y=series["cumulative_clv"],
            mode="lines+markers", name="Cumulative CLV %",
            line=dict(color="#3498db", width=2),
            marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=series["timestamp"], y=series["cumulative_pl"],
            mode="lines+markers", name="Cumulative P/L (units)",
            line=dict(color="#27ae60", width=2, dash="dash"),
            marker=dict(size=6),
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="#95a5a6")
        fig.update_layout(
            title="CLV & P/L Over Time",
            xaxis_title="", yaxis_title="",
            template="plotly_white", height=300,
            margin=dict(l=40, r=20, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        chart_div = dcc.Graph(figure=fig)

    # Bet log table
    table_div = html.Div()
    if not all_bets.empty:
        display_df = all_bets[["id", "timestamp", "home_team", "away_team",
                                "bet_side", "odds_at_bet", "closing_odds",
                                "edge", "profit_loss", "settled"]].copy()
        display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        display_df["edge"] = display_df["edge"].apply(
            lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "-")
        display_df["odds_at_bet"] = display_df["odds_at_bet"].apply(lambda x: f"{x:.2f}")
        display_df["closing_odds"] = display_df["closing_odds"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x > 0 else "-")
        display_df["profit_loss"] = display_df["profit_loss"].apply(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "pending")
        display_df["settled"] = display_df["settled"].apply(
            lambda x: "Yes" if x else "No")
        display_df.columns = ["#", "Time", "Home", "Away", "Side", "Odds",
                              "Close", "Edge", "P/L", "Settled"]

        table_div = html.Div([
            html.H4("Bet Log", style={"textAlign": "center", "color": "#2c3e50",
                                       "marginBottom": "10px"}),
            dash_table.DataTable(
                data=display_df.to_dict("records"),
                columns=[{"name": c, "id": c} for c in display_df.columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "fontSize": "12px", "padding": "5px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#ecf0f1"},
                style_data_conditional=[
                    {"if": {"filter_query": '{P/L} contains "+"'},
                     "backgroundColor": "#e8f8f5", "color": "#27ae60"},
                    {"if": {"filter_query": '{P/L} contains "-"'},
                     "backgroundColor": "#fdedec", "color": "#e74c3c"},
                ],
                page_size=10,
                sort_action="native",
            ),
        ])

    status_div = html.Span(status_msg, style={"color": "#7f8c8d"}) if status_msg else ""
    return summary_div, chart_div, table_div, status_div


def _clv_card(title, value, color):
    """Build a small summary card for CLV dashboard."""
    return html.Div([
        html.P(title, style={"fontSize": "11px", "color": "#7f8c8d",
                              "marginBottom": "2px", "textAlign": "center"}),
        html.P(value, style={"fontSize": "20px", "fontWeight": "bold",
                              "color": color, "textAlign": "center", "margin": "0"}),
    ], style={"backgroundColor": "#fafafa", "borderRadius": "8px",
              "padding": "10px 18px", "minWidth": "90px"})


if __name__ == "__main__":
    port = 8052
    url = f"http://127.0.0.1:{port}/"
    print(f"Starting dashboard at {url}")
    webbrowser.open(url)
    app.run(debug=False, port=port, use_reloader=False)
