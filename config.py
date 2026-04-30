"""
Centralized configuration for the Over/Under 2.5 Goals Prediction System.

Supports multi-league operation via ``ACTIVE_LEAGUE`` env var (default ``"PL"``).
League-specific structural parameters live in ``league_config.py``; this module
re-exports the active league's values for backward compatibility.
"""
import os

from dotenv import load_dotenv

load_dotenv()

from league_config import get_league_config

# ── Active league ──
ACTIVE_LEAGUE = os.environ.get("ACTIVE_LEAGUE", "PL")
LEAGUE_CFG = get_league_config(ACTIVE_LEAGUE)

# ── Paths ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = LEAGUE_CFG["csv_path"]
ENRICHED_DATA_PATH = LEAGUE_CFG["enriched_csv_path"]
MATCHES_2425_PATH = os.path.join(PROJECT_DIR, "New Project Data", "matches2425.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
# PL keeps original names for backward compat; other leagues get a suffix
_LEAGUE_SUFFIX = "" if ACTIVE_LEAGUE == "PL" else f"_{ACTIVE_LEAGUE.lower()}"
MODEL_PATH = os.path.join(MODEL_DIR, f"over_under_model{_LEAGUE_SUFFIX}.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, f"scaler{_LEAGUE_SUFFIX}.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, f"feature_list{_LEAGUE_SUFFIX}.pkl")

# ── Temporal split boundaries ──
# Train on seasons 14-23 initially; val (season 24) used for early stopping / pruning.
# After early stopping, model.py retrains on train+val (14-24) for final models.
# Walk-forward CV (in model.py) provides robust multi-fold evaluation.
# Season 25 (2025/26, in progress) is the held-out test set.
TRAIN_SEASONS = list(range(0, 24))       # Seasons 0-23 (2000/01 - 2023/24)
VAL_SEASONS = [24]                       # Season 24 (2024/25) — early stopping, then folded into train
TEST_SEASONS = [25]                      # Season 25 (2025/26 - current)

# ── Features ──
# Columns from CompleteDSPL_CSV.csv to use directly
EXISTING_FEATURES = [
    # Factor (contains potential leakage in X1-X3 but we'll address in pipeline)
    "Home Factor", "Away Factor",

    # Shot stats (rolling 5-game)
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    "Home_ShotRatio_5", "Away_ShotRatio_5",
    "Home_ShotsPerGoal_5", "Away_ShotsPerGoal_5",

    # Conversion rates
    "Home_CR_5", "Home_CR_20",
    "Away_CR_5", "Away_CR_20",
    "Home_SOT_CR_5", "Home_SOT_CR_20",
    "Away_SOT_CR_5", "Away_SOT_CR_20",

    # Defensive
    "Home_DefensiveStrength_5", "Away_DefensiveStrength_5",
    "Home_DefensiveStrength_SOT", "Away_DefensiveStrength_SOT",

    # League position
    "Home_LeaguePosition", "Away_LeaguePosition",

    # Form (rolling 5-game goals)
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
    "Home_Past5Corners", "Away_Past5Corners",
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",

    # Context
    "Home_Promoted", "Away_Promoted",
    "Local Derby", "Historical Derby",

    # H2H
    "H2H_HomeWins", "H2H_AwayWins", "H2H_Draws",
    "H2H_AvgGoals_5", "H2HAvgGoals",
]

# Features we'll derive in pipeline.py
# Betting odds EXCLUDED from features - they defeat the model's purpose.
# Odds are used only as a benchmark to find edge, not as input.
DERIVED_FEATURES = [
    "LeaguePosition_Diff",
    "Home_RestDays",
    "Away_RestDays",
    "Home_GoalDiff_5",   # Past5Goals - Past5Conceded
    "Away_GoalDiff_5",
    # Option 3 Step 2a: Corner efficiency — rolling goals / rolling corners.
    # Hypothesis: higher efficiency => team converts set-piece pressure into
    # goals more reliably, sustaining goal expectancy.
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    "Home_CornerEfficiency_10", "Away_CornerEfficiency_10",
]

# xG features computed from Understat data (available seasons 14-23)
XG_FEATURES = [
    "Home_RollingXG_5",        # Avg xG created over last 5 matches
    "Away_RollingXG_5",
    "Home_RollingXGAgainst_5", # Avg xG conceded over last 5 matches
    "Away_RollingXGAgainst_5",
    "Home_XGOverperf_5",       # Goals minus xG (regression signal)
    "Away_XGOverperf_5",
    "Home_RollingTotalXG_5",   # Avg total match xG in recent games
    "Away_RollingTotalXG_5",
]

# Player availability features from FPL data (seasons 20-24).
# XGBoost handles NaN natively, so earlier seasons without data are fine.
PLAYER_FEATURES = [
    "Home_InjuryBurden",
    "Away_InjuryBurden",
    "Home_KeyAbsences",
    "Away_KeyAbsences",
]

# FPL pre-season team strength ratings (seasons 19-24).
# Static features known before GW1 — critical cold-start signal for early season.
FPL_STRENGTH_FEATURES = [
    "Home_FPL_Attack",       # Home team's attacking strength (normalised 0-1)
    "Away_FPL_Attack",       # Away team's attacking strength (normalised 0-1)
    "Home_FPL_Defence",      # Home team's defensive strength (normalised 0-1)
    "Away_FPL_Defence",      # Away team's defensive strength (normalised 0-1)
    "FPL_Openness",          # Combined attack - defence (positive = more open match)
    "FPL_HomeDominance",     # Home strength advantage over away
]

# Advanced features computed in pipeline (goals-based, Elo, Poisson)
ADVANCED_FEATURES = [
    "Home_Over25_5", "Away_Over25_5",        # Over 2.5 rate in last 5 games
    "Home_BTTS_5", "Away_BTTS_5",            # Both teams to score rate
    "Home_CS_5", "Away_CS_5",                # Clean sheet rate
    "Home_TGAvg_5", "Away_TGAvg_5",          # Total goals per game avg
    "Home_GPG_20", "Away_GPG_20",            # Long-term goals per game
    "Home_GAPG_20", "Away_GAPG_20",          # Long-term goals conceded per game
    # Option 3 Step 2c: short-horizon _3 windows
    "Home_Over25_3", "Away_Over25_3",        # Over 2.5 rate in last 3 games
    "Home_BTTS_3", "Away_BTTS_3",            # BTTS rate in last 3 games
    "Home_TGAvg_3", "Away_TGAvg_3",          # Total goals avg in last 3 games
    "Home_Past3Goals", "Away_Past3Goals",    # Sum of goals in last 3 games
    "Home_CornersAvg_3", "Away_CornersAvg_3",  # Avg corners won last 3
    "Elo_Diff",                               # Elo rating difference + home advantage
    # Poisson: Shots (lowest variance per Wheatcroft) + consensus features
    # Note: Poisson_DC removed — real DC signal enters via stacker's 3rd column,
    # pipeline's Poisson_DC (flat 20-game rolling) was only 0.41 correlated with it
    "Poisson_Shots",                          # P(over 2.5) from shots-based lambdas (Wheatcroft)
    "Poisson_Consensus",                      # Mean of 3 Poisson variants (xG, Goals, Shots)
    "Expected_TG_Consensus",                  # Mean of all 4 Expected_TG variants
    "Combined_Over25",                        # Avg of both teams' over 2.5 rates
    "Combined_BTTS",                          # Avg of both teams' BTTS rates
    "Attack_Power",                           # Combined long-term GPG
    # Corner features (Wheatcroft: corners outperform goals as inputs)
    "Home_CornersAvg_5", "Away_CornersAvg_5",        # Avg corners won last 5
    "Home_CornersConcAvg_5", "Away_CornersConcAvg_5", # Avg corners conceded last 5
    "Corner_Dominance",                                # Combined corner differential
    # EWM (span=10) versions: lower variance than hard 5-game cutoff
    "Home_Over25_EWM10", "Away_Over25_EWM10",    # EWM over 2.5 rate
    "Home_TGAvg_EWM10", "Away_TGAvg_EWM10",      # EWM total goals avg
    "Home_BTTS_EWM10", "Away_BTTS_EWM10",        # EWM BTTS rate
    "Home_GPG_EWM10", "Away_GPG_EWM10",          # EWM goals per game
]

# Squad-level features from FPL-Core-Insights player data (seasons 24-25+)
# Built from real match stats (xG, xA, defensive actions), NOT FPL fantasy points
SQUAD_FEATURES = [
    "Home_AvailableXG", "Away_AvailableXG",       # Available xG / best XI xG ratio
    "Home_AvailableXA", "Away_AvailableXA",       # Available xA / best XI xA ratio
    "Home_AttackMissing", "Away_AttackMissing",   # Fraction of attacking xG missing
    "Home_DefenceMissing", "Away_DefenceMissing", # Fraction of defensive actions missing
    "Home_StarAvailable", "Away_StarAvailable",   # Star player availability (xG-weighted)
    "Home_FormXG5", "Away_FormXG5",               # Avg per-90 xG of available players
    "Home_SquadDepth", "Away_SquadDepth",         # Unique players used / 20
    "AvailableXG_Diff",                           # Home - Away available xG ratio
    "DefenceMissing_Diff",                        # Home - Away defence missing
]

# Shot-level features from Understat per-shot data (seasons 14-25)
# Built from granular shot xG, position, situation — not available from match totals
SHOT_LEVEL_FEATURES = [
    # Original shot quality features
    "Home_xGPerShot_8", "Away_xGPerShot_8",         # Avg xG per shot (shot quality)
    "Home_BigChanceRate_8", "Away_BigChanceRate_8",   # % shots with xG > 0.3
    "Home_SetPieceXG_8", "Away_SetPieceXG_8",        # xG from set pieces per match
    "Home_OpenPlayXG_8", "Away_OpenPlayXG_8",         # xG from open play per match
    # Option 3 Step 2b: set-play xG dependency ratio (interaction of SetPieceXG + OpenPlayXG).
    # Higher = more reliant on set pieces = narrower goal distribution.
    "Home_SetPieceXG_Ratio_8", "Away_SetPieceXG_Ratio_8",
    "Home_AvgShotDist_8", "Away_AvgShotDist_8",      # Avg distance to goal centre
    "Home_CornerXG_8", "Away_CornerXG_8",             # xG from corner situations per match
    # Shot volume & accuracy
    "Home_ShotVolume_8", "Away_ShotVolume_8",         # Avg shots per match
    "Home_SOTRate_8", "Away_SOTRate_8",               # Shots on target %
    # Playing style indicators
    "Home_SetPieceShare_8", "Away_SetPieceShare_8",   # % shots from set pieces (dependency)
    "Home_HeadedRate_8", "Away_HeadedRate_8",         # % headed shots (aerial/crossing style)
    "Home_CrossRate_8", "Away_CrossRate_8",           # % shots from crosses (wing play)
    "Home_ThroughballRate_8", "Away_ThroughballRate_8",  # % shots from throughballs (direct play)
    "Home_TakeOnRate_8", "Away_TakeOnRate_8",         # % shots from take-ons (individual skill)
    "Home_ReboundRate_8", "Away_ReboundRate_8",       # % shots from rebounds (chaos/second balls)
    # Timing
    "Home_LateXGShare_8", "Away_LateXGShare_8",       # % of xG in last 15 mins (late pressure)
    # Defensive / variance
    "Home_BlockedRate_8", "Away_BlockedRate_8",       # % shots blocked (opponent organisation)
    "Home_xGStd_8", "Away_xGStd_8",                   # xG variance (consistency of chances)
]

# Roster features from Understat match roster data (xGChain, xGBuildup, xA)
# Measures chance creation quality and buildup play — seasons 14+
ROSTER_FEATURES = [
    "Home_xGChain_8", "Away_xGChain_8",           # Total xG of possessions involved in
    "Home_xGBuildup_8", "Away_xGBuildup_8",       # xGChain minus shot+assist (buildup quality)
    "Home_KeyPasses_8", "Away_KeyPasses_8",        # Passes leading directly to shots
    "Home_xA_8", "Away_xA_8",                     # Expected assists (chance creation quality)
    "Home_BuildupRatio_8", "Away_BuildupRatio_8", # xGBuildup/xGChain (midfield contribution)
    "Home_xAPerKP_8", "Away_xAPerKP_8",           # xA per key pass (quality of final ball)
]

# Tactical features from Understat match_info (PPDA, deep completions)
# Measures pressing intensity and attacking penetration — seasons 14+
TACTICAL_FEATURES = [
    "Home_PPDA_8", "Away_PPDA_8",                 # Passes Per Defensive Action (pressing)
    "Home_PPDAAgainst_8", "Away_PPDAAgainst_8",   # Opponent pressing intensity faced
    "Home_Deep_8", "Away_Deep_8",                 # Deep completions (passes within 20m of goal)
    "Home_DeepAgainst_8", "Away_DeepAgainst_8",   # Deep completions conceded
]

# Detailed tactical features from matches2425.csv (2024-25 season only)
# NaN for all training seasons — XGBoost handles natively
DETAILED_MATCH_FEATURES = [
    "Home_PossessionAvg_8", "Away_PossessionAvg_8",           # Possession % (match openness)
    "Home_BigChancesAvg_8", "Away_BigChancesAvg_8",           # Big chances created
    "Home_TouchesInBoxAvg_8", "Away_TouchesInBoxAvg_8",       # Touches in opp box
    "Home_xGSetPlayAvg_8", "Away_xGSetPlayAvg_8",             # xG from set pieces
]

# Match context features (relegation/title/European proximity + season progress)
# Computed from running standings table — available for all seasons
CONTEXT_FEATURES = [
    "Season_Progress",
    "Home_RelegationProximity", "Away_RelegationProximity",
    "Home_TitleProximity", "Away_TitleProximity",
    "Home_EuroProximity", "Away_EuroProximity",
]

# Fixture congestion features (matches in last 14 days, avg rest across 3 matches)
CONGESTION_FEATURES = [
    "Home_MatchesLast14Days", "Away_MatchesLast14Days",
    "Home_AvgRest3", "Away_AvgRest3",
]

# Discipline features (rolling yellow/red cards and fouls)
DISCIPLINE_FEATURES = [
    "Home_YellowCards_5", "Away_YellowCards_5",
    "Home_RedCards_10", "Away_RedCards_10",
    "Home_Fouls_5", "Away_Fouls_5",
]

# Weather features from Open-Meteo (temperature, precipitation, wind at home stadium)
WEATHER_FEATURES = [
    "Match_Temperature",
    "Match_Precipitation",
    "Match_WindSpeed",
]

# Combined feature list for the main model (squad features excluded — used as post-model adjustment)
# Note: CONGESTION_FEATURES and DISCIPLINE_FEATURES tested but did not improve accuracy — excluded
ALL_FEATURES = (EXISTING_FEATURES + DERIVED_FEATURES + XG_FEATURES + PLAYER_FEATURES +
                FPL_STRENGTH_FEATURES + ADVANCED_FEATURES + SHOT_LEVEL_FEATURES +
                ROSTER_FEATURES + TACTICAL_FEATURES + DETAILED_MATCH_FEATURES +
                CONTEXT_FEATURES + WEATHER_FEATURES)

# ═══════════════════════════════════════════════════════════════════════════════
# BTTS-specific features (used by BTTS model only, not Over/Under 2.5)
# ═══════════════════════════════════════════════════════════════════════════════

BTTS_SPECIFIC_FEATURES = [
    # Failed-to-score / scoring reliability
    "Home_FTS_5", "Away_FTS_5",               # Failed to score rate (last 5)
    "Home_FTS_10", "Away_FTS_10",             # Failed to score rate (last 10)
    "Home_ScoredAtLeast1_5", "Away_ScoredAtLeast1_5",  # Complement of FTS
    "Home_GoalStd_10", "Away_GoalStd_10",     # Scoring consistency (std dev)
    # Clean sheet streaks
    "Home_CSStreak", "Away_CSStreak",         # Consecutive clean sheets
    # Longer-window BTTS rate
    "Home_BTTS_10", "Away_BTTS_10",           # BTTS rate over 10 games
    # Half-time scoring tendency
    "Home_HT_Scored_5", "Away_HT_Scored_5",   # Scored in first half (last 5)
    "Home_HT_Conceded_5", "Away_HT_Conceded_5",  # Conceded in first half (last 5)
    # Combination / interaction features
    "Combined_FTS",                            # Avg blanking risk
    "Blanking_Risk",                           # P(at least one team fails to score)
    "Poisson_BTTS",                            # Poisson P(BTTS) from goal lambdas
    "Poisson_BTTS_xG",                         # Poisson P(BTTS) from xG lambdas
    "Poisson_BTTS_Consensus",                  # Mean of Poisson BTTS variants
    "BTTS_Attack_Power",                       # Product of GPG rates (joint scoring)
    "CS_Risk",                                 # Combined clean sheet risk
]

# Shared features that are BTTS-relevant (curated subset of ALL_FEATURES)
# Excludes total-goals features (Over25, TGAvg, Expected_TG, corner-based)
BTTS_SHARED_FEATURES = [
    # Option 3 Step 2a: corner efficiency — scoring productivity per corner
    "Home_CornerEfficiency_5", "Away_CornerEfficiency_5",
    "Home_CornerEfficiency_10", "Away_CornerEfficiency_10",
    # Option 3 Step 2c: short-horizon _3 BTTS-relevant features
    # (total-goal _3 features deliberately excluded for BTTS per the
    # design distinction between BTTS and O/U)
    "Home_Past3Goals", "Away_Past3Goals",
    "Home_BTTS_3", "Away_BTTS_3",
    # Individual team scoring/conceding rates
    "Home_Past5Goals", "Away_Past5Goals",
    "Home_Past5Conceded", "Away_Past5Conceded",
    "Home_CR_5", "Home_CR_20", "Away_CR_5", "Away_CR_20",
    "Home_SOT_CR_5", "Home_SOT_CR_20", "Away_SOT_CR_5", "Away_SOT_CR_20",
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    "Home_ShotRatio_5", "Away_ShotRatio_5",
    "Home_ShotsPerGoal_5", "Away_ShotsPerGoal_5",
    "Home_DefensiveStrength_5", "Away_DefensiveStrength_5",
    "Home_DefensiveStrength_SOT", "Away_DefensiveStrength_SOT",
    # League position & context
    "Home_LeaguePosition", "Away_LeaguePosition",
    "LeaguePosition_Diff",
    "Home_RestDays", "Away_RestDays",
    "Home_Promoted", "Away_Promoted",
    "Local Derby", "Historical Derby",
    # Existing BTTS/CS rolling features
    "Home_BTTS_5", "Away_BTTS_5",
    "Home_CS_5", "Away_CS_5",
    "Combined_BTTS",
    "Home_BTTS_EWM10", "Away_BTTS_EWM10",
    # xG (per-team, not total)
    "Home_RollingXG_5", "Away_RollingXG_5",
    "Home_RollingXGAgainst_5", "Away_RollingXGAgainst_5",
    "Home_XGOverperf_5", "Away_XGOverperf_5",
    # Per-team goal rates
    "Home_GPG_20", "Away_GPG_20",
    "Home_GAPG_20", "Away_GAPG_20",
    "Home_GPG_EWM10", "Away_GPG_EWM10",
    "Attack_Power",
    # Elo (quality differential)
    "Elo_Diff",
    # FPL Strength
    "Home_FPL_Attack", "Away_FPL_Attack",
    "Home_FPL_Defence", "Away_FPL_Defence",
    "FPL_Openness",
    # Player availability
    "Home_InjuryBurden", "Away_InjuryBurden",
    "Home_KeyAbsences", "Away_KeyAbsences",
    # Shot quality (per-team)
    "Home_xGPerShot_8", "Away_xGPerShot_8",
    "Home_BigChanceRate_8", "Away_BigChanceRate_8",
    "Home_OpenPlayXG_8", "Away_OpenPlayXG_8",
    "Home_SetPieceXG_8", "Away_SetPieceXG_8",
    "Home_ShotVolume_8", "Away_ShotVolume_8",
    "Home_SOTRate_8", "Away_SOTRate_8",
    # Roster
    "Home_xA_8", "Away_xA_8",
    "Home_KeyPasses_8", "Away_KeyPasses_8",
    # Tactical
    "Home_PPDA_8", "Away_PPDA_8",
    "Home_Deep_8", "Away_Deep_8",
    "Home_DeepAgainst_8", "Away_DeepAgainst_8",
    # Context
    "Season_Progress",
    "Home_RelegationProximity", "Away_RelegationProximity",
]

# Full BTTS feature set: shared + BTTS-specific
BTTS_ALL_FEATURES = BTTS_SHARED_FEATURES + BTTS_SPECIFIC_FEATURES

# ═══════════════════════════════════════════════════════════════════════════════
# Corners O/U 10.5 features (used by corners model only)
# ═══════════════════════════════════════════════════════════════════════════════

CORNERS_SPECIFIC_FEATURES = [
    # Per-team rolling corner stats (10-game windows for stability)
    "Home_CornersAvg_10", "Away_CornersAvg_10",
    "Home_CornersConcAvg_10", "Away_CornersConcAvg_10",
    "Home_TotalCorners_10", "Away_TotalCorners_10",
    "Home_CornersStd_10", "Away_CornersStd_10",
    # Long-term baseline
    "Home_TotalCorners_20", "Away_TotalCorners_20",
    # Corner differential (dominance signal)
    "Home_CornerDiff_5", "Away_CornerDiff_5",
    # EWM versions
    "Home_CornersEWM_10", "Away_CornersEWM_10",
    "Home_CornersConcEWM_10", "Away_CornersConcEWM_10",
    # Over 10.5 rate
    "Home_Over105_5", "Away_Over105_5",
    "Home_Over105_10", "Away_Over105_10",
    # Match-level
    "Combined_TotalCorners",
    "Corner_Poisson_Over105",
]

# Shared features relevant to corners (pressing, shot volume, tactical)
CORNERS_SHARED_FEATURES = [
    # Existing corner stats from add_advanced_features
    "Home_CornersAvg_5", "Away_CornersAvg_5",
    "Home_CornersConcAvg_5", "Away_CornersConcAvg_5",
    "Corner_Dominance",
    "Home_Past5Corners", "Away_Past5Corners",
    "Home_Past5CornersConceded", "Away_Past5CornersConceded",
    # Elo (quality differential drives corner counts)
    "Elo_Diff",
    # Tactical: pressing & deep completions drive corners
    "Home_PPDA_8", "Away_PPDA_8",
    "Home_PPDAAgainst_8", "Away_PPDAAgainst_8",
    "Home_Deep_8", "Away_Deep_8",
    "Home_DeepAgainst_8", "Away_DeepAgainst_8",
    # Shot volume (more shots -> more corners from blocks/saves)
    "Home_ShotVolume_8", "Away_ShotVolume_8",
    "Home_SOTRate_8", "Away_SOTRate_8",
    "Home_AvgShotsOnTarget_5", "Away_AvgShotsOnTarget_5",
    # Set piece style (corner-heavy teams)
    "Home_SetPieceShare_8", "Away_SetPieceShare_8",
    "Home_SetPieceXG_8", "Away_SetPieceXG_8",
    "Home_CornerXG_8", "Away_CornerXG_8",
    # Crossing style (wing play -> corners)
    "Home_CrossRate_8", "Away_CrossRate_8",
    # Possession & touches (possession teams force more corners)
    "Home_PossessionAvg_8", "Away_PossessionAvg_8",
    "Home_TouchesInBoxAvg_8", "Away_TouchesInBoxAvg_8",
    # FPL strength
    "Home_FPL_Attack", "Away_FPL_Attack",
    "Home_FPL_Defence", "Away_FPL_Defence",
    "FPL_Openness",
    # League position (proxy for team quality)
    "Home_LeaguePosition", "Away_LeaguePosition",
    "LeaguePosition_Diff",
    # Context
    "Home_Promoted", "Away_Promoted",
    "Season_Progress",
]

# Full corners feature set
CORNERS_ALL_FEATURES = CORNERS_SHARED_FEATURES + CORNERS_SPECIFIC_FEATURES

# Training: only use seasons with xG data (14+) for better signal
TRAIN_MIN_SEASON = 14

# ── League structure (re-exported for backward compat) ──
LEAGUE_NAME = LEAGUE_CFG["name"]
TEAMS_COUNT = LEAGUE_CFG["teams_count"]
GAMES_PER_SEASON = LEAGUE_CFG["games_per_season"]
MAX_POINTS = LEAGUE_CFG["max_points"]
RELEGATION_POS = LEAGUE_CFG["relegation_pos"]
EURO_POS = LEAGUE_CFG.get("euro_pos")
PROMOTION_POS = LEAGUE_CFG.get("promotion_pos")
PLAYOFF_RANGE = LEAGUE_CFG.get("playoff_range")
HAS_FPL = LEAGUE_CFG["has_fpl"]
HAS_UNDERSTAT = LEAGUE_CFG["has_understat"]
UNDERSTAT_LEAGUE = LEAGUE_CFG.get("understat_league")
HAS_FBREF = LEAGUE_CFG["has_fbref"]
FBREF_COMP_ID = LEAGUE_CFG.get("fbref_comp_id")

# ── Dashboard ──
DASHBOARD_DB_PATH = LEAGUE_CFG["db_path"]

# ── Alternative O/U Lines ──
# Only lines with proven backtest edge are evaluated live.
# O/U 1.5 Over: +92.3% ROI in backtest; O/U 2.5 handled by main model.
ALLOWED_ALT_LINES = {1.5}
ALT_LINES_OVER_ONLY = True  # Only Over bets — backtest showed Under has no edge

# ── Selective Market Targeting ──
# De-vigged soft-book edges are noisier than Pinnacle-derived edges.
# Discount multiplier applied to the *edge* when edge_source == "devig"
# (0.80 = treat a 5% de-vig edge as 4% effective).
DEVIG_DISCOUNT = 0.80

# Market/side confidence multipliers (applied to Kelly stake).
# Values < 1.0 reduce sizing for historically weaker segments.
# Values > 1.0 boost sizing for historically stronger segments.
# Keys: (market, side) tuple. Missing combos default to 1.0.
MARKET_MULTIPLIERS: dict[tuple[str, str], float] = {
    # O/U 2.5 — Over historically strongest edge
    ("ou25", "over"):  1.00,
    ("ou25", "under"): 0.80,
    # O/U 1.5 — Over only (Under disabled), DC-only model
    ("ou15", "over"):  0.90,
    ("ou15", "under"): 0.50,
    # BTTS — slightly less reliable than O/U 2.5
    ("btts", "yes"):   0.90,
    ("btts", "no"):    0.75,
}

# ── Regime Detection (Phase C) ──
# Live in-season adjustment: when the current season's rolling Over/BTTS rate
# deviates meaningfully from the training base rate, shift the calibration
# target to track the new environment (S23 Over rate spiked to 64.7% but
# training prior was 52% — regime detection prevents months of mis-calibration).
#
# Keys: market code. Value: (lo, hi) clamp on the adjusted base rate.
# Regime is applied ONLY for markets in this dict — absence = disabled.
# BTTS deliberately omitted: goal totals trend seasonally, BTTS does not.
REGIME_CLAMPS: dict[str, tuple[float, float]] = {
    "ou25_pl":  (0.35, 0.70),   # PL Over/Under 2.5 — training prior ~0.52
    "ou25_efl": (0.32, 0.68),   # EFL Over/Under 2.5 — training prior ~0.48
    "ou15_efl": (0.55, 0.92),   # EFL Over/Under 1.5 — training prior ~0.73
}

# Rolling window parameters passed to RegimeDetector
REGIME_WINDOW = 40              # last N league matches for rolling rate
REGIME_BLEND_SPEED = 0.40       # fraction of deviation to apply (0=none, 1=full)
REGIME_TRIGGER_THRESHOLD = 0.04 # min deviation (4pp) before adjustment activates
REGIME_MIN_MATCHES = 15         # matches played before detector activates at all

# ── Two-Phase Early-Season Strategy ──
# First N matches of a season, use tighter betting parameters:
# lean harder on the market, demand a larger edge, stake smaller.
# Matches the validated backtest config (backtest.py DEFAULT_CONFIG).
EARLY_SEASON_MATCHES = 60         # ~6 PL matchweeks, ~5 EFL matchweeks
EARLY_BLEND_WEIGHT = 0.20         # lean harder on market (vs normal 0.35)
EARLY_MIN_EDGE = 0.03             # stricter edge threshold (vs normal 0.02)
EARLY_KELLY_FRACTION = 0.15       # smaller stakes (vs normal 0.25)

# ── Option 5: Staking Refinement ──

# Step 1: portfolio Kelly — cap total daily stake as a % of bankroll.
# Same-day bets are correlated (same matchday environment, same weather,
# sometimes same fixture). Kelly's independence assumption is violated,
# so we cap the aggregate exposure. If proposed sum exceeds this, every
# bet on that matchday gets pro-rated down.
#
# Phase 4a ROI validation (reports/roi_findings.md) showed the 10% cap
# is DORMANT in 5 of 6 market cells over S19–S24 — current staking rarely
# aggregates above 10% per matchday. It fires on EFL O/U 1.5 only, where
# bet density is ~9/matchday. Kept at 10% as a rare-event safety net
# rather than tightened; tightening to 7% was considered and deferred
# (action A2 in roi_findings.md).
MAX_MATCHDAY_STAKE_PCT = 0.10

# Step 2c: same-match correlation discount factors.
# RETAINED AS HISTORICAL RECORD — no longer referenced by production.
# Phase 4a showed this discount fired zero times across 6 cells × 6
# seasons (different markets rarely identify value on the same fixture
# simultaneously), so it was removed from apply_portfolio_constraints
# post-Phase-5. Empirical correlations from scripts/measure_correlation.py
# (see reports/same_match_correlations.csv). Re-referenced if we ever
# reintroduce a same-match combination logic.
SAME_MATCH_DISCOUNT: dict[frozenset[str], float] = {
    frozenset({"ou25", "btts"}): 0.83,  # rho=+0.56
    frozenset({"ou15", "btts"}): 0.79,  # rho=+0.61
    frozenset({"ou15", "ou25"}): 0.81,  # rho=+0.58
}

# Step 3: Bayesian edge uncertainty shrinkage.
# Raw edges from a ~0.55-0.58 AUC model carry real noise; without
# shrinkage we overbet. shrunk = raw * (n / (n + N_PRIOR)), where
# n varies by model agreement count (higher agreement = larger n =
# less shrinkage). N_PRIOR represents "virtual observations of a
# zero-edge world" pulling the estimate toward zero.
USE_EDGE_SHRINKAGE = True
EDGE_SHRINKAGE_N_PRIOR = 2
# n per agreement count (4 models for PL O/U 2.5 / BTTS; 3 for EFL)
EDGE_SHRINKAGE_N_BY_AGREE: dict[int, int] = {
    0: 1, 1: 2,    # effectively 33-50% shrink (discouraged to bet anyway)
    2: 3,          # 60% of raw edge kept
    3: 6,          # ~75% kept
    4: 10,         # ~83% kept
}

# ── Sparse Feature Pruning (Option 3 Step 3) ──
# Groups identified by the feature audit (scripts/run_feature_audit.py,
# reports/feature_audit_*.csv) as contributing zero detectable signal
# above the Gaussian-noise baseline across multiple markets.
#
# Keys: group name (for logging). Values: the list of features in that group.
#
# Full audit evidence (mean permutation AUC drop, averaged across 10
# shuffle samples per feature per market):
#   DETAILED_MATCH   — 4%  coverage, 0.00000 perm drop on every market
#   ROSTER           — 38% coverage, 0.00000 perm drop on every market
#   SHOT_LEVEL       — 38% coverage, 0.00000 perm drop on every market
#   TACTICAL         — 38% coverage, 0.00000 perm drop on every market
#
# XG deliberately excluded for now — mean perm drop is mildly negative
# on PL markets (suggesting noise contribution) but xG has strong
# theoretical support and pruning it would be more controversial. Revisit
# after the diagnose_option3 A/B shows use_sparse=False doesn't regress.
#
# Default flag USE_SPARSE_FEATURES=True keeps existing behaviour unchanged
# (Option 3 is opt-in). Setting it to False, or passing
# use_sparse_features=False to a predictor constructor, activates pruning.
SPARSE_FEATURE_GROUPS: dict[str, list[str]] = {
    "DETAILED_MATCH": DETAILED_MATCH_FEATURES,
    "ROSTER":         ROSTER_FEATURES,
    "SHOT_LEVEL":     SHOT_LEVEL_FEATURES,
    "TACTICAL":       TACTICAL_FEATURES,
}

# Default: keep sparse feature groups in the feature set (preserves existing
# behaviour — Option 3 is opt-in, not flag-day). Individual predictor
# instances can override via their `use_sparse_features` constructor param
# and the `diagnose_option3.py` A/B script uses that to compare variants.
USE_SPARSE_FEATURES = True


def get_active_features(
    base_features: list[str],
    use_sparse: bool = True,
) -> list[str]:
    """Return the active feature list after optional sparse-group pruning.

    When use_sparse is True, returns base_features unchanged (current
    behaviour). When False, excludes any feature that appears in any
    group listed in SPARSE_FEATURE_GROUPS.

    Args:
        base_features: The feature list to filter (e.g. ALL_FEATURES).
        use_sparse: Whether to keep sparse-group features. Default True.

    Returns:
        Filtered feature list preserving original order.
    """
    if use_sparse:
        return list(base_features)
    excluded: set[str] = set()
    for group_feats in SPARSE_FEATURE_GROUPS.values():
        excluded.update(group_feats)
    return [f for f in base_features if f not in excluded]


# Dixon-Coles lambda clipping bounds for numerical stability
DC_LAMBDA_MIN = 0.1
DC_LAMBDA_MAX = 5.0

# ── OddsPapi ──
ODDSPAPI_CACHE_TTL_MINUTES = 60  # Longer TTL than odds_api since quota is tighter
ODDSPAPI_SHARP_BOOKS = {"pinnacle"}
ODDSPAPI_TIER1_BOOKS = {"bet365", "betfair-ex", "unibet", "betsson", "betway"}

# ── Option β-tight: OddsPapi usage strategy ──
# Phase 4a + API audit (April 2026) showed OddsPapi was burning ~430 credits/month
# (158 in 11 days) because every dashboard load + every predict refresh fetched
# fresh data, even though only the week-ahead snapshot meaningfully drives bet
# selection. Option β-tight restricts OddsPapi to the week-ahead snapshot only;
# matchday/lineup/CLV refreshes go Odds-API-only (still 14 sharp books, plenty
# for line-movement tracking and CLV).
#
# Snapshot type semantics:
#   "week_ahead"  → Monday morning sweep. Both APIs fetch.
#   "refresh"     → Matchday morning, KO-1h, KO-5m. Odds-API only.
# Default is "refresh" (the conservative cheap path); the Monday scheduler job
# explicitly passes snapshot_type="week_ahead".
DEFAULT_SNAPSHOT_TYPE = "refresh"

# Dashboard read-only mode: when False, dashboard scans serve the most-recent
# cached merged snapshot from the predictor and never fire a fresh OddsPapi
# fetch on page load. The Refresh button + scheduled jobs are the only paths
# that hit the API. Set True only for diagnostic use.
DASHBOARD_FETCH_ODDSPAPI = False

# Hard quota guardrail. When this fraction of either provider's monthly cap
# is reached, the fetch sites refuse to send further requests until the
# month rolls over or the operator swaps the API key. The dashboard widget
# colours quota red at >=0.90 — set guardrail slightly above so the operator
# has a window to react before the system stops fetching entirely.
QUOTA_GUARDRAIL_THRESHOLD = 0.95

# ── Path B selective per-event fetching ──
# Per-event Odds API calls cost 1 credit each. Rather than fetching every
# market for every fixture, we let the model decide which fixtures are worth
# spending credits on. The thresholds below define "model conviction"
# windows where edge is plausible enough to spend a credit on real odds.
#
# Tuning notes:
#   - Stricter (closer to 0.5 / further from 1) → fewer fetches, fewer bets.
#   - More permissive → more fetches, higher API cost, more bet candidates.
#   - Live ROI feedback over a few weeks should drive these higher or lower.

# BTTS: fetch only when |model_prob − 0.50| > BTTS_FETCH_PROB_DELTA.
# A delta of 0.10 means we fetch when model says <40% or >60%. Below that,
# the BTTS market is essentially a coin flip and bookies price it tightly.
BTTS_FETCH_PROB_DELTA = 0.10

# O/U 1.5: fetch only when model_prob (Over) > threshold.
# Under is disabled per ALT_LINES_OVER_ONLY, so we only care about Over edge.
# Per-league because the two leagues have different scoring profiles:
#   - PL: training base rate ~0.78. Books offer 1.15-1.30 (implied 77-87%).
#     Need model_prob > 0.82 to leave room for >2% edge.
#   - EFL: training base rate ~0.73. Books offer 1.25-1.50 (implied 67-80%).
#     Need model_prob > 0.70 to leave room for >2% edge.
# A single global threshold either over-fetches in PL or never fires in EFL.
OU15_FETCH_PROB_THRESHOLD: dict[str, float] = {
    "PL":  0.82,
    "EFL": 0.70,
}

# ── Path B Match Centre display filter ──
# By default the Match Centre hides rows where edge is far below break-even.
# The toggle in the UI flips this off to show every evaluated market for
# transparency (so the operator can see what the model considered and
# rejected). EDGE_DISPLAY_THRESHOLD is the minimum edge_pct (in %) for a
# row to be shown by default. -2.0 means "show all rows with edge >= -2%",
# i.e. include near-misses but hide definitely-no rows.
EDGE_DISPLAY_THRESHOLD = -2.0

# ── Phase 4a baseline ROI per (league, market) cell ──
# Source: scripts/run_phase4a_matrix.py, validated April 2026.
# These are the simulated ROI numbers from the historical OOF cache —
# the dashboard's "Live ROI vs Simulation" panel uses them as the
# benchmark to detect live drift. Trigger Phase 4b re-spin when any cell
# drifts > 3pp below baseline for two consecutive months.
PHASE_4A_BASELINE_ROI: dict[tuple[str, str], float] = {
    ("PL",  "ou25"): 0.0804,   # +8.04%
    ("PL",  "btts"): 0.0649,   # +6.49%
    ("PL",  "ou15"): -0.0203,  # -2.03%  (under-performing cell — watch closely)
    ("EFL", "ou25"): 0.0120,   # +1.20%
    ("EFL", "ou15"): 0.0035,   # +0.35%
    ("EFL", "btts"): 0.1384,   # +13.84%
}

# ── API Keys ──
# football-data.org: sign up at https://www.football-data.org/ for free key
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")

# FPL API - no key needed
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

# football-data.org
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
