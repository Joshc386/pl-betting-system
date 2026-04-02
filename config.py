"""
Centralized configuration for the Over/Under 2.5 Goals Prediction System.
"""
import os

# ── Paths ──
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")
ENRICHED_DATA_PATH = os.path.join(PROJECT_DIR, "CompleteDSPL_enriched.csv")
MATCHES_2425_PATH = os.path.join(PROJECT_DIR, "New Project Data", "matches2425.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "over_under_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.pkl")

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

# ── API Keys ──
# football-data.org: sign up at https://www.football-data.org/ for free key
FOOTBALL_DATA_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")

# FPL API - no key needed
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

# football-data.org
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
