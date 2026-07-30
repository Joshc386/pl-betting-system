"""
League configuration registry.

Each league is defined as a dict containing structural parameters, API
identifiers, data paths, and feature-availability flags.  Every pipeline
component that previously hard-coded PL values should read them from here
via ``get_league_config()``.
"""
from __future__ import annotations

import os
from typing import Any

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════════════════════════════════════════
# League definitions
# ═══════════════════════════════════════════════════════════════════════════════

LEAGUES: dict[str, dict[str, Any]] = {

    # ── Premier League ──────────────────────────────────────────────────────
    "PL": {
        "name": "Premier League",
        "country": "England",

        # Structure
        "teams_count": 20,
        "games_per_season": 38,
        "max_points": 114,           # 38 * 3

        # Position thresholds (1-indexed)
        "relegation_pos": 18,        # 18th-20th go down
        "euro_pos": 7,               # Top 7 qualify for Europe (approx)
        "promotion_pos": None,       # N/A for top flight
        "playoff_range": None,

        # API identifiers
        "odds_api_sport": "soccer_epl",
        "oddspapi_tournament": "premier-league",
        "football_data_code": "PL",
        "football_data_co_uk_div": "E0",

        # Data sources
        "csv_path": os.path.join(_PROJECT_DIR, "CompleteDSPL_CSV.csv"),
        "enriched_csv_path": os.path.join(_PROJECT_DIR, "CompleteDSPL_enriched.csv"),
        "db_path": os.path.join(_PROJECT_DIR, "data", "dashboard.db"),

        # Feature availability
        "has_fpl": True,
        "has_understat": True,
        "understat_league": "EPL",
        "has_fbref": True,
        "fbref_comp_id": 9,          # FBref competition ID for PL
        "has_detailed_match": True,   # matches2425.csv available

        # Derby definitions — canonical CSV names, matched exactly by
        # data/build_canonical_dataset.py. These reproduce the derby flags
        # already in the PL canonical, so a rebuild is value-neutral.
        "derbies_local": {
            ("Arsenal FC", "Tottenham Hotspur FC"),
            ("Aston Villa FC", "Birmingham City FC"),
            ("Brighton & Hove Albion FC", "Crystal Palace FC"),
            ("Chelsea FC", "Fulham FC"),
            ("Everton FC", "Liverpool FC"),
            ("Manchester City FC", "Manchester United FC"),
            ("Newcastle United FC", "Sunderland AFC"),
            ("West Bromwich Albion FC", "Wolverhampton Wanderers FC"),
        },
        "derbies_historical": {
            ("Arsenal FC", "Chelsea FC"),
            ("Arsenal FC", "Manchester United FC"),
            ("Chelsea FC", "Liverpool FC"),
            ("Chelsea FC", "Manchester United FC"),
            ("Liverpool FC", "Manchester City FC"),
            ("Liverpool FC", "Manchester United FC"),
        },

        # Historical seasons available (SeasonIndex range)
        "first_season_idx": 0,       # 2000/01
        "current_season_idx": 25,    # 2025/26
    },

    # ── English Championship ────────────────────────────────────────────────
    "EFL": {
        "name": "English Championship",
        "country": "England",

        # Structure
        "teams_count": 24,
        "games_per_season": 46,
        "max_points": 138,           # 46 * 3

        # Position thresholds (1-indexed)
        "relegation_pos": 22,        # 22nd-24th go down
        "euro_pos": None,            # No European qualification
        "promotion_pos": 2,          # Top 2 auto-promote
        "playoff_range": (3, 6),     # 3rd-6th enter playoffs

        # API identifiers
        "odds_api_sport": "soccer_efl_champ",
        "oddspapi_tournament": "championship",
        "football_data_code": "ELC",
        "football_data_co_uk_div": "E1",

        # Data sources
        "csv_path": os.path.join(_PROJECT_DIR, "CompleteDSChamp_CSV.csv"),
        "enriched_csv_path": os.path.join(_PROJECT_DIR, "CompleteDSChamp_enriched.csv"),
        "db_path": os.path.join(_PROJECT_DIR, "data", "dashboard_efl.db"),

        # Feature availability
        "has_fpl": False,            # FPL only covers PL
        "has_understat": False,      # Understat only covers top 5 leagues
        "understat_league": None,
        "has_fbref": True,
        "fbref_comp_id": 10,         # FBref competition ID for Championship
        "has_detailed_match": False,  # No matches2425.csv equivalent yet

        # Derby definitions — football-data.co.uk short forms, exactly as the
        # EFL canonical stores them ("Sheffield Weds", not "Sheffield
        # Wednesday"). The previous long-form list here was read by no code
        # and named 18 clubs that do not exist in the canonical, so wiring it
        # up would have silently zeroed most EFL derby flags.
        "derbies_local": {
            ("Sheffield Weds", "Sheffield United"),
            # "Nott'm Forest" with the apostrophe — the canonical's own form.
            # Configured as "Nottm Forest" this matched nothing for 33
            # East Midlands derbies across 17 seasons, and fuzzy matching did
            # not save it either: the apostrophe defeats substring matching in
            # both directions, which is why fuzzy and exact agreed.
            ("Nott'm Forest", "Derby"),
            ("Leeds", "Sheffield United"),
            ("Sunderland", "Middlesbrough"),
            ("Bristol City", "Cardiff"),
            ("Norwich", "Ipswich"),
            ("Burnley", "Blackburn"),
            ("QPR", "Millwall"),
            ("Watford", "Luton"),
        },
        "derbies_historical": {
            ("West Brom", "Wolves"),
            ("Leeds", "Sheffield Weds"),
            ("Preston", "Blackpool"),
            ("Stoke", "Port Vale"),
            ("Coventry", "Leicester"),
            ("Hull", "Sheffield United"),
            ("Derby", "Leicester"),
            ("Sunderland", "Newcastle"),
            ("Huddersfield", "Leeds"),
            ("Birmingham", "West Brom"),
            ("Charlton", "Millwall"),
        },

        # Historical seasons available
        "first_season_idx": 0,       # 2000/01 (football-data.co.uk)
        "current_season_idx": 25,    # 2025/26
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def get_league_config(league: str | None = None) -> dict[str, Any]:
    """Return the config dict for the given league key.

    Args:
        league: League key (e.g. ``"PL"``, ``"EFL"``).  If *None*, reads
                 from the ``ACTIVE_LEAGUE`` environment variable (default ``"PL"``).

    Returns:
        League configuration dictionary.

    Raises:
        KeyError: If the league key is not registered.
    """
    if league is None:
        league = os.environ.get("ACTIVE_LEAGUE", "PL")
    if league not in LEAGUES:
        raise KeyError(
            f"Unknown league '{league}'. Available: {sorted(LEAGUES.keys())}"
        )
    return LEAGUES[league]


def list_leagues() -> list[str]:
    """Return sorted list of registered league keys."""
    return sorted(LEAGUES.keys())
