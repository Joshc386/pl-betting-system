"""
Weather data fetcher using Open-Meteo API (free, no key required).
Fetches historical weather for PL match days at home stadium locations.
Caches results to data/weather_cache.csv to avoid repeated API calls.
"""
import os
import time
import json
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "weather_cache.csv")

# Stadium coordinates for all PL teams (2000/01 - 2025/26)
# Using team names as they appear in the CSV (with "FC" suffix)
STADIUM_COORDS = {
    # Current PL teams (2025/26)
    "Arsenal FC": (51.555, -0.108),           # Emirates Stadium
    "Aston Villa FC": (52.509, -1.885),       # Villa Park
    "AFC Bournemouth": (50.735, -1.838),      # Vitality Stadium
    "Brentford FC": (51.491, -0.289),         # Gtech Community Stadium
    "Brighton & Hove Albion FC": (50.862, -0.084),  # Amex Stadium
    "Chelsea FC": (51.482, -0.191),           # Stamford Bridge
    "Crystal Palace FC": (51.398, -0.086),    # Selhurst Park
    "Everton FC": (53.439, -2.966),           # Goodison Park
    "Fulham FC": (51.475, -0.222),            # Craven Cottage
    "Ipswich Town FC": (52.055, 1.145),       # Portman Road
    "Leicester City FC": (52.620, -1.142),    # King Power Stadium
    "Liverpool FC": (53.431, -2.961),         # Anfield
    "Manchester City FC": (53.483, -2.200),   # Etihad Stadium
    "Manchester United FC": (53.463, -2.291), # Old Trafford
    "Newcastle United FC": (54.976, -1.622),  # St James' Park
    "Nottingham Forest FC": (52.940, -1.133), # City Ground
    "Southampton FC": (50.906, -1.391),       # St Mary's Stadium
    "Tottenham Hotspur FC": (51.604, -0.066), # Tottenham Hotspur Stadium
    "West Ham United FC": (51.539, -0.017),   # London Stadium
    "Wolverhampton Wanderers FC": (52.590, -2.130),  # Molineux

    # Historic PL teams (relegated/promoted over 2000-2025)
    "Leeds United FC": (53.778, -1.572),      # Elland Road
    "Sunderland AFC": (54.914, -1.388),       # Stadium of Light
    "West Bromwich Albion FC": (52.509, -1.964),  # The Hawthorns
    "Stoke City FC": (52.988, -2.175),        # bet365 Stadium
    "Swansea City AFC": (51.643, -3.935),     # Liberty Stadium
    "Hull City FC": (53.746, -0.368),         # MKM Stadium
    "Burnley FC": (53.789, -2.230),           # Turf Moor
    "Watford FC": (51.650, -0.402),           # Vicarage Road
    "Sheffield United FC": (53.370, -1.471),  # Bramall Lane
    "Norwich City FC": (52.622, 1.309),       # Carrow Road
    "Luton Town FC": (51.884, -0.432),        # Kenilworth Road
    "Middlesbrough FC": (54.578, -1.217),     # Riverside Stadium
    "Blackburn Rovers FC": (53.729, -2.489),  # Ewood Park
    "Bolton Wanderers FC": (53.581, -2.536),  # Reebok/Toughsheet Stadium
    "Charlton Athletic FC": (51.487, 0.036),  # The Valley
    "Birmingham City FC": (52.476, -1.868),   # St Andrew's
    "Portsmouth FC": (50.796, -1.064),        # Fratton Park
    "Wigan Athletic FC": (53.548, -2.654),    # DW Stadium
    "Reading FC": (51.422, -0.983),           # Madejski Stadium
    "Derby County FC": (52.915, -1.447),      # Pride Park
    "Blackpool FC": (53.805, -3.048),         # Bloomfield Road
    "Queens Park Rangers FC": (51.509, -0.232),  # Loftus Road
    "Cardiff City FC": (51.473, -3.203),      # Cardiff City Stadium
    "Sheffield Wednesday FC": (53.411, -1.501),  # Hillsborough
    "Huddersfield Town AFC": (53.654, -1.768),  # John Smith's Stadium
    "Coventry City FC": (52.448, -1.496),     # Coventry Building Society Arena
    "Bradford City AFC": (53.804, -1.759),    # Valley Parade
    "Oldham Athletic FC": (53.555, -2.129),   # Boundary Park
    "Oxford United FC": (51.716, -1.209),     # Kassam Stadium
}

# Default coords (central England) for any team not in the mapping
DEFAULT_COORDS = (52.5, -1.5)


def fetch_weather_for_dates(lat, lon, start_date, end_date):
    """
    Fetch daily weather from Open-Meteo historical API.
    Returns list of {date, temperature, precipitation, wind_speed}.
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,precipitation_sum,wind_speed_10m_max"
        f"&timezone=Europe/London"
    )
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        data = json.loads(resp.read())
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        temps = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])
        wind = daily.get("wind_speed_10m_max", [])

        results = []
        for i, d in enumerate(dates):
            results.append({
                "date": d,
                "temperature": temps[i] if i < len(temps) else None,
                "precipitation": precip[i] if i < len(precip) else None,
                "wind_speed": wind[i] if i < len(wind) else None,
            })
        return results
    except Exception as e:
        print(f"  Weather API error for ({lat},{lon}) {start_date}-{end_date}: {e}")
        return []


def build_weather_cache(df):
    """
    Build weather cache for all matches in the dataset.
    Fetches weather per (home_team, season) in batches to minimize API calls.
    Saves to data/weather_cache.csv.
    """
    # Load existing cache if available
    if os.path.exists(CACHE_PATH):
        existing = pd.read_csv(CACHE_PATH)
        cached_keys = set(zip(existing["date"], existing["home_team"]))
        print(f"  Existing weather cache: {len(existing)} entries")
    else:
        existing = pd.DataFrame()
        cached_keys = set()

    # Find matches that need weather data
    needed = []
    for _, row in df.iterrows():
        date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])[:10]
        key = (date_str, row["Home_Team"])
        if key not in cached_keys:
            needed.append(row)

    if not needed:
        print("  Weather cache is up to date.")
        return existing

    print(f"  Need weather for {len(needed)} matches...")

    # Group by (home_team, season) for batch fetching
    needed_df = pd.DataFrame(needed)
    new_rows = []

    # Group by team to batch API calls per team per date range
    for team in needed_df["Home_Team"].unique():
        team_matches = needed_df[needed_df["Home_Team"] == team].sort_values("Date")
        coords = STADIUM_COORDS.get(team, DEFAULT_COORDS)

        # Batch by year to minimize API calls
        for year, year_group in team_matches.groupby(team_matches["Date"].dt.year):
            dates = year_group["Date"].sort_values()
            start = dates.iloc[0].strftime("%Y-%m-%d")
            end = dates.iloc[-1].strftime("%Y-%m-%d")

            weather_data = fetch_weather_for_dates(coords[0], coords[1], start, end)
            weather_lookup = {w["date"]: w for w in weather_data}

            for _, match_row in year_group.iterrows():
                date_str = match_row["Date"].strftime("%Y-%m-%d")
                w = weather_lookup.get(date_str, {})
                new_rows.append({
                    "date": date_str,
                    "home_team": match_row["Home_Team"],
                    "temperature": w.get("temperature"),
                    "precipitation": w.get("precipitation"),
                    "wind_speed": w.get("wind_speed"),
                })

            time.sleep(0.5)  # Rate limiting: ~2 req/sec (Open-Meteo free tier)

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(
        subset=["date", "home_team"], keep="last"
    )
    combined.to_csv(CACHE_PATH, index=False)
    print(f"  Weather cache updated: {len(combined)} total entries ({len(new_rows)} new)")
    return combined


def load_weather_cache():
    """Load weather cache from disk."""
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame()


def get_live_weather(team_name):
    """
    Get current weather for a team's stadium (for live predictions).
    Uses Open-Meteo forecast API.
    """
    coords = STADIUM_COORDS.get(team_name, DEFAULT_COORDS)
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={coords[0]}&longitude={coords[1]}"
        f"&current=temperature_2m,precipitation,wind_speed_10m"
        f"&timezone=Europe/London"
    )
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        data = json.loads(resp.read())
        current = data.get("current", {})
        return {
            "temperature": current.get("temperature_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
        }
    except Exception:
        return {"temperature": None, "precipitation": None, "wind_speed": None}
