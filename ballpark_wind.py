"""
ballpark_wind.py
Precise wind effect modeling using actual stadium CF orientations.
A 10 mph tailwind to CF is worth ~15% more HRs. A headwind cuts them.
"""

import math
import requests
from datetime import datetime, timedelta


# ── Stadium CF Orientations ───────────────────────────────────────────────────
# Direction (degrees) a ball travels when hit to center field.
# 0° = North, 90° = East, 180° = South, 270° = West.
# If wind blows FROM this direction, it's a headwind (bad for HRs).
# If wind blows TOWARD this direction, it's a tailwind (great for HRs).

STADIUM_CF_ORIENTATION = {
    "COL": 335,   # Coors — CF points NNW
    "CIN": 20,    # GABP — CF points NNE
    "PHI": 75,    # Citizens Bank — CF points ENE
    "NYY": 60,    # Yankee Stadium — CF points ENE
    "BOS": 90,    # Fenway — CF points E (tricky)
    "TEX": 20,    # Globe Life — CF points NNE
    "MIL": 10,    # AmFam Field — CF points N
    "BAL": 330,   # Camden — CF points NNW
    "ATL": 35,    # Truist — CF points NNE
    "CHC": 135,   # Wrigley — CF points SE (famous wind direction)
    "HOU": 30,    # Minute Maid — retractable, less wind effect
    "TOR": 0,     # Rogers Centre — dome
    "MIN": 355,   # Target Field — CF points N
    "LAA": 30,    # Angel Stadium — CF points NNE
    "CLE": 20,    # Progressive — CF points NNE
    "DET": 10,    # Comerica — CF points N
    "WSH": 15,    # Nationals Park — CF points NNE
    "STL": 25,    # Busch — CF points NNE
    "NYM": 65,    # Citi Field — CF points ENE
    "ARI": 0,     # Chase Field — retractable dome
    "TBR": 0,     # Tropicana — dome
    "KCR": 15,    # Kauffman — CF points NNE
    "CWS": 350,   # Guaranteed Rate — CF points N
    "PIT": 330,   # PNC — CF points NNW (Allegheny River behind)
    "MIA": 0,     # loanDepot — retractable dome
    "SFG": 270,   # Oracle — CF points W (infamous wind)
    "LAD": 295,   # Dodger Stadium — CF points WNW
    "OAK": 290,   # CF points WNW
    "SEA": 330,   # T-Mobile — retractable
    "SDP": 305,   # Petco — CF points WNW
}

# Dome/retractable roof stadiums (wind has no effect)
DOME_STADIUMS = {"TOR", "TBR", "ARI", "MIA", "SEA", "HOU", "MIL"}

# Stadium coordinates (lat, lon)
STADIUM_COORDS = {
    "COL": (39.7559, -104.9942),
    "CIN": (39.0979, -84.5082),
    "PHI": (39.9061, -75.1665),
    "NYY": (40.8296, -73.9262),
    "BOS": (42.3467, -71.0972),
    "TEX": (32.7473, -97.0824),
    "MIL": (43.0280, -87.9712),
    "BAL": (39.2838, -76.6218),
    "ATL": (33.8908, -84.4678),
    "CHC": (41.9484, -87.6553),
    "HOU": (29.7573, -95.3555),
    "TOR": (43.6414, -79.3894),
    "MIN": (44.9817, -93.2775),
    "LAA": (33.8003, -117.8827),
    "CLE": (41.4962, -81.6852),
    "DET": (42.3390, -83.0485),
    "WSH": (38.8730, -77.0074),
    "STL": (38.6226, -90.1928),
    "NYM": (40.7571, -73.8458),
    "ARI": (33.4453, -112.0667),
    "TBR": (27.7682, -82.6534),
    "KCR": (39.0517, -94.4803),
    "CWS": (41.8299, -87.6338),
    "PIT": (40.4469, -80.0058),
    "MIA": (25.7781, -80.2197),
    "SFG": (37.7786, -122.3893),
    "LAD": (34.0739, -118.2400),
    "OAK": (37.7516, -122.2005),
    "SEA": (47.5914, -122.3325),
    "SDP": (32.7076, -117.1570),
}


def precise_wind_boost(wind_speed_mph: float, wind_from_deg: float,
                        team: str, temp_f: float = 72) -> dict:
    """
    Compute precise wind effect on HR probability for a given stadium.

    wind_from_deg: meteorological convention — direction wind is COMING FROM.
    So wind_from=180 means wind blowing FROM south, TOWARD north.

    Returns dict with:
        wind_component_mph: positive=tailwind, negative=headwind
        wind_boost: HR probability multiplier adjustment (-0.05 to +0.08)
        wind_label: human-readable description
    """
    if team in DOME_STADIUMS:
        return {
            "wind_component_mph": 0,
            "wind_boost": 0,
            "wind_label": "Dome — no wind effect",
            "temp_boost": _temp_boost(temp_f),
        }

    cf_orientation = STADIUM_CF_ORIENTATION.get(team, 0)

    # Wind blows FROM wind_from_deg, so it travels TOWARD (wind_from_deg + 180) % 360
    wind_to_deg = (wind_from_deg + 180) % 360

    # Component of wind in the direction of the CF
    angle_diff = math.radians(wind_to_deg - cf_orientation)
    wind_component = wind_speed_mph * math.cos(angle_diff)

    # ~10 mph direct tailwind = +8% HR rate boost (research-backed estimate)
    # ~10 mph headwind = -5% HR rate
    if wind_component >= 0:
        boost = (wind_component / 10) * 0.08
    else:
        boost = (wind_component / 10) * 0.05

    # Label
    if wind_component > 7:
        label = f"Strong tailwind ({wind_component:.1f} mph out)"
    elif wind_component > 3:
        label = f"Light tailwind ({wind_component:.1f} mph out)"
    elif wind_component < -7:
        label = f"Strong headwind ({abs(wind_component):.1f} mph in)"
    elif wind_component < -3:
        label = f"Light headwind ({abs(wind_component):.1f} mph in)"
    else:
        label = f"Crosswind ({wind_speed_mph:.1f} mph)"

    return {
        "wind_component_mph": round(wind_component, 1),
        "wind_boost": round(boost, 4),
        "wind_label": label,
        "temp_boost": _temp_boost(temp_f),
    }


def _temp_boost(temp_f: float) -> float:
    """
    Temperature effect on HR probability.
    Ball travels further in warm air (lower density).
    Baseline = 72°F. ~1% boost per 10°F above baseline.
    """
    return ((temp_f - 72) / 10) * 0.01


def fetch_weather_precise(team: str, game_date: str, game_hour_local: int = 19) -> dict:
    """
    Fetch weather for a game using Open-Meteo, then compute precise wind boost.
    game_hour_local: local hour of first pitch (default 7pm = 19)
    """
    if team not in STADIUM_COORDS:
        return _default_weather(team)

    lat, lon = STADIUM_COORDS[team]
    today_dt = __import__("datetime").datetime.today()
    game_dt = __import__("datetime").datetime.strptime(game_date, "%Y-%m-%d")

    try:
        if game_dt.date() <= today_dt.date():
            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={game_date}&end_date={game_date}"
                f"&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation"
                f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto"
            )
            r = requests.get(url, timeout=10)
            data = r.json()
            hourly = data.get("hourly", {})
            idx = min(game_hour_local, len(hourly.get("temperature_2m", [72])) - 1)

            temp_f = hourly["temperature_2m"][idx]
            wind_mph = hourly["windspeed_10m"][idx]
            wind_from = hourly["winddirection_10m"][idx]
            precip = hourly["precipitation"][idx]
        else:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,windspeed_10m,winddirection_10m,precipitation"
                f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto"
                f"&start_date={game_date}&end_date={game_date}"
            )
            r = requests.get(url, timeout=10)
            data = r.json()
            hourly = data.get("hourly", {})
            idx = min(game_hour_local, len(hourly.get("temperature_2m", [72])) - 1)

            temp_f = hourly["temperature_2m"][idx]
            wind_mph = hourly["windspeed_10m"][idx]
            wind_from = hourly["winddirection_10m"][idx]
            precip = hourly["precipitation"][idx]

        wind_data = precise_wind_boost(wind_mph, wind_from, team, temp_f)

        return {
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_from_deg": wind_from,
            "precip_mm": precip,
            **wind_data,
        }

    except Exception as e:
        print(f"  Weather error for {team} on {game_date}: {e}")
        return _default_weather(team)


def _default_weather(team: str) -> dict:
    """Default neutral weather when API fails."""
    return {
        "temp_f": 72,
        "wind_mph": 5,
        "wind_from_deg": 180,
        "precip_mm": 0,
        "wind_component_mph": 0,
        "wind_boost": 0,
        "wind_label": "Unknown",
        "temp_boost": 0,
    }
