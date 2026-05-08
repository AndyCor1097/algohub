"""
data_collection.py
Pulls 2025 season data entirely from Baseball Savant via pybaseball.
No FanGraphs, no Baseball Reference — just Statcast which never blocks.
Handles missing columns (barrel, launch_speed, etc.) gracefully.
Saves progress every 10 batters so restarts pick up where they left off.
"""

import pandas as pd
import numpy as np
import pybaseball as pb
import requests
import time
import os
from datetime import datetime

pb.cache.enable()

SEASON = 2025
SEASON_START = f"{SEASON}-03-20"
SEASON_END = f"{SEASON}-10-01"

# ── Park Factors (2026) ───────────────────────────────────────────────────────
# HR park factors — 1.0 = league average, updated for 2026 changes:
# KCR: fences moved in 9-10 ft for 2026, was 0.85 HR factor historically, now ~1.02
# TBR: returned to Tropicana Field (same dimensions as pre-2025)
PARK_FACTORS = {
    "COL": 1.22,  "CIN": 1.15,  "PHI": 1.12,  "NYY": 1.10,
    "BOS": 1.08,  "TEX": 1.07,  "MIL": 1.06,  "BAL": 1.05,
    "ATL": 1.04,  "CHC": 1.03,  "HOU": 1.02,  "KCR": 1.02,
    "TOR": 1.01,  "MIN": 1.00,  "LAA": 1.00,  "CLE": 0.99,
    "DET": 0.98,  "WSH": 0.98,  "STL": 0.97,  "NYM": 0.97,
    "ARI": 0.97,  "TBR": 0.96,  "CWS": 0.96,  "PIT": 0.95,
    "MIA": 0.94,  "SFG": 0.93,  "LAD": 0.93,  "OAK": 0.92,
    "SEA": 0.91,  "SDP": 0.90,
}

# ── Stadium Coordinates ───────────────────────────────────────────────────────
STADIUM_COORDS = {
    "COL": (39.7559, -104.9942), "CIN": (39.0979, -84.5082),
    "PHI": (39.9061, -75.1665),  "NYY": (40.8296, -73.9262),
    "BOS": (42.3467, -71.0972),  "TEX": (32.7473, -97.0824),
    "MIL": (43.0280, -87.9712),  "BAL": (39.2838, -76.6218),
    "ATL": (33.8908, -84.4678),  "CHC": (41.9484, -87.6553),
    "HOU": (29.7573, -95.3555),  "TOR": (43.6414, -79.3894),
    "MIN": (44.9817, -93.2775),  "LAA": (33.8003, -117.8827),
    "CLE": (41.4962, -81.6852),  "DET": (42.3390, -83.0485),
    "WSH": (38.8730, -77.0074),  "STL": (38.6226, -90.1928),
    "NYM": (40.7571, -73.8458),  "ARI": (33.4453, -112.0667),
    "TBR": (27.7682, -82.6534),  "KCR": (39.0517, -94.4803),
    "CWS": (41.8299, -87.6338),  "PIT": (40.4469, -80.0058),
    "MIA": (25.7781, -80.2197),  "SFG": (37.7786, -122.3893),
    "LAD": (34.0739, -118.2400), "OAK": (37.7516, -122.2005),
    "SEA": (47.5914, -122.3325), "SDP": (32.7076, -117.1570),
}


# ── Weather ───────────────────────────────────────────────────────────────────

def fetch_weather(team_abbr: str, game_date: str) -> dict:
    """Fetch weather for a stadium via Open-Meteo (free, no API key)."""
    default = {"temp_f": 72, "wind_mph": 5, "wind_deg": 180, "precip_mm": 0}
    if team_abbr not in STADIUM_COORDS:
        return default
    lat, lon = STADIUM_COORDS[team_abbr]
    date_obj = datetime.strptime(game_date, "%Y-%m-%d")
    today = datetime.today()
    try:
        if date_obj.date() <= today.date():
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
            idx = min(19, len(hourly.get("temperature_2m", [72])) - 1)
            return {
                "temp_f": hourly["temperature_2m"][idx],
                "wind_mph": hourly["windspeed_10m"][idx],
                "wind_deg": hourly["winddirection_10m"][idx],
                "precip_mm": hourly["precipitation"][idx],
            }
        else:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&daily=temperature_2m_max,windspeed_10m_max,winddirection_10m_dominant,precipitation_sum"
                f"&temperature_unit=fahrenheit&windspeed_unit=mph&timezone=auto"
                f"&start_date={game_date}&end_date={game_date}"
            )
            r = requests.get(url, timeout=10)
            data = r.json()
            daily = data.get("daily", {})
            return {
                "temp_f": daily["temperature_2m_max"][0],
                "wind_mph": daily["windspeed_10m_max"][0],
                "wind_deg": daily["winddirection_10m_dominant"][0],
                "precip_mm": daily["precipitation_sum"][0],
            }
    except Exception:
        return default


def wind_hr_boost(wind_mph: float, wind_deg: float, stadium_orientation: float = 90) -> float:
    import math
    diff = abs(wind_deg - stadium_orientation) % 360
    if diff > 180:
        diff = 360 - diff
    return math.cos(math.radians(diff)) * wind_mph * 0.005


# ── Safe column helpers ───────────────────────────────────────────────────────

def safe_mean(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Return mean of a column if it exists and has data, else default."""
    if col in df.columns and len(df) > 0:
        val = df[col].mean()
        return float(val) if not pd.isna(val) else default
    return default


def safe_rate(df: pd.DataFrame, col: str, condition, default: float = 0.0) -> float:
    """Return fraction of rows meeting condition if column exists, else default."""
    if col in df.columns and len(df) > 0:
        return float((condition(df[col])).mean())
    return default


def safe_sum(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    """Return sum of a column if it exists, else default."""
    if col in df.columns and len(df) > 0:
        val = df[col].sum()
        return float(val) if not pd.isna(val) else default
    return default


# ── Get player list from Statcast leaderboard ─────────────────────────────────

def get_qualified_batters(season: int, min_bbe: int = 20) -> pd.DataFrame:
    """
    Pull Statcast exit velocity / barrel leaderboard to get qualified hitters.
    This is a Baseball Savant endpoint — never blocks.
    """
    print(f"Pulling Statcast batting leaderboard for {season}...")
    try:
        df = pb.statcast_batter_exitvelo_barrels(season, minBBE=min_bbe)
        print(f"  Got {len(df)} qualified batters")
        return df
    except Exception as e:
        print(f"  Leaderboard pull failed: {e}")
        return pd.DataFrame()


# ── Per-player game logs ──────────────────────────────────────────────────────

def fetch_batter_game_logs(player_id: int, season: int) -> pd.DataFrame:
    """
    Pull all Statcast PAs for a batter and roll up to game level.
    Each row = one player-game with HR outcome + Statcast metrics.
    Handles missing columns (barrel, launch_speed, etc.) safely.
    """
    start = f"{season}-03-20"
    end = f"{season}-10-01"

    df = pb.statcast_batter(start, end, player_id=player_id)
    if df.empty:
        return pd.DataFrame()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["hr_flag"] = (df["events"] == "home_run").astype(int)

    # ── Season-level aggregates (computed safely) ─────────────────────────────
    batted = df[df["type"] == "X"].copy()
    total_batted = len(batted)
    fly_balls = batted[batted["bb_type"].isin(["fly_ball", "popup"])] if "bb_type" in batted.columns else pd.DataFrame()
    hrs = batted[batted["events"] == "home_run"] if "events" in batted.columns else pd.DataFrame()

    season_barrel_rate  = safe_mean(batted, "barrel")
    season_hard_hit     = safe_rate(batted, "launch_speed", lambda x: x >= 95)
    season_avg_ev       = safe_mean(batted, "launch_speed")
    season_avg_la       = safe_mean(batted, "launch_angle")
    season_hr_fb        = len(hrs) / len(fly_balls) if len(fly_balls) > 0 else 0
    season_fb_rate      = len(fly_balls) / total_batted if total_batted > 0 else 0
    season_total_hr     = len(hrs)
    season_pa           = df["at_bat_number"].nunique() if "at_bat_number" in df.columns else len(df)

    # ── Game-level aggregation ────────────────────────────────────────────────
    agg_dict = {
        "pa": ("at_bat_number", "nunique"),
        "hr": ("hr_flag", "max"),
        "avg_ev": ("launch_speed", "mean"),
        "avg_la": ("launch_angle", "mean"),
    }

    # Only include barrel if the column exists
    if "barrel" in df.columns:
        agg_dict["barrel_count"] = ("barrel", "sum")

    # Only include launch_speed hard hit if column exists
    group_cols = ["game_date", "home_team", "away_team"]
    missing_group = [c for c in group_cols if c not in df.columns]
    if missing_group:
        return pd.DataFrame()

    try:
        game_logs = df.groupby(group_cols).agg(**agg_dict).reset_index()
    except Exception as e:
        return pd.DataFrame()

    # Fill barrel_count with 0 if it wasn't computed
    if "barrel_count" not in game_logs.columns:
        game_logs["barrel_count"] = 0

    # Fill missing ev/la with 0
    game_logs["avg_ev"] = game_logs["avg_ev"].fillna(0)
    game_logs["avg_la"] = game_logs["avg_la"].fillna(0)

    # Attach identifiers and season aggregates
    game_logs["player_id"]           = player_id
    game_logs["season"]              = season
    game_logs["season_barrel_rate"]  = season_barrel_rate
    game_logs["season_hard_hit_rate"]= season_hard_hit
    game_logs["season_avg_ev"]       = season_avg_ev
    game_logs["season_avg_la"]       = season_avg_la
    game_logs["season_hr_fb_rate"]   = season_hr_fb
    game_logs["season_fb_rate"]      = season_fb_rate
    game_logs["season_total_hr"]     = season_total_hr
    game_logs["season_pa"]           = season_pa

    return game_logs


# ── Master Build ──────────────────────────────────────────────────────────────

def build_training_data(
    season: int = SEASON,
    top_n_batters: int = 200,
    output_path: str = "data/training_data.csv"
) -> pd.DataFrame:
    """
    Build full training dataset from Baseball Savant only.
    Saves progress every 10 batters — safe to restart anytime.
    """
    os.makedirs("data", exist_ok=True)

    season_cache  = f"data/season_{season}.csv"
    progress_path = "data/progress_log.csv"

    # Already fully done
    if os.path.exists(season_cache):
        print(f"Season {season} already complete — loading from cache")
        df = pd.read_csv(season_cache)
        df.to_csv(output_path, index=False)
        print(f"Loaded {len(df)} records. Done!")
        return df

    # Get player list
    leaderboard = get_qualified_batters(season)
    if leaderboard.empty:
        print("Could not get player list. Exiting.")
        return pd.DataFrame()

    # Find PA / attempts column for sorting
    pa_col = next((c for c in ["pa", "PA", "attempts"] if c in leaderboard.columns), None)
    leaderboard = (
        leaderboard.nlargest(top_n_batters, pa_col)
        if pa_col else leaderboard.head(top_n_batters)
    )

    # Find player ID column
    id_col = next((c for c in ["player_id", "batter", "IDfg"] if c in leaderboard.columns), None)
    if id_col is None:
        print(f"Cannot find player ID column. Available: {list(leaderboard.columns)}")
        return pd.DataFrame()

    print(f"\nProcessing {len(leaderboard)} batters...\n")

    # Resume from checkpoint if it exists
    done_ids = set()
    existing_logs = []
    if os.path.exists(progress_path):
        prev = pd.read_csv(progress_path)
        done_ids = set(prev["player_id"].unique())
        existing_logs.append(prev)
        remaining = len(leaderboard) - len(done_ids)
        print(f"Resuming — {len(done_ids)} done, {remaining} remaining\n")

    new_logs = []

    for j, (_, row) in enumerate(leaderboard.iterrows()):
        player_id = row.get(id_col)
        if pd.isna(player_id):
            continue
        player_id = int(player_id)

        if player_id in done_ids:
            continue

        player_name = row.get("last_name, first_name", row.get("name", str(player_id)))

        try:
            game_logs = fetch_batter_game_logs(player_id, season)

            if not game_logs.empty:
                # Attach authoritative barrel rate from leaderboard
                # (overrides the per-PA calculation which can be 0 for missing barrel column)
                if "brl_percent" in row and not pd.isna(row["brl_percent"]):
                    game_logs["season_barrel_rate"] = row["brl_percent"] / 100
                if "ev95percent" in row and not pd.isna(row["ev95percent"]):
                    game_logs["season_hard_hit_rate"] = row["ev95percent"] / 100
                if "avg_hit_speed" in row and not pd.isna(row["avg_hit_speed"]):
                    game_logs["season_avg_ev"] = row["avg_hit_speed"]

                # Also attach other leaderboard columns
                for col in ["avg_distance", "avg_hr_distance", "brl_pa"]:
                    if col in row and not pd.isna(row[col]):
                        game_logs[f"lb_{col}"] = row[col]

                game_logs["batter_name"] = player_name
                new_logs.append(game_logs)

            done_ids.add(player_id)

            # Progress + checkpoint save every 10 batters
            if len(done_ids) % 10 == 0:
                print(f"  {len(done_ids)}/{len(leaderboard)} batters done...")
                if new_logs:
                    chunk = pd.concat(existing_logs + new_logs, ignore_index=True)
                    chunk["home_team"] = chunk["home_team"].str.upper()
                    chunk["park_factor"] = chunk["home_team"].map(PARK_FACTORS).fillna(1.0)
                    chunk.to_csv(progress_path, index=False)

            time.sleep(0.5)

        except Exception as e:
            print(f"  Skipping {player_name} ({player_id}): {e}")
            done_ids.add(player_id)
            continue

    # Final combine
    all_logs = existing_logs + new_logs
    if not all_logs:
        print("No data collected!")
        return pd.DataFrame()

    df = pd.concat(all_logs, ignore_index=True)
    df["home_team"] = df["home_team"].str.upper()
    df["park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(1.0)

    print(f"\n{'='*40}")
    print(f"Total player-game records : {len(df)}")
    print(f"Unique players            : {df['player_id'].nunique()}")
    print(f"HR rate                   : {df['hr'].mean():.3f}")
    print(f"{'='*40}")

    df.to_csv(season_cache, index=False)
    df.to_csv(output_path, index=False)

    if os.path.exists(progress_path):
        os.remove(progress_path)

    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    print(f"Starting data collection for {SEASON} season...")
    print("Source: Baseball Savant only — no FanGraphs, no B-Ref\n")
    df = build_training_data(
        season=SEASON,
        top_n_batters=400,
        output_path="data/training_data.csv"
    )
    print(f"\nDone. Shape: {df.shape}")
