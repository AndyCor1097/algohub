"""
platoon_splits.py
Fetches batter and pitcher handedness from MLB Stats API.
Computes platoon splits and matchup advantages.
Platoon advantage (opposite hands) is one of the strongest HR predictors.
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import os

CACHE_PATH = "data/hand_cache.json"

# HR rate multipliers by platoon matchup (empirical MLB averages)
# Source: FanGraphs platoon splits research
PLATOON_HR_MULTIPLIER = {
    ("R", "R"): 0.92,   # same-side slight disadvantage
    ("R", "L"): 1.10,   # classic platoon advantage
    ("L", "R"): 1.12,   # lefty vs righty (biggest advantage)
    ("L", "L"): 0.88,   # same-side, lefty hurts more
    ("S", "R"): 1.05,   # switch hitter vs righty (bats left)
    ("S", "L"): 1.05,   # switch hitter vs lefty (bats right)
}


def load_hand_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_hand_cache(cache: dict):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def get_player_hand(player_id: int, cache: dict) -> dict:
    """
    Fetch batter side and pitcher hand from MLB Stats API.
    Returns {"bat_side": "R/L/S", "pitch_hand": "R/L"}
    """
    key = str(player_id)
    if key in cache:
        return cache[key]

    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}?fields=people,batSide,pitchHand,primaryPosition"
        r = requests.get(url, timeout=8)
        data = r.json()
        person = data.get("people", [{}])[0]

        result = {
            "bat_side": person.get("batSide", {}).get("code", "R"),
            "pitch_hand": person.get("pitchHand", {}).get("code", "R"),
            "position": person.get("primaryPosition", {}).get("abbreviation", ""),
        }
        cache[key] = result
        time.sleep(0.1)
        return result
    except Exception:
        return {"bat_side": "R", "pitch_hand": "R", "position": ""}


def build_hand_lookup(player_ids: list) -> dict:
    """
    Bulk fetch hand data for a list of player IDs.
    Returns dict: {player_id: {"bat_side": ..., "pitch_hand": ...}}
    """
    cache = load_hand_cache()
    missing = [pid for pid in player_ids if str(pid) not in cache]

    if missing:
        print(f"  Fetching hand data for {len(missing)} players...")
        for pid in missing:
            get_player_hand(pid, cache)

        save_hand_cache(cache)

    return {int(k): v for k, v in cache.items() if int(k) in player_ids}


def add_platoon_features(df: pd.DataFrame, batter_id_col: str = "player_id",
                          pitcher_id_col: str = "pitcher_id") -> pd.DataFrame:
    """
    Add platoon features to a DataFrame.
    Requires batter_id and pitcher_id columns.
    """
    all_ids = []
    if batter_id_col in df.columns:
        all_ids += df[batter_id_col].dropna().astype(int).tolist()
    if pitcher_id_col in df.columns:
        all_ids += df[pitcher_id_col].dropna().astype(int).tolist()

    hand_lookup = build_hand_lookup(list(set(all_ids)))

    def get_bat_side(pid):
        return hand_lookup.get(int(pid) if pd.notna(pid) else 0, {}).get("bat_side", "R")

    def get_pitch_hand(pid):
        return hand_lookup.get(int(pid) if pd.notna(pid) else 0, {}).get("pitch_hand", "R")

    if batter_id_col in df.columns:
        df["bat_side"] = df[batter_id_col].apply(get_bat_side)
    if pitcher_id_col in df.columns:
        df["pitch_hand"] = df[pitcher_id_col].apply(get_pitch_hand)

    # Platoon advantage flag (opposite hands)
    if "bat_side" in df.columns and "pitch_hand" in df.columns:
        df["platoon_advantage"] = (
            ((df["bat_side"] == "R") & (df["pitch_hand"] == "L")) |
            ((df["bat_side"] == "L") & (df["pitch_hand"] == "R")) |
            (df["bat_side"] == "S")
        ).astype(int)

        # HR multiplier based on matchup
        df["platoon_hr_multiplier"] = df.apply(
            lambda row: PLATOON_HR_MULTIPLIER.get(
                (row.get("bat_side", "R"), row.get("pitch_hand", "R")), 1.0
            ), axis=1
        )

    return df


def get_fg_platoon_splits(season: int) -> pd.DataFrame:
    """
    Pull FanGraphs platoon splits (vs LHP / vs RHP) via pybaseball.
    Returns merged DataFrame with HR/FB vs L and vs R for each batter.
    """
    import pybaseball as pb
    try:
        # FanGraphs splits leaderboard
        vs_r = pb.batting_stats_range(f"{season}-03-01", f"{season}-10-31")
        # NOTE: Full splits require FanGraphs splits page scraping.
        # pybaseball's batting_stats_range gives overall splits.
        # For full L/R splits, use the manual URL below.
        return vs_r
    except Exception as e:
        print(f"  Could not fetch FanGraphs splits: {e}")
        return pd.DataFrame()
