"""
pitch_mix.py
Pulls pitcher Statcast pitch mix data — pitch types, velocities, usage rates.
Slider-heavy and fastball-up-in-zone pitchers give up significantly more HRs.
"""

import pybaseball as pb
import pandas as pd
import numpy as np
import os

CACHE_PATH = "data/pitch_mix_cache.csv"


def fetch_pitcher_statcast(pitcher_id: int, season: int) -> pd.DataFrame:
    """Pull all Statcast pitches thrown by a pitcher in a season."""
    start = f"{season}-03-20"
    end = f"{season}-11-01"
    try:
        df = pb.statcast_pitcher(start, end, player_id=pitcher_id)
        return df
    except Exception as e:
        print(f"    Statcast pitcher fetch failed for {pitcher_id}: {e}")
        return pd.DataFrame()


def compute_pitch_mix_features(df: pd.DataFrame) -> dict:
    """
    Compute pitch mix HR-relevant features from raw Statcast pitch data.

    Key signals:
    - High slider% → more HRs (sliders that hang get crushed)
    - High 4-seam FB% thrown up in zone → more HRs
    - Low avg velo → more HRs
    - High spin rate (certain pitches) → can be either way
    """
    if df.empty:
        return {}

    total = len(df)

    # Pitch type groups
    fb_types = ["FF", "SI", "FC"]      # fastballs
    breaking = ["SL", "CU", "KC", "CS"]  # breaking balls
    offspeed = ["CH", "FS", "FO"]        # changeups/splitters

    fb_df = df[df["pitch_type"].isin(fb_types)]
    sl_df = df[df["pitch_type"] == "SL"]
    cu_df = df[df["pitch_type"].isin(["CU", "KC"])]
    ch_df = df[df["pitch_type"].isin(["CH", "FS"])]

    # Zone-based: upper third of strike zone (zone 1,2,3 in Statcast)
    upper_zone = df[df["zone"].isin([1, 2, 3])]
    fb_up = fb_df[fb_df["zone"].isin([1, 2, 3])]

    # Barrel / hard hit allowed
    batted = df[df["type"] == "X"].copy()
    hr_pitches = df[df["events"] == "home_run"]

    features = {
        # Usage rates
        "pitch_fb_pct": len(fb_df) / total if total else 0,
        "pitch_sl_pct": len(sl_df) / total if total else 0,
        "pitch_cu_pct": len(cu_df) / total if total else 0,
        "pitch_ch_pct": len(ch_df) / total if total else 0,

        # Velocity
        "pitch_avg_fb_velo": fb_df["release_speed"].mean() if len(fb_df) > 0 else 92,
        "pitch_avg_velo": df["release_speed"].mean() if total > 0 else 90,

        # Location tendencies
        "pitch_fb_up_zone_pct": len(fb_up) / max(len(fb_df), 1),
        "pitch_upper_zone_pct": len(upper_zone) / total if total else 0,

        # Results
        "pitch_barrel_rate_allowed": (
            batted["barrel"].sum() / len(batted)
            if len(batted) > 0 and "barrel" in batted.columns else 0
        ),
        "pitch_hard_hit_rate_allowed": (
            len(batted[batted["launch_speed"] >= 95]) / len(batted)
            if len(batted) > 0 and "launch_speed" in batted.columns else 0
        ),
        "pitch_hr_per_9_statcast": len(hr_pitches) / max(total / 27, 0.1),

        # Spin (high-spin sliders hang more)
        "pitch_avg_sl_spin": sl_df["release_spin_rate"].mean() if len(sl_df) > 10 else 2400,

        # Total pitches (sample size indicator)
        "pitch_sample_size": total,
    }
    return features


def build_pitcher_pitch_mix(pitcher_ids: list, season: int,
                             use_cache: bool = True) -> pd.DataFrame:
    """
    Build a DataFrame of pitch mix features for all pitchers.
    """
    if use_cache and os.path.exists(CACHE_PATH):
        cached = pd.read_csv(CACHE_PATH)
        cached_season = cached[cached["season"] == season] if "season" in cached.columns else cached
        cached_ids = set(cached_season["pitcher_id"].tolist()) if "pitcher_id" in cached_season.columns else set()
        missing = [pid for pid in pitcher_ids if pid not in cached_ids]

        if not missing:
            print(f"  Pitch mix: all {len(pitcher_ids)} pitchers loaded from cache")
            return cached_season
    else:
        missing = pitcher_ids
        cached_season = pd.DataFrame()

    print(f"  Fetching Statcast pitch mix for {len(missing)} pitchers...")
    rows = []
    for i, pid in enumerate(missing):
        print(f"    [{i+1}/{len(missing)}] pitcher {pid}")
        df = fetch_pitcher_statcast(pid, season)
        features = compute_pitch_mix_features(df)
        if features:
            features["pitcher_id"] = pid
            features["season"] = season
            rows.append(features)

    if rows:
        new_df = pd.DataFrame(rows)
        if not cached_season.empty:
            result = pd.concat([cached_season, new_df], ignore_index=True)
        else:
            result = new_df
        result.to_csv(CACHE_PATH, index=False)
        return result

    return cached_season if not cached_season.empty else pd.DataFrame()


def hr_risk_score(pitch_mix_row: pd.Series) -> float:
    """
    Compute a composite 'pitcher HR risk' score from pitch mix features.
    Higher = more HR-prone pitcher.
    Weights derived from empirical HR correlation research.
    """
    score = 0.0

    # Slider-heavy pitchers are more HR prone
    score += pitch_mix_row.get("pitch_sl_pct", 0.20) * 1.5

    # Fastballs up in zone
    score += pitch_mix_row.get("pitch_fb_up_zone_pct", 0.30) * 1.2

    # Low velocity = more HRs
    avg_velo = pitch_mix_row.get("pitch_avg_fb_velo", 93)
    score += max(0, (93 - avg_velo) / 10) * 0.8

    # Barrel rate allowed
    score += pitch_mix_row.get("pitch_barrel_rate_allowed", 0.06) * 3.0

    # Normalize to 0–1
    return min(max(score / 3.0, 0), 1)
