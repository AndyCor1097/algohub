"""
build_zone_maps.py — One-time zone map builder
Pulls full 2025 Statcast season and builds zone HR maps for all batters/pitchers.
Saves to data/zone_maps.pkl for use in algohub.py

Run once:
  python build_zone_maps.py

Takes ~5-10 minutes. After that algohub loads from disk instantly.
"""

import pybaseball as pb
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

pb.cache.enable()

ZONE_MAPS_PATH = "data/zone_maps.pkl"

# Pitch type groups
FASTBALL_TYPES = {"FF", "SI", "FC"}
BREAKING_TYPES = {"SL", "CU", "KC", "CS", "SV", "ST"}
OFFSPEED_TYPES = {"CH", "FS", "FO"}


def calc_barrel(ev, la):
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return 0
    if ev >= 116:
        return int(8 <= la <= 50)
    min_la = 26 - (116 - ev)
    max_la = 30 + (116 - ev)
    return int(min_la <= la <= max_la)


def build_zone_maps(raw: pd.DataFrame) -> dict:
    """Build zone HR maps for all batters and pitchers."""
    print("  Processing batted balls...")
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    raw["is_hr"] = (raw["events"] == "home_run").fillna(False).astype("int8")

    batted = raw[raw["type"] == "X"].copy()

    # Fix dtypes
    batted["batter"]  = batted["batter"].astype("int64")
    batted["pitcher"] = batted["pitcher"].astype("int64")
    raw["batter"]     = raw["batter"].astype("int64")
    raw["pitcher"]    = raw["pitcher"].astype("int64")

    # Barrel
    if "launch_speed" in batted.columns and "launch_angle" in batted.columns:
        ls = pd.to_numeric(batted["launch_speed"], errors="coerce")
        la = pd.to_numeric(batted["launch_angle"], errors="coerce")
        batted["launch_speed"] = ls
        batted["launch_angle"] = la
        batted["barrel"] = batted.apply(
            lambda r: calc_barrel(r.get("launch_speed"), r.get("launch_angle")), axis=1
        )
    else:
        batted["barrel"] = 0

    # Hard hit
    if "launch_speed" in batted.columns:
        batted["hard_hit"] = (pd.to_numeric(batted["launch_speed"], errors="coerce") >= 95).fillna(False).astype("int8")

    # Sweet spot
    if "launch_angle" in batted.columns:
        la_num = pd.to_numeric(batted["launch_angle"], errors="coerce")
        batted["sweet_spot"] = ((la_num >= 8) & (la_num <= 32)).fillna(False).astype("int8")

    # ── Batter zone maps ──────────────────────────────────────────────────────
    print("  Building batter zone maps...")
    batter_zones = {}
    batter_pt_barrels = {}

    for pid, grp in batted.groupby("batter"):
        games = raw[raw["batter"] == pid]["game_date"].nunique()

        # Zone HR map
        zone_hrs = {}
        if "zone" in grp.columns:
            for z, zgrp in grp[grp["is_hr"] == 1].groupby("zone"):
                try:
                    zone_hrs[int(z)] = int(len(zgrp))
                except:
                    pass

        # Pitch type barrel rates
        pt_barrels = {}
        if "pitch_type" in grp.columns:
            for pt_group, types in [("fb", FASTBALL_TYPES), ("brk", BREAKING_TYPES), ("off", OFFSPEED_TYPES)]:
                sub = grp[grp["pitch_type"].isin(types)]
                if len(sub) >= 10:
                    pt_barrels[pt_group] = {
                        "barrel_rate": float(sub["barrel"].mean()) if not pd.isna(sub["barrel"].mean()) else 0,
                        "hr_rate":     float(sub["is_hr"].sum() / max(games, 1)),
                        "avg_ev":      float(pd.to_numeric(sub["launch_speed"], errors="coerce").fillna(0).mean()),
                        "n":           len(sub),
                    }

        # FB% and HR/FB
        fb_rate = 0.0
        hr_fb_rate = 0.0
        if "bb_type" in grp.columns:
            fb_rate = float(grp["bb_type"].isin(["fly_ball"]).mean())
            fly_balls = grp[grp["bb_type"] == "fly_ball"]
            if len(fly_balls) >= 5:
                hr_fb_rate = float(fly_balls["is_hr"].mean())

        batter_zones[int(pid)] = {
            "zone_hrs":    zone_hrs,
            "pt_barrels":  pt_barrels,
            "games":       int(games),
            "bip":         len(grp),
            "hr_count":    int(grp["is_hr"].sum()),
            "barrel_rate": float(grp["barrel"].mean()) if not pd.isna(grp["barrel"].mean()) else 0,
            "hard_hit":    float(grp["hard_hit"].mean()) if "hard_hit" in grp.columns and not pd.isna(grp["hard_hit"].mean()) else 0,
            "sweet_spot":  float(grp["sweet_spot"].mean()) if "sweet_spot" in grp.columns and not pd.isna(grp["sweet_spot"].mean()) else 0,
            "fb_rate":     fb_rate,
            "hr_fb_rate":  hr_fb_rate,
            "avg_ev":      float(pd.to_numeric(grp["launch_speed"], errors="coerce").fillna(0).mean()) if "launch_speed" in grp.columns else 0,
            "avg_la":      float(pd.to_numeric(grp["launch_angle"], errors="coerce").fillna(0).mean()) if "launch_angle" in grp.columns else 0,
        }

    print(f"  Built zone maps for {len(batter_zones)} batters")

    # ── Pitcher zone maps ─────────────────────────────────────────────────────
    print("  Building pitcher zone maps...")
    pitcher_zones = {}

    for pid, grp in batted.groupby("pitcher"):
        pitcher_raw = raw[raw["pitcher"] == pid]
        games = pitcher_raw["game_date"].nunique()

        # Zone HR allowed map
        zone_hrs_allowed = {}
        if "zone" in grp.columns:
            for z, zgrp in grp[grp["is_hr"] == 1].groupby("zone"):
                try:
                    zone_hrs_allowed[int(z)] = int(len(zgrp))
                except:
                    pass

        # Pitch mix
        pitch_mix = {}
        primary_pitch = None
        primary_vel   = 0
        if "pitch_type" in pitcher_raw.columns:
            pt_counts = pitcher_raw["pitch_type"].value_counts(normalize=True)
            for pt, pct in pt_counts.head(6).items():
                if pd.notna(pt) and pt != "":
                    sub = pitcher_raw[pitcher_raw["pitch_type"] == pt]
                    avg_vel = float(pd.to_numeric(sub["release_speed"], errors="coerce").mean()) if "release_speed" in sub.columns else 0
                    hr_sub  = grp[grp["pitch_type"] == pt]
                    pitch_mix[str(pt)] = {
                        "pct":     round(float(pct), 3),
                        "avg_vel": round(avg_vel, 1) if not pd.isna(avg_vel) else 0,
                        "hr_rate": float(hr_sub["is_hr"].sum() / max(games, 1)),
                    }
            if len(pt_counts) > 0:
                primary_pitch = str(pt_counts.index[0])
                if primary_pitch in pitch_mix:
                    primary_vel = pitch_mix[primary_pitch]["avg_vel"]

        # Velocity band
        if primary_vel >= 96:   vel_band = "elite"
        elif primary_vel >= 92: vel_band = "hard"
        elif primary_vel >= 88: vel_band = "med"
        else:                   vel_band = "soft"

        # SwStr%
        swstr = 0
        if "description" in pitcher_raw.columns:
            swings = pitcher_raw[pitcher_raw["description"].isin(
                ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
            )]
            swstr = len(swings) / max(len(pitcher_raw), 1)

        fb_alw = grp[grp["bb_type"].isin(["fly_ball","popup"])].shape[0] if "bb_type" in grp.columns else 1

        pitcher_zones[int(pid)] = {
            "zone_hrs_allowed": zone_hrs_allowed,
            "pitch_mix":        pitch_mix,
            "primary_pitch":    primary_pitch,
            "primary_vel":      primary_vel,
            "vel_band":         vel_band,
            "swstr_rate":       round(swstr, 3),
            "hr_allowed":       int(grp["is_hr"].sum()),
            "hr_per_fb":        float(grp["is_hr"].sum() / max(fb_alw, 1)),
            "barrel_allowed":   float(grp["barrel"].mean()) if not pd.isna(grp["barrel"].mean()) else 0,
            "hard_hit_allowed": float(grp["hard_hit"].mean()) if "hard_hit" in grp.columns and not pd.isna(grp["hard_hit"].mean()) else 0,
            "avg_ev_allowed":   float(pd.to_numeric(grp["launch_speed"], errors="coerce").fillna(0).mean()) if "launch_speed" in grp.columns else 0,
            "bip":              len(grp),
            "games":            int(games),
        }

    print(f"  Built zone maps for {len(pitcher_zones)} pitchers")

    return {
        "batter_zones":  batter_zones,
        "pitcher_zones": pitcher_zones,
        "built_at":      datetime.now().isoformat(),
        "season":        2025,
    }


def main():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  AlgoHub Zone Map Builder — 2025 Season")
    print("=" * 60)

    # Pull full 2025 season
    print("\nPulling 2025 Statcast data (full season)...")
    print("This will take 5-10 minutes...\n")

    try:
        raw = pb.statcast("2025-03-20", "2025-10-01")
        print(f"\n  Got {len(raw):,} pitch records")
        print(f"  Date range: {raw['game_date'].min()} → {raw['game_date'].max()}")
    except Exception as e:
        print(f"ERROR pulling Statcast: {e}")
        return

    # Build maps
    print("\nBuilding zone maps...")
    maps = build_zone_maps(raw)

    # Save
    print(f"\nSaving to {ZONE_MAPS_PATH}...")
    with open(ZONE_MAPS_PATH, "wb") as f:
        pickle.dump(maps, f)

    size_mb = os.path.getsize(ZONE_MAPS_PATH) / 1024 / 1024
    print(f"  Saved {size_mb:.1f} MB")
    print(f"  Batters: {len(maps['batter_zones']):,}")
    print(f"  Pitchers: {len(maps['pitcher_zones']):,}")
    print("\n✓ Zone maps ready. Launch algohub.py to use them.")


if __name__ == "__main__":
    main()
