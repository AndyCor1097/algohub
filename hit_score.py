"""
hit_score.py — AlgoHub HIT Score Engine
HR Intelligence Target Score — proprietary metric

Combines:
- Zone crush rate (batter HR rate in pitcher's primary zones)
- Pitch type barrel rate (batter barrel% vs pitcher's primary pitch)
- Velocity matchup (batter performance vs pitch velocity band)
- Platoon edge
- Pitcher HR exposure
- Park + weather environment
- Hot bat form
- Value score (HIT Score vs market implied probability)

Zero extra API calls — all computed from bulk Statcast data.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Pitch Type Groups ──────────────────────────────────────────────────────────
FASTBALL_TYPES  = {"FF", "SI", "FC"}
BREAKING_TYPES  = {"SL", "CU", "KC", "CS", "SV", "ST"}
OFFSPEED_TYPES  = {"CH", "FS", "FO"}

# Velocity bands
VEL_ELITE  = 96   # 96+ mph
VEL_HARD   = 92   # 92-95
VEL_MED    = 88   # 88-91
VEL_SOFT   = 0    # <88

# Statcast zone map (1-9 strike zone, 11-14 chase)
ZONE_MAP = {
    1:(0,0), 2:(0,1), 3:(0,2),
    4:(1,0), 5:(1,1), 6:(1,2),
    7:(2,0), 8:(2,1), 9:(2,2),
}

# HR multipliers by platoon matchup
PLATOON_MULT = {
    ("R","R"): 0.92, ("R","L"): 1.10,
    ("L","R"): 1.12, ("L","L"): 0.88,
    ("S","R"): 1.05, ("S","L"): 1.05,
}


def calc_barrel(ev, la):
    """Statcast barrel definition."""
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return 0
    if ev >= 116:
        return int(8 <= la <= 50)
    min_la = 26 - (116 - ev)
    max_la = 30 + (116 - ev)
    return int(min_la <= la <= max_la)


class HITScoreEngine:
    """
    Proprietary HIT Score calculator.
    Initialize once with bulk Statcast data, query per matchup.
    """

    def __init__(self, raw_df: pd.DataFrame, zone_maps_path: str = "data/zone_maps.pkl"):
        print("  Building HIT Score engine...")
        self.raw = raw_df.copy()
        self._prepare()
        self._build_indexes()

        # Merge 2025 historical zone maps if available
        self._merge_historical_zones(zone_maps_path)

        print(f"  Engine ready: {len(self._batter_index)} batters, {len(self._pitcher_index)} pitchers")

    def _merge_historical_zones(self, path: str):
        """
        Load 2025 zone maps and merge with 2026 data.
        2026 data takes priority for recent form, but 2025 fills in zone gaps.
        """
        import pickle, os
        if not os.path.exists(path):
            print("  No historical zone maps found — run build_zone_maps.py for better ZF scores")
            return

        try:
            with open(path, "rb") as f:
                historical = pickle.load(f)

            hist_batters  = historical.get("batter_zones", {})
            hist_pitchers = historical.get("pitcher_zones", {})
            season        = historical.get("season", 2025)

            # Merge batter zones — add historical zone_hrs where 2026 is sparse
            merged_b = 0
            for pid, hist_data in hist_batters.items():
                if pid in self._batter_index:
                    curr = self._batter_index[pid]
                    # Merge zone_hrs: combine 2025 + 2026 counts
                    curr_zones = curr.get("zone_hrs", {})
                    hist_zones = hist_data.get("zone_hrs", {})
                    merged_zones = {}
                    all_zones = set(curr_zones.keys()) | set(hist_zones.keys())
                    for z in all_zones:
                        merged_zones[z] = curr_zones.get(z, 0) + hist_zones.get(z, 0)
                    curr["zone_hrs"] = merged_zones

                    # Merge pt_barrels — average 2025 + 2026 if both exist
                    curr_pt = curr.get("pt_barrels", {})
                    hist_pt = hist_data.get("pt_barrels", {})
                    for pt_group in set(list(curr_pt.keys()) + list(hist_pt.keys())):
                        if pt_group in curr_pt and pt_group in hist_pt:
                            # Weighted average — 2026 gets 60%, 2025 gets 40%
                            curr_pt[pt_group]["barrel_rate"] = (
                                curr_pt[pt_group]["barrel_rate"] * 0.6 +
                                hist_pt[pt_group]["barrel_rate"] * 0.4
                            )
                        elif pt_group in hist_pt and pt_group not in curr_pt:
                            curr_pt[pt_group] = hist_pt[pt_group]
                    curr["pt_barrels"] = curr_pt
                    merged_b += 1
                else:
                    # Player not in 2026 yet — add from historical
                    self._batter_index[pid] = hist_data

            # Merge pitcher zones
            merged_p = 0
            for pid, hist_data in hist_pitchers.items():
                if pid in self._pitcher_index:
                    curr = self._pitcher_index[pid]
                    curr_zones = curr.get("zone_hrs_allowed", {})
                    hist_zones = hist_data.get("zone_hrs_allowed", {})
                    merged_zones = {}
                    all_zones = set(curr_zones.keys()) | set(hist_zones.keys())
                    for z in all_zones:
                        merged_zones[z] = curr_zones.get(z, 0) + hist_zones.get(z, 0)
                    curr["zone_hrs_allowed"] = merged_zones

                    # If pitch mix empty in 2026, use 2025
                    if not curr.get("pitch_mix") and hist_data.get("pitch_mix"):
                        curr["pitch_mix"]     = hist_data["pitch_mix"]
                        curr["primary_pitch"] = hist_data["primary_pitch"]
                        curr["primary_vel"]   = hist_data["primary_vel"]
                        curr["vel_band"]      = hist_data["vel_band"]
                    merged_p += 1
                else:
                    self._pitcher_index[pid] = hist_data

            print(f"  Merged {season} zone maps: {merged_b} batters, {merged_p} pitchers enriched")

        except Exception as e:
            print(f"  Zone map merge failed: {e}")

    def _prepare(self):
        df = self.raw
        df["game_date"] = pd.to_datetime(df["game_date"])
        df["is_hr"] = (df["events"] == "home_run").fillna(False).astype("int8")

        # Batted ball events only
        batted = df[df["type"] == "X"].copy()
        batted["is_hr"] = batted["is_hr"].fillna(0).astype("int8")

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

        # Sweet spot (8-32 degrees)
        if "launch_angle" in batted.columns:
            la_num = pd.to_numeric(batted["launch_angle"], errors="coerce")
            batted["sweet_spot"] = ((la_num >= 8) & (la_num <= 32)).fillna(False).astype("int8")

        # Hard hit (95+ mph)
        if "launch_speed" in batted.columns:
            ls_num = pd.to_numeric(batted["launch_speed"], errors="coerce")
            batted["hard_hit"] = (ls_num >= 95).fillna(False).astype("int8")

        # Pulled ball
        if "hc_x" in batted.columns and "stand" in batted.columns:
            hc = pd.to_numeric(batted["hc_x"], errors="coerce")
            batted["pulled"] = (
                ((batted["stand"] == "R") & (hc < 125)) |
                ((batted["stand"] == "L") & (hc > 125))
            ).fillna(False).astype("int8")
        else:
            batted["pulled"] = 0

        # Velocity band
        if "release_speed" in df.columns:
            df["vel_band"] = pd.cut(
                df["release_speed"].fillna(88),
                bins=[0, 88, 92, 96, 120],
                labels=["soft", "med", "hard", "elite"]
            )

        self.batted = batted
        self.raw = df

    def _build_indexes(self):
        """Pre-build per-batter and per-pitcher indexes for fast lookup."""

        # ── Batter index ──────────────────────────────────────────────────────
        self._batter_index = {}
        batted_clean = self.batted.copy()
        batted_clean["batter"] = batted_clean["batter"].astype("int64")
        raw_clean = self.raw.copy()
        raw_clean["batter"] = raw_clean["batter"].astype("int64")

        for pid, grp in batted_clean.groupby("batter"):
            games = raw_clean[raw_clean["batter"] == pid]["game_date"].nunique()
            hrs   = grp["is_hr"].sum()
            batter_all = raw_clean[raw_clean["batter"] == pid]

            # SwStr% — swing and miss rate (lower = better contact = more HRs)
            swstr_rate = 0.0
            if "description" in batter_all.columns:
                swings     = batter_all[batter_all["description"].isin(["swinging_strike","swinging_strike_blocked","foul_tip"])]
                total_p    = len(batter_all)
                swstr_rate = float(len(swings) / max(total_p, 1))

            # SwStr% by pitch type — key matchup signal
            swstr_by_pt = {}
            if "description" in batter_all.columns and "pitch_type" in batter_all.columns:
                for pt_group, types in [("fb", FASTBALL_TYPES), ("brk", BREAKING_TYPES), ("off", OFFSPEED_TYPES)]:
                    sub = batter_all[batter_all["pitch_type"].isin(types)]
                    if len(sub) >= 10:
                        sw = sub[sub["description"].isin(["swinging_strike","swinging_strike_blocked","foul_tip"])]
                        swstr_by_pt[pt_group] = float(len(sw) / max(len(sub), 1))

            # Zone HR map
            zone_hrs = {}
            if "zone" in grp.columns:
                for z, zgrp in grp.groupby("zone"):
                    zone_hrs[int(z)] = int(zgrp["is_hr"].sum())

            # Pitch type barrel rates
            pt_barrels = {}
            if "pitch_type" in grp.columns:
                for pt_group, types in [("fb", FASTBALL_TYPES), ("brk", BREAKING_TYPES), ("off", OFFSPEED_TYPES)]:
                    sub = grp[grp["pitch_type"].isin(types)]
                    if len(sub) >= 5:
                        pt_barrels[pt_group] = {
                            "barrel_rate": float(sub["barrel"].mean()),
                            "hr_rate":     float(sub["is_hr"].sum() / max(games, 1)),
                            "avg_ev":      float(sub["launch_speed"].mean()) if "launch_speed" in sub.columns else 0,
                            "n":           len(sub),
                        }

            # Velocity matchup
            vel_hrs = {}
            if "vel_band" in self.raw.columns:
                batter_raw = self.raw[self.raw["batter"] == pid]
                for band in ["soft", "med", "hard", "elite"]:
                    sub = batter_raw[batter_raw["vel_band"] == band]
                    if len(sub) >= 5:
                        vel_hrs[band] = float(sub["is_hr"].sum() / max(games, 1))

            # Launch angle consistency (15-35 degree sweet spot — wider HR window)
            la_consistency = 0.0
            if "launch_angle" in grp.columns:
                la_num = pd.to_numeric(grp["launch_angle"], errors="coerce")
                la_consistency = float(((la_num >= 15) & (la_num <= 35)).fillna(False).mean())

            # ── Rolling 7-day trending stats ──────────────────────────────────
            # Compare last 7 days vs full sample to detect heating up / cooling down
            cutoff_7 = grp["game_date"].max() - pd.Timedelta(days=7) if "game_date" in grp.columns else None
            if cutoff_7 is not None:
                recent = grp[grp["game_date"] >= cutoff_7]
            else:
                recent = grp.tail(10)

            # Recent barrel rate (last 7 days)
            barrel_7 = float(recent["barrel"].mean()) if len(recent) >= 3 else float(grp["barrel"].mean())

            # Recent hard hit % (last 7 days)
            hh_7 = float(recent["hard_hit"].mean()) if "hard_hit" in recent.columns and len(recent) >= 3 else float(grp.get("hard_hit", pd.Series([0])).mean())

            # Recent xwOBA (last 7 days)
            xwoba_7 = None
            if "estimated_woba_using_speedangle" in recent.columns and len(recent) >= 3:
                xwoba_7 = float(pd.to_numeric(recent["estimated_woba_using_speedangle"], errors="coerce").fillna(0).mean())

            # Recent EV (last 7 days)
            ev_7 = float(pd.to_numeric(recent["launch_speed"], errors="coerce").fillna(0).mean()) if "launch_speed" in recent.columns and len(recent) >= 3 else 0

            # Season averages for comparison
            barrel_season = float(grp["barrel"].mean())
            hh_season     = float(grp["hard_hit"].mean()) if "hard_hit" in grp.columns else 0
            xwoba_season  = float(pd.to_numeric(grp["estimated_woba_using_speedangle"], errors="coerce").fillna(0).mean()) if "estimated_woba_using_speedangle" in grp.columns else 0.300
            ev_season     = float(pd.to_numeric(grp["launch_speed"], errors="coerce").fillna(0).mean()) if "launch_speed" in grp.columns else 85

            # Trend scores — positive = heating up, negative = cooling down
            # Scale so +0.10 = 10% above avg
            barrel_trend = (barrel_7 - barrel_season) / max(barrel_season, 0.01)
            hh_trend     = (hh_7 - hh_season) / max(hh_season, 0.01)
            xwoba_trend  = ((xwoba_7 or xwoba_season) - xwoba_season) / max(xwoba_season, 0.01)
            ev_trend     = (ev_7 - ev_season) / max(ev_season, 0.01)

            # Composite heat score — weighted average of trends
            heat_score = (
                barrel_trend * 0.35 +
                hh_trend     * 0.25 +
                xwoba_trend  * 0.25 +
                ev_trend     * 0.15
            )
            heat_score = round(float(heat_score), 3)

            # Fly ball % — direct from bb_type
            fb_rate = 0.0
            if "bb_type" in grp.columns:
                fb_balls = grp["bb_type"].isin(["fly_ball"])
                fb_rate  = float(fb_balls.mean())

            # HR/FB — how often fly balls become HRs
            hr_fb_rate = 0.0
            if "bb_type" in grp.columns:
                fly_balls = grp[grp["bb_type"] == "fly_ball"]
                if len(fly_balls) >= 5:
                    hr_fb_rate = float(fly_balls["is_hr"].mean())

            # ISO proxy — from Statcast estimated SLG - estimated BA
            # Use woba as proxy since we don't have direct SLG/BA
            # Instead calculate from HR rate and hard hit — higher = more ISO
            iso_proxy = float(hrs / max(len(grp), 1)) * 4 + float(grp["barrel"].mean()) * 2

            self._batter_index[int(pid)] = {
                "games":          int(games),
                "bip":            len(grp),
                "hr_count":       int(hrs),
                "hr_rate":        float(hrs / max(games, 1)),
                "barrel_rate":    float(grp["barrel"].mean()),
                "sweet_spot":     float(grp["sweet_spot"].mean()) if "sweet_spot" in grp.columns else 0,
                "la_consistency": la_consistency,
                "heat_score":     heat_score,
                "barrel_7":       round(barrel_7 * 100, 1),
                "hh_7":           round(hh_7 * 100, 1),
                "xwoba_7":        round(xwoba_7, 3) if xwoba_7 else None,
                "ev_7":           round(ev_7, 1),
                "barrel_trend":   round(barrel_trend, 3),
                "hh_trend":       round(hh_trend, 3),
                "fb_rate":        fb_rate,
                "hr_fb_rate":     hr_fb_rate,
                "iso_proxy":      round(iso_proxy, 3),
                "swstr_rate":     swstr_rate,
                "swstr_by_pt":    swstr_by_pt,
                "hard_hit":       float(grp["hard_hit"].mean()) if "hard_hit" in grp.columns else 0,
                "avg_ev":         float(pd.to_numeric(grp["launch_speed"], errors="coerce").fillna(0).mean()) if "launch_speed" in grp.columns else 0,
                "avg_la":         float(pd.to_numeric(grp["launch_angle"], errors="coerce").fillna(0).mean()) if "launch_angle" in grp.columns else 0,
                "pulled_rate":    float(grp["pulled"].mean()) if "pulled" in grp.columns else 0,
                "xwoba":          float(pd.to_numeric(grp["estimated_woba_using_speedangle"], errors="coerce").fillna(0).mean()) if "estimated_woba_using_speedangle" in grp.columns else None,
                "zone_hrs":       zone_hrs,
                "pt_barrels":     pt_barrels,
                "vel_hrs":        vel_hrs,
            }

        # ── Pitcher index ─────────────────────────────────────────────────────
        self._pitcher_index = {}
        pitcher_clean = self.batted.copy()
        pitcher_clean["pitcher"] = pitcher_clean["pitcher"].astype("int64")
        pitcher_raw_clean = self.raw.copy()
        pitcher_raw_clean["pitcher"] = pitcher_raw_clean["pitcher"].astype("int64")

        for pid, grp in pitcher_clean.groupby("pitcher"):
            pitcher_raw = pitcher_raw_clean[pitcher_raw_clean["pitcher"] == pid]
            games  = pitcher_raw["game_date"].nunique()
            hr_alw = grp["is_hr"].sum()
            fb_alw = grp[grp["bb_type"].isin(["fly_ball","popup"])].shape[0] if "bb_type" in grp.columns else 1

            # Zone HR allowed map
            zone_hrs_allowed = {}
            if "zone" in grp.columns:
                for z, zgrp in grp.groupby("zone"):
                    zone_hrs_allowed[int(z)] = int(zgrp["is_hr"].sum())

            # Pitch mix
            pitch_mix = {}
            primary_pitch = None
            primary_vel   = 0
            if "pitch_type" in pitcher_raw.columns:
                pt_counts = pitcher_raw["pitch_type"].value_counts(normalize=True)
                for pt, pct in pt_counts.head(5).items():
                    if pd.notna(pt) and pt != "":
                        sub = pitcher_raw[pitcher_raw["pitch_type"] == pt]
                        avg_vel = float(sub["release_speed"].mean()) if "release_speed" in sub.columns else 0
                        pitch_mix[str(pt)] = {
                            "pct":     round(float(pct), 3),
                            "avg_vel": round(avg_vel, 1),
                            "hr_rate": float(grp[grp["pitch_type"] == pt]["is_hr"].sum() / max(games, 1)),
                        }
                if pt_counts.index[0] if len(pt_counts) > 0 else None:
                    primary_pitch = str(pt_counts.index[0])
                    if primary_pitch in pitch_mix:
                        primary_vel = pitch_mix[primary_pitch]["avg_vel"]

            # Primary velocity band
            vel_band = "med"
            if primary_vel >= VEL_ELITE:   vel_band = "elite"
            elif primary_vel >= VEL_HARD:  vel_band = "hard"
            elif primary_vel >= VEL_MED:   vel_band = "med"
            else:                           vel_band = "soft"

            # SwStr% (swing and miss)
            swstr = 0
            if "description" in pitcher_raw.columns:
                swings    = pitcher_raw[pitcher_raw["description"].isin(["swinging_strike","swinging_strike_blocked","foul_tip"])]
                all_pitches = len(pitcher_raw)
                swstr = len(swings) / max(all_pitches, 1)

            self._pitcher_index[int(pid)] = {
                "games":            int(games),
                "bip":              len(grp),
                "hr_allowed":       int(hr_alw),
                "hr_per_fb":        float(hr_alw / max(fb_alw, 1)),
                "barrel_allowed":   float(grp["barrel"].mean()),
                "hard_hit_allowed": float(grp["hard_hit"].mean()) if "hard_hit" in grp.columns else 0,
                "zone_hrs_allowed": zone_hrs_allowed,
                "pitch_mix":        pitch_mix,
                "primary_pitch":    primary_pitch,
                "primary_vel":      primary_vel,
                "vel_band":         vel_band,
                "swstr_rate":       round(swstr, 3),
                "avg_ev_allowed":   float(grp["launch_speed"].mean()) if "launch_speed" in grp.columns else 0,
            }

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_batter(self, batter_id: int) -> dict:
        return self._batter_index.get(int(batter_id), {})

    def get_pitcher(self, pitcher_id: int) -> dict:
        return self._pitcher_index.get(int(pitcher_id), {})

    def compute_zone_fit(self, batter_id: int, pitcher_id: int) -> dict:
        """
        Zone overlap — where pitcher lives vs where batter crushes.
        Returns zone_fit score (0-1) and zone_count (overlap zones).
        """
        b = self._batter_index.get(int(batter_id), {})
        p = self._pitcher_index.get(int(pitcher_id), {})

        bz = set(k for k,v in b.get("zone_hrs", {}).items() if v > 0)
        pz = set(k for k,v in p.get("zone_hrs_allowed", {}).items() if v > 0)

        if not bz or not pz:
            return {"zone_fit": 0.0, "zone_count": 0, "kill_zones": set()}

        overlap  = bz & pz
        union    = bz | pz
        fit      = len(overlap) / len(union) if union else 0

        return {
            "zone_fit":   round(fit, 3),
            "zone_count": len(overlap),
            "kill_zones": overlap,
            "batter_zones":  bz,
            "pitcher_zones": pz,
        }

    def compute_pitch_matchup(self, batter_id: int, pitcher_id: int) -> dict:
        """
        Pitch type matchup — batter barrel rate vs pitcher's primary pitches.
        This is the key edge signal.
        """
        b = self._batter_index.get(int(batter_id), {})
        p = self._pitcher_index.get(int(pitcher_id), {})

        if not b or not p:
            return {"pitch_matchup_score": 0.5, "primary_matchup": "unknown", "edge_pitch": None}

        pitch_mix  = p.get("pitch_mix", {})
        pt_barrels = b.get("pt_barrels", {})

        # Weight batter's barrel rate by pitcher's pitch usage
        fb_pct  = sum(v["pct"] for k,v in pitch_mix.items() if k in FASTBALL_TYPES)
        brk_pct = sum(v["pct"] for k,v in pitch_mix.items() if k in BREAKING_TYPES)
        off_pct = sum(v["pct"] for k,v in pitch_mix.items() if k in OFFSPEED_TYPES)

        fb_brl  = pt_barrels.get("fb",  {}).get("barrel_rate", 0.06)
        brk_brl = pt_barrels.get("brk", {}).get("barrel_rate", 0.06)
        off_brl = pt_barrels.get("off", {}).get("barrel_rate", 0.06)

        weighted_barrel = fb_pct * fb_brl + brk_pct * brk_brl + off_pct * off_brl

        # Velocity matchup bonus
        vel_band = p.get("vel_band", "med")
        vel_hrs  = b.get("vel_hrs", {})
        vel_bonus = vel_hrs.get(vel_band, 0) / max(b.get("hr_rate", 0.05), 0.01)
        vel_bonus = min(vel_bonus, 2.0)

        # Find edge pitch (where batter has biggest barrel rate vs pitcher's usage)
        edge_pitch = None
        edge_score = 0
        for pt_group, pt_data in pt_barrels.items():
            group_pct = {"fb": fb_pct, "brk": brk_pct, "off": off_pct}.get(pt_group, 0)
            score = pt_data["barrel_rate"] * group_pct
            if score > edge_score:
                edge_score = score
                edge_pitch = pt_group

        score = min(weighted_barrel / 0.12, 1.0) * 0.7 + min((vel_bonus - 1) * 0.5, 0.3)

        return {
            "pitch_matchup_score": round(score, 3),
            "weighted_barrel":     round(weighted_barrel, 3),
            "fb_pct":              round(fb_pct, 2),
            "brk_pct":             round(brk_pct, 2),
            "off_pct":             round(off_pct, 2),
            "fb_barrel":           round(fb_brl, 3),
            "brk_barrel":          round(brk_brl, 3),
            "off_barrel":          round(off_brl, 3),
            "vel_band":            vel_band,
            "vel_bonus":           round(vel_bonus, 2),
            "edge_pitch":          edge_pitch,
        }

    def compute_hit_score(
        self,
        batter_id:    int,
        pitcher_id:   int,
        bat_side:     str   = "R",
        pitch_hand:   str   = "R",
        park_factor:  float = 1.0,
        wind_boost:   float = 0.0,
        temp_f:       float = 70.0,
        batter_iso:   float = 0.15,
        pitcher_era:  float = 4.50,
        pitcher_hr9:  float = 1.10,
        pitcher_hrfb: float = 0.12,
        pitcher_hard: float = 0.35,
        hr_odds:      Optional[int] = None,
    ) -> dict:
        """
        Full HIT Score — 0 to 100.
        Weights:
          Barrel rate       20pts  — best single HR predictor
          Hard hit %        15pts  — sustained quality contact
          Fly ball %        10pts  — HR requires fly ball
          Exit velo         10pts  — raw power
          Pitcher HR vuln   20pts  — ERA + HR/9 + HR/FB
          Platoon edge      10pts  — handedness matchup
          Park + weather    10pts  — environment
          Hot bat            5pts  — recent form
        """
        b = self._batter_index.get(int(batter_id), {})
        p = self._pitcher_index.get(int(pitcher_id), {})
        zone  = self.compute_zone_fit(batter_id, pitcher_id)
        pitch = self.compute_pitch_matchup(batter_id, pitcher_id)

        # ── Batter signals ────────────────────────────────────────────────────
        # Thresholds set so league avg scores ~40-45, elite scores ~70-80

        # 1. Barrel rate (15pts) — elite = 18%+, avg = 8%
        barrel = b.get("barrel_rate", 0)
        barrel_score = min(max((barrel - 0.04) / 0.18, 0), 1.0) * 15

        # 2. Hard hit % (10pts) — elite = 55%+, avg = 38%
        hard_hit = b.get("hard_hit", 0)
        hh_score = min(max((hard_hit - 0.30) / 0.30, 0), 1.0) * 10

        # 3. xwOBA (10pts) — elite = .400+, avg = .310
        xwoba = b.get("xwoba") or 0.300
        xwoba_score = min(max((xwoba - 0.280) / 0.150, 0), 1.0) * 10

        # 4. Launch angle % (6pts) — elite = 40%+, avg = 25%
        la_cons = b.get("la_consistency", 0)
        la_score = min(max((la_cons - 0.15) / 0.30, 0), 1.0) * 6

        # 5. FB% (5pts) — elite = 45%+, avg = 30%
        fb_rate = b.get("fb_rate", 0)
        fb_score = min(max((fb_rate - 0.20) / 0.25, 0), 1.0) * 5

        # 6. HR/FB (7pts) — elite = 20%+, avg = 10%
        hr_fb = b.get("hr_fb_rate", 0)
        hrfb_score = min(max((hr_fb - 0.05) / 0.18, 0), 1.0) * 7

        # 7. Exit velo (5pts) — elite = 95mph+, avg = 88mph
        avg_ev = b.get("avg_ev", 85)
        ev_score = min(max((avg_ev - 86) / 12, 0), 1.0) * 5

        # 8. Pull rate (3pts) — weighted by park
        pull_rate = b.get("pulled_rate", 0)
        pull_score = min(max((pull_rate - 0.30) / 0.25, 0), 1.0) * 3 * min(park_factor / 1.05, 1.0)

        # SwStr% matchup bonus (up to +4)
        swstr_bonus = 0.0
        swstr_by_pt = b.get("swstr_by_pt", {})
        primary_pitch = p.get("primary_pitch") if p else None
        if primary_pitch and swstr_by_pt:
            pt_group = "fb" if primary_pitch in FASTBALL_TYPES else "brk" if primary_pitch in BREAKING_TYPES else "off"
            batter_swstr = swstr_by_pt.get(pt_group, b.get("swstr_rate", 0.10))
        else:
            batter_swstr = b.get("swstr_rate", 0.10)
        swstr_bonus = min(max((0.12 - batter_swstr) / 0.08, 0), 1.0) * 4

        # ── Pitcher signals ───────────────────────────────────────────────────

        # Pitcher HR vulnerability (18pts) — elite = ERA 5.5+, avg = 4.2
        p_hr9  = p.get("hr_per_fb", pitcher_hrfb) if p else pitcher_hrfb
        p_era  = pitcher_era
        p_hard = p.get("hard_hit_allowed", pitcher_hard) if p else pitcher_hard

        era_score  = min(max((p_era - 3.50) / 3.0, 0), 1.0) * 8
        hr9_score  = min(max((p_hr9 - 0.08) / 0.14, 0), 1.0) * 6
        hard_score = min(max((p_hard - 0.30) / 0.18, 0), 1.0) * 4
        pitcher_score = era_score + hr9_score + hard_score

        # ── Platoon (8pts) ────────────────────────────────────────────────────
        platoon_mult  = PLATOON_MULT.get((bat_side, pitch_hand), 1.0)
        platoon_score = (platoon_mult - 0.88) / (1.12 - 0.88) * 8

        # ── Environment (8pts) ───────────────────────────────────────────────
        park_score = min(max((park_factor - 0.90) / 0.35, 0), 1.0) * 4
        wind_score = min(max(wind_boost / 12, 0), 1.0) * 3
        temp_score = min(max((temp_f - 65) / 25, 0), 1.0) * 1
        env_score  = park_score + wind_score + temp_score

        # ── Hot bat (5pts) — HR rate + Statcast trending ─────────────────────
        hr_rate    = b.get("hr_rate", 0)
        heat_score_val = b.get("heat_score", 0)
        # HR rate component (2.5pts) + trending component (2.5pts)
        form_score = min(max((hr_rate - 0.05) / 0.20, 0), 1.0) * 2.5 + \
                     min(max(heat_score_val / 0.25, 0), 1.0) * 2.5

        # ── Zone bonus (up to +8) ─────────────────────────────────────────────
        zone_count = zone.get("zone_count", 0)
        zone_bonus = min(zone_count * 1.5, 8)

        # ── Composite ─────────────────────────────────────────────────────────
        base_score = (barrel_score + hh_score + xwoba_score + la_score +
                      fb_score + hrfb_score + ev_score + pull_score +
                      pitcher_score + platoon_score + env_score + form_score)
        hit_score  = round(min(base_score + zone_bonus + swstr_bonus, 100), 1)

        # ── Grade ─────────────────────────────────────────────────────────────
        if hit_score >= 65:   grade = "ELITE"
        elif hit_score >= 50: grade = "STRONG"
        elif hit_score >= 35: grade = "MODERATE"
        else:                 grade = "FADE"

        # ── Projected HR% ─────────────────────────────────────────────────────
        base_rate = max(hr_rate, 0.03)
        proj_hr   = base_rate * platoon_mult * park_factor * (1 + wind_boost/30)
        proj_hr   = round(min(proj_hr * 100, 35), 1)

        # ── Value vs market ───────────────────────────────────────────────────
        value_score = None
        if hr_odds is not None:
            try:
                if hr_odds > 0:
                    implied_prob = 100 / (hr_odds + 100)
                else:
                    implied_prob = abs(hr_odds) / (abs(hr_odds) + 100)
                our_prob = proj_hr / 100
                if implied_prob > 0:
                    value_score = round((our_prob / implied_prob - 1) * 100, 1)
            except:
                pass

        return {
            "hit_score":      hit_score,
            "grade":          grade,
            "proj_hr_pct":    proj_hr,
            "value_score":    value_score,

            # Component breakdown
            "barrel_score":   round(barrel_score, 1),
            "hh_score":       round(hh_score, 1),
            "xwoba_score":    round(xwoba_score, 1),
            "la_score":       round(la_score, 1),
            "fb_score":       round(fb_score, 1),
            "hrfb_score":     round(hrfb_score, 1),
            "ev_score":       round(ev_score, 1),
            "pull_score":     round(pull_score, 1),
            "swstr_bonus":    round(swstr_bonus, 1),
            "pitcher_score":  round(pitcher_score, 1),
            "platoon_score":  round(platoon_score, 1),
            "env_score":      round(env_score, 1),
            "form_score":     round(form_score, 1),
            "zone_bonus":     round(zone_bonus, 1),

            # Zone data
            "zone_count":     zone_count,
            "zone_fit":       zone.get("zone_fit", 0),
            "kill_zones":     zone.get("kill_zones", set()),
            "batter_zones":   zone.get("batter_zones", set()),
            "pitcher_zones":  zone.get("pitcher_zones", set()),

            # Pitch matchup
            "edge_pitch":      pitch.get("edge_pitch"),
            "weighted_barrel": pitch.get("weighted_barrel", 0),
            "fb_pct":          pitch.get("fb_pct", 0),
            "vel_band":        pitch.get("vel_band", "med"),
            "primary_pitch":   p.get("primary_pitch") if p else None,
            "pitch_mix":       p.get("pitch_mix", {}) if p else {},

            # Batter raw stats
            "barrel_rate":    round(barrel * 100, 1),
            "hard_hit_pct":   round(hard_hit * 100, 1),
            "xwoba":          round(xwoba, 3) if xwoba else None,
            "la_consistency": round(la_cons * 100, 1),
            "fb_rate":        round(fb_rate * 100, 1),
            "hr_fb_rate":     round(hr_fb * 100, 1),
            "pull_rate":      round(pull_rate * 100, 1),
            "swstr_rate":     round(batter_swstr * 100, 1),
            "avg_ev":         round(avg_ev, 1),
            "avg_la":         round(b.get("avg_la", 0), 1),
            "hr_rate":        round(hr_rate * 100, 2),
            "heat_score":     round(heat_score_val, 3),
            "barrel_7":       b.get("barrel_7", 0),
            "hh_7":           b.get("hh_7", 0),
            "xwoba_7":        b.get("xwoba_7"),
            "ev_7":           b.get("ev_7", 0),
            "barrel_trend":   b.get("barrel_trend", 0),
            "bip":            b.get("bip", 0) if b else 0,

            # Pitcher raw stats
            "pitcher_era":          round(p_era, 2),
            "pitcher_hr_fb":        round(p_hr9 * 100, 1),
            "pitcher_hard_allowed": round(p_hard * 100, 1),
            "pitcher_swstr":        round(p.get("swstr_rate", 0) * 100, 1) if p else 0,
            "pitcher_primary_vel":  p.get("primary_vel", 0) if p else 0,
            "pitcher_bip":          p.get("bip", 0) if p else 0,
            "pitcher_data_quality": "good" if p and p.get("bip", 0) >= 50 else "limited" if p and p.get("bip", 0) >= 10 else "none",
        }

    def get_k_score(self, pitcher_id: int, batter_id: int, bat_side: str = "R") -> dict:
        """
        Strikeout probability score for K props.
        Higher = more likely this batter strikes out vs this pitcher.
        """
        p = self._pitcher_index.get(int(pitcher_id), {})
        b = self._batter_index.get(int(batter_id), {})

        if not p:
            return {"k_score": 50, "grade": "MODERATE"}

        # Pitcher SwStr% (primary signal for K props)
        swstr = p.get("swstr_rate", 0.10)
        swstr_score = min(swstr / 0.15, 1.0) * 40

        # Pitch mix danger — high breaking ball usage = more Ks
        pitch_mix = p.get("pitch_mix", {})
        brk_pct   = sum(v["pct"] for k,v in pitch_mix.items() if k in BREAKING_TYPES)
        brk_score = min(brk_pct / 0.35, 1.0) * 20

        # Velocity — elite velo = more Ks
        primary_vel = p.get("primary_vel", 88)
        vel_score   = min(max((primary_vel - 88) / 12, 0), 1.0) * 20

        # Platoon matchup for Ks (same side = harder to make contact)
        platoon_mult = PLATOON_MULT.get((bat_side, p.get("pitch_hand", "R")), 1.0)
        platoon_k    = (1.12 - platoon_mult) / (1.12 - 0.88) * 20  # inverse of HR platoon

        k_score = swstr_score + brk_score + vel_score + platoon_k
        k_score = round(min(k_score, 100), 1)

        if k_score >= 65:   grade = "ELITE"
        elif k_score >= 50: grade = "STRONG"
        elif k_score >= 35: grade = "MODERATE"
        else:               grade = "FADE"

        return {
            "k_score":    k_score,
            "grade":      grade,
            "swstr_pct":  round(swstr * 100, 1),
            "brk_pct":    round(brk_pct * 100, 1),
            "primary_vel": primary_vel,
            "swstr_score": round(swstr_score, 1),
            "brk_score":   round(brk_score, 1),
            "vel_score":   round(vel_score, 1),
        }
