"""
zone_engine.py — AlgoHub Zone Overlap Engine
Calculates kHR, zone fit, sweet spot%, xwOBA, pulled barrel%, 
matchup score, and ceiling from bulk Statcast data.

All computed from the existing bulk Statcast pull — zero extra API calls.
"""

import pandas as pd
import numpy as np
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

SWEET_SPOT_MIN = 8    # degrees
SWEET_SPOT_MAX = 32   # degrees

# Statcast zones 1-9 are strike zone, 11-14 are outside
# 1=top-left, 2=top-middle, 3=top-right
# 4=mid-left, 5=middle,     6=mid-right  
# 7=bot-left, 8=bot-middle, 9=bot-right
# 11-14 = chase zones (corners)
HEART_ZONES    = {5}           # Middle middle
SHADOW_ZONES   = {2, 4, 6, 8} # Edge of zone
CHASE_ZONES    = {11, 12, 13, 14}
ALL_ZONES      = set(range(1, 10)) | CHASE_ZONES


# ── Core Zone Engine ───────────────────────────────────────────────────────────

class ZoneEngine:
    """
    Computes all advanced HR metrics from bulk Statcast pitch data.
    Initialize once, query per batter/pitcher pair.
    """

    def __init__(self, raw_df: pd.DataFrame):
        """
        raw_df: full bulk Statcast pull (all pitches, all games)
        """
        self.raw = raw_df.copy()
        self._prepare()
        self._build_indexes()

    def _prepare(self):
        """Clean and enrich raw Statcast data."""
        df = self.raw

        # Barrel calculation from EV + LA
        def calc_barrel(ev, la):
            if pd.isna(ev) or pd.isna(la) or ev < 98:
                return 0
            if ev >= 116:
                return int(8 <= la <= 50)
            min_la = 26 - (116 - ev)
            max_la = 30 + (116 - ev)
            return int(min_la <= la <= max_la)

        # Only compute on batted balls
        batted_mask = df["type"] == "X"
        self.batted = df[batted_mask].copy()

        if "launch_speed" in self.batted.columns and "launch_angle" in self.batted.columns:
            self.batted["barrel"] = self.batted.apply(
                lambda r: calc_barrel(r.get("launch_speed"), r.get("launch_angle")), axis=1
            )
        else:
            self.batted["barrel"] = 0

        # Sweet spot
        if "launch_angle" in self.batted.columns:
            self.batted["sweet_spot"] = (
                (self.batted["launch_angle"] >= SWEET_SPOT_MIN) &
                (self.batted["launch_angle"] <= SWEET_SPOT_MAX)
            ).astype(int)
        else:
            self.batted["sweet_spot"] = 0

        # HR flag
        self.batted["is_hr"] = (self.batted["events"] == "home_run").astype(int)
        df["is_hr"] = (df["events"] == "home_run").astype(int)
        self.raw = df

        # Pulled ball (approximate from hc_x)
        # hc_x: RHB pulls to left (low x), LHB pulls to right (high x)
        if "hc_x" in self.batted.columns and "stand" in self.batted.columns:
            self.batted["is_pulled"] = (
                ((self.batted["stand"] == "R") & (self.batted["hc_x"] < 125)) |
                ((self.batted["stand"] == "L") & (self.batted["hc_x"] > 125))
            ).astype(int)
        else:
            self.batted["is_pulled"] = 0

    def _build_indexes(self):
        """Pre-build per-pitcher and per-batter zone HR maps for fast lookup."""
        # Pitcher HR zones: which zones does this pitcher give up HRs in?
        if "zone" in self.batted.columns:
            pitcher_hr = self.batted[self.batted["is_hr"] == 1].groupby(
                ["pitcher", "zone"]
            ).size().reset_index(name="hr_count")
            self._pitcher_zone_hrs = pitcher_hr

            # Batter HR zones: which zones does this batter hit HRs in?
            batter_hr = self.batted[self.batted["is_hr"] == 1].groupby(
                ["batter", "zone"]
            ).size().reset_index(name="hr_count")
            self._batter_zone_hrs = batter_hr
        else:
            self._pitcher_zone_hrs = pd.DataFrame()
            self._batter_zone_hrs = pd.DataFrame()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_batter_stats(self, batter_id: int) -> dict:
        """All stats for a batter from bulk data."""
        b = self.batted[self.batted["batter"] == batter_id]
        raw_b = self.raw[self.raw["batter"] == batter_id]

        if b.empty:
            return self._empty_batter_stats()

        n = len(b)
        games = raw_b["game_date"].nunique() if "game_date" in raw_b.columns else 1

        avg_ev  = b["launch_speed"].mean() if "launch_speed" in b.columns else 0
        avg_la  = b["launch_angle"].mean()  if "launch_angle" in b.columns else 0
        barrel_rate   = b["barrel"].mean()
        sweet_spot_pct = b["sweet_spot"].mean()
        hard_hit_pct  = (b["launch_speed"] >= 95).mean() if "launch_speed" in b.columns else 0
        hr_count      = b["is_hr"].sum()
        hr_rate       = hr_count / max(games, 1)
        pulled_pct    = b["is_pulled"].mean()
        pulled_barrel = (b["barrel"] & b["is_pulled"]).mean() if "is_pulled" in b.columns else 0

        # xwOBA from Statcast
        xwoba = b["estimated_woba_using_speedangle"].mean() \
            if "estimated_woba_using_speedangle" in b.columns else None

        # ISO proxy from SLG-like metric
        iso = None  # Will be filled from leaderboard data

        # Batter HR zones
        batter_zones = set()
        if not self._batter_zone_hrs.empty:
            bz = self._batter_zone_hrs[self._batter_zone_hrs["batter"] == batter_id]
            batter_zones = set(bz["zone"].tolist())

        return {
            "batter_id":       batter_id,
            "games":           int(games),
            "bip":             int(n),
            "hr_count":        int(hr_count),
            "hr_rate":         float(hr_rate),
            "avg_ev":          float(avg_ev) if not pd.isna(avg_ev) else 0,
            "avg_la":          float(avg_la) if not pd.isna(avg_la) else 0,
            "barrel_rate":     float(barrel_rate) if not pd.isna(barrel_rate) else 0,
            "sweet_spot_pct":  float(sweet_spot_pct) if not pd.isna(sweet_spot_pct) else 0,
            "hard_hit_pct":    float(hard_hit_pct) if not pd.isna(hard_hit_pct) else 0,
            "pulled_pct":      float(pulled_pct) if not pd.isna(pulled_pct) else 0,
            "pulled_barrel":   float(pulled_barrel) if not pd.isna(pulled_barrel) else 0,
            "xwoba":           float(xwoba) if xwoba is not None and not pd.isna(xwoba) else None,
            "batter_hr_zones": batter_zones,
        }

    def get_pitcher_stats(self, pitcher_id: int) -> dict:
        """HR vulnerability stats for a pitcher."""
        b = self.batted[self.batted["pitcher"] == pitcher_id]

        if b.empty:
            return self._empty_pitcher_stats()

        hr_count     = b["is_hr"].sum()
        bip          = len(b)
        fb           = b[b["bb_type"].isin(["fly_ball", "popup"])] if "bb_type" in b.columns else pd.DataFrame()
        hr_per_fb    = hr_count / max(len(fb), 1)
        barrel_rate  = b["barrel"].mean()
        hard_hit_pct = (b["launch_speed"] >= 95).mean() if "launch_speed" in b.columns else 0

        # Pitcher HR zones
        pitcher_zones = set()
        if not self._pitcher_zone_hrs.empty:
            pz = self._pitcher_zone_hrs[self._pitcher_zone_hrs["pitcher"] == pitcher_id]
            pitcher_zones = set(pz["zone"].tolist())

        return {
            "pitcher_id":       pitcher_id,
            "bip":              int(bip),
            "hr_count":         int(hr_count),
            "hr_per_fb":        float(hr_per_fb) if not pd.isna(hr_per_fb) else 0,
            "barrel_allowed":   float(barrel_rate) if not pd.isna(barrel_rate) else 0,
            "hard_hit_allowed": float(hard_hit_pct) if not pd.isna(hard_hit_pct) else 0,
            "pitcher_hr_zones": pitcher_zones,
        }

    def compute_zone_fit(self, batter_id: int, pitcher_id: int) -> float:
        """
        Zone fit: how well do the batter's HR zones overlap with the pitcher's HR zones?
        Returns 0.0 - 1.0 (1.0 = perfect overlap)
        """
        if self._batter_zone_hrs.empty or self._pitcher_zone_hrs.empty:
            return 0.0

        bz = set(self._batter_zone_hrs[
            self._batter_zone_hrs["batter"] == batter_id
        ]["zone"].tolist())

        pz = set(self._pitcher_zone_hrs[
            self._pitcher_zone_hrs["pitcher"] == pitcher_id
        ]["zone"].tolist())

        if not bz or not pz:
            return 0.0

        overlap = len(bz & pz)
        union   = len(bz | pz)
        return round(overlap / union, 3) if union > 0 else 0.0

    def compute_khr(self, batter_id: int, pitcher_id: int) -> float:
        """
        kHR: batter's HR rate weighted by zone fit with this pitcher.
        Higher = better matchup-specific HR threat.
        """
        batter = self.get_batter_stats(batter_id)
        zone_fit = self.compute_zone_fit(batter_id, pitcher_id)
        hr_rate = batter.get("hr_rate", 0)

        # kHR = HR rate × (1 + zone_fit boost)
        # Zone fit of 0.5 = 50% boost to base HR rate
        khr = hr_rate * (1 + zone_fit)
        return round(khr, 4)

    def compute_matchup_score(
        self,
        batter_id: int,
        pitcher_id: int,
        park_factor: float = 1.0,
        wind_boost: float = 0.0,
        temp_f: float = 70.0,
        batter_iso: float = 0.15,
        pitcher_era: float = 4.0,
        platoon_adv: bool = False,
    ) -> dict:
        """
        Full matchup score combining all factors.
        Returns dict with individual scores + composite.
        """
        batter  = self.get_batter_stats(batter_id)
        pitcher = self.get_pitcher_stats(pitcher_id)
        zone_fit = self.compute_zone_fit(batter_id, pitcher_id)
        khr = self.compute_khr(batter_id, pitcher_id)

        # --- Individual component scores (0-100) ---

        # 1. Zone fit score (0-25 pts) — most important
        zone_score = min(zone_fit * 100, 25)

        # 2. ISO score (0-20 pts)
        iso_score = min((batter_iso / 0.300) * 20, 20)

        # 3. Pitcher ERA vuln (0-20 pts)
        era_score = min(((pitcher_era - 2.0) / 6.0) * 20, 20)
        era_score = max(era_score, 0)

        # 4. Barrel rate (0-15 pts)
        barrel_score = min((batter.get("barrel_rate", 0) / 0.15) * 15, 15)

        # 5. Park + weather (0-10 pts)
        park_score = min(((park_factor - 0.85) / 0.4) * 5, 5)
        wind_score = min(max(wind_boost / 15, 0) * 3, 3)
        temp_score = min(max((temp_f - 60) / 30, 0) * 2, 2)
        env_score  = park_score + wind_score + temp_score

        # 6. HR form (0-10 pts)
        form_score = min((batter.get("hr_rate", 0) / 0.3) * 10, 10)

        # 7. Platoon advantage bonus (0-5 pts)
        platoon_score = 5 if platoon_adv else 0

        # 8. Exit velo (0-5 pts)
        ev_score = min(max((batter.get("avg_ev", 85) - 80) / 20, 0) * 5, 5)

        composite = (
            zone_score + iso_score + era_score + barrel_score +
            env_score + form_score + platoon_score + ev_score
        )
        composite = round(min(composite, 100), 1)

        # Grade
        if composite >= 70:
            grade = "TARGET"
        elif composite >= 55:
            grade = "STRONG"
        elif composite >= 40:
            grade = "MODERATE"
        else:
            grade = "TOUGH"

        # Ceiling = composite + upside factors
        ceiling = composite + (
            min(batter.get("sweet_spot_pct", 0) * 20, 5) +
            min(batter.get("pulled_barrel", 0) * 30, 5)
        )
        ceiling = round(min(ceiling, 100), 1)

        # Projected HR%
        base_hr_rate = batter.get("hr_rate", 0.05)
        proj_hr_pct  = base_hr_rate * (1 + zone_fit) * park_factor * (1 + wind_boost / 30)
        proj_hr_pct  = round(min(proj_hr_pct * 100, 35), 1)

        return {
            "batter_id":        batter_id,
            "pitcher_id":       pitcher_id,
            "composite_score":  composite,
            "ceiling":          ceiling,
            "grade":            grade,
            "proj_hr_pct":      proj_hr_pct,
            "zone_fit":         round(zone_fit, 3),
            "khr":              round(khr * 100, 2),
            "zone_score":       round(zone_score, 1),
            "iso_score":        round(iso_score, 1),
            "era_score":        round(era_score, 1),
            "barrel_score":     round(barrel_score, 1),
            "env_score":        round(env_score, 1),
            "form_score":       round(form_score, 1),
            "platoon_score":    platoon_score,
            "ev_score":         round(ev_score, 1),
            # Batter stats
            "avg_ev":           round(batter.get("avg_ev", 0), 1),
            "avg_la":           round(batter.get("avg_la", 0), 1),
            "barrel_rate":      round(batter.get("barrel_rate", 0) * 100, 1),
            "sweet_spot_pct":   round(batter.get("sweet_spot_pct", 0) * 100, 1),
            "hard_hit_pct":     round(batter.get("hard_hit_pct", 0) * 100, 1),
            "hr_rate":          round(batter.get("hr_rate", 0) * 100, 2),
            "hr_count":         batter.get("hr_count", 0),
            "xwoba":            batter.get("xwoba"),
            "pulled_barrel":    round(batter.get("pulled_barrel", 0) * 100, 1),
            # Pitcher stats
            "pitcher_era_vuln": round(pitcher.get("hr_per_fb", 0) * 100, 1),
            "pitcher_barrel_allowed": round(pitcher.get("barrel_allowed", 0) * 100, 1),
            # Zone count (number of overlapping zones)
            "zone_count":       len(
                self._get_batter_zones(batter_id) &
                self._get_pitcher_zones(pitcher_id)
            ),
        }

    def _get_batter_zones(self, batter_id: int) -> set:
        if self._batter_zone_hrs.empty:
            return set()
        return set(self._batter_zone_hrs[
            self._batter_zone_hrs["batter"] == batter_id
        ]["zone"].tolist())

    def _get_pitcher_zones(self, pitcher_id: int) -> set:
        if self._pitcher_zone_hrs.empty:
            return set()
        return set(self._pitcher_zone_hrs[
            self._pitcher_zone_hrs["pitcher"] == pitcher_id
        ]["zone"].tolist())

    def _empty_batter_stats(self) -> dict:
        return {
            "batter_id": None, "games": 0, "bip": 0, "hr_count": 0,
            "hr_rate": 0, "avg_ev": 0, "avg_la": 0, "barrel_rate": 0,
            "sweet_spot_pct": 0, "hard_hit_pct": 0, "pulled_pct": 0,
            "pulled_barrel": 0, "xwoba": None, "batter_hr_zones": set(),
        }

    def _empty_pitcher_stats(self) -> dict:
        return {
            "pitcher_id": None, "bip": 0, "hr_count": 0,
            "hr_per_fb": 0, "barrel_allowed": 0,
            "hard_hit_allowed": 0, "pitcher_hr_zones": set(),
        }
