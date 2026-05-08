"""
daily_predictions.py
Generates today's HR probability rankings for all players in the day's slate.
Now includes: platoon splits, pitch mix, precise wind, injury/rest, Vegas lines.

Usage:
    python daily_predictions.py
    python daily_predictions.py --date 2026-04-15
    python daily_predictions.py --top 50 --no-vegas
"""

import argparse
import pandas as pd
import numpy as np
import pybaseball as pb
import requests
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")

from data_collection import PARK_FACTORS
from feature_engineering import get_feature_columns
from model_training import load_model
from ballpark_wind import fetch_weather_precise, DOME_STADIUMS
from platoon_splits import add_platoon_features, build_hand_lookup
from pitch_mix import build_pitcher_pitch_mix, hr_risk_score
from injury_rest import (get_current_il_list, get_il_transactions,
                          add_injury_rest_features, get_days_since_last_game_bulk)
from vegas_lines import add_vegas_features
from h2h_splits import add_h2h_features_bulk

pb.cache.enable()


# ── Today's Schedule ──────────────────────────────────────────────────────────

def get_todays_schedule(date_str: str) -> pd.DataFrame:
    """
    Pull today's MLB schedule. Returns DataFrame with game info.
    Uses pybaseball schedule or falls back to MLB Stats API.
    """
    print(f"Fetching schedule for {date_str}...")
    try:
        # MLB Stats API (free, reliable)
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,team,linescore"
        r = requests.get(url, timeout=10)
        data = r.json()

        games = []
        for date_data in data.get("dates", []):
            for game in date_data.get("games", []):
                home = game["teams"]["home"]
                away = game["teams"]["away"]

                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})

                games.append({
                    "game_pk": game["gamePk"],
                    "game_date": date_str,
                    "home_team": home["team"]["abbreviation"],
                    "away_team": away["team"]["abbreviation"],
                    "home_team_id": home["team"]["id"],
                    "away_team_id": away["team"]["id"],
                    "home_pitcher_id": home_pitcher.get("id"),
                    "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                    "venue": game.get("venue", {}).get("name", ""),
                    "game_time": game.get("gameDate", ""),
                })

        print(f"  Found {len(games)} games")
        return pd.DataFrame(games)

    except Exception as e:
        print(f"  Schedule fetch error: {e}")
        return pd.DataFrame()


# ── Active Rosters ────────────────────────────────────────────────────────────

def get_team_roster(team_id: int) -> list:
    """Fetch active roster for a team via MLB Stats API.
    Tries multiple roster types to handle early season / spring training edge cases.
    """
    roster_types = ["active", "fullRoster", "40Man"]
    for roster_type in roster_types:
        try:
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType={roster_type}"
            r = requests.get(url, timeout=10)
            data = r.json()
            all_players = data.get("roster", [])
            roster = []
            for p in all_players:
                pos = p.get("position", {}).get("abbreviation", "")
                if pos not in ("P", "SP", "RP"):
                    roster.append({
                        "player_id": p["person"]["id"],
                        "player_name": p["person"]["fullName"],
                        "position": pos,
                    })
            if all_players:
                print(f"    Team {team_id} ({roster_type}): {len(all_players)} total, {len(roster)} non-pitchers")
                return roster
            else:
                print(f"    Team {team_id} ({roster_type}): empty response")
        except Exception as e:
            print(f"    Team {team_id} ({roster_type}) error: {e}")
            continue
    return []


# ── Recent Statcast for Current Form ─────────────────────────────────────────

# ── Bulk Statcast Cache (replaces per-player calls) ───────────────────────────
_STATCAST_BULK_CACHE = None
_STATCAST_RAW_CACHE = None  # Keep raw data for pitch type splits

# Pitch type groupings
FASTBALL_TYPES  = {"FF", "SI", "FC"}
BREAKING_TYPES  = {"SL", "CU", "KC", "CS", "SV"}
OFFSPEED_TYPES  = {"CH", "FS", "FO"}

def load_bulk_statcast_cache(days: int = 30):
    """
    Pull last N days of Statcast data for ALL players in one bulk call.
    Calculates barrel from launch_speed + launch_angle.
    Also stores raw data for pitch type split calculations.
    Called once at startup — takes ~10 seconds vs 20 minutes for per-player.
    """
    global _STATCAST_BULK_CACHE, _STATCAST_RAW_CACHE
    if _STATCAST_BULK_CACHE is not None:
        return _STATCAST_BULK_CACHE

    end = datetime.today()
    start = end - timedelta(days=days)
    print(f"\nBulk Statcast pull: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}...")

    try:
        raw = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if raw is None or raw.empty:
            print("  No Statcast data returned — using empty cache")
            _STATCAST_BULK_CACHE = {}
            _STATCAST_RAW_CACHE = pd.DataFrame()
            return _STATCAST_BULK_CACHE

        raw["game_date"] = pd.to_datetime(raw["game_date"])
        raw["hr_flag"] = (raw["events"] == "home_run").astype(int)
        batted = raw[raw["type"] == "X"].copy()

        # Calculate barrel from EV + LA
        def calc_barrel(ev, la):
            if pd.isna(ev) or pd.isna(la) or ev < 98:
                return 0
            if ev >= 116:
                return int(8 <= la <= 50)
            min_la = 26 - (116 - ev) * 1.0
            max_la = 30 + (116 - ev) * 1.0
            return int(min_la <= la <= max_la)

        if "launch_speed" in batted.columns and "launch_angle" in batted.columns:
            batted["barrel_calc"] = batted.apply(
                lambda r: calc_barrel(r["launch_speed"], r["launch_angle"]), axis=1
            )
        else:
            batted["barrel_calc"] = 0

        # Store raw for pitch type splits
        _STATCAST_RAW_CACHE = batted.copy()

        cache = {}
        for pid, grp in raw.groupby("batter"):
            batted_grp = batted[batted["batter"] == pid]
            games = grp["game_date"].nunique()
            hr_count = grp["hr_flag"].sum()
            fly_balls = batted_grp[batted_grp["bb_type"].isin(["fly_ball", "popup"])] if "bb_type" in batted_grp.columns else pd.DataFrame()
            hr_batted = batted_grp[batted_grp["events"] == "home_run"] if "events" in batted_grp.columns else pd.DataFrame()

            barrel_rate = batted_grp["barrel_calc"].mean() if not batted_grp.empty else 0
            avg_ev = batted_grp["launch_speed"].mean() if "launch_speed" in batted_grp.columns and not batted_grp.empty else 0
            avg_la = batted_grp["launch_angle"].mean() if "launch_angle" in batted_grp.columns and not batted_grp.empty else 0

            cache[int(pid)] = {
                "hr_rate_last30":     hr_count / max(games, 1),
                "barrel_rate_last30": float(barrel_rate) if not pd.isna(barrel_rate) else 0,
                "avg_ev_last30":      float(avg_ev) if not pd.isna(avg_ev) else 0,
                "avg_la_last30":      float(avg_la) if not pd.isna(avg_la) else 0,
                "hr_fb_rate_last30":  len(hr_batted) / max(len(fly_balls), 1),
                "games_last30":       games,
            }

        print(f"  Cached {len(cache)} players from {raw['game_date'].nunique()} days")
        _STATCAST_BULK_CACHE = cache
        return cache

    except Exception as e:
        print(f"  Bulk Statcast pull failed: {e}")
        _STATCAST_BULK_CACHE = {}
        _STATCAST_RAW_CACHE = pd.DataFrame()
        return {}


def get_batter_pitch_splits(player_id: int) -> dict:
    """
    Get batter's barrel rate and EV broken down by pitch type group
    (fastball, breaking, offspeed) from the bulk cache.
    Zero extra API calls — uses already-pulled data.
    """
    global _STATCAST_RAW_CACHE
    if _STATCAST_RAW_CACHE is None or _STATCAST_RAW_CACHE.empty:
        return {}

    batted = _STATCAST_RAW_CACHE[_STATCAST_RAW_CACHE["batter"] == player_id]
    if batted.empty or "pitch_type" not in batted.columns:
        return {}

    def split_stats(subset):
        if subset.empty:
            return {"barrel_rate": 0, "avg_ev": 0, "hr_rate": 0, "n": 0}
        hrs = (subset["events"] == "home_run").sum() if "events" in subset.columns else 0
        return {
            "barrel_rate": float(subset["barrel_calc"].mean()) if "barrel_calc" in subset.columns else 0,
            "avg_ev":      float(subset["launch_speed"].mean()) if "launch_speed" in subset.columns else 0,
            "hr_rate":     float(hrs / max(len(subset), 1)),
            "n":           len(subset),
        }

    fb  = batted[batted["pitch_type"].isin(FASTBALL_TYPES)]
    brk = batted[batted["pitch_type"].isin(BREAKING_TYPES)]
    off = batted[batted["pitch_type"].isin(OFFSPEED_TYPES)]

    fb_stats  = split_stats(fb)
    brk_stats = split_stats(brk)
    off_stats = split_stats(off)

    return {
        "batter_fb_barrel_rate":      fb_stats["barrel_rate"],
        "batter_fb_avg_ev":           fb_stats["avg_ev"],
        "batter_fb_hr_rate":          fb_stats["hr_rate"],
        "batter_brk_barrel_rate":     brk_stats["barrel_rate"],
        "batter_brk_avg_ev":          brk_stats["avg_ev"],
        "batter_brk_hr_rate":         brk_stats["hr_rate"],
        "batter_off_barrel_rate":     off_stats["barrel_rate"],
        "batter_off_avg_ev":          off_stats["avg_ev"],
        "batter_off_hr_rate":         off_stats["hr_rate"],
    }


def compute_pitch_matchup_score(batter_splits: dict, pitcher_mix: dict) -> float:
    """
    Combine batter's pitch type splits with pitcher's usage rates
    to generate a matchup score.

    Higher score = batter's strengths align with pitcher's tendencies.
    Scale: 0.0 (terrible matchup) to 1.0 (elite matchup)
    """
    if not batter_splits or not pitcher_mix:
        return 0.5  # neutral default

    fb_pct  = pitcher_mix.get("pitch_fb_pct",  0.50)
    brk_pct = pitcher_mix.get("pitch_sl_pct", 0.20) + pitcher_mix.get("pitch_cu_pct", 0.10)
    off_pct = pitcher_mix.get("pitch_ch_pct",  0.15)

    # Weighted barrel rate against pitcher's actual pitch mix
    matchup_barrel = (
        fb_pct  * batter_splits.get("batter_fb_barrel_rate",  0.06) +
        brk_pct * batter_splits.get("batter_brk_barrel_rate", 0.06) +
        off_pct * batter_splits.get("batter_off_barrel_rate", 0.06)
    )

    # Weighted HR rate against pitcher's pitch mix
    matchup_hr = (
        fb_pct  * batter_splits.get("batter_fb_hr_rate",  0.05) +
        brk_pct * batter_splits.get("batter_brk_hr_rate", 0.05) +
        off_pct * batter_splits.get("batter_off_hr_rate", 0.05)
    )

    # Normalize: league avg barrel ~6%, HR rate ~5%
    barrel_score = min(matchup_barrel / 0.12, 1.0)
    hr_score     = min(matchup_hr / 0.10, 1.0)

    return round((barrel_score * 0.6 + hr_score * 0.4), 3)


def get_recent_statcast(player_id: int, days: int = 30) -> dict:
    """Look up player from bulk cache instead of making per-player API call."""
    cache = load_bulk_statcast_cache(days=days)
    return cache.get(int(player_id), {})


# ── Season Stats for Players ──────────────────────────────────────────────────

def get_season_batting_stats(season: int) -> pd.DataFrame:
    """Pull Statcast batting leaderboard. Falls back to prior season if current is empty."""
    for s in [season, season - 1]:
        try:
            df = pb.statcast_batter_exitvelo_barrels(s, minBBE=20)
            if df is not None and not df.empty:
                rename = {
                    "player_id":            "player_id",
                    "last_name, first_name":"Name",
                    "brl_percent":          "season_barrel_rate",
                    "ev95percent":          "season_hard_hit_rate",
                    "avg_hit_speed":        "season_avg_ev",
                    "attempts":             "pa",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                # Convert barrel rate from percentage to decimal
                if "season_barrel_rate" in df.columns:
                    df["season_barrel_rate"] = df["season_barrel_rate"] / 100
                if "season_hard_hit_rate" in df.columns:
                    df["season_hard_hit_rate"] = df["season_hard_hit_rate"] / 100

                # Estimate HR/FB rate from leaderboard data
                # barrels / fly balls is a reasonable HR/FB proxy
                if "barrels" in df.columns and "fbld" in df.columns:
                    df["season_hr_fb_rate"] = (df["barrels"] / df["fbld"].replace(0, np.nan)).clip(0, 0.6).fillna(0.12)
                else:
                    df["season_hr_fb_rate"] = 0.12

                # Estimate launch angle category from avg_hit_angle
                if "avg_hit_angle" in df.columns:
                    df["season_avg_la"] = df["avg_hit_angle"]
                else:
                    df["season_avg_la"] = 12.0
                print(f"  Savant batting leaderboard ({s}): {len(df)} players")
                return df
        except Exception as e:
            print(f"  Savant batting {s} failed: {e}")
    return pd.DataFrame()


def get_pitcher_season_stats(season: int) -> pd.DataFrame:
    """Pull Statcast pitching leaderboard. Falls back to prior season if current is empty."""
    for s in [season, season - 1]:
        try:
            df = pb.statcast_pitcher_exitvelo_barrels(s, minBBE=20)
            if df is not None and not df.empty:
                rename = {
                    "player_id":            "pitcher_savant_id",
                    "last_name, first_name":"Name",
                    "brl_percent":          "pitcher_barrel_allowed",
                    "ev95percent":          "pitcher_hard_pct",
                    "avg_hit_speed":        "pitcher_avg_ev_allowed",
                }
                df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
                if "pitcher_barrel_allowed" in df.columns:
                    df["pitcher_barrel_allowed"] = df["pitcher_barrel_allowed"] / 100
                if "pitcher_hard_pct" in df.columns:
                    df["pitcher_hard_pct"] = df["pitcher_hard_pct"] / 100
                print(f"  Savant pitching leaderboard ({s}): {len(df)} pitchers")
                return df
        except Exception as e:
            print(f"  Savant pitching {s} failed: {e}")
    return pd.DataFrame()


# ── Team abbreviation normalization ──────────────────────────────────────────
# MLB Stats API uses different abbreviations than our park/weather lookups
TEAM_ABBR_MAP = {
    "SF":  "SFG", "SD":  "SDP", "KC":  "KCR", "TB":  "TBR",
    "CWS": "CWS", "WSH": "WSH", "NYY": "NYY", "NYM": "NYM",
    "LAD": "LAD", "LAA": "LAA", "STL": "STL", "CHC": "CHC",
    "CLE": "CLE", "MIL": "MIL", "MIN": "MIN", "ATL": "ATL",
    "HOU": "HOU", "BOS": "BOS", "PHI": "PHI", "COL": "COL",
    "TEX": "TEX", "BAL": "BAL", "DET": "DET", "PIT": "PIT",
    "MIA": "MIA", "OAK": "OAK", "SEA": "SEA", "TOR": "TOR",
    "CIN": "CIN", "ARI": "ARI",
}

def normalize_team(abbr: str) -> str:
    return TEAM_ABBR_MAP.get(abbr, abbr)


# ── Build Prediction Rows ─────────────────────────────────────────────────────

def build_prediction_rows(schedule: pd.DataFrame, date_str: str,
                           season: int = 2026, use_vegas: bool = True) -> pd.DataFrame:
    """
    For each game in today's slate, build a feature row per batter.
    Now includes: real platoon splits, pitch mix, precise wind, rest/IL flags.
    """
    print("\nBuilding prediction feature rows...")

    season_bat = get_season_batting_stats(season)
    season_pit = get_pitcher_season_stats(season)

    # Pre-fetch IL list and recent transactions
    print("  Checking injury list...")
    il_players = get_current_il_list()
    il_transactions = get_il_transactions(days_back=30)

    # Collect all pitcher IDs for pitch mix pull
    pitcher_ids = []
    for _, game in schedule.iterrows():
        for pid in [game.get("home_pitcher_id"), game.get("away_pitcher_id")]:
            if pid and not pd.isna(pid):
                pitcher_ids.append(int(pid))

    # Fetch pitch mix for all starters
    print("  Fetching pitcher pitch mix data...")
    pitch_mix_df = build_pitcher_pitch_mix(pitcher_ids, season=season - 1)  # use prior season for full sample

    rows = []

    for _, game in schedule.iterrows():
        home_team = normalize_team(game["home_team"])
        away_team = normalize_team(game["away_team"])

        # Precise weather + wind orientation for this stadium
        weather = fetch_weather_precise(home_team, date_str)
        park_factor = PARK_FACTORS.get(home_team, 1.0)

        # Process both teams' lineups
        for batting_team_id, batting_team, pitcher_id, pitcher_name, is_home in [
            (game["home_team_id"], home_team, game["away_pitcher_id"], game["away_pitcher_name"], True),
            (game["away_team_id"], away_team, game["home_pitcher_id"], game["home_pitcher_name"], False),
        ]:
            # ── Pitcher Features ──────────────────────────────────────────────
            pitcher_stats = {}
            if not season_pit.empty and pitcher_id:
                pit_row = season_pit[season_pit["Name"].str.contains(
                    pitcher_name.split()[-1] if pitcher_name not in ("TBD", "") else "ZZZZZ",
                    case=False, na=False
                )]
                if not pit_row.empty:
                    pit_row = pit_row.iloc[0]
                    pitcher_stats = {
                        "pitcher_hrfb": pit_row.get("HR/FB", 0.12),
                        "pitcher_hr9": pit_row.get("HR9", 1.1),
                        "pitcher_hard_pct": pit_row.get("Hard%", 0.35),
                        "pitcher_fip": pit_row.get("FIP", 4.0),
                        "pitcher_xfip": pit_row.get("xFIP", 4.0),
                    }

            pitcher_stats.setdefault("pitcher_hrfb", 0.12)
            pitcher_stats.setdefault("pitcher_hr9", 1.1)
            pitcher_stats.setdefault("pitcher_hard_pct", 0.35)
            pitcher_stats.setdefault("pitcher_fip", 4.0)
            pitcher_stats.setdefault("pitcher_xfip", 4.0)

            # Pitch mix risk score for this starter
            pitch_risk = 0.5  # neutral default
            if not pitch_mix_df.empty and pitcher_id and not pd.isna(pitcher_id):
                pm_row = pitch_mix_df[pitch_mix_df["pitcher_id"] == int(pitcher_id)]
                if not pm_row.empty:
                    pitch_risk = hr_risk_score(pm_row.iloc[0])
            pitcher_stats["pitcher_pitch_risk"] = pitch_risk

            # Pitcher hand for platoon
            safe_pitcher_id = int(pitcher_id) if pitcher_id and not pd.isna(float(pitcher_id)) else None
            pitcher_hand_data = build_hand_lookup([safe_pitcher_id] if safe_pitcher_id else [])
            pitcher_hand = pitcher_hand_data.get(safe_pitcher_id, {}).get("pitch_hand", "R") if safe_pitcher_id else "R"

            # ── Roster + Batter Features ──────────────────────────────────────
            roster = get_team_roster(batting_team_id)
            # Active roster already excludes IL — just keep everyone
            print(f"  {batting_team} ({len(roster)} active batters) vs {pitcher_name} ({pitcher_hand}HP)")

            # Bulk fetch rest days for this roster
            batter_ids = [p["player_id"] for p in roster]
            rest_days_map = get_days_since_last_game_bulk(batter_ids, date_str, season)

            # Bulk fetch hand data for batters
            batter_hand_lookup = build_hand_lookup(batter_ids)

            for player in roster:
                pid = player["player_id"]

                # ── Recent Statcast Form ──────────────────────────────────────
                recent = get_recent_statcast(pid, days=30)

                # ── Pitch Type Splits + Matchup Score ────────────────────────
                pitch_splits = get_batter_pitch_splits(pid)
                pitch_matchup_score = compute_pitch_matchup_score(pitch_splits, pm_row.iloc[0].to_dict() if not pm_row.empty else {})

                # ── Season Stats ─────────────────────────────────────────────
                # Step 1: Savant leaderboard (barrel rate, hard hit%, exit velo)
                # Step 2: Training data supplements with HR/FB rate, launch angle
                # Step 3: League average defaults for anything still missing
                batter_season = {}

                # Load training data cache once
                try:
                    if not hasattr(build_prediction_rows, '_td_cache'):
                        td_path = "data/training_data.csv"
                        build_prediction_rows._td_cache = pd.read_csv(td_path) if os.path.exists(td_path) else pd.DataFrame()
                    td = build_prediction_rows._td_cache
                    td_player = td[td["player_id"] == pid] if not td.empty else pd.DataFrame()
                    if not td_player.empty:
                        batter_season["season_hr_fb_rate"] = float(td_player["season_hr_fb_rate"].mean())
                        batter_season["season_fb_rate"]    = float(td_player["season_fb_rate"].mean())
                        batter_season["season_avg_la"]     = float(td_player["season_avg_la"].mean())
                        # Use training data barrel/EV as baseline (will be overridden by leaderboard below)
                        batter_season["season_barrel_rate"]   = float(td_player["season_barrel_rate"].mean())
                        batter_season["season_hard_hit_rate"] = float(td_player["season_hard_hit_rate"].mean())
                        batter_season["season_avg_ev"]        = float(td_player["season_avg_ev"].mean())
                        batter_season["season_weight"]        = 0.8
                except Exception:
                    pass

                # Always override with Savant leaderboard for barrel/EV — covers everyone
                # including injured/part-time players not in our training data
                if not season_bat.empty and "player_id" in season_bat.columns:
                    bat_match = season_bat[season_bat["player_id"] == pid]
                    if not bat_match.empty:
                        bat_row = bat_match.iloc[0]
                        pa_val = float(bat_row.get("pa", bat_row.get("attempts", 50)))
                        batter_season["season_barrel_rate"]   = float(bat_row.get("season_barrel_rate",   batter_season.get("season_barrel_rate",   0.06)))
                        batter_season["season_hard_hit_rate"] = float(bat_row.get("season_hard_hit_rate", batter_season.get("season_hard_hit_rate", 0.35)))
                        batter_season["season_avg_ev"]        = float(bat_row.get("season_avg_ev",        batter_season.get("season_avg_ev",        88.0)))
                        batter_season["season_hr_fb_rate"]    = float(bat_row.get("season_hr_fb_rate",    batter_season.get("season_hr_fb_rate",    0.12)))
                        batter_season["season_avg_la"]        = float(bat_row.get("season_avg_la",        batter_season.get("season_avg_la",        12.0)))
                        batter_season["season_weight"]        = min(pa_val / 300, 1.0)

                # League average defaults for anything still missing
                batter_season.setdefault("season_barrel_rate",   0.06)
                batter_season.setdefault("season_hard_hit_rate", 0.35)
                batter_season.setdefault("season_avg_ev",        88.0)
                batter_season.setdefault("season_avg_la",        12.0)
                batter_season.setdefault("season_hr_fb_rate",    0.12)
                batter_season.setdefault("season_fb_rate",       0.35)
                batter_season.setdefault("season_weight",        0.3)

                # ── Platoon ───────────────────────────────────────────────────
                from platoon_splits import PLATOON_HR_MULTIPLIER
                bat_side = batter_hand_lookup.get(pid, {}).get("bat_side", "R")
                platoon_adv = int(
                    (bat_side == "R" and pitcher_hand == "L") or
                    (bat_side == "L" and pitcher_hand == "R") or
                    bat_side == "S"
                )
                platoon_mult = PLATOON_HR_MULTIPLIER.get((bat_side, pitcher_hand), 1.0)

                # ── Rest / IL ─────────────────────────────────────────────────
                days_rest = rest_days_map.get(pid, 1)
                from injury_rest import rest_hr_multiplier, days_since_il_activation, il_return_hr_multiplier
                rest_mult = rest_hr_multiplier(days_rest)
                days_il = days_since_il_activation(pid, date_str, il_transactions)
                il_mult = il_return_hr_multiplier(days_il)

                row = {
                    "player_id": pid,
                    "player_name": player["player_name"],
                    "position": player["position"],
                    "batting_team": batting_team,
                    "home_team": home_team,
                    "opposing_pitcher": pitcher_name,
                    "pitcher_id": int(pitcher_id) if pitcher_id and not pd.isna(pitcher_id) else None,
                    "game_date": date_str,
                    "bat_side": bat_side,
                    "pitch_hand": pitcher_hand,
                }

                # Early season baseline: use 2025 season stats until 2026 data builds up
                baseline_barrel = batter_season.get("season_barrel_rate", 0.06)
                baseline_ev     = batter_season.get("season_avg_ev", 88.0)
                baseline_hr     = batter_season.get("season_hr_fb_rate", 0.05)

                row.update({
                    # Rolling recent form
                    "hr_rate_last7":      recent.get("hr_rate_last30", baseline_hr),
                    "hr_rate_last15":     recent.get("hr_rate_last30", baseline_hr),
                    "hr_rate_last30":     recent.get("hr_rate_last30", baseline_hr),
                    "barrel_rate_last7":  recent.get("barrel_rate_last30", baseline_barrel),
                    "barrel_rate_last15": recent.get("barrel_rate_last30", baseline_barrel),
                    "barrel_rate_last30": recent.get("barrel_rate_last30", baseline_barrel),
                    "avg_ev_last7":       recent.get("avg_ev_last30", baseline_ev),
                    "avg_ev_last15":      recent.get("avg_ev_last30", baseline_ev),
                    "avg_ev_last30":      recent.get("avg_ev_last30", baseline_ev),

                    # Park + precise weather
                    "park_factor": park_factor,
                    "temp_f": weather["temp_f"],
                    "wind_mph": weather["wind_mph"],
                    "wind_boost": weather["wind_boost"],
                    "wind_component_mph": weather.get("wind_component_mph", 0),
                    "temp_boost": weather.get("temp_boost", 0),
                    "precip_mm": weather["precip_mm"],
                    "wind_label": weather.get("wind_label", ""),

                    # Platoon
                    "platoon_advantage": platoon_adv,
                    "platoon_hr_multiplier": platoon_mult,

                    # Rest / IL
                    "days_rest": days_rest,
                    "rest_hr_multiplier": rest_mult,
                    "days_since_il": days_il,
                    "il_return_multiplier": il_mult,

                    # Context
                    "expected_pa": 3.8,
                    "month": datetime.strptime(date_str, "%Y-%m-%d").month,
                })

                row.update(batter_season)
                row.update(pitcher_stats)
                # Pitch type splits and matchup score
                row.update(pitch_splits)
                row["pitch_matchup_score"] = pitch_matchup_score
                rows.append(row)

    df = pd.DataFrame(rows)

    # Apply multiplier adjustments as post-model scaling factors
    # (stored separately so the model score and adjustment are both visible)
    if not df.empty:
        df["adjustment_factor"] = (
            df.get("platoon_hr_multiplier", 1.0) *
            df.get("rest_hr_multiplier", 1.0) *
            df.get("il_return_multiplier", 1.0)
        )

    return df


# ── Run Predictions ───────────────────────────────────────────────────────────

def run_daily_predictions(date_str: str, top_n: int = 30, output_path: str = None,
                           use_vegas: bool = True):
    """Full daily prediction pipeline with all enhancements."""

    print(f"\n{'='*60}")
    print(f"  MLB HR MODEL — Daily Predictions: {date_str}")
    print(f"  Platoon ✓  Pitch Mix ✓  Precise Wind ✓  Rest/IL ✓  Vegas ✓")
    print(f"{'='*60}\n")

    # Load model
    if not os.path.exists("models/hr_model.pkl"):
        print("ERROR: No trained model found. Run model_training.py first.")
        return pd.DataFrame()

    model, feature_cols = load_model()
    print(f"Model loaded. Features: {len(feature_cols)}")

    # Pre-load bulk Statcast cache — one API call for all players (~10 seconds)
    load_bulk_statcast_cache(days=30)

    # Get schedule
    schedule = get_todays_schedule(date_str)
    if schedule.empty:
        print("No games found for today.")
        return pd.DataFrame()

    # Build features
    pred_df = build_prediction_rows(schedule, date_str, use_vegas=use_vegas)
    if pred_df.empty:
        print("No prediction rows built.")
        return pd.DataFrame()

    # Vegas lines
    if use_vegas:
        print("\nFetching Vegas implied totals...")
        pred_df = add_vegas_features(pred_df, date_str)

    # Head-to-head batter vs pitcher history
    print("\nFetching batter vs pitcher head-to-head history...")
    pred_df = add_h2h_features_bulk(pred_df)

    # Align feature columns
    available_features = [c for c in feature_cols if c in pred_df.columns]
    X_pred = pred_df[available_features].fillna(pred_df[available_features].median())

    # Base model prediction
    base_probs = model.predict_proba(X_pred)[:, 1]
    pred_df["model_prob"] = base_probs

    # Apply post-model multipliers (platoon, rest, IL, vegas, h2h)
    adj = pred_df.get("adjustment_factor", pd.Series(1.0, index=pred_df.index))
    vegas_mult = pred_df.get("vegas_hr_multiplier", pd.Series(1.0, index=pred_df.index))
    wind_adj = 1.0 + pred_df.get("wind_boost", pd.Series(0.0, index=pred_df.index))
    temp_adj = 1.0 + pred_df.get("temp_boost", pd.Series(0.0, index=pred_df.index))

    # H2H adjustment: if we have meaningful sample, nudge toward blended h2h rate
    # credibility_weight=0 means ignore h2h, =1 means fully use it
    h2h_weight = pred_df.get("h2h_credibility_weight", pd.Series(0.0, index=pred_df.index))
    h2h_blended = pred_df.get("h2h_blended_hr_rate", pd.Series(0.0, index=pred_df.index))
    h2h_adj = 1.0 + (h2h_weight * (h2h_blended - base_probs).clip(-0.05, 0.05))

    adjusted_probs = base_probs * adj * vegas_mult * wind_adj * temp_adj * h2h_adj

    # Clip to valid probability range
    adjusted_probs = adjusted_probs.clip(0.01, 0.60)

    pred_df["hr_probability"] = adjusted_probs
    pred_df["hr_pct"] = (adjusted_probs * 100).round(1)
    pred_df["model_pct"] = (base_probs * 100).round(1)

    # Sort and display
    results = pred_df.sort_values("hr_probability", ascending=False)

    # Deduplicate — keep best row per player per day
    results = results.drop_duplicates(subset=["player_id", "game_date"], keep="first")

    # Filter out players with no recent data (all rolling windows are baseline)
    # Games played filter — require at least 5 games of Statcast data
    if "avg_ev_last30" in results.columns and "season_avg_ev" in results.columns:
        # If EV exactly matches season EV it means no recent data was found — flag those
        no_recent = (results["avg_ev_last30"] == results["season_avg_ev"])
        results.loc[no_recent, "hr_probability"] = results.loc[no_recent, "hr_probability"] * 0.7

    results = results.reset_index(drop=True)
    results.index += 1

    display_cols = [
        "player_name", "batting_team", "opposing_pitcher", "bat_side", "pitch_hand",
        "hr_pct", "model_pct", "barrel_rate_last15", "avg_ev_last15",
        "park_factor", "wind_label", "days_rest",
        "implied_team_runs", "platoon_advantage",
        "h2h_ab", "h2h_hr", "h2h_blended_hr_rate", "h2h_is_owned"
    ]
    display_cols = [c for c in display_cols if c in results.columns]

    print(f"\n🏟️  TOP {top_n} HR CANDIDATES — {date_str}")
    print("─" * 100)
    top = results.head(top_n)[display_cols].copy()
    top.columns = [c.replace("_", " ").title() for c in top.columns]
    print(top.to_string())
    print("─" * 100)
    print(f"\nTotal players analyzed: {len(results)}")
    print(f"  (Model Pct = raw XGBoost | HR Pct = after platoon/rest/wind/vegas adjustments)")

    # Save
    if output_path is None:
        os.makedirs("predictions", exist_ok=True)
        output_path = f"predictions/hr_predictions_{date_str}.csv"

    results.to_csv(output_path, index=False)
    print(f"\nFull predictions saved to {output_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate daily MLB HR predictions")
    parser.add_argument("--date", default=datetime.today().strftime("%Y-%m-%d"),
                        help="Date to predict (YYYY-MM-DD), defaults to today")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top players to display")
    parser.add_argument("--no-vegas", action="store_true",
                        help="Skip Vegas lines (use if no Odds API key)")
    args = parser.parse_args()

    run_daily_predictions(date_str=args.date, top_n=args.top, use_vegas=not args.no_vegas)
