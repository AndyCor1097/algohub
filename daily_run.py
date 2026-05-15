"""
daily_run.py — AlgoHub Daily Pre-Compute
Pulls Statcast, builds HIT scores for every batter on today's slate,
saves to data/today.json, then auto-pushes to GitHub.

Run every morning:
  python daily_run.py

Streamlit Cloud picks up the new file and serves it instantly to followers.
"""

import json
import os
import subprocess
import sys
import time
import requests
import pybaseball as pb
import pandas as pd
from datetime import datetime, timedelta, timezone

# Eastern Time helper
_ET = timezone(timedelta(hours=-4))
def et_now(): return datetime.now(_ET)
def et_today(): return et_now().strftime("%Y-%m-%d")

pb.cache.enable()

OUTPUT_PATH = "data/today.json"


# ── Park Factors ───────────────────────────────────────────────────────────────
PARK_FACTORS = {
    "COL":1.22,"CIN":1.15,"PHI":1.12,"NYY":1.10,"BOS":1.08,"TEX":1.07,
    "MIL":1.06,"BAL":1.05,"ATL":1.04,"CHC":1.03,"HOU":1.02,"KCR":1.02,
    "TOR":1.01,"MIN":1.00,"LAA":1.00,"CLE":0.99,"DET":0.98,"WSH":0.98,
    "STL":0.97,"NYM":0.97,"ARI":0.97,"TBR":0.96,"CWS":0.96,"PIT":0.95,
    "MIA":0.94,"SFG":0.93,"LAD":0.93,"ATH":0.92,"SEA":0.91,"SDP":0.90,
}
DOME_PARKS = {"TBR","TOR","HOU","MIA","ARI","MIL","ATH","TEX"}

TEAM_MAP = {
    108:"LAA", 109:"ARI", 110:"BAL", 111:"BOS", 112:"CHC",
    113:"CIN", 114:"CLE", 115:"COL", 116:"DET", 117:"HOU",
    118:"KCR", 119:"LAD", 120:"WSH", 121:"NYM", 133:"ATH",
    134:"PIT", 135:"SDP", 136:"SEA", 137:"SFG", 138:"STL",
    139:"TBR", 140:"TEX", 141:"TOR", 142:"MIN", 143:"PHI",
    144:"ATL", 145:"CWS", 146:"MIA", 147:"NYY", 158:"MIL",
}


def log(msg): print(f"  {msg}")


# ── Schedule ───────────────────────────────────────────────────────────────────
def get_schedule():
    today = et_today()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,team"
    try:
        r = requests.get(url, timeout=10)
        games = []
        for date in r.json().get("dates", []):
            for g in date.get("games", []):
                home = g["teams"]["home"]
                away = g["teams"]["away"]
                home_id = home["team"]["id"]
                away_id = away["team"]["id"]
                home_abbr = TEAM_MAP.get(home_id, home["team"].get("abbreviation","???"))
                away_abbr = TEAM_MAP.get(away_id, away["team"].get("abbreviation","???"))
                hp = home.get("probablePitcher", {})
                ap = away.get("probablePitcher", {})
                raw_time = g.get("gameDate", "")
                try:
                    dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    et = timezone(timedelta(hours=-4))
                    game_time_str = dt.astimezone(et).strftime("%I:%M %p ET").lstrip("0")
                except:
                    game_time_str = raw_time[11:16] if raw_time else "TBD"
                games.append({
                    "game_pk":         g["gamePk"],
                    "home_team":       home_abbr,
                    "away_team":       away_abbr,
                    "home_team_id":    home_id,
                    "away_team_id":    away_id,
                    "home_pitcher":    hp.get("fullName", "TBD"),
                    "away_pitcher":    ap.get("fullName", "TBD"),
                    "home_pitcher_id": hp.get("id"),
                    "away_pitcher_id": ap.get("id"),
                    "venue":           g.get("venue", {}).get("name", ""),
                    "game_time":       game_time_str,
                })
        return games
    except Exception as e:
        log(f"Schedule failed: {e}")
        return []


# ── Roster ─────────────────────────────────────────────────────────────────────
def get_roster(team_id: int) -> list:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    try:
        r = requests.get(url, timeout=10)
        players = []
        for p in r.json().get("roster", []):
            pos = p.get("position", {}).get("type", "")
            if pos != "Pitcher":
                players.append({
                    "player_id":   p["person"]["id"],
                    "player_name": p["person"]["fullName"],
                    "position":    p.get("position", {}).get("abbreviation", ""),
                })
        return players
    except:
        return []


# ── Player Hand ────────────────────────────────────────────────────────────────
_hand_cache = {}
def prefetch_hands(player_ids: list):
    """Bulk fetch handedness for a list of player IDs in one API call."""
    uncached = [pid for pid in player_ids if pid and pid not in _hand_cache]
    if not uncached:
        return
    # MLB API supports comma-separated player IDs
    chunks = [uncached[i:i+50] for i in range(0, len(uncached), 50)]
    for chunk in chunks:
        try:
            ids_str = ",".join(str(p) for p in chunk)
            r = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}", timeout=15)
            for p in r.json().get("people", []):
                pid  = p.get("id")
                bat  = p.get("batSide", {}).get("code", "R")
                hand = p.get("pitchHand", {}).get("code", "R")
                bat  = bat  if bat  in ("L","R","S") else "R"
                hand = hand if hand in ("L","R")     else "R"
                _hand_cache[int(pid)] = {"bat_side": bat, "pitch_hand": hand}
            time.sleep(0.3)
        except Exception as e:
            log(f"Bulk hand fetch failed: {e}")


def get_hand(player_id: int) -> dict:
    if not player_id: return {"bat_side": "R", "pitch_hand": "R"}
    if player_id in _hand_cache: return _hand_cache[player_id]
    try:
        time.sleep(0.15)
        r = requests.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=8)
        p = r.json().get("people", [{}])[0]
        bat  = p.get("batSide", {}).get("code", "R")
        hand = p.get("pitchHand", {}).get("code", "R")
        bat  = bat  if bat  in ("L","R","S") else "R"
        hand = hand if hand in ("L","R")     else "R"
        result = {"bat_side": bat, "pitch_hand": hand}
        _hand_cache[player_id] = result
        return result
    except:
        return {"bat_side": "R", "pitch_hand": "R"}


# ── Pitcher Stats ──────────────────────────────────────────────────────────────
_pitcher_stats_cache = {}
def get_pitcher_stats(pitcher_id: int) -> dict:
    if not pitcher_id: return {"era": 4.50, "hr9": 1.10, "hrfb": 0.12}
    if pitcher_id in _pitcher_stats_cache: return _pitcher_stats_cache[pitcher_id]
    season = et_now().year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
    try:
        r = requests.get(url, timeout=8)
        stat = r.json().get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
        era  = float(stat.get("era", 4.50) or 4.50)
        hr9  = float(stat.get("homeRunsPer9", 1.10) or 1.10)
        ip   = float(stat.get("inningsPitched", 0) or 0)
        hrs  = int(stat.get("homeRuns", 0) or 0)
        hrfb = hrs / max(ip / 9 * 0.35, 1)
        result = {"era": era, "hr9": hr9, "hrfb": round(hrfb, 3)}
        _pitcher_stats_cache[pitcher_id] = result
        return result
    except:
        return {"era": 4.50, "hr9": 1.10, "hrfb": 0.12}


# ── Weather ────────────────────────────────────────────────────────────────────
VENUE_COORDS = {
    "Yankee Stadium": (40.829, -73.926),
    "Fenway Park": (42.347, -71.097),
    "Wrigley Field": (41.948, -87.655),
    "Citizens Bank Park": (39.906, -75.166),
    "Truist Park": (33.890, -84.468),
    "Great American Ball Park": (39.097, -84.507),
    "Busch Stadium": (38.623, -90.193),
    "Dodger Stadium": (34.074, -118.240),
    "Oracle Park": (37.778, -122.389),
    "Kauffman Stadium": (39.051, -94.480),
    "Camden Yards": (39.284, -76.622),
    "Globe Life Field": (32.751, -97.083),
    "Minute Maid Park": (29.757, -95.355),
    "T-Mobile Park": (47.591, -122.333),
    "Angel Stadium": (33.800, -117.883),
    "Coors Field": (39.756, -104.994),
    "PNC Park": (40.447, -80.006),
    "American Family Field": (43.028, -87.971),
    "Target Field": (44.982, -93.278),
    "Guaranteed Rate Field": (41.830, -87.634),
    "Progressive Field": (41.496, -81.685),
    "Comerica Park": (42.339, -83.049),
    "Nationals Park": (38.873, -77.007),
    "loanDepot park": (25.778, -80.220),
    "Petco Park": (32.707, -117.157),
    "Chase Field": (33.445, -112.067),
    "Citi Field": (40.757, -73.846),
    "Tropicana Field": (27.768, -82.653),
    "Rogers Centre": (43.641, -79.389),
}

# Park CF orientation in degrees (direction CF faces FROM home plate)
# Wind blowing FROM this direction = blowing IN, blowing TO = blowing OUT
PARK_CF_DIRECTION = {
    "Yankee Stadium":              315,  # CF faces NW
    "Fenway Park":                  60,  # CF faces NE
    "Wrigley Field":                45,  # CF faces NE
    "Citizens Bank Park":          315,  # CF faces NW
    "Truist Park":                 330,  # CF faces NNW
    "Great American Ball Park":    300,  # CF faces WNW
    "Busch Stadium":               315,  # CF faces NW
    "Dodger Stadium":               45,  # CF faces NE
    "Oracle Park":                 315,  # CF faces NW
    "Kauffman Stadium":            315,  # CF faces NW
    "Oriole Park at Camden Yards": 330,  # CF faces NNW
    "Globe Life Field":            300,  # CF faces WNW (dome-ish)
    "Minute Maid Park":            330,  # CF faces NNW
    "T-Mobile Park":                 0,  # CF faces N
    "Angel Stadium":                45,  # CF faces NE
    "Coors Field":                 315,  # CF faces NW
    "PNC Park":                    315,  # CF faces NW
    "American Family Field":       315,  # CF faces NW
    "Target Field":                  0,  # CF faces N
    "Guaranteed Rate Field":       315,  # CF faces NW
    "Progressive Field":           330,  # CF faces NNW
    "Comerica Park":               315,  # CF faces NW
    "Nationals Park":              315,  # CF faces NW
    "loanDepot park":              315,  # CF faces NW
    "Petco Park":                  315,  # CF faces NW
    "Chase Field":                 315,  # CF faces NW
    "Citi Field":                    0,  # CF faces N
    "Sutter Health Park":          270,  # CF faces W
    "Rate Field":                  315,  # CF faces NW
}


def wind_direction_boost(wind_mph: float, wind_deg: float, venue: str) -> tuple:
    """
    Calculate wind boost based on direction relative to park orientation.
    Returns (boost_value, direction_label)
    boost > 0 = blowing out (HR friendly)
    boost < 0 = blowing in (HR suppressing)
    """
    if wind_mph < 3:
        return 0.0, f"💨 {wind_mph:.0f}mph"

    # Find park CF direction
    cf_dir = None
    for name, deg in PARK_CF_DIRECTION.items():
        if name.lower() in venue.lower() or venue.lower() in name.lower():
            cf_dir = deg
            break

    if cf_dir is None:
        # Unknown park — use neutral boost
        boost = round(wind_mph * 0.06, 2)
        return boost, f"💨 {wind_mph:.0f}mph"

    # Calculate angle between wind direction and CF direction
    # Wind direction = where wind is coming FROM
    # If wind comes FROM direction opposite to CF = blowing OUT to CF = good
    wind_from = wind_deg
    cf_facing = cf_dir

    # Angle difference between wind FROM and CF direction
    diff = abs(((wind_from - cf_facing) + 180) % 360 - 180)

    # diff = 0 → wind blowing straight out to CF (best)
    # diff = 180 → wind blowing straight in from CF (worst)
    # diff = 90 → crosswind (neutral)
    import math
    direction_factor = math.cos(math.radians(diff))  # 1.0 = out, -1.0 = in

    boost = round(wind_mph * direction_factor * 0.15, 2)

    if direction_factor > 0.5:
        arrow = "⬆️"  # blowing out
        label = f"⬆️ {wind_mph:.0f}mph OUT"
    elif direction_factor < -0.5:
        arrow = "⬇️"  # blowing in
        label = f"⬇️ {wind_mph:.0f}mph IN"
    else:
        label = f"↔️ {wind_mph:.0f}mph X"

    return boost, label


def get_weather(venue: str, is_dome: bool) -> dict:
    if is_dome:
        return {"temp_f": 72, "wind_mph": 0, "wind_boost": 0, "wind_label": "🏟️ DOME",
                "wind_dir": 0, "wind_direction_label": "DOME"}
    coords = None
    for name, c in VENUE_COORDS.items():
        if name.lower() in venue.lower() or venue.lower() in name.lower():
            coords = c
            break
    if not coords:
        return {"temp_f": 70, "wind_mph": 0, "wind_boost": 0, "wind_label": "Unknown",
                "wind_dir": 0, "wind_direction_label": ""}
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&current=temperature_2m,wind_speed_10m,wind_direction_10m,precipitation&temperature_unit=fahrenheit&wind_speed_unit=mph"
        r = requests.get(url, timeout=8)
        d = r.json().get("current", {})
        temp     = round(float(d.get("temperature_2m", 70)), 1)
        wind     = round(float(d.get("wind_speed_10m", 0)), 1)
        wind_dir = float(d.get("wind_direction_10m", 0))

        boost, dir_label = wind_direction_boost(wind, wind_dir, venue)

        label = f"🌡️ {temp:.0f}°F {dir_label}"
        return {
            "temp_f":               temp,
            "wind_mph":             wind,
            "wind_boost":           boost,
            "wind_label":           label,
            "wind_dir":             wind_dir,
            "wind_direction_label": dir_label,
        }
    except:
        return {"temp_f": 70, "wind_mph": 0, "wind_boost": 0, "wind_label": "Unknown",
                "wind_dir": 0, "wind_direction_label": ""}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"  AlgoHub Daily Run — {et_now().strftime('%A, %B %d %Y')}")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)

    # 1. Build HIT Score engine
    print("\n[1/5] Loading Statcast data (30 days)...")
    end   = et_now()
    start = end - timedelta(days=30)
    raw = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    log(f"Got {len(raw):,} pitch records")

    from hit_score import HITScoreEngine
    engine = HITScoreEngine(raw)

    # 2. Get today's schedule
    print("\n[2/5] Fetching schedule...")
    games = get_schedule()
    log(f"Found {len(games)} games")

    # 3. Compute scores for every batter in every game
    print("\n[3/5] Computing HIT scores...")
    output_games = []

    for g in games:
        home_team = g["home_team"]
        away_team = g["away_team"]
        is_dome   = home_team in DOME_PARKS
        pf        = PARK_FACTORS.get(home_team, 1.0)
        weather   = get_weather(g["venue"], is_dome)

        home_pitcher_id = g.get("home_pitcher_id")
        away_pitcher_id = g.get("away_pitcher_id")
        home_pitcher_hand = get_hand(home_pitcher_id).get("pitch_hand", "R") if home_pitcher_id else "R"
        away_pitcher_hand = get_hand(away_pitcher_id).get("pitch_hand", "R") if away_pitcher_id else "R"

        home_p_stats = get_pitcher_stats(home_pitcher_id)
        away_p_stats = get_pitcher_stats(away_pitcher_id)

        home_roster = get_roster(g["home_team_id"])
        away_roster = get_roster(g["away_team_id"])

        # Bulk prefetch all player hands in one API call
        all_pids = [p["player_id"] for p in home_roster + away_roster]
        if home_pitcher_id: all_pids.append(home_pitcher_id)
        if away_pitcher_id: all_pids.append(away_pitcher_id)
        prefetch_hands(all_pids)

        def score_lineup(batters, pitcher_id, pitcher_hand, p_stats):
            results = []
            for player in batters:
                pid  = player["player_id"]
                hand = get_hand(pid)
                hit_data = engine.compute_hit_score(
                    batter_id    = pid,
                    pitcher_id   = pitcher_id or 0,
                    bat_side     = hand["bat_side"],
                    pitch_hand   = pitcher_hand,
                    park_factor  = pf,
                    wind_boost   = weather["wind_boost"],
                    temp_f       = weather["temp_f"],
                    pitcher_era  = p_stats["era"],
                    pitcher_hr9  = p_stats["hr9"],
                    pitcher_hrfb = p_stats["hrfb"],
                )
                k_data = engine.get_k_score(
                    pitcher_id = pitcher_id or 0,
                    batter_id  = pid,
                    bat_side   = hand["bat_side"],
                )
                results.append({
                    "player_id":      pid,
                    "player_name":    player["player_name"],
                    "position":       player["position"],
                    "bat_side":       hand["bat_side"],
                    "hit_score":      hit_data["hit_score"],
                    "grade":          hit_data["grade"],
                    "zone_fit":       hit_data["zone_fit"],
                    "zone_count":     hit_data["zone_count"],
                    "barrel_rate":    hit_data["barrel_rate"],
                    "hard_hit_pct":   hit_data["hard_hit_pct"],
                    "xwoba":          hit_data["xwoba"],
                    "la_consistency": hit_data["la_consistency"],
                    "fb_rate":        hit_data.get("fb_rate", 0),
                    "hr_fb_rate":     hit_data.get("hr_fb_rate", 0),
                    "pull_rate":      hit_data["pull_rate"],
                    "avg_ev":         hit_data["avg_ev"],
                    "avg_la":         hit_data["avg_la"],
                    "hr_rate":        hit_data["hr_rate"],
                    "proj_hr_pct":    hit_data["proj_hr_pct"],
                    "edge_pitch":     hit_data["edge_pitch"],
                    "pitch_matchup":  hit_data["weighted_barrel"],
                    "platoon_score":  hit_data["platoon_score"],
                    "heat_score":     hit_data.get("heat_score", 0),
                    "barrel_7":       hit_data.get("barrel_7", 0),
                    "hh_7":           hit_data.get("hh_7", 0),
                    "xwoba_7":        hit_data.get("xwoba_7"),
                    "ev_7":           hit_data.get("ev_7", 0),
                    "k_score":        k_data.get("k_score", 50),
                    "k_grade":        k_data.get("grade", "MODERATE"),
                    # Component scores for Zone Maps breakdown
                    "barrel_score":   hit_data.get("barrel_score", 0),
                    "hh_score":       hit_data.get("hh_score", 0),
                    "xwoba_score":    hit_data.get("xwoba_score", 0),
                    "la_score":       hit_data.get("la_score", 0),
                    "fb_score":       hit_data.get("fb_score", 0),
                    "hrfb_score":     hit_data.get("hrfb_score", 0),
                    "ev_score":       hit_data.get("ev_score", 0),
                    "pull_score":     hit_data.get("pull_score", 0),
                    "swstr_bonus":    hit_data.get("swstr_bonus", 0),
                    "bat_speed_bonus": hit_data.get("bat_speed_bonus", 0),
                    "squared_bonus":  hit_data.get("squared_bonus", 0),
                    "chase_penalty":  hit_data.get("chase_penalty", 0),
                    "pitcher_score":  hit_data.get("pitcher_score", 0),
                    "env_score":      hit_data.get("env_score", 0),
                    "form_score":     hit_data.get("form_score", 0),
                })
            results.sort(key=lambda x: x["hit_score"], reverse=True)
            return results

        # Away pitcher vs home batters
        home_scored = score_lineup(home_roster, away_pitcher_id, away_pitcher_hand, away_p_stats)
        # Home pitcher vs away batters
        away_scored = score_lineup(away_roster, home_pitcher_id, home_pitcher_hand, home_p_stats)

        log(f"{away_team} @ {home_team} — {len(home_scored)} + {len(away_scored)} batters scored")

        output_games.append({
            "game_pk":            g["game_pk"],
            "home_team":          home_team,
            "away_team":          away_team,
            "venue":              g["venue"],
            "game_time":          g["game_time"],
            "park_factor":        pf,
            "is_dome":            is_dome,
            "weather":            weather,
            "home_pitcher":       g["home_pitcher"],
            "away_pitcher":       g["away_pitcher"],
            "home_pitcher_id":    home_pitcher_id,
            "away_pitcher_id":    away_pitcher_id,
            "home_pitcher_hand":  home_pitcher_hand,
            "away_pitcher_hand":  away_pitcher_hand,
            "home_pitcher_era":   home_p_stats["era"],
            "away_pitcher_era":   away_p_stats["era"],
            "home_pitcher_bip":   engine._pitcher_index.get(int(home_pitcher_id), {}).get("bip", 0) if home_pitcher_id and engine else 0,
            "away_pitcher_bip":   engine._pitcher_index.get(int(away_pitcher_id), {}).get("bip", 0) if away_pitcher_id and engine else 0,
            "home_pitcher_krate": engine._pitcher_index.get(int(home_pitcher_id), {}).get("k_rate", 0) if home_pitcher_id and engine else 0,
            "away_pitcher_krate": engine._pitcher_index.get(int(away_pitcher_id), {}).get("k_rate", 0) if away_pitcher_id and engine else 0,
            "home_proj_ks":       engine.compute_proj_ks(home_pitcher_id, away_scored) if home_pitcher_id and engine else 0,
            "away_proj_ks":       engine.compute_proj_ks(away_pitcher_id, home_scored) if away_pitcher_id and engine else 0,
            # home batters face away pitcher
            "home_batters":       home_scored,
            # away batters face home pitcher
            "away_batters":       away_scored,
        })

    # 4. Save to JSON
    print("\n[4/5] Saving data...")
    output = {
        "date":       et_today(),
        "generated":  datetime.now().isoformat(),
        "games":      output_games,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f)
    log(f"Saved {len(output_games)} games to {OUTPUT_PATH}")

    # 5. Git push
    print("\n[5/5] Pushing to GitHub...")
    try:
        subprocess.run(["git", "add", OUTPUT_PATH], check=True)
        subprocess.run(["git", "commit", "-m", f"Daily data {et_now().strftime('%Y-%m-%d')}"], check=True)
        subprocess.run(["git", "push"], check=True)
        log("Pushed to GitHub ✓")
    except subprocess.CalledProcessError as e:
        log(f"Git push failed: {e}")
        log("Make sure git is configured and you have a remote set up")

    print(f"\n✓ Done! {len(output_games)} games ready.")
    print(f"  Share: https://thealgohub.streamlit.app")


if __name__ == "__main__":
    main()
