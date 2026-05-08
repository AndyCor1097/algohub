"""
algohub.py — AlgoHub Full Dashboard
Game-first navigation. HIT Score. Zone heatmaps. K props. Parlay builder.

Run with: python -m streamlit run algohub.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
import os
import sys
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AlgoHub",
    page_icon="💣",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:      #06080f;
    --s1:      #0c1018;
    --s2:      #111827;
    --border:  #1c2333;
    --accent:  #3b82f6;
    --green:   #22c55e;
    --red:     #ef4444;
    --amber:   #f59e0b;
    --purple:  #a855f7;
    --cyan:    #06b6d4;
    --text:    #f1f5f9;
    --muted:   #475569;
}

* { box-sizing: border-box; }
html, body, [class*="css"] { background: var(--bg) !important; color: var(--text); font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

/* Header */
.ah-logo { font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem; letter-spacing: 0.15em; color: var(--text); }
.ah-logo span { color: var(--accent); }
.ah-tagline { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); letter-spacing: 0.2em; text-transform: uppercase; }

/* Game cards */
.game-card {
    background: var(--s1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.game-card:hover { border-color: var(--accent); }
.game-card.active { border-color: var(--accent); background: #0f1929; }
.game-matchup { font-family: 'Bebas Neue', sans-serif; font-size: 1.1rem; letter-spacing: 0.08em; }
.game-info { font-size: 0.7rem; color: var(--muted); margin-top: 2px; font-family: 'DM Mono', monospace; }
.game-ou { font-size: 0.75rem; color: var(--cyan); }

/* Section headers */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    padding: 6px 0 4px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

/* Batter rows */
.batter-row {
    display: grid;
    grid-template-columns: 28px 185px 65px 80px 65px 65px 65px 70px 55px 60px 75px;
    align-items: center;
    padding: 7px 10px;
    border-radius: 6px;
    margin-bottom: 3px;
    background: var(--s1);
    border: 1px solid transparent;
    transition: border-color 0.1s;
    gap: 4px;
}
.batter-row:hover { border-color: var(--border); }
.batter-row.elite { border-left: 3px solid var(--red) !important; }
.batter-row.strong { border-left: 3px solid var(--amber) !important; }
.batter-row.moderate { border-left: 3px solid var(--accent) !important; }

.batter-rank { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--muted); }
.batter-name { font-weight: 600; font-size: 0.88rem; }
.batter-stat { font-family: 'DM Mono', monospace; font-size: 0.78rem; text-align: right; }
.batter-header { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--muted); text-align: right; }

/* HIT Score badge */
.hit-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-family: 'Bebas Neue', sans-serif; font-size: 0.9rem; letter-spacing: 0.05em; }
.hit-elite    { background: #450a0a33; color: #fca5a5; border: 1px solid #ef444466; }
.hit-strong   { background: #45260333; color: #fcd34d; border: 1px solid #f59e0b66; }
.hit-moderate { background: #0c2d4833; color: #7dd3fc; border: 1px solid #3b82f666; }
.hit-fade     { background: #11182733; color: #475569; border: 1px solid #1c233366; }

/* Zone badge */
.zone-badge { background: #0c2d48; color: #60a5fa; border: 1px solid #1e4d7a; padding: 1px 6px; border-radius: 3px; font-family: 'DM Mono', monospace; font-size: 0.72rem; font-weight: 600; }

/* Hot bat */
.hot  { color: #ef4444; font-size: 0.72rem; }
.warm { color: #f59e0b; font-size: 0.72rem; }
.cold { color: #475569; font-size: 0.72rem; }

/* Value badge */
.value-pos { color: var(--green); font-family: 'DM Mono', monospace; font-size: 0.75rem; font-weight: 600; }
.value-neg { color: var(--muted); font-family: 'DM Mono', monospace; font-size: 0.75rem; }

/* Odds */
.odds-display { font-family: 'DM Mono', monospace; font-size: 0.8rem; color: var(--amber); }

/* Pitcher card */
.pitcher-card {
    background: var(--s2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.pitcher-name { font-family: 'Bebas Neue', sans-serif; font-size: 1.2rem; letter-spacing: 0.08em; }
.pitcher-stats { font-family: 'DM Mono', monospace; font-size: 0.72rem; color: var(--muted); margin-top: 4px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--s1) !important; border-radius: 8px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif !important; font-size: 0.82rem !important; font-weight: 500 !important; color: var(--muted) !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"] { background: var(--s2) !important; color: var(--text) !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding: 12px 0 0 !important; }

/* Parlay */
.parlay-leg { background: var(--s1); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 6px; }
.parlay-odds { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: var(--green); letter-spacing: 0.05em; }

/* Metric cells */
.stat-cell-green { color: #22c55e; }
.stat-cell-amber { color: #f59e0b; }
.stat-cell-red   { color: #ef4444; }
.stat-cell-muted { color: #475569; }

.stDataFrame { font-family: 'DM Mono', monospace !important; }
div[data-testid="metric-container"] { background: var(--s1); border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_schedule():
    """Load today's MLB schedule from MLB Stats API."""
    today = datetime.today().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineScore,team"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        games = []
        for date in data.get("dates", []):
            for g in date.get("games", []):
                home = g.get("teams", {}).get("home", {})
                away = g.get("teams", {}).get("away", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})
                game_time = g.get("gameDate", "")
                if game_time:
                    try:
                        from datetime import timezone
                        dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                        local = dt.astimezone().strftime("%-I:%M %p")
                    except:
                        local = game_time[11:16]
                else:
                    local = "TBD"

                games.append({
                    "game_pk":         g.get("gamePk"),
                    "home_team":       home.get("team", {}).get("abbreviation", ""),
                    "away_team":       away.get("team", {}).get("abbreviation", ""),
                    "home_team_id":    home.get("team", {}).get("id"),
                    "away_team_id":    away.get("team", {}).get("id"),
                    "home_pitcher":    home_pitcher.get("fullName", "TBD"),
                    "away_pitcher":    away_pitcher.get("fullName", "TBD"),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "game_time":       local,
                    "venue":           g.get("venue", {}).get("name", ""),
                    "status":          g.get("status", {}).get("abstractGameState", ""),
                })
        return games
    except Exception as e:
        st.error(f"Schedule load failed: {e}")
        return []


@st.cache_data(ttl=300)
def load_roster(team_id: int) -> list:
    """Load active roster for a team."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        players = []
        for p in data.get("roster", []):
            person = p.get("person", {})
            pos    = p.get("position", {})
            if pos.get("type") not in ["Pitcher"]:
                players.append({
                    "player_id":   person.get("id"),
                    "player_name": person.get("fullName", ""),
                    "position":    pos.get("abbreviation", ""),
                })
        return players
    except:
        return []


_HAND_CACHE = {}

def load_player_hand(player_id: int) -> dict:
    """Get bat side and pitch hand from MLB Stats API."""
    if not player_id:
        return {"bat_side": "R", "pitch_hand": "R"}
    if player_id in _HAND_CACHE:
        return _HAND_CACHE[player_id]
    # No fields filter — it strips nested objects
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    try:
        r = requests.get(url, timeout=8)
        person = r.json().get("people", [{}])[0]
        bat  = person.get("batSide",   {}).get("code", "R")
        hand = person.get("pitchHand", {}).get("code", "R")
        bat  = bat  if bat  in ("L","R","S") else "R"
        hand = hand if hand in ("L","R")     else "R"
        result = {"bat_side": bat, "pitch_hand": hand}
        _HAND_CACHE[player_id] = result
        return result
    except:
        return {"bat_side": "R", "pitch_hand": "R"}


@st.cache_data(ttl=600)
def load_statcast_engine():
    """Load bulk Statcast data and build HIT Score engine."""
    try:
        import pybaseball as pb
        from hit_score import HITScoreEngine
        pb.cache.enable()
        end   = datetime.today()
        start = end - timedelta(days=30)
        st.info("Loading Statcast data (30 days)...")
        raw = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        engine = HITScoreEngine(raw)
        return engine
    except Exception as e:
        st.warning(f"Statcast engine failed: {e}")
        return None


@st.cache_data(ttl=3600)
def load_pitcher_stats(pitcher_id: int) -> dict:
    """Pull pitcher season stats from MLB Stats API."""
    if not pitcher_id:
        return {}
    season = datetime.today().year
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&season={season}&group=pitching"
    try:
        r = requests.get(url, timeout=8)
        stats = r.json().get("stats", [{}])[0].get("splits", [{}])[0].get("stat", {})
        era  = float(stats.get("era", 4.50))
        hr9  = float(stats.get("homeRunsPer9", 1.10))
        ip   = float(stats.get("inningsPitched", 0) or 0)
        hrs  = int(stats.get("homeRuns", 0) or 0)
        bf   = int(stats.get("battersFaced", 1) or 1)
        # Hard hit % not in MLB API — use default
        return {
            "era":    era,
            "hr9":    hr9,
            "ip":     ip,
            "hrs":    hrs,
            "hrfb":   hrs / max(ip / 9 * 0.35, 1),  # rough HR/FB estimate
        }
    except:
        return {"era": 4.50, "hr9": 1.10, "ip": 0, "hrs": 0, "hrfb": 0.12}


@st.cache_data(ttl=300)
def load_bovada_odds() -> dict:
    """Pull HR prop odds from Bovada. Returns {player_name_lower: odds}"""
    try:
        from bovada_odds import get_mlb_hr_props
        props = get_mlb_hr_props()
        if not props.empty:
            return dict(zip(props["player_name"].str.lower(), props["hr_odds"]))
    except:
        pass
    return {}


@st.cache_data(ttl=300)
def load_weather(venue: str) -> dict:
    """Get weather for a venue."""
    # Venue coordinate lookup
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
    }
    coords = None
    for name, c in VENUE_COORDS.items():
        if name.lower() in venue.lower() or venue.lower() in name.lower():
            coords = c
            break

    if not coords:
        return {"temp_f": 70, "wind_mph": 0, "wind_dir": "", "precip": 0, "conditions": "Unknown"}

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords[0]}&longitude={coords[1]}&current=temperature_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
        r = requests.get(url, timeout=8)
        d = r.json().get("current", {})
        return {
            "temp_f":     round(d.get("temperature_2m", 70), 1),
            "wind_mph":   round(d.get("wind_speed_10m", 0), 1),
            "wind_dir":   d.get("wind_direction_10m", 0),
            "precip":     d.get("precipitation", 0),
            "conditions": "Clear" if d.get("weather_code", 0) < 3 else "Cloudy",
        }
    except:
        return {"temp_f": 70, "wind_mph": 0, "wind_dir": 0, "precip": 0, "conditions": "Unknown"}


# ── Park Factors ───────────────────────────────────────────────────────────────
PARK_FACTORS = {
    "COL":1.22,"CIN":1.15,"PHI":1.12,"NYY":1.10,"BOS":1.08,"TEX":1.07,
    "MIL":1.06,"BAL":1.05,"ATL":1.04,"CHC":1.03,"HOU":1.02,"KCR":1.02,
    "TOR":1.01,"MIN":1.00,"LAA":1.00,"CLE":0.99,"DET":0.98,"WSH":0.98,
    "STL":0.97,"NYM":0.97,"ARI":0.97,"TBR":0.96,"CWS":0.96,"PIT":0.95,
    "MIA":0.94,"SFG":0.93,"LAD":0.93,"ATH":0.92,"SEA":0.91,"SDP":0.90,
}

DOME_PARKS = {"TBR","TOR","HOU","MIA","ARI","MIL","ATH","TEX"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def hot_bat(hr_rate):
    if hr_rate >= 0.25:   return "🔥", "hot"
    elif hr_rate >= 0.10: return "🌡️", "warm"
    else:                 return "❄️", "cold"


def color_stat(val, low, high):
    if pd.isna(val) or val == 0: return "stat-cell-muted"
    if val >= high:  return "stat-cell-green"
    if val >= low:   return "stat-cell-amber"
    return "stat-cell-red"


def build_zone_fig(zone_data: dict, title: str, colorscale="Blues", height=180):
    grid = np.zeros((3, 3))
    zone_map = {1:(0,0),2:(0,1),3:(0,2),4:(1,0),5:(1,1),6:(1,2),7:(2,0),8:(2,1),9:(2,2)}
    for z,(r,c) in zone_map.items():
        grid[r][c] = zone_data.get(z, 0)
    text = [[str(int(grid[r][c])) if grid[r][c]>0 else "·" for c in range(3)] for r in range(3)]
    fig = go.Figure(go.Heatmap(
        z=grid, text=text, texttemplate="%{text}",
        textfont={"size":16,"color":"white","family":"DM Mono"},
        colorscale=colorscale, showscale=False,
        zmin=0, zmax=max(1, np.max(grid)),
    ))
    fig.add_shape(type="rect",x0=-0.5,y0=-0.5,x1=2.5,y1=2.5,line=dict(color="#3b82f6",width=2))
    for i in [0.5,1.5]:
        fig.add_shape(type="line",x0=i,y0=-0.5,x1=i,y1=2.5,line=dict(color="#1c2333",width=1))
        fig.add_shape(type="line",x0=-0.5,y0=i,x1=2.5,y1=i,line=dict(color="#1c2333",width=1))
    fig.update_layout(
        title=dict(text=title,font=dict(family="Bebas Neue",size=12,color="#475569"),x=0.5),
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        margin=dict(l=5,r=5,t=25,b=5), height=height,
        plot_bgcolor="#0c1018", paper_bgcolor="#0c1018",
    )
    return fig


def build_overlap_fig(batter_zones: dict, pitcher_zones: dict, height=180):
    zone_map = {1:(0,0),2:(0,1),3:(0,2),4:(1,0),5:(1,1),6:(1,2),7:(2,0),8:(2,1),9:(2,2)}
    grid = np.zeros((3,3))
    text = [[""]*3 for _ in range(3)]
    hover = [[""]*3 for _ in range(3)]
    for z,(r,c) in zone_map.items():
        b = batter_zones.get(z,0); p = pitcher_zones.get(z,0)
        if b>0 and p>0:
            grid[r][c]=3; text[r][c]="⚡"; hover[r][c]=f"KILL ZONE\nBatter:{b}HR\nPitcher:{p}HR"
        elif p>0:
            grid[r][c]=2; text[r][c]="●"; hover[r][c]=f"Pitcher weak:{p}HR"
        elif b>0:
            grid[r][c]=1; text[r][c]="○"; hover[r][c]=f"Batter crush:{b}HR"
    cs = [[0.0,"#0c1018"],[0.33,"#1e3a5f"],[0.66,"#7f1d1d"],[1.0,"#dc2626"]]
    fig = go.Figure(go.Heatmap(
        z=grid, text=text, customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{text}", textfont={"size":14,"color":"white"},
        colorscale=cs, showscale=False, zmin=0, zmax=3,
    ))
    fig.add_shape(type="rect",x0=-0.5,y0=-0.5,x1=2.5,y1=2.5,line=dict(color="#3b82f6",width=2))
    for i in [0.5,1.5]:
        fig.add_shape(type="line",x0=i,y0=-0.5,x1=i,y1=2.5,line=dict(color="#1c2333",width=1))
        fig.add_shape(type="line",x0=-0.5,y0=i,x1=2.5,y1=i,line=dict(color="#1c2333",width=1))
    fig.update_layout(
        title=dict(text="⚡ KILL ZONES",font=dict(family="Bebas Neue",size=12,color="#475569"),x=0.5),
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        margin=dict(l=5,r=5,t=25,b=5), height=height,
        plot_bgcolor="#0c1018", paper_bgcolor="#0c1018",
    )
    return fig


def render_batter_row(rank, player, hit_data, odds=None):
    name    = player.get("player_name", "—")
    grade   = hit_data.get("grade", "MODERATE").lower()
    score   = hit_data.get("hit_score", 0)
    brl     = hit_data.get("barrel_rate", 0)
    ev      = hit_data.get("avg_ev", 0)
    hr_r    = hit_data.get("hr_rate", 0)
    zone_fit = hit_data.get("zone_fit", 0)
    pitch_ms = hit_data.get("pitch_matchup_score", hit_data.get("weighted_barrel", 0))
    proj    = hit_data.get("proj_hr_pct", 0)
    edge    = hit_data.get("edge_pitch", "")
    bat_side = hit_data.get("bat_side", "R")
    bat_icon, _ = hot_bat(hr_r / 100 if hr_r > 1 else hr_r)
    edge_str = {"fb":"🔥 FB","brk":"🌀 BRK","off":"🎯 OFF"}.get(edge or "","")
    hand_str = f"<span style='font-size:.65rem;color:#475569'>{bat_side}HB</span>"
    hh    = hit_data.get("hard_hit_pct", 0)
    xwoba = hit_data.get("xwoba") or 0
    la_c  = hit_data.get("la_consistency", 0)
    pull  = hit_data.get("pull_rate", 0)

    # ZF score
    zf = min((zone_fit * 0.6) + (min(pitch_ms / 0.15, 1.0) * 0.4), 1.0)
    zf_color = "#22c55e" if zf >= 0.7 else "#f59e0b" if zf >= 0.4 else "#475569"

    st.markdown(f"""
    <div class="batter-row {grade}">
        <span class="batter-rank">{rank}</span>
        <span class="batter-name">{name} {bat_icon} {hand_str}</span>
        <span class="hit-badge hit-{grade}">{score:.0f}</span>
        <span class="batter-stat" style="color:{zf_color};font-family:'DM Mono',monospace;font-weight:600">ZF {zf:.3f}</span>
        <span class="batter-stat {'stat-cell-green' if brl>=15 else 'stat-cell-amber' if brl>=8 else 'stat-cell-muted'}">{brl:.1f}%</span>
        <span class="batter-stat {'stat-cell-green' if hh>=50 else 'stat-cell-amber' if hh>=40 else 'stat-cell-muted'}">{hh:.1f}%</span>
        <span class="batter-stat {'stat-cell-green' if ev>=92 else 'stat-cell-amber' if ev>=88 else 'stat-cell-muted'}">{ev:.1f}</span>
        <span class="batter-stat {'stat-cell-green' if xwoba>=0.360 else 'stat-cell-amber' if xwoba>=0.300 else 'stat-cell-muted'}">{xwoba:.3f}</span>
        <span class="batter-stat {'stat-cell-green' if la_c>=35 else 'stat-cell-amber' if la_c>=25 else 'stat-cell-muted'}">{la_c:.0f}%</span>
        <span class="batter-stat">{proj:.1f}%</span>
        <span style="font-size:0.7rem;color:#475569">{edge_str}</span>
    </div>
    """, unsafe_allow_html=True)


def render_lineup_section(title, pitcher_name, pitcher_id, pitcher_hand, batters, home_team, engine, odds_lookup, park_factor, weather):
    """Render one side of a game matchup."""
    is_dome = home_team in DOME_PARKS
    wind_boost = 0 if is_dome else max(weather.get("wind_mph", 0) * 0.1, 0)

    st.markdown(f"""
    <div class="pitcher-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:0.7rem;color:#475569;font-family:'DM Mono',monospace;letter-spacing:.15em">PITCHER</span><br>
                <span class="pitcher-name">{pitcher_name}</span>
                <span style="font-size:0.75rem;color:#475569;margin-left:8px">{'RHP' if pitcher_hand=='R' else 'LHP'}</span>
            </div>
            <div style="text-align:right;font-family:'DM Mono',monospace;font-size:0.72rem;color:#475569">
                PF {park_factor:.2f} &nbsp;|&nbsp;
                {'🏟️ DOME' if is_dome else f'🌡️ {weather.get("temp_f",70):.0f}°F  💨 {weather.get("wind_mph",0):.0f}mph'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Column headers
    st.markdown("""
    <div class="batter-row" style="border:none;background:transparent;padding-bottom:2px;">
        <span class="batter-header">#</span>
        <span class="batter-header">BATTER</span>
        <span class="batter-header">HIT</span>
        <span class="batter-header">ZF</span>
        <span class="batter-header">BBL%</span>
        <span class="batter-header">HH%</span>
        <span class="batter-header">EV</span>
        <span class="batter-header">xwOBA</span>
        <span class="batter-header">LA%</span>
        <span class="batter-header">PROJ%</span>
        <span class="batter-header">EDGE</span>
    </div>
    """, unsafe_allow_html=True)

    results = []
    for player in batters:
        pid = player.get("player_id")
        if not pid: continue
        hand = load_player_hand(pid)
        bat_side = hand.get("bat_side", "R")

        if engine:
            p_stats = load_pitcher_stats(pitcher_id) if pitcher_id else {}
            hit_data = engine.compute_hit_score(
                batter_id    = pid,
                pitcher_id   = pitcher_id or 0,
                bat_side     = bat_side,
                pitch_hand   = pitcher_hand,
                park_factor  = park_factor,
                wind_boost   = wind_boost,
                temp_f       = weather.get("temp_f", 70),
                pitcher_era  = p_stats.get("era", 4.50),
                pitcher_hr9  = p_stats.get("hr9", 1.10),
                pitcher_hrfb = p_stats.get("hrfb", 0.12),
            )
        else:
            hit_data = {"hit_score": 0, "grade": "MODERATE", "barrel_rate": 0,
                       "avg_ev": 0, "hr_rate": 0, "zone_count": 0, "proj_hr_pct": 0}

        hit_data["player_name"] = player.get("player_name","")
        hit_data["player_id"]   = pid
        hit_data["bat_side"]    = bat_side
        results.append(hit_data)

    # Sort by HIT Score
    results.sort(key=lambda x: x.get("hit_score", 0), reverse=True)

    for i, r in enumerate(results):
        render_batter_row(i+1, r, r)

    return results


# ── Main App ───────────────────────────────────────────────────────────────────

def main():
    # Header
    col_logo, col_date, col_stats = st.columns([3, 2, 2])
    with col_logo:
        st.markdown('<div class="ah-logo">ALGO<span>HUB</span></div><div class="ah-tagline">HR Intelligence Platform</div>', unsafe_allow_html=True)
    with col_date:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.8rem;color:#475569;padding-top:12px">{datetime.today().strftime("%A, %B %d %Y")}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading schedule..."):
        games = load_schedule()

    if not games:
        st.error("No games found today.")
        return

    with st.spinner("Loading Statcast engine (30 days)..."):
        engine = load_statcast_engine()

    with st.spinner("Fetching odds..."):
        odds_lookup = load_bovada_odds()

    # ── Game selector ──────────────────────────────────────────────────────────
    st.markdown("#### TODAY'S SLATE")
    game_cols = st.columns(min(len(games), 8))
    selected_game_idx = st.session_state.get("selected_game", 0)

    for i, g in enumerate(games):
        with game_cols[i % len(game_cols)]:
            park = g.get("home_team","")
            pf   = PARK_FACTORS.get(park, 1.0)
            is_dome = park in DOME_PARKS
            active = "active" if i == selected_game_idx else ""
            if st.button(f"{g['away_team']} @ {g['home_team']}\n{g['game_time']}", key=f"game_{i}"):
                st.session_state["selected_game"] = i
                selected_game_idx = i

    st.divider()

    # ── Selected game ──────────────────────────────────────────────────────────
    g = games[selected_game_idx]
    home_team = g.get("home_team","")
    away_team = g.get("away_team","")
    venue     = g.get("venue","")
    pf        = PARK_FACTORS.get(home_team, 1.0)
    is_dome   = home_team in DOME_PARKS

    with st.spinner("Loading weather..."):
        weather = {} if is_dome else load_weather(venue)

    # Game header
    wind_str = "🏟️ DOME" if is_dome else f"🌡️ {weather.get('temp_f',70):.0f}°F  💨 {weather.get('wind_mph',0):.0f}mph"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding:10px 0;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:.1em">{away_team} @ {home_team}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.8rem;color:#475569">{venue}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.8rem;color:#06b6d4">{wind_str}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.8rem;color:#f59e0b">PF {pf:.2f}</span>
        <span style="font-family:'DM Mono',monospace;font-size:.8rem;color:#475569">{g.get('game_time','')}</span>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💣 HR Board", "⚡ Zone Maps", "🎳 K Props", "🎰 Parlay Builder"])

    # Load rosters
    with st.spinner("Loading rosters..."):
        home_batters = load_roster(g.get("home_team_id", 0))
        away_batters = load_roster(g.get("away_team_id", 0))

    # Declare pitcher variables before tabs
    away_pitcher_name = g.get("away_pitcher", "TBD")
    away_pitcher_id   = g.get("away_pitcher_id")
    away_pitcher_hand = load_player_hand(away_pitcher_id).get("pitch_hand","R") if away_pitcher_id else "R"
    home_pitcher_name = g.get("home_pitcher", "TBD")
    home_pitcher_id   = g.get("home_pitcher_id")
    home_pitcher_hand = load_player_hand(home_pitcher_id).get("pitch_hand","R") if home_pitcher_id else "R"

    # ── TAB 1: HR Board ────────────────────────────────────────────────────────
    with tab1:
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown(f'<div class="section-header">{away_pitcher_name} → {home_team} BATTERS</div>',
                       unsafe_allow_html=True)
            home_results = render_lineup_section(
                f"{away_pitcher_name} vs {home_team}",
                away_pitcher_name, away_pitcher_id, away_pitcher_hand,
                home_batters, home_team, engine, odds_lookup, pf, weather
            )

        with right_col:
            st.markdown(f'<div class="section-header">{home_pitcher_name} → {away_team} BATTERS</div>',
                       unsafe_allow_html=True)
            away_results = render_lineup_section(
                f"{home_pitcher_name} vs {away_team}",
                home_pitcher_name, home_pitcher_id, home_pitcher_hand,
                away_batters, home_team, engine, odds_lookup, pf, weather
            )

    # ── TAB 2: Zone Maps ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### ⚡ Zone Analysis")
        all_batters = home_batters + away_batters
        if all_batters:
            sel_name = st.selectbox("Select batter", [p["player_name"] for p in all_batters])
            sel_player = next((p for p in all_batters if p["player_name"] == sel_name), None)
            if sel_player and engine:
                bid = sel_player["player_id"]
                is_home = sel_player in home_batters
                pitcher_id   = away_pitcher_id if is_home else home_pitcher_id
                pitcher_name = g.get("away_pitcher","TBD") if is_home else g.get("home_pitcher","TBD")

                hit_data = engine.compute_hit_score(bid, pitcher_id or 0, park_factor=pf)
                zone_data = engine.compute_zone_fit(bid, pitcher_id or 0)

                # Stats row
                m1,m2,m3,m4,m5,m6 = st.columns(6)
                m1.metric("HIT Score", f"{hit_data['hit_score']:.0f}/100")
                m2.metric("Grade", hit_data["grade"])
                m3.metric("Zone Count", f"⚡{zone_data['zone_count']}")
                m4.metric("Zone Fit", f"{zone_data['zone_fit']*100:.0f}%")
                m5.metric("Barrel%", f"{hit_data['barrel_rate']:.1f}%")
                m6.metric("Proj HR%", f"{hit_data['proj_hr_pct']:.1f}%")

                # Zone heatmaps
                z1,z2,z3 = st.columns(3)
                bz = engine._batter_index.get(bid, {}).get("zone_hrs", {})
                pz = engine._pitcher_index.get(pitcher_id or 0, {}).get("zone_hrs_allowed", {}) if pitcher_id else {}

                with z1:
                    st.plotly_chart(build_zone_fig(bz, f"{sel_name}\nHR ZONES", "Blues"), use_container_width=True)
                with z2:
                    st.plotly_chart(build_overlap_fig(bz, pz), use_container_width=True)
                with z3:
                    st.plotly_chart(build_zone_fig(pz, f"{pitcher_name}\nHR ZONES ALLOWED", "Reds"), use_container_width=True)

                # Score breakdown
                st.markdown("#### Score Breakdown")
                breakdown = pd.DataFrame({
                    "Component": ["Barrel","Hard Hit","xwOBA","LA%","Exit Velo","Pull","Pitcher","Platoon","Env","Hot Bat"],
                    "Score": [
                        hit_data.get("barrel_score", 0),
                        hit_data.get("hh_score", 0),
                        hit_data.get("xwoba_score", 0),
                        hit_data.get("la_score", 0),
                        hit_data.get("ev_score", 0),
                        hit_data.get("pull_score", 0),
                        hit_data.get("pitcher_score", 0),
                        hit_data.get("platoon_score", 0),
                        hit_data.get("env_score", 0),
                        hit_data.get("form_score", 0),
                    ],
                    "Max": [18, 12, 12, 10, 8, 5, 20, 10, 10, 5],
                })
                breakdown["Pct"] = breakdown["Score"] / breakdown["Max"]
                fig = px.bar(breakdown, x="Component", y="Score", color="Pct",
                             color_continuous_scale=["#7f1d1d","#f59e0b","#22c55e"],
                             range_color=[0,1], text="Score")
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig.update_layout(
                    height=220, plot_bgcolor="#0c1018", paper_bgcolor="#0c1018",
                    showlegend=False, coloraxis_showscale=False,
                    margin=dict(l=0,r=0,t=10,b=0),
                    xaxis=dict(tickfont=dict(size=10,color="#94a3b8",family="DM Mono")),
                    yaxis=dict(showgrid=False,tickfont=dict(size=9,color="#94a3b8")),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Pitch mix detail
                if engine and pitcher_id:
                    pitcher_data = engine.get_pitcher(pitcher_id)
                    pitch_mix = pitcher_data.get("pitch_mix", {})
                    if pitch_mix:
                        st.markdown("#### Pitcher Pitch Mix")
                        mix_df = pd.DataFrame([
                            {"Pitch": pt, "Usage%": f"{v['pct']*100:.0f}%",
                             "Avg Velo": f"{v['avg_vel']:.1f}",
                             "HR Rate": f"{v['hr_rate']*100:.2f}%"}
                            for pt, v in sorted(pitch_mix.items(), key=lambda x: x[1]["pct"], reverse=True)
                        ])
                        st.dataframe(mix_df, use_container_width=True, hide_index=True)

    # ── TAB 3: K Props ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### 🎳 Strikeout Prop Analysis")

        k1, k2 = st.columns(2)

        for col, pitcher_name, pitcher_id, pitcher_hand, opp_batters, opp_team in [
            (k1, g.get("away_pitcher","TBD"), g.get("away_pitcher_id"), "R", home_batters, home_team),
            (k2, g.get("home_pitcher","TBD"), g.get("home_pitcher_id"), "R", away_batters, away_team),
        ]:
            with col:
                if pitcher_id:
                    ph = load_player_hand(pitcher_id).get("pitch_hand","R")
                else:
                    ph = "R"

                st.markdown(f"""
                <div class="pitcher-card">
                    <span class="pitcher-name">{pitcher_name}</span>
                    <span style="font-size:.75rem;color:#475569;margin-left:8px">{'RHP' if ph=='R' else 'LHP'} vs {opp_team}</span>
                </div>
                """, unsafe_allow_html=True)

                if engine and pitcher_id:
                    pitcher_data = engine.get_pitcher(pitcher_id)
                    swstr = pitcher_data.get("swstr_rate", 0)
                    vel   = pitcher_data.get("primary_vel", 0)

                    sc1,sc2,sc3 = st.columns(3)
                    sc1.metric("SwStr%", f"{swstr*100:.1f}%")
                    sc2.metric("Primary Velo", f"{vel:.1f}")
                    sc3.metric("Primary Pitch", pitcher_data.get("primary_pitch","—") or "—")

                    # K scores per batter
                    k_results = []
                    for batter in opp_batters:
                        bid = batter.get("player_id")
                        if not bid: continue
                        hand = load_player_hand(bid)
                        k_data = engine.get_k_score(pitcher_id, bid, hand.get("bat_side","R"))
                        k_results.append({
                            "name":    batter.get("player_name",""),
                            "k_score": k_data.get("k_score",0),
                            "grade":   k_data.get("grade","MODERATE"),
                            "swstr":   k_data.get("swstr_pct",0),
                        })

                    k_results.sort(key=lambda x: x["k_score"], reverse=True)

                    for kr in k_results[:8]:
                        grade_color = {"ELITE":"#fca5a5","STRONG":"#fcd34d","MODERATE":"#7dd3fc","FADE":"#475569"}.get(kr["grade"],"#475569")
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;padding:5px 8px;background:#0c1018;border-radius:5px;margin-bottom:3px;">
                            <span style="font-size:.84rem;font-weight:500">{kr['name']}</span>
                            <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:{grade_color}">{kr['k_score']:.0f} K-Score</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Engine needed for K analysis")

    # ── TAB 4: Parlay Builder ──────────────────────────────────────────────────
    with tab4:
        st.markdown("#### 🎰 Parlay Builder")
        st.caption("Build across different games for maximum value")

        all_player_names = [p["player_name"] for p in home_batters + away_batters]
        selected = st.multiselect("Add legs", all_player_names, max_selections=4)

        if selected:
            legs = []
            for name in selected:
                player = next((p for p in home_batters + away_batters if p["player_name"] == name), None)
                if not player: continue
                pid  = player["player_id"]
                hand = load_player_hand(pid)
                is_home = player in home_batters
                opp_pid  = away_pitcher_id if is_home else home_pitcher_id
                opp_hand = away_pitcher_hand if is_home else home_pitcher_hand

                if engine and opp_pid:
                    hit_data = engine.compute_hit_score(
                        pid, opp_pid, hand.get("bat_side","R"), opp_hand,
                        park_factor=pf, wind_boost=0 if is_dome else weather.get("wind_mph",0)*0.1,
                        temp_f=weather.get("temp_f",70),
                        hr_odds=odds_lookup.get(name.lower())
                    )
                else:
                    hit_data = {"hit_score":0,"grade":"MODERATE","proj_hr_pct":5,"barrel_rate":0,"avg_ev":0}

                odds = odds_lookup.get(name.lower())
                legs.append({**hit_data, "player_name": name, "odds": odds})

            for leg in legs:
                odds = leg.get("odds")
                odds_str = f"+{odds}" if odds and odds > 0 else (str(odds) if odds else "N/A")
                st.markdown(f"""
                <div class="parlay-leg">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="font-family:'Bebas Neue',sans-serif;font-size:1.1rem">{leg['player_name']}</span>
                            <span style="font-size:.72rem;color:#475569;margin-left:8px">{leg.get('grade','—')} · HIT {leg.get('hit_score',0):.0f}</span>
                        </div>
                        <span class="odds-display">{odds_str}</span>
                    </div>
                    <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:#475569;margin-top:4px">
                        BBL {leg.get('barrel_rate',0):.1f}% · EV {leg.get('avg_ev',0):.1f} · ⚡{leg.get('zone_count',0)} zones · Proj {leg.get('proj_hr_pct',0):.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Parlay odds
            if len(legs) >= 2:
                combined = 1.0
                for leg in legs:
                    p = leg.get("proj_hr_pct", 5) / 100
                    combined *= max(p, 0.02)
                impl = (1/combined)-1
                amer = int(impl*100) if impl>=1 else int(-100/impl)
                odds_disp = f"+{amer:,}" if amer>0 else str(amer)

                st.markdown(f"""
                <div style="background:#052e16;border:1px solid #166534;border-radius:10px;padding:20px;margin-top:12px;text-align:center;">
                    <div style="font-family:'DM Mono',monospace;font-size:.72rem;color:#86efac;letter-spacing:.15em">ESTIMATED PARLAY ODDS</div>
                    <div class="parlay-odds">{odds_disp}</div>
                    <div style="font-size:.72rem;color:#4a5568;margin-top:4px">Combined: {combined*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Twitter
                legs_text = "\n".join([f"{l['player_name']} {'+'+str(l['odds']) if l.get('odds') and l['odds']>0 else ''} ✅" for l in legs])
                tweet = f"💣 HR PARLAY\n\n{legs_text}\n\n{odds_disp} 🔒 AlgoHub locked in\n\n@TheAlgoHub | #MLBProps #HomeRun"
                st.code(tweet, language=None)

    # Footer
    st.divider()
    st.markdown('<div style="text-align:center;font-family:DM Mono,monospace;font-size:.7rem;color:#1c2333">ALGOHUB · @TheAlgoHub · Powered by Baseball Savant Statcast</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
