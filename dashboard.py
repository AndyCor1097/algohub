"""
dashboard.py — AlgoHub HR Matchup Dashboard v2
Full system: zone heatmaps, hot bat tracking, Bovada odds, parlay builder.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
import sys
from datetime import datetime

st.set_page_config(
    page_title="AlgoHub · HR Intel",
    page_icon="💣",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --bg:#080c14; --surface:#0d1321; --border:#1a2540;
    --accent:#00e5ff; --green:#00ff88; --amber:#fbbf24;
    --red:#ef4444; --muted:#4a5568; --text:#e2e8f0;
}
html,body,[class*="css"]{background:var(--bg)!important;color:var(--text);font-family:'Inter',sans-serif;}
.stApp{background:var(--bg);}
.algo-header{display:flex;align-items:baseline;gap:12px;padding:8px 0 4px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.algo-title{font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;letter-spacing:.12em;background:linear-gradient(90deg,#00e5ff,#00ff88);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.algo-sub{font-size:.75rem;color:var(--muted);letter-spacing:.2em;text-transform:uppercase;font-family:'JetBrains Mono',monospace;}
.pick-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;height:100%;position:relative;overflow:hidden;}
.pick-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.pick-card.hot::before{background:var(--red);}
.pick-card.warm::before{background:var(--amber);}
.pick-card.cold::before{background:var(--muted);}
.pick-name{font-family:'Rajdhani',sans-serif;font-size:1.05rem;font-weight:700;color:var(--text);}
.pick-vs{font-size:.72rem;color:var(--muted);margin:2px 0 8px;}
.pick-prob{font-size:1.6rem;font-weight:700;color:var(--green);font-family:'JetBrains Mono',monospace;}
.pick-stats{font-size:.7rem;color:#94a3b8;margin-top:4px;}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.7rem;font-weight:700;letter-spacing:.05em;font-family:'JetBrains Mono',monospace;}
.badge-TARGET{background:#450a0a;color:#fca5a5;border:1px solid #7f1d1d;}
.badge-STRONG{background:#451a03;color:#fcd34d;border:1px solid #92400e;}
.badge-MODERATE{background:#052e16;color:#86efac;border:1px solid #166534;}
.badge-TOUGH{background:#0f172a;color:#64748b;border:1px solid #1e293b;}
</style>
""", unsafe_allow_html=True)


def load_predictions():
    files = sorted(glob.glob("predictions/hr_predictions_*.csv"), reverse=True)
    if not files:
        return None, None
    df = pd.read_csv(files[0])
    date_str = files[0].split("hr_predictions_")[1].replace(".csv", "")
    return df, date_str


def load_bovada():
    try:
        from bovada_odds import get_mlb_hr_props
        return get_mlb_hr_props()
    except:
        return pd.DataFrame()


def derive_grade(prob):
    if prob >= 0.18:   return "TARGET"
    elif prob >= 0.12: return "STRONG"
    elif prob >= 0.07: return "MODERATE"
    else:              return "TOUGH"


def hot_bat_label(l7):
    if l7 >= 0.25:   return "🔥 HOT BAT", "hot"
    elif l7 >= 0.10: return "🌡️ WARM", "warm"
    else:            return "❄️ COLD", "cold"


def build_zone_heatmap(zone_data, title, colorscale="Reds"):
    grid = np.zeros((3, 3))
    zone_map = {1:(0,0),2:(0,1),3:(0,2),4:(1,0),5:(1,1),6:(1,2),7:(2,0),8:(2,1),9:(2,2)}
    for z,(r,c) in zone_map.items():
        grid[r][c] = zone_data.get(z, 0)
    text = [[str(int(grid[r][c])) if grid[r][c]>0 else "" for c in range(3)] for r in range(3)]
    fig = go.Figure(go.Heatmap(
        z=grid, text=text, texttemplate="%{text}",
        textfont={"size":20,"color":"white","family":"JetBrains Mono"},
        colorscale=colorscale, showscale=False,
        zmin=0, zmax=max(1, np.max(grid)),
    ))
    fig.add_shape(type="rect",x0=-0.5,y0=-0.5,x1=2.5,y1=2.5,line=dict(color="#60a5fa",width=2))
    for i in [0.5,1.5]:
        fig.add_shape(type="line",x0=i,y0=-0.5,x1=i,y1=2.5,line=dict(color="#1e2a3a",width=1))
        fig.add_shape(type="line",x0=-0.5,y0=i,x1=2.5,y1=i,line=dict(color="#1e2a3a",width=1))
    fig.update_layout(
        title=dict(text=title,font=dict(family="Rajdhani",size=13,color="#94a3b8"),x=0.5),
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        margin=dict(l=5,r=5,t=30,b=5), height=200,
        plot_bgcolor="#0d1321", paper_bgcolor="#0d1321",
    )
    return fig


def build_overlap_heatmap(batter_zones, pitcher_zones):
    zone_map = {1:(0,0),2:(0,1),3:(0,2),4:(1,0),5:(1,1),6:(1,2),7:(2,0),8:(2,1),9:(2,2)}
    grid = np.zeros((3,3))
    hover = [[""]*3 for _ in range(3)]
    text  = [[""]*3 for _ in range(3)]
    for z,(r,c) in zone_map.items():
        b = batter_zones.get(z,0)
        p = pitcher_zones.get(z,0)
        if b>0 and p>0:
            grid[r][c]=3; text[r][c]="⚡"; hover[r][c]=f"KILL ZONE\nBatter:{b}HR Pitcher:{p}HR"
        elif p>0:
            grid[r][c]=2; text[r][c]="●"; hover[r][c]=f"Pitcher weak:{p}HR"
        elif b>0:
            grid[r][c]=1; text[r][c]="○"; hover[r][c]=f"Batter strength:{b}HR"
    colorscale=[[0.0,"#0d1321"],[0.33,"#1e3a5f"],[0.66,"#7f1d1d"],[1.0,"#dc2626"]]
    fig = go.Figure(go.Heatmap(
        z=grid, text=text, customdata=hover,
        hovertemplate="%{customdata}<extra></extra>",
        texttemplate="%{text}", textfont={"size":16,"color":"white"},
        colorscale=colorscale, showscale=False, zmin=0, zmax=3,
    ))
    fig.add_shape(type="rect",x0=-0.5,y0=-0.5,x1=2.5,y1=2.5,line=dict(color="#60a5fa",width=2))
    for i in [0.5,1.5]:
        fig.add_shape(type="line",x0=i,y0=-0.5,x1=i,y1=2.5,line=dict(color="#1e2a3a",width=1))
        fig.add_shape(type="line",x0=-0.5,y0=i,x1=2.5,y1=i,line=dict(color="#1e2a3a",width=1))
    fig.update_layout(
        title=dict(text="⚡ Kill Zones",font=dict(family="Rajdhani",size=13,color="#94a3b8"),x=0.5),
        xaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
        margin=dict(l=5,r=5,t=30,b=5), height=200,
        plot_bgcolor="#0d1321", paper_bgcolor="#0d1321",
    )
    return fig


def main():
    st.markdown('<div class="algo-header"><span class="algo-title">💣 ALGOHUB</span><span class="algo-sub">HR Intelligence · @TheAlgoHub</span></div>', unsafe_allow_html=True)

    df, date_str = load_predictions()
    if df is None:
        st.error("No predictions found. Run `python daily_predictions.py` first.")
        return

    if "grade" not in df.columns and "hr_probability" in df.columns:
        df["grade"] = df["hr_probability"].apply(derive_grade)

    with st.spinner("Fetching Bovada odds..."):
        bovada = load_bovada()

    if not bovada.empty and "player_name" in df.columns:
        lookup = dict(zip(bovada["player_name"].str.lower(), bovada["hr_odds"]))
        df["bovada_odds"] = df["player_name"].str.lower().map(lookup)
    else:
        df["bovada_odds"] = None

    hot_count = int((df.get("hr_rate_last7", pd.Series(dtype=float)) >= 0.25).sum()) if "hr_rate_last7" in df.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📅 Date", date_str)
    c2.metric("⚾ Players", len(df))
    c3.metric("🔥 Hot Bats", hot_count)
    c4.metric("📡 Bovada Props", f"{len(bovada)}" if not bovada.empty else "Offline")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Board", "🔥 Hot Bats", "⚡ Zone Maps", "🎰 Parlay Builder"])

    # ── TAB 1: Board ──────────────────────────────────────────────────────────
    with tab1:
        with st.sidebar:
            st.markdown("### 🎯 Filters")
            grade_filter = st.multiselect("Grade", ["TARGET","STRONG","MODERATE","TOUGH"], default=["TARGET","STRONG","MODERATE"])
            min_barrel = st.slider("Min Barrel% L15", 0.0, 0.40, 0.0, 0.01) if "barrel_rate_last15" in df.columns else 0.0
            min_ev = st.slider("Min EV L15", 80.0, 100.0, 85.0, 0.5) if "avg_ev_last15" in df.columns else 85.0
            min_park = st.slider("Min Park Factor", 0.85, 1.25, 0.90, 0.01) if "park_factor" in df.columns else 0.90
            wind_out = st.checkbox("Wind out only")
            platoon_only = st.checkbox("Platoon advantage")
            hot_only = st.checkbox("Hot bats only")

        filt = df[df["grade"].isin(grade_filter)].copy()
        if "barrel_rate_last15" in filt.columns: filt = filt[filt["barrel_rate_last15"] >= min_barrel]
        if "avg_ev_last15" in filt.columns: filt = filt[filt["avg_ev_last15"] >= min_ev]
        if "park_factor" in filt.columns: filt = filt[filt["park_factor"] >= min_park]
        if wind_out and "wind_component_mph" in filt.columns: filt = filt[filt["wind_component_mph"] > 0]
        if platoon_only and "platoon_advantage" in filt.columns: filt = filt[filt["platoon_advantage"] == 1]
        if hot_only and "hr_rate_last7" in filt.columns: filt = filt[filt["hr_rate_last7"] >= 0.25]
        filt = filt.sort_values("hr_probability", ascending=False)

        st.markdown("#### 🔝 Top Picks")
        top_cols = st.columns(5)
        for i, (_, row) in enumerate(filt.head(5).iterrows()):
            with top_cols[i]:
                prob = row.get("hr_probability", 0)
                l7 = row.get("hr_rate_last7", 0)
                barrel = row.get("barrel_rate_last15", 0)
                ev = row.get("avg_ev_last15", 0)
                grade = row.get("grade", "MODERATE")
                bat_label, bat_class = hot_bat_label(l7)
                odds = row.get("bovada_odds", None)
                odds_str = f"+{int(odds)}" if odds and not pd.isna(odds) and odds > 0 else "N/A"
                st.markdown(f"""<div class="pick-card {bat_class}">
                    <div class="pick-name">{row.get('player_name','—')}</div>
                    <div class="pick-vs">vs {row.get('opposing_pitcher','—')} · {row.get('batting_team','')}</div>
                    <div class="pick-prob">{prob*100:.1f}%</div>
                    <div class="pick-stats">EV {ev:.1f} · BBL {barrel*100:.1f}%<br>{bat_label} · <span style="color:#fbbf24">{odds_str}</span></div>
                    <div style="margin-top:6px"><span class="badge badge-{grade}">{grade}</span></div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### Full Board — {len(filt)} players")

        cols_map = {"player_name":"Player","batting_team":"Team","opposing_pitcher":"Pitcher",
                    "grade":"Grade","hr_probability":"HR%","bovada_odds":"Odds",
                    "hr_rate_last7":"HR L7","hr_rate_last15":"HR L15",
                    "barrel_rate_last15":"BBL%","avg_ev_last15":"EV",
                    "park_factor":"PF","wind_label":"Wind","temp_f":"°F",
                    "platoon_advantage":"Platoon","h2h_hr":"H2H HR"}
        avail = {k:v for k,v in cols_map.items() if k in filt.columns}
        tdf = filt[list(avail.keys())].copy()
        tdf.columns = list(avail.values())
        for col, fn in [("HR%", lambda x: f"{x*100:.1f}%"),("HR L7", lambda x: f"{x*100:.1f}%"),
                        ("HR L15", lambda x: f"{x*100:.1f}%"),("BBL%", lambda x: f"{x*100:.1f}%"),
                        ("EV", lambda x: f"{x:.1f}"),("PF", lambda x: f"{x:.2f}"),
                        ("Platoon", lambda x: "✅" if x==1 else ""),
                        ("H2H HR", lambda x: str(int(x)) if not pd.isna(x) else ""),
                        ("Odds", lambda x: f"+{int(x)}" if not pd.isna(x) and x>0 else ("" if pd.isna(x) else str(int(x))))]:
            if col in tdf.columns:
                tdf[col] = tdf[col].apply(fn)
        st.dataframe(tdf, use_container_width=True, hide_index=True, height=500)

    # ── TAB 2: Hot Bats ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### 🔥 Hot Bat Leaderboard")
        st.caption("Ranked by recent HR form — L7/L15/L30")
        if "hr_rate_last7" not in df.columns:
            st.warning("HR rate data not available")
        else:
            hot_df = df.copy()
            hot_df["hot_score"] = (hot_df.get("hr_rate_last7",0)*0.5 +
                                   hot_df.get("hr_rate_last15",0)*0.3 +
                                   hot_df.get("barrel_rate_last15",0)*0.2)
            hot_df = hot_df.sort_values("hot_score", ascending=False).head(30)
            for _, row in hot_df.iterrows():
                l7 = row.get("hr_rate_last7",0); l15 = row.get("hr_rate_last15",0)
                l30 = row.get("hr_rate_last30",0); bbl = row.get("barrel_rate_last15",0)
                ev = row.get("avg_ev_last15",0)
                label, _ = hot_bat_label(l7)
                bar_pct = min(l7/0.5, 1.0)
                bar_color = "#ef4444" if l7>=0.25 else "#fbbf24" if l7>=0.10 else "#4a5568"
                st.markdown(f"""<div style="background:#0d1321;border:1px solid #1a2540;border-radius:8px;padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;gap:16px;">
                    <div style="min-width:180px"><span style="font-family:Rajdhani;font-weight:700;font-size:1rem">{row.get('player_name','—')}</span>
                    <span style="color:#4a5568;font-size:.75rem;margin-left:8px">{row.get('batting_team','')} vs {row.get('opposing_pitcher','')}</span></div>
                    <div style="flex:1;background:#0a0e1a;border-radius:4px;height:8px;overflow:hidden">
                    <div style="width:{bar_pct*100:.0f}%;height:100%;background:{bar_color};border-radius:4px;"></div></div>
                    <div style="min-width:260px;font-family:JetBrains Mono;font-size:.75rem;color:#94a3b8;">
                    L7:<span style="color:{bar_color}"> {l7*100:.0f}%</span> L15:{l15*100:.0f}% L30:{l30*100:.0f}% BBL:{bbl*100:.0f}% EV:{ev:.1f}</div>
                    <div>{label}</div></div>""", unsafe_allow_html=True)

    # ── TAB 3: Zone Maps ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### ⚡ Zone Heatmaps")
        st.caption("Batter HR zones vs pitcher weakness zones — overlap = kill zone")
        if "player_name" not in df.columns:
            st.warning("Player data not available")
        else:
            sel = st.selectbox("Select player", df.sort_values("hr_probability",ascending=False)["player_name"].tolist())
            row = df[df["player_name"]==sel].iloc[0]
            try:
                from zone_engine import ZoneEngine
                import pybaseball as pb
                from datetime import timedelta
                with st.spinner("Loading zone data..."):
                    end = datetime.today()
                    start = end - timedelta(days=30)
                    raw = pb.statcast(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
                    engine = ZoneEngine(raw)

                bid = int(row["player_id"]) if "player_id" in row else None
                pid = int(row["pitcher_id"]) if "pitcher_id" in row else None

                if bid and pid:
                    bz_df = engine._batter_zone_hrs
                    pz_df = engine._pitcher_zone_hrs
                    batter_zones  = dict(zip(bz_df[bz_df["batter"]==bid]["zone"], bz_df[bz_df["batter"]==bid]["hr_count"])) if not bz_df.empty else {}
                    pitcher_zones = dict(zip(pz_df[pz_df["pitcher"]==pid]["zone"], pz_df[pz_df["pitcher"]==pid]["hr_count"])) if not pz_df.empty else {}
                    zone_fit = engine.compute_zone_fit(bid, pid)
                    khr = engine.compute_khr(bid, pid)
                    matchup = engine.compute_matchup_score(bid, pid,
                        park_factor=float(row.get("park_factor",1.0)),
                        wind_boost=float(row.get("wind_component_mph",0)),
                        temp_f=float(row.get("temp_f",70)),
                        batter_iso=float(row.get("season_barrel_rate",0.10)))

                    m1,m2,m3,m4,m5 = st.columns(5)
                    m1.metric("Score", f"{matchup['composite_score']:.0f}/100")
                    m2.metric("Grade", matchup["grade"])
                    m3.metric("Zone Fit", f"{zone_fit*100:.0f}%")
                    m4.metric("kHR", f"{khr*100:.2f}")
                    m5.metric("Proj HR%", f"{matchup['proj_hr_pct']:.1f}%")

                    z1,z2,z3 = st.columns(3)
                    with z1: st.plotly_chart(build_zone_heatmap(batter_zones, f"{sel} HR Zones", "Blues"), use_container_width=True)
                    with z2: st.plotly_chart(build_overlap_heatmap(batter_zones, pitcher_zones), use_container_width=True)
                    with z3: st.plotly_chart(build_zone_heatmap(pitcher_zones, f"{row.get('opposing_pitcher','Pitcher')} HR Zones Allowed", "Reds"), use_container_width=True)

                    st.markdown("#### Score Breakdown")
                    score_df = pd.DataFrame({
                        "Factor": ["Zone Fit","ISO/Power","Pitcher ERA","Barrel Rate","Park+Weather","HR Form","Platoon","Exit Velo"],
                        "Score":  [matchup["zone_score"],matchup["iso_score"],matchup["era_score"],matchup["barrel_score"],matchup["env_score"],matchup["form_score"],matchup["platoon_score"],matchup["ev_score"]],
                        "Max":    [25,20,20,15,10,10,5,5],
                    })
                    score_df["Pct"] = score_df["Score"] / score_df["Max"]
                    fig = px.bar(score_df, x="Factor", y="Score", color="Pct",
                                 color_continuous_scale=["#7f1d1d","#fbbf24","#14532d"], range_color=[0,1])
                    fig.update_layout(height=220, plot_bgcolor="#0d1321", paper_bgcolor="#0d1321",
                                      showlegend=False, coloraxis_showscale=False,
                                      margin=dict(l=0,r=0,t=10,b=0),
                                      xaxis=dict(tickfont=dict(size=10,color="#94a3b8")),
                                      yaxis=dict(showgrid=False,tickfont=dict(size=9,color="#94a3b8")))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Player/pitcher IDs needed for zone analysis")
            except ImportError:
                st.info("Install dependencies: `pip install pybaseball plotly --break-system-packages`")
            except Exception as e:
                st.warning(f"Zone data unavailable: {e}")

    # ── TAB 4: Parlay Builder ─────────────────────────────────────────────────
    with tab4:
        st.markdown("#### 🎰 Parlay Builder")
        st.caption("3-leg recommended · Spread across different games")
        parlay_pool = df[df["grade"].isin(["TARGET","STRONG","MODERATE"])].sort_values("hr_probability",ascending=False)
        selected = st.multiselect("Add legs", parlay_pool["player_name"].tolist(), max_selections=4)
        if selected:
            legs = parlay_pool[parlay_pool["player_name"].isin(selected)]
            if "home_team" in legs.columns:
                dupes = legs["home_team"].value_counts()
                dupes = dupes[dupes>1].index.tolist()
                if dupes: st.error(f"⚠️ Same game conflict: {', '.join(dupes)}")
            for _, leg in legs.iterrows():
                prob = leg.get("hr_probability",0); l7 = leg.get("hr_rate_last7",0)
                barrel = leg.get("barrel_rate_last15",0); ev = leg.get("avg_ev_last15",0)
                grade = leg.get("grade","MODERATE"); odds = leg.get("bovada_odds",None)
                odds_str = f"+{int(odds)}" if odds and not pd.isna(odds) and odds>0 else "N/A"
                label, _ = hot_bat_label(l7)
                st.markdown(f"""<div style="background:#0d1321;border:1px solid #1a2540;border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div><span style="font-family:Rajdhani;font-weight:700;font-size:1.1rem">{leg.get('player_name','—')}</span>
                    <span style="color:#4a5568;font-size:.8rem;margin-left:8px">vs {leg.get('opposing_pitcher','—')} · {leg.get('wind_label','')}</span></div>
                    <div style="text-align:right"><span style="font-family:JetBrains Mono;font-size:1.2rem;color:#00ff88">{prob*100:.1f}%</span>
                    <span style="color:#fbbf24;font-family:JetBrains Mono;margin-left:12px">{odds_str}</span></div></div>
                    <div style="margin-top:6px;font-size:.75rem;color:#94a3b8;font-family:JetBrains Mono">
                    BBL {barrel*100:.1f}% · EV {ev:.1f} · {label} &nbsp;<span class="badge badge-{grade}">{grade}</span></div></div>""", unsafe_allow_html=True)

            probs = [float(p) for p in legs["hr_probability"].tolist()]
            if len(probs) >= 2:
                combined = 1.0
                for p in probs: combined *= p
                impl = (1/combined)-1
                amer = int(impl*100) if impl>=1 else int(-100/impl)
                odds_disp = f"+{amer:,}" if amer>0 else str(amer)
                st.markdown(f"""<div style="background:#052e16;border:1px solid #166534;border-radius:8px;padding:16px;margin-top:12px;text-align:center;">
                    <div style="font-family:Rajdhani;font-size:1rem;color:#86efac;letter-spacing:.1em">ESTIMATED PARLAY ODDS</div>
                    <div style="font-family:JetBrains Mono;font-size:2rem;font-weight:700;color:#00ff88">{odds_disp}</div>
                    <div style="font-size:.75rem;color:#4a5568;margin-top:4px">Combined probability: {combined*100:.2f}%</div></div>""", unsafe_allow_html=True)

                legs_text = "\n".join([f"{l.get('player_name','—')} ✅" for _,l in legs.iterrows()])
                tweet = f"🚨 HR Algo Alert 🚨\n\n{legs_text}\n\n{len(selected)}-leg parlay {odds_disp} 🔒\nThe algo locked in. Let's cash 💰\n\n@TheAlgoHub | #MLBProps #HomeRun"
                st.markdown("#### Twitter Caption")
                st.code(tweet, language=None)

    st.divider()
    st.markdown("<div style='text-align:center;color:#4a5568;font-size:.75rem;font-family:JetBrains Mono'>@TheAlgoHub · Powered by Baseball Savant Statcast · Not financial advice</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
