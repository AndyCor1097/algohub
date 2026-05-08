"""
bovada_odds.py — Free odds scraper using Bovada's public API
No API key required. Pulls HR prop odds and game totals.
"""

import requests
import pandas as pd
from datetime import datetime
import time

BOVADA_BASE = "https://www.bovada.lv/services/sports/event/v2/events/A/description"
BOVADA_MLB  = f"{BOVADA_BASE}/baseball/mlb"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.bovada.lv/",
}

TEAM_MAP = {
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "New York Yankees": "NYY", "Tampa Bay Rays": "TBR",
    "Toronto Blue Jays": "TOR", "Chicago White Sox": "CWS",
    "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
    "Kansas City Royals": "KCR", "Minnesota Twins": "MIN",
    "Houston Astros": "HOU", "Los Angeles Angels": "LAA",
    "Oakland Athletics": "ATH", "Seattle Mariners": "SEA",
    "Texas Rangers": "TEX", "Atlanta Braves": "ATL",
    "Miami Marlins": "MIA", "New York Mets": "NYM",
    "Philadelphia Phillies": "PHI", "Washington Nationals": "WSH",
    "Chicago Cubs": "CHC", "Cincinnati Reds": "CIN",
    "Milwaukee Brewers": "MIL", "Pittsburgh Pirates": "PIT",
    "St. Louis Cardinals": "STL", "Arizona Diamondbacks": "ARI",
    "Colorado Rockies": "COL", "Los Angeles Dodgers": "LAD",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
}


def get_mlb_game_totals() -> pd.DataFrame:
    """Pull MLB game totals from Bovada. Returns one row per game."""
    try:
        resp = requests.get(BOVADA_MLB, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            print(f"  Bovada returned {resp.status_code}")
            return pd.DataFrame()

        events = resp.json()
        rows = []

        for event in events:
            if not isinstance(event, dict):
                continue

            teams = event.get("competitors", [])
            if len(teams) < 2:
                continue

            home = next((t["name"] for t in teams if t.get("home")), "")
            away = next((t["name"] for t in teams if not t.get("home")), "")

            home_abbr = TEAM_MAP.get(home, home[:3].upper())
            away_abbr = TEAM_MAP.get(away, away[:3].upper())

            total = None
            home_ml = None
            away_ml = None
            home_implied = None
            away_implied = None

            for market in event.get("displayGroups", []):
                for group in market.get("markets", []):
                    desc = group.get("description", "")

                    # Game total
                    if "Total" in desc and total is None:
                        for outcome in group.get("outcomes", []):
                            if outcome.get("type") == "O":
                                try:
                                    total = float(outcome.get("price", {}).get("handicap", 0))
                                except:
                                    pass

                    # Moneyline
                    if desc == "Moneyline":
                        for outcome in group.get("outcomes", []):
                            price = outcome.get("price", {}).get("american", "")
                            name  = outcome.get("description", "")
                            try:
                                ml = int(str(price).replace("+", ""))
                                if home in name:
                                    home_ml = ml
                                elif away in name:
                                    away_ml = ml
                            except:
                                pass

            # Implied totals from moneyline
            if home_ml is not None and away_ml is not None and total is not None:
                def ml_to_prob(ml):
                    if ml > 0:
                        return 100 / (ml + 100)
                    else:
                        return abs(ml) / (abs(ml) + 100)

                home_prob = ml_to_prob(home_ml)
                away_prob = ml_to_prob(away_ml)
                norm = home_prob + away_prob
                home_prob /= norm
                away_prob /= norm
                home_implied = round(total * home_prob, 2)
                away_implied = round(total * away_prob, 2)

            rows.append({
                "home_team":      home_abbr,
                "away_team":      away_abbr,
                "total_line":     total,
                "home_ml":        home_ml,
                "away_ml":        away_ml,
                "home_implied":   home_implied,
                "away_implied":   away_implied,
            })

        df = pd.DataFrame(rows)
        print(f"  Bovada: {len(df)} games found")
        return df

    except Exception as e:
        print(f"  Bovada pull failed: {e}")
        return pd.DataFrame()


def get_hr_prop_odds(player_name: str) -> dict:
    """
    Try to pull HR prop odds for a specific player from Bovada props.
    Returns dict with best odds or empty dict.
    """
    try:
        props_url = f"{BOVADA_BASE}/baseball/mlb/player-props"
        resp = requests.get(props_url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}

        events = resp.json()
        name_lower = player_name.lower()

        for event in events:
            for group in event.get("displayGroups", []):
                for market in group.get("markets", []):
                    if "home run" in market.get("description", "").lower():
                        for outcome in market.get("outcomes", []):
                            if name_lower in outcome.get("description", "").lower():
                                price = outcome.get("price", {}).get("american", "")
                                try:
                                    return {"odds": int(str(price).replace("+", ""))}
                                except:
                                    pass
        return {}

    except Exception as e:
        return {}


def get_mlb_hr_props() -> pd.DataFrame:
    """Pull all available MLB HR props from Bovada."""
    try:
        props_url = f"{BOVADA_BASE}/baseball/mlb/player-props"
        resp = requests.get(props_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        events = resp.json()
        rows = []

        for event in events:
            teams = event.get("competitors", [])
            home = next((TEAM_MAP.get(t["name"], t["name"][:3].upper())
                        for t in teams if t.get("home")), "")
            away = next((TEAM_MAP.get(t["name"], t["name"][:3].upper())
                        for t in teams if not t.get("home")), "")

            for group in event.get("displayGroups", []):
                for market in group.get("markets", []):
                    if "home run" not in market.get("description", "").lower():
                        continue
                    for outcome in market.get("outcomes", []):
                        player = outcome.get("description", "")
                        price  = outcome.get("price", {}).get("american", "")
                        try:
                            odds = int(str(price).replace("+", ""))
                            rows.append({
                                "player_name": player,
                                "home_team":   home,
                                "away_team":   away,
                                "hr_odds":     odds,
                            })
                        except:
                            pass

        df = pd.DataFrame(rows)
        print(f"  Bovada HR props: {len(df)} players found")
        return df

    except Exception as e:
        print(f"  Bovada HR props failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing Bovada odds...")
    totals = get_mlb_game_totals()
    print(totals)
    props = get_mlb_hr_props()
    print(props.head(10))
