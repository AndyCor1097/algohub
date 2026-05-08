"""
vegas_lines.py
Pulls Vegas implied run totals and lines from The Odds API (free tier available).
Implied team total is one of the strongest signals for expected offensive output.

Sign up for a free API key at: https://the-odds-api.com
Free tier: 500 requests/month (more than enough for daily use)

Set your key as env variable: ODDS_API_KEY=your_key_here
Or paste it directly in ODDS_API_KEY below.
"""

import requests
import pandas as pd
import os
from datetime import datetime
import time

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "YOUR_KEY_HERE")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Fallback: DraftKings public lines via undocumented endpoint (no key needed)
DK_FALLBACK = True


def get_mlb_game_lines(date_str: str = None) -> pd.DataFrame:
    """
    Pull MLB moneyline, run total, and team implied totals for today's games.
    Returns DataFrame with one row per game.
    """
    if ODDS_API_KEY and ODDS_API_KEY != "YOUR_KEY_HERE":
        return _fetch_from_odds_api(date_str)
    elif DK_FALLBACK:
        print("  No Odds API key set — using DraftKings public fallback")
        return _fetch_dk_fallback()
    else:
        print("  No odds source available. Set ODDS_API_KEY env variable.")
        return pd.DataFrame()


def _fetch_from_odds_api(date_str: str = None) -> pd.DataFrame:
    """Fetch from The Odds API."""
    try:
        url = f"{ODDS_API_BASE}/sports/baseball_mlb/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "totals,h2h",
            "oddsFormat": "american",
            "bookmakers": "draftkings,fanduel,betmgm",
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        rows = []
        for game in data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            commence = game.get("commence_time", "")

            total_line = None
            home_ml = None
            away_ml = None

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals" and total_line is None:
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                total_line = outcome.get("point", 8.5)
                    if market["key"] == "h2h" and home_ml is None:
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home:
                                home_ml = outcome.get("price")
                            elif outcome["name"] == away:
                                away_ml = outcome.get("price")

            if total_line:
                home_implied, away_implied = split_implied_total(total_line, home_ml, away_ml)
                rows.append({
                    "home_team_full": home,
                    "away_team_full": away,
                    "game_time": commence,
                    "total_line": total_line,
                    "home_implied_runs": home_implied,
                    "away_implied_runs": away_implied,
                    "home_ml": home_ml,
                    "away_ml": away_ml,
                })

        df = pd.DataFrame(rows)
        print(f"  Pulled Vegas lines for {len(df)} games")
        return df

    except Exception as e:
        print(f"  Odds API error: {e}")
        return pd.DataFrame()


def _fetch_dk_fallback() -> pd.DataFrame:
    """
    DraftKings public endpoint fallback.
    Returns approximate implied totals.
    """
    try:
        url = "https://sportsbook.draftkings.com//sites/US-SB/api/v5/eventgroups/84240/categories/743/subcategories/6589?format=json"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()

        rows = []
        events = (data.get("eventGroup", {})
                      .get("offerCategories", [{}])[0]
                      .get("offerSubcategoryDescriptors", [{}])[0]
                      .get("offerSubcategory", {})
                      .get("offers", []))

        for event_offers in events:
            for offer in event_offers:
                if offer.get("label") == "Total Runs":
                    outcomes = offer.get("outcomes", [])
                    for o in outcomes:
                        if o.get("label") == "Over":
                            total = o.get("line", 8.5)
                            # Without ML, split 50/50
                            implied = total / 2
                            rows.append({
                                "total_line": total,
                                "home_implied_runs": implied,
                                "away_implied_runs": implied,
                            })
                            break

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    except Exception as e:
        print(f"  DK fallback error: {e}")
        return pd.DataFrame()


def split_implied_total(total: float, home_ml: float = None,
                         away_ml: float = None) -> tuple:
    """
    Split the game total into home/away implied run totals.
    Uses moneyline to estimate win probability, then backs into team totals.
    If no ML available, splits 50/50.
    """
    if home_ml is None or away_ml is None:
        return total / 2, total / 2

    home_prob = ml_to_prob(home_ml)
    away_prob = 1 - home_prob

    # Typical home run scoring advantage for favorites
    # If home is 60% favorite, they score ~55% of runs
    home_run_share = 0.5 + (home_prob - 0.5) * 0.4
    away_run_share = 1 - home_run_share

    return round(total * home_run_share, 2), round(total * away_run_share, 2)


def ml_to_prob(ml: float) -> float:
    """Convert American moneyline to implied probability."""
    if ml is None:
        return 0.5
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


def implied_total_hr_multiplier(implied_team_runs: float,
                                 baseline_runs: float = 4.5) -> float:
    """
    Scale HR probability by implied team total.
    High-run environments = more HRs.
    ~+3% per implied run above baseline.
    """
    diff = implied_team_runs - baseline_runs
    multiplier = 1.0 + (diff * 0.03)
    return round(max(0.75, min(multiplier, 1.40)), 4)


def normalize_team_name(full_name: str) -> str:
    """Map full team name to abbreviation."""
    mapping = {
        "Colorado Rockies": "COL", "Cincinnati Reds": "CIN",
        "Philadelphia Phillies": "PHI", "New York Yankees": "NYY",
        "Boston Red Sox": "BOS", "Texas Rangers": "TEX",
        "Milwaukee Brewers": "MIL", "Baltimore Orioles": "BAL",
        "Atlanta Braves": "ATL", "Chicago Cubs": "CHC",
        "Houston Astros": "HOU", "Toronto Blue Jays": "TOR",
        "Minnesota Twins": "MIN", "Los Angeles Angels": "LAA",
        "Cleveland Guardians": "CLE", "Detroit Tigers": "DET",
        "Washington Nationals": "WSH", "St. Louis Cardinals": "STL",
        "New York Mets": "NYM", "Arizona Diamondbacks": "ARI",
        "Tampa Bay Rays": "TBR", "Kansas City Royals": "KCR",
        "Chicago White Sox": "CWS", "Pittsburgh Pirates": "PIT",
        "Miami Marlins": "MIA", "San Francisco Giants": "SFG",
        "Los Angeles Dodgers": "LAD", "Oakland Athletics": "OAK",
        "Seattle Mariners": "SEA", "San Diego Padres": "SDP",
    }
    return mapping.get(full_name, full_name[:3].upper())


def add_vegas_features(pred_df: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """
    Fetch and join Vegas features to prediction DataFrame.
    Requires 'batting_team' and 'home_team' columns.
    """
    lines = get_mlb_game_lines(game_date)

    if lines.empty:
        pred_df["implied_team_runs"] = 4.5
        pred_df["total_line"] = 9.0
        pred_df["vegas_hr_multiplier"] = 1.0
        return pred_df

    # Normalize team names to abbreviations
    if "home_team_full" in lines.columns:
        lines["home_team"] = lines["home_team_full"].apply(normalize_team_name)
        lines["away_team"] = lines["away_team_full"].apply(normalize_team_name)

    # Build lookup: team -> implied runs
    team_implied = {}
    for _, row in lines.iterrows():
        team_implied[row.get("home_team", "")] = row.get("home_implied_runs", 4.5)
        team_implied[row.get("away_team", "")] = row.get("away_implied_runs", 4.5)

    pred_df["implied_team_runs"] = pred_df["batting_team"].map(team_implied).fillna(4.5)
    pred_df["total_line"] = pred_df["home_team"].map(
        {row["home_team"]: row["total_line"] for _, row in lines.iterrows()
         if "home_team" in row}
    ).fillna(9.0)

    pred_df["vegas_hr_multiplier"] = pred_df["implied_team_runs"].apply(
        implied_total_hr_multiplier
    )

    return pred_df
