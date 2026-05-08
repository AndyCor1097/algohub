"""
injury_rest.py
Pulls injury/IL status and rest day flags from MLB Stats API.
Fatigued or returning-from-IL players hit fewer HRs.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time


# ── IL / Injury Status ────────────────────────────────────────────────────────

def get_il_transactions(days_back: int = 30) -> pd.DataFrame:
    """
    Pull recent IL transactions from MLB Stats API.
    Returns DataFrame of players placed on or activated from IL.
    """
    end = datetime.today()
    start = end - timedelta(days=days_back)

    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/transactions"
            f"?startDate={start.strftime('%Y-%m-%d')}"
            f"&endDate={end.strftime('%Y-%m-%d')}"
            f"&sportId=1"
        )
        r = requests.get(url, timeout=10)
        data = r.json()

        rows = []
        for t in data.get("transactions", []):
            rows.append({
                "player_id": t.get("person", {}).get("id"),
                "player_name": t.get("person", {}).get("fullName"),
                "transaction_type": t.get("typeDesc", ""),
                "transaction_date": t.get("date", ""),
                "description": t.get("description", ""),
                "team": t.get("team", {}).get("abbreviation", ""),
                "from_team": t.get("fromTeam", {}).get("abbreviation", ""),
                "to_team": t.get("toTeam", {}).get("abbreviation", ""),
            })

        df = pd.DataFrame(rows)
        return df

    except Exception as e:
        print(f"  IL transactions fetch error: {e}")
        return pd.DataFrame()


def get_current_il_list() -> set:
    """
    Returns set of player_ids currently on the 10-day or 15-day IL.
    Uses the injuredList roster type which is scoped to the active roster IL only.
    """
    try:
        url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
        r = requests.get(url, timeout=10)
        teams = r.json().get("teams", [])

        il_players = set()
        for team in teams:
            team_id = team["id"]
            try:
                # Use 'injuries' roster type which returns only MLB-level IL players
                roster_url = (
                    f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                    f"?rosterType=injuries"
                )
                rr = requests.get(roster_url, timeout=8)
                data = rr.json()
                for p in data.get("roster", []):
                    # Only add if status is 10-day or 15-day IL, not 60-day or minors
                    status = p.get("status", {}).get("description", "")
                    if "10-Day" in status or "15-Day" in status or status == "":
                        il_players.add(p["person"]["id"])
                time.sleep(0.1)
            except Exception:
                continue

        print(f"  Found {len(il_players)} players on active IL")
        return il_players

    except Exception as e:
        print(f"  IL list fetch error: {e}")
        return set()


# ── Rest Days ─────────────────────────────────────────────────────────────────

def compute_rest_days(player_game_log: pd.DataFrame,
                       game_date: str,
                       player_id: int) -> int:
    """
    Compute days of rest before a given game for a player.
    Uses their historical game log.
    Returns days since last game (0 = back-to-back, 1 = one day rest, etc.)
    """
    player_games = player_game_log[
        (player_game_log["player_id"] == player_id) &
        (player_game_log["game_date"] < game_date)
    ].sort_values("game_date", ascending=False)

    if player_games.empty:
        return 2  # no data, assume normal rest

    last_game = player_games.iloc[0]["game_date"]
    delta = (pd.to_datetime(game_date) - pd.to_datetime(last_game)).days
    return min(delta, 10)  # cap at 10


def get_days_since_last_game_bulk(player_ids: list, game_date: str,
                                   season: int = 2026) -> dict:
    """
    For a list of player IDs, compute days since last game via MLB Stats API.
    Returns dict: {player_id: days_rest}
    """
    end_date = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")

    rest_dict = {}

    for pid in player_ids:
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                f"?stats=gameLog&season={season}&group=hitting"
            )
            r = requests.get(url, timeout=8)
            data = r.json()

            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                rest_dict[pid] = 2
                continue

            # Find most recent game before today
            game_dates = []
            for s in splits:
                gd = s.get("date", "")
                if gd and gd < game_date:
                    game_dates.append(gd)

            if game_dates:
                last = max(game_dates)
                delta = (datetime.strptime(game_date, "%Y-%m-%d") -
                         datetime.strptime(last, "%Y-%m-%d")).days
                rest_dict[pid] = min(delta, 10)
            else:
                rest_dict[pid] = 2

            time.sleep(0.05)

        except Exception:
            rest_dict[pid] = 2

    return rest_dict


# ── HR Rate by Rest Days (empirical MLB averages) ─────────────────────────────

# Based on Retrosheet analysis — more rest = slightly fewer HRs (pitcher-like fatigue isn't factor,
# but rhythm/timing matters; back-to-back is slightly worse than 1 day rest)
REST_HR_MULTIPLIER = {
    0: 0.93,   # back-to-back (tired, less reactive)
    1: 1.00,   # one day rest (typical, baseline)
    2: 1.03,   # two days rest (fresh, peak)
    3: 1.01,   # three days (still fresh)
    4: 0.98,   # 4+ days (timing can get rusty)
    5: 0.96,
    6: 0.94,
    7: 0.92,   # week off — timing way off
}


def rest_hr_multiplier(days_rest: int) -> float:
    """Return HR rate multiplier for given days of rest."""
    return REST_HR_MULTIPLIER.get(min(days_rest, 7), 0.92)


# ── Return from IL Adjustment ─────────────────────────────────────────────────

def days_since_il_activation(player_id: int, game_date: str,
                               il_transactions: pd.DataFrame) -> int:
    """
    Check how many days ago a player was activated from IL.
    Players in first 7 days back from IL show reduced power.
    Returns days since activation (99 if not recently activated).
    """
    if il_transactions.empty:
        return 99

    activations = il_transactions[
        (il_transactions["player_id"] == player_id) &
        (il_transactions["transaction_type"].str.contains("Activated", case=False, na=False))
    ].copy()

    if activations.empty:
        return 99

    activations["transaction_date"] = pd.to_datetime(activations["transaction_date"])
    game_dt = pd.to_datetime(game_date)

    recent = activations[activations["transaction_date"] <= game_dt]
    if recent.empty:
        return 99

    latest = recent["transaction_date"].max()
    return (game_dt - latest).days


def il_return_hr_multiplier(days_since_activation: int) -> float:
    """
    HR rate reduction for players recently back from IL.
    Full power usually returns after 2 weeks.
    """
    if days_since_activation >= 14:
        return 1.0
    elif days_since_activation >= 7:
        return 0.92
    elif days_since_activation >= 3:
        return 0.85
    else:
        return 0.78  # first few games back — significant power reduction


def add_injury_rest_features(df: pd.DataFrame, game_date: str,
                              il_transactions: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add rest and injury features to prediction DataFrame.
    """
    if "days_rest" not in df.columns:
        df["days_rest"] = 1  # default

    df["rest_hr_multiplier"] = df["days_rest"].apply(rest_hr_multiplier)

    if il_transactions is not None and "player_id" in df.columns:
        df["days_since_il"] = df["player_id"].apply(
            lambda pid: days_since_il_activation(pid, game_date, il_transactions)
        )
        df["il_return_multiplier"] = df["days_since_il"].apply(il_return_hr_multiplier)
    else:
        df["days_since_il"] = 99
        df["il_return_multiplier"] = 1.0

    return df
