"""
feature_engineering.py
Transforms raw game logs into model-ready features.
Handles rolling windows, platoon splits, matchup features, weather, park factors.
"""

import pandas as pd
import numpy as np
from data_collection import PARK_FACTORS, fetch_weather, wind_hr_boost


# ── Rolling Window Features ───────────────────────────────────────────────────

def add_rolling_batter_features(df: pd.DataFrame, windows: list = [7, 15, 30]) -> pd.DataFrame:
    """
    For each batter, compute rolling stats prior to each game.
    Uses shift(1) so there's no data leakage (today's game not included).
    Calculates barrel_count from launch speed/angle if all zeros.
    """
    df = df.sort_values(["player_id", "game_date"]).copy()

    # If barrel_count is all zeros, recalculate from avg_ev and avg_la
    # using the per-game averages as a proxy
    if "barrel_count" in df.columns and df["barrel_count"].sum() == 0:
        if "avg_ev" in df.columns and "avg_la" in df.columns:
            def estimate_barrel_games(row):
                ev = row.get("avg_ev", 0)
                la = row.get("avg_la", 0)
                if pd.isna(ev) or pd.isna(la) or ev < 98:
                    return 0
                if ev >= 116:
                    return int(8 <= la <= 50)
                min_la = 26 - (116 - ev) * 1.0
                max_la = 30 + (116 - ev) * 1.0
                return int(min_la <= la <= max_la)
            df["barrel_count"] = df.apply(estimate_barrel_games, axis=1)

    for w in windows:
        min_p = max(3, w // 3)  # e.g. w=7 → min 3, w=15 → min 5, w=30 → min 10
        df[f"hr_rate_last{w}"] = (
            df.groupby("player_id")["hr"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=min_p).mean())
        )
        df[f"barrel_rate_last{w}"] = (
            df.groupby("player_id")["barrel_count"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=min_p).mean())
        )
        df[f"avg_ev_last{w}"] = (
            df.groupby("player_id")["avg_ev"]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=min_p).mean())
        )

    return df


def add_season_pa_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Blend rolling recent form with season-long stats.
    Early in season, lean on prior year; later, lean on current year.
    Uses a simple weight: min(season_pa / 300, 1.0)
    """
    df["season_pa_cumulative"] = df.groupby(["player_id", "season"])["pa"].cumsum()
    df["season_weight"] = (df["season_pa_cumulative"] / 300).clip(upper=1.0)
    return df


# ── Pitcher Matchup Features ──────────────────────────────────────────────────

def build_pitcher_features(fg_pitching: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and rename pitcher FanGraphs features for joining.
    """
    rename = {
        "IDfg": "pitcher_fg_id",
        "Name": "pitcher_name",
        "Team": "pitcher_team",
        "HR9": "pitcher_hr9",
        "HR/FB": "pitcher_hrfb",
        "Hard%": "pitcher_hard_pct",
        "K%": "pitcher_k_pct",
        "FIP": "pitcher_fip",
        "xFIP": "pitcher_xfip",
        "FB%": "pitcher_fb_pct",
    }
    available = {k: v for k, v in rename.items() if k in fg_pitching.columns}
    return fg_pitching.rename(columns=available)[list(available.values())].copy()


def add_platoon_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple platoon advantage flag.
    Assumes batter_hand and pitcher_hand columns exist (add them during data collection).
    If not available, defaults to 0.
    """
    if "batter_hand" in df.columns and "pitcher_hand" in df.columns:
        # Platoon advantage: batter and pitcher opposite hands
        df["platoon_advantage"] = (df["batter_hand"] != df["pitcher_hand"]).astype(int)
    else:
        df["platoon_advantage"] = 0
    return df


# ── Park & Weather Features ───────────────────────────────────────────────────

def add_park_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure park factor is present and add park tier."""
    if "park_factor" not in df.columns:
        df["park_factor"] = df["home_team"].map(PARK_FACTORS).fillna(1.0)

    # Tier: hitter-friendly (>1.05), neutral, pitcher-friendly (<0.95)
    df["park_tier"] = pd.cut(
        df["park_factor"],
        bins=[0, 0.95, 1.05, 99],
        labels=["pitchers_park", "neutral", "hitters_park"]
    )
    return df


def add_weather_features(df: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    """
    Fetch weather for each unique team/date combo and join to df.
    Cached to avoid repeated API calls.
    """
    weather_cache = {}
    results = []

    unique_combos = df[["home_team", "game_date"]].drop_duplicates()
    print(f"  Fetching weather for {len(unique_combos)} team/date combos...")

    for _, row in unique_combos.iterrows():
        team = row["home_team"]
        date = str(row["game_date"])[:10]
        key = f"{team}_{date}"

        if key not in weather_cache:
            w = fetch_weather(team, date)
            w["wind_boost"] = wind_hr_boost(w["wind_mph"], w["wind_deg"])
            weather_cache[key] = w

        r = weather_cache[key].copy()
        r["home_team"] = team
        r["game_date"] = row["game_date"]
        results.append(r)

    weather_df = pd.DataFrame(results)
    df = df.merge(weather_df, on=["home_team", "game_date"], how="left")
    return df


# ── Lineup Position ───────────────────────────────────────────────────────────

def add_lineup_pa_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate expected PA per game based on lineup spot.
    If batting_order is available, map to typical PA; else use historical pa.
    """
    lineup_pa_map = {1: 4.5, 2: 4.3, 3: 4.2, 4: 4.1, 5: 3.9,
                     6: 3.7, 7: 3.5, 8: 3.4, 9: 3.2}

    if "batting_order" in df.columns:
        df["expected_pa"] = df["batting_order"].map(lineup_pa_map).fillna(3.7)
    else:
        df["expected_pa"] = df["pa"].clip(lower=1, upper=6)

    return df


# ── Master Feature Builder ────────────────────────────────────────────────────

def build_features(raw_df: pd.DataFrame, pitcher_df: pd.DataFrame = None,
                   add_weather: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input: raw game log DataFrame from data_collection.py
    Output: feature matrix ready for modeling
    """
    print("Building features...")
    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Rolling batter features
    print("  Adding rolling batter features...")
    df = add_rolling_batter_features(df)

    # Season PA weight (recency blending)
    df = add_season_pa_weight(df)

    # Park features
    print("  Adding park features...")
    df = add_park_features(df)

    # Weather
    if add_weather:
        print("  Adding weather features...")
        df = add_weather_features(df)
    else:
        df["temp_f"] = 72
        df["wind_mph"] = 5
        df["wind_boost"] = 0
        df["precip_mm"] = 0

    # Lineup PA estimate
    df = add_lineup_pa_estimate(df)

    # Platoon
    df = add_platoon_split(df)

    # Pitcher join
    if pitcher_df is not None:
        pitcher_features = build_pitcher_features(pitcher_df)
        if "pitcher_fg_id" in df.columns:
            df = df.merge(pitcher_features, on="pitcher_fg_id", how="left")

    # Month of season (HR rates vary by month)
    df["month"] = df["game_date"].dt.month

    # Drop rows with too many nulls in key features
    key_features = ["barrel_rate_last15", "avg_ev_last15", "park_factor"]
    df = df.dropna(subset=[f for f in key_features if f in df.columns])

    print(f"  Feature matrix shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    """Returns the ordered list of feature columns used for model training."""
    return [
        # Season-level Statcast (most important — always populated)
        "season_barrel_rate", "season_hard_hit_rate", "season_avg_ev",
        "season_avg_la", "season_hr_fb_rate", "season_fb_rate",
        "season_weight",

        # Rolling batter form (fills in as season progresses)
        "hr_rate_last7", "hr_rate_last15", "hr_rate_last30",
        "barrel_rate_last7", "barrel_rate_last15", "barrel_rate_last30",
        "avg_ev_last7", "avg_ev_last15", "avg_ev_last30",

        # Pitcher matchup
        "pitcher_hrfb", "pitcher_hr9", "pitcher_hard_pct", "pitcher_fip", "pitcher_xfip",

        # Park & context
        "park_factor", "platoon_advantage", "expected_pa", "month",

        # Weather
        "temp_f", "wind_mph", "wind_boost", "precip_mm",

        # H2H batter vs pitcher history
        "h2h_ab", "h2h_hr_rate", "h2h_blended_hr_rate",
        "h2h_recent_blended_hr_rate", "h2h_credibility_weight",
        "h2h_ops", "h2h_k_rate", "h2h_is_dominated", "h2h_is_owned",
    ]


if __name__ == "__main__":
    import os

    if not os.path.exists("data/training_data.csv"):
        print("Run data_collection.py first to generate training_data.csv")
    else:
        raw = pd.read_csv("data/training_data.csv")
        features = build_features(raw, add_weather=False)  # set True for full weather
        features.to_csv("data/features.csv", index=False)
        print("Features saved to data/features.csv")
