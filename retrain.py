"""
retrain.py
Weekly model retraining script incorporating new 2026 season data.
Run once a week (e.g. every Monday morning) to keep the model fresh.

Usage:
    python retrain.py
    python retrain.py --full    # full retrain from scratch (slower)
    python retrain.py --eval    # evaluate current model performance only
"""

import argparse
import pandas as pd
import numpy as np
import pybaseball as pb
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from feature_engineering import build_features, get_feature_columns
from model_training import prepare_training_data, train_model, save_model, load_model

pb.cache.enable()

PERFORMANCE_LOG = "models/performance_log.json"
CURRENT_SEASON = 2026


def load_performance_log() -> list:
    if os.path.exists(PERFORMANCE_LOG):
        with open(PERFORMANCE_LOG) as f:
            return json.load(f)
    return []


def save_performance_log(log: list):
    os.makedirs("models", exist_ok=True)
    with open(PERFORMANCE_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── Pull New 2026 Data ────────────────────────────────────────────────────────

def fetch_2026_data_to_date() -> pd.DataFrame:
    """
    Pull 2026 Statcast game logs from season start through yesterday.
    Uses bulk statcast() pull instead of per-player calls — much faster.
    """
    season_start = f"{CURRENT_SEASON}-03-20"
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Pulling 2026 bulk Statcast data from {season_start} to {yesterday}...")

    try:
        raw = pb.statcast(season_start, yesterday)
        if raw.empty:
            print("  No Statcast data returned")
            return pd.DataFrame()

        print(f"  Got {len(raw)} pitch records across {raw['game_date'].nunique()} days")

        raw["game_date"] = pd.to_datetime(raw["game_date"])
        raw["hr_flag"] = (raw["events"] == "home_run").astype(int)

        # Filter to batted ball events only for aggregation
        batted = raw[raw["type"] == "X"].copy()

        # Calculate barrel from launch_speed and launch_angle
        # Statcast barrel definition: EV >= 98 mph with optimal launch angle
        # Angle range expands as velocity increases
        def is_barrel(row):
            ev = row.get("launch_speed", 0)
            la = row.get("launch_angle", 0)
            if pd.isna(ev) or pd.isna(la) or ev < 98:
                return 0
            if ev >= 116:
                return int(8 <= la <= 50)
            min_la = 26 - (116 - ev) * 1.0
            max_la = 30 + (116 - ev) * 1.0
            return int(min_la <= la <= max_la)

        if "launch_speed" in batted.columns and "launch_angle" in batted.columns:
            batted["barrel"] = batted.apply(is_barrel, axis=1)
        else:
            batted["barrel"] = 0

        # Game-level aggregation per batter — PA/HR from all pitches, EV/barrel from batted balls only
        pa_hr = (
            raw.groupby(["batter", "game_date", "home_team", "away_team"])
            .agg(
                pa=("at_bat_number", "nunique"),
                hr=("hr_flag", "max"),
            )
            .reset_index()
        )

        batted_agg_dict = {
            "avg_ev": ("launch_speed", "mean"),
            "avg_la": ("launch_angle", "mean"),
        }
        if "barrel" in batted.columns:
            batted_agg_dict["barrel_count"] = ("barrel", "sum")

        batted_agg = (
            batted.groupby(["batter", "game_date", "home_team", "away_team"])
            .agg(**batted_agg_dict)
            .reset_index()
        )

        game_logs = pa_hr.merge(batted_agg, on=["batter", "game_date", "home_team", "away_team"], how="left")
        game_logs = game_logs.rename(columns={"batter": "player_id"})

        if "barrel_count" not in game_logs.columns:
            game_logs["barrel_count"] = 0
        game_logs["barrel_count"] = game_logs["barrel_count"].fillna(0)

        # Attach season-level Statcast stats from leaderboard
        print("  Pulling Statcast leaderboard for season stats...")
        try:
            lb = pb.statcast_batter_exitvelo_barrels(CURRENT_SEASON, minBBE=5)
            if lb.empty:
                lb = pb.statcast_batter_exitvelo_barrels(CURRENT_SEASON - 1, minBBE=20)

            lb_lookup = {}
            for _, row in lb.iterrows():
                pid = row.get("player_id")
                if pd.notna(pid):
                    lb_lookup[int(pid)] = {
                        "season_barrel_rate":   row.get("brl_percent", 0) / 100,
                        "season_hard_hit_rate": row.get("ev95percent", 0) / 100,
                        "season_avg_ev":        row.get("avg_hit_speed", 88.0),
                    }

            game_logs["season_barrel_rate"]   = game_logs["player_id"].map(lambda x: lb_lookup.get(x, {}).get("season_barrel_rate", 0.06))
            game_logs["season_hard_hit_rate"] = game_logs["player_id"].map(lambda x: lb_lookup.get(x, {}).get("season_hard_hit_rate", 0.35))
            game_logs["season_avg_ev"]        = game_logs["player_id"].map(lambda x: lb_lookup.get(x, {}).get("season_avg_ev", 88.0))

        except Exception as e:
            print(f"  Leaderboard pull failed: {e}")
            game_logs["season_barrel_rate"]   = 0.06
            game_logs["season_hard_hit_rate"] = 0.35
            game_logs["season_avg_ev"]        = 88.0

        game_logs["season"] = CURRENT_SEASON
        game_logs["home_team"] = game_logs["home_team"].str.upper()

        # Add missing season columns with defaults
        if "season_avg_la" not in game_logs.columns:
            game_logs["season_avg_la"] = 12.0
        if "season_hr_fb_rate" not in game_logs.columns:
            game_logs["season_hr_fb_rate"] = 0.12
        if "season_fb_rate" not in game_logs.columns:
            game_logs["season_fb_rate"] = 0.35

        from data_collection import PARK_FACTORS
        game_logs["park_factor"] = game_logs["home_team"].map(PARK_FACTORS).fillna(1.0)

        if "barrel_count" not in game_logs.columns:
            game_logs["barrel_count"] = 0

        print(f"  Built {len(game_logs)} player-game records for {game_logs['player_id'].nunique()} players")
        game_logs.to_csv(f"data/training_data_{CURRENT_SEASON}.csv", index=False)
        return game_logs

    except Exception as e:
        print(f"  Bulk Statcast pull failed: {e}")
        return pd.DataFrame()


def merge_with_historical(df_new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new 2026 data with existing 2023–2025 training data.
    """
    historical_path = "data/training_data.csv"
    if os.path.exists(historical_path):
        df_hist = pd.read_csv(historical_path)
        df_hist["game_date"] = pd.to_datetime(df_hist["game_date"])
        print(f"  Historical data: {len(df_hist)} records")

        # Remove any previously stored 2026 data to avoid duplication
        df_hist = df_hist[df_hist["season"] != CURRENT_SEASON]

        df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_combined = df_new

    print(f"  Combined dataset: {len(df_combined)} records")
    return df_combined


# ── Evaluate Current Model ────────────────────────────────────────────────────

def evaluate_model_on_recent(days: int = 14) -> dict:
    """
    Evaluate model performance on the most recent N days of 2026 data.
    Returns AUC, Brier score, calibration error.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.calibration import calibration_curve

    recent_path = f"data/training_data_{CURRENT_SEASON}.csv"
    if not os.path.exists(recent_path):
        print("No 2026 data available for evaluation")
        return {}

    df = pd.read_csv(recent_path)
    df["game_date"] = pd.to_datetime(df["game_date"])

    cutoff = datetime.today() - timedelta(days=days)
    df_recent = df[df["game_date"] >= cutoff]

    if len(df_recent) < 50:
        print(f"  Only {len(df_recent)} records in last {days} days — too few to evaluate")
        return {}

    try:
        model, feature_cols = load_model()
    except Exception:
        print("  No model loaded")
        return {}

    # Build features — add missing columns with defaults if needed
    for col, default in [("season_avg_la", 12.0), ("season_hr_fb_rate", 0.12), ("season_fb_rate", 0.35)]:
        if col not in df_recent.columns:
            df_recent[col] = default

    features = build_features(df_recent, add_weather=False)
    avail_cols = [c for c in feature_cols if c in features.columns]
    X = features[avail_cols].fillna(features[avail_cols].median())
    y = features["hr"].astype(int)

    if len(y) < 50:
        return {}

    preds = model.predict_proba(X)[:, 1]

    auc = roc_auc_score(y, preds)
    brier = brier_score_loss(y, preds)

    # Calibration: expected vs actual HR rate in deciles
    prob_true, prob_pred = calibration_curve(y, preds, n_bins=5)
    cal_error = float(np.mean(np.abs(prob_true - prob_pred)))

    results = {
        "date": datetime.today().strftime("%Y-%m-%d"),
        "n_samples": len(y),
        "hr_rate": float(y.mean()),
        "auc": round(auc, 4),
        "brier": round(brier, 4),
        "calibration_error": round(cal_error, 4),
        "mean_predicted_prob": round(float(preds.mean()), 4),
    }

    print(f"\n  📊 Model Evaluation (last {days} days, n={len(y)})")
    print(f"     AUC:               {auc:.4f}")
    print(f"     Brier Score:       {brier:.4f}")
    print(f"     Calibration Error: {cal_error:.4f}")
    print(f"     HR Rate (actual):  {y.mean():.3f}")
    print(f"     HR Rate (pred):    {preds.mean():.3f}")

    return results


# ── Main Retrain Flow ─────────────────────────────────────────────────────────

def run_retraining(full: bool = False):
    """
    Full weekly retraining pipeline:
    1. Pull new 2026 data
    2. Evaluate current model performance
    3. Merge with historical data
    4. Build features
    5. Retrain model
    6. Log performance
    """
    print(f"\n{'='*55}")
    print(f"  WEEKLY RETRAIN — {datetime.today().strftime('%Y-%m-%d')}")
    print(f"{'='*55}\n")

    # Evaluate current model first
    print("Evaluating current model on recent games...")
    perf = evaluate_model_on_recent(days=14)

    # Pull 2026 data
    df_2026 = fetch_2026_data_to_date()

    if df_2026.empty and not full:
        print("No new data to retrain on. Exiting.")
        return

    # Merge
    df_all = merge_with_historical(df_2026) if not df_2026.empty else pd.read_csv("data/training_data.csv")

    # Feature engineering
    print("\nBuilding features...")
    features = build_features(df_all, add_weather=False)  # weather too slow for full retrain

    # Train
    X, y, groups, feature_cols = prepare_training_data(features)
    model, cv_scores = train_model(X, y, groups)
    save_model(model, feature_cols)

    # Log
    perf["post_retrain_auc"] = round(float(np.mean(cv_scores["cv_auc"])), 4)
    perf["post_retrain_brier"] = round(float(np.mean(cv_scores["cv_brier"])), 4)
    perf["training_samples"] = len(X)

    log = load_performance_log()
    log.append(perf)
    save_performance_log(log)

    print(f"\n✓ Retraining complete.")
    print(f"  New model AUC: {perf['post_retrain_auc']:.4f}")
    print(f"  Training samples: {len(X)}")
    print(f"  Performance log updated ({len(log)} entries)")


def print_performance_history():
    """Print all historical model performance metrics."""
    log = load_performance_log()
    if not log:
        print("No performance history found.")
        return

    print(f"\n{'='*65}")
    print(f"  MODEL PERFORMANCE HISTORY")
    print(f"{'='*65}")
    print(f"{'Date':<12} {'AUC':>8} {'Brier':>8} {'Cal Err':>10} {'Samples':>10}")
    print("-" * 55)
    for entry in log:
        print(
            f"{entry.get('date','?'):<12} "
            f"{entry.get('auc', entry.get('post_retrain_auc', '?')):>8} "
            f"{entry.get('brier', entry.get('post_retrain_brier', '?')):>8} "
            f"{entry.get('calibration_error', '?'):>10} "
            f"{entry.get('training_samples', '?'):>10}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly model retraining")
    parser.add_argument("--full", action="store_true", help="Full retrain from scratch")
    parser.add_argument("--eval", action="store_true", help="Evaluate current model only")
    parser.add_argument("--history", action="store_true", help="Print performance history")
    args = parser.parse_args()

    if args.history:
        print_performance_history()
    elif args.eval:
        evaluate_model_on_recent(days=14)
    else:
        run_retraining(full=args.full)
