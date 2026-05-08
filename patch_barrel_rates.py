"""
patch_barrel_rates.py
One-time script to fix season_barrel_rate and season_hard_hit_rate
in the existing training_data.csv using the authoritative Statcast leaderboard.
Run this once, then retrain the model.
"""

import pandas as pd
import pybaseball as pb

pb.cache.enable()

print("Loading training data...")
df = pd.read_csv("data/training_data.csv")
print(f"  {len(df)} records, {df['player_id'].nunique()} players")
print(f"  Current avg barrel rate: {df['season_barrel_rate'].mean():.4f}")

print("\nPulling 2025 Statcast leaderboard...")
lb = pb.statcast_batter_exitvelo_barrels(2025, minBBE=20)
print(f"  Got {len(lb)} players from leaderboard")

# Build lookup: player_id -> correct stats
lb_lookup = {}
for _, row in lb.iterrows():
    pid = row.get("player_id")
    if pd.notna(pid):
        lb_lookup[int(pid)] = {
            "season_barrel_rate":   row.get("brl_percent", 0) / 100,
            "season_hard_hit_rate": row.get("ev95percent", 0) / 100,
            "season_avg_ev":        row.get("avg_hit_speed", 88.0),
        }

print(f"\nPatching {len(lb_lookup)} players...")
patched = 0
for pid, stats in lb_lookup.items():
    mask = df["player_id"] == pid
    if mask.any():
        for col, val in stats.items():
            df.loc[mask, col] = val
        patched += 1

print(f"  Patched {patched} players")
print(f"  New avg barrel rate: {df['season_barrel_rate'].mean():.4f}")

# Verify Judge
judge = df[df["player_id"] == 592450]
if not judge.empty:
    print(f"\n  Aaron Judge: barrel={judge['season_barrel_rate'].iloc[0]:.3f}, "
          f"hard_hit={judge['season_hard_hit_rate'].iloc[0]:.3f}, "
          f"ev={judge['season_avg_ev'].iloc[0]:.1f}")

df.to_csv("data/training_data.csv", index=False)
# Also update season cache
df.to_csv("data/season_2025.csv", index=False)
print("\nSaved. Now run: python feature_engineering.py && python model_training.py")
