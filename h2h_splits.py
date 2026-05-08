"""
h2h_splits.py
Pulls career and recent batter vs pitcher head-to-head stats from MLB Stats API.
Small samples are blended with season-level stats using a credibility weighting system.
More ABs vs a pitcher = more weight on the actual h2h numbers.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time

CACHE_PATH = "data/h2h_cache.json"

# Minimum ABs before we start trusting h2h data meaningfully
MIN_AB_THRESHOLD = 10
# ABs at which we fully trust h2h over season stats
FULL_TRUST_AB = 60


def load_h2h_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_h2h_cache(cache: dict):
    os.makedirs("data", exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def fetch_h2h_stats(batter_id: int, pitcher_id: int, cache: dict) -> dict:
    """
    Fetch career batter vs pitcher stats from MLB Stats API.
    Returns dict with AB, H, HR, BB, SO, avg, slg, ops.
    Results are cached to avoid repeat API calls.
    """
    key = f"{batter_id}_{pitcher_id}"
    if key in cache:
        return cache[key]

    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
            f"?stats=vsPlayer&opposingPlayerId={pitcher_id}&group=hitting&sportId=1"
        )
        r = requests.get(url, timeout=10)
        data = r.json()

        splits = data.get("stats", [{}])[0].get("splits", [])

        if not splits:
            result = _empty_h2h()
        else:
            s = splits[0].get("stat", {})
            ab = int(s.get("atBats", 0))
            hr = int(s.get("homeRuns", 0))
            h = int(s.get("hits", 0))
            bb = int(s.get("baseOnBalls", 0))
            so = int(s.get("strikeOuts", 0))
            pa = ab + bb

            result = {
                "h2h_ab": ab,
                "h2h_pa": pa,
                "h2h_hr": hr,
                "h2h_hits": h,
                "h2h_bb": bb,
                "h2h_so": so,
                "h2h_avg": round(h / ab, 3) if ab > 0 else 0,
                "h2h_hr_rate": round(hr / ab, 4) if ab > 0 else 0,
                "h2h_bb_rate": round(bb / pa, 3) if pa > 0 else 0,
                "h2h_k_rate": round(so / pa, 3) if pa > 0 else 0,
                "h2h_slg": round(float(s.get("slugging", 0)), 3),
                "h2h_ops": round(float(s.get("ops", 0)), 3),
            }

        cache[key] = result
        time.sleep(0.15)
        return result

    except Exception as e:
        result = _empty_h2h()
        cache[key] = result
        return result


def fetch_h2h_recent(batter_id: int, pitcher_id: int,
                      seasons: list = [2024, 2025, 2026]) -> dict:
    """
    Fetch recent (last 2-3 seasons) h2h stats only.
    More relevant than career for pitchers who've changed their stuff.
    """
    all_ab = 0
    all_hr = 0
    all_pa = 0
    all_h = 0

    for season in seasons:
        try:
            url = (
                f"https://statsapi.mlb.com/api/v1/people/{batter_id}/stats"
                f"?stats=vsPlayer&opposingPlayerId={pitcher_id}"
                f"&group=hitting&sportId=1&season={season}"
            )
            r = requests.get(url, timeout=8)
            data = r.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                all_ab += int(s.get("atBats", 0))
                all_hr += int(s.get("homeRuns", 0))
                all_pa += all_ab + int(s.get("baseOnBalls", 0))
                all_h += int(s.get("hits", 0))
            time.sleep(0.1)
        except Exception:
            continue

    return {
        "h2h_recent_ab": all_ab,
        "h2h_recent_hr": all_hr,
        "h2h_recent_hr_rate": round(all_hr / all_ab, 4) if all_ab > 0 else 0,
        "h2h_recent_avg": round(all_h / all_ab, 3) if all_ab > 0 else 0,
    }


def _empty_h2h() -> dict:
    """Return zeroed-out h2h dict when no data is available."""
    return {
        "h2h_ab": 0, "h2h_pa": 0, "h2h_hr": 0, "h2h_hits": 0,
        "h2h_bb": 0, "h2h_so": 0, "h2h_avg": 0, "h2h_hr_rate": 0,
        "h2h_bb_rate": 0, "h2h_k_rate": 0, "h2h_slg": 0, "h2h_ops": 0,
    }


def credibility_weight(ab: int) -> float:
    """
    How much to trust h2h data vs season-level stats.
    0.0 = fully trust season stats (tiny sample)
    1.0 = fully trust h2h (large sample)
    Uses a smooth sigmoid-like curve.
    """
    if ab < MIN_AB_THRESHOLD:
        return 0.0
    elif ab >= FULL_TRUST_AB:
        return 1.0
    else:
        # Linear blend between thresholds
        return (ab - MIN_AB_THRESHOLD) / (FULL_TRUST_AB - MIN_AB_THRESHOLD)


def blended_hr_rate(h2h_hr_rate: float, h2h_ab: int,
                     season_hr_rate: float) -> float:
    """
    Blend h2h HR rate with season-level HR rate using credibility weighting.
    Small sample = mostly season stats.
    Large sample = mostly h2h.
    """
    w = credibility_weight(h2h_ab)
    return round((w * h2h_hr_rate) + ((1 - w) * season_hr_rate), 4)


def build_h2h_features(batter_id: int, pitcher_id: int,
                        season_hr_rate: float = 0.05) -> dict:
    """
    Full h2h feature builder for a single batter/pitcher matchup.
    Returns dict of features ready to add to prediction row.
    """
    cache = load_h2h_cache()

    # Career h2h
    career = fetch_h2h_stats(batter_id, pitcher_id, cache)
    save_h2h_cache(cache)

    # Recent h2h (last 3 seasons)
    recent = fetch_h2h_recent(batter_id, pitcher_id)

    # Blended HR rate
    blended = blended_hr_rate(
        career["h2h_hr_rate"],
        career["h2h_ab"],
        season_hr_rate
    )

    # Recent blend (even smaller sample, so tighter weight)
    recent_w = credibility_weight(recent["h2h_recent_ab"] * 2)  # weight recent more
    recent_blended = (recent_w * recent["h2h_recent_hr_rate"]) + ((1 - recent_w) * season_hr_rate)

    # Dominance flags
    is_dominated = (
        career["h2h_ab"] >= MIN_AB_THRESHOLD and
        career["h2h_hr_rate"] == 0 and
        career["h2h_k_rate"] > 0.35
    )
    is_owned = (
        career["h2h_ab"] >= MIN_AB_THRESHOLD and
        career["h2h_hr_rate"] > 0.08
    )

    features = {
        **career,
        **recent,
        "h2h_blended_hr_rate": blended,
        "h2h_recent_blended_hr_rate": recent_blended,
        "h2h_credibility_weight": credibility_weight(career["h2h_ab"]),
        "h2h_is_dominated": int(is_dominated),   # pitcher owns this batter
        "h2h_is_owned": int(is_owned),            # batter owns this pitcher
    }

    return features


def add_h2h_features_bulk(pred_df: pd.DataFrame,
                           batter_id_col: str = "player_id",
                           pitcher_id_col: str = "pitcher_id",
                           season_hr_rate_col: str = "batter_HR/FB") -> pd.DataFrame:
    """
    Add h2h features to a full prediction DataFrame.
    Processes each unique batter/pitcher combo once (deduped).
    """
    if batter_id_col not in pred_df.columns or pitcher_id_col not in pred_df.columns:
        print("  H2H: Missing batter or pitcher ID columns, skipping.")
        return pred_df

    # Get unique matchups
    matchups = pred_df[[batter_id_col, pitcher_id_col]].dropna().drop_duplicates()
    print(f"  Fetching h2h stats for {len(matchups)} unique matchups...")

    h2h_rows = []
    cache = load_h2h_cache()

    for _, row in matchups.iterrows():
        bid = int(row[batter_id_col])
        pid = row[pitcher_id_col]
        if pd.isna(pid):
            continue
        pid = int(pid)

        # Get season HR rate for this batter as the baseline
        batter_rows = pred_df[pred_df[batter_id_col] == bid]
        season_hr = float(batter_rows[season_hr_rate_col].iloc[0]) if (
            season_hr_rate_col in batter_rows.columns and not batter_rows.empty
        ) else 0.05

        career = fetch_h2h_stats(bid, pid, cache)
        recent = fetch_h2h_recent(bid, pid)

        blended = blended_hr_rate(career["h2h_hr_rate"], career["h2h_ab"], season_hr)
        recent_w = credibility_weight(recent["h2h_recent_ab"] * 2)
        recent_blended = (recent_w * recent["h2h_recent_hr_rate"]) + ((1 - recent_w) * season_hr)

        h2h_rows.append({
            batter_id_col: bid,
            pitcher_id_col: pid,
            **career,
            **recent,
            "h2h_blended_hr_rate": blended,
            "h2h_recent_blended_hr_rate": recent_blended,
            "h2h_credibility_weight": credibility_weight(career["h2h_ab"]),
            "h2h_is_dominated": int(career["h2h_ab"] >= MIN_AB_THRESHOLD and career["h2h_hr_rate"] == 0 and career["h2h_k_rate"] > 0.35),
            "h2h_is_owned": int(career["h2h_ab"] >= MIN_AB_THRESHOLD and career["h2h_hr_rate"] > 0.08),
        })

    save_h2h_cache(cache)

    if not h2h_rows:
        return pred_df

    h2h_df = pd.DataFrame(h2h_rows)
    pred_df = pred_df.merge(h2h_df, on=[batter_id_col, pitcher_id_col], how="left")

    # Fill missing (no pitcher ID) with neutral values
    h2h_fill_cols = [c for c in h2h_df.columns if c.startswith("h2h_")]
    for col in h2h_fill_cols:
        if col in pred_df.columns:
            pred_df[col] = pred_df[col].fillna(0)

    print(f"  H2H features added. Matchups with 10+ career ABs: "
          f"{(pred_df['h2h_ab'] >= MIN_AB_THRESHOLD).sum()}")

    return pred_df
