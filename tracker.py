"""
tracker.py — AlgoHub Pick Tracker
Logs HR prop picks, tracks results, generates performance stats.
Run daily to log picks, update results, and view ROI.

Usage:
  python tracker.py log       — log today's picks
  python tracker.py result    — update results for yesterday's picks
  python tracker.py stats     — view performance summary
  python tracker.py export    — export Twitter-ready stats
"""

import json
import os
import argparse
from datetime import datetime, timedelta

TRACKER_FILE = "data/pick_tracker.json"


# ── Data Layer ─────────────────────────────────────────────────────────────────

def load_tracker() -> dict:
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"picks": [], "parlays": []}


def save_tracker(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Logging ────────────────────────────────────────────────────────────────────

def log_pick(data: dict):
    """Log a single HR prop pick."""
    print("\n── Log Single Pick ──────────────────────────────")
    date = input("Date (YYYY-MM-DD, enter for today): ").strip()
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    player     = input("Player name: ").strip()
    pitcher    = input("Opposing pitcher: ").strip()
    park       = input("Park: ").strip()
    odds       = input("American odds (e.g. +460): ").strip()
    bet_amount = input("Bet amount ($, enter to skip): ").strip()
    notes      = input("Notes (barrel rate, wind, etc.): ").strip()

    pick = {
        "id":         len(data["picks"]) + 1,
        "date":       date,
        "type":       "single",
        "player":     player,
        "pitcher":    pitcher,
        "park":       park,
        "odds":       odds,
        "bet":        float(bet_amount) if bet_amount else None,
        "notes":      notes,
        "result":     None,  # "hit" or "miss" — fill in later
        "logged_at":  datetime.now().isoformat(),
    }

    data["picks"].append(pick)
    save_tracker(data)
    print(f"✓ Logged: {player} @ {odds}")


def log_parlay(data: dict):
    """Log a multi-leg HR parlay."""
    print("\n── Log Parlay ───────────────────────────────────")
    date = input("Date (YYYY-MM-DD, enter for today): ").strip()
    if not date:
        date = datetime.today().strftime("%Y-%m-%d")

    legs_input = input("How many legs? ").strip()
    n_legs = int(legs_input)

    legs = []
    for i in range(n_legs):
        print(f"\n  Leg {i+1}:")
        player  = input("  Player: ").strip()
        pitcher = input("  Pitcher: ").strip()
        odds    = input("  Odds: ").strip()
        legs.append({"player": player, "pitcher": pitcher, "odds": odds})

    parlay_odds = input("\nParlay odds (e.g. +16709): ").strip()
    bet_amount  = input("Bet amount ($, enter to skip): ").strip()
    notes       = input("Notes: ").strip()

    parlay = {
        "id":         len(data["parlays"]) + 1,
        "date":       date,
        "type":       "parlay",
        "legs":       legs,
        "odds":       parlay_odds,
        "bet":        float(bet_amount) if bet_amount else None,
        "notes":      notes,
        "result":     None,
        "logged_at":  datetime.now().isoformat(),
    }

    data["parlays"].append(parlay)
    save_tracker(data)
    print(f"✓ Logged {n_legs}-leg parlay @ {parlay_odds}")


# ── Results ────────────────────────────────────────────────────────────────────

def update_results(data: dict):
    """Update hit/miss for pending picks."""
    print("\n── Update Results ───────────────────────────────")

    pending_singles = [p for p in data["picks"] if p["result"] is None]
    pending_parlays = [p for p in data["parlays"] if p["result"] is None]

    if not pending_singles and not pending_parlays:
        print("No pending picks to update.")
        return

    # Singles
    for pick in pending_singles:
        print(f"\n  {pick['date']} | {pick['player']} vs {pick['pitcher']} @ {pick['odds']}")
        if pick.get("notes"):
            print(f"  Notes: {pick['notes']}")
        result = input("  Result (h=hit, m=miss, s=skip): ").strip().lower()
        if result == "h":
            pick["result"] = "hit"
        elif result == "m":
            pick["result"] = "miss"

    # Parlays
    for parlay in pending_parlays:
        legs_str = " + ".join(l["player"] for l in parlay["legs"])
        print(f"\n  {parlay['date']} | {len(parlay['legs'])}-leg: {legs_str} @ {parlay['odds']}")
        result = input("  Result (h=hit, m=miss, s=skip): ").strip().lower()
        if result == "h":
            parlay["result"] = "hit"
        elif result == "m":
            parlay["result"] = "miss"

    save_tracker(data)
    print("\n✓ Results updated")


# ── Stats ──────────────────────────────────────────────────────────────────────

def american_to_decimal(odds_str: str) -> float:
    """Convert American odds string to decimal multiplier."""
    try:
        odds = int(odds_str.replace("+", "").replace(" ", ""))
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    except:
        return 1.0


def compute_stats(data: dict) -> dict:
    """Compute performance stats across all picks."""
    all_picks = [p for p in data["picks"] if p["result"] in ("hit", "miss")]
    all_parlays = [p for p in data["parlays"] if p["result"] in ("hit", "miss")]

    # Singles stats
    single_hits   = sum(1 for p in all_picks if p["result"] == "hit")
    single_misses = sum(1 for p in all_picks if p["result"] == "miss")
    single_total  = single_hits + single_misses
    single_rate   = single_hits / single_total if single_total else 0

    # Parlay stats
    parlay_hits   = sum(1 for p in all_parlays if p["result"] == "hit")
    parlay_misses = sum(1 for p in all_parlays if p["result"] == "miss")
    parlay_total  = parlay_hits + parlay_misses
    parlay_rate   = parlay_hits / parlay_total if parlay_total else 0

    # ROI (for picks with bet amounts)
    total_bet = 0
    total_return = 0

    for pick in all_picks:
        if pick.get("bet"):
            total_bet += pick["bet"]
            if pick["result"] == "hit":
                dec = american_to_decimal(pick["odds"])
                total_return += pick["bet"] * dec
            # miss = just lost bet amount

    for parlay in all_parlays:
        if parlay.get("bet"):
            total_bet += parlay["bet"]
            if parlay["result"] == "hit":
                dec = american_to_decimal(parlay["odds"])
                total_return += parlay["bet"] * dec

    roi = ((total_return - total_bet) / total_bet * 100) if total_bet > 0 else None

    # Recent form (last 7 days)
    cutoff = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_picks = [p for p in all_picks if p["date"] >= cutoff]
    recent_hits  = sum(1 for p in recent_picks if p["result"] == "hit")
    recent_rate  = recent_hits / len(recent_picks) if recent_picks else 0

    return {
        "single_total":  single_total,
        "single_hits":   single_hits,
        "single_rate":   single_rate,
        "parlay_total":  parlay_total,
        "parlay_hits":   parlay_hits,
        "parlay_rate":   parlay_rate,
        "total_bet":     total_bet,
        "total_return":  total_return,
        "roi":           roi,
        "recent_picks":  len(recent_picks),
        "recent_hits":   recent_hits,
        "recent_rate":   recent_rate,
        "pending_singles": len([p for p in data["picks"] if p["result"] is None]),
        "pending_parlays": len([p for p in data["parlays"] if p["result"] is None]),
    }


def show_stats(data: dict):
    """Print performance summary."""
    s = compute_stats(data)

    print("\n" + "="*50)
    print("  🎯 AlgoHub HR Pick Tracker")
    print("="*50)

    print(f"\n  SINGLES ({s['single_total']} graded)")
    print(f"    Hit Rate:  {s['single_hits']}/{s['single_total']} ({s['single_rate']:.1%})")

    print(f"\n  PARLAYS ({s['parlay_total']} graded)")
    print(f"    Hit Rate:  {s['parlay_hits']}/{s['parlay_total']} ({s['parlay_rate']:.1%})")

    print(f"\n  LAST 7 DAYS (singles)")
    print(f"    {s['recent_hits']}/{s['recent_picks']} ({s['recent_rate']:.1%})")

    if s["roi"] is not None:
        profit = s["total_return"] - s["total_bet"]
        print(f"\n  ROI")
        print(f"    Wagered:  ${s['total_bet']:.2f}")
        print(f"    Returned: ${s['total_return']:.2f}")
        print(f"    Profit:   ${profit:+.2f}")
        print(f"    ROI:      {s['roi']:+.1f}%")

    print(f"\n  Pending: {s['pending_singles']} singles, {s['pending_parlays']} parlays")
    print("="*50)


def export_twitter(data: dict):
    """Generate Twitter-ready performance post."""
    s = compute_stats(data)

    lines = ["📊 @TheAlgoHub HR Prop Performance\n"]

    if s["single_total"] > 0:
        lines.append(f"Singles: {s['single_hits']}/{s['single_total']} ({s['single_rate']:.0%}) ✅")

    if s["parlay_total"] > 0:
        lines.append(f"Parlays: {s['parlay_hits']}/{s['parlay_total']} ({s['parlay_rate']:.0%}) 🎯")

    if s["recent_picks"] > 0:
        lines.append(f"Last 7 days: {s['recent_hits']}/{s['recent_picks']} ({s['recent_rate']:.0%}) 🔥")

    if s["roi"] is not None:
        profit = s["total_return"] - s["total_bet"]
        lines.append(f"ROI: {s['roi']:+.1f}% (${profit:+.2f})")

    lines.append("\nThe algo doesn't lie. 🤖")
    lines.append("#MLBProps #HomeRun #AlgoHub")

    tweet = "\n".join(lines)
    print("\n── Twitter Caption ──────────────────────────────")
    print(tweet)
    print("─────────────────────────────────────────────────")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AlgoHub Pick Tracker")
    parser.add_argument("command", choices=["log", "parlay", "result", "stats", "export"],
                        help="Command to run")
    args = parser.parse_args()

    data = load_tracker()

    if args.command == "log":
        log_pick(data)
    elif args.command == "parlay":
        log_parlay(data)
    elif args.command == "result":
        update_results(data)
    elif args.command == "stats":
        show_stats(data)
    elif args.command == "export":
        export_twitter(data)


if __name__ == "__main__":
    main()
