"""
Closing Line Value (CLV) Tracker.

Logs bets, captures closing odds before kickoff, records results,
and calculates CLV metrics — the single best predictor of long-term
betting profitability.

If you consistently beat the closing line, you have a real edge.
If not, any profits are likely variance.
"""
import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "bets.db")


def _get_conn():
    """Get SQLite connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            commence_time TEXT,
            bet_side TEXT NOT NULL CHECK (bet_side IN ('over', 'under')),
            odds_at_bet REAL NOT NULL,
            bookmaker TEXT,
            model_prob REAL,
            blended_prob REAL,
            fair_implied REAL,
            edge REAL,
            ev REAL,
            kelly_stake REAL,
            n_agree INTEGER,
            sharp_fair REAL,
            closing_odds REAL,
            closing_pinnacle REAL,
            closing_timestamp TEXT,
            actual_goals INTEGER,
            outcome TEXT CHECK (outcome IN ('over', 'under', NULL)),
            profit_loss REAL,
            settled INTEGER DEFAULT 0,
            notes TEXT
        )
    """)
    conn.commit()
    return conn


def log_bet(home_team, away_team, bet_side, odds_at_bet,
            bookmaker=None, model_prob=None, blended_prob=None,
            fair_implied=None, edge=None, ev=None, kelly_stake=None,
            n_agree=None, sharp_fair=None, commence_time=None, notes=None):
    """Log a new bet. Returns the bet ID."""
    conn = _get_conn()
    try:
        cur = conn.execute("""
            INSERT INTO bets (timestamp, home_team, away_team, commence_time,
                              bet_side, odds_at_bet, bookmaker, model_prob,
                              blended_prob, fair_implied, edge, ev,
                              kelly_stake, n_agree, sharp_fair, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            home_team, away_team, commence_time,
            bet_side, odds_at_bet, bookmaker, model_prob,
            blended_prob, fair_implied, edge, ev,
            kelly_stake, n_agree, sharp_fair, notes,
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_closing_odds(bet_id, closing_odds, closing_pinnacle=None):
    """Record closing odds for a bet (fetched just before kickoff)."""
    conn = _get_conn()
    try:
        conn.execute("""
            UPDATE bets SET closing_odds = ?, closing_pinnacle = ?,
                            closing_timestamp = ?
            WHERE id = ?
        """, (closing_odds, closing_pinnacle, datetime.now().isoformat(), bet_id))
        conn.commit()
    finally:
        conn.close()


def settle_bet(bet_id, actual_goals):
    """Settle a bet with the actual match result."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM bets WHERE id = ?", (bet_id,)).fetchone()
        if not row:
            return None

        outcome = "over" if actual_goals > 2 else "under"
        won = (outcome == row["bet_side"])
        profit_loss = (row["odds_at_bet"] - 1.0) if won else -1.0

        conn.execute("""
            UPDATE bets SET actual_goals = ?, outcome = ?,
                            profit_loss = ?, settled = 1
            WHERE id = ?
        """, (actual_goals, outcome, profit_loss, bet_id))
        conn.commit()
        return {"won": won, "profit_loss": profit_loss, "outcome": outcome}
    finally:
        conn.close()


def fetch_and_store_closing_odds():
    """Fetch current odds for all unsettled bets and store as closing odds.

    Should be called ~5-15 minutes before kickoff for best accuracy.
    Can also be called periodically — the latest fetch before kickoff wins.
    """
    conn = _get_conn()
    try:
        unsettled = conn.execute("""
            SELECT id, home_team, away_team FROM bets WHERE settled = 0
        """).fetchall()
    finally:
        conn.close()

    if not unsettled:
        return 0

    try:
        from api.odds_api import fetch_epl_odds, get_best_odds, match_to_our_teams
        matches = fetch_epl_odds(force_refresh=True)
    except Exception as e:
        print(f"  CLV: Failed to fetch closing odds: {e}")
        return 0

    # Build lookup by team names
    odds_lookup = {}
    for m in matches:
        best = get_best_odds(m)
        if best is None:
            continue
        # Store by both API names and try to match
        key = (m["home_team"], m["away_team"])
        odds_lookup[key] = {
            "best_over": best["best_over"],
            "best_under": best["best_under"],
            "pinnacle_over": best.get("pinnacle_over"),
            "pinnacle_under": best.get("pinnacle_under"),
        }

    updated = 0
    conn = _get_conn()
    try:
        for bet in unsettled:
            # Try to find matching fixture
            for key, odds in odds_lookup.items():
                api_home, api_away = key
                # Simple substring match since names may differ slightly
                if (_name_match(bet["home_team"], api_home) and
                        _name_match(bet["away_team"], api_away)):
                    side = "over"  # We need to know the bet side
                    row = conn.execute("SELECT bet_side FROM bets WHERE id = ?",
                                       (bet["id"],)).fetchone()
                    if row:
                        side = row["bet_side"]

                    closing = odds[f"best_{side}"]
                    pin = odds.get(f"pinnacle_{side}")
                    # Convert Pinnacle odds to fair prob then back if available
                    closing_pin = None
                    if pin:
                        closing_pin = pin

                    conn.execute("""
                        UPDATE bets SET closing_odds = ?, closing_pinnacle = ?,
                                        closing_timestamp = ?
                        WHERE id = ?
                    """, (closing, closing_pin, datetime.now().isoformat(), bet["id"]))
                    updated += 1
                    break
        conn.commit()
    finally:
        conn.close()

    return updated


def _name_match(our_name, api_name):
    """Fuzzy match between our team name and API team name."""
    our_words = set(our_name.lower().replace("fc", "").replace("afc", "").split())
    api_words = set(api_name.lower().replace("fc", "").replace("afc", "").split())
    overlap = len(our_words & api_words)
    return overlap >= 1 and overlap >= len(api_words) * 0.5


def get_all_bets(settled_only=False):
    """Get all bets as a DataFrame."""
    conn = _get_conn()
    try:
        query = "SELECT * FROM bets"
        if settled_only:
            query += " WHERE settled = 1"
        query += " ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_unsettled_bets():
    """Get unsettled bets as a DataFrame."""
    conn = _get_conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM bets WHERE settled = 0 ORDER BY timestamp DESC", conn)
        return df
    finally:
        conn.close()


def calculate_clv():
    """Calculate CLV metrics across all settled bets with closing odds.

    CLV = (odds_at_bet / closing_odds) - 1
    Positive CLV means you consistently got better odds than the closing line.

    Returns dict with summary statistics.
    """
    conn = _get_conn()
    try:
        df = pd.read_sql_query("""
            SELECT * FROM bets
            WHERE settled = 1 AND closing_odds IS NOT NULL AND closing_odds > 1
        """, conn)
    finally:
        conn.close()

    if df.empty:
        return {"n_bets": 0, "message": "No settled bets with closing odds yet."}

    # CLV = how much better your odds were vs closing line
    df["clv"] = (df["odds_at_bet"] / df["closing_odds"]) - 1
    df["clv_pct"] = df["clv"] * 100

    # Closing implied probability
    df["closing_implied"] = 1 / df["closing_odds"]
    df["bet_implied"] = 1 / df["odds_at_bet"]

    # Did we beat the closing line?
    df["beat_close"] = df["odds_at_bet"] > df["closing_odds"]

    # Pinnacle CLV (sharper benchmark)
    pin_df = df[df["closing_pinnacle"].notna() & (df["closing_pinnacle"] > 1)]
    pin_clv = None
    pin_beat_rate = None
    if not pin_df.empty:
        pin_df = pin_df.copy()
        pin_df["pin_clv"] = (pin_df["odds_at_bet"] / pin_df["closing_pinnacle"]) - 1
        pin_clv = float(pin_df["pin_clv"].mean() * 100)
        pin_beat_rate = float(pin_df["odds_at_bet"].gt(pin_df["closing_pinnacle"]).mean() * 100)

    # Split by side
    over_df = df[df["bet_side"] == "over"]
    under_df = df[df["bet_side"] == "under"]

    result = {
        "n_bets": len(df),
        "mean_clv_pct": float(df["clv_pct"].mean()),
        "median_clv_pct": float(df["clv_pct"].median()),
        "beat_close_rate": float(df["beat_close"].mean() * 100),
        "mean_edge_at_bet": float(df["edge"].mean() * 100) if "edge" in df and df["edge"].notna().any() else None,
        "actual_roi": float(df["profit_loss"].sum() / len(df) * 100) if df["profit_loss"].notna().any() else None,
        "win_rate": float((df["profit_loss"] > 0).mean() * 100) if df["profit_loss"].notna().any() else None,
        "pinnacle_clv_pct": pin_clv,
        "pinnacle_beat_rate": pin_beat_rate,
        "over_clv_pct": float(over_df["clv_pct"].mean()) if not over_df.empty else None,
        "under_clv_pct": float(under_df["clv_pct"].mean()) if not under_df.empty else None,
        "over_n": len(over_df),
        "under_n": len(under_df),
        # For time-series plotting
        "clv_series": df[["timestamp", "clv_pct", "bet_side", "profit_loss",
                          "odds_at_bet", "closing_odds"]].to_dict("records"),
    }

    return result


def clv_report():
    """Print a formatted CLV report to console."""
    stats = calculate_clv()
    if stats["n_bets"] == 0:
        print("No settled bets with closing odds yet.")
        return stats

    print("=" * 60)
    print("CLOSING LINE VALUE REPORT")
    print("=" * 60)
    print(f"  Settled bets with closing odds: {stats['n_bets']}")
    print(f"  Mean CLV:         {stats['mean_clv_pct']:+.2f}%")
    print(f"  Median CLV:       {stats['median_clv_pct']:+.2f}%")
    print(f"  Beat close rate:  {stats['beat_close_rate']:.1f}%")
    if stats["pinnacle_clv_pct"] is not None:
        print(f"  Pinnacle CLV:     {stats['pinnacle_clv_pct']:+.2f}%")
        print(f"  Beat Pinnacle:    {stats['pinnacle_beat_rate']:.1f}%")
    if stats["actual_roi"] is not None:
        print(f"  Actual ROI:       {stats['actual_roi']:+.1f}%")
        print(f"  Win rate:         {stats['win_rate']:.1f}%")
    if stats["over_n"] > 0:
        print(f"  Over CLV:         {stats['over_clv_pct']:+.2f}% ({stats['over_n']} bets)")
    if stats["under_n"] > 0:
        print(f"  Under CLV:        {stats['under_clv_pct']:+.2f}% ({stats['under_n']} bets)")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    # Quick test / report
    stats = clv_report()
    if stats["n_bets"] == 0:
        print("\nTo get started:")
        print("  1. Use the dashboard to log bets (click 'Log Bet' on value bets)")
        print("  2. Run: python clv_tracker.py --fetch-closing  (before kickoff)")
        print("  3. Run: python clv_tracker.py --settle <bet_id> <goals>")
        print("\nOr import and use programmatically:")
        print("  from clv_tracker import log_bet, settle_bet, calculate_clv")

    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--fetch-closing":
            n = fetch_and_store_closing_odds()
            print(f"\nUpdated closing odds for {n} bets.")
        elif sys.argv[1] == "--settle" and len(sys.argv) >= 4:
            bet_id = int(sys.argv[2])
            goals = int(sys.argv[3])
            result = settle_bet(bet_id, goals)
            if result:
                print(f"\nBet #{bet_id}: {'WON' if result['won'] else 'LOST'} "
                      f"(P/L: {result['profit_loss']:+.2f} units)")
            else:
                print(f"\nBet #{bet_id} not found.")
        elif sys.argv[1] == "--settle-all":
            _settle_from_results()
        elif sys.argv[1] == "--list":
            df = get_all_bets()
            if df.empty:
                print("No bets logged yet.")
            else:
                for _, row in df.iterrows():
                    status = "SETTLED" if row["settled"] else "PENDING"
                    clv = ""
                    if row["closing_odds"] and row["closing_odds"] > 1:
                        clv_val = (row["odds_at_bet"] / row["closing_odds"] - 1) * 100
                        clv = f" CLV:{clv_val:+.1f}%"
                    pl = f" P/L:{row['profit_loss']:+.2f}" if pd.notna(row["profit_loss"]) else ""
                    print(f"  #{row['id']} [{status}] {row['home_team']} vs {row['away_team']} "
                          f"{row['bet_side'].upper()} @{row['odds_at_bet']:.2f}{clv}{pl}")
        elif sys.argv[1] == "--unsettled":
            df = get_unsettled_bets()
            if df.empty:
                print("No unsettled bets.")
            else:
                print(f"\n{len(df)} unsettled bets:")
                for _, row in df.iterrows():
                    print(f"  #{row['id']} {row['home_team']} vs {row['away_team']} "
                          f"{row['bet_side'].upper()} @{row['odds_at_bet']:.2f}")


def _settle_from_results():
    """Auto-settle unsettled bets by looking up actual results in the dataset."""
    try:
        from config import DATA_PATH
        df_data = pd.read_csv(DATA_PATH)
        df_data["Date"] = pd.to_datetime(df_data["Date"], format="mixed", dayfirst=True)
    except Exception as e:
        print(f"Could not load results data: {e}")
        return

    unsettled = get_unsettled_bets()
    if unsettled.empty:
        print("No unsettled bets to settle.")
        return

    settled_count = 0
    for _, bet in unsettled.iterrows():
        # Find matching result
        mask = (
            (df_data["Home_Team"] == bet["home_team"]) &
            (df_data["Away_Team"] == bet["away_team"])
        )
        matches = df_data[mask].sort_values("Date", ascending=False)
        if matches.empty:
            continue

        # Use the most recent match
        match = matches.iloc[0]
        total_goals = int(match["TG"])
        result = settle_bet(bet["id"], total_goals)
        if result:
            settled_count += 1
            status = "WON" if result["won"] else "LOST"
            print(f"  #{bet['id']} {bet['home_team']} vs {bet['away_team']} "
                  f"→ {total_goals} goals → {status} ({result['profit_loss']:+.2f})")

    print(f"\nSettled {settled_count}/{len(unsettled)} bets.")
