"""
Bet settlement: pulls completed match results and settles open recommendations.

Uses football-data.org API for match scores. Determines win/loss for each
unsettled bet and updates the dashboard database.
"""
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from api.football_data import _get
from api.team_mapping import normalize
from dashboard import DB_PATH

logger = logging.getLogger(__name__)


def get_finished_matches(days_back: int = 7) -> list[dict]:
    """Fetch recently completed PL matches from football-data.org.

    Args:
        days_back: How many days back to look for finished matches.

    Returns:
        List of dicts with home_team, away_team, home_goals, away_goals, date.
    """
    try:
        data = _get("/competitions/PL/matches?status=FINISHED")
        matches = data.get("matches", [])
    except Exception as e:
        logger.error(f"Failed to fetch finished matches: {e}")
        return []

    cutoff = datetime.utcnow() - timedelta(days=days_back)
    results = []

    for m in matches:
        match_date = m.get("utcDate", "")
        try:
            dt = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
            if dt.replace(tzinfo=None) < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        home_goals = m.get("score", {}).get("fullTime", {}).get("home")
        away_goals = m.get("score", {}).get("fullTime", {}).get("away")

        if home_goals is None or away_goals is None:
            continue

        results.append({
            "home_team": normalize(m["homeTeam"]["name"]),
            "away_team": normalize(m["awayTeam"]["name"]),
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "date": match_date,
            "total_goals": int(home_goals) + int(away_goals),
            "btts": int(home_goals) > 0 and int(away_goals) > 0,
        })

    logger.info(f"Fetched {len(results)} finished matches (last {days_back} days)")
    return results


def _determine_outcome(
    market: str,
    side: str,
    home_goals: int,
    away_goals: int,
) -> tuple[bool, str]:
    """Determine if a bet won based on match result.

    Args:
        market: 'ou25' or 'btts'
        side: 'over'/'under' or 'yes'/'no'
        home_goals: Final home score
        away_goals: Final away score

    Returns:
        Tuple of (won: bool, actual_result: str)
    """
    total_goals = home_goals + away_goals
    btts = home_goals > 0 and away_goals > 0

    if market == "ou25":
        actual = "over" if total_goals > 2.5 else "under"
        won = (side == actual)
        result_str = f"{home_goals}-{away_goals} ({total_goals} goals, {actual})"
    elif market == "btts":
        actual = "yes" if btts else "no"
        won = (side == actual)
        result_str = f"{home_goals}-{away_goals} (BTTS {'Yes' if btts else 'No'})"
    else:
        won = False
        result_str = f"{home_goals}-{away_goals} (unknown market)"

    return won, result_str


def settle_bets(days_back: int = 7, verbose: bool = True) -> dict:
    """Settle all open recommendations against actual match results.

    Args:
        days_back: How many days back to fetch results.
        verbose: Print progress to stdout.

    Returns:
        Dict with settlement summary: settled, won, lost, profit.
    """
    finished = get_finished_matches(days_back=days_back)
    if not finished:
        if verbose:
            print("No finished matches found to settle against.")
        return {"settled": 0, "won": 0, "lost": 0, "profit": 0.0}

    # Build lookup: (home_team, away_team) -> result
    result_lookup = {}
    for m in finished:
        key = (m["home_team"], m["away_team"])
        result_lookup[key] = m

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    unsettled = conn.execute(
        "SELECT * FROM recommendations WHERE settled=0"
    ).fetchall()

    if not unsettled:
        if verbose:
            print("No unsettled bets to process.")
        conn.close()
        return {"settled": 0, "won": 0, "lost": 0, "profit": 0.0}

    settled_count = 0
    won_count = 0
    lost_count = 0
    total_profit = 0.0
    now = datetime.now().isoformat()

    for bet in unsettled:
        key = (bet["home_team"], bet["away_team"])
        result = result_lookup.get(key)

        if result is None:
            # Match not yet played or not found
            continue

        won, result_str = _determine_outcome(
            bet["market"], bet["side"],
            result["home_goals"], result["away_goals"],
        )

        stake = bet["stake_pct"] or 0
        odds = bet["odds"] or 0
        profit = stake * (odds - 1) if won else -stake

        conn.execute(
            """UPDATE recommendations
               SET settled=1, won=?, profit_pct=?, actual_result=?, settled_at=?
               WHERE id=?""",
            (int(won), profit, result_str, now, bet["id"]),
        )

        settled_count += 1
        if won:
            won_count += 1
        else:
            lost_count += 1
        total_profit += profit

        if verbose:
            status = "WON" if won else "LOST"
            print(f"  [{status}] {bet['home_team']} v {bet['away_team']} | "
                  f"{bet['market'].upper()} {bet['side']} @ {odds:.2f} | "
                  f"{result_str} | P/L: {profit:+.4f}")

    conn.commit()

    # Update bankroll table
    if settled_count > 0:
        last_balance = conn.execute(
            "SELECT balance FROM bankroll ORDER BY id DESC LIMIT 1"
        ).fetchone()
        current_balance = (last_balance["balance"] if last_balance else 1.0) + total_profit
        conn.execute(
            "INSERT INTO bankroll (timestamp, balance, event) VALUES (?, ?, ?)",
            (now, current_balance, f"Settled {settled_count} bets ({won_count}W/{lost_count}L)"),
        )
        conn.commit()

    conn.close()

    summary = {
        "settled": settled_count,
        "won": won_count,
        "lost": lost_count,
        "profit": total_profit,
    }

    if verbose:
        print(f"\nSettlement complete: {settled_count} bets settled")
        print(f"  Won: {won_count} | Lost: {lost_count}")
        print(f"  Profit: {total_profit:+.4f} units")

    return summary


if __name__ == "__main__":
    print("=== Bet Settlement ===\n")
    settle_bets(days_back=7, verbose=True)
