"""
Bet settlement: pulls completed match results and settles open recommendations.

Uses ESPN public API for match scores (no API key required). Determines
win/loss for each unsettled bet and updates the dashboard database.
"""
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from api.espn_scores import fetch_completed_matches
from api.team_mapping import normalize
from dashboard import LEAGUE_DB_PATHS
from config import ACTIVE_LEAGUE

logger = logging.getLogger(__name__)


def get_finished_matches(
    days_back: int = 7,
    competitions: list[str] | None = None,
) -> list[dict]:
    """Fetch recently completed matches from ESPN.

    Args:
        days_back: How many days back to look for finished matches.
        competitions: Competition codes to fetch. Defaults to ['PL', 'ELC']
                      (Premier League + Championship).

    Returns:
        List of dicts with home_team, away_team, home_goals, away_goals,
        total_goals, date, competition.
    """
    if competitions is None:
        competitions = ["PL", "ELC"]

    results = fetch_completed_matches(
        days_back=days_back, competitions=competitions)

    # Add btts field for settlement logic
    for r in results:
        r["btts"] = r["home_goals"] > 0 and r["away_goals"] > 0

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

    if market.startswith("ou"):
        # Parse goal line from market code: ou25 -> 2.5, ou15 -> 1.5, ou35 -> 3.5
        try:
            line = int(market[2:]) / 10.0
        except (ValueError, IndexError):
            logger.warning("Failed to parse goal line from market '%s', defaulting to 2.5", market)
            line = 2.5
        actual = "over" if total_goals > line else "under"
        won = (side == actual)
        result_str = (f"{home_goals}-{away_goals} ({total_goals} goals, "
                      f"{actual} {line})")
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

    db_path = LEAGUE_DB_PATHS.get(ACTIVE_LEAGUE, LEAGUE_DB_PATHS["PL"])
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        unsettled = conn.execute(
            "SELECT * FROM recommendations WHERE settled=0"
        ).fetchall()

        if not unsettled:
            if verbose:
                print("No unsettled bets to process.")
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
    finally:
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


def settle_predictions(days_back: int = 7, verbose: bool = True) -> dict:
    """Settle all open predictions against actual match results.

    Unlike settle_bets (which settles recommended bets with stakes),
    this settles ALL positive-edge predictions regardless of whether
    they were bet on. Used for model accuracy tracking.

    Args:
        days_back: How many days back to fetch results.
        verbose: Print progress to stdout.

    Returns:
        Dict with settlement summary: settled, won, lost.
    """
    finished = get_finished_matches(days_back=days_back)
    if not finished:
        if verbose:
            print("No finished matches found to settle predictions against.")
        return {"settled": 0, "won": 0, "lost": 0}

    result_lookup = {}
    for m in finished:
        key = (m["home_team"], m["away_team"])
        result_lookup[key] = m

    settled_count = 0
    won_count = 0
    lost_count = 0
    now = datetime.now().isoformat()

    for league, db_path in LEAGUE_DB_PATHS.items():
        if not os.path.exists(db_path):
            continue

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Check if predictions table exists
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
            ).fetchone()
            if not tables:
                continue

            unsettled = conn.execute(
                "SELECT * FROM predictions WHERE settled=0"
            ).fetchall()

            for pred in unsettled:
                key = (pred["home_team"], pred["away_team"])
                result = result_lookup.get(key)
                if result is None:
                    continue

                won, result_str = _determine_outcome(
                    pred["market"], pred["side"],
                    result["home_goals"], result["away_goals"],
                )

                conn.execute(
                    """UPDATE predictions
                       SET settled=1, won=?, actual_result=?, settled_at=?
                       WHERE id=?""",
                    (int(won), result_str, now, pred["id"]),
                )

                settled_count += 1
                if won:
                    won_count += 1
                else:
                    lost_count += 1

                if verbose:
                    status_str = "WON" if won else "LOST"
                    taken = " [TAKEN]" if pred["taken"] else ""
                    edge_str = (f"{pred['edge_pct']:.1f}%"
                                if pred['edge_pct'] is not None else "N/A")
                    print(f"  [{status_str}]{taken} {pred['home_team']} v "
                          f"{pred['away_team']} | "
                          f"{pred['market'].upper()} {pred['side']} | "
                          f"Edge: {edge_str} | {result_str}")

            conn.commit()
        finally:
            conn.close()

    summary = {
        "settled": settled_count,
        "won": won_count,
        "lost": lost_count,
    }

    if verbose:
        print(f"\nPrediction settlement: {settled_count} predictions settled")
        if settled_count > 0:
            print(f"  Won: {won_count} | Lost: {lost_count} | "
                  f"Hit rate: {won_count/settled_count:.1%}")

    return summary


if __name__ == "__main__":
    print("=== Bet Settlement ===\n")
    settle_bets(days_back=7, verbose=True)
    print("\n=== Prediction Settlement ===\n")
    settle_predictions(days_back=7, verbose=True)
