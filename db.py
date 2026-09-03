"""
Database layer for the betting system.

Owns all SQLite schema creation, connection management, and CRUD operations
for recommendations, predictions, match analysis, logged bets, and bankroll.
Every module that reads or writes betting data goes through this module —
dashboard.py, scheduler.py, settlement.py, predict.py, edge_analytics.py.

Each league has its own SQLite database; the league key ("PL", "EFL") selects
which file to open.  Tables are created lazily on first connection via
``get_db()``.
"""
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import pandas as pd

from league_config import LEAGUES

logger = logging.getLogger(__name__)

# ── League DB paths ──────────────────────────────────────────────────────────
LEAGUE_DB_PATHS: dict[str, str] = {
    key: cfg["db_path"] for key, cfg in LEAGUES.items()
}
LEAGUE_DISPLAY_NAMES: dict[str, str] = {
    key: cfg["name"] for key, cfg in LEAGUES.items()
}


# ═════════════════════════════════════════════════════════════════════════════
# Connection management
# ═════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_db(league: str = "PL") -> Generator[sqlite3.Connection, None, None]:
    """Get SQLite connection for the specified league, creating tables if needed.

    Args:
        league: League key ("PL", "EFL", etc.).

    Yields:
        sqlite3.Connection with row_factory = sqlite3.Row and WAL mode.

    Raises:
        KeyError: If *league* is not in LEAGUE_DB_PATHS.
    """
    db_path = LEAGUE_DB_PATHS.get(league)
    if db_path is None:
        raise KeyError(
            f"Unknown league '{league}'. Available: {list(LEAGUE_DB_PATHS)}"
        )

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")

        # ── recommendations (legacy — kept for settlement compatibility) ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                model_prob REAL,
                blended_prob REAL,
                fair_prob REAL,
                odds REAL NOT NULL,
                edge REAL,
                ev REAL,
                stake_pct REAL,
                confidence TEXT,
                best_bookmaker TEXT,
                n_books INTEGER,
                n_agree INTEGER,
                per_model_json TEXT,
                settled INTEGER DEFAULT 0,
                won INTEGER,
                profit_pct REAL,
                actual_result TEXT,
                settled_at TEXT
            )
        """)

        # ── match_analysis: ALL fixture-market combos from latest scan ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS match_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff TEXT,
                matchday TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                best_odds REAL,
                best_bookmaker TEXT,
                model_prob REAL,
                fair_odds REAL,
                edge_pct REAL,
                confidence TEXT,
                n_books INTEGER,
                per_model_json TEXT,
                bookmaker_odds_json TEXT
            )
        """)

        # Migrate: add bookmaker_odds_json if missing (existing DBs)
        try:
            conn.execute("SELECT bookmaker_odds_json FROM match_analysis LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE match_analysis ADD COLUMN bookmaker_odds_json TEXT"
            )

        # Migrate: add edge_source column (pinnacle vs devig)
        try:
            conn.execute("SELECT edge_source FROM match_analysis LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE match_analysis ADD COLUMN edge_source TEXT"
            )

        # Migrate: outcomes for every evaluated market, not just the ones we
        # liked. `predictions` stores a row only when edge_pct > 0 and
        # `recommendations` only when the gate passed, so both are selected
        # samples — and conditioning on the model's own positive edge inflates
        # its apparent confidence by ~6.3 points whatever its real calibration
        # (measured walk-forward over the OOF caches). Settling this table is
        # what makes an unbiased calibration measurement possible at all.
        for _col, _type in (("settled", "INTEGER DEFAULT 0"),
                            ("won", "INTEGER"),
                            ("actual_result", "TEXT"),
                            ("settled_at", "TEXT")):
            try:
                conn.execute(f"SELECT {_col} FROM match_analysis LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(
                    f"ALTER TABLE match_analysis ADD COLUMN {_col} {_type}"
                )

        # ── predictions: every positive-edge prediction, settled automatically ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                model_prob REAL,
                fair_odds REAL,
                edge_pct REAL,
                best_odds REAL,
                best_bookmaker TEXT,
                confidence TEXT,
                bookmaker_odds_json TEXT,
                taken INTEGER DEFAULT 0,
                settled INTEGER DEFAULT 0,
                won INTEGER,
                actual_result TEXT,
                settled_at TEXT
            )
        """)

        # ── logged_bets: manually logged bets ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logged_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                kickoff TEXT,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                odds REAL NOT NULL,
                stake REAL NOT NULL,
                bookmaker TEXT,
                potential_return REAL,
                notes TEXT,
                settled INTEGER DEFAULT 0,
                won INTEGER,
                profit REAL,
                actual_result TEXT,
                settled_at TEXT
            )
        """)

        # Migrate: add CLV tracking + edge source columns to logged_bets
        for col, col_type in [
            ("closing_odds", "REAL"),
            ("closing_pinnacle", "REAL"),
            ("closing_timestamp", "TEXT"),
            ("model_prob", "REAL"),
            ("edge_pct", "REAL"),
            ("edge_source", "TEXT"),
        ]:
            try:
                conn.execute(f"SELECT {col} FROM logged_bets LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute(
                    f"ALTER TABLE logged_bets ADD COLUMN {col} {col_type}"
                )

        # ── bankroll ──
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bankroll (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                balance REAL NOT NULL,
                event TEXT
            )
        """)

        conn.commit()
        yield conn
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# Match analysis CRUD
# ═════════════════════════════════════════════════════════════════════════════

def save_match_analysis(rows: list[dict], league: str = "PL") -> int:
    """Upsert match_analysis with fresh scan data, preserving model_prob.

    Only replaces rows for fixtures covered by the new scan. Existing rows
    for fixtures NOT in the scan are kept untouched — this prevents a
    depleted API cache from wiping out good data.

    When new rows have model_prob=None but existing rows have a value,
    the existing model_prob (and derived edge/confidence) are carried
    forward.

    Args:
        rows: List of dicts, one per fixture-market-side.
        league: League key.

    Returns:
        Number of rows upserted.
    """
    if not rows:
        return 0

    with get_db(league) as conn:
        # Read existing data for carry-forward
        existing: dict[tuple, dict] = {}
        try:
            cursor = conn.execute(
                "SELECT home_team, away_team, market, side, "
                "model_prob, edge_pct, confidence, per_model_json, "
                "settled, won, actual_result, settled_at "
                "FROM match_analysis"
            )
            for row in cursor.fetchall():
                key = (row[0], row[1], row[2], row[3])
                existing[key] = {
                    "model_prob": row[4],
                    "edge_pct": row[5],
                    "confidence": row[6],
                    "per_model_json": row[7],
                    # A rescan must never unsettle a fixture. This function
                    # deletes and re-inserts per fixture, so without carrying
                    # the outcome forward a settled row would come back blank
                    # — silently, and the calibration series would just get
                    # shorter. Same reasoning as the model_prob rescue below.
                    "settled": row[8],
                    "won": row[9],
                    "actual_result": row[10],
                    "settled_at": row[11],
                }
        except Exception:
            pass

        # Only delete rows for fixtures that this scan covers
        new_fixtures = {(r["home_team"], r["away_team"]) for r in rows}
        for home, away in new_fixtures:
            conn.execute(
                "DELETE FROM match_analysis WHERE home_team=? AND away_team=?",
                (home, away),
            )

        now = datetime.now().isoformat()
        for r in rows:
            # Outcome carried forward for every row, unconditionally — the
            # delete-and-reinsert above would otherwise blank a settled
            # fixture without raising anything.
            settled_prev = existing.get(
                (r["home_team"], r["away_team"], r["market"], r["side"])) or {}
            model_p = r.get("model_prob")
            edge = r.get("edge_pct")
            conf = r.get("confidence")
            per_model = r.get("per_model_probs") or {}

            # Carry the per-model breakdown forward on its own terms, not as a
            # side effect of the model_prob rescue below. Two paths write here:
            # the predictor's own _match_analysis (scan.py:510) supplies
            # per_model_probs for every evaluated line, while the rows scan.py
            # rebuilds (scan.py:817) supply model_prob but no breakdown. Since
            # this function deletes and re-inserts per fixture, the rebuild used
            # to overwrite the predictor's rows with per_model_json = "{}" —
            # and the `if model_p is None` guard never fired to stop it,
            # because the rebuild always has a model_prob. That is why the
            # Match Centre's AGREE column read "—" on every row.
            if not per_model:
                _key = (r["home_team"], r["away_team"], r["market"], r["side"])
                _prev = existing.get(_key)
                if _prev:
                    try:
                        per_model = json.loads(
                            _prev.get("per_model_json") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        per_model = {}

            # If this row has no model_prob, carry forward from existing data
            if model_p is None:
                key = (r["home_team"], r["away_team"], r["market"], r["side"])
                prev = existing.get(key)
                if prev and prev["model_prob"] is not None:
                    model_p = prev["model_prob"]
                    # Recalculate edge with preserved model_prob and fresh odds
                    fair_odds = r.get("fair_odds")
                    if fair_odds and fair_odds > 0:
                        fair_p = 1.0 / fair_odds
                        edge = (model_p - fair_p) * 100
                        if edge > 4:
                            conf = "high"
                        elif edge > 2.5:
                            conf = "medium"
                        elif edge > 0:
                            conf = "low"
                        else:
                            conf = "negative"
                    else:
                        edge = prev["edge_pct"]
                        conf = prev["confidence"]
                    # Only when this row still has nothing. The block above
                    # already carried the breakdown forward on its own terms;
                    # overwriting unconditionally here would discard a good
                    # breakdown the row supplied whenever the stored value was
                    # "{}" — the same erasure the carry-forward exists to stop,
                    # arriving from the other direction. Note `or "{}"`, not
                    # `.get(k, "{}")`: the key exists with a NULL value on rows
                    # written before per_model_json was populated.
                    if not per_model:
                        try:
                            per_model = json.loads(
                                prev.get("per_model_json") or "{}"
                            )
                        except (json.JSONDecodeError, TypeError):
                            pass

            conn.execute(
                """INSERT INTO match_analysis
                   (scanned_at, home_team, away_team, kickoff, matchday,
                    market, side, best_odds, best_bookmaker,
                    model_prob, fair_odds, edge_pct, confidence,
                    n_books, per_model_json, bookmaker_odds_json,
                    edge_source,
                    settled, won, actual_result, settled_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?)""",
                (
                    now,
                    r["home_team"], r["away_team"],
                    r.get("kickoff", ""), r.get("matchday", ""),
                    r["market"], r["side"],
                    r.get("best_odds"), r.get("best_bookmaker"),
                    model_p, r.get("fair_odds"),
                    edge, conf,
                    r.get("n_books"),
                    json.dumps(per_model),
                    json.dumps(r.get("bookmaker_odds", {})),
                    r.get("edge_source"),
                    # Settlement is never re-derived from a scan; it is only
                    # ever carried forward. A fixture that has been settled
                    # stays settled however many times it is rescanned.
                    #
                    # Looked up on its own rather than reusing `prev` above:
                    # that name is bound only when the row arrives without a
                    # model_prob, so reading it here works until the first
                    # scan that carries one — which is the normal case.
                    settled_prev.get("settled") or 0,
                    settled_prev.get("won"),
                    settled_prev.get("actual_result"),
                    settled_prev.get("settled_at"),
                ),
            )
        conn.commit()
        return len(rows)


def get_match_analysis(league: str = "PL") -> pd.DataFrame:
    """Get current match analysis for a league.

    Filters out:
      - Fixtures from previous days (removed at end of day, not at kickoff,
        so today's in-play/completed games remain visible).
      - Fixtures more than 7 days ahead (predictions degrade with distance
        due to injuries, form changes, and line movements).
    """
    with get_db(league) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM match_analysis ORDER BY kickoff, home_team, market, side",
            conn,
        )
    if df.empty:
        return df

    try:
        kickoff_dt = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
        now_utc = pd.Timestamp.now(tz="UTC")
        today_start = now_utc.normalize()  # midnight UTC today
        max_lookahead = now_utc + pd.Timedelta(days=7)

        # Keep: unparseable dates OR (today or later AND within 7-day window)
        keep_mask = (
            kickoff_dt.isna()
            | ((kickoff_dt >= today_start) & (kickoff_dt <= max_lookahead))
        )
        df = df[keep_mask].reset_index(drop=True)
    except Exception:
        pass  # If parsing fails, return all rows as fallback

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Prediction tracking
# ═════════════════════════════════════════════════════════════════════════════

def log_predictions(rows: list[dict], league: str = "PL") -> int:
    """Log positive-edge predictions to the predictions table.

    Only logs predictions with edge > 0 that haven't been logged yet
    for the same fixture+market+side (avoids duplicates on re-scan).

    Args:
        rows: List of match analysis dicts (from scan or predictor).
        league: League key.

    Returns:
        Number of new predictions logged.
    """
    logged = 0
    with get_db(league) as conn:
        for r in rows:
            edge = r.get("edge_pct")
            if edge is None or edge <= 0:
                continue

            # Skip if already logged for this fixture+market+side
            existing = conn.execute(
                """SELECT id FROM predictions
                   WHERE home_team=? AND away_team=? AND market=? AND side=?
                   AND settled=0""",
                (r["home_team"], r["away_team"], r["market"], r["side"]),
            ).fetchone()
            if existing:
                continue

            conn.execute(
                """INSERT INTO predictions
                   (created_at, home_team, away_team, kickoff, market, side,
                    model_prob, fair_odds, edge_pct, best_odds, best_bookmaker,
                    confidence, bookmaker_odds_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    r["home_team"], r["away_team"],
                    r.get("kickoff", ""), r["market"], r["side"],
                    r.get("model_prob"), r.get("fair_odds"),
                    edge, r.get("best_odds"), r.get("best_bookmaker"),
                    r.get("confidence", ""),
                    json.dumps(r.get("bookmaker_odds", {})),
                ),
            )
            logged += 1
        conn.commit()
    return logged


def get_predictions(
    league: str = "PL", settled_only: bool = False,
) -> pd.DataFrame:
    """Get predictions from the tracking table.

    Args:
        league: League key.
        settled_only: If True, return only settled predictions.

    Returns:
        DataFrame of predictions.
    """
    where = "WHERE settled=1" if settled_only else ""
    with get_db(league) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM predictions {where} ORDER BY created_at DESC",
            conn,
        )


def toggle_prediction_taken(
    prediction_id: int, taken: bool, league: str = "PL",
) -> None:
    """Mark a prediction as taken or not taken.

    Args:
        prediction_id: ID of the prediction row.
        taken: True if bet was taken, False otherwise.
        league: League key.
    """
    with get_db(league) as conn:
        conn.execute(
            "UPDATE predictions SET taken=? WHERE id=?",
            (int(taken), prediction_id),
        )
        conn.commit()


# ═════════════════════════════════════════════════════════════════════════════
# Logged bets CRUD
# ═════════════════════════════════════════════════════════════════════════════

def save_logged_bet(bet: dict, league: str = "PL") -> int:
    """Save a manually logged bet.

    Args:
        bet: Dict with home_team, away_team, market, side, odds, stake, etc.
        league: League key.

    Returns:
        The new bet's row ID.
    """
    with get_db(league) as conn:
        cursor = conn.execute(
            """INSERT INTO logged_bets
               (created_at, home_team, away_team, kickoff, market, side,
                odds, stake, bookmaker, potential_return, notes,
                model_prob, edge_pct, edge_source)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                bet["home_team"], bet["away_team"],
                bet.get("kickoff", ""), bet["market"], bet["side"],
                bet["odds"], bet["stake"], bet.get("bookmaker", ""),
                bet["stake"] * (bet["odds"] - 1),
                bet.get("notes", ""),
                bet.get("model_prob"),
                bet.get("edge_pct"),
                bet.get("edge_source"),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_open_bets(league: str = "PL") -> pd.DataFrame:
    """Get unsettled logged bets."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM logged_bets WHERE settled=0 ORDER BY created_at DESC",
            conn,
        )


def get_settled_bets(league: str = "PL") -> pd.DataFrame:
    """Get settled logged bets."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM logged_bets WHERE settled=1 ORDER BY settled_at DESC",
            conn,
        )


def get_all_bets(league: str = "PL") -> pd.DataFrame:
    """Get all logged bets."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM logged_bets ORDER BY created_at DESC", conn,
        )


def settle_logged_bet(
    bet_id: int, won: bool, actual_result: str = "", league: str = "PL",
) -> None:
    """Settle a single logged bet.

    Args:
        bet_id: Row ID in logged_bets.
        won: True if the bet won.
        actual_result: Score string (e.g. "2-1 (3 goals, over 2.5)").
        league: League key.
    """
    with get_db(league) as conn:
        row = conn.execute(
            "SELECT odds, stake FROM logged_bets WHERE id=?", (bet_id,)
        ).fetchone()
        if row is None:
            return
        profit = row["stake"] * (row["odds"] - 1) if won else -row["stake"]
        conn.execute(
            """UPDATE logged_bets
               SET settled=1, won=?, profit=?, actual_result=?, settled_at=?
               WHERE id=?""",
            (int(won), profit, actual_result, datetime.now().isoformat(), bet_id),
        )
        conn.commit()


def fetch_closing_odds_for_logged_bets(league: str = "PL") -> int:
    """Fetch current odds for unsettled logged bets and store as closing odds.

    Should be called ~5 minutes before kickoff. Can be called multiple
    times — latest fetch before kickoff wins. Handles O/U 2.5, O/U 1.5,
    and BTTS markets.

    Args:
        league: "PL" or "EFL".

    Returns:
        Number of bets updated with closing odds.
    """
    with get_db(league) as conn:
        unsettled = conn.execute(
            """SELECT id, home_team, away_team, market, side
               FROM logged_bets WHERE settled = 0"""
        ).fetchall()

    if not unsettled:
        return 0

    # Fetch current odds from The-Odds-API
    try:
        from api.odds_api import fetch_epl_odds, get_best_odds
        if league == "EFL":
            from championship_predict import _fetch_champ_odds
            matches = _fetch_champ_odds(force_refresh=True)
        else:
            matches = fetch_epl_odds(force_refresh=True)
    except Exception as e:
        logger.error(f"CLV closing odds fetch failed ({league}): {e}")
        return 0

    if not matches:
        return 0

    # Build odds lookup by normalised team names
    from api.odds_api import get_best_btts_odds
    odds_lookup: dict[tuple[str, str], dict] = {}
    for m in matches:
        h = m["home_team"].lower().replace(" fc", "").strip()
        a = m["away_team"].lower().replace(" fc", "").strip()
        entry: dict = {}

        # O/U totals
        best_ou = get_best_odds(m)
        if best_ou:
            entry["best_over"] = best_ou.get("best_over")
            entry["best_under"] = best_ou.get("best_under")
            entry["pinnacle_over"] = best_ou.get("pinnacle_over")
            entry["pinnacle_under"] = best_ou.get("pinnacle_under")

        # BTTS
        best_btts = get_best_btts_odds(m)
        if best_btts:
            entry["best_yes"] = best_btts.get("best_yes")
            entry["best_no"] = best_btts.get("best_no")
            entry["pinnacle_yes"] = best_btts.get("pinnacle_yes")
            entry["pinnacle_no"] = best_btts.get("pinnacle_no")

        if entry:
            odds_lookup[(h, a)] = entry

    updated = 0
    now_iso = datetime.now().isoformat()

    with get_db(league) as conn:
        for bet in unsettled:
            bet_home = bet["home_team"].lower().replace(" fc", "").strip()
            bet_away = bet["away_team"].lower().replace(" fc", "").strip()
            market = bet["market"]
            side = bet["side"]

            # Find matching fixture
            matched = None
            for (h, a), data in odds_lookup.items():
                h_words = set(h.split())
                a_words = set(a.split())
                bh_words = set(bet_home.split())
                ba_words = set(bet_away.split())
                if (len(bh_words & h_words) >= 1 and
                        len(ba_words & a_words) >= 1):
                    matched = data
                    break

            if matched is None:
                continue

            # Get closing odds for the specific market/side
            if market in ("ou25", "ou15", "ou35", "ou45"):
                closing = matched.get(f"best_{side}")
                pin = matched.get(f"pinnacle_{side}")
            elif market == "btts":
                closing = matched.get(f"best_{side}")
                pin = matched.get(f"pinnacle_{side}")
            else:
                continue

            if closing is None:
                continue

            conn.execute(
                """UPDATE logged_bets
                   SET closing_odds = ?, closing_pinnacle = ?,
                       closing_timestamp = ?
                   WHERE id = ?""",
                (closing, pin, now_iso, bet["id"]),
            )
            updated += 1

        conn.commit()

    logger.info(
        f"CLV: Updated closing odds for {updated}/{len(unsettled)} "
        f"{league} bets"
    )
    return updated


def calculate_logged_bet_clv(league: str = "PL") -> dict:
    """Calculate CLV metrics across settled logged bets with closing odds.

    CLV = (odds_at_bet / closing_odds) - 1
    Positive CLV means you consistently got better odds than the closing line.

    Args:
        league: "PL" or "EFL".

    Returns:
        Dict with CLV summary statistics.
    """
    with get_db(league) as conn:
        df = pd.read_sql_query(
            """SELECT * FROM logged_bets
               WHERE settled = 1 AND closing_odds IS NOT NULL
                     AND closing_odds > 1""",
            conn,
        )

    if df.empty:
        return {"n_bets": 0, "message": "No settled bets with closing odds yet."}

    # CLV: how much better your odds were vs closing line
    df["clv"] = (df["odds"] / df["closing_odds"]) - 1
    df["clv_pct"] = df["clv"] * 100
    df["beat_close"] = df["odds"] > df["closing_odds"]

    # Pinnacle CLV (sharper benchmark)
    pin_df = df[df["closing_pinnacle"].notna() & (df["closing_pinnacle"] > 1)]
    pin_clv = None
    pin_beat_rate = None
    if not pin_df.empty:
        pin_df = pin_df.copy()
        pin_df["pin_clv"] = (pin_df["odds"] / pin_df["closing_pinnacle"]) - 1
        pin_clv = float(pin_df["pin_clv"].mean() * 100)
        pin_beat_rate = float(
            pin_df["odds"].gt(pin_df["closing_pinnacle"]).mean() * 100
        )

    # Per-market breakdown
    market_clv = {}
    for mkt in df["market"].unique():
        mkt_df = df[df["market"] == mkt]
        market_clv[mkt] = {
            "n": len(mkt_df),
            "mean_clv_pct": float(mkt_df["clv_pct"].mean()),
            "beat_rate": float(mkt_df["beat_close"].mean() * 100),
        }

    # Per-side breakdown
    side_clv = {}
    for s in df["side"].unique():
        s_df = df[df["side"] == s]
        side_clv[s] = {
            "n": len(s_df),
            "mean_clv_pct": float(s_df["clv_pct"].mean()),
            "beat_rate": float(s_df["beat_close"].mean() * 100),
        }

    total_profit = (
        float(df["profit"].sum()) if df["profit"].notna().any() else 0
    )

    return {
        "n_bets": len(df),
        "mean_clv_pct": float(df["clv_pct"].mean()),
        "median_clv_pct": float(df["clv_pct"].median()),
        "beat_close_rate": float(df["beat_close"].mean() * 100),
        "pinnacle_clv_pct": pin_clv,
        "pinnacle_beat_rate": pin_beat_rate,
        "actual_roi": float(total_profit / df["stake"].sum() * 100)
        if df["stake"].sum() > 0 else None,
        "win_rate": float((df["won"] == 1).mean() * 100)
        if df["won"].notna().any() else None,
        "total_profit": total_profit,
        "total_staked": float(df["stake"].sum()),
        "market_clv": market_clv,
        "side_clv": side_clv,
        # Per-bet data for time series
        "clv_series": df[[
            "created_at", "home_team", "away_team", "market", "side",
            "odds", "closing_odds", "closing_pinnacle", "clv_pct",
            "won", "profit",
        ]].to_dict("records"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Recommendation CRUD (legacy — kept for settlement compatibility)
# ═════════════════════════════════════════════════════════════════════════════

def save_recommendations(recs: list[dict], league: str = "PL") -> int:
    """Save new recommendations to database.

    Args:
        recs: List of recommendation dicts.
        league: League key.

    Returns:
        Number of new recommendations saved (duplicates skipped).
    """
    with get_db(league) as conn:
        saved = 0
        for r in recs:
            existing = conn.execute(
                "SELECT id FROM recommendations WHERE home_team=? AND away_team=? "
                "AND market=? AND side=? AND settled=0",
                (r["home_team"], r["away_team"], r["market"], r["side"]),
            ).fetchone()
            if existing:
                # KO-1h re-scan: refresh the existing unsettled row in place so
                # late-scan price/edge/stake moves are not lost. An update is
                # not a new pick, so it is not counted in the return value.
                conn.execute(
                    """UPDATE recommendations SET
                         kickoff=?, model_prob=?, blended_prob=?, fair_prob=?,
                         odds=?, edge=?, ev=?, stake_pct=?, confidence=?,
                         best_bookmaker=?, n_books=?, n_agree=?, per_model_json=?
                       WHERE id=?""",
                    (
                        r.get("kickoff", ""),
                        r.get("model_prob"), r.get("blended_prob"),
                        r.get("fair_prob"),
                        r["odds"], r.get("edge"), r.get("ev"),
                        r.get("stake_pct"), r.get("confidence"),
                        r.get("best_bookmaker"),
                        r.get("n_books"), r.get("n_agree"),
                        json.dumps(r.get("per_model_probs", {})),
                        existing[0],
                    ),
                )
                continue
            conn.execute(
                """INSERT INTO recommendations
                   (created_at, home_team, away_team, kickoff, market, side,
                    model_prob, blended_prob, fair_prob, odds, edge, ev,
                    stake_pct, confidence, best_bookmaker, n_books, n_agree,
                    per_model_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    r["home_team"], r["away_team"], r.get("kickoff", ""),
                    r["market"], r["side"],
                    r.get("model_prob"), r.get("blended_prob"),
                    r.get("fair_prob"),
                    r["odds"], r.get("edge"), r.get("ev"),
                    r.get("stake_pct"), r.get("confidence"),
                    r.get("best_bookmaker"),
                    r.get("n_books"), r.get("n_agree"),
                    json.dumps(r.get("per_model_probs", {})),
                ),
            )
            saved += 1
        conn.commit()
        return saved


def get_active_recommendations(league: str = "PL") -> pd.DataFrame:
    """Get all unsettled recommendations for a league."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM recommendations WHERE settled=0 ORDER BY kickoff",
            conn,
        )


def get_settled_recommendations(league: str = "PL") -> pd.DataFrame:
    """Get all settled recommendations for a league."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM recommendations WHERE settled=1 "
            "ORDER BY settled_at DESC",
            conn,
        )


def get_all_recommendations(league: str = "PL") -> pd.DataFrame:
    """Get all recommendations for a league."""
    with get_db(league) as conn:
        return pd.read_sql_query(
            "SELECT * FROM recommendations ORDER BY created_at DESC", conn,
        )
