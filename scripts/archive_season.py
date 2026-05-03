"""
Season-end archiver.

Snapshots a finished league's dashboard SQLite into `data/archive/<season>-<league>/`,
exports the same data as CSVs, copies the model artifacts that produced the
predictions, and auto-generates a markdown season report. Optionally MOVES the
season's rows out of the live DB so the dashboard stays fast across years.

Usage:
    # Dry-run on EFL — produces archive without touching live DB
    python scripts/archive_season.py --league=EFL --season=2025-26 \\
        --start=2025-08-08 --end=2026-05-02

    # Actually move data out of live DB after dry-run looks correct
    python scripts/archive_season.py --league=EFL --season=2025-26 \\
        --start=2025-08-08 --end=2026-05-02 --confirm-delete

Default behaviour is DRY-RUN. The --confirm-delete flag is the explicit
opt-in to mutate the live DB. A timestamped .bak is taken before any DELETE.
"""
import argparse
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project paths — script lives at scripts/, project root is one up
ROOT = Path(__file__).resolve().parent.parent
LIVE_DBS = {
    "PL": ROOT / "data" / "dashboard.db",
    "EFL": ROOT / "data" / "dashboard_efl.db",
}
MODEL_FILES = {
    "PL": [
        ROOT / "models" / "pl_pipeline_cache.pkl",
        ROOT / "models" / "pl_trained_state.pkl",
    ],
    "EFL": [
        ROOT / "models" / "championship" / "efl_pipeline_cache.pkl",
        ROOT / "models" / "championship" / "efl_trained_state.pkl",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--league", required=True, choices=["PL", "EFL"])
    p.add_argument("--season", required=True,
                   help="Season label, e.g. 2025-26")
    p.add_argument("--start", required=True,
                   help="Season start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end", required=True,
                   help="Season end date (YYYY-MM-DD), inclusive")
    p.add_argument("--confirm-delete", action="store_true",
                   help="Move (not copy) — DELETE rows from live DB after archive. "
                        "Default is dry-run: archive only.")
    return p.parse_args()


def snapshot_db(src: Path, dst: Path) -> None:
    """Use sqlite3 backup API for a consistent snapshot (handles in-flight writes)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        src_conn.close()
        dst_conn.close()


def export_csvs(snapshot_db_path: Path, archive_dir: Path,
                start: str, end: str) -> dict:
    """Export the 3 in-season tables to CSV. Returns row counts."""
    conn = sqlite3.connect(str(snapshot_db_path))
    counts = {}
    try:
        for table, where in [
            ("predictions", f"WHERE kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"),
            ("recommendations", f"WHERE kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"),
            ("bankroll", f"WHERE timestamp >= '{start}' AND timestamp <= '{end}T23:59:59'"),
        ]:
            df = pd.read_sql_query(f"SELECT * FROM {table} {where}", conn)
            df.to_csv(archive_dir / f"{table}.csv", index=False)
            counts[table] = len(df)
            logger.info("Exported %d rows from %s", len(df), table)
    finally:
        conn.close()
    return counts


def copy_model_artifacts(league: str, archive_dir: Path) -> int:
    """Copy the league's pickled model files into the archive. Returns count copied."""
    n = 0
    for src in MODEL_FILES.get(league, []):
        if src.exists():
            shutil.copy2(src, archive_dir / src.name)
            n += 1
            logger.info("Copied model artifact %s", src.name)
        else:
            logger.warning("Model artifact missing: %s", src)
    return n


def render_season_report(snapshot_db_path: Path, archive_dir: Path,
                         league: str, season: str,
                         start: str, end: str) -> Path:
    """Generate season_report.md from the archived snapshot.

    Pure summary — sources its numbers from the snapshot, not the live DB.
    """
    conn = sqlite3.connect(str(snapshot_db_path))
    try:
        date_filter = f"kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"
        preds = pd.read_sql_query(
            f"SELECT * FROM predictions WHERE {date_filter}", conn)
        recs = pd.read_sql_query(
            f"SELECT * FROM recommendations WHERE {date_filter}", conn)
        bankroll = pd.read_sql_query(
            f"SELECT * FROM bankroll WHERE timestamp >= '{start}' "
            f"AND timestamp <= '{end}T23:59:59' ORDER BY timestamp", conn)
        logged = pd.read_sql_query("SELECT COUNT(*) AS n FROM logged_bets", conn)
        n_logged = int(logged["n"].iloc[0])
    finally:
        conn.close()

    # Headline numbers
    n_preds_total = len(preds)
    settled_preds = preds[preds["settled"] == 1] if "settled" in preds.columns else pd.DataFrame()
    n_preds_settled = len(settled_preds)
    pred_hit_rate = settled_preds["won"].mean() if n_preds_settled > 0 else None

    n_recs_total = len(recs)
    settled_recs = recs[recs["settled"] == 1] if "settled" in recs.columns else pd.DataFrame()
    n_recs_settled = len(settled_recs)
    rec_hit_rate = settled_recs["won"].mean() if n_recs_settled > 0 else None
    rec_total_staked = settled_recs["stake_pct"].sum() if n_recs_settled > 0 else 0
    rec_total_profit = settled_recs["profit_pct"].sum() if n_recs_settled > 0 else 0
    rec_roi = rec_total_profit / rec_total_staked if rec_total_staked > 0 else 0

    # Drawdown
    max_dd = 0.0
    if not bankroll.empty and "balance" in bankroll.columns:
        running_peak = bankroll["balance"].cummax()
        drawdown = (running_peak - bankroll["balance"]) / running_peak.replace(0, 1) * 100
        max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    # By-market breakdown (settled recommendations)
    by_market_lines = []
    if n_recs_settled > 0 and "market" in settled_recs.columns:
        for mkt in sorted(settled_recs["market"].unique()):
            sub = settled_recs[settled_recs["market"] == mkt]
            staked = sub["stake_pct"].sum()
            profit = sub["profit_pct"].sum()
            roi = profit / staked if staked > 0 else 0
            by_market_lines.append(
                f"| {mkt} | {len(sub)} | {sub['won'].mean():.1%} | {roi:+.1%} |"
            )

    # Calibration: 3 edge buckets
    cal_lines = []
    if n_recs_settled > 0 and "edge" in settled_recs.columns:
        sr = settled_recs.copy()
        sr["edge_pct"] = pd.to_numeric(sr["edge"], errors="coerce") * 100
        bins = [0, 2, 4, 100]
        labels = ["0–2%", "2–4%", "4%+"]
        sr["bucket"] = pd.cut(sr["edge_pct"], bins=bins, labels=labels, right=False)
        for label in labels:
            b = sr[sr["bucket"] == label]
            if len(b) == 0:
                continue
            actual = b["won"].mean()
            staked = b["stake_pct"].sum()
            roi = b["profit_pct"].sum() / staked if staked > 0 else 0
            cal_lines.append(f"| {label} | {len(b)} | {actual:.1%} | {roi:+.1%} |")

    # Compose markdown
    lines = [
        f"# Season Report — {season} {league}",
        "",
        f"**Window:** {start} → {end}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        "",
    ]
    if n_logged == 0:
        lines += [
            "> **Disclaimer:** No live bets were logged for this season "
            "(`logged_bets = 0`). Numbers below are from the recommendation "
            "engine running in paper-trading mode — they reflect what *would* "
            "have been bet, not what was actually staked.",
            "",
        ]

    lines += [
        "## Headline numbers",
        "",
        f"- **Predictions made**: {n_preds_total} total, {n_preds_settled} settled",
        f"- **Model hit rate** (settled predictions): "
        f"{pred_hit_rate:.1%}" if pred_hit_rate is not None else "- Model hit rate: —",
        f"- **Recommendations**: {n_recs_total} total, {n_recs_settled} settled",
        f"- **Recommendation hit rate**: "
        f"{rec_hit_rate:.1%}" if rec_hit_rate is not None else "- Recommendation hit rate: —",
        f"- **Recommendation ROI**: {rec_roi:+.2%} (per £ staked)",
        f"- **Max drawdown**: -{max_dd:.2f}%",
        "",
    ]

    if by_market_lines:
        lines += [
            "## By market (settled recommendations)",
            "",
            "| Market | Bets | Win rate | ROI |",
            "|---|---|---|---|",
            *by_market_lines,
            "",
        ]

    if cal_lines:
        lines += [
            "## Calibration by edge bucket",
            "",
            "| Edge | Bets | Actual win rate | ROI |",
            "|---|---|---|---|",
            *cal_lines,
            "",
        ]

    lines += [
        "## Provenance",
        "",
        "- DB snapshot: `dashboard.db` in this directory (full SQLite, queryable)",
        "- CSV exports: `predictions.csv`, `recommendations.csv`, `bankroll.csv`",
        "- Model artifacts: pickled trained state + pipeline cache as of season-end",
        "- This report is auto-generated by `scripts/archive_season.py`. Re-run "
        "the same command to regenerate.",
        "",
    ]

    out = archive_dir / "season_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote season report: %s", out)
    return out


def delete_from_live(live_db: Path, start: str, end: str) -> dict:
    """MOVE-step: DELETE the season's rows from the live DB.

    Takes a .bak of the DB first as a safety net. Only called when
    --confirm-delete is passed. Returns counts of rows deleted per table.
    """
    bak = live_db.with_suffix(
        live_db.suffix + f".pre-archive.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(live_db, bak)
    logger.info("Pre-delete backup: %s", bak.name)

    conn = sqlite3.connect(str(live_db))
    deleted = {}
    try:
        for table, where in [
            ("predictions", f"kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"),
            ("recommendations", f"kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"),
            ("match_analysis", f"kickoff >= '{start}' AND kickoff <= '{end}T23:59:59'"),
            ("bankroll", f"timestamp >= '{start}' AND timestamp <= '{end}T23:59:59'"),
        ]:
            try:
                cur = conn.execute(f"DELETE FROM {table} WHERE {where}")
                deleted[table] = cur.rowcount
            except sqlite3.OperationalError as exc:
                logger.warning("Skipping %s: %s", table, exc)
                deleted[table] = 0
        conn.commit()
    finally:
        conn.close()
    return deleted


def main() -> int:
    args = parse_args()
    live_db = LIVE_DBS[args.league]
    if not live_db.exists():
        logger.error("Live DB not found: %s", live_db)
        return 1

    archive_dir = ROOT / "data" / "archive" / f"{args.season}-{args.league}"
    if archive_dir.exists():
        logger.error("Archive dir already exists — refusing to overwrite: %s",
                     archive_dir)
        return 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    logger.info("=== Archiving %s season %s ===", args.league, args.season)
    logger.info("Window: %s → %s", args.start, args.end)
    logger.info("Live DB: %s", live_db)
    logger.info("Archive: %s", archive_dir)

    snapshot_path = archive_dir / "dashboard.db"
    snapshot_db(live_db, snapshot_path)
    logger.info("DB snapshot taken")

    counts = export_csvs(snapshot_path, archive_dir, args.start, args.end)
    n_models = copy_model_artifacts(args.league, archive_dir)
    render_season_report(snapshot_path, archive_dir, args.league, args.season,
                         args.start, args.end)

    logger.info("=== Archive summary ===")
    for table, n in counts.items():
        logger.info("  %s: %d rows", table, n)
    logger.info("  model artifacts: %d files", n_models)

    if args.confirm_delete:
        logger.info("\n--confirm-delete is set → deleting season rows from live DB")
        deleted = delete_from_live(live_db, args.start, args.end)
        logger.info("=== Live DB delete summary ===")
        for table, n in deleted.items():
            logger.info("  %s: %d rows deleted", table, n)
    else:
        logger.info("\nDRY RUN — live DB untouched. To MOVE the data, re-run "
                    "with --confirm-delete after verifying %s", archive_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
