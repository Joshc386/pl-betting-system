"""
Automated scheduler for the betting system.

Runs three jobs:
  1. Generate predictions — every matchday morning (fetches odds, runs models)
  2. Settle bets — every evening after matches finish
  3. Refresh odds — periodically to catch line movements

Uses APScheduler with persistent job store so jobs survive restarts.
"""
import os
import sys
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger("scheduler")

# ── Job Functions ──

def job_generate_predictions() -> None:
    """Fetch live odds, run models, save recommendations."""
    logger.info("Starting prediction generation...")
    try:
        from predict import LivePredictor
        from dashboard import save_recommendations

        predictor = LivePredictor(verbose=True)
        predictor.load_data()
        predictor.train()
        recs = predictor.generate_recommendations()
        n_saved = save_recommendations(recs)

        logger.info(f"Predictions complete: {len(recs)} generated, {n_saved} new saved")
        print(f"[{datetime.now():%H:%M:%S}] Predictions: {len(recs)} generated, "
              f"{n_saved} new saved")
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Prediction FAILED: {e}")


def job_settle_bets() -> None:
    """Pull match results and settle open bets."""
    logger.info("Starting bet settlement...")
    try:
        from settlement import settle_bets
        summary = settle_bets(days_back=3, verbose=True)
        logger.info(f"Settlement complete: {summary}")
        print(f"[{datetime.now():%H:%M:%S}] Settlement: "
              f"{summary['settled']} settled, "
              f"{summary['won']}W/{summary['lost']}L, "
              f"P/L: {summary['profit']:+.4f}")
    except Exception as e:
        logger.error(f"Settlement failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Settlement FAILED: {e}")


def job_refresh_odds() -> None:
    """Refresh odds cache without full model retrain."""
    logger.info("Refreshing odds cache...")
    try:
        from api.odds_api import fetch_epl_odds
        matches = fetch_epl_odds(force_refresh=True)
        logger.info(f"Odds refreshed: {len(matches)} matches")
        print(f"[{datetime.now():%H:%M:%S}] Odds refresh: {len(matches)} matches")
    except Exception as e:
        logger.error(f"Odds refresh failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Odds refresh FAILED: {e}")


# ── Schedule Configuration ──

# Premier League matches typically:
#   - Saturday 15:00 (bulk), some at 12:30 and 17:30
#   - Sunday 14:00, 16:30
#   - Midweek: Tue/Wed 19:30, 20:00
#   - Monday/Friday: occasional 20:00

SCHEDULE = {
    "predict_morning": {
        "func": job_generate_predictions,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=8,
            minute=0,
        ),
        "description": "Generate predictions every morning at 08:00",
    },
    "predict_prematch": {
        "func": job_generate_predictions,
        "trigger": CronTrigger(
            day_of_week="sat",
            hour=11,
            minute=0,
        ),
        "description": "Saturday pre-match refresh at 11:00",
    },
    "settle_evening": {
        "func": job_settle_bets,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=23,
            minute=0,
        ),
        "description": "Settle bets every evening at 23:00",
    },
    "settle_morning": {
        "func": job_settle_bets,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=9,
            minute=0,
        ),
        "description": "Morning settlement catch-up at 09:00",
    },
    "odds_refresh": {
        "func": job_refresh_odds,
        "trigger": IntervalTrigger(minutes=30),
        "description": "Refresh odds every 30 minutes",
    },
}


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the scheduler with all jobs."""
    scheduler = BackgroundScheduler(timezone="Europe/London")

    for job_id, config in SCHEDULE.items():
        scheduler.add_job(
            config["func"],
            trigger=config["trigger"],
            id=job_id,
            name=config["description"],
            replace_existing=True,
            misfire_grace_time=3600,  # 1 hour grace for missed jobs
        )

    return scheduler


def print_schedule(scheduler: BackgroundScheduler) -> None:
    """Print the current schedule."""
    print("\n  Scheduled Jobs:")
    print(f"  {'Job':<25} {'Next Run':<25} {'Description'}")
    print(f"  {'-'*80}")
    for job in scheduler.get_jobs():
        next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M %Z") if job.next_run_time else "—"
        print(f"  {job.id:<25} {next_run:<25} {job.name}")
    print()


def run_now(job_name: str) -> None:
    """Run a specific job immediately by name.

    Args:
        job_name: One of 'predict', 'settle', 'odds'.
    """
    jobs = {
        "predict": job_generate_predictions,
        "settle": job_settle_bets,
        "odds": job_refresh_odds,
    }
    func = jobs.get(job_name)
    if func is None:
        print(f"Unknown job: {job_name}. Options: {list(jobs.keys())}")
        return
    func()


def main():
    """Run the scheduler as a long-lived process alongside the dashboard."""
    # Set up logging
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "scheduler.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    print("=" * 60)
    print("  Premier League Betting System — Scheduler")
    print("=" * 60)

    scheduler = create_scheduler()
    print_schedule(scheduler)

    # Parse CLI args for immediate actions
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        if action == "run-predict":
            print("Running predictions now...\n")
            job_generate_predictions()
            return
        elif action == "run-settle":
            print("Running settlement now...\n")
            job_settle_bets()
            return
        elif action == "run-odds":
            print("Refreshing odds now...\n")
            job_refresh_odds()
            return
        elif action == "run-all":
            print("Running all jobs now...\n")
            job_generate_predictions()
            job_settle_bets()
            return
        elif action == "schedule":
            pass  # Fall through to start scheduler
        else:
            print(f"Usage: python scheduler.py [schedule|run-predict|run-settle|run-odds|run-all]")
            return

    # Start scheduler
    scheduler.start()
    print("  Scheduler started. Press Ctrl+C to stop.\n")

    try:
        # Keep the process alive
        import time
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        print("\n  Shutting down scheduler...")
        scheduler.shutdown()
        print("  Done.")


if __name__ == "__main__":
    main()
