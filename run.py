"""
Main entry point: starts the dashboard and scheduler together.

Usage:
    python run.py              — Start dashboard + scheduler
    python run.py predict      — Run predictions once (no server)
    python run.py settle       — Run settlement once (no server)
    python run.py dashboard    — Start dashboard only (no scheduler)
"""
import sys
import os
import logging
import threading

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "system.log")),
        logging.StreamHandler(sys.stdout),
    ],
)


def main():
    action = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if action == "predict":
        from scheduler import job_generate_predictions
        print("\n  Running predictions...\n")
        job_generate_predictions()

    elif action == "settle":
        from scheduler import job_settle_bets
        print("\n  Running settlement...\n")
        job_settle_bets()

    elif action == "dashboard":
        from dashboard import main as run_dashboard
        run_dashboard()

    elif action == "retrain":
        from scheduler import job_weekly_retrain
        print("\n  Running weekly retrain (PL + Championship)...\n")
        job_weekly_retrain()

    elif action == "fetch":
        league = sys.argv[2].upper() if len(sys.argv) > 2 else "PL"
        from scheduler import job_matchday_fetch
        print(f"\n  Running light fetch for {league}...\n")
        job_matchday_fetch(league=league)

    elif action in ("all", "start"):
        print("=" * 60)
        print("  Premier League & Championship Betting System")
        print("  Dashboard: http://127.0.0.1:8050")
        print("  Dynamic scheduler: fixture-aware odds fetching")
        print("=" * 60)

        # Start scheduler in background thread
        from scheduler import create_scheduler, print_schedule, job_plan_today
        scheduler = create_scheduler()
        scheduler.start()
        print_schedule(scheduler)

        # Run fixture planner immediately to schedule today's jobs
        print("  Checking today's fixtures...")
        job_plan_today()
        print("  Scheduler running in background.\n")

        # Start dashboard in main thread (blocking)
        from dashboard import create_app
        app = create_app()
        try:
            app.run(debug=False, host="127.0.0.1", port=8050)
        except KeyboardInterrupt:
            pass
        finally:
            print("\n  Shutting down scheduler...")
            scheduler.shutdown()
            print("  Done.")

    else:
        print("Usage: python run.py [all|predict|settle|dashboard|retrain|fetch]")


if __name__ == "__main__":
    main()
