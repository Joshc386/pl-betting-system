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

    elif action in ("all", "start"):
        print("=" * 60)
        print("  Premier League Betting System")
        print("  Dashboard: http://127.0.0.1:8050")
        print("=" * 60)

        # Start scheduler in background thread
        from scheduler import create_scheduler, print_schedule
        scheduler = create_scheduler()
        print_schedule(scheduler)
        scheduler.start()
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
        print("Usage: python run.py [all|predict|settle|dashboard]")


if __name__ == "__main__":
    main()
