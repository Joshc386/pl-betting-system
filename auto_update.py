"""
Automatic dataset updater.
Fetches latest match results, retrains the model, and logs the outcome.

Designed to run via Windows Task Scheduler (daily at 7am).
Usage:
    python auto_update.py
"""
import os
import sys
import logging
import joblib
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# ── Logging ──
LOG_PATH = os.path.join(PROJECT_DIR, "logs", "auto_update.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("Auto-update started")

    # Step 0: Refresh player data from FPL-Core-Insights
    log.info("Step 0: Refreshing player data from FPL-Core-Insights...")
    try:
        from api.player_features import download_latest_player_data, build_squad_features_csv
        download_latest_player_data()
        build_squad_features_csv()
        log.info("  Player data refreshed and squad features recomputed.")
    except Exception as e:
        log.warning(f"  Player data refresh failed (non-critical): {e}")

    # Step 1: Fetch latest matches
    log.info("Step 1: Fetching latest match data from Understat...")
    try:
        from data.live_updater import update_dataset, get_current_season_year
        season_year = get_current_season_year()
        log.info(f"  Season: {season_year}/{season_year + 1}")
        result = update_dataset(season_year)
        if result is None:
            log.info("  No new matches found. Skipping retrain.")
            return
        new_count = len(result[result["SeasonIndex"] == (season_year - 2000)])
        log.info(f"  Dataset updated. Season has {new_count} matches.")
    except Exception as e:
        log.error(f"  Data fetch failed: {e}", exc_info=True)
        return

    # Step 2: Retrain model
    log.info("Step 2: Retraining model...")
    try:
        from model import main as train_main

        ensemble, features, test_metrics = train_main(tune=False)
        log.info(f"  Model retrained. Test AUC: {test_metrics['auc']:.4f}, "
                 f"Acc: {test_metrics['accuracy']:.4f}")

    except Exception as e:
        log.error(f"  Retrain failed: {e}", exc_info=True)
        return

    # Step 2b: Retrain squad adjuster
    log.info("Step 2b: Retraining squad adjuster...")
    try:
        from squad_adjuster import build_adjustment_dataset, train_adjuster, ADJUSTER_PATH
        X_train_adj, y_train_adj, X_test_adj, y_test_adj = build_adjustment_dataset()
        if X_train_adj is not None and len(X_train_adj) > 20:
            adj = train_adjuster(X_train_adj, y_train_adj)
            joblib.dump(adj, ADJUSTER_PATH)
            log.info("  Squad adjuster retrained.")
        else:
            log.info("  Not enough squad data for adjuster.")
    except Exception as e:
        log.warning(f"  Squad adjuster retrain failed (non-critical): {e}")

    log.info("Auto-update completed successfully.")


if __name__ == "__main__":
    main()
