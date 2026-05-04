"""
Dynamic fixture-aware scheduler for the betting system.

Three-tier job architecture:
  Tier 1 — Weekly heavy retrain (Sunday 23:30): full pipeline + model training
  Tier 2 — Matchday light fetch: load pickled models + fresh odds + edge calc
  Tier 3 — Edge recalculation (no API): pure computation from cached data

Uses APScheduler with:
  - CronTrigger for static recurring jobs (weekly retrain, daily planner, settlement)
  - DateTrigger for one-off pre-kickoff odds refreshes (auto-removed after firing)

API budget: ~150 calls/month (well within 500/month free tier).
"""
import os
import sys
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

logger = logging.getLogger("scheduler")
UK_TZ = ZoneInfo("Europe/London")

# Reference to the global scheduler instance (set in create_scheduler)
_scheduler: BackgroundScheduler | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 1: Heavy Retrain (Weekly)
# ═══════════════════════════════════════════════════════════════════════════════

def _refresh_player_data() -> None:
    """Refresh squad availability data from FPL-Core-Insights.

    Downloads latest player data and rebuilds the squad features CSV.
    Non-critical — failures are logged as warnings and do not block
    downstream jobs.
    """
    try:
        from api.player_features import (
            download_latest_player_data,
            build_squad_features_csv,
        )
        download_latest_player_data()
        build_squad_features_csv()
        logger.info("Player data refreshed and squad features recomputed")
        print(f"[{datetime.now():%H:%M:%S}] Player data refreshed")
    except Exception as e:
        logger.warning(f"Player data refresh failed (non-critical): {e}")
        print(f"[{datetime.now():%H:%M:%S}] Player data refresh FAILED "
              f"(non-critical): {e}")


def _fetch_latest_matches() -> bool:
    """Fetch newly completed matches from Understat and update the dataset.

    Returns:
        True if new matches were found and appended, False otherwise.
    """
    try:
        from data.live_updater import update_dataset, get_current_season_year
        season_year = get_current_season_year()
        result = update_dataset(season_year)
        if result is None:
            logger.info("No new matches found on Understat")
            print(f"[{datetime.now():%H:%M:%S}] No new matches found")
            return False
        new_count = len(
            result[result["SeasonIndex"] == (season_year - 2000)]
        )
        logger.info(f"Dataset updated: season {season_year}/{season_year + 1}, "
                     f"{new_count} matches")
        print(f"[{datetime.now():%H:%M:%S}] Dataset updated: {new_count} matches "
              f"in {season_year}/{season_year + 1}")
        return True
    except Exception as e:
        logger.warning(f"Understat match fetch failed (non-critical): {e}")
        print(f"[{datetime.now():%H:%M:%S}] Match fetch FAILED "
              f"(non-critical): {e}")
        return False


def _retrain_squad_adjuster() -> None:
    """Retrain the squad availability adjuster model.

    Learns the relationship between squad disruption features and match
    outcomes. Only retrains if sufficient training data exists (>20 rows).
    Non-critical — failures are logged as warnings.
    """
    try:
        import joblib
        from squad_adjuster import (
            build_adjustment_dataset,
            train_adjuster,
            ADJUSTER_PATH,
        )
        X_train, y_train, X_test, y_test = build_adjustment_dataset()
        if X_train is not None and len(X_train) > 20:
            adj = train_adjuster(X_train, y_train)
            joblib.dump(adj, ADJUSTER_PATH)
            logger.info("Squad adjuster retrained")
            print(f"[{datetime.now():%H:%M:%S}] Squad adjuster retrained")
        else:
            logger.info("Not enough squad data for adjuster retrain")
            print(f"[{datetime.now():%H:%M:%S}] Squad adjuster: "
                  f"insufficient data, skipped")
    except Exception as e:
        logger.warning(f"Squad adjuster retrain failed (non-critical): {e}")
        print(f"[{datetime.now():%H:%M:%S}] Squad adjuster FAILED "
              f"(non-critical): {e}")


def job_daily_data_refresh() -> None:
    """Daily data ingestion: refresh player data and fetch new match results.

    Runs at 06:30 daily, before the fixture planner (07:00). Ensures
    matchday predictions use current squad availability and that the
    dataset includes any matches completed since the last fetch.

    Both steps are non-critical — failures log warnings but do not
    prevent downstream jobs from running.
    """
    logger.info("Starting daily data refresh...")
    print(f"[{datetime.now():%H:%M:%S}] Daily data refresh starting...")

    _refresh_player_data()
    _fetch_latest_matches()

    print(f"[{datetime.now():%H:%M:%S}] Daily data refresh complete.")


def job_weekly_retrain() -> None:
    """Full retrain + pickle save for both PL and Championship.

    Runs data ingestion first (Understat matches + player data) to
    guarantee fresh inputs, then retrains all models, saves trained
    state and pipeline cache, and retrains the squad adjuster.
    """
    logger.info("Starting weekly retrain...")
    print(f"[{datetime.now():%H:%M:%S}] Weekly retrain starting...")

    # ── Data ingestion (guarantees fresh inputs before training) ──
    _fetch_latest_matches()
    _refresh_player_data()

    # Off-season gates: each league has an explicit boolean in config.py.
    # Skipping a league here is the supported way to pause retraining over
    # the summer break — no new match data means a retrain would refit on
    # stale features and waste CPU. Flip the flag back to True a few weeks
    # before the new season kicks off.
    from config import PL_RETRAIN_ENABLED, EFL_RETRAIN_ENABLED

    # ── PL ──
    if not PL_RETRAIN_ENABLED:
        logger.info("PL retrain skipped — PL_RETRAIN_ENABLED is False (off-season)")
        print(f"[{datetime.now():%H:%M:%S}] PL retrain SKIPPED (off-season gate)")
    else:
        try:
            from predict import LivePredictor
            from dashboard import (save_recommendations, save_match_analysis,
                                   log_predictions)

            # Option β-tight: weekly retrain IS the week-ahead snapshot — fetch
            # the full OddsPapi sweep here so bet selection has 300+ bookmaker
            # best-price coverage. Matchday refreshes go Odds-API only.
            predictor = LivePredictor(verbose=True, snapshot_type="week_ahead")
            predictor.load_data()
            predictor.train()
            predictor.save_trained_state()
            predictor.save_pipeline_cache()

            recs = predictor.generate_recommendations()
            n_saved = save_recommendations(recs)
            analysis = getattr(predictor, "_match_analysis", [])
            if analysis:
                save_match_analysis(analysis, league="PL")
                log_predictions(analysis, league="PL")

            logger.info(f"PL retrain complete: {len(recs)} recs, {n_saved} saved")
            print(f"[{datetime.now():%H:%M:%S}] PL retrain: {len(recs)} recs, "
                  f"{n_saved} saved")
        except Exception as e:
            logger.error(f"PL retrain failed: {e}", exc_info=True)
            print(f"[{datetime.now():%H:%M:%S}] PL retrain FAILED: {e}")

    # ── Championship ──
    if not EFL_RETRAIN_ENABLED:
        logger.info("EFL retrain skipped — EFL_RETRAIN_ENABLED is False (off-season)")
        print(f"[{datetime.now():%H:%M:%S}] EFL retrain SKIPPED (off-season gate)")
    else:
        try:
            from championship_predict import ChampionshipPredictor
            from dashboard import (save_recommendations, save_match_analysis,
                                   log_predictions)

            # Option β-tight: weekly retrain = week-ahead snapshot (see PL above).
            champ = ChampionshipPredictor(verbose=True, snapshot_type="week_ahead")
            champ.load_data()
            champ.train()
            champ.save_trained_state()
            champ.save_pipeline_cache()

            recs = champ.generate_recommendations()
            n_saved = save_recommendations(recs, league="EFL")
            analysis = getattr(champ, "_match_analysis", [])
            if analysis:
                save_match_analysis(analysis, league="EFL")
                log_predictions(analysis, league="EFL")

            logger.info(f"EFL retrain complete: {len(recs)} recs, {n_saved} saved")
            print(f"[{datetime.now():%H:%M:%S}] EFL retrain: {len(recs)} recs, "
                  f"{n_saved} saved")
        except Exception as e:
            logger.error(f"EFL retrain failed: {e}", exc_info=True)
            print(f"[{datetime.now():%H:%M:%S}] EFL retrain FAILED: {e}")

    # ── Squad adjuster ──
    _retrain_squad_adjuster()

    print(f"[{datetime.now():%H:%M:%S}] Weekly retrain complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tier 2: Light Matchday Fetch
# ═══════════════════════════════════════════════════════════════════════════════

def job_matchday_fetch(
    league: str = "PL",
    markets: tuple[str, ...] = ("totals", "btts"),
) -> None:
    """Light prediction: load pickled models + fresh odds + save analysis.

    Args:
        league: "PL" or "EFL".
        markets: Which odds markets to fetch.
    """
    logger.info(f"Matchday fetch for {league} (markets={markets})...")
    try:
        from dashboard import (save_recommendations, save_match_analysis,
                               log_predictions)

        # Option β-tight: matchday morning + KO-1h refreshes use Odds-API
        # only. snapshot_type="refresh" gates out the OddsPapi fetch path.
        if league == "PL":
            from predict import LivePredictor
            predictor = LivePredictor(verbose=True, snapshot_type="refresh")
        else:
            from championship_predict import ChampionshipPredictor
            predictor = ChampionshipPredictor(
                verbose=True, snapshot_type="refresh")

        recs = predictor.light_refresh(markets=markets)

        if league == "PL":
            n_saved = save_recommendations(recs)
        else:
            n_saved = save_recommendations(recs, league="EFL")

        analysis = getattr(predictor, "_match_analysis", [])
        lg = "PL" if league == "PL" else "EFL"
        if analysis:
            save_match_analysis(analysis, league=lg)
            log_predictions(analysis, league=lg)

        logger.info(f"{league} matchday fetch: {len(recs)} recs, {n_saved} saved, "
                    f"{len(analysis)} analysis rows")
        print(f"[{datetime.now():%H:%M:%S}] {league} fetch: {len(recs)} recs, "
              f"{n_saved} saved, {len(analysis)} analysis rows")
    except Exception as e:
        logger.error(f"{league} matchday fetch failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] {league} fetch FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLV: Closing Odds Capture
# ═══════════════════════════════════════════════════════════════════════════════

def job_fetch_closing_odds() -> None:
    """Fetch closing odds for all unsettled logged bets (both leagues).

    Should fire ~5 minutes before kickoff. Updates logged_bets with
    the latest available odds so CLV can be calculated after settlement.
    """
    logger.info("Fetching closing odds for logged bets...")
    try:
        from dashboard import fetch_closing_odds_for_logged_bets

        total = 0
        for league in ("PL", "EFL"):
            n = fetch_closing_odds_for_logged_bets(league)
            if n > 0:
                logger.info(f"  {league}: updated closing odds for {n} bets")
            total += n

        print(f"[{datetime.now():%H:%M:%S}] Closing odds: "
              f"{total} bets updated")
    except Exception as e:
        logger.error(f"Closing odds fetch failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Closing odds FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Settlement
# ═══════════════════════════════════════════════════════════════════════════════

def job_settle_bets() -> None:
    """Pull match results and settle open bets + predictions."""
    logger.info("Starting bet settlement...")
    try:
        from settlement import settle_bets, settle_predictions
        summary = settle_bets(days_back=3, verbose=True)
        logger.info(f"Settlement complete: {summary}")
        print(f"[{datetime.now():%H:%M:%S}] Settlement: "
              f"{summary['settled']} settled, "
              f"{summary['won']}W/{summary['lost']}L, "
              f"P/L: {summary['profit']:+.4f}")

        # Also settle prediction tracking (model accuracy, not stakes)
        pred_summary = settle_predictions(days_back=3, verbose=True)
        logger.info(f"Prediction settlement: {pred_summary}")
    except Exception as e:
        logger.error(f"Settlement failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Settlement FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Fixture-Aware Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

def schedule_matchday_jobs(
    scheduler: BackgroundScheduler,
    league: str,
    kickoffs: list[datetime],
) -> None:
    """Schedule matchday-specific jobs for a set of kickoff times.

    Creates:
      - Morning fetch at 09:00 (all markets including BTTS)
      - Pre-kickoff fetch 60 min before each distinct kickoff (totals only)

    Uses DateTrigger for one-off execution (auto-removed after firing).

    Args:
        scheduler: The APScheduler instance.
        league: "PL" or "EFL".
        kickoffs: List of distinct kickoff datetimes (UK time).
    """
    if not kickoffs:
        return

    matchday = kickoffs[0].date()
    now = datetime.now(UK_TZ)
    jobs_added = 0

    # Morning fetch at 09:00 on matchday (all markets)
    morning_time = datetime(
        matchday.year, matchday.month, matchday.day,
        9, 0, tzinfo=UK_TZ,
    )
    morning_id = f"matchday_{league}_{matchday}_morning"

    if morning_time > now:
        # Remove existing job with same ID if it exists (e.g. re-run of planner)
        if scheduler.get_job(morning_id):
            scheduler.remove_job(morning_id)
        scheduler.add_job(
            job_matchday_fetch,
            trigger=DateTrigger(run_date=morning_time),
            id=morning_id,
            name=f"{league} morning fetch ({matchday})",
            kwargs={"league": league, "markets": ("totals", "btts")},
            replace_existing=True,
            misfire_grace_time=3600,
        )
        jobs_added += 1
        logger.info(f"Scheduled {league} morning fetch at {morning_time:%H:%M}")

    # Pre-kickoff fetch 60 min before each distinct kickoff (totals only)
    for ko in kickoffs:
        pre_ko_time = ko - timedelta(minutes=60)
        pre_ko_id = f"matchday_{league}_{matchday}_{ko:%H%M}_preko"

        if pre_ko_time > now:
            if scheduler.get_job(pre_ko_id):
                scheduler.remove_job(pre_ko_id)
            scheduler.add_job(
                job_matchday_fetch,
                trigger=DateTrigger(run_date=pre_ko_time),
                id=pre_ko_id,
                name=f"{league} pre-kickoff {ko:%H:%M} ({matchday})",
                kwargs={"league": league, "markets": ("totals",)},
                replace_existing=True,
                misfire_grace_time=1800,
            )
            jobs_added += 1
            logger.info(f"Scheduled {league} pre-kickoff at {pre_ko_time:%H:%M} "
                        f"(KO {ko:%H:%M})")

    # CLV: capture closing odds 5 min before earliest kickoff
    earliest_ko = min(kickoffs)
    clv_time = earliest_ko - timedelta(minutes=5)
    clv_id = f"matchday_{league}_{matchday}_clv"

    if clv_time > now:
        if scheduler.get_job(clv_id):
            scheduler.remove_job(clv_id)
        scheduler.add_job(
            job_fetch_closing_odds,
            trigger=DateTrigger(run_date=clv_time),
            id=clv_id,
            name=f"CLV closing odds ({matchday})",
            replace_existing=True,
            misfire_grace_time=1800,
        )
        jobs_added += 1
        logger.info(f"Scheduled CLV closing odds at {clv_time:%H:%M} "
                    f"(earliest KO {earliest_ko:%H:%M})")

    if jobs_added:
        print(f"[{datetime.now():%H:%M:%S}] Scheduled {jobs_added} "
              f"{league} jobs for {matchday}")


def job_plan_today() -> None:
    """Daily planner: check for today's fixtures and schedule matchday jobs.

    Runs at 07:00 every day. If matches are found, creates morning and
    pre-kickoff jobs via schedule_matchday_jobs().
    """
    global _scheduler
    if _scheduler is None:
        logger.warning("No scheduler instance — cannot plan matchday jobs")
        return

    logger.info("Running daily fixture planner...")

    try:
        from fixture_schedule import get_matchday_kickoffs
    except ImportError as e:
        logger.error(f"Cannot import fixture_schedule: {e}")
        return

    today = date.today()
    any_matches = False

    for league in ("PL", "EFL"):
        try:
            kickoffs = get_matchday_kickoffs(league, today)
            if kickoffs:
                any_matches = True
                print(f"[{datetime.now():%H:%M:%S}] {league}: "
                      f"{len(kickoffs)} kickoff(s) today — "
                      f"{', '.join(ko.strftime('%H:%M') for ko in kickoffs)}")
                schedule_matchday_jobs(_scheduler, league, kickoffs)
            else:
                logger.info(f"No {league} matches today")
        except Exception as e:
            logger.error(f"Fixture check failed for {league}: {e}",
                         exc_info=True)

    if not any_matches:
        print(f"[{datetime.now():%H:%M:%S}] No matches today for any league")


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy Job Functions (kept for CLI compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def job_generate_predictions() -> None:
    """Full PL prediction (retrain + odds + save). Used by CLI run-predict."""
    logger.info("Starting PL prediction generation...")
    try:
        from predict import LivePredictor
        from dashboard import (save_recommendations, save_match_analysis,
                               log_predictions)

        # CLI run-predict is treated as a week-ahead sweep (operator running
        # manually expects full OddsPapi enrichment).
        predictor = LivePredictor(verbose=True, snapshot_type="week_ahead")
        predictor.load_data()
        predictor.train()
        predictor.save_trained_state()
        predictor.save_pipeline_cache()
        recs = predictor.generate_recommendations()
        n_saved = save_recommendations(recs)

        analysis = getattr(predictor, "_match_analysis", [])
        if analysis:
            save_match_analysis(analysis, league="PL")
            log_predictions(analysis, league="PL")

        logger.info(f"Predictions complete: {len(recs)} generated, {n_saved} new saved, "
                    f"{len(analysis)} analysis rows")
        print(f"[{datetime.now():%H:%M:%S}] Predictions: {len(recs)} generated, "
              f"{n_saved} saved, {len(analysis)} analysis rows")
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Prediction FAILED: {e}")


def job_generate_champ_predictions() -> None:
    """Full Championship prediction. Used by CLI run-champ."""
    logger.info("Starting Championship prediction generation...")
    try:
        from championship_predict import ChampionshipPredictor
        from dashboard import (save_recommendations, save_match_analysis,
                               log_predictions)

        # CLI run-champ is treated as a week-ahead sweep (see PL above).
        predictor = ChampionshipPredictor(
            verbose=True, snapshot_type="week_ahead")
        predictor.load_data()
        predictor.train()
        predictor.save_trained_state()
        predictor.save_pipeline_cache()
        recs = predictor.generate_recommendations()
        n_saved = save_recommendations(recs, league="EFL")

        analysis = getattr(predictor, "_match_analysis", [])
        if analysis:
            save_match_analysis(analysis, league="EFL")
            log_predictions(analysis, league="EFL")

        logger.info(f"Championship predictions: {len(recs)} generated, {n_saved} saved")
        print(f"[{datetime.now():%H:%M:%S}] Championship: {len(recs)} recs, "
              f"{n_saved} saved, {len(analysis)} analysis rows")
    except Exception as e:
        logger.error(f"Championship prediction failed: {e}", exc_info=True)
        print(f"[{datetime.now():%H:%M:%S}] Championship prediction FAILED: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Static jobs that run regardless of fixtures
STATIC_SCHEDULE = {
    "daily_data_refresh": {
        "func": job_daily_data_refresh,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=6,
            minute=30,
            timezone=UK_TZ,
        ),
        "description": "Daily data refresh — player data + match results (06:30)",
    },
    "weekly_retrain": {
        "func": job_weekly_retrain,
        "trigger": CronTrigger(
            day_of_week="sun",
            hour=23,
            minute=30,
            timezone=UK_TZ,
        ),
        "description": "Weekly full retrain (Sunday 23:30)",
    },
    "daily_planner": {
        "func": job_plan_today,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=7,
            minute=0,
            timezone=UK_TZ,
        ),
        "description": "Daily fixture planner (07:00)",
    },
    "settle_evening": {
        "func": job_settle_bets,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=23,
            minute=0,
            timezone=UK_TZ,
        ),
        "description": "Settle bets every evening (23:00)",
    },
    "settle_morning": {
        "func": job_settle_bets,
        "trigger": CronTrigger(
            day_of_week="mon,tue,wed,thu,fri,sat,sun",
            hour=9,
            minute=0,
            timezone=UK_TZ,
        ),
        "description": "Morning settlement catch-up (09:00)",
    },
}


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the scheduler with static jobs.

    Dynamic matchday jobs are added by job_plan_today() at 07:00 daily.
    """
    global _scheduler
    scheduler = BackgroundScheduler(timezone="Europe/London")

    for job_id, config in STATIC_SCHEDULE.items():
        scheduler.add_job(
            config["func"],
            trigger=config["trigger"],
            id=job_id,
            name=config["description"],
            replace_existing=True,
            misfire_grace_time=3600,
        )

    _scheduler = scheduler
    return scheduler


def print_schedule(scheduler: BackgroundScheduler) -> None:
    """Print the current schedule."""
    print("\n  Scheduled Jobs:")
    print(f"  {'Job':<35} {'Next Run':<25} {'Description'}")
    print(f"  {'-'*90}")
    for job in scheduler.get_jobs():
        next_run = (job.next_run_time.strftime("%Y-%m-%d %H:%M %Z")
                    if job.next_run_time else "—")
        print(f"  {job.id:<35} {next_run:<25} {job.name}")
    print()


def run_now(job_name: str) -> None:
    """Run a specific job immediately by name.

    Args:
        job_name: One of 'predict', 'predict-champ', 'settle',
                  'retrain', 'plan', 'fetch-pl', 'fetch-efl',
                  'closing-odds', 'refresh'.
    """
    jobs = {
        "predict": job_generate_predictions,
        "predict-champ": job_generate_champ_predictions,
        "settle": job_settle_bets,
        "retrain": job_weekly_retrain,
        "plan": job_plan_today,
        "fetch-pl": lambda: job_matchday_fetch("PL"),
        "fetch-efl": lambda: job_matchday_fetch("EFL"),
        "closing-odds": job_fetch_closing_odds,
        "refresh": job_daily_data_refresh,
    }
    func = jobs.get(job_name)
    if func is None:
        print(f"Unknown job: {job_name}. Options: {list(jobs.keys())}")
        return
    func()


def main() -> None:
    """Run the scheduler as a long-lived process alongside the dashboard."""
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
    print("  Betting System — Dynamic Scheduler (PL + Championship)")
    print("  API budget: ~150 calls/month (500 limit)")
    print("=" * 60)

    scheduler = create_scheduler()

    # Parse CLI args for immediate actions
    if len(sys.argv) > 1:
        action = sys.argv[1].lower()
        immediate_actions = {
            "run-predict": ("Running PL predictions...", job_generate_predictions),
            "run-settle": ("Running settlement...", job_settle_bets),
            "run-champ": ("Running Championship predictions...",
                          job_generate_champ_predictions),
            "run-retrain": ("Running weekly retrain...", job_weekly_retrain),
            "run-plan": ("Running daily fixture planner...", job_plan_today),
            "run-fetch-pl": ("Running PL light fetch...",
                             lambda: job_matchday_fetch("PL")),
            "run-fetch-efl": ("Running EFL light fetch...",
                              lambda: job_matchday_fetch("EFL")),
            "run-closing-odds": ("Fetching closing odds for logged bets...",
                                 job_fetch_closing_odds),
            "run-refresh": ("Running daily data refresh...",
                            job_daily_data_refresh),
            "run-all": ("Running all predictions...", None),
        }

        if action in immediate_actions:
            msg, func = immediate_actions[action]
            print(f"\n{msg}\n")
            if action == "run-all":
                job_generate_predictions()
                job_generate_champ_predictions()
                job_settle_bets()
            else:
                func()
            return
        elif action == "schedule":
            pass  # Fall through to start scheduler
        else:
            print(f"Usage: python scheduler.py [schedule|run-predict|run-settle|"
                  f"run-champ|run-retrain|run-plan|run-fetch-pl|run-fetch-efl|"
                  f"run-refresh|run-all]")
            return

    # Start scheduler
    scheduler.start()
    print_schedule(scheduler)

    # Also run the daily planner immediately on startup to schedule today's jobs
    print("  Running fixture planner for today...")
    job_plan_today()

    print("  Scheduler started. Press Ctrl+C to stop.\n")

    try:
        import time
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        print("\n  Shutting down scheduler...")
        scheduler.shutdown()
        print("  Done.")


if __name__ == "__main__":
    main()
