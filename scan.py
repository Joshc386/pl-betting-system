"""
Odds scan pipeline: fetch odds → resolve teams → run predictor → save analysis.

Called by the dashboard "Refresh Odds" button and potentially by the scheduler
for on-demand scans.  All odds-fetching, OddsPapi integration, team
resolution, and match-analysis row construction lives here — dashboard.py
just calls ``run_scan(league)`` and displays the returned status string.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from db import (
    get_match_analysis,
    save_match_analysis,
    log_predictions,
    get_active_recommendations,
    save_recommendations,
    LEAGUE_DISPLAY_NAMES,
)
from freshness import FreshnessError
from league_config import get_league_config

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# OddsPapi ↔ Odds-API format converters
# ═════════════════════════════════════════════════════════════════════════════

def oddspapi_to_matches(oddspapi_fixtures: list[dict]) -> list[dict]:
    """Convert OddsPapi fixture data to The-Odds-API match format.

    This lets the scan pipeline process OddsPapi data identically to
    The-Odds-API data, so all downstream helpers (get_best_odds,
    get_best_btts_odds, get_best_odds_all_lines) work unchanged.

    Args:
        oddspapi_fixtures: List of OddsPapi fixture dicts from
            fetch_epl_all_odds().

    Returns:
        List of match dicts in The-Odds-API format.
    """
    matches = []
    for fx in oddspapi_fixtures:
        bookmakers: dict[str, dict] = {}
        btts_bookmakers: dict[str, dict] = {}

        # ── Build bookmakers from ou_lines ──
        # OddsPapi gives aggregated best prices per line. We create a
        # single synthetic "best-price" bookmaker that has both over AND
        # under for every line, so get_best_odds_all_lines() works.
        # We also create individual bookmaker entries for the actual
        # best-price books so the bookmaker column shows real names.
        ou_lines = fx.get("ou_lines") or {}
        all_lines_combined: dict[float, dict] = {}
        for line_str, line_data in ou_lines.items():
            line_f = float(line_str)
            best_over = line_data.get("best_over")
            best_under = line_data.get("best_under")
            if not best_over or not best_under:
                continue
            all_lines_combined[line_f] = {
                "over": best_over, "under": best_under,
            }

            # Track individual bookmakers for display
            for side, bk_key_field, odds_field in [
                ("over", "best_over_book", "best_over"),
                ("under", "best_under_book", "best_under"),
            ]:
                bk = line_data.get(bk_key_field, "oddspapi")
                if bk not in bookmakers:
                    bookmakers[bk] = {
                        "title": bk, "over": 0, "under": 0,
                        "is_sharp": "pinnacle" in bk.lower(),
                        "is_major": True, "all_lines": {},
                    }
                bookmakers[bk]["all_lines"].setdefault(line_f, {})
                bookmakers[bk]["all_lines"][line_f][side] = (
                    line_data.get(odds_field))
                if line_f == 2.5:
                    bookmakers[bk][side] = line_data.get(odds_field)

        # Synthetic bookmaker with ALL lines having both sides — this
        # ensures get_best_odds_all_lines() finds every line.
        if all_lines_combined:
            ou25 = ou_lines.get("2.5") or ou_lines.get(2.5) or {}
            bookmakers["_oddspapi_best"] = {
                "title": "OddsPapi Best",
                "over": ou25.get("best_over", 0),
                "under": ou25.get("best_under", 0),
                "is_sharp": False, "is_major": True,
                "all_lines": all_lines_combined,
            }

        # Ensure every real bookmaker has both over and under for 2.5
        ou25 = ou_lines.get("2.5") or ou_lines.get(2.5) or {}
        for bk_data in bookmakers.values():
            if bk_data["over"] == 0 and ou25.get("best_over"):
                bk_data["over"] = ou25["best_over"]
            if bk_data["under"] == 0 and ou25.get("best_under"):
                bk_data["under"] = ou25["best_under"]

        # ── Build btts_bookmakers ──
        btts = fx.get("btts")
        if btts and btts.get("best_yes") and btts.get("best_no"):
            yes_bk = btts.get("best_yes_book", "oddspapi")
            no_bk = btts.get("best_no_book", "oddspapi")
            # Both sides need to be on each bookmaker entry for the
            # helper to iterate correctly
            for bk_key in {yes_bk, no_bk}:
                btts_bookmakers[bk_key] = {
                    "title": bk_key,
                    "yes": btts["best_yes"],
                    "no": btts["best_no"],
                    "is_sharp": "pinnacle" in bk_key.lower(),
                    "is_major": True,
                }

        # Only include fixtures that have at least some odds data
        if not bookmakers and not btts_bookmakers:
            continue

        start_time = fx.get("start_time", "")
        # Normalise timestamp: OddsPapi uses ".000Z", The-Odds-API uses "Z"
        commence_time = start_time.replace(".000Z", "Z")

        matches.append({
            "id": fx.get("fixture_id", ""),
            "home_team": fx.get("home_team", ""),
            "away_team": fx.get("away_team", ""),
            "commence_time": commence_time,
            "bookmakers": bookmakers,
            "btts_bookmakers": btts_bookmakers,
        })

    return matches


def _normalise_team_for_merge(name: str) -> str:
    """Strip common suffixes for fuzzy matching between OddsPapi and The-Odds-API.

    OddsPapi uses 'Southampton FC', The-Odds-API uses 'Southampton'.
    This normalises both to the same base form for fixture matching.

    Args:
        name: Raw team name from either API.

    Returns:
        Lowercased name with trailing FC/AFC stripped.
    """
    n = name.strip()
    # Remove trailing " FC" or " AFC" (case-insensitive)
    for suffix in (" FC", " AFC"):
        if n.upper().endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    return n.lower()


def merge_oddspapi_into_matches(
    oa_matches: list[dict],
    op_matches: list[dict],
) -> list[dict]:
    """Merge OddsPapi market data into The-Odds-API match dicts.

    For each OddsPapi match that corresponds to a The-Odds-API match:
    - Injects missing all_lines (e.g. O/U 1.5) into bookmaker entries
    - Fills missing btts_bookmakers when The-Odds-API has no BTTS

    Does NOT overwrite existing data — The-Odds-API values take priority
    since they tend to be more recent.

    Args:
        oa_matches: The-Odds-API match dicts (mutated in-place).
        op_matches: OddsPapi match dicts (from oddspapi_to_matches).

    Returns:
        The oa_matches list (same reference, now enriched).
    """
    # Build lookup for OddsPapi matches by normalised team names
    op_lookup: dict[tuple[str, str], dict] = {}
    for m in op_matches:
        key = (
            _normalise_team_for_merge(m.get("home_team", "")),
            _normalise_team_for_merge(m.get("away_team", "")),
        )
        op_lookup[key] = m

    merged_count = 0
    for oa_match in oa_matches:
        key = (
            _normalise_team_for_merge(oa_match.get("home_team", "")),
            _normalise_team_for_merge(oa_match.get("away_team", "")),
        )
        op_match = op_lookup.get(key)
        if not op_match:
            continue

        merged_count += 1

        # ── Merge bookmaker all_lines (fills O/U 1.5 gaps) ──
        oa_books = oa_match.get("bookmakers", {})
        op_books = op_match.get("bookmakers", {})

        for bk_key, op_bk in op_books.items():
            op_lines = op_bk.get("all_lines", {})
            if not op_lines:
                continue

            if bk_key in oa_books:
                # Bookmaker exists in both — add missing lines only
                existing_lines = oa_books[bk_key].get("all_lines", {})
                for line_pt, line_odds in op_lines.items():
                    if line_pt not in existing_lines:
                        existing_lines[line_pt] = line_odds
                oa_books[bk_key]["all_lines"] = existing_lines
            else:
                # Bookmaker only in OddsPapi — add entire entry
                oa_books[bk_key] = op_bk

        # ── Merge BTTS (fill gaps where The-Odds-API has none) ──
        oa_btts = oa_match.get("btts_bookmakers", {})
        op_btts = op_match.get("btts_bookmakers", {})
        if not oa_btts and op_btts:
            # No BTTS from The-Odds-API — use OddsPapi entirely
            oa_match["btts_bookmakers"] = op_btts
        elif oa_btts and op_btts:
            # Both have BTTS — add missing bookmakers only
            for bk_key, bk_data in op_btts.items():
                if bk_key not in oa_btts:
                    oa_btts[bk_key] = bk_data

    logger.info(
        "OddsPapi merge: %d/%d fixtures enriched",
        merged_count, len(oa_matches),
    )
    unmatched = len(oa_matches) - merged_count
    if unmatched > 0:
        logger.warning(
            "OddsPapi merge: %d fixtures had no OddsPapi match", unmatched)

    return oa_matches


# ═════════════════════════════════════════════════════════════════════════════
# Main scan pipeline
# ═════════════════════════════════════════════════════════════════════════════

def run_scan(league: str) -> str:
    """Fetch upcoming fixtures and odds, then save to match_analysis.

    This is a fast operation (seconds) — it only fetches odds from APIs.
    Model predictions are populated by running predict.py separately
    (via CLI or scheduler), which saves recommendations to the DB.

    Args:
        league: League key ("PL", "EFL").

    Returns:
        Status string for the UI.
    """
    league_name = LEAGUE_DISPLAY_NAMES.get(league, league)
    now = datetime.now().strftime("%H:%M:%S")

    os.environ["ACTIVE_LEAGUE"] = league
    league_cfg = get_league_config(league)

    try:
        from dotenv import load_dotenv
        load_dotenv()

        from api.odds_api import (
            fetch_epl_odds, get_best_odds, get_best_btts_odds,
            get_best_odds_all_lines, match_to_our_teams,
        )
        from api.team_mapping import normalize
        import api.odds_api as odds_mod

        # Temporarily override sport + cache for this league
        original_sport = odds_mod.SPORT
        original_cache = odds_mod.CACHE_FILE
        odds_mod.SPORT = league_cfg.get("odds_api_sport", original_sport)
        if league != "PL":
            odds_mod.CACHE_FILE = original_cache.replace(
                "odds_cache", f"odds_cache_{league.lower()}")

        try:
            matches = fetch_epl_odds(force_refresh=False)
        finally:
            odds_mod.SPORT = original_sport
            odds_mod.CACHE_FILE = original_cache

        # ── OddsPapi integration ──
        # Option β-tight: gated behind DASHBOARD_FETCH_ODDSPAPI (default False).
        # Dashboard serves cached merged snapshots from the most recent
        # predictor run; the predictor itself is the only path that fetches
        # OddsPapi (and only on the week-ahead snapshot). This avoids burning
        # ~14 OddsPapi credits per dashboard load. Set the flag to True only
        # for diagnostic use.
        from config import DASHBOARD_FETCH_ODDSPAPI as _DB_FETCH_OP
        if _DB_FETCH_OP:
            try:
                from api.oddspapi import (
                    fetch_epl_all_odds as _fetch_op_epl,
                    fetch_championship_all_odds as _fetch_op_efl,
                    CACHE_FILE as _OP_CACHE_PL,
                    CHAMPIONSHIP_CACHE_FILE as _OP_CACHE_EFL,
                )
                _op_fetcher = (
                    _fetch_op_efl if league == "EFL" else _fetch_op_epl
                )
                _op_cache = (
                    _OP_CACHE_EFL if league == "EFL" else _OP_CACHE_PL
                )

                # Try normal fetch (respects TTL)
                op_fixtures = _op_fetcher(force_refresh=False)

                # If that returns nothing (API + cache both expired),
                # load stale cache directly — old odds are better than none
                if not op_fixtures:
                    try:
                        with open(_op_cache, "r") as _f:
                            _raw = json.load(_f)
                        op_fixtures = _raw.get("data", [])
                        if op_fixtures:
                            logger.info(
                                "Loaded stale OddsPapi cache (%s): %d fixtures "
                                "(cached %s)", league, len(op_fixtures),
                                _raw.get("timestamp", "unknown"),
                            )
                    except (FileNotFoundError, json.JSONDecodeError):
                        pass

                if op_fixtures:
                    op_matches = oddspapi_to_matches(op_fixtures)
                    if not matches:
                        # Full fallback — The-Odds-API returned nothing
                        matches = op_matches
                        logger.info(
                            "OddsPapi fallback (%s): %d fixtures converted",
                            league, len(matches),
                        )
                    else:
                        # Merge — enrich The-Odds-API data with OddsPapi
                        # alt lines + BTTS
                        matches = merge_oddspapi_into_matches(
                            matches, op_matches)
            except Exception as e:
                logger.warning("OddsPapi integration failed (%s): %s",
                               league, e)
        else:
            # Option β-tight: dashboard read-only. If The-Odds-API returned
            # nothing (quota exhausted etc.), fall back to whatever the
            # OddsPapi cache last contained — but never trigger a fresh
            # network fetch from this code path.
            if not matches:
                try:
                    from api.oddspapi import (
                        CACHE_FILE as _OP_CACHE_PL,
                        CHAMPIONSHIP_CACHE_FILE as _OP_CACHE_EFL,
                    )
                    _op_cache = (
                        _OP_CACHE_EFL if league == "EFL" else _OP_CACHE_PL
                    )
                    with open(_op_cache, "r") as _f:
                        _raw = json.load(_f)
                    op_fixtures = _raw.get("data", [])
                    if op_fixtures:
                        matches = oddspapi_to_matches(op_fixtures)
                        logger.info(
                            "Dashboard read-only fallback to OddsPapi cache "
                            "(%s): %d fixtures (cached %s)",
                            league, len(matches),
                            _raw.get("timestamp", "unknown"),
                        )
                except (FileNotFoundError, ValueError, ImportError):
                    pass

        if not matches:
            return f"{league_name} | No upcoming fixtures found | {now}"

        # Filter to fixtures within today → 7-day window
        _now_utc = datetime.now(timezone.utc)
        _today_start = _now_utc.replace(
            hour=0, minute=0, second=0, microsecond=0)
        _cutoff = _now_utc + timedelta(days=7)
        _filtered = []
        for m in matches:
            ct = m.get("commence_time", "")
            if ct:
                try:
                    ko = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if ko < _today_start or ko > _cutoff:
                        continue
                except (ValueError, TypeError):
                    pass
            _filtered.append(m)
        matches = _filtered

        # For EFL, use Championship-specific team resolver
        _champ_teams: set[str] | None = None
        if league == "EFL":
            try:
                from championship_predict import _resolve_champ_team
                from championship_pipeline import load_championship_data
                champ_df = load_championship_data()
                latest_s = champ_df["SeasonIndex"].max()
                latest = champ_df[champ_df["SeasonIndex"] == latest_s]
                _champ_teams = (set(latest["Home_Team"].unique())
                                | set(latest["Away_Team"].unique()))
            except Exception as e:
                logger.warning(
                    "EFL team resolver failed, using normalize: %s", e)

        def _resolve_team(api_name: str) -> str:
            """Resolve API team name to dataset name for this league."""
            if league == "EFL" and _champ_teams is not None:
                resolved = _resolve_champ_team(api_name, _champ_teams)
                return resolved if resolved else api_name
            return normalize(api_name)

        # Run model predictions if no recent data exists for these fixtures.
        # This ensures clicking "Refresh Odds" always shows model output.
        _scan_fixture_names = set()
        for m in matches:
            _scan_fixture_names.add((
                _resolve_team(m.get("home_team", "")),
                _resolve_team(m.get("away_team", "")),
            ))

        # Check if we already have model predictions for ALL scanned fixtures
        # AND at least one ou15 row for any current fixture. The second check
        # is a Path B safety net: pre-Path-B scans produced ou25 + btts but
        # never ou15 (broken bulk alt_totals call), so per-fixture coverage
        # was misleading. Without this guard, a stale "covered" fixture set
        # blocks Refresh Odds from ever rerunning the predictor — which
        # means the selective alt_totals fetch never gets a chance to fire.
        existing_analysis = get_match_analysis(league)
        _fixtures_with_model: set[tuple] = set()
        _has_any_current_ou15 = False
        _current_fixture_keys = set(_scan_fixture_names)
        if not existing_analysis.empty:
            for _, _ea in existing_analysis.iterrows():
                if pd.notna(_ea.get("model_prob")):
                    _fixtures_with_model.add(
                        (_ea["home_team"], _ea["away_team"]))
                # ou15 rows for any current fixture means Path B selective
                # fetch has already considered the slate. We don't require
                # ou15 for *every* current fixture (selective fetch
                # legitimately skips low-conviction ones).
                if (_ea.get("market") == "ou15"
                    and (_ea["home_team"], _ea["away_team"])
                        in _current_fixture_keys):
                    _has_any_current_ou15 = True
        _missing_fixtures = _scan_fixture_names - _fixtures_with_model
        _has_model_data = (
            len(_missing_fixtures) == 0 and _has_any_current_ou15
        )

        if not _has_model_data:
            try:
                logger.info(
                    "Missing model data for %d/%d fixtures — running "
                    "predictor...", len(_missing_fixtures),
                    len(_scan_fixture_names))
                # Freshness Gate (ADR 0005) before either branch, because both
                # retrain inline when load_trained_state() fails — a scan is a
                # Data Refresh too, which ADR 0005 did not anticipate.
                from freshness import assert_fresh
                assert_fresh(league)

                if league == "EFL":
                    from championship_predict import ChampionshipPredictor
                    _predictor = ChampionshipPredictor(verbose=False)
                    if not _predictor.load_trained_state():
                        _predictor.load_data()
                        _predictor.train()
                        _predictor.save_trained_state()
                    else:
                        if not _predictor.load_pipeline_cache():
                            _predictor.load_data()
                            _predictor.save_pipeline_cache()
                    _recs = _predictor.generate_recommendations()
                else:
                    from predict import LivePredictor
                    _predictor = LivePredictor(verbose=False)
                    if not _predictor.load_trained_state():
                        _predictor.load_data()
                        _predictor.train()
                        _predictor.save_trained_state()
                    else:
                        if not _predictor.load_pipeline_cache():
                            _predictor.load_data()
                            _predictor.save_pipeline_cache()
                    _recs = _predictor.generate_recommendations(
                        prefetched_matches=matches,
                    )

                if _recs:
                    save_recommendations(_recs, league=league)
                _analysis = getattr(_predictor, "_match_analysis", [])
                if _analysis:
                    save_match_analysis(_analysis, league=league)
                    log_predictions(_analysis, league=league)
                logger.info(
                    "Predictor run complete: %d recs, %d analysis rows",
                    len(_recs), len(_analysis),
                )
            except FreshnessError as e:
                # Caught before the generic handler so the reason survives.
                # "Predictor run during scan failed" buries the one thing the
                # operator needs: which fixtures are missing, and that odds
                # below are fine while recommendations are not.
                logger.error(
                    "Freshness Gate blocked %s — no recommendations this scan. "
                    "%s", league, e,
                )
            except Exception as e:
                logger.error("Predictor run during scan failed: %s", e,
                             exc_info=True)

        # Load model predictions from two sources:
        # 1. recommendations table (positive-edge bets only)
        # 2. match_analysis table (all rows from prior predictor runs)
        # This ensures model_prob is carried forward even for negative-edge
        # rows.
        model_prob_lookup: dict[tuple, float] = {}
        edge_source_lookup: dict[tuple, str] = {}

        # Source 1: existing match_analysis rows (from predictor runs)
        existing_analysis = get_match_analysis(league)
        if not existing_analysis.empty:
            for _, row in existing_analysis.iterrows():
                key = (row["home_team"], row["away_team"],
                       row["market"], row["side"])
                if pd.notna(row.get("model_prob")):
                    model_prob_lookup[key] = row["model_prob"]
                if pd.notna(row.get("edge_source")):
                    edge_source_lookup[key] = row["edge_source"]

        # Source 2: recommendations table (overrides with latest if available)
        rec_df = get_active_recommendations(league)
        rec_lookup: dict[tuple, dict] = {}
        if not rec_df.empty:
            for _, row in rec_df.iterrows():
                key = (row["home_team"], row["away_team"],
                       row["market"], row["side"])
                rec_lookup[key] = row.to_dict()
                if pd.notna(row.get("model_prob")):
                    model_prob_lookup[key] = row["model_prob"]

        def _extract_bookmaker_odds(match_data: dict,
                                     market: str, side: str) -> dict:
            """Extract per-bookmaker odds for a specific market+side.

            Includes ALL bookmakers with valid odds (> 1.0).
            For O/U markets, checks both the default line (2.5) and
            alt lines via all_lines. For BTTS, uses btts_bookmakers.
            """
            book_odds = {}
            if market == "btts":
                for bk, bm in match_data.get("btts_bookmakers", {}).items():
                    title = bm.get("title", bk)
                    val = bm.get("yes" if side == "yes" else "no")
                    if val and val > 1:
                        book_odds[title] = round(val, 2)
            else:
                # O/U market — determine the goal line from market code
                line = 2.5  # default
                if market == "ou15":
                    line = 1.5
                elif market == "ou35":
                    line = 3.5
                elif market == "ou45":
                    line = 4.5

                side_key = "over" if side == "over" else "under"
                for bk, bm in match_data.get("bookmakers", {}).items():
                    title = bm.get("title", bk)
                    # Check all_lines for the specific line
                    all_lines = bm.get("all_lines", {})
                    line_data = all_lines.get(line) or all_lines.get(
                        str(line))
                    if line_data:
                        val = line_data.get(side_key)
                        if val and val > 1:
                            book_odds[title] = round(val, 2)
                    elif line == 2.5:
                        # Fallback to top-level over/under for 2.5
                        val = bm.get(side_key)
                        if val and val > 1:
                            book_odds[title] = round(val, 2)
            return book_odds

        rows = []
        for match in matches:
            home_raw = match.get("home_team", "")
            away_raw = match.get("away_team", "")
            home = _resolve_team(home_raw)
            away = _resolve_team(away_raw)
            kickoff = match.get("commence_time", "")

            # ── O/U 2.5 ──
            best_ou = get_best_odds(match)
            if best_ou:
                raw_over_imp = 1.0 / best_ou["best_over"]
                raw_under_imp = 1.0 / best_ou["best_under"]
                overround = raw_over_imp + raw_under_imp
                fair_over = raw_over_imp / overround
                fair_under = raw_under_imp / overround

                for side, odds, book, fair_p in [
                    ("over", best_ou["best_over"],
                     best_ou["best_over_book"], fair_over),
                    ("under", best_ou["best_under"],
                     best_ou["best_under_book"], fair_under),
                ]:
                    fair_odds_val = 1 / fair_p if fair_p > 0 else None

                    rec = rec_lookup.get((home, away, "ou25", side))
                    model_p = (rec["model_prob"] if rec else
                               model_prob_lookup.get(
                                   (home, away, "ou25", side)))
                    edge = ((model_p - fair_p) * 100
                            if model_p else None)
                    conf = None
                    if rec:
                        conf = rec.get("confidence")
                    if not conf and edge is not None:
                        conf = ("high" if edge > 4 else
                                "medium" if edge > 2.5 else
                                "low" if edge > 0 else "negative")

                    rows.append({
                        "home_team": home, "away_team": away,
                        "kickoff": kickoff, "market": "ou25",
                        "side": side,
                        "best_odds": odds, "best_bookmaker": book,
                        "model_prob": model_p,
                        "fair_odds": fair_odds_val,
                        "edge_pct": edge, "confidence": conf,
                        "edge_source": edge_source_lookup.get(
                            (home, away, "ou25", side)),
                        "n_books": best_ou.get("n_books"),
                        "bookmaker_odds": _extract_bookmaker_odds(
                            match, "ou25", side),
                    })

            # ── BTTS ──
            best_btts = get_best_btts_odds(match)
            if best_btts:
                raw_yes_imp = 1.0 / best_btts["best_yes"]
                raw_no_imp = 1.0 / best_btts["best_no"]
                overround = raw_yes_imp + raw_no_imp
                fair_yes = raw_yes_imp / overround
                fair_no = raw_no_imp / overround

                for side, odds, book, fair_p in [
                    ("yes", best_btts["best_yes"],
                     best_btts["best_yes_book"], fair_yes),
                    ("no", best_btts["best_no"],
                     best_btts["best_no_book"], fair_no),
                ]:
                    fair_odds_val = 1 / fair_p if fair_p > 0 else None

                    rec = rec_lookup.get((home, away, "btts", side))
                    model_p = (rec["model_prob"] if rec else
                               model_prob_lookup.get(
                                   (home, away, "btts", side)))
                    edge = ((model_p - fair_p) * 100
                            if model_p else None)
                    conf = None
                    if rec:
                        conf = rec.get("confidence")
                    if not conf and edge is not None:
                        conf = ("high" if edge > 4 else
                                "medium" if edge > 2.5 else
                                "low" if edge > 0 else "negative")

                    rows.append({
                        "home_team": home, "away_team": away,
                        "kickoff": kickoff, "market": "btts",
                        "side": side,
                        "best_odds": odds, "best_bookmaker": book,
                        "model_prob": model_p,
                        "fair_odds": fair_odds_val,
                        "edge_source": edge_source_lookup.get(
                            (home, away, "btts", side)),
                        "edge_pct": edge, "confidence": conf,
                        "n_books": best_btts.get("n_books"),
                        "bookmaker_odds": _extract_bookmaker_odds(
                            match, "btts", side),
                    })

            # ── O/U 1.5 (from all_lines data) ──
            all_lines = get_best_odds_all_lines(match)
            ou15 = all_lines.get(1.5)
            if (ou15 and ou15.get("best_over", 0) > 1
                    and ou15.get("best_under", 0) > 1):
                raw_o15 = 1.0 / ou15["best_over"]
                raw_u15 = 1.0 / ou15["best_under"]
                orr = raw_o15 + raw_u15
                fair_o15 = raw_o15 / orr
                fair_u15 = raw_u15 / orr

                for side, odds, book, fair_p in [
                    ("over", ou15["best_over"],
                     ou15["best_over_book"], fair_o15),
                    ("under", ou15["best_under"],
                     ou15["best_under_book"], fair_u15),
                ]:
                    fair_odds_val = 1 / fair_p if fair_p > 0 else None

                    rec = rec_lookup.get((home, away, "ou15", side))
                    model_p = (rec["model_prob"] if rec else
                               model_prob_lookup.get(
                                   (home, away, "ou15", side)))
                    edge = ((model_p - fair_p) * 100
                            if model_p else None)
                    conf = None
                    if rec:
                        conf = rec.get("confidence")
                    if not conf and edge is not None:
                        conf = ("high" if edge > 4 else
                                "medium" if edge > 2.5 else
                                "low" if edge > 0 else "negative")

                    rows.append({
                        "home_team": home, "away_team": away,
                        "kickoff": kickoff, "market": "ou15",
                        "side": side,
                        "best_odds": odds, "best_bookmaker": book,
                        "model_prob": model_p,
                        "fair_odds": fair_odds_val,
                        "edge_pct": edge, "confidence": conf,
                        "edge_source": edge_source_lookup.get(
                            (home, away, "ou15", side)),
                        "n_books": ou15.get("n_books"),
                        "bookmaker_odds": _extract_bookmaker_odds(
                            match, "ou15", side),
                    })

        # Carry forward model predictions for markets the scan didn't cover.
        # The scan only creates rows where the API has odds data, but the
        # model may have predictions for markets the API doesn't list
        # (e.g. O/U 1.5 when only O/U 2.5 odds are available).
        scan_keys = {(r["home_team"], r["away_team"],
                      r["market"], r["side"]) for r in rows}
        scan_fixtures = {(r["home_team"], r["away_team"]) for r in rows}

        # Check recommendations for predictions not in scan
        for key, rec in rec_lookup.items():
            if key not in scan_keys and (key[0], key[1]) in scan_fixtures:
                home, away, market, side = key
                model_p = rec.get("model_prob")
                if model_p is None:
                    continue
                best_odds = rec.get("odds")
                fair_p = (1.0 / best_odds
                          if best_odds and best_odds > 1 else None)
                edge = (model_p - fair_p) * 100 if fair_p else None
                conf = None
                if edge is not None:
                    conf = ("high" if edge > 4 else
                            "medium" if edge > 2.5 else
                            "low" if edge > 0 else "negative")
                rows.append({
                    "home_team": home, "away_team": away,
                    "kickoff": rec.get("kickoff", ""),
                    "market": market, "side": side,
                    "best_odds": best_odds,
                    "best_bookmaker": rec.get("best_bookmaker", ""),
                    "model_prob": model_p,
                    "fair_odds": (1.0 / fair_p
                                  if fair_p and fair_p > 0 else None),
                    "edge_pct": edge, "confidence": conf,
                    "n_books": rec.get("n_books"),
                    "bookmaker_odds": {},
                })

        # Also check existing match_analysis for predictions not in scan
        existing_analysis = get_match_analysis(league)
        if not existing_analysis.empty:
            for _, ea_row in existing_analysis.iterrows():
                key = (ea_row["home_team"], ea_row["away_team"],
                       ea_row["market"], ea_row["side"])
                if (key not in scan_keys
                        and (key[0], key[1]) in scan_fixtures
                        and pd.notna(ea_row.get("model_prob"))):
                    # Don't duplicate if already added from recommendations
                    if key in {(r["home_team"], r["away_team"],
                                r["market"], r["side"]) for r in rows}:
                        continue
                    rows.append({
                        "home_team": key[0], "away_team": key[1],
                        "kickoff": ea_row.get("kickoff", ""),
                        "market": key[2], "side": key[3],
                        "best_odds": ea_row.get("best_odds"),
                        "best_bookmaker": ea_row.get(
                            "best_bookmaker", ""),
                        "model_prob": ea_row["model_prob"],
                        "fair_odds": ea_row.get("fair_odds"),
                        "edge_pct": ea_row.get("edge_pct"),
                        "confidence": ea_row.get("confidence"),
                        "n_books": ea_row.get("n_books"),
                        "bookmaker_odds": {},
                    })

        n_fixtures = len({(r["home_team"], r["away_team"]) for r in rows})
        n_analysis = save_match_analysis(rows, league=league)
        n_predictions = log_predictions(rows, league=league)

        return (f"{league_name} | Scanned {now} | "
                f"{n_fixtures} fixtures, {n_analysis} lines, "
                f"{n_predictions} new predictions logged")

    except Exception as e:
        logger.error("Scan failed for %s: %s", league, e, exc_info=True)
        return f"{league_name} | Scan failed: {str(e)[:60]}"
