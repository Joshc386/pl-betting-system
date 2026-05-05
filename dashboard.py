"""
Unified betting dashboard for O/U and BTTS markets.

Three main views:
  1. Match Centre  — all upcoming fixtures with model vs bookmaker odds
  2. Bet Tracker   — log bets, view open / settled, running P&L
  3. Performance   — bankroll curve, market breakdown, win-rate charts

Supports multiple leagues (Premier League, Championship) via league selector.
Each league has its own SQLite database.
"""
import os
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Kickoff timestamps are stored in UTC (ISO with trailing 'Z'). All UK
# football fixtures kick off in UK local time, so display conversion to
# Europe/London is required — during BST this adds 1 hour, during GMT
# no offset. Otherwise the dashboard shows times 1 hour before actual
# kickoff during BST (the entire football season's first half is BST).
_UK_TZ = ZoneInfo("Europe/London")
_UTC_TZ = ZoneInfo("UTC")
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import (
    Dash, dcc, html, Input, Output, State, ALL, MATCH,
    dash_table, callback_context, no_update,
)
import dash_bootstrap_components as dbc

from league_config import LEAGUES, get_league_config
from scan import run_scan
from db import (
    get_db,
    LEAGUE_DB_PATHS,
    LEAGUE_DISPLAY_NAMES,
    save_match_analysis,
    get_match_analysis,
    log_predictions,
    get_predictions,
    toggle_prediction_taken,
    save_logged_bet,
    get_open_bets,
    get_settled_bets,
    get_all_bets,
    settle_logged_bet,
    fetch_closing_odds_for_logged_bets,
    calculate_logged_bet_clv,
    save_recommendations,
    get_active_recommendations,
    get_settled_recommendations,
    get_all_recommendations,
)

logger = logging.getLogger(__name__)

# Backward-compat alias: internal code and edge_analytics used the
# private name ``_get_db``; keep it available as a thin wrapper.
_get_db = get_db

# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette (dark theme)
# ═══════════════════════════════════════════════════════════════════════════════
_COLOURS = {
    "bg": "#0f0f23",
    "card": "#16213e",
    "header": "#1a1a2e",
    "green": "#1b4332",
    "red": "#3d2020",
    "blue": "#375a7f",
    "text": "#e0e0e0",
    "muted": "#8898aa",
    "accent": "#00d4aa",
    "warn": "#ffd43b",
}


def _format_market(code: str) -> str:
    """Convert market code to display label.

    Args:
        code: Market code like 'ou25', 'ou15', 'btts'.

    Returns:
        Human-readable label like 'O/U 2.5', 'O/U 1.5', 'BTTS'.
    """
    if code == "btts":
        return "BTTS"
    if isinstance(code, str) and code.startswith("ou"):
        try:
            line = int(code[2:]) / 10.0
            return f"O/U {line:.1f}"
        except (ValueError, IndexError):
            return code
    return str(code) if code is not None else ""


def _clv_card(label: str, value: str, colour: str) -> dbc.Card:
    """Create a small summary card for CLV metrics.

    Args:
        label: Card title (e.g. "Mean CLV").
        value: Display value (e.g. "+1.23%").
        colour: One of "green", "red", "blue".
    """
    colour_map = {
        "green": _COLOURS.get("green", "#28a745"),
        "red": _COLOURS.get("red", "#dc3545"),
        "blue": _COLOURS.get("blue", "#0d6efd"),
    }
    text_colour = colour_map.get(colour, _COLOURS.get("text", "#e0e0e0"))

    return dbc.Card([
        dbc.CardBody([
            html.P(label, className="text-muted small mb-1"),
            html.H4(value, style={
                "color": text_colour,
                "fontFamily": "monospace",
                "fontWeight": "bold",
                "marginBottom": "0",
            }),
        ], className="py-2 px-3"),
    ], className="bg-dark border-secondary text-center")


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard UI Components
# ═══════════════════════════════════════════════════════════════════════════════

def _stat_card(title: str, value: str, color: str = "primary",
               subtitle: str | None = None) -> dbc.Card:
    """Single stat card.

    Args:
        title: small uppercase label above the value
        value: the headline number (rendered large + monospace)
        color: bootstrap color class for the value
        subtitle: optional muted text below the value — used for things
            like "95% CI: -7% to +20%" so the eye gets uncertainty info
            alongside the point estimate. Backwards compatible — every
            existing call site that omits it still renders cleanly.
    """
    body = [
        html.H6(title, className="card-title text-muted mb-1",
                 style={"fontSize": "11px", "textTransform": "uppercase",
                        "letterSpacing": "0.5px"}),
        html.H4(value, className=f"text-{color} mb-0",
                 style={"fontFamily": "monospace", "fontWeight": "600"}),
    ]
    if subtitle:
        body.append(html.Div(
            subtitle,
            className="text-muted",
            style={"fontSize": "10px", "fontStyle": "italic",
                   "marginTop": "4px"},
        ))
    return dbc.Card(
        dbc.CardBody(body),
        className="bg-dark border-secondary",
    )


def _format_age(ts_iso: str | None) -> str:
    """Render an ISO timestamp as a short relative-age string.

    Examples: "12m ago", "1h 23m ago", "2d ago", "never".
    Used by the dashboard freshness indicator — keeps wording compact
    enough to fit on one line of the header.
    """
    if not ts_iso:
        return "never"
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        # Drop tz to compare with naive datetime.now()
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        delta = datetime.now() - ts
    except (TypeError, ValueError):
        return "?"
    secs = int(delta.total_seconds())
    if secs < 0:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        rem = mins % 60
        return f"{hours}h {rem}m ago" if rem else f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _cache_timestamp(path: str) -> str | None:
    """Read the ``timestamp`` field from a cache JSON file (or None)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f).get("timestamp")
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def _make_odds_status(league: str) -> html.Div:
    """Top-of-dashboard widget: freshness + monthly API quota.

    Renders one line summarising:
      * when each provider's cache for this league was last refreshed
      * monthly credit usage vs cap for both providers

    Pulled from cache file timestamps + ``api/quota_tracker``. Read-only;
    does not trigger any network calls.
    """
    from api.quota_tracker import read_quota, QUOTA_LIMITS

    # Cache file paths — match the conventions used elsewhere in the project
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = "" if league == "PL" else "_efl"
    odds_cache = os.path.join(proj_dir, "data", f"odds_cache{suffix}.json")
    op_cache = os.path.join(proj_dir, "data", f"oddspapi_cache{suffix}.json")

    odds_age = _format_age(_cache_timestamp(odds_cache))
    op_age = _format_age(_cache_timestamp(op_cache))

    quota = read_quota()
    oa = quota.get("odds_api", {})
    op = quota.get("oddspapi", {})

    def _fmt_quota(used, limit):
        if used is None:
            return f"–/{limit}"
        # Visual warning thresholds — yellow at 70%, red at 90%
        return f"{used}/{limit}"

    def _quota_colour(used, limit):
        if used is None:
            return "text-muted"
        ratio = used / limit
        if ratio >= 0.9:
            return "text-danger"
        if ratio >= 0.7:
            return "text-warning"
        return "text-success"

    oa_used = oa.get("used")
    op_used = op.get("used")

    return html.Div([
        html.Span("Odds last updated: ", className="text-muted small"),
        html.Span(f"Odds API {odds_age}", className="small me-2",
                  style={"fontWeight": "600"}),
        html.Span("·", className="text-muted small me-2"),
        html.Span(f"OddsPapi {op_age}", className="small me-3",
                  style={"fontWeight": "600"}),
        html.Span("│ Quota: ", className="text-muted small"),
        html.Span(f"Odds API {_fmt_quota(oa_used, QUOTA_LIMITS['odds_api'])}",
                  className=f"small me-2 {_quota_colour(oa_used, QUOTA_LIMITS['odds_api'])}",
                  style={"fontWeight": "600"}),
        html.Span("·", className="text-muted small me-2"),
        html.Span(f"OddsPapi {_fmt_quota(op_used, QUOTA_LIMITS['oddspapi'])}",
                  className=f"small {_quota_colour(op_used, QUOTA_LIMITS['oddspapi'])}",
                  style={"fontWeight": "600"}),
    ], className="mb-2", style={"fontFamily": "monospace"})


def _make_stats_row(league: str) -> html.Div:
    """Build summary stat cards for the selected league."""
    all_bets = get_all_bets(league)
    settled = all_bets[all_bets["settled"] == 1] if not all_bets.empty else pd.DataFrame()
    open_bets = all_bets[all_bets["settled"] == 0] if not all_bets.empty else pd.DataFrame()

    total = len(settled)
    if total == 0:
        return dbc.Row([
            dbc.Col(_stat_card("Open Bets", str(len(open_bets)), "info"), md=2),
            dbc.Col(_stat_card("Settled", "0", "secondary"), md=2),
            dbc.Col(_stat_card("Win Rate", "--", "secondary"), md=2),
            dbc.Col(_stat_card("ROI", "--", "secondary"), md=2),
            dbc.Col(_stat_card("P&L", "--", "secondary"), md=2),
            dbc.Col(_stat_card("Bankroll", "--", "secondary"), md=2),
        ], className="g-2 mb-3")

    wins = int(settled["won"].sum())
    win_rate = wins / total
    total_staked = settled["stake"].sum()
    total_profit = settled["profit"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0

    return dbc.Row([
        dbc.Col(_stat_card("Open Bets", str(len(open_bets)), "info"), md=2),
        dbc.Col(_stat_card("Settled", str(total), "secondary"), md=2),
        dbc.Col(_stat_card("Win Rate", f"{win_rate:.1%}",
                           "success" if win_rate > 0.5 else "danger"), md=2),
        dbc.Col(_stat_card("ROI", f"{roi:+.1%}",
                           "success" if roi > 0 else "danger"), md=2),
        dbc.Col(_stat_card("P&L", f"{total_profit:+.2f}",
                           "success" if total_profit > 0 else "danger"), md=2),
        dbc.Col(_stat_card("Total Staked", f"{total_staked:.2f}",
                           "secondary"), md=2),
    ], className="g-2 mb-3")


# ── Match Centre ──

def _match_centre_legend() -> html.Div:
    """Inline colour-coding legend rendered above the Match Centre table.

    Mirrors the rules in the match-centre-table's `style_data_conditional`
    block (currently dashboard.py:1363-1431). When you change a threshold
    or a hex colour there, update the corresponding swatch here so the
    legend stays truthful — there is no programmatic derivation.
    """
    def swatch(bg: str, text: str | None = None) -> html.Span:
        # Filled square in the actual hex used by the table. For Edge tiers
        # the colour is the *text* colour (cells aren't tinted), so we still
        # render a filled swatch for visual clarity rather than a tiny
        # coloured letter.
        style = {
            "display": "inline-block", "width": "12px", "height": "12px",
            "borderRadius": "2px", "backgroundColor": bg,
            "marginRight": "5px", "verticalAlign": "middle",
            "border": "1px solid #2d3a4a",
        }
        return html.Span(style=style)

    def item(bg: str, label: str) -> html.Span:
        return html.Span([
            swatch(bg),
            html.Span(label, className="text-muted small me-3",
                      style={"verticalAlign": "middle"}),
        ])

    return html.Div([
        html.Div([
            html.Span("Edge:", className="text-muted small me-2",
                      style={"fontWeight": "600"}),
            item("#00d4aa", ">4%"),
            item("#69db7c", "0–4%"),
            item("#ff6b6b", "negative"),
        ], className="mb-1"),
        html.Div([
            html.Span("Stake:", className="text-muted small me-2",
                      style={"fontWeight": "600"}),
            item("#1a3322", "<1%"),
            item("#1f4a2c", "1–3%"),
            item("#1a5e36", "3–5%"),
            item("#5e4a1a", "≥5% ⚠"),
        ]),
    ], className="text-end")


def _build_match_centre(league: str, show_all: bool = False) -> html.Div:
    """Build the Match Centre view showing all fixtures and markets.

    Features a bookmaker dropdown that defaults to 'Best Edge' (auto-selects
    the bookmaker offering the highest odds / largest edge per row). Selecting
    a specific bookmaker recalculates odds and edge for that book across all
    rows. Edge is always displayed regardless of sign.

    Args:
        league: ``"PL"`` or ``"EFL"`` selected league.
        show_all: When False (default), rows whose ``edge_pct`` is below
            ``config.EDGE_DISPLAY_THRESHOLD`` are filtered out — the
            "useless markets the model says no edge on" the operator
            asked us to suppress. The toggle flips this off to reveal
            every evaluated market for transparency / spot-checking.
    """
    from config import EDGE_DISPLAY_THRESHOLD

    analysis = get_match_analysis(league)

    # Path B display filter — hide low-edge rows by default. The toggle
    # at the top of the dashboard exposes them when the operator wants
    # to see what the model considered and rejected. Rows with NULL
    # edge_pct are kept so brand-new markets without odds yet still show.
    if not show_all and not analysis.empty and "edge_pct" in analysis.columns:
        keep_mask = (
            analysis["edge_pct"].isna()
            | (analysis["edge_pct"] >= EDGE_DISPLAY_THRESHOLD)
        )
        analysis = analysis[keep_mask].reset_index(drop=True)

    if analysis.empty:
        return html.Div([
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "No match data available. Click ",
                html.Strong("Refresh Odds"),
                " to fetch upcoming matches and run the model.",
            ], color="info", className="mt-3"),
        ])

    # Get scan timestamp
    scan_time = analysis["scanned_at"].iloc[0] if not analysis.empty else ""
    try:
        scan_dt = datetime.fromisoformat(scan_time)
        scan_label = scan_dt.strftime("%d %b %Y, %H:%M")
    except (ValueError, TypeError):
        scan_label = scan_time

    # Discover which bookmakers have data — build dropdown options
    # Source 1: per-bookmaker odds JSON (populated by scan with full API data)
    _all_book_names: set[str] = set()
    for _, row in analysis.iterrows():
        bm_json = row.get("bookmaker_odds_json", "")
        if bm_json and isinstance(bm_json, str) and bm_json != "{}":
            try:
                bm = json.loads(bm_json)
                _all_book_names.update(bm.keys())
            except (json.JSONDecodeError, TypeError):
                pass
        # Source 2: best_bookmaker column (always available from recommendations)
        best_bk = row.get("best_bookmaker", "")
        if best_bk and isinstance(best_bk, str) and best_bk.strip():
            _all_book_names.add(best_bk.strip())

    # Preferred display order
    _PREFERRED_ORDER = ["Bet365", "Paddy Power", "William Hill",
                        "Unibet", "Unibet (NL)", "Betfair",
                        "Pinnacle", "Betsson", "1xBet",
                        "BetOnline.ag", "LeoVegas"]
    _DISPLAY_BOOKS = [b for b in _PREFERRED_ORDER if b in _all_book_names]
    _DISPLAY_BOOKS += sorted(b for b in _all_book_names
                             if b not in _DISPLAY_BOOKS)

    # Load existing predictions to check "taken" state
    _predictions_taken: dict[tuple, int] = {}
    try:
        pred_df = get_predictions(league)
        for _, p in pred_df.iterrows():
            _key = (p["home_team"], p["away_team"], p["market"], p["side"])
            _predictions_taken[_key] = int(p.get("taken", 0))
    except Exception:
        pass

    # Load recommendations to flag formally suggested bets.
    # A "recommendation" passed stricter filters (min edge, model agreement,
    # positive EV, Kelly stake > 0) beyond just having a positive edge.
    # We also harvest stake_pct here for the Stake % column — same lookup,
    # one DB hit instead of two.
    _recommended: set[tuple] = set()
    _stake_lookup: dict[tuple, float] = {}
    try:
        rec_df = get_active_recommendations(league)
        settled_rec_df = get_settled_recommendations(league)
        for _rdf in (rec_df, settled_rec_df):
            if not _rdf.empty:
                for _, r in _rdf.iterrows():
                    _key = (r["home_team"], r["away_team"],
                            r["market"], r["side"])
                    _recommended.add(_key)
                    _sp = r.get("stake_pct")
                    if pd.notna(_sp) and _sp is not None:
                        # Active recs win over settled (latest-wins): only
                        # write if no entry yet OR we're now on rec_df.
                        # rec_df iterates first so settled won't overwrite.
                        if _key not in _stake_lookup:
                            _stake_lookup[_key] = float(_sp)
    except Exception:
        pass

    # Build display rows with per-bookmaker odds stored in hidden _bm_odds col
    display_rows = []
    for _, row in analysis.iterrows():
        model_p = row["model_prob"]

        # Parse per-bookmaker odds from JSON
        bm_json = row.get("bookmaker_odds_json", "")
        bm_odds = {}
        if bm_json and isinstance(bm_json, str):
            try:
                bm_odds = json.loads(bm_json)
            except (json.JSONDecodeError, TypeError):
                pass

        # "Best Edge" = bookmaker with highest odds (largest edge)
        if bm_odds:
            best_bk = max(bm_odds, key=lambda k: bm_odds[k])
            best_odds = bm_odds[best_bk]
        else:
            best_bk = row.get("best_bookmaker", "")
            best_odds = row["best_odds"]

        fair_odds = row["fair_odds"]
        fair_odds_from_db = pd.notna(fair_odds) and fair_odds is not None
        # Derive fair odds from model probability for display when DB value is missing
        if not fair_odds_from_db and pd.notna(model_p) and model_p > 0:
            fair_odds = 1.0 / model_p
        if pd.notna(best_odds) and best_odds > 1:
            implied_p = 1.0 / best_odds
            # Edge calc: use DB fair_odds if available, otherwise bookmaker implied prob
            if fair_odds_from_db:
                fair_p = 1.0 / fair_odds
                # Detect corrupt fair_odds (stored as 1/model_prob instead of
                # de-vigged bookmaker odds). If fair_p ≈ model_p, fair_odds is
                # wrong — fall back to raw implied prob from bookmaker odds.
                if (pd.notna(model_p)
                        and abs(fair_p - model_p) < 0.001):
                    fair_p = implied_p
                    fair_odds = 1.0 / implied_p
            else:
                fair_p = implied_p
            edge = (model_p - fair_p) * 100 if pd.notna(model_p) else None
        else:
            edge = row["edge_pct"]

        # Always assign confidence — even for negative edges
        conf = row.get("confidence", "") or ""
        if not conf and edge is not None:
            if edge > 4:
                conf = "high"
            elif edge > 2.5:
                conf = "medium"
            elif edge > 0:
                conf = "low"
            else:
                conf = "negative"

        entry = {
            "fixture": f"{row['home_team']} v {row['away_team']}",
            "kickoff": _format_kickoff(row.get("kickoff", "")),
            "market": _format_market(row["market"]),
            "side": row["side"].capitalize(),
            "odds": round(best_odds, 2) if pd.notna(best_odds) else None,
            "bookmaker": best_bk,
            "model_prob": round(model_p * 100, 1) if pd.notna(model_p) else None,
            "fair_odds": round(fair_odds, 2) if pd.notna(fair_odds) else None,
            "edge": round(edge, 1) if pd.notna(edge) else None,
            # stake_pct is stored as a fraction (0.0184 = 1.84% of
            # bankroll); convert to percentage for display so the bin
            # filter queries (>= 1, >= 3, >= 5) match user-facing units.
            "stake_pct": (
                round(_stake_lookup[(row["home_team"], row["away_team"],
                                     row["market"], row["side"])] * 100.0, 2)
                if (row["home_team"], row["away_team"],
                    row["market"], row["side"]) in _stake_lookup
                else None
            ),
            "confidence": conf,
            "n_books": int(row["n_books"]) if pd.notna(row.get("n_books")) else None,
            # Hidden: bookmaker odds JSON for callback recalculation
            "_bm_odds": json.dumps(bm_odds) if bm_odds else "{}",
            # Hidden: original best bookmaker/odds (survives filter switching)
            "_orig_bookmaker": best_bk,
            "_orig_odds": round(best_odds, 2) if pd.notna(best_odds) else None,
            # Hidden fields for bet logging
            "_home": row["home_team"],
            "_away": row["away_team"],
            "_market": row["market"],
            "_side": row["side"],
            "_odds": round(best_odds, 2) if pd.notna(best_odds) else None,
            "_kickoff": row.get("kickoff", ""),
            "_model_prob": model_p if pd.notna(model_p) else None,
            "_fair_odds": float(fair_odds) if pd.notna(fair_odds) else None,
            "rec": ("\u2713" if (row["home_team"], row["away_team"],
                                row["market"], row["side"]) in _recommended
                    else ""),
            "taken": ("Yes" if _predictions_taken.get(
                (row["home_team"], row["away_team"],
                 row["market"], row["side"])) else ""),
        }

        display_rows.append(entry)

    # Default sort: best edges first
    display_rows.sort(key=lambda r: r["edge"] if r["edge"] is not None else -9999, reverse=True)

    from dash.dash_table.Format import Format, Scheme, Sign

    cols = [
        {"name": "Fixture", "id": "fixture", "type": "text"},
        {"name": "Kickoff", "id": "kickoff", "type": "text"},
        {"name": "Market", "id": "market", "type": "text"},
        {"name": "Side", "id": "side", "type": "text"},
        {"name": "Odds", "id": "odds", "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "Bookmaker", "id": "bookmaker", "type": "text"},
        {"name": "Model %", "id": "model_prob", "type": "numeric",
         "format": Format(precision=1, scheme=Scheme.fixed).symbol_suffix("%")},
        {"name": "Fair Odds", "id": "fair_odds", "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.fixed)},
        {"name": "Edge %", "id": "edge", "type": "numeric",
         "format": Format(precision=1, scheme=Scheme.fixed, sign=Sign.positive)
                  .symbol_suffix("%")},
        {"name": "Stake %", "id": "stake_pct", "type": "numeric",
         "format": Format(precision=2, scheme=Scheme.fixed).symbol_suffix("%")},
        {"name": "Conf", "id": "confidence", "type": "text"},
        {"name": "Books", "id": "n_books", "type": "numeric"},
        {"name": "Rec", "id": "rec", "type": "text"},
        {"name": "Taken", "id": "taken", "type": "text",
         "editable": True, "presentation": "dropdown"},
    ]

    # Bookmaker dropdown options: individual books only (empty = best edge)
    book_options = [{"label": b, "value": b} for b in _DISPLAY_BOOKS]

    return html.Div([
        # Controls row: scan info, bookmaker dropdown, sort, filter
        dbc.Row([
            dbc.Col([
                html.Span(f"Last scan: {scan_label}", className="text-muted small"),
                html.Span(f" | {len(analysis)} market lines across "
                          f"{analysis[['home_team','away_team']].drop_duplicates().shape[0]} fixtures",
                          className="text-muted small ms-2"),
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Label("Bookmaker:", className="text-muted small me-2",
                               style={"display": "inline-block",
                                      "verticalAlign": "middle"}),
                    dcc.Dropdown(
                        id="bookmaker-dropdown",
                        options=book_options,
                        value=[],
                        multi=True,
                        searchable=True,
                        placeholder="Best Edge (all)",
                        className="sort-dropdown",
                        style={"width": "340px", "display": "inline-block",
                               "verticalAlign": "middle",
                               "color": "white", "backgroundColor": "#1a1a2e"},
                    ),
                    dbc.Button(
                        "Reset", id="bookmaker-reset-btn", size="sm",
                        color="secondary", outline=True,
                        className="ms-2",
                        style={"verticalAlign": "middle", "height": "36px"},
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], width=4),
            dbc.Col([
                html.Div([
                    html.Label("Sort by:", className="text-muted small me-2",
                               style={"display": "inline-block", "verticalAlign": "middle"}),
                    dcc.Dropdown(
                        id="match-sort-dropdown",
                        options=[
                            {"label": "Edge: High \u2192 Low", "value": "edge_desc"},
                            {"label": "Edge: Low \u2192 High", "value": "edge_asc"},
                            {"label": "Model %: High \u2192 Low", "value": "model_desc"},
                            {"label": "Odds: High \u2192 Low", "value": "odds_desc"},
                            {"label": "Kickoff Time", "value": "kickoff_asc"},
                            {"label": "Fixture A \u2192 Z", "value": "fixture_asc"},
                        ],
                        value="edge_desc",
                        clearable=False,
                        className="sort-dropdown",
                        style={"width": "210px", "display": "inline-block",
                               "verticalAlign": "middle",
                               "color": "white", "backgroundColor": "#1a1a2e"},
                    ),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "flex-end"}),
            ], width=3),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("All", id="filter-all", color="outline-light",
                               size="sm", active=True),
                    dbc.Button("Edges Only", id="filter-edges",
                               color="outline-success", size="sm"),
                ]),
            ], width=3, className="text-end"),
        ], className="mb-3 align-items-center"),

        # Colour-coding legend — right-aligned, sits between the controls row
        # and the table. See `_match_centre_legend` for the swatch source of
        # truth (kept in sync by hand with `style_data_conditional` below).
        dbc.Row([
            dbc.Col(_match_centre_legend()),
        ], className="mb-2"),

        # Hidden store for the full unfiltered data (used by Edges Only toggle)
        dcc.Store(id="match-centre-full-data", data=display_rows),
        html.Div(id="taken-persist-output", style={"display": "none"}),

        # Main table
        dash_table.DataTable(
            id="match-centre-table",
            data=display_rows,
            columns=cols,
            row_selectable="single",
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "left", "padding": "10px 12px",
                "fontFamily": "'JetBrains Mono', monospace", "fontSize": "13px",
                "border": "1px solid #2d3a4a",
            },
            style_header={
                "backgroundColor": _COLOURS["header"], "color": "white",
                "fontWeight": "bold", "border": "1px solid #2d3a4a",
                "fontSize": "12px", "textTransform": "uppercase",
                "letterSpacing": "0.5px",
            },
            style_data={
                "backgroundColor": _COLOURS["card"], "color": _COLOURS["text"],
            },
            style_data_conditional=[
                # Strong positive edge (>4%) — bright cyan, bold
                {
                    "if": {"filter_query": "{edge} > 4",
                           "column_id": "edge"},
                    "color": "#00d4aa", "fontWeight": "bold",
                },
                # Moderate positive edge (0-4%) — green
                {
                    "if": {"filter_query": "{edge} > 0 && {edge} <= 4",
                           "column_id": "edge"},
                    "color": "#69db7c",
                },
                # Negative edge — red
                {
                    "if": {"filter_query": "{edge} < 0",
                           "column_id": "edge"},
                    "color": "#ff6b6b",
                },
                # Recommended bet marker — green checkmark
                {
                    "if": {"filter_query": '{rec} = "\u2713"',
                           "column_id": "rec"},
                    "color": "#69db7c", "fontWeight": "bold", "fontSize": "16px",
                },
                # Stake bins -- only colour cells with a value (None renders
                # blank, so non-recommended bets stay neutral).
                # Tier 1: tiny conviction (<1%) -- faint green hint.
                {
                    "if": {"filter_query": "{stake_pct} > 0 && {stake_pct} < 1",
                           "column_id": "stake_pct"},
                    "backgroundColor": "#1a3322",
                    "color": "#a3e6b4",
                },
                # Tier 2: standard recommendation (1-3%) -- clear green.
                {
                    "if": {"filter_query": "{stake_pct} >= 1 && {stake_pct} < 3",
                           "column_id": "stake_pct"},
                    "backgroundColor": "#1f4a2c",
                    "color": "#69db7c",
                    "fontWeight": "bold",
                },
                # Tier 3: high conviction (3-5%) -- deeper green.
                {
                    "if": {"filter_query": "{stake_pct} >= 3 && {stake_pct} < 5",
                           "column_id": "stake_pct"},
                    "backgroundColor": "#1a5e36",
                    "color": "#00d4aa",
                    "fontWeight": "bold",
                },
                # Tier 4: very high (>=5%) -- amber warning. Kelly rarely
                # asks for this much; worth a manual sanity check.
                {
                    "if": {"filter_query": "{stake_pct} >= 5",
                           "column_id": "stake_pct"},
                    "backgroundColor": "#5e4a1a",
                    "color": "#ffd966",
                    "fontWeight": "bold",
                },
                # High confidence row highlight
                {
                    "if": {"filter_query": '{confidence} = "high"'},
                    "backgroundColor": "#1a2e1a",
                },
                # Alternating row colours
                {
                    "if": {"row_index": "odd"},
                    "backgroundColor": "#1a2640",
                },
            ],
            style_cell_conditional=[
                {"if": {"column_id": "fixture"}, "width": "220px"},
                {"if": {"column_id": "kickoff"}, "width": "145px"},
                {"if": {"column_id": "market"}, "width": "80px", "textAlign": "center"},
                {"if": {"column_id": "side"}, "width": "70px", "textAlign": "center"},
                {"if": {"column_id": "odds"}, "width": "80px", "textAlign": "center"},
                {"if": {"column_id": "bookmaker"}, "width": "130px"},
                {"if": {"column_id": "model_prob"}, "width": "90px", "textAlign": "center"},
                {"if": {"column_id": "fair_odds"}, "width": "80px", "textAlign": "center"},
                {"if": {"column_id": "edge"}, "width": "70px", "textAlign": "center"},
                {"if": {"column_id": "stake_pct"}, "width": "80px", "textAlign": "center"},
                {"if": {"column_id": "confidence"}, "width": "60px", "textAlign": "center"},
                {"if": {"column_id": "n_books"}, "width": "60px", "textAlign": "center"},
                {"if": {"column_id": "rec"}, "width": "45px", "textAlign": "center"},
                {"if": {"column_id": "taken"}, "width": "70px", "textAlign": "center"},
            ],
            dropdown={
                "taken": {
                    "options": [
                        {"label": "Yes", "value": "Yes"},
                        {"label": "", "value": ""},
                    ],
                    "clearable": True,
                },
            },
            style_filter={
                "backgroundColor": "white", "color": "black",
            },
            filter_action="native",
            sort_action="none",
            page_size=50,
            hidden_columns=["_bm_odds", "_orig_bookmaker", "_orig_odds",
                            "_home", "_away", "_market", "_side",
                            "_odds", "_kickoff", "_model_prob", "_fair_odds"],
        ),
    ])


def _format_kickoff(kickoff_str: str) -> str:
    """Format kickoff datetime to short display string in UK local time.

    Stored kickoffs are UTC (e.g. '2026-04-11T11:30:00Z'); we convert to
    Europe/London so the displayed clock time matches the actual UK
    kickoff. During BST that's +1 hour; during GMT no offset. The DB
    column itself stays in UTC — only the display layer shifts.
    """
    if not kickoff_str:
        return "--"
    try:
        dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Defensive: rows missing 'Z' are assumed UTC, never naive-local.
            dt = dt.replace(tzinfo=_UTC_TZ)
        dt = dt.astimezone(_UK_TZ)
        return dt.strftime("%a %d %b %H:%M")
    except (ValueError, TypeError):
        return kickoff_str[:16] if len(kickoff_str) > 16 else kickoff_str


# ── Bet Tracker ──

def _build_bet_tracker(league: str) -> html.Div:
    """Build the Bet Tracker view."""
    open_bets = get_open_bets(league)
    settled = get_settled_bets(league)
    clv_stats = calculate_logged_bet_clv(league)

    # CLV summary cards (only show if we have data)
    clv_section = []
    if clv_stats["n_bets"] > 0:
        clv_cards = []
        clv_cards.append(_clv_card(
            "Mean CLV",
            f"{clv_stats['mean_clv_pct']:+.2f}%",
            "green" if clv_stats["mean_clv_pct"] > 0 else "red",
        ))
        clv_cards.append(_clv_card(
            "Beat Close Rate",
            f"{clv_stats['beat_close_rate']:.1f}%",
            "green" if clv_stats["beat_close_rate"] > 50 else "red",
        ))
        if clv_stats["pinnacle_clv_pct"] is not None:
            clv_cards.append(_clv_card(
                "Pinnacle CLV",
                f"{clv_stats['pinnacle_clv_pct']:+.2f}%",
                "green" if clv_stats["pinnacle_clv_pct"] > 0 else "red",
            ))
        if clv_stats["actual_roi"] is not None:
            clv_cards.append(_clv_card(
                "Actual ROI",
                f"{clv_stats['actual_roi']:+.1f}%",
                "green" if clv_stats["actual_roi"] > 0 else "red",
            ))
        clv_cards.append(_clv_card(
            "Settled Bets",
            str(clv_stats["n_bets"]),
            "blue",
        ))

        # Per-market CLV breakdown
        market_items = []
        for mkt, mdata in clv_stats.get("market_clv", {}).items():
            clv_val = mdata["mean_clv_pct"]
            colour = _COLOURS["green"] if clv_val > 0 else _COLOURS["red"]
            market_items.append(
                html.Span([
                    html.Span(f"{_format_market(mkt)}: ",
                              className="text-muted"),
                    html.Span(f"{clv_val:+.2f}%",
                              style={"color": colour, "fontWeight": "bold"}),
                    html.Span(f" ({mdata['n']})", className="text-muted small"),
                ], className="me-4")
            )

        clv_section = [
            dbc.Card([
                dbc.CardHeader(
                    html.H5("Closing Line Value", className="mb-0 text-light"),
                    className="bg-dark border-secondary",
                ),
                dbc.CardBody([
                    dbc.Row([dbc.Col(c, md=True) for c in clv_cards],
                            className="mb-3"),
                    html.Div(market_items, className="mb-2")
                    if market_items else html.Div(),
                    html.P(
                        "Positive CLV = you're consistently getting better "
                        "odds than the closing line. This is the single best "
                        "predictor of long-term profitability.",
                        className="text-muted small mb-0",
                    ),
                ], className="bg-dark"),
            ], className="border-secondary mb-4"),
        ]

    return html.Div([
        # CLV summary (if data exists)
        *clv_section,

        # Log bet form
        dbc.Card([
            dbc.CardHeader(
                html.H5("Log a Bet", className="mb-0 text-light"),
                className="bg-dark border-secondary",
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Home Team", className="text-muted small"),
                        dbc.Input(id="bet-home", type="text",
                                  placeholder="e.g. Arsenal FC",
                                  className="bg-dark text-light border-secondary"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Away Team", className="text-muted small"),
                        dbc.Input(id="bet-away", type="text",
                                  placeholder="e.g. Chelsea FC",
                                  className="bg-dark text-light border-secondary"),
                    ], md=3),
                    dbc.Col([
                        dbc.Label("Market", className="text-muted small"),
                        dbc.Select(
                            id="bet-market",
                            options=[
                                {"label": "O/U 2.5", "value": "ou25"},
                                {"label": "O/U 1.5", "value": "ou15"},
                                {"label": "O/U 3.5", "value": "ou35"},
                                {"label": "O/U 4.5", "value": "ou45"},
                                {"label": "BTTS", "value": "btts"},
                            ],
                            value="ou25",
                            className="bg-dark text-light border-secondary",
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Side", className="text-muted small"),
                        dbc.Select(
                            id="bet-side",
                            options=[
                                {"label": "Over", "value": "over"},
                                {"label": "Under", "value": "under"},
                                {"label": "Yes", "value": "yes"},
                                {"label": "No", "value": "no"},
                            ],
                            value="over",
                            className="bg-dark text-light border-secondary",
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Kickoff", className="text-muted small"),
                        dbc.Input(id="bet-kickoff", type="text",
                                  placeholder="2026-04-12T15:00",
                                  className="bg-dark text-light border-secondary"),
                    ], md=2),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Odds", className="text-muted small"),
                        dbc.Input(id="bet-odds", type="number", step=0.01,
                                  min=1.01, placeholder="e.g. 1.95",
                                  className="bg-dark text-light border-secondary"),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Stake", className="text-muted small"),
                        dbc.Input(id="bet-stake", type="number", step=0.01,
                                  min=0.01, placeholder="e.g. 10.00",
                                  className="bg-dark text-light border-secondary"),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Bookmaker", className="text-muted small"),
                        dbc.Input(id="bet-bookmaker", type="text",
                                  placeholder="e.g. Pinnacle",
                                  className="bg-dark text-light border-secondary"),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Potential Return", className="text-muted small"),
                        html.Div(id="bet-return-display",
                                 className="text-success",
                                 style={"fontSize": "20px", "fontFamily": "monospace",
                                        "paddingTop": "4px"}),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("Notes", className="text-muted small"),
                        dbc.Input(id="bet-notes", type="text",
                                  placeholder="Optional",
                                  className="bg-dark text-light border-secondary"),
                    ], md=2),
                    dbc.Col([
                        dbc.Label("\u00a0", className="small"),  # spacer
                        html.Div([
                            dbc.Button("Log Bet", id="btn-log-bet",
                                       color="success", className="w-100"),
                        ]),
                    ], md=2),
                ]),
                html.Div(id="bet-log-feedback", className="mt-2"),
            ], className="bg-dark"),
        ], className="border-secondary mb-4"),

        # Open bets
        html.H5(f"Open Bets ({len(open_bets)})", className="text-light mb-3"),
        _make_open_bets_table(open_bets),

        html.Hr(className="border-secondary my-4"),

        # Settled bets
        html.H5(f"Settled Bets ({len(settled)})", className="text-light mb-3"),
        _make_settled_bets_table(settled),
    ])


def _make_open_bets_table(df: pd.DataFrame) -> html.Div:
    """Create table for open (unsettled) logged bets."""
    if df.empty:
        return dbc.Alert("No open bets.", color="secondary", className="text-center")

    display_df = df.copy()
    display_df["fixture"] = display_df["home_team"] + " v " + display_df["away_team"]
    display_df["market_label"] = display_df["market"].apply(_format_market)
    display_df["side_label"] = display_df["side"].str.capitalize()
    display_df["odds_fmt"] = display_df["odds"].round(2)
    display_df["stake_fmt"] = display_df["stake"].round(2)
    display_df["return_fmt"] = (display_df["stake"] * display_df["odds"]).round(2)
    display_df["logged"] = pd.to_datetime(display_df["created_at"]).dt.strftime("%d %b %H:%M")

    cols = [
        {"name": "Fixture", "id": "fixture"},
        {"name": "Market", "id": "market_label"},
        {"name": "Side", "id": "side_label"},
        {"name": "Odds", "id": "odds_fmt"},
        {"name": "Stake", "id": "stake_fmt"},
        {"name": "Pot. Return", "id": "return_fmt"},
        {"name": "Bookmaker", "id": "bookmaker"},
        {"name": "Logged", "id": "logged"},
    ]

    return dash_table.DataTable(
        data=display_df[["fixture", "market_label", "side_label", "odds_fmt",
                         "stake_fmt", "return_fmt", "bookmaker", "logged"]].to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left", "padding": "8px 12px",
            "fontFamily": "monospace", "fontSize": "13px",
            "border": "1px solid #2d3a4a",
        },
        style_header={
            "backgroundColor": _COLOURS["header"], "color": "white",
            "fontWeight": "bold", "border": "1px solid #2d3a4a",
        },
        style_data={"backgroundColor": _COLOURS["card"], "color": _COLOURS["text"]},
    )


def _make_settled_bets_table(df: pd.DataFrame) -> html.Div:
    """Create table for settled logged bets."""
    if df.empty:
        return dbc.Alert("No settled bets yet.", color="secondary", className="text-center")

    display_df = df.copy()
    display_df["fixture"] = display_df["home_team"] + " v " + display_df["away_team"]
    display_df["market_label"] = display_df["market"].apply(_format_market)
    display_df["side_label"] = display_df["side"].str.capitalize()
    display_df["odds_fmt"] = display_df["odds"].round(2)
    display_df["stake_fmt"] = display_df["stake"].round(2)
    display_df["profit_fmt"] = display_df["profit"].apply(
        lambda x: f"{x:+.2f}" if pd.notna(x) else "--"
    )
    display_df["result"] = display_df["won"].map({1: "WON", 0: "LOST"})
    display_df["settled_date"] = pd.to_datetime(
        display_df["settled_at"], errors="coerce"
    ).dt.strftime("%d %b %H:%M")

    # CLV column: show if closing odds exist
    if "closing_odds" in display_df.columns:
        display_df["clv_fmt"] = display_df.apply(
            lambda r: f"{((r['odds'] / r['closing_odds']) - 1) * 100:+.1f}%"
            if pd.notna(r.get("closing_odds")) and r.get("closing_odds", 0) > 1
            else "--",
            axis=1,
        )
    else:
        display_df["clv_fmt"] = "--"

    cols = [
        {"name": "Fixture", "id": "fixture"},
        {"name": "Market", "id": "market_label"},
        {"name": "Side", "id": "side_label"},
        {"name": "Odds", "id": "odds_fmt"},
        {"name": "Close", "id": "close_fmt"},
        {"name": "CLV", "id": "clv_fmt"},
        {"name": "Stake", "id": "stake_fmt"},
        {"name": "Result", "id": "result"},
        {"name": "P&L", "id": "profit_fmt"},
        {"name": "Settled", "id": "settled_date"},
    ]

    # Closing odds column
    if "closing_odds" in display_df.columns:
        display_df["close_fmt"] = display_df["closing_odds"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) and x > 1 else "--"
        )
    else:
        display_df["close_fmt"] = "--"

    return dash_table.DataTable(
        data=display_df[["fixture", "market_label", "side_label", "odds_fmt",
                         "close_fmt", "clv_fmt", "stake_fmt", "result",
                         "profit_fmt", "settled_date"]].to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left", "padding": "8px 12px",
            "fontFamily": "monospace", "fontSize": "13px",
            "border": "1px solid #2d3a4a",
        },
        style_header={
            "backgroundColor": _COLOURS["header"], "color": "white",
            "fontWeight": "bold", "border": "1px solid #2d3a4a",
        },
        style_data={"backgroundColor": _COLOURS["card"], "color": _COLOURS["text"]},
        style_data_conditional=[
            {"if": {"filter_query": '{result} = "WON"'},
             "backgroundColor": _COLOURS["green"], "color": "white"},
            {"if": {"filter_query": '{result} = "LOST"'},
             "backgroundColor": _COLOURS["red"], "color": "white"},
        ],
    )


# ── Performance ──

def _build_performance(league: str) -> html.Div:
    """Build the Performance view with charts and stats."""
    all_bets = get_all_bets(league)
    settled = all_bets[all_bets["settled"] == 1] if not all_bets.empty else pd.DataFrame()

    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Graph(figure=_make_bankroll_chart(settled)),
            ], md=12),
        ]),
        html.Hr(className="border-secondary"),
        # Live ROI vs Phase 4a simulation — the diagnostic for whether
        # the live system tracks the simulated numbers it was deployed on.
        _make_live_vs_sim_panel(league, settled),
        html.Hr(className="border-secondary"),
        dbc.Row([
            dbc.Col([
                _make_market_breakdown(settled),
            ], md=6),
            dbc.Col([
                _make_monthly_breakdown(settled),
            ], md=6),
        ]),
        html.Hr(className="border-secondary"),
        dbc.Row([
            dbc.Col([
                _make_side_breakdown(settled),
            ], md=6),
            dbc.Col([
                _make_edge_source_breakdown(settled),
            ], md=6),
        ]),
    ])


# ── Live ROI vs Phase 4a baseline ──

# Bet count threshold below which we display ROI as low-confidence —
# per-market n_bets must clear this before the drift signal is meaningful.
_LIVE_ROI_MIN_BETS = 20

# Drift threshold: when live ROI is more than this far below baseline
# *and* n_bets >= _LIVE_ROI_MIN_BETS, show the row in red as a Phase 4b
# re-spin trigger candidate.
_LIVE_ROI_DRIFT_TRIGGER_PP = 0.03


def _compute_live_roi_rows(league: str, settled: pd.DataFrame) -> list[dict]:
    """Build per-market rows for the Live ROI vs Simulation table.

    Each row reports the live ROI for that (league, market) cell,
    the Phase 4a simulated ROI baseline, and the drift between them.
    Rows are emitted for *every* market with a baseline entry — even
    those with no live bets — so the operator sees the full slate of
    cells the system was validated on.

    Returns a list of dicts with these keys:
        market, n_bets, win_pct, live_roi, sim_roi, drift_pp, status

    ``status`` is one of:
        "ok"       → enough data, drift is within tolerance (or ahead)
        "drift"    → enough data, drift > trigger threshold (red flag)
        "low_n"    → too few bets to interpret ROI yet
        "no_data"  → no settled bets in this cell at all
        "no_baseline" → live cell with no Phase 4a baseline (rare —
                        e.g. a market we started betting outside the validated set)
    """
    from config import PHASE_4A_BASELINE_ROI

    # Markets to display: all baselines for this league, plus any live
    # markets we have data for that aren't in the baseline.
    baseline_for_league = {
        m: roi for (lg, m), roi in PHASE_4A_BASELINE_ROI.items() if lg == league
    }
    live_markets = (
        set(settled["market"].dropna().unique()) if not settled.empty else set()
    )
    markets = sorted(set(baseline_for_league) | live_markets)

    rows = []
    for mkt in markets:
        sub = settled[settled["market"] == mkt] if not settled.empty else pd.DataFrame()
        n_bets = len(sub)
        win_pct = (sub["won"].sum() / n_bets) if n_bets else None
        total_staked = sub["stake"].sum() if n_bets else 0
        total_profit = sub["profit"].sum() if n_bets else 0
        live_roi = (total_profit / total_staked) if total_staked > 0 else None
        sim_roi = baseline_for_league.get(mkt)
        drift_pp = (live_roi - sim_roi) if (live_roi is not None and sim_roi is not None) else None

        # Status classification — drives the colour and the "should I act" signal.
        if sim_roi is None:
            status = "no_baseline"
        elif n_bets == 0:
            status = "no_data"
        elif n_bets < _LIVE_ROI_MIN_BETS:
            status = "low_n"
        elif drift_pp is not None and drift_pp < -_LIVE_ROI_DRIFT_TRIGGER_PP:
            status = "drift"
        else:
            status = "ok"

        rows.append({
            "market": mkt,
            "n_bets": n_bets,
            "win_pct": win_pct,
            "live_roi": live_roi,
            "sim_roi": sim_roi,
            "drift_pp": drift_pp,
            "status": status,
        })
    return rows


def _make_live_vs_sim_panel(league: str, settled: pd.DataFrame) -> html.Div:
    """Render the Live ROI vs Phase 4a baseline panel."""
    rows = _compute_live_roi_rows(league, settled)

    # Format helpers — handle the None states explicitly so cells read
    # "—" instead of "nan%" or similar
    def _pct(x: float | None, signed: bool = False) -> str:
        if x is None:
            return "—"
        return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"

    # Bright text-friendly shades — _COLOURS["green"]/["red"] are dark
    # backgrounds (used elsewhere for table row fills) so we use lighter
    # foreground variants here for legibility against the card background.
    _TEXT_GREEN = "#51cf66"
    _TEXT_RED = "#ff6b6b"

    def _drift_cell(drift: float | None, status: str) -> html.Span:
        if drift is None:
            return html.Span("—", className="text-muted")
        colour = {
            "ok": _TEXT_GREEN,
            "drift": _TEXT_RED,
            "low_n": _COLOURS["text"],
        }.get(status, _COLOURS["text"])
        return html.Span(_pct(drift, signed=True),
                         style={"color": colour, "fontWeight": "600"})

    def _status_label(status: str) -> html.Span:
        labels = {
            "ok": ("on track", _TEXT_GREEN),
            "drift": ("DRIFT — investigate", _TEXT_RED),
            "low_n": (f"low n (<{_LIVE_ROI_MIN_BETS})", _COLOURS["warn"]),
            "no_data": ("no bets yet", _COLOURS["muted"]),
            "no_baseline": ("no baseline", _COLOURS["warn"]),
        }
        text, colour = labels.get(status, (status, _COLOURS["text"]))
        return html.Span(text,
                         style={"color": colour, "fontSize": "12px",
                                "fontWeight": "600"})

    # Build the table rows
    table_rows = []
    for r in rows:
        table_rows.append(html.Tr([
            html.Td(_format_market(r["market"]),
                    style={"fontWeight": "600"}),
            html.Td(str(r["n_bets"]) if r["n_bets"] else "—",
                    className="text-end"),
            html.Td(_pct(r["win_pct"]) if r["win_pct"] is not None else "—",
                    className="text-end"),
            html.Td(_pct(r["live_roi"], signed=True) if r["live_roi"] is not None else "—",
                    className="text-end",
                    style={"fontFamily": "monospace"}),
            html.Td(_pct(r["sim_roi"], signed=True) if r["sim_roi"] is not None else "—",
                    className="text-end text-muted",
                    style={"fontFamily": "monospace"}),
            html.Td(_drift_cell(r["drift_pp"], r["status"]),
                    className="text-end",
                    style={"fontFamily": "monospace"}),
            html.Td(_status_label(r["status"])),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Market"),
            html.Th("Bets", className="text-end"),
            html.Th("Win%", className="text-end"),
            html.Th("Live ROI", className="text-end"),
            html.Th("Sim ROI", className="text-end"),
            html.Th("Drift", className="text-end"),
            html.Th("Status"),
        ])),
        html.Tbody(table_rows),
    ], color="dark", hover=True, size="sm", className="mb-2")

    # Caveat line: written so a reviewer can interpret the table without
    # context. The two thresholds are config constants — change there.
    caveat = html.P(
        f"Drift = Live ROI − Phase 4a simulated baseline. Cells turn "
        f"red when bets ≥ {_LIVE_ROI_MIN_BETS} *and* drift "
        f"< −{_LIVE_ROI_DRIFT_TRIGGER_PP * 100:.0f}pp — the trigger to "
        f"investigate or schedule a Phase 4b re-spin. Ignore single-game "
        f"swings; this is a sustained-trend diagnostic.",
        className="text-muted small mb-0",
    )

    return dbc.Card([
        dbc.CardHeader(
            html.H5("Live ROI vs Simulation",
                    className="mb-0 text-light"),
            className="bg-dark border-secondary",
        ),
        dbc.CardBody([table, caveat], className="bg-dark"),
    ], className="border-secondary mb-4")


def _make_bankroll_chart(settled: pd.DataFrame) -> go.Figure:
    """Create bankroll / cumulative P&L chart."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_COLOURS["bg"],
        plot_bgcolor=_COLOURS["bg"],
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=20, t=50, b=40),
    )

    if settled.empty:
        fig.update_layout(title="Cumulative P&L (no settled bets yet)")
        return fig

    df = settled.sort_values("settled_at").copy()
    df["cum_profit"] = df["profit"].cumsum()

    fig.add_trace(go.Scatter(
        x=list(range(1, len(df) + 1)),
        y=df["cum_profit"],
        mode="lines",
        name="Cumulative P&L",
        line=dict(color=_COLOURS["accent"], width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.1)",
    ))

    # Per-market breakdown
    market_colors = {
        "ou25": "#4dabf7", "btts": "#ff6b6b", "ou15": "#ffd43b",
        "ou35": "#69db7c", "ou45": "#da77f2",
    }
    for mkt in sorted(df["market"].unique()):
        mkt_df = df[df["market"] == mkt].copy()
        mkt_df["cum_profit"] = mkt_df["profit"].cumsum()
        color = market_colors.get(mkt, "#adb5bd")
        fig.add_trace(go.Scatter(
            x=list(range(1, len(mkt_df) + 1)),
            y=mkt_df["cum_profit"],
            mode="lines",
            name=_format_market(mkt),
            line=dict(color=color, width=1, dash="dot"),
        ))

    fig.update_layout(
        title="Cumulative P&L",
        xaxis_title="Bet #",
        yaxis_title="Profit (units)",
    )
    return fig


def _make_market_breakdown(settled: pd.DataFrame) -> html.Div:
    """Per-market performance breakdown table."""
    if settled.empty:
        return html.Div([
            html.H5("Market Breakdown", className="text-light mb-3"),
            html.P("No settled bets yet.", className="text-muted"),
        ])

    rows = []
    for mkt in sorted(settled["market"].unique()):
        mkt_df = settled[settled["market"] == mkt]
        total_staked = mkt_df["stake"].sum()
        total_profit = mkt_df["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0
        rows.append({
            "Market": _format_market(mkt),
            "Bets": len(mkt_df),
            "Wins": int(mkt_df["won"].sum()),
            "Win %": f"{mkt_df['won'].mean():.1%}",
            "Staked": f"{total_staked:.2f}",
            "P&L": f"{total_profit:+.2f}",
            "ROI": f"{roi:+.1%}",
        })

    return html.Div([
        html.H5("Market Breakdown", className="text-light mb-3"),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": k, "id": k} for k in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center", "padding": "8px",
                "fontFamily": "monospace", "fontSize": "13px",
                "border": "1px solid #2d3a4a",
            },
            style_header={
                "backgroundColor": _COLOURS["header"], "color": "white",
                "fontWeight": "bold",
            },
            style_data={
                "backgroundColor": _COLOURS["card"], "color": _COLOURS["text"],
            },
        ),
    ])


def _make_side_breakdown(settled: pd.DataFrame) -> html.Div:
    """Per-market, per-side performance breakdown table."""
    if settled.empty:
        return html.Div([
            html.H5("Side Breakdown", className="text-light mb-3"),
            html.P("No settled bets yet.", className="text-muted"),
        ])

    rows = []
    for mkt in sorted(settled["market"].unique()):
        mkt_df = settled[settled["market"] == mkt]
        for sd in sorted(mkt_df["side"].dropna().unique()):
            s_df = mkt_df[mkt_df["side"] == sd]
            total_staked = s_df["stake"].sum()
            total_profit = s_df["profit"].sum()
            roi = total_profit / total_staked if total_staked > 0 else 0
            rows.append({
                "Market": _format_market(mkt),
                "Side": sd.title(),
                "Bets": len(s_df),
                "Win %": f"{s_df['won'].mean():.1%}" if len(s_df) > 0 else "—",
                "Staked": f"{total_staked:.2f}",
                "P&L": f"{total_profit:+.2f}",
                "ROI": f"{roi:+.1%}",
            })

    if not rows:
        return html.Div()

    return html.Div([
        html.H5("Side Breakdown", className="text-light mb-3"),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": k, "id": k} for k in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center", "padding": "8px",
                "fontFamily": "monospace", "fontSize": "13px",
                "border": "1px solid #2d3a4a",
            },
            style_header={
                "backgroundColor": _COLOURS["header"], "color": "white",
                "fontWeight": "bold",
            },
            style_data={
                "backgroundColor": _COLOURS["card"], "color": _COLOURS["text"],
            },
        ),
    ])


def _make_edge_source_breakdown(settled: pd.DataFrame) -> html.Div:
    """Performance breakdown by edge source (Pinnacle vs de-vig)."""
    if settled.empty or "edge_source" not in settled.columns:
        return html.Div([
            html.H5("Edge Source Breakdown", className="text-light mb-3"),
            html.P("No edge source data yet.", className="text-muted"),
        ])

    df = settled.dropna(subset=["edge_source"])
    if df.empty:
        return html.Div([
            html.H5("Edge Source Breakdown", className="text-light mb-3"),
            html.P("No edge source data yet.", className="text-muted"),
        ])

    rows = []
    for src in sorted(df["edge_source"].unique()):
        s_df = df[df["edge_source"] == src]
        total_staked = s_df["stake"].sum()
        total_profit = s_df["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0
        rows.append({
            "Source": src.title(),
            "Bets": len(s_df),
            "Win %": f"{s_df['won'].mean():.1%}",
            "Staked": f"{total_staked:.2f}",
            "P&L": f"{total_profit:+.2f}",
            "ROI": f"{roi:+.1%}",
        })

    return html.Div([
        html.H5("Edge Source Breakdown", className="text-light mb-3"),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": k, "id": k} for k in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center", "padding": "8px",
                "fontFamily": "monospace", "fontSize": "13px",
                "border": "1px solid #2d3a4a",
            },
            style_header={
                "backgroundColor": _COLOURS["header"], "color": "white",
                "fontWeight": "bold",
            },
            style_data={
                "backgroundColor": _COLOURS["card"], "color": _COLOURS["text"],
            },
        ),
    ])


def _make_monthly_breakdown(settled: pd.DataFrame) -> html.Div:
    """Monthly P&L breakdown."""
    if settled.empty:
        return html.Div([
            html.H5("Monthly P&L", className="text-light mb-3"),
            html.P("No settled bets yet.", className="text-muted"),
        ])

    df = settled.copy()
    df["month"] = pd.to_datetime(df["settled_at"], errors="coerce").dt.to_period("M")
    df = df.dropna(subset=["month"])

    rows = []
    for month in sorted(df["month"].unique()):
        m_df = df[df["month"] == month]
        total_staked = m_df["stake"].sum()
        total_profit = m_df["profit"].sum()
        roi = total_profit / total_staked if total_staked > 0 else 0
        rows.append({
            "Month": str(month),
            "Bets": len(m_df),
            "W/L": f"{int(m_df['won'].sum())}/{len(m_df) - int(m_df['won'].sum())}",
            "Staked": f"{total_staked:.2f}",
            "P&L": f"{total_profit:+.2f}",
            "ROI": f"{roi:+.1%}",
        })

    return html.Div([
        html.H5("Monthly P&L", className="text-light mb-3"),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": k, "id": k} for k in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={
                "textAlign": "center", "padding": "8px",
                "fontFamily": "monospace", "fontSize": "13px",
                "border": "1px solid #2d3a4a",
            },
            style_header={
                "backgroundColor": _COLOURS["header"], "color": "white",
                "fontWeight": "bold",
            },
            style_data={
                "backgroundColor": _COLOURS["card"], "color": _COLOURS["text"],
            },
        ),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Model Analytics Tab
# ═══════════════════════════════════════════════════════════════════════════════

def _build_analytics(league: str) -> html.Div:
    """Build the Model Analytics view.

    Shows edge validation, calibration, and per-confidence breakdown
    from settled recommendations in the database.
    """
    from edge_analytics import (
        edge_bucket_analysis, calibration_curve,
        confidence_validation, side_analysis,
    )

    sections = []

    # ══════════════════════════════════════════════════════════════════════
    # Prediction Tracking — model accuracy independent of betting
    # ══════════════════════════════════════════════════════════════════════
    try:
        all_preds = get_predictions(league)
        settled_preds = get_predictions(league, settled_only=True)
    except Exception:
        all_preds = pd.DataFrame()
        settled_preds = pd.DataFrame()

    # Build set of recommended (home, away, market, side) tuples from
    # both active and settled recommendations for cross-referencing.
    _rec_keys: set[tuple] = set()
    try:
        for _rdf in (get_active_recommendations(league),
                     get_settled_recommendations(league)):
            if not _rdf.empty:
                for _, _r in _rdf.iterrows():
                    _rec_keys.add((
                        _r["home_team"], _r["away_team"],
                        _r["market"], _r["side"],
                    ))
    except Exception:
        pass

    if not all_preds.empty:
        n_total = len(all_preds)
        n_settled = len(settled_preds)
        n_pending = n_total - n_settled

        if not settled_preds.empty:
            settled_preds["won"] = pd.to_numeric(
                settled_preds["won"], errors="coerce"
            ).fillna(0).astype(int)
            settled_preds["taken"] = pd.to_numeric(
                settled_preds["taken"], errors="coerce"
            ).fillna(0).astype(int)
            settled_preds["edge_pct"] = pd.to_numeric(
                settled_preds["edge_pct"], errors="coerce"
            )

            # Flag predictions that were formally recommended (passed
            # stricter filters: min edge, model agreement, positive EV,
            # Kelly stake > 0) vs just having a positive edge.
            settled_preds["recommended"] = settled_preds.apply(
                lambda r: 1 if (r["home_team"], r["away_team"],
                                r["market"], r["side"]) in _rec_keys
                else 0, axis=1,
            )

            model_hit = settled_preds["won"].mean()
            taken_mask = settled_preds["taken"] == 1
            n_taken = taken_mask.sum()
            n_untaken = (~taken_mask).sum()
            taken_hit = (
                settled_preds.loc[taken_mask, "won"].mean()
                if n_taken > 0 else None
            )
            untaken_hit = (
                settled_preds.loc[~taken_mask, "won"].mean()
                if n_untaken > 0 else None
            )

            rec_mask = settled_preds["recommended"] == 1
            n_rec = rec_mask.sum()
            n_not_rec = (~rec_mask).sum()
            rec_hit = (
                settled_preds.loc[rec_mask, "won"].mean()
                if n_rec > 0 else None
            )
            not_rec_hit = (
                settled_preds.loc[~rec_mask, "won"].mean()
                if n_not_rec > 0 else None
            )

            # Wilson CIs on each hit-rate card so small-n results don't
            # masquerade as proven edges. The card colour stays driven by
            # the point estimate; the CI in the subtitle lets the eye
            # discount uncertain-looking results.
            from edge_analytics import wilson_ci
            model_wins = int(settled_preds["won"].sum())
            mh_lo, mh_hi = wilson_ci(model_wins, n_settled)

            cards = [
                dbc.Col(_stat_card("Predictions", str(n_total), "primary"), width=2),
                dbc.Col(_stat_card("Settled", str(n_settled), "info"), width=2),
                dbc.Col(_stat_card("Pending", str(n_pending), "warning"), width=2),
                dbc.Col(_stat_card(
                    "Model Hit Rate",
                    f"{model_hit:.1%}",
                    "success" if model_hit > 0.5 else "danger",
                    subtitle=f"95% CI: {mh_lo:.0%} – {mh_hi:.0%}",
                ), width=2),
            ]
            if n_rec > 0:
                rec_wins = int(settled_preds.loc[rec_mask, "won"].sum())
                rh_lo, rh_hi = wilson_ci(rec_wins, int(n_rec))
                cards.append(dbc.Col(_stat_card(
                    f"Rec'd ({n_rec})",
                    f"{rec_hit:.1%}",
                    "success" if rec_hit and rec_hit > 0.5 else "danger",
                    subtitle=f"95% CI: {rh_lo:.0%} – {rh_hi:.0%}",
                ), width=2))
            if n_not_rec > 0 and n_rec > 0:
                nr_wins = int(settled_preds.loc[~rec_mask, "won"].sum())
                nh_lo, nh_hi = wilson_ci(nr_wins, int(n_not_rec))
                cards.append(dbc.Col(_stat_card(
                    f"Not Rec'd ({n_not_rec})",
                    f"{not_rec_hit:.1%}",
                    "success" if not_rec_hit and not_rec_hit > 0.5 else "danger",
                    subtitle=f"95% CI: {nh_lo:.0%} – {nh_hi:.0%}",
                ), width=2))

            sections.append(html.Div([
                html.H5("Prediction Tracking", className="text-light mb-2"),
                html.P(
                    "Every positive-edge prediction tracked — independent of "
                    "whether you placed the bet.",
                    className="text-muted small mb-2",
                ),
                dbc.Row(cards, className="mb-3"),
            ]))

            # ══════════════════════════════════════════════════════════════
            # NEW: Strategy Counterfactuals
            # Compares Kelly-rec'd vs flat-rec'd vs flat-all-positive-edge
            # on the same settled outcomes. Directly answers "did the
            # recommendation filter and Kelly sizing each earn their keep?"
            # ══════════════════════════════════════════════════════════════
            try:
                from edge_analytics import counterfactual_strategies
                # Pass settled recs only — counterfactual_strategies
                # filters by settled==1 internally, and using only
                # settled rows here means avg_kelly is a stable
                # historical mean rather than being skewed by whatever
                # active recs happen to be open right now.
                _settled_recs_df = get_settled_recommendations(league)
                cf_df = counterfactual_strategies(all_preds, _settled_recs_df)
            except Exception as exc:
                logger.warning("counterfactual_strategies failed: %s", exc)
                cf_df = pd.DataFrame()

            if not cf_df.empty:
                # Table rows — one per strategy. We render the CIs inline
                # under each ROI as muted small text so the user reads
                # "+6.4% (CI -7% to +20%)" as a single visual unit.
                cf_table_rows = []
                for _, r in cf_df.iterrows():
                    if pd.isna(r["roi"]):
                        roi_cell = "—"
                        wr_cell = "—"
                        pl_cell = "—"
                    else:
                        roi_cell = html.Div([
                            html.Div(f"{r['roi']*100:+.2f}%",
                                     style={"fontWeight": "bold",
                                            "color": "#69db7c" if r["roi"] > 0 else "#ff6b6b"}),
                            html.Div(
                                f"CI {r['roi_lo']*100:+.1f}% to {r['roi_hi']*100:+.1f}%"
                                if not pd.isna(r["roi_lo"]) else "CI —",
                                style={"fontSize": "10px", "color": "#888",
                                       "fontStyle": "italic"},
                            ),
                        ])
                        wr_cell = html.Div([
                            html.Div(f"{r['win_rate']*100:.1f}%",
                                     style={"fontWeight": "bold"}),
                            html.Div(
                                f"CI {r['win_rate_lo']*100:.0f}% – {r['win_rate_hi']*100:.0f}%",
                                style={"fontSize": "10px", "color": "#888",
                                       "fontStyle": "italic"},
                            ),
                        ])
                        pl_cell = html.Div(
                            f"{r['pl_pct']*100:+.2f}%",
                            style={"color": "#69db7c" if r["pl_pct"] > 0 else "#ff6b6b",
                                   "fontWeight": "bold"},
                        )
                    cf_table_rows.append(html.Tr([
                        html.Td(r["strategy"], style={"fontWeight": "500"}),
                        html.Td(int(r["n_bets"]), style={"textAlign": "center"}),
                        html.Td(wr_cell, style={"textAlign": "center"}),
                        html.Td(roi_cell, style={"textAlign": "center"}),
                        html.Td(pl_cell, style={"textAlign": "center"}),
                    ]))

                cf_table = dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Strategy"),
                        html.Th("Bets", style={"textAlign": "center"}),
                        html.Th("Win Rate", style={"textAlign": "center"}),
                        html.Th("ROI (per £ risked)", style={"textAlign": "center"}),
                        html.Th("P/L (bankroll)", style={"textAlign": "center"}),
                    ])),
                    html.Tbody(cf_table_rows),
                ], bordered=True, hover=True, color="dark",
                   className="table-sm mb-2")

                sections.append(html.Div([
                    html.H5("Strategy Counterfactuals",
                            className="text-light mt-4 mb-2"),
                    html.P([
                        "Same settled outcomes, three strategies. ",
                        html.Strong("Row 1 vs Row 2"),
                        " tells you whether Kelly sizing earned its variance. ",
                        html.Strong("Row 2 vs Row 3"),
                        " tells you whether the recommendation filter "
                        "(n_agree, EV-after-vig, Kelly>0) actually picks winners. ",
                        "Differences within overlapping CIs aren't meaningful yet."
                    ], className="text-muted small mb-2"),
                    cf_table,
                ]))

            # ══════════════════════════════════════════════════════════════
            # NEW: Cumulative P/L Over Time
            # Bankroll trajectory from settled recommendations only.
            # Shape (steady slope vs lucky-spike vs decay) reveals whether
            # the headline ROI came from a real edge or a single fluke.
            # ══════════════════════════════════════════════════════════════
            try:
                _settled_recs = get_settled_recommendations(league)
            except Exception:
                _settled_recs = pd.DataFrame()

            if not _settled_recs.empty and "profit_pct" in _settled_recs.columns:
                _sr = _settled_recs.copy()
                _sr["kickoff_dt"] = pd.to_datetime(_sr["kickoff"], errors="coerce")
                _sr = _sr.dropna(subset=["kickoff_dt", "profit_pct"])
                _sr = _sr.sort_values("kickoff_dt").reset_index(drop=True)

                if not _sr.empty:
                    _sr["cum_pl"] = _sr["profit_pct"].cumsum() * 100  # %
                    # Drawdown: max(running peak - current)
                    running_peak = _sr["cum_pl"].cummax()
                    drawdown_series = running_peak - _sr["cum_pl"]
                    max_dd = float(drawdown_series.max()) if len(drawdown_series) else 0.0
                    if max_dd > 0:
                        dd_end_idx = int(drawdown_series.idxmax())
                        dd_start_idx = int(_sr.loc[:dd_end_idx, "cum_pl"].idxmax())
                        dd_start = _sr.loc[dd_start_idx, "kickoff_dt"]
                        dd_end = _sr.loc[dd_end_idx, "kickoff_dt"]
                        dd_label = (
                            f"Max drawdown: -{max_dd:.2f}% "
                            f"({dd_start.strftime('%d %b')} – {dd_end.strftime('%d %b')})"
                        )
                    else:
                        dd_label = "Max drawdown: 0% (no peak yet)"

                    # Counterfactual line: flat-stake all positive edge
                    if not all_preds.empty:
                        _ap = all_preds[
                            (all_preds.get("settled", 0) == 1)
                            & (all_preds["edge_pct"].fillna(-1) > 0)
                        ].copy()
                        _ap["kickoff_dt"] = pd.to_datetime(
                            _ap["kickoff"], errors="coerce")
                        _ap = _ap.dropna(subset=["kickoff_dt"])
                        _ap = _ap.sort_values("kickoff_dt").reset_index(drop=True)
                        if not _ap.empty:
                            avg_kelly_for_cf = (
                                float(_settled_recs["stake_pct"].dropna().mean())
                                if _settled_recs["stake_pct"].notna().any()
                                else 0.01
                            )
                            won_arr = pd.to_numeric(
                                _ap["won"], errors="coerce").fillna(0).to_numpy()
                            odds_arr = pd.to_numeric(
                                _ap["best_odds"], errors="coerce").fillna(0).to_numpy()
                            _ap["profit_cf"] = np.where(
                                won_arr == 1,
                                (odds_arr - 1) * avg_kelly_for_cf,
                                -avg_kelly_for_cf,
                            )
                            _ap["cum_cf"] = _ap["profit_cf"].cumsum() * 100
                        else:
                            _ap = pd.DataFrame()
                    else:
                        _ap = pd.DataFrame()

                    fig_cum = go.Figure()
                    fig_cum.add_trace(go.Scatter(
                        x=_sr["kickoff_dt"], y=_sr["cum_pl"],
                        mode="lines+markers", name="Recommended (Kelly)",
                        line={"color": "#00d4aa", "width": 2.5},
                        marker={"size": 5},
                    ))
                    if not _ap.empty:
                        fig_cum.add_trace(go.Scatter(
                            x=_ap["kickoff_dt"], y=_ap["cum_cf"],
                            mode="lines", name="All +edge (flat)",
                            line={"color": "#888", "width": 1.5, "dash": "dot"},
                        ))
                    fig_cum.add_hline(
                        y=0, line_dash="dash", line_color="#666",
                        annotation_text="0%", annotation_position="right",
                    )
                    fig_cum.update_layout(
                        title="Cumulative P/L Over Time (% of bankroll)",
                        xaxis_title="Kickoff date",
                        yaxis_title="Cumulative P/L %",
                        plot_bgcolor=_COLOURS["card"],
                        paper_bgcolor=_COLOURS["card"],
                        font={"color": _COLOURS["text"]},
                        height=350,
                        margin={"l": 50, "r": 30, "t": 50, "b": 50},
                        legend={"orientation": "h", "y": -0.18},
                        annotations=[{
                            "x": 0.02, "y": 0.95, "xref": "paper", "yref": "paper",
                            "text": dd_label, "showarrow": False,
                            "bgcolor": "#2a1a1a",
                            "bordercolor": "#ff6b6b",
                            "borderwidth": 1, "borderpad": 6,
                            "font": {"size": 11, "color": "#ff9999"},
                        }],
                    )

                    sections.append(html.Div([
                        html.H5("Cumulative P/L Over Time",
                                className="text-light mt-4 mb-2"),
                        html.P(
                            "Bankroll trajectory from settled recommendations. "
                            "Steady upward slope = a real edge being harvested. "
                            "Flat then a single spike = one lucky match. "
                            "Peak then decay = model decaying or market caught on. "
                            "Dotted line is the all-positive-edge counterfactual.",
                            className="text-muted small mb-2",
                        ),
                        dcc.Graph(figure=fig_cum, config={"displayModeBar": False}),
                    ]))

            # ══════════════════════════════════════════════════════════════
            # NEW: Closing Line Value (CLV)
            # Pulled from the same calculator the Bet Tracker uses.
            # Leading indicator of sustainability — stabilises faster than
            # ROI, so it's a more honest read at small samples.
            # ══════════════════════════════════════════════════════════════
            try:
                clv_stats = calculate_logged_bet_clv(league)
            except Exception:
                clv_stats = {"n_bets": 0}

            if clv_stats.get("n_bets", 0) > 0:
                clv_cards = []
                clv_cards.append(_clv_card(
                    "Mean CLV",
                    f"{clv_stats['mean_clv_pct']:+.2f}%",
                    "green" if clv_stats["mean_clv_pct"] > 0 else "red",
                ))
                clv_cards.append(_clv_card(
                    "Beat Close Rate",
                    f"{clv_stats['beat_close_rate']:.1f}%",
                    "green" if clv_stats["beat_close_rate"] > 50 else "red",
                ))
                if clv_stats.get("pinnacle_clv_pct") is not None:
                    clv_cards.append(_clv_card(
                        "Pinnacle CLV",
                        f"{clv_stats['pinnacle_clv_pct']:+.2f}%",
                        "green" if clv_stats["pinnacle_clv_pct"] > 0 else "red",
                    ))
                if clv_stats.get("actual_roi") is not None:
                    clv_cards.append(_clv_card(
                        "Actual ROI",
                        f"{clv_stats['actual_roi']:+.1f}%",
                        "green" if clv_stats["actual_roi"] > 0 else "red",
                    ))
                clv_cards.append(_clv_card(
                    "Settled Bets",
                    str(clv_stats["n_bets"]),
                    "blue",
                ))

                sections.append(html.Div([
                    html.H5("Closing Line Value",
                            className="text-light mt-4 mb-2"),
                    html.P(
                        "Difference between your taken price and the eventual "
                        "market close. Positive CLV is the strongest leading "
                        "indicator that ROI will hold up — it stabilises in "
                        "30-50 bets, while ROI takes 200+. If ROI is flat but "
                        "CLV is positive, you're unlucky, not edgeless.",
                        className="text-muted small mb-2",
                    ),
                    dbc.Row([dbc.Col(c, md=True) for c in clv_cards]),
                ]))

            # Edge bucket breakdown for predictions
            if "edge_pct" in settled_preds.columns:
                bins = [0, 2, 4, 6, 100]
                labels = ["0-2%", "2-4%", "4-6%", "6%+"]
                settled_preds["_edge_bucket"] = pd.cut(
                    settled_preds["edge_pct"], bins=bins, labels=labels,
                    right=False,
                )
                pred_bucket_rows = []
                for label in labels:
                    bucket = settled_preds[settled_preds["_edge_bucket"] == label]
                    if bucket.empty:
                        continue
                    b_rec = bucket[bucket["recommended"] == 1]
                    pred_bucket_rows.append({
                        "edge_bucket": label,
                        "predictions": len(bucket),
                        "hit_rate": bucket["won"].mean(),
                        "recommended": len(b_rec),
                        "rec_hit": (
                            b_rec["won"].mean() if len(b_rec) > 0
                            else None
                        ),
                    })

                if pred_bucket_rows:
                    pred_bucket_df = pd.DataFrame(pred_bucket_rows)
                    # Chart
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Bar(
                        x=pred_bucket_df["edge_bucket"],
                        y=pred_bucket_df["hit_rate"] * 100,
                        name="Model Hit Rate",
                        marker_color=_COLOURS["accent"],
                        text=[f"{r:.1f}%" for r in pred_bucket_df["hit_rate"] * 100],
                        textposition="outside",
                    ))
                    rec_vals = pred_bucket_df["rec_hit"].fillna(0) * 100
                    if pred_bucket_df["recommended"].sum() > 0:
                        fig_pred.add_trace(go.Bar(
                            x=pred_bucket_df["edge_bucket"],
                            y=rec_vals,
                            name="Recommended Hit Rate",
                            marker_color="#69db7c",
                            text=[
                                f"{r:.1f}%" if pd.notna(v) else "--"
                                for r, v in zip(
                                    rec_vals,
                                    pred_bucket_df["rec_hit"],
                                )
                            ],
                            textposition="outside",
                        ))
                    fig_pred.update_layout(
                        title="Prediction Hit Rate by Edge Bucket",
                        xaxis_title="Edge Bucket",
                        yaxis_title="Hit Rate %",
                        barmode="group",
                        plot_bgcolor=_COLOURS["card"],
                        paper_bgcolor=_COLOURS["bg"],
                        font_color=_COLOURS["text"],
                        legend=dict(orientation="h", y=1.1),
                        height=320,
                    )
                    for _, row in pred_bucket_df.iterrows():
                        fig_pred.add_annotation(
                            x=row["edge_bucket"], y=-5,
                            text=f"n={row['predictions']}",
                            showarrow=False,
                            font=dict(size=10, color=_COLOURS["muted"]),
                        )
                    sections.append(html.Div([
                        html.H6("Prediction Edge Analysis",
                                className="text-light mt-3 mb-2"),
                        dcc.Graph(figure=fig_pred),
                    ]))

                    # Table
                    cols = [
                        {"name": "Edge", "id": "edge_bucket"},
                        {"name": "Predictions", "id": "predictions"},
                        {"name": "Hit Rate", "id": "hit_rate",
                         "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "Rec'd", "id": "recommended"},
                        {"name": "Rec'd Hit", "id": "rec_hit",
                         "type": "numeric", "format": {"specifier": ".1%"}},
                    ]
                    sections.append(_make_analytics_table(
                        pred_bucket_df, "pred-bucket-table", columns=cols,
                    ))

            # Market breakdown for predictions
            if "market" in settled_preds.columns:
                mkt_rows = []
                for mkt in sorted(settled_preds["market"].dropna().unique()):
                    mb = settled_preds[settled_preds["market"] == mkt]
                    br = mb[mb["recommended"] == 1]
                    mkt_rows.append({
                        "market": _format_market(mkt),
                        "predictions": len(mb),
                        "hit_rate": mb["won"].mean(),
                        "recommended": len(br),
                        "rec_hit": br["won"].mean() if len(br) > 0 else None,
                    })
                if mkt_rows:
                    mkt_df = pd.DataFrame(mkt_rows)
                    sections.append(html.Div([
                        html.H6("Prediction Market Breakdown",
                                className="text-light mt-4 mb-2"),
                    ]))
                    sections.append(_make_analytics_table(
                        mkt_df, "pred-market-table",
                        columns=[
                            {"name": "Market", "id": "market"},
                            {"name": "Predictions", "id": "predictions"},
                            {"name": "Hit Rate", "id": "hit_rate",
                             "type": "numeric", "format": {"specifier": ".1%"}},
                            {"name": "Rec'd", "id": "recommended"},
                            {"name": "Rec'd Hit", "id": "rec_hit",
                             "type": "numeric", "format": {"specifier": ".1%"}},
                        ],
                    ))

            # ── Fixture-level results ──────────────────────────────────
            fixture_rows = []
            for _, p in settled_preds.iterrows():
                # Parse kickoff to short date in UK local time. Date-only
                # display still cares about timezone for fixtures near
                # midnight UTC (rare but possible) — keeps consistency
                # with `_format_kickoff` above.
                ko = p.get("kickoff", "")
                try:
                    ko_dt = datetime.fromisoformat(
                        str(ko).replace("Z", "+00:00")
                    )
                    if ko_dt.tzinfo is None:
                        ko_dt = ko_dt.replace(tzinfo=_UTC_TZ)
                    ko_dt = ko_dt.astimezone(_UK_TZ)
                    date_str = ko_dt.strftime("%d %b %Y")
                except (ValueError, TypeError):
                    date_str = str(ko)[:10] if ko else ""

                fixture_rows.append({
                    "date": date_str,
                    "fixture": f"{p['home_team']} v {p['away_team']}",
                    "market": _format_market(p["market"]),
                    "side": str(p["side"]).capitalize(),
                    "edge": round(float(p["edge_pct"]), 1) if pd.notna(p["edge_pct"]) else None,
                    "odds": round(float(p["best_odds"]), 2) if pd.notna(p.get("best_odds")) else None,
                    "result": p.get("actual_result", ""),
                    "outcome": "\u2705 Won" if int(p["won"]) == 1 else "\u274c Lost",
                    "rec": "\u2713" if int(p.get("recommended", 0)) == 1 else "",
                    "_won": int(p["won"]),
                    "_kickoff": str(ko),
                })
            if fixture_rows:
                # Sort: most recent date first, then outcome, then edge
                fixture_rows.sort(
                    key=lambda r: (r.get("_kickoff", ""), r.get("_won", 0),
                                   r.get("edge") or 0),
                    reverse=True,
                )
                fixture_df = pd.DataFrame(fixture_rows)
                sections.append(html.Div([
                    html.H6("Settled Predictions — Fixture Detail",
                            className="text-light mt-4 mb-2"),
                    html.P(
                        f"{len(fixture_rows)} settled predictions. "
                        f"Sorted by date, then outcome, then edge.",
                        className="text-muted small",
                    ),
                    dash_table.DataTable(
                        id="pred-fixture-table",
                        data=fixture_df.drop(
                            columns=["_won", "_kickoff"],
                        ).to_dict("records"),
                        columns=[
                            {"name": "Date", "id": "date"},
                            {"name": "Fixture", "id": "fixture"},
                            {"name": "Market", "id": "market"},
                            {"name": "Side", "id": "side"},
                            {"name": "Edge %", "id": "edge", "type": "numeric"},
                            {"name": "Odds", "id": "odds", "type": "numeric"},
                            {"name": "Result", "id": "result"},
                            {"name": "Outcome", "id": "outcome"},
                            {"name": "Rec", "id": "rec"},
                        ],
                        style_table={"overflowX": "auto"},
                        style_header={
                            "backgroundColor": _COLOURS["header"],
                            "color": "white",
                            "fontWeight": "bold",
                        },
                        style_data={
                            "backgroundColor": _COLOURS["card"],
                            "color": _COLOURS["text"],
                        },
                        style_cell={"textAlign": "center", "padding": "8px"},
                        style_cell_conditional=[
                            {"if": {"column_id": "fixture"}, "textAlign": "left",
                             "width": "220px"},
                            {"if": {"column_id": "result"}, "textAlign": "left",
                             "width": "180px"},
                        ],
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": '{outcome} contains "Won"',
                                },
                                "backgroundColor": "#1b4332",
                            },
                            {
                                "if": {
                                    "filter_query": '{outcome} contains "Lost"',
                                },
                                "backgroundColor": "#3d2020",
                            },
                        ],
                        page_size=25,
                        sort_action="native",
                        filter_action="native",
                    ),
                ], className="mb-3"))

        else:
            # Have predictions but none settled yet
            sections.append(html.Div([
                html.H5("Prediction Tracking", className="text-light mb-2"),
                html.P(
                    f"{n_total} predictions logged, awaiting settlement. "
                    "Results will appear after matches complete.",
                    className="text-muted",
                ),
            ]))

        # Divider before bet analytics
        sections.append(html.Hr(
            style={"borderColor": _COLOURS["muted"], "margin": "2rem 0"},
        ))

    # ══════════════════════════════════════════════════════════════════════
    # Bet Analytics (existing) — settled recommendations with stakes
    # ══════════════════════════════════════════════════════════════════════

    # Load settled recommendations
    settled = get_settled_recommendations(league)

    if settled.empty:
        sections.append(html.Div([
            html.H5("Bet Analytics", className="text-light mb-3"),
            html.P("No settled bets yet. Bet analytics will appear after "
                    "bets are settled.", className="text-muted"),
        ]))
        return html.Div(sections)

    # Ensure numeric types
    for col in ["model_prob", "edge", "odds", "stake_pct", "profit_pct"]:
        if col in settled.columns:
            settled[col] = pd.to_numeric(settled[col], errors="coerce")
    if "won" in settled.columns:
        settled["won"] = settled["won"].fillna(0).astype(int)

    # Derive confidence if missing
    if "confidence" not in settled.columns or settled["confidence"].isna().all():
        settled["confidence"] = settled["edge"].apply(
            lambda e: "high" if e and e > 0.04 else
                      "medium" if e and e > 0.025 else "low"
        )

    # ── Summary Stats ──
    n_bets = len(settled)
    win_rate = settled["won"].mean()
    total_staked = settled["stake_pct"].sum()
    total_profit = settled["profit_pct"].sum()
    roi = total_profit / total_staked if total_staked > 0 else 0
    avg_edge = settled["edge"].mean() if "edge" in settled.columns else 0
    avg_odds = settled["odds"].mean() if "odds" in settled.columns else 0

    sections.append(html.Div([
        html.H5("Bet Analytics", className="text-light mb-3"),
        html.P(f"Based on {n_bets} settled bets", className="text-muted mb-2"),
        dbc.Row([
            dbc.Col(_stat_card("Settled", str(n_bets), "primary"), width=2),
            dbc.Col(_stat_card("Win Rate", f"{win_rate:.1%}", "success" if win_rate > 0.5 else "danger"), width=2),
            dbc.Col(_stat_card("ROI", f"{roi:+.1%}", "success" if roi > 0 else "danger"), width=2),
            dbc.Col(_stat_card("Avg Edge", f"{avg_edge:.1%}", "info"), width=2),
            dbc.Col(_stat_card("Avg Odds", f"{avg_odds:.2f}", "info"), width=2),
            dbc.Col(_stat_card("P/L", f"{total_profit:+.2%}", "success" if total_profit > 0 else "danger"), width=2),
        ], className="mb-4"),
    ]))

    # ── Edge Bucket Chart ──
    edge_df = edge_bucket_analysis(settled)
    if not edge_df.empty:
        fig_edge = go.Figure()
        fig_edge.add_trace(go.Bar(
            x=edge_df["bucket"],
            y=edge_df["win_rate"] * 100,
            name="Win Rate %",
            marker_color=_COLOURS["accent"],
            text=[f"{r:.1f}%" for r in edge_df["win_rate"] * 100],
            textposition="outside",
        ))
        fig_edge.add_trace(go.Bar(
            x=edge_df["bucket"],
            y=edge_df["roi"] * 100,
            name="ROI %",
            marker_color=_COLOURS["blue"],
            text=[f"{r:+.1f}%" for r in edge_df["roi"] * 100],
            textposition="outside",
        ))
        fig_edge.update_layout(
            title="Hit Rate & ROI by Edge Bucket",
            xaxis_title="Edge Bucket",
            yaxis_title="%",
            barmode="group",
            plot_bgcolor=_COLOURS["card"],
            paper_bgcolor=_COLOURS["bg"],
            font_color=_COLOURS["text"],
            legend=dict(orientation="h", y=1.1),
            height=350,
        )
        # Add sample size annotations
        for _, row in edge_df.iterrows():
            fig_edge.add_annotation(
                x=row["bucket"], y=-5,
                text=f"n={row['n_bets']}", showarrow=False,
                font=dict(size=10, color=_COLOURS["muted"]),
            )

        sections.append(html.Div([
            html.H6("Edge Validation", className="text-light mt-3 mb-2"),
            html.P("Does higher model edge correspond to higher win rate?",
                    className="text-muted small"),
            dcc.Graph(figure=fig_edge),
        ]))

        # Edge bucket table
        sections.append(_make_analytics_table(
            edge_df, "edge-bucket-table",
            columns=[
                {"name": "Edge Bucket", "id": "bucket"},
                {"name": "Bets", "id": "n_bets"},
                {"name": "Win Rate", "id": "win_rate",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "ROI", "id": "roi",
                 "type": "numeric", "format": {"specifier": "+.1%"}},
                {"name": "Avg Edge", "id": "avg_edge",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "Avg Odds", "id": "avg_odds",
                 "type": "numeric", "format": {"specifier": ".2f"}},
            ],
        ))

    # ── Calibration Chart ──
    # Drops bins with n < 5 (those are pure noise — a single bet can swing
    # actual win rate from 0% to 100%) and renders a Wilson 95% CI as
    # vertical error bars on each surviving point. Wide bars = "ignore
    # this point", tight bars = "this calibration is real".
    cal_df = calibration_curve(settled)
    if not cal_df.empty:
        from edge_analytics import wilson_ci
        # Filter out under-powered bins
        cal_df = cal_df[cal_df["n_bets"] >= 5].reset_index(drop=True)

    if not cal_df.empty:
        fig_cal = go.Figure()
        # Perfect calibration line
        fig_cal.add_trace(go.Scatter(
            x=[0.4, 1.0], y=[0.4, 1.0],
            mode="lines", name="Perfect",
            line=dict(dash="dash", color=_COLOURS["muted"]),
        ))
        # Compute Wilson CIs for error bars. Each bin's actual win rate
        # is wins / n_bets, so we need wins per bin — derive from actual
        # rate × n_bets (both columns already in cal_df).
        ci_lo = []
        ci_hi = []
        for _, r in cal_df.iterrows():
            wins = int(round(r["actual"] * r["n_bets"]))
            lo, hi = wilson_ci(wins, int(r["n_bets"]))
            ci_lo.append(r["actual"] - lo)
            ci_hi.append(hi - r["actual"])

        # Model calibration with error bars
        fig_cal.add_trace(go.Scatter(
            x=cal_df["predicted"],
            y=cal_df["actual"],
            error_y=dict(
                type="data", symmetric=False,
                array=ci_hi, arrayminus=ci_lo,
                color=_COLOURS["accent"], thickness=1.5,
            ),
            mode="lines+markers", name="Model",
            marker=dict(size=cal_df["n_bets"].clip(upper=20),
                        color=_COLOURS["accent"]),
            line=dict(color=_COLOURS["accent"]),
            text=[f"{r['bin_label']}: {r['n_bets']} bets"
                  for _, r in cal_df.iterrows()],
        ))
        fig_cal.update_layout(
            title="Calibration: Predicted vs Actual Win Rate",
            xaxis_title="Model Predicted Probability",
            yaxis_title="Actual Win Rate",
            plot_bgcolor=_COLOURS["card"],
            paper_bgcolor=_COLOURS["bg"],
            font_color=_COLOURS["text"],
            height=350,
            xaxis=dict(range=[0.35, 1.0]),
            yaxis=dict(range=[0.35, 1.0]),
        )

        sections.append(html.Div([
            html.H6("Calibration", className="text-light mt-4 mb-2"),
            html.P("Points on the dashed line = perfectly calibrated. "
                    "Above = model underestimates. Below = model overestimates.",
                    className="text-muted small"),
            dcc.Graph(figure=fig_cal),
        ]))

    # ── Confidence Level Breakdown ──
    conf_df = confidence_validation(settled)
    if not conf_df.empty:
        sections.append(html.Div([
            html.H6("Confidence Level Validation",
                     className="text-light mt-4 mb-2"),
            html.P("Do 'high confidence' bets actually perform better?",
                    className="text-muted small"),
        ]))
        sections.append(_make_analytics_table(
            conf_df, "confidence-table",
            columns=[
                {"name": "Confidence", "id": "confidence"},
                {"name": "Bets", "id": "n_bets"},
                {"name": "Win Rate", "id": "win_rate",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "ROI", "id": "roi",
                 "type": "numeric", "format": {"specifier": "+.1%"}},
                {"name": "Avg Edge", "id": "avg_edge",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "Avg Odds", "id": "avg_odds",
                 "type": "numeric", "format": {"specifier": ".2f"}},
            ],
        ))

    # ── Side Breakdown ──
    side_df = side_analysis(settled)
    if not side_df.empty:
        sections.append(html.Div([
            html.H6("Side Breakdown", className="text-light mt-4 mb-2"),
        ]))
        sections.append(_make_analytics_table(
            side_df, "side-table",
            columns=[
                {"name": "Side", "id": "side"},
                {"name": "Bets", "id": "n_bets"},
                {"name": "Win Rate", "id": "win_rate",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "ROI", "id": "roi",
                 "type": "numeric", "format": {"specifier": "+.1%"}},
                {"name": "Avg Edge", "id": "avg_edge",
                 "type": "numeric", "format": {"specifier": ".1%"}},
                {"name": "Avg Odds", "id": "avg_odds",
                 "type": "numeric", "format": {"specifier": ".2f"}},
            ],
        ))

    # ── Market Breakdown ──
    if "market" in settled.columns:
        from edge_analytics import bootstrap_ci, adequacy_label
        market_rows = []
        for market in sorted(settled["market"].dropna().unique()):
            mb = settled[settled["market"] == market]
            st = mb["stake_pct"].sum()
            pr = mb["profit_pct"].sum()
            # Adequacy badge: per-bet ROI bootstrapped, then flagged by
            # n + CI width. Stops the "+28% on 18 bets" trap.
            stake_arr = mb["stake_pct"].fillna(0).to_numpy(dtype=float)
            profit_arr = mb["profit_pct"].fillna(0).to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                per_bet_roi = profit_arr / np.where(stake_arr > 0, stake_arr, np.nan)
            roi_lo, roi_hi, _ = bootstrap_ci(per_bet_roi)
            adequacy = adequacy_label(len(mb), roi_lo, roi_hi)
            badge = {"ok": "🟢 OK", "marginal": "🟡 Marginal",
                     "noise": "🔴 Noise"}[adequacy]
            market_rows.append({
                "market": _format_market(market),
                "n_bets": len(mb),
                "win_rate": mb["won"].mean(),
                "roi": pr / st if st > 0 else 0,
                "avg_edge": mb["edge"].mean() if "edge" in mb.columns else 0,
                "avg_odds": mb["odds"].mean(),
                "adequacy": badge,
            })
        if market_rows:
            market_df = pd.DataFrame(market_rows)
            sections.append(html.Div([
                html.H6("Market Breakdown", className="text-light mt-4 mb-2"),
                html.P([
                    "Adequacy badge: ",
                    html.Span("🟢 OK", style={"marginRight": "8px"}),
                    "= CI clears zero (real edge detectable). ",
                    html.Span("🟡 Marginal", style={"marginRight": "8px"}),
                    "= n ≥ 30 but CI straddles zero. ",
                    html.Span("🔴 Noise"),
                    " = n < 30 or CI wider than ±15pp.",
                ], className="text-muted small"),
            ]))
            sections.append(_make_analytics_table(
                market_df, "market-analytics-table",
                columns=[
                    {"name": "Market", "id": "market"},
                    {"name": "Bets", "id": "n_bets"},
                    {"name": "Win Rate", "id": "win_rate",
                     "type": "numeric", "format": {"specifier": ".1%"}},
                    {"name": "ROI", "id": "roi",
                     "type": "numeric", "format": {"specifier": "+.1%"}},
                    {"name": "Avg Edge", "id": "avg_edge",
                     "type": "numeric", "format": {"specifier": ".1%"}},
                    {"name": "Avg Odds", "id": "avg_odds",
                     "type": "numeric", "format": {"specifier": ".2f"}},
                    {"name": "Adequacy", "id": "adequacy"},
                ],
            ))

    return html.Div(sections)


def _make_analytics_table(
    df: pd.DataFrame,
    table_id: str,
    columns: list[dict],
) -> html.Div:
    """Create a styled DataTable for analytics display."""
    return html.Div([
        dash_table.DataTable(
            id=table_id,
            data=df.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": _COLOURS["header"],
                "color": "white",
                "fontWeight": "bold",
            },
            style_data={
                "backgroundColor": _COLOURS["card"],
                "color": _COLOURS["text"],
            },
            style_cell={"textAlign": "center", "padding": "8px"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{roi} > 0", "column_id": "roi"},
                    "color": _COLOURS["accent"],
                },
                {
                    "if": {"filter_query": "{roi} < 0", "column_id": "roi"},
                    "color": "#ff6b6b",
                },
            ],
        ),
    ], className="mb-3")


# ═══════════════════════════════════════════════════════════════════════════════
# Build App
# ═══════════════════════════════════════════════════════════════════════════════

def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
        ],
        title="Betting Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    # Dark theme CSS for dropdown components
    app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .sort-dropdown div[class*="control"] {
                background-color: #1a2640 !important;
                border-color: #2d3a4a !important;
            }
            .sort-dropdown div[class*="singleValue"],
            .sort-dropdown div[class*="placeholder"],
            .sort-dropdown div[class*="input"] {
                color: white !important;
            }
            .sort-dropdown div[class*="menu"] {
                background-color: #1a2640 !important;
            }
            .sort-dropdown div[class*="option"] {
                background-color: #1a2640 !important;
                color: white !important;
            }
            .sort-dropdown div[class*="option"]:hover,
            .sort-dropdown div[class*="option--is-focused"] {
                background-color: #2d3a4a !important;
            }
            .sort-dropdown div[class*="indicatorSeparator"] {
                background-color: #2d3a4a !important;
            }
            .sort-dropdown svg {
                fill: white !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

    # League selector tabs
    league_tabs = []
    for key in LEAGUES:
        name = LEAGUE_DISPLAY_NAMES[key]
        league_tabs.append(
            dbc.Tab(label=name, tab_id=f"league-{key}",
                    tab_style={"marginRight": "4px"},
                    active_tab_style={"backgroundColor": _COLOURS["blue"],
                                      "borderColor": _COLOURS["blue"]})
        )

    app.layout = dbc.Container([
        # ── Header ──
        dbc.Row([
            dbc.Col([
                html.H2("Betting Dashboard",
                         className="text-light mb-0",
                         style={"fontWeight": "700"}),
                html.P("O/U 1.5 \u2022 O/U 2.5 \u2022 BTTS \u2022 Multi-League",
                        className="text-muted mb-0", style={"fontSize": "13px"}),
            ], md=4),
            dbc.Col([
                dbc.Tabs(league_tabs, id="league-selector",
                         active_tab="league-PL", className="nav-pills"),
            ], md=4, className="d-flex align-items-center justify-content-center"),
            dbc.Col([
                dbc.Button([html.I(className="bi bi-arrow-clockwise me-1"),
                            "Refresh Odds"],
                           id="btn-scan", color="primary", size="sm",
                           className="me-2",
                           title=("Refresh fixtures + odds for the active "
                                  "league (Odds-API only — OddsPapi week-"
                                  "ahead sweep runs on Sunday).")),
                # Path B display filter — when off (default), the Match
                # Centre hides rows whose edge is below
                # config.EDGE_DISPLAY_THRESHOLD. Toggle on to reveal every
                # evaluated market for transparency.
                dbc.Switch(
                    id="toggle-show-all",
                    label="Show all markets",
                    value=False,
                    className="d-inline-block ms-2 me-2 align-middle",
                    style={"fontSize": "12px"},
                ),
                html.Span(id="status-text", className="text-muted small"),
            ], md=4, className="text-end pt-2"),
        ], className="py-3 border-bottom border-secondary mb-3"),

        # ── Cache freshness + API quota strip ──
        # Updated alongside stats-row whenever the league/scan/interval
        # callback fires. Read-only; never triggers network calls.
        html.Div(id="odds-status-row"),

        # ── Stats row ──
        html.Div(id="stats-row"),

        # ── Content tabs ──
        dbc.Tabs([
            dbc.Tab(label="Match Centre", tab_id="tab-matches",
                    className="bg-dark"),
            dbc.Tab(label="Bet Tracker", tab_id="tab-bets",
                    className="bg-dark"),
            dbc.Tab(label="Performance", tab_id="tab-performance",
                    className="bg-dark"),
            dbc.Tab(label="Model Analytics", tab_id="tab-analytics",
                    className="bg-dark"),
        ], id="content-tabs", active_tab="tab-matches", className="mt-2"),

        # Wrap tab-content in dcc.Loading so any long-running callback
        # (e.g. the first scan after a clean repo when models haven't
        # been pickled yet — full pipeline + train can take 3-5 minutes
        # for EFL) shows a spinner instead of looking frozen.
        dcc.Loading(
            id="tab-content-loading",
            type="circle",
            color="#00d4aa",
            children=html.Div(id="tab-content", className="mt-3"),
        ),

        # ── Auto-refresh interval (30 min) ──
        dcc.Interval(id="interval-refresh", interval=30 * 60 * 1000, n_intervals=0),

        # ── Hidden stores ──
        dcc.Store(id="store-selected-row"),
        dcc.Store(id="bet-model-prob"),
        dcc.Store(id="bet-edge-pct"),
    ], fluid=True, className="bg-dark min-vh-100 pb-5")

    # ══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════════

    @app.callback(
        [Output("tab-content", "children"),
         Output("stats-row", "children"),
         Output("status-text", "children"),
         Output("odds-status-row", "children")],
        [Input("content-tabs", "active_tab"),
         Input("league-selector", "active_tab"),
         Input("btn-scan", "n_clicks"),
         Input("interval-refresh", "n_intervals"),
         Input("toggle-show-all", "value")],
    )
    def update_main(active_tab, league_tab, n_clicks, n_intervals, show_all):
        league = league_tab.replace("league-", "") if league_tab else "PL"
        if league not in LEAGUES:
            league = "PL"

        league_name = LEAGUE_DISPLAY_NAMES.get(league, league)
        triggered = callback_context.triggered_id

        # Handle scan button
        if triggered == "btn-scan" and n_clicks:
            try:
                status = run_scan(league)
            except Exception as e:
                status = f"Scan failed: {str(e)[:80]}"
                logger.error("Scan failed for %s: %s", league, e, exc_info=True)
        else:
            status = f"{league_name} | {datetime.now().strftime('%H:%M:%S')}"

        # Build content
        stats = _make_stats_row(league)
        # Recompute the freshness/quota strip every callback so post-scan
        # refreshes pick up the newly written cache timestamp + quota delta.
        odds_status = _make_odds_status(league)

        if active_tab == "tab-matches":
            # Path B: pass the show_all toggle through so the Match Centre
            # can hide low-edge rows by default (clean view) and expose
            # them when the operator wants transparency.
            content = _build_match_centre(league, show_all=bool(show_all))
        elif active_tab == "tab-bets":
            content = _build_bet_tracker(league)
        elif active_tab == "tab-performance":
            content = _build_performance(league)
        elif active_tab == "tab-analytics":
            content = _build_analytics(league)
        else:
            content = html.P("Select a tab.", className="text-muted")

        return content, stats, status, odds_status

    # ── Reset bookmaker dropdown ──────────────────────────────────────────
    @app.callback(
        Output("bookmaker-dropdown", "value"),
        Input("bookmaker-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_bookmaker_dropdown(_n):
        """Clear bookmaker selection → reverts to Best Edge mode."""
        return []

    # ── Bookmaker dropdown: recalculate odds + edge per selected books ────
    @app.callback(
        Output("match-centre-full-data", "data"),
        [Input("bookmaker-dropdown", "value")],
        [State("match-centre-full-data", "data")],
        prevent_initial_call=True,
    )
    def switch_bookmaker(selected_books, full_data):
        """Recalculate odds/edge when bookmaker selection changes.

        Args:
            selected_books: List of selected bookmaker names (empty = best edge).
            full_data: Current full dataset from hidden Store.
        """
        if not full_data:
            return full_data

        # Normalise: None or single string → list
        if not selected_books:
            selected_books = []
        elif isinstance(selected_books, str):
            selected_books = [selected_books]

        rows = []
        for row in full_data:
            r = dict(row)
            bm_odds = {}
            try:
                bm_odds = json.loads(r.get("_bm_odds", "{}"))
            except (json.JSONDecodeError, TypeError):
                pass

            model_p = r.get("_model_prob")
            fair_odds = r.get("_fair_odds")

            # Original bookmaker and odds from the row (before any filter)
            orig_bk = r.get("_orig_bookmaker") or r.get("bookmaker", "")
            orig_odds = r.get("_orig_odds") or r.get("_odds") or r.get("odds")

            has_per_book_data = bool(bm_odds)

            if not selected_books:
                # Best Edge mode: pick bookmaker with highest odds
                if has_per_book_data:
                    best_bk = max(bm_odds, key=lambda k: bm_odds[k])
                    odds_val = bm_odds[best_bk]
                else:
                    best_bk = orig_bk
                    odds_val = orig_odds
                r["bookmaker"] = best_bk
                r["odds"] = round(odds_val, 2) if odds_val else None
            else:
                # Filter to selected bookmakers, pick best among them
                filtered = {k: v for k, v in bm_odds.items()
                            if k in selected_books}

                # Also check if the row's original best_bookmaker matches
                if orig_bk in selected_books and orig_odds:
                    filtered[orig_bk] = orig_odds

                if filtered:
                    best_bk = max(filtered, key=lambda k: filtered[k])
                    r["bookmaker"] = best_bk
                    r["odds"] = round(filtered[best_bk], 2)
                elif not has_per_book_data:
                    # No per-bookmaker data yet — keep showing the
                    # best-available odds so the row isn't blanked out.
                    # Mark bookmaker as "Best avail" so user knows
                    # it isn't specifically from their selected book.
                    r["bookmaker"] = f"{orig_bk}*"
                    r["odds"] = round(orig_odds, 2) if orig_odds else None
                else:
                    # Per-book data exists but selected books aren't in it
                    r["bookmaker"] = ""
                    r["odds"] = None

            # Recalculate edge using bookmaker's actual odds (remove their
            # overround to get fair implied probability)
            if r["odds"] and r["odds"] > 1 and model_p:
                implied_p = 1.0 / r["odds"]
                r["edge"] = round((model_p - implied_p) * 100, 1)
            else:
                r["edge"] = None

            # Update confidence
            edge = r["edge"]
            if edge is not None:
                if edge > 4:
                    r["confidence"] = "high"
                elif edge > 2.5:
                    r["confidence"] = "medium"
                elif edge > 0:
                    r["confidence"] = "low"
                else:
                    r["confidence"] = "negative"

            # Update hidden _odds for bet form auto-fill
            r["_odds"] = r["odds"]
            rows.append(r)

        return rows

    # ── Sort + filter → updates table data ────────────────────────────────
    @app.callback(
        Output("match-centre-table", "data"),
        [Input("match-centre-full-data", "data"),
         Input("match-sort-dropdown", "value"),
         Input("filter-all", "n_clicks"),
         Input("filter-edges", "n_clicks")],
        prevent_initial_call=True,
    )
    def sort_and_filter_match_centre(full_data, sort_value, _all_clicks, _edge_clicks):
        """Sort and optionally filter the match centre table."""
        if not full_data:
            return []

        rows = list(full_data)

        # Determine which filter button was clicked
        triggered = callback_context.triggered_id
        edges_only = triggered == "filter-edges"

        if edges_only:
            rows = [r for r in rows if r.get("edge") is not None and r["edge"] > 0]

        # Sort
        def _safe(row, col, default=0):
            v = row.get(col)
            return v if v is not None else default

        if not sort_value:
            sort_value = "edge_desc"

        if sort_value == "edge_desc":
            rows.sort(key=lambda r: _safe(r, "edge", -9999), reverse=True)
        elif sort_value == "edge_asc":
            rows.sort(key=lambda r: _safe(r, "edge", -9999))
        elif sort_value == "model_desc":
            rows.sort(key=lambda r: _safe(r, "model_prob", -9999), reverse=True)
        elif sort_value == "odds_desc":
            rows.sort(key=lambda r: _safe(r, "odds", -9999), reverse=True)
        elif sort_value == "kickoff_asc":
            rows.sort(key=lambda r: r.get("_kickoff") or "")
        elif sort_value == "fixture_asc":
            rows.sort(key=lambda r: r.get("fixture", ""))

        return rows

    # ── Toggle active state on filter buttons ─────────────────────────────
    @app.callback(
        [Output("filter-all", "active"),
         Output("filter-edges", "active")],
        [Input("filter-all", "n_clicks"),
         Input("filter-edges", "n_clicks")],
        prevent_initial_call=True,
    )
    def toggle_filter_buttons(_all_clicks, _edge_clicks):
        triggered = callback_context.triggered_id
        if triggered == "filter-edges":
            return False, True
        return True, False

    @app.callback(
        Output("bet-return-display", "children"),
        [Input("bet-odds", "value"),
         Input("bet-stake", "value")],
    )
    def calc_return(odds, stake):
        if odds and stake and odds > 1:
            potential = float(stake) * float(odds)
            profit = potential - float(stake)
            return f"\u00a3{potential:.2f} (+{profit:.2f})"
        return "--"

    @app.callback(
        [Output("bet-log-feedback", "children"),
         Output("bet-home", "value"),
         Output("bet-away", "value"),
         Output("bet-odds", "value"),
         Output("bet-stake", "value"),
         Output("bet-bookmaker", "value"),
         Output("bet-notes", "value")],
        Input("btn-log-bet", "n_clicks"),
        [State("bet-home", "value"),
         State("bet-away", "value"),
         State("bet-market", "value"),
         State("bet-side", "value"),
         State("bet-odds", "value"),
         State("bet-stake", "value"),
         State("bet-bookmaker", "value"),
         State("bet-kickoff", "value"),
         State("bet-notes", "value"),
         State("league-selector", "active_tab"),
         State("bet-model-prob", "data"),
         State("bet-edge-pct", "data")],
        prevent_initial_call=True,
    )
    def log_bet(n_clicks, home, away, market, side, odds, stake,
                bookmaker, kickoff, notes, league_tab, model_prob, edge_pct):
        if not all([home, away, market, side, odds, stake]):
            return (
                dbc.Alert("Please fill in all required fields.",
                          color="warning", duration=3000),
                no_update, no_update, no_update, no_update, no_update, no_update,
            )

        league = league_tab.replace("league-", "") if league_tab else "PL"

        try:
            bet_id = save_logged_bet({
                "home_team": home.strip(),
                "away_team": away.strip(),
                "kickoff": kickoff or "",
                "market": market,
                "side": side,
                "odds": float(odds),
                "stake": float(stake),
                "bookmaker": bookmaker or "",
                "notes": notes or "",
                "model_prob": float(model_prob) if model_prob else None,
                "edge_pct": float(edge_pct) if edge_pct else None,
            }, league=league)

            return (
                dbc.Alert(
                    f"Bet #{bet_id} logged: {home} v {away} | "
                    f"{_format_market(market)} {side} @ {float(odds):.2f} | "
                    f"Stake: {float(stake):.2f}",
                    color="success", duration=5000,
                ),
                "", "", None, None, "", "",  # Clear form
            )
        except Exception as e:
            return (
                dbc.Alert(f"Error logging bet: {e}", color="danger", duration=5000),
                no_update, no_update, no_update, no_update, no_update, no_update,
            )

    # Auto-fill bet form when a match centre row is selected
    @app.callback(
        [Output("bet-home", "value", allow_duplicate=True),
         Output("bet-away", "value", allow_duplicate=True),
         Output("bet-market", "value"),
         Output("bet-side", "value"),
         Output("bet-odds", "value", allow_duplicate=True),
         Output("bet-kickoff", "value"),
         Output("bet-model-prob", "data"),
         Output("bet-edge-pct", "data")],
        Input("match-centre-table", "selected_rows"),
        State("match-centre-table", "data"),
        prevent_initial_call=True,
    )
    def fill_from_selection(selected_rows, table_data):
        if not selected_rows or not table_data:
            return (no_update, no_update, no_update, no_update,
                    no_update, no_update, no_update, no_update)

        row = table_data[selected_rows[0]]
        return (
            row.get("_home", ""),
            row.get("_away", ""),
            row.get("_market", "ou25"),
            row.get("_side", "over"),
            row.get("_odds", ""),
            row.get("_kickoff", ""),
            row.get("_model_prob"),
            row.get("Edge %", "").replace("%", "").replace("+", "")
            if row.get("Edge %") else None,
        )

    # ── Persist "Taken" dropdown changes to predictions DB ────────────
    @app.callback(
        [Output("taken-persist-output", "children"),
         Output("match-centre-full-data", "data", allow_duplicate=True)],
        Input("match-centre-table", "data_timestamp"),
        [State("match-centre-table", "data"),
         State("match-centre-full-data", "data"),
         State("league-selector", "active_tab")],
        prevent_initial_call=True,
    )
    def persist_taken_changes(_ts, table_data, full_data, league_tab):
        """Sync taken column edits to the predictions table and Store."""
        if not table_data:
            return "", no_update

        league = league_tab.replace("league-", "") if league_tab else "PL"

        # Build a lookup of taken values from the rendered table
        table_taken: dict[tuple, str] = {}
        for row in table_data:
            key = (row.get("_home", ""), row.get("_away", ""),
                   row.get("_market", ""), row.get("_side", ""))
            table_taken[key] = row.get("taken", "")

        # Persist to DB
        try:
            pred_df = get_predictions(league)
            if not pred_df.empty:
                pred_lookup: dict[tuple, int] = {}
                pred_taken_db: dict[tuple, int] = {}
                for _, p in pred_df.iterrows():
                    key = (p["home_team"], p["away_team"],
                           p["market"], p["side"])
                    pred_lookup[key] = int(p["id"])
                    pred_taken_db[key] = int(p.get("taken", 0))

                for key, taken_str in table_taken.items():
                    pred_id = pred_lookup.get(key)
                    if pred_id is None:
                        continue
                    new_taken = 1 if taken_str == "Yes" else 0
                    if new_taken != pred_taken_db.get(key, 0):
                        toggle_prediction_taken(pred_id, bool(new_taken), league)
        except Exception:
            pass

        # Sync taken values back to the full Store so sort/filter preserves them
        if full_data:
            updated = []
            for row in full_data:
                r = dict(row)
                key = (r.get("_home", ""), r.get("_away", ""),
                       r.get("_market", ""), r.get("_side", ""))
                if key in table_taken:
                    r["taken"] = table_taken[key]
                updated.append(r)
            return "", updated

        return "", no_update

    return app



# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the dashboard server."""
    app = create_app()
    print("\n  Dashboard running at http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
