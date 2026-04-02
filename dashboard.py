"""
Unified betting dashboard for O/U 2.5 and BTTS markets.

Displays upcoming fixture recommendations, historical performance,
and bankroll tracking across both strategies.
"""
import os
import json
import sqlite3
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

# ── Database ──
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "dashboard.db")


def _get_db():
    """Get SQLite connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            kickoff TEXT,
            market TEXT NOT NULL CHECK (market IN ('ou25', 'btts')),
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            balance REAL NOT NULL,
            event TEXT
        )
    """)
    conn.commit()
    return conn


def save_recommendations(recs: list[dict]) -> int:
    """Save new recommendations to database. Returns number saved."""
    conn = _get_db()
    saved = 0
    for r in recs:
        # Check for duplicate (same fixture + market + side)
        existing = conn.execute(
            "SELECT id FROM recommendations WHERE home_team=? AND away_team=? "
            "AND market=? AND side=? AND settled=0",
            (r["home_team"], r["away_team"], r["market"], r["side"])
        ).fetchone()
        if existing:
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
                r.get("model_prob"), r.get("blended_prob"), r.get("fair_prob"),
                r["odds"], r.get("edge"), r.get("ev"),
                r.get("stake_pct"), r.get("confidence"), r.get("best_bookmaker"),
                r.get("n_books"), r.get("n_agree"),
                json.dumps(r.get("per_model_probs", {})),
            ),
        )
        saved += 1
    conn.commit()
    conn.close()
    return saved


def get_active_recommendations() -> pd.DataFrame:
    """Get all unsettled recommendations."""
    conn = _get_db()
    df = pd.read_sql_query(
        "SELECT * FROM recommendations WHERE settled=0 ORDER BY kickoff", conn
    )
    conn.close()
    return df


def get_settled_recommendations() -> pd.DataFrame:
    """Get all settled recommendations."""
    conn = _get_db()
    df = pd.read_sql_query(
        "SELECT * FROM recommendations WHERE settled=1 ORDER BY settled_at DESC", conn
    )
    conn.close()
    return df


def get_all_recommendations() -> pd.DataFrame:
    """Get all recommendations."""
    conn = _get_db()
    df = pd.read_sql_query(
        "SELECT * FROM recommendations ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


# ── Dashboard Layout ──

def _make_rec_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Create a styled DataTable for recommendations."""
    if df.empty:
        return html.P("No recommendations available.", className="text-muted p-3")

    display_df = df[[
        "home_team", "away_team", "market", "side", "edge", "odds",
        "stake_pct", "confidence", "best_bookmaker", "n_agree",
    ]].copy()
    display_df["fixture"] = display_df["home_team"] + " v " + display_df["away_team"]
    display_df["edge"] = (display_df["edge"] * 100).round(1).astype(str) + "%"
    display_df["stake_pct"] = (display_df["stake_pct"] * 100).round(2).astype(str) + "%"
    display_df["odds"] = display_df["odds"].round(2)
    display_df["market"] = display_df["market"].map({"ou25": "O/U 2.5", "btts": "BTTS"})

    cols = [
        {"name": "Fixture", "id": "fixture"},
        {"name": "Market", "id": "market"},
        {"name": "Side", "id": "side"},
        {"name": "Edge", "id": "edge"},
        {"name": "Odds", "id": "odds"},
        {"name": "Stake", "id": "stake_pct"},
        {"name": "Conf", "id": "confidence"},
        {"name": "Agree", "id": "n_agree"},
        {"name": "Bookmaker", "id": "best_bookmaker"},
    ]

    return dash_table.DataTable(
        data=display_df[["fixture", "market", "side", "edge", "odds",
                         "stake_pct", "confidence", "n_agree",
                         "best_bookmaker"]].to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px",
                     "fontFamily": "monospace", "fontSize": "13px"},
        style_header={"backgroundColor": "#1a1a2e", "color": "white",
                       "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": '{confidence} = "high"'},
             "backgroundColor": "#1b4332", "color": "white"},
            {"if": {"filter_query": '{confidence} = "medium"'},
             "backgroundColor": "#2d3a4a", "color": "white"},
            {"if": {"filter_query": '{confidence} = "low"'},
             "backgroundColor": "#3d2020", "color": "white"},
        ],
        style_data={"backgroundColor": "#16213e", "color": "#e0e0e0"},
    )


def _make_history_table(df: pd.DataFrame) -> dash_table.DataTable:
    """Create a styled DataTable for settled bets."""
    if df.empty:
        return html.P("No settled bets yet.", className="text-muted p-3")

    display_df = df[[
        "home_team", "away_team", "market", "side", "edge", "odds",
        "stake_pct", "won", "profit_pct", "settled_at",
    ]].copy()
    display_df["fixture"] = display_df["home_team"] + " v " + display_df["away_team"]
    display_df["edge"] = (display_df["edge"] * 100).round(1).astype(str) + "%"
    display_df["stake_pct"] = (display_df["stake_pct"] * 100).round(2).astype(str) + "%"
    display_df["profit_pct"] = (display_df["profit_pct"] * 100).round(2).astype(str) + "%"
    display_df["odds"] = display_df["odds"].round(2)
    display_df["result"] = display_df["won"].map({1: "✓ Won", 0: "✗ Lost"})
    display_df["market"] = display_df["market"].map({"ou25": "O/U 2.5", "btts": "BTTS"})

    cols = [
        {"name": "Fixture", "id": "fixture"},
        {"name": "Market", "id": "market"},
        {"name": "Side", "id": "side"},
        {"name": "Odds", "id": "odds"},
        {"name": "Result", "id": "result"},
        {"name": "Profit", "id": "profit_pct"},
        {"name": "Settled", "id": "settled_at"},
    ]

    return dash_table.DataTable(
        data=display_df[["fixture", "market", "side", "odds", "result",
                         "profit_pct", "settled_at"]].to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "8px",
                     "fontFamily": "monospace", "fontSize": "13px"},
        style_header={"backgroundColor": "#1a1a2e", "color": "white",
                       "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": '{result} contains "Won"'},
             "backgroundColor": "#1b4332", "color": "white"},
            {"if": {"filter_query": '{result} contains "Lost"'},
             "backgroundColor": "#3d2020", "color": "white"},
        ],
        style_data={"backgroundColor": "#16213e", "color": "#e0e0e0"},
    )


def _make_bankroll_chart(df: pd.DataFrame) -> go.Figure:
    """Create bankroll curve from settled bets."""
    if df.empty or "settled" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title="Bankroll Growth (no data yet)",
            paper_bgcolor="#0f0f23",
            plot_bgcolor="#0f0f23",
        )
        return fig

    settled = df[df["settled"] == 1].sort_values("settled_at").copy()
    if settled.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            title="Bankroll Growth (no settled bets yet)",
            paper_bgcolor="#0f0f23",
            plot_bgcolor="#0f0f23",
        )
        return fig

    settled["cum_profit"] = settled["profit_pct"].cumsum()
    settled["bankroll"] = 1 + settled["cum_profit"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(settled))),
        y=settled["bankroll"],
        mode="lines",
        name="Combined",
        line=dict(color="#00d4aa", width=2),
    ))

    # Separate by market
    for mkt, color, label in [("ou25", "#4dabf7", "O/U 2.5"),
                               ("btts", "#ff6b6b", "BTTS")]:
        mkt_df = settled[settled["market"] == mkt]
        if not mkt_df.empty:
            mkt_df = mkt_df.copy()
            mkt_df["cum_profit"] = mkt_df["profit_pct"].cumsum()
            mkt_df["bankroll"] = 1 + mkt_df["cum_profit"]
            fig.add_trace(go.Scatter(
                x=list(range(len(mkt_df))),
                y=mkt_df["bankroll"],
                mode="lines",
                name=label,
                line=dict(color=color, width=1, dash="dot"),
            ))

    fig.update_layout(
        template="plotly_dark",
        title="Bankroll Growth",
        xaxis_title="Bet #",
        yaxis_title="Bankroll (units)",
        paper_bgcolor="#0f0f23",
        plot_bgcolor="#0f0f23",
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
    )
    return fig


def _make_stats_cards(df: pd.DataFrame) -> html.Div:
    """Create summary stat cards from all recommendations."""
    if df.empty or "settled" not in df.columns:
        settled = pd.DataFrame()
    else:
        settled = df[df["settled"] == 1]

    total_bets = len(settled)
    if total_bets == 0:
        return html.Div([
            dbc.Row([
                dbc.Col(_stat_card("Total Bets", "0", "secondary"), width=3),
                dbc.Col(_stat_card("Win Rate", "—", "secondary"), width=3),
                dbc.Col(_stat_card("ROI", "—", "secondary"), width=3),
                dbc.Col(_stat_card("Bankroll", "1.000x", "secondary"), width=3),
            ])
        ])

    wins = settled["won"].sum()
    win_rate = wins / total_bets
    staked = settled["stake_pct"].sum()
    profit = settled["profit_pct"].sum()
    roi = profit / staked if staked > 0 else 0
    bankroll = 1 + profit

    active = df[df["settled"] == 0] if "settled" in df.columns else pd.DataFrame()

    return html.Div([
        dbc.Row([
            dbc.Col(_stat_card("Active Picks", str(len(active)), "info"), width=2),
            dbc.Col(_stat_card("Settled", str(total_bets), "secondary"), width=2),
            dbc.Col(_stat_card("Win Rate",
                               f"{win_rate:.1%}",
                               "success" if win_rate > 0.5 else "danger"), width=2),
            dbc.Col(_stat_card("ROI",
                               f"{roi:+.1%}",
                               "success" if roi > 0 else "danger"), width=2),
            dbc.Col(_stat_card("Bankroll",
                               f"{bankroll:.3f}x",
                               "success" if bankroll > 1 else "danger"), width=2),
            dbc.Col(_stat_card("Profit",
                               f"{profit:+.3f}u",
                               "success" if profit > 0 else "danger"), width=2),
        ])
    ])


def _stat_card(title: str, value: str, color: str = "primary") -> dbc.Card:
    """Single stat card."""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-title text-muted mb-1",
                     style={"fontSize": "11px"}),
            html.H4(value, className=f"text-{color} mb-0",
                     style={"fontFamily": "monospace"}),
        ])
    ], className="bg-dark border-secondary")


# ── Build App ──

def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Betting Dashboard",
        update_title=None,
    )

    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("Premier League Betting Dashboard",
                         className="text-light mb-0"),
                html.P("O/U 2.5 + BTTS | Live Predictions",
                        className="text-muted"),
            ], width=8),
            dbc.Col([
                dbc.Button("Refresh Predictions", id="btn-refresh",
                           color="primary", className="me-2"),
                html.Span(id="last-refresh", className="text-muted small"),
            ], width=4, className="text-end pt-3"),
        ], className="py-3 border-bottom border-secondary mb-3"),

        # Stats row
        html.Div(id="stats-cards"),

        # Tabs
        dbc.Tabs([
            dbc.Tab(label="Active Picks", tab_id="tab-active",
                    className="bg-dark"),
            dbc.Tab(label="History", tab_id="tab-history",
                    className="bg-dark"),
            dbc.Tab(label="Performance", tab_id="tab-performance",
                    className="bg-dark"),
        ], id="tabs", active_tab="tab-active", className="mt-3"),

        html.Div(id="tab-content", className="mt-3"),

        # Auto-refresh every 30 minutes
        dcc.Interval(id="interval-refresh", interval=30 * 60 * 1000,
                     n_intervals=0),

        # Store for recommendations
        dcc.Store(id="store-recs"),
    ], fluid=True, className="bg-dark min-vh-100")

    # ── Callbacks ──

    @app.callback(
        [Output("tab-content", "children"),
         Output("stats-cards", "children"),
         Output("last-refresh", "children")],
        [Input("tabs", "active_tab"),
         Input("btn-refresh", "n_clicks"),
         Input("interval-refresh", "n_intervals")],
    )
    def update_content(active_tab, n_clicks, n_intervals):
        all_recs = get_all_recommendations()
        active = get_active_recommendations()
        settled = get_settled_recommendations()

        stats = _make_stats_cards(all_recs)
        refresh_text = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

        triggered = callback_context.triggered_id
        if triggered == "btn-refresh":
            try:
                from predict import LivePredictor
                predictor = LivePredictor(verbose=False)
                predictor.load_data()
                predictor.train()
                recs = predictor.generate_recommendations()
                n_saved = save_recommendations(recs)
                refresh_text = (f"Refreshed at {datetime.now().strftime('%H:%M:%S')} "
                                f"({n_saved} new picks)")
                active = get_active_recommendations()
            except Exception as e:
                refresh_text = f"Refresh failed: {str(e)[:50]}"
                logger.error(f"Refresh failed: {e}", exc_info=True)

        if active_tab == "tab-active":
            content = html.Div([
                html.H5(f"Active Recommendations ({len(active)})",
                         className="text-light mb-3"),
                _make_rec_table(active),
            ])
        elif active_tab == "tab-history":
            content = html.Div([
                html.H5(f"Settled Bets ({len(settled)})",
                         className="text-light mb-3"),
                _make_history_table(settled),
            ])
        elif active_tab == "tab-performance":
            content = html.Div([
                dcc.Graph(figure=_make_bankroll_chart(all_recs)),
                html.Hr(className="border-secondary"),
                _make_market_breakdown(all_recs),
            ])
        else:
            content = html.P("Select a tab.")

        return content, stats, refresh_text

    return app


def _make_market_breakdown(df: pd.DataFrame) -> html.Div:
    """Per-market performance breakdown."""
    if df.empty or "settled" not in df.columns:
        return html.P("No settled bets for breakdown.", className="text-muted")
    settled = df[df["settled"] == 1]
    if settled.empty:
        return html.P("No settled bets for breakdown.", className="text-muted")

    rows = []
    for mkt, label in [("ou25", "O/U 2.5"), ("btts", "BTTS")]:
        mkt_df = settled[settled["market"] == mkt]
        if mkt_df.empty:
            continue
        staked = mkt_df["stake_pct"].sum()
        profit = mkt_df["profit_pct"].sum()
        roi = profit / staked if staked > 0 else 0
        rows.append({
            "Market": label,
            "Bets": len(mkt_df),
            "Wins": int(mkt_df["won"].sum()),
            "Win Rate": f"{mkt_df['won'].mean():.1%}",
            "ROI": f"{roi:+.1%}",
            "Profit": f"{profit:+.3f}u",
        })

    if not rows:
        return html.P("No breakdown available.", className="text-muted")

    return html.Div([
        html.H5("Market Breakdown", className="text-light mb-3"),
        dash_table.DataTable(
            data=rows,
            columns=[{"name": k, "id": k} for k in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "center", "padding": "8px",
                         "fontFamily": "monospace", "fontSize": "13px"},
            style_header={"backgroundColor": "#1a1a2e", "color": "white",
                           "fontWeight": "bold"},
            style_data={"backgroundColor": "#16213e", "color": "#e0e0e0"},
        ),
    ])


def main():
    """Run the dashboard server."""
    app = create_app()
    print("\n  Dashboard running at http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop\n")
    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
