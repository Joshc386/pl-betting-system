"""ROI validator — replays walk-forward OOF predictions through ``decide_bet``
with configurable toggles, measures realised P&L, and reports per-season
ROI with block-bootstrap confidence intervals.

Built in Phase 3 to close the backtest-vs-live divergence identified in
Option 5. Every bet-selection decision flows through the shared
``staking.decide_bet`` function so the validator and the live path
(``predict.py`` / ``championship_predict.py``) make identical decisions
given identical inputs.

Input: OOF cache parquet files produced by ``scripts/generate_oof_cache.py``.
Output: per-run JSON under ``reports/roi_validate/runs/<run_id>.json`` plus
a console summary.

Bootstrap: **block-bootstrap by matchday** (kickoff date). Same-match
correlation + matchday portfolio cap mean per-bet resampling would
understate variance. Resampling whole matchdays preserves the dependency.

Run:
  python scripts/roi_validate.py --league PL --market ou25
  python scripts/roi_validate.py --league PL --market ou25 --shrinkage off
  python scripts/roi_validate.py --league PL --market ou25 --baseline
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Make pickled model classes unpickle-able if parquet→loaded downstream
from model import DixonColesPredictor  # noqa: F401

from staking import (
    decide_bet, apply_portfolio_constraints,
    PL_AGREE_SCALE, EFL_AGREE_SCALE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OOF_DIR = PROJECT_ROOT / "reports" / "roi_validate" / "oof_cache"
RUNS_DIR = PROJECT_ROOT / "reports" / "roi_validate" / "runs"

LEAGUE_TO_AGREE_SCALE = {"PL": PL_AGREE_SCALE, "EFL": EFL_AGREE_SCALE}

DEFAULT_CONFIG = {
    "blend_weight": 0.35,
    "min_edge": 0.02,
    "min_agree": 2,
    "kelly_fraction": 0.25,
    "max_stake_pct": 0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Toggles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    """All knobs that affect decision-making. Mapped to CLI flags."""
    shrinkage: bool = True
    portfolio_cap: bool = True
    same_match_discount: bool = True
    market_multipliers: bool = True
    edge_scaling_fix: bool = True
    devig_discount: float | None = None      # None -> use config.DEVIG_DISCOUNT
    edge_source: str = "pinnacle"            # "pinnacle" | "devig"
    min_edge: float = DEFAULT_CONFIG["min_edge"]
    min_agree: int = DEFAULT_CONFIG["min_agree"]
    blend_weight: float = DEFAULT_CONFIG["blend_weight"]
    kelly_fraction: float = DEFAULT_CONFIG["kelly_fraction"]
    max_stake_pct: float = DEFAULT_CONFIG["max_stake_pct"]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReplayResult:
    """Full output of one replay run, serialisable to JSON."""
    run_id: str
    league: str
    market: str
    config: dict
    seasons: list[int]
    n_fixtures: int
    n_bets: int
    gross_roi: float
    bet_win_rate: float
    max_drawdown: float
    roi_ci_lo_95: float
    roi_ci_hi_95: float
    per_season: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core replay
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate_single(raw_prob: float, shift: float) -> float:
    """Apply logit-shift calibration (same as backtest.py)."""
    if raw_prob <= 0:
        return 0.0
    if raw_prob >= 1:
        return 1.0
    logit = np.log(raw_prob / (1 - raw_prob))
    return 1.0 / (1.0 + np.exp(-(logit + shift)))


def _apply_overrides_in_config_scope(vc: ValidationConfig):
    """Yield a context where config.USE_EDGE_SHRINKAGE etc. reflect toggles.

    Rather than rewrite decide_bet, we temporarily patch the config module
    since decide_bet/shrink_edge read from it. Restored on exit.
    """
    import config as cfg

    class _Scope:
        def __enter__(self_):
            self_._orig_shrink = cfg.USE_EDGE_SHRINKAGE
            self_._orig_devig = cfg.DEVIG_DISCOUNT
            self_._orig_mult = dict(cfg.MARKET_MULTIPLIERS)
            cfg.USE_EDGE_SHRINKAGE = vc.shrinkage
            if vc.devig_discount is not None:
                cfg.DEVIG_DISCOUNT = vc.devig_discount
            if not vc.market_multipliers:
                # replace with all-1.0 map so every market/side has mult=1.0
                cfg.MARKET_MULTIPLIERS = {}
            return self_

        def __exit__(self_, *exc):
            cfg.USE_EDGE_SHRINKAGE = self_._orig_shrink
            cfg.DEVIG_DISCOUNT = self_._orig_devig
            cfg.MARKET_MULTIPLIERS = self_._orig_mult
            return False

    return _Scope()


def _implied_fair_prob(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove overround: return (fair_a, fair_b) that sum to 1."""
    if odds_a is None or odds_b is None or odds_a <= 1 or odds_b <= 1:
        return float("nan"), float("nan")
    ia = 1.0 / odds_a
    ib = 1.0 / odds_b
    total = ia + ib
    return ia / total, ib / total


def replay_league_market(
    oof_df: pd.DataFrame,
    league: str,
    market: str,
    vc: ValidationConfig,
) -> tuple[list[dict], dict]:
    """Replay every fixture in the OOF cache through decide_bet.

    Returns (bets, per_season_stats). Each bet dict has keys:
      season, date, home_team, away_team, side, stake_pct, edge,
      model_prob, fair_prob, odds, outcome, pnl.
    """
    agree_scale = LEAGUE_TO_AGREE_SCALE[league]
    cfg = {
        "blend_weight": vc.blend_weight,
        "min_edge": vc.min_edge,
        "min_agree": vc.min_agree,
        "kelly_fraction": vc.kelly_fraction,
        "max_stake_pct": vc.max_stake_pct,
    }

    per_season_stats: dict[int, dict] = {}
    bets: list[dict] = []

    with _apply_overrides_in_config_scope(vc):
        # Group by matchday so we can call apply_portfolio_constraints per day
        day_groups = oof_df.groupby("date", sort=True)
        for day, day_df in day_groups:
            day_bets: list[dict] = []
            for _, row in day_df.iterrows():
                # Calibrate raw per-model probs
                per_model = []
                for mdl in ("xgb", "lgb", "dc", "lr"):
                    raw = row.get(f"{mdl}_prob")
                    shift = row.get(f"{mdl}_shift")
                    if raw is None or pd.isna(raw):
                        continue
                    per_model.append(_calibrate_single(float(raw), float(shift)))
                per_model_arr = np.array(per_model, dtype=float)

                # Evaluate side_a and side_b
                odds_a = row.get("odds_a")
                odds_b = row.get("odds_b")
                fair_a, fair_b = _implied_fair_prob(odds_a, odds_b)
                if np.isnan(fair_a) or np.isnan(fair_b):
                    continue

                # Model probability for side_a = mean of calibrated per-model
                model_a = float(per_model_arr.mean())
                model_b = 1.0 - model_a

                # Evaluate both sides
                for side_label, model_p, fair_p, odds, side_col in (
                    (row["side_a_label"], model_a, fair_a, odds_a, "a"),
                    (row["side_b_label"], model_b, fair_p if False else fair_b,
                     odds_b, "b"),
                ):
                    if odds is None or pd.isna(odds) or odds <= 1:
                        continue
                    # per_model must reflect side being evaluated
                    pm_side = (per_model_arr if side_col == "a"
                               else 1.0 - per_model_arr)
                    decision = decide_bet(
                        model_p=model_p,
                        fair_p=fair_p,
                        odds=float(odds),
                        per_model=pm_side,
                        fair_threshold=fair_p,
                        config=cfg,
                        edge_source=vc.edge_source,
                        market=market,
                        side=side_label,
                        agree_scale=agree_scale,
                    )
                    if decision is None:
                        continue

                    # Apply edge-scaling-fix toggle (pre-Option-5 behaviour)
                    # — if disabled, stake as if >0.06 branch unreachable.
                    # We approximate by capping stake via refined_kelly from
                    # the old code path; cheap way = refine in-place.
                    if not vc.edge_scaling_fix and decision["edge"] > 0.06:
                        # Old code: 1.15 applied, not 1.25. Undo the extra.
                        decision["stake_pct"] *= (1.15 / 1.25)

                    won = (int(row["outcome"]) == 1 and side_col == "a") \
                          or (int(row["outcome"]) == 0 and side_col == "b")
                    day_bets.append({
                        "season": int(row["season"]),
                        "date": row["date"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "market": market,
                        "side": side_label,
                        "stake_pct": float(decision["stake_pct"]),
                        "edge": float(decision["edge"]),
                        "model_prob": float(model_p),
                        "fair_prob": float(fair_p),
                        "odds": float(odds),
                        "outcome": int(row["outcome"]),
                        "won": bool(won),
                    })

            # Portfolio constraints / same-match discount (matchday-level)
            if vc.portfolio_cap or vc.same_match_discount:
                # Save original same-match discount if disabled
                import config as cfg_mod
                orig_smd = dict(cfg_mod.SAME_MATCH_DISCOUNT)
                orig_cap = cfg_mod.MAX_MATCHDAY_STAKE_PCT
                if not vc.same_match_discount:
                    cfg_mod.SAME_MATCH_DISCOUNT = {}
                if not vc.portfolio_cap:
                    cfg_mod.MAX_MATCHDAY_STAKE_PCT = 1e9  # effectively off
                try:
                    apply_portfolio_constraints(day_bets)
                finally:
                    cfg_mod.SAME_MATCH_DISCOUNT = orig_smd
                    cfg_mod.MAX_MATCHDAY_STAKE_PCT = orig_cap

            # Settle P&L per bet
            for b in day_bets:
                stake = b["stake_pct"]
                if b["won"]:
                    b["pnl"] = stake * (b["odds"] - 1)
                else:
                    b["pnl"] = -stake
                bets.append(b)

    # Per-season aggregate stats
    per_season = []
    if bets:
        bets_df = pd.DataFrame(bets)
        for s, sub in bets_df.groupby("season", sort=True):
            staked = sub["stake_pct"].sum()
            pnl = sub["pnl"].sum()
            roi = pnl / staked if staked > 0 else 0.0
            per_season.append({
                "season": int(s),
                "n_bets": int(len(sub)),
                "staked": float(staked),
                "pnl": float(pnl),
                "roi": float(roi),
                "win_rate": float(sub["won"].mean()),
            })

    return bets, {"per_season": per_season}


# ─────────────────────────────────────────────────────────────────────────────
# Block bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def block_bootstrap_roi(
    bets: list[dict], n_resamples: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Resample whole matchdays and recompute ROI. Return (mean, lo95, hi95).

    Resampling matchdays (not individual bets) preserves same-match
    correlation and matchday-portfolio-cap structure — which ``decide_bet``
    explicitly models.
    """
    if not bets:
        return 0.0, 0.0, 0.0

    df = pd.DataFrame(bets)
    days = df["date"].unique()
    rng = random.Random(seed)
    rois: list[float] = []
    for _ in range(n_resamples):
        sample_days = [rng.choice(list(days)) for _ in days]
        counts = pd.Series(sample_days).value_counts().to_dict()
        parts: list[pd.DataFrame] = []
        for d, k in counts.items():
            day_df = df[df["date"] == d]
            parts.extend([day_df] * k)
        sampled = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
        staked = sampled["stake_pct"].sum()
        pnl = sampled["pnl"].sum()
        rois.append(pnl / staked if staked > 0 else 0.0)
    arr = np.array(rois)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), \
           float(np.percentile(arr, 97.5))


def compute_max_drawdown(bets: list[dict]) -> float:
    """Running-peak drawdown across cumulative P&L."""
    if not bets:
        return 0.0
    df = pd.DataFrame(bets).sort_values("date")
    cum = df["pnl"].cumsum()
    peak = cum.cummax()
    dd = (cum - peak).min()
    return float(abs(dd))


# ─────────────────────────────────────────────────────────────────────────────
# CLI + driver
# ─────────────────────────────────────────────────────────────────────────────

def _bool_flag(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def _build_config_from_args(args) -> ValidationConfig:
    vc = ValidationConfig()
    if args.baseline:
        vc.shrinkage = False
        vc.portfolio_cap = False
        vc.same_match_discount = False
        vc.market_multipliers = False
        vc.edge_scaling_fix = False
        vc.devig_discount = 1.0  # no discount in baseline
        return vc
    if args.shrinkage is not None:
        vc.shrinkage = _bool_flag(args.shrinkage)
    if args.portfolio_cap is not None:
        vc.portfolio_cap = _bool_flag(args.portfolio_cap)
    if args.same_match_discount is not None:
        vc.same_match_discount = _bool_flag(args.same_match_discount)
    if args.market_multipliers is not None:
        vc.market_multipliers = _bool_flag(args.market_multipliers)
    if args.edge_scaling_fix is not None:
        vc.edge_scaling_fix = _bool_flag(args.edge_scaling_fix)
    if args.devig_discount is not None:
        vc.devig_discount = args.devig_discount
    if args.edge_source is not None:
        vc.edge_source = args.edge_source
    return vc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=("PL", "EFL"))
    ap.add_argument("--market", required=True,
                    choices=("ou25", "btts", "btts_betfair", "ou15"))
    ap.add_argument("--oof-path", default=None,
                    help="Override OOF parquet path")
    ap.add_argument("--seasons", default="19-24")
    ap.add_argument("--baseline", action="store_true",
                    help="All toggles OFF (pre-roadmap reference)")
    ap.add_argument("--shrinkage", default=None)
    ap.add_argument("--portfolio-cap", default=None)
    ap.add_argument("--same-match-discount", default=None)
    ap.add_argument("--market-multipliers", default=None)
    ap.add_argument("--edge-scaling-fix", default=None)
    ap.add_argument("--devig-discount", type=float, default=None)
    ap.add_argument("--edge-source", choices=("pinnacle", "devig"), default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.oof_path:
        oof_path = Path(args.oof_path)
    else:
        # Prefer parquet, fall back to pickle produced by the generator's
        # defensive write.
        candidates = [
            OOF_DIR / f"{args.league.lower()}_{args.market}.parquet",
            OOF_DIR / f"{args.league.lower()}_{args.market}.pkl",
        ]
        oof_path = next((c for c in candidates if c.exists()), candidates[0])
    if not oof_path.exists():
        print(f"[ERROR] OOF cache not found: {oof_path}", file=sys.stderr)
        print("Run `python scripts/generate_oof_cache.py --league ... "
              "--market ...` first.", file=sys.stderr)
        return 2

    print(f"Reading OOF cache: {oof_path}")
    oof_df = (pd.read_parquet(oof_path)
              if oof_path.suffix == ".parquet"
              else pd.read_pickle(oof_path))
    print(f"  {len(oof_df):,} fixtures across "
          f"S{oof_df['season'].min()}..S{oof_df['season'].max()}")

    # Filter to requested seasons if --seasons differs from full span
    parts = args.seasons.split("-")
    if len(parts) == 2:
        s_lo, s_hi = int(parts[0]), int(parts[1])
        oof_df = oof_df[(oof_df["season"] >= s_lo)
                        & (oof_df["season"] <= s_hi)].copy()

    vc = _build_config_from_args(args)
    print(f"Config: {vc.to_dict()}")

    print("\nReplaying ...")
    bets, stats = replay_league_market(oof_df, args.league, args.market, vc)

    total_staked = sum(b["stake_pct"] for b in bets)
    total_pnl = sum(b["pnl"] for b in bets)
    gross_roi = total_pnl / total_staked if total_staked > 0 else 0.0
    win_rate = np.mean([b["won"] for b in bets]) if bets else 0.0
    dd = compute_max_drawdown(bets)

    print("\nRunning block bootstrap ...")
    mean_roi, lo95, hi95 = block_bootstrap_roi(
        bets, n_resamples=args.n_bootstrap,
    )
    print(f"  ROI: {gross_roi:+.3%}  CI95 [{lo95:+.3%}, {hi95:+.3%}]  "
          f"(bootstrap mean {mean_roi:+.3%})")

    run_id = args.run_id or f"{args.league}_{args.market}_{uuid.uuid4().hex[:8]}"
    result = ReplayResult(
        run_id=run_id,
        league=args.league,
        market=args.market,
        config=vc.to_dict(),
        seasons=sorted({b["season"] for b in bets}),
        n_fixtures=len(oof_df),
        n_bets=len(bets),
        gross_roi=gross_roi,
        bet_win_rate=float(win_rate),
        max_drawdown=dd,
        roi_ci_lo_95=lo95,
        roi_ci_hi_95=hi95,
        per_season=stats["per_season"],
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"  {args.league} {args.market} — Phase 4a run {run_id}")
    print("=" * 70)
    print(f"  Fixtures:      {result.n_fixtures:,}")
    print(f"  Bets:          {result.n_bets:,}")
    print(f"  ROI:           {result.gross_roi:+.3%}")
    print(f"  95% CI:        [{result.roi_ci_lo_95:+.3%}, "
          f"{result.roi_ci_hi_95:+.3%}]")
    print(f"  Win rate:      {result.bet_win_rate:.3%}")
    print(f"  Max drawdown:  {result.max_drawdown:.3%}")
    print(f"\nPer-season:")
    for ps in result.per_season:
        print(f"  S{ps['season']}: {ps['n_bets']:>4} bets, "
              f"ROI {ps['roi']:+.3%}, win {ps['win_rate']:.1%}")
    print(f"\nSaved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
