"""Phase 4a — run the per-toggle ROI-validation matrix across all 6 primary
matrix cells, using the OOF caches produced in Phase 3.

For each (league, market) cell, evaluates these variants:
  baseline         — all Option-5 toggles OFF (pre-roadmap reference)
  all_on           — all toggles ON (current production)
  no_shrinkage     — everything ON except Bayesian edge shrinkage
  no_portfolio_cap — everything ON except matchday portfolio cap
  no_same_match    — everything ON except same-match correlation discount
  no_mkt_mult      — everything ON except market/side confidence multipliers
  no_edge_scaling  — everything ON except the edge-scaling bug fix

Contribution of each toggle =  (all_on) vs (no_<toggle>)  deltas on ROI / DD.

Output:
  reports/roi_validate/phase4a_matrix.csv  — one row per (cell, variant)
  reports/roi_validate/phase4a_matrix.md   — consolidated markdown tables
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Register pickled classes for joblib.load via OOF parquet reads
from model import DixonColesPredictor  # noqa: F401

import roi_validate as RV

OOF_DIR = PROJECT_ROOT / "reports" / "roi_validate" / "oof_cache"
OUT_DIR = PROJECT_ROOT / "reports" / "roi_validate"

CELLS: list[tuple[str, str]] = [
    ("PL", "ou25"),
    ("PL", "btts"),
    ("EFL", "ou25"),
    ("PL", "ou15"),
    ("EFL", "ou15"),
    ("EFL", "btts"),
]


def _base_config() -> RV.ValidationConfig:
    """Current production defaults — all toggles ON."""
    return RV.ValidationConfig(
        shrinkage=True, portfolio_cap=True, same_match_discount=True,
        market_multipliers=True, edge_scaling_fix=True,
        devig_discount=None, edge_source="pinnacle",
    )


def _baseline_config() -> RV.ValidationConfig:
    """Pre-roadmap reference — everything OFF."""
    return RV.ValidationConfig(
        shrinkage=False, portfolio_cap=False, same_match_discount=False,
        market_multipliers=False, edge_scaling_fix=False,
        devig_discount=1.0, edge_source="pinnacle",
    )


VARIANTS: dict[str, callable] = {
    "baseline":          _baseline_config,
    "all_on":            _base_config,
    "no_shrinkage":      lambda: _with(_base_config(), shrinkage=False),
    "no_portfolio_cap":  lambda: _with(_base_config(), portfolio_cap=False),
    "no_same_match":     lambda: _with(_base_config(), same_match_discount=False),
    "no_mkt_mult":       lambda: _with(_base_config(), market_multipliers=False),
    "no_edge_scaling":   lambda: _with(_base_config(), edge_scaling_fix=False),
}


def _with(cfg: RV.ValidationConfig, **kw) -> RV.ValidationConfig:
    """Return a copy of cfg with kw overrides applied."""
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _load_oof(league: str, market: str) -> pd.DataFrame | None:
    path = OOF_DIR / f"{league.lower()}_{market}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def run_one(league: str, market: str, variant: str,
            oof_df: pd.DataFrame) -> dict:
    """Run one (cell, variant) combination. Returns a results dict."""
    vc = VARIANTS[variant]()
    bets, _ = RV.replay_league_market(oof_df, league, market, vc)
    total_staked = sum(b["stake_pct"] for b in bets)
    total_pnl = sum(b["pnl"] for b in bets)
    roi = total_pnl / total_staked if total_staked > 0 else 0.0
    win_rate = float(np.mean([b["won"] for b in bets])) if bets else 0.0
    dd = RV.compute_max_drawdown(bets)
    mean_roi, lo95, hi95 = RV.block_bootstrap_roi(bets, n_resamples=500)
    return {
        "league": league,
        "market": market,
        "variant": variant,
        "n_bets": len(bets),
        "staked": float(total_staked),
        "pnl": float(total_pnl),
        "roi": float(roi),
        "ci_lo_95": float(lo95),
        "ci_hi_95": float(hi95),
        "win_rate": win_rate,
        "max_drawdown": float(dd),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()
    print(f"Phase 4a matrix — {len(CELLS)} cells × {len(VARIANTS)} variants "
          f"= {len(CELLS) * len(VARIANTS)} runs\n")

    for league, market in CELLS:
        oof_df = _load_oof(league, market)
        if oof_df is None:
            print(f"[SKIP] {league} {market}: OOF parquet missing")
            continue
        print(f"--- {league} {market} ({len(oof_df):,} fixtures) ---")
        for variant in VARIANTS:
            t0 = time.time()
            res = run_one(league, market, variant, oof_df)
            rows.append(res)
            print(f"  {variant:<18} bets={res['n_bets']:>5}  "
                  f"ROI={res['roi']:+6.2%}  "
                  f"CI95=[{res['ci_lo_95']:+6.2%}, {res['ci_hi_95']:+6.2%}]  "
                  f"DD={res['max_drawdown']:>6.1%}  "
                  f"({time.time()-t0:.1f}s)")
        print()

    csv_path = OUT_DIR / "phase4a_matrix.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    _write_markdown(rows, OUT_DIR / "phase4a_matrix.md")

    print(f"\nTotal time: {time.time()-started:.0f}s")
    return 0


def _write_markdown(rows: list[dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Phase 4a — Per-toggle ROI matrix\n\n")
        f.write("Walk-forward window S19-S24. Variants are evaluated against "
                "the same OOF cache per cell, so numeric differences are "
                "entirely attributable to the decision-logic toggle.\n\n")
        f.write("**Contribution of each toggle** = (all_on) − (no_<toggle>). "
                "Positive ROI contribution = the toggle helps. "
                "Positive DD contribution = the toggle reduces drawdown.\n\n")

        # ── Per-cell tables ──
        for (league, market), group in df.groupby(["league", "market"], sort=False):
            f.write(f"## {league} {market}\n\n")
            f.write("| Variant | Bets | ROI | CI95 | Win rate | Max DD |\n")
            f.write("|---------|------|-----|------|----------|--------|\n")
            for _, r in group.iterrows():
                f.write(f"| `{r['variant']}` | {int(r['n_bets']):>5} | "
                        f"{r['roi']:+6.2%} | "
                        f"[{r['ci_lo_95']:+6.2%}, {r['ci_hi_95']:+6.2%}] | "
                        f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} |\n")
            f.write("\n")

        # ── Per-toggle contribution summary ──
        f.write("## Per-toggle contribution summary (delta vs all_on)\n\n")
        f.write("For each toggle, positive ROI Δ means *turning it off hurts* "
                "(i.e. the toggle contributes ROI). Positive DD Δ means "
                "*turning it off raises drawdown* (i.e. the toggle reduces "
                "risk).\n\n")

        pivot_roi = df.pivot(index=["league", "market"], columns="variant",
                              values="roi")
        pivot_dd = df.pivot(index=["league", "market"], columns="variant",
                             values="max_drawdown")

        toggle_variants = ["no_shrinkage", "no_portfolio_cap",
                           "no_same_match", "no_mkt_mult", "no_edge_scaling"]

        f.write("### ROI contribution (all_on − no_<toggle>)\n\n")
        hdr = "| Cell | " + " | ".join(t.replace("no_", "") for t in toggle_variants) + " |\n"
        sep = "|" + "---|" * (len(toggle_variants) + 1) + "\n"
        f.write(hdr); f.write(sep)
        for idx, r in pivot_roi.iterrows():
            league, market = idx
            row = f"| {league} {market} |"
            for t in toggle_variants:
                if t in r and "all_on" in r:
                    d = r["all_on"] - r[t]
                    row += f" {d:+5.2%} |"
                else:
                    row += " n/a |"
            f.write(row + "\n")
        f.write("\n")

        f.write("### Drawdown contribution (no_<toggle> − all_on)\n\n")
        f.write("Positive = the toggle reduces drawdown by that much.\n\n")
        hdr = "| Cell | " + " | ".join(t.replace("no_", "") for t in toggle_variants) + " |\n"
        f.write(hdr); f.write(sep)
        for idx, r in pivot_dd.iterrows():
            league, market = idx
            row = f"| {league} {market} |"
            for t in toggle_variants:
                if t in r and "all_on" in r:
                    d = r[t] - r["all_on"]
                    row += f" {d:+5.1%} |"
                else:
                    row += " n/a |"
            f.write(row + "\n")
        f.write("\n")

        # ── all_on vs baseline overall Δ ──
        f.write("## All-on vs baseline summary\n\n")
        f.write("| Cell | ROI all_on | ROI baseline | ROI Δ | DD all_on | DD baseline | DD reduction |\n")
        f.write("|------|-----------|---------------|-------|-----------|-------------|--------------|\n")
        for idx in pivot_roi.index:
            if "all_on" in pivot_roi.columns and "baseline" in pivot_roi.columns:
                league, market = idx
                r_on = pivot_roi.loc[idx, "all_on"]
                r_base = pivot_roi.loc[idx, "baseline"]
                d_on = pivot_dd.loc[idx, "all_on"]
                d_base = pivot_dd.loc[idx, "baseline"]
                dd_red = (d_base - d_on) / d_base if d_base > 0 else 0.0
                f.write(f"| {league} {market} | {r_on:+6.2%} | "
                        f"{r_base:+6.2%} | {r_on - r_base:+6.2%} | "
                        f"{d_on:.1%} | {d_base:.1%} | "
                        f"{dd_red:+.0%} |\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
