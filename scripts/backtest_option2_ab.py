"""
Option 2 full-backtest A/B comparison.

Runs PL O/U 2.5 backtest and EFL O/U 2.5 backtest twice each:
  - PRE-Option-2:  use_shrinkage=False (hard-threshold fallback).
                   Simulates the pre-partial-pooling behaviour.
  - POST-Option-2: use_shrinkage=True (partial pooling).
                   Current default.

Note: Step 1 (per-market tuning) affects BTTS most, which isn't exercised
by these backtest scripts (they're O/U 2.5-focused). So this A/B mainly
isolates Step 2's shrinkage effect at the ROI level.

Reports ROI, win rate, bet count, and bankroll growth side-by-side.

Runtime: ~30 minutes total (two backtests x two configs).
Run:  python scripts/backtest_option2_ab.py
"""
from __future__ import annotations

import os
import sys
import io
import contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import model  # imported so we can monkeypatch
import backtest as pl_backtest
import championship_backtest as efl_backtest


class DCShrinkageContext:
    """Context manager that forces use_shrinkage on or off for all DC
    instances created inside the with-block. We do this by monkeypatching
    the class default rather than changing every call site.
    """

    def __init__(self, use_shrinkage: bool):
        self.use_shrinkage = use_shrinkage
        self._original_init = None

    def __enter__(self):
        original_init = model.DixonColesPredictor.__init__
        force = self.use_shrinkage

        def patched_init(self, rho=-0.13, half_life=30, use_xg=False,
                         use_mle=False, mle_alpha=0.01, use_shrinkage=True):
            # Force the shrinkage choice for A/B isolation
            original_init(self, rho=rho, half_life=half_life,
                          use_xg=use_xg, use_mle=use_mle,
                          mle_alpha=mle_alpha, use_shrinkage=force)

        self._original_init = original_init
        model.DixonColesPredictor.__init__ = patched_init
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        model.DixonColesPredictor.__init__ = self._original_init


def extract_pl_metrics(captured: str) -> dict:
    """Parse the PL backtest's summary output into structured metrics."""
    # The PL backtest prints a summary table at the end. Extract key lines.
    metrics = {}
    for line in captured.splitlines():
        line = line.strip()
        if line.startswith("Total bets:"):
            try:
                metrics["total_bets"] = int(line.split()[-1])
            except (ValueError, IndexError):
                pass
        elif line.startswith("Win rate:"):
            try:
                metrics["win_rate"] = float(line.split()[-1].rstrip("%")) / 100
            except (ValueError, IndexError):
                pass
        elif line.startswith("Total ROI:"):
            try:
                metrics["roi"] = float(
                    line.split()[-1].rstrip("%").lstrip("+")) / 100
            except (ValueError, IndexError):
                pass
        elif line.startswith("Final bankroll:"):
            try:
                metrics["final_bankroll"] = float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    return metrics


def run_and_capture(run_fn, use_shrinkage: bool) -> tuple[str, dict]:
    """Run a backtest under the shrinkage context, return (stdout, metrics)."""
    buf = io.StringIO()
    with DCShrinkageContext(use_shrinkage):
        with contextlib.redirect_stdout(buf):
            try:
                run_fn()
            except Exception as e:
                # Don't let one backtest crash the A/B — print to stderr
                print(f"\n[ERROR] Backtest crashed: {e}", file=sys.stderr)
    captured = buf.getvalue()
    metrics = extract_pl_metrics(captured)
    return captured, metrics


def format_delta(post: float, pre: float) -> str:
    delta = post - pre
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.4f}"


def main():
    print("\n" + "#" * 70)
    print("#  Option 2 Full-Backtest A/B — Pre vs Post Option 2")
    print("#" * 70)
    print("\n  PRE  = use_shrinkage=False (hard-threshold fallback)")
    print("  POST = use_shrinkage=True  (partial pooling, Option 2)")
    print("\n  Note: This isolates Step 2's shrinkage. Step 1 (per-market"
          "\n  tuning) is market-level infrastructure in predict.py, not"
          "\n  in these backtest scripts (which are O/U 2.5 specific).")

    results = {}

    # ── PL O/U 2.5 backtest ──
    print("\n" + "=" * 70)
    print("  PL O/U 2.5 backtest")
    print("=" * 70)

    print("\n  [1/2] Running with use_shrinkage=False (PRE)...")
    pre_output, pre_metrics = run_and_capture(
        pl_backtest.run_backtest, use_shrinkage=False)
    print(f"    PRE metrics: {pre_metrics}")

    print("\n  [2/2] Running with use_shrinkage=True (POST)...")
    post_output, post_metrics = run_and_capture(
        pl_backtest.run_backtest, use_shrinkage=True)
    print(f"    POST metrics: {post_metrics}")

    results["PL_ou25"] = (pre_metrics, post_metrics,
                           pre_output, post_output)

    # ── EFL O/U 2.5 backtest ──
    print("\n" + "=" * 70)
    print("  EFL O/U 2.5 backtest")
    print("=" * 70)

    print("\n  [1/2] Running with use_shrinkage=False (PRE)...")
    pre_output_e, pre_metrics_e = run_and_capture(
        efl_backtest.run_backtest, use_shrinkage=False)
    print(f"    PRE metrics: {pre_metrics_e}")

    print("\n  [2/2] Running with use_shrinkage=True (POST)...")
    post_output_e, post_metrics_e = run_and_capture(
        efl_backtest.run_backtest, use_shrinkage=True)
    print(f"    POST metrics: {post_metrics_e}")

    results["EFL_ou25"] = (pre_metrics_e, post_metrics_e,
                            pre_output_e, post_output_e)

    # ── Final summary ──
    print("\n" + "#" * 70)
    print("#  SUMMARY TABLE")
    print("#" * 70)

    for market, (pre, post, _, _) in results.items():
        print(f"\n  {market}")
        print(f"  {'metric':<20} {'PRE':<15} {'POST':<15} {'delta':<15}")
        for key in ("total_bets", "win_rate", "roi", "final_bankroll"):
            if key in pre and key in post:
                pre_v = pre[key]
                post_v = post[key]
                if isinstance(pre_v, float):
                    delta = format_delta(post_v, pre_v)
                    print(f"  {key:<20} {pre_v:<15.4f} {post_v:<15.4f} {delta:<15}")
                else:
                    d = post_v - pre_v
                    sign = "+" if d > 0 else ""
                    print(f"  {key:<20} {pre_v:<15} {post_v:<15} {sign}{d}")

    # Save full captured output for reference
    out_path = os.path.join(
        os.path.dirname(__file__), "option2_backtest_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for market, (pre, post, pre_out, post_out) in results.items():
            f.write(f"\n{'='*70}\n{market} PRE (use_shrinkage=False)\n{'='*70}\n")
            f.write(pre_out)
            f.write(f"\n{'='*70}\n{market} POST (use_shrinkage=True)\n{'='*70}\n")
            f.write(post_out)
    print(f"\n  Full outputs saved to: {out_path}")

    print("\n" + "#" * 70)
    print("#  Interpretation:")
    print("#    POST ROI > PRE ROI by > 1pp: Option 2 meaningfully improves ROI")
    print("#    POST ROI ≈ PRE ROI (±1pp):   Option 2 neutral on ROI (theory still sound)")
    print("#    POST ROI < PRE ROI by > 1pp: Option 2 regression — investigate")
    print("#" * 70)


if __name__ == "__main__":
    main()
