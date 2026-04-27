"""
Option 5 Step 4: walk-forward validation of staking refinement.

Runs the PL O/U 2.5 backtest twice using the existing backtest
infrastructure:
  PRE-Option-5:  shrinkage disabled, portfolio constraints bypassed,
                 edge-scaling bug restored (for a true pre-Option-5 comparison).
  POST-Option-5: current defaults (shrinkage on, portfolio constraints
                 applied, edge-scaling bug fixed).

Reports per-season and aggregate ROI, win rate, bet count, and final
bankroll. Writes outputs to reports/diagnose_option5.txt.

Runtime: ~20 minutes (two backtests).
Run:     python scripts/diagnose_option5.py
"""
from __future__ import annotations

import io
import contextlib
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest as pl_backtest
import config


class Option5OffContext:
    """Temporarily disable all Option 5 changes for the PRE variant.

    - USE_EDGE_SHRINKAGE = False  (no shrinkage)
    - apply_portfolio_constraints becomes a no-op (no cap, no same-match discount)
    - refined_kelly's edge-scaling branches get reverted to the old
      (broken) order so the >0.06 branch is unreachable — this matches
      the pre-Option-5 behaviour exactly.
    """

    def __enter__(self):
        self._original_shrink = config.USE_EDGE_SHRINKAGE
        config.USE_EDGE_SHRINKAGE = False

        # Bypass portfolio constraints: replace with an identity function
        self._original_apply = pl_backtest.apply_portfolio_constraints
        pl_backtest.apply_portfolio_constraints = lambda recs: recs

        # Restore the pre-fix edge-scaling order by patching refined_kelly.
        # We keep all other logic identical; only the scaling branches differ.
        self._original_refined_kelly = pl_backtest.refined_kelly

        def _pre_fix_refined_kelly(blended_prob, odds, n_agree, edge,
                                    kelly_fraction=0.25,
                                    max_stake_pct=0.05,
                                    drawdown_factor=1.0):
            import numpy as np
            if odds <= 1 or blended_prob <= 0:
                return 0.0
            kelly = (blended_prob * odds - 1) / (odds - 1)
            if kelly <= 0:
                return 0.0
            stake = kelly * kelly_fraction
            agree_scale = {0: 0.0, 1: 0.0, 2: 0.70, 3: 0.90, 4: 1.10}
            stake *= agree_scale.get(n_agree, 0.70)
            # Pre-fix (buggy) branching order
            if edge > 0.04:
                stake *= 1.15
            elif edge > 0.06:
                stake *= 1.25
            stake *= drawdown_factor
            stake = min(stake, max_stake_pct)
            if stake < 0.003:
                return 0.0
            return stake

        pl_backtest.refined_kelly = _pre_fix_refined_kelly
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        config.USE_EDGE_SHRINKAGE = self._original_shrink
        pl_backtest.apply_portfolio_constraints = self._original_apply
        pl_backtest.refined_kelly = self._original_refined_kelly


def _extract_summary_metrics(captured_output: str) -> dict:
    """Parse run_backtest's final summary into structured numbers."""
    metrics = {}
    patterns = [
        ("total_bets", r"Total bets:\s+(\d+)"),
        ("win_rate", r"Win rate:\s+([\d.]+)%"),
        ("roi", r"Total ROI:\s+([+\-\d.]+)%"),
        ("final_bankroll", r"Final bankroll:\s+([\d.]+)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, captured_output)
        if m:
            try:
                val = float(m.group(1))
                if key == "win_rate":
                    val /= 100
                elif key == "roi":
                    val /= 100
                metrics[key] = val
            except ValueError:
                pass
    return metrics


def _run_variant(use_option5: bool) -> tuple[str, dict]:
    """Run the PL O/U 2.5 backtest under the given Option 5 setting."""
    buf = io.StringIO()

    def _go():
        with contextlib.redirect_stdout(buf):
            try:
                pl_backtest.run_backtest()
            except Exception as e:
                print(f"\n[ERROR] Backtest crashed: {e}", file=sys.stderr)

    if use_option5:
        _go()
    else:
        with Option5OffContext():
            _go()

    captured = buf.getvalue()
    return captured, _extract_summary_metrics(captured)


def main():
    print("\n" + "#" * 70)
    print("#  Option 5 Step 4: walk-forward validation  -  PL O/U 2.5 A/B")
    print("#" * 70)
    print("\n  PRE  = shrinkage off, portfolio off, pre-fix refined_kelly")
    print("  POST = current (shrinkage on, portfolio cap, same-match discount)")
    print("\n  Expected direction: lower volatility, possibly lower mean ROI")
    print("  but better risk-adjusted return.\n")

    # PRE
    print("=" * 70)
    print("  [1/2] Running PRE-Option-5 variant")
    print("=" * 70 + "\n")
    pre_out, pre_metrics = _run_variant(use_option5=False)
    print(f"\n  PRE metrics: {pre_metrics}")

    # POST
    print("\n" + "=" * 70)
    print("  [2/2] Running POST-Option-5 variant")
    print("=" * 70 + "\n")
    post_out, post_metrics = _run_variant(use_option5=True)
    print(f"\n  POST metrics: {post_metrics}")

    # Summary
    print("\n" + "#" * 70)
    print("#  SUMMARY  -  PL O/U 2.5 backtest")
    print("#" * 70)
    print(f"\n  {'Metric':<20} {'PRE':>12} {'POST':>12} {'delta':>12}")
    for key in ("total_bets", "win_rate", "roi", "final_bankroll"):
        if key in pre_metrics and key in post_metrics:
            pre_v = pre_metrics[key]
            post_v = post_metrics[key]
            delta = post_v - pre_v
            fmt = ".4f" if isinstance(pre_v, float) and abs(pre_v) < 10 else ".1f"
            sign = "+" if delta > 0 else ""
            print(f"  {key:<20} {pre_v:>12{fmt}} {post_v:>12{fmt}} "
                  f"{sign}{delta:>11{fmt}}")

    # Write both captured outputs + summary
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/diagnose_option5.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Option 5 Step 4 walk-forward validation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"PRE metrics:  {pre_metrics}\n")
        f.write(f"POST metrics: {post_metrics}\n\n")
        f.write("=" * 70 + "\n")
        f.write("PRE backtest output (Option 5 disabled)\n")
        f.write("=" * 70 + "\n")
        f.write(pre_out)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("POST backtest output (Option 5 enabled)\n")
        f.write("=" * 70 + "\n")
        f.write(post_out)

    print(f"\n  Full outputs saved to: {out_path}")

    print("\n" + "#" * 70)
    print("#  Interpretation:")
    print("#    POST ROI > PRE by >1pp             -> clear alpha gain")
    print("#    |POST - PRE| <= 1pp, Brier better  -> neutral ROI, cleaner risk")
    print("#    POST ROI < PRE by >2pp             -> staking is over-conservative,")
    print("#                                          consider loosening caps")
    print("#" * 70)


if __name__ == "__main__":
    main()
