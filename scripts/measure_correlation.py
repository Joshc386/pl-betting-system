"""
Option 5 Step 2a: empirical measurement of same-match market correlations.

Computes the joint outcome distribution and Pearson correlation coefficient
between pairs of same-match bet outcomes, using the full historical
DataFrame from both PL and EFL pipelines. The output drives the
same-match discount factor for Step 2c.

Market pairs measured (only include ones we actually bet):
  - (Over 2.5, BTTS)  - primary concern on most matchdays
  - (Over 1.5, BTTS)  - PL alt lines + EFL primary
  - (Over 1.5, Over 2.5) - nested O/U lines (mostly PL alt lines scan)
  - (Over 2.5, Over 3.5) - for completeness

Runtime: <30 seconds.
Run:     python scripts/measure_correlation.py
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _derive_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Derive all binary market outcomes from Home_Goals / Away_Goals."""
    df = df.copy()
    # Drop unplayed/future fixtures
    df = df[df["Home_Goals"].notna() & df["Away_Goals"].notna()].copy()
    df["total_goals"] = df["Home_Goals"] + df["Away_Goals"]
    df["over05"] = (df["total_goals"] > 0.5).astype(int)
    df["over15"] = (df["total_goals"] > 1.5).astype(int)
    df["over25"] = (df["total_goals"] > 2.5).astype(int)
    df["over35"] = (df["total_goals"] > 3.5).astype(int)
    df["btts"] = ((df["Home_Goals"] > 0) & (df["Away_Goals"] > 0)).astype(int)
    return df


def _joint_stats(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Compute joint + marginal + conditional stats for two binary outcomes."""
    a = df[col_a].values
    b = df[col_b].values
    n = len(df)

    p_a = a.mean()
    p_b = b.mean()
    p_both = ((a == 1) & (b == 1)).mean()
    p_neither = ((a == 0) & (b == 0)).mean()

    # Pearson correlation for binary variables
    rho = float(np.corrcoef(a, b)[0, 1])

    # Conditional probabilities
    p_b_given_a = p_both / p_a if p_a > 0 else np.nan
    p_b_given_not_a = ((a == 0) & (b == 1)).mean() / (1 - p_a) if p_a < 1 else np.nan

    # Independence-implied joint vs actual
    p_both_indep = p_a * p_b
    joint_lift = p_both / p_both_indep if p_both_indep > 0 else np.nan

    return {
        "n": n,
        "p_a": p_a, "p_b": p_b,
        "p_both": p_both, "p_both_indep": p_both_indep,
        "joint_lift": joint_lift,
        "p_b_given_a": p_b_given_a,
        "p_b_given_not_a": p_b_given_not_a,
        "rho": rho,
    }


def _kelly_discount_from_rho(rho: float) -> float:
    """Convert Pearson correlation to an approximate Kelly discount factor.

    Rationale: for two positively correlated binary bets with correlation rho,
    the joint variance grows as (1 + rho), so the effective "independent"
    Kelly stake per leg should be scaled down by approximately sqrt(1 - rho^2)
    to match the risk of a single uncorrelated bet at the same aggregate
    exposure. This is a first-order approximation and deliberately
    conservative (under-discounts as rho -> 0 where effect is small).
    """
    r = max(min(rho, 0.99), -0.99)  # guard against rho=1 / -1 edge cases
    return float(np.sqrt(1.0 - r * r))


def _print_pair_report(league: str, pair_name: str, stats: dict) -> None:
    discount = _kelly_discount_from_rho(stats["rho"])
    print(f"\n  {league} / {pair_name}")
    print(f"    n={stats['n']}  P(A)={stats['p_a']:.3f}  P(B)={stats['p_b']:.3f}")
    print(f"    P(A and B):       {stats['p_both']:.3f}  "
          f"(independent would be {stats['p_both_indep']:.3f}, "
          f"lift = {stats['joint_lift']:.2f}x)")
    print(f"    P(B|A):           {stats['p_b_given_a']:.3f}  "
          f"P(B|not A): {stats['p_b_given_not_a']:.3f}  "
          f"(delta = {stats['p_b_given_a'] - stats['p_b_given_not_a']:+.3f})")
    print(f"    Pearson rho:      {stats['rho']:+.4f}")
    print(f"    Kelly discount:   {discount:.3f}  "
          f"(applied when BOTH legs are bet on same fixture)")


def main():
    print("\n" + "#" * 70)
    print("#  Option 5 Step 2a: same-match market correlation measurement")
    print("#" * 70)

    pairs = [
        ("Over 2.5 vs BTTS",    "over25", "btts"),
        ("Over 1.5 vs BTTS",    "over15", "btts"),
        ("Over 1.5 vs Over 2.5", "over15", "over25"),
        ("Over 2.5 vs Over 3.5", "over25", "over35"),
    ]

    all_stats: list[dict] = []

    # --- PL ---
    print("\n[pipeline] Loading PL...")
    from pipeline import run_pipeline as pl_pipeline
    pl_df = _derive_outcomes(pl_pipeline(verbose=False)["full_df"])
    print(f"[pipeline] PL settled matches: {len(pl_df)}")
    print(f"\nPL results:")
    for label, a, b in pairs:
        stats = _joint_stats(pl_df, a, b)
        _print_pair_report("PL", label, stats)
        all_stats.append({"league": "PL", "pair": label, **stats})

    # --- EFL ---
    print("\n[pipeline] Loading EFL...")
    from championship_pipeline import run_pipeline as efl_pipeline
    efl_df = _derive_outcomes(efl_pipeline(verbose=False)["full_df"])
    print(f"[pipeline] EFL settled matches: {len(efl_df)}")
    print(f"\nEFL results:")
    for label, a, b in pairs:
        stats = _joint_stats(efl_df, a, b)
        _print_pair_report("EFL", label, stats)
        all_stats.append({"league": "EFL", "pair": label, **stats})

    # --- Summary table ---
    print("\n" + "#" * 70)
    print("#  SUMMARY: correlation and implied Kelly discount per pair")
    print("#" * 70)
    print(f"\n  {'League':<5} {'Pair':<26} {'rho':>7} {'joint_lift':>11} {'discount':>10}")
    for row in all_stats:
        discount = _kelly_discount_from_rho(row["rho"])
        print(f"  {row['league']:<5} {row['pair']:<26} "
              f"{row['rho']:>+7.4f} {row['joint_lift']:>11.2f} {discount:>10.3f}")

    # --- Persist ---
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/same_match_correlations.csv"
    df = pd.DataFrame(all_stats)
    df["kelly_discount"] = df["rho"].apply(_kelly_discount_from_rho)
    df.to_csv(out_path, index=False)
    print(f"\n  Full results saved to: {out_path}")

    # --- Recommendation ---
    print("\n" + "#" * 70)
    print("#  RECOMMENDATION for SAME_MATCH_DISCOUNT config")
    print("#" * 70)
    print("\n  Based on measured correlations, populate config.py with:")
    print("    SAME_MATCH_DISCOUNT = {")
    for row in all_stats:
        pair = row["pair"].replace(" vs ", "__")
        key = _pair_key(row["pair"])
        discount = _kelly_discount_from_rho(row["rho"])
        print(f"        {key:<40}: {discount:.3f},  "
              f"# {row['league']} rho={row['rho']:+.3f}")
    print("    }")


def _pair_key(pair_label: str) -> str:
    """Convert pair label to a Python-legal dict key (frozenset string)."""
    label_map = {
        "Over 2.5": "ou25",
        "Over 1.5": "ou15",
        "Over 3.5": "ou35",
        "BTTS":     "btts",
    }
    parts = [label_map[p.strip()] for p in pair_label.split(" vs ")]
    return f'frozenset({{"{parts[0]}", "{parts[1]}"}})'


if __name__ == "__main__":
    main()
