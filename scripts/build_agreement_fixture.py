"""Build the fixture used by tests/test_agreement_analysis.py.

Snapshots the replayed pre-gate rows for one OOF cell so the known-output
regression test runs on any checkout. `reports/` is gitignored, so the OOF
caches themselves are build artefacts the test cannot rely on.

Only the columns `agreement_bins` reads are kept — no model internals.

Run once locally; commit the resulting CSV. Re-run only when the gate
arithmetic or the cache schema changes, and expect the regression values
in the test to move if it does.

Usage:
    python scripts/build_agreement_fixture.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_analytics import load_oof_cell, replay_oof_gate  # noqa: E402

LEAGUE, MARKET = "PL", "ou25"
KEEP = ["fixture", "season", "side_col", "n_models", "n_agree",
        "fair_prob", "edge", "won", "passes"]
FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "agreement_pl_ou25_replayed.csv",
)


def main() -> int:
    oof = load_oof_cell(LEAGUE, MARKET)
    if oof is None:
        print(f"[ERROR] No OOF cache for {LEAGUE} {MARKET}. "
              f"Run scripts/generate_oof_cache.py first.", file=sys.stderr)
        return 1

    replayed = replay_oof_gate(oof)
    # side_a only — the shipped pre-gate view keeps one side per fixture.
    out = replayed[replayed["side_col"] == "a"][KEEP]

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    out.to_csv(FIXTURE_PATH, index=False)
    print(f"Wrote {FIXTURE_PATH}: {len(out):,} rows, "
          f"{out['fixture'].nunique():,} fixtures")
    print(f"Seasons: {sorted(out['season'].unique())}")
    print("\nPre-gate bins (side_a only):")
    for n, grp in out.groupby("n_agree"):
        re_ = grp["won"].mean() - grp["fair_prob"].mean()
        print(f"  n_agree={n}: {len(grp):>5,} rows  realised={re_:+.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
