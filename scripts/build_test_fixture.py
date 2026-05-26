"""Build the test fixture used by tests/test_stacker.py.

Snapshots a small, recent slice of the PL pipeline output to disk
so tests can exercise the real model contract without re-running
the full pipeline (which hits external APIs and takes ~90s).

Run once locally; commit the resulting pickle. Re-run only when the
pipeline schema changes in a way that breaks the snapshot.

Usage:
    python scripts/build_test_fixture.py
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "fixtures", "pl_ou_fixture.pkl",
)

MIN_SEASON_INDEX = 22  # Snapshot last 4 seasons (22-25)

REQUIRED_META_COLS = [
    "Over_2_5", "SeasonIndex", "Date",
    "Home_Team", "Away_Team",
    "Home_Goals", "Away_Goals",
    "home_xg", "away_xg",
]


def main():
    print("Running pipeline (this hits external APIs and takes ~90s)...")
    result = run_pipeline(verbose=False)
    full_df = result["full_df"]
    features = result["features"]

    snap = full_df[full_df["SeasonIndex"] >= MIN_SEASON_INDEX].copy()
    cols_to_keep = list(dict.fromkeys(features + REQUIRED_META_COLS))
    snap = snap[cols_to_keep].reset_index(drop=True)

    print(f"Snapshot shape: {snap.shape}")
    print(f"Season range: {snap['SeasonIndex'].min()} to {snap['SeasonIndex'].max()}")
    print(f"Rows per season:\n{snap.groupby('SeasonIndex').size()}")

    payload = {"df": snap, "features": features}

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(FIXTURE_PATH) / (1024 * 1024)
    print(f"Wrote {FIXTURE_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
