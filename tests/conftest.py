"""Shared pytest fixtures for the test suite.

The PL O/U fixture is loaded from a pickled snapshot of the pipeline
output (see scripts/build_test_fixture.py). All heavy work — pickle
load, walk-forward CV, stacker training — is session-scoped so it
runs once per pytest invocation regardless of how many tests use it.
"""
import os
import pickle

import numpy as np
import pytest


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "pl_ou_fixture.pkl"
)


@pytest.fixture(scope="session")
def pl_fixture():
    """Load the PL O/U 2.5 snapshot (DataFrame + features list)."""
    with open(FIXTURE_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="session")
def oof_df_for_stacker(pl_fixture):
    """Run walk_forward_cv on the fixture once and cache the OOF predictions."""
    from model import walk_forward_cv
    df = pl_fixture["df"]
    features = pl_fixture["features"]
    _, oof_df = walk_forward_cv(
        df, features,
        min_train_season=22,
        start_val_season=23,
    )
    return oof_df


@pytest.fixture(scope="session")
def trained_stacker(oof_df_for_stacker):
    """A LogReg stacker fit on real OOF predictions from the fixture."""
    from sklearn.linear_model import LogisticRegression
    X = oof_df_for_stacker[["xgb", "lgb", "lr", "dc"]].values
    y = oof_df_for_stacker["y"].values
    stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    stacker.fit(X, y)
    return stacker
