"""Guard: only football-data.co.uk may write Facts into a Canonical Dataset.

PL season 25 spent a whole season with no shots, corners, half-time scores or
odds because ``live_updater.update_dataset()`` scraped Understat, hardcoded
every stat column to ``NaN``, and wrote those rows into
``CompleteDSPL_CSV.csv`` — daily from the scheduler, and again immediately
before every weekly retrain.

ADR 0004 makes football-data.co.uk the sole Facts authority and demotes
Understat to xG enrichment. These tests hold that boundary, because the
failure mode is silent: the rows look fine, they are just structurally
poorer, and nothing in the schema records it.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from league_config import LEAGUES

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CANONICAL_BASENAMES = {
    os.path.basename(cfg["csv_path"]) for cfg in LEAGUES.values()
} | {
    os.path.basename(cfg["enriched_csv_path"]) for cfg in LEAGUES.values()
}


def _source(relpath: str) -> str:
    with open(os.path.join(PROJECT_DIR, relpath), encoding="utf-8") as f:
        return f.read()


def _strip_docstrings_and_comments(src: str) -> str:
    """Remove comments and string literals so only executable code remains.

    The retired write path is described in prose in these modules; prose must
    not trip the guard, but a real reference must.
    """
    import io
    import tokenize

    out = []
    prev_type = None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            # A string that stands alone as a statement is a docstring;
            # keep genuine string values (they could name a file).
            if prev_type in (tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
                             tokenize.DEDENT, None):
                continue
        out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            prev_type = tok.type
        else:
            prev_type = tok.type
    return "\n".join(out)


def test_live_updater_never_names_a_canonical() -> None:
    """The Understat module must not reference any canonical artefact in code."""
    code = _strip_docstrings_and_comments(_source("data/live_updater.py"))
    offenders = [name for name in CANONICAL_BASENAMES if name in code]
    assert not offenders, (
        f"data/live_updater.py references {offenders} in executable code. "
        "Understat is an xG source only (ADR 0004) and must never write Facts."
    )


def test_live_updater_exposes_no_dataset_writer() -> None:
    """The removed entry points must not come back under their old names."""
    import data.live_updater as lu

    for gone in ("update_dataset", "fetch_latest_matches", "load_main_csv"):
        assert not hasattr(lu, gone), (
            f"data.live_updater.{gone} is back. It wrote goals-only rows into "
            "the PL canonical; results come from build_canonical_dataset.py."
        )
    assert hasattr(lu, "refresh_xg"), "the xG scrape must be kept (ADR 0004)"


def test_scheduler_does_not_ingest_results() -> None:
    """Neither scheduled job may call the retired results path."""
    code = _strip_docstrings_and_comments(_source("scheduler.py"))
    for gone in ("update_dataset", "_fetch_latest_matches"):
        assert gone not in code, (
            f"scheduler.py calls {gone}. Match results are ingested by "
            "scripts/daily_ingest.py under Task Scheduler (ADR 0006)."
        )
    assert "_refresh_understat_xg" in code


@pytest.mark.parametrize("league", [
    "EFL",
    # PL season 25 is still goals-only in the live canonical: the Understat
    # writer has been retired, but the merge that repairs the damage it did
    # has not been published yet. strict=True so this turns into a failure
    # the moment the merge lands — that is the signal to delete this marker,
    # not a reason to loosen the test.
    pytest.param("PL", marks=pytest.mark.xfail(
        strict=True,
        reason="PL canonical not yet repaired — publish the Facts merge, "
               "then remove this xfail",
    )),
])
def test_canonical_has_no_facts_only_rows(league: str) -> None:
    """No season may carry goals while every other Fact column is empty.

    That signature — results present, all match stats absent — is what a
    thinner source leaves behind, and it is invisible unless you look for it.
    """
    path = LEAGUES[league]["csv_path"]
    if not os.path.exists(path):
        pytest.skip(f"{league} canonical not present")

    df = pd.read_csv(path, low_memory=False)
    stat_cols = [c for c in ("Home_Shots", "Away_Shots", "Home_Corners",
                             "Away_Corners", "HTHG", "HTAG")
                 if c in df.columns]
    assert stat_cols, "canonical is missing the match-stat columns entirely"

    empty_seasons = []
    for season, grp in df.groupby("SeasonIndex"):
        if grp["Home_Goals"].notna().sum() == 0:
            continue  # no results at all — a different problem
        if all(grp[c].notna().sum() == 0 for c in stat_cols):
            empty_seasons.append(int(season))

    assert not empty_seasons, (
        f"{league} season(s) {empty_seasons} have results but no match stats "
        f"({stat_cols}) — the signature of ingestion from a thinner source."
    )
