"""A feature-schema change must not reach the live canonical unattended.

`scripts/daily_ingest.py` rebuilds and publishes the EFL canonical every
morning under Task Scheduler. On 2026-07-29 it picked up ADR 0007 decision 2
and published a 74-column canonical — correct work, but it went live overnight
with no human in the loop, against a model trained on the previous schema.

`_facts_regressed` already refuses to publish when the *Facts* regress. This is
the same idea for the *schema*: adding or removing a feature column is a
deliberate act that belongs with a retrain, so it needs an explicit override
rather than arriving by cron.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.daily_ingest import _schema_changed


def _frame(columns: list[str]) -> pd.DataFrame:
    """Minimal frame with the columns the publish path actually reads."""
    df = pd.DataFrame({c: [1, 2] for c in columns})
    if "Date" in columns:
        df["Date"] = ["2026-05-01", "2026-05-02"]
    return df


# "Date" is present because ingest_league logs the latest fixture from it.
BASE = ["Date", "SeasonIndex", "Home_Team", "Away_Team", "Home_Goals"]


def test_identical_schema_is_allowed(tmp_path):
    """The normal daily case publishes without complaint."""
    live = tmp_path / "canonical.csv"
    _frame(BASE).to_csv(live, index=False)
    assert _schema_changed(str(live), _frame(BASE)) is None


def test_added_column_is_refused(tmp_path):
    """A new feature column must not go live unattended."""
    live = tmp_path / "canonical.csv"
    _frame(BASE).to_csv(live, index=False)

    problem = _schema_changed(str(live), _frame(BASE + ["Home_Relegated"]))

    assert problem is not None
    assert "Home_Relegated" in problem


def test_removed_column_is_refused(tmp_path):
    """Dropping a column the model may train on is equally deliberate."""
    live = tmp_path / "canonical.csv"
    _frame(BASE + ["Home_Promoted"]).to_csv(live, index=False)

    problem = _schema_changed(str(live), _frame(BASE))

    assert problem is not None
    assert "Home_Promoted" in problem


def test_first_build_has_nothing_to_compare(tmp_path):
    """No live canonical yet — the gate cannot and should not fire."""
    assert _schema_changed(str(tmp_path / "absent.csv"), _frame(BASE)) is None


def test_column_order_alone_is_not_a_change(tmp_path):
    """Reordering is not a schema change; the column *set* is what matters."""
    live = tmp_path / "canonical.csv"
    _frame(BASE).to_csv(live, index=False)
    assert _schema_changed(str(live), _frame(list(reversed(BASE)))) is None


# ─────────────────────────────────────────────────────────────────────────────
# The gate has to be wired into the publish path, not merely defined
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point ingest_league at a temp canonical and a stubbed build()."""
    import scripts.daily_ingest as mod

    live = tmp_path / "canonical.csv"
    _frame(BASE).to_csv(live, index=False)

    monkeypatch.setattr(mod, "_settings",
                        lambda league: {"output_path": str(live)})
    # Facts gate must not be what stops us — we are testing the schema gate.
    monkeypatch.setattr(mod, "_facts_regressed", lambda path, df: None)
    monkeypatch.setattr(mod, "_prune_backups", lambda path, keep=5: 0)
    return mod, live


def _stub_build(columns: list[str]):
    """Stand in for build(): writes the candidate out, as the real one does."""
    def _build(league, output):
        df = _frame(columns)
        df.to_csv(output, index=False)
        return df
    return _build


def test_publish_is_refused_when_the_schema_changed(wired, monkeypatch):
    """An unattended run must not put a new feature column live."""
    mod, live = wired
    monkeypatch.setattr(mod, "build", _stub_build(BASE + ["Home_Relegated"]))

    assert mod.ingest_league("EFL") is False
    # The live canonical is untouched.
    assert "Home_Relegated" not in pd.read_csv(live, nrows=0).columns


def test_override_allows_the_schema_change(wired, monkeypatch):
    """With the flag set, the same publish goes through."""
    mod, live = wired
    monkeypatch.setattr(mod, "build", _stub_build(BASE + ["Home_Relegated"]))

    assert mod.ingest_league("EFL", allow_schema_change=True) is True
    assert "Home_Relegated" in pd.read_csv(live, nrows=0).columns


def test_unchanged_schema_still_publishes(wired, monkeypatch):
    """The ordinary daily run is unaffected by the gate."""
    mod, live = wired
    monkeypatch.setattr(mod, "build", _stub_build(BASE))

    assert mod.ingest_league("EFL") is True


# ── Facts compare by fixture key, not row position ──────────────────────────
#
# The builder's output order is not part of the contract: preserved rows
# (absent upstream) can land at different positions depending on which live
# file they were read back from, and a reordering is not a fact change.
# Date compares at day precision — seasons 24-25 of the PL canonical carried
# FotMob kickoff times that football-data.co.uk does not serve.

from scripts.daily_ingest import _facts_regressed


def _facts_frame():
    return pd.DataFrame({
        "SeasonIndex": [3, 3, 4],
        "Home_Team": ["A", "B", "A"],
        "Away_Team": ["B", "A", "B"],
        "Date": ["2003-08-16", "2003-12-01", "2004-08-14"],
        "Home_Goals": [1, 0, 2],
        "Away_Goals": [0, 0, 2],
        "TG": [1, 0, 4],
        "FTR": ["H", "D", "D"],
        # a current season so the rows above count as historical
        **{},
    })


def _with_current(df):
    cur = pd.DataFrame({
        "SeasonIndex": [9], "Home_Team": ["A"], "Away_Team": ["B"],
        "Date": ["2009-08-15"], "Home_Goals": [1], "Away_Goals": [1],
        "TG": [2], "FTR": ["D"],
    })
    return pd.concat([df, cur], ignore_index=True)


def test_row_order_alone_is_not_a_fact_change(tmp_path):
    live = _with_current(_facts_frame())
    path = tmp_path / "live.csv"
    live.to_csv(path, index=False)
    shuffled = _with_current(_facts_frame().iloc[::-1])
    assert _facts_regressed(str(path), shuffled) is None


def test_a_changed_fact_value_is_still_refused(tmp_path):
    live = _with_current(_facts_frame())
    path = tmp_path / "live.csv"
    live.to_csv(path, index=False)
    tampered = _facts_frame()
    tampered.loc[0, "Home_Goals"] = 3
    problem = _facts_regressed(str(path), _with_current(tampered))
    assert problem is not None and "Home_Goals" in problem


def test_date_compares_at_day_precision(tmp_path):
    """A kickoff time on one side of the comparison is not a moved match."""
    live = _facts_frame()
    live.loc[0, "Date"] = "2003-08-16 19:00:00"
    path = tmp_path / "live.csv"
    _with_current(live).to_csv(path, index=False)
    assert _facts_regressed(str(path), _with_current(_facts_frame())) is None
