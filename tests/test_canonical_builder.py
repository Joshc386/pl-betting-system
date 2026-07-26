"""Regression tests for the league-parameterised canonical dataset builder.

The builder was generalised from an EFL-only script into a two-league one
(ADR 0004). EFL is the league we are *not* changing, so rebuilding it and
comparing against the live canonical proves the generalisation is
behaviour-preserving — the refactor's regression gate.

The EFL rebuild reads the cached raw CSVs in ``data/championship_raw`` and
performs no network I/O when that cache is complete.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from data.build_canonical_dataset import _LEAGUES, _settings, build

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFL_CANONICAL = os.path.join(PROJECT_DIR, "CompleteDSChamp_CSV.csv")

# Facts are copied verbatim from the raw source and must be byte-identical
# across a regeneration (ADR 0001). Everything else is a computed feature
# and may legitimately drift when build logic improves.
FACT_COLUMNS = [
    "Date", "Home_Team", "Away_Team", "Home_Goals", "Away_Goals",
    "TG", "FTR", "HTHG", "HTAG", "HTR", "SeasonIndex",
]


@pytest.fixture(scope="module")
def efl_rebuild(tmp_path_factory) -> pd.DataFrame:
    """Rebuild the EFL canonical to a temp path (never overwrites the live file)."""
    if not os.path.exists(EFL_CANONICAL):
        pytest.skip("EFL canonical not present — nothing to compare against")
    raw_dir = _LEAGUES["EFL"]["raw_dir"]
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        pytest.skip("EFL raw cache missing — rebuild would require network I/O")

    out = tmp_path_factory.mktemp("canonical") / "efl_rebuild.csv"
    # refresh_current_season=False keeps the test hermetic — it must compare
    # against the same raws that built the live canonical, not whatever
    # football-data.co.uk is serving today.
    return build(league="EFL", output=str(out), refresh_current_season=False)


@pytest.fixture(scope="module")
def efl_live() -> pd.DataFrame:
    if not os.path.exists(EFL_CANONICAL):
        pytest.skip("EFL canonical not present")
    return pd.read_csv(EFL_CANONICAL)


def _normalise_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort and type-normalise so comparison is order- and dtype-insensitive."""
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], format="mixed", dayfirst=True,
                                 errors="coerce")
    out = out.sort_values(["Date", "Home_Team", "Away_Team"])
    return out.reset_index(drop=True)


def test_row_count_unchanged(efl_rebuild, efl_live):
    """Hard gate (ADR 0001): no historical matches lost, duplicated or shifted."""
    assert len(efl_rebuild) == len(efl_live)


def test_per_season_row_counts_unchanged(efl_rebuild, efl_live):
    """Row counts must match season by season, not just in total."""
    got = efl_rebuild.groupby("SeasonIndex").size()
    want = efl_live.groupby("SeasonIndex").size()
    pd.testing.assert_series_equal(got, want, check_names=False)


def test_schema_unchanged(efl_rebuild, efl_live):
    """The rebuild must emit exactly the canonical's columns."""
    assert set(efl_rebuild.columns) == set(efl_live.columns)


def test_facts_are_byte_identical(efl_rebuild, efl_live):
    """Hard stop (ADR 0001): a Facts difference means source or mapping drift."""
    got = _normalise_for_compare(efl_rebuild)
    want = _normalise_for_compare(efl_live)
    for col in FACT_COLUMNS:
        assert col in got.columns, f"rebuild is missing Fact column {col!r}"
        pd.testing.assert_series_equal(
            got[col], want[col],
            check_dtype=False, check_names=False,
            obj=f"Fact column {col!r}",
        )


def test_efl_team_names_are_not_normalised(efl_rebuild):
    """EFL keeps football-data.co.uk's short forms.

    Normalising would rewrite 23 of 24 names and break the ESPN/odds mappings
    and the Betfair League Split allowlists (ADR 0004).
    """
    assert _LEAGUES["EFL"]["normalize_names"] is False
    names = set(efl_rebuild["Home_Team"])
    assert "Blackburn" in names
    assert "Blackburn Rovers FC" not in names


def test_pl_settings_select_e0_and_normalise():
    """PL must read division E0 and normalise names to canonical long form."""
    s = _settings("PL")
    assert s["div"] == "E0"
    assert s["normalize_names"] is True
    assert s["output_path"].endswith("CompleteDSPL_CSV.csv")


def test_efl_settings_select_e1_verbatim():
    s = _settings("EFL")
    assert s["div"] == "E1"
    assert s["normalize_names"] is False
    assert s["output_path"].endswith("CompleteDSChamp_CSV.csv")


def test_output_override_does_not_touch_live_canonical():
    """Dry runs must never write over the live canonical."""
    s = _settings("PL", output="/tmp/scratch.csv")
    assert s["output_path"] == "/tmp/scratch.csv"


def test_unknown_league_rejected():
    with pytest.raises(ValueError, match="Unknown league"):
        _settings("LaLiga")


def test_current_season_bypasses_cache(monkeypatch):
    """The current season must re-download; finished seasons must not.

    Without this the canonical silently freezes at whatever gameweek the
    current-season raw was first cached — the failure mode that makes
    automated ingestion useless.
    """
    from data import build_canonical_dataset as mod

    calls: list[tuple[int, bool]] = []

    def fake_download(season_idx, div, raw_dir, use_cache=True):
        calls.append((season_idx, use_cache))
        return None  # skip the rest of the build

    monkeypatch.setattr(mod, "download_season", fake_download)
    with pytest.raises(RuntimeError):  # no data -> build aborts, as designed
        mod.build(league="EFL", output="/tmp/unused.csv")

    current = _settings("EFL")["last_season"]
    by_season = dict(calls)
    assert by_season[current] is False, "current season must bypass the cache"
    assert all(v is True for k, v in by_season.items() if k != current), \
        "finished seasons must use the cache"


def test_no_refresh_flag_caches_every_season(monkeypatch):
    """refresh_current_season=False keeps the build fully offline."""
    from data import build_canonical_dataset as mod

    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        mod, "download_season",
        lambda season_idx, div, raw_dir, use_cache=True: (
            calls.append((season_idx, use_cache)) or None),
    )
    with pytest.raises(RuntimeError):
        mod.build(league="EFL", output="/tmp/unused.csv",
                  refresh_current_season=False)

    assert all(use_cache for _, use_cache in calls)
