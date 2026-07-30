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

from data.build_canonical_dataset import (
    _LEAGUES,
    _backfill_canonical_values,
    _preserve_canonical_only_rows,
    _settings,
    build,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFL_CANONICAL = os.path.join(PROJECT_DIR, "CompleteDSChamp_CSV.csv")
PL_CANONICAL = os.path.join(PROJECT_DIR, "CompleteDSPL_CSV.csv")

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


# Columns the rebuild emits that the published canonical does not yet carry.
# Empty because the EFL canonical caught up: scripts/daily_ingest.py published
# the Relegated columns on its 2026-07-29 run, the morning after ADR 0007
# decision 2 landed. The assertion below is strict in both directions, so a
# stale entry here fails the suite — which is how that publish was noticed.
PENDING_NEW_COLUMNS: set[str] = set()

# The same idea in reverse: columns the rebuild deliberately drops that the
# published canonical still carries. ADR 0007 decision 6 removes the H2H win
# counts, and the canonical loses them at its next publish — which now needs
# `--allow-schema-change`, because daily_ingest's schema gate refuses an
# unattended column change. Empty this set once that publish has happened.
PENDING_REMOVED_COLUMNS = {"H2H_HomeWins", "H2H_AwayWins", "H2H_Draws"}


def test_schema_unchanged(efl_rebuild, efl_live):
    """The rebuild may only add or drop columns we have declared.

    Strict in both directions: an undeclared change fails, and so does a
    declaration that has gone stale.
    """
    added = set(efl_rebuild.columns) - set(efl_live.columns)
    lost = set(efl_live.columns) - set(efl_rebuild.columns)
    assert added == PENDING_NEW_COLUMNS, (
        f"rebuild added {sorted(added)}, expected {sorted(PENDING_NEW_COLUMNS)}")
    assert lost == PENDING_REMOVED_COLUMNS, (
        f"rebuild dropped {sorted(lost)}, "
        f"expected {sorted(PENDING_REMOVED_COLUMNS)}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Canonical row preservation
#
# football-data.co.uk serves only 335 rows for E0 seasons 3 and 4 against the
# 380 actually played. A rebuild that trusts the source destroys 90 real PL
# fixtures, which ADR 0001 forbids.
# ─────────────────────────────────────────────────────────────────────────────

def _row(season: int, home: str, away: str, goals: int = 1) -> dict:
    return {
        "SeasonIndex": season, "Date": f"0{season + 1}/05/200{season}",
        "Home_Team": home, "Away_Team": away,
        "Home_Goals": goals, "Away_Goals": 0, "TG": goals, "FTR": "H",
    }


def test_preserve_restores_rows_absent_upstream(tmp_path):
    """A canonical fixture the source no longer serves comes back."""
    canonical = pd.DataFrame([
        _row(3, "A", "B"), _row(3, "C", "D"), _row(3, "E", "F"),
    ])
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(3, "A", "B"), _row(3, "C", "D")])
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _preserve_canonical_only_rows(upstream, str(path), 0, 25)

    assert len(out) == 3
    assert ("E", "F") in set(zip(out["Home_Team"], out["Away_Team"]))


def test_preserve_is_noop_when_source_is_complete(tmp_path):
    """Nothing is added — or duplicated — when upstream serves every row."""
    rows = [_row(3, "A", "B"), _row(3, "C", "D")]
    path = tmp_path / "canonical.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    upstream = pd.DataFrame(rows)
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _preserve_canonical_only_rows(upstream, str(path), 0, 25)
    assert len(out) == 2


def test_preserve_carries_only_source_columns(tmp_path):
    """Computed features are not copied across — they are recomputed."""
    canonical = pd.DataFrame([_row(3, "A", "B"), _row(3, "E", "F")])
    canonical["Home Factor"] = 99.0  # a computed column, must not survive
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(3, "A", "B")])
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _preserve_canonical_only_rows(upstream, str(path), 0, 25)
    assert "Home Factor" not in out.columns


def test_preserve_ignores_seasons_outside_the_build_range(tmp_path):
    """A season the build did not attempt is not resurrected."""
    canonical = pd.DataFrame([_row(3, "A", "B"), _row(9, "X", "Y")])
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(3, "A", "B")])
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _preserve_canonical_only_rows(upstream, str(path), 0, 5)
    assert len(out) == 1


def test_preserve_tolerates_canonical_with_no_rows_in_range(tmp_path):
    """A canonical holding only out-of-range seasons must not break the mask.

    Pins a pandas behaviour the membership test leans on: ``.apply(tuple,
    axis=1)`` yields an empty *Series* here. Were it ever to yield a DataFrame,
    the mask would silently stop being a boolean index.
    """
    canonical = pd.DataFrame([_row(9, "X", "Y")])
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(3, "A", "B")])
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _preserve_canonical_only_rows(upstream, str(path), 0, 5)
    assert len(out) == 1


def test_preserve_tolerates_missing_canonical(tmp_path):
    """A first-ever build has nothing to preserve from and must not crash."""
    upstream = pd.DataFrame([_row(3, "A", "B")])
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)
    out = _preserve_canonical_only_rows(
        upstream, str(tmp_path / "absent.csv"), 0, 25)
    assert len(out) == 1


def test_backfill_restores_values_the_source_omits(tmp_path):
    """Preserving rows is not enough — cells go missing too.

    football-data.co.uk carries no B365 columns for E0 seasons 0-1, but the
    canonical holds 380 match-odds for each and 374 O/U prices for season 1.
    Those rows *are* served, so row preservation never sees them, and a
    rebuild silently blanks the odds the backtests price against.
    """
    canonical = pd.DataFrame([_row(0, "A", "B"), _row(0, "C", "D")])
    canonical["B365H"] = [2.5, 3.0]
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(0, "A", "B"), _row(0, "C", "D")])
    upstream["B365H"] = [float("nan"), float("nan")]
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _backfill_canonical_values(upstream, str(path), 0, 25)

    assert out["B365H"].tolist() == [2.5, 3.0]


def test_backfill_never_overwrites_what_the_source_provides(tmp_path):
    """The source is the authority where it speaks; backfill only fills gaps.

    Otherwise a genuine upstream correction would be silently reverted to the
    stale canonical value on every rebuild.
    """
    canonical = pd.DataFrame([_row(0, "A", "B")])
    canonical["B365H"] = [2.5]
    canonical["Home_Goals"] = [9]  # stale, since corrected upstream
    path = tmp_path / "canonical.csv"
    canonical.to_csv(path, index=False)

    upstream = pd.DataFrame([_row(0, "A", "B")])
    upstream["B365H"] = [float("nan")]
    upstream["Home_Goals"] = [1]  # the corrected value
    upstream["Date"] = pd.to_datetime(upstream["Date"], format="mixed",
                                      dayfirst=True)

    out = _backfill_canonical_values(upstream, str(path), 0, 25)

    assert out["B365H"].tolist() == [2.5], "gap should be filled"
    assert out["Home_Goals"].tolist() == [1], "correction must survive"


def test_preserve_reads_live_canonical_even_on_a_dry_run():
    """A dry run must preserve from the real canonical, not its own output.

    Otherwise `--output /tmp/x.csv` finds nothing to preserve and "reproduces"
    the build by dropping the very rows preservation exists to keep.
    """
    s = _settings("PL", output="/tmp/dry_run.csv")
    assert s["output_path"] == "/tmp/dry_run.csv"
    assert s["canonical_path"].endswith("CompleteDSPL_CSV.csv")


def test_preservation_precedes_feature_computation(monkeypatch):
    """Order is the whole point: rolling windows must see the completed frame.

    Restoring rows after feature computation would leave correct Facts
    carrying features derived from an incomplete season — the same corruption
    in a subtler form.
    """
    from data import build_canonical_dataset as mod

    order: list[str] = []

    def fake_preserve(df, canonical_path, first_season, last_season):
        order.append("preserve")
        return df

    def fake_rolling(df):
        order.append("rolling")
        raise RuntimeError("stop the build here")

    monkeypatch.setattr(mod, "_preserve_canonical_only_rows", fake_preserve)
    monkeypatch.setattr(mod, "_add_rolling_features", fake_rolling)

    with pytest.raises(RuntimeError, match="stop the build here"):
        mod.build(league="EFL", output="/tmp/unused.csv",
                  refresh_current_season=False)

    assert order == ["preserve", "rolling"]


def test_pl_rebuild_keeps_every_canonical_fixture(tmp_path):
    """End-to-end gate: the PL rebuild loses no historical match.

    Seasons 3 and 4 are the ones upstream truncates, so they are the
    assertion that matters.
    """
    if not os.path.exists(PL_CANONICAL):
        pytest.skip("PL canonical not present")
    raw_dir = _LEAGUES["PL"]["raw_dir"]
    if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
        pytest.skip("PL raw cache missing — rebuild would require network I/O")

    out = tmp_path / "pl_rebuild.csv"
    rebuilt = build(league="PL", output=str(out), refresh_current_season=False)
    live = pd.read_csv(PL_CANONICAL, low_memory=False)

    assert len(rebuilt) == len(live)
    got = rebuilt.groupby("SeasonIndex").size()
    want = live.groupby("SeasonIndex").size()
    pd.testing.assert_series_equal(got, want, check_names=False)
    assert got.loc[3] == 380 and got.loc[4] == 380

    key = ["SeasonIndex", "Home_Team", "Away_Team"]
    assert (set(map(tuple, rebuilt[key].itertuples(index=False, name=None)))
            == set(map(tuple, live[key].itertuples(index=False, name=None))))

    # Losslessness is about cells as well as rows: no source column may come
    # back with fewer values than the canonical already holds. Without the
    # backfill this loses 760 match prices and 375 O/U prices in seasons 0-1.
    source_columns = [
        "Home_Goals", "Away_Goals", "TG", "FTR", "HTHG", "HTAG", "HTR",
        "Home_Shots", "Away_Shots", "Home_Shots_Target", "Away_Shots_Target",
        "HF", "AF", "Home_Corners", "Away_Corners", "HY", "AY", "HR", "AR",
        "B365H", "B365D", "B365A", "B365Greater2.5", "B365LessThan2.5",
    ]
    shortfalls = {
        col: (int(live[col].notna().sum()), int(rebuilt[col].notna().sum()))
        for col in source_columns
        if col in live.columns
        and rebuilt[col].notna().sum() < live[col].notna().sum()
    }
    assert not shortfalls, f"rebuild lost values (live, rebuilt): {shortfalls}"


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
