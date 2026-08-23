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
    ColumnCoverageError,
    assert_column_coverage,
    coverage_regressions,
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
# Both ledgers are EMPTY as of the 2026-08-03 publish, which carried ADR
# 0007's four new Scoring columns in and its nine retired ones out under
# `--allow-schema-change`. The assertion below is strict in both
# directions, so a stale entry here fails the suite just as an undeclared
# change does — that is how the 2026-07-29 Relegated publish was noticed.
#
# Empty again: the ADR 0003 odds columns (Odds_Over/Under/Source_{1.5,2.5})
# were declared here and published on 2026-08-04, so the declaration went
# stale the moment the publish landed and this test said so. B365Greater2.5
# / B365LessThan2.5 remain in both canonicals — they are Facts from
# football-data.co.uk and the fallback those columns coalesce over, not
# duplicates of them.
PENDING_NEW_COLUMNS: set[str] = set()
PENDING_REMOVED_COLUMNS: set[str] = set()


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


# ── ADR 0007 decision 7: Factor retired, ScoringRate_10 + ScoringIndex_10 ──
#
# "Factor" denoted the rolling-10 mean in the PL canonical and that mean
# divided by the season's league average in the EFL one. Both quantities are
# now emitted under names that say which is which; the old column is gone.

def _scoring_fixture() -> pd.DataFrame:
    """A hand-computable mini-season.

    A hosts B four times; C hosts D twice in between. Before A's fourth home
    match, A's home history is [2, 4, 0] and B's away history is [1, 1, 1] —
    both exactly at the 3-match minimum.
    """
    rows = []
    fixtures = [
        ("A", "B", 2, 1),
        ("C", "D", 1, 0),
        ("A", "B", 4, 1),
        ("C", "D", 1, 0),
        ("A", "B", 0, 1),
        ("A", "B", 3, 2),  # features computed BEFORE this match
    ]
    for i, (h, a, hg, ag) in enumerate(fixtures):
        rows.append({
            "SeasonIndex": 3,
            "Date": pd.Timestamp("2003-08-01") + pd.Timedelta(days=i),
            "Home_Team": h, "Away_Team": a,
            "Home_Goals": hg, "Away_Goals": ag,
        })
    return pd.DataFrame(rows)


def test_scoring_rate_is_the_rolling_mean():
    """ScoringRate_10 is the raw rolling-10 venue mean (the PL semantic)."""
    from data.build_canonical_dataset import _add_scoring_features

    out = _add_scoring_features(_scoring_fixture())
    last = out.iloc[-1]
    assert last["Home_ScoringRate_10"] == pytest.approx((2 + 4 + 0) / 3)
    assert last["Away_ScoringRate_10"] == pytest.approx(1.0)


def test_scoring_index_is_the_rate_over_league_average():
    """ScoringIndex_10 divides the rate by the season's venue average
    (the EFL semantic). League home goals seen before the last match are
    A's [2, 4, 0] and C's [1, 1]; away are B's [1, 1, 1] and D's [0, 0]."""
    from data.build_canonical_dataset import _add_scoring_features

    out = _add_scoring_features(_scoring_fixture())
    last = out.iloc[-1]
    assert last["Home_ScoringIndex_10"] == pytest.approx(2.0 / 1.6)
    assert last["Away_ScoringIndex_10"] == pytest.approx(1.0 / 0.6)


def test_scoring_features_need_three_prior_venue_matches():
    """Below the 3-match minimum every scoring column stays NaN."""
    from data.build_canonical_dataset import _add_scoring_features

    out = _add_scoring_features(_scoring_fixture())
    early = out.iloc[:-1]
    for col in ("Home_ScoringRate_10", "Home_ScoringIndex_10",
                "Away_ScoringRate_10", "Away_ScoringIndex_10"):
        assert early[col].isna().all(), f"{col} appeared before 3 matches"


def test_the_factor_name_is_retired():
    """The builder must not emit the retired column under either name."""
    from data.build_canonical_dataset import _add_scoring_features

    out = _add_scoring_features(_scoring_fixture())
    assert "Home Factor" not in out.columns
    assert "Away Factor" not in out.columns


# ── ADR 0007 decision 3 / ADR 0002: matchday-1 seeding by previous season ──
#
# Before any games are played every team is on zero points, and the old code
# ranked them alphabetically — pure noise as a model feature. The seed is the
# previous season's outcome: returning teams keep their finishing position,
# promoted teams enter at the bottom by route (champion 18th PL / runner-up
# 19th / play-off winner 20th), relegated teams enter the EFL at the top in
# their PL finishing order, and League One arrivals (whose table this system
# does not hold) take the neutral 2nd-from-bottom seed. Once games are
# played, the live table applies — points, goal difference, goals scored —
# with the seed as the final tie-break in place of the alphabet.

def _m(date, home, away, hg, ag, season):
    ftr = "H" if hg > ag else ("A" if hg < ag else "D")
    return {"Date": pd.Timestamp(date), "SeasonIndex": season,
            "Home_Team": home, "Away_Team": away,
            "Home_Goals": hg, "Away_Goals": ag, "FTR": ftr}


def _pl_two_seasons(tmp_path):
    """A complete synthetic PL season 1 and a season-2 opening day.

    Season 1: fixture i is A{i} beating A{21-i} by i goals to nil, so the
    final table is A10 (GD +10) down to A01 in 1st-10th, then A20 (GD -1)
    down to A11 in 11th-20th. Relegated: A13, A12, A11.

    Season 2 arrivals B01, B02, B03 come from a sibling EFL whose season-1
    table reads B01 champion, B02 runner-up, B03 3rd (the play-off winner
    here), B04 4th.
    """
    rows = []
    for i in range(1, 11):
        rows.append(_m(f"2001-01-{i:02d}", f"A{i:02d}", f"A{21 - i:02d}", i, 0, 1))
    # Season 2, matchday 1. The table is live even inside a matchday, so
    # every day-1 result is a 0-0 draw: drawn sides gain a point but no GD
    # or goals, leaving the seed-ordering of later openers intact.
    rows.append(_m("2001-08-01", "A10", "A01", 0, 0, 2))   # champion v 10th
    rows.append(_m("2001-08-01", "B01", "B02", 0, 0, 2))   # the two arrivals
    rows.append(_m("2001-08-01", "A02", "B03", 0, 0, 2))   # play-off arrival
    # Fillers so every season-2 team is in the table from day one.
    fillers = [("A03", "A04"), ("A05", "A06"), ("A07", "A08"),
               ("A09", "A14"), ("A15", "A16"), ("A17", "A18"),
               ("A19", "A20")]
    for h, a in fillers:
        rows.append(_m("2001-08-02", h, a, 0, 0, 2))
    # Day 2: A10 and A01 each win 1-0 — 4 points, +1 GD, 1 goal scored apiece.
    rows.append(_m("2001-08-03", "A10", "A14", 1, 0, 2))
    rows.append(_m("2001-08-03", "A20", "A01", 0, 1, 2))
    # Day 3: dead level on points, GD and goals scored — the seed must break
    # the tie, not the alphabet.
    rows.append(_m("2001-08-05", "A01", "A10", 0, 0, 2))
    df = pd.DataFrame(rows)

    sib_rows = [
        _m("2001-01-01", "B01", "B04", 3, 0, 1),
        _m("2001-01-02", "B02", "B03", 1, 0, 1),
    ]
    sib_path = tmp_path / "sibling_efl.csv"
    pd.DataFrame(sib_rows).to_csv(sib_path, index=False)
    return df, str(sib_path)


def _seeded_pl(tmp_path, monkeypatch):
    from data import build_canonical_dataset as mod

    df, sib_path = _pl_two_seasons(tmp_path)
    # The synthetic league has 20 distinct teams in season 1, matching the
    # real PL count, so the completeness check passes unmodified.
    return mod._add_league_position(df, "PL", sib_path)


def test_matchday1_returning_teams_keep_their_finishing_position(tmp_path, monkeypatch):
    out = _seeded_pl(tmp_path, monkeypatch)
    opener = out[(out["SeasonIndex"] == 2) & (out["Home_Team"] == "A10")].iloc[0]
    assert opener["Home_LeaguePosition"] == 1          # last season's champion
    assert opener["Away_LeaguePosition"] == 10         # A01 finished 10th


def test_matchday1_promoted_teams_seed_by_route(tmp_path, monkeypatch):
    out = _seeded_pl(tmp_path, monkeypatch)
    s2 = out[out["SeasonIndex"] == 2]
    b_row = s2[s2["Home_Team"] == "B01"].iloc[0]
    assert b_row["Home_LeaguePosition"] == 18          # EFL champion
    assert b_row["Away_LeaguePosition"] == 19          # runner-up (B02)
    assert s2[s2["Home_Team"] == "A02"].iloc[0]["Away_LeaguePosition"] == 20  # play-off (B03)


def test_equal_points_break_on_seed_not_alphabet(tmp_path, monkeypatch):
    """By day 3, A01 and A10 both have 4 points, +1 GD, 1 goal scored. A10's
    seed (champion, 1) must rank it above A01 (10th) — alphabetically A01
    would win, which is the old noise this decision removes."""
    out = _seeded_pl(tmp_path, monkeypatch)
    decider = out[(out["SeasonIndex"] == 2) & (out["Home_Team"] == "A01")
                  & (out["Away_Team"] == "A10")].iloc[0]
    assert decider["Away_LeaguePosition"] == 1
    assert decider["Home_LeaguePosition"] == 2


def test_matchday1_positions_span_the_full_roster(tmp_path, monkeypatch):
    """Openers used to rank only the teams already seen — the first fixture's
    away side always showed 2nd. The whole roster is in the table from day
    one, so A01 shows its seeded 10th, not 2nd."""
    out = _seeded_pl(tmp_path, monkeypatch)
    opener = out[(out["SeasonIndex"] == 2) & (out["Home_Team"] == "A10")].iloc[0]
    assert opener["Away_LeaguePosition"] != 2


def test_goals_scored_breaks_ties_before_the_seed(tmp_path):
    """Mid-season rank is points, GD, then goals scored (ADR 0002). C and E
    both won by two; E scored three to C's two, so E ranks above C even
    though C is alphabetically first and no seed exists (single season)."""
    from data import build_canonical_dataset as mod

    df = pd.DataFrame([
        _m("2001-01-01", "C", "D", 2, 0, 1),
        _m("2001-01-01", "E", "F", 3, 1, 1),
        _m("2001-01-08", "C", "E", 0, 0, 1),
    ])
    out = mod._add_league_position(df, "PL", None)
    decider = out.iloc[-1]
    assert decider["Away_LeaguePosition"] == 1     # E, on goals scored
    assert decider["Home_LeaguePosition"] == 2     # C


def test_efl_relegated_arrivals_seed_top_in_pl_finish_order(tmp_path):
    """EFL season 2: P01-P03 came down from the PL (P01 finished best), so
    they seed 1, 2, 3. L01-L03 came up from League One, whose table this
    system does not hold: all take the neutral 23rd, ranked between the
    returning sides by alphabet."""
    from data import build_canonical_dataset as mod

    rows = []
    for i in range(1, 13):
        rows.append(_m(f"2001-01-{i:02d}", f"E{i:02d}", f"E{25 - i:02d}", i, 0, 1))
    # Season 2 openers: relegated P01 hosts League One arrival L01;
    # P02 hosts P03.
    rows.append(_m("2001-08-01", "P01", "L01", 1, 0, 2))
    rows.append(_m("2001-08-01", "P02", "P03", 1, 0, 2))
    # Fillers: the 18 returning sides (E10-E12 went up, E13-E15 went down)
    # plus L02 and L03, so the full 24-team roster is in the table.
    fillers = [("E01", "E02"), ("E03", "E04"), ("E05", "E06"),
               ("E07", "E08"), ("E09", "E16"), ("E17", "E18"),
               ("E19", "E20"), ("E21", "E22"), ("E23", "E24"),
               ("L02", "L03")]
    for h, a in fillers:
        rows.append(_m("2001-08-02", h, a, 0, 0, 2))
    df = pd.DataFrame(rows)

    sib_rows = [
        _m("2001-01-01", "P01", "P04", 3, 0, 1),
        _m("2001-01-02", "P02", "P05", 2, 0, 1),
        _m("2001-01-03", "P03", "P06", 1, 0, 1),
    ]
    sib_path = tmp_path / "sibling_pl.csv"
    pd.DataFrame(sib_rows).to_csv(sib_path, index=False)

    out = mod._add_league_position(df, "EFL", str(sib_path))
    s2 = out[out["SeasonIndex"] == 2]
    p_row = s2[s2["Home_Team"] == "P01"].iloc[0]
    assert p_row["Home_LeaguePosition"] == 1
    assert p_row["Away_LeaguePosition"] == 22      # first of the three 23-seeds
    pp_row = s2[s2["Home_Team"] == "P02"].iloc[0]
    assert pp_row["Home_LeaguePosition"] == 2
    assert pp_row["Away_LeaguePosition"] == 3


def test_first_season_has_no_seed_and_keeps_alphabetical_order(tmp_path):
    """Season 0 has nothing to seed from — the documented default is the
    old behaviour: every team equal, alphabet decides."""
    from data import build_canonical_dataset as mod

    df = pd.DataFrame([
        _m("2000-08-01", "Z", "M", 0, 0, 0),
    ])
    out = mod._add_league_position(df, "PL", None)
    assert out.iloc[0]["Home_LeaguePosition"] == 2  # Z after M
    assert out.iloc[0]["Away_LeaguePosition"] == 1


# ── Division guard: football-data.co.uk mod_speling redirects ──────────────
#
# Requesting a season football-data.co.uk has not published yet does not 404.
# Apache's mod_speling finds the nearest filename and 301-redirects to it, so
# `/mmz4281/2627/E0.csv` answers with `EC.csv` — the National League. requests
# follows redirects by default and `raise_for_status()` only ever sees the
# final 200, so the download path cannot tell the difference. Confirmed live
# on 2026-08-14, the day the EFL season opened:
#
#     HTTP/1.1 301 Moved Permanently
#     Location: https://www.football-data.co.uk/mmz4281/2627/EC.csv
#
# The rows that come back are well-formed — real teams, real scores, every
# column the malformed-row filter checks — so nothing downstream rejects them.
# They would be stamped with the requested SeasonIndex and published as
# Premier League Facts, which ADR 0004 makes football-data.co.uk's `E0` alone.

_EC_PAYLOAD = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR\n"
    "EC,08/08/2026,15:00,Altrincham,Southend,1,3,A,1,1,D\n"
    "EC,08/08/2026,15:00,Boreham Wood,Tamworth,3,3,D,1,1,D\n"
)

_E0_PAYLOAD = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HTR\n"
    "E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1,0,H\n"
)


class _FakeResponse:
    """Only the surface ``download_season`` touches, decoded as requests does.

    football-data.co.uk sends its CSVs with a UTF-8 BOM and no charset in the
    Content-Type, so requests falls back to ISO-8859-1 (RFC 2616) and
    ``.text`` renders those three BOM bytes as the characters ``ï»¿`` — the
    first column arrives named ``ï»¿Div``, not ``Div``. Modelling that is the
    entire point of this fake: an earlier version handed back clean text, and
    every test below passed while the guard rejected all four real seasons.
    """

    def __init__(self, body: str) -> None:
        self.content = b"\xef\xbb\xbf" + body.encode("utf-8")
        self.text = self.content.decode("iso-8859-1")

    def raise_for_status(self) -> None:
        return None  # a followed 301 lands on 200


def _serve(monkeypatch, body: str) -> None:
    """Serve *body* the way the real source does, BOM and all."""
    from data import build_canonical_dataset as mod
    monkeypatch.setattr(mod.requests, "get",
                        lambda *a, **k: _FakeResponse(body))
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)


def test_download_rejects_a_different_division(monkeypatch, tmp_path, capsys):
    """Asking for E0 and being served EC yields nothing, not National League.

    The reason is asserted, not just the outcome: a guard that cannot read the
    division column at all also returns None, and that would pass this test
    while being blind to the thing it exists to catch.
    """
    from data import build_canonical_dataset as mod

    _serve(monkeypatch, _EC_PAYLOAD)
    got = mod.download_season(26, "E0", str(tmp_path), use_cache=False)

    assert got is None
    assert "served ['EC']" in capsys.readouterr().out, \
        "rejected, but not as a division mismatch"


def test_download_accepts_the_requested_division(monkeypatch, tmp_path):
    """The guard must not cost us the seasons that are genuinely there.

    The payload is BOM-prefixed like the real one, so this also pins the
    encoding: read as plain utf-8 the column is named "ï»¿Div", the guard sees
    no division at all, and every published season is rejected. That was the
    state of this code until the live check caught it — all five unit tests
    green, all four real seasons refused.
    """
    from data import build_canonical_dataset as mod

    _serve(monkeypatch, _E0_PAYLOAD)
    got = mod.download_season(25, "E0", str(tmp_path), use_cache=False)

    assert got is not None
    assert len(got) == 1
    assert got.iloc[0]["HomeTeam"] == "Liverpool"


def test_rejected_download_is_not_left_in_the_raw_cache(monkeypatch, tmp_path):
    """A rejected payload must not persist and be trusted later.

    The raw is written before it is parsed, and the cache-read path does no
    validation, so without this a single bad download poisons every
    subsequent build from disk — no network needed to stay wrong.
    """
    from data import build_canonical_dataset as mod

    _serve(monkeypatch, _EC_PAYLOAD)
    assert mod.download_season(26, "E0", str(tmp_path), use_cache=False) is None

    assert not (tmp_path / "E0_2627.csv").exists(), \
        "rejected payload was cached and will be served silently next build"


def test_wrong_division_in_cache_is_treated_as_corruption(monkeypatch, tmp_path):
    """A bad raw already on disk must not be trusted, and must self-heal.

    Finished seasons read from cache without touching the network, so a raw
    poisoned before this guard existed would be believed forever. Falling
    through to a re-download matches how the function already handles an
    unparseable cache, and recovers the moment upstream publishes.
    """
    from data import build_canonical_dataset as mod

    (tmp_path / "E0_2627.csv").write_text(_EC_PAYLOAD, encoding="utf-8")
    _serve(monkeypatch, _E0_PAYLOAD)

    got = mod.download_season(26, "E0", str(tmp_path), use_cache=True)

    assert got is not None
    assert got.iloc[0]["HomeTeam"] == "Liverpool", \
        "served the poisoned cache instead of re-downloading"


def test_payload_without_a_division_column_is_rejected(monkeypatch, tmp_path):
    """Unverifiable provenance is not the same as verified provenance.

    Same principle as the Freshness Gate's UNKNOWN verdict: "could not
    determine" must not collapse into "fine". Every one of the 52 cached raws
    carries a single Div value, so this only fires if the source's shape
    changes — exactly when believing it would be worst.
    """
    from data import build_canonical_dataset as mod

    _serve(monkeypatch, "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"
                        "15/08/2026,Liverpool,Bournemouth,4,2\n")

    assert mod.download_season(26, "E0", str(tmp_path), use_cache=False) is None
    assert not (tmp_path / "E0_2627.csv").exists()


def test_byte_order_mark_in_cached_raw_is_read(monkeypatch, tmp_path):
    """Same for the cache: the raw is saved verbatim, BOM bytes included."""
    from data import build_canonical_dataset as mod

    (tmp_path / "E0_2526.csv").write_bytes(
        b"\xef\xbb\xbf" + _E0_PAYLOAD.encode("utf-8"))
    got = mod.download_season(25, "E0", str(tmp_path), use_cache=True)

    assert got is not None
    assert got.iloc[0]["HomeTeam"] == "Liverpool"


def test_stray_non_utf8_byte_does_not_lose_the_season(monkeypatch, tmp_path):
    """The 2004/05 raws carry a latin-1 nbsp (0xa0) that is invalid UTF-8.

    Decoding strictly turns one stray byte into a lost season: the download
    path returns None, and the cache path raises into the catch-all and
    re-downloads on every single build. Neither is worth a non-breaking space,
    so undecodable bytes are replaced rather than fatal.
    """
    from data import build_canonical_dataset as mod

    body = (b"\xef\xbb\xbfDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG\n"
            b"E0,15/08/2004,15:00,Arsenal,Everton\xa0,4,1\n")

    class _Raw:
        content = body
        text = body.decode("iso-8859-1")

        def raise_for_status(self):
            return None

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Raw())
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    got = mod.download_season(4, "E0", str(tmp_path), use_cache=False)
    assert got is not None, "one stray byte lost the whole season"
    assert len(got) == 1

    # And the same raw read back from cache, which is how every finished
    # season is loaded.
    (tmp_path / "E0_0405.csv").write_bytes(body)
    cached = mod.download_season(4, "E0", str(tmp_path), use_cache=True)
    assert cached is not None
    assert len(cached) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Column coverage on season addition
#
# `_map_columns` reads match stats with `df.get(...)`, so an upstream rename
# yields an all-NaN column and the build still succeeds. These tests pin the
# check that catches it before the rows reach training.
# ─────────────────────────────────────────────────────────────────────────────


def _seasons(spec: dict[int, dict[str, list]]) -> pd.DataFrame:
    """Build a frame of {season_idx: {column: values}}."""
    frames = []
    for idx, cols in spec.items():
        f = pd.DataFrame(cols)
        f["SeasonIndex"] = idx
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def test_column_that_arrives_empty_is_flagged():
    """The rename case: populated for years, all-NaN in the new season."""
    df = _seasons({
        24: {"Home_Shots": [12.0, 9.0, 14.0, 11.0]},
        25: {"Home_Shots": [10.0, 13.0, 8.0, 15.0]},
        26: {"Home_Shots": [None, None, None, None]},
    })

    flagged = coverage_regressions(df, season_idx=26)

    assert [c for c, _, _ in flagged] == ["Home_Shots"]


def test_legitimately_sparse_enrichment_is_not_flagged():
    """xG and injury columns only exist for seasons upstream covers.

    Flagging those would block every season addition, and the check would be
    turned off — which is worse than not having it.
    """
    df = _seasons({
        24: {"home_xg": [None, None, 1.2, None]},
        25: {"home_xg": [None, None, None, 0.9]},
        26: {"home_xg": [None, None, None, None]},
    })

    assert coverage_regressions(df, season_idx=26) == []


def test_a_complete_new_season_passes():
    df = _seasons({
        25: {"Home_Shots": [12.0, 9.0, 14.0, 11.0]},
        26: {"Home_Shots": [10.0, 13.0, 8.0, 15.0]},
    })

    assert coverage_regressions(df, season_idx=26) == []


def test_the_first_season_has_nothing_to_compare_against():
    df = _seasons({0: {"Home_Shots": [None, None]}})

    assert coverage_regressions(df, season_idx=0) == []


def test_only_prior_seasons_form_the_reference():
    """A later season must not vouch for an earlier one's missing column."""
    df = _seasons({
        25: {"Home_Shots": [None, None, None, None]},
        26: {"Home_Shots": [10.0, 13.0, 8.0, 15.0]},
    })

    assert coverage_regressions(df, season_idx=25) == []


def test_assert_names_every_regressed_column_and_both_rates():
    """With no bypass flag, the message is the only route to action."""
    df = _seasons({
        25: {"Home_Shots": [12.0, 9.0], "Home_Corners": [4.0, 6.0]},
        26: {"Home_Shots": [None, None], "Home_Corners": [None, None]},
    })

    with pytest.raises(ColumnCoverageError) as exc:
        assert_column_coverage(df, season_idx=26)

    assert "Home_Shots" in str(exc.value)
    assert "Home_Corners" in str(exc.value)
    assert "0%" in str(exc.value)


def test_assert_is_silent_when_the_season_is_well_formed():
    df = _seasons({
        25: {"Home_Shots": [12.0, 9.0]},
        26: {"Home_Shots": [10.0, 13.0]},
    })

    assert_column_coverage(df, season_idx=26)


def test_betfair_populated_columns_are_not_judged():
    """They are filled monthly by a separate job, not by the season CSV.

    Judging them would fail every rollover added mid-cycle, and a check that
    cries wolf at every rollover is a check that gets switched off.
    """
    df = _seasons({
        25: {"Odds_Over_1.5": [1.2, 1.3], "Home_Shots": [12.0, 9.0]},
        26: {"Odds_Over_1.5": [None, None], "Home_Shots": [10.0, 13.0]},
    })

    assert coverage_regressions(df, season_idx=26) == []


def test_partial_season_does_not_flag_rolling_window_features():
    """A season one round old has no 10-match window filled — and neither had
    any prior season at the same point.

    Judging a partial season against complete ones flagged every window
    feature: EFL 2026/27's opening 12 matches read 0% on Home_ScoringRate_10
    against 87% across three complete seasons, which blocked the rollover
    rebuild on the day upstream published. That is an artefact of comparing
    unlike slices, not the upstream rename this check exists to catch.

    The reference fill rate here is deliberately 87.5% — above the 80%
    `populated` threshold. Below it the column is never judged at all and the
    test would pass whatever the slicing does.
    """
    # Unfilled for the opening 2 matches, filled for the remaining 14: 87.5%,
    # matching the real Home_ScoringRate_10 rate that triggered the block.
    window = [None, None] + [1.5] * 14
    df = _seasons({
        24: {"Home_ScoringRate_10": window, "Home_Shots": [11.0] * 16},
        25: {"Home_ScoringRate_10": window, "Home_Shots": [12.0] * 16},
        26: {"Home_ScoringRate_10": [None, None], "Home_Shots": [10.0, 9.0]},
    })

    assert coverage_regressions(df, season_idx=26) == []


def test_partial_season_still_flags_a_genuinely_missing_column():
    """The slice must not blunt the check.

    Same two-match new season as above, but a column upstream populates from
    the first whistle arrives empty. That is the rename case and must survive.
    """
    df = _seasons({
        24: {"Home_ScoringRate_10": [None, None, 1.4, 1.6], "Home_Shots": [11.0] * 4},
        25: {"Home_ScoringRate_10": [None, None, 1.5, 1.3], "Home_Shots": [12.0] * 4},
        26: {"Home_ScoringRate_10": [None, None], "Home_Shots": [None, None]},
    })

    flagged = coverage_regressions(df, season_idx=26)

    assert [c for c, _, _ in flagged] == ["Home_Shots"]


def test_complete_new_season_compares_against_whole_reference_seasons():
    """Slicing to len(new) must be a no-op once the season is complete."""
    df = _seasons({
        24: {"Home_Shots": [11.0, 12.0, 13.0, 14.0]},
        25: {"Home_Shots": [10.0, 11.0, 12.0, 13.0]},
        26: {"Home_Shots": [None, None, None, None]},
    })

    flagged = coverage_regressions(df, season_idx=26)

    assert [c for c, _, _ in flagged] == ["Home_Shots"]


class TestTheGuardJudgesTheSeasonThatWasAskedFor:
    """Step 9 checked `SeasonIndex.max()`, not the season the config wants.

    `build()` step 1 drops a season whose download returns nothing
    (`if raw is not None and len(raw) > 0`). So an empty or failed download
    for the configured season produced a canonical ending at the season
    before, `newest` resolved to that, the guard compared it against seasons
    older still, passed, and the build printed "coverage matches prior
    seasons" over a canonical missing the entire current season.

    That is the same silent-success shape the guard exists to catch, one
    level up: it verified the shape of whatever arrived, never that the thing
    asked for arrived at all.
    """

    @staticmethod
    def _canonical(seasons):
        rows = []
        for idx, n in seasons:
            for i in range(n):
                rows.append({
                    "SeasonIndex": idx,
                    "Date": f"20{20 + idx}-01-{i % 28 + 1:02d}",
                    "Home_Team": f"T{i % 12:02d}",
                    "Away_Team": f"T{(i + 1) % 12:02d}",
                    "HS": 10, "AS": 9,
                })
        return pd.DataFrame(rows)

    def test_a_season_that_never_arrived_is_reported(self):
        from data.build_canonical_dataset import (
            MissingSeasonError, assert_season_present,
        )

        df = self._canonical([(23, 132), (24, 132), (25, 132)])

        with pytest.raises(MissingSeasonError) as excinfo:
            assert_season_present(df, 26)

        assert "26" in str(excinfo.value)

    def test_a_season_that_did_arrive_passes(self):
        from data.build_canonical_dataset import assert_season_present

        df = self._canonical([(24, 132), (25, 132), (26, 12)])

        assert_season_present(df, 26)  # one round is still arrival

    def test_the_guard_does_not_fall_back_to_an_older_season(self):
        """The vacuous pass: 25 vs 22-24 says nothing about 26."""
        from data.build_canonical_dataset import (
            MissingSeasonError, assert_season_present,
        )

        df = self._canonical([(23, 132), (24, 132), (25, 132)])
        assert int(df["SeasonIndex"].max()) == 25

        with pytest.raises(MissingSeasonError):
            assert_season_present(df, 26)


class TestColumnsTheBuilderNullsItself:
    """The guard flagged the builder's own deliberate nulling as a regression.

    `_add_promotion_flags` sets Home_/Away_Promoted and Home_/Away_Relegated
    to NaN for a season whose roster is not yet complete — "a part-loaded
    season cannot say who is new, and guessing would assert something false".
    The reference seasons have all four at 100%, so the new season reads 0%
    against 100%, below the floor, and the rebuild aborts.

    It fires between a season's opening fixture and the completion of round 1
    (EFL round 1 routinely opens on a Friday night), and for as long as a
    postponement leaves one side unplayed. These columns have a known owner
    inside the builder, which is the same reason `_LAGGING_COLUMNS` exists.
    """

    @staticmethod
    def _frame(flags_null):
        import numpy as np
        rows = []
        for idx, n in ((23, 132), (24, 132), (25, 132), (26, 12)):
            for i in range(n):
                null = flags_null and idx == 26
                rows.append({
                    "SeasonIndex": idx,
                    "Date": f"20{20 + idx}-01-{i % 28 + 1:02d}",
                    "Home_Team": f"T{i % 12:02d}",
                    "Away_Team": f"T{(i + 1) % 12:02d}",
                    "HS": 10, "AS": 9,
                    "Home_Promoted": np.nan if null else 0,
                    "Away_Promoted": np.nan if null else 0,
                    "Home_Relegated": np.nan if null else 0,
                    "Away_Relegated": np.nan if null else 0,
                })
        return pd.DataFrame(rows)

    def test_a_part_loaded_season_does_not_abort_the_rebuild(self):
        assert_column_coverage(self._frame(flags_null=True), 26)

    def test_a_real_column_loss_is_still_caught(self):
        """Guards the test above: the exclusion must be narrow."""
        import numpy as np

        df = self._frame(flags_null=True)
        df.loc[df["SeasonIndex"] == 26, "HS"] = np.nan

        with pytest.raises(ColumnCoverageError) as excinfo:
            assert_column_coverage(df, 26)

        assert "HS" in str(excinfo.value)
