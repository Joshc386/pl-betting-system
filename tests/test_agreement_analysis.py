"""Tests for the Historical Agreement analysis (ADR 0010).

Covers realised edge, the fixture-clustered bootstrap, and the OOF gate
replay that answers "does model agreement predict a better bet?".
"""
from pathlib import Path

import pandas as pd
import pytest

from edge_analytics import (
    agreement_bins,
    clustered_bootstrap_ci,
    load_oof_cell,
    realised_edge,
    replay_oof_gate,
)


def _oof_frame(rows: list[dict], n_models: int = 4) -> pd.DataFrame:
    """Build a frame shaped like an OOF cache from per-fixture model probs.

    ``rows`` entries give ``probs`` (raw P(side_a) per base model),
    ``odds_a``, ``odds_b`` and ``outcome``. A 3-model (EFL) cell is
    produced by passing n_models=3, which leaves ``lr_prob`` null.
    """
    out = []
    for i, r in enumerate(rows):
        probs = list(r["probs"])
        rec = {
            "season": 20, "date": f"2020-01-{i+1:02d}",
            "home_team": f"H{i}", "away_team": f"A{i}",
            "base_rate": 0.5,
            "odds_a": r["odds_a"], "odds_b": r["odds_b"],
            "bookie_a": "Bet365", "bookie_b": "Bet365",
            "side_a_label": "over", "side_b_label": "under",
            "outcome": r["outcome"],
        }
        for name, p in zip(("xgb", "lgb", "dc", "lr"), probs + [None]):
            rec[f"{name}_prob"] = p
            rec[f"{name}_shift"] = 0.0
        if n_models == 3:
            rec["lr_prob"] = None
        out.append(rec)
    return pd.DataFrame(out)


class TestRealisedEdge:
    """mean(won) - mean(fair_prob) over a set of bets."""

    def test_is_outcome_rate_minus_mean_fair_price(self) -> None:
        # 2 of 3 won against a market pricing every bet at 50%
        assert realised_edge([1, 1, 0], [0.5, 0.5, 0.5]) == pytest.approx(
            2 / 3 - 0.5)

    def test_is_zero_when_bets_win_at_exactly_their_fair_rate(self) -> None:
        # The property the metric exists for: a market that wins 75% of the
        # time and is priced at 0.75 shows no edge, despite a 75% hit rate.
        won = [1] * 75 + [0] * 25
        assert realised_edge(won, [0.75] * 100) == pytest.approx(0.0)


class TestClusteredBootstrapCI:
    """Resamples fixtures, not rows — bets on one match are correlated."""

    def test_is_deterministic_and_brackets_the_point_estimate(self) -> None:
        won = [1, 0, 1, 0, 1, 1, 0, 1] * 10
        fair = [0.5] * 80
        fixtures = [f"f{i}" for i in range(80)]

        first = clustered_bootstrap_ci(won, fair, fixtures, seed=7)
        again = clustered_bootstrap_ci(won, fair, fixtures, seed=7)

        assert first == again
        lo, hi = first
        assert lo <= realised_edge(won, fair) <= hi

    def test_correlated_bets_widen_the_interval(self) -> None:
        # Same 80 observations. Treated as 80 independent fixtures, then as
        # 8 fixtures of 10 perfectly-correlated bets each. The second case
        # carries far less information and the CI must say so.
        won = ([1] * 10 + [0] * 10) * 4
        fair = [0.5] * 80
        independent = [f"f{i}" for i in range(80)]
        clustered = [f"f{i // 10}" for i in range(80)]

        wide = clustered_bootstrap_ci(won, fair, clustered, seed=3)
        narrow = clustered_bootstrap_ci(won, fair, independent, seed=3)

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestReplayOOFGate:
    """Replays every fixture x side through the live gate arithmetic."""

    def test_agreement_and_edge_are_antisymmetric_across_sides(self) -> None:
        # The property that forces the pre-gate view to keep one side per
        # fixture: each model is either above fair_a or its complement is
        # above fair_b, so the two sides' counts always sum to n_models and
        # their edges are exact negatives.
        df = _oof_frame([
            {"probs": [0.7, 0.65, 0.6, 0.55], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 1},
            {"probs": [0.3, 0.35, 0.45, 0.2], "odds_a": 1.8, "odds_b": 2.2,
             "outcome": 0},
        ])

        out = replay_oof_gate(df)

        for fixture, pair in out.groupby("fixture"):
            assert len(pair) == 2
            assert pair["n_agree"].sum() == pair["n_models"].iloc[0]
            a = pair[pair.side_col == "a"].iloc[0]
            b = pair[pair.side_col == "b"].iloc[0]
            assert a["edge"] == pytest.approx(-b["edge"])

    def test_n_models_is_read_from_the_data_not_assumed(self) -> None:
        # EFL cells carry no lr_prob — a 3-model ensemble. Hardcoding 4
        # would misreport every EFL agreement count.
        efl = replay_oof_gate(_oof_frame(
            [{"probs": [0.7, 0.65, 0.6], "odds_a": 2.0, "odds_b": 2.0,
              "outcome": 1}], n_models=3))
        assert set(efl["n_models"]) == {3}

    def test_at_most_one_side_of_a_fixture_can_pass_the_gate(self) -> None:
        # Falls out of edge antisymmetry: min_edge is positive, and the two
        # sides' edges sum to zero, so they cannot both clear it. This is
        # why the gated view needs no de-duplication.
        df = _oof_frame([
            {"probs": [0.8, 0.75, 0.7, 0.72], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 1},
            {"probs": [0.2, 0.25, 0.3, 0.28], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 0},
        ])
        out = replay_oof_gate(df)
        assert out.groupby("fixture")["passes"].sum().max() <= 1


class TestAgreementBins:
    """Bins by n_agree. Pre-gate keeps one side per fixture."""

    def _replayed(self):
        return replay_oof_gate(_oof_frame([
            # unanimous for over
            {"probs": [0.8, 0.75, 0.7, 0.72], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 1},
            # unanimous against over
            {"probs": [0.2, 0.25, 0.3, 0.28], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 0},
            # split
            {"probs": [0.6, 0.55, 0.4, 0.45], "odds_a": 2.0, "odds_b": 2.0,
             "outcome": 1},
        ]))

    def test_pre_gate_emits_exactly_one_row_per_fixture(self) -> None:
        # Counting both sides would double every fixture and force bins 0
        # and N to be mirror images with equal-and-opposite realised edges.
        bins = agreement_bins(self._replayed(), gated=False)
        assert bins["n_rows"].sum() == 3
        assert bins["n_fixtures"].sum() == 3

    def test_pre_gate_shows_agreement_levels_the_gate_rejects(self) -> None:
        bins = agreement_bins(self._replayed(), gated=False)
        assert 0 in set(bins["n_agree"])

    def test_gated_view_cannot_contain_sub_threshold_agreement(self) -> None:
        bins = agreement_bins(self._replayed(), gated=True)
        assert bins.empty or bins["n_agree"].min() >= 2


class TestLoadOOFCell:
    """The caches are gitignored build artefacts and may not exist."""

    def test_returns_none_when_the_cache_has_not_been_generated(
            self, tmp_path) -> None:
        # Must be distinguishable from "generated but empty" so the
        # dashboard can say so rather than rendering a silent blank.
        assert load_oof_cell("PL", "ou25", cache_dir=tmp_path) is None


class TestKnownOutputRegression:
    """Fixed input -> fixed output, so a refactor cannot silently move it.

    Runs against a committed snapshot of the replayed PL O/U 2.5 rows
    (`scripts/build_agreement_fixture.py`), because the OOF caches
    themselves are gitignored build artefacts.
    """

    @pytest.fixture
    def replayed(self) -> pd.DataFrame:
        path = (Path(__file__).parent / "fixtures"
                / "agreement_pl_ou25_replayed.csv")
        return pd.read_csv(path)

    def test_snapshot_is_one_row_per_fixture(self, replayed) -> None:
        assert len(replayed) == 2280
        assert replayed["fixture"].nunique() == 2280

    def test_unanimous_support_for_over_beats_the_market(
            self, replayed) -> None:
        bins = agreement_bins(replayed, gated=False).set_index("n_agree")
        assert bins.loc[4, "n_rows"] == 334
        assert bins.loc[4, "realised_edge"] == pytest.approx(0.025580,
                                                             abs=1e-6)

    def test_unanimous_opposition_to_over_loses_more_than_support_gains(
            self, replayed) -> None:
        # Visible only because the pre-gate view keeps one side per
        # fixture. Counting both sides forces bin 0 to be bin 4's exact
        # negative, which would conceal that the ensemble identifies bad
        # Over bets more sharply than good ones.
        bins = agreement_bins(replayed, gated=False).set_index("n_agree")
        assert bins.loc[0, "n_rows"] == 392
        assert bins.loc[0, "realised_edge"] == pytest.approx(-0.058487,
                                                             abs=1e-6)
        assert abs(bins.loc[0, "realised_edge"]) > bins.loc[4, "realised_edge"]


class TestHistoricalAgreementSection:
    """The dashboard section must never render as a silent blank."""

    def test_says_so_when_the_caches_have_not_been_generated(
            self, tmp_path) -> None:
        from dashboard import _make_historical_agreement

        div = _make_historical_agreement("PL", cache_dir=tmp_path)

        assert "not yet generated" in str(div).lower()
