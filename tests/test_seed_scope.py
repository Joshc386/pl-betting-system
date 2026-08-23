"""What an arriving side's row is built from when it has no history.

A side with no rows in the division has nothing of its own to score, so the
row is assembled from a template plus a seed. The template is the *opposing*
side's last fixture, which means every column the seed does not cover belongs
to whichever club last played that opponent.

Only 11 of the PL's 110 ``Home_`` model features were seeded, so 99 carried a
third club's values. `Home_Elo` swung 1900 vs 1010 on nothing but a bystander's
rating, and Elo is one of the strongest inputs the ensemble has. Hull, Ipswich
and Coventry all had zero rows in season 26 when this was found.

The EFL path was already immune for prefixed columns, because it sweeps every
``{side}_`` column rather than an explicit list — the looser code was the safer
code. These tests hold both leagues to the EFL behaviour.

Derived features with no side prefix (``Elo_Diff``, ``Combined_Over25``,
``Poisson_Consensus`` ...) are computed from both clubs and cannot be seeded
from one side's cohort. Those that are a pure function of already-seeded
inputs are recomputed; the rest remain an approximation, and
``test_a_two_sided_derivation_is_not_silently_claimed_correct`` records exactly
which ones so the limitation is visible rather than assumed away.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Two clubs with very different ratings. Which of them last hosted the
# arrival's opponent decides the template - and must decide nothing else.
_WEAK_HOST = "T01"     # Elo 1000
_STRONG_HOST = "T03"   # Elo 1900

_ELOS = {"T01": 1000.0, "T02": 1100.0, "T03": 1900.0,
         "T04": 1050.0, "T05": 1060.0, "T06": 1070.0,
         "T07": 1080.0, "T08": 1090.0}


def _frame(last_host: str) -> pd.DataFrame:
    """Two seasons of a division, identical but for who hosted T02 last.

    Every club and every value is the same in both variants, so the cohort
    the seed is drawn from is identical. The only thing that moves is which
    fixture ends up as the template for an arrival with no history of its
    own. Nothing about the arrival's row may follow it.
    """
    elos = _ELOS
    sides, rows, k = list(elos), [], 0
    for season in (24, 25):
        for home in sides:
            for away in sides:
                if home == away:
                    continue
                k += 1
                rows.append({
                    "SeasonIndex": season,
                    "Date": f"20{20 + season}-01-{k:03d}",
                    "Home_Team": home, "Away_Team": away,
                    "Home_LeaguePosition": sides.index(home) + 1,
                    "Away_LeaguePosition": sides.index(away) + 1,
                    "Home_Past5Goals": 1.0, "Away_Past5Goals": 1.0,
                    "Home_Past5Conceded": 1.0, "Away_Past5Conceded": 1.0,
                    "Home_Elo": elos[home], "Away_Elo": elos[away],
                    "Home_Over25_5": 0.5, "Away_Over25_5": 0.5,
                    "Elo_Diff": elos[home] - elos[away],
                    "LeaguePosition_Diff": (sides.index(home)
                                            - sides.index(away)),
                    "Home_Promoted": 0, "Away_Promoted": 0,
                })
    # The deciding fixture, latest in the frame.
    rows.append({
        "SeasonIndex": 25, "Date": "2045-12-31",
        "Home_Team": last_host, "Away_Team": "T02",
        "Home_LeaguePosition": sides.index(last_host) + 1,
        "Away_LeaguePosition": sides.index("T02") + 1,
        "Home_Past5Goals": 1.0, "Away_Past5Goals": 1.0,
        "Home_Past5Conceded": 1.0, "Away_Past5Conceded": 1.0,
        "Home_Elo": elos[last_host], "Away_Elo": elos["T02"],
        "Home_Over25_5": 0.5, "Away_Over25_5": 0.5,
        "Elo_Diff": elos[last_host] - elos["T02"],
        "LeaguePosition_Diff": (sides.index(last_host)
                                - sides.index("T02")),
        "Home_Promoted": 0, "Away_Promoted": 0,
    })
    return pd.DataFrame(rows)


def _pl_row(df: pd.DataFrame) -> pd.Series:
    from predict import LivePredictor

    predictor = LivePredictor.__new__(LivePredictor)
    predictor.verbose = False
    predictor._full_df = df
    return predictor._fixture_feature_row(
        "NEWCO", "T02", df, season=26, arrivals={"NEWCO"})


def _efl_row(df: pd.DataFrame) -> pd.Series:
    from championship_predict import ChampionshipPredictor
    from division_movement import PROMOTED, SeedParams

    predictor = ChampionshipPredictor.__new__(ChampionshipPredictor)
    predictor.verbose = False
    predictor._full_df = df
    predictor._pl_df = pd.DataFrame(
        columns=["SeasonIndex", "Date", "Home_Team", "Away_Team",
                 "Home_LeaguePosition", "Away_LeaguePosition"])
    predictor._seed_params_cache = SeedParams(priors={}, n_events=0)
    return predictor._fixture_feature_row(
        "NEWCO", "T02", df, season=26, arrivals={"NEWCO": PROMOTED})


_BUILDERS = [("PL", _pl_row), ("EFL", _efl_row)]


class TestNoBystandersValueReachesAnArrival:
    """Change which fixture supplies the template; the row must not move."""

    @pytest.mark.parametrize("league,build", _BUILDERS)
    @pytest.mark.parametrize("feature", [
        "Home_Elo", "Home_Over25_5", "Home_Past5Goals",
    ])
    def test_a_prefixed_feature_is_independent_of_a_bystander(
            self, league, build, feature):
        strong, weak = build(_frame(_STRONG_HOST)), build(_frame(_WEAK_HOST))

        assert strong is not None and weak is not None
        assert strong[feature] == pytest.approx(weak[feature]), (
            f"{league}: {feature} moved when only the template fixture "
            f"changed - the arriving side inherited a bystander's value "
            f"because the seed does not cover this column")

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_the_seeded_row_is_still_the_arrival(self, league, build):
        row = build(_frame(_STRONG_HOST))

        assert row["Home_Team"] == "NEWCO"
        assert row["Home_Promoted"] == 1


class TestDerivationsRecomputedFromSeededInputs:
    """A difference of two seeded numbers can be made correct; it must be."""

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_elo_diff_follows_the_seeded_elos(self, league, build):
        row = build(_frame(_STRONG_HOST))

        assert row["Elo_Diff"] == pytest.approx(
            row["Home_Elo"] - row["Away_Elo"]), (
            f"{league}: Elo_Diff still describes two clubs that are not "
            f"playing in this fixture")

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_elo_diff_is_independent_of_a_bystander(self, league, build):
        strong, weak = build(_frame(_STRONG_HOST)), build(_frame(_WEAK_HOST))

        assert strong["Elo_Diff"] == pytest.approx(weak["Elo_Diff"])

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_league_position_diff_follows_the_seeded_positions(
            self, league, build):
        row = build(_frame(_STRONG_HOST))

        assert row["LeaguePosition_Diff"] == pytest.approx(
            row["Home_LeaguePosition"] - row["Away_LeaguePosition"])


class TestTheRemainingLimitationIsRecorded:
    """Two-sided derivations that are not a function of seeded inputs.

    These cannot be repaired from one side's cohort without reimplementing
    pipeline formulas in the predictor - a second definition of each, which is
    what ADR 0011 exists to prevent. They stay approximate. This test names
    them so the gap is visible in the suite rather than assumed away, and
    fails if the list grows without a decision.
    """

    KNOWN_APPROXIMATE = {
        "Attack_Power", "AvailableXG_Diff", "BTTS_Attack_Power",
        "Blanking_Risk", "CS_Risk", "Combined_BTTS", "Combined_FTS",
        "Combined_Over25", "Combined_TotalCorners", "Corner_Dominance",
        "Corner_Poisson_Over105", "DefenceMissing_Diff",
        "Expected_TG_Consensus", "FPL_HomeDominance", "FPL_Openness",
        "Poisson_BTTS", "Poisson_BTTS_Consensus", "Poisson_BTTS_xG",
        "Poisson_Consensus", "Poisson_Shots",
    }

    def test_the_approximate_set_has_not_grown(self):
        import config

        union = []
        for name in dir(config):
            value = getattr(config, name)
            if ("FEATURE" in name.upper()
                    and isinstance(value, (list, tuple))):
                for feat in value:
                    if isinstance(feat, str) and feat not in union:
                        union.append(feat)

        # Fixture-level facts are correct for any pair of sides, and the two
        # repaired differences are asserted above.
        fixture_level = {
            "H2HAvgGoals", "H2H_AvgGoals_5", "Historical Derby", "Local Derby",
            "Match_Precipitation", "Match_Temperature", "Match_WindSpeed",
            "Season_Progress",
        }
        repaired = {"Elo_Diff", "LeaguePosition_Diff"}
        derived = {f for f in union
                   if not f.startswith(("Home_", "Away_"))}

        unaccounted = derived - fixture_level - repaired - self.KNOWN_APPROXIMATE
        assert not unaccounted, (
            f"new two-sided derived features are unaccounted for: "
            f"{sorted(unaccounted)}. Decide whether each is fixture-level, "
            f"repairable from seeded inputs, or approximate.")


class TestTheWiderFillDoesNotWidenTheBlend:
    """Fill and blend are different sets, and must stay different.

    Widening the fill to every numeric feature fixed the bystander leak. If
    the *blend* widened with it, an arriving side's non-rolling features would
    be part-cohort for five matches while training left them entirely actual —
    the same train/serve divergence, arriving from the opposite direction.
    """

    @staticmethod
    def _with_history(matches: int) -> pd.DataFrame:
        """A frame where the arrival has played, with its own distinct values."""
        df = _frame(_WEAK_HOST)
        extra = []
        for n in range(matches):
            extra.append({
                "SeasonIndex": 26, "Date": f"2046-02-{n + 1:02d}",
                "Home_Team": "NEWCO", "Away_Team": "T04",
                "Home_LeaguePosition": 20, "Away_LeaguePosition": 4,
                "Home_Past5Goals": 9.0, "Away_Past5Goals": 1.0,
                "Home_Past5Conceded": 9.0, "Away_Past5Conceded": 1.0,
                "Home_Elo": 777.0, "Away_Elo": 1050.0,
                "Home_Over25_5": 0.9, "Away_Over25_5": 0.5,
                "Elo_Diff": 777.0 - 1050.0, "LeaguePosition_Diff": 16,
                "Home_Promoted": 1, "Away_Promoted": 0,
            })
        return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_a_non_blended_feature_is_the_sides_own_value(self, league, build):
        """Home_Elo is filled when there is nothing, never blended when there is."""
        row = build(self._with_history(matches=2))

        assert row["Home_Elo"] == pytest.approx(777.0), (
            f"{league}: Home_Elo was blended with the cohort. Training never "
            f"blends it, so the live row would mean something the trained "
            f"rows do not")

    @pytest.mark.parametrize("league,build", _BUILDERS)
    def test_a_blended_feature_is_still_blended(self, league, build):
        """Guards the test above: absence of blending must be selective."""
        from division_movement import seed_weight

        row = build(self._with_history(matches=2))
        weight = seed_weight(2)

        assert weight > 0
        assert row["Home_Past5Goals"] != pytest.approx(9.0), (
            f"{league}: a feature training does blend was left entirely "
            f"actual")
