"""Odds columns in the Canonical Datasets: exchange price preferred, soft
book as fallback, provenance recorded (ADR 0003).

The O/U 2.5 backtest priced off Bet365 for both the fair reference and the
execution price, gross — neither the live base nor the venue the system
actually trades at. ADR 0003 calls that out and says "'re-run with
commission' is really 're-source the backtest onto Betfair'". These columns
are that re-sourcing.

Three things they must get right, each of which silently ruins a backtest:

* **Never the last traded price.** ``over_ltp`` is contaminated by in-play
  trading — across 66k settled O/U 2.5 markets its median is 1.53 when the
  over won and 15.00 when it lost, with 18% above 100. A pre-match price
  cannot know the result. Only ``*_ltp_first`` is safe.
* **Never a women's, youth or reserve fixture.** Betfair carries those under
  near-identical names on the same day.
* **Never a silent name miss.** Betfair spells five EFL clubs differently
  from football-data.co.uk; without a bridge the join drops ~1,000 fixtures
  and simply reports thinner coverage, which looks like a data gap.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.build_canonical_dataset import _add_odds


def _betfair(rows) -> pd.DataFrame:
    """Build a Betfair goal-O/U frame from (event, market, first, ltp) rows."""
    out = []
    for event, mkt, over_first, under_first, over_ltp, under_ltp in rows:
        out.append({
            "event_name": event,
            "market_type": mkt,
            "goal_line": 2.5 if mkt.endswith("25") else 1.5,
            "market_time": "2024-08-17T14:00:00.000Z",
            "over_ltp": over_ltp, "under_ltp": under_ltp,
            "over_ltp_first": over_first, "under_ltp_first": under_first,
            "winner": "over", "country_code": "GB",
        })
    return pd.DataFrame(out)


def _canonical(home, away, b365_over=None, b365_under=None) -> pd.DataFrame:
    return pd.DataFrame([{
        "Date": pd.Timestamp("2024-08-17"),
        "SeasonIndex": 24,
        "Home_Team": home, "Away_Team": away,
        "Home_Goals": 1, "Away_Goals": 1,
        "B365Greater2.5": b365_over, "B365LessThan2.5": b365_under,
    }])


def test_betfair_price_is_preferred_over_bet365(tmp_path):
    bf = _betfair([("Arsenal v Chelsea", "OVER_UNDER_25", 1.90, 2.00, 5.0, 1.1)])
    path = tmp_path / "bf.csv"
    bf.to_csv(path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC", 1.75, 2.10),
                    "PL", str(path))
    row = out.iloc[0]
    assert row["Odds_Over_2.5"] == pytest.approx(1.90)
    assert row["Odds_Under_2.5"] == pytest.approx(2.00)
    assert row["Odds_Source_2.5"] == "betfair"


def test_the_in_play_contaminated_price_is_never_used(tmp_path):
    """over_ltp is the LAST traded price. Using it leaks the result."""
    bf = _betfair([("Arsenal v Chelsea", "OVER_UNDER_25", 1.90, 2.00, 1.01, 500.0)])
    path = tmp_path / "bf.csv"
    bf.to_csv(path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC"), "PL", str(path))
    assert out.iloc[0]["Odds_Over_2.5"] == pytest.approx(1.90)
    assert out.iloc[0]["Odds_Over_2.5"] != pytest.approx(1.01)


def test_bet365_is_the_fallback_when_betfair_is_absent(tmp_path):
    """Seasons 0-15 predate Betfair entirely — the soft book carries them."""
    path = tmp_path / "bf.csv"
    _betfair([("Someone v Else", "OVER_UNDER_25", 1.5, 2.5, 1.5, 2.5)]).to_csv(
        path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC", 1.75, 2.10),
                    "PL", str(path))
    row = out.iloc[0]
    assert row["Odds_Over_2.5"] == pytest.approx(1.75)
    assert row["Odds_Under_2.5"] == pytest.approx(2.10)
    assert row["Odds_Source_2.5"] == "b365"


def test_no_odds_anywhere_is_null_with_no_source(tmp_path):
    path = tmp_path / "bf.csv"
    _betfair([("Someone v Else", "OVER_UNDER_25", 1.5, 2.5, 1.5, 2.5)]).to_csv(
        path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC"), "PL", str(path))
    row = out.iloc[0]
    assert pd.isna(row["Odds_Over_2.5"])
    assert pd.isna(row["Odds_Source_2.5"])


def test_ou15_has_no_soft_book_fallback(tmp_path):
    """football-data.co.uk serves no 1.5 line, so 1.5 is Betfair or nothing —
    it must never silently inherit the 2.5 price."""
    bf = _betfair([("Arsenal v Chelsea", "OVER_UNDER_25", 1.90, 2.00, 1.9, 2.0)])
    path = tmp_path / "bf.csv"
    bf.to_csv(path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC", 1.75, 2.10),
                    "PL", str(path))
    row = out.iloc[0]
    assert pd.isna(row["Odds_Over_1.5"])
    assert pd.isna(row["Odds_Source_1.5"])


def test_both_lines_are_carried_independently(tmp_path):
    bf = _betfair([
        ("Arsenal v Chelsea", "OVER_UNDER_25", 1.90, 2.00, 1.9, 2.0),
        ("Arsenal v Chelsea", "OVER_UNDER_15", 1.30, 3.60, 1.3, 3.6),
    ])
    path = tmp_path / "bf.csv"
    bf.to_csv(path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC"), "PL", str(path))
    row = out.iloc[0]
    assert row["Odds_Over_2.5"] == pytest.approx(1.90)
    assert row["Odds_Over_1.5"] == pytest.approx(1.30)
    assert row["Odds_Source_1.5"] == "betfair"


@pytest.mark.parametrize("betfair_name,efl_name", [
    ("Peterborough", "Peterboro"),
    ("Nottm Forest", "Nott'm Forest"),
    ("Oxford United", "Oxford"),
    ("Sheff Utd", "Sheffield United"),
    ("Sheff Wed", "Sheffield Weds"),
])
def test_the_five_efl_clubs_betfair_spells_differently(
        tmp_path, betfair_name, efl_name):
    """Each of these sat at exactly 0% match before the bridge existed."""
    bf = _betfair([(f"{betfair_name} v Millwall", "OVER_UNDER_25",
                    1.90, 2.00, 1.9, 2.0)])
    path = tmp_path / "bf.csv"
    bf.to_csv(path, index=False)

    out = _add_odds(_canonical(efl_name, "Millwall"), "EFL", str(path))
    assert out.iloc[0]["Odds_Over_2.5"] == pytest.approx(1.90)
    assert out.iloc[0]["Odds_Source_2.5"] == "betfair"


@pytest.mark.parametrize("event", [
    "Arsenal (W) v Chelsea (W)",
    "Arsenal U21 v Chelsea U21",
    "Arsenal U23 v Chelsea U23",
    "Arsenal FC (Res) v Chelsea FC (Res)",
])
def test_womens_youth_and_reserve_fixtures_never_merge(tmp_path, event):
    """Same clubs, same day, a completely different match."""
    path = tmp_path / "bf.csv"
    _betfair([(event, "OVER_UNDER_25", 1.10, 8.00, 1.1, 8.0)]).to_csv(
        path, index=False)

    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC"), "PL", str(path))
    assert pd.isna(out.iloc[0]["Odds_Over_2.5"])


def test_a_missing_betfair_file_leaves_bet365_prices_intact(tmp_path):
    """The builder must not fail when the monthly download has not run."""
    out = _add_odds(_canonical("Arsenal FC", "Chelsea FC", 1.75, 2.10),
                    "PL", str(tmp_path / "absent.csv"))
    assert out.iloc[0]["Odds_Over_2.5"] == pytest.approx(1.75)
    assert out.iloc[0]["Odds_Source_2.5"] == "b365"
