"""Two leagues fetching at once must not contaminate each other.

`api.odds_api` held the sport key and the cache path as **module-level
globals**, and callers swapped them in place around a fetch, restoring in a
`finally`. Dash serves on Flask with `threaded=True`, so two `update_main`
callbacks — one per league tab — run concurrently against that shared state.

Live on 2026-09-01 the result was a complete swap: `odds_cache.json` held 24
Championship fixtures and `odds_cache_efl.json` held 20 Premier League ones.
Every Championship team mapping then failed, the EFL produced zero
recommendations for a week, and nothing raised.

Two interleavings each produce it, and both are covered here:

1. The PL branch of `scan.py` never set `CACHE_FILE` at all — it inherited
   whatever the global held — so a PL fetch starting mid-EFL-fetch wrote PL
   data into the EFL file.
2. A PL fetch's `finally` restored ``SPORT`` to the PL key while an EFL fetch
   was still in flight, so the EFL request hit the PL endpoint.

A sequential test passes against the broken code — three separate sequential
probes did during the diagnosis. The seam has to be the concurrent one.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

import api.odds_api as odds

PL_SPORT = "soccer_epl"
EFL_SPORT = "soccer_efl_champ"


@pytest.fixture()
def slow_api(monkeypatch):
    """Mock the one HTTP seam, holding long enough for threads to overlap.

    The event is named after whichever sport key the call actually used, so a
    contaminated fetch is visible in the returned data rather than inferred.
    """
    def _fetch_market(market_key, sport=None, **kwargs):
        used = sport if sport is not None else odds.SPORT
        time.sleep(0.05)
        return [{
            "id": f"evt_{used}", "sport_key": used,
            "commence_time": "2026-09-06T15:30:00Z",
            "home_team": f"HOME_{used}", "away_team": f"AWAY_{used}",
            "bookmakers": [{"key": "bet365", "title": "Bet365", "markets": [
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 2.5, "price": 1.9},
                    {"name": "Under", "point": 2.5, "price": 1.9}]}]}],
        }]
    monkeypatch.setattr(odds, "_fetch_market", _fetch_market)


def _sports_in(matches) -> set[str]:
    return {m["home_team"].replace("HOME_", "") for m in matches or []}


def test_concurrent_league_fetches_do_not_contaminate_each_other(
        slow_api, tmp_path):
    """The regression. Each league gets its own fixtures and its own file."""
    pl_cache = str(tmp_path / "odds_cache.json")
    efl_cache = str(tmp_path / "odds_cache_efl.json")
    results: dict[str, list] = {}

    def run(name, sport, cache_file, delay):
        time.sleep(delay)
        results[name] = odds.fetch_epl_odds(
            force_refresh=True, markets=("totals",),
            sport=sport, cache_file=cache_file)

    threads = [
        threading.Thread(target=run, args=("EFL", EFL_SPORT, efl_cache, 0.0)),
        threading.Thread(target=run, args=("PL", PL_SPORT, pl_cache, 0.02)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _sports_in(results["PL"]) == {PL_SPORT}, (
        f"the PL fetch returned {_sports_in(results['PL'])} — a concurrent "
        f"EFL fetch changed the sport key underneath it")
    assert _sports_in(results["EFL"]) == {EFL_SPORT}, (
        f"the EFL fetch returned {_sports_in(results['EFL'])} — a concurrent "
        f"PL fetch changed the sport key underneath it")

    with open(pl_cache) as fh:
        assert _sports_in(json.load(fh)["data"]) == {PL_SPORT}, (
            "the PL cache file received another league's fixtures")
    with open(efl_cache) as fh:
        assert _sports_in(json.load(fh)["data"]) == {EFL_SPORT}, (
            "the EFL cache file received another league's fixtures")


def test_an_explicit_fetch_does_not_disturb_the_module_defaults(
        slow_api, tmp_path):
    """Passing sport and cache_file must not mutate shared state at all.

    The defect was not that the swap was wrong, but that there was a swap.
    """
    before_sport, before_cache = odds.SPORT, odds.CACHE_FILE

    odds.fetch_epl_odds(
        force_refresh=True, markets=("totals",),
        sport=EFL_SPORT, cache_file=str(tmp_path / "odds_cache_efl.json"))

    assert odds.SPORT == before_sport, (
        "fetching for one league changed the module-wide sport key")
    assert odds.CACHE_FILE == before_cache, (
        "fetching for one league changed the module-wide cache path")
