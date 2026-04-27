"""Walk-forward OOF prediction cache generator for ROI validation (Phase 3).

For each test season in the walk-forward window, trains base models on all
prior seasons, produces per-fixture calibrated predictions + odds + outcome,
and writes a Parquet cache file. The validator (`scripts/roi_validate.py`)
reads these caches and applies `decide_bet` with toggle variations — without
ever needing to retrain.

**Crucial property**: for each test season N, the model was trained only on
data from seasons < N. No look-ahead leakage.

Pass 1 scope: **PL O/U 2.5 only**. Other markets land in Passes 2-5.

Output: `reports/roi_validate/oof_cache/<league>_<market>.parquet`

Parquet schema (every row = one fixture):
  season             int
  date               str (YYYY-MM-DD)
  home_team          str
  away_team          str
  xgb_prob           float   (raw P(side_a) before calibration)
  lgb_prob           float
  dc_prob            float
  lr_prob            float   (NaN for 3-model ensembles)
  xgb_shift          float   (per-season logit shift)
  lgb_shift          float
  dc_shift           float
  lr_shift           float
  base_rate          float   (per-season target mean)
  odds_a             float   (decimal odds for side_a: over / yes)
  odds_b             float   (decimal odds for side_b: under / no)
  bookie_a           str
  bookie_b           str
  side_a_label       str     ("over" | "yes")
  side_b_label       str     ("under" | "no")
  outcome            int     (1 if side_a wins, 0 if side_b)

Run:
  python scripts/generate_oof_cache.py --league PL --market ou25 \
      --seasons 19-24
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Register unpicklable classes in __main__ for joblib.load compatibility
# (DixonColesPredictor etc. were pickled under __main__ the first time)
from model import DixonColesPredictor  # noqa: F401

OUT_DIR = PROJECT_ROOT / "reports" / "roi_validate" / "oof_cache"


def generate_pl_ou25(seasons: range) -> pd.DataFrame:
    """Generate OOF cache rows for PL O/U 2.5.

    Calls existing `backtest.precompute_season()` per test season.
    """
    from pipeline import run_pipeline
    from backtest import precompute_season
    from model import tune_dc_params

    print(f"[PL ou25] Loading pipeline ...")
    t0 = time.time()
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])
    print(f"  ...loaded {len(full_df):,} rows in {time.time()-t0:.0f}s")

    print("[PL ou25] Tuning Dixon-Coles hyperparameters ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["B365Greater2.5"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue

        print(f"  S{season}: training on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        cached = precompute_season(train_df, test_df, features, dc_kwargs)
        print(f"    ...{time.time()-t0:.0f}s")

        # Each match_data entry references an index into the prediction arrays
        for md in cached["match_data"]:
            pi = md["pred_idx"]
            rows.append({
                "season": int(md["season"]),
                "date": str(md["date"])[:10],
                "home_team": md["home"],
                "away_team": md["away"],
                "xgb_prob": float(cached["xgb_raw"][pi]),
                "lgb_prob": float(cached["lgb_raw"][pi]),
                "dc_prob":  float(cached["dc_raw"][pi]),
                "lr_prob":  float(cached["lr_raw"][pi]),
                "xgb_shift": float(cached["xgb_shift"]),
                "lgb_shift": float(cached["lgb_shift"]),
                "dc_shift":  float(cached["dc_shift"]),
                "lr_shift":  float(cached["lr_shift"]),
                "base_rate": float(cached["base_rate"]),
                "odds_a":   float(md["odds_over"]),
                "odds_b":   float(md["odds_under"]),
                "bookie_a": "Bet365",
                "bookie_b": "Bet365",
                "side_a_label": "over",
                "side_b_label": "under",
                "outcome":  int(md["actual"]),
            })

    df = pd.DataFrame(rows)
    return df


def generate_pl_btts(seasons: range, odds_source: str = "footiqo") -> pd.DataFrame:
    """Generate OOF cache rows for PL BTTS.

    Args:
        seasons: range of test seasons to include.
        odds_source: "footiqo" (default, uses BTTSY/BTTSN from the footiqo
            CSVs via btts_data.load_btts_odds()) or "betfair" (uses
            yes_ltp/no_ltp from data/betfair_pl_btts.csv — Phase 2.4
            Betfair cross-check source).
    """
    from pipeline import run_pipeline
    from btts_backtest import precompute_btts_season
    from btts_data import load_btts_odds, merge_btts_odds
    from model import tune_dc_params

    print(f"[PL btts / {odds_source}] Loading pipeline ...")
    t0 = time.time()
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])
    print(f"  ...loaded {len(full_df):,} rows in {time.time()-t0:.0f}s")

    # Merge the chosen odds source onto the pipeline.
    if odds_source == "footiqo":
        print("[PL btts] Merging footiqo BTTS odds ...")
        full_df = merge_btts_odds(full_df, load_btts_odds())
    elif odds_source == "betfair":
        print("[PL btts] Merging Betfair BTTS odds (data/betfair_pl_btts.csv)"
              " ...")
        bf = pd.read_csv(PROJECT_ROOT / "data" / "betfair_pl_btts.csv")
        bf["_date"] = pd.to_datetime(bf["Date"]).dt.date
        bf = bf[["_date", "Home_Team", "Away_Team", "yes_ltp", "no_ltp"]]
        bf = bf.rename(columns={"yes_ltp": "BTTSY", "no_ltp": "BTTSN"})
        full_df["_date"] = pd.to_datetime(full_df["Date"]).dt.date
        full_df = full_df.merge(bf, on=["_date", "Home_Team", "Away_Team"],
                                how="left")
        full_df = full_df.drop(columns=["_date"])
    else:
        raise ValueError(f"Unknown odds_source={odds_source!r}")

    print("[PL btts] Tuning Dixon-Coles hyperparameters ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    # btts_backtest.py tunes DC on Over_2_5 target (same as O/U 2.5)
    # — the resulting half_life/rho are used for both markets.
    dc_kwargs = tune_dc_params(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["BTTSY"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue
        print(f"  S{season}: training on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        cached = precompute_btts_season(train_df, test_df, features, dc_kwargs)
        print(f"    ...{time.time()-t0:.0f}s")
        for md in cached["match_data"]:
            pi = md["pred_idx"]
            rows.append({
                "season": int(md["season"]),
                "date": str(md["date"])[:10],
                "home_team": md["home"],
                "away_team": md["away"],
                "xgb_prob": float(cached["xgb_raw"][pi]),
                "lgb_prob": float(cached["lgb_raw"][pi]),
                "dc_prob":  float(cached["dc_raw"][pi]),
                "lr_prob":  float(cached["lr_raw"][pi]),
                "xgb_shift": float(cached["xgb_shift"]),
                "lgb_shift": float(cached["lgb_shift"]),
                "dc_shift":  float(cached["dc_shift"]),
                "lr_shift":  float(cached["lr_shift"]),
                "base_rate": float(cached["base_rate"]),
                "odds_a":   float(md["odds_yes"]),
                "odds_b":   float(md["odds_no"]),
                "bookie_a": odds_source,
                "bookie_b": odds_source,
                "side_a_label": "yes",
                "side_b_label": "no",
                "outcome":  int(md["actual"]),
            })

    return pd.DataFrame(rows)


def generate_pl_btts_footiqo(seasons: range) -> pd.DataFrame:
    return generate_pl_btts(seasons, odds_source="footiqo")


def generate_pl_btts_betfair(seasons: range) -> pd.DataFrame:
    return generate_pl_btts(seasons, odds_source="betfair")


def generate_efl_ou25(seasons: range) -> pd.DataFrame:
    """Generate OOF cache rows for EFL O/U 2.5 (3-model ensemble)."""
    from championship_pipeline import run_pipeline as run_champ_pipeline
    from championship_backtest import precompute_season_efl
    from championship_model import tune_dc_params_champ

    print(f"[EFL ou25] Loading Championship pipeline ...")
    t0 = time.time()
    data = run_champ_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])
    print(f"  ...loaded {len(full_df):,} rows in {time.time()-t0:.0f}s")

    print("[EFL ou25] Tuning Dixon-Coles (champ) hyperparameters ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_champ(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["B365Greater2.5"].notna().sum()
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue
        print(f"  S{season}: training on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        cached = precompute_season_efl(train_df, test_df, features, dc_kwargs)
        print(f"    ...{time.time()-t0:.0f}s")
        for md in cached["match_data"]:
            pi = md["pred_idx"]
            rows.append({
                "season": int(md["season"]),
                "date": str(md["date"])[:10],
                "home_team": md["home"],
                "away_team": md["away"],
                "xgb_prob": float(cached["xgb_raw"][pi]),
                "lgb_prob": float(cached["lgb_raw"][pi]),
                "dc_prob":  float(cached["dc_raw"][pi]),
                "lr_prob":  float("nan"),              # 3-model ensemble
                "xgb_shift": float(cached["xgb_shift"]),
                "lgb_shift": float(cached["lgb_shift"]),
                "dc_shift":  float(cached["dc_shift"]),
                "lr_shift":  float("nan"),
                "base_rate": float(cached["base_rate"]),
                "odds_a":   float(md["odds_over"]),
                "odds_b":   float(md["odds_under"]),
                "bookie_a": "Bet365",
                "bookie_b": "Bet365",
                "side_a_label": "over",
                "side_b_label": "under",
                "outcome":  int(md["actual"]),
            })
    return pd.DataFrame(rows)


def _fit_dc_and_predict_ou15(
    train_df: pd.DataFrame, test_df: pd.DataFrame, dc_kwargs: dict,
) -> np.ndarray:
    """Train DC on train_df, return P(Over 1.5) per fixture in test_df."""
    from model import DixonColesPredictor
    dc = DixonColesPredictor(**dc_kwargs)
    dc.fit(train_df)
    return dc.predict_proba_ou15_df(test_df)


def _make_ou15_rows(
    test_df: pd.DataFrame, probs: np.ndarray, base_rate: float,
    shift: float, odds_a_col: str, odds_b_col: str,
    bookie_a: str, bookie_b: str, n_model_slots: int = 4,
) -> list[dict]:
    """Construct OOF rows for O/U 1.5.

    Alt-lines are DC-only (no XGB/LGB/LR), so we replicate dc_prob into
    ``n_model_slots`` model slots in the schema. That makes the
    validator's n_agree count reach the league-appropriate max
    (4 for PL, 3 for EFL) when DC agrees with the bet direction.

    Args:
        n_model_slots: 4 for PL (XGB+LGB+DC+LR), 3 for EFL (XGB+LGB+DC).
            Must match the league's agree_scale key range in staking.py,
            or stake will be 0 via the default-first-value fallback.
    """
    assert n_model_slots in (3, 4), f"n_model_slots must be 3 or 4, got {n_model_slots}"
    rows = []
    for i, (_, row) in enumerate(test_df.reset_index(drop=True).iterrows()):
        home_g = row.get("Home_Goals")
        away_g = row.get("Away_Goals")
        if pd.isna(home_g) or pd.isna(away_g):
            continue
        outcome = 1 if (float(home_g) + float(away_g)) > 1.5 else 0
        odds_a = row.get(odds_a_col)
        odds_b = row.get(odds_b_col)
        if pd.isna(odds_a) or pd.isna(odds_b):
            continue
        p = float(probs[i])
        # For EFL, leave lr_prob/lr_shift as NaN so the validator's
        # per_model array length matches the 3-model agree scale.
        rec = {
            "season": int(row.get("SeasonIndex", 0)),
            "date": str(row.get("Date", ""))[:10],
            "home_team": row.get("Home_Team", ""),
            "away_team": row.get("Away_Team", ""),
            "xgb_prob": p, "lgb_prob": p, "dc_prob": p,
            "xgb_shift": shift, "lgb_shift": shift, "dc_shift": shift,
            "lr_prob": p if n_model_slots == 4 else float("nan"),
            "lr_shift": shift if n_model_slots == 4 else float("nan"),
            "base_rate": float(base_rate),
            "odds_a": float(odds_a),
            "odds_b": float(odds_b),
            "bookie_a": bookie_a,
            "bookie_b": bookie_b,
            "side_a_label": "over",
            "side_b_label": "under",
            "outcome": outcome,
        }
        rows.append(rec)
    return rows


def generate_pl_ou15(seasons: range) -> pd.DataFrame:
    """PL O/U 1.5 via DC scoreline distribution + footiqo O15/U15 odds."""
    from pipeline import run_pipeline
    from btts_data import load_btts_odds
    from model import tune_dc_params

    print("[PL ou15] Loading pipeline ...")
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]

    # btts_data.merge_btts_odds only carries BTTSY/BTTSN across; for
    # O/U 1.5 we need O15/U15 too. Do the merge directly here.
    print("[PL ou15] Merging footiqo O15/U15 odds ...")
    footiqo = load_btts_odds()
    footiqo = footiqo[["DateOnly", "Home_Team", "Away_Team",
                       "O15", "U15"]].copy()
    footiqo["_merge_date"] = footiqo["DateOnly"].astype(str)
    full_df["_merge_date"] = pd.to_datetime(full_df["Date"]).dt.strftime("%Y-%m-%d")
    full_df = full_df.merge(
        footiqo[["_merge_date", "Home_Team", "Away_Team", "O15", "U15"]],
        on=["_merge_date", "Home_Team", "Away_Team"],
        how="left",
    )
    full_df = full_df.drop(columns=["_merge_date"])

    print("[PL ou15] Tuning Dixon-Coles hyperparameters ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["O15"].notna().sum() \
            if "O15" in test_df.columns else 0
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue
        print(f"  S{season}: training DC on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        probs = _fit_dc_and_predict_ou15(train_df, test_df, dc_kwargs)
        # Per-season base rate from recent 2 seasons
        recent_s = sorted(train_df["SeasonIndex"].unique())[-2:]
        recent = train_df[train_df["SeasonIndex"].isin(recent_s)]
        base_rate = ((recent["Home_Goals"] + recent["Away_Goals"]) > 1.5).mean() \
            if len(recent) > 100 else 0.75
        from backtest import _calibrate
        _, shift = _calibrate(probs, base_rate)
        print(f"    ...{time.time()-t0:.0f}s (base={base_rate:.3f}, "
              f"shift={shift:+.3f})")
        rows.extend(_make_ou15_rows(test_df, probs, base_rate, shift,
                                    "O15", "U15", "footiqo", "footiqo",
                                    n_model_slots=4))
    return pd.DataFrame(rows)


def generate_efl_ou15(seasons: range) -> pd.DataFrame:
    """EFL O/U 1.5 via Championship DC + Betfair OU1.5 odds (Phase 2.3)."""
    from championship_pipeline import run_pipeline as run_champ_pipeline
    from championship_model import tune_dc_params_champ
    from backtest import _calibrate

    print("[EFL ou15] Loading Championship pipeline ...")
    data = run_champ_pipeline(verbose=False)
    full_df = data["full_df"]

    print("[EFL ou15] Merging Betfair EFL O/U 1.5 odds ...")
    bf = pd.read_csv(PROJECT_ROOT / "data" / "betfair_efl_ou15.csv")
    bf["_date"] = pd.to_datetime(bf["Date"]).dt.date
    bf = bf[["_date", "Home_Team", "Away_Team", "over_ltp", "under_ltp"]]
    bf = bf.rename(columns={"over_ltp": "O15", "under_ltp": "U15"})
    full_df["_date"] = pd.to_datetime(full_df["Date"]).dt.date
    full_df = full_df.merge(bf, on=["_date", "Home_Team", "Away_Team"],
                             how="left")
    full_df = full_df.drop(columns=["_date"])

    print("[EFL ou15] Tuning Dixon-Coles (champ) ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_champ(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["O15"].notna().sum() \
            if "O15" in test_df.columns else 0
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue
        print(f"  S{season}: training DC on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        probs = _fit_dc_and_predict_ou15(train_df, test_df, dc_kwargs)
        recent_s = sorted(train_df["SeasonIndex"].unique())[-2:]
        recent = train_df[train_df["SeasonIndex"].isin(recent_s)]
        base_rate = ((recent["Home_Goals"] + recent["Away_Goals"]) > 1.5).mean() \
            if len(recent) > 100 else 0.75
        _, shift = _calibrate(probs, base_rate)
        print(f"    ...{time.time()-t0:.0f}s (base={base_rate:.3f}, "
              f"shift={shift:+.3f})")
        rows.extend(_make_ou15_rows(test_df, probs, base_rate, shift,
                                    "O15", "U15", "Betfair", "Betfair",
                                    n_model_slots=3))
    return pd.DataFrame(rows)


def generate_efl_btts(seasons: range) -> pd.DataFrame:
    """EFL BTTS via new walk-forward harness + Betfair YES/NO odds.

    No existing EFL BTTS backtest existed — this is the Phase 3 Pass 5
    addition. Uses ``precompute_btts_season_efl`` and the Betfair-sourced
    per-match yes/no prices from ``data/betfair_efl_btts.csv``.

    Following the Pass 4 lesson: Betfair LTP at settlement time is mostly
    in-play adulterated, so we use ``yes_ltp_first`` / ``no_ltp_first``
    (first-ever trade) as a pre-match proxy. Imperfect but orders of
    magnitude cleaner than settlement LTP. Phase 5 action item: re-extract
    with true pre-kickoff close-time snapshot.
    """
    from championship_pipeline import run_pipeline as run_champ_pipeline
    from championship_backtest import precompute_btts_season_efl
    from championship_model import tune_dc_params_champ

    print("[EFL btts] Loading Championship pipeline ...")
    t0 = time.time()
    data = run_champ_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data.get("btts_features", data["features"]))
    print(f"  ...loaded {len(full_df):,} rows, {len(features)} BTTS features "
          f"in {time.time()-t0:.0f}s")

    # Merge Betfair BTTS odds (first-trade proxy for pre-match)
    print("[EFL btts] Merging Betfair BTTS odds (first-trade proxy) ...")
    bf = pd.read_csv(PROJECT_ROOT / "data" / "betfair_efl_btts.csv")
    bf["_date"] = pd.to_datetime(bf["Date"]).dt.date
    bf = bf[["_date", "Home_Team", "Away_Team",
             "yes_ltp_first", "no_ltp_first"]].rename(
        columns={"yes_ltp_first": "BTTSY", "no_ltp_first": "BTTSN"})
    full_df["_date"] = pd.to_datetime(full_df["Date"]).dt.date
    full_df = full_df.merge(bf, on=["_date", "Home_Team", "Away_Team"],
                             how="left")
    full_df = full_df.drop(columns=["_date"])

    print("[EFL btts] Tuning Dixon-Coles (champ) ...")
    t0 = time.time()
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params_champ(tune_df)
    print(f"  ...tuned in {time.time()-t0:.0f}s: {dc_kwargs}")

    rows: list[dict] = []
    for season in seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                           & (full_df["SeasonIndex"] < season)].copy()
        test_df = full_df[full_df["SeasonIndex"] == season].copy()
        has_odds = test_df["BTTSY"].notna().sum() \
            if "BTTSY" in test_df.columns else 0
        if has_odds < 50 or len(train_df) < 500:
            print(f"  S{season}: skipping — odds={has_odds}, train={len(train_df)}")
            continue
        print(f"  S{season}: training on {len(train_df):,} rows, "
              f"predicting {len(test_df):,} fixtures ...", flush=True)
        t0 = time.time()
        cached = precompute_btts_season_efl(
            train_df, test_df, features, dc_kwargs)
        print(f"    ...{time.time()-t0:.0f}s")
        for md in cached["match_data"]:
            pi = md["pred_idx"]
            rows.append({
                "season": int(md["season"]),
                "date": str(md["date"])[:10],
                "home_team": md["home"],
                "away_team": md["away"],
                "xgb_prob": float(cached["xgb_raw"][pi]),
                "lgb_prob": float(cached["lgb_raw"][pi]),
                "dc_prob":  float(cached["dc_raw"][pi]),
                "lr_prob":  float("nan"),              # EFL is 3-model
                "xgb_shift": float(cached["xgb_shift"]),
                "lgb_shift": float(cached["lgb_shift"]),
                "dc_shift":  float(cached["dc_shift"]),
                "lr_shift":  float("nan"),
                "base_rate": float(cached["base_rate"]),
                "odds_a":   float(md["odds_yes"]),
                "odds_b":   float(md["odds_no"]),
                "bookie_a": "Betfair",
                "bookie_b": "Betfair",
                "side_a_label": "yes",
                "side_b_label": "no",
                "outcome":  int(md["actual"]),
            })
    return pd.DataFrame(rows)


GENERATORS: dict[tuple[str, str], callable] = {
    ("PL", "ou25"): generate_pl_ou25,
    ("PL", "btts"): generate_pl_btts_footiqo,
    ("PL", "btts_betfair"): generate_pl_btts_betfair,
    ("EFL", "ou25"): generate_efl_ou25,
    ("PL", "ou15"): generate_pl_ou15,
    ("EFL", "ou15"): generate_efl_ou15,
    ("EFL", "btts"): generate_efl_btts,
}


def _parse_seasons(s: str) -> range:
    """Parse '19-24' into range(19, 25)."""
    parts = s.split("-")
    if len(parts) != 2:
        raise ValueError(f"Bad seasons spec: {s} (expected START-END)")
    a, b = int(parts[0]), int(parts[1])
    return range(a, b + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", required=True, choices=("PL", "EFL"))
    ap.add_argument("--market", required=True,
                    choices=("ou25", "ou15", "btts", "btts_betfair"))
    ap.add_argument("--seasons", default="19-24",
                    help="Test-season range, e.g. 19-24 = S19..S24 inclusive")
    ap.add_argument("--out-dir", default=None,
                    help="Override default output directory")
    args = ap.parse_args()

    key = (args.league, args.market)
    if key not in GENERATORS:
        print(f"[ERROR] No generator for {key}. Available: "
              f"{sorted(GENERATORS.keys())}", file=sys.stderr)
        print("Will land in a later Phase 3 pass.", file=sys.stderr)
        return 2

    seasons = _parse_seasons(args.seasons)
    print(f"Generating OOF cache: {args.league} {args.market} "
          f"seasons S{seasons[0]}..S{seasons[-1]}")

    gen = GENERATORS[key]
    df = gen(seasons)

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defensive: persist to pickle first (always works with stdlib) so a
    # downstream format issue can't cost us another full training run.
    pkl_path = out_dir / f"{args.league.lower()}_{args.market}.pkl"
    df.to_pickle(pkl_path)
    print(f"\nWrote pickle fallback: {pkl_path} ({len(df):,} rows)")
    if len(df) == 0:
        print("[ERROR] Empty output — no fixtures had odds + train data.",
              file=sys.stderr)
        return 3

    # Then write the primary parquet output.
    out_path = out_dir / f"{args.league.lower()}_{args.market}.parquet"
    try:
        df.to_parquet(out_path, index=False)
        print(f"Wrote parquet: {out_path}")
    except ImportError as e:
        print(f"[WARN] Parquet write failed ({e}); pickle fallback is in place.",
              flush=True)
        out_path = pkl_path
    print(f"Per-season counts:")
    for s, n in df["season"].value_counts().sort_index().items():
        with_odds = df[(df["season"] == s) & df["odds_a"].notna()
                        & df["odds_b"].notna()].shape[0]
        print(f"  S{s}: {n:,} fixtures, {with_odds:,} with complete odds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
