"""LR Ablation Test: determine whether Logistic Regression should remain in the PL O/U 2.5 pipeline.

Tests 4 configurations of the PL O/U 2.5 ensemble:
  A) 4-model equal average + 4-model agreement (current backtest)
  B) 3-model stacker + 4-model agreement (current live system)
  C) 3-model stacker + 3-model agreement (clean LR removal)
  D) 4-model stacker + 4-model agreement (let stacker learn LR weight)

All configs use identical walk-forward CV, staking logic, and random seeds.
Output: comparison table with ROI, bets, max drawdown, win rate, avg edge, AUC.

Usage:
    python scripts/lr_ablation_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from pipeline import run_pipeline
from model import (
    train_xgb, train_lgb, train_logreg, DixonColesPredictor, tune_dc_params,
)
from backtest import (
    _calibrate, _calibrate_single, _lr_predict, RegimeDetector,
    compute_drawdown_factor, DEFAULT_CONFIG,
)
from staking import (
    shrink_edge, refined_kelly as _staking_refined_kelly,
    PL_AGREE_SCALE, EFL_AGREE_SCALE,
)
from config import DEVIG_DISCOUNT


START_SEASON = 19
END_SEASON = 25


def _train_models_for_season(train_df, test_df, features, dc_kwargs):
    """Train all 4 base models and return raw predictions on test set."""
    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values

    train_seasons = sorted(train_df["SeasonIndex"].unique())
    last_season = train_seasons[-1]
    es_val_mask = train_df["SeasonIndex"] == last_season
    es_train_mask = ~es_val_mask

    X_es_train = train_df.loc[es_train_mask, features].values
    y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
    X_es_val = train_df.loc[es_val_mask, features].values
    y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

    if len(X_es_train) < 100 or len(X_es_val) < 50:
        n_val = min(380, len(train_df) // 5)
        X_es_train, y_es_train = X_train[:-n_val], y_train[:-n_val]
        X_es_val, y_es_val = X_train[-n_val:], y_train[-n_val:]

    xgb_m = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
    lgb_m = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val, feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    X_test = test_df[features].values
    xgb_raw = xgb_m.predict_proba(X_test)[:, 1]
    lgb_raw = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_raw = _lr_predict(lr_m, lr_scaler, X_test)
    dc_raw = dc_m.predict_proba_df(test_df)

    return {
        "xgb_raw": xgb_raw, "lgb_raw": lgb_raw,
        "lr_raw": lr_raw, "dc_raw": dc_raw,
        "y_test": test_df["Over_2_5"].values,
    }


def _train_stacker_oof(full_df, features, dc_kwargs, include_lr=False):
    """Train a logistic stacker on walk-forward OOF predictions.

    Args:
        include_lr: If True, stacker takes 4 inputs (XGB+LGB+DC+LR).
                    If False, stacker takes 3 inputs (XGB+LGB+DC).

    Returns:
        Fitted LogisticRegression or None if insufficient OOF data.
    """
    oof_records = []

    for val_season in range(START_SEASON, END_SEASON + 1):
        train_df = full_df[
            (full_df["SeasonIndex"] >= 14) & (full_df["SeasonIndex"] < val_season)
        ].copy()
        val_df = full_df[full_df["SeasonIndex"] == val_season].copy()

        if len(train_df) < 100 or len(val_df) < 50:
            continue

        X_tr = train_df[features].values
        y_tr = train_df["Over_2_5"].values
        X_v = val_df[features].values
        y_v = val_df["Over_2_5"].values

        xgb_m = train_xgb(X_tr, y_tr, X_v, y_v)
        lgb_m = train_lgb(X_tr, y_tr, X_v, y_v, feature_names=features)
        dc_m = DixonColesPredictor(**dc_kwargs)
        dc_m.fit(train_df)

        xgb_p = xgb_m.predict_proba(X_v)[:, 1]
        lgb_p = lgb_m.predict_proba(pd.DataFrame(X_v, columns=features))[:, 1]
        dc_p = dc_m.predict_proba_df(val_df)

        if include_lr:
            lr_m, lr_scaler = train_logreg(X_tr, y_tr)
            lr_p = _lr_predict(lr_m, lr_scaler, X_v)

        for i in range(len(val_df)):
            rec = {"xgb": xgb_p[i], "lgb": lgb_p[i], "dc": dc_p[i], "y": y_v[i]}
            if include_lr:
                rec["lr"] = lr_p[i]
            oof_records.append(rec)

    oof_df = pd.DataFrame(oof_records)
    if len(oof_df) < 100:
        return None

    if include_lr:
        X_stack = oof_df[["xgb", "lgb", "dc", "lr"]].values
    else:
        X_stack = oof_df[["xgb", "lgb", "dc"]].values

    stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    stacker.fit(X_stack, oof_df["y"].values)

    cols = ["XGB", "LGB", "DC"] + (["LR"] if include_lr else [])
    coefs = stacker.coef_[0]
    print(f"    Stacker coefs: {', '.join(f'{c}={v:.3f}' for c, v in zip(cols, coefs))}")
    auc = roc_auc_score(oof_df["y"], stacker.predict_proba(X_stack)[:, 1])
    print(f"    Stacker OOF AUC: {auc:.4f}")

    return stacker


def _simulate_season(test_df, raw_preds, config, stacker, stacker_mode,
                     agree_models, agree_scale, cumulative_bankroll, peak_bankroll):
    """Simulate betting on one season with a given configuration.

    Args:
        raw_preds: dict with xgb_raw, lgb_raw, lr_raw, dc_raw, y_test
        stacker: fitted LogisticRegression (or None for average mode)
        stacker_mode: 'avg4' | 'stack3' | 'stack4' — how to compute ensemble prob
        agree_models: list of model keys to include in agreement vote
        agree_scale: PL_AGREE_SCALE or EFL_AGREE_SCALE
        cumulative_bankroll, peak_bankroll: for drawdown tracking

    Returns:
        (bets_list, cumulative_bankroll, peak_bankroll)
    """
    xgb_raw = raw_preds["xgb_raw"]
    lgb_raw = raw_preds["lgb_raw"]
    lr_raw = raw_preds["lr_raw"]
    dc_raw = raw_preds["dc_raw"]
    y_test = raw_preds["y_test"]

    # Calibration
    train_seasons = sorted(test_df["SeasonIndex"].unique())
    base_rate = y_test.mean() if len(y_test) > 0 else 0.5
    # Use a simple 2-season base rate approximation (same as backtest_season)
    recent_mask = test_df["SeasonIndex"] == test_df["SeasonIndex"].max()
    base_rate_approx = 0.52  # PL historical average as fallback

    _, xgb_shift = _calibrate(xgb_raw, base_rate_approx)
    _, lgb_shift = _calibrate(lgb_raw, base_rate_approx)
    _, lr_shift = _calibrate(lr_raw, base_rate_approx)
    _, dc_shift = _calibrate(dc_raw, base_rate_approx)

    xgb_cal = np.array([_calibrate_single(p, xgb_shift) for p in xgb_raw])
    lgb_cal = np.array([_calibrate_single(p, lgb_shift) for p in lgb_raw])
    lr_cal = np.array([_calibrate_single(p, lr_shift) for p in lr_raw])
    dc_cal = np.array([_calibrate_single(p, dc_shift) for p in dc_raw])

    # Regime detector
    regime = RegimeDetector(
        prior_base_rate=base_rate_approx,
        window=config.get("regime_window", 40),
        blend_speed=config.get("regime_blend_speed", 0.4),
        trigger_threshold=config.get("regime_trigger", 0.04),
        min_matches=config.get("regime_min_matches", 15),
    )

    blend_w = config.get("blend_weight", 0.35)
    min_edge = config.get("min_edge", 0.02)
    min_agree = config.get("min_agree", 2)
    kelly_fraction = config.get("kelly_fraction", 0.25)
    max_stake_pct = config.get("max_stake_pct", 0.05)

    use_early = config.get("early_season", True)
    early_matches = config.get("early_season_matches", 60)
    early_blend_w = config.get("early_blend_weight", 0.20)
    early_min_edge = config.get("early_min_edge", 0.03)
    early_kelly = config.get("early_kelly_fraction", 0.15)

    test_sorted = test_df.sort_values("Date").reset_index(drop=True)
    sorted_positions = test_df.reset_index(drop=True).sort_values("Date").index.tolist()

    bets = []

    for match_num, (_, row) in enumerate(test_sorted.iterrows()):
        pred_idx = sorted_positions[match_num]
        actual = y_test[pred_idx]
        regime.update(actual)

        # Get calibrated per-model predictions (with regime adjustment)
        if regime.regime_shift_detected():
            adj_rate = regime.get_adjusted_base_rate()
            _, xgb_s = _calibrate(xgb_raw, adj_rate)
            _, lgb_s = _calibrate(lgb_raw, adj_rate)
            _, lr_s = _calibrate(lr_raw, adj_rate)
            _, dc_s = _calibrate(dc_raw, adj_rate)
            xgb_p = _calibrate_single(xgb_raw[pred_idx], xgb_s)
            lgb_p = _calibrate_single(lgb_raw[pred_idx], lgb_s)
            lr_p = _calibrate_single(lr_raw[pred_idx], lr_s)
            dc_p = _calibrate_single(dc_raw[pred_idx], dc_s)
        else:
            xgb_p = xgb_cal[pred_idx]
            lgb_p = lgb_cal[pred_idx]
            lr_p = lr_cal[pred_idx]
            dc_p = dc_cal[pred_idx]

        # Compute ensemble probability based on mode
        if stacker_mode == "avg4":
            model_over = (xgb_p + lgb_p + lr_p + dc_p) / 4.0
        elif stacker_mode == "stack3" and stacker is not None:
            base = np.array([[xgb_p, lgb_p, dc_p]])
            model_over = float(stacker.predict_proba(base)[:, 1][0])
        elif stacker_mode == "stack4" and stacker is not None:
            base = np.array([[xgb_p, lgb_p, dc_p, lr_p]])
            model_over = float(stacker.predict_proba(base)[:, 1][0])
        else:
            model_over = (xgb_p + lgb_p + dc_p) / 3.0

        model_under = 1 - model_over

        # Agreement: count from specified model set only
        all_probs = {"xgb": xgb_p, "lgb": lgb_p, "lr": lr_p, "dc": dc_p}
        per_model = np.array([all_probs[k] for k in agree_models])

        # Odds
        odds_over = row.get("B365Greater2.5", np.nan)
        odds_under = row.get("B365LessThan2.5", np.nan)
        if pd.isna(odds_over) or pd.isna(odds_under):
            continue
        if odds_over <= 1.0 or odds_under <= 1.0:
            continue

        raw_o = 1.0 / odds_over
        raw_u = 1.0 / odds_under
        overround = raw_o + raw_u
        fair_over = raw_o / overround
        fair_under = raw_u / overround

        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)

        is_early = use_early and match_num < early_matches
        active_blend_w = early_blend_w if is_early else blend_w
        active_min_edge = early_min_edge if is_early else min_edge
        active_kelly = early_kelly if is_early else kelly_fraction

        for side, model_p, fair_p, odds in [
            ("over", model_over, fair_over, odds_over),
            ("under", model_under, fair_under, odds_under),
        ]:
            blended_p = active_blend_w * model_p + (1 - active_blend_w) * fair_p
            edge = blended_p - fair_p
            ev = blended_p * odds - 1

            if side == "over":
                n_agree = int(np.sum(per_model > fair_over))
            else:
                n_agree = int(np.sum((1 - per_model) > fair_under))

            edge = shrink_edge(edge, n_agree)

            if ev <= 0 or edge < active_min_edge or n_agree < min_agree:
                continue

            stake = _staking_refined_kelly(
                blended_p, odds, n_agree, edge,
                agree_scale=agree_scale,
                kelly_fraction=active_kelly,
                max_stake_pct=max_stake_pct,
                drawdown_factor=dd_factor,
            )

            if stake <= 0:
                continue

            won = (side == "over" and actual == 1) or (side == "under" and actual == 0)
            actual_stake = stake * cumulative_bankroll
            if won:
                cumulative_bankroll += actual_stake * (odds - 1)
            else:
                cumulative_bankroll -= actual_stake
            peak_bankroll = max(peak_bankroll, cumulative_bankroll)

            profit = stake * (odds - 1) if won else -stake

            bets.append({
                "season": row.get("SeasonIndex", 0),
                "side": side,
                "odds": odds,
                "edge": edge,
                "n_agree": n_agree,
                "stake_pct": stake,
                "won": won,
                "profit_pct": profit,
            })

    return bets, cumulative_bankroll, peak_bankroll


def _compute_metrics(all_bets, final_bankroll):
    """Compute summary metrics from all bets across seasons."""
    if not all_bets:
        return {"n_bets": 0, "roi": 0, "max_dd": 0, "win_rate": 0, "avg_edge": 0}

    df = pd.DataFrame(all_bets)
    total_staked = df["stake_pct"].sum()
    total_profit = df["profit_pct"].sum()

    # Max drawdown from cumulative P&L curve
    cumulative = df["profit_pct"].cumsum()
    running_max = cumulative.cummax()
    drawdowns = running_max - cumulative
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0

    return {
        "n_bets": len(df),
        "roi": total_profit / total_staked if total_staked > 0 else 0,
        "total_profit": total_profit,
        "max_dd": max_dd,
        "win_rate": df["won"].mean(),
        "avg_edge": df["edge"].mean(),
        "avg_odds": df["odds"].mean(),
        "avg_stake": df["stake_pct"].mean(),
        "final_bankroll": final_bankroll,
        "n_over": len(df[df["side"] == "over"]),
        "n_under": len(df[df["side"] == "under"]),
    }


def _per_model_auc(full_df, features, dc_kwargs):
    """Compute per-model AUC on the test season (S25) for reporting."""
    train_df = full_df[(full_df["SeasonIndex"] >= 14) & (full_df["SeasonIndex"] < 25)].copy()
    test_df = full_df[full_df["SeasonIndex"] == 25].copy()

    if len(test_df) < 50:
        return {}

    X_train = train_df[features].values
    y_train = train_df["Over_2_5"].values
    X_test = test_df[features].values
    y_test = test_df["Over_2_5"].values

    n_val = min(380, len(train_df) // 5)
    xgb_m = train_xgb(X_train[:-n_val], y_train[:-n_val], X_train[-n_val:], y_train[-n_val:])
    lgb_m = train_lgb(X_train[:-n_val], y_train[:-n_val], X_train[-n_val:], y_train[-n_val:],
                      feature_names=features)
    lr_m, lr_scaler = train_logreg(X_train, y_train)
    dc_m = DixonColesPredictor(**dc_kwargs)
    dc_m.fit(train_df)

    xgb_p = xgb_m.predict_proba(X_test)[:, 1]
    lgb_p = lgb_m.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    lr_p = _lr_predict(lr_m, lr_scaler, X_test)
    dc_p = dc_m.predict_proba_df(test_df)

    return {
        "XGB": roc_auc_score(y_test, xgb_p),
        "LGB": roc_auc_score(y_test, lgb_p),
        "LR": roc_auc_score(y_test, lr_p),
        "DC": roc_auc_score(y_test, dc_p),
        "Avg3": roc_auc_score(y_test, (xgb_p + lgb_p + dc_p) / 3),
        "Avg4": roc_auc_score(y_test, (xgb_p + lgb_p + lr_p + dc_p) / 4),
    }


def main():
    print("=" * 80)
    print("LR ABLATION TEST — PL O/U 2.5")
    print("=" * 80)

    print("\n[1/5] Loading pipeline data...")
    data = run_pipeline(verbose=False)
    full_df = data["full_df"]
    features = list(data["features"])

    print("[2/5] Tuning Dixon-Coles hyperparameters...")
    tune_df = full_df[full_df["SeasonIndex"] >= 14].copy()
    dc_kwargs = tune_dc_params(tune_df)

    print("\n[3/5] Per-model AUC on test season (S25):")
    aucs = _per_model_auc(full_df, features, dc_kwargs)
    for name, auc in aucs.items():
        marker = " <-- SUB-RANDOM" if auc < 0.5 else ""
        print(f"    {name:>5s}: {auc:.4f}{marker}")

    print("\n[4/5] Training stackers...")
    print("  Training 3-model stacker (XGB+LGB+DC):")
    stacker_3 = _train_stacker_oof(full_df, features, dc_kwargs, include_lr=False)
    print("  Training 4-model stacker (XGB+LGB+DC+LR):")
    stacker_4 = _train_stacker_oof(full_df, features, dc_kwargs, include_lr=True)

    # Config definitions
    configs = {
        "A: 4-avg + 4-agree": {
            "stacker": None, "stacker_mode": "avg4",
            "agree_models": ["xgb", "lgb", "lr", "dc"],
            "agree_scale": PL_AGREE_SCALE,
        },
        "B: 3-stack + 4-agree": {
            "stacker": stacker_3, "stacker_mode": "stack3",
            "agree_models": ["xgb", "lgb", "lr", "dc"],
            "agree_scale": PL_AGREE_SCALE,
        },
        "C: 3-stack + 3-agree": {
            "stacker": stacker_3, "stacker_mode": "stack3",
            "agree_models": ["xgb", "lgb", "dc"],
            "agree_scale": EFL_AGREE_SCALE,
        },
        "D: 4-stack + 4-agree": {
            "stacker": stacker_4, "stacker_mode": "stack4",
            "agree_models": ["xgb", "lgb", "lr", "dc"],
            "agree_scale": PL_AGREE_SCALE,
        },
    }

    print(f"\n[5/5] Running walk-forward backtest (S{START_SEASON}–S{END_SEASON})...")
    print(f"  Config: blend={DEFAULT_CONFIG['blend_weight']}, "
          f"min_edge={DEFAULT_CONFIG['min_edge']}, "
          f"kelly={DEFAULT_CONFIG['kelly_fraction']}")

    results = {}

    for cfg_name, cfg in configs.items():
        print(f"\n  --- {cfg_name} ---")
        all_bets = []
        cumulative_bankroll = 1.0
        peak_bankroll = 1.0

        for season in range(START_SEASON, END_SEASON + 1):
            train_df = full_df[
                (full_df["SeasonIndex"] >= 14) & (full_df["SeasonIndex"] < season)
            ].copy()
            test_df = full_df[full_df["SeasonIndex"] == season].copy()

            has_odds = test_df["B365Greater2.5"].notna().sum()
            if has_odds < 50 or len(train_df) < 500:
                continue

            raw_preds = _train_models_for_season(train_df, test_df, features, dc_kwargs)

            season_bets, cumulative_bankroll, peak_bankroll = _simulate_season(
                test_df, raw_preds, DEFAULT_CONFIG,
                stacker=cfg["stacker"],
                stacker_mode=cfg["stacker_mode"],
                agree_models=cfg["agree_models"],
                agree_scale=cfg["agree_scale"],
                cumulative_bankroll=cumulative_bankroll,
                peak_bankroll=peak_bankroll,
            )
            all_bets.extend(season_bets)
            n = len(season_bets)
            print(f"    S{season}: {n} bets, bankroll={cumulative_bankroll:.4f}")

        metrics = _compute_metrics(all_bets, cumulative_bankroll)
        results[cfg_name] = metrics

    # Print comparison table
    print("\n")
    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n{'Config':<25s} {'Bets':>5s} {'ROI':>7s} {'WinR':>6s} "
          f"{'AvgEdge':>8s} {'MaxDD':>7s} {'AvgOdds':>8s} {'Bankroll':>9s}")
    print("-" * 80)

    for name, m in results.items():
        print(f"{name:<25s} {m['n_bets']:>5d} {m['roi']:>+6.1%} "
              f"{m['win_rate']:>5.1%} {m['avg_edge']:>7.3%} "
              f"{m['max_dd']:>6.1%} {m['avg_odds']:>7.3f} "
              f"{m['final_bankroll']:>9.4f}")

    print("-" * 80)
    print(f"\n{'Config':<25s} {'Over':>5s} {'Under':>5s} {'Profit':>8s}")
    print("-" * 45)
    for name, m in results.items():
        print(f"{name:<25s} {m['n_over']:>5d} {m['n_under']:>5d} "
              f"{m['total_profit']:>+7.3f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
  - If C >= A on ROI (within 1pp) with similar/lower drawdown:
      → Remove LR entirely from PL O/U 2.5 (simplest, cleanest)
  - If B >= A:
      → Current live system is fine, update backtest to use stacker
  - If D > all:
      → Add LR to stacker, let it learn the weight
  - If A > all:
      → Stacker was a regression, revert live to 4-model average
    """)


if __name__ == "__main__":
    main()
