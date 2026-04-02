"""
Live prediction engine for O/U 2.5 and BTTS markets.

Loads trained models, generates predictions for upcoming fixtures,
blends with live odds, and outputs bet recommendations.
"""
import logging
import numpy as np
import pandas as pd
from typing import Optional

from pipeline import run_pipeline
from config import ALL_FEATURES, BTTS_ALL_FEATURES
from model import (
    train_xgb, train_lgb, train_logreg, train_xgb_btts, train_lgb_btts,
    DixonColesPredictor, tune_dc_params, _fill_nan_median,
)
from backtest import (
    _calibrate, _calibrate_single, _lr_predict, DEFAULT_CONFIG,
    RegimeDetector, refined_kelly, compute_drawdown_factor,
)
from btts_backtest import BTTS_DEFAULT_CONFIG
from api.odds_api import (
    fetch_epl_odds, get_best_odds, get_best_btts_odds, match_to_our_teams,
)

logger = logging.getLogger(__name__)


class LivePredictor:
    """Generates bet recommendations for upcoming fixtures using trained models.

    Trains on all available historical data, calibrates probabilities,
    and compares against live bookmaker odds.
    """

    def __init__(
        self,
        ou_config: Optional[dict] = None,
        btts_config: Optional[dict] = None,
        verbose: bool = True,
    ) -> None:
        self.ou_config = ou_config or DEFAULT_CONFIG.copy()
        self.btts_config = btts_config or BTTS_DEFAULT_CONFIG.copy()
        self.verbose = verbose

        self._pipeline_data: Optional[dict] = None
        self._full_df: Optional[pd.DataFrame] = None
        self._ou_features: list[str] = []
        self._btts_features: list[str] = []

        # Trained models (set by train())
        self._ou_models: Optional[dict] = None
        self._btts_models: Optional[dict] = None
        self._ou_base_rate: float = 0.5
        self._btts_base_rate: float = 0.5
        self._dc_kwargs: dict = {}
        self._train_medians: Optional[pd.Series] = None
        self._our_teams: set[str] = set()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}")
        logger.info(msg)

    def load_data(self) -> None:
        """Load pipeline data and identify available features."""
        self._log("Loading pipeline data...")
        self._pipeline_data = run_pipeline(verbose=self.verbose)
        self._full_df = self._pipeline_data["full_df"]
        self._train_medians = self._pipeline_data.get("train_medians")

        self._ou_features = [f for f in ALL_FEATURES if f in self._full_df.columns]
        self._btts_features = [f for f in BTTS_ALL_FEATURES if f in self._full_df.columns]

        # Get current team names
        latest_season = self._full_df["SeasonIndex"].max()
        latest = self._full_df[self._full_df["SeasonIndex"] == latest_season]
        self._our_teams = set(latest["Home_Team"].unique()) | set(latest["Away_Team"].unique())

        self._log(f"O/U features: {len(self._ou_features)}")
        self._log(f"BTTS features: {len(self._btts_features)}")
        self._log(f"Current teams: {len(self._our_teams)}")

    def train(self) -> None:
        """Train all models on full historical data."""
        if self._full_df is None:
            self.load_data()

        df = self._full_df
        train_df = df[df["SeasonIndex"] >= 14].copy()

        self._log("Tuning Dixon-Coles parameters...")
        self._dc_kwargs = tune_dc_params(train_df)
        self._log(f"DC params: {self._dc_kwargs}")

        # Use last 2 seasons for early stopping validation
        train_seasons = sorted(train_df["SeasonIndex"].unique())
        last_season = train_seasons[-1]
        es_val_mask = train_df["SeasonIndex"] == last_season
        es_train_mask = ~es_val_mask

        # ── O/U 2.5 Models ──
        self._log("Training O/U 2.5 models...")
        X_train_ou = train_df[self._ou_features].values
        y_train_ou = train_df["Over_2_5"].values

        X_es_train = train_df.loc[es_train_mask, self._ou_features].values
        y_es_train = train_df.loc[es_train_mask, "Over_2_5"].values
        X_es_val = train_df.loc[es_val_mask, self._ou_features].values
        y_es_val = train_df.loc[es_val_mask, "Over_2_5"].values

        ou_xgb = train_xgb(X_es_train, y_es_train, X_es_val, y_es_val)
        ou_lgb = train_lgb(X_es_train, y_es_train, X_es_val, y_es_val,
                           feature_names=self._ou_features)
        ou_lr, ou_lr_scaler = train_logreg(X_train_ou, y_train_ou)
        ou_dc = DixonColesPredictor(**self._dc_kwargs)
        ou_dc.fit(train_df)

        # Base rate from last 2 seasons
        recent_mask = train_df["SeasonIndex"].isin(train_seasons[-2:])
        self._ou_base_rate = train_df.loc[recent_mask, "Over_2_5"].mean()

        self._ou_models = {
            "xgb": ou_xgb, "lgb": ou_lgb,
            "lr": ou_lr, "lr_scaler": ou_lr_scaler,
            "dc": ou_dc,
        }
        self._log(f"O/U base rate: {self._ou_base_rate:.3f}")

        # ── BTTS Models ──
        self._log("Training BTTS models...")
        X_train_btts = train_df[self._btts_features].values
        y_train_btts = ((train_df["Home_Goals"] > 0) &
                        (train_df["Away_Goals"] > 0)).astype(int).values

        X_es_train_b = train_df.loc[es_train_mask, self._btts_features].values
        y_es_train_b = ((train_df.loc[es_train_mask, "Home_Goals"] > 0) &
                        (train_df.loc[es_train_mask, "Away_Goals"] > 0)).astype(int).values
        X_es_val_b = train_df.loc[es_val_mask, self._btts_features].values
        y_es_val_b = ((train_df.loc[es_val_mask, "Home_Goals"] > 0) &
                      (train_df.loc[es_val_mask, "Away_Goals"] > 0)).astype(int).values

        btts_xgb = train_xgb_btts(X_es_train_b, y_es_train_b, X_es_val_b, y_es_val_b)
        btts_lgb = train_lgb_btts(X_es_train_b, y_es_train_b, X_es_val_b, y_es_val_b,
                                   feature_names=self._btts_features)
        btts_lr, btts_lr_scaler = train_logreg(X_train_btts, y_train_btts)
        btts_dc = DixonColesPredictor(**self._dc_kwargs)
        btts_dc.fit(train_df)

        recent_btts = ((train_df.loc[recent_mask, "Home_Goals"] > 0) &
                       (train_df.loc[recent_mask, "Away_Goals"] > 0)).mean()
        self._btts_base_rate = recent_btts

        self._btts_models = {
            "xgb": btts_xgb, "lgb": btts_lgb,
            "lr": btts_lr, "lr_scaler": btts_lr_scaler,
            "dc": btts_dc,
        }
        self._log(f"BTTS base rate: {self._btts_base_rate:.3f}")
        self._log("All models trained.")

    def _predict_ou(self, fixture_row: pd.Series) -> dict:
        """Generate O/U 2.5 probability from all 4 models for a single fixture."""
        m = self._ou_models
        feats = np.array([fixture_row[self._ou_features].values], dtype=float)

        # Fill NaN with training medians
        if self._train_medians is not None:
            for i, f in enumerate(self._ou_features):
                if np.isnan(feats[0, i]) and f in self._train_medians.index:
                    feats[0, i] = self._train_medians[f]

        xgb_raw = float(m["xgb"].predict_proba(feats)[:, 1][0])
        lgb_raw = float(m["lgb"].predict_proba(
            pd.DataFrame(feats, columns=self._ou_features))[:, 1][0])
        lr_raw = float(_lr_predict(m["lr"], m["lr_scaler"], feats)[0])
        dc_raw = float(m["dc"].predict_proba_df(
            fixture_row.to_frame().T)[0])

        # Calibrate
        base = self._ou_base_rate
        xgb_p = _calibrate_single(xgb_raw, _calibrate(np.array([xgb_raw]), base)[1])
        lgb_p = _calibrate_single(lgb_raw, _calibrate(np.array([lgb_raw]), base)[1])
        lr_p = _calibrate_single(lr_raw, _calibrate(np.array([lr_raw]), base)[1])
        dc_p = _calibrate_single(dc_raw, _calibrate(np.array([dc_raw]), base)[1])

        # Equal weight ensemble for O/U
        ensemble = (xgb_p + lgb_p + lr_p + dc_p) / 4.0

        return {
            "xgb": xgb_p, "lgb": lgb_p, "lr": lr_p, "dc": dc_p,
            "ensemble": ensemble,
            "per_model": np.array([xgb_p, lgb_p, lr_p, dc_p]),
        }

    def _predict_btts(self, fixture_row: pd.Series) -> dict:
        """Generate BTTS probability from all 4 models for a single fixture."""
        m = self._btts_models
        feats = np.array([fixture_row[self._btts_features].values], dtype=float)

        if self._train_medians is not None:
            for i, f in enumerate(self._btts_features):
                if np.isnan(feats[0, i]) and f in self._train_medians.index:
                    feats[0, i] = self._train_medians[f]

        xgb_raw = float(m["xgb"].predict_proba(feats)[:, 1][0])
        lgb_raw = float(m["lgb"].predict_proba(
            pd.DataFrame(feats, columns=self._btts_features))[:, 1][0])
        lr_raw = float(_lr_predict(m["lr"], m["lr_scaler"], feats)[0])
        dc_raw = float(m["dc"].predict_proba_btts_df(
            fixture_row.to_frame().T)[0])

        base = self._btts_base_rate
        xgb_p = _calibrate_single(xgb_raw, _calibrate(np.array([xgb_raw]), base)[1])
        lgb_p = _calibrate_single(lgb_raw, _calibrate(np.array([lgb_raw]), base)[1])
        lr_p = _calibrate_single(lr_raw, _calibrate(np.array([lr_raw]), base)[1])
        dc_p = _calibrate_single(dc_raw, _calibrate(np.array([dc_raw]), base)[1])

        # Weighted ensemble for BTTS (LR + DC get more weight)
        weights = np.array([0.20, 0.20, 0.30, 0.30])
        ensemble = float(np.dot([xgb_p, lgb_p, lr_p, dc_p], weights))

        return {
            "xgb": xgb_p, "lgb": lgb_p, "lr": lr_p, "dc": dc_p,
            "ensemble": ensemble,
            "per_model": np.array([xgb_p, lgb_p, lr_p, dc_p]),
        }

    def _evaluate_bet(
        self,
        model_p: float,
        fair_p: float,
        odds: float,
        per_model: np.ndarray,
        fair_threshold: float,
        config: dict,
        cumulative_bankroll: float = 1.0,
        peak_bankroll: float = 1.0,
    ) -> Optional[dict]:
        """Evaluate a single betting opportunity. Returns bet dict or None."""
        blend_w = config.get("blend_weight", 0.35)
        min_edge = config.get("min_edge", 0.02)
        min_agree = config.get("min_agree", 2)
        kelly_fraction = config.get("kelly_fraction", 0.25)
        max_stake_pct = config.get("max_stake_pct", 0.05)

        blended_p = blend_w * model_p + (1 - blend_w) * fair_p
        edge = blended_p - fair_p
        ev = blended_p * odds - 1

        n_agree = int(np.sum(per_model > fair_threshold))

        if ev <= 0 or edge < min_edge or n_agree < min_agree:
            return None

        dd_factor = compute_drawdown_factor(cumulative_bankroll, peak_bankroll)
        stake = refined_kelly(
            blended_p, odds, n_agree, edge,
            kelly_fraction=kelly_fraction,
            max_stake_pct=max_stake_pct,
            drawdown_factor=dd_factor,
        )
        if stake <= 0:
            return None

        return {
            "model_prob": model_p,
            "blended_prob": blended_p,
            "fair_prob": fair_p,
            "odds": odds,
            "edge": edge,
            "ev": ev,
            "n_agree": n_agree,
            "stake_pct": stake,
            "confidence": "high" if n_agree >= 3 and edge > 0.04 else
                          "medium" if n_agree >= 2 and edge > 0.025 else "low",
        }

    def generate_recommendations(self) -> list[dict]:
        """Fetch live odds, generate predictions, return bet recommendations.

        Returns list of recommendation dicts, each containing:
          - fixture info (home, away, kickoff)
          - market (ou25 or btts)
          - side (over/under or yes/no)
          - model_prob, fair_prob, blended_prob, edge, ev
          - stake_pct, confidence, best_bookmaker
        """
        if self._ou_models is None or self._btts_models is None:
            raise RuntimeError("Models not trained. Call train() first.")

        self._log("Fetching live odds...")
        matches = fetch_epl_odds(force_refresh=True)
        self._log(f"Found {len(matches)} upcoming fixtures")

        if not matches:
            return []

        # Build fixture features from historical data
        df = self._full_df
        latest_season = df["SeasonIndex"].max()
        latest_df = df[df["SeasonIndex"] == latest_season].copy()

        recommendations = []

        for match in matches:
            home_mapped, away_mapped = match_to_our_teams(match, self._our_teams)
            if home_mapped is None or away_mapped is None:
                self._log(f"Skipping {match['home_team']} vs {match['away_team']} "
                          f"(team mapping failed)")
                continue

            # Find most recent row for this fixture to get features
            fixture_mask = (
                (latest_df["Home_Team"] == home_mapped) &
                (latest_df["Away_Team"] == away_mapped)
            )
            if fixture_mask.sum() == 0:
                # Try to build features from most recent home/away rows
                home_rows = latest_df[latest_df["Home_Team"] == home_mapped]
                away_rows = latest_df[latest_df["Away_Team"] == away_mapped]
                if len(home_rows) == 0 or len(away_rows) == 0:
                    self._log(f"Skipping {home_mapped} vs {away_mapped} "
                              f"(no recent data)")
                    continue
                fixture_row = home_rows.iloc[-1].copy()
                # Override away-specific features from away team's latest row
                away_row = away_rows.iloc[-1]
                for col in fixture_row.index:
                    if col.startswith("Away_"):
                        fixture_row[col] = away_row[col]
            else:
                fixture_row = latest_df[fixture_mask].iloc[-1]

            kickoff = match.get("commence_time", "")

            # ── O/U 2.5 ──
            best_ou = get_best_odds(match)
            if best_ou:
                ou_pred = self._predict_ou(fixture_row)

                # Fair probabilities from market
                raw_over_imp = 1.0 / best_ou["best_over"]
                raw_under_imp = 1.0 / best_ou["best_under"]
                overround = raw_over_imp + raw_under_imp
                fair_over = raw_over_imp / overround
                fair_under = raw_under_imp / overround

                for side, model_p, fair_p, odds, book in [
                    ("over", ou_pred["ensemble"], fair_over,
                     best_ou["best_over"], best_ou["best_over_book"]),
                    ("under", 1 - ou_pred["ensemble"], fair_under,
                     best_ou["best_under"], best_ou["best_under_book"]),
                ]:
                    if side == "over":
                        fair_threshold = fair_over
                        per_model = ou_pred["per_model"]
                    else:
                        fair_threshold = fair_under
                        per_model = 1 - ou_pred["per_model"]

                    bet = self._evaluate_bet(
                        model_p, fair_p, odds, per_model, fair_threshold,
                        self.ou_config,
                    )
                    if bet:
                        bet.update({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": "ou25",
                            "side": side,
                            "best_bookmaker": book,
                            "n_books": best_ou["n_books"],
                            "per_model_probs": {
                                "xgb": ou_pred["xgb"],
                                "lgb": ou_pred["lgb"],
                                "lr": ou_pred["lr"],
                                "dc": ou_pred["dc"],
                            },
                        })
                        recommendations.append(bet)

            # ── BTTS ──
            best_btts = get_best_btts_odds(match)
            if best_btts:
                btts_pred = self._predict_btts(fixture_row)

                raw_yes_imp = 1.0 / best_btts["best_yes"]
                raw_no_imp = 1.0 / best_btts["best_no"]
                overround = raw_yes_imp + raw_no_imp
                fair_yes = raw_yes_imp / overround
                fair_no = raw_no_imp / overround

                btts_weights = np.array([0.20, 0.20, 0.30, 0.30])

                for side, model_p, fair_p, odds, book in [
                    ("yes", btts_pred["ensemble"], fair_yes,
                     best_btts["best_yes"], best_btts["best_yes_book"]),
                    ("no", 1 - btts_pred["ensemble"], fair_no,
                     best_btts["best_no"], best_btts["best_no_book"]),
                ]:
                    if side == "yes":
                        fair_threshold = fair_yes
                        per_model = btts_pred["per_model"]
                    else:
                        fair_threshold = fair_no
                        per_model = 1 - btts_pred["per_model"]

                    bet = self._evaluate_bet(
                        model_p, fair_p, odds, per_model, fair_threshold,
                        self.btts_config,
                    )
                    if bet:
                        bet.update({
                            "home_team": home_mapped,
                            "away_team": away_mapped,
                            "kickoff": kickoff,
                            "market": "btts",
                            "side": side,
                            "best_bookmaker": book,
                            "n_books": best_btts["n_books"],
                            "per_model_probs": {
                                "xgb": btts_pred["xgb"],
                                "lgb": btts_pred["lgb"],
                                "lr": btts_pred["lr"],
                                "dc": btts_pred["dc"],
                            },
                        })
                        recommendations.append(bet)

        # Sort by edge descending
        recommendations.sort(key=lambda x: x["edge"], reverse=True)
        self._log(f"Generated {len(recommendations)} recommendations")
        return recommendations


def run_predictions(verbose: bool = True) -> list[dict]:
    """Convenience function: load, train, predict in one call."""
    predictor = LivePredictor(verbose=verbose)
    predictor.load_data()
    predictor.train()
    return predictor.generate_recommendations()


if __name__ == "__main__":
    recs = run_predictions()
    if not recs:
        print("\nNo recommendations (no upcoming fixtures or no edges found)")
    else:
        print(f"\n{'='*80}")
        print(f"{'MATCH':<35} {'MKT':<6} {'SIDE':<6} {'EDGE':>6} "
              f"{'ODDS':>5} {'STAKE':>6} {'CONF':<6} {'BOOK':<15}")
        print(f"{'='*80}")
        for r in recs:
            fixture = f"{r['home_team']} v {r['away_team']}"
            if len(fixture) > 34:
                fixture = fixture[:31] + "..."
            print(f"{fixture:<35} {r['market']:<6} {r['side']:<6} "
                  f"{r['edge']:>+5.1%} {r['odds']:>5.2f} "
                  f"{r['stake_pct']:>5.1%} {r['confidence']:<6} "
                  f"{r['best_bookmaker']:<15}")

        # Save to dashboard database
        from dashboard import save_recommendations
        n_saved = save_recommendations(recs)
        print(f"\nSaved {n_saved} new recommendations to dashboard DB")
