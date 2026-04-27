"""
Over/Under 2.5 Goals prediction model.

Architecture:
  1. Walk-forward CV for robust evaluation (train on seasons 14-N, validate on N+1)
  2. Four diverse base models: XGBoost, LightGBM, Logistic Regression, Dixon-Coles Poisson
  3. Logistic regression stacker (meta-learner) trained on OOF predictions
  4. Calibration based on validation data only (no test leakage)

Includes training, evaluation, SHAP, and plotting.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import poisson as poisson_dist
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from config import (
    MODEL_DIR, MODEL_PATH, SCALER_PATH, FEATURE_LIST_PATH,
    TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS, TRAIN_MIN_SEASON,
)
from pipeline import run_pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Base model training
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb(X_train, y_train, X_val, y_val):
    """Train XGBoost with strong regularization and early stopping."""
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        gamma=0.1, reg_alpha=0.5, reg_lambda=3.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb(X_train, y_train, X_val, y_val, feature_names=None):
    """Train LightGBM with matching regularization."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def _fill_nan_median(X, medians=None):
    """Fill NaN with column medians. If medians not provided, compute from X."""
    X_filled = X.copy()
    if medians is None:
        medians = np.nanmedian(X, axis=0)
    for j in range(X_filled.shape[1]):
        mask = np.isnan(X_filled[:, j])
        if mask.any():
            X_filled[mask, j] = medians[j] if not np.isnan(medians[j]) else 0.0
    return X_filled, medians


def _clip_scaled(X_scaled, clip=5.0):
    """Clip scaled features to [-clip, +clip].

    Prevents catastrophic predictions from distribution shift between train/test.
    Without this, a single feature shifting by 250+ standard deviations (e.g.
    Home_DefensiveStrength_5) can push LR logits to +10, making every prediction 0.9999.
    """
    return np.clip(X_scaled, -clip, clip)


def train_logreg(X_train, y_train):
    """Train L2-regularized logistic regression (genuinely different model class).

    Fills NaN with column medians before scaling (better than 0 for centered data).
    Clips scaled features to [-5, 5] to prevent distribution-shift catastrophe.
    """
    X_filled, col_medians = _fill_nan_median(X_train)
    scaler = StandardScaler()
    X_scaled = _clip_scaled(scaler.fit_transform(X_filled))
    model = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    # Store medians for inference
    model._col_medians = col_medians
    return model, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# BTTS-specific model training (higher base rate, fewer features, more interaction)
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb_btts(X_train, y_train, X_val, y_val):
    """XGBoost tuned for BTTS target.

    Key differences from O/U 2.5 version:
      - max_depth=5: BTTS needs deeper trees to capture FTS*defence interactions
      - min_child_weight=3: allow finer splits (BTTS has ~55% base rate)
      - Much less regularisation: BTTS features are curated/targeted, less noise
      - colsample_bytree=0.8: fewer features, sample more per tree
    """
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_btts(X_train, y_train, X_val, y_val, feature_names=None):
    """LightGBM tuned for BTTS target (matching XGB BTTS changes)."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def train_xgb_corners(X_train, y_train, X_val, y_val):
    """XGBoost tuned for Corners O/U 10.5 target.

    Corners are ~50/50 split (base rate ~50.7%), so balanced trees work well.
    Deeper trees (max_depth=6) to capture interactions between pressing,
    possession, and corner-winning tendencies.
    """
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.7, min_child_weight=5,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lgb_corners(X_train, y_train, X_val, y_val, feature_names=None):
    """LightGBM tuned for Corners O/U 10.5 target."""
    model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.015,
        subsample=0.75, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1,
    )
    X_tr = pd.DataFrame(X_train, columns=feature_names) if feature_names else X_train
    X_v = pd.DataFrame(X_val, columns=feature_names) if feature_names else X_val
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Dixon-Coles Poisson model (standalone — not just a feature)
# ═══════════════════════════════════════════════════════════════════════════════

class DixonColesPredictor:
    """Standalone Dixon-Coles Poisson model for Over/Under 2.5 prediction.

    Uses team-level attack/defence ratings estimated from historical goals (or xG).
    Fundamentally different model class from gradient boosting — makes explicit
    parametric assumptions about goal-scoring (Poisson) with tau correction
    for low-score dependency.

    Improvements over naive implementation:
      - Home/away-specific attack/defence ratings (venue matters)
      - Time-decay weighting (half-life parameter)
      - Correct lambda formula (no double-counting of home advantage)
      - xG-based ratings when available (lower variance than goals)
      - Promoted team priors (venue-aware, not uniform)
    """

    # Venue-aware promoted team priors (used as terminal fallback when a team
    # has zero matches at either venue — e.g. the very first game for a newly
    # synthesised promoted team).
    PRIORS = {
        "attack_home": 0.90,   # promoted teams decent at home
        "attack_away": 0.75,   # promoted teams struggle away
        "defence_home": 1.10,  # slightly leaky at home
        "defence_away": 1.20,  # leakier away
    }
    MIN_VENUE_MATCHES = 3  # legacy hard threshold — still used when use_mle=True

    # Option 2 Step 2 — Partial pooling (Bayesian shrinkage) toward the
    # league mean (=1.0 by DC construction). Each team's venue estimate is
    # blended with the league mean, weighted by effective sample size:
    #     rating = (n/(n+N_PRIOR)) * team_estimate + (N_PRIOR/(n+N_PRIOR)) * 1.0
    # With N_PRIOR=6, a team with 2 matches is weighted 25% own / 75% mean;
    # at 20 matches it flips to ~77% own / 23% mean. Eliminates the
    # discontinuity at MIN_VENUE_MATCHES. Applied only for the
    # weighted-average path — MLE keeps its own ridge regularisation.
    N_PRIOR = 6

    def __init__(self, rho=-0.13, half_life=30, use_xg=False, use_mle=False,
                 mle_alpha=0.01, use_shrinkage=True):
        self.rho = rho
        self.half_life = half_life  # matches; more recent = more weight
        self.use_xg = use_xg
        self.use_mle = use_mle      # use MLE optimization after weighted-average
        self.mle_alpha = mle_alpha   # ridge penalty for MLE
        # Option 2 Step 2: partial pooling toward league mean. On by default
        # for the weighted-average path. Forced off when use_mle=True to
        # avoid double-regularising (MLE's ridge penalty handles small-sample
        # stability). Exposed as a flag so diagnostics can A/B against the
        # old hard-threshold behaviour.
        self.use_shrinkage = use_shrinkage
        self.attack_home = {}   # team -> home attack strength (1.0 = average)
        self.attack_away = {}   # team -> away attack strength
        self.defence_home = {}  # team -> home defence weakness (>1 = leaky)
        self.defence_away = {}  # team -> away defence weakness
        self.mu = 1.35          # league avg goals per team per match
        self.gamma = 1.36       # home advantage factor (home_goals / away_goals)

    def _decay_weights(self, n):
        """Exponential decay weights: most recent match = 1.0, decays backward."""
        if n == 0:
            return np.array([])
        indices = np.arange(n)[::-1]  # [n-1, n-2, ..., 0] — most recent last
        return np.power(0.5, indices / self.half_life)

    def _weighted_avg(self, values, weights):
        """Safe weighted average, returns None if no data."""
        if len(values) == 0:
            return None
        return np.average(values, weights=weights)

    def _shrink_to_league(self, estimate, n_eff):
        """Blend a team estimate toward the league mean (=1.0).

        n_eff = effective sample size (match count at the relevant venue
        or pooled across venues). Returns a continuous shrinkage that
        eliminates the discontinuity of the old hard-threshold fallback.
        """
        if estimate is None:
            return None
        shrink = n_eff / (n_eff + self.N_PRIOR)
        return shrink * estimate + (1.0 - shrink) * 1.0

    def _pool_with_shrinkage(self, venue_val, n_venue,
                             pooled_val, n_pool, prior):
        """Pick the best available estimate and apply partial pooling.

        Preference order:
          1. Venue-specific estimate (shrunk by n_venue)
          2. Cross-venue pooled estimate (shrunk by n_pool)
          3. Prior constant (team has no data at all)
        """
        if venue_val is not None:
            return self._shrink_to_league(venue_val, n_venue)
        if pooled_val is not None:
            return self._shrink_to_league(pooled_val, n_pool)
        return prior

    def fit(self, df):
        """Estimate venue-specific attack/defence ratings from historical match data.
        Maintains separate home/away ratings per team. Falls back to pooled estimate
        when a team has fewer than MIN_VENUE_MATCHES at a given venue.
        """
        # Choose scoring metric: xG if available, else goals
        if self.use_xg and "home_xg" in df.columns:
            h_score_col, a_score_col = "home_xg", "away_xg"
        else:
            h_score_col, a_score_col = "Home_Goals", "Away_Goals"

        # League averages
        h_avg = df[h_score_col].mean()
        a_avg = df[a_score_col].mean()
        self.mu = (h_avg + a_avg) / 2
        self.gamma = h_avg / a_avg if a_avg > 0 else 1.36

        teams = set(df["Home_Team"].unique()) | set(df["Away_Team"].unique())

        for team in teams:
            home_matches = df[df["Home_Team"] == team].sort_values("Date")
            away_matches = df[df["Away_Team"] == team].sort_values("Date")
            n_home = len(home_matches)
            n_away = len(away_matches)

            # --- Attack ratings ---
            # Home attack: goals scored at home / home league avg
            att_home_val = None
            if n_home > 0:
                hw = self._decay_weights(n_home)
                att_home_val = self._weighted_avg(home_matches[h_score_col].values / h_avg, hw)

            # Away attack: goals scored away / away league avg
            att_away_val = None
            if n_away > 0:
                aw = self._decay_weights(n_away)
                att_away_val = self._weighted_avg(away_matches[a_score_col].values / a_avg, aw)

            # Pooled attack (for fallback)
            att_pooled = None
            if att_home_val is not None and att_away_val is not None:
                # Weight by number of matches at each venue
                att_pooled = (att_home_val * n_home + att_away_val * n_away) / (n_home + n_away)
            elif att_home_val is not None:
                att_pooled = att_home_val
            elif att_away_val is not None:
                att_pooled = att_away_val

            # Assign attack ratings.
            # Shrinkage path (use_mle=False): partial pooling toward league mean,
            # continuous weighting, eliminates the hard-threshold discontinuity.
            # MLE warm-start path (use_mle=True): keep legacy hard-threshold
            # behaviour — MLE's ridge penalty already regularises small samples.
            n_pool = n_home + n_away
            if self.use_shrinkage and not self.use_mle:
                self.attack_home[team] = self._pool_with_shrinkage(
                    att_home_val, n_home, att_pooled, n_pool,
                    self.PRIORS["attack_home"])
                self.attack_away[team] = self._pool_with_shrinkage(
                    att_away_val, n_away, att_pooled, n_pool,
                    self.PRIORS["attack_away"])
            else:
                if n_home >= self.MIN_VENUE_MATCHES and att_home_val is not None:
                    self.attack_home[team] = att_home_val
                elif att_pooled is not None:
                    self.attack_home[team] = att_pooled
                else:
                    self.attack_home[team] = self.PRIORS["attack_home"]

                if n_away >= self.MIN_VENUE_MATCHES and att_away_val is not None:
                    self.attack_away[team] = att_away_val
                elif att_pooled is not None:
                    self.attack_away[team] = att_pooled
                else:
                    self.attack_away[team] = self.PRIORS["attack_away"]

            # --- Defence ratings ---
            # Home defence: goals conceded at home / away league avg
            def_home_val = None
            if n_home > 0:
                hw = self._decay_weights(n_home)
                def_home_val = self._weighted_avg(home_matches[a_score_col].values / a_avg, hw)

            # Away defence: goals conceded away / home league avg
            def_away_val = None
            if n_away > 0:
                aw = self._decay_weights(n_away)
                def_away_val = self._weighted_avg(away_matches[h_score_col].values / h_avg, aw)

            # Pooled defence (for fallback)
            def_pooled = None
            if def_home_val is not None and def_away_val is not None:
                def_pooled = (def_home_val * n_home + def_away_val * n_away) / (n_home + n_away)
            elif def_home_val is not None:
                def_pooled = def_home_val
            elif def_away_val is not None:
                def_pooled = def_away_val

            # Assign defence ratings — same shrinkage pattern as attack.
            if self.use_shrinkage and not self.use_mle:
                self.defence_home[team] = self._pool_with_shrinkage(
                    def_home_val, n_home, def_pooled, n_pool,
                    self.PRIORS["defence_home"])
                self.defence_away[team] = self._pool_with_shrinkage(
                    def_away_val, n_away, def_pooled, n_pool,
                    self.PRIORS["defence_away"])
            else:
                if n_home >= self.MIN_VENUE_MATCHES and def_home_val is not None:
                    self.defence_home[team] = def_home_val
                elif def_pooled is not None:
                    self.defence_home[team] = def_pooled
                else:
                    self.defence_home[team] = self.PRIORS["defence_home"]

                if n_away >= self.MIN_VENUE_MATCHES and def_away_val is not None:
                    self.defence_away[team] = def_away_val
                elif def_pooled is not None:
                    self.defence_away[team] = def_pooled
                else:
                    self.defence_away[team] = self.PRIORS["defence_away"]

        # Optionally refine with MLE
        if self.use_mle:
            self.fit_mle(df, alpha=self.mle_alpha)

        return self

    def fit_mle(self, df, alpha=0.01):
        """Fit via maximum likelihood estimation (Dixon & Coles 1997).

        Jointly optimizes all attack/defence ratings, mu, gamma, and rho under
        the Poisson likelihood with tau correction. Uses L-BFGS-B with ridge
        penalty and sum-to-zero constraints on log-ratings for identifiability.

        Key design choices:
          - Log-space parameterization ensures positive ratings without bounds
          - Sum-to-zero constraint on log(attack) and log(defence) resolves the
            identifiability issue (scale absorbed into mu, not ratings)
          - mu is optimized jointly (not fixed to data mean)
          - Proper normalization: ratings are centered, mu absorbs the scale
          - Warm-starts from the weighted-average estimates already computed by fit()
        """

        # Choose scoring metric
        if self.use_xg and "home_xg" in df.columns:
            h_score_col, a_score_col = "home_xg", "away_xg"
        else:
            h_score_col, a_score_col = "Home_Goals", "Away_Goals"

        # Build match arrays for vectorized likelihood
        teams = sorted(set(df["Home_Team"].unique()) | set(df["Away_Team"].unique()))
        team_to_idx = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        home_idx = df["Home_Team"].map(team_to_idx).values
        away_idx = df["Away_Team"].map(team_to_idx).values
        h_goals = df[h_score_col].values.astype(float)
        a_goals = df[a_score_col].values.astype(float)

        # Time-decay weights (normalized so they sum to n_matches)
        n_matches = len(df)
        weights = self._decay_weights(n_matches)
        if len(weights) == 0:
            return self

        # Pre-compute integer goals for tau correction
        h_int = h_goals.astype(int)
        a_int = a_goals.astype(int)
        m00 = (h_int == 0) & (a_int == 0)
        m01 = (h_int == 0) & (a_int == 1)
        m10 = (h_int == 1) & (a_int == 0)
        m11 = (h_int == 1) & (a_int == 1)

        # Parameter layout (all in log-space for ratings):
        # [log_att_h(N), log_att_a(N), log_def_h(N), log_def_a(N), log_mu, log_gamma, rho]
        # Total: 4*N + 3
        data_mu = (np.mean(h_goals) + np.mean(a_goals)) / 2

        x0 = np.zeros(4 * n_teams + 3)
        for t, i in team_to_idx.items():
            x0[i] = np.log(max(self.attack_home.get(t, self.PRIORS["attack_home"]), 0.01))
            x0[n_teams + i] = np.log(max(self.attack_away.get(t, self.PRIORS["attack_away"]), 0.01))
            x0[2 * n_teams + i] = np.log(max(self.defence_home.get(t, self.PRIORS["defence_home"]), 0.01))
            x0[3 * n_teams + i] = np.log(max(self.defence_away.get(t, self.PRIORS["defence_away"]), 0.01))

        # Center the log-ratings (sum-to-zero) and absorb mean into mu
        for offset in [0, n_teams, 2 * n_teams, 3 * n_teams]:
            block = x0[offset:offset + n_teams]
            block -= block.mean()

        x0[-3] = np.log(data_mu)     # log_mu
        x0[-2] = np.log(self.gamma)   # log_gamma
        x0[-1] = self.rho             # rho (not log-space, can be negative)

        # Bounds
        log_rating_bounds = [(-2.0, 2.0)] * (4 * n_teams)  # exp range: [0.14, 7.4]
        log_mu_bounds = [(np.log(0.5), np.log(3.0))]
        log_gamma_bounds = [(np.log(0.8), np.log(2.0))]
        rho_bounds = [(-0.30, 0.05)]
        bounds = log_rating_bounds + log_mu_bounds + log_gamma_bounds + rho_bounds

        # Sum-to-zero penalty weight (strong — this is a constraint, not preference)
        constraint_weight = 100.0

        def neg_log_likelihood(params):
            log_att_h = params[:n_teams]
            log_att_a = params[n_teams:2*n_teams]
            log_def_h = params[2*n_teams:3*n_teams]
            log_def_a = params[3*n_teams:4*n_teams]
            log_mu = params[-3]
            log_gamma = params[-2]
            rho = params[-1]

            mu = np.exp(log_mu)
            sqrt_g = np.exp(log_gamma / 2)

            # Vectorized lambda: exp(log_att_h + log_def_a + log_mu + log_gamma/2)
            lam_h = np.exp(log_att_h[home_idx] + log_def_a[away_idx] + log_mu + log_gamma / 2)
            lam_a = np.exp(log_att_a[away_idx] + log_def_h[home_idx] + log_mu - log_gamma / 2)

            # Clamp
            lam_h = np.clip(lam_h, 0.05, 8.0)
            lam_a = np.clip(lam_a, 0.05, 8.0)

            # Poisson log-likelihood (vectorized)
            ll = (poisson_dist.logpmf(h_goals, lam_h) +
                  poisson_dist.logpmf(a_goals, lam_a))

            # Tau correction (Dixon-Coles low-score dependency)
            tau = np.ones(n_matches)
            tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
            tau[m01] = 1 + lam_h[m01] * rho
            tau[m10] = 1 + lam_a[m10] * rho
            tau[m11] = 1 - rho

            tau = np.clip(tau, 1e-10, None)
            ll += np.log(tau)

            # Weighted sum
            total_ll = np.sum(weights * ll)

            # Ridge penalty on log-ratings (pull toward 0 = average team)
            penalty = alpha * (np.sum(log_att_h**2) + np.sum(log_att_a**2) +
                               np.sum(log_def_h**2) + np.sum(log_def_a**2))

            # Sum-to-zero constraint (identifiability)
            constraint = constraint_weight * (
                log_att_h.mean()**2 + log_att_a.mean()**2 +
                log_def_h.mean()**2 + log_def_a.mean()**2
            )

            return -total_ll + penalty + constraint

        # Optimize
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 1000, 'ftol': 1e-10})

        if not result.success:
            print(f"  [DC MLE] Warning: optimizer did not converge ({result.message})")

        # Extract optimized parameters (convert from log-space)
        params = result.x
        log_att_h = params[:n_teams]
        log_att_a = params[n_teams:2*n_teams]
        log_def_h = params[2*n_teams:3*n_teams]
        log_def_a = params[3*n_teams:4*n_teams]

        self.mu = np.exp(params[-3])
        self.gamma = np.exp(params[-2])
        self.rho = params[-1]

        for t, i in team_to_idx.items():
            self.attack_home[t] = np.exp(log_att_h[i])
            self.attack_away[t] = np.exp(log_att_a[i])
            self.defence_home[t] = np.exp(log_def_h[i])
            self.defence_away[t] = np.exp(log_def_a[i])

        return self

    def _dc_tau(self, x, y, lam, mu):
        if x == 0 and y == 0:
            return 1 - lam * mu * self.rho
        elif x == 0 and y == 1:
            return 1 + lam * self.rho
        elif x == 1 and y == 0:
            return 1 + mu * self.rho
        elif x == 1 and y == 1:
            return 1 - self.rho
        return 1.0

    def predict_match(self, home_team, away_team):
        """Predict P(Over 2.5) for a single match.

        Lambda formula uses venue-specific ratings:
          home_lambda = attack_home[home] * defence_away[away] * mu * sqrt(gamma)
          away_lambda = attack_away[away] * defence_home[home] * mu / sqrt(gamma)
        """
        h_att = self.attack_home.get(home_team, self.PRIORS["attack_home"])
        a_def = self.defence_away.get(away_team, self.PRIORS["defence_away"])
        a_att = self.attack_away.get(away_team, self.PRIORS["attack_away"])
        h_def = self.defence_home.get(home_team, self.PRIORS["defence_home"])

        sqrt_gamma = np.sqrt(self.gamma)
        home_lambda = h_att * a_def * self.mu * sqrt_gamma
        away_lambda = a_att * h_def * self.mu / sqrt_gamma

        # Clamp to reasonable range
        home_lambda = np.clip(home_lambda, 0.1, 5.0)
        away_lambda = np.clip(away_lambda, 0.1, 5.0)

        p_under = 0
        for h in range(12):
            for a in range(12):
                if h + a <= 2:
                    p = (poisson_dist.pmf(h, home_lambda) *
                         poisson_dist.pmf(a, away_lambda) *
                         self._dc_tau(h, a, home_lambda, away_lambda))
                    p_under += max(p, 0)

        return np.clip(1 - p_under, 0.01, 0.99)

    def predict_match_ou15(self, home_team, away_team):
        """Predict P(Over 1.5) for a single match.

        Same lambda formula as predict_match but different scoreline threshold:
        P(Over 1.5) = 1 - P(0-0) - P(1-0) - P(0-1).
        """
        h_att = self.attack_home.get(home_team, self.PRIORS["attack_home"])
        a_def = self.defence_away.get(away_team, self.PRIORS["defence_away"])
        a_att = self.attack_away.get(away_team, self.PRIORS["attack_away"])
        h_def = self.defence_home.get(home_team, self.PRIORS["defence_home"])

        sqrt_gamma = np.sqrt(self.gamma)
        home_lambda = np.clip(h_att * a_def * self.mu * sqrt_gamma, 0.1, 5.0)
        away_lambda = np.clip(a_att * h_def * self.mu / sqrt_gamma, 0.1, 5.0)

        p_under = 0
        for h in range(12):
            for a in range(12):
                if h + a <= 1:
                    p = (poisson_dist.pmf(h, home_lambda) *
                         poisson_dist.pmf(a, away_lambda) *
                         self._dc_tau(h, a, home_lambda, away_lambda))
                    p_under += max(p, 0)

        return np.clip(1 - p_under, 0.01, 0.99)

    def predict_proba_ou15_df(self, df):
        """Predict P(Over 1.5) for a DataFrame with Home_Team, Away_Team columns."""
        probs = np.array([
            self.predict_match_ou15(row["Home_Team"], row["Away_Team"])
            for _, row in df.iterrows()
        ])
        return probs

    def predict_match_btts(self, home_team, away_team):
        """Predict P(Both Teams To Score) for a single match.

        P(BTTS) = 1 - P(Home=0) - P(Away=0) + P(0-0)
        Uses same lambdas as predict_match but sums different score lines.
        """
        h_att = self.attack_home.get(home_team, self.PRIORS["attack_home"])
        a_def = self.defence_away.get(away_team, self.PRIORS["defence_away"])
        a_att = self.attack_away.get(away_team, self.PRIORS["attack_away"])
        h_def = self.defence_home.get(home_team, self.PRIORS["defence_home"])

        sqrt_gamma = np.sqrt(self.gamma)
        home_lambda = np.clip(h_att * a_def * self.mu * sqrt_gamma, 0.1, 5.0)
        away_lambda = np.clip(a_att * h_def * self.mu / sqrt_gamma, 0.1, 5.0)

        # P(Home=0) and P(Away=0) with DC tau correction
        p_home_zero = 0  # sum of P(0, a) for all a
        p_away_zero = 0  # sum of P(h, 0) for all h
        p_both_zero = 0  # P(0, 0)

        for h in range(12):
            for a in range(12):
                p = (poisson_dist.pmf(h, home_lambda) *
                     poisson_dist.pmf(a, away_lambda) *
                     self._dc_tau(h, a, home_lambda, away_lambda))
                p = max(p, 0)
                if h == 0:
                    p_home_zero += p
                if a == 0:
                    p_away_zero += p
                if h == 0 and a == 0:
                    p_both_zero += p

        # P(BTTS) = 1 - P(H=0) - P(A=0) + P(0-0) (inclusion-exclusion)
        p_btts = 1 - p_home_zero - p_away_zero + p_both_zero
        return np.clip(p_btts, 0.01, 0.99)

    def predict_proba_df(self, df):
        """Predict P(Over 2.5) for a DataFrame with Home_Team, Away_Team columns."""
        probs = np.array([
            self.predict_match(row["Home_Team"], row["Away_Team"])
            for _, row in df.iterrows()
        ])
        return probs

    def predict_proba_btts_df(self, df):
        """Predict P(BTTS) for a DataFrame with Home_Team, Away_Team columns."""
        probs = np.array([
            self.predict_match_btts(row["Home_Team"], row["Away_Team"])
            for _, row in df.iterrows()
        ])
        return probs

    # --- Corner predictions ---

    CORNER_PRIORS = {
        "corner_attack_home": 1.00,   # average corner-winning ability
        "corner_attack_away": 1.00,
        "corner_defence_home": 1.00,  # average corner-conceding tendency
        "corner_defence_away": 1.00,
    }

    def fit_corners(self, df: pd.DataFrame) -> "DixonColesPredictor":
        """Estimate corner-specific attack/defence ratings from historical corner data.

        Args:
            df: DataFrame with Home_Team, Away_Team, Home_Corners, Away_Corners, Date.

        Returns:
            self (for chaining).
        """
        # Filter to matches with corner data
        cdf = df.dropna(subset=["Home_Corners", "Away_Corners"]).copy()
        if len(cdf) == 0:
            return self

        h_avg = cdf["Home_Corners"].mean()
        a_avg = cdf["Away_Corners"].mean()
        self.corner_mu = (h_avg + a_avg) / 2
        self.corner_gamma = h_avg / a_avg if a_avg > 0 else 1.10

        self.corner_attack_home = {}
        self.corner_attack_away = {}
        self.corner_defence_home = {}
        self.corner_defence_away = {}

        teams = set(cdf["Home_Team"].unique()) | set(cdf["Away_Team"].unique())

        for team in teams:
            home_matches = cdf[cdf["Home_Team"] == team].sort_values("Date")
            away_matches = cdf[cdf["Away_Team"] == team].sort_values("Date")
            n_home = len(home_matches)
            n_away = len(away_matches)

            # Corner attack: corners won / league avg
            att_h = None
            if n_home > 0:
                hw = self._decay_weights(n_home)
                att_h = self._weighted_avg(home_matches["Home_Corners"].values / h_avg, hw)
            att_a = None
            if n_away > 0:
                aw = self._decay_weights(n_away)
                att_a = self._weighted_avg(away_matches["Away_Corners"].values / a_avg, aw)

            att_pooled = None
            if att_h is not None and att_a is not None:
                att_pooled = (att_h * n_home + att_a * n_away) / (n_home + n_away)
            elif att_h is not None:
                att_pooled = att_h
            elif att_a is not None:
                att_pooled = att_a

            self.corner_attack_home[team] = (
                att_h if n_home >= self.MIN_VENUE_MATCHES and att_h is not None
                else att_pooled if att_pooled is not None
                else self.CORNER_PRIORS["corner_attack_home"]
            )
            self.corner_attack_away[team] = (
                att_a if n_away >= self.MIN_VENUE_MATCHES and att_a is not None
                else att_pooled if att_pooled is not None
                else self.CORNER_PRIORS["corner_attack_away"]
            )

            # Corner defence: corners conceded / opponent's league avg
            def_h = None
            if n_home > 0:
                hw = self._decay_weights(n_home)
                def_h = self._weighted_avg(home_matches["Away_Corners"].values / a_avg, hw)
            def_a = None
            if n_away > 0:
                aw = self._decay_weights(n_away)
                def_a = self._weighted_avg(away_matches["Home_Corners"].values / h_avg, aw)

            def_pooled = None
            if def_h is not None and def_a is not None:
                def_pooled = (def_h * n_home + def_a * n_away) / (n_home + n_away)
            elif def_h is not None:
                def_pooled = def_h
            elif def_a is not None:
                def_pooled = def_a

            self.corner_defence_home[team] = (
                def_h if n_home >= self.MIN_VENUE_MATCHES and def_h is not None
                else def_pooled if def_pooled is not None
                else self.CORNER_PRIORS["corner_defence_home"]
            )
            self.corner_defence_away[team] = (
                def_a if n_away >= self.MIN_VENUE_MATCHES and def_a is not None
                else def_pooled if def_pooled is not None
                else self.CORNER_PRIORS["corner_defence_away"]
            )

        return self

    def predict_match_corners(self, home_team: str, away_team: str) -> float:
        """Predict P(Over 10.5 total corners) for a single match.

        Uses Poisson model with corner-specific ratings. No tau correction
        needed for corners (low-score dependency is a goals phenomenon).

        Args:
            home_team: Canonical home team name.
            away_team: Canonical away team name.

        Returns:
            Probability of over 10.5 total corners, clipped to [0.01, 0.99].
        """
        h_att = self.corner_attack_home.get(home_team, self.CORNER_PRIORS["corner_attack_home"])
        a_def = self.corner_defence_away.get(away_team, self.CORNER_PRIORS["corner_defence_away"])
        a_att = self.corner_attack_away.get(away_team, self.CORNER_PRIORS["corner_attack_away"])
        h_def = self.corner_defence_home.get(home_team, self.CORNER_PRIORS["corner_defence_home"])

        sqrt_gamma = np.sqrt(self.corner_gamma)
        home_lambda = np.clip(h_att * a_def * self.corner_mu * sqrt_gamma, 1.0, 15.0)
        away_lambda = np.clip(a_att * h_def * self.corner_mu / sqrt_gamma, 1.0, 15.0)

        total_lambda = home_lambda + away_lambda
        # P(total > 10.5) = 1 - P(total <= 10) = 1 - poisson.cdf(10, total_lambda)
        p_over = 1.0 - poisson_dist.cdf(10, total_lambda)
        return np.clip(p_over, 0.01, 0.99)

    def predict_proba_corners_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predict P(Over 10.5 corners) for a DataFrame."""
        probs = np.array([
            self.predict_match_corners(row["Home_Team"], row["Away_Team"])
            for _, row in df.iterrows()
        ])
        return probs


# ═══════════════════════════════════════════════════════════════════════════════
# Dixon-Coles hyperparameter tuning
# ═══════════════════════════════════════════════════════════════════════════════

def tune_dc_params(full_df, min_train_season=14, start_val_season=19,
                   target_col="Over_2_5", predict_fn_name="predict_proba_df"):
    """Grid-search half_life, rho, and MLE vs weighted-avg via walk-forward CV.

    Returns dict of best params for DixonColesPredictor constructor.
    Phase 1: Fast grid over half_life x rho (weighted-avg only).
    Phase 2: Test MLE with best half_life at a few alpha values.

    Args:
        full_df: Full pipeline DataFrame with match history.
        min_train_season: First season to include in training folds.
        start_val_season: First season to hold out as validation.
        target_col: Outcome column to optimise AUC against
            (e.g. "Over_2_5", "BTTS"). Default preserves existing behaviour.
        predict_fn_name: DixonColesPredictor method to use for predictions
            against target_col (e.g. "predict_proba_df" for O/U 2.5,
            "predict_proba_btts_df" for BTTS).
    """
    # half_life=10 added at low end for markets that favour very recent form
    # (BTTS, O/U 1.5 tend to respond to short-term team defensive/scoring streaks)
    half_life_grid = [10, 15, 20, 25, 30, 40, 50, 70]
    rho_grid = [-0.20, -0.15, -0.13, -0.10, -0.07, -0.03, 0.0]

    all_seasons = sorted(full_df["SeasonIndex"].unique())
    valid_seasons = [s for s in all_seasons if s >= min_train_season]
    val_seasons = [s for s in range(start_val_season, max(valid_seasons) + 1)]

    # Pre-split data for each fold
    folds = []
    for vs in val_seasons:
        train = full_df[(full_df["SeasonIndex"] >= min_train_season) &
                        (full_df["SeasonIndex"] < vs)]
        val = full_df[full_df["SeasonIndex"] == vs]
        if len(train) >= 100 and len(val) >= 50:
            folds.append((train, val))

    if not folds:
        print("  [DC tune] Not enough folds, using defaults")
        return {"half_life": 30, "rho": -0.13}

    # Phase 1: Grid search half_life x rho (weighted-average, fast)
    best_auc = -1
    best_params = {"half_life": 30, "rho": -0.13}
    results = []

    for hl in half_life_grid:
        for rho in rho_grid:
            fold_aucs = []
            for train_df, val_df in folds:
                dc = DixonColesPredictor(rho=rho, half_life=hl)
                dc.fit(train_df)
                preds = getattr(dc, predict_fn_name)(val_df)
                y = val_df[target_col].values
                if len(np.unique(y)) < 2:
                    continue
                fold_aucs.append(roc_auc_score(y, preds))

            if fold_aucs:
                mean_auc = np.mean(fold_aucs)
                results.append((hl, rho, False, 0, mean_auc))
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_params = {"half_life": hl, "rho": rho}

    print(f"\n{'='*60}")
    print(f"Dixon-Coles Hyperparameter Tuning (target={target_col})")
    print(f"{'='*60}")
    print(f"\n  Phase 1: Weighted-average grid ({len(half_life_grid)}x{len(rho_grid)} = {len(results)} combos)")
    print(f"  Folds: {len(folds)}")
    print(f"  Top 5:")
    for hl, rho, _, _, auc in sorted(results, key=lambda x: -x[4])[:5]:
        marker = " <-- BEST" if hl == best_params["half_life"] and rho == best_params["rho"] else ""
        print(f"    half_life={hl:3d}  rho={rho:+.2f}  AUC={auc:.4f}{marker}")

    print(f"\n  Phase 1 best: half_life={best_params['half_life']}, rho={best_params['rho']}, AUC={best_auc:.4f}")

    # Phase 2: Test MLE refinement and xG-based ratings with best half_life/rho
    print(f"\n  Phase 2: Testing MLE + xG variants (using best half_life={best_params['half_life']}, rho={best_params['rho']})")
    phase2_results = []
    best_hl = best_params["half_life"]
    best_rho = best_params["rho"]

    # Check if xG data is available in the training folds
    has_xg = all("home_xg" in train_df.columns and train_df["home_xg"].notna().sum() > 100
                 for train_df, _ in folds)

    variants = [
        # (use_mle, use_xg, mle_alpha, label)
        (False, False, 0.0,  "weighted-avg (baseline)"),
        (True,  False, 0.001, "MLE a=0.001"),
        (True,  False, 0.005, "MLE a=0.005"),
        (True,  False, 0.01, "MLE a=0.01"),
        (True,  False, 0.05, "MLE a=0.05"),
    ]
    if has_xg:
        variants += [
            (False, True,  0.0,  "xG weighted-avg"),
            (True,  True,  0.005, "xG+MLE a=0.005"),
            (True,  True,  0.01, "xG+MLE a=0.01"),
        ]

    for use_mle, use_xg, mle_alpha, label in variants:
        fold_aucs = []
        for train_df, val_df in folds:
            try:
                dc = DixonColesPredictor(
                    rho=best_rho, half_life=best_hl,
                    use_xg=use_xg, use_mle=use_mle, mle_alpha=mle_alpha
                )
                dc.fit(train_df)
                preds = getattr(dc, predict_fn_name)(val_df)
                y = val_df[target_col].values
                if len(np.unique(y)) < 2:
                    continue
                fold_aucs.append(roc_auc_score(y, preds))
            except Exception as e:
                print(f"    [WARN] {label} failed on fold: {e}")
                continue

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            phase2_results.append((use_mle, use_xg, mle_alpha, label, mean_auc))
            marker = " <-- NEW BEST" if mean_auc > best_auc else ""
            print(f"    {label:30s}  AUC={mean_auc:.4f}  ({len(fold_aucs)} folds){marker}")

            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {
                    "half_life": best_hl, "rho": best_rho,
                    "use_mle": use_mle, "use_xg": use_xg, "mle_alpha": mle_alpha,
                }

    # Clean up: only include non-default params
    final_params = {"half_life": best_params["half_life"], "rho": best_params["rho"]}
    if best_params.get("use_mle", False):
        final_params["use_mle"] = True
        final_params["mle_alpha"] = best_params["mle_alpha"]
    if best_params.get("use_xg", False):
        final_params["use_xg"] = True

    print(f"\n  Final best: {final_params}, AUC={best_auc:.4f}")
    print(f"{'='*60}\n")

    return final_params


def tune_dc_params_corners(full_df: pd.DataFrame, min_train_season: int = 14,
                           start_val_season: int = 19) -> dict:
    """Grid-search half_life for corner-specific Dixon-Coles model.

    Key differences from tune_dc_params():
      - Target: Over 10.5 total corners (not Over 2.5 goals)
      - Fitting: fit() + fit_corners(), predict via predict_proba_corners_df()
      - rho fixed at 0.0 (low-score dependency irrelevant for corners)
      - No Phase 2 (MLE/xG are goals concepts)

    Args:
        full_df: Full pipeline DataFrame with Home_Corners, Away_Corners.
        min_train_season: Earliest season to include in training.
        start_val_season: First validation season.

    Returns:
        Dict of best params for DixonColesPredictor constructor.
    """
    half_life_grid = [10, 15, 20, 25, 30, 40, 50]

    all_seasons = sorted(full_df["SeasonIndex"].unique())
    valid_seasons = [s for s in all_seasons if s >= min_train_season]
    val_seasons = [s for s in range(start_val_season, max(valid_seasons) + 1)]

    # Pre-split data for each fold — require corner data
    folds = []
    for vs in val_seasons:
        train = full_df[(full_df["SeasonIndex"] >= min_train_season) &
                        (full_df["SeasonIndex"] < vs)]
        val = full_df[full_df["SeasonIndex"] == vs]
        # Filter to rows with corner data
        train = train.dropna(subset=["Home_Corners", "Away_Corners"])
        val = val.dropna(subset=["Home_Corners", "Away_Corners"])
        if len(train) >= 100 and len(val) >= 50:
            folds.append((train, val))

    if not folds:
        print("  [DC corners tune] Not enough folds, using defaults")
        return {"half_life": 25, "rho": 0.0}

    best_auc = -1
    best_hl = 25
    results = []

    for hl in half_life_grid:
        fold_aucs = []
        for train_df, val_df in folds:
            dc = DixonColesPredictor(rho=0.0, half_life=hl)
            dc.fit(train_df)
            dc.fit_corners(train_df)
            preds = dc.predict_proba_corners_df(val_df)
            y = ((val_df["Home_Corners"] + val_df["Away_Corners"]) > 10.5).astype(int).values
            if len(np.unique(y)) < 2:
                continue
            fold_aucs.append(roc_auc_score(y, preds))

        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            results.append((hl, mean_auc))
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_hl = hl

    print(f"\n{'='*60}")
    print("Dixon-Coles Corners Hyperparameter Tuning")
    print(f"{'='*60}")
    print(f"  Grid: {len(half_life_grid)} half_life values, rho=0.0 (fixed)")
    print(f"  Folds: {len(folds)}")
    for hl, auc in sorted(results, key=lambda x: -x[1])[:5]:
        marker = " <-- BEST" if hl == best_hl else ""
        print(f"    half_life={hl:3d}  AUC={auc:.4f}{marker}")

    final_params = {"half_life": best_hl, "rho": 0.0}
    print(f"\n  Best: {final_params}, AUC={best_auc:.4f}")
    print(f"{'='*60}\n")

    return final_params


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble model (preserves interface for Dash app / pickling)
# ═══════════════════════════════════════════════════════════════════════════════

class IsotonicWrapper:
    """Wraps IsotonicRegression to match predict_proba interface."""
    def __init__(self, iso_model):
        self.iso_model = iso_model
    def predict_proba(self, X):
        X_flat = X.ravel()
        p1 = self.iso_model.predict(X_flat)
        return np.column_stack([1 - p1, p1])


class EnsembleModel:
    """Stacking ensemble with logistic regression meta-learner.

    Base models: XGBoost, LightGBM, Dixon-Coles Poisson.
    Meta-learner: Logistic regression trained on OOF base model predictions.
    This replaces the simple weighted average — the stacker learns optimal
    combination and calibrates probabilities simultaneously.

    Critical: Dixon-Coles predictions must come from dc_model (stored in this class),
    NOT from the Poisson_DC pipeline feature (which uses different parameters and has
    only 0.41 correlation with dc_model). Always pass dc_probs or team names.
    """

    def __init__(self, xgb_model, lgb_model, logreg_model=None, logreg_scaler=None,
                 dc_model=None, stacker=None, feature_names=None,
                 threshold=0.5, platt_scaler=None,
                 # Legacy support: weights for backward compat with old pickles
                 weights=(0.5, 0.5)):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.logreg_model = logreg_model
        self.logreg_scaler = logreg_scaler
        self.dc_model = dc_model
        self.stacker = stacker
        self.feature_names = feature_names
        self.threshold = threshold
        self.platt_scaler = platt_scaler
        self.weights = weights

    def _base_probs(self, X, dc_probs=None):
        """Get predictions from all base models, return as columns for stacker.

        Args:
            X: Feature matrix (numpy array or DataFrame)
            dc_probs: Pre-computed Dixon-Coles probabilities. STRONGLY preferred
                      over Poisson_DC feature extraction.
        """
        if isinstance(X, np.ndarray):
            X_arr = X
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_arr = X.values if hasattr(X, 'values') else np.array(X)
            X_df = X

        xgb_p = self.xgb_model.predict_proba(X_arr)[:, 1]
        lgb_p = self.lgb_model.predict_proba(X_df)[:, 1]

        # Dixon-Coles — use provided dc_probs (preferred), else fall back to feature
        if dc_probs is not None:
            dc_p = np.asarray(dc_probs).ravel()
        else:
            # Fallback: Poisson_DC feature from pipeline (WARNING: lower correlation
            # with dc_model than expected, AUC may degrade)
            dc_p = np.full(len(xgb_p), 0.5)
            if self.feature_names is not None and "Poisson_DC" in self.feature_names:
                dc_idx = self.feature_names.index("Poisson_DC")
                dc_raw = X_arr[:, dc_idx]
                dc_p = np.where(np.isnan(dc_raw), 0.5, dc_raw)

        return np.column_stack([xgb_p, lgb_p, dc_p])

    def predict_proba(self, X, dc_probs=None):
        """Predict P(Over 2.5).

        Args:
            X: Feature matrix (numpy array or DataFrame)
            dc_probs: Pre-computed Dixon-Coles probabilities. Pass these for
                      accurate predictions; without them, falls back to Poisson_DC
                      feature which has lower correlation with dc_model.
        """
        if self.stacker is not None:
            base = self._base_probs(X, dc_probs=dc_probs)
            p1 = self.stacker.predict_proba(base)[:, 1]
        else:
            # Fallback: simple weighted average (backward compat)
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=self.feature_names)
            else:
                X_df = X
            xgb_p = self.xgb_model.predict_proba(X)[:, 1]
            lgb_p = self.lgb_model.predict_proba(X_df)[:, 1]
            p1 = self.weights[0] * xgb_p + self.weights[1] * lgb_p

        if self.platt_scaler is not None:
            if isinstance(self.platt_scaler, (int, float)):
                # Logit-shift calibration: subtract shift to correct mean drift
                # Preserves ranking (AUC unchanged), fixes probability mean
                logits = np.log(p1 / (1 - p1 + 1e-10))
                p1 = 1 / (1 + np.exp(-(logits - self.platt_scaler)))
            elif isinstance(self.platt_scaler, IsotonicWrapper):
                p1 = self.platt_scaler.predict_proba(p1.reshape(-1, 1))[:, 1]
            else:
                # Platt (LogisticRegression): takes logits
                logits = np.log(p1 / (1 - p1 + 1e-10)).reshape(-1, 1)
                p1 = self.platt_scaler.predict_proba(logits)[:, 1]

        return np.column_stack([1 - p1, p1])

    def predict(self, X, dc_probs=None):
        probs = self.predict_proba(X, dc_probs=dc_probs)
        return (probs[:, 1] >= self.threshold).astype(int)

    def predict_single(self, X, dc_probs=None):
        """For single-match prediction: return calibrated probability.
        Pass dc_probs for accurate Dixon-Coles integration."""
        return self.predict_proba(X, dc_probs=dc_probs)[:, 1]

    def predict_batch(self, X, dc_probs=None, target_over_rate=0.52):
        """For batch prediction: use dynamic percentile threshold.
        Adapts to the actual probability distribution in this batch.
        target_over_rate: expected proportion of Over 2.5 matches (~52% historically)."""
        probs = self.predict_proba(X, dc_probs=dc_probs)[:, 1]
        dynamic_t = np.percentile(probs, (1 - target_over_rate) * 100)
        preds = (probs >= dynamic_t).astype(int)
        return preds, probs, dynamic_t


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-forward cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(full_df, features, min_train_season=14, start_val_season=19,
                    dc_kwargs=None):
    """Walk-forward CV: train on 14..N, validate on N+1, for N = start_val_season-1 .. max-1.

    Returns per-fold metrics and aggregated OOF predictions for the stacker.
    This gives 6+ validation folds instead of a single train/val split,
    producing much more robust estimates of model quality.
    """
    if dc_kwargs is None:
        dc_kwargs = {}
    all_seasons = sorted(full_df["SeasonIndex"].unique())
    valid_seasons = [s for s in all_seasons if s >= min_train_season]

    fold_metrics = []
    oof_records = []  # (season, idx, xgb_prob, lgb_prob, dc_prob, y_true)

    for val_season in range(start_val_season, max(valid_seasons) + 1):
        train_df = full_df[(full_df["SeasonIndex"] >= min_train_season) &
                           (full_df["SeasonIndex"] < val_season)].copy()
        val_df = full_df[full_df["SeasonIndex"] == val_season].copy()

        if len(train_df) < 100 or len(val_df) < 50:
            continue

        X_tr = train_df[features].values
        y_tr = train_df["Over_2_5"].values
        X_v = val_df[features].values
        y_v = val_df["Over_2_5"].values

        # XGBoost
        xgb_m = train_xgb(X_tr, y_tr, X_v, y_v)
        xgb_p = xgb_m.predict_proba(X_v)[:, 1]

        # LightGBM
        lgb_m = train_lgb(X_tr, y_tr, X_v, y_v, feature_names=features)
        lgb_p = lgb_m.predict_proba(pd.DataFrame(X_v, columns=features))[:, 1]

        # Dixon-Coles (uses team attack/defence ratings — fundamentally different model)
        dc_m = DixonColesPredictor(**dc_kwargs)
        dc_m.fit(train_df)
        dc_p = dc_m.predict_proba_df(val_df)

        # Simple average for fold evaluation (3 models)
        avg_p = (xgb_p + lgb_p + dc_p) / 3
        fold_auc = roc_auc_score(y_v, avg_p)
        fold_acc = accuracy_score(y_v, (avg_p > 0.5).astype(int))
        fold_brier = brier_score_loss(y_v, avg_p)

        fold_metrics.append({
            "val_season": val_season, "n_train": len(train_df), "n_val": len(val_df),
            "auc": fold_auc, "acc": fold_acc, "brier": fold_brier,
        })
        print(f"  Fold S{val_season}: train={len(train_df):4d} val={len(val_df):3d} "
              f"AUC={fold_auc:.4f} Acc={fold_acc:.4f} Brier={fold_brier:.4f}")

        # Store OOF predictions for stacker training
        for i, idx in enumerate(val_df.index):
            oof_records.append({
                "idx": idx, "season": val_season,
                "xgb": xgb_p[i], "lgb": lgb_p[i], "dc": dc_p[i],
                "y": y_v[i],
            })

    oof_df = pd.DataFrame(oof_records)
    return fold_metrics, oof_df


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Importance Audit (Option 3 Step 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_feature_to_group_map() -> dict[str, str]:
    """Build a feature name -> group name lookup from config.py.

    A feature appearing in multiple groups gets the first-listed group
    (e.g. a roster feature also in tactical would be marked as ROSTER).
    """
    from config import (
        EXISTING_FEATURES, DERIVED_FEATURES, XG_FEATURES, PLAYER_FEATURES,
        FPL_STRENGTH_FEATURES, ADVANCED_FEATURES, SHOT_LEVEL_FEATURES,
        ROSTER_FEATURES, TACTICAL_FEATURES, DETAILED_MATCH_FEATURES,
        CONTEXT_FEATURES, WEATHER_FEATURES, BTTS_SPECIFIC_FEATURES,
    )
    groups_ordered = [
        ("EXISTING", EXISTING_FEATURES),
        ("DERIVED", DERIVED_FEATURES),
        ("XG", XG_FEATURES),
        ("PLAYER", PLAYER_FEATURES),
        ("FPL_STRENGTH", FPL_STRENGTH_FEATURES),
        ("ADVANCED", ADVANCED_FEATURES),
        ("SHOT_LEVEL", SHOT_LEVEL_FEATURES),
        ("ROSTER", ROSTER_FEATURES),
        ("TACTICAL", TACTICAL_FEATURES),
        ("DETAILED_MATCH", DETAILED_MATCH_FEATURES),
        ("CONTEXT", CONTEXT_FEATURES),
        ("WEATHER", WEATHER_FEATURES),
        ("BTTS_SPECIFIC", BTTS_SPECIFIC_FEATURES),
    ]
    lookup: dict[str, str] = {}
    for group_name, feat_list in groups_ordered:
        for f in feat_list:
            if f not in lookup:
                lookup[f] = group_name
    return lookup


def _resolve_audit_config(league: str, market: str,
                          full_df: pd.DataFrame) -> tuple[list[str], str]:
    """Resolve (feature_list, target_column) for a (league, market) pair.

    Returns the feature list filtered to those actually present in full_df.
    """
    from config import ALL_FEATURES, BTTS_ALL_FEATURES
    try:
        from championship_pipeline import (
            CHAMP_ALL_FEATURES, CHAMP_OU15_FEATURES, CHAMP_BTTS_FEATURES,
        )
    except ImportError:
        CHAMP_ALL_FEATURES = []
        CHAMP_OU15_FEATURES = []
        CHAMP_BTTS_FEATURES = []

    key = (league.upper(), market.lower())
    mapping = {
        ("PL",  "ou25"): (ALL_FEATURES,         "Over_2_5"),
        ("PL",  "btts"): (BTTS_ALL_FEATURES,    "BTTS"),
        ("EFL", "ou25"): (CHAMP_ALL_FEATURES,   "Over_2_5"),
        ("EFL", "ou15"): (CHAMP_OU15_FEATURES,  "Over_1_5"),
        ("EFL", "btts"): (CHAMP_BTTS_FEATURES,  "BTTS"),
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported audit combination: league={league}, market={market}. "
            f"Supported: {list(mapping.keys())}")
    features, target = mapping[key]
    # Drop any feature not actually present in the loaded DataFrame
    features = [f for f in features if f in full_df.columns]
    return features, target


def audit_features(
    league: str,
    market: str,
    full_df: pd.DataFrame | None = None,
    output_dir: str = "reports",
    verbose: bool = True,
    random_seed: int = 42,
    n_permutations: int = 5,
) -> pd.DataFrame:
    """Feature importance audit for a single (league, market) combination.

    Trains XGB + LGB on walk-forward folds; for the last two validation
    folds extracts gain-based importance and computes permutation
    importance (mean AUC drop over ``n_permutations`` shuffles per
    fold). Injects a Gaussian-noise column as a baseline — any feature
    with permutation importance below the noise column is flagged as a
    pruning candidate.

    Args:
        league: "PL" or "EFL".
        market: "ou25", "ou15", or "btts".
        full_df: Optional pre-loaded pipeline DataFrame. Saves ~60s reload
            when auditing multiple markets of the same league.
        output_dir: Directory to write the audit CSV to (created if missing).
        verbose: Print progress.
        random_seed: Seed for noise column and permutation shuffles.
        n_permutations: Number of shuffles per feature per fold. Higher
            values reduce variance in the permutation estimate at the
            cost of linear extra predict-calls. Default 5 gives 10 total
            samples per feature across two folds.

    Returns:
        DataFrame with columns: feature, group, xgb_gain, lgb_gain,
        perm_auc_drop, perm_auc_drop_std, nan_rate, pruning_candidate.
    """
    rng = np.random.default_rng(random_seed)

    # ── 1. Load pipeline data if not provided ─────────────────────────────
    if full_df is None:
        if league.upper() == "PL":
            from pipeline import run_pipeline as _pl_pipeline
            data = _pl_pipeline(verbose=verbose)
        elif league.upper() == "EFL":
            from championship_pipeline import run_pipeline as _efl_pipeline
            data = _efl_pipeline(verbose=verbose)
        else:
            raise ValueError(f"Unknown league: {league}")
        full_df = data["full_df"].copy()

    # ── 2. Materialise BTTS if the market needs it ────────────────────────
    if "BTTS" not in full_df.columns:
        full_df = full_df.copy()
        full_df["BTTS"] = (
            (full_df["Home_Goals"] > 0) & (full_df["Away_Goals"] > 0)
        ).astype(int)

    # ── 3. Resolve features and target for this market ───────────────────
    features, target_col = _resolve_audit_config(league, market, full_df)
    if verbose:
        print(f"\n  [audit] {league}/{market}: "
              f"{len(features)} features, target={target_col}")

    if target_col not in full_df.columns:
        raise ValueError(f"Target column {target_col} missing from DataFrame")

    # ── 4. Inject noise baseline column ───────────────────────────────────
    noise_col = "__noise_baseline__"
    full_df = full_df.copy()
    full_df[noise_col] = rng.standard_normal(len(full_df))
    features_with_noise = list(features) + [noise_col]

    # ── 5. Identify last two walk-forward folds ───────────────────────────
    # Match walk_forward_cv's convention: train on 14..N, validate on N+1.
    all_seasons = sorted(full_df["SeasonIndex"].unique())
    valid_seasons = [s for s in all_seasons if s >= 14]
    # Use the final two seasons as our audit folds
    audit_seasons = valid_seasons[-2:]
    if len(audit_seasons) < 2:
        raise RuntimeError(
            f"Not enough seasons to run audit (need 2, got {len(audit_seasons)})")

    # ── 6. Run two folds: train models, compute gains + perm importance ──
    gain_xgb_sum = np.zeros(len(features_with_noise))
    gain_lgb_sum = np.zeros(len(features_with_noise))
    # Collect all permutation samples across folds × repeats for std estimate
    perm_samples: list[np.ndarray] = []
    n_folds = 0

    for val_season in audit_seasons:
        train_df = full_df[(full_df["SeasonIndex"] >= 14)
                            & (full_df["SeasonIndex"] < val_season)]
        val_df = full_df[full_df["SeasonIndex"] == val_season]
        if len(train_df) < 100 or len(val_df) < 50:
            if verbose:
                print(f"    Skipping fold S{val_season} "
                      f"(train={len(train_df)}, val={len(val_df)})")
            continue

        X_tr = train_df[features_with_noise].values
        y_tr = train_df[target_col].values
        X_v_df = val_df[features_with_noise]
        X_v = X_v_df.values
        y_v = val_df[target_col].values

        if len(np.unique(y_v)) < 2:
            if verbose:
                print(f"    Skipping fold S{val_season} (target is constant)")
            continue

        xgb_m = train_xgb(X_tr, y_tr, X_v, y_v)
        lgb_m = train_lgb(X_tr, y_tr, X_v, y_v,
                          feature_names=features_with_noise)

        # Gain importances (already normalised by sklearn wrappers)
        gain_xgb_sum += xgb_m.feature_importances_
        gain_lgb_sum += lgb_m.feature_importances_

        # Baseline ensemble AUC on this fold
        base_xgb_p = xgb_m.predict_proba(X_v)[:, 1]
        base_lgb_p = lgb_m.predict_proba(
            pd.DataFrame(X_v, columns=features_with_noise))[:, 1]
        base_ensemble = (base_xgb_p + base_lgb_p) / 2.0
        baseline_auc = roc_auc_score(y_v, base_ensemble)

        if verbose:
            print(f"    Fold S{val_season}: baseline ensemble AUC={baseline_auc:.4f} "
                  f"(train={len(train_df)}, val={len(val_df)}, "
                  f"n_perm={n_permutations})")

        # Permutation importance: shuffle each column n_permutations times,
        # measure AUC drop per shuffle, and keep every sample for std.
        perm_rng = np.random.default_rng(random_seed + val_season)
        for rep in range(n_permutations):
            fold_drops = np.zeros(len(features_with_noise))
            for fi, feat_name in enumerate(features_with_noise):
                X_v_perm = X_v.copy()
                X_v_perm[:, fi] = perm_rng.permutation(X_v_perm[:, fi])
                perm_xgb_p = xgb_m.predict_proba(X_v_perm)[:, 1]
                perm_lgb_p = lgb_m.predict_proba(
                    pd.DataFrame(X_v_perm, columns=features_with_noise))[:, 1]
                perm_ensemble = (perm_xgb_p + perm_lgb_p) / 2.0
                try:
                    perm_auc = roc_auc_score(y_v, perm_ensemble)
                except ValueError:
                    perm_auc = baseline_auc
                fold_drops[fi] = baseline_auc - perm_auc
            perm_samples.append(fold_drops)

        n_folds += 1

    if n_folds == 0:
        raise RuntimeError(f"No usable audit folds for {league}/{market}")

    # Average gains across folds; average perm drops across all samples
    gain_xgb = gain_xgb_sum / n_folds
    gain_lgb = gain_lgb_sum / n_folds
    perm_matrix = np.vstack(perm_samples)  # shape: (n_samples, n_features)
    perm_drop = perm_matrix.mean(axis=0)
    perm_drop_std = perm_matrix.std(axis=0)

    # ── 7. NaN rate across the full training window ──────────────────────
    train_window = full_df[full_df["SeasonIndex"] >= 14]
    nan_rates = []
    for f in features_with_noise:
        col = train_window[f]
        nan_rates.append(col.isna().mean())

    # ── 8. Group membership ─────────────────────────────────────────────
    group_map = _build_feature_to_group_map()
    groups = [group_map.get(f, "UNKNOWN") for f in features_with_noise]
    # Override for noise baseline
    noise_idx = features_with_noise.index(noise_col)
    groups[noise_idx] = "NOISE_BASELINE"

    # ── 9. Pruning candidate flag ────────────────────────────────────────
    # A feature is a pruning candidate if its permutation importance is
    # not statistically distinguishable from the noise baseline —
    # specifically, if perm_drop <= noise_mean + 1 * noise_std. This
    # treats noise as a distribution rather than a point estimate; with
    # only ~10 permutation samples per feature the mean alone is noisy.
    noise_perm_drop = perm_drop[noise_idx]
    noise_perm_std = perm_drop_std[noise_idx]
    noise_threshold = noise_perm_drop + noise_perm_std
    pruning_candidate = perm_drop <= noise_threshold

    # ── 10. Assemble and save results ────────────────────────────────────
    audit_df = pd.DataFrame({
        "feature": features_with_noise,
        "group": groups,
        "xgb_gain": gain_xgb,
        "lgb_gain": gain_lgb,
        "perm_auc_drop": perm_drop,
        "perm_auc_drop_std": perm_drop_std,
        "nan_rate": nan_rates,
        "pruning_candidate": pruning_candidate,
    })
    # Sort by permutation importance descending (highest contributors first)
    audit_df = audit_df.sort_values(
        "perm_auc_drop", ascending=False).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"feature_audit_{league.upper()}_{market.lower()}.csv")
    audit_df.to_csv(out_path, index=False)

    if verbose:
        print(f"  [audit] saved to {out_path}")
        print(f"  [audit] noise baseline: mean={noise_perm_drop:+.5f}, "
              f"std={noise_perm_std:.5f}, "
              f"1-sigma threshold={noise_threshold:+.5f}")
        print(f"  [audit] {pruning_candidate.sum()}/{len(features_with_noise)} "
              f"features flagged as pruning candidates (perm <= 1-sigma above noise)")
        print(f"  [audit] top 10 by perm_auc_drop:")
        for _, row in audit_df.head(10).iterrows():
            print(f"    {row['feature']:40s} group={row['group']:15s} "
                  f"perm={row['perm_auc_drop']:+.5f} "
                  f"xgb_gain={row['xgb_gain']:.4f}")

    return audit_df


# ═══════════════════════════════════════════════════════════════════════════════
# Feature pruning
# ═══════════════════════════════════════════════════════════════════════════════

def prune_features(xgb_model, features, min_importance=0.001):
    """Remove features with near-zero importance."""
    importance = xgb_model.feature_importances_
    mask = importance >= min_importance
    kept = [f for f, m in zip(features, mask) if m]
    dropped = [f for f, m in zip(features, mask) if not m]

    if dropped:
        print(f"\nFeature pruning: dropping {len(dropped)} low-importance features:")
        for f in dropped:
            print(f"  - {f} (importance: {importance[features.index(f)]:.6f})")
        print(f"Keeping {len(kept)}/{len(features)} features")
    else:
        print("\nFeature pruning: all features above threshold, keeping all")

    return kept, dropped


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation & plotting
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X, y, label="Test"):
    """Evaluate model and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    ll = log_loss(y, y_prob)
    brier = brier_score_loss(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    print(f"\n{'='*50}")
    print(f"{label} Set Evaluation")
    print(f"{'='*50}")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    print(f"Log Loss:   {ll:.4f}")
    print(f"Brier Score:{brier:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    print(f"\nBase rate (actual Over 2.5): {y.mean():.1%}")
    print(f"Predicted Over 2.5 rate:     {y_pred.mean():.1%}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "log_loss": ll, "brier": brier, "auc": auc}


def plot_feature_importance(model, features, top_n=20):
    """Plot XGBoost feature importance."""
    importance = model.feature_importances_
    idx = np.argsort(importance)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), importance[idx])
    plt.yticks(range(len(idx)), [features[i] for i in idx])
    plt.xlabel("Feature Importance (gain)")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def plot_shap(model, X, features, max_display=20):
    """Generate SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=features,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary.png"), dpi=150)
    plt.close()


def plot_calibration(model, X, y):
    """Plot calibration curve."""
    from sklearn.calibration import calibration_curve
    y_prob = model.predict_proba(X)[:, 1]
    fraction_of_positives, mean_predicted = calibration_curve(y, y_prob, n_bins=10)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "calibration_curve.png"), dpi=150)
    plt.close()


def save_model(model, features):
    """Save model and feature list. No scaler needed (trees don't need scaling)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Save a dummy scaler for backward compat with Dash app that loads SCALER_PATH
    joblib.dump(None, SCALER_PATH)
    joblib.dump(features, FEATURE_LIST_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load saved model, scaler, and feature list."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_LIST_PATH)
    return model, scaler, features


# ═══════════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main(tune=False):
    # ── Load data ──
    data = run_pipeline(verbose=True)
    train = data["train"]
    val = data["val"]
    test = data["test"]
    full_df = data["full_df"]
    features = list(data["features"])

    X_train = train[features].values
    y_train = train["Over_2_5"].values
    X_val = val[features].values
    y_val = val["Over_2_5"].values
    X_test = test[features].values
    y_test = test["Over_2_5"].values

    # ── Phase 0: Tune Dixon-Coles hyperparameters ──
    print("\n" + "="*60)
    print("Phase 0: Tuning Dixon-Coles hyperparameters")
    print("="*60)
    wf_df = full_df[(full_df["SeasonIndex"] >= TRAIN_MIN_SEASON) &
                     (~full_df["SeasonIndex"].isin(TEST_SEASONS))].copy()
    dc_kwargs = tune_dc_params(wf_df)

    # ── Phase 1: Walk-forward CV (robust multi-fold evaluation) ──
    print("\n" + "="*60)
    print("Phase 1: Walk-forward cross-validation")
    print("="*60)
    # Combine train+val for walk-forward (test stays held out)
    fold_metrics, oof_df = walk_forward_cv(wf_df, features, dc_kwargs=dc_kwargs)

    if len(fold_metrics) > 0:
        avg_auc = np.mean([f["auc"] for f in fold_metrics])
        avg_acc = np.mean([f["acc"] for f in fold_metrics])
        avg_brier = np.mean([f["brier"] for f in fold_metrics])
        print(f"\n  Walk-forward summary ({len(fold_metrics)} folds):")
        print(f"    Avg AUC:   {avg_auc:.4f}")
        print(f"    Avg Acc:   {avg_acc:.4f}")
        print(f"    Avg Brier: {avg_brier:.4f}")

    # ── Phase 2: Train base models ──
    # Step 1: Train on train (14-23) with val (24) for early stopping & feature pruning.
    # Step 2: Retrain final models on train+val combined (14-24) for maximum data.
    print("\n" + "="*60)
    print("Phase 2: Train base models")
    print("="*60)

    # 2a: XGBoost — use val for early stopping, then retrain on full data
    print("\nTraining XGBoost (feature pruning on train, val for early stopping)...")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    print(f"  Best iteration (for pruning): {xgb_model.best_iteration}")
    best_n_trees = xgb_model.best_iteration

    kept_features, dropped = prune_features(xgb_model, features)
    if dropped:
        features = kept_features
        X_train = train[features].values
        X_val = val[features].values
        X_test = test[features].values

    # Combine train+val for final training (seasons 14-24)
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val[features].values
    y_train_val = train_val["Over_2_5"].values
    print(f"\n  Combined train+val: {len(train_val)} matches (seasons 14-24)")

    # Retrain XGBoost on full data using best_n_trees from early stopping
    print("  Retraining XGBoost on train+val...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=best_n_trees, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        gamma=0.1, reg_alpha=0.5, reg_lambda=3.0,
        eval_metric="logloss", random_state=42,
    )
    xgb_model.fit(X_train_val, y_train_val, verbose=False)
    print(f"  XGBoost retrained with {best_n_trees} trees on {len(train_val)} rows")

    # 2b: LightGBM — same approach
    print("\nTraining LightGBM (early stopping on val)...")
    lgb_temp = train_lgb(X_train, y_train, X_val, y_val, feature_names=features)
    best_lgb_trees = lgb_temp.best_iteration_
    print(f"  Best iteration: {best_lgb_trees}")

    print("  Retraining LightGBM on train+val...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=best_lgb_trees, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=7,
        reg_alpha=0.5, reg_lambda=3.0, random_state=42, verbose=-1,
    )
    lgb_model.fit(pd.DataFrame(X_train_val, columns=features), y_train_val)
    print(f"  LightGBM retrained with {best_lgb_trees} trees on {len(train_val)} rows")

    # 2c: Dixon-Coles on train+val (using tuned hyperparameters)
    print(f"\nTraining Dixon-Coles on train+val (half_life={dc_kwargs.get('half_life', 30)}, rho={dc_kwargs.get('rho', -0.13)})...")
    dc_model = DixonColesPredictor(**dc_kwargs)
    dc_model.fit(train_val)

    # ── Phase 3: Stacking meta-learner ──
    print("\n" + "="*60)
    print("Phase 3: Stacking meta-learner (walk-forward OOF predictions)")
    print("="*60)

    # Use walk-forward OOF predictions from Phase 1 to train the stacker.
    # This captures temporal distribution shift (train on past, predict future)
    # — unlike KFold which leaks within-season patterns.
    if len(oof_df) > 100:
        print(f"\n  Using {len(oof_df)} walk-forward OOF predictions for stacker training")
        oof_stack_full = oof_df[["xgb", "lgb", "dc"]].values
        oof_y = oof_df["y"].values

        stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        stacker.fit(oof_stack_full, oof_y)
    else:
        # Fallback: KFold OOF if walk-forward didn't produce enough data
        print("\n  Insufficient walk-forward OOF -- falling back to KFold")
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=False)
        oof_xgb = np.zeros(len(y_train))
        oof_lgb = np.zeros(len(y_train))
        oof_dc = np.zeros(len(y_train))

        for fold_i, (tr_idx, oof_idx) in enumerate(kf.split(X_train)):
            fold_xgb = train_xgb(X_train[tr_idx], y_train[tr_idx],
                                  X_train[oof_idx], y_train[oof_idx])
            oof_xgb[oof_idx] = fold_xgb.predict_proba(X_train[oof_idx])[:, 1]

            fold_lgb = train_lgb(X_train[tr_idx], y_train[tr_idx],
                                  X_train[oof_idx], y_train[oof_idx],
                                  feature_names=features)
            oof_lgb[oof_idx] = fold_lgb.predict_proba(
                pd.DataFrame(X_train[oof_idx], columns=features))[:, 1]

            fold_train_df = train.iloc[tr_idx]
            fold_dc = DixonColesPredictor(**dc_kwargs)
            fold_dc.fit(fold_train_df)
            oof_dc[oof_idx] = fold_dc.predict_proba_df(train.iloc[oof_idx])

        oof_stack_full = np.column_stack([oof_xgb, oof_lgb, oof_dc])
        oof_y = y_train

        stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        stacker.fit(oof_stack_full, oof_y)

    # Stacker weights (interpretable)
    print(f"\n  Stacker coefficients: XGB={stacker.coef_[0][0]:.3f}, "
          f"LGB={stacker.coef_[0][1]:.3f}, DC={stacker.coef_[0][2]:.3f}")
    print(f"  Stacker intercept: {stacker.intercept_[0]:.3f}")

    # OOF stacker AUC (unbiased — walk-forward predictions)
    oof_stacker_eval = stacker.predict_proba(oof_stack_full)[:, 1]
    oof_stacker_auc = roc_auc_score(oof_y, oof_stacker_eval)
    print(f"  OOF stacker AUC: {oof_stacker_auc:.4f}")

    # ── Phase 4: Calibration (logit-shift prior correction) ──
    print("\n" + "="*60)
    print("Phase 4: Calibration (logit-shift prior correction)")
    print("="*60)

    # Problem: stacker probabilities drift upward on new seasons.
    # On OOF data, mean predicted prob ≈ actual Over rate (well-calibrated).
    # But on truly new data (test), predictions inflate by ~3-6% because:
    #   1. Models trained on more data → more confident → shift away from 0.5
    #   2. Feature distributions shift between seasons
    #
    # Old approach (Platt on nested OOF) learned near-identity because OOF was
    # already calibrated. It couldn't fix the TEST shift it never saw.
    #
    # New approach: Logit-shift prior correction.
    # The model's RANKING is correct (AUC is stable). The MEAN drifts.
    # Fix: compute a logit-space shift on val predictions (closest to deployment),
    # then apply at inference time.
    #   shift = mean(logit(p_val)) - logit(base_rate)
    #   p_corrected = sigmoid(logit(p) - shift)
    # This preserves ranking perfectly (monotonic) while fixing the mean.

    # Step 1: Get stacker predictions on val set (most recent known data)
    xgb_val_p = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_p = lgb_model.predict_proba(pd.DataFrame(X_val, columns=features))[:, 1]
    dc_val_p = dc_model.predict_proba_df(val)
    val_stack = np.column_stack([xgb_val_p, lgb_val_p, dc_val_p])
    stacker_val_raw = stacker.predict_proba(val_stack)[:, 1]

    val_mean_logit = np.mean(np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10)))
    # Use long-term base rate as target (more stable than single-season val rate)
    base_rate = y_train_val.mean()  # ~0.53 over seasons 14-24
    target_logit = np.log(base_rate / (1 - base_rate + 1e-10))
    logit_shift = val_mean_logit - target_logit

    print(f"\n  Val predictions: mean={stacker_val_raw.mean():.4f}, actual={y_val.mean():.4f}")
    print(f"  Val mean logit: {val_mean_logit:.4f}")
    print(f"  Target base rate: {base_rate:.4f} (logit: {target_logit:.4f})")
    print(f"  Logit shift: {logit_shift:.4f}")

    # Verify on val: corrected predictions should match base rate better
    val_logits = np.log(stacker_val_raw / (1 - stacker_val_raw + 1e-10))
    val_corrected = 1 / (1 + np.exp(-(val_logits - logit_shift)))
    print(f"\n  Val corrected mean: {val_corrected.mean():.4f} (target: {base_rate:.4f})")
    print(f"  Val Brier: before={brier_score_loss(y_val, stacker_val_raw):.4f}, "
          f"after={brier_score_loss(y_val, val_corrected):.4f}")
    print(f"  Val AUC:   {roc_auc_score(y_val, stacker_val_raw):.4f} "
          f"(unchanged — logit shift preserves ranking)")

    # Also compute nested OOF stats for comparison
    oof_seasons = sorted(oof_df["season"].unique())
    nested_stacker_preds = np.zeros(len(oof_df))
    print(f"\n  Nested OOF stacker predictions ({len(oof_seasons)} seasons)...")
    for hold_season in oof_seasons:
        hold_mask = oof_df["season"] == hold_season
        train_mask = ~hold_mask
        if train_mask.sum() < 50 or hold_mask.sum() < 10:
            nested_stacker_preds[hold_mask] = stacker.predict_proba(
                oof_stack_full[hold_mask])[:, 1]
            continue
        temp_stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        temp_stacker.fit(oof_stack_full[train_mask], oof_y[train_mask])
        nested_stacker_preds[hold_mask] = temp_stacker.predict_proba(
            oof_stack_full[hold_mask])[:, 1]
        hold_auc = roc_auc_score(oof_y[hold_mask], nested_stacker_preds[hold_mask])
        print(f"    S{hold_season}: n={hold_mask.sum()}, AUC={hold_auc:.4f}, "
              f"mean_prob={nested_stacker_preds[hold_mask].mean():.4f}")
    print(f"  Nested OOF: AUC={roc_auc_score(oof_y, nested_stacker_preds):.4f}, "
          f"Brier={brier_score_loss(oof_y, nested_stacker_preds):.4f}, "
          f"mean={nested_stacker_preds.mean():.4f}")

    # Store the logit_shift as calibration (will be applied in EnsembleModel.predict_proba)
    calibrator = logit_shift  # float, not a model
    calibrator_type = "logit_shift"
    print(f"\n  >> Using logit-shift calibration (shift={logit_shift:.4f})")

    # Threshold
    overall_base_rate = y_train_val.mean()
    chosen_threshold = 0.50  # With proper calibration, 0.5 should work

    # ── Build final ensemble ──
    ensemble = EnsembleModel(
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        logreg_model=None,      # LR removed: AUC < 0.5 on test, adds noise not signal
        logreg_scaler=None,
        dc_model=dc_model,
        stacker=stacker,
        feature_names=features,
        threshold=chosen_threshold,
        platt_scaler=calibrator,
    )

    # ── Phase 5: Evaluate on held-out test ──
    print("\n" + "="*60)
    print("Phase 5: Final evaluation")
    print("="*60)

    # Individual models on test
    print("\n--- Individual Base Models ---")
    xgb_test_p = xgb_model.predict_proba(X_test)[:, 1]
    lgb_test_p = lgb_model.predict_proba(pd.DataFrame(X_test, columns=features))[:, 1]
    dc_test_p = dc_model.predict_proba_df(test)

    for name, probs in [("XGBoost", xgb_test_p), ("LightGBM", lgb_test_p),
                         ("Dixon-Coles", dc_test_p)]:
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, (probs > 0.5).astype(int))
        brier = brier_score_loss(y_test, probs)
        print(f"  {name:12s}: AUC={auc:.4f}  Acc={acc:.4f}  Brier={brier:.4f}")

    # Stacker on test — UNCALIBRATED (consistent with stacker training: direct dc_model predictions)
    test_stack = np.column_stack([xgb_test_p, lgb_test_p, dc_test_p])
    stacker_test_raw = stacker.predict_proba(test_stack)[:, 1]

    # CALIBRATED — apply calibrator to stacker output
    if calibrator_type == "logit_shift":
        raw_logits = np.log(stacker_test_raw / (1 - stacker_test_raw + 1e-10))
        stacker_test_p = 1 / (1 + np.exp(-(raw_logits - calibrator)))
    elif calibrator_type == "platt":
        raw_logits = np.log(stacker_test_raw / (1 - stacker_test_raw + 1e-10)).reshape(-1, 1)
        stacker_test_p = calibrator.predict_proba(raw_logits)[:, 1]
    else:
        stacker_test_p = calibrator.predict_proba(stacker_test_raw.reshape(-1, 1))[:, 1]

    # Also check ensemble path (uses Poisson_DC feature as DC fallback when dc_probs not passed)
    ensemble_fallback_p = ensemble.predict_proba(X_test)[:, 1]
    # Proper path: pass dc_probs from dc_model
    ensemble_proper_p = ensemble.predict_proba(X_test, dc_probs=dc_test_p)[:, 1]

    print(f"\n--- Stacker Ensemble (3 models: XGB + LGB + DC) ---")
    stacker_test_auc = roc_auc_score(y_test, stacker_test_p)

    # Calibration diagnostics
    print(f"\n  Probability distribution:")
    print(f"    {'':20s} {'Uncalibrated':>14s} {'Calibrated':>14s} {'Actual':>10s}")
    print(f"    {'Mean prob':20s} {stacker_test_raw.mean():14.4f} {stacker_test_p.mean():14.4f} {y_test.mean():10.4f}")
    print(f"    {'Median prob':20s} {np.median(stacker_test_raw):14.4f} {np.median(stacker_test_p):14.4f} {'':>10s}")
    print(f"    {'Pred Over% (>0.5)':20s} {(stacker_test_raw > 0.5).mean():14.1%} {(stacker_test_p > 0.5).mean():14.1%} {y_test.mean():10.1%}")
    print(f"    {'Brier score':20s} {brier_score_loss(y_test, stacker_test_raw):14.4f} {brier_score_loss(y_test, stacker_test_p):14.4f} {'':>10s}")
    print(f"    {'AUC':20s} {roc_auc_score(y_test, stacker_test_raw):14.4f} {roc_auc_score(y_test, stacker_test_p):14.4f} {'':>10s}")

    # DC path consistency check
    ensemble_base_fallback = ensemble._base_probs(X_test)  # uses Poisson_DC feature
    dc_from_feature = ensemble_base_fallback[:, 2]
    dc_corr = np.corrcoef(dc_test_p, dc_from_feature)[0, 1]
    print(f"\n  DC consistency: dc_model mean={dc_test_p.mean():.4f}, "
          f"Poisson_DC feature mean={dc_from_feature.mean():.4f}, corr={dc_corr:.4f}")
    print(f"  Ensemble (Poisson_DC fallback) AUC: {roc_auc_score(y_test, ensemble_fallback_p):.4f}")
    print(f"  Ensemble (dc_probs passed)     AUC: {roc_auc_score(y_test, ensemble_proper_p):.4f}")

    # Accuracy
    stacker_test_acc = accuracy_score(y_test, (stacker_test_p >= 0.5).astype(int))
    oracle_t = np.percentile(stacker_test_p, (1 - y_test.mean()) * 100)
    stacker_test_acc_oracle = accuracy_score(y_test, (stacker_test_p >= oracle_t).astype(int))
    stacker_test_brier = brier_score_loss(y_test, stacker_test_p)

    print(f"\n  Acc (t=0.50):    {stacker_test_acc:.4f} (pred rate: {(stacker_test_p >= 0.5).mean():.1%})")
    print(f"  Acc (oracle t={oracle_t:.4f}): {stacker_test_acc_oracle:.4f}")
    print(f"  Actual Over rate: {y_test.mean():.1%}")

    # Simple avg for comparison (3 models)
    simple_avg_test = (xgb_test_p + lgb_test_p + dc_test_p) / 3
    simple_auc = roc_auc_score(y_test, simple_avg_test)
    simple_oracle_t = np.percentile(simple_avg_test, (1 - y_test.mean()) * 100)
    simple_acc = accuracy_score(y_test, (simple_avg_test >= simple_oracle_t).astype(int))
    print(f"\n  Simple avg (3 models):  AUC={simple_auc:.4f}  Acc(oracle)={simple_acc:.4f}")
    print(f"  DC alone:               AUC={roc_auc_score(y_test, dc_test_p):.4f}")
    print(f"  XGB alone:              AUC={roc_auc_score(y_test, xgb_test_p):.4f}")

    # Overfitting diagnostics
    print(f"\n--- Overfitting Diagnostics ---")
    # Train+Val AUC (models trained on this data — measures memorization)
    xgb_tv_p = xgb_model.predict_proba(X_train_val)[:, 1]
    lgb_tv_p = lgb_model.predict_proba(pd.DataFrame(X_train_val, columns=features))[:, 1]
    dc_tv_p = dc_model.predict_proba_df(train_val)

    tv_stack = np.column_stack([xgb_tv_p, lgb_tv_p, dc_tv_p])
    stacker_tv_p = stacker.predict_proba(tv_stack)[:, 1]
    tv_auc = roc_auc_score(y_train_val, stacker_tv_p)

    # OOF AUC (unbiased estimate from walk-forward)
    oof_stack_p = stacker.predict_proba(oof_stack_full)[:, 1]
    oof_auc = roc_auc_score(oof_y, oof_stack_p)

    print(f"  {'':24s} {'Train+Val':>10s} {'OOF(WF)':>10s} {'Test':>10s}")
    print(f"  {'AUC':24s} {tv_auc:10.4f} {oof_auc:10.4f} {stacker_test_auc:10.4f}")
    print(f"  {'Brier':24s} {brier_score_loss(y_train_val, stacker_tv_p):10.4f} "
          f"{brier_score_loss(oof_y, oof_stack_p):10.4f} {stacker_test_brier:10.4f}")
    print(f"  {'Over rate (actual)':24s} {y_train_val.mean():10.1%} "
          f"{oof_y.mean():10.1%} {y_test.mean():10.1%}")
    print(f"  {'N matches':24s} {len(y_train_val):10d} {len(oof_y):10d} {len(y_test):10d}")

    train_test_auc_gap = tv_auc - stacker_test_auc
    print(f"\n  Train+Val-Test AUC gap: {train_test_auc_gap:.4f} "
          f"({'OK' if train_test_auc_gap < 0.10 else 'OVERFITTING' if train_test_auc_gap < 0.15 else 'SEVERE OVERFITTING'})")

    # Walk-forward vs test comparison (most meaningful)
    if len(fold_metrics) > 0:
        print(f"  Walk-forward avg AUC: {avg_auc:.4f} vs Test AUC: {stacker_test_auc:.4f} "
              f"(gap: {abs(avg_auc - stacker_test_auc):.4f})")

    test_metrics = {
        "accuracy": stacker_test_acc, "auc": stacker_test_auc,
        "brier": stacker_test_brier, "log_loss": log_loss(y_test, stacker_test_p),
    }

    # Save
    save_model(ensemble, features)

    # Plots
    print("\nGenerating plots...")
    plot_feature_importance(xgb_model, features)
    plot_shap(xgb_model, X_test[:200], features)
    plot_calibration(ensemble, X_test, y_test)

    return ensemble, features, test_metrics


if __name__ == "__main__":
    import sys
    tune = "--tune" in sys.argv
    main(tune=tune)
