---
name: dixon-coles-methodology
description: "Statistical methodology reference for the Dixon-Coles model and 4-model stacking ensemble, covering theory, implementation, and evaluation."
user-invocable: false
---

## Model Architecture Reference

### Overview

The system uses a **4-model stacking ensemble**, not a standalone Dixon-Coles model. Dixon-Coles is one of three base models feeding into a logistic regression meta-learner.

### Base Models (in `model.py`)

1. **XGBoost** — gradient boosted trees on 150+ engineered features (stacker weight: 2.16)
2. **LightGBM** — gradient boosted trees on same features (stacker weight: 1.18)
3. **Dixon-Coles Poisson** (`DixonColesPredictor`) — attack/defence strengths with tau correction (stacker weight: 1.10)

LR was removed from base models — AUC < 0.5 on test, added noise.

### Meta-Learner

**LogisticRegression stacker** trained on walk-forward out-of-fold (OOF) predictions from the 3 base models. Critical: `dc_probs` must be passed to `EnsembleModel.predict_proba(X, dc_probs=dc_probs)` for accurate DC contribution (Poisson_DC pipeline feature only has 0.41 correlation with dc_model).

### Dixon-Coles Theory

The Dixon-Coles (1997) model extends independent Poisson regression for football match scores by introducing a tau (tau) correlation parameter that adjusts probabilities for low-scoring outcomes (0-0, 1-0, 0-1, 1-1).

**Key parameters:**
- **Attack strength (alpha)**: per-team offensive capability
- **Defence strength (beta)**: per-team defensive capability
- **Home advantage (gamma)**: league-wide home effect
- **Tau correlation**: adjustment for low-scoring outcomes, typically small and negative

### Market Derivation from Goal Matrix

From the bivariate Poisson goal distribution (with tau correction):

- **Over/Under X goals**: sum all (i,j) cells where i+j > X (over) or i+j <= X (under)
- **BTTS Yes**: sum all (i,j) cells where i >= 1 AND j >= 1
- **BTTS No**: 1 - P(BTTS Yes)
- **O/U 1.5 specifically uses Dixon-Coles Poisson only** (not the full ensemble)

### Edge Calculation

In `predict.py` `_evaluate_bet()`:
```
blended_p = blend_weight * model_prob + (1 - blend_weight) * fair_prob
edge = blended_p - fair_prob
ev = blended_p * odds - 1
```
Default `blend_weight = 0.35` (35% model, 65% market).

Fair probability derived by removing overround:
```
implied_prob = 1 / decimal_odds
overround = sum(implied_probs_for_both_sides)
fair_prob = implied_prob / overround
```

### Kelly Criterion Staking (in `backtest.py`)

`refined_kelly()` implements confidence-scaled quarter-Kelly:
1. Raw Kelly: `(blended_prob * odds - 1) / (odds - 1)`
2. Base fraction: `kelly * 0.25` (quarter-Kelly)
3. Agreement scaling: 2/4 agree = 0.7x, 3/4 = 0.9x, 4/4 = 1.1x
4. Edge magnitude scaling: edge > 4% = 1.15x, > 6% = 1.25x
5. Drawdown protection: reduces stakes based on bankroll vs peak
6. Hard cap: 5% of bankroll maximum
7. Minimum filter: below 0.3% = no bet

### Walk-Forward Cross-Validation

- Train on seasons 14..N, validate on N+1 (6 folds, seasons 19-24)
- OOF predictions used for stacker training — captures temporal distribution shift
- No scaler needed for trees; LR has its own internal `_clip_scaled()` StandardScaler
- Feature pruning: drops zero-importance features after initial XGBoost training

### Calibration

- Platt scaling (logistic on stacker logits), fit on nested temporal CV of OOF predictions
- Known issue: probability distribution shifts ~3-6% upward on test (mean 0.586 vs actual 0.528)
- AUC/ranking is reliable; absolute probabilities slightly inflated
- Edge detection still works because both sides shift proportionally

### Current Performance

- Test AUC: 0.5799 (stacker with dc_probs)
- DC standalone: 0.597 AUC
- Walk-forward average AUC: 0.583
- Backtest (10% edge threshold): +9.7% ROI over 5 seasons, Under bets +26.9% ROI

### Championship Differences

- Uses 3-model ensemble (XGB + LGB + DC, no LR base)
- No FPL or Understat features (not available for Championship)
- Uses FBref for some features (fbref_comp_id: 10)
- Higher average goals per game — model parameters reflect this
- Greater variance in team quality
- Synthesises features for newly promoted teams using league averages

### Academic References

- Dixon, M.J. & Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
- Fischer, K. & Heuer, A. (2024). Hybrid Poisson-ML approaches
- Wheatcroft, E. — corners and shots outperform goals as predictive inputs
- Rue, H. & Salvesen, O. (2000). Prediction and retrospective analysis
