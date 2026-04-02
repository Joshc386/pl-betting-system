"""
Alternative Total Lines Engine.

Computes model probabilities for any Over/Under line (1.5, 2.0, 2.25, 2.5,
2.75, 3.0, 3.5, etc.) using the Poisson goal distribution, then finds the
best edge across all available lines.

Handles Asian handicap (quarter-lines) correctly:
  - 2.25 = half on Over 2.0 + half on Over 2.5
  - 2.75 = half on Over 2.5 + half on Over 3.0
  - Settlement: win, half-win, push, half-loss, loss
"""
import numpy as np
from scipy.stats import poisson as poisson_dist


# ── Poisson Goal Distribution ──

def build_goal_matrix(home_lambda, away_lambda, max_goals=10, rho=0.0):
    """Build bivariate Poisson probability matrix with optional DC correction.

    Args:
        home_lambda: expected home goals
        away_lambda: expected away goals
        max_goals: max goals per team to consider
        rho: Dixon-Coles dependence parameter (0 = independent Poisson)

    Returns:
        2D numpy array where mat[h][a] = P(home=h, away=a)
    """
    home_lambda = max(home_lambda, 0.01)
    away_lambda = max(away_lambda, 0.01)

    mat = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = poisson_dist.pmf(h, home_lambda) * poisson_dist.pmf(a, away_lambda)
            # Dixon-Coles tau correction for low scores
            if rho != 0 and h <= 1 and a <= 1:
                p *= _dc_tau(h, a, home_lambda, away_lambda, rho)
            mat[h][a] = max(p, 0)

    # Renormalize
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def _dc_tau(x, y, lx, ly, rho):
    """Dixon-Coles tau adjustment for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - lx * ly * rho
    elif x == 0 and y == 1:
        return 1 + lx * rho
    elif x == 1 and y == 0:
        return 1 + ly * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def total_goals_distribution(goal_matrix):
    """Convert goal matrix to total goals probability distribution.

    Returns:
        dict: {total_goals: probability} for total 0..2*max_goals
    """
    max_g = goal_matrix.shape[0] - 1
    dist = {}
    for total in range(2 * max_g + 1):
        p = 0
        for h in range(min(total, max_g) + 1):
            a = total - h
            if 0 <= a <= max_g:
                p += goal_matrix[h][a]
        dist[total] = p
    return dist


def prob_over_line(goal_dist, line):
    """Compute P(Over) for any line, including Asian quarter-lines.

    Args:
        goal_dist: dict {total_goals: probability}
        line: the betting line (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, etc.)

    Returns:
        dict with over/under probabilities and settlement details
    """
    line = float(line)
    max_total = max(goal_dist.keys())

    # Check if this is a quarter-line (Asian handicap)
    frac = line % 1
    is_quarter = frac in (0.25, 0.75)
    is_whole = frac == 0.0
    is_half = frac == 0.5

    if is_half:
        # Standard half-line (1.5, 2.5, 3.5): no push possible
        p_over = sum(p for g, p in goal_dist.items() if g > line)
        p_under = 1 - p_over
        return {
            "line": line,
            "type": "standard",
            "p_over": p_over,
            "p_under": p_under,
            "p_push": 0.0,
        }

    elif is_whole:
        # Whole number line (2.0, 3.0): push possible on exact total
        exact = int(line)
        p_over = sum(p for g, p in goal_dist.items() if g > line)
        p_under = sum(p for g, p in goal_dist.items() if g < line)
        p_push = goal_dist.get(exact, 0)
        return {
            "line": line,
            "type": "whole",
            "p_over": p_over,
            "p_under": p_under,
            "p_push": p_push,
        }

    elif is_quarter:
        # Asian quarter-line: split bet between two adjacent lines
        if frac == 0.25:
            # e.g., 2.25 = half on 2.0 + half on 2.5
            lower = int(line)           # 2.0 (whole number, push possible)
            upper = int(line) + 0.5     # 2.5 (standard half-line)
        else:  # frac == 0.75
            # e.g., 2.75 = half on 2.5 + half on 3.0
            lower = int(line) + 0.5     # 2.5 (standard half-line)
            upper = int(line) + 1       # 3.0 (whole number, push possible)

        lower_result = prob_over_line(goal_dist, lower)
        upper_result = prob_over_line(goal_dist, upper)

        # Effective probability is the average of the two legs
        # But for EV/settlement, we need to track both legs separately
        eff_p_over = (lower_result["p_over"] + upper_result["p_over"]) / 2
        eff_p_under = (lower_result["p_under"] + upper_result["p_under"]) / 2
        eff_p_push = (lower_result.get("p_push", 0) + upper_result.get("p_push", 0)) / 2

        return {
            "line": line,
            "type": "asian_quarter",
            "p_over": eff_p_over,
            "p_under": eff_p_under,
            "p_push": eff_p_push,
            "lower_leg": lower_result,
            "upper_leg": upper_result,
        }

    else:
        # Unusual line — treat as half-line
        p_over = sum(p for g, p in goal_dist.items() if g > line)
        p_under = 1 - p_over
        return {
            "line": line,
            "type": "standard",
            "p_over": p_over,
            "p_under": p_under,
            "p_push": 0.0,
        }


# ── Settlement ──

def settle_asian(bet_side, line, actual_goals, odds, stake=1.0):
    """Settle an Asian handicap total bet.

    Args:
        bet_side: "over" or "under"
        line: the betting line
        actual_goals: actual total goals
        odds: decimal odds at bet placement
        stake: stake amount

    Returns:
        dict with profit_loss, outcome description
    """
    line = float(line)
    frac = line % 1

    if frac == 0.5:
        # Standard half-line: win or lose, no push
        if bet_side == "over":
            won = actual_goals > line
        else:
            won = actual_goals < line
        pl = stake * (odds - 1) if won else -stake
        return {"profit_loss": pl, "outcome": "win" if won else "loss"}

    elif frac == 0.0:
        # Whole number: push possible
        exact = int(line)
        if actual_goals == exact:
            return {"profit_loss": 0.0, "outcome": "push"}
        if bet_side == "over":
            won = actual_goals > line
        else:
            won = actual_goals < line
        pl = stake * (odds - 1) if won else -stake
        return {"profit_loss": pl, "outcome": "win" if won else "loss"}

    elif frac in (0.25, 0.75):
        # Asian quarter-line: two legs at half stake each
        if frac == 0.25:
            lower = int(line)       # whole number
            upper = int(line) + 0.5 # half
        else:
            lower = int(line) + 0.5 # half
            upper = int(line) + 1   # whole number

        half_stake = stake / 2
        leg1 = settle_asian(bet_side, lower, actual_goals, odds, half_stake)
        leg2 = settle_asian(bet_side, upper, actual_goals, odds, half_stake)
        total_pl = leg1["profit_loss"] + leg2["profit_loss"]

        # Classify the overall outcome
        if leg1["outcome"] == "win" and leg2["outcome"] == "win":
            outcome = "win"
        elif leg1["outcome"] == "loss" and leg2["outcome"] == "loss":
            outcome = "loss"
        elif "win" in (leg1["outcome"], leg2["outcome"]):
            outcome = "half_win"
        elif "loss" in (leg1["outcome"], leg2["outcome"]):
            outcome = "half_loss"
        else:
            outcome = "push"

        return {"profit_loss": total_pl, "outcome": outcome,
                "leg1": leg1, "leg2": leg2}

    else:
        # Fallback: treat as standard
        if bet_side == "over":
            won = actual_goals > line
        else:
            won = actual_goals < line
        pl = stake * (odds - 1) if won else -stake
        return {"profit_loss": pl, "outcome": "win" if won else "loss"}


# ── Expected Value ──

def expected_value_asian(bet_side, line_probs, odds):
    """Calculate expected value for an Asian total bet.

    Correctly handles pushes and half-wins for Asian lines.

    Args:
        bet_side: "over" or "under"
        line_probs: dict from prob_over_line() with p_over, p_under, p_push, type
        odds: decimal odds

    Returns:
        float: expected value per unit staked
    """
    line_type = line_probs.get("type", "standard")

    if bet_side == "over":
        p_win = line_probs["p_over"]
        p_lose = line_probs["p_under"]
    else:
        p_win = line_probs["p_under"]
        p_lose = line_probs["p_over"]

    p_push = line_probs.get("p_push", 0)

    if line_type == "standard":
        # Half-line: no push
        return p_win * (odds - 1) - p_lose

    elif line_type == "whole":
        # Whole number: push returns stake
        return p_win * (odds - 1) - p_lose + p_push * 0

    elif line_type == "asian_quarter":
        # Quarter-line: need to compute EV from both legs
        lower = line_probs.get("lower_leg", {})
        upper = line_probs.get("upper_leg", {})

        # Each leg is half the stake
        ev1 = expected_value_asian(bet_side, lower, odds) / 2
        ev2 = expected_value_asian(bet_side, upper, odds) / 2
        return ev1 + ev2

    return p_win * (odds - 1) - p_lose


# ── Multi-Line Edge Scanner ──

def scan_all_lines(home_lambda, away_lambda, odds_by_line, rho=0.0,
                   blend_weight=0.35, min_edge=0.02):
    """Scan all available lines and find the best betting opportunities.

    Args:
        home_lambda: expected home goals (from model)
        away_lambda: expected away goals (from model)
        odds_by_line: dict from get_best_odds_all_lines()
                      {line: {"best_over", "best_under", "sharp_fair_over", ...}}
        rho: Dixon-Coles dependence parameter
        blend_weight: model weight in model-market blend (0.35 = 35% model)
        min_edge: minimum edge to flag a bet

    Returns:
        list of opportunities sorted by EV, each with:
            line, side, model_prob, blended_prob, fair_prob, edge, ev, odds, book
    """
    mat = build_goal_matrix(home_lambda, away_lambda, rho=rho)
    goal_dist = total_goals_distribution(mat)

    opportunities = []

    for line, odds_data in odds_by_line.items():
        line = float(line)
        line_probs = prob_over_line(goal_dist, line)

        for side in ("over", "under"):
            best_odds = odds_data.get(f"best_{side}")
            book = odds_data.get(f"best_{side}_book", "")
            if not best_odds or best_odds <= 1:
                continue

            # Model probability for this side
            if side == "over":
                model_prob = line_probs["p_over"]
            else:
                model_prob = line_probs["p_under"]

            # Market fair probability (prefer Pinnacle sharp)
            sharp_key = f"sharp_fair_{side}"
            fair_prob = odds_data.get(sharp_key)
            if fair_prob is None:
                # Fall back to removing overround from best odds
                impl_over = 1.0 / odds_data["best_over"]
                impl_under = 1.0 / odds_data["best_under"]
                total = impl_over + impl_under
                fair_prob = (1.0 / best_odds) / total

            # Model-market blend
            blended_prob = blend_weight * model_prob + (1 - blend_weight) * fair_prob

            # Edge and EV using blended probability
            edge = blended_prob - fair_prob
            ev = expected_value_asian(side, line_probs, best_odds)
            # Adjust EV with blended prob rather than pure model
            ev_blended = blended_prob * (best_odds - 1) - (1 - blended_prob)
            if line_probs.get("type") in ("whole", "asian_quarter"):
                # For lines with push/half-win, use proper EV calculation
                # but scale by blend ratio
                ev_blended = expected_value_asian(
                    side,
                    _scale_probs(line_probs, side, blend_weight, fair_prob),
                    best_odds)

            # Quarter-Kelly
            if best_odds > 1 and blended_prob > 0:
                kelly = (blended_prob * best_odds - 1) / (best_odds - 1)
                kelly = min(max(kelly * 0.25, 0), 0.05)
            else:
                kelly = 0

            opportunities.append({
                "line": line,
                "side": side,
                "line_type": line_probs.get("type", "standard"),
                "model_prob": round(model_prob, 4),
                "blended_prob": round(blended_prob, 4),
                "fair_prob": round(fair_prob, 4),
                "edge": round(edge, 4),
                "ev": round(ev_blended, 4),
                "odds": best_odds,
                "book": book,
                "kelly": round(kelly, 4),
                "p_push": round(line_probs.get("p_push", 0), 4),
            })

    # Sort by EV descending
    opportunities.sort(key=lambda x: x["ev"], reverse=True)
    return opportunities


def _scale_probs(line_probs, side, blend_weight, fair_prob):
    """Create a scaled version of line_probs using blended probability."""
    result = dict(line_probs)
    if side == "over":
        model_p = line_probs["p_over"]
        blended = blend_weight * model_p + (1 - blend_weight) * fair_prob
        ratio = blended / max(model_p, 0.001) if model_p > 0 else 1
        result["p_over"] = blended
        result["p_under"] = 1 - blended - line_probs.get("p_push", 0)
    else:
        model_p = line_probs["p_under"]
        blended = blend_weight * model_p + (1 - blend_weight) * fair_prob
        result["p_under"] = blended
        result["p_over"] = 1 - blended - line_probs.get("p_push", 0)
    return result


def get_value_bets(opportunities, min_edge=0.02, min_ev=0.0):
    """Filter opportunities down to value bets only.

    Args:
        opportunities: list from scan_all_lines()
        min_edge: minimum blended edge
        min_ev: minimum expected value

    Returns:
        filtered list of value bets
    """
    return [o for o in opportunities
            if o["edge"] >= min_edge and o["ev"] > min_ev and o["kelly"] > 0]


# ── Quick Test ──
if __name__ == "__main__":
    # Example: Man City vs Southampton
    # City expected ~2.2 goals, Southampton ~0.8 goals
    home_l, away_l = 2.2, 0.8
    mat = build_goal_matrix(home_l, away_l, rho=-0.13)
    goal_dist = total_goals_distribution(mat)

    print(f"Expected total: {home_l + away_l:.1f}")
    print(f"\nGoal distribution:")
    for g in range(8):
        print(f"  {g} goals: {goal_dist[g]*100:.1f}%")

    print(f"\nLine probabilities:")
    for line in [0.5, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.5]:
        probs = prob_over_line(goal_dist, line)
        push_str = f"  push={probs['p_push']*100:.1f}%" if probs["p_push"] > 0.001 else ""
        type_str = f"  [{probs['type']}]"
        print(f"  {line:>4}: Over {probs['p_over']*100:.1f}%  Under {probs['p_under']*100:.1f}%{push_str}{type_str}")

    # Test settlement
    print(f"\nSettlement tests (Over 2.25 @ 1.90, 3 goals):")
    s = settle_asian("over", 2.25, 3, 1.90, stake=10)
    print(f"  {s}")

    print(f"Settlement tests (Over 2.25 @ 1.90, 2 goals):")
    s = settle_asian("over", 2.25, 2, 1.90, stake=10)
    print(f"  {s}")

    print(f"Settlement tests (Under 2.75 @ 1.85, 3 goals):")
    s = settle_asian("under", 2.75, 3, 1.85, stake=10)
    print(f"  {s}")
