# 3. Exchange execution, commission, and commission-aware minimum odds

Date: 2026-06-24

## Status

Accepted (spec agreed via `/grill-with-docs`; implementation pending — see Consequences).
Sequenced **behind** pre-live hardening item #5 (settlement concurrency), which rewrites the
same settlement profit path.

## Context

The system is moving from paper validation toward live betting at scale. Soft bookmakers
limit or ban winning bettors, which is the practical death of most profitable models — so
the long-term execution venue is the **betting exchanges** (Betfair, Matchbook), which
profit on commission regardless of who wins and therefore do not limit winners. Exchanges
also allow earlier market access.

This forces three things the system does not currently handle:

1. **Commission.** Soft books charge none; exchanges charge a fee on winnings. The system
   models no commission anywhere. Every backtest ROI figure (PL O/U 2.5 +8%, EFL O/U 2.5
   +1.2%, …) is therefore **gross**, and settlement (`settlement.py`) computes gross P&L.
   Thin edges that look positive gross can be negative net of commission.
2. **A fair-price reference vs an execution venue.** On a soft book the price you get and
   the sharp fair price are genuinely different numbers. On an exchange they are nearly the
   same number, which makes it tempting — and wrong — to grade a bet against the very price
   you execute into (circular; erases edge by construction).
3. **A way to act at the exchange.** The user wants to place limit orders at or above the
   worst price still worth taking, rather than only taking whatever is currently offered.

A code check found the O/U 2.5 backtest (`backtest.py:416`) uses **Bet365** odds for both
the fair reference and execution, gross — neither the live base (Pinnacle) nor the intended
execution venue (Betfair). So "re-run with commission" is really "re-source the backtest
onto Betfair."

## Decision

1. **The exchange is an Execution Venue, not the Fair Probability source.** Its price feeds
   `odds` → EV → Kelly and the Minimum Odds, **net of commission**. The Fair Probability
   (edge benchmark) stays an independent sharp reference. Live source hierarchy:
   **(1) Pinnacle** (preferred), **(2) Exchange de-vigged midpoint** as fallback (accepting
   the circularity only when Pinnacle is unavailable), **(3) soft de-vig** (discounted 80%).
   Keeping Pinnacle as an independent base is the antidote to grading a bet against the venue
   you trade on.

2. **Commission is a first-class, per-venue config constant** applied to net winnings:
   **Betfair 5%** (new account, no discount earned), **Matchbook 2%**. It surfaces in exactly
   two places — **(a) pre-bet** it sets the per-venue Minimum Odds shown on the dashboard for
   *both* exchanges; **(b) post-bet** it is deducted from winnings when a **logged bet** is
   settled, using the venue actually chosen, with that rate snapshotted onto the settled row.
   The advisory/paper track (`recommendations`, `predictions`) is **not** commission-netted;
   `logged_bets` is the net-of-fees source of truth for real P&L.

3. **Minimum Odds is anchored on `blended_prob`**: `O_min = 1 + (1 − p_blended)/(p_blended·(1 − c))`.
   It is the commission-aware break-even floor (never accept below it); the **Post Target** is
   a strictly higher price that preserves the required edge after fill risk / adverse selection,
   and is what you actually post. Both are computed **per venue** and displayed for both. Minimum
   Odds is an **execution floor for already-qualified Recommendations, not a selection filter** —
   the Edge/Agreement gates still decide what becomes a Recommendation.

4. **Execution is advisory/manual.** The system computes and surfaces Minimum Odds + Post
   Target; the human places and manages the order, recording the chosen venue. **Automated
   order placement is out of scope**, deferred behind item #5 and a future security review —
   uncoordinated processes placing real exchange orders would amplify exactly the concurrency
   hazard #5 exists to contain.

5. **The backtest is re-sourced onto Betfair-closing**, net of the conservative 5% (Betfair)
   commission, used as both the base/reference and the execution price (historical Pinnacle is
   not available beyond O/U 2.5). This makes the backtest a conservative *"does the model beat
   the Betfair closing line, net of fees?"* test. Live keeps the Pinnacle base and bets earlier
   than the close, so live performance is expected to be **at least** as good (the earlier price
   is the upside; CLV measures it). The historical Pinnacle O/U 2.5 data is used to *validate*
   that Betfair-closing ≈ Pinnacle-closing, not as a competing base.

6. **Markets are re-ranked Active vs Monitored from the commission-net backtest.** A
   **Monitored Market** is still priced and calibration-tracked but never staked. EFL O/U 2.5
   (~+1.2% gross, negative after 5%) is the first; the final Active set is data-driven from the
   re-run and is expected to shrink toward PL markets + EFL BTTS.

7. **The recommendation engine is unchanged.** The ensemble, blend (35/65), edge definition,
   agreement, Kelly, drawdown, and market multipliers are untouched — this ADR adds an
   execution layer around them, it does not alter strategy. The only change to *what the user
   sees* is a smaller Active market set (a market-status change, not a logic change).

## Consequences

- A single **shared commission helper** must be authored and used by *both* settlement (live
  net P&L) and the backtests (validation), so "how commission works" lives in one place and
  cannot drift between what is backtested and what is realised.
- `settlement.py`'s profit path becomes commission-aware (net winnings, rate snapshotted).
  This edits the same code item #5 rewrites for idempotency — **do #5 first**, then layer
  commission on top, to avoid rewriting the bankroll-write path twice.
- `recommendations` gains an `exchange_targets_json` column (per-venue min-odds + post-target),
  matching the existing `per_model_json` idiom. Venue logging reuses `logged_bets.bookmaker`
  (no new column). A `commission_rate` snapshot is added to the settled logged-bet row.
- Dashboard changes are additive: per-venue floors/targets on Active Picks, a venue selector
  when logging a bet, net-of-commission performance, and Monitored-market badges. Goes through
  the `dashboard-tester` render-regression check.
- The soft-book odds APIs are **not** retired — their role narrows to the independent Pinnacle
  base (OddsPapi) and opportunistic soft-book price scouting (time-limited, sunset as scaling
  brings limits). Near-term, soft-book bulk fetch usage can be trimmed to save quota.
- Money-adjacent + strategy-touching work → TDD throughout, and any change to blend weight,
  edge definition, or agreement remains gated behind explicit user sign-off (none is proposed
  here).

## Considered alternatives

- **Exchange as the fair-price source (Betfair-only base).** Rejected: grading a bet against
  the price you execute into is circular and collapses edge to the bid-ask spread. Pinnacle as
  an independent base is retained.
- **Automated order placement now.** Rejected for v1: highest-risk surface in the system,
  amplifies item #5's concurrency hazard, and must validate the commission/min-odds math
  against real fills first.
- **Backtest at the optimistic 2% (Matchbook).** Rejected: Matchbook liquidity is not
  guaranteed (thin on EFL/alt-lines), so the conservative 5% gate is what a market must clear
  to earn Active status; 2%-only survivors are flagged fragile, not Active.
- **Net the paper track too.** Deferred: `logged_bets` is the real-money truth; the paper
  track stays gross as a diagnostic upper bound.
