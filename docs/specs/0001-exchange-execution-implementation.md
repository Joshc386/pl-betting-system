# Spec: Exchange execution + commission-aware minimum odds

**Status:** Approved spec, not started. Implementation gated behind pre-live hardening item #5.
**Decisions:** see [ADR 0003](../adr/0003-exchange-execution-and-commission.md).
**Vocabulary:** see `CONTEXT.md` — Execution Venue, Commission, Minimum Odds, Post Target, Monitored Market.
**Approach:** TDD throughout (money-adjacent). No strategy logic (blend weight, edge, agreement, Kelly) changes — those stay gated behind explicit sign-off.

## Guiding principle

The recommendation engine is untouched. Every item below is an **execution layer** wrapped
around it, plus a backtest re-source and a market re-rank. If any item finds itself editing
the ensemble / blend / edge / agreement / Kelly, stop — that is out of scope.

---

## Phase 0 — Prerequisite (separate work)

- [ ] **Item #5 ships first** (settlement idempotency + quota filelock). The commission-net
      settlement change (Phase 2) edits the same profit path; doing #5 first avoids rewriting
      the bankroll-write path twice.

## Phase 1 — Commission model (shared helper)

- [ ] Config constants: `BETFAIR_COMMISSION = 0.05`, `MATCHBOOK_COMMISSION = 0.02`, plus a
      `VENUE_COMMISSION = {"betfair": 0.05, "matchbook": 0.02}` lookup.
- [ ] A single pure helper, e.g. `net_profit(stake, odds, won, commission)` →
      `stake*(odds-1)*(1-commission)` on a win, `-stake` on a loss. **Used by both settlement
      and the backtests.**
- [ ] A `min_odds(p_blended, commission)` helper → `1 + (1-p)/(p*(1-c))`, and a
      `post_target(p_blended, commission, required_edge)` helper (price clearing the required
      net edge).
- **Tests:** known p/odds/commission → known net profit and known O_min; c=0 collapses to
      gross / fair odds; both venues.

## Phase 2 — Commission-aware settlement (logged bets only)

- [ ] `settle_bets` (logged-bet path) computes **net** P&L via the Phase-1 helper, keyed off
      `logged_bets.bookmaker` (the chosen venue).
- [ ] Add `commission_rate REAL` to `logged_bets` (migration, same pattern as the CLV columns);
      snapshot the venue rate onto the row at settlement (immutable financial record).
- [ ] Paper track (`recommendations`, `predictions`) settlement stays **gross** — unchanged.
- **Tests:** Betfair win nets 5% off winnings; Matchbook win nets 2%; losses unaffected;
      rate is snapshotted; bankroll reflects net. Build on top of #5's idempotent settle.

## Phase 3 — Per-venue Minimum Odds + Post Target

- [ ] During scan/predict, for each Recommendation compute Minimum Odds + Post Target for
      **both** venues from `blended_prob` + each venue commission.
- [ ] Store as `exchange_targets_json` on `recommendations` (new column, JSON, matching the
      `per_model_json` idiom): `{"betfair":{"min_odds":…,"post_target":…},"matchbook":{…}}`.
- [ ] This is an annotation on already-qualified Recommendations — it does **not** gate
      selection.
- **Tests:** floors differ by venue (Betfair floor > Matchbook floor for the same bet);
      post_target > min_odds; JSON round-trips.

## Phase 4 — Dashboard (additive)

- [ ] Active Picks: show per-venue floor + post target per Recommendation.
- [ ] Bet logging: venue selector (Betfair/Matchbook) → `logged_bets.bookmaker`.
- [ ] Performance/P&L: net-of-commission figures; optional per-venue breakdown.
- [ ] Monitored markets: clear "monitor-only, not staked" badge; keep calibration display.
- **Tests:** `dashboard-tester` render-regression pass; no silent-empty sections.

## Phase 5 — Backtest re-source onto Betfair

- [ ] Re-source `backtest.py` (O/U 2.5) off Bet365 onto **Betfair-closing** for base +
      execution; apply the Phase-1 commission helper at the conservative **5%**.
- [ ] Confirm BTTS / alt-line backtests use Betfair-closing + commission consistently (they
      already read `betfair_goal_ou.csv` etc.).
- [ ] **Base-proxy validation:** using the historical Pinnacle O/U 2.5 odds
      (football-data.co.uk), confirm Betfair-closing ≈ Pinnacle-closing, so Betfair-closing is
      a sound base for the markets without Pinnacle. Record the result.
- **Tests:** regression test on a fixed slice (known Betfair odds + outcome → known net ROI).

## Phase 6 — Market re-rank (Active vs Monitored)

- [ ] Run all market backtests net of 5%; produce the Active/Monitored table.
- [ ] Set EFL O/U 2.5 → **Monitored** (confirmed); set others per the re-run (EFL O/U 1.5
      likely Monitored; PL O/U 2.5 / BTTS / O/U 1.5 Over and EFL BTTS likely Active).
- [ ] Flag any "Matchbook-only" (survives 2%, not 5%) markets distinctly.
- **Output:** the data-driven Active set. This is a **market-status** change (approved), not a
      strategy-logic change.

---

## Out of scope (named, not forgotten)

- Automated order placement on the exchange (gated behind #5 + security review).
- Unfilled-order / fill-rate / adverse-selection measurement (revisit at the automated phase).
- Per-market commission nuance and Betfair's discount-rate machinery (single per-venue rate
  for now).
- A live Betfair Exchange price feed (future; OddsPapi-Pinnacle remains the live base).
- Any change to blend weight, edge definition, agreement, or Kelly (strategy-locked).

## Suggested skills

- `/tdd` for every phase (money-adjacent).
- `/gstack-cso` before this goes from paper to live capital.
- `dashboard-tester` agent for Phase 4.
