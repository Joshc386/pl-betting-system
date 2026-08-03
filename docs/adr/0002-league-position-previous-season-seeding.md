# 2. League position seeded by previous-season outcome at matchday 1

Date: 2026-06-13

## Status

Accepted. **Implemented 2026-08-02** in the shared builder
(`_add_league_position` / `_matchday1_seeds`, ADR 0007 decision 3), with
three conventions this document left open, recorded in `CONTEXT.md`:
sides relegated into the EFL seed 1, 2, 3 in their PL finishing order;
the neutral promoted seed is 2nd-from-bottom; the dataset's first season
(nothing to seed from) keeps the alphabetical order as its documented
default. `PROMOTED_TEAMS` was never extended for routes — it was deleted
(decision 10), and routes derive from the sibling canonical's final table
instead.

## Context

`Home_LeaguePosition` / `Away_LeaguePosition` (and `LeaguePosition_Diff`) are model
features in both the PL and EFL pipelines. They represent a team's league rank *before*
the fixture is played.

Mid-season this is well-defined: rank by points, then goal difference, then goals scored.
But at **matchday 1, before any games are played, every team is on zero points**, so rank
is undefined. The build scripts historically resolved this arbitrarily — one version ranked
only teams that had already appeared in the fixture list (giving season-opener teams
positions like 1–2 despite zero games), another defaulted unplayed teams to mid-table (12).
Both are noise: they encode no real information about relative strength at season start.

This surfaced during the EFL 2025/26 data refresh, where ~92 of the largest historical
league-position diffs between two build-script versions were season openers (zero games
played) — confirming the matchday-1 value was arbitrary, not a real signal.

## Decision

At matchday 1 (zero games played), league position is **seeded by the previous season's
outcome**, not left undefined:

- **Returning teams** are seeded at their **finishing position from the previous season**.
- **Promoted teams** (up from the division below, so no previous-season position in this
  league) are seeded at the bottom by **promotion route**:
  - Division-below **champions** → 3rd-from-bottom (22nd EFL Championship / 18th PL)
  - **Runners-up** (2nd, auto-promoted) → 2nd-from-bottom (23rd EFL / 19th PL)
  - **Play-off winners** → bottom (24th EFL / 20th PL)

Once games are played, position is the live points table as before. The seed also serves as
the tie-breaker at equal points (replacing the alphabetical tie-break), so it continues to
carry mild signal beyond matchday 1.

Rationale: the previous-season finish is the best available prior for relative team strength
at season start — it is the convention bookmakers and football data providers use. Promoted
teams are genuinely the weakest-priored sides, and promotion route is a real (if weak)
strength signal (champions > runners-up > play-off winners on average). This replaces pure
noise with an informative prior in the exact region (early season) where the model otherwise
has the least information.

## Consequences

- Both build scripts (`data/build_championship_dataset.py` for EFL, and the PL equivalent)
  must compute end-of-season standings for season N-1 and seed season N's matchday-1 table
  from them.
- `PROMOTED_TEAMS` must be extended from a flat set of names to encode **promotion route**
  (champion / runner-up / play-off winner) per team per season, for every season where the
  seed matters. Seasons lacking route data fall back to a neutral promoted-team seed.
- The very first season in the dataset (no prior season) needs a documented default seed.
- This is a cross-league feature-engineering change that alters the matchday-1 rows of every
  historical season; it warrants test-first development (known previous-season tables → known
  seeds) and a retrain of both leagues afterward.
- Because the change is confined to early-season rows and the affected feature is one of many,
  the expected model impact is modest but directionally positive (replacing noise with a prior).
