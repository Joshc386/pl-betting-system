# 8. One team-resolution contract across odds feeds

Date: 2026-07-30

## Status

Accepted. Implemented 2026-07-30 (`e82d4d1`), following the wrong-club fix in
`f30d366`. Extended the same day to `api/team_mapping.py::normalize()`
(`25881a2`, `4cc05cf`).

Applies to odds-feed resolution the principle
[0007](0007-one-feature-contract-per-name.md) establishes for canonical
features.

`normalize()` keeps its own **resolution mechanism** — an alias table, rather
than feed names weighed against a live `our_teams` set — and is not
consolidated onto `resolve_feed_team`. It is now held to the same **contract**:
resolve to a club, or say so. Both halves are recorded in Consequences.

## Context

Every odds feed spells clubs its own way, so each feed carries a hand-maintained
mapping dict from its names to the names a league's Canonical Dataset holds.
Where the dict misses, a fuzzy word-overlap fallback guesses.

Three copies of that fallback existed:

| Function | Module | Feed |
|---|---|---|
| `match_to_our_teams` | `api/odds_api.py` | The-Odds-API |
| `map_team` | `api/oddspapi.py` | OddsPapi |
| `_resolve_champ_team` | `championship_predict.py` | both, Championship |

All three accepted `best_score >= 1` — any single shared word was a match. "City"
is shared by Manchester City, Leicester City, Hull City, Coventry City, Norwich
City and Stoke City. With the 2026/27 feed live, "Coventry City" resolved to
"Manchester City FC", so `Arsenal v Coventry City` would have been priced, staked
and recommended as `Arsenal v Manchester City`.

`f30d366` fixed that in the first two and left the third, which is the point of
this ADR. The three were *near*-duplicates, and the differences were real enough
to make the third look like a separate problem: it stripped the literal substring
`"city"` from the feed name before comparing, stripped `"'m"` from candidates,
and returned explicit mappings even for clubs absent from `our_teams`. Under
deadline pressure that reads as a third implementation to be dealt with
separately. It was the same contract wearing different clothes.

What made them look different was **name format**, which is genuinely per-feed
and per-league: the PL canonical holds `"Arsenal FC"`, the EFL canonical keeps
football-data.co.uk short forms like `"Blackburn"` and `"Nott'm Forest"`. The
*matching rule* was never per-feed. Nothing forced the three to agree, and they
did not.

The cost of the drift, measured after the fact: `_resolve_champ_team` was still
resolving an unmapped `"Sheffield Utd"` to `"Sheffield Weds"`. Worse than wrong,
it was non-deterministic — `overlap > best_score` keeps the *first* candidate
reaching the best score while iterating a `set`, so which Sheffield club came
back depended on hash ordering. This is the live path into recommendations and
Kelly staking.

## Decision

**The matching rule is one contract with one implementation.**
`api/team_resolver.py::resolve_feed_team(api_name, our_teams, explicit)` is that
implementation. The three functions above remain as named entry points and keep
their callers, but each is now a single line passing its own mapping dict.

1. **Per-feed data stays per-feed; per-feed logic does not exist.** The mapping
   dicts are feed-specific *data* and stay in their modules. The resolver is
   name-format agnostic by construction — it never sees a format, because format
   lives in the dicts and in `our_teams`. `test_the_resolver_is_name_format_agnostic`
   runs one feed name against both canonicals to hold that line.

2. **An explicit mapping is authoritative, even when the club is absent from
   `our_teams`.** A newly promoted side is unknown until its season is built into
   the canonical; downstream logs "no recent data" and skips it, then starts
   working by itself once the season lands. Falling through to the fuzzy path
   instead is what returned Manchester City for Coventry City. All three feeds
   had converged on this independently; consolidation recognised the agreement
   rather than imposing it.

3. **A distinctive word is required.** A name sharing only generic words
   (city, united, town, albion, wanderers, rovers, athletic, county, fc, afc,
   and, the) with a candidate does not match it. Generic words still count
   towards ranking once something distinctive has qualified a candidate.

4. **Mutual disagreement disqualifies.** Ignoring noise (fc/afc/and/the), if the
   feed name carries a word the candidate lacks *and* the candidate carries a
   word the feed name lacks, each asserts something the other denies: they are
   two clubs sharing a place name. Bristol Rovers is not Bristol City,
   Manchester United is not Manchester City, Sheffield United is not Sheffield
   Wednesday. The candidate is **dropped, not down-ranked** — a shared city is
   not a shared club at any score.

5. **No list of club surnames is kept.** "Weds", "Forest", "Hotspur" and
   "Argyle" are surnames exactly as "City" and "United" are. A list of them
   would be one more thing to keep current, which is the drift this ADR exists
   to end; the first implementation of decision 4 used one and immediately
   resolved "Sheffield United" to "Sheffield Weds" because "Weds" was not on it.
   A word the other name lacks is disagreement enough, whatever the word.

6. **A tie resolves to `None`, and `None` means skip the fixture.** No match is
   recoverable and visible; the wrong match is neither. Callers already skip
   unresolved fixtures.

Verified across all 109 names in the three mapping dicts: **no resolution
changes**. Full suite unchanged against the base commit — the 21 failures and 3
errors in `tests/test_championship.py` are pre-existing and unrelated.

## Consequences

- The wrong-club class of bug now has one place to be fixed and one place to
  regress. `tests/test_team_resolution.py` asserts identity — all three modules
  must hold the *same function object* — so a future third copy fails a test
  rather than going unnoticed for two days.

- **Abbreviations move to the mapping dicts.** Two EFL pairs no longer match on
  the fuzzy path: "Nottingham Forest"/"Nott'm Forest" and "West Bromwich
  Albion"/"West Brom". Both are explicitly mapped, so live behaviour is
  unchanged. Guessing that "Bromwich" means "Brom" is not the resolver's
  business, and an unmapped abbreviation is now skipped rather than guessed.

- **A club promoted into either league needs a mapping entry**, not a code
  change. That was already true; it is now the only true path.

- **`api/team_mapping.py::normalize()` remains a separate contract, but its
  version of this failure is now fixed** (`25881a2`). It maps *data-source*
  names (FPL, football-data.org, Understat, ESPN) to canonical names via an
  alias table, rather than feed names against a live `our_teams` set, so it is
  not consolidated onto `resolve_feed_team`. It had the same failure by a
  different mechanism: a substring fallback that found short alias codes inside
  longer names, returning `"Stockport County FC"` for `"Bristol Rovers"` (via
  `"sto"` in "bri**sto**l rovers") and `"Chelsea FC"` for `"Manchester"` (via
  `"che"` in "man**che**ster"). Of 22 real EFL clubs absent from the table, 11
  resolved to the wrong club.

  It was reachable, and by the same route as everything else in this ADR: a
  club absent from the table is what promotion produces, and
  [0007](0007-one-feature-contract-per-name.md) decision 3 already cites this
  fallback as what hid the Bradford City gap. Matching is now on whole names
  only — the codes stay, since a code is a name for its own club, but cannot be
  found inside another club's name.

- **An unresolved name is now distinguishable from a resolved one** (`4cc05cf`).
  Removing the fragment matching fixed the wrong answers but not the silence:
  `normalize()` returns its input when it cannot resolve it, so `"Manchester"`
  — a city no club is called — came back looking exactly like a canonical team
  name. Half the contract was still missing. No caller could tell the two
  apart, and the string went on to be written into the canonical datasets, used
  as a pipeline merge key, and matched against open bets.

  `is_known_team()` answers the question the return value cannot;
  `assert_known_teams()` raises, naming what did not resolve; and `normalize()`
  warns once per unrecognised name — once per *name*, not per call, because
  `pipeline.py` normalises whole columns and a single unmapped club would
  otherwise log thousands of times and be worse than silence.

  Strict where an unknown cannot simply be skipped:

  | Boundary | Why it cannot skip |
  |---|---|
  | `data/build_canonical_dataset.py`, `data/add_season.py` | The canonical is the training data. A name stored unresolved becomes a team of its own, for every spelling the feed uses. |
  | `api/espn_scores.py` | Settlement keys bets on `(home_team, away_team)`. A name in the wrong format matches nothing, so the bet stays open with nothing said. |

  A guard on a build path has to be verified against the build, not only
  against tests. Two defects surfaced only that way, both in code the tests
  were green on: `E0_1415.csv` carries a blank row, which `astype(str)` renders
  as the string `"nan"` — checking the mapped column rather than the source
  values would have failed every rebuild on a file that has always been fine —
  and normalising that `"nan"` logged an unrecognised-team warning on every
  run, which is how a team learns to ignore warnings.

- **The ESPN Championship resolver had no valid fallback and now has none.**
  An EFL club missing from `_ESPN_TO_CHAMP` fell through to `normalize()`,
  which answers in Premier League canonical names (`"Blackburn Rovers FC"`)
  while the EFL database holds football-data.co.uk short forms
  (`"Blackburn"`). Those cannot match, so such a club's bets were
  unsettleable and nothing said so. Masked only because that hand-maintained
  25-entry map happens to cover the current division — the next promoted club
  would have hit it. Unresolved now returns `None`, and the match is skipped
  and logged.

  **The shared principle, stated once:** a name resolves to a club or it
  resolves to nothing, and *nothing* must be visible as nothing. Fragments of a
  name are not evidence about which club it is — not a shared city
  ("Bristol"), a shared surname ("City"), or a substring that happens to spell
  another club's code ("sto") — and a name that resolved to nothing must not be
  returned in a shape that reads as an answer.
