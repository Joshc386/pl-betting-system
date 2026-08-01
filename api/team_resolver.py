"""One resolver from odds-feed team names to canonical names.

Every odds feed spells clubs its own way, and every league canonical spells
them differently again — the PL canonical holds "Arsenal FC", the EFL one
keeps football-data.co.uk short forms like "Blackburn". Each feed therefore
carries its own explicit mapping dict. The *matching rule* is not per-feed:
it is one contract, and it lives here.

It did not always. Three near-identical resolvers drifted apart, the
wrong-club bug was found and fixed in two of them, and the third kept
resolving "Sheffield Utd" to Sheffield Wednesday. That is the failure
docs/adr/0007-one-feature-contract-per-name.md documents for canonical
features: one name, two implementations, nothing forcing them to agree.

The rule, in order:

  1. An explicit mapping is authoritative, even when the club is not in
     `our_teams`. A newly promoted side is unknown until its season is built
     into the canonical; downstream logs "no recent data" and skips it, then
     starts working by itself once the season lands.
  2. An exact name match.
  3. Word overlap — but only where the overlap actually identifies a club.

Resolution is either correct or absent. A missing fixture is visible; a
confident prediction for the wrong fixture is not, and this path feeds
recommendations and Kelly staking.
"""
from __future__ import annotations

from typing import Iterable, Mapping

# Words too common to identify a club on their own. "City" is shared by
# Manchester City, Leicester City, Hull City, Coventry City, Norwich City and
# Stoke City; "United" is no better.
_GENERIC_TEAM_WORDS = frozenset({
    "city", "united", "town", "albion", "wanderers", "rovers",
    "athletic", "county", "fc", "afc", "and", "the",
})

# Carry no meaning at all: present or absent, the club is the same. Sunderland
# AFC and Sunderland FC are one club, so these can never signal a disagreement.
_NOISE_WORDS = frozenset({"fc", "afc", "and", "the"})


def _team_words(name: str) -> set[str]:
    """Lower-cased word set for comparison, with '&' folded to 'and'."""
    return {w for w in name.lower().replace("&", " and ").split() if w}


def _resolve_by_overlap(api_name: str, our_teams: Iterable[str]) -> str | None:
    """Best candidate sharing a distinctive word — None if absent or ambiguous.

    Two rules do the work.

    *A distinctive word is required.* Requiring one is what stops "Coventry
    City" matching "Manchester City FC" on "City" alone.

    *Mutual disagreement disqualifies.* Ignoring noise, if the feed name
    carries a word the candidate lacks and the candidate carries a word the
    feed name lacks, each is asserting something the other denies — they are
    two clubs sharing a place name. Bristol Rovers is not Bristol City;
    Manchester United is not Manchester City; Sheffield United is not
    Sheffield Wednesday. The candidate is out, rather than merely losing on
    score, because a shared city is not a shared club at any score.

    Note what this deliberately does not do: it keeps no list of club
    surnames. "Weds", "Forest", "Hotspur" and "Argyle" are surnames just as
    much as "City" and "United" are, and a list of them would be one more
    thing to keep current — which is how the resolvers this replaces drifted.
    A word the other name lacks is disagreement enough, whatever the word.

    The price is that an abbreviation the two names spell differently
    ("Nottingham Forest" against "Nott'm Forest") no longer resolves here.
    Those belong in the feed's mapping dict, where they already are, and
    refusing to guess that "Bromwich" means "Brom" is the safer failure.

    Whatever survives is ranked by total overlap, generic words included. A
    tie returns None: no match is recoverable, the wrong match is not.
    """
    api_words = _team_words(api_name)
    distinctive = api_words - _GENERIC_TEAM_WORDS
    if not distinctive:
        return None
    api_meaning = api_words - _NOISE_WORDS

    scored: list[tuple[int, str]] = []
    for candidate in our_teams:
        candidate_words = _team_words(candidate)
        if not (distinctive & candidate_words):
            continue
        candidate_meaning = candidate_words - _NOISE_WORDS
        if (api_meaning - candidate_meaning) and (candidate_meaning - api_meaning):
            continue
        scored.append((len(api_words & candidate_words), candidate))

    if not scored:
        return None
    best = max(score for score, _ in scored)
    winners = [team for score, team in scored if score == best]
    return winners[0] if len(winners) == 1 else None


def resolve_feed_team(
    api_name: str,
    our_teams: Iterable[str],
    explicit: Mapping[str, str],
) -> str | None:
    """Resolve one odds-feed team name against one league's canonical names.

    Args:
        api_name: Team name as the feed spells it.
        our_teams: Canonical names the league's dataset holds. Format is the
            caller's business — long PL names or EFL short forms both work.
        explicit: That feed's hand-maintained name mapping.

    Returns:
        The canonical name, or None when no name resolves unambiguously.
        Callers skip a None: pricing the wrong fixture is far worse than
        pricing none.
    """
    if api_name in explicit:
        return explicit[api_name]
    if api_name in our_teams:
        return api_name
    return _resolve_by_overlap(api_name, our_teams)
