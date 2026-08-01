"""
Team name normalization across data sources.
All functions return the canonical name used in CompleteDSPL_CSV.csv (e.g. "Arsenal FC").

An alias is a whole name for the club: "Man Utd" for Manchester United,
"Spurs" for Tottenham, "ARS" for Arsenal. Matching is on whole names only.

It used to fall back to substring matching, which is a different thing
entirely — it took a club's long name apart and matched a fragment. Every
short code sits inside some real name: "BUR" is in "Blackburn", "CHE" in
"Manchester City", "STO" in "Bristol Rovers", "MIL" in "Milton Keynes Dons".
So "Bristol Rovers" normalised to Stockport County and "Manchester" to
Chelsea. A code is a name for one club, never a fragment of another's.
A name this table does not recognise is returned untouched, and warned about.
Use `is_known_team()` to tell the two apart before trusting the result.
"""
import logging

logger = logging.getLogger(__name__)

# Canonical names in CompleteDSPL_CSV.csv -> aliases from other sources
_ALIASES = {
    "Arsenal FC": ["Arsenal", "ARS"],
    "Aston Villa FC": ["Aston Villa", "Villa", "AVL"],
    "AFC Bournemouth": ["Bournemouth", "BOU"],
    "Brentford FC": ["Brentford", "BRE"],
    "Brighton & Hove Albion FC": ["Brighton", "Brighton and Hove Albion", "BHA"],
    "Burnley FC": ["Burnley", "BUR"],
    "Chelsea FC": ["Chelsea", "CHE"],
    "Crystal Palace FC": ["Crystal Palace", "C Palace", "Palace", "CRY"],
    "Everton FC": ["Everton", "EVE"],
    "Fulham FC": ["Fulham", "FUL"],
    "Ipswich Town FC": ["Ipswich", "Ipswich Town", "IPS"],
    "Leeds United FC": ["Leeds", "Leeds United", "LEE"],
    "Leicester City FC": ["Leicester", "Leicester City", "LEI"],
    "Liverpool FC": ["Liverpool", "LIV"],
    "Luton Town FC": ["Luton", "Luton Town", "LUT"],
    "Manchester City FC": ["Man City", "Manchester City", "MCI"],
    "Manchester United FC": ["Man United", "Manchester United", "Man Utd", "Manchester Utd", "MUN"],
    "Newcastle United FC": ["Newcastle", "Newcastle United", "Newcastle Utd", "NEW"],
    "Nottingham Forest FC": ["Nottingham Forest", "Nott'ham Forest", "Nott'm Forest", "Nottm Forest", "Forest", "NFO"],
    "Sheffield United FC": ["Sheffield United", "Sheffield Utd", "Sheff Utd", "Sheff United", "SHU"],
    "Southampton FC": ["Southampton", "SOU"],
    "Tottenham Hotspur FC": ["Tottenham", "Spurs", "Tottenham Hotspurs", "TOT"],
    "Watford FC": ["Watford", "WAT"],
    "West Ham United FC": ["West Ham", "West Ham United", "West Ham Utd", "WHU"],
    "Wolverhampton Wanderers FC": ["Wolverhampton Wanderers", "Wolverhampton", "Wolves", "WOL"],
    "West Bromwich Albion FC": ["West Bromwich Albion", "West Brom", "West Bromwich", "WBA"],
    "Norwich City FC": ["Norwich", "Norwich City", "NOR"],
    "Swansea City AFC": ["Swansea", "Swansea City", "SWA"],
    "Stoke City FC": ["Stoke", "Stoke City", "STK"],
    "Sunderland AFC": ["Sunderland", "SUN"],
    "Hull City AFC": ["Hull", "Hull City", "HUL"],
    "Cardiff City FC": ["Cardiff", "Cardiff City", "CAR"],
    "Queens Park Rangers FC": ["Queens Park Rangers", "QPR"],
    "Reading FC": ["Reading", "REA"],
    "Wigan Athletic FC": ["Wigan Athletic", "Wigan", "WIG"],
    "Bolton Wanderers FC": ["Bolton Wanderers", "Bolton", "BOL"],
    "Blackburn Rovers FC": ["Blackburn Rovers", "Blackburn", "BLB"],
    "Birmingham City FC": ["Birmingham City", "Birmingham", "BIR"],
    "Charlton Athletic FC": ["Charlton Athletic", "Charlton", "CHA"],
    "Derby County FC": ["Derby County", "Derby", "DER"],
    "Middlesbrough FC": ["Middlesbrough", "Boro", "MID"],
    "Portsmouth FC": ["Portsmouth", "POR"],
    "Blackpool FC": ["Blackpool", "BLP"],
    "Huddersfield Town AFC": ["Huddersfield", "Huddersfield Town", "HUD"],
    "Bradford City AFC": ["Bradford", "Bradford City", "BRA"],
    # ── Championship-specific teams (not in PL during 2000-2025) ──
    "Millwall FC": ["Millwall", "MIL"],
    "Plymouth Argyle FC": ["Plymouth", "Plymouth Argyle", "PLY"],
    "Preston North End FC": ["Preston", "Preston North End", "PNE"],
    "Rotherham United FC": ["Rotherham", "Rotherham United", "ROT"],
    "Barnsley FC": ["Barnsley", "BAR"],
    "Coventry City FC": ["Coventry", "Coventry City", "COV"],
    "Bristol City FC": ["Bristol City", "BRC"],
    "Crewe Alexandra FC": ["Crewe", "Crewe Alexandra", "CRE"],
    "Gillingham FC": ["Gillingham", "GIL"],
    "Grimsby Town FC": ["Grimsby", "Grimsby Town", "GRI"],
    "Wimbledon FC": ["Wimbledon", "WIM"],
    "Stockport County FC": ["Stockport", "Stockport County", "STO"],
    "Walsall FC": ["Walsall", "WAL"],
    "Sheffield Wednesday FC": ["Sheffield Wednesday", "Sheffield Weds", "Sheff Wed", "Sheff Weds", "SHW"],
    "Peterborough United FC": ["Peterborough", "Peterboro", "Peterborough United", "PET"],
    "Doncaster Rovers FC": ["Doncaster", "Doncaster Rovers", "DON"],
    "Scunthorpe United FC": ["Scunthorpe", "Scunthorpe United", "SCU"],
    "Colchester United FC": ["Colchester", "Colchester United", "COL"],
    # Southend's own code is SOU, which Southampton holds. Two clubs cannot
    # share one alias, so Southend keeps its names and no code.
    "Southend United FC": ["Southend", "Southend United"],
    "Yeovil Town FC": ["Yeovil", "Yeovil Town", "YEO"],
    "MK Dons FC": ["MK Dons", "Milton Keynes Dons", "MKD"],
    "Burton Albion FC": ["Burton", "Burton Albion", "BUA"],
    "Oxford United FC": ["Oxford", "Oxford United", "OXF"],
    "Wycombe Wanderers FC": ["Wycombe", "Wycombe Wanderers", "WYC"],
    "Exeter City FC": ["Exeter", "Exeter City", "EXE"],
    "Tranmere Rovers FC": ["Tranmere", "Tranmere Rovers", "TRA"],
    # In the Championship from 2025/26. The odds feeds send "Wrexham AFC".
    "Wrexham AFC": ["Wrexham", "WRE"],
}

# FC, AFC and CF are decoration, not part of the name: "Arsenal" and
# "Arsenal FC" are one club, and so are "AFC Bournemouth" and "Bournemouth".
_CLUB_AFFIXES = frozenset({"fc", "afc", "cf"})

def _lookup_key(name: str) -> str:
    """Comparison key: case-folded, with any FC/AFC/CF dropped.

    Lets "Bournemouth", "AFC Bournemouth" and "Bournemouth FC" meet without
    each having to be listed. It compares whole words only — the fragment
    matching this replaced is what resolved "Bristol Rovers" to Stockport.
    """
    words = [w for w in name.lower().split() if w not in _CLUB_AFFIXES]
    return " ".join(words)


# Build reverse lookup: alias -> canonical
_REVERSE = {}
for canonical, aliases in _ALIASES.items():
    _REVERSE[canonical.lower()] = canonical
    for alias in aliases:
        _REVERSE[alias.lower()] = canonical

# The same lookup with club affixes dropped. A key two clubs could both claim
# is dropped rather than awarded to whichever was inserted first.
_BY_KEY: dict[str, str] = {}
_ambiguous: set[str] = set()
for _name, _canonical in _REVERSE.items():
    _key = _lookup_key(_name)
    if _BY_KEY.setdefault(_key, _canonical) != _canonical:
        _ambiguous.add(_key)
for _key in _ambiguous:
    del _BY_KEY[_key]


def _resolve(name: str) -> str | None:
    """The canonical name, or None when this table has never been taught it."""
    clean = name.strip().rstrip("*").strip()
    direct = _REVERSE.get(clean.lower())
    if direct is not None:
        return direct
    return _BY_KEY.get(_lookup_key(clean))


def is_known_team(name: str) -> bool:
    """Whether `name` resolves to a club.

    `normalize()` hands back what it was given when it cannot resolve it, so
    its return value cannot answer this on its own: "Manchester" comes back
    looking exactly like a canonical club name, and no club is called that.
    Anything writing team names into stored data or matching them against
    stored data should ask this first.
    """
    return _resolve(name) is not None


def assert_known_teams(names, context: str) -> None:
    """Raise unless every name resolves to a club.

    For the boundaries where an unrecognised name cannot simply be skipped:
    anything writing team names into stored data. The canonical datasets are
    the training data, and a name written there is a team until someone
    rebuilds them — an unresolved one becomes a club of its own, distinct
    from every other appearance of the same side.

    Args:
        names: The names about to be stored.
        context: What is being built, named in the error.

    Raises:
        ValueError: Listing every name that did not resolve.
    """
    unknown = sorted({str(n) for n in names if not is_known_team(str(n))})
    if unknown:
        raise ValueError(
            f"{len(unknown)} team name(s) in {context} do not resolve to a "
            f"club: {unknown}. Add them to _ALIASES in api/team_mapping.py — "
            f"storing them unresolved would create a separate team per "
            f"spelling.")


# One warning per unrecognised name, not per row: pipeline.py normalises whole
# columns, so a single unmapped club would otherwise log thousands of times.
_unknown_names_logged: set[str] = set()


def _reset_unknown_name_log() -> None:
    """Forget which names have been warned about. For tests."""
    _unknown_names_logged.clear()


def normalize(name: str) -> str:
    """Convert any team name variant to the canonical CSV name.

    An unrecognised name is returned as it came in, and warned about once.
    Callers see the original string and can flag it; what they must never see
    is a different club's name in place of one this table has never held.

    The string it returns is not proof of anything — use `is_known_team()`
    before writing a name into stored data or matching one against it.
    """
    resolved = _resolve(name)
    if resolved is not None:
        return resolved

    clean = name.strip().rstrip("*").strip()
    if clean not in _unknown_names_logged:
        _unknown_names_logged.add(clean)
        logger.warning(
            "Unrecognised team name %r — returned unchanged. It will read as "
            "a team name downstream and match nothing. Add it to _ALIASES in "
            "api/team_mapping.py.", clean)
    return clean


def get_all_current_teams():
    """Return list of all canonical team names (useful for dropdowns)."""
    return sorted(_ALIASES.keys())


def from_fpl(fpl_name: str) -> str:
    """Normalize a team name from the FPL API."""
    return normalize(fpl_name)


def from_football_data(fd_name: str) -> str:
    """Normalize a team name from football-data.org API."""
    return normalize(fd_name)


if __name__ == "__main__":
    # Test
    tests = [
        "Arsenal", "Man City", "Man Utd", "Spurs", "Wolves",
        "Brighton", "Nott'ham Forest", "West Ham", "QPR",
        "Arsenal FC", "Chelsea *"
    ]
    for t in tests:
        print(f"  {t:30s} -> {normalize(t)}")
