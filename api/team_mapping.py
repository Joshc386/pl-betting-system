"""
Team name normalization across data sources.
All functions return the canonical name used in CompleteDSPL_CSV.csv (e.g. "Arsenal FC").
"""

# Canonical names in CompleteDSPL_CSV.csv -> aliases from other sources
_ALIASES = {
    "Arsenal FC": ["Arsenal", "ARS"],
    "Aston Villa FC": ["Aston Villa", "AVL"],
    "AFC Bournemouth": ["Bournemouth", "BOU"],
    "Brentford FC": ["Brentford", "BRE"],
    "Brighton & Hove Albion FC": ["Brighton", "Brighton and Hove Albion", "BHA"],
    "Burnley FC": ["Burnley", "BUR"],
    "Chelsea FC": ["Chelsea", "CHE"],
    "Crystal Palace FC": ["Crystal Palace", "C Palace", "CRY"],
    "Everton FC": ["Everton", "EVE"],
    "Fulham FC": ["Fulham", "FUL"],
    "Ipswich Town FC": ["Ipswich", "Ipswich Town", "IPS"],
    "Leeds United FC": ["Leeds", "Leeds United", "LEE"],
    "Leicester City FC": ["Leicester", "Leicester City", "LEI"],
    "Liverpool FC": ["Liverpool", "LIV"],
    "Luton Town FC": ["Luton", "Luton Town", "LUT"],
    "Manchester City FC": ["Man City", "Manchester City", "MCI"],
    "Manchester United FC": ["Man United", "Manchester United", "Man Utd", "Manchester Utd", "MUN"],
    "Newcastle United FC": ["Newcastle", "Newcastle United", "NEW"],
    "Nottingham Forest FC": ["Nottingham Forest", "Nott'ham Forest", "Nott'm Forest", "Nottm Forest", "NFO"],
    "Sheffield United FC": ["Sheffield United", "Sheffield Utd", "SHU"],
    "Southampton FC": ["Southampton", "SOU"],
    "Tottenham Hotspur FC": ["Tottenham", "Spurs", "TOT"],
    "Watford FC": ["Watford", "WAT"],
    "West Ham United FC": ["West Ham", "West Ham United", "WHU"],
    "Wolverhampton Wanderers FC": ["Wolverhampton Wanderers", "Wolverhampton", "Wolves", "WOL"],
    "West Bromwich Albion FC": ["West Bromwich Albion", "West Brom", "WBA"],
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
    "Middlesbrough FC": ["Middlesbrough", "MID"],
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
    "Sheffield Wednesday FC": ["Sheffield Wednesday", "Sheffield Weds", "Sheff Wed", "SHW"],
    "Peterborough United FC": ["Peterborough", "Peterboro", "PET"],
    "Doncaster Rovers FC": ["Doncaster", "DON"],
    "Scunthorpe United FC": ["Scunthorpe", "SCU"],
    "Colchester United FC": ["Colchester", "COL"],
    "Southend United FC": ["Southend", "SOU_EFL"],
    "Yeovil Town FC": ["Yeovil", "Yeovil Town", "YEO"],
    "MK Dons FC": ["MK Dons", "Milton Keynes Dons", "MKD"],
    "Burton Albion FC": ["Burton", "Burton Albion", "BUA"],
    "Oxford United FC": ["Oxford", "Oxford United", "OXF"],
    "Wycombe Wanderers FC": ["Wycombe", "Wycombe Wanderers", "WYC"],
    "Exeter City FC": ["Exeter", "Exeter City", "EXE"],
}

# Build reverse lookup: alias -> canonical
_REVERSE = {}
for canonical, aliases in _ALIASES.items():
    _REVERSE[canonical.lower()] = canonical
    for alias in aliases:
        _REVERSE[alias.lower()] = canonical


def normalize(name: str) -> str:
    """Convert any team name variant to the canonical CSV name."""
    # Strip whitespace and promoted-team asterisk
    clean = name.strip().rstrip("*").strip()
    result = _REVERSE.get(clean.lower())
    if result is None:
        # Try partial match
        for key, val in _REVERSE.items():
            if clean.lower() in key or key in clean.lower():
                return val
        return clean  # Return as-is if no match
    return result


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
