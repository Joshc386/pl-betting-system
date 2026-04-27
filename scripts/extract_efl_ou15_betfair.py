"""Extract EFL Championship O/U 1.5 closing odds from the existing
Betfair GB extraction (`data/betfair_goal_ou.csv`).

This is part of Phase 2.3 of the Option C ROI-validation effort. The
Betfair historical archives (`BetFairData_OU/`) were downloaded as a
country-wide GB feed filtered to Over/Under markets; `extract_goal_odds.py`
already parsed those into `data/betfair_goal_ou.csv` with country_code,
market_type, event_name, over_ltp, under_ltp and settlement info.

This script:
  1. Filters that CSV to `OVER_UNDER_15` market_type.
  2. Applies an explicit senior-men's-first-team allowlist mapping from
     Betfair naming to our canonical `CompleteDSChamp_CSV.csv` naming.
     Women's / reserve / U21 / U23 / B-team entries are excluded.
  3. Joins each Betfair event to a canonical EFL Championship fixture
     in `CompleteDSChamp_CSV.csv` on (canonical home, canonical away,
     calendar date ± 1 day to absorb UTC/local mismatch).
  4. Writes `data/betfair_efl_ou15.csv` with columns:
     Date, SeasonIndex, Home_Team, Away_Team, over_ltp, under_ltp,
     over_ltp_first, under_ltp_first, market_time, settled_time,
     Over_Goals (outcome).
  5. Prints a per-season coverage report and lists any unmatched
     Betfair events + unmatched canonical fixtures so the user can
     review the mapping before we trust the join.

Run: ``python scripts/extract_efl_ou15_betfair.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "betfair_goal_ou.csv"
EFL_CSV = PROJECT_ROOT / "CompleteDSChamp_CSV.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "betfair_efl_ou15.csv"
REPORT_MD = PROJECT_ROOT / "reports" / "efl_ou15_coverage.md"

# ─────────────────────────────────────────────────────────────────────────────
# Senior men's first-team mapping: Betfair name → canonical EFL name
# ─────────────────────────────────────────────────────────────────────────────
# Anything not in this dict is silently dropped. Women / reserves / U21 / U23
# / B-teams / academy / non-league entries never appear here by design.

BETFAIR_TO_CANONICAL: dict[str, str] = {
    # Championship-regular teams in the 2016/17 - 2025/26 window.
    "Aston Villa": "Aston Villa",
    "Aston VIlla FC": "Aston Villa",  # observed typo variant
    "Barnsley": "Barnsley",
    "Birmingham": "Birmingham",
    "Birmingham City": "Birmingham",
    "Blackburn": "Blackburn",
    "Blackburn Rovers": "Blackburn",
    "Blackpool": "Blackpool",
    "Bolton": "Bolton",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Bristol City": "Bristol City",
    "Burnley": "Burnley",
    "Burton": "Burton",
    "Burton Albion": "Burton",
    "Cardiff": "Cardiff",
    "Cardiff City": "Cardiff",
    "Charlton": "Charlton",
    "Coventry": "Coventry",
    "Coventry City": "Coventry",
    "Derby": "Derby",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Huddersfield Town": "Huddersfield",
    "Hull": "Hull",
    "Hull City": "Hull",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leeds": "Leeds",
    "Leeds Utd": "Leeds",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Luton": "Luton",
    "Luton Town": "Luton",
    "Middlesbrough": "Middlesbrough",
    "Millwall": "Millwall",
    "Newcastle": "Newcastle",
    "Norwich": "Norwich",
    "Norwich City": "Norwich",
    "Nottm Forest": "Nott'm Forest",
    "Oxford United": "Oxford",
    "Oxford Utd": "Oxford",
    "Peterborough": "Peterboro",
    "Plymouth": "Plymouth",
    "Plymouth Argyle": "Plymouth",
    "Portsmouth": "Portsmouth",
    "Preston": "Preston",
    "Preston North End": "Preston",
    "QPR": "QPR",
    "Reading": "Reading",
    "Rotherham": "Rotherham",
    "Rotherham United": "Rotherham",
    "Rotherham Utd": "Rotherham",
    "Sheff Utd": "Sheffield United",
    "Sheff Wed": "Sheffield Weds",
    "Southampton": "Southampton",
    "Stoke": "Stoke",
    "Stoke City": "Stoke",
    "Sunderland": "Sunderland",
    "Swansea": "Swansea",
    "Swansea City": "Swansea",
    "Watford": "Watford",
    "West Brom": "West Brom",
    "Wigan": "Wigan",
    "Wigan Athletic": "Wigan",
    "Wolves": "Wolves",
    "Wycombe": "Wycombe",
    "Wycombe Wanderers": "Wycombe",
}


def _parse_event(event_name: str) -> tuple[str, str] | None:
    """Split 'HomeName v AwayName' into (home, away)."""
    if " v " not in event_name:
        return None
    parts = event_name.split(" v ", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def main() -> int:
    print(f"Reading Betfair OU: {INPUT_CSV}")
    bf = pd.read_csv(INPUT_CSV)
    bf = bf[bf["market_type"] == "OVER_UNDER_15"].copy()
    print(f"  OVER_UNDER_15 rows: {len(bf):,}")

    # Parse event_name → home/away
    hw = bf["event_name"].apply(_parse_event)
    bf["_bf_home"] = hw.apply(lambda t: t[0] if t else None)
    bf["_bf_away"] = hw.apply(lambda t: t[1] if t else None)
    bf = bf[bf["_bf_home"].notna() & bf["_bf_away"].notna()].copy()

    # Map to canonical names. Rows where EITHER side fails lookup are
    # dropped — this filters out women's, reserves, youth, non-league.
    bf["Home_Team"] = bf["_bf_home"].map(BETFAIR_TO_CANONICAL)
    bf["Away_Team"] = bf["_bf_away"].map(BETFAIR_TO_CANONICAL)
    mapped = bf[bf["Home_Team"].notna() & bf["Away_Team"].notna()].copy()
    print(f"  After canonical mapping (both sides match): {len(mapped):,}")

    # Parse Betfair market_time into a date (UTC).
    mapped["_bf_date"] = pd.to_datetime(
        mapped["market_time"], utc=True, errors="coerce"
    ).dt.date
    mapped = mapped[mapped["_bf_date"].notna()].copy()

    # Load canonical EFL fixtures.
    print(f"\nReading canonical EFL fixtures: {EFL_CSV}")
    efl = pd.read_csv(
        EFL_CSV, usecols=["Date", "Home_Team", "Away_Team", "SeasonIndex",
                          "Home_Goals", "Away_Goals"],
    )
    efl["_efl_date"] = pd.to_datetime(
        efl["Date"], format="ISO8601", errors="coerce"
    ).dt.date
    efl = efl[efl["_efl_date"].notna()].copy()
    # Walk-forward window we care about for ROI validation: S16 onwards.
    efl = efl[efl["SeasonIndex"] >= 16].copy()
    print(f"  EFL fixtures S16+: {len(efl):,}")

    # Join on (home, away) then ±1 day date window.
    merged = mapped.merge(
        efl,
        left_on=["Home_Team", "Away_Team"],
        right_on=["Home_Team", "Away_Team"],
        how="left",
        suffixes=("", "_efl"),
    )
    # Keep rows where the Betfair date is within 1 day of the canonical date.
    merged["_date_diff"] = (
        pd.to_datetime(merged["_efl_date"]) - pd.to_datetime(merged["_bf_date"])
    ).dt.days.abs()
    matched = merged[merged["_date_diff"] <= 1].copy()
    # In the ±1 day window we may have duplicates; keep the closest-date match.
    matched = matched.sort_values(["event_name", "_date_diff"]).drop_duplicates(
        subset=["event_name", "market_time"], keep="first"
    )
    print(f"\nMatched Betfair rows to canonical fixtures: {len(matched):,}")

    # Outcome from canonical goals.
    matched["Over_Goals"] = (
        matched["Home_Goals"] + matched["Away_Goals"] > 1.5
    ).astype(int)

    # Write output.
    out_cols = [
        "Date", "SeasonIndex", "Home_Team", "Away_Team",
        "over_ltp", "under_ltp", "over_ltp_first", "under_ltp_first",
        "market_time", "settled_time", "Over_Goals",
    ]
    out = matched[out_cols].sort_values(["Date", "Home_Team"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV}: {len(out):,} rows")

    # Per-season coverage vs canonical fixtures.
    print("\nPer-season coverage (Betfair O/U 1.5 matched / canonical fixtures):")
    lines_md = []
    for s in sorted(efl["SeasonIndex"].unique()):
        total = (efl["SeasonIndex"] == s).sum()
        got = (out["SeasonIndex"] == s).sum()
        pct = 100.0 * got / total if total else 0.0
        line = f"  S{s}: {got}/{total}  ({pct:.1f}%)"
        print(line)
        lines_md.append(line)

    # Unmatched Betfair event names (for diagnostic — helps user spot
    # mapping gaps). Only report names that appeared in OVER_UNDER_15
    # but were dropped because they weren't in BETFAIR_TO_CANONICAL and
    # the event looks like senior first-team football (excludes the
    # clearly-marked reserves/women/U21/U23).
    exclude_markers = ("(W)", "(Res)", "U21", "U23", "LFC",
                       " B)", " FC B", " XI", "(w)")
    bf_all_names = pd.concat([bf["_bf_home"], bf["_bf_away"]]).dropna().drop_duplicates()
    unmatched_senior = [
        n for n in bf_all_names
        if n not in BETFAIR_TO_CANONICAL
        and not any(m in n for m in exclude_markers)
    ]
    print(f"\nUnmatched Betfair names that look senior (review):")
    print(f"  (These are names with no age/women/reserve markers that")
    print(f"   didn't map. Most will be PL-only teams or non-league.)")
    for n in sorted(unmatched_senior)[:40]:
        print(f"    {n}")

    # Canonical EFL fixtures with NO matched Betfair row — these are
    # fixtures where the mapping failed or no Betfair market existed.
    efl_with_match = matched[["_efl_date", "Home_Team", "Away_Team"]].drop_duplicates()
    efl["_has_bf"] = efl[["_efl_date", "Home_Team", "Away_Team"]].apply(
        tuple, axis=1
    ).isin(
        efl_with_match[["_efl_date", "Home_Team", "Away_Team"]].apply(tuple, axis=1)
    )
    missing_fixtures = efl[~efl["_has_bf"]]
    print(f"\nCanonical EFL fixtures with NO Betfair O/U 1.5 match: {len(missing_fixtures):,}")
    if len(missing_fixtures) > 0:
        miss_by_season = missing_fixtures.groupby("SeasonIndex").size()
        print("  By season:")
        for s, n in miss_by_season.items():
            print(f"    S{s}: {n} missing")

    # Write the markdown report.
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# EFL O/U 1.5 — Betfair coverage (Phase 2.3)\n\n")
        f.write(f"Extracted from `{INPUT_CSV.name}` filtered to\n"
                f"`OVER_UNDER_15` market_type, joined to\n"
                f"`CompleteDSChamp_CSV.csv` S16+ on canonical team names\n"
                f"+ ±1 day date tolerance.\n\n")
        f.write(f"**Output:** `{OUTPUT_CSV.relative_to(PROJECT_ROOT)}` "
                f"({len(out):,} rows)\n\n")
        f.write("## Per-season coverage\n\n")
        f.write("| Season | Betfair matches | Canonical fixtures | Coverage |\n")
        f.write("|--------|-----------------|--------------------|----------|\n")
        for s in sorted(efl["SeasonIndex"].unique()):
            total = (efl["SeasonIndex"] == s).sum()
            got = (out["SeasonIndex"] == s).sum()
            pct = 100.0 * got / total if total else 0.0
            f.write(f"| S{s} | {got} | {total} | {pct:.1f}% |\n")
        f.write(f"\n## Unmatched senior-looking Betfair names\n\n")
        f.write("These aren't in the allowlist and don't have age/reserve/women\n"
                "markers. Most should be PL-only teams or non-league. Review\n"
                "and add to `BETFAIR_TO_CANONICAL` if any are EFL-relevant.\n\n")
        for n in sorted(unmatched_senior)[:80]:
            f.write(f"  - `{n}`\n")
    print(f"\nReport written to {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
