"""Split the full-GB Betfair BTTS extraction (`data/betfair_btts.csv`)
into PL-only and EFL-only per-match files via canonical team-name mapping.

Reads:
  * ``data/betfair_btts.csv`` — from ``extract_btts_odds.py``
  * ``CompleteDSPL_CSV.csv``   — canonical PL fixtures + results
  * ``CompleteDSChamp_CSV.csv`` — canonical EFL fixtures + results

Writes:
  * ``data/betfair_pl_btts.csv``
  * ``data/betfair_efl_btts.csv``
  * ``reports/pl_btts_coverage.md``
  * ``reports/efl_btts_coverage.md``

For each Betfair BTTS row, the script attempts BOTH mappings (PL canonical
and EFL canonical). Whichever maps to a real fixture (both sides mappable
+ the fixture exists in the canonical CSV within ±1 day of market_time)
gets the row. Rows that map to neither are dropped and logged. A row
should never map to both leagues because a single match can only be in
one league.

Run: ``python scripts/extract_btts_by_league.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "betfair_btts.csv"
EFL_CSV = PROJECT_ROOT / "CompleteDSChamp_CSV.csv"
PL_CSV = PROJECT_ROOT / "CompleteDSPL_CSV.csv"

PL_OUTPUT_CSV = PROJECT_ROOT / "data" / "betfair_pl_btts.csv"
EFL_OUTPUT_CSV = PROJECT_ROOT / "data" / "betfair_efl_btts.csv"
PL_REPORT_MD = PROJECT_ROOT / "reports" / "pl_btts_coverage.md"
EFL_REPORT_MD = PROJECT_ROOT / "reports" / "efl_btts_coverage.md"

# EFL mapping is identical to the one in extract_efl_ou15_betfair
from scripts.extract_efl_ou15_betfair import BETFAIR_TO_CANONICAL as EFL_MAP

# ─────────────────────────────────────────────────────────────────────────────
# PL senior men's first-team mapping
# ─────────────────────────────────────────────────────────────────────────────
# PL canonical names come from CompleteDSPL_CSV.csv and typically include an
# "FC" or "AFC" suffix (except "AFC Bournemouth"). Betfair uses the short
# form in its event feed (e.g. "Man City v Arsenal"). This dict lists every
# senior first-team Betfair name we accept for the PL-matching leg.
PL_MAP: dict[str, str] = {
    "AFC Bournemouth": "AFC Bournemouth",
    "Bournemouth": "AFC Bournemouth",
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Aston VIlla FC": "Aston Villa FC",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Burnley": "Burnley FC",
    "Cardiff": "Cardiff City FC",
    "Cardiff City": "Cardiff City FC",
    "Chelsea": "Chelsea FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Huddersfield": "Huddersfield Town AFC",
    "Huddersfield Town": "Huddersfield Town AFC",
    "Hull": "Hull City AFC",
    "Hull City": "Hull City AFC",
    "Ipswich": "Ipswich Town FC",
    "Ipswich Town": "Ipswich Town FC",
    "Leeds": "Leeds United FC",
    "Leeds Utd": "Leeds United FC",
    "Leicester": "Leicester City FC",
    "Leicester City": "Leicester City FC",
    "Liverpool": "Liverpool FC",
    "Luton": "Luton Town FC",
    "Luton Town": "Luton Town FC",
    "Man City": "Manchester City FC",
    "Manchester City": "Manchester City FC",
    "Man Utd": "Manchester United FC",
    "Manchester United": "Manchester United FC",
    "Manchester Utd": "Manchester United FC",
    "Middlesbrough": "Middlesbrough FC",
    "Newcastle": "Newcastle United FC",
    "Norwich": "Norwich City FC",
    "Norwich City": "Norwich City FC",
    "Nottm Forest": "Nottingham Forest FC",
    "Nottingham Forest": "Nottingham Forest FC",
    "Sheff Utd": "Sheffield United FC",
    "Sheffield United": "Sheffield United FC",
    "Southampton": "Southampton FC",
    "Stoke": "Stoke City FC",
    "Stoke City": "Stoke City FC",
    "Sunderland": "Sunderland AFC",
    "Swansea": "Swansea City AFC",
    "Swansea City": "Swansea City AFC",
    "Tottenham": "Tottenham Hotspur FC",
    "Spurs": "Tottenham Hotspur FC",
    "Watford": "Watford FC",
    "West Brom": "West Bromwich Albion FC",
    "West Bromwich Albion": "West Bromwich Albion FC",
    "West Ham": "West Ham United FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers FC",
}


def _parse_event(event_name: str) -> tuple[str, str] | None:
    if " v " not in event_name:
        return None
    parts = event_name.split(" v ", 1)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def _load_canonical(csv_path: Path, league: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["Date", "Home_Team", "Away_Team", "SeasonIndex",
                 "Home_Goals", "Away_Goals"],
    )
    # PL CSV mixes "2022-05-08" and "2022-05-08 00:00:00" across seasons;
    # pandas 2+ needs ISO8601 mode to parse both without dropping rows.
    df["_date"] = pd.to_datetime(
        df["Date"], format="ISO8601", errors="coerce"
    ).dt.date
    df = df[df["_date"].notna() & (df["SeasonIndex"] >= 16)].copy()
    df["_league"] = league
    return df


def _join_and_write(
    bf_mapped: pd.DataFrame, canonical: pd.DataFrame, output_csv: Path,
    report_md: Path, league: str,
) -> int:
    """Inner-join mapped Betfair BTTS rows to canonical fixtures.

    Returns: number of rows written.
    """
    merged = bf_mapped.merge(
        canonical, on=["Home_Team", "Away_Team"], how="inner",
        suffixes=("", "_can"),
    )
    merged["_date_diff"] = (
        pd.to_datetime(merged["_date"]) - pd.to_datetime(merged["_bf_date"])
    ).dt.days.abs()
    merged = merged[merged["_date_diff"] <= 1].copy()
    # Keep closest-date match per (event_name, market_time)
    merged = merged.sort_values(
        ["event_name", "_date_diff"]
    ).drop_duplicates(subset=["event_name", "market_time"], keep="first")

    # Outcome from canonical goals
    merged["BTTS_Result"] = (
        (merged["Home_Goals"] > 0) & (merged["Away_Goals"] > 0)
    ).astype(int)

    out_cols = [
        "Date", "SeasonIndex", "Home_Team", "Away_Team",
        "yes_ltp", "no_ltp", "yes_ltp_first", "no_ltp_first",
        "market_time", "settled_time", "BTTS_Result",
    ]
    out = merged[out_cols].sort_values(["Date", "Home_Team"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}: {len(out):,} rows")

    # Per-season coverage report
    report_md.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md, "w", encoding="utf-8") as f:
        f.write(f"# {league} BTTS — Betfair coverage (Phase 2.4)\n\n")
        f.write(f"Source: {INPUT_CSV.name}, filtered to rows that map to "
                f"canonical {league} teams + fixtures within ±1 day.\n\n")
        f.write(f"**Output:** `{output_csv.relative_to(PROJECT_ROOT)}` "
                f"({len(out):,} rows)\n\n")
        f.write("## Per-season coverage\n\n")
        f.write("| Season | Betfair matches | Canonical fixtures | Coverage |\n")
        f.write("|--------|-----------------|--------------------|----------|\n")
        for s in sorted(canonical["SeasonIndex"].unique()):
            total = (canonical["SeasonIndex"] == s).sum()
            got = (out["SeasonIndex"] == s).sum()
            pct = 100.0 * got / total if total else 0.0
            f.write(f"| S{s} | {got} | {total} | {pct:.1f}% |\n")
        f.write("\n")
    print(f"Report: {report_md}")
    return len(out)


def main() -> int:
    print(f"Reading Betfair BTTS: {INPUT_CSV}")
    bf = pd.read_csv(INPUT_CSV)
    print(f"  Total GB BTTS rows: {len(bf):,}")

    # Parse event name
    hw = bf["event_name"].apply(_parse_event)
    bf["_bf_home"] = hw.apply(lambda t: t[0] if t else None)
    bf["_bf_away"] = hw.apply(lambda t: t[1] if t else None)
    bf = bf[bf["_bf_home"].notna() & bf["_bf_away"].notna()].copy()
    bf["_bf_date"] = pd.to_datetime(
        bf["market_time"], utc=True, errors="coerce"
    ).dt.date
    bf = bf[bf["_bf_date"].notna()].copy()

    # Try EFL mapping
    bf_efl = bf.copy()
    bf_efl["Home_Team"] = bf_efl["_bf_home"].map(EFL_MAP)
    bf_efl["Away_Team"] = bf_efl["_bf_away"].map(EFL_MAP)
    bf_efl_mapped = bf_efl[
        bf_efl["Home_Team"].notna() & bf_efl["Away_Team"].notna()
    ].copy()
    print(f"  EFL-mapped candidates: {len(bf_efl_mapped):,}")

    # Try PL mapping (on the same unparsed Betfair rows)
    bf_pl = bf.copy()
    bf_pl["Home_Team"] = bf_pl["_bf_home"].map(PL_MAP)
    bf_pl["Away_Team"] = bf_pl["_bf_away"].map(PL_MAP)
    bf_pl_mapped = bf_pl[
        bf_pl["Home_Team"].notna() & bf_pl["Away_Team"].notna()
    ].copy()
    print(f"  PL-mapped candidates: {len(bf_pl_mapped):,}")

    # Load canonical fixtures
    pl_canonical = _load_canonical(PL_CSV, "PL")
    efl_canonical = _load_canonical(EFL_CSV, "EFL")
    print(f"  Canonical PL fixtures (S16+): {len(pl_canonical):,}")
    print(f"  Canonical EFL fixtures (S16+): {len(efl_canonical):,}")

    # Join each side to the right canonical CSV
    print("\n--- PL ---")
    pl_rows = _join_and_write(
        bf_pl_mapped, pl_canonical, PL_OUTPUT_CSV, PL_REPORT_MD, "PL",
    )
    print("\n--- EFL ---")
    efl_rows = _join_and_write(
        bf_efl_mapped, efl_canonical, EFL_OUTPUT_CSV, EFL_REPORT_MD, "EFL",
    )

    # Unmatched senior-looking Betfair names
    matched_bf = set()
    if pl_rows:
        pl_out = pd.read_csv(PL_OUTPUT_CSV)
        matched_bf.update(set(pl_out["Home_Team"]) | set(pl_out["Away_Team"]))
    if efl_rows:
        efl_out = pd.read_csv(EFL_OUTPUT_CSV)
        matched_bf.update(set(efl_out["Home_Team"]) | set(efl_out["Away_Team"]))

    bf_names_all = set(bf["_bf_home"].dropna()) | set(bf["_bf_away"].dropna())
    exclude_markers = ("(W)", "(Res)", "U21", "U23", "LFC",
                       " B)", " FC B", " XI", "(w)", "XI")
    unmatched_senior = sorted(
        n for n in bf_names_all
        if n not in PL_MAP and n not in EFL_MAP
        and not any(m in n for m in exclude_markers)
    )
    print(f"\nUnmatched senior-looking Betfair names: {len(unmatched_senior)}")
    print("First 40:")
    for n in unmatched_senior[:40]:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
