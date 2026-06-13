---
description: "Run a full health check across the betting bot — model validation, data pipeline QA, and dashboard review."
---

## Full Project Health Check

Run the following three-agent review sequence for the betting bot:

### Step 1: Data Pipeline QA
Use the **data-qa** agent to:
- Validate all API response parsing against expected schemas (The-Odds-API, OddsPapi, ESPN)
- Check for null handling gaps in the OddsPapi/Odds-API merge (`_merge_oddspapi_into_matches`)
- Verify team name mapping consistency across all sources (PL normalize, Championship _ODDS_API_TO_CHAMP, ESPN _ESPN_TO_PL/_ESPN_TO_CHAMP, merge _normalise_team_for_merge)
- Confirm ESPN settlement correctly matches fixtures and handles postponed matches
- Check SQLite schema alignment between `dashboard.db` and `dashboard_efl.db`

### Step 2: Model Validation
Use the **model-scientist** agent to:
- Review the 4-model stacking ensemble (XGB + LGB + DC + LogReg stacker) for correctness
- Analyse recent model analytics for performance drift across O/U 1.5, O/U 2.5, and BTTS markets
- Check calibration: are predicted probabilities matching observed frequencies (known ~3-6% upward shift)
- Verify walk-forward CV has no data leakage and respects temporal ordering
- Confirm `dc_probs` are correctly passed through predict pipeline

### Step 3: Dashboard Review
Use the **dashboard-reviewer** agent to:
- Trace one fixture end-to-end: The-Odds-API fetch -> OddsPapi merge -> model prediction -> edge calculation -> DataTable display -> ESPN settlement -> analytics logging
- Verify edge calculations displayed match backend values (model_prob - fair_prob vs blended edge)
- Confirm all three markets handled consistently in display, "Rec" flagging, and analytics tracking
- Check that the "Rec" column correctly cross-references the recommendations table
- Verify the analytics tab splits by "Rec'd" / "Not Rec'd" correctly

### Output
Provide a consolidated report with:
- Critical issues (must fix)
- Warnings (should investigate)
- Suggestions (nice to have)
- Overall system health assessment
