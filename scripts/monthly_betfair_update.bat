@echo off
REM ----------------------------------------------------------------------
REM  Monthly Betfair historical data download for Windows Task Scheduler.
REM
REM  Downloads the previous month's BTTS, O/U 1.5, and O/U 2.5 market
REM  data from the Betfair Historical Data API (GB only), parses the bz2
REM  files, appends new rows to the master CSVs, and re-runs the
REM  league-split scripts to regenerate PL/EFL-specific files.
REM
REM  Triggered by the "Betting Bot Monthly Betfair Update" Scheduled
REM  Task -- 5th of each month at 10:00 with catch-up enabled if the
REM  laptop is off at that time.
REM
REM  Manual run for testing:
REM      scripts\monthly_betfair_update.bat
REM
REM  Log location:
REM      logs\monthly_betfair.log               (rolling -- last run)
REM      logs\monthly_betfair_YYYYMMDD.log      (timestamped archive)
REM ----------------------------------------------------------------------

cd /d "C:\Users\joshc\OneDrive\Documents\Project"

REM Build a date stamp for the archive log (YYYYMMDD format).
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%I

REM Make sure the logs dir exists
if not exist logs mkdir logs

set LOG_FILE=logs\monthly_betfair_%TODAY%.log
echo ============================================================ > "%LOG_FILE%"
echo  Monthly Betfair update started: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
python scripts\betfair_monthly_update.py >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo  Monthly Betfair update finished: %DATE% %TIME% (exit=%EXIT_CODE%) >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM Copy the archive to the rolling "last run" log
copy /Y "%LOG_FILE%" "logs\monthly_betfair.log" > nul

exit /b %EXIT_CODE%
