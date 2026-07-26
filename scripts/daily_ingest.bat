@echo off
REM ----------------------------------------------------------------------
REM  Daily canonical-dataset ingestion for Windows Task Scheduler.
REM
REM  Rebuilds each league's Canonical Dataset from football-data.co.uk,
REM  re-downloading the current season (which gains fixtures every
REM  matchday), validates that historical Facts have not moved, publishes
REM  only if they haven't, then re-derives the Betfair League Splits.
REM
REM  Task Scheduler runs data-critical jobs rather than APScheduler because
REM  StartWhenAvailable catches up jobs missed while the laptop slept,
REM  whereas APScheduler's misfire_grace_time silently discards them.
REM  See docs/adr/0006-task-scheduler-for-data-critical-jobs.md.
REM
REM  Triggered by the "Betting Bot Daily Ingest" Scheduled Task -- daily
REM  at 06:00 with catch-up enabled.
REM
REM  Manual run for testing:
REM      scripts\daily_ingest.bat
REM
REM  Log location:
REM      logs\daily_ingest.log              (rolling -- last run)
REM      logs\daily_ingest_YYYYMMDD.log     (timestamped archive)
REM ----------------------------------------------------------------------

cd /d "C:\Users\joshc\OneDrive\Documents\Project"

REM Build a date stamp for the archive log (YYYYMMDD format).
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%I

REM Make sure the logs dir exists
if not exist logs mkdir logs

set LOG_FILE=logs\daily_ingest_%TODAY%.log
echo ============================================================ > "%LOG_FILE%"
echo  Daily ingest started: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
python scripts\daily_ingest.py >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo  Daily ingest finished: %DATE% %TIME% (exit=%EXIT_CODE%) >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM Copy the archive to the rolling "last run" log
copy /Y "%LOG_FILE%" "logs\daily_ingest.log" > nul

exit /b %EXIT_CODE%
