@echo off
REM ──────────────────────────────────────────────────────────────────────
REM  Weekly retrain wrapper for Windows Task Scheduler.
REM
REM  Runs the full PL + EFL retrain via run.py, captures stdout/stderr to
REM  a rolling log so you can audit what happened. Triggered by the
REM  "Betting Bot Weekly Retrain" Scheduled Task — Sunday 23:30 with
REM  catch-up enabled if the laptop is off at that time.
REM
REM  Manual run for testing:
REM      scripts\weekly_retrain.bat
REM
REM  Log location:
REM      logs\weekly_retrain.log    (last run only — overwritten each week)
REM      logs\weekly_retrain_YYYYMMDD.log  (timestamped archive of each run)
REM ──────────────────────────────────────────────────────────────────────

cd /d "C:\Users\joshc\OneDrive\Documents\Project"

REM Build a date stamp for the archive log (YYYYMMDD format).
REM Use PowerShell rather than wmic — wmic is deprecated and absent
REM by default on Windows 11.
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%I

REM Make sure the logs dir exists
if not exist logs mkdir logs

REM Run the retrain. Two log targets:
REM   - weekly_retrain.log  (always the most recent, easy to tail)
REM   - weekly_retrain_<date>.log  (archive)
REM Tee-style logging in pure batch is awkward, so we write to the archive
REM and copy at the end.

set LOG_FILE=logs\weekly_retrain_%TODAY%.log
echo ============================================================ > "%LOG_FILE%"
echo  Weekly retrain started: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
python run.py retrain >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo  Weekly retrain finished: %DATE% %TIME% (exit=%EXIT_CODE%) >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM Copy the archive to the rolling "last run" log
copy /Y "%LOG_FILE%" "logs\weekly_retrain.log" > nul

exit /b %EXIT_CODE%
