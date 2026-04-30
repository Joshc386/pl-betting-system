@echo off
REM ──────────────────────────────────────────────────────────────────────
REM  Daily settlement wrapper for Windows Task Scheduler.
REM
REM  Runs ``python run.py settle`` which calls ``job_settle_bets`` —
REM  pulls finished matches from football-data.org for the last 3 days,
REM  marks any open ``logged_bets`` rows with their actual outcomes,
REM  and settles every model ``prediction`` for the same window.
REM
REM  Triggered by the "Betting Bot Daily Settlement" Scheduled Task —
REM  fires daily at 23:30 with catch-up enabled if the laptop was off.
REM
REM  Manual run for testing:
REM      scripts\daily_settle.bat
REM
REM  Log location:
REM      logs\daily_settle.log              (rolling — last run, easy to tail)
REM      logs\daily_settle_YYYYMMDD.log     (timestamped archive)
REM
REM  Cost: ~14 football-data.org calls (well under 10 req/min free tier
REM  limit) + ~14 ESPN scoreboard calls (unlimited). Zero impact on
REM  Odds API / OddsPapi quotas.
REM ──────────────────────────────────────────────────────────────────────

cd /d "C:\Users\joshc\OneDrive\Documents\Project"

REM Build a date stamp for the archive log (YYYYMMDD format).
REM Use PowerShell rather than wmic — wmic is deprecated and absent
REM by default on Windows 11.
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%I

REM Make sure the logs dir exists
if not exist logs mkdir logs

set LOG_FILE=logs\daily_settle_%TODAY%.log
echo ============================================================ > "%LOG_FILE%"
echo  Daily settlement started: %DATE% %TIME% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
python run.py settle >> "%LOG_FILE%" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo. >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo  Daily settlement finished: %DATE% %TIME% (exit=%EXIT_CODE%) >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

REM Copy the archive to the rolling "last run" log
copy /Y "%LOG_FILE%" "logs\daily_settle.log" > nul

exit /b %EXIT_CODE%
