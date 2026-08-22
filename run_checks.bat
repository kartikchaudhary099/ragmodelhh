@echo off
REM ============================================================================
REM ThinkZen — one-click verification for brief steps 11/15 (pytest) and 12/16
REM (live server + endpoints). Runs everything, writes a full log, prints it.
REM
REM Usage (double-click, or from a terminal at the repo root):
REM     run_checks.bat
REM
REM Paste the resulting checks_output.log back if you want help reading it.
REM ============================================================================
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "LOG=checks_output.log"

if not exist "%PY%" (
    echo Could not find %PY%
    echo Create the venv and install dev deps first:
    echo     py -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
    exit /b 1
)

(
    echo ================================================================
    echo ThinkZen verification run
    echo ================================================================
    "%PY%" --version
    echo.
    echo === STEP 15: full pytest suite ===
    "%PY%" -m pytest -q
    echo.
    echo === STEP 16a: in-process nine-case verify ^(TestClient^) ===
    "%PY%" scripts\verify_pipeline.py
    echo.
    echo === STEP 16b: live-server smoke ^(real uvicorn over HTTP^) ===
    "%PY%" scripts\smoke_server.py
) > "%LOG%" 2>&1

type "%LOG%"
echo.
echo ----------------------------------------------------------------
echo Full log saved to %LOG%
echo ----------------------------------------------------------------
endlocal
