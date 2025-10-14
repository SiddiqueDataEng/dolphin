@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Start HospiAnalytics Pro application

REM Ensure virtual environment exists; if not, run setup
if not exist .venv\Scripts\python.exe (
  echo Virtual environment not found. Running setup_env.bat...
  if exist setup_env.bat (
    call setup_env.bat
    if %ERRORLEVEL% NEQ 0 (
      echo Setup failed. Exiting.
      exit /b 1
    )
  ) else (
    echo setup_env.bat not found. Please run dependency setup manually.
    exit /b 1
  )
)

REM Activate venv
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
  echo Failed to activate virtual environment.
  exit /b 1
)

REM Ensure required packages installed (quick import test)
python -c "import chardet, duckdb, pandas, streamlit" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Installing Python dependencies from requirements.txt...
  python -m pip install -r requirements.txt
  REM Verify imports after install
  python -c "import chardet, duckdb, pandas, streamlit" >nul 2>nul
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to install dependencies.
    exit /b 1
  )
)

REM Optional: warn if data file missing
if not exist "sample sales.csv" (
  echo WARNING: Data file "sample sales.csv" not found in project root.
)

REM Run the Python entrypoint
python -m hospi.main %*
set EXITCODE=%ERRORLEVEL%

echo.
echo Application exited with code %EXITCODE%
exit /b %EXITCODE%
