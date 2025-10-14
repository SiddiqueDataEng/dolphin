@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Start Streamlit dashboard for HospiAnalytics Pro

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

REM Prepend project root to PYTHONPATH so 'hospi' is importable
set PYTHONPATH=%CD%;%PYTHONPATH%

REM Launch Streamlit dashboard
streamlit run hospi\dashboard.py --server.headless true %*
set EXITCODE=%ERRORLEVEL%

echo.
echo Dashboard exited with code %EXITCODE%
exit /b %EXITCODE%
