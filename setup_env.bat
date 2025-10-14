@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Windows setup: create venv, upgrade pip, install deps

REM Detect Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo Python not found in PATH. Please install Python 3.10+ and retry.
  exit /b 1
)

REM Create virtual environment in .venv
if not exist .venv (
  echo Creating virtual environment in .venv
  python -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo Virtual environment already exists: .venv
)

REM Activate venv
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
  echo Failed to activate virtual environment.
  exit /b 1
)

REM Upgrade pip and install build tools
python -m pip install --upgrade pip setuptools wheel

REM Create default requirements.txt if missing
if not exist requirements.txt (
  echo Creating default requirements.txt
  > requirements.txt echo pandas>=2.0.0
  >> requirements.txt echo streamlit>=1.35.0
  >> requirements.txt echo plotly>=5.20.0
  >> requirements.txt echo numpy>=1.24.0
  >> requirements.txt echo scikit-learn>=1.4.0
  >> requirements.txt echo pytest>=8.0.0
  >> requirements.txt echo openpyxl>=3.1.2
  >> requirements.txt echo SQLAlchemy>=2.0.0
  >> requirements.txt echo duckdb>=1.0.0
  >> requirements.txt echo python-dateutil>=2.8.2
  >> requirements.txt echo pytz>=2024.1
)

REM Install dependencies
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
  echo Failed to install dependencies.
  exit /b 1
)

echo.
echo Environment ready. To activate later, run:
echo   call .venv\Scripts\activate.bat
endlocal
