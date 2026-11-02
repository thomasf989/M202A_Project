@echo off
setlocal
cd /d "%~dp0"

REM Prefer an explicitly activated Conda environment (for example: conda activate m202a).
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
  echo [INFO] Using active Conda Python: %CONDA_PREFIX%\python.exe
  "%CONDA_PREFIX%\python.exe" -m pip install -r requirements-enhanced.txt || exit /b 1
  "%CONDA_PREFIX%\python.exe" -m wifi_localization web --output outputs\web
  exit /b %errorlevel%
)

REM Otherwise use the normal `python` command. Do not use the Windows `py`
REM launcher because stale Python registrations can point at deleted projects.
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python was not found. Install Python 3.11/3.12 or activate your m202a Conda environment.
  exit /b 1
)
python -c "import sys; assert (3,11) <= sys.version_info[:2] <= (3,12)" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] This project expects Python 3.11 or 3.12.
  python --version
  exit /b 1
)

if not exist .venv\Scripts\python.exe (
  echo [INFO] Creating project-local .venv using python...
  python -m venv .venv || exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-enhanced.txt || exit /b 1
.venv\Scripts\python.exe -m wifi_localization web --output outputs\web
endlocal
