@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3.12 -c "import sys" >nul 2>nul
  if %errorlevel%==0 (
    set PY=py -3.12
  ) else (
    set PY=py -3.11
  )
) else (
  set PY=python
)
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv || exit /b 1
)
.venv\Scripts\python.exe -m pip install -r requirements-enhanced.txt || exit /b 1
.venv\Scripts\python.exe -m wifi_localization web --output outputs\web
endlocal
