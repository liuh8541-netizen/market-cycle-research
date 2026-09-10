@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo Market Forecast - FinMind Secure Login
echo Project folder:
echo %CD%
echo.

set "SECURE_RUNNER=%CD%\scripts\run_predict_secure.ps1"
if not exist "%SECURE_RUNNER%" (
  echo Cannot find scripts\run_predict_secure.ps1
  echo.
  pause
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SECURE_RUNNER%"
if not %ERRORLEVEL%==0 (
  echo.
  echo Forecast failed. Please review the error above.
  echo.
  pause
  exit /b 1
)

echo.
echo Session token cleared. Saved cloud token remains available for the next run.
echo.
pause
