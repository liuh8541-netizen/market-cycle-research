@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "REPORT=%PROJECT_DIR%reports\today_market_forecast.md"

if not exist "%REPORT%" (
    echo Latest forecast report was not found:
    echo "%REPORT%"
    pause
    exit /b 1
)

start "" "%REPORT%"
exit /b 0
