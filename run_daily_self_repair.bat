@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "RUNNER=%CD%\scripts\daily_self_repair_monitor.ps1"
if not exist "%RUNNER%" (
  echo Cannot find scripts\daily_self_repair_monitor.ps1
  pause
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%RUNNER%"
if not %ERRORLEVEL%==0 (
  echo.
  echo Daily self-repair monitor failed. Check reports\automation_logs.
  pause
  exit /b 1
)

echo.
echo Daily self-repair monitor completed.
echo Reports:
echo - reports\today_market_forecast.md
echo - reports\error_review.md
echo - reports\model_reliability_trend_audit.md
echo - reports\semantic_psychology_layer_audit.md
echo - reports\global_news_risk_impact_audit.md
echo - reports\market_breath_monitor.html
echo.
pause
