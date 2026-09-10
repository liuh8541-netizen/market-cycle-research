@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set "TASK_NAME=TWII Market Self Repair Monitor"
set "RUNNER=%CD%\scripts\daily_self_repair_monitor.ps1"

if not exist "%RUNNER%" (
  echo Cannot find scripts\daily_self_repair_monitor.ps1
  pause
  exit /b 1
)

schtasks /Create /TN "%TASK_NAME%" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 15:10 /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""%RUNNER%""" /F
if not %ERRORLEVEL%==0 (
  echo.
  echo Failed to install scheduled task.
  pause
  exit /b 1
)

echo.
echo Installed scheduled task:
echo %TASK_NAME%
echo Schedule: Monday-Friday 15:10
echo Action: run daily self-repair monitor after market close.
echo.
pause
