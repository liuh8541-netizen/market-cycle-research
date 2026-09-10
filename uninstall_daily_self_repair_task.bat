@echo off
setlocal
chcp 65001 >nul

set "TASK_NAME=TWII Market Self Repair Monitor"
schtasks /Delete /TN "%TASK_NAME%" /F
if not %ERRORLEVEL%==0 (
  echo.
  echo Failed to delete scheduled task or task was not installed.
  pause
  exit /b 1
)

echo.
echo Deleted scheduled task:
echo %TASK_NAME%
echo.
pause
