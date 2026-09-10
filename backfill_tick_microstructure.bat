@echo off
setlocal
cd /d "%~dp0"
echo Starting resumable FinMind TX tick microstructure backfill...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\backfill_tick_microstructure.ps1"
if errorlevel 1 (
  echo.
  echo Backfill failed. Review reports\tick_microstructure_backfill.log
) else (
  echo.
  echo Backfill completed.
)
pause
