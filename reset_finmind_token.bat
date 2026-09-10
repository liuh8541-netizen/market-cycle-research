@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "TOKEN_FILE=%CD%\config\.secrets\finmind_token.txt"
if exist "%TOKEN_FILE%" (
  del /q "%TOKEN_FILE%"
  echo FinMind Token removed. Run run_predict_market.bat to enter a new token.
) else (
  echo No saved FinMind Token was found.
)
pause
