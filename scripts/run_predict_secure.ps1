$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python was not found." }
$secretDir = Join-Path $projectDir "config\.secrets"
$tokenFile = Join-Path $secretDir "finmind_token.txt"
$pointer = [IntPtr]::Zero

try {
    if (Test-Path $tokenFile) {
        $env:FINMIND_TOKEN = (Get-Content -Raw -Encoding UTF8 $tokenFile).Trim()
        Write-Host "FinMind Token loaded from the cloud project."
    }
    else {
        $secureToken = Read-Host "First setup: paste FinMind Sponsor API Token" -AsSecureString
        $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        $env:FINMIND_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        if ([string]::IsNullOrWhiteSpace($env:FINMIND_TOKEN)) {
            throw "FinMind Token was not entered."
        }
        New-Item -ItemType Directory -Force -Path $secretDir | Out-Null
        $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        [IO.File]::WriteAllText($tokenFile, $env:FINMIND_TOKEN, $utf8NoBom)
        Write-Host "Token saved. Future runs will load it automatically."
    }

    if ([string]::IsNullOrWhiteSpace($env:FINMIND_TOKEN)) {
        throw "Saved Token is empty. Run reset_finmind_token.bat and try again."
    }

    Set-Location $projectDir
    $taipeiZone = [TimeZoneInfo]::FindSystemTimeZoneById("Taipei Standard Time")
    $now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
    $today = $now.ToString("yyyy-MM-dd")
    $officialDailyEnd = $now.Date.AddDays(-1).ToString("yyyy-MM-dd")

    Write-Host "Updating FinMind Sponsor data..."
    & $python "$projectDir\scripts\fetch_finmind_factors.py" --end-date $today
    if ($LASTEXITCODE -ne 0) { throw "FinMind data update failed." }

    Write-Host "Updating cross-sectional breadth..."
    $breadthStart = [DateTime]::Now.AddDays(-14).ToString("yyyy-MM-dd")
    & $python "$projectDir\scripts\fetch_finmind_cross_sectional_breadth.py" --start-date $breadthStart --end-date $today
    if ($LASTEXITCODE -ne 0) { throw "Cross-sectional breadth update failed." }

    Write-Host "Updating official TAIFEX Taiwan VIX daily history..."
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $vixOutput = & $python "$projectDir\scripts\fetch_taifex_vix_daily.py" --end-date $officialDailyEnd 2>&1
    $vixExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($vixExitCode -ne 0) {
        $vixReason = ($vixOutput | ForEach-Object { $_.ToString() }) -join " | "
        Write-Warning "TAIFEX VIX update failed; forecast will continue with the last stored official VIX history. Reason: $vixReason"
    }
    elseif ($vixOutput) {
        $vixOutput | ForEach-Object { Write-Host $_ }
    }

    $tickStart = [DateTime]::Now.AddDays(-7).ToString("yyyy-MM-dd")
    $tickLock = "$projectDir\data\processed\factors\.futures_tick_fetch.lock"
    Write-Host "Updating incremental TX tick microstructure cache..."
    & $python "$projectDir\scripts\fetch_finmind_futures_tick_bars.py" --start-date $tickStart --end-date $today
    if ($LASTEXITCODE -eq 0 -and !(Test-Path $tickLock)) {
        & $python "$projectDir\scripts\build_night_microstructure_features.py"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "TX microstructure feature build failed; forecast will continue without changing the formal model."
        }
    }
    elseif ($LASTEXITCODE -eq 0) {
        Write-Host "Historical TX tick backfill is active; retained the last complete microstructure feature file."
    }
    else {
        Write-Warning "TX tick microstructure update failed; forecast will continue without changing the formal model."
    }

    Write-Host "Recording validated night-spread opening-gap forecast..."
    & $python "$projectDir\scripts\forecast_locked_night_spread_gap.py"
    if ($LASTEXITCODE -ne 0) { throw "Night-spread opening-gap forecast failed." }

    Write-Host "Recording validated opening-gap amplitude risk..."
    & $python "$projectDir\scripts\forecast_locked_gap_amplitude.py"
    if ($LASTEXITCODE -ne 0) { throw "Opening-gap amplitude forecast failed." }

    Write-Host "Updating prospective Q85 opening-gap candidate status..."
    & $python "$projectDir\scripts\forecast_prospective_open_gap_cash_close.py"
    if ($LASTEXITCODE -ne 0) { throw "Prospective Q85 candidate logger failed." }

    $captureDeadline = $now.Date.AddHours(13).AddMinutes(25)
    if ($now.DayOfWeek -notin @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday) -and $now -lt $captureDeadline) {
        $captureScript = "$projectDir\scripts\capture_open_gap_candidate.ps1"
        $captureArguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$captureScript`"")
        Start-Process powershell.exe -ArgumentList $captureArguments -WindowStyle Hidden
        Write-Host "Prospective Q85 capture monitor started; it will stop after the cash-open record is locked."
    }

    Write-Host "Running today's market forecast..."
    & $python "$projectDir\scripts\predict_market.py" --date $today --factor-dir "$projectDir\data\processed\factors"
    if ($LASTEXITCODE -ne 0) { throw "Market forecast failed." }

    Write-Host "Rebuilding causal psychology replay before error analysis..."
    & $python "$projectDir\scripts\backtest_psychology_state.py"
    if ($LASTEXITCODE -ne 0) { throw "Psychology replay update failed." }

    Write-Host "Analyzing latest resolved one-day forecast error..."
    & $python "$projectDir\scripts\analyze_previous_forecast_error.py"
    if ($LASTEXITCODE -ne 0) { throw "Previous-day forecast error analysis failed." }

    Write-Host "Updating complete market clinical ledger..."
    & $python "$projectDir\scripts\build_market_clinical_ledger.py"
    if ($LASTEXITCODE -ne 0) { throw "Market clinical ledger update failed." }

    Write-Host "Updating append-only prospective clinical registry..."
    & $python "$projectDir\scripts\update_prospective_clinical_registry.py"
    if ($LASTEXITCODE -ne 0) { throw "Prospective clinical registry update failed." }

    $report = "$projectDir\reports\today_market_forecast.md"
    if (!(Test-Path $report)) { throw "Forecast report was not found." }
    Start-Process $report
    Write-Host "Completed: $report"
}
finally {
    $env:FINMIND_TOKEN = $null
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    Remove-Variable secureToken -ErrorAction SilentlyContinue
}
