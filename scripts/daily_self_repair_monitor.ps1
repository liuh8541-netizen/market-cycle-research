$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
}
else {
    throw "Python was not found."
}

$logDir = Join-Path $projectDir "reports\automation_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$taipeiZone = [TimeZoneInfo]::FindSystemTimeZoneById("Taipei Standard Time")
$taipeiNow = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
$stamp = $taipeiNow.ToString("yyyyMMdd_HHmmss")
$logFile = Join-Path $logDir "daily_self_repair_$stamp.log"
$statusFile = Join-Path $projectDir "reports\daily_self_repair_status.json"
$secretDir = Join-Path $projectDir "config\.secrets"
$tokenFile = Join-Path $secretDir "finmind_token.txt"

function Write-StatusFile {
    param(
        [string]$Status,
        [string]$Message
    )
    $payload = [ordered]@{
        status = $Status
        message = $Message
        checked_at = ([TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)).ToString("s")
        timezone = "Asia/Taipei"
        log_file = $logFile
        report = (Join-Path $projectDir "reports\today_market_forecast.md")
        daily_global_news_risk = (Join-Path $projectDir "reports\daily_global_news_risk.json")
        error_review = (Join-Path $projectDir "reports\error_review.md")
        previous_day_error_analysis = (Join-Path $projectDir "reports\previous_day_forecast_error_analysis.md")
        forecast_error_pathology_ledger = (Join-Path $projectDir "reports\forecast_error_pathology_ledger.md")
        breath_monitor = (Join-Path $projectDir "reports\market_breath_monitor.html")
        research_report = (Join-Path $projectDir "reports\periodic_deep_market_research.md")
        research_database = (Join-Path $projectDir "data\processed\market_research_stats.sqlite")
        reliability_report = (Join-Path $projectDir "reports\model_reliability_trend_audit.md")
        semantic_psychology_layer_audit = (Join-Path $projectDir "reports\semantic_psychology_layer_audit.md")
        global_news_risk_impact_audit = (Join-Path $projectDir "reports\global_news_risk_impact_audit.md")
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $statusFile
}

Start-Transcript -Path $logFile -Force | Out-Null
try {
    Set-Location $projectDir
    $nowTaipei = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
    $today = $nowTaipei.ToString("yyyy-MM-dd")
    $officialDailyEnd = $nowTaipei.Date.AddDays(-1).ToString("yyyy-MM-dd")
    $factorDir = Join-Path $projectDir "data\processed\factors"

    if (Test-Path $tokenFile) {
        $env:FINMIND_TOKEN = (Get-Content -Raw -Encoding UTF8 $tokenFile).Trim()
        Write-Host "FinMind token loaded."
    }
    else {
        Write-Warning "FinMind token file not found. Running with public/Yahoo sources and existing factor cache."
    }

    if (-not [string]::IsNullOrWhiteSpace($env:FINMIND_TOKEN)) {
        Write-Host "Updating FinMind factor data..."
        & $python "$projectDir\scripts\fetch_finmind_factors.py" --end-date $today
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "FinMind factor update failed; continuing with existing factor cache."
        }

        Write-Host "Updating TX tick microstructure cache..."
        $tickStart = $nowTaipei.AddDays(-7).ToString("yyyy-MM-dd")
        & $python "$projectDir\scripts\fetch_finmind_futures_tick_bars.py" --start-date $tickStart --end-date $today
        if ($LASTEXITCODE -eq 0) {
            & $python "$projectDir\scripts\build_night_microstructure_features.py"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Microstructure feature build failed; continuing."
            }
        }
        else {
            Write-Warning "TX tick microstructure update failed; continuing."
        }
    }

    Write-Host "Updating official TAIFEX Taiwan VIX daily history..."
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $vixOutput = & $python "$projectDir\scripts\fetch_taifex_vix_daily.py" --end-date $officialDailyEnd 2>&1
    $vixExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($vixExitCode -ne 0) {
        $vixReason = ($vixOutput | ForEach-Object { $_.ToString() }) -join " | "
        Write-Warning "TAIFEX VIX update failed; continuing with existing VIX cache. Reason: $vixReason"
    }
    elseif ($vixOutput) {
        $vixOutput | ForEach-Object { Write-Host $_ }
    }

    Write-Host "Fetching daily global finance/political news risk scaffold..."
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $newsOutput = & $python "$projectDir\scripts\fetch_global_news_risk.py" --date $today 2>&1
    $newsExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($newsExitCode -ne 0) {
        $newsReason = ($newsOutput | ForEach-Object { $_.ToString() }) -join " | "
        Write-Warning "Global news risk scaffold failed; continuing with existing or heartbeat-browsed cache. Reason: $newsReason"
    }
    elseif ($newsOutput) {
        $newsOutput | ForEach-Object { Write-Host $_ }
    }

    Write-Host "Running market forecast, error review, next-day validation, and breath monitor..."
    if (Test-Path $factorDir) {
        & $python "$projectDir\scripts\predict_market.py" --date $today --factor-dir $factorDir
    }
    else {
        & $python "$projectDir\scripts\predict_market.py" --date $today
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Market forecast/self-repair run failed."
    }

    Write-Host "Running previous-day forecast error pathology analysis..."
    & $python "$projectDir\scripts\analyze_previous_forecast_error.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Previous-day forecast error pathology analysis failed."
    }

    Write-Host "Running periodic deep market research database update..."
    & $python "$projectDir\scripts\periodic_deep_market_research.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Periodic deep market research update failed."
    }

    Write-Host "Running model reliability trend audit..."
    & $python "$projectDir\scripts\audit_model_reliability_trend.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Model reliability trend audit failed."
    }

    Write-Host "Running semantic psychology layer effect audit..."
    & $python "$projectDir\scripts\audit_semantic_psychology_layer.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic psychology layer effect audit failed."
    }

    Write-Host "Running global news risk impact audit..."
    & $python "$projectDir\scripts\audit_global_news_risk_impact.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Global news risk impact audit failed."
    }

    Write-StatusFile -Status "success" -Message "Daily self-repair monitor completed."
}
catch {
    Write-StatusFile -Status "failed" -Message $_.Exception.Message
    throw
}
finally {
    $env:FINMIND_TOKEN = $null
    Stop-Transcript | Out-Null
}
