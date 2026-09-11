param(
    [ValidateSet("Auto", "Night", "Close")]
    [string]$Phase = "Auto",
    [switch]$NoUpdate
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $projectDir "reports\automation_logs"
$statusFile = Join-Path $projectDir "reports\daily_update_status.json"
$lockFile = Join-Path $projectDir "data\processed\.daily_market_update.lock"
$tokenFile = Join-Path $projectDir "config\.secrets\finmind_token.txt"
$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python was not found." }
$taipeiZone = [TimeZoneInfo]::FindSystemTimeZoneById("Taipei Standard Time")
$now = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)

if ($Phase -eq "Auto") {
    $Phase = if ($now.Hour -lt 12) { "Night" } else { "Close" }
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("{0}-{1}.log" -f $now.ToString("yyyy-MM-dd-HHmmss"), $Phase.ToLowerInvariant())
$startedAt = $now.ToString("o")
$success = $false
$message = ""

function Invoke-CheckedPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    $previousErrorAction = $ErrorActionPreference
    try {
        # Windows PowerShell can promote native stderr into a terminating
        # NativeCommandError. Capture the full stream, then trust exit code.
        $ErrorActionPreference = "Continue"
        & $python @Arguments 2>&1 | ForEach-Object {
            $_.ToString() | Tee-Object -FilePath $logFile -Append
        }
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($nativeExitCode -ne 0) {
        throw "Python command failed ($nativeExitCode): $($Arguments -join ' ')"
    }
}

function Invoke-OptionalPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    try {
        Invoke-CheckedPython $Arguments
    }
    catch {
        "WARNING: optional diagnostic update failed: $($Arguments -join ' '): $($_.Exception.Message)" | Tee-Object -FilePath $logFile -Append
    }
}

try {
    if ($now.DayOfWeek -in @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday)) {
        $message = "Weekend: no market update was written."
        $success = $true
        return
    }
    if (Test-Path $lockFile) {
        $age = $now - (Get-Item $lockFile).LastWriteTime
        if ($age.TotalHours -lt 3) { throw "Another market update is still active." }
        Remove-Item -LiteralPath $lockFile -Force
    }
    Set-Content -LiteralPath $lockFile -Value "$PID|$startedAt|$Phase" -Encoding utf8
    Set-Location $projectDir
    $env:PYTHONPATH = "src;scripts"
    $today = $now.ToString("yyyy-MM-dd")
    $officialDailyEnd = $now.Date.AddDays(-1).ToString("yyyy-MM-dd")

    if (-not $NoUpdate) {
        if (Test-Path $tokenFile) {
            $env:FINMIND_TOKEN = (Get-Content -Raw -Encoding utf8 $tokenFile).Trim()
        }
        if (-not [string]::IsNullOrWhiteSpace($env:FINMIND_TOKEN)) {
            Invoke-CheckedPython @("$projectDir\scripts\fetch_finmind_factors.py", "--end-date", $today)
            Invoke-CheckedPython @("$projectDir\scripts\fetch_finmind_cross_sectional_breadth.py", "--start-date", $now.AddDays(-14).ToString("yyyy-MM-dd"), "--end-date", $today)
            Invoke-CheckedPython @("$projectDir\scripts\fetch_taifex_vix_daily.py", "--end-date", $officialDailyEnd)
            Invoke-CheckedPython @("$projectDir\scripts\fetch_finmind_futures_tick_bars.py", "--start-date", $now.AddDays(-7).ToString("yyyy-MM-dd"), "--end-date", $today)
            Invoke-CheckedPython @("$projectDir\scripts\build_night_microstructure_features.py")
            if ($Phase -eq "Close") {
                Invoke-OptionalPython @("$projectDir\scripts\fetch_finmind_taiex_early_pulse.py", "--start-date", $now.AddDays(-14).ToString("yyyy-MM-dd"), "--end-date", $today)
                Invoke-OptionalPython @("$projectDir\scripts\fetch_finmind_market_cap_structure.py", "--start-date", $now.AddDays(-14).ToString("yyyy-MM-dd"), "--end-date", $today, "--workers", "2")
            }
        }
        else {
            "FinMind token unavailable; continuing with public market sources." | Tee-Object -FilePath $logFile -Append
        }
    }

    $predictArgs = @(
        "$projectDir\scripts\predict_market.py", "--date", $today,
        "--factor-dir", "$projectDir\data\processed\factors"
    )
    if ($NoUpdate) { $predictArgs += "--no-update" }
    Invoke-CheckedPython $predictArgs
    Invoke-CheckedPython @("$projectDir\scripts\backtest_psychology_state.py")
    Invoke-CheckedPython @("$projectDir\scripts\analyze_previous_forecast_error.py")
    Invoke-CheckedPython @("$projectDir\scripts\track_night_cash_impact.py")
    Invoke-CheckedPython @("$projectDir\scripts\build_market_clinical_ledger.py")
    Invoke-CheckedPython @("$projectDir\scripts\update_prospective_clinical_registry.py")

    $tracking = Get-Content (Join-Path $projectDir "reports\night_cash_impact_tracking.json") -Raw -Encoding utf8 | ConvertFrom-Json
    $latestExpected = if ($Phase -eq "Close") { $today } else { $tracking.latest_date }
    $message = "Completed $Phase update. Tracking latest date: $($tracking.latest_date); expected checkpoint: $latestExpected."
    $success = $true
}
catch {
    $message = $_.Exception.Message
    "ERROR: $message" | Tee-Object -FilePath $logFile -Append
    throw
}
finally {
    $finished = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
    $status = [ordered]@{
        phase = $Phase
        success = $success
        started_at = $startedAt
        finished_at = $finished.ToString("o")
        message = $message
        log_file = $logFile
        no_update_test = [bool]$NoUpdate
    }
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding utf8
    $env:FINMIND_TOKEN = $null
    if (Test-Path $lockFile) { Remove-Item -LiteralPath $lockFile -Force }
}
