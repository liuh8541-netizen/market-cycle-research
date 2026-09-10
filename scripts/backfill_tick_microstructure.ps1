param(
    [string]$StartDate = "2020-01-01",
    [string]$EndDate = ""
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python was not found." }
$tokenFile = Join-Path $projectDir "config\.secrets\finmind_token.txt"
$logFile = Join-Path $projectDir "reports\tick_microstructure_backfill.log"

try {
    if (!(Test-Path $tokenFile)) {
        throw "Saved FinMind Token was not found. Run the normal forecast once for first-time setup."
    }
    $env:FINMIND_TOKEN = (Get-Content -Raw -Encoding UTF8 $tokenFile).Trim()
    if ([string]::IsNullOrWhiteSpace($env:FINMIND_TOKEN)) {
        throw "Saved FinMind Token is empty."
    }
    Set-Location $projectDir
    $today = if ([string]::IsNullOrWhiteSpace($EndDate)) { [DateTime]::Now.ToString("yyyy-MM-dd") } else { $EndDate }
    "Backfill started: $([DateTime]::Now.ToString('s'))" | Add-Content -Encoding UTF8 $logFile
    & $python -u "$projectDir\scripts\fetch_finmind_futures_tick_bars.py" --start-date $StartDate --end-date $today *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "Tick-bar backfill failed. See $logFile" }
    & $python -u "$projectDir\scripts\build_night_microstructure_features.py" *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "Microstructure feature build failed. See $logFile" }
    & $python -u "$projectDir\scripts\audit_night_microstructure_integrity.py" *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "Microstructure integrity audit failed. See $logFile" }
    & $python -u "$projectDir\scripts\research_night_microstructure_gap.py" *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "Microstructure walk-forward research failed. See $logFile" }
    "Backfill completed: $([DateTime]::Now.ToString('s'))" | Add-Content -Encoding UTF8 $logFile
}
catch {
    "Backfill error: $($_.Exception.Message)" | Add-Content -Encoding UTF8 $logFile
    throw
}
finally {
    $env:FINMIND_TOKEN = $null
}
