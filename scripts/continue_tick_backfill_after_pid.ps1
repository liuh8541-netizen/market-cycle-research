param(
    [Parameter(Mandatory = $true)]
    [int]$PriorProcessId
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$logFile = Join-Path $projectDir "reports\tick_microstructure_backfill.log"

Wait-Process -Id $PriorProcessId -ErrorAction SilentlyContinue
"Extended 2020 backfill starting after PID ${PriorProcessId}: $([DateTime]::Now.ToString('s'))" | Add-Content -Encoding UTF8 $logFile
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PSScriptRoot\backfill_tick_microstructure.ps1" -StartDate "2020-01-01"
if ($LASTEXITCODE -ne 0) {
    "Extended 2020 backfill failed with exit code $LASTEXITCODE." | Add-Content -Encoding UTF8 $logFile
    exit $LASTEXITCODE
}
