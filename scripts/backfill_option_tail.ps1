param(
    [string]$StartDate = "2020-01-01",
    [string]$EndDate = ""
)

$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { throw "Python was not found." }
$tokenFile = Join-Path $projectDir "config\.secrets\finmind_token.txt"
$logFile = Join-Path $projectDir "reports\option_tail_backfill.log"

try {
    $env:FINMIND_TOKEN = (Get-Content -Raw -Encoding UTF8 $tokenFile).Trim()
    Set-Location $projectDir
    $today = if ([string]::IsNullOrWhiteSpace($EndDate)) { [DateTime]::Now.ToString("yyyy-MM-dd") } else { $EndDate }
    "Option-tail backfill started: $([DateTime]::Now.ToString('s'))" | Add-Content -Encoding UTF8 $logFile
    & $python -u "$projectDir\scripts\fetch_finmind_option_tail_features.py" --start-date $StartDate --end-date $today *>> $logFile
    if ($LASTEXITCODE -ne 0) { throw "Option-tail backfill failed. See $logFile" }
    "Option-tail backfill completed: $([DateTime]::Now.ToString('s'))" | Add-Content -Encoding UTF8 $logFile
}
finally {
    $env:FINMIND_TOKEN = $null
}
