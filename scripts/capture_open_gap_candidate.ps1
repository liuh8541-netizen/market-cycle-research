$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$mutex = New-Object Threading.Mutex($false, "Global\TWII_OpenGap_Q85_Capture")
$ownsMutex = $false

try {
    $ownsMutex = $mutex.WaitOne(0)
    if (-not $ownsMutex) { exit 0 }

    $python = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { exit 0 }
    $logger = Join-Path $projectDir "scripts\forecast_prospective_open_gap_cash_close.py"
    $taipeiZone = [TimeZoneInfo]::FindSystemTimeZoneById("Taipei Standard Time")
    $taipeiNow = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
    $deadline = $taipeiNow.Date.AddHours(13).AddMinutes(25)

    while ($taipeiNow -lt $deadline) {
        $now = $taipeiNow
        if ($now.DayOfWeek -in @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday)) { exit 0 }
        if ($now.TimeOfDay -ge [TimeSpan]::FromHours(9)) {
            Set-Location $projectDir
            & $python $logger
            $logPath = Join-Path $projectDir "reports\prospective_open_gap_cash_close_log.json"
            if (Test-Path $logPath) {
                $today = $now.ToString("yyyy-MM-dd")
                $records = Get-Content -Raw -Encoding UTF8 $logPath | ConvertFrom-Json
                if ($records | Where-Object { $_.signal_date -eq $today }) { exit 0 }
            }
        }
        Start-Sleep -Seconds 120
        $taipeiNow = [TimeZoneInfo]::ConvertTimeFromUtc([DateTime]::UtcNow, $taipeiZone)
    }
}
finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
