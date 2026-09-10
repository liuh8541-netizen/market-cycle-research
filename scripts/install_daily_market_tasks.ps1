$ErrorActionPreference = "Stop"
$projectDir = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $PSScriptRoot "run_daily_market_update.ps1"
$powershell = (Get-Command powershell.exe).Source
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)

function Install-MarketTask {
    param([string]$Name, [string]$Phase, [datetime]$At)
    $arguments = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$runner`" -Phase $Phase"
    $action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $projectDir
    $trigger = New-ScheduledTaskTrigger -Weekly -WeeksInterval 1 -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At $At
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description "Update Taiwan night/cash market tracking and reports." -Force | Out-Null
}

Install-MarketTask -Name "TaiwanMarket-NightUpdate" -Phase "Night" -At "05:20"
Install-MarketTask -Name "TaiwanMarket-CloseUpdate" -Phase "Close" -At "14:20"
Write-Host "Installed TaiwanMarket-NightUpdate (05:20) and TaiwanMarket-CloseUpdate (14:20), weekdays."
