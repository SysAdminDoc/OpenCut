<#
.SYNOPSIS
Starts an installed OpenCut-Server.exe and checks /health answers before
capability discovery finishes.

.DESCRIPTION
The server answers /health from cheap facts and probes optional packages
(Torch, ONNX Runtime, Whisper and the rest) in a child process. This starts
the given server on a free loopback port with a disposable data folder,
expects the first /health answer while discovery is still pending, then waits
for discovery to finish and fails on any probe that crashed or hung. The
server and its probe children are stopped afterwards.

The installer smokes call this against the server they just installed. It
can also run on its own against dist\OpenCut-Server\OpenCut-Server.exe.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ServerExe,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [int]$StartupSeconds = 90,
    [int]$DiscoverySeconds = 300
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$serverFull = [System.IO.Path]::GetFullPath($ServerExe)
if (-not (Test-Path -LiteralPath $serverFull)) {
    throw "Server executable missing at $serverFull"
}
$dataFull = [System.IO.Path]::GetFullPath($DataDir)
New-Item -ItemType Directory -Force -Path $dataFull | Out-Null

# A free port, so the server never runs its kill sequence on a live instance.
$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = $listener.LocalEndpoint.Port
$listener.Stop()

$startInfo = [System.Diagnostics.ProcessStartInfo]::new($serverFull)
$startInfo.Arguments = "--host 127.0.0.1 --port $port"
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.WorkingDirectory = Split-Path -Parent $serverFull
$startInfo.Environment["OPENCUT_HOME"] = $dataFull
foreach ($name in @("PYTHONPATH", "PYTHONHOME", "OPENCUT_PORT", "OPENCUT_HOST")) {
    [void]$startInfo.Environment.Remove($name)
}

function Get-LogTail {
    $lines = @()
    foreach ($name in @("server.log", "crash.log")) {
        $path = Join-Path $dataFull $name
        if (Test-Path -LiteralPath $path) {
            $lines += "--- $name"
            $lines += Get-Content -LiteralPath $path -Tail 20
        }
    }
    return ($lines -join [Environment]::NewLine)
}

function Get-Health {
    param([int]$TimeoutSec = 5)
    return Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec $TimeoutSec -UseBasicParsing
}

# Strict mode throws on a missing property, which would hide the real failure
# behind a PropertyNotFoundException.
function Get-Field {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($property) { return $property.Value }
    return $null
}

Write-Host "[F447] Starting $serverFull on 127.0.0.1:$port"
$started = [DateTime]::UtcNow
$process = [System.Diagnostics.Process]::Start($startInfo)
try {
    $health = $null
    $deadline = $started.AddSeconds($StartupSeconds)
    while (-not $health) {
        if ($process.HasExited) {
            throw "Server exited with code $($process.ExitCode) before /health answered.`n$(Get-LogTail)"
        }
        if ([DateTime]::UtcNow -gt $deadline) {
            throw "/health did not answer within $StartupSeconds seconds.`n$(Get-LogTail)"
        }
        try {
            $health = Get-Health
        }
        catch {
            Start-Sleep -Milliseconds 250
        }
    }
    $firstAnswer = ([DateTime]::UtcNow - $started).TotalSeconds
    $status = Get-Field $health "status"
    if ($status -ne "ok") {
        throw "/health answered status=$status, expected ok"
    }
    $discovery = Get-Field $health "capability_discovery"
    if (-not $discovery) {
        throw "/health has no capability_discovery block"
    }
    # A fresh data folder has no saved probe results, and discovery starts on
    # the first /health, so that answer must come before any probe finished.
    $state = Get-Field $discovery "state"
    $pending = @(Get-Field $discovery "pending")
    if (($state -eq "ready") -or ($pending.Count -eq 0)) {
        throw "The first /health answer came after discovery finished (state=$state); it should not wait for the probes"
    }
    Write-Host ("[F447] /health answered after {0:N1} s with discovery {1}, {2} probes pending" -f $firstAnswer, $state, $pending.Count)

    $deadline = [DateTime]::UtcNow.AddSeconds($DiscoverySeconds)
    while ($state -ne "ready") {
        if ($process.HasExited) {
            throw "Server exited with code $($process.ExitCode) during capability discovery.`n$(Get-LogTail)"
        }
        if ([DateTime]::UtcNow -gt $deadline) {
            throw "Capability discovery did not finish within $DiscoverySeconds seconds (state=$state, pending: $(@(Get-Field $discovery 'pending') -join ', '))"
        }
        Start-Sleep -Seconds 1
        $discovery = Get-Field (Get-Health -TimeoutSec 10) "capability_discovery"
        if (-not $discovery) {
            throw "/health stopped reporting capability_discovery"
        }
        $state = Get-Field $discovery "state"
    }
    $total = ([DateTime]::UtcNow - $started).TotalSeconds

    $broken = @()
    $failed = Get-Field $discovery "failed"
    $entries = if ($failed) { @($failed.PSObject.Properties) } else { @() }
    foreach ($entry in $entries) {
        $result = $entry.Value
        $resultState = Get-Field $result "state"
        Write-Host "[F447] probe $($entry.Name): $resultState $(Get-Field $result 'reason')"
        if (($resultState -eq "crashed") -or ($resultState -eq "timeout")) {
            $broken += $entry.Name
        }
    }
    if ($broken.Count -gt 0) {
        throw "Capability probes crashed or hung: $($broken -join ', ').`n$(Get-LogTail)"
    }
    Write-Host ("[F447] Capability discovery finished after {0:N1} s" -f $total)
}
finally {
    if (-not $process.HasExited) {
        # /T takes the probe children with it. Process.Kill($true) would do
        # the same but needs PowerShell 7.
        & taskkill.exe /PID $process.Id /T /F | Out-Null
        [void]$process.WaitForExit(15000)
    }
    $process.Dispose()
}

Write-Host "[F447] Server health smoke passed"
