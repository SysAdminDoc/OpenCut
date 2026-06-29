<#
.SYNOPSIS
Local smoke test for the legacy Inno Setup installer.

.DESCRIPTION
Installs the generated OpenCut-Setup-*.exe into a temporary directory,
verifies the user-visible payload and machine-readable installer manifest,
then runs the generated uninstaller and verifies cleanup.

The Inno uninstaller removes the user's ~/.opencut directory by design.
For that reason this script refuses to run unless OPENCUT_INSTALLER_SMOKE=1
is set or -AllowLocalProfileMutation is passed explicitly.
#>

[CmdletBinding()]
param(
    [string]$InstallerPath = "",
    [string]$InstallRoot = "",
    [switch]$KeepArtifacts,
    [switch]$AllowLocalProfileMutation
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-FullPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    return [System.IO.Path]::GetFullPath($Path)
}

function Assert-UnderDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Child,
        [Parameter(Mandatory = $true)][string]$Parent,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $childFull = Get-FullPath $Child
    $parentFull = (Get-FullPath $Parent).TrimEnd('\', '/')
    $prefix = $parentFull + [System.IO.Path]::DirectorySeparatorChar
    if (($childFull -ne $parentFull) -and (-not $childFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "$Label must stay under $parentFull; got $childFull"
    }
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Label
    )
    Write-Host "[F213] $Label"
    $process = Start-Process -FilePath $FilePath -ArgumentList $Arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($process.ExitCode)"
    }
}

function Assert-Exists {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label missing at $Path"
    }
}

function Assert-Removed {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (Test-Path -LiteralPath $Path) {
        throw "$Label should have been removed: $Path"
    }
}

$automationOptIn = $env:OPENCUT_INSTALLER_SMOKE -eq "1"
if ((-not $automationOptIn) -and (-not $AllowLocalProfileMutation)) {
    throw "Refusing to run without explicit local smoke opt-in because the Inno uninstaller deletes ~/.opencut. Pass -AllowLocalProfileMutation or set OPENCUT_INSTALLER_SMOKE=1 for disposable release validation."
}

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $InstallerPath) {
    $candidate = Get-ChildItem -Path (Join-Path $repoRoot "installer\dist") -Filter "OpenCut-Setup-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No OpenCut-Setup-*.exe found under installer\dist; build the Inno installer first."
    }
    $InstallerPath = $candidate.FullName
}

$installerFull = Get-FullPath $InstallerPath
Assert-Exists $installerFull "Inno installer"

$tempBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$tempFull = Get-FullPath $tempBase

if (-not $InstallRoot) {
    $InstallRoot = Join-Path $tempFull ("OpenCut-Inno-Smoke-" + [System.Guid]::NewGuid().ToString("N"))
}

$installFull = Get-FullPath $InstallRoot
Assert-UnderDirectory -Child $installFull -Parent $tempFull -Label "InstallRoot"

$logDir = Join-Path $tempFull "OpenCut-Inno-Smoke-Logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$installLog = Join-Path $logDir "install.log"
$uninstallLog = Join-Path $logDir "uninstall.log"

try {
    if (Test-Path -LiteralPath $installFull) {
        Remove-Item -LiteralPath $installFull -Recurse -Force
    }

    $installArgs = @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/TASKS=""""",
        "/DIR=$installFull",
        "/LOG=$installLog"
    )
    Invoke-CheckedProcess -FilePath $installerFull -Arguments $installArgs -Label "Installing OpenCut with Inno"

    $expectedPayload = @(
        "server\OpenCut-Server.exe",
        "ffmpeg\ffmpeg.exe",
        "ffmpeg\ffprobe.exe",
        "OpenCut-Launcher.vbs",
        "extension\com.opencut.panel",
        "logo.ico"
    )
    foreach ($relative in $expectedPayload) {
        Assert-Exists (Join-Path $installFull $relative) "Installed payload $relative"
    }

    $manifestPath = Join-Path $env:USERPROFILE ".opencut\installer.json"
    Assert-Exists $manifestPath "Inno installer manifest"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.installer_kind -ne "inno") {
        throw "installer.json should report installer_kind=inno; got $($manifest.installer_kind)"
    }
    if ((Get-FullPath $manifest.install_path) -ne $installFull) {
        throw "installer.json install_path mismatch: expected $installFull, got $($manifest.install_path)"
    }
    if (-not $manifest.bundled_ffmpeg_version) {
        throw "installer.json missing bundled_ffmpeg_version"
    }

    $registryPath = "HKCU:\Software\OpenCut"
    Assert-Exists $registryPath "OpenCut HKCU registry key"
    $registryInstallPath = (Get-ItemProperty -LiteralPath $registryPath).InstallPath
    if ((Get-FullPath $registryInstallPath) -ne $installFull) {
        throw "Registry InstallPath mismatch: expected $installFull, got $registryInstallPath"
    }

    $uninstaller = Get-ChildItem -LiteralPath $installFull -Filter "unins*.exe" |
        Sort-Object Name |
        Select-Object -First 1
    if (-not $uninstaller) {
        throw "Generated Inno uninstaller not found under $installFull"
    }

    $uninstallArgs = @(
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/LOG=$uninstallLog"
    )
    Invoke-CheckedProcess -FilePath $uninstaller.FullName -Arguments $uninstallArgs -Label "Uninstalling OpenCut with Inno"

    Assert-Removed (Join-Path $installFull "server") "Installed server directory"
    Assert-Removed (Join-Path $installFull "ffmpeg") "Installed FFmpeg directory"
    Assert-Removed $registryPath "OpenCut HKCU registry key"
    Assert-Removed $manifestPath "Inno installer manifest"
}
finally {
    if ((-not $KeepArtifacts) -and (Test-Path -LiteralPath $installFull)) {
        Assert-UnderDirectory -Child $installFull -Parent $tempFull -Label "Cleanup target"
        Remove-Item -LiteralPath $installFull -Recurse -Force
    }
}

Write-Host "[F213] Inno installer install/uninstall smoke passed"
