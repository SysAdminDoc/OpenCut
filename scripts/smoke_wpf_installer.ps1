<#
.SYNOPSIS
Local smoke test for the recommended WPF installer.

.DESCRIPTION
Runs the built self-extracting WPF installer in quiet mode against a temporary
install root and temporary OpenCut profile, verifies the installed payload and
installer manifest, then runs the quiet uninstaller.
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
    Write-Host "[F212] $Label"
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
    throw "Refusing to run without explicit local smoke opt-in. Pass -AllowLocalProfileMutation or set OPENCUT_INSTALLER_SMOKE=1 for disposable release validation."
}

if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    throw "smoke_wpf_installer.ps1 requires Windows."
}

$repoRoot = Split-Path -Parent $PSScriptRoot

if (-not $InstallerPath) {
    $candidate = Get-ChildItem -Path (Join-Path $repoRoot "installer\dist\wpf") -Filter "OpenCut-WPF-Setup-*.exe" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No OpenCut-WPF-Setup-*.exe found under installer\dist\wpf; build the WPF installer first."
    }
    $InstallerPath = $candidate.FullName
}

$installerFull = Get-FullPath $InstallerPath
Assert-Exists $installerFull "WPF installer"

$tempBase = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [System.IO.Path]::GetTempPath() }
$tempFull = Get-FullPath $tempBase

if (-not $InstallRoot) {
    $InstallRoot = Join-Path $tempFull ("OpenCut-WPF-Smoke-" + [System.Guid]::NewGuid().ToString("N"))
}

$installFull = Get-FullPath $InstallRoot
Assert-UnderDirectory -Child $installFull -Parent $tempFull -Label "InstallRoot"

$userDataDir = Join-Path $tempFull ("OpenCut-WPF-Profile-" + [System.Guid]::NewGuid().ToString("N"))
Assert-UnderDirectory -Child $userDataDir -Parent $tempFull -Label "UserDataDir"

$logDir = Join-Path $tempFull "OpenCut-WPF-Smoke-Logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$installLog = Join-Path $logDir "install.log"
$uninstallLog = Join-Path $logDir "uninstall.log"

try {
    if (Test-Path -LiteralPath $installFull) {
        Remove-Item -LiteralPath $installFull -Recurse -Force
    }

    $commonSafetyArgs = @(
        "--dir=$installFull",
        "--user-data-dir=$userDataDir",
        "--no-desktop-shortcut",
        "--no-start-menu-shortcut",
        "--no-startup-shortcut",
        "--no-cep-extension",
        "--no-player-debug-mode",
        "--no-path-update",
        "--no-register-uninstaller",
        "--no-whisper-model",
        "--no-optional-deps"
    )

    $installArgs = @("--install", "--quiet", "--log=$installLog") + $commonSafetyArgs
    Invoke-CheckedProcess -FilePath $installerFull -Arguments $installArgs -Label "Installing OpenCut with WPF quiet mode"

    $expectedPayload = @(
        "server\OpenCut-Server.exe",
        "ffmpeg\ffmpeg.exe",
        "ffmpeg\ffprobe.exe",
        "release-metadata\release-composition.json",
        "release-metadata\opencut-artifact-sbom.cyclonedx.json",
        "release-metadata\THIRD-PARTY-NOTICES.txt",
        "release-metadata\ffmpeg-provenance.json",
        "LICENSE",
        "OpenCut-Launcher.vbs",
        "extension\com.opencut.panel",
        "logo.ico",
        "OpenCut-Uninstall.exe"
    )
    foreach ($relative in $expectedPayload) {
        Assert-Exists (Join-Path $installFull $relative) "Installed payload $relative"
    }

    $manifestPath = Join-Path $userDataDir "installer.json"
    Assert-Exists $manifestPath "WPF installer manifest"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.installer_kind -ne "wpf") {
        throw "installer.json should report installer_kind=wpf; got $($manifest.installer_kind)"
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

    $uninstaller = Join-Path $installFull "OpenCut-Uninstall.exe"
    $uninstallArgs = @("--uninstall", "--quiet", "--log=$uninstallLog") + $commonSafetyArgs
    Invoke-CheckedProcess -FilePath $uninstaller -Arguments $uninstallArgs -Label "Uninstalling OpenCut with WPF quiet mode"

    Assert-Removed (Join-Path $installFull "server") "Installed server directory"
    Assert-Removed (Join-Path $installFull "ffmpeg") "Installed FFmpeg directory"
    Assert-Removed $registryPath "OpenCut HKCU registry key"
    Assert-Removed $manifestPath "WPF installer manifest"
}
finally {
    Remove-Item -LiteralPath "HKCU:\Software\OpenCut" -Recurse -Force -ErrorAction SilentlyContinue
    if ((-not $KeepArtifacts) -and (Test-Path -LiteralPath $installFull)) {
        Assert-UnderDirectory -Child $installFull -Parent $tempFull -Label "Cleanup target"
        Remove-Item -LiteralPath $installFull -Recurse -Force
    }
    if ((-not $KeepArtifacts) -and (Test-Path -LiteralPath $userDataDir)) {
        Assert-UnderDirectory -Child $userDataDir -Parent $tempFull -Label "Profile cleanup target"
        Remove-Item -LiteralPath $userDataDir -Recurse -Force
    }
}

Write-Host "[F212] WPF installer quiet install/uninstall smoke passed"
