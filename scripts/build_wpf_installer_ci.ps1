<#
.SYNOPSIS
    Build the recommended WPF installer for local release validation.
.DESCRIPTION
    F201 compatibility wrapper around installer/InstallerBuilder.ps1. The
    local release build runs PyInstaller first, then this script copies
    FFmpeg/ffprobe from PATH into the repo-local ffmpeg/ payload folder and
    archives the WPF output in installer/dist/wpf/ so the later Inno fallback
    build cannot overwrite it.
#>

param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RepoPath([string]$RelativePath) {
    return Join-Path $RepoRoot $RelativePath
}

function Copy-ToolFromPath([string]$ToolName, [string]$TargetDir) {
    $target = Join-Path $TargetDir "$ToolName.exe"
    if (Test-Path $target) {
        Write-Host "[wpf-installer] using existing $target"
        return
    }

    $command = Get-Command "$ToolName.exe" -ErrorAction Stop
    Copy-Item -LiteralPath $command.Source -Destination $target -Force
    Write-Host "[wpf-installer] copied $ToolName from $($command.Source)"
}

if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    throw "build_wpf_installer_ci.ps1 requires Windows."
}

$builder = Resolve-RepoPath "installer\InstallerBuilder.ps1"
$serverDist = Resolve-RepoPath "dist\OpenCut-Server"
$ffmpegDir = Resolve-RepoPath "ffmpeg"
$distDir = Resolve-RepoPath "installer\dist"
$wpfArchiveDir = Join-Path $distDir "wpf"

if (-not (Test-Path $builder)) {
    throw "Installer builder missing: $builder"
}
if (-not (Test-Path $serverDist)) {
    throw "PyInstaller payload missing: $serverDist. Run pyinstaller opencut_server.spec first."
}

New-Item -ItemType Directory -Force -Path $ffmpegDir | Out-Null
Copy-ToolFromPath "ffmpeg" $ffmpegDir
Copy-ToolFromPath "ffprobe" $ffmpegDir

Write-Host "[wpf-installer] running $builder"
& $builder
if ($LASTEXITCODE -ne 0) {
    throw "InstallerBuilder.ps1 failed with exit code $LASTEXITCODE"
}

$installer = Get-ChildItem -Path $distDir -Filter "OpenCut-Setup-*.exe" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $installer) {
    throw "WPF installer output not found in $distDir"
}

$version = $installer.BaseName -replace "^OpenCut-Setup-", ""
New-Item -ItemType Directory -Force -Path $wpfArchiveDir | Out-Null
$archived = Join-Path $wpfArchiveDir "OpenCut-WPF-Setup-$version.exe"
Copy-Item -LiteralPath $installer.FullName -Destination $archived -Force

if ($env:OPENCUT_WPF_INSTALLER_OUTPUT) {
    "wpf_installer=$archived" | Out-File -FilePath $env:OPENCUT_WPF_INSTALLER_OUTPUT -Append -Encoding utf8
}

Write-Host "[wpf-installer] archived WPF installer: $archived"
