<#
.SYNOPSIS
    OpenCut Custom Installer Build Script
.DESCRIPTION
    Builds the custom WPF installer, stages payload, creates self-extracting exe.
    Run from the OpenCut repo root.
#>

param(
    [switch]$SkipPublish,
    [switch]$PayloadOnly
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$InstallerProj = Join-Path $ScriptDir "src\OpenCut.Installer\OpenCut.Installer.csproj"
$PublishDir = Join-Path $ScriptDir "publish"
$StagingDir = Join-Path $ScriptDir "staging"
$OutputDir = Join-Path $ScriptDir "dist"

# Read version from opencut/__init__.py
$InitPy = Join-Path $RepoRoot "opencut\__init__.py"
$Version = "1.3.0"
if (Test-Path $InitPy) {
    $match = Select-String -Path $InitPy -Pattern '__version__\s*=\s*"([^"]+)"'
    if ($match) { $Version = $match.Matches[0].Groups[1].Value }
}

Write-Host ""
Write-Host "  OpenCut Installer Builder" -ForegroundColor Cyan
Write-Host "  Version: $Version" -ForegroundColor DarkCyan
Write-Host "  =========================" -ForegroundColor DarkGray
Write-Host ""

function Write-Step($msg) { Write-Host "  [*] $msg" -ForegroundColor White }
function Write-Ok($msg) { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "  [!!] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [!] $msg" -ForegroundColor Yellow }

# ── Step 1: Verify prerequisites ──────────────────────────────────────────
Write-Step "Checking prerequisites..."

# Check dotnet SDK
$dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
if (-not $dotnet) {
    Write-Err "dotnet SDK not found. Install .NET 9 SDK from https://dot.net"
    exit 1
}
$sdkVersion = & dotnet --version 2>$null
Write-Ok "dotnet SDK: $sdkVersion"

# Check csproj exists
if (-not (Test-Path $InstallerProj)) {
    Write-Err "Installer project not found: $InstallerProj"
    exit 1
}
Write-Ok "Installer project found"

# Check payload sources
$distServer = Join-Path $RepoRoot "dist\OpenCut-Server"
$ffmpegDir = Join-Path $RepoRoot "ffmpeg"
$extensionDir = Join-Path $RepoRoot "extension\com.opencut.panel"
$launcherVbs = Join-Path $RepoRoot "OpenCut-Launcher.vbs"
$logoIco = Join-Path $RepoRoot "img\logo.ico"

$payloadReady = $true
if (-not (Test-Path $distServer)) { Write-Warn "dist/OpenCut-Server/ not found (run PyInstaller first)"; $payloadReady = $false }
if (-not (Test-Path $ffmpegDir)) { Write-Warn "ffmpeg/ not found"; $payloadReady = $false }
if (-not (Test-Path $extensionDir)) { Write-Warn "extension/com.opencut.panel/ not found"; $payloadReady = $false }
if (-not (Test-Path $launcherVbs)) { Write-Warn "OpenCut-Launcher.vbs not found"; $payloadReady = $false }
if (-not (Test-Path $logoIco)) { Write-Warn "img/logo.ico not found"; $payloadReady = $false }

if ($payloadReady) { Write-Ok "All payload sources found" }
else { Write-Warn "Some payload sources missing - payload.zip will be incomplete" }

Write-Host ""

# ── Step 2: Publish installer ─────────────────────────────────────────────
if (-not $SkipPublish -and -not $PayloadOnly) {
    Write-Step "Publishing installer (self-contained, win-x64)..."

    if (Test-Path $PublishDir) { Remove-Item $PublishDir -Recurse -Force }

    & dotnet publish $InstallerProj `
        -c Release `
        -r win-x64 `
        --self-contained true `
        -p:PublishSingleFile=true `
        -p:IncludeNativeLibrariesForSelfExtract=true `
        -p:EnableCompressionInSingleFile=true `
        -o $PublishDir

    if ($LASTEXITCODE -ne 0) {
        Write-Err "dotnet publish failed with exit code $LASTEXITCODE"
        exit 1
    }

    $publishedExe = Join-Path $PublishDir "OpenCut-Setup.exe"
    if (-not (Test-Path $publishedExe)) {
        Write-Err "Published exe not found at $publishedExe"
        exit 1
    }

    $exeSize = [math]::Round((Get-Item $publishedExe).Length / 1MB, 2)
    Write-Ok "Installer published ($exeSize MB)"
}

if ($PayloadOnly) {
    Write-Step "Skipping installer publish (PayloadOnly mode)"
}

Write-Host ""

# ── Step 3: Stage payload ─────────────────────────────────────────────────
Write-Step "Staging payload directory..."

if (Test-Path $StagingDir) { Remove-Item $StagingDir -Recurse -Force }
New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

# Copy server
if (Test-Path $distServer) {
    Write-Step "  Copying server files..."
    Copy-Item -Path $distServer -Destination (Join-Path $StagingDir "server") -Recurse
    $serverFiles = (Get-ChildItem (Join-Path $StagingDir "server") -Recurse -File).Count
    Write-Ok "  Server: $serverFiles files"
}

# Copy ffmpeg
if (Test-Path $ffmpegDir) {
    Write-Step "  Copying FFmpeg..."
    $ffmpegDest = Join-Path $StagingDir "ffmpeg"
    New-Item -ItemType Directory -Path $ffmpegDest -Force | Out-Null
    Copy-Item -Path (Join-Path $ffmpegDir "ffmpeg.exe") -Destination $ffmpegDest -ErrorAction SilentlyContinue
    Copy-Item -Path (Join-Path $ffmpegDir "ffprobe.exe") -Destination $ffmpegDest -ErrorAction SilentlyContinue
    Write-Ok "  FFmpeg copied"
}

# Copy extension
if (Test-Path $extensionDir) {
    Write-Step "  Copying CEP extension..."
    $extDest = Join-Path $StagingDir "extension\com.opencut.panel"
    Copy-Item -Path $extensionDir -Destination $extDest -Recurse
    Write-Ok "  Extension copied"
}

# Copy launcher + icon
if (Test-Path $launcherVbs) {
    Copy-Item -Path $launcherVbs -Destination $StagingDir
    Write-Ok "  Launcher VBS copied"
}
if (Test-Path $logoIco) {
    Copy-Item -Path $logoIco -Destination $StagingDir
    Write-Ok "  Logo icon copied"
}

Write-Host ""

# ── Step 4: Create payload.zip ────────────────────────────────────────────
Write-Step "Creating payload.zip..."

$payloadZip = Join-Path $OutputDir "payload.zip"
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }
if (Test-Path $payloadZip) { Remove-Item $payloadZip -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($StagingDir, $payloadZip, [System.IO.Compression.CompressionLevel]::Optimal, $false)

$zipSize = [math]::Round((Get-Item $payloadZip).Length / 1MB, 2)
Write-Ok "Payload zip created ($zipSize MB)"

Write-Host ""

if ($PayloadOnly) {
    Write-Ok "Payload-only build complete: $payloadZip"
    # Cleanup staging
    Remove-Item $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
    exit 0
}

# ── Step 5: Create self-extracting exe ────────────────────────────────────
Write-Step "Creating self-extracting installer..."

$publishedExe = Join-Path $PublishDir "OpenCut-Setup.exe"
$finalExe = Join-Path $OutputDir "OpenCut-Setup-$Version.exe"

if (-not (Test-Path $publishedExe)) {
    Write-Warn "Published installer not found, copying payload.zip alongside for fallback mode"
    # Just copy the published exe without payload appended
    Write-Ok "Build complete (adjacent payload.zip mode)"
} else {
    # Concatenate: installer.exe + payload.zip + size(8 bytes LE) + "OCPAYLOAD"(9 bytes)
    $payloadBytes = [System.IO.File]::ReadAllBytes($payloadZip)
    $sizeBytes = [BitConverter]::GetBytes([long]$payloadBytes.Length)
    $magicBytes = [System.Text.Encoding]::ASCII.GetBytes("OCPAYLOAD")

    $installerBytes = [System.IO.File]::ReadAllBytes($publishedExe)

    $totalSize = $installerBytes.Length + $payloadBytes.Length + $sizeBytes.Length + $magicBytes.Length

    $fs = [System.IO.File]::Create($finalExe)
    try {
        $fs.Write($installerBytes, 0, $installerBytes.Length)
        $fs.Write($payloadBytes, 0, $payloadBytes.Length)
        $fs.Write($sizeBytes, 0, $sizeBytes.Length)
        $fs.Write($magicBytes, 0, $magicBytes.Length)
    } finally {
        $fs.Dispose()
    }

    $finalSize = [math]::Round((Get-Item $finalExe).Length / 1MB, 2)
    Write-Ok "Self-extracting installer created: $finalExe ($finalSize MB)"
}

Write-Host ""

# ── Cleanup ───────────────────────────────────────────────────────────────
Write-Step "Cleaning up..."
Remove-Item $StagingDir -Recurse -Force -ErrorAction SilentlyContinue
Write-Ok "Staging directory removed"

Write-Host ""
Write-Host "  Build complete!" -ForegroundColor Green
Write-Host "  Output: $finalExe" -ForegroundColor Cyan
Write-Host ""
