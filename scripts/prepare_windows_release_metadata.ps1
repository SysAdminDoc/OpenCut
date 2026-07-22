param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$OutputDirectory = "",
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $RepoRoot "dist\release-metadata"
}

if (-not $PythonExe) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Release Python interpreter is missing: $PythonExe"
}
$server = Join-Path $RepoRoot "dist\OpenCut-Server"
$ffmpeg = Join-Path $RepoRoot "ffmpeg\ffmpeg.exe"
$ffprobe = Join-Path $RepoRoot "ffmpeg\ffprobe.exe"
$extension = Join-Path $RepoRoot "extension\com.opencut.panel"
$provenance = Join-Path $OutputDirectory "ffmpeg-provenance.json"

foreach ($required in @($server, $ffmpeg, $ffprobe, $extension)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Release metadata input is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

& $PythonExe (Join-Path $RepoRoot "scripts\verify_ffmpeg_provenance.py") $ffmpeg `
    --ffprobe $ffprobe `
    --release `
    --source-url "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz" `
    --source-sha256 "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c" `
    --build-origin "gyan.dev release essentials 2026-06-27; FFmpeg commit 38b88335f9" `
    --corresponding-source "Download and verify the FFmpeg 8.1.2 archive named by source.url/source.sha256. The exact binary configuration is in this manifest; gyan.dev/builds identifies release 8.1.2 commit 38b88335f9 and its MSYS2 toolchain." `
    --manifest $provenance
if ($LASTEXITCODE -ne 0) {
    throw "FFmpeg release provenance is incomplete."
}

& $PythonExe (Join-Path $RepoRoot "scripts\release_composition.py") `
    --lane windows `
    --artifact $server `
    --artifact $ffmpeg `
    --artifact $ffprobe `
    --artifact $extension `
    --build-lock (Join-Path $RepoRoot "requirements-build-lock.txt") `
    --ffmpeg-provenance $provenance `
    --output-dir $OutputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Resolved release composition is incomplete."
}

Write-Host "[release-metadata] complete: $OutputDirectory"
