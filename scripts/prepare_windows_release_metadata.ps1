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
    --source-url "https://github.com/FFmpeg/FFmpeg/archive/23c454005319dee4a5bcae5c370f75fdeb325230.tar.gz" `
    --source-sha256 "81c206f525b8206a5e3406fd0aacd92fd7070dff6a5cf28842b82e65640aa1e1" `
    --package-url "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-2026-11-11-git-23c4540053-full_build.7z" `
    --package-sha256 "6fa482b30da6b3a5b51ae4776d1873488a949724d0302aa5e96cad143e483776" `
    --build-origin "gyan.dev git-full snapshot 2026-11-11; FFmpeg commit 23c454005319dee4a5bcae5c370f75fdeb325230" `
    --corresponding-source "Download and verify the exact FFmpeg source archive at source.url/source.sha256, then build commit 23c454005319dee4a5bcae5c370f75fdeb325230 with the recorded configuration. The Windows payload is the Gyan git-full package at package.url/package.sha256." `
    --require-pinned-snapshot `
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
