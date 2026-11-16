<#
.SYNOPSIS
    Build the Windows release (panel bundle, frozen server, WPF installer) in one command.
.DESCRIPTION
    Builds with a Python environment made only from the hashed locks
    (requirements-release-lock.txt and requirements-build-lock.txt), kept in
    build\release-venv and rebuilt whenever either lock or the base Python
    changes. A machine-wide Python carries whatever else is installed: two Qt
    bindings stop PyInstaller, and InstallerBuilder.ps1's composition check
    refuses anything that isn't an exact lock match.

    Needs the .NET 10 SDK, Node.js with npm, and the pinned FFmpeg build in
    ffmpeg\ffmpeg.exe and ffmpeg\ffprobe.exe (see installer\InstallerBuilder.ps1
    for the package URL and hash). Output lands in installer\dist\.
#>

param(
    # python.exe to create the build environment from. Defaults to `py -3.13`, then python.exe on PATH.
    [string]$BasePython = "",
    # Keep the existing extension\com.opencut.panel\client\dist bundle instead of rebuilding it.
    [switch]$SkipPanel,
    # Rebuild build\release-venv even when the locks haven't changed.
    [switch]$RecreateVenv
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvDir = Join-Path $RepoRoot 'build\release-venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$StampPath = Join-Path $VenvDir 'opencut-lock-stamp.txt'
$ReleaseLock = Join-Path $RepoRoot 'requirements-release-lock.txt'
$BuildLock = Join-Path $RepoRoot 'requirements-build-lock.txt'
$PanelDir = Join-Path $RepoRoot 'extension\com.opencut.panel'
$Builder = Join-Path $RepoRoot 'installer\InstallerBuilder.ps1'

function Write-BuildStep([string]$Message) {
    Write-Host "[build-release] $Message"
}

function Invoke-Checked([string]$What, [scriptblock]$Command) {
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

function Resolve-BasePython {
    if ($BasePython) {
        return (Resolve-Path -LiteralPath $BasePython).Path
    }
    $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($launcher) {
        $found = & $launcher.Source -3.13 -c 'import sys; print(sys.executable)'
        if ($LASTEXITCODE -eq 0 -and $found) {
            return ([string]$found).Trim()
        }
    }
    $onPath = Get-Command 'python.exe' -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }
    throw 'No Python found. Install Python 3.13 or pass -BasePython <path to python.exe>.'
}

if (-not [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)) {
    throw 'build_release.ps1 builds the Windows installer and needs Windows.'
}

# Fail before the long steps when a payload input the installer needs is missing.
foreach ($tool in @('ffmpeg.exe', 'ffprobe.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $RepoRoot "ffmpeg\$tool") -PathType Leaf)) {
        throw "ffmpeg\$tool is missing. Extract the pinned FFmpeg package named in installer\InstallerBuilder.ps1 into ffmpeg\ first."
    }
}

$base = Resolve-BasePython
$baseVersion = ([string](& $base -c 'import sys; print(sys.version)')).Trim()
$stampText = @(
    "base=$baseVersion",
    "release-lock=$((Get-FileHash -LiteralPath $ReleaseLock -Algorithm SHA256).Hash)",
    "build-lock=$((Get-FileHash -LiteralPath $BuildLock -Algorithm SHA256).Hash)"
) -join "`n"
$currentStamp = ''
if (Test-Path -LiteralPath $StampPath) {
    $currentStamp = ([IO.File]::ReadAllText($StampPath)).Trim()
}

if ($RecreateVenv -or $currentStamp -ne $stampText -or -not (Test-Path -LiteralPath $VenvPython)) {
    Write-BuildStep "creating build\release-venv from the hashed locks ($base)"
    if (Test-Path -LiteralPath $VenvDir) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    }
    Invoke-Checked 'venv creation' { & $base -m venv $VenvDir }
    Invoke-Checked 'locked install' {
        & $VenvPython -m pip install --disable-pip-version-check --require-hashes --no-deps `
            -r $ReleaseLock -r $BuildLock
    }
    Invoke-Checked 'dependency check' { & $VenvPython -m pip check --disable-pip-version-check }
    [IO.File]::WriteAllText($StampPath, $stampText)
} else {
    Write-BuildStep 'reusing build\release-venv (locks unchanged)'
}

if (-not $SkipPanel) {
    Write-BuildStep 'building the panel bundle'
    Push-Location -LiteralPath $PanelDir
    try {
        Invoke-Checked 'npm ci' { & npm ci --no-audit --no-fund }
        Invoke-Checked 'panel build' { & npm run build }
        Invoke-Checked 'panel build check' { & node scripts/verify-build.mjs --require-build }
    } finally {
        Pop-Location
    }
}

Write-BuildStep 'freezing the server with PyInstaller'
Push-Location -LiteralPath $RepoRoot
try {
    Invoke-Checked 'PyInstaller' { & $VenvPython -m PyInstaller --noconfirm --clean opencut_server.spec }
} finally {
    Pop-Location
}

Write-BuildStep 'building the WPF installer'
$global:LASTEXITCODE = 0
Invoke-Checked 'InstallerBuilder.ps1' { & $Builder -PythonExe $VenvPython }

# InstallerBuilder.ps1 can finish without the self-extracting exe (it falls
# back to an adjacent payload.zip), so check for the files a release ships.
$initText = [IO.File]::ReadAllText((Join-Path $RepoRoot 'opencut\__init__.py'))
$versionMatch = [regex]::Match($initText, '__version__\s*=\s*"([^"]+)"')
if (-not $versionMatch.Success) {
    throw 'Could not read __version__ from opencut\__init__.py'
}
$distDir = Join-Path $RepoRoot 'installer\dist'
foreach ($name in @("OpenCut-Setup-$($versionMatch.Groups[1].Value).exe", 'payload.zip')) {
    $output = Join-Path $distDir $name
    if (-not (Test-Path -LiteralPath $output -PathType Leaf)) {
        throw "Expected release output is missing: $output"
    }
    Write-BuildStep "output: $output ($([math]::Round((Get-Item -LiteralPath $output).Length / 1MB, 1)) MB)"
}
