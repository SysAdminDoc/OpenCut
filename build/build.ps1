<#
.SYNOPSIS
    OpenCut Build Script - Creates a distributable installer from source.

.DESCRIPTION
    Automates the entire build pipeline:
    1. Installs build dependencies (PyInstaller)
    2. Bundles the Python backend into a standalone exe
    3. Generates an application icon
    4. Creates a Windows installer via Inno Setup

    Prerequisites:
    - Python 3.9+ with pip
    - Inno Setup 6+ (optional, for installer creation)

.EXAMPLE
    .\build\build.ps1              # Full build
    .\build\build.ps1 -SkipInstaller   # Just build the exe
    .\build\build.ps1 -Clean           # Clean build artifacts first
#>

param(
    [switch]$SkipInstaller,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = $PSScriptRoot
$DistDir = Join-Path $ProjectRoot "dist"
$AppVersion = "1.0.0"

function Write-Step { param([string]$Text)
    Write-Host ""
    Write-Host "  [*] $Text" -ForegroundColor Cyan
}
function Write-OK { param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Green
}
function Write-Warn { param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Yellow
}
function Write-Fail { param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Red
}

Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host "  OpenCut Build System v0.7.0" -ForegroundColor Cyan
Write-Host "  Premium Edition" -ForegroundColor DarkGray
Write-Host "  ============================================" -ForegroundColor DarkCyan

# ---- Clean ----
if ($Clean) {
    Write-Step "Cleaning previous build..."
    if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
    $specBuild = Join-Path $ProjectRoot "build" "opencut-server"
    if (Test-Path $specBuild) { Remove-Item $specBuild -Recurse -Force }
    Write-OK "Clean complete"
}

# ---- Check Python ----
Write-Step "Checking Python..."
$py = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 9) {
                $py = $cmd
                Write-OK "Found: $ver ($cmd)"
                break
            }
        }
    } catch {}
}
if (-not $py) {
    Write-Fail "Python 3.9+ is required. Install from python.org"
    exit 1
}

# ---- Install build dependencies ----
Write-Step "Installing build dependencies..."
& $py -m pip install --upgrade pyinstaller --break-system-packages -q 2>$null
if (-not $?) { & $py -m pip install --upgrade pyinstaller -q }
Write-OK "PyInstaller ready"

# ---- Install project dependencies ----
Write-Step "Installing project dependencies..."
Push-Location $ProjectRoot
& $py -m pip install -e ".[captions]" --break-system-packages -q 2>$null
if (-not $?) { & $py -m pip install -e ".[captions]" -q 2>$null }
if (-not $?) { & $py -m pip install -e . -q }
Pop-Location
Write-OK "Dependencies installed"

# ---- Generate icon if missing ----
$IconPath = Join-Path $BuildDir "icon.ico"
if (-not (Test-Path $IconPath)) {
    Write-Step "Generating application icon..."
    $iconScript = @"
import sys
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit(1)

sizes = [256, 128, 64, 48, 32, 16]
images = []
for sz in sizes:
    img = Image.new('RGBA', (sz, sz), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Dark circle background
    pad = max(1, sz // 16)
    draw.ellipse([pad, pad, sz - pad - 1, sz - pad - 1], fill=(15, 15, 30, 255))
    draw.ellipse([pad, pad, sz - pad - 1, sz - pad - 1], outline=(79, 195, 247, 200), width=max(1, sz // 32))
    # Play triangle + cut line
    cx, cy = sz // 2, sz // 2
    ts = sz // 4  # triangle size
    # Play triangle (left side)
    tri = [(cx - ts + ts//4, cy - ts), (cx + ts//2, cy), (cx - ts + ts//4, cy + ts)]
    draw.polygon(tri, fill=(79, 195, 247, 220))
    # Cut line (vertical, right of center)
    lx = cx + ts // 3
    lw = max(1, sz // 32)
    draw.line([(lx, cy - ts), (lx, cy + ts)], fill=(255, 255, 255, 180), width=lw)
    images.append(img)

images[0].save(sys.argv[1], format='ICO', sizes=[(s, s) for s in sizes], append_images=images[1:])
print(f'Icon saved: {sys.argv[1]}')
"@
    $iconScript | & $py - $IconPath
    if (Test-Path $IconPath) { Write-OK "Icon generated" }
    else { Write-Warn "Icon generation failed (Pillow may not be installed). Continuing without icon." }
}

# ---- Build with PyInstaller ----
Write-Step "Building standalone executable with PyInstaller..."
Write-Host "      This may take 1-3 minutes..." -ForegroundColor DarkGray
Push-Location $ProjectRoot
$specFile = Join-Path $BuildDir "opencut.spec"
& $py -m PyInstaller --noconfirm --clean $specFile
$buildSuccess = $?
Pop-Location

if (-not $buildSuccess) {
    Write-Fail "PyInstaller build failed!"
    Write-Host ""
    Write-Host "  Try running manually:" -ForegroundColor Yellow
    Write-Host "    cd $ProjectRoot" -ForegroundColor DarkGray
    Write-Host "    pyinstaller build/opencut.spec" -ForegroundColor DarkGray
    exit 1
}

$exePath = Join-Path $DistDir "opencut-server" "opencut-server.exe"
if (Test-Path $exePath) {
    $exeSize = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
    Write-OK "Built: opencut-server.exe ($exeSize MB)"
} else {
    Write-Fail "Expected exe not found at: $exePath"
    exit 1
}

# ---- Create Installer (Inno Setup) ----
if (-not $SkipInstaller) {
    Write-Step "Creating installer..."

    # Find Inno Setup
    $iscc = $null
    $issLocations = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        (Get-Command "iscc.exe" -ErrorAction SilentlyContinue).Source
    )
    foreach ($loc in $issLocations) {
        if ($loc -and (Test-Path $loc)) { $iscc = $loc; break }
    }

    if ($iscc) {
        $issFile = Join-Path $BuildDir "installer.iss"
        & $iscc $issFile
        if ($?) {
            $installer = Get-ChildItem (Join-Path $DistDir "OpenCut-Setup-*.exe") -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($installer) {
                $instSize = [math]::Round($installer.Length / 1MB, 1)
                Write-OK "Installer created: $($installer.Name) ($instSize MB)"
            }
        } else {
            Write-Warn "Inno Setup build failed. Exe is still available in dist/"
        }
    } else {
        Write-Warn "Inno Setup not found. Install from: https://jrsoftware.org/isinfo.php"
        Write-Warn "Then run: iscc build\installer.iss"
        Write-Host ""
        Write-Host "  The standalone exe is ready in:" -ForegroundColor White
        Write-Host "    $DistDir\opencut-server\" -ForegroundColor DarkGray
    }
}

# ---- Summary ----
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "  ============================================" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Standalone exe: dist\opencut-server\opencut-server.exe" -ForegroundColor White
if (Test-Path (Join-Path $DistDir "OpenCut-Setup-*.exe")) {
    Write-Host "  Installer:      dist\OpenCut-Setup-$($AppVersion).exe" -ForegroundColor White
}
Write-Host ""
Write-Host "  To test the exe:" -ForegroundColor DarkGray
Write-Host "    .\dist\opencut-server\opencut-server.exe" -ForegroundColor DarkGray
Write-Host ""
