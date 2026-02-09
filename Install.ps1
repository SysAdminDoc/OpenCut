<#
.SYNOPSIS
    OpenCut Installer - One-click setup for the OpenCut video editing automation tool.

.DESCRIPTION
    Installs all prerequisites, configures the Premiere Pro CEP extension,
    and creates launcher scripts. Run this once and everything is ready.

    What this installer does:
    1. Cleans up old OpenCut installations (kills processes, removes packages)
    2. Checks/installs FFmpeg (via winget)
    3. Checks Python 3.9+ is available
    4. Installs Python dependencies (click, rich, flask, flask-cors)
    5. Optionally installs Whisper for caption generation
    6. Copies the CEP extension to Adobe's extensions folder
    7. Sets the PlayerDebugMode registry key for unsigned extensions
    8. Creates Start-OpenCut.bat launcher for the backend server
    9. Validates the installation

.NOTES
    Run as Administrator for best results (required for registry + program files access).
    If not admin, will attempt user-level installation.
#>

param(
    [switch]$SkipFFmpeg,
    [switch]$SkipWhisper,
    [switch]$SkipExtension,
    [switch]$SkipCleanup,
    [switch]$Uninstall,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$script:InstallDir = $PSScriptRoot
$script:ExitCode = 0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "  ========================================" -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "  ========================================" -ForegroundColor DarkCyan
}

function Write-Step {
    param([string]$Text)
    Write-Host ""
    Write-Host "  [*] $Text" -ForegroundColor White
}

function Write-Ok {
    param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Text)
    Write-Host "      $Text" -ForegroundColor Red
}

function Write-Info {
    param([string]$Text)
    Write-Host "      $Text" -ForegroundColor DarkGray
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-CommandExists {
    param([string]$Command)
    return [bool](Get-Command $Command -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if ($Uninstall) {
    Write-Header "OpenCut Uninstaller"

    # Kill running processes
    Write-Step "Stopping OpenCut processes..."
    Get-Process -Name "python*" -ErrorAction SilentlyContinue | Where-Object {
        try { $_.CommandLine -like "*opencut*" } catch { $false }
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    # Kill by port
    $portPids = netstat -ano 2>$null | Select-String ":5679 " | ForEach-Object {
        ($_ -split '\s+')[-1]
    } | Where-Object { $_ -match '^\d+$' } | Sort-Object -Unique
    foreach ($pid in $portPids) {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
    Write-Ok "Processes stopped"

    # Remove CEP extension
    $extPaths = @(
        "$env:APPDATA\Adobe\CEP\extensions\com.opencut.panel",
        "${env:ProgramFiles(x86)}\Common Files\Adobe\CEP\extensions\com.opencut.panel"
    )
    foreach ($p in $extPaths) {
        if (Test-Path $p) {
            Remove-Item $p -Recurse -Force
            Write-Ok "Removed CEP extension: $p"
        }
    }

    # Remove pip packages
    Write-Step "Removing Python packages..."
    try {
        pip uninstall opencut -y 2>$null
        Write-Ok "OpenCut package removed"
    } catch {
        Write-Warn "Could not remove OpenCut package (may not be installed)"
    }

    # Clean up model cache
    $modelPaths = @(
        "$env:LOCALAPPDATA\OpenCut",
        "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper*",
        "$env:USERPROFILE\.cache\whisper"
    )
    foreach ($mp in $modelPaths) {
        if (Test-Path $mp) {
            Remove-Item $mp -Recurse -Force -ErrorAction SilentlyContinue
            Write-Ok "Removed: $mp"
        }
    }

    Write-Ok "Uninstall complete."
    Write-Host ""
    exit 0
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "   ___                    ____      _   " -ForegroundColor Cyan
Write-Host "  / _ \ _ __   ___ _ __  / ___|   _| |_ " -ForegroundColor Cyan
Write-Host " | | | | '_ \ / _ \ '_ \| |  | | | | __|" -ForegroundColor Cyan
Write-Host " | |_| | |_) |  __/ | | | |__| |_| | |_ " -ForegroundColor Cyan
Write-Host "  \___/| .__/ \___|_| |_|\____\__,_|\__|" -ForegroundColor Cyan
Write-Host "       |_|                               " -ForegroundColor Cyan
Write-Host ""
Write-Host "  Open Source Video Editing Automation" -ForegroundColor DarkGray
Write-Host "  Installer v0.5.0" -ForegroundColor DarkGray

$isAdmin = Test-IsAdmin
if ($isAdmin) {
    Write-Info "Running as Administrator"
} else {
    Write-Warn "Not running as Administrator - some features may require elevation"
}

# ---------------------------------------------------------------------------
# Step 1: Cleanup Old Installation
# ---------------------------------------------------------------------------
if (-not $SkipCleanup) {
    Write-Header "Step 1/7: Cleanup Old Installation"

    # Kill any running OpenCut server processes
    Write-Step "Stopping any running OpenCut processes..."
    
    # Method 1: Kill python processes on port 5679-5689
    $killed = 0
    for ($port = 5679; $port -le 5689; $port++) {
        $portPids = netstat -ano 2>$null | Select-String ":$port " | ForEach-Object {
            ($_ -split '\s+')[-1]
        } | Where-Object { $_ -match '^\d+$' -and $_ -ne '0' } | Sort-Object -Unique
        
        foreach ($pid in $portPids) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                $killed++
            } catch {}
        }
    }
    
    # Method 2: Kill by PID file
    $pidFile = "$env:USERPROFILE\.opencut\server.pid"
    if (Test-Path $pidFile) {
        $oldPid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($oldPid -match '^\d+$') {
            try {
                Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue
                $killed++
            } catch {}
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    
    if ($killed -gt 0) {
        Write-Ok "Stopped $killed process(es)"
    } else {
        Write-Ok "No running processes found"
    }
    
    # Uninstall old Python package
    Write-Step "Removing old OpenCut Python package..."
    try {
        $pipList = pip list 2>$null | Select-String "^opencut\s"
        if ($pipList) {
            pip uninstall opencut -y 2>$null | Out-Null
            Write-Ok "Removed old opencut package"
        } else {
            Write-Ok "No existing opencut package found"
        }
    } catch {
        Write-Info "Could not check for old package"
    }
    
    # Clear potentially corrupted model cache
    Write-Step "Clearing model cache (prevents CUDA errors)..."
    $cachePaths = @(
        "$env:LOCALAPPDATA\OpenCut\models",
        "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-tiny*",
        "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-base*",
        "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper-small*"
    )
    $cleared = 0
    foreach ($cachePath in $cachePaths) {
        $items = Get-Item $cachePath -ErrorAction SilentlyContinue
        foreach ($item in $items) {
            try {
                Remove-Item $item.FullName -Recurse -Force -ErrorAction SilentlyContinue
                $cleared++
            } catch {}
        }
    }
    if ($cleared -gt 0) {
        Write-Ok "Cleared $cleared cached model folder(s)"
        Write-Info "Models will re-download on first use (ensures clean state)"
    } else {
        Write-Ok "No cached models to clear"
    }
    
    # Remove old CEP extension
    Write-Step "Removing old CEP extension..."
    $extPaths = @(
        "$env:APPDATA\Adobe\CEP\extensions\com.opencut.panel",
        "${env:ProgramFiles(x86)}\Common Files\Adobe\CEP\extensions\com.opencut.panel"
    )
    $removed = 0
    foreach ($p in $extPaths) {
        if (Test-Path $p) {
            Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
            $removed++
        }
    }
    if ($removed -gt 0) {
        Write-Ok "Removed $removed old extension folder(s)"
    } else {
        Write-Ok "No old extensions found"
    }
    
    # Brief pause to let file handles release
    Start-Sleep -Milliseconds 500
    
} else {
    Write-Info "Skipping cleanup (--SkipCleanup)"
}

# ---------------------------------------------------------------------------
# Step 2: FFmpeg
# ---------------------------------------------------------------------------
if (-not $SkipFFmpeg) {
    Write-Header "Step 2/7: FFmpeg"

    if (Test-CommandExists "ffmpeg") {
        $ffVer = (ffmpeg -version 2>&1 | Select-Object -First 1) -replace "ffmpeg version\s+", "" -replace "\s.*", ""
        Write-Ok "FFmpeg found: $ffVer"
    } else {
        Write-Step "FFmpeg not found. Installing..."

        $installed = $false

        # Try winget first
        if (Test-CommandExists "winget") {
            Write-Info "Using winget to install FFmpeg..."
            try {
                winget install Gyan.FFmpeg --accept-package-agreements --accept-source-agreements --silent 2>$null
                # Refresh PATH
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                if (Test-CommandExists "ffmpeg") {
                    Write-Ok "FFmpeg installed via winget"
                    $installed = $true
                }
            } catch {
                Write-Warn "winget install failed, trying alternate method..."
            }
        }

        # Try chocolatey
        if (-not $installed -and (Test-CommandExists "choco")) {
            Write-Info "Using Chocolatey to install FFmpeg..."
            try {
                choco install ffmpeg -y 2>$null
                $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                if (Test-CommandExists "ffmpeg") {
                    Write-Ok "FFmpeg installed via Chocolatey"
                    $installed = $true
                }
            } catch {
                Write-Warn "Chocolatey install failed"
            }
        }

        if (-not $installed) {
            Write-Err "Could not auto-install FFmpeg."
            Write-Err "Please install manually: https://ffmpeg.org/download.html"
            Write-Err "Or run: winget install Gyan.FFmpeg"
            $script:ExitCode = 1
        }
    }
} else {
    Write-Info "Skipping FFmpeg check (--SkipFFmpeg)"
}

# ---------------------------------------------------------------------------
# Step 3: Python
# ---------------------------------------------------------------------------
Write-Header "Step 3/7: Python"

$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Test-CommandExists $cmd) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 9) {
                    $pythonCmd = $cmd
                    Write-Ok "Python found: $ver (using '$cmd')"
                    break
                } else {
                    Write-Warn "$cmd is version $ver (need 3.9+)"
                }
            }
        } catch {
            continue
        }
    }
}

if (-not $pythonCmd) {
    Write-Err "Python 3.9+ not found."
    Write-Err "Install from: https://www.python.org/downloads/"
    Write-Err "Or run: winget install Python.Python.3.12"
    Write-Err ""
    Write-Err "IMPORTANT: Check 'Add Python to PATH' during installation."
    $script:ExitCode = 1
    Write-Host ""
    exit $script:ExitCode
}

# ---------------------------------------------------------------------------
# Step 4: Python Dependencies
# ---------------------------------------------------------------------------
Write-Header "Step 4/7: Python Dependencies"

Write-Step "Installing core dependencies..."
& $pythonCmd -m pip install --upgrade pip --quiet 2>$null
& $pythonCmd -m pip install click rich flask flask-cors --quiet 2>&1 | Out-Null
Write-Ok "Core packages installed (click, rich, flask, flask-cors)"

# Install OpenCut package
Write-Step "Installing OpenCut package..."
$opencutDir = Join-Path $script:InstallDir "."
if (Test-Path (Join-Path $opencutDir "pyproject.toml")) {
    & $pythonCmd -m pip install $opencutDir --quiet 2>&1 | Out-Null
    Write-Ok "OpenCut package installed"
} else {
    Write-Err "pyproject.toml not found in $opencutDir"
    Write-Err "Make sure you're running this from the OpenCut project folder."
    $script:ExitCode = 1
}

# Optional: Whisper
if (-not $SkipWhisper) {
    Write-Step "Installing Whisper for caption generation..."
    Write-Info "This may take several minutes (downloads ~150MB+ of ML libraries)..."
    Write-Info ""

    $whisperInstalled = $false

    # Strategy 1: Pre-built wheels only (fastest, avoids Rust requirement)
    Write-Info "  Trying pre-built wheels..."
    try {
        $pipOutput = & $pythonCmd -m pip install faster-whisper --only-binary :all: --progress-bar on 2>&1
        $pipExit = $LASTEXITCODE
        foreach ($line in ($pipOutput -split "`n")) {
            $trimmed = $line.Trim()
            if ($trimmed -match "^(Downloading|Installing|Successfully|ERROR|error|failed)") {
                Write-Info "  $trimmed"
            }
        }
        if ($pipExit -eq 0) {
            $verifyResult = & $pythonCmd -c "from faster_whisper import WhisperModel; print('ok')" 2>&1
            if ($verifyResult -match "ok") {
                Write-Ok "faster-whisper installed (pre-built wheels)"
                $whisperInstalled = $true
            }
        }
    } catch {}

    # Strategy 2: Prefer binary but allow source build
    if (-not $whisperInstalled) {
        Write-Info "  Trying with prefer-binary flag..."
        try {
            & $pythonCmd -m pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
            $pipOutput = & $pythonCmd -m pip install faster-whisper --prefer-binary --progress-bar on 2>&1
            $pipExit = $LASTEXITCODE
            foreach ($line in ($pipOutput -split "`n")) {
                $trimmed = $line.Trim()
                if ($trimmed -match "^(Downloading|Installing|Successfully|Building|ERROR|error|failed|metadata)") {
                    Write-Info "  $trimmed"
                }
            }
            if ($pipExit -eq 0) {
                $verifyResult = & $pythonCmd -c "from faster_whisper import WhisperModel; print('ok')" 2>&1
                if ($verifyResult -match "ok") {
                    Write-Ok "faster-whisper installed and verified"
                    $whisperInstalled = $true
                }
            }
        } catch {}
    }

    # Strategy 3: Fallback to openai-whisper (no tokenizers dependency)
    if (-not $whisperInstalled) {
        Write-Warn "faster-whisper failed (likely needs Rust for tokenizers)."
        Write-Info "  Trying openai-whisper as fallback..."
        try {
            $pipOutput = & $pythonCmd -m pip install openai-whisper --progress-bar on 2>&1
            $pipExit = $LASTEXITCODE
            foreach ($line in ($pipOutput -split "`n")) {
                $trimmed = $line.Trim()
                if ($trimmed -match "^(Downloading|Installing|Successfully|ERROR|error|failed)") {
                    Write-Info "  $trimmed"
                }
            }
            if ($pipExit -eq 0) {
                $verifyResult = & $pythonCmd -c "import whisper; print('ok')" 2>&1
                if ($verifyResult -match "ok") {
                    Write-Ok "openai-whisper installed as fallback (caption support enabled)"
                    $whisperInstalled = $true
                }
            }
        } catch {}
    }

    if (-not $whisperInstalled) {
        Write-Warn "Could not install Whisper automatically."
        Write-Warn "Common fix: Update Python to 3.10-3.12, or install Rust from https://rustup.rs/"
        Write-Warn "You can retry from the OpenCut panel in Premiere Pro."
    }
} else {
    Write-Info "Skipping Whisper installation (--SkipWhisper)"
}

# ---------------------------------------------------------------------------
# Step 5: CEP Extension
# ---------------------------------------------------------------------------
if (-not $SkipExtension) {
    Write-Header "Step 5/7: Premiere Pro Extension"

    $extensionSrc = Join-Path $script:InstallDir "extension\com.opencut.panel"

    if (-not (Test-Path $extensionSrc)) {
        Write-Err "Extension source not found: $extensionSrc"
        $script:ExitCode = 1
    } else {
        # Determine target directory
        # Install to user-level (always works, no admin needed)
        $userExtDir = "$env:APPDATA\Adobe\CEP\extensions"
        $systemExtDir = "${env:ProgramFiles(x86)}\Common Files\Adobe\CEP\extensions"

        # Always install to user dir
        $userTarget = Join-Path $userExtDir "com.opencut.panel"
        if (-not (Test-Path $userExtDir)) {
            New-Item -Path $userExtDir -ItemType Directory -Force | Out-Null
        }
        if (Test-Path $userTarget) { Remove-Item $userTarget -Recurse -Force }
        Copy-Item $extensionSrc $userTarget -Recurse -Force
        Write-Ok "Extension installed to: $userTarget"

        # Also install to system dir if admin
        if ($isAdmin) {
            $sysTarget = Join-Path $systemExtDir "com.opencut.panel"
            if (-not (Test-Path $systemExtDir)) {
                New-Item -Path $systemExtDir -ItemType Directory -Force | Out-Null
            }
            if (Test-Path $sysTarget) { Remove-Item $sysTarget -Recurse -Force }
            Copy-Item $extensionSrc $sysTarget -Recurse -Force
            Write-Ok "Extension also installed to: $sysTarget"
        }
    }

    # Set PlayerDebugMode registry key (allows unsigned extensions)
    Write-Step "Configuring Adobe CEP debug mode..."

    $regPaths = @(
        "HKCU:\Software\Adobe\CSXS.12",
        "HKCU:\Software\Adobe\CSXS.11",
        "HKCU:\Software\Adobe\CSXS.10",
        "HKCU:\Software\Adobe\CSXS.9",
        "HKCU:\Software\Adobe\CSXS.8",
        "HKCU:\Software\Adobe\CSXS.7"
    )

    foreach ($regPath in $regPaths) {
        try {
            if (-not (Test-Path $regPath)) {
                New-Item -Path $regPath -Force | Out-Null
            }
            Set-ItemProperty -Path $regPath -Name "PlayerDebugMode" -Value "1" -Type String -Force
        } catch {
            # Silently continue - some may not exist
        }
    }
    Write-Ok "PlayerDebugMode enabled for CSXS 7-12"
    Write-Info "This allows unsigned extensions to load in Premiere Pro."

} else {
    Write-Info "Skipping extension installation (--SkipExtension)"
}

# ---------------------------------------------------------------------------
# Step 6: Create Launcher Scripts
# ---------------------------------------------------------------------------
Write-Header "Step 6/7: Launcher Scripts"

# Start-OpenCut.bat
$launcherPath = Join-Path $script:InstallDir "Start-OpenCut.bat"
$launcherContent = @"
@echo off
title OpenCut Backend Server
echo.
echo   OpenCut Backend Server
echo   =====================
echo.
echo   Starting on http://127.0.0.1:5679
echo   Keep this window open while using the Premiere Pro panel.
echo   Press Ctrl+C to stop.
echo.
$pythonCmd -m opencut.server
pause
"@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII
Write-Ok "Created: Start-OpenCut.bat"

# Start-OpenCut-Hidden.vbs (runs backend without visible window)
$vbsPath = Join-Path $script:InstallDir "Start-OpenCut-Hidden.vbs"
$vbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "$pythonCmd -m opencut.server", 0, False
"@
Set-Content -Path $vbsPath -Value $vbsContent -Encoding ASCII
Write-Ok "Created: Start-OpenCut-Hidden.vbs (runs backend silently)"

# Desktop shortcut
try {
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = Join-Path $desktopPath "Start OpenCut.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $launcherPath
    $shortcut.WorkingDirectory = $script:InstallDir
    $shortcut.Description = "Start OpenCut Backend Server"
    $shortcut.Save()
    Write-Ok "Desktop shortcut created: Start OpenCut"
} catch {
    Write-Warn "Could not create desktop shortcut"
}

# ---------------------------------------------------------------------------
# Step 7: Validation
# ---------------------------------------------------------------------------
Write-Header "Step 7/7: Validation"

$allGood = $true

# Check FFmpeg
if (Test-CommandExists "ffmpeg") {
    Write-Ok "FFmpeg .............. OK"
} else {
    Write-Err "FFmpeg .............. MISSING"
    $allGood = $false
}

# Check Python
if ($pythonCmd) {
    Write-Ok "Python .............. OK ($pythonCmd)"
} else {
    Write-Err "Python .............. MISSING"
    $allGood = $false
}

# Check OpenCut
try {
    $ocVer = & $pythonCmd -c "import opencut; print(opencut.__version__)" 2>&1
    Write-Ok "OpenCut package ..... OK (v$ocVer)"
} catch {
    Write-Err "OpenCut package ..... MISSING"
    $allGood = $false
}

# Check Flask
try {
    & $pythonCmd -c "import flask" 2>&1 | Out-Null
    Write-Ok "Flask server ........ OK"
} catch {
    Write-Err "Flask server ........ MISSING"
    $allGood = $false
}

# Check Whisper
try {
    $whisperCheck = & $pythonCmd -c "from faster_whisper import WhisperModel; print('ok')" 2>&1
    if ($whisperCheck -eq "ok") {
        Write-Ok "Whisper (captions) .. OK (faster-whisper)"
    } else {
        Write-Warn "Whisper (captions) .. NOT INSTALLED (optional)"
    }
} catch {
    Write-Warn "Whisper (captions) .. NOT INSTALLED (optional)"
}

# Check CEP extension
$extCheck = $false
foreach ($dir in @("$env:APPDATA\Adobe\CEP\extensions\com.opencut.panel", "${env:ProgramFiles(x86)}\Common Files\Adobe\CEP\extensions\com.opencut.panel")) {
    if (Test-Path (Join-Path $dir "CSXS\manifest.xml")) {
        Write-Ok "CEP extension ....... OK ($dir)"
        $extCheck = $true
        break
    }
}
if (-not $extCheck) {
    Write-Err "CEP extension ....... NOT FOUND"
    $allGood = $false
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
if ($allGood -or $script:ExitCode -eq 0) {
    Write-Host "  ========================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "  ========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  HOW TO USE:" -ForegroundColor White
    Write-Host "  1. Double-click 'Start-OpenCut.bat' (or the desktop shortcut)" -ForegroundColor Gray
    Write-Host "  2. Open Premiere Pro" -ForegroundColor Gray
    Write-Host "  3. Go to Window > Extensions > OpenCut" -ForegroundColor Gray
    Write-Host "  4. Select a clip and click 'Remove Silences'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  The generated XML imports directly into your project." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  CLI is also available: opencut silence video.mp4" -ForegroundColor DarkGray
} else {
    Write-Host "  ========================================" -ForegroundColor Yellow
    Write-Host "  Installation completed with warnings." -ForegroundColor Yellow
    Write-Host "  ========================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Some components may need manual installation." -ForegroundColor Gray
    Write-Host "  See messages above for details." -ForegroundColor Gray
}

Write-Host ""
exit $script:ExitCode
