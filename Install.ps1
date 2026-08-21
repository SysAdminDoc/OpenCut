<#
.SYNOPSIS
    OpenCut Installer - One-click setup for the OpenCut video editing automation tool.

.DESCRIPTION
    Installs all prerequisites, configures the Premiere Pro CEP extension,
    and creates launcher scripts. Run this once and everything is ready.

    What this installer does:
    1. Cleans up old OpenCut installations (kills processes, removes packages)
    2. Checks/installs FFmpeg (via winget)
    3. Checks Python 3.11-3.14 is available
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

function Resolve-OpenCutPython {
    <#
    .SYNOPSIS
        Resolve a supported Python executable before any pip operation.

    .DESCRIPTION
        Returning the executable path (rather than relying on a separate
        ``pip`` command) keeps cleanup and installation on the same Python
        environment on machines with multiple Python installations.
    #>
    foreach ($candidateName in @("python", "python3", "py")) {
        $command = Get-Command $candidateName -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $command) { continue }
        if ($command.CommandType -notin @("Application", "ExternalScript")) {
            continue
        }

        $candidatePath = if ($command.Source) { $command.Source } else { $command.Definition }
        if (-not $candidatePath -or -not [IO.Path]::IsPathRooted($candidatePath)) {
            continue
        }
        try {
            $versionOutput = & $candidatePath --version 2>&1
            if ($versionOutput -match "(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 11 -and $minor -le 14) {
                    return $candidatePath
                }
            }
        } catch {
            continue
        }
    }
    return $null
}

function Remove-OpenCutArtifact {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) { return $true }
    try {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        Write-Ok "Removed ${Label}: $Path"
        return $true
    } catch {
        $script:ExitCode = 1
        Write-Warn "Could not remove ${Label}: $Path"
        Write-Warn "Close Premiere Pro and any OpenCut process, then retry the uninstall."
        return $false
    }
}

function Get-FFmpegSecurityStatus {
    $required = "release>=8.1.3 OR git-master>=2026-07-06"
    $command = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if (-not $command) {
        return [pscustomobject]@{ Safe = $false; Version = "missing"; Reason = "FFmpeg was not found" }
    }

    try {
        $banner = (& $command.Source -version 2>&1 | Select-Object -First 1) -as [string]
    } catch {
        return [pscustomobject]@{ Safe = $false; Version = "unknown"; Reason = "FFmpeg could not be executed: $_" }
    }

    if ($banner -notmatch "\bversion\s+([^\s]+)") {
        return [pscustomobject]@{ Safe = $false; Version = "unknown"; Reason = "Version banner could not be parsed" }
    }

    $token = $Matches[1]
    if ($token -match "^(\d{4}-\d{2}-\d{2})-git-[0-9a-f]{7,40}") {
        $safe = $Matches[1] -ge "2026-07-06"
        $reason = if ($safe) { "Post-fix snapshot lane (CVE-2026-64832/64833/64835/66041)" } else { "Snapshot predates 2026-07-06 (CVE-2026-64832/64833/64835/66041)" }
        return [pscustomobject]@{ Safe = $safe; Version = $token; Reason = $reason }
    }

    if ($token -match "^n?(\d+)\.(\d+)(?:\.(\d+))?") {
        $patch = if ($Matches[3]) { [int]$Matches[3] } else { 0 }
        $version = [version]::new([int]$Matches[1], [int]$Matches[2], $patch)
        $safe = $version -ge [version]::new(8, 1, 3)
        $reason = if ($safe) { "Release lane clears $required" } else { "Release predates 8.1.3 (CVE-2026-64832/64833/64835/66041)" }
        return [pscustomobject]@{ Safe = $safe; Version = $token; Reason = $reason }
    }

    return [pscustomobject]@{ Safe = $false; Version = $token; Reason = "Unrecognized FFmpeg build token" }
}

function Get-OpenCutListenerPids {
    <#
    .SYNOPSIS
        PIDs of the processes LISTENING on a TCP port - never the clients.

    .DESCRIPTION
        `netstat -ano | Select-String ":5679 "` also matches ESTABLISHED rows,
        and the last column of an ESTABLISHED row is the PID of whichever end
        of the connection the row describes. When the CEP panel is connected,
        one of those rows belongs to Premiere Pro - so the old parse killed
        Premiere and lost the user's unsaved project.

        Prefer `Get-NetTCPConnection`, which reports state as a typed value and
        does not depend on the console locale. Fall back to parsing `netstat`
        column-wise: a killable row is TCP, in state LISTENING, and its *local*
        address (field 2, not the foreign address) ends with the port.

    .PARAMETER NetstatOutput
        Recorded `netstat -ano` lines. For tests; production callers omit it.
    #>
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [string[]]$NetstatOutput
    )

    if (-not $NetstatOutput) {
        try {
            $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
            return @($listeners |
                ForEach-Object { [string]$_.OwningProcess } |
                Where-Object { $_ -match '^\d+$' -and $_ -ne '0' } |
                Sort-Object -Unique)
        } catch {
            # Get-NetTCPConnection is unavailable or returned nothing usable;
            # fall through to the netstat parse below.
        }
        $NetstatOutput = @(netstat -ano 2>$null)
    }

    $found = @()
    foreach ($line in $NetstatOutput) {
        if (-not $line) { continue }
        $fields = ([string]$line).Trim() -split '\s+'
        # TCP  <local address>  <foreign address>  LISTENING  <pid>
        if ($fields.Count -lt 5) { continue }
        if ($fields[0] -ne 'TCP') { continue }
        if ($fields[3] -ne 'LISTENING') { continue }
        if ($fields[1] -notmatch ":$Port$") { continue }
        $procId = $fields[4]
        if ($procId -match '^\d+$' -and $procId -ne '0') { $found += $procId }
    }
    return @($found | Sort-Object -Unique)
}

# Resolve Python before cleanup/uninstall so every pip operation uses the
# interpreter that will also install and validate OpenCut.
$pythonCmd = Resolve-OpenCutPython

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
    
    # Kill by port - listeners only, never connected clients (Premiere holds
    # the client end of :5679 while the panel is open). NB: ``$pid`` is a
    # read-only automatic PowerShell variable (the current process PID).
    # Assigning to it in ``foreach`` raises a ``Cannot overwrite variable PID``
    # runtime error and, with ``$ErrorActionPreference = "Stop"`` set above,
    # halts the entire uninstall. Use ``$procId`` instead.
    $portPids = Get-OpenCutListenerPids -Port 5679
    foreach ($procId in $portPids) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Write-Ok "Processes stopped"

    # Remove CEP extension
    $extPaths = @(
        "$env:APPDATA\Adobe\CEP\extensions\com.opencut.panel",
        "${env:ProgramFiles(x86)}\Common Files\Adobe\CEP\extensions\com.opencut.panel"
    )
    foreach ($p in $extPaths) {
        Remove-OpenCutArtifact -Path $p -Label "CEP extension"
    }

    # Remove pip packages
    Write-Step "Removing Python packages..."
    if ($pythonCmd) {
        try {
            & $pythonCmd -m pip uninstall opencut -y 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "OpenCut package removed (using $pythonCmd)"
            } else {
                $script:ExitCode = 1
                Write-Warn "Could not remove OpenCut package using $pythonCmd"
            }
        } catch {
            $script:ExitCode = 1
            Write-Warn "Could not remove OpenCut package using $pythonCmd"
        }
    } else {
        $script:ExitCode = 1
        Write-Warn "No supported Python 3.11-3.14 interpreter was found; OpenCut package was not removed."
    }

    # Clean up model cache
    $modelPaths = @(
        "$env:LOCALAPPDATA\OpenCut",
        "$env:USERPROFILE\.cache\huggingface\hub\models--Systran--faster-whisper*",
        "$env:USERPROFILE\.cache\whisper"
    )
    foreach ($modelPattern in $modelPaths) {
        foreach ($modelPath in @(Get-Item -Path $modelPattern -ErrorAction SilentlyContinue)) {
            Remove-OpenCutArtifact -Path $modelPath.FullName -Label "model cache"
        }
    }

    # Remove files and shortcuts generated by the source installer.
    $launcherPaths = @(
        (Join-Path $script:InstallDir "Start-OpenCut.bat"),
        (Join-Path $script:InstallDir "Start-OpenCut-Hidden.vbs"),
        (Join-Path ([Environment]::GetFolderPath("Desktop")) "Start OpenCut.lnk")
    )
    foreach ($launcherArtifact in $launcherPaths) {
        Remove-OpenCutArtifact -Path $launcherArtifact -Label "launcher artifact"
    }

    # Remove only the debug-mode value written by this installer. Leave the
    # Adobe CSXS key itself intact for other CEP extensions.
    $regPaths = @(
        "HKCU:\Software\Adobe\CSXS.18",
        "HKCU:\Software\Adobe\CSXS.17",
        "HKCU:\Software\Adobe\CSXS.16",
        "HKCU:\Software\Adobe\CSXS.15",
        "HKCU:\Software\Adobe\CSXS.14",
        "HKCU:\Software\Adobe\CSXS.13",
        "HKCU:\Software\Adobe\CSXS.12",
        "HKCU:\Software\Adobe\CSXS.11",
        "HKCU:\Software\Adobe\CSXS.10",
        "HKCU:\Software\Adobe\CSXS.9",
        "HKCU:\Software\Adobe\CSXS.8",
        "HKCU:\Software\Adobe\CSXS.7"
    )
    foreach ($regPath in $regPaths) {
        try {
            if (Test-Path -LiteralPath $regPath) {
                $regValue = Get-ItemProperty -LiteralPath $regPath -Name "PlayerDebugMode" -ErrorAction SilentlyContinue
                if ($null -ne $regValue) {
                    Remove-ItemProperty -LiteralPath $regPath -Name "PlayerDebugMode" -Force -ErrorAction Stop
                    Write-Ok "Removed PlayerDebugMode: $regPath"
                }
            }
        } catch {
            $script:ExitCode = 1
            Write-Warn "Could not remove PlayerDebugMode: $regPath"
            Write-Warn "Close Premiere Pro and retry the uninstall if the registry value is in use."
        }
    }

    if ($script:ExitCode -eq 0) {
        Write-Ok "Uninstall complete."
    } else {
        Write-Warn "Uninstall completed with warnings; retry after closing Premiere Pro and OpenCut."
    }
    Write-Host ""
    exit $script:ExitCode
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
Write-Host "  Installer v1.50.0" -ForegroundColor DarkGray

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
    
    # Method 1: Kill whatever is LISTENING on ports 5679-5689. Connected
    # clients (Premiere, a browser tab) must be left alone.
    $killed = 0
    for ($port = 5679; $port -le 5689; $port++) {
        $portPids = Get-OpenCutListenerPids -Port $port

        # ``$pid`` is reserved (current process PID, read-only); use ``$procId``.
        foreach ($procId in $portPids) {
            try {
                Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
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
    if ($pythonCmd) {
        try {
            $pipList = & $pythonCmd -m pip list 2>$null | Select-String "^opencut\s"
            if ($pipList) {
                & $pythonCmd -m pip uninstall opencut -y 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Ok "Removed old opencut package"
                } else {
                    Write-Warn "Could not remove old opencut package using $pythonCmd"
                }
            } else {
                Write-Ok "No existing opencut package found"
            }
        } catch {
            Write-Warn "Could not check for old package using $pythonCmd"
        }
    } else {
        Write-Warn "No supported Python 3.11-3.14 interpreter found; skipped old package cleanup."
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
            if (Remove-OpenCutArtifact -Path $p -Label "old CEP extension") {
                $removed++
            }
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
# Step 2: FFmpeg security floor
# ---------------------------------------------------------------------------
Write-Header "Step 2/7: FFmpeg"
$ffmpegStatus = Get-FFmpegSecurityStatus
if ($ffmpegStatus.Safe) {
    Write-Ok "FFmpeg verified: $($ffmpegStatus.Version) ($($ffmpegStatus.Reason))"
} elseif ($SkipFFmpeg) {
    Write-Err "-SkipFFmpeg skips auto-install only; it cannot bypass the security floor."
    Write-Err "$($ffmpegStatus.Version): $($ffmpegStatus.Reason)"
    exit 1
} else {
    Write-Warn "Existing FFmpeg is unavailable: $($ffmpegStatus.Version) — $($ffmpegStatus.Reason)"
    $installed = $false

    if (Test-CommandExists "winget") {
        Write-Info "Installing/upgrading the verified Gyan.FFmpeg package..."
        try {
            winget upgrade --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements --silent 2>$null
            if ($LASTEXITCODE -ne 0) {
                winget install --id Gyan.FFmpeg --exact --accept-package-agreements --accept-source-agreements --silent 2>$null
            }
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            $installed = (Get-FFmpegSecurityStatus).Safe
        } catch {
            Write-Warn "winget could not install a verified FFmpeg build."
        }
    }

    if (-not $installed -and (Test-CommandExists "choco")) {
        Write-Info "Upgrading FFmpeg with Chocolatey..."
        try {
            choco upgrade ffmpeg -y 2>$null
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            $installed = (Get-FFmpegSecurityStatus).Safe
        } catch {
            Write-Warn "Chocolatey could not install a verified FFmpeg build."
        }
    }

    $ffmpegStatus = Get-FFmpegSecurityStatus
    if (-not $ffmpegStatus.Safe) {
        Write-Err "FFmpeg remains blocked: $($ffmpegStatus.Version) — $($ffmpegStatus.Reason)"
        Write-Err "Install FFmpeg 8.1.3+ or a dated post-fix full snapshot from https://www.gyan.dev/ffmpeg/builds/"
        exit 1
    }
    Write-Ok "FFmpeg verified after install: $($ffmpegStatus.Version)"
}

# ---------------------------------------------------------------------------
# Step 3: Python
# ---------------------------------------------------------------------------
Write-Header "Step 3/7: Python"

$pythonCmd = Resolve-OpenCutPython
if ($pythonCmd) {
    $ver = & $pythonCmd --version 2>&1
    Write-Ok "Detected $ver (using '$pythonCmd'); supported Python 3.11-3.14"
}

if (-not $pythonCmd) {
    Write-Err "No supported interpreter found; OpenCut requires Python 3.11-3.14."
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
$releaseLock = Join-Path $script:InstallDir "requirements-release-lock.txt"
$reqFile = Join-Path $script:InstallDir "requirements.txt"
if (Test-Path $releaseLock) {
    & $pythonCmd -m pip install --require-hashes -r $releaseLock --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Core packages installed from the hash-locked release set"
    } else {
        Write-Err "Hash-locked dependency install failed (exit code $LASTEXITCODE)."
        Write-Err "Check your network connection, then re-run: $pythonCmd -m pip install --require-hashes -r `"$releaseLock`""
        $script:ExitCode = 1
    }
} elseif (Test-Path $reqFile) {
    & $pythonCmd -m pip install -r $reqFile --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Core packages installed from requirements.txt"
    } else {
        Write-Err "pip install -r requirements.txt failed (exit code $LASTEXITCODE)."
        Write-Err "Check your network connection, then re-run: $pythonCmd -m pip install -r `"$reqFile`""
        $script:ExitCode = 1
    }
} else {
    Write-Err "No dependency lock or requirements file was found."
    Write-Err "Re-download the complete OpenCut release and retry."
    $script:ExitCode = 1
}

# Install OpenCut package
Write-Step "Installing OpenCut package..."
$opencutDir = Join-Path $script:InstallDir "."
if (Test-Path (Join-Path $opencutDir "pyproject.toml")) {
    & $pythonCmd -m pip install $opencutDir --quiet 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "OpenCut package installed"
    } else {
        Write-Err "OpenCut package install failed (exit code $LASTEXITCODE)."
        Write-Err "Re-run manually to see the full error: $pythonCmd -m pip install `"$opencutDir`""
        $script:ExitCode = 1
    }
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
    $fasterWhisperRequirement = "faster-whisper>=1.1,<2"

    # Strategy 1: Pre-built wheels only (fastest, avoids Rust requirement)
    Write-Info "  Trying pre-built wheels..."
    try {
        $pipOutput = & $pythonCmd -m pip install $fasterWhisperRequirement --only-binary :all: --progress-bar on 2>&1
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
            $pipOutput = & $pythonCmd -m pip install $fasterWhisperRequirement --prefer-binary --progress-bar on 2>&1
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

    if (-not $whisperInstalled) {
        Write-Warn "Could not install Whisper automatically."
        Write-Warn "Verify Python 3.11-3.14 and retry the supported captions extra."
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
        $userTargetReady = Remove-OpenCutArtifact -Path $userTarget -Label "existing CEP extension"
        if ($userTargetReady) {
            Copy-Item $extensionSrc $userTarget -Recurse -Force
            Write-Ok "Extension installed to: $userTarget"
        } else {
            Write-Err "User CEP extension was not replaced; close Premiere Pro and retry."
        }

        # Also install to system dir if admin
        if ($isAdmin) {
            $sysTarget = Join-Path $systemExtDir "com.opencut.panel"
            if (-not (Test-Path $systemExtDir)) {
                New-Item -Path $systemExtDir -ItemType Directory -Force | Out-Null
            }
            $sysTargetReady = Remove-OpenCutArtifact -Path $sysTarget -Label "existing system CEP extension"
            if ($sysTargetReady) {
                Copy-Item $extensionSrc $sysTarget -Recurse -Force
                Write-Ok "Extension also installed to: $sysTarget"
            } else {
                Write-Err "System CEP extension was not replaced; close Premiere Pro and retry."
            }
        }
    }

    # Set PlayerDebugMode registry key (allows unsigned extensions)
    Write-Step "Configuring Adobe CEP debug mode..."

    # Cover CSXS 7 (CC 2014) through 18 (current PPro 2025+). Old script
    # stopped at 12 which left modern Premiere installs (CC 2023+ on
    # CSXS 13+) without PlayerDebugMode set, so the panel never loaded.
    # Mirrors the install.py + install_cep_extension matrix.
    $regPaths = @(
        "HKCU:\Software\Adobe\CSXS.18",
        "HKCU:\Software\Adobe\CSXS.17",
        "HKCU:\Software\Adobe\CSXS.16",
        "HKCU:\Software\Adobe\CSXS.15",
        "HKCU:\Software\Adobe\CSXS.14",
        "HKCU:\Software\Adobe\CSXS.13",
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
    Write-Ok "PlayerDebugMode enabled for CSXS 7-18"
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
$quotedPythonCmd = '"{0}"' -f $pythonCmd
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
$quotedPythonCmd -m opencut.server
pause
"@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII
Write-Ok "Created: Start-OpenCut.bat"

# Start-OpenCut-Hidden.vbs (runs backend without visible window)
$vbsPath = Join-Path $script:InstallDir "Start-OpenCut-Hidden.vbs"
$vbsCommand = ('"{0}" -m opencut.server' -f $pythonCmd).Replace('"', '""')
$vbsContent = @"
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "$vbsCommand", 0, False
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
if ((Get-FFmpegSecurityStatus).Safe) {
    Write-Ok "FFmpeg .............. OK (security floor verified)"
} else {
    Write-Err "FFmpeg .............. BLOCKED (requires 8.1.3+ or post-fix snapshot >= 2026-07-06)"
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
# NOTE: native commands do not throw in PS 5.1, so try/catch never sees an
# import failure — each probe must test $LASTEXITCODE instead.
$ocVer = & $pythonCmd -c "import opencut; print(opencut.__version__)" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Ok "OpenCut package ..... OK (v$ocVer)"
} else {
    Write-Err "OpenCut package ..... MISSING"
    $allGood = $false
}

# Check Flask
& $pythonCmd -c "import flask" 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Ok "Flask server ........ OK"
} else {
    Write-Err "Flask server ........ MISSING"
    $allGood = $false
}

# Check Whisper
$whisperCheck = & $pythonCmd -c "from faster_whisper import WhisperModel; print('ok')" 2>&1
if ($LASTEXITCODE -eq 0 -and "$whisperCheck" -match "ok") {
    Write-Ok "Whisper (captions) .. OK (faster-whisper)"
} else {
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
if (-not $allGood) {
    $script:ExitCode = 1
}
if ($script:ExitCode -eq 0) {
    Write-Host "  ========================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "  ========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  HOW TO USE:" -ForegroundColor White
    Write-Host "  1. Double-click 'Start-OpenCut.bat' (or the desktop shortcut)" -ForegroundColor Gray
    Write-Host "  2. Open Premiere Pro" -ForegroundColor Gray
    Write-Host "  3. Go to Window > Extensions > OpenCut" -ForegroundColor Gray
    Write-Host "     On Premiere 2026+ the panel is under Window > Extensions (Legacy)." -ForegroundColor Gray
    Write-Host "  4. Select a clip and click 'Remove Silences'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  The generated XML imports directly into your project." -ForegroundColor Gray
    Write-Host ""
    Write-Host "  CLI is also available: opencut silence video.mp4" -ForegroundColor DarkGray
} else {
    Write-Host "  ========================================" -ForegroundColor Red
    Write-Host "  Installation FAILED." -ForegroundColor Red
    Write-Host "  ========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "  One or more required components failed to install or validate." -ForegroundColor Gray
    Write-Host "  Review the error messages above, fix the issue, and re-run Install.ps1." -ForegroundColor Gray
    Write-Host "  Exit code: $script:ExitCode" -ForegroundColor Gray
}

Write-Host ""
exit $script:ExitCode
