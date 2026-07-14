@echo off
setlocal

set "OPENCUT_HOME=%~dp0"
set "OPENCUT_HOME=%OPENCUT_HOME:~0,-1%"

if exist "%OPENCUT_HOME%\python\python.exe" (
    set "PYTHON=%OPENCUT_HOME%\python\python.exe"
    set "PYTHON_ARGS="
    set "PATH=%OPENCUT_HOME%\python;%OPENCUT_HOME%\python\Scripts;%PATH%"
) else (
    set "PYTHON=python"
    set "PYTHON_ARGS="
)

set "DETECTED_PYTHON=not found"
for /f "delims=" %%V in ('"%PYTHON%" %PYTHON_ARGS% --version 2^>^&1') do set "DETECTED_PYTHON=%%V"
"%PYTHON%" %PYTHON_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo  OpenCut could not start.
    echo  Detected: %DETECTED_PYTHON%
    echo  Required: Python 3.11 or later
    echo  Install a supported version from https://www.python.org/downloads/
    echo  Windows: winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

if exist "%OPENCUT_HOME%\ffmpeg" set "PATH=%OPENCUT_HOME%\ffmpeg;%PATH%"

if exist "%OPENCUT_HOME%\models" (
    set "OPENCUT_BUNDLED=true"
    set "WHISPER_MODELS_DIR=%OPENCUT_HOME%\models\whisper"
    set "TORCH_HOME=%OPENCUT_HOME%\models\demucs"
    set "OPENCUT_FLORENCE_DIR=%OPENCUT_HOME%\models\florence"
    set "OPENCUT_LAMA_DIR=%OPENCUT_HOME%\models\lama"
)

echo.
echo  OpenCut Server - Debug Mode
echo  ===========================
echo  This window shows server output for troubleshooting.
echo  Close this window to stop the server.
echo.

"%PYTHON%" %PYTHON_ARGS% -m opencut.server
pause
