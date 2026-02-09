@echo off
setlocal

set "OPENCUT_HOME=%~dp0"
set "OPENCUT_HOME=%OPENCUT_HOME:~0,-1%"

if exist "%OPENCUT_HOME%\python\python.exe" (
    set "PYTHON=%OPENCUT_HOME%\python\python.exe"
    set "PATH=%OPENCUT_HOME%\python;%OPENCUT_HOME%\python\Scripts;%PATH%"
) else (
    set "PYTHON=python"
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

"%PYTHON%" -m opencut.server
pause
