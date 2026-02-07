@echo off
:: ============================================
:: OpenCut Installer - Double-click to install
:: ============================================
:: This launches the PowerShell installer with elevated privileges.
:: If prompted by UAC, click "Yes" to allow.

echo.
echo   OpenCut Installer
echo   =================
echo.

:: Check if we have admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo   Running as Administrator...
    powershell -ExecutionPolicy Bypass -File "%~dp0Install.ps1"
) else (
    echo   Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d \"%~dp0\" && powershell -ExecutionPolicy Bypass -File \"%~dp0Install.ps1\"' -Verb RunAs"
)

echo.
pause
