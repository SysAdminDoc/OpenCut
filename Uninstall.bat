@echo off
:: ============================================
:: OpenCut Uninstaller
:: ============================================

echo.
echo   OpenCut Uninstaller
echo   ===================
echo.

powershell -ExecutionPolicy Bypass -File "%~dp0Install.ps1" -Uninstall

pause
