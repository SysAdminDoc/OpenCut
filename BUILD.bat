@echo off
REM Build the OpenCut Windows installer.
REM Delegates to the maintained WPF builder, which reads the version from
REM opencut/__init__.py and writes installer\dist\OpenCut-Setup-<version>.exe.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\InstallerBuilder.ps1" %*
exit /b %ERRORLEVEL%
