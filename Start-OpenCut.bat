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
python -m opencut.server
pause
