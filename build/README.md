# OpenCut Build Guide

## Quick Start

```powershell
.\build\build.ps1
```

This creates `dist\OpenCut-Setup-0.7.0.exe` -- a standalone installer that requires no Python on the end user's machine.

## What the build does

1. **Checks Python 3.9+** is available
2. **Installs PyInstaller** and project dependencies
3. **Generates an icon** (icon.ico) if missing (requires Pillow)
4. **Bundles the server** into `dist\opencut-server\opencut-server.exe` via PyInstaller
5. **Creates the installer** via Inno Setup (if installed)

## Prerequisites

| Tool | Required For | Install |
|------|-------------|---------|
| Python 3.9+ | Building the exe | [python.org](https://python.org) |
| Inno Setup 6.2+ | Creating the installer | [jrsoftware.org](https://jrsoftware.org/isinfo.php) |
| Pillow | Icon generation | `pip install Pillow` (optional) |

## Build Options

```powershell
.\build\build.ps1                  # Full build
.\build\build.ps1 -SkipInstaller   # Just the exe (no Inno Setup needed)
.\build\build.ps1 -Clean           # Clean artifacts first
```

## Output

```
dist/
  opencut-server/           # Standalone server (can run without Python)
    opencut-server.exe
    ...
  OpenCut-Setup-0.7.0.exe  # Windows installer
```

## What the installer does

1. Copies the server exe to `%LOCALAPPDATA%\OpenCut\`
2. Copies the CEP extension to `%APPDATA%\Adobe\CEP\extensions\`
3. Sets `PlayerDebugMode=1` in the registry for CSXS 6-12
4. Creates Start Menu shortcuts
5. Checks for FFmpeg and prompts if missing
6. Optionally launches the server

## Bundling FFmpeg (optional)

Place `ffmpeg.exe` and `ffprobe.exe` in `build\ffmpeg\` before building, then uncomment the FFmpeg line in `installer.iss`. This embeds FFmpeg into the installer so users don't need to install it separately.
