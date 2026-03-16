# Development Guide

## Prerequisites
- Python 3.9+
- FFmpeg (must be on PATH)
- Adobe Premiere Pro (for CEP panel testing)

## Backend Setup
```bash
# Clone and install in development mode
git clone https://github.com/SysAdminDoc/OpenCut.git
cd OpenCut
pip install -e ".[dev]"

# Install optional features as needed
pip install -e ".[standard]"  # Common deps (whisper, opencv, etc.)
pip install -e ".[ai]"       # AI tools (CPU)
pip install -e ".[ai-gpu]"   # AI tools (GPU with onnxruntime-gpu)
pip install -e ".[all]"      # Everything

# Run the server
python -m opencut.server
# Server starts on http://localhost:5679
```

## CEP Extension Development
1. Enable unsigned extensions by setting registry key:
   - Windows: `HKCU\Software\Adobe\CSXS.XX\PlayerDebugMode = 1` (for each CC version)
   - macOS: `defaults write com.adobe.CSXS.XX PlayerDebugMode 1`
2. The extension is at `extension/com.opencut.panel/`
3. Remote debug at `http://localhost:8842` (configured in `.debug`)
4. Reload extension in Premiere: Window > Extensions > OpenCut

## Project Structure
```
opencut/                    # Python backend
  server.py                 # Flask app, startup
  security.py               # CSRF, path validation, rate limiting
  jobs.py                   # Job state management
  helpers.py                # Shared utilities
  user_data.py              # Thread-safe file I/O
  routes/                   # Flask Blueprints
    system.py, audio.py, captions.py, video.py, jobs_routes.py, settings.py
extension/com.opencut.panel/ # CEP panel
  client/                   # HTML/CSS/JS frontend
  host/                     # ExtendScript (ES3 JSX)
  CSXS/manifest.xml         # Extension manifest
```

## Building

### Server Executable
```bash
pip install pyinstaller
pyinstaller opencut_server.spec
# Output: dist/OpenCut-Server/
```

### Windows Installer
Requires [Inno Setup 6](https://jrsoftware.org/isinfo.php) and `ffmpeg/ffmpeg.exe` + `ffprobe.exe` in project root.
```bash
iscc OpenCut.iss
```

## Linting
```bash
ruff check opencut/
ruff format opencut/
```

## Version Management
Version is defined in `opencut/__init__.py` and must match across 7+ files.
Use the sync script to update all at once:
```bash
python scripts/sync_version.py --set 1.3.0
```
