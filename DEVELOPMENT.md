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

## UXP Panel Development

The UXP panel at `extension/com.opencut.uxp/` targets Premiere Pro 25.6+.

### Setup
1. Install Adobe UXP Developer Tool from the Creative Cloud app
2. In UXP Developer Tool: Add Plugin → select `extension/com.opencut.uxp/`
3. Click "Load" to sideload the panel into Premiere Pro
4. The UXP panel connects to the same Python backend on port 5679

### Key differences from CEP panel
- Uses modern ES modules (`import`/`export`) — no ES3 restrictions
- Direct `fetch()` API — no HTTP bridge needed
- Timeline write-back uses `premierepro` UXP module (Premiere 25.6+)
- CEP panel handles write-back for older versions

### Manifests
- CEP manifest: `extension/com.opencut.panel/CSXS/manifest.xml`
- UXP manifest: `extension/com.opencut.uxp/manifest.json`
- Both must be kept in sync for version numbers

## New Optional Dependency Groups (v1.5.0+)

Install groups individually as needed:

```bash
# LLM features (chapters, NLP commands)
pip install openai>=1.0.0          # for OpenAI provider
pip install anthropic>=0.20.0      # for Anthropic provider
# Ollama: install from https://ollama.ai (no pip needed)

# Color matching (OpenCV required)
pip install opencv-python-headless>=4.8.0

# Auto-zoom face detection (OpenCV required)
pip install opencv-python-headless>=4.8.0

# Footage search (stdlib only — no pip required)

# Deliverables (stdlib csv only — no pip required)

# Loudness matching (FFmpeg required — external tool)
# Download from https://ffmpeg.org/download.html
```

## Version Sync

When releasing a new version, update these files:
- `opencut/__init__.py` — `__version__`
- `pyproject.toml` — `version`
- `extension/com.opencut.panel/CSXS/manifest.xml` — `ExtensionBundleVersion` and `Version`
- `extension/com.opencut.uxp/manifest.json` — `version`
- `extension/com.opencut.uxp/main.js` — `VERSION` constant
- `README.md` — version badge
- `CHANGELOG.md` — new section
