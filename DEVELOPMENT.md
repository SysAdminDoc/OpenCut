# Development Guide

## Prerequisites
- Python 3.11+
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
pip install -e ".[ai]"       # CPU enhancement extras
pip install -e ".[ai-gpu]"   # GPU enhancement extras with onnxruntime-gpu
pip install -e ".[all]"      # Release-audited convenience extras
pip install -e ".[all,torch-stack]"  # Add Torch-backed backends explicitly

# Run the server
python -m opencut.server
# Server starts on http://localhost:5679
```

## Bootstrap Verification

Run the stdlib-only bootstrap verifier before release work or when a local
checkout behaves inconsistently:

```bash
python scripts/bootstrap_check.py
python scripts/bootstrap_check.py --metadata-only  # skip installed dependency checks
python scripts/bootstrap_check.py --json           # machine-readable output
```

The verifier checks Python version, local `opencut` import path, version-sync
state, `requirements-lock.txt` auditability, runtime imports, and
`opencut.server.create_app` import readiness.

## CEP Extension Development
1. Enable unsigned extensions by setting registry key:
   - Windows: `HKCU\Software\Adobe\CSXS.XX\PlayerDebugMode = 1` (for each CC version)
   - macOS: `defaults write com.adobe.CSXS.XX PlayerDebugMode 1`
2. The extension is at `extension/com.opencut.panel/`
3. Remote debug at `http://localhost:8842` (configured in `.debug`, which release builds leave out)
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
The WPF installer is the recommended path, and one command builds it. You need the .NET 10 SDK, Node.js, and the pinned FFmpeg build in `ffmpeg/ffmpeg.exe` and `ffmpeg/ffprobe.exe` (the package URL and hash are in `installer/InstallerBuilder.ps1`):

```powershell
pwsh -NoProfile -File scripts/build_release.ps1
```

It makes `build/release-venv` from `requirements-release-lock.txt` and `requirements-build-lock.txt` with `--require-hashes`, and reuses it until either lock changes. Then it builds the panel bundle, freezes the server with PyInstaller and runs `installer/InstallerBuilder.ps1` with that Python. The installer and `payload.zip` land in `installer/dist/`. Pass `-SkipPanel` to keep an existing panel bundle, or `-RecreateVenv` to start the environment over.

#### Signing the panel

The installer builder signs the Premiere panel and stops before the slow steps if it can't. It needs two things:

- Adobe's ZXPSignCmd. Put the x64 `ZXPSignCmd.exe` from [Adobe-CEP/CEP-Resources](https://github.com/Adobe-CEP/CEP-Resources/tree/master/ZXPSignCMD) in `%LOCALAPPDATA%\Programs\ZXPSignCmd\<version>\`, on `PATH`, or at the path in `OPENCUT_ZXPSIGNCMD`.
- A signing certificate. Set `OPENCUT_CEP_CERT` to the `.p12` and `OPENCUT_CEP_CERT_PASSWORD` to its password, as process or user environment variables. Adobe accepts a self-signed certificate for panels:

```powershell
ZXPSignCmd -selfSignedCert US NA OpenCut OpenCut <password> "$env:LOCALAPPDATA\OpenCut-Signing\opencut-cep.p12" -validityDays 3650
```

Keep the certificate outside the checkout. `.gitignore` refuses `*.p12` and `*.pfx` in case one lands there.

The builder copies the panel without `.debug`, signs it into `installer/dist/OpenCut-Panel-<version>.zxp`, and ships the signed files (with `mimetype` and `META-INF/signatures.xml`) inside the installer. The signature gets a timestamp from DigiCert's server, or from `OPENCUT_CEP_TSA_URL`, so it stays valid after the certificate expires. `scripts/smoke_wpf_installer.ps1` fails unless the installed panel matches its signature file for file and carries that timestamp. The installers still turn on `PlayerDebugMode` until a signed panel has been seen loading in Premiere without it.

The installer builder publishes the WPF installer itself. If you only changed the installer project and want its exe without the rest, this is the command it runs:

```powershell
dotnet publish installer/src/OpenCut.Installer/OpenCut.Installer.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:EnableCompressionInSingleFile=true
```

Don't build a release with your everyday Python. Anything else installed there either ends up in the bundle or stops the build (PyInstaller quits when it finds two Qt bindings), and the installer's composition check wants an exact lock match.

The Inno Setup fallback remains available for local recovery builds:

```powershell
iscc OpenCut.iss
```

## Linting
```bash
ruff check opencut/
ruff format opencut/
```

## Version Management
Version is defined in `opencut/__init__.py` and must match across release,
extension, installer, and package files.
Use the sync script to update all at once:
```bash
python scripts/sync_version.py --set 1.3.0
python scripts/sync_version.py --check
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
pip install opencv-python-headless>=4.13.0

# Auto-zoom face detection (OpenCV required)
pip install opencv-python-headless>=4.13.0

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
