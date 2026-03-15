# OpenCut - CLAUDE.md

## Tech Stack
- **Backend**: Python 3.9+ / Flask, runs on localhost:5679
- **Frontend**: CEP panel (HTML/CSS/JS) for Adobe Premiere Pro
- **ExtendScript**: ES3 JSX for Premiere Pro automation (host/index.jsx)
- **Build**: PyInstaller for exe, Inno Setup for Windows installer
- **AI**: faster-whisper, demucs, pedalboard, MusicGen, Real-ESRGAN, rembg, etc.

## Key Files
- `opencut/server.py` (~7550 lines) - Flask backend, all API routes, job system
- `extension/com.opencut.panel/client/main.js` (~5340 lines) - Frontend controller
- `extension/com.opencut.panel/client/index.html` (~2490 lines) - UI layout (6 tabs)
- `extension/com.opencut.panel/client/style.css` (~3690 lines) - Themes & styles
- `extension/com.opencut.panel/host/index.jsx` (~1150 lines) - ExtendScript host
- `opencut_server.spec` - PyInstaller spec
- `OpenCut.iss` - Inno Setup installer script
- `install.py` - Cross-platform dev installer

## Architecture
- Backend runs as standalone process (exe or `python -m opencut.server`)
- Panel communicates via XHR to localhost:5679
- Job system: `_new_job()` creates job, background thread processes, SSE/polling for status
- `subprocess` aliased as `_sp` throughout server.py
- Custom dropdown system replaces native `<select>` elements in CEP
- 6 main tabs: Cut, Captions, Audio, Video, Export, Settings
- localStorage for settings persistence (`opencut_settings` key)

## Build & Run
- Dev: `python -m opencut.server` (or `pip install -e .` then `opencut-server`)
- CEP dev: Set `PlayerDebugMode = 1` in registry for all CC versions
- Build exe: `pyinstaller opencut_server.spec`
- Build installer: Inno Setup 6 compile `OpenCut.iss`

## Version
- Current: **v1.2.0**
- All version strings: pyproject.toml, __init__.py, server.py health + banner, install.py, requirements.txt, index.html header + about, main.js header, style.css header

## Gotchas
- `subprocess.run` must use `_sp.run` (the alias) except in routes with local imports
- ExtendScript is ES3: no let/const, no arrow functions, no template literals
- CEP Chromium needs `user-select: text` override for inputs
- VBS launcher must quote paths for `C:\Program Files\OpenCut`
- SSE race condition: always copy job dict inside `job_lock`
- `video_codec == "copy"` with concat must use CRF 18 instead

## v1.1.0 Features Added
- Preset save/load system (backend + UI)
- AI model management panel (list/delete downloaded models)
- GPU auto-detection & recommendation
- Job queue system (sequential processing)
- Enhanced job history with re-run capability
- Keyboard shortcuts (Enter to run, 1-6 for tabs, Escape to cancel)
- Enhanced drag-and-drop (whole panel, not just drop zone)
- Toast notifications for job completion
- Transcript search with navigation
- Social platform export presets (YouTube Shorts, TikTok, Instagram, etc.)
- Premiere Pro theme sync (CSInterface skin detection)
- Universal auto-import ExtendScript function

## v1.2.0 Features Added
- Waveform preview with draggable threshold (canvas-based, FFmpeg PCM extraction)
- Side-by-side before/after preview modal (FFmpeg frame extraction, base64 JPEG)
- Dependency health dashboard (24 optional deps + FFmpeg status grid)
- First-run wizard overlay (3-step onboarding, localStorage dismissal)
- Output file browser (recent outputs from completed jobs, Import button)
- Favorites bar (pinned operations as chips, persisted to `~/.opencut/favorites.json`)
- Batch multi-select file picker (add selected, add all, clear)
- Parameter tooltips on range sliders
- Custom workflow builder (chain operations, save/load/delete to `~/.opencut/workflows.json`)
- Audio preview player (floating player for generated audio)
- Settings import/export (JSON bundle of presets + favorites + workflows)
- Right-click context menu on clip selector
- Collapsible card headers
- Job time estimates (historical ratio-based, stored in `~/.opencut/job_times.json`)
- i18n language selector placeholder
- Video reframe tool (resize/crop for TikTok, Shorts, Reels, square, custom dims; crop/pad/stretch modes)
- Clip preview thumbnail + metadata on selection
- Command palette (Ctrl+K) with fuzzy search across all 28+ operations
- Recent clips dropdown (last 10, persisted to localStorage)
- Trim tool with in/out points (stream copy or re-encode)
- Merge/concatenate clips (fast demux or re-encoded filter modes)
- Auto-crop detect for reframe (FFmpeg cropdetect → smart anchor)
- Waveform preview buttons on denoise/normalize tabs
- Per-operation preset save/load (localStorage)
- FFmpeg real progress parsing via `-progress pipe:1`
- Job cancel kills FFmpeg subprocess (Popen terminate/kill)
- Temp file cleanup on server startup
- Server health ping with reconnect banner (10s interval)
- Output file deduplication (auto-increment suffix)
- Sub-tab visibility persistence infrastructure
