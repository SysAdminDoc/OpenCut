# OpenCut - CLAUDE.md

## Tech Stack
- **Backend**: Python 3.9+ / Flask with Blueprints, runs on localhost:5679
- **Frontend**: CEP panel (HTML/CSS/JS) for Adobe Premiere Pro
- **ExtendScript**: ES3 JSX for Premiere Pro automation (host/index.jsx)
- **Build**: PyInstaller for exe, Inno Setup for Windows installer
- **AI**: faster-whisper, demucs, pedalboard, MusicGen, Real-ESRGAN, rembg, etc.

## Key Files

### Backend (Python)
- `opencut/server.py` (~400 lines) - Flask app creation, startup, port management, download_models
- `opencut/security.py` (~180 lines) - Path validation, CSRF tokens, safe_pip_install, safe_float/safe_int (with range clamp + inf/nan rejection), validate_filepath, VALID_WHISPER_MODELS, rate_limit/require_rate_limit
- `opencut/jobs.py` (~190 lines) - Job state, _new_job, _update_job, _kill_job_process, _get_job_copy, _list_jobs_copy, _unregister_job_process, TooManyJobsError, MAX_CONCURRENT_JOBS=10, async_job decorator
- `opencut/helpers.py` (~430 lines) - _try_import, output paths, FFmpegCmd builder, FFmpeg progress runner, deferred temp cleanup, job time tracking, compute_estimate
- `opencut/errors.py` (~60 lines) - OpenCutError exception class with typed codes (MISSING_DEPENDENCY, FILE_NOT_FOUND, GPU_OUT_OF_MEMORY, INVALID_INPUT, OPERATION_FAILED), register_error_handlers
- `opencut/checks.py` (~50 lines) - Centralized dependency availability checks (demucs, watermark, pedalboard, audiocraft, edge_tts, rembg, upscale, scenedetect)
- `opencut/user_data.py` (~100 lines) - Thread-safe JSON file access for user settings (per-file locks, normalized lock keys)
- `opencut/data/social_presets.json` - Social platform export presets (YouTube Shorts, TikTok, etc.)

### Route Blueprints (`opencut/routes/`)
- `__init__.py` - `register_blueprints(app)` registers all 6 Blueprints
- `system.py` (~1030 lines) - /health, /shutdown, /info, /gpu/*, /dependencies, /file, /whisper/*
- `audio.py` (~1650 lines) - /silence, /fillers, /audio/*, /audio/pro/*, /audio/tts/*, /audio/gen/*
- `captions.py` (~1120 lines) - /captions/*, /transcript/*, /full, /captions/burnin/*
- `video.py` (~2920 lines) - /video/*, /fx/*, /ai/*, /export/*
- `jobs_routes.py` (~190 lines) - /status/*, /cancel/*, /cancel-all, /jobs, /stream/*, /queue/*
- `settings.py` (~200 lines) - /presets/*, /favorites/*, /workflows/*, /settings/import|export

### Frontend (CEP Panel)
- `extension/com.opencut.panel/client/main.js` (~5400 lines) - Frontend controller
- `extension/com.opencut.panel/client/index.html` (~2490 lines) - UI layout (6 tabs)
- `extension/com.opencut.panel/client/style.css` (~3690 lines) - Themes & styles
- `extension/com.opencut.panel/host/index.jsx` (~1150 lines) - ExtendScript host

### Build
- `opencut_server.spec` - PyInstaller spec
- `OpenCut.iss` - Inno Setup installer script
- `install.py` - Cross-platform dev installer

## Architecture
- Backend runs as standalone process (exe or `python -m opencut.server`)
- Panel communicates via XHR to localhost:5679
- **Blueprint-based route organization**: 6 Blueprints (system, audio, captions, video, jobs, settings)
- **Shared modules**: security.py (CSRF + path validation), jobs.py (job state), helpers.py (utilities), user_data.py (thread-safe file I/O)
- **CSRF protection**: Token generated at startup in security.py, returned via /health, sent as `X-OpenCut-Token` header on mutations. `@require_csrf` decorator applied to ALL POST routes.
- **Path validation**: `validate_path()` checks realpath, null bytes, `..` components, symlinks. `validate_filepath()` adds isfile check. Applied to ALL routes accepting file paths.
- **Input validation**: `safe_float()`/`safe_int()` with optional `min_val`/`max_val` range clamping and inf/nan rejection, `VALID_WHISPER_MODELS` frozenset for model name validation
- **Rate limiting**: `require_rate_limit(key)` decorator prevents concurrent expensive ops (e.g. model installs share `"model_install"` key)
- **Error taxonomy**: `OpenCutError` with typed codes (`MISSING_DEPENDENCY`, `GPU_OUT_OF_MEMORY`, etc.) — frontend `enhanceError()` adds actionable hints
- **Job safety**: `TooManyJobsError` (429), `_get_job_copy()`/`_list_jobs_copy()` for thread-safe reads, `_unregister_job_process()` for cleanup
- **Async job decorator**: `@async_job("type")` wraps routes in standard thread + try/catch + update pattern
- Job system: `_new_job()` creates job, background thread processes, SSE/polling for status
- **Request size limit**: 100 MB `MAX_CONTENT_LENGTH` with 413 error handler
- **Crash logging**: 500 errors append to `~/.opencut/crash.log` with endpoint, method, traceback
- **Subprocess tracking**: Install routes register Popen processes for cancel support via `_register_job_process`
- **Model cache**: `/models/list` caches results for 5 min TTL, invalidated on model delete
- `subprocess` aliased as `_sp` in route files
- Custom dropdown system replaces native `<select>` elements in CEP
- 6 main tabs: Cut, Captions, Audio, Video, Export, Settings
- localStorage for settings persistence (`opencut_settings` key)

## Build & Run
- Dev: `python -m opencut.server` (or `pip install -e .` then `opencut-server`)
- CEP dev: Set `PlayerDebugMode = 1` in registry for all CC versions
- Build exe: `pyinstaller opencut_server.spec`
- Build installer: Inno Setup 6 compile `OpenCut.iss` (requires `ffmpeg/ffmpeg.exe` + `ffprobe.exe` in project root)
- FFmpeg is bundled in installer and auto-added to user PATH; server also auto-detects bundled ffmpeg dir
- CI/CD: `.github/workflows/build.yml` — PyInstaller builds for Windows/macOS/Linux on `workflow_dispatch` or `v*` tag push, uploads artifacts + release tarballs
- Optional deps: `pip install -e ".[ai]"` (CPU), `pip install -e ".[ai-gpu]"` (GPU with onnxruntime-gpu), `pip install -e ".[all]"` (everything)

## Dev Tooling
- `scripts/sync_version.py` - Syncs version from `__init__.py` to all 6 target locations (`python scripts/sync_version.py --set X.Y.Z`)
- `.editorconfig` - Editor indent/encoding rules (4-space Python, 2-space JS/CSS/HTML)
- `.pre-commit-config.yaml` - ruff lint+format, trailing whitespace, EOF fixer, YAML/JSON checks
- `DEVELOPMENT.md` - Developer setup guide (backend, CEP, building, linting)
- CI: `.github/workflows/build.yml` includes ruff lint + import smoke tests before PyInstaller build

## Version
- Current: **v1.2.0**
- All version strings: pyproject.toml, __init__.py, server.py banner, install.py, requirements.txt, index.html header + about, main.js header, style.css header
- Use `python scripts/sync_version.py --set X.Y.Z` to update all at once

## Gotchas
- `subprocess.run` must use `_sp.run` (the alias) in route files
- ExtendScript is ES3: no let/const, no arrow functions, no template literals
- CEP Chromium needs `user-select: text` override for inputs
- VBS launcher must quote paths for `C:\Program Files\OpenCut`
- SSE race condition: always copy job dict inside `job_lock`
- `video_codec == "copy"` with concat must use CRF 18 instead
- Job state (`jobs` dict, `job_lock`) lives in `opencut/jobs.py` — import from there, not server.py
- User data files use per-file locks in `user_data.py` — always use the wrappers, never raw file I/O
- `_kill_job_process()` does graceful terminate → 3s wait → force kill
- Health check uses exponential backoff (4s → 60s cap) on failure, resets on success
- main.js `api()` timeout is 120s default, 10s for health checks
- Lazy DOM proxy in main.js (`el.xxx`) caches getElementById calls
- GET request deduplication in `api()` — concurrent identical GETs share one XHR
- Event delegation for batch files, workflow steps, favorites — don't attach per-element listeners
- DocumentFragment batching for DOM rebuilds (batch files, deps grid, favorites, workflow steps)
- `FFmpegCmd` builder in helpers.py — use `.build()` for new FFmpeg commands, don't construct raw lists
- Dependency checks live in `opencut/checks.py` — don't duplicate `check_X_available()` in route files
- Deferred temp cleanup: `_schedule_temp_cleanup(path)` retries with exponential backoff on Windows

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
- Server health ping with reconnect banner (exponential backoff)
- Output file deduplication (auto-increment suffix)
- Sub-tab visibility persistence infrastructure
- CSRF token protection on all POST/PUT/DELETE endpoints
- Path traversal prevention on all file-accepting endpoints
- Cancel All jobs endpoint
- Thread-safe user data file access

## v1.2.0 Reliability & DX Improvements
- 100 MB request size limit (MAX_CONTENT_LENGTH + 413 handler)
- Runtime crash logging to `~/.opencut/crash.log` (500 handler)
- safe_float/safe_int range clamping with inf/nan rejection
- Rate limiting on model install endpoints (one install at a time)
- Deferred temp file cleanup with exponential backoff retry
- Subprocess tracking in install routes for cancel support
- Model list caching (5 min TTL, invalidated on delete)
- Error taxonomy (OpenCutError with typed codes + frontend enhanceError hints)
- Async job decorator (`@async_job`) for standardized job boilerplate
- FFmpegCmd builder class for constructing FFmpeg commands
- Centralized dependency checks module (checks.py)
- GET request deduplication in frontend api() function
- Event delegation for dynamic lists (batch files, favorites, workflow steps)
- DocumentFragment batching for DOM rebuilds
- ARIA attributes on custom dropdown components (role, aria-expanded)
- Actionable error messages (pattern-matched hints for common failures)
- Version sync script (`scripts/sync_version.py`)
- .editorconfig + .pre-commit-config.yaml (ruff lint/format)
- CI lint + smoke test steps in build workflow
- DEVELOPMENT.md developer setup guide
