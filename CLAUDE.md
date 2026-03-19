# OpenCut - CLAUDE.md

## Tech Stack
- **Backend**: Python 3.9+ / Flask with Blueprints, runs on localhost:5679
- **Frontend**: CEP panel (HTML/CSS/JS) for Adobe Premiere Pro
- **ExtendScript**: ES3 JSX for Premiere Pro automation (host/index.jsx)
- **Build**: PyInstaller for exe, custom WPF installer (C# .NET 9) — legacy Inno Setup still present
- **AI**: faster-whisper, demucs, pedalboard, MusicGen, Real-ESRGAN, rembg, etc.

## Key Files

### Backend (Python)
- `opencut/server.py` (~720 lines) - Flask app creation, startup, port management, download_models, `_setup_system_site_packages()` for frozen builds
- `opencut/security.py` (~280 lines) - Path validation, CSRF tokens, safe_pip_install (frozen-build aware via `_find_system_python()`), safe_float/safe_int (with range clamp + inf/nan rejection), validate_filepath, VALID_WHISPER_MODELS, rate_limit/require_rate_limit
- `opencut/jobs.py` (~215 lines) - Job state, _new_job, _update_job, _kill_job_process, _get_job_copy, _list_jobs_copy, _unregister_job_process, TooManyJobsError, MAX_CONCURRENT_JOBS=10, async_job decorator
- `opencut/helpers.py` (~530 lines) - _try_import, output paths, FFmpegCmd builder, FFmpeg progress runner, deferred temp cleanup, job time tracking, compute_estimate, `run_ffmpeg()`, `ensure_package()`, `get_video_info()`
- `opencut/errors.py` (~60 lines) - OpenCutError exception class with typed codes (MISSING_DEPENDENCY, FILE_NOT_FOUND, GPU_OUT_OF_MEMORY, INVALID_INPUT, OPERATION_FAILED), register_error_handlers
- `opencut/checks.py` (~90 lines) - Centralized dependency availability checks (demucs, watermark, pedalboard, audiocraft, edge_tts, rembg, upscale, scenedetect, auto-editor, transnetv2, resemble-enhance, ollama)
- `opencut/user_data.py` (~100 lines) - Thread-safe JSON file access for user settings (per-file locks, normalized lock keys)
- `opencut/data/social_presets.json` - Social platform export presets (YouTube Shorts, TikTok, etc.)

### Core Modules (`opencut/core/`)
- `llm.py` (~300 lines) - LLM abstraction layer (Ollama/OpenAI/Anthropic). LLMConfig, LLMResponse, query_llm(), check_llm_reachable(). Zero pip deps (stdlib urllib).
- `silence.py` (~490 lines) - Silence detection + speed_up_silences() for sped-up (not cut) silent segments via FFmpeg filter_complex with chained atempo.
- `scene_detect.py` (~460 lines) - FFmpeg threshold + TransNetV2 ML scene detection. detect_scenes_ml() with lazy import.
- `auto_edit.py` (~400 lines) - auto-editor CLI integration for motion/audio-based editing. Parses JSON output, optional Premiere XML export.
- `audio_enhance.py` (~240 lines) - Resemble Enhance speech super-resolution. Lazy-loaded optional dependency.
- `face_reframe.py` (~390 lines) - MediaPipe face-tracking auto-framing for vertical/social video reframe. Smoothed crop path with per-second FFmpeg between() expressions.
- `highlights.py` (~350 lines) - LLM-powered highlight extraction and video summarization. Parses JSON from LLM with fallback regex.
- `shorts_pipeline.py` (~350 lines) - One-click shorts pipeline: transcribe → LLM highlights → trim → face-reframe → caption burn-in → export.
- `lut_library.py` (~580 lines) - LUT management + generate_lut_from_reference() for AI LUT generation from reference images using PIL/numpy histogram matching.

### Route Blueprints (`opencut/routes/`)
- `__init__.py` - `register_blueprints(app)` registers all 6 Blueprints
- `system.py` (~1130 lines) - /health, /shutdown, /info, /gpu/*, /dependencies, /file, /whisper/*, /llm/*
- `audio.py` (~1840 lines) - /silence, /silence/speed-up, /fillers, /audio/*, /audio/enhance, /audio/pro/*, /audio/tts/*, /audio/gen/*
- `captions.py` (~1255 lines) - /captions/*, /transcript/*, /transcript/summarize, /full, /captions/burnin/*
- `video.py` (~3475 lines) - /video/*, /video/auto-edit, /video/reframe/face, /video/highlights, /video/lut/generate-from-ref, /video/shorts-pipeline, /fx/*, /ai/*, /export/*
- `jobs_routes.py` (~280 lines) - /status/*, /cancel/*, /cancel-all, /jobs, /stream/*, /queue/*
- `settings.py` (~200 lines) - /presets/*, /favorites/*, /workflows/*, /settings/import|export

### Frontend (CEP Panel)
- `extension/com.opencut.panel/client/main.js` (~5770 lines) - Frontend controller (includes PremiereBridge UXP abstraction)
- `extension/com.opencut.panel/client/index.html` (~2710 lines) - UI layout (sidebar + content-area, 6 tabs)
- `extension/com.opencut.panel/client/style.css` (~3910 lines) - Themes & styles (sidebar navigation)
- `extension/com.opencut.panel/host/index.jsx` (~1190 lines) - ExtendScript host

### Build
- `opencut_server.spec` - PyInstaller spec
- `OpenCut.iss` - Inno Setup installer script
- `install.py` - Cross-platform dev installer

## Architecture
- Backend runs as standalone process (exe or `python -m opencut.server`)
- Panel communicates via XHR to localhost:5679
- **Blueprint-based route organization**: 6 Blueprints (system, audio, captions, video, jobs, settings)
- **Shared modules**: security.py (CSRF + path validation), jobs.py (job state), helpers.py (utilities + `run_ffmpeg` + `ensure_package` + `get_video_info`), user_data.py (thread-safe file I/O)
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
- **PremiereBridge abstraction**: All jsx/csInterface calls go through `PremiereBridge` object for future UXP migration
- `subprocess` aliased as `_sp` in route files
- Custom dropdown system replaces native `<select>` elements in CEP
- **Sidebar navigation**: 52px icon-only left sidebar (CapCut-style), CSS tooltips on hover, active tab = left accent bar
- **Layout**: `.app` = flex row → `aside.sidebar` (52px) + `.content-area` (flex:1 column → `.content-header` + banners + `main.main` + `.content-footer`)
- 6 main tabs: Cut, Captions, Audio, Video, Export, Settings
- localStorage for settings persistence (`opencut_settings` key)

## Build & Run
- Dev: `python -m opencut.server` (or `pip install -e .` then `opencut-server`)
- CEP dev: Set `PlayerDebugMode = 1` in registry for all CC versions
- Build exe: `pyinstaller opencut_server.spec`
- Build installer (legacy): Inno Setup 6 compile `OpenCut.iss`
- Build installer (custom): `cd installer && pwsh InstallerBuilder.ps1` (requires .NET 9 SDK + PyInstaller dist)
- FFmpeg is bundled in installer and auto-added to user PATH; server also auto-detects bundled ffmpeg dir
- CI/CD: `.github/workflows/build.yml` — PyInstaller builds for Windows/macOS/Linux on `workflow_dispatch` or `v*` tag push, uploads artifacts + release tarballs
- Optional deps: `pip install -e ".[ai]"` (CPU), `pip install -e ".[ai-gpu]"` (GPU with onnxruntime-gpu), `pip install -e ".[all]"` (everything)

## Dev Tooling
- `scripts/sync_version.py` - Syncs version from `__init__.py` to all 6 target locations (`python scripts/sync_version.py --set X.Y.Z`)
- `.editorconfig` - Editor indent/encoding rules (4-space Python, 2-space JS/CSS/HTML)
- `.pre-commit-config.yaml` - ruff lint+format, trailing whitespace, EOF fixer, YAML/JSON checks
- `DEVELOPMENT.md` - Developer setup guide (backend, CEP, building, linting)
- CI: `.github/workflows/build.yml` includes ruff lint (`--select E,F,I --ignore E501`) + import smoke tests before PyInstaller build
- Lint: `ruff check opencut/` — codebase is fully clean, pre-commit enforces on every commit

## Version
- Current: **v1.3.1**
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
- `FFmpegCmd` builder in helpers.py — use `.build()` for new FFmpeg commands in routes. Core modules use `run_ffmpeg(cmd, timeout=N)` directly.
- Dependency checks live in `opencut/checks.py` — don't duplicate `check_X_available()` in route files
- **Consolidated helpers** — `run_ffmpeg()`, `ensure_package()`, `get_video_info()` live in `opencut/helpers.py`. All core modules import from there. Never define local `_run_ffmpeg`/`_ensure_package`/`_get_video_info` copies.
- `ensure_package()` routes through `safe_pip_install()` from security.py — never bypass this with raw `subprocess.run(["pip", ...])` in core modules
- `get_video_info()` includes format-duration fallback for containers where stream-level duration is unavailable
- Deferred temp cleanup: `_schedule_temp_cleanup(path)` retries with exponential backoff on Windows
- **Never `git add -A`** — `installer/bin/`, `installer/obj/`, `installer/publish/` are build artifacts NOT in `.gitignore` (they're tracked in the repo). Use specific file paths when staging.
- **Frozen builds** — `sys.executable` points to the exe, not Python. `safe_pip_install()` and `_setup_system_site_packages()` detect frozen state and find system Python from PATH instead.
- **Ruff CI rules** — CI runs `ruff check opencut/ --select E,F,I --ignore E501`. Codebase is fully lint-clean as of v1.3.0. Use `# noqa: F401` for intentional lazy imports, `# noqa: E402` for delayed imports, `# noqa: F821` for closure-scoped forward refs.
- **PyInstaller spec paths** — `opencut_server.spec` must use `os.path.join()` for all paths (not backslashes). Backslash paths break Linux/macOS CI runners.

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

## v1.3.0 Features Added (New)
- **UXP Bridge abstraction** (`PremiereBridge` object in main.js) — wraps all 13 jsx/csInterface calls for future CEP→UXP migration
- **Speed-up-silence mode** — alternative to hard-cut: speeds silent segments 1.5-8x via FFmpeg atempo chains, preserving context
- **LLM abstraction module** (`opencut/core/llm.py`) — Ollama/OpenAI/Anthropic provider support, zero pip deps (stdlib urllib)
- **auto-editor integration** — motion/audio-based editing via auto-editor CLI, detects boring/static segments
- **Resemble Enhance** — speech super-resolution (upsamples low-quality speech audio to studio quality)
- **TransNetV2 ML scene detection** — neural network scene detector as alternative to FFmpeg threshold filter
- **MediaPipe face-tracking auto-framing** — auto-crop to keep face centered for vertical reframe (TikTok/Shorts/Reels)
- **LLM highlight extraction** — send transcript to LLM, get ranked viral/interesting clip timestamps
- **Video summarization** — transcript → LLM → text summary with bullet points and key topics
- **AI LUT generation from reference image** — analyze reference image color palette and generate matching .cube LUT
- **One-click shorts pipeline** — automates: transcribe → highlight → reframe → burn captions → export
- **LLM Configuration panel** in Settings tab (provider select, model, API key, base URL, test button)
- 11 new backend routes across 4 Blueprints
- 6 new core modules, 4 modified core modules

## v1.3.0 UI Overhaul
- **Sidebar navigation** — CapCut-style 52px icon-only left sidebar replaces horizontal tab bar
- CSS tooltips on hover show tab names; active tab = animated left accent bar
- `.app` layout changed from `flex-direction: column` to `row`
- New elements: `.sidebar`, `.sidebar-brand`, `.sidebar-footer`, `.content-area`, `.content-header`, `.content-title`, `.content-actions`, `.content-footer`
- Increased spacing variables: `--sp-xs:6px`, `--sp-sm:8px`, `--sp-md:12px`, `--sp-lg:16px`, `--sp-xl:20px`
- Sub-tab overflow fade indicator (`.has-overflow` class via `checkSubTabOverflow()`)
- Compact clip section (reduced glass morphism)
- Responsive: sidebar shrinks to 44px at ≤380px panel width
- Content header shows dynamic tab title (updated by `setupNavTabs()` click handler)

## v1.3.0 Code Quality Audit Fixes
- **GPU detection caching** — `_detect_gpu()` with 30s TTL replaces duplicate nvidia-smi calls in system.py
- **GPU cache thread safety** — `_gpu_cache_lock` protects `_detect_gpu()` reads/writes across concurrent requests
- **VideoCapture try/finally** — video.py watermark route now properly releases OpenCV captures + VideoWriter on error with `try/finally` on both passes
- **GPU VRAM preflight** — watermark removal checks free VRAM (≥2 GB) before loading Florence-2, falls back to CPU
- **Atomic file writes** — user_data.py writes to temp file + `os.replace()` to prevent corruption on crash
- **Queue thread safety** — jobs_routes.py entry status updates now happen under `job_queue_lock`; `_get_job_copy()` called inside lock
- **Queue dispatch timeout** — `_dispatch_queue_entry()` runs view function in sub-thread with 60s timeout
- **Pip install timeout** — 10-minute safety timer kills hung pip subprocess; `proc.wait(timeout=30)` prevents zombie
- **Dead code removal** — removed unused `_deferred_cleanup` list from helpers.py
- **XSS prevention** — favorites chip and workflow step labels now escaped with `esc()` in main.js
- **SSE cleanup** — `beforeunload` listener closes active EventSource on panel close
- **MutationObserver tracking** — custom dropdown observers stored for potential cleanup
- **Z-index hierarchy** — sidebar tooltips (500), context menu (9998), command palette (9999), toasts (10000)
- **Disabled button UX** — `filter: grayscale(0.5)` + `pointer-events: none` for clearer disabled state
- **Button accessibility** — all 179 buttons now have explicit `type="button"`
- **Version sync** — OpenCut.iss, install.py, requirements.txt, README.md all synced to v1.3.0
- **Registry InstallPath** — Inno Setup now writes HKCU `Software\OpenCut\InstallPath`
- **PyInstaller spec** — added mediapipe, auto_editor, transnetv2, resemble_enhance hidden imports
- **ExtendScript safety** — getProjectFolder() null-checks `f.parent` before accessing `.fsName`; `isProjectSaved()` added
- **Frozen-build pip install** — `safe_pip_install` uses system Python from PATH when running as PyInstaller exe
- **System site-packages discovery** — `_setup_system_site_packages()` in server.py appends system Python's site-packages to `sys.path` for frozen builds
- **Installer optional deps** — OptionsPage has "Optional Tools" section (auto-editor, edge-tts, mediapipe) with DependencyInstaller service
- **Condensed media section** — reduced padding/margins throughout #clipSection for tighter vertical layout
- **`.btn-ghost` CSS class** — added transparent/bordered button style for the "Recent" button
- **Ruff lint cleanup** — fixed 275+ lint errors across 51 files (unused imports/variables, import sorting, noqa annotations for intentional patterns)
- **Temp cleanup dedup** — `_schedule_temp_cleanup()` tracks scheduled paths to prevent duplicate timers
- **Music gen validation** — `generate_tone()` validates waveform against WAVEFORMS list, clamps frequency 20-20000Hz, duration to MAX_DURATION (3600s); `generate_sfx()` validates preset against SFX_PRESETS
- **Stderr truncation** — `_run_ffmpeg()` in music_gen.py caps stderr to 10 KB
- **Disk space preflight** — `check_disk_space()` utility in helpers.py; export-video route checks ≥500 MB free before rendering
- **Job time file safety** — `_schedule_record_time()` checks file exists before probing duration
- **Input bounds** — `max_bbox_percent` clamped 1-100 in watermark removal
- **Log export** — `/logs/export` + `/logs/clear` endpoints; UI buttons in Settings tab
- **Job retry info** — `/jobs/retry/<job_id>` endpoint returns original job params for re-run
- **ExtendScript error logging** — PremiereBridge import callbacks now `console.error()` instead of silent catch
- **Project save detection** — `isProjectSaved()` JSX function + `PremiereBridge.isProjectSaved()` + one-time toast warning on unsaved projects
- **Waveform caching** — `_waveformCache` keyed by filepath (max 10 entries, LRU eviction) avoids redundant API calls
- **validate_path() crash fix** — audio.py (musicgen, melody) and video.py (title render) had `valid, msg = validate_path()` tuple unpacking on a function that returns a string; converted to try/except
- **SSE connection race fix** — `_sse_connections` counter moved to mutable dict `_sse_state`; check + increment now atomic under single lock acquisition
- **Silence generator validation** — `generate_silence()` now clamps duration to MAX_DURATION (same as tone/sfx)
- **FFmpeg stderr cap** — `_run_ffmpeg_with_progress()` limits collected stderr to 32 KB to prevent memory bloat on long-running jobs
- **Face reframe safety** — `face_reframe()` validates target_w/target_h > 0, validates src_w/src_h > 0, releases VideoCapture on `isOpened()` failure
- **LUT path traversal fix** — `ensure_lut()` now validates user LUT names against `..`, `/`, `\` and verifies resolved path stays within LUTS_DIR
- **LUT CDF bounds** — `_apply_cdf_transfer()` clamps input value to 0.0-1.0 before bin index calculation
- **Shorts pipeline time clamp** — adjusted segment start/end times clamped to >= 0 to prevent negative timestamps
- **Highlight JSON robustness** — `_parse_highlights()` skips malformed items instead of crashing; `summarize_transcript()` validates bullet_points/topics are lists
- **Silence detection bounds** — threshold clamped -60..0 dB, min_duration 0.05..30s, padding 0..5s, min_speech 0.05..10s
- **Demucs format validation** — output_format validated against {"wav", "mp3", "flac"}, defaults to wav
- **VideoCapture/Writer isOpened()** — video.py watermark route checks `isOpened()` on both passes, fails fast on corrupt files
- **Demucs sys.executable** — audio.py uses `sys.executable` instead of hardcoded `'python'` for venv/frozen builds
- **ffprobe returncode check** — waveform route checks returncode before `json.loads` to prevent crash on probe failure
- **output_dir validation** — 6 TTS/gen/mix routes now validate `output_dir` via `validate_path()` (path traversal prevention)
- **GPU comma parse fix** — `_detect_gpu()` uses `rsplit(",", 1)` to handle GPU names containing commas
- **pyannote.audio false positive fix** — dependency check imports `pyannote.audio.pipelines` instead of just `pyannote` namespace stub
- **LLM provider allowlist** — `/llm/test` validates provider against `{"ollama", "openai", "anthropic"}`
- **Queue deadlock fix** — jobs_routes.py moves `_get_job_copy()` outside `job_queue_lock` to prevent nested lock deadlock
- **Import size limits** — settings import caps favorites (200) and workflows (100)
- **Captions import resilience** — `captions.py` uses None defaults + nested try/except for import fallback
- **GET dedup callback fix** — main.js queues pending callbacks instead of calling with `(null, null)` on deduped GETs
- **Batch poll error limit** — batch poll interval auto-clears after 10 consecutive errors
- **Blob URL revocation timing** — settings export defers `URL.revokeObjectURL()` via 5s `setTimeout`
- **HTML attribute escaping** — data-path attributes now escape `<`, `>`, `&`, `"` (XSS prevention)
- **insertClip Time object** — index.jsx uses `new Time()` instead of string `"0"` for caption/clip insertion
- **Batch import verification** — index.jsx counts items before/after `importFiles()` to report actual import count
- **Helper consolidation** — `run_ffmpeg()` (from 22 files), `ensure_package()` (from 15 files), `get_video_info()` (from 14 files) consolidated into `opencut/helpers.py`. ~886 lines of duplicate code removed. All core modules now import these shared helpers instead of defining local copies. `ensure_package()` routes through `safe_pip_install()` for security.

## v1.3.1 Batch 4 Bug Fixes
- **PyTorch API fix** — `.total_mem` → `.total_memory` in VRAM preflight check (video.py)
- **TooManyJobsError handler** — registered in `register_error_handlers()`, returns 429 JSON instead of unhandled 500
- **Export format allowlists** — `output_format` and `audio_format` validated against allowlists to prevent path injection
- **safeFixed() helper** — null-safe `.toFixed()` replacement in main.js `showResults()`; all numeric server values coerced with `Number()` before innerHTML
- **detection_prompt cap** — capped to 200 chars in watermark removal route
- **None-callable guard** — `detect_speech` checked for None before use in silence_remove and filler_removal (audio.py)
- **Time object for insertClip** — index.jsx uses `new Time()` instead of string ticks for Premiere Pro API
- **Bare subprocess removed** — deleted `import subprocess` inside `_process()`, uses `_sp` alias consistently (video.py)

## v1.3.1 Batch 5 Bug Fixes
- **ffprobe returncode check** — `detect_scenes()` and `detect_scenes_ml()` in scene_detect.py now warn on non-zero ffprobe returncode before json.loads
- **safeFixed() expansion** — 7 additional `.toFixed()` call sites in main.js converted to `safeFixed()` (file info fps/sample_rate/file_size, loudness LUFS/dBTP/LU, beat bpm/confidence, scene avg_scene_length, GPU vram_mb)
- **SSRF fix** — `list_ollama_models()` in llm.py now validates base_url hostname is localhost (matches `_query_ollama()` guard)
- **Stale version fix** — `/health` endpoint in system.py returns `__version__` instead of hardcoded "1.3.0"
- **Highlight regex fix** — `_parse_highlights_json()` fallback regex changed from non-greedy `\[[\s\S]*?\]` to greedy `\[\s*\{[\s\S]*\}\s*\]` to match JSON arrays of objects instead of short bracket expressions
- **Dead code cleanup** — removed 2 orphaned `data.get()` calls in captions.py `captions_translate()`

## v1.3.1 Batch 6 Bug Fixes
- **safeFixed() on model sizes** — model list display uses `safeFixed()` for `size_mb` and `total_mb` (crash on null API data)
- **`.closest()` null guard** — `updateSilenceModeUI()` guards `.closest(".form-group")` result before accessing `.style` (3 sites)
- **Audio tracks in getSequenceClips()** — ExtendScript now iterates `seq.audioTracks` in addition to `seq.videoTracks`, returning audio clips with `trackType: "audio"`
- **Trim time validation** — `/video/trim` validates `start_time`/`end_time` format against `HH:MM:SS(.xxx)` regex before passing to FFmpeg
- **Route parameter allowlists (video.py)** — denoise method (`nlmeans|highpass|gate`), face blur method+detector, upscale model, rembg model, watermark method (`delogo|lama`), scene detect method (`ffmpeg|ml`), LLM provider (`ollama|openai|anthropic`)
- **Route parameter allowlists (audio.py)** — denoise method (`afftdn|anlmdn|rnnoise`), stem separation stems validated against known set, filler removal model validated against `VALID_WHISPER_MODELS`
- **Thread handle tracking** — 4 routes (reframe, merge, trim, preview-frame) now store thread handles in jobs dict for cancellation support, matching the established pattern used by other routes

## v1.3.1 Batch 7 Bug Fixes
- **Stale version in settings export** — `settings.py` export bundle uses `__version__` instead of hardcoded "1.3.0"
- **Stale version in CLI** — `cli.py` `@click.version_option` uses `__version__` instead of hardcoded "1.3.0"
- **face_reframe `best_conf` NameError** — `best_conf` initialized to `0.5` before detection loop to prevent NameError when used in `FaceTrack()`
- **shorts_pipeline clip overshoot** — highlight start clamped to `total_dur - 0.1` so end never exceeds file duration
- **Caption format allowlist** — `sub_format` validated against `{srt, vtt, json, ass}` in both `/captions/generate` and `/transcript/export` routes
- **ffprobe crash in `get_video_info()`** — checks returncode and empty `streams[]` before array access; returns safe defaults on failure
- **parentNode null guard** — `addAudioWaveformButtons()` guards `parentNode` before calling `querySelector()`

## v1.3.1 Batch 8 Bug Fixes
- **selectedIndex crash guard** — workflow builder and clip select guard `selectedIndex >= 0` before accessing `options[]` (prevents TypeError on empty selects)
- **XSS: esc() for data-path** — output browser and recent clips replace inline manual escaping (missing `'`) with `esc()` helper for consistent HTML attribute escaping
- **GPU rec undefined display** — `whisper_model`, `caption_quality`, `whisper_device` fallback to "N/A" when API omits fields
- **Null-safe beat/scene counts** — `total_beats` and `total_scenes` display "--" when null/undefined instead of literal "undefined"
- **err.message type guard** — LLM test error handler checks `typeof err === "object"` before accessing `.message`
- **Keyframe validation** — `speed_ramp.py` validates each keyframe dict has `time` and `speed` keys before processing
- **TOCTOU os.unlink fix** — `diarize.py` wraps temp file unlink in try/except to prevent race condition crash
- **music_gen.py unlink safety** — concat temp list file unlink wrapped in try/except to prevent masking original errors
- **Dead ffprobe call removed** — `color_management.py` removed unused signalstats ffprobe call; added returncode + empty streams check on remaining probe
- **sys.executable for frozen builds** — `auto_edit.py` uses `sys.executable` instead of `shutil.which("python")` for PyInstaller compatibility

## v1.3.1 Batch 9 Bug Fixes
- **Blend mode allowlist** — `/video/blend` validates mode against 14 valid FFmpeg blend modes (prevents filter expression injection)
- **Style transfer allowlist** — `/video/style/apply` validates style name against 9 known styles
- **Title preset allowlist** — `/video/title/render` and `/video/title/overlay` validate preset against {fade_center, slide_left, typewriter}
- **Shorts pipeline validation** — `whisper_model` validated against `VALID_WHISPER_MODELS`, `llm_provider` validated against {ollama, openai, anthropic}
- **Stabilize crop allowlist** — `crop` param validated against {keep, black} (prevents FFmpeg filter injection)
- **Letterbox aspect allowlist** — `aspect` param validated against 7 standard ratios
- **FX route safe_float/safe_int** — 7 params (smoothing, zoom, similarity, blend, intensity x3) now use `safe_float()`/`safe_int()` with bounds clamping, matching the batch processing path
- **MusicGen model allowlist** — `model_size` validated against {small, medium, large} (prevents arbitrary HuggingFace downloads)
- **Mix duration_mode allowlist** — validated against {longest, shortest, first}
- **Summarize llm_provider** — `/transcript/summarize` validates provider (was missing, unlike `/video/highlights`)
- **Whisper model persistence** — `/whisper/settings` POST validates model against `VALID_WHISPER_MODELS` before saving
- **ExtendScript project-open guards** — `applyEditsToTimeline`, `importFileToProject`, `autoImportResult` check `app.project.rootItem` before access
- **Segment value coercion** — `applyEditsToTimeline` coerces `seg.start`/`seg.end` via `Number()` with `isNaN` guard
- **File.open() return check** — startup script writers (Windows bat, macOS sh) check return value and report error
- **Case-insensitive path dedup** — `getSelectedClips` audio track dedup uses `.toLowerCase()` for Windows path matching

## v1.3.1 Batch 10 Bug Fixes
- **Burn-in style allowlist sync** — `_VALID_BURNIN_STYLES` in captions.py updated to match actual `BURNIN_STYLES` keys in `caption_burnin.py`
- **ASS override tag injection** — `_write_ass_file()` strips `{\tag}` override expressions from user-provided caption text via regex
- **SAM2 prompt validation** — `generate_masks_sam2()` uses `.get()` with key-existence checks and `float()` coercion on point/box coordinates (prevents KeyError + type injection)
- **Delogo coord coercion** — `remove_watermark_delogo()` coerces region coords to `int()` via `.get()` with defaults (prevents type injection into FFmpeg filter)
- **Track volume coercion** — `mix_audio_tracks()` coerces track volume to `float()` before FFmpeg filter interpolation (prevents string injection)
- **FFmpeg semicolon escaping** — `render_title_card()` and `overlay_title()` escape `;` in title/subtitle text to prevent filter chain injection
- **Subtitle length cap in overlay** — title overlay route caps subtitle to 500 chars (matching render route guard)

## v1.3.1 Batch 11 Bug Fixes
- **Queue dispatch crash** — `adapter.match()` returns `(endpoint_string, view_args)`, not `(Rule, view_args)`. Fixed `rule.endpoint` → direct endpoint string usage; entire queue system was non-functional (jobs_routes.py)
- **Queue request context** — sub-thread now creates its own Flask `test_request_context` so view functions can access `request.get_json()` (jobs_routes.py)
- **Duplicate error handler** — removed redundant `TooManyJobsError` handler in server.py that clobbered the one in errors.py (lost error code field)
- **Socket leak** — `urlopen` in `_kill_via_shutdown_endpoint` wrapped in `with` (server.py)
- **Lock eviction safety** — removed lock eviction in `_get_lock()` that could break thread safety if an in-use lock was evicted (user_data.py)
- **LUT Windows path fix** — removed colon escaping (`\:`) inside single-quoted FFmpeg filter values, which corrupted Windows drive-letter paths (lut_library.py)
- **Dead code** — removed unreachable `if not tempos:` branch (silence.py), unused `info` variable + `get_video_info` import (export_presets.py)
- **Preset count limit** — `save_preset()` now caps at 500 presets, matching import route (settings.py)
- **SSE leak on cancel** — `cancelJob()` closes `activeStream` EventSource to prevent connection leak (main.js)
- **Cancel failure recovery** — cancel API error callback resets UI instead of leaving permanent "Cancelling..." state (main.js)

## v1.3.1 Batch 12 Bug Fixes
- **WaitForExit timeout crash** — `DependencyInstaller.cs` and `WhisperDownloader.cs` now check `WaitForExit()` return value; kill process on timeout instead of accessing `ExitCode` on still-running process (InvalidOperationException)
- **Pipe deadlock prevention** — both installer services consume stdout/stderr asynchronously to prevent pipe buffer deadlock when output exceeds 4 KB
- **ZIP Slip path traversal** — `PayloadExtractor.cs` validates that each extracted entry resolves within `targetDir` (prevents `../` escape in malicious ZIP entries)
- **PATH segment comparison** — `RegistryManager.cs` splits PATH by `;` for exact segment match instead of substring `Contains()` (prevents false positive on partial path match like `C:\OpenCut-Old` matching `C:\OpenCut`)
- **Process handle leak** — `ProcessKiller.cs` uses `using var proc` for `GetProcessById()` to ensure handle disposal even if `Kill()`/`WaitForExit()` throw
- **cancelJob UI deadlock** — `currentJob = null; hideProgress()` moved outside `if (err)` block so cancel success also resets UI (main.js)
- **esc(number) crash** — model list `esc(item.size_mb)` replaced with `safeFixed(item.size_mb, 1)` and `esc(item.type || "")` for null safety (main.js)
- **NaN beat count** — beat detection result guards `total_beats` with `!= null` + `Number()` coercion (main.js)
- **Batch poll timer leak** — `pollInterval` → module-level `batchPollTimer` variable, cleared on `beforeunload` (main.js)
- **Slider toFixed crash** — 3 slider `oninput` handlers use `safeFixed()` instead of raw `.toFixed()` (main.js)
- **Installer version sync** — `AppConstants.cs` version updated from "1.3.0" to "1.3.1"

## v1.3.1 Batch 13 Bug Fixes
- **LUT path double-backslash colon** — `video_fx.py` used `"\\\\:"` producing `\\:` in FFmpeg (literal backslash + separator), corrupting Windows LUT paths. Now uses single-quoted path with no colon escaping (matching lut_library.py batch 11 fix).
- **LUT colon inside single quotes** — `color_management.py` escaped colons with `\:` inside single-quoted FFmpeg filter values where `\` is literal, corrupting `C:/` drive paths. Removed colon escaping.
- **Subtitle path colon inside quotes** — `caption_burnin.py` same colon-inside-quotes bug for ass/subtitles FFmpeg filter paths. Removed colon escaping.
- **Temp file leak in audio_pro.py** — `temp_output` was excluded from cleanup when `is_video=False`, leaking temp WAV files for non-wav/flac/aiff audio processing. Now cleans all temp files that differ from `output_path`.

## v1.3.1 Batch 14 Bug Fixes
- **Transition acrossfade crash** — `apply_transition()` now probes both clips for audio streams before building filter_complex; omits `acrossfade` and maps `-an` when either clip lacks audio (prevents FFmpeg "matches no streams" crash)
- **Transition duration guard** — warns when `dur_a <= 0` from ffprobe failure instead of silently computing wrong offset
- **Style transfer allowlist sync** — route allowlists in `video.py` (lines 1280, 1629) replaced phantom "feathers"/"composition_vii" with "pointilism" to match actual `STYLE_MODELS` keys in `style_transfer.py`
- **Particles isOpened() check** — `overlay_particles()` now checks `cap.isOpened()` after `cv2.VideoCapture()` and raises early on failure (matching face_swap.py pattern)
- **Face swap temp file leak** — both `enhance_faces()` and `swap_face()` now clean up temp video file when `VideoWriter.isOpened()` fails (was leaking orphan files)

## v1.3.1 Batch 15 Security Fixes
- **Chromakey color injection** — `/video/fx/apply` chromakey effect now validates `color` param against `0x[0-9A-Fa-f]{6}` regex; prevents FFmpeg filter expression injection via crafted hex strings
- **Chromakey background path traversal** — `/video/fx/apply` chromakey `background` param now validated via `validate_filepath()` (standalone `/video/chromakey` route already validated, but fx/apply was unprotected)
- **Transition name allowlist** — `/video/transitions/apply` and `/video/transitions/join` routes validate `transition` param against `XFADE_TRANSITIONS` dict at route level (defense-in-depth; downstream already falls back to "fade")

## v1.3.1 Batch 16 Bug Fixes
- **vid.stab Windows path crash** — `stabilize_video()` transforms_file path now wrapped in single quotes with forward slashes, preventing Windows drive colon (`C:/`) from breaking FFmpeg filter option parsing (same class as LUT colon fixes in batches 11/13)
- **ensure_package crash** — `apply_pedalboard_effect()` and `deepfilter_denoise()` in audio_pro.py now check `ensure_package()` return value before importing; raises RuntimeError on install failure instead of unhandled ImportError
- **PCM alignment crash** — beat detection and audio ducking in audio_suite.py truncate PCM data to even byte length before `array.array("h")` — prevents ValueError on truncated FFmpeg output with odd byte count
- **Letterbox color injection** — `apply_letterbox()` validates color against named color set + hex regex to prevent FFmpeg pad filter injection
- **Chromakey defense-in-depth** — `chromakey()` module-level color validation via regex (complements route-level fix from batch 15)
- **ExtendScript children[i] crash** — `_findProjectItemByPath()` moved `children[i]` access inside try/catch to prevent crash on invalidated project items
- **ExtendScript rootItem guards** — added `!app || !app.project || !app.project.rootItem` guards to 5 import functions: `importXMLToProject`, `importAndOpenXml`, `importCaptions`, `importFilesToProject`, `importCaptionOverlay`
- **importCaptionOverlay try/catch** — entire function body wrapped in try/catch (was previously unguarded, crash on null project)

## v1.3.1 Batch 17 Bug Fixes
- **face_reframe ensure_package** — 3 `ensure_package()` calls now check return value; raises RuntimeError on install failure. Zero duration/fps from probe now raises ValueError instead of producing garbage output.
- **style_transfer atomic download** — `urlretrieve()` now uses 120s socket timeout, writes to `.tmp` then `os.replace()` to prevent corrupt partial downloads from persisting on disk
- **auto_edit temp dir leak** — changed `except` cleanup to `finally` so temp dir is cleaned on both success and failure paths
- **auto_edit 30fps hardcode** — `_export_premiere_xml()` now accepts `fps` param from actual ffprobe result; XML timebase matches source video instead of hardcoded 30fps
- **diarize GPU memory** — `del pipeline` + `torch.cuda.empty_cache()` in finally block prevents VRAM accumulation across successive diarize jobs
- **shorts_pipeline trim crash** — `_trim_clip()` wrapped in try/except with `continue` so single clip failure doesn't abort entire pipeline
- **speed_ramp double encoding** — concat step changed from `libx264` re-encode to `-c copy` (segments already encoded); eliminates 2x CPU time and quality loss
- **audio.py info.duration crash** — filler removal no-silence path used `_fdur` fallback instead of `info.duration` which crashes when `_finfo` is None
- **TTS rate/pitch validation** — regex validates rate (`[+-]?\d{1,3}%`) and pitch (`[+-]?\d{1,4}Hz`) before passing to edge_tts
- **audio mix tracks cap** — `/audio/mix` caps tracks list at 32 to prevent FFmpeg filter_complex resource exhaustion
- **TTS text length cap** — both `/audio/tts/generate` and `/audio/tts/subtitled` cap text at 50000 chars
- **remap_segments validation** — captions animated route validates remap_segments_raw items as dicts with start/end keys + safe_float coercion
- **word_segments cap** — animated caption render caps word_segments at 50000 entries
- **workflow count limit** — `save_workflow()` caps at 100 workflows (matching import route limit)
- **parseTimeToSec NaN** — returns 0 on NaN instead of propagating garbage values from malformed time input
- **createEvent modernization** — preset load event dispatch uses `new Event()` with CEP fallback
- **context menu overflow** — viewport-clamped positioning prevents menu from rendering off-screen
- **custom workflow queue fix** — changed `file` to `filepath` key (all custom workflow queue jobs were failing silently)
- **runBatch file selection** — uses `_batchFiles` array when populated instead of ignoring user's batch picker selection
- **Dead code + play() promise** — removed unused `_translations` variable; `showAudioPreview()` catches play() rejection

## v1.3.1 Batch 18 Bug Fixes
- **apply_lut() FFmpeg crash** — partial-intensity LUT uses filter_complex graph syntax (named pads/semicolons) but was passed via `-vf`; now correctly switches to `-filter_complex` when intensity < 1.0 (video_fx.py)
- **concat path quoting** — `concatenate_audio()` escapes single quotes in file paths for FFmpeg concat demuxer list file (music_gen.py)
- **whisper/reinstall SECURITY** — `/whisper/reinstall` backend param was unvalidated, allowing arbitrary pip install; added allowlist matching `/install-whisper` route (system.py)
- **reframe "auto" dead code** — `"auto"` position was missing from `_VALID_POSITIONS` set, making auto-crop detection unreachable; added to allowlist (video.py)
- **title overlay text cap** — `/video/title/overlay` now caps text at 500 chars, matching `/video/title/render` (video.py)
- **merge/join file cap** — `/video/merge` and `/video/transitions/join` now enforce `MAX_BATCH_FILES` limit (video.py)
- **scanForServer CSRF** — port-scan reconnection now extracts `csrf_token` from health response and resets `healthBackoff` + restarts health interval; prevents stale CSRF and 60s backoff after backend restart (main.js)
- **updateClipPreview null guards** — null-safe access on `clipThumb`, `clipMetaRes`, `clipMetaDur`, `clipMetaSize` elements (main.js)
- **command palette null guards** — `openCommandPalette`, `renderPaletteResults`, `initCommandPalette`, `paletteNavigate` all guard `commandPaletteInput` and `commandPaletteResults` (main.js)
- **clearWhisperCache crash** — `data.cleared.length` guarded with `(data.cleared ? data.cleared.length : 0)` (main.js)
- **recent clips click-outside** — document click handler closes recent clips dropdown when clicking outside (main.js)
- **refreshOutputs array check** — `Array.isArray(data)` guard prevents "undefined" display when API returns non-array (main.js)

## v1.3.0 New Optional Dependencies
```toml
auto-edit = ["auto-editor>=24.0"]
scene-ml = ["transnetv2>=1.0"]
enhance = ["resemble-enhance>=0.0.1"]
```

## Custom Installer (`installer/src/OpenCut.Installer/`)
- **Tech**: C# WPF / .NET 9, self-contained single-file exe, win-x64, requireAdministrator
- **Theme**: Catppuccin Mocha dark theme (Base #1e1e2e, Accent #89b4fa)
- **UI Flow**: Welcome -> License -> Options -> Progress -> Complete (forward-only wizard)
- **Payload**: Self-extracting exe (ZIP appended with `[data][8-byte size][OCPAYLOAD]` trailer) or adjacent `payload.zip` fallback
- **Key Files**:
  - `Models/AppConstants.cs` — version, GUIDs, registry paths
  - `Services/InstallEngine.cs` — orchestrator: 17 install steps (step 16 = optional Python deps)
  - `Services/DependencyInstaller.cs` — finds system Python, pip installs optional deps (auto-editor, edge-tts, mediapipe)
  - `Services/UninstallEngine.cs` — reverse all operations, schedule self-delete
  - `Themes/CatppuccinMocha.xaml` — full theme with all control styles
  - `Controls/LogPanel.xaml` — auto-scrolling color-coded log
  - `Controls/StepIndicator.xaml` — dot/line wizard progress
- **Build**: `cd installer && pwsh InstallerBuilder.ps1` (publishes exe, stages payload, creates self-extracting installer)
- **Uninstall**: Triggered via `--uninstall` CLI arg (registered in Add/Remove Programs)
- **Gotchas**:
  - WPF implicit usings don't include `System.IO` — added `GlobalUsings.cs`
  - `LibraryImport` requires `AllowUnsafeBlocks=true` in csproj
  - `OpenFolderDialog` is .NET 8+ only (replaces WinForms FolderBrowserDialog)
  - COM interop for `WScript.Shell` shortcut creation uses `dynamic` + `Marshal.ReleaseComObject`

## v1.3.1 Batch 19 Bug Fixes
- **safe_int OverflowError** — `int(float('inf'))` raised uncaught `OverflowError` in `safe_int()`; added to except tuple and changed `int(value)` to `int(float(value))` to also handle float strings like `"3.7"` (security.py)
- **FFmpeg progress pipe deadlock** — `_run_ffmpeg_with_progress()` consumed stdout line-by-line while stderr was deferred; if stderr exceeded pipe buffer (4KB Windows), classic deadlock. Now drains stderr in background thread (helpers.py)
- **Job times atomic write** — `_record_job_time()` used raw `open()+json.dump()` risking corruption on crash; now uses `tempfile.mkstemp()` + `os.replace()` matching `user_data.py` pattern (helpers.py)
- **Dead variable cleanup** — removed unused `_stderr_size` variable from progress runner (helpers.py)
- **particles.py ensure_package** — `ensure_package()` return value unchecked; `import cv2` crashed with unhandled `ImportError` on failed install. Added return guard + `RuntimeError`
- **particles.py writer.isOpened()** — `VideoWriter` creation not checked; silent corrupt output on codec/resolution failure. Added guard with cleanup
- **chromakey.py ensure_package** — same `ensure_package()` return value bug as particles.py
- **chromakey.py writer.isOpened()** — same `VideoWriter` guard as particles.py, plus `fg_cap`/`bg_cap` cleanup on failure
- **motion_graphics.py drawtext escaping** — POSIX shell `'\\''` trick doesn't work in FFmpeg filter expressions; changed to FFmpeg-native `\\'` escaping. Text with apostrophes (e.g., "don't") no longer breaks the filter parser
- **motion_graphics.py filename sanitization** — output filename only stripped spaces and `/`; now strips all Windows-reserved characters (`\:*?"<>|`) via regex
- **index.jsx .command open check** — macOS `.command` fallback didn't check `open("w")` return value (`.bat` and `.sh` paths did); added matching guard
- **index.jsx importCaptions polling** — `_findProjectItemByPath()` ran immediately after `importFiles()` with no delay; added polling loop (20 attempts × 50ms) matching `importFileToProject()` pattern

## v1.3.1 Batch 20 Bug Fixes
- **Crop position expressions inverted** — reframe crop mode used `(ow-iw)/2` which evaluates negative (clamped to 0), pinning all crops to top-left. Changed to `(iw-ow)/2` for correct centering; same fix on all 5 named positions (video.py)
- **Queue allowlist 15 phantom routes** — `_ALLOWED_QUEUE_ENDPOINTS` had 15 paths that don't match actual Flask routes (e.g. `/captions/styled` → `/styled-captions`, `/export/video` → `/export-video`, `/fx/*` → `/video/fx/apply`, `/ai/*` → `/video/ai/*`). ~40% of queue operations silently failed (jobs_routes.py)
- **face_blur batch allowlist mismatch** — batch processing path validated "fill" instead of "black" for face blur method; `blur_faces()` accepts "black" (video.py)
- **`.rebuild()` TypeError** — preset list refresh called `.rebuild()` on custom dropdown which doesn't exist; changed to `.update()` (main.js)
- **styled_captions PCM truncation** — `array.array("h")` crashed on odd-length PCM bytes from FFmpeg; truncates to even byte count before parsing (same class as batch 16 audio_suite fix) (styled_captions.py)
- **face_tools temp file leak** — face detection extracted first frame to temp PNG but didn't clean up on ffmpeg/imread error; wrapped in try/finally with os.unlink (face_tools.py)
- **style_transfer ensure_package** — `ensure_package()` return value unchecked; `import cv2` crashed with ImportError on failed install. Added return guard + RuntimeError (style_transfer.py)
- **Preset slider display** — loading per-operation presets fired "change" event which doesn't trigger slider `oninput` display handlers; now dispatches "input" for range elements (main.js)

## v1.3.1 Batch 21 Bug Fixes
- **Rate limit async bug** — `@require_rate_limit("model_install")` decorator releases rate limit slot immediately when async install routes return (before background thread finishes), making concurrent protection useless. 4 routes fixed: `install_whisper` + `whisper_reinstall` (system.py — also fixed double-release since thread's finally also called `rate_limit_release`), `audio_pro_install` + `tts_install` (audio.py — added missing `rate_limit_release` in thread's finally). All 4 now use manual `rate_limit()`/`rate_limit_release()` instead of decorator, with early-return paths also releasing the slot.
- **ensure_package return guards** — 16 unchecked `ensure_package()` calls across 5 core modules crashed with `ImportError` when pip install failed (return value not checked before subsequent import). Added `if not ensure_package(...): raise RuntimeError(...)` guards: voice_gen.py (4: edge_tts x2, kokoro, soundfile — soundfile also gained missing pip_name and on_progress args), face_swap.py (3: cv2, cv2+onnxruntime), animated_captions.py (2: cv2, Pillow), video_ai.py (5: realesrgan, basicsr, cv2 x3, rembg), face_tools.py (2: cv2 in blur_faces + detect_faces_in_frame)

## v1.3.1 Batch 22 Bug Fixes
- **captions_enhanced ensure_package** — 6 unchecked `ensure_package()` calls (whisperx, ctranslate2 x2, sentencepiece x2, huggingface_hub x2, pysubs2 x2) now check return value and raise RuntimeError on failure
- **WhisperX GPU memory leak** — model + align_model wrapped in try/finally with `del` + `torch.cuda.empty_cache()` to free VRAM between jobs
- **CTranslate2 translator leak** — `translate_text()` and `translate_segments()` translator objects wrapped in try/finally with `del translator` to prevent memory accumulation
- **Empty translation crash** — both translate functions guard `if not results or not results[0].hypotheses` before accessing output tokens
- **object_removal ensure_package** — 2 unchecked cv2 `ensure_package()` calls in `generate_masks_sam2()` and `remove_watermark_lama()`
- **SAM2 temp dir leak** — frames extraction dir wrapped in try/finally with `shutil.rmtree()` (was only cleaned on success path)
- **mask_region KeyError** — `remove_watermark_lama()` changed `mask_region["x"]` to `.get("x", 0)` with `int()` coercion (4 keys)
- **VideoCapture/Writer isOpened** — `remove_watermark_lama()` checks `cap.isOpened()` and `writer.isOpened()` after construction
- **Zero-dimension validation** — `remove_watermark_lama()` validates w/h/fps > 0 from `get_video_info()` before processing
- **upscale_pro ensure_package** — 1 unchecked cv2 `ensure_package()` in `upscale_realesrgan()`
- **thumbnail ensure_package** — 1 unchecked cv2 `ensure_package()` in `generate_thumbnails()`
- **Rule-of-thirds break bug** — inner `break` only exited inner `ty` loop, not outer `tx` loop; face at multiple thirds intersections inflated score. Fixed with `thirds_bonus` flag pattern
- **Rate limit async (video.py)** — `video_ai_install` and `face_install` used `@require_rate_limit` decorator which releases on route return, not thread completion. Converted to manual `rate_limit()`/`rate_limit_release()` with early-return + finally release
- **Segment type coercion** — export_video segment start/end now coerced via `safe_float()` in both audio-only and video+audio paths (prevents TypeError from JSON string values)
- **bg_color injection** — rembg route validates bg_color against `^[a-zA-Z0-9#]+$` regex; rejects injection payloads
- **Phantom colorspace removal** — removed "aces", "bt709", "bt2020" from colorspace allowlist (silently passed validation but produced wrong FFmpeg output)
- **Dead position cleanup** — removed "face" from `_VALID_POSITIONS` (unreachable code path) and `\d+:\d+` regex fallback (never matched valid user input)

## v1.3.1 Batch 23 Bug Fixes
- **audio_enhance GPU leak** — Resemble Enhance `audio` tensor and torch CUDA cache freed in finally block (`del audio` + `torch.cuda.empty_cache()`)
- **auto_edit master clip fps** — FCP XML master clip `<rate><timebase>` used hardcoded 30; now uses actual source fps via `{timebase}` variable
- **auto_edit consolidated probe** — replaced local `_probe_duration()`/`_probe_fps()` with `_probe_media_info()` wrapping consolidated `get_video_info()`
- **auto_edit URL encoding** — `<pathurl>` in Premiere XML now uses `urllib.parse.quote()` instead of `html.escape()` (spaces/special chars broke import)
- **auto_edit float coercion** — threshold/margin/min_clip_length coerced to `float()` before interpolation into command args
- **speed_ramp float coercion** — keyframe `time`/`speed` values coerced to `float()` (prevents TypeError from JSON string values)
- **speed max clamp sync** — speed change route max_val changed from 100.0 to 8.0 to match `speed_ramp.py` module clamp
- **silence returncode check** — FFmpeg silencedetect result.returncode checked with logger.warning on failure
- **silence float coercion** — `threshold_db`/`min_duration` coerced to `float()` before FFmpeg filter interpolation
- **voice_gen path traversal** — voice param sanitized with `re.sub(r'[^a-zA-Z0-9_]', '_', voice)` to prevent directory traversal via `../../` in output filenames (3 sites: edge_tts_generate, edge_tts_with_subtitles, kokoro_generate)
- **voice_gen empty output** — edge_tts_with_subtitles validates audio file exists and is non-empty before returning
- **audio_pro GPU cache** — added `torch.cuda.empty_cache()` after `del model, df_state` in DeepFilterNet finally block
- **audio_suite denoise allowlist** — route allowlist synced from `("afftdn", "anlmdn", "rnnoise")` to `("afftdn", "highpass", "gate")` to match actual module methods
- **audio_suite stderr truncation** — `normalize_loudness` error message truncated to last 500 chars (was exposing full stderr)
- **face_swap GPU leaks** — `enhance_faces()` frees GFPGANer restorer, `swap_face()` frees FaceAnalysis app + swapper model, both with `del` + `torch.cuda.empty_cache()` in finally blocks
- **face_reframe sample_rate guard** — `sample_rate <= 0` clamped to 1 to prevent ZeroDivisionError in `frame_idx % sample_rate`
- **face detector name fix** — route allowlist changed `"haarcascade"` to `"haar"` to match `face_tools.py` module's expected value (was causing `face_det = None` → AttributeError)
- **face_detect allowlist** — `/video/face/detect` route now validates detector param against `("mediapipe", "haar")` (was unvalidated)
- **easing allowlist** — added `"exponential"` to `_VALID_EASING` set in video.py

## v1.3.1 Batch 25 Bug Fixes (Full Audit)
- **Batch denoise TypeError** — `_execute_batch_item` passed `output_dir=` kwarg to `denoise_audio()` which doesn't accept it; now constructs explicit `output_path` (video.py)
- **Batch normalize ImportError** — imported non-existent `normalize_audio`; fixed to `normalize_loudness` with explicit `output_path` (video.py)
- **Rate limit slot leak** — `_new_job()` after `rate_limit()` could raise `TooManyJobsError`, permanently leaking the rate limit slot; wrapped in try/except with release on all 4 affected routes (video.py: video_ai_install, face_install; system.py: install_whisper, whisper_reinstall)
- **XSS via innerHTML** — `updateClipPreview()` injected `data.image` into innerHTML; switched to `document.createElement("img")` + `.src` assignment (main.js)
- **getSequenceClips project guard** — added `!app || !app.project` check before accessing `activeSequence` (index.jsx)
- **Queue allowlist +15 endpoints** — added `/video/pip`, `/video/blend`, `/video/merge`, `/video/trim`, `/video/face/enhance`, `/video/face/swap`, `/video/object/remove`, `/video/watermark`, `/video/title/overlay`, `/audio/separate`, `/audio/tts/generate`, `/audio/tts/subtitled`, `/audio/duck`, `/transcript/summarize` (jobs_routes.py)
- **SSE CORS origin** — hardcoded `"null"` replaced with request origin echo for `"null"` and `"file://"` (jobs_routes.py)
- **fetchTimeEstimate duration** — regex expected `\d+s` but `fmtDur()` outputs `M:SS`; now parses `M:SS` and `H:MM:SS` formats (main.js)
- **updateClipPreview waveform abuse** — replaced heavyweight `/audio/waveform` call (PCM extraction) with lightweight `/info` probe for clip metadata (main.js)
- **Merge file listener leak** — `renderMergeFiles()` attached per-button click listeners on every render; switched to event delegation via `ensureMergeDelegation()` (main.js)
- **CSS animation GPU waste** — 8 infinite animations paused via `animation-play-state: paused` when panel hidden (Page Visibility API) (style.css + main.js)
- **MutationObserver leak** — `refreshClipDropdown()` and `populateDropdown()` now call `observer.disconnect()` before recreating custom dropdowns (main.js)
- **Transcript editor delegation** — replaced per-textarea input listeners with single delegated handler on container (main.js)
- **importXMLToProject sleep** — replaced blocking `$.sleep(500)` with 50ms polling loop (20 iterations max) (index.jsx)
- **Sub-tab scroll affordance** — replaced `scrollbar-width: none` with thin 3px scrollbar so mouse users can discover overflow tabs (style.css)
- **Content header backdrop-filter** — wrapped in `@supports` with solid color fallback for CC 2019-2020 (style.css)
- **Card hover jank** — removed `translateY(-3px)` from `.card:hover` (style.css)
- **Export video stderr loss** — split stdout/stderr pipes with background drain thread; error messages now include last 500 chars of stderr (video.py)
- **Deduplicated _output_path** — moved to `helpers.output_path()`, replaced local copies in `video_fx.py` and `video_ai.py`; removed dead `_probe_duration()` and unused `subprocess` import from video_fx.py
- **_MAX_FILE_LOCKS enforcement** — `user_data.py` now evicts oldest lock when at capacity (was defined but never checked)
- **Version sync script** — added 8 new targets: CSXS manifest (bundle + extension), `.csproj` (Version + FileVersion), `AppConstants.cs`, `install.py`, `requirements.txt`, `OpenCut.iss`; fixed stale regex for manifest Extension Id; removed dead index.html `<span class="version">` pattern
- **Version desync fixed** — all 15 version locations now at v1.3.1 (were stale at 1.3.0)
- **Waveform cache comment** — corrected "LRU" to "FIFO" (main.js)
- **jobStarting safety** — wrapped `api()` call in try/catch to clear `jobStarting` flag on synchronous throw (main.js)

## v1.3.1 Batch 24 Bug Fixes (Previously Pending, Now Applied)
- **styled_captions stderr timeout** — `proc.stderr.read()` capped at 64KB; `proc.wait()` wrapped in try/except for `TimeoutExpired` with kill (styled_captions.py)
- **styled_captions fps validation** — fps <= 0 or NaN/inf now defaults to 30.0 (styled_captions.py)
- **lut_library NaN intensity** — NaN/inf/non-numeric intensity now defaults to 1.0 before clamping (lut_library.py)
- **shorts_pipeline float coercion** — `_trim_clip()` start/end coerced to float with NaN/inf/negative validation (shorts_pipeline.py)
- **shorts_pipeline int coercion** — `target_w`/`target_h` coerced to `int()` in crop/scale filter (shorts_pipeline.py)
- **animated_captions VideoWriter** — `writer.isOpened()` check added; fps <= 0 fallback to `CAP_PROP_FPS` or 30 (animated_captions.py)
- **motion_graphics ffprobe crash** — `overlay_title()` lower_third ffprobe wrapped in try/except for FileNotFoundError/TimeoutExpired (motion_graphics.py)
- **motion_graphics semicolon escape** — kinetic_bounce per-word text now escapes `;` to prevent FFmpeg filter chain injection (motion_graphics.py)
- **motion_graphics typewriter div/zero** — `chars` clamped to max(1,...), `char_dur` clamped to max(0.01,...) (motion_graphics.py)
- **animation preset allowlist sync** — route `"slide"`→`"slide_up"`, `"wave"`→`"highlight_box"` to match module ANIMATION_PRESETS (captions.py)
- **title preset allowlist expanded** — added `lower_third`, `countdown`, `kinetic_bounce` to both render and overlay route allowlists (video.py)
- **video_fx float coercion** — `smoothing`/`zoom` coerced to int, `similarity`/`blend` coerced to float in FFmpeg filter strings (video_fx.py)
- **color_management dual filter fix** — exposure+contrast+saturation merged into single `eq=` filter; temperature+shadows+midtones+highlights merged into single `colorbalance=` filter (color_management.py)
- **style_transfer intensity clamp** — route max changed from 2.0 to 1.0 (>1.0 caused image inversion via `cv2.addWeighted`) (video.py)
- **whisper_reinstall proc.wait timeout** — `proc.wait()` given 600s timeout with kill on expiry (system.py)
- **favorites max limit sync** — save route raised to 200 (matching import route) (settings.py)
- **style_transfer thread-safe download** — replaced `socket.setdefaulttimeout()` (process-global) with `urlopen(timeout=120)` per-request; removed dead `socket` import (style_transfer.py)
- **video_fx dead code removed** — `_probe_duration()` (never called) + unused `subprocess` import cleaned up (video_fx.py, done in batch 25)
- **Installer Close button** — ProgressPage shows Close button on install failure (ProgressPage.xaml/.cs)
- **Installer path validation** — OptionsPage validates path characters, length (max 200), and disk space (min 500 MB) (OptionsPage.xaml.cs)

## v1.3.1 Batch 26 Bug Fixes
- **Dynamic button type** — added `type="button"` to 5 dynamically created buttons (job history re-run, output import, batch remove, workflow remove, merge remove) (main.js)
- **getProjectInfo guard** — added `!app || !app.project` guard before accessing project properties (index.jsx)
- **getProjectFolder guard** — added `!app || !app.project` guard (index.jsx)
- **isProjectSaved guard** — added `!app || !app.project` guard (index.jsx)
- **showRecentClips race** — replaced `classList.toggle("hidden")` with explicit `classList.remove("hidden")` to avoid race with outside-click dismiss (main.js)
- **ProcessKiller deprecated wmic** — replaced `wmic` with `powershell Get-CimInstance Win32_Process` for Windows 11 23H2+ compatibility (ProcessKiller.cs)
- **Job history delegation** — replaced per-button addEventListener with single delegated handler on `el.jobHistory` container (main.js)
- **captions_enhanced_install rate limit** — converted from `@require_rate_limit` decorator (releases immediately on route return) to manual `rate_limit()`/`rate_limit_release()` with try/except on `_new_job()` and finally release in thread (captions.py)
- **Install.ps1 version** — updated from "0.5.0" to "1.3.1" (Install.ps1)
- **app.manifest version** — updated from "1.3.0.0" to "1.3.1.0" (app.manifest)
- **Version sync +2 targets** — added `app.manifest` and `Install.ps1` to `sync_version.py` (now 17 targets total)
- **Legacy InstallerBuilder.ps1** — marked as DEPRECATED with note to use `installer/InstallerBuilder.ps1` instead

## Phase 1 — Competitive Upgrades (Quick Wins)
- **rembg default → BiRefNet** — Changed default model from `u2net` to `birefnet-general` in route + module + UI. Added `birefnet-massive` as highest-quality option. Dramatically sharper edge detection.
- **Whisper default → turbo** — Changed default model from `base` to `turbo` (6x faster, near large-v3 accuracy). Updated all 3 UI selectors + user_data defaults + installer.
- **Distil-Whisper models** — Added `distil-large-v3.5`, `distil-large-v3`, `distil-medium.en`, `distil-small.en` to `VALID_WHISPER_MODELS`. Added distil-large-v3.5 and distil-large-v3 to all UI model selectors. These are 6x faster, 49% smaller, within 1% WER of full Whisper.
- **faster-whisper min version** — Bumped from `>=1.0` to `>=1.1` in pyproject.toml (required for turbo + distil model support)

## Competitive Upgrade Roadmap (March 2026 Research)

### Phase 2 — Dependency Swaps (Medium Effort)
- [x] **Audio separation**: Added `python-audio-separator` backend with Mel-Band RoFormer, BS-RoFormer, SCNet, MDX23C models alongside Demucs (backend param in `/audio/separate`)
- [x] **Speech enhancement**: Added `ClearerVoice-Studio` as recommended backend (MossFormer2/FRCRN) alongside Resemble Enhance. `backend` param in `/audio/enhance`
- [x] **Style transfer**: Added `arbitrary_style_transfer()` — any image as style reference via AdaIN color transfer in LAB space. New `/video/style/arbitrary` route. Original .t7 preset styles retained.
- [x] **Object removal**: Added `inpaint_video_propainter()` for temporally coherent video inpainting (ICCV 2023). LAMA retained as per-frame fallback.
- [x] **Face enhancement**: Added `CodeFormer` alongside GFPGAN — tunable fidelity slider (0=quality, 1=identity), model param in `/video/face/enhance`
- [x] **Face detection**: Added InsightFace `buffalo_l` as `"insightface"` detector option in face_tools (highest accuracy). Route allowlists updated.

### Phase 3 — New Features (Higher Effort)
- [x] **Music generation**: Added `ACE-Step 1.5` — full songs WITH vocals+lyrics, `/audio/music-ai/ace-step` route
- [x] **TTS tiers**: Kokoro already existed; added `Chatterbox` (voice cloning, emotion, 23 langs, MIT) as `"chatterbox"` engine in `/audio/tts/generate`
- [x] **Voice cloning**: Via Chatterbox `voice_ref` param — zero-shot from 5s audio, emotion control
- [x] **AI color grading**: Added `generate_lut_ai()` — LAB perceptual percentile matching (inspired by Image-Adaptive-3DLUT). New `/video/lut/generate-ai` route
- [ ] **Motion graphics**: Add `Remotion` render service — React-based, After Effects quality titles/animations vs FFmpeg drawtext
- [x] **Video denoising**: Added `BasicVSR++` as `"basicvsr"` method in `/video/ai/denoise` — GPU temporal propagation, chunk-based processing, strength-blended output
- [x] **Scene detection**: Added `PySceneDetect` as `"pyscenedetect"` method in `/video/scenes` — heuristic, fast, ContentDetector
- [x] **Neural LUT blending**: Added `blend_luts()` — linearly interpolate between any two .cube LUTs with a slider. New `/video/lut/blend` route
- [x] **Translation**: Added `SeamlessM4T v2` via `translate_text_seamless()` — 20% BLEU improvement over NLLB. `backend` param in `/captions/translate`
- [x] **Caption NLP emphasis**: Added `detect_keywords_nlp()` — TF-IDF-like frequency analysis + POS heuristics for auto-emphasis. Integrated into `get_action_word_indices()`

### Phase 4 — Architecture (Long-term)
- [ ] **UXP migration** — CEP deprecated, removal late 2026. PremiereBridge abstraction already in place. Test with UXP samples.
- [ ] **MCP server exposure** — Expose OpenCut's 81 endpoints as MCP server for AI client integration (Claude Code, Cursor, etc.)
- [ ] **Vision-augmented highlights** — GPT-4o/Claude frame sampling alongside transcript for visual-only highlights
- [x] **Transcription slicing** — Added `_transcript_cache` with FIFO eviction (max 20). `cache_transcript()` / `get_cached_transcript()` in captions routes. Keyed by filepath+mtime. `force_retranscribe` param to bypass.

### Keep As-Is (Already Best-in-Class)
- faster-whisper (transcription engine), WhisperX (alignment), Real-ESRGAN (upscaling), InsightFace (face swap), auto-editor (auto-editing), pedalboard (audio effects), pyannote.audio (diarization — update to v4.0.4)
