# OpenCut - CLAUDE.md

## Tech Stack
- **Backend**: Python 3.9+ / Flask with Blueprints, runs on localhost:5679
- **Frontend**: CEP panel (HTML/CSS/JS) for Adobe Premiere Pro
- **ExtendScript**: ES3 JSX for Premiere Pro automation (host/index.jsx)
- **Build**: PyInstaller for exe, custom WPF installer (C# .NET 9) — legacy Inno Setup still present
- **AI**: faster-whisper, demucs, pedalboard, MusicGen, Real-ESRGAN, rembg, etc.

## Key Files

### Backend (Python)
- `opencut/server.py` (~800 lines) - Flask app creation, startup, port management, download_models, `_setup_system_site_packages()` for frozen builds. JSON structured logging via python-json-logger (file handler; console stays plain text). `[job_id]` correlation via _JobLogFilter. `/logs/tail` endpoint for filtered log viewing.
- `opencut/security.py` (~280 lines) - Path validation, CSRF tokens, safe_pip_install (frozen-build aware via `_find_system_python()`), safe_float/safe_int (with range clamp + inf/nan rejection), validate_filepath, VALID_WHISPER_MODELS, rate_limit/require_rate_limit
- `opencut/jobs.py` (~240 lines) - Job state, _new_job, _update_job, _kill_job_process, _get_job_copy, _list_jobs_copy, _unregister_job_process, TooManyJobsError, MAX_CONCURRENT_JOBS=10, async_job decorator. Thread-local job_id for log correlation (_thread_local.job_id set in _process, cleared in finally). _safe_error delegates to errors.safe_error for structured classification.
- `opencut/helpers.py` (~530 lines) - _try_import, output paths, FFmpegCmd builder, FFmpeg progress runner, deferred temp cleanup, job time tracking, compute_estimate, `run_ffmpeg()`, `ensure_package()`, `get_video_info()`
- `opencut/errors.py` (~230 lines) - Structured error taxonomy. OpenCutError exception class with code/message/suggestion. `error_response()` helper for routes. `safe_error(exc, context)` classifies exceptions (MemoryError→GPU_OUT_OF_MEMORY, TimeoutError→OPERATION_TIMEOUT, PermissionError→PERMISSION_DENIED, ImportError→MISSING_DEPENDENCY, etc.) and returns structured JSON with recovery suggestions. Factory constructors: missing_dependency, file_not_found, gpu_out_of_memory, invalid_input, invalid_model, operation_failed, rate_limited, queue_full, module_not_available, file_permission_denied, too_many_items, server_busy, install_failed. All errors return `{error, code, suggestion}` JSON.
- `opencut/checks.py` (~90 lines) - Centralized dependency availability checks (demucs, watermark, pedalboard, audiocraft, edge_tts, rembg, upscale, scenedetect, auto-editor, transnetv2, resemble-enhance, ollama)
- `opencut/user_data.py` (~100 lines) - Thread-safe JSON file access for user settings (per-file locks, normalized lock keys)
- `opencut/data/social_presets.json` - Social platform export presets (13 platforms: YouTube Shorts/Long, TikTok, Instagram Reel/Story/Post, Twitter/X, LinkedIn, Snapchat, Facebook Reel/Post, Pinterest, Podcast MP3)

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

**New in v1.5.0:**
- `repeat_detect.py` - Jaccard similarity sliding-window to detect fumbled/repeated takes. detect_repeated_takes(segments, threshold=0.6, gap_tolerance=2.0) → {repeats, clean_ranges}. merge_repeat_ranges() for overlapping range merging.
- `chapter_gen.py` - YouTube chapter generation via LLM (with heuristic pause-detection fallback). generate_chapters(segments, llm_config, max_chapters) → {chapters, description_block}. LLM response JSON parsed with regex fallback.
- `footage_search.py` - BM25-lite transcript index at `~/.opencut/footage_index.json`. index_file(), search_footage(), clear_index(), get_index_stats(). Cross-platform file locking (_FileLock) + atomic write (os.replace). Always returns structured results, never crashes on empty index.
- `deliverables.py` - Post-production CSV documents from sequence_data dict. generate_vfx_sheet/adr_list/music_cue_sheet/asset_list() all return {output: path, rows: N}. Music detection: audio track index >= 2. Helpers: _seconds_to_tc(secs, fps) → "HH:MM:SS:FF", _seconds_to_readable(secs) → "H:MM:SS".
- `color_match.py` - YCbCr histogram matching with cv2/numpy. color_match_video(source, reference, output, strength=1.0, on_progress=None). extract_color_stats() for profiling. cv2/numpy wrapped in try/except.
- `multicam.py` - Speaker diarization → multicam cut list. generate_multicam_cuts(segments, speaker_map, min_cut_duration) → {cuts, total_cuts, speaker_to_track}. auto_assign_speakers() assigns tracks in order of first appearance. merge_diarization_segments(gap_tolerance=0.5) merges same-speaker gaps. Pure Python, no optional deps.
- `auto_zoom.py` - Face-tracked zoom keyframes via OpenCV Haar cascade. generate_zoom_keyframes(path, zoom_amount=1.15, easing, on_progress=None) → {keyframes, fps, duration}. _ease(t, mode) for linear/ease_in/ease_out/ease_in_out curves — boundary: _ease(0)=0, _ease(1)=1. Center-crop fallback if no face detected. cv2 wrapped in try/except.
- `loudness_match.py` - FFmpeg loudnorm two-pass LUFS normalization. measure_loudness(file) → {lufs, lra, peak, true_peak} parsed from stderr JSON. normalize_to_lufs(input, output, target=-14.0). batch_loudness_match(files, output_dir, target_lufs, on_progress=None) → [{input, output, original_lufs, job_ok}].
- `nlp_command.py` - Natural language → API route mapping. COMMAND_MAP with 19 entries. parse_command_keyword(text) → {route, params, confidence, matched_keyword} or None. parse_command_llm(text, config, routes). parse_command(text, llm_config) tries LLM then keyword. extract_params_from_text(text) extracts numbers/language/intensity hints.

### Route Blueprints (`opencut/routes/`)
- `__init__.py` - `register_blueprints(app)` registers all 11 Blueprints (added workflow_bp)
- `system.py` (~1130 lines) - /health, /shutdown, /info, /gpu/*, /dependencies (includes color_match, auto_zoom, footage_search, loudness_match, deliverables, nlp_command checks), /file, /whisper/*, /llm/*
- `audio.py` (~2175 lines) - /silence, /fillers, /audio/*, /audio/beat-markers (→ beat timestamps for ExtendScript markers), /audio/loudness-match (async, on_progress)
- `captions.py` (~1590 lines) - /captions/*, /captions/chapters (LLMConfig object, not dict), /captions/repeat-detect (word-level timestamps → detect_repeated_takes)
- `video.py` (~4021 lines) - /video/*, /video/color-match (async, on_progress), /video/auto-zoom (dynamic resolution via probe — no hardcoded hd1080), /video/multicam-cuts (result.get("cuts") — dict not list)
- `jobs_routes.py` (~330 lines) - /status/*, /cancel/*, /cancel-all, /jobs, /stream/*, /queue/*, /jobs/history (SQLite-backed), /jobs/stats, /jobs/interrupted
- `settings.py` (~440 lines) - /presets/*, /favorites/*, /workflows/*, /settings/import|export, /settings/llm (GET masks key, POST preserves masked), /settings/loudness-target, /settings/auto-zoom, /settings/chapters, /settings/multicam, /settings/footage-index, /logs/tail (filtered log viewer), /templates/list, /templates/save, /templates/apply

**New in v1.5.0:**
- `timeline.py` - /timeline/export-from-markers (FFmpeg clip extraction per marker), /timeline/batch-rename (validates renames for ExtendScript), /timeline/smart-bins (validates rules), /timeline/srt-to-captions (parse SRT or pass-through segments), GET /timeline/index-status
- `search.py` - POST /search/index (transcribe → index_file, async job), POST /search/footage (search_footage), DELETE /search/index (clear_index)
- `deliverables.py` - POST /deliverables/vfx-sheet|adr-list|music-cue-sheet|asset-list → {output, rows}. All handle dict return from core functions via isinstance guard.
- `nlp.py` - POST /nlp/command → LLMConfig object (not dict), explanation falls back to param_source
- `workflow.py` - POST /workflow/run (chained multi-step processing), GET /workflow/presets, POST /workflow/save, DELETE /workflow/delete
- `context.py` - POST /context/analyze (clip metadata → feature scores + guidance), GET /context/features (list all features)
- `plugins.py` - GET /plugins/list, GET /plugins/loaded, POST /plugins/install, POST /plugins/uninstall

**New in v1.9.0:**
- `context_awareness.py` (`core/`) — 35-feature relevance scoring based on clip metadata. classify_clip(metadata) → tags, score_features(tags) → scored list, get_guidance_message() → contextual help.
- `plugins.py` (`core/`) — Plugin loader. discover_plugins() scans ~/.opencut/plugins/, load_all_plugins(app) registers Flask Blueprints under /plugins/<name>/. Validates plugin.json manifests.
- `multicam_xml.py` (`core/`) — FCP XML generation from multicam cuts. generate_multicam_xml(cuts, source_files, ...) → {xml, output, cuts_count, duration}.
- `footage_index_db.py` (`core/`) — SQLite FTS5 footage index at ~/.opencut/footage_index.db. WAL mode, thread-local connections. index_file, search (FTS5 with LIKE fallback), needs_reindex (mtime comparison), get_stats, clear_index, remove_missing_files.
- `streaming.py` (`core/`) — NDJSON streaming utilities. ndjson_generator (batched), ndjson_item_generator (per-item), ndjson_progress_generator, make_ndjson_response.

### Persistence (`opencut/`)
- `job_store.py` (~200 lines) - SQLite job persistence at ~/.opencut/jobs.db. WAL mode, thread-local connections. save_job, get_job, list_jobs, mark_interrupted (on startup), cleanup_old_jobs (7-day TTL), get_job_stats. Jobs auto-persisted on terminal status via _persist_job in jobs.py.

### Data Files (`opencut/data/`)
- `workflow_presets.json` - 6 built-in workflow presets (Clean Interview, Podcast Polish, Social Clip, YouTube Upload, Documentary Rough, Studio Audio)
- `project_templates.json` - 6 project templates (YouTube, Shorts, TikTok/Reels, Podcast, Cinema, Broadcast)
- `example_plugins/timecode-watermark/` - Example plugin with plugin.json manifest and routes.py Blueprint

### Frontend (CEP Panel)
- `extension/com.opencut.panel/client/main.js` (~7730 lines) - Frontend controller. New systems: keyboard shortcut registry (DEFAULT_SHORTCUTS + localStorage persistence + matchesShortcut), lazy tab rendering (_tabRendered + initTabOnFirstVisit), cut review panel (showCutReview + formatTimecode), status bar (pollSystemStatus), i18n (loadLocale + t() + applyI18nToDOM), workflow preset loader (loadWorkflowPresets + server-side POST /workflow/run), project templates (initProjectTemplates + loadTemplateList), preset export/import (exportPresetFile + importPresetFile), quick action buttons (one-click workflows on Cut/Captions/Audio/Video tabs), toast reflow (_reflowToasts), enhanced error display (showAlert reads structured error.suggestion), job history loads from backend on init, interrupted jobs alert on first connect.
- `extension/com.opencut.panel/client/index.html` (~3210 lines) - Quick action bars on Cut/Captions/Audio/Video tabs. Cut review panel. Status bar with role="status" + aria-label. Keyboard shortcuts reference card. Project templates card. Preset export/import buttons. Zero inline styles. data-i18n attributes on ~25 elements.
- `extension/com.opencut.panel/client/style.css` (~4900 lines) - 10 complete themes (6 dark + 4 light: snowlight, latte, solarized, paper). Light-theme overrides for ~40 components including scrollbar (--scrollbar-thumb-color/--scrollbar-hover-color tokens), hovers (--light-hover/--light-border-accent), toast, command palette, job history. 4px spacing rhythm enforced (eliminated 10.5px/11.5px font sizes, normalized 10px/14px/18px/28px margins). 4 responsive breakpoints (800px wide, 480px compact, 440px mid-narrow, 380px very small). Complete interactive states for all button variants (btn-outline :disabled/:active, btn-ghost :active/:disabled, btn-text :disabled, range :focus-visible/:disabled, checkbox :focus-visible/:disabled). Focus-visible rings on 15 interactive elements. Quick action button styles. Shortcut reference card styles. Unified input styling (.text-input matches standard inputs).
- `extension/com.opencut.panel/host/index.jsx` (~2230 lines) - ExtendScript host. **New functions (lines 1315–2230):** ocGetSequenceInfo, ocAddSequenceMarkers, ocGetSequenceMarkers, ocApplySequenceCuts, ocApplyClipKeyframes, ocBatchRenameProjectItems, ocCreateSmartBins, ocAddNativeCaptionTrack, ocGetProjectBins, ocExportSequenceRange. Private helpers: _findByNodeId, _collectMediaItems, _collectBins. Markers use getFirstMarker/getNextMarker iterator (not indexed access).

### UXP Panel (Premiere Pro 25.6+)
- `extension/com.opencut.uxp/manifest.json` - UXP plugin manifest, targets PPRO minVersion 25.6, network domains 5679–5689, localFileSystem fullAccess
- `extension/com.opencut.uxp/index.html` (~771 lines) - 7-tab panel (Cut & Clean, Captions, Audio, Video, Timeline, Search, Deliverables)
- `extension/com.opencut.uxp/style.css` (~909 lines) - Dark theme with CSS variables matching CEP aesthetic
- `extension/com.opencut.uxp/main.js` (~1523 lines) - ES module controller. Internal modules: PProBridge (premierepro UXP lazy import), BackendClient (fetch + CSRF), JobPoller (async poll + cancel), UIController (tabs, toasts, progress). Auto port-scan 5679–5689 via detectBackend(). Loads LLM settings on startup via loadLlmSettings() → window._llmSettings.
- `extension/com.opencut.uxp/uxp-api-notes.md` - API status notes and CEP vs UXP comparison

### Build & Config
- `opencut_server.spec` - PyInstaller spec
- `OpenCut.iss` - Inno Setup installer script
- `install.py` - Cross-platform dev installer
- `Dockerfile` - Multi-stage build (Python 3.12 + FFmpeg + optional deps)
- `docker-compose.yml` - Service definition with GPU variant, named volume for ~/.opencut
- `.dockerignore` - Excludes .git, tests, extension, docs from Docker builds
- `extension/com.opencut.panel/package.json` - Vite dev dependency for future build pipeline
- `extension/com.opencut.panel/vite.config.js` - Vite bundler config (source maps, terser, predictable filenames for CEP)
- `extension/com.opencut.panel/tsconfig.json` - TypeScript config (allowJs, strict:false for incremental migration)
- `extension/com.opencut.panel/client/locales/en.json` - English locale (~150 i18n strings)
- `ROADMAP.md` - 7-phase implementation roadmap with priority matrix and success metrics

### Tests
- `tests/conftest.py` - Flask test client + CSRF fixtures + test media generators
- `tests/test_core.py` - Core module unit tests (silence, export, config)
- `tests/test_integration.py` - Route integration tests (health, CSRF, search, NLP, settings, timeline)
- `tests/test_new_modules.py` - v1.5 module tests (repeat_detect, chapter_gen, footage_search, deliverables, multicam, nlp_command, loudness_match, auto_zoom, color_match)
- `tests/test_route_smoke.py` (~950 lines) - 107+ smoke tests across all 11 blueprints + structured error tests + CSRF enforcement
- `tests/test_job_store.py` - SQLite persistence tests (save, get, list, filter, update, mark_interrupted, cleanup, stats, pagination)
- `tests/test_workflow.py` - Workflow engine tests (validation, presets, save/delete, built-in protection)
- `tests/test_context.py` - Context awareness tests (classify_clip, score_features, guidance, route integration)
- `tests/test_plugins.py` - Plugin discovery, manifest validation, route endpoint tests
- `tests/test_multicam_xml.py` - Multicam XML generation (basic, file output, NTSC, path-to-url)
- `tests/test_footage_index_db.py` - SQLite FTS5 index (index, search, upsert, reindex, stats, cleanup)
- `tests/test_streaming.py` - NDJSON streaming (batched, per-item, progress generators)
- `tests/test_batch_parallel.py` - Parallel batch processing (ThreadPoolExecutor, GPU/CPU workers, error isolation, cancellation)
- `tests/test_batch_executor.py` - BatchExecutor class tests (OperationSpec dispatch, combined progress, cancellation, partial failure)
- `tests/test_clip_notes_plugin.py` - Clip Notes plugin tests (CRUD notes, export text/CSV)

### Example Plugins (`opencut/data/example_plugins/`)
- `timecode-watermark/` - FFmpeg drawtext timecode overlay plugin
- `clip-notes/` - SQLite-backed per-clip timestamped notes with export (text/CSV)

## Architecture
- Backend runs as standalone process (exe or `python -m opencut.server`)
- Panel communicates via XHR to localhost:5679
- **Blueprint-based route organization**: 13 Blueprints (system, audio, captions, video, jobs, settings, timeline, search, deliverables, nlp, workflow, context, plugins) + dynamically loaded plugin blueprints
- **Shared modules**: security.py (CSRF + path validation), jobs.py (job state), helpers.py (utilities + `run_ffmpeg` + `ensure_package` + `get_video_info`), user_data.py (thread-safe file I/O)
- **CSRF protection**: Token generated at startup in security.py, returned via /health, sent as `X-OpenCut-Token` header on mutations. `@require_csrf` decorator applied to ALL POST routes.
- **Path validation**: `validate_path()` checks realpath, null bytes, `..` components, symlinks. `validate_filepath()` adds isfile check. Applied to ALL routes accepting file paths.
- **Input validation**: `safe_float()`/`safe_int()` with optional `min_val`/`max_val` range clamping and inf/nan rejection, `VALID_WHISPER_MODELS` frozenset for model name validation
- **Rate limiting**: `require_rate_limit(key)` decorator prevents concurrent expensive ops (e.g. model installs share `"model_install"` key)
- **Error taxonomy**: `OpenCutError` with typed codes (`MISSING_DEPENDENCY`, `GPU_OUT_OF_MEMORY`, etc.) — frontend `enhanceError()` adds actionable hints
- **Job safety**: `TooManyJobsError` (429), `_get_job_copy()`/`_list_jobs_copy()` for thread-safe reads, `_unregister_job_process()` for cleanup
- **Async job decorator**: `@async_job("type")` wraps routes in standard thread + try/catch + update pattern. Sets thread-local job_id for log correlation. Auto-persists terminal jobs to SQLite via _persist_job.
- Job system: `_new_job()` creates job, thread pool processes, SSE/polling for status. SQLite persistence at `~/.opencut/jobs.db` (WAL mode). `mark_interrupted()` on startup recovers jobs from previous crashes.
- **Workflow engine**: `core/workflow.py` chains steps sequentially via Flask test client, polls sub-jobs to completion, checks parent cancellation between steps. `routes/workflow.py` serves 6 built-in presets + user custom workflows.
- **Request size limit**: 100 MB `MAX_CONTENT_LENGTH` with 413 error handler
- **Crash logging**: 500 errors append to `~/.opencut/crash.log` with endpoint, method, traceback
- **Subprocess tracking**: Install routes register Popen processes for cancel support via `_register_job_process`
- **Model cache**: `/models/list` caches results for 5 min TTL, invalidated on model delete
- **PremiereBridge abstraction**: All jsx/csInterface calls go through `PremiereBridge` object for future UXP migration
- `subprocess` aliased as `_sp` in route files
- Custom dropdown system replaces native `<select>` elements in CEP
- **Sidebar navigation**: 52px icon-only left sidebar (CapCut-style), CSS tooltips on hover, active tab = left accent bar
- **Layout**: `.app` = flex row → `aside.sidebar` (52px) + `.content-area` (flex:1 column → `.content-header` + banners + `main.main` + `.content-footer`)
- 8 main tabs (CEP): Cut, Captions, Audio, Video, Export, Timeline, NLP, Settings
- 7 main tabs (UXP): Cut & Clean, Captions, Audio, Video, Timeline, Search, Deliverables
- **Timeline write-back**: ExtendScript functions ocApplySequenceCuts, ocAddSequenceMarkers, ocApplyClipKeyframes, ocBatchRenameProjectItems, ocCreateSmartBins, ocAddNativeCaptionTrack write directly to the active Premiere sequence. Python routes return structured data; frontend JS calls evalScript() to apply it. Never call ExtendScript from Python — always via frontend bridge.
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
- Current: **v1.9.0**
- All version strings: `pyproject.toml`, `__init__.py`, `CSXS/manifest.xml` (ExtensionBundleVersion + Version), `com.opencut.uxp/manifest.json`, `com.opencut.uxp/main.js` (VERSION const), `index.html` version display, README badge
- Use `python scripts/sync_version.py --set X.Y.Z` to update all 18 targets at once (including UXP files)

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
- Dependency checks live in `opencut/checks.py` — don't duplicate `check_X_available()` in route files. New checks: check_color_match_available (cv2+numpy), check_auto_zoom_available (cv2), check_loudness_match_available (ffmpeg in PATH), check_footage_search_available (always True, stdlib)
- **Consolidated helpers** — `run_ffmpeg()`, `ensure_package()`, `get_video_info()` live in `opencut/helpers.py`. All core modules import from there. Never define local `_run_ffmpeg`/`_ensure_package`/`_get_video_info` copies.
- `ensure_package()` routes through `safe_pip_install()` from security.py — never bypass this with raw `subprocess.run(["pip", ...])` in core modules
- `get_video_info()` includes format-duration fallback for containers where stream-level duration is unavailable
- Deferred temp cleanup: `_schedule_temp_cleanup(path)` retries with exponential backoff on Windows
- **LLMConfig is a dataclass, not a dict** — routes must instantiate `LLMConfig(provider=..., model=..., api_key=...)`, never pass `{"provider": ...}` dicts to core functions. Import pattern: try relative `..core.llm`, fallback absolute `opencut.core.llm`, guard with `if LLMConfig is not None`.
- **Deliverables return dict, not string** — `generate_vfx_sheet/adr_list/music_cue_sheet/asset_list()` return `{"output": path, "rows": N}`. Routes use `isinstance(result, dict)` guard with `.get("output", fallback)`.
- **Multicam result is a dict** — `generate_multicam_cuts()` returns `{"cuts": [...], "total_cuts": N, "speaker_to_track": {...}}`. Unpack with `result.get("cuts", [])`, not direct iteration.
- **on_progress callbacks** — Core modules call `on_progress(int_pct)` with **1 argument**. All route `_on_progress(pct, msg="")` closures must default `msg` to `""`. Never define `_on_progress(pct, msg)` without the default — it will crash when core modules call with 1 arg.
- **Auto-zoom FFmpeg resolution** — zoompan filter uses dynamic `s={src_w}x{src_h}` from probe(), not hardcoded `s=hd1080`. Always probe source before building the filter string.
- **footage_search file locking** — index writes use cross-platform `_FileLock` + `os.replace()` atomic swap. Never write the index JSON file directly; always use `save_index()`.
- **LLM API key security** — `/settings/llm` GET masks key as `***{last4}`. POST preserves stored key if client echoes back a masked value (starts with `***`). Never log LLMConfig objects or api_key values.
- **User preferences** — New setting files in `~/.opencut/`: llm_settings.json, footage_index_config.json, loudness_settings.json, color_profiles.json, multicam_config.json, auto_zoom_presets.json, chapter_defaults.json. Always use `load_X()` / `save_X()` wrappers from user_data.py.
- **MCP server** — 8 new tools added (opencut_repeat_detect, opencut_chapters, opencut_footage_search, opencut_index_footage, opencut_color_match, opencut_loudness_match, opencut_auto_zoom, opencut_multicam_cuts). Filepath validation in handle_tool_call covers "file", "source", "reference" keys in addition to existing "filepath".
- **Command palette** — 9 new entries at end of `_commandIndex` array (lines ~5162–5171). Tab values must match `data-nav` attributes in HTML. Sub-tab IDs must match `data-sub` attributes in HTML — verify before adding entries.
- **Route param name convention** — CEP panel sends `filepath` (established convention). New routes MUST accept `filepath` as the primary key (use `data.get("filepath", data.get("file", ""))` fallback pattern for compat). Never use only `"file"` — it breaks the frontend.
- **UXP endpoint convention** — UXP handlers MUST use full Blueprint-prefixed paths (`/audio/denoise`, `/video/color-match`, `/captions/chapters`, `/timeline/batch-rename`, `/search/footage`, `/nlp/command`, `/deliverables/vfx-sheet`). Never use bare paths like `/denoise` or `/chapters`.
- **ExtendScript data format** — `ocAddSequenceMarkers` expects a JSON array of `{time, name, type}` objects, NOT a wrapper object. `ocBatchRenameProjectItems` reads `{nodeId, newName}`. `ocCreateSmartBins` reads `{binName, rule, field, value}`. Always verify field names match between JS and JSX.
- **Queue allowlist** — New async routes MUST be added to `_ALLOWED_QUEUE_ENDPOINTS` in jobs_routes.py, or queue operations silently fail with "Endpoint not queueable".
- **`run_ffmpeg()` returns `str`, not CompletedProcess** — It raises `RuntimeError` on non-zero exit and returns stderr as a string. Never access `.returncode` or `.stderr` on the return value. If you need the `subprocess.CompletedProcess` fallback path, use `subprocess.run()` directly.
- **ExtendScript MarkerCollection** — Use `getFirstMarker()`/`getNextMarker()` iterator, NOT `markers[i]` indexed access (unreliable across Premiere versions). Pattern: `var m = markers.getFirstMarker(); while (m) { ... try { m = markers.getNextMarker(m); } catch(e) { m = null; } }`
- **CSS `hidden` class vs `style.display`** — Setting `style.display=""` does NOT override a `display:none` from a CSS class. Use `classList.toggle("hidden", ...)` or `classList.remove("hidden")` instead.
- **Workflow preset steps** — Each step must be a self-contained `{endpoint, payload, label}` object. Do NOT rely on `pendingTranslate`/`pendingBurnin` flags for chaining — those only work for manual button clicks, not workflow presets.
- **Duplicate HTML `class` attributes** — HTML parser silently discards the second `class=` attribute. Always merge into a single `class="cls1 cls2"`. Grep with `class=".+" class="` to detect.
- **pip `--target` fallback** — `safe_pip_install()` has 3 strategies: normal → `--user` → `--target ~/.opencut/packages`. The `--target` dir is created at install time and added to `sys.path` both in security.py (immediately) and server.py (on startup). Packages installed via `--target` are importable without restart.
- **`cancelJob()` order** — Close SSE/poll streams BEFORE nulling `currentJob` to prevent in-flight events from triggering `onJobDone` after cancel.
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

## v1.5.0 Features Added
- **Repeated Take Detection** (`core/repeat_detect.py`) — Jaccard similarity sliding-window; detect_repeated_takes() returns {repeats, clean_ranges}; merge_repeat_ranges() for overlapping ranges
- **YouTube Chapter Generation** (`core/chapter_gen.py`) — LLM-powered topic boundary detection; heuristic fallback on long pauses; outputs description_block ready to paste
- **Footage Search** (`core/footage_search.py`) — BM25-lite JSON index; cross-platform atomic writes; index_file, search_footage, clear_index, get_index_stats
- **Post-Production Deliverables** (`core/deliverables.py`) — VFX sheets, ADR lists, music cue sheets, asset lists as CSV; all return {output, rows}
- **Color Match** (`core/color_match.py`) — YCbCr histogram matching with strength blending; on_progress callback; cv2/numpy optional
- **Multicam Auto-Switching** (`core/multicam.py`) — diarization segments → cut list; auto_assign_speakers; merge_diarization_segments; pure Python
- **Auto Zoom Keyframes** (`core/auto_zoom.py`) — Haar cascade face detection; 4 easing modes; on_progress callback; center-crop fallback; _ease() boundary-correct
- **Loudness Match** (`core/loudness_match.py`) — FFmpeg loudnorm two-pass; measure_loudness parses stderr JSON; batch_loudness_match with on_progress
- **NLP Command Parser** (`core/nlp_command.py`) — 19-entry COMMAND_MAP; keyword + LLM dispatch; extract_params_from_text
- **Timeline Write-Back** (ExtendScript `index.jsx` lines 1315–2177) — 10 new ocXxx functions for direct sequence manipulation; ES3 compliant; all return JSON strings
- **4 new Flask Blueprints** — timeline, search, deliverables, nlp
- **New routes** — /audio/beat-markers, /audio/loudness-match, /captions/chapters, /captions/repeat-detect, /video/color-match, /video/auto-zoom, /video/multicam-cuts + full timeline/search/deliverables/nlp suites
- **CEP panel** — 2 new tabs (Timeline, NLP), 12 new feature cards, LLM settings + Audio/Zoom Defaults in Settings tab, loadLlmSettings() on startup
- **UXP Panel** (`extension/com.opencut.uxp/`) — full parallel implementation for Premiere Pro 25.6+; auto port-scan; PProBridge/BackendClient/JobPoller/UIController modules; 1523-line main.js
- **8 new MCP tools** — repeat_detect, chapters, footage_search, index_footage, color_match, loudness_match, auto_zoom, multicam_cuts
- **8 new CLI commands** — chapters, repeat-detect, search index/query, color-match, loudness-match, auto-zoom, deliverables, nlp
- **9 command palette entries** — all new features searchable via Ctrl+K
- **7 user preference groups** — LLM, footage index, loudness, color profiles, multicam, auto-zoom, chapters (load_X/save_X in user_data.py)

## v2.0 Features Added

### Backend Infrastructure
- **Structured error taxonomy** (`errors.py`) — 13 error types with codes, user messages, and recovery suggestions. `safe_error(exc)` auto-classifies exceptions (MemoryError→GPU_OOM, TimeoutError→OPERATION_TIMEOUT, PermissionError→PERMISSION_DENIED, ImportError→MISSING_DEPENDENCY). All `_safe_error()` calls across all routes now return structured `{error, code, suggestion}` JSON.
- **SQLite job persistence** (`job_store.py`) — Jobs at `~/.opencut/jobs.db` (WAL mode). Auto-persist on terminal status. `mark_interrupted()` on startup. 7-day cleanup. Endpoints: GET /jobs/history, /jobs/stats, /jobs/interrupted.
- **Structured logging** — Log format `[job_id]` correlation via thread-local. GET /logs/tail with level/job_id filtering.
- **Workflow engine** (`core/workflow.py` + `routes/workflow.py`) — Server-side sequential step execution with output chaining, cancellation checks between steps, sub-job polling. 6 built-in presets (Clean Interview, Podcast Polish, Social Clip, YouTube Upload, Documentary Rough, Studio Audio). POST /workflow/run, GET /workflow/presets, POST /workflow/save, DELETE /workflow/delete.
- **Pip install fix** — `safe_pip_install()` reordered: `--target ~/.opencut/packages` is strategy #1 (always writable). Errno 13 / WinError 5 detection skips to next strategy immediately. Post-install verification. `~/.opencut/packages` added to `sys.path` with priority at startup.
- **Project templates** — 6 built-in templates (YouTube, Shorts, TikTok/Reels, Podcast, Cinema, Broadcast). GET /templates/list, POST /templates/save, POST /templates/apply.
- **Health monitoring** — GET /system/status (CPU, RAM, GPU VRAM with 30s cache, disk, jobs, uptime). `psutil` optional with graceful degradation.
- **Route migration** — 5 audio routes migrated from raw `threading.Thread` to `@async_job` decorator (denoise, isolate, normalize, beats, deepfilter).
- **Docker** — `Dockerfile` (multi-stage), `docker-compose.yml` (GPU variant), `.dockerignore`.
- **Test coverage** — 107+ smoke tests covering all 11 blueprints, structured error classification, CSRF enforcement, job store persistence.

### Frontend Features
- **Quick Actions** — One-click workflow buttons at top of Cut (Clean Up, YouTube Ready, Podcast Polish), Captions (Auto Subtitle, Translate), Audio (Studio Polish, Quick Denoise), and Video (Auto Color, Social Reframe) tabs. Wired to server-side POST /workflow/run.
- **Cut review panel** — After silence/filler/auto-edit/highlights detection, shows checkbox table with timecodes. Select All / Deselect All / Apply Selected. Users review and selectively apply cuts before committing.
- **Keyboard shortcuts** — 8 default bindings (Ctrl+Shift+S/C/N/D/E/W, Ctrl+K, Escape). Registry in localStorage. Dynamic matching via matchesShortcut(). Shortcut reference card in Settings tab. Title hints on primary buttons.
- **Lazy tab rendering** — 5 heavy init functions deferred until first tab visit (initCaptionNewFeatures, initAudioNewFeatures, initTimelineFeatures, initNlpFeatures, initDeliverablesFeatures).
- **Status bar** — Persistent footer: green/yellow/red dot, uptime, CPU%, RAM, GPU VRAM, job count. 5-second polling with 60-retry cap. `role="status"` + `aria-label` for accessibility.
- **i18n framework** — `locales/en.json` (~150 strings). `t(key, fallback)` function. English loaded as base, target locale merged on top. `data-i18n` attributes on ~25 elements. Language persisted in localStorage.
- **Project templates UI** — Settings tab card with dropdown (built-in + custom), description, Apply Template button. Applies export/audio/caption settings to panel controls.
- **Preset export/import** — Export as `.opencut-preset` JSON file, import with validation. Buttons in Settings tab.
- **Workflow UI upgrade** — Preset dropdown loads from GET /workflow/presets (built-in + custom, with step counts). Run uses server-side POST /workflow/run. Custom save/delete use new /workflow/* routes. Workflow completion toast shows step count and output filename.
- **Job history persistence** — initJobHistory() loads last 20 jobs from GET /jobs/history. Interrupted jobs alert on first connect.
- **Toast reflow** — `_reflowToasts()` recalculates positions after removal to eliminate visual gaps.
- **Enhanced errors** — `showAlert()` and `enhanceError()` read structured `suggestion` field from server. Install errors (Demucs, WhisperX) show recovery guidance.
- **Command palette** — 4 new entries (Workflow Presets, Project Templates, Keyboard Shortcuts, Job History).

### CSS/Theme Improvements
- **4 light themes** — Snowlight (clean white/blue), Catppuccin Latte (warm cream), Solarized Light, Paper (monochrome). `color-scheme: light` + scrollbar tokens + ~40 component overrides.
- **Light-theme scrollbar** — `--scrollbar-thumb-color` / `--scrollbar-hover-color` tokens. Dark themes: rgba(255,255,255,...), light themes: rgba(0,0,0,...). Visible on all 10 themes.
- **20+ light-theme hover overrides** — nav-tab, header-btn, job-history-item, model-item, fav-chip, batch-file-item, merge-file-item, recent-clip-item, context-menu-item, custom-dropdown-item, btn-ghost, btn-outline. Uses `--light-hover` / `--light-border-accent` tokens.
- **Typography cleanup** — Eliminated 10.5px→11px and 11.5px→12px fractional sizes. Clean integer scale: 9/10/11/12/13/14/16/17/18/20/22px.
- **4px spacing rhythm** — Normalized card-header (14px→12px), alert-banner (18px→16px), hints (14px→12px), wizard (28px→24px), slider-row (14px→12px), checkbox-row (10px→8px), margins (10px→8px, 14px→12px).
- **4 responsive breakpoints** — 800px (wide: 4-col stat grid, wider cards), 480px (compact), 440px (mid-narrow: stacked buttons, single-col), 380px (tiny sidebar).
- **Complete interactive states** — btn-outline :disabled/:active, btn-ghost :active/:disabled, btn-text :disabled, range :focus-visible/:disabled, checkbox :focus-visible/:disabled. Focus-visible rings on 15 elements. Tokenized focus rings (var(--cyan-subtle)).
- **Input consistency** — .text-input unified with standard inputs (9px 12px padding). Consistent focus rings across all input types.

### Gotchas (v2.0 additions)
- **Workflow cancellation** — `run_workflow()` checks `_is_cancelled(parent_job_id)` between steps. Cancelled sub-jobs also stop the workflow. Pass `parent_job_id` when calling from routes.
- **Job persistence** — `_update_job()` auto-calls `_persist_job()` for terminal statuses (complete/error/cancelled). `_cleanup_old_jobs()` also persists stuck-job error status. Background thread avoids I/O under lock.
- **`get_interrupted_jobs()`** queries `status='interrupted'` (not 'running') — `mark_interrupted()` changes status on startup.
- **Workflow step output chaining** — `_extract_output_path(resp_data)` checks `output`, `output_file`, `filepath`, `result.output` keys. If no output found, reuses current_input for next step.
- **Quick action buttons** — Added to `_clipButtons` array so `updateButtons()` enables/disables them with connection + file selection state. Click handlers call `_quickWorkflow(name)` which looks up preset from `_workflowPresets` loaded from backend.
- **Scrollbar tokens** — Use `var(--scrollbar-thumb-color)` and `var(--scrollbar-hover-color)` for scrollbar styling. Light themes define dark-alpha versions. Never use hardcoded rgba for scrollbar colors.
- **Light-theme hover pattern** — Use `var(--light-hover)` / `var(--light-border-accent)` tokens (defined in light theme shared block). For new components, add `:root[data-theme="snowlight/latte/solarized/paper"] .component:hover { background: var(--light-hover); }`.
- **safe_int for query params** — All GET endpoints accepting numeric query params (limit, offset, lines) must use `safe_int()` from security.py, not bare `int()`. Bare int() causes 500 on non-numeric input.
- **14 new settings routes** — GET+POST pairs for llm, loudness-target, auto-zoom, chapters, multicam, footage-index
- **6 new social presets** — Snapchat Story, Facebook Reel, Facebook Post, YouTube Long Form, Pinterest Video, Podcast MP3 (13 total)
- **84 new tests** (`tests/test_new_modules.py`, 1064 lines) — 9 test classes, all mocked correctly against actual function signatures
- **4 new dependency checks** in checks.py; 6 new entries in /system/dependencies dashboard
- **Dynamic auto-zoom resolution** — zoompan filter uses probed source dimensions, not hardcoded hd1080

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
- [x] **Motion graphics**: Added `render_remotion_title()` — Remotion CLI integration via npx with fallback to FFmpeg drawtext. `check_remotion_available()` for Node.js detection
- [x] **Video denoising**: Added `BasicVSR++` as `"basicvsr"` method in `/video/ai/denoise` — GPU temporal propagation, chunk-based processing, strength-blended output
- [x] **Scene detection**: Added `PySceneDetect` as `"pyscenedetect"` method in `/video/scenes` — heuristic, fast, ContentDetector
- [x] **Neural LUT blending**: Added `blend_luts()` — linearly interpolate between any two .cube LUTs with a slider. New `/video/lut/blend` route
- [x] **Translation**: Added `SeamlessM4T v2` via `translate_text_seamless()` — 20% BLEU improvement over NLLB. `backend` param in `/captions/translate`
- [x] **Caption NLP emphasis**: Added `detect_keywords_nlp()` — TF-IDF-like frequency analysis + POS heuristics for auto-emphasis. Integrated into `get_action_word_indices()`

### Phase 4 — Architecture (Long-term)
- [ ] **UXP migration** — CEP deprecated, removal late 2026. PremiereBridge abstraction already in place. Test with UXP samples.
- [x] **MCP server exposure** — Added `opencut/mcp_server.py` — stdio JSON-RPC MCP server with 10 tools (transcribe, silence, export, highlights, separate, TTS, style, face enhance, music, job status). Run via `python -m opencut.mcp_server`.
- [x] **Vision-augmented highlights** — Added `extract_highlights_with_vision()` + `extract_frames_for_vision()`. Samples keyframes at intervals, sends alongside transcript to LLM. `use_vision` param in `/video/highlights`.
- [x] **Transcription slicing** — Added `_transcript_cache` with FIFO eviction (max 20). `cache_transcript()` / `get_cached_transcript()` in captions routes. Keyed by filepath+mtime. `force_retranscribe` param to bypass.

### Keep As-Is (Already Best-in-Class)
- faster-whisper (transcription engine), WhisperX (alignment), Real-ESRGAN (upscaling), InsightFace (face swap), auto-editor (auto-editing), pedalboard (audio effects), pyannote.audio (diarization — update to v4.0.4)

## v1.5.1 Batch 28 Bug Fixes
- **UXP panel 5 P0 bugs** — entire panel was non-functional: CSRF header X-CSRF-Token→X-OpenCut-Token, fetchCsrf /csrf→/health csrf_token field, job poll /jobs/{id}→/status/{id}, cancel DELETE→POST /cancel/{id}, LLM settings BackendClient.fetch()→.get()
- **chapter_gen JSON regex** — non-greedy `\[.*?\]` truncated multi-element LLM arrays; changed to greedy `\[\s*\{[\s\S]*\}\s*\]` (same class as highlights batch 5)
- **nlp_command** — greedy JSON regex for nested params, route allowlist on LLM output, word-boundary keyword matching (`\b` regex prevents "um" matching "volume"), float coercion safety on confidence, removed unused `string` import
- **color_match** — VideoWriter.isOpened() check, on_progress(100) at completion, use consolidated run_ffmpeg helper, fix YCrCb channel labels (OpenCV order is Y,Cr,Cb not Y,Cb,Cr)
- **auto_zoom** — VideoCapture try/finally, easing no-op fix (_ease(1.0,mode) always=1.0 → proper fractional blending), negative fps guard
- **loudness_match** — use consolidated helper, pass 1 returncode check, validate measured values as floats before FFmpeg filter interpolation, clamp target_lufs [-70,0] and true_peak [-10,0]
- **footage_search** — Windows LK_NBLCK→LK_LOCK (blocking), write 1 byte before lock, explicit msvcrt.LK_UNLCK before close
- **multicam** — float coercion on start/end via .get(), min_cut_duration clamp
- **deliverables** — int(fps)→int(round(fps)) for 29.97fps timecodes
- **checks.py** — all check functions return bool consistently (new ones returned Tuple, making `if check_X():` always truthy)
- **user_data.py** — removed lock eviction that could delete in-use locks (re-introduced bug from batch 11/25)
- **CLI** — chapters uses dict.get() not attribute access, repeat-detect unpacks dict result, search index/query import from footage_search (was non-existent .core.search)
- **nlp route** — guard parse_command() returning None, LLM provider allowlist
- **timeline route** — safe_float instead of float() on SRT segments
- **ExtendScript** — smart bins "video" type matches AV files (hasVid, not hasVid&&!hasAud), temp SRT cleanup after caption import

## v1.5.1 Batch 29 Bug Fixes
- **video.py auto-zoom crash** — `probe.get("width")` on non-dict probe object; replaced with `get_video_info()` from helpers.py. `import subprocess as _sp2` → use module-level `_sp`. Keyframe result dict unpacking fixed (`keyframes.get("keyframes", [])` not raw list check).
- **audio.py loudness-match** — `batch_loudness_match()` returns a list, not dict; `result.get("outputs")` always returned empty. Fixed to handle list directly.
- **settings.py 6 POST routes** — `request.get_json(silent=True)` → `force=True` (malformed JSON silently became {}, returning success without changes)
- **captions.py chapters** — missing LLM provider allowlist, missing segments list type+size validation (max 50000)
- **video.py multicam** — missing segments list type+size validation, diarization file 50 MB size cap before JSON.load
- **MCP server** — `files` array items validated for path traversal, UNC path (`\\` and `//`) rejection in filepath validation

## v1.5.2 Batch 30 Bug Fixes
- **CEP panel 7 P0 param mismatches** — beat-markers/repeat-detect/chapters/NLP sent `filepath` but new routes read `file`; routes now accept both via `data.get("filepath", data.get("file", ""))` fallback. Loudness-match/search-index sent `filepaths` but routes read `files`; frontend fixed. Footage search sent `max_results` but route reads `top_k`; frontend fixed. Loudness result table read `r.clips` but API returns `outputs`; fixed.
- **on_progress crash (48 callbacks)** — all `_on_progress(pct, msg)` closures in video.py/audio.py/captions.py took 2 required args, but core modules (color_match, auto_zoom, loudness_match) call `on_progress(int)` with 1 arg → TypeError. Fixed `msg=""` default on all 48 callbacks.
- **Queue allowlist** — 9 new v1.5.0 routes missing from `_ALLOWED_QUEUE_ENDPOINTS` (beat-markers, loudness-match, chapters, repeat-detect, color-match, auto-zoom, multicam-cuts, export-from-markers, search/index)
- **Command palette** — "Color Match" pointed to non-existent `tl-colormatch` → `vid-color`; "Auto Zoom" pointed to `tl-autozoom` → `vid-effects`
- **Dependencies dashboard** — multicam missing from `/system/dependencies` response

## v1.5.3 Batch 31 Bug Fixes
- **UXP panel 18 P0 — every feature was broken** — all handlers posted to wrong endpoints missing Blueprint prefixes (/silence, /denoise, /chapters, /color-match → /silence, /audio/denoise, /captions/chapters, /video/color-match, etc.) and sent wrong param names (file_path→filepath, sensitivity→subdivisions, filler_words→custom_words, caption_style→format, etc.). All 18+ handlers fixed.
- **CEP marker format** — `addBeatMarkersToSequence` and `addChaptersAsMarkers` sent `{times: [...]}` wrapper object but `ocAddSequenceMarkers` expects `[{time, name, type}]` array. Fixed to build proper marker array.
- **CEP rename fields** — sent `{id, old_name, new_name}` but `ocBatchRenameProjectItems` reads `{nodeId, newName}`. Fixed field mapping.
- **CEP smart bin fields** — sent `{bin_name, rule_type}` but `ocCreateSmartBins` reads `{binName, rule}`. Added field transformation before ExtendScript call.
- **CEP deliverables result** — read `data.output_path` but route returns `data.output`. Fixed with fallback.
- **ExtendScript export range** — `ocExportSequenceRange` setInPoint/setOutPoint failures now abort with error instead of silently exporting wrong range.

## v1.5.4 Batch 32 Bug Fixes
- **CEP repeat-detect result** — read `r.cuts`/`r.ranges` but route returns `r.repeats`. Fixed with fallback chain.
- **CEP chapters api_key** — sent `llm_api_key` but route reads `api_key` — OpenAI/Anthropic key was silently lost.
- **CEP export-from-markers result** — read `r.exported` but route returns `r.count`.
- **CEP CSRF header in raw fetch()** — 4 `fetch()` calls used `X-CSRF-Token` instead of `X-OpenCut-Token` — settings saves were 403 failures.
- **UXP result field mismatches** — color-match/auto-zoom/denoise result handlers read `result.output_path` but routes return `result.output`.
- **UXP chapters timecodes** — read `c.start` (undefined) instead of `c.seconds` for display.
- **UXP batch-rename/smart-bins/export-from-markers** — sent wrong param schemas for route validation.
- **UXP deliverables** — empty `{}` sequence_data was falsy in Python, causing 400 errors.
- **Route fallbacks** — `export-from-markers` accepts `filepath` for `input_file`, `smart-bins` accepts both `binName`/`bin_name` and `rule`/`rule_type`.

## v1.5.5 Batch 33 Fixes
- **Multicam route** — now accepts `filepath` param and auto-transcribes with Whisper to generate diarization segments (CEP multicam was completely broken — sent filepath but route only accepted diarization_file/segments)
- **UXP Full Report** — replaced non-existent `/project-report` endpoint with sequential calls to all 4 deliverables endpoints, with UXP PProBridge sequence info
- **Search/Index folder support** — route now accepts `folder` param alongside `files`, auto-scans for media files

## v1.5.6 Batch 34 Fixes
- **SECURITY: UNC path blocking** — `validate_path()` now blocks `\\server\share` and `//server/share` paths (SSRF/NTLM hash leak prevention). Checked pre and post `normpath()`.
- **SECURITY: Registry command injection** — ExtendScript `startOpenCutBackend()` sanitizes registry-sourced `exePath` against batch metacharacters
- **Stuck jobs auto-expire** — "running" jobs stuck >2 hours marked as error (prevents permanent TooManyJobsError)
- **Zombie process reaping** — `proc.wait(5)` after `proc.kill()` in `_kill_job_process()`
- **async_job thread tracking** — decorator stores thread handle in job dict (`_thread` was always None)
- **ExtendScript projectItem safety** — in/out point reset guaranteed even if insert loop throws
- **Test fixes** — 5 broken tests fixed (mock paths updated for consolidated helpers, tempdir context scope)

## v1.5.6 Batch 35 Bug Fixes (Subtle Bug Hunt)
- **color_match.py AttributeError CRASH** — `run_ffmpeg()` returns `str` (stderr) but code accessed `.returncode`/`.stderr` (CompletedProcess attributes). Color matching was **completely broken**. Fixed: `run_ffmpeg` raises on failure (no return check needed), fallback `subprocess.run` checks returncode separately.
- **auto_edit.py temp dir destroys XML** — `finally: shutil.rmtree(temp_dir)` deleted the XML result file before caller could use it. Fixed: copy XML to persistent temp dir before cleanup.
- **ExtendScript marker iteration** — `seqMarkers[i]` indexed access is unreliable on Premiere's MarkerCollection. Switched both `ocGetSequenceInfo` and `ocGetSequenceMarkers` to `getFirstMarker()`/`getNextMarker()` iterator pattern.
- **silenceSpeedGroup permanently hidden** — Element has CSS class `hidden` (`display:none`) but JS used `style.display=""` to show it — inline style removal doesn't override class rule. Speed-up slider was **always invisible**. Fixed: `classList.toggle("hidden", ...)`.
- **Translate Pipeline workflow** — Only transcribed, never translated. Relied on `pendingTranslate` flag that workflow runner never set. Added explicit `/captions/translate` step.
- **Social Ready workflow** — Dropdown promised "Burn-in Captions" but workflow had no captions step. Added `/captions/burn-in` step.
- **Styled captions checkboxes dead** — `captionWordHighlight` and `captionAutoEmoji` checkbox values never included in API payload. Added both to `startJob` payload.
- **Workflow denoise method mismatch** — `clean_audio` and `pro_video` presets hardcoded `method: "rnnoise"` but UI only offers `afftdn`/`bandpass`. Changed to `afftdn`.
- **Cancel/complete race condition** — SSE could deliver "complete" event after `cancelJob()` nulled `currentJob` but before it closed the stream, causing unwanted auto-import. Fixed: close SSE/poll **before** nulling `currentJob`.
- **NLP auto-execute stale selectedPath** — `selectedPath` read at callback time, not command time. If user switches clips during NLP API call, job runs on wrong clip. Fixed: snapshot at command invocation.
- **ocApplySequenceCuts NaN sort corruption** — Sort comparator `b.start - a.start` ran before Number coercion; string/null/undefined values produce NaN, corrupting sort order and causing wrong clips to be deleted. Fixed: pre-coerce all cut times before sort.
- **ocExportSequenceRange persistent side effect** — `setInPoint`/`setOutPoint` persist on the sequence after export, constraining all future playback and exports. Fixed: save/restore original in/out points.
- **ocAddNativeCaptionTrack inflated count** — Returned `segments.length` (original array size) instead of `srtIndex` (actual written count after skipping invalid segments).
- **Keyframe time coercion** — `kf.time || 0` used without `Number()` coercion; string time values from JSON could cause wrong keyframe placement. Added `Number()`.
- **export_video no progress** — Missing `-progress pipe:1` flag meant FFmpeg wrote progress to stderr (not stdout), so progress parsing never matched. Progress bar was stuck at 20%.
- **export_video no timeout** — `proc.wait()` had no timeout; hung FFmpeg process blocked the thread forever. Added scaled timeout with kill.
- **export_video partial file leak** — Failed FFmpeg runs left corrupt partial output files on disk. Added cleanup on non-zero exit.
- **10 duplicate class attributes in HTML** — 10 elements had two `class=` attributes; HTML parser silently ignores the second, losing spacing utilities (mt-xs, mt-sm, mb-sm, mt-md). All merged into single attributes.
- **pip install permission denied** — `safe_pip_install()` failed on Windows when both normal and `--user` installs hit Errno 13 (Microsoft Store Python, OneDrive-synced user dirs, restrictive ACLs). Added `--target ~/.opencut/packages` as third fallback strategy. server.py adds `~/.opencut/packages` to `sys.path` at startup.

## v1.9.0 Features Added

### Backend Infrastructure
- **Contextual Awareness** (`core/context_awareness.py` + `routes/context.py`) — 35-feature relevance scoring based on clip metadata. `classify_clip()` produces tags (talking_head, audio_only, long_duration, etc.), `score_features()` scores 0-100, `get_guidance_message()` generates contextual help text. POST /context/analyze, GET /context/features.
- **Plugin System** (`core/plugins.py` + `routes/plugins.py`) — Plugin loader scans `~/.opencut/plugins/` for directories with `plugin.json` manifests. Validates name/version/description, loads Flask Blueprints under `/plugins/<name>/`. Auto-loaded on server startup via `load_all_plugins(app)`. GET /plugins/list, /plugins/loaded, POST /plugins/install, /plugins/uninstall.
- **Multicam XML Export** (`core/multicam_xml.py`) — FCP XML generation for Premiere Pro import. `generate_multicam_xml(cuts, source_files, ...)` produces `<xmeml>` with video/audio tracks, NTSC detection, file URL references. POST /video/multicam-xml. CLI multicam command now exports XML.
- **SQLite FTS5 Footage Index** (`core/footage_index_db.py`) — Replaces JSON index with SQLite + FTS5 at `~/.opencut/footage_index.db`. WAL mode, thread-local connections, triggers for FTS sync. `needs_reindex()` compares mtime for incremental updates. POST /search/auto-index, /search/db-search, GET /search/db-stats, POST /search/cleanup.
- **NDJSON Response Streaming** (`core/streaming.py`) — Batched and per-item streaming for large result sets. `ndjson_generator()` batches items, `make_ndjson_response()` wraps in Flask Response with correct headers. GET /jobs/stream-result/<job_id> streams completed job arrays.

### Security & Reliability Hardening
- **TooManyJobsError handling** — All 96 manual `_new_job()` calls across video.py, audio.py, captions.py now catch `TooManyJobsError` and return 429 instead of 500.
- **AI GPU rate limiting** — `rate_limit("ai_gpu")` on 6 GPU-heavy routes (upscale, rembg, shorts pipeline, 3 music AI routes) prevents concurrent GPU OOM.
- **Settings import validation** — Workflow steps validated (require endpoint + label) before import.
- **Input bounds** — preview_frame width (32-3840), detection_skip (1-30).
- **Secure tempfile** — preview_frame uses `tempfile.mkstemp()` not predictable paths.

### Gotchas (v1.9.0 additions)
- **Plugin route prefix** — Plugin Blueprints are registered under `/plugins/<plugin_name>/`. Plugin routes.py defines `plugin_bp` (not any other name). The Blueprint's internal URL rules are relative, so a plugin route `@plugin_bp.route("/apply")` becomes `/plugins/timecode-watermark/apply`.
- **Plugin loading order** — Plugins load AFTER all built-in blueprints in `create_app()`. Plugin route collisions with built-in routes will silently fail — Flask uses the first registered route.
- **FTS5 query syntax** — `footage_index_db.search()` wraps words in quotes for safety. Raw FTS5 operators (AND, OR, NOT, NEAR) in user queries may cause OperationalError — the function falls back to LIKE search automatically.
- **`needs_reindex()` uses mtime** — If a file is re-transcribed without changing on disk, `needs_reindex()` returns False. Force re-index by calling `index_file()` directly.
- **NDJSON response mime type** — `application/x-ndjson`. Frontend must parse line-by-line, not `JSON.parse()` the whole body. Each line is a valid JSON object.
- **Context awareness `score_features` capability check** — The scoring checks for "audio" and "video" capabilities by inferring from tags (e.g., `audio_only` implies has audio but no video). Tags like `talking_head` imply both audio and video are present.
- **`rate_limit("ai_gpu")` scope** — Shared across all AI routes. If upscale is running, music generation returns 429. This is intentional to prevent GPU memory conflicts.
- **13 Blueprints** — Route registration list in `__init__.py` is now: system, audio, captions, video, jobs, settings, timeline, search, deliverables, nlp, workflow, context, plugins. Plus dynamic plugin blueprints.
