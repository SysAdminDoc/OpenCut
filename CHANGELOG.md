# Changelog

## [1.9.8] - 2026-03-27

### Added
- **9 new response schemas** — `WorkflowResult`, `ContextAnalysisResult`, `VideoAIResult`, `ShortsPipelineResult`, `DepthMapResult`, `BrollPlanResult`, `BatchResult`, `PluginListResult` (22 total schemas, up from 13). All mapped into OpenAPI spec.
- **OpenAPI endpoint coverage +15** — 15 more async routes registered in `_JOB_ENDPOINTS` and `_ENDPOINT_SCHEMAS` for typed OpenAPI responses.

### Fixed (Batch 40)
- **3 more bare `"ffprobe"` in routes/utils** — `audio.py` waveform probe, `video_editing.py` reframe probe, `utils/media.py` media probe.
- **2 broken tests fixed** — `test_info_with_csrf` expected 200 from `/info` with no filepath (returns 400), `test_file_no_path` expected 400 but route returns 404.
- **pytest hanging on exit** — WorkerPool daemon threads prevented clean exit. Added session-scoped `_shutdown_worker_pool` fixture in conftest.py.

## [1.9.7] - 2026-03-27

### Added
- **5 new MCP tools** — `opencut_denoise_audio`, `opencut_upscale`, `opencut_scene_detect`, `opencut_depth_map`, `opencut_shorts_pipeline` (23 total, up from 18). AI clients can now drive denoising, upscaling, scene detection, depth mapping, and the full shorts pipeline.
- **WorkerPool priority queue** — Replaced FIFO `ThreadPoolExecutor` with custom `PriorityQueue`-backed pool. `JobPriority.CRITICAL` (0) jobs now leapfrog `BACKGROUND` (200) work when all workers are busy. Equal-priority jobs still run in FIFO order.

### Fixed (Batch 39)
- **3 more bare `"ffprobe"` in routes/utils** — `audio.py` waveform probe, `video_editing.py` reframe probe, `utils/media.py` media probe — all crash on bundled installs where ffprobe isn't in PATH.
- **FFmpeg/ffprobe path warning** — `get_ffmpeg_path()` and `get_ffprobe_path()` now log WARNING when binary not found in PATH (was silently falling back to bare name, causing confusing subprocess errors later).

## [1.9.6] - 2026-03-27

### Fixed (Batch 38 — Route Cleanup & Remaining Crash Bugs)
- **10 remaining `_p(pct, msg)` closures missing `msg=""` default** — Found in `audio.py` (3), `video_fx.py` (5), `captions.py` (1), `workflow.py` (1). All core modules call `on_progress(pct)` with 1 arg, causing TypeError crash on every job using these routes. All now have `msg=""` default.
- **Redundant filepath re-validation in 7 routes** — `face_enhance`, `face_swap`, `upscale_run`, `remove_watermark`, `title_overlay`, `particle_apply`, `color_correct`, `color_convert`, `color_external_lut` all re-read `data.get("filepath")` and re-validated despite `@async_job` already doing this. Removed ~80 lines of dead validation code, now using the decorator-provided `filepath` param directly.
- **Unused `validate_filepath` import** — Removed from `video_specialty.py` after watermark/title routes stopped re-validating.

## [1.9.5] - 2026-03-27

### Fixed (Batch 37 — Infrastructure & Hardening)
- **Bare ffprobe in 8 core modules** — `audio_enhance`, `color_management`, `highlights`, `motion_graphics`, `scene_detect` (2 sites), `shorts_pipeline`, `transitions_3d`, and `video_ai` all used `"ffprobe"` directly in subprocess commands. On systems where ffprobe is bundled (not in PATH), all of these would crash. Fixed to use `get_ffprobe_path()`.
- **GPU VRAM check can hang** — `check_vram()` in `gpu.py` called `torch.cuda.mem_get_info()` with no timeout. On hung NVIDIA drivers, this blocked forever. Added 5-second timeout via ThreadPoolExecutor.
- **safe_error GPU OOM false positives** — `errors.py` pattern `"cuda" in lower and "memory" in lower` matched unrelated errors like "CUDA device not found in memory". Tightened to exact phrases `"cuda out of memory"` / `"cuda error: out of memory"`.
- **_unique_output_path 10K stat loop** — `helpers.py` looped up to 9,998 times trying filenames. On full disk or permission errors, wasted thousands of stat() calls. Capped at 100.
- **get_video_info silent defaults** — Fallback to 1920x1080@30fps was logged at DEBUG level (invisible). Upgraded all 3 fallback paths to WARNING so users know when probe data is missing.
- **Job cleanup on every creation** — `_new_job()` called `_cleanup_old_jobs()` synchronously on every job creation. With 1000+ old jobs, blocked the HTTP handler. Removed — periodic cleanup thread (already running every 5 minutes) handles this.
- **proc.poll() crash on stale handles** — `_cleanup_old_jobs()` called `.poll()` on potentially corrupt Popen objects. Added try/except per handle.
- **async_job future not stored for cancel** — Race window between `_new_job()` returning job_id and storing the future. Cancel requests during that window couldn't find the future. Now stores `_thread` reference immediately after submit.
- **multicam_xml path encoding** — `_path_to_url()` didn't URI-encode spaces or special characters in filenames. Premiere Pro rejected the XML. Added `urllib.parse.quote()`.
- **main.js parentNode null guard** — `select.parentNode.insertBefore()` could crash if select element had no parent. Added null check.

## [1.9.4] - 2026-03-27

### Fixed (Batch 36 Audit)
- **P0: face_enhance/face_swap/upscale `_p(pct, msg)` crash** — 3 `on_progress` closures in video_ai.py missing `msg=""` default; core modules call with 1 arg → TypeError. Added default.
- **P0: engagement attribute crash** — shorts pipeline response accessed `c.engagement.hook_strength` directly; switched to `getattr()` with defaults for all 5 engagement fields.
- **P0: broll_plan inconsistent response** — empty segments returned 2 keys but success path returned 4; frontend expects all 4. Fixed + added `plan is None` guard + `getattr()` for all window fields.
- **Security: style_arbitrary path traversal** — `/video/style/arbitrary` accepted `style_image` without `validate_filepath()`; attacker could read arbitrary files. Added validation.
- **Security: plugin install path traversal** — `/plugins/install` accepted arbitrary `source` directory without `validate_path()`. Added validation.
- **Security: plugin uninstall symlink escape** — `/plugins/uninstall` didn't verify resolved `plugin_dir` stays within `PLUGINS_DIR`. Added `os.path.realpath()` containment check.
- **Security: depth_effects model_id injection** — `model_size` param interpolated into HuggingFace model ID without validation in `apply_bokeh_effect()` and `apply_parallax_zoom()`. Added allowlist in all 3 functions.
- **GPU rate limiting gaps** — `/video/ai/interpolate` and `/video/ai/denoise` (basicvsr method) bypassed `rate_limit("ai_gpu")`, allowing concurrent GPU OOM. Added guards.
- **Queue allowlist +9 routes** — 9 newer routes missing from `_ALLOWED_QUEUE_ENDPOINTS`: interpolate, depth/map/bokeh/parallax, broll-plan, remove/watermark, upscale/run, multicam-xml, search/auto-index.
- **title_overlay preset allowlist** — Missing `lower_third`, `countdown`, `kinetic_bounce` presets (present in title_render but not title_overlay).
- **Engine registry cache race** — `_availability_cache` reads/writes in `get_available_engines()` were outside `_lock`; added lock protection.
- **main.js timer leak** — `_scanDebounceTimer` missing from `cleanupTimers()`; leaked on panel close.
- **main.js safeFixed** — `defaultZoomVal` slider used raw `toFixed()` instead of `safeFixed()` wrapper.
- **FFmpeg stderr truncation UX** — `run_ffmpeg()` truncated stderr silently; now prepends `"...[truncated] "` marker when truncating.
- **Test fix** — `test_system_gpu` expected `gpu_available` key but API returns `available`.

## [1.9.3] - 2026-03-27

### Added
- **JSON Structured Logging** (Phase 0.3) — File handler now outputs JSON via `python-json-logger`. Each log line includes `timestamp`, `level`, `module`, `job_id`, `message`. Console handler stays plain text. Graceful fallback if `python-json-logger` not installed.
- **CI Coverage Enforcement** (Phase 0.1) — `.github/workflows/build.yml` now runs `pytest-cov` with `--cov-fail-under=50` threshold. CI also triggers on PRs and pushes to main. PyInstaller build skipped for non-release runs.
- **Structured Error Migration** (Phase 0.2) — All route error handlers migrated from bare `{"error": str(e)}` to `safe_error()` from `opencut/errors.py`. Zero bare 500 error patterns remain across all 13 route files. Every API error returns `{error, code, suggestion}`.
- **Smart Tab Reordering** (Phase 3.2) — Sub-tabs now physically reorder in the DOM based on contextual relevance scores. Highest-scoring features move to front within each tab group. `resetTabOrder()` restores original order on clip deselection.
- **Frontend Error Code Mapper** (Phase 0.2) — `enhanceError()` now reads `data.code` field before regex fallbacks. Code-to-action mapping for GPU_OOM, MISSING_DEPENDENCY, FILE_NOT_FOUND, etc. with navigable settings links.
- **Core Module Unit Tests** (Phase 0.1) — 15 additional core modules tested (silence, fillers, scene_detect, auto_edit, highlights, workflow, speed_ramp, audio, face_reframe, chromakey, video_fx, export_presets, diarize, audio_duck, thumbnail).
- **i18n String Extraction** (Phase 6.2) — ~200 additional `data-i18n` attributes added to buttons, headers, labels, and tabs in index.html with corresponding en.json keys.
- **Pre-commit Hooks** (Phase 0.1) — `.pre-commit-config.yaml` now includes ruff lint/format + pytest smoke suite (on pre-push). `pre-commit` added to dev dependencies.
- **Log Levels Audit** (Phase 0.3) — 22 log calls across 10 files corrected: verbose processing steps downgraded INFO→DEBUG, fallback/degraded paths upgraded INFO→WARNING. No secrets found in logging.
- **Core Module Unit Tests Batch 2** (Phase 0.1) — 135 additional tests across 28 previously untested core modules (animated_captions, audio_enhance, audio_pro, audio_suite, caption_burnin, captions_enhanced, color_management, emotion_highlights, face_swap, face_tools, lut_library, motion_graphics, music_ai, music_gen, object_removal, particles, shorts_pipeline, style_transfer, styled_captions, transitions_3d, upscale_pro, video_ai, voice_gen, zoom, broll_insert, broll_generate, multimodal_diarize, social_post).
- **ExtendScript Mock Harness** (Phase 0.1) — `tests/jsx_mock.js` provides fake Premiere Pro DOM (app.project, activeSequence, tracks, markers, ProjectItem). Tests 38 ExtendScript functions including ocApplySequenceCuts, ocBatchRenameProjectItems, ocAddSequenceMarkers, ocAddNativeCaptionTrack. 35 assertions pass under Node.js.

### Refactored
- **`async_job` decorator adoption** — All 97 manual `_new_job()` + `threading.Thread` + `_update_job(status="complete")` patterns across 6 route files converted to `@async_job` decorator. Fixes race condition where cancelled jobs could be overwritten to "complete". Removes ~2,800 lines of boilerplate. Decorator extended with `filepath_required` and `filepath_param` parameters.
- **Split `video.py` into 5 domain blueprints** — Monolithic 3636-line `video.py` split into `video_core.py` (1395), `video_editing.py` (750), `video_fx.py` (687), `video_specialty.py` (489), `video_ai.py` (443). No file exceeds 1400 lines. All URL paths unchanged.
- **App factory pattern** — `server.py` now exposes `create_app(config)` for isolated Flask instances. New `opencut/config.py` centralizes env var reads into `OpenCutConfig` dataclass. Tests use independent app instances. Module-level `app = create_app()` preserved for backward compat.
- **Install route factory** — `make_install_route()` in `opencut/jobs.py` replaces 6 identical install endpoint handlers (depth, emotion, multimodal-diarize, broll-generate, face, crisper-whisper) with a single factory call per route.
- **Version sync CI enforcement** — `scripts/sync_version.py` now includes `package.json` in targets and supports `--check` flag for CI. Added to `.github/workflows/build.yml`.

### Security
- Replaced all `__import__()` calls with `importlib.import_module()` / `importlib.util.find_spec()` in helpers.py, system.py, engine_registry.py, plugins.py

### Fixed
- **`_safe_error` undefined** — Fixed `_safe_error` reference in video.py multicam XML (F821 undefined name bug)
- **`safe_int` missing import** — Added missing import in settings.py log tail endpoint (F821)
- **Timecode watermark plugin** — `start_tc` variable now used in FFmpeg drawtext filter (was extracted but unused)
- **Version sync** — All version strings (pyproject.toml, __init__.py, CEP manifest, UXP manifest, server startup banner, installer AppConstants.cs) now read from single source. Server banner uses `__version__` instead of hardcoded string.
- **TooManyJobsError handling** — 7 remaining `_new_job()` calls in system.py, search.py, timeline.py now properly catch `TooManyJobsError` and return HTTP 429 instead of 500.
- **audio_separate cleanup crash** — `temp_audio` variable moved above try block to prevent `NameError` in finally clause if `_resolve_output_dir()` fails.
- **GPUContext memory leak** — `__exit__` now calls `.cpu()` on registered models before clearing, actually freeing GPU VRAM instead of only deleting a loop variable.
- **batch_id collision risk** — Batch IDs extended from 8 to 16 hex chars, reducing collision probability from 1-in-65K to 1-in-4-billion batches.
- **XSS in engine registry** — All API-sourced values (domain, engine name, display_name, quality, speed) now wrapped in `esc()` before innerHTML insertion.
- **elapsedTimer leak** — `startJob()` now clears any existing elapsed timer before creating a new one, preventing interval accumulation on rapid re-starts.
- **mediaScanTimer leak** — Periodic media scan `setInterval` now assigned to tracked variable and cleared on `beforeunload`.
- **pollTimer leak** — `trackJobPoll()` now clears any existing poll timer before creating a new one; SSE fallback no longer creates duplicate trackers.
- **SSE parse errors silent** — SSE `onmessage` handler now logs JSON parse failures to console instead of silently swallowing them.
- **Reframe dimension DoS** — `/video/reframe` now enforces `min_val=16, max_val=7680` on width/height parameters, preventing absurd allocation requests.
- **ASS subtitle injection** — Caption burn-in now strips backslash sequences from source text before ASS formatting, preventing style override injection via subtitle content.
- **23 import sorting violations** fixed (I001)
- **15 unused imports** removed (F401)
- **4 unused variables** removed (F841)
- Codebase at **0 lint warnings** matching CI config

## [1.9.1] - 2026-03-26

### Added
- **Frontend Context Awareness** (Phase 3.2) — CEP panel now calls `POST /context/analyze` after clip selection. Guidance banner shows clip-specific recommendations. Sub-tabs for high-scoring features get visual highlights. Dismiss button hides banner.
- **Parallel Batch Processing** (Phase 5.4) — `process_batch_parallel()` in `batch_process.py` uses `ThreadPoolExecutor` for concurrent multi-clip operations. GPU ops limited to 1 worker, CPU ops scale to core count. Partial failure isolation — one item crash doesn't kill the batch.
- **Clip Notes Plugin** (Phase 6.1) — Second example plugin at `opencut/data/example_plugins/clip-notes/`. SQLite-backed per-clip notes with `POST /note`, `GET /notes`, `DELETE /note`, `GET /export` (text/CSV). Shipped alongside timecode-watermark as reference plugins.
- **Route Smoke Tests** (Phase 0.1) — Comprehensive smoke test suite in `tests/test_route_smoke.py` covering all 175+ endpoints across 13 blueprints. Verifies routes don't crash with minimal payloads.

### Fixed
- **Ruff lint cleanup** — Fixed unused imports in new test files (8 auto-fixed F401 violations)

## [1.9.0] - 2026-03-26

### Added
- **Contextual Awareness** (Phase 3.2) — Clip type detection and feature relevance scoring. `POST /context/analyze` accepts clip metadata and returns scored features, guidance messages, and per-tab relevance scores. 35 features scored across 4 tabs based on clip tags (talking_head, audio_only, long_duration, etc.)
- **Plugin System** (Phase 6.1) — Plugin loader discovers/validates/loads plugins from `~/.opencut/plugins/`. Each plugin has a `plugin.json` manifest and optional Flask Blueprint routes registered under `/plugins/<name>/`. Endpoints: `GET /plugins/list`, `GET /plugins/loaded`, `POST /plugins/install`, `POST /plugins/uninstall`. Includes example timecode-watermark plugin.
- **Multicam XML Export** — Generate Premiere Pro compatible FCP XML from multicam diarization cut data. `POST /video/multicam-xml` endpoint. CLI `multicam` command now exports XML instead of showing placeholder message.
- **Background Indexing with SQLite FTS5** (Phase 5.3) — Persistent footage index at `~/.opencut/footage_index.db` with full-text search. Incremental re-indexing via mtime comparison. Endpoints: `POST /search/auto-index`, `POST /search/db-search`, `GET /search/db-stats`, `POST /search/cleanup`.
- **Response Streaming** (Phase 5.2) — NDJSON streaming utilities for progressively delivering large result sets. `GET /jobs/stream-result/<job_id>` streams completed job results (segments, scenes, thumbnails) in batches of 50.
- **Context blueprint** (`routes/context.py`) — 2 new endpoints
- **Plugins blueprint** (`routes/plugins.py`) — 4 new endpoints
- **4 new search endpoints** — auto-index, db-search, db-stats, cleanup

### Fixed
- **TooManyJobsError handling** — All 96 manual `_new_job()` calls across video.py (47), audio.py (23), and captions.py (13) now catch `TooManyJobsError` and return proper 429 responses instead of 500 errors
- **AI GPU rate limiting** — Added `rate_limit("ai_gpu")` to 6 GPU-heavy routes: video AI upscale, background removal, shorts pipeline, and 3 music AI generation routes. Prevents concurrent GPU OOM crashes
- **Settings import validation** — `/settings/import` now validates workflow steps (requires endpoint + label per step) before saving, preventing malformed workflow injection
- **Preview frame bounds** — `width` parameter bounded to 32-3840, `detection_skip` bounded to 1-30
- **Secure tempfile** — Preview frame extraction uses `tempfile.mkstemp()` instead of predictable path construction

### Security
- 96 routes hardened against job limit bypass (TooManyJobsError → 429)
- Input bounds added to prevent resource exhaustion via unbounded parameters
- Workflow import step validation prevents malformed workflow injection
- Fixed double rate-limit release in demucs/watermark install routes (decorator + explicit call)

## [1.5.0] - 2026-03-23

### Added
- **Repeated Take Detection** — Jaccard similarity sliding-window to find and remove fumbled/repeated takes from transcriptions
- **YouTube Chapter Generation** — LLM-powered (Ollama/OpenAI/Anthropic) topic detection and chapter timestamp generation
- **Footage Search** — Index media library by spoken content; search by keyword or phrase across all clips
- **Post-Production Deliverables** — Generate VFX sheets, ADR lists, music cue sheets, and asset lists as CSV from sequence data
- **Color Match** — YCbCr histogram matching to match color profile of one clip to a reference clip
- **Multicam Auto-Switching** — Diarization-driven cut generation for podcast multicam editing
- **Auto Zoom Keyframes** — Face-detected keyframe generation for dynamic push-in zoom effect
- **Loudness Match** — FFmpeg two-pass LUFS normalization to match loudness across multiple clips
- **NLP Command Parser** — Natural language command dispatch via keyword matching and LLM
- **Timeline Write-Back** — Apply cuts, markers, and keyframes directly to the active Premiere Pro sequence via ExtendScript
- **Beat Markers** — Sync detected beat timestamps as sequence markers in Premiere Pro
- **Batch Rename** — Rename project panel clips with find/replace pattern support
- **Smart Bins** — Auto-sort project items into bins by rule (name contains, type, duration)
- **SRT to Native Captions** — Import SRT files as native Premiere Pro caption tracks
- **Batch Export from Markers** — Export individual clips defined by sequence markers
- **UXP Panel** — New parallel panel for Premiere Pro 25.6+ using modern UXP APIs (com.opencut.uxp)
- **Timeline Tab** — New CEP panel tab with all timeline write-back operations
- **Search & NLP Tab** — New CEP panel tab with footage search and AI command input
- **Deliverables Section** — Post-production document generation in the Export tab

### Fixed
- Improved progress callback support in color match, auto zoom, and loudness match operations
- LLMConfig now properly instantiated as dataclass object across all routes

## v1.2.0 (2026-03-15)

### New Features
- **Waveform Preview**: Visual waveform display on Silence tab with draggable threshold line synced to slider
- **Side-by-Side Preview**: Before/after frame comparison modal for video effects
- **Dependency Health Dashboard**: Grid view of all 24 optional dependencies with install status in Settings
- **First-Run Wizard**: Animated 3-step onboarding overlay for new users (dismissible, persisted)
- **Output File Browser**: Browse recent output files with Import-to-Premiere button
- **Favorites Bar**: Pin frequently-used operations as quick-access chips below the nav tabs
- **Batch Multi-Select**: Multi-file picker for batch operations (add selected, add all, clear)
- **Parameter Tooltips**: Hover tooltips on range sliders explaining what each parameter controls
- **Custom Workflow Builder**: Chain multiple operations into named reusable workflows (saved to `~/.opencut/workflows.json`)
- **Audio Preview Player**: Floating audio player to preview generated audio before importing
- **Settings Import/Export**: Export all settings (presets, favorites, workflows) as JSON, import on another machine
- **Right-Click Context Menu**: Quick-action context menu on clip selector (silence, transcribe, denoise, etc.)
- **Collapsible Cards**: Click card headers to collapse/expand dense form sections
- **Job Time Estimates**: Estimated processing time in the progress banner based on historical job data
- **Localization Framework**: Language selector in Settings (placeholder for future i18n support)
- **Video Reframe**: Resize/crop video for TikTok, Shorts, Reels, Instagram, Square, or custom dimensions with crop/pad/stretch modes
- **Clip Preview Thumbnail**: Visual thumbnail + duration/resolution metadata when selecting a clip
- **Command Palette**: Ctrl+K fuzzy search across all 28+ operations with keyboard navigation
- **Recent Clips**: Dropdown of last 10 used clips, persisted across sessions
- **Trim Tool**: Set in/out points to extract a clip portion (stream copy or re-encode)
- **Merge/Concatenate**: Join multiple clips into one (fast stream copy or re-encoded for mixed formats)
- **Auto-Crop Detect**: Smart reframe anchor using FFmpeg cropdetect for talking-head content
- **Audio Waveform Everywhere**: Waveform preview buttons on Denoise and Normalize tabs (reuses Silence tab infrastructure)
- **Per-Operation Presets**: Save/load settings per operation (persisted to localStorage)
- **Server Health Monitor**: 10-second heartbeat with reconnect banner and toast notification
- **Output Deduplication**: Auto-increment suffix prevents overwriting previous outputs

### Backend Improvements
- **FFmpeg Progress Parsing**: `_run_ffmpeg_with_progress()` helper parses `-progress pipe:1` for real percentage updates
- **Subprocess Kill on Cancel**: Job cancellation now terminates the running FFmpeg process via `_kill_job_process()`
- **Temp File Cleanup**: Stale `opencut_preview_*.jpg` files cleaned up on server startup
- **File Serving Endpoint**: `GET /file` serves local audio/video for preview player (path-restricted)

### Backend (14 new endpoints)

- `POST /audio/waveform` - Extract waveform peak data via FFmpeg PCM extraction
- `POST /video/preview-frame` - Extract single frame as base64 JPEG at timestamp
- `GET /system/dependencies` - Check 24 optional deps + FFmpeg install status
- `GET /outputs/recent` - List recent output files from completed jobs
- `GET/POST /favorites` - Favorites persistence to `~/.opencut/favorites.json`
- `GET/POST/DELETE /workflows` - Custom workflow CRUD to `~/.opencut/workflows.json`
- `GET/POST /settings/export|import` - Bundle presets + favorites + workflows
- `POST /system/estimate-time` - Historical ratio-based time estimation
- `POST /video/reframe` - Resize/crop video for target aspect ratio
- `GET /video/reframe/presets` - Available reframe platform presets
- `GET /file` - Serve local files for audio/video preview
- `POST /video/merge` - Concatenate multiple clips (demux or filter modes)
- `POST /video/trim` - Extract clip portion by in/out timecodes

## v1.1.0 (2026-03-15)

### New Features
- **Preset Save/Load**: Save and load all panel settings as named presets (persisted to `~/.opencut/user_presets.json`)
- **AI Model Manager**: View all downloaded AI models (HuggingFace, Torch, Whisper) with sizes, delete unused models to free space
- **GPU Auto-Detection & Recommendation**: Automatically recommend optimal model sizes and settings based on detected GPU VRAM
- **Job Queue System**: Queue multiple jobs for sequential processing instead of waiting for each to finish
- **Enhanced Job History**: Re-run previous jobs directly from history, toast notifications on completion
- **Keyboard Shortcuts**: Enter to run active operation, 1-6 to switch tabs, Escape to cancel
- **Enhanced Drag-and-Drop**: Drop files anywhere on the panel, not just the drop zone
- **Toast Notifications**: Non-intrusive slide-in notifications for job completion/errors
- **Transcript Search**: Search and navigate through transcript segments with highlight
- **Social Platform Export Presets**: Quick export presets for YouTube Shorts, TikTok, Instagram Reel/Story/Post, Twitter/X, LinkedIn
- **Premiere Theme Sync**: Detect Premiere Pro's UI brightness via CSInterface
- **Universal Auto-Import**: Smart ExtendScript import function that routes files to appropriate project bins

### Bug Fixes (v1.0.0 audit)
- Fixed 6 broken install routes using `subprocess.run` instead of `_sp.run`
- Fixed SSE race condition (dict copy + status read now inside `job_lock`)
- Fixed VBS launcher path quoting for `C:\Program Files\OpenCut`
- Replaced `eval()` with `JSON.parse()` in ExtendScript
- Fixed wrong `/cut/silence` endpoint in social_ready workflow
- Fixed `jobStarting` guard and `no_input` flag in all applicable routes
- Fixed temp file leaks in audio_separate and watermark routes
- Added `user-select: text` override for CEP Chromium inputs

## v1.0.0-beta (2026-02-09)

First public beta release with full feature set.

### Core (34 modules, 116 routes)
- Silence detection and removal with configurable thresholds
- Filler word detection (um, uh, like, you know, so, actually)
- Speaker diarization via pyannote.audio
- Job queue with async processing and progress polling

### Captions (5 modules)
- WhisperX / faster-whisper transcription with word-level timestamps
- 19 caption styles (YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, etc.)
- 7 animated caption presets (pop, fade, slide, typewriter, bounce, glow, highlight)
- Caption burn-in with style selection
- Translation to 50+ languages

### Audio (8 modules)
- Stem separation via Demucs (vocals, drums, bass, other)
- AI noise reduction + spectral gating
- Loudness normalization (LUFS targeting)
- Beat detection and BPM analysis
- Pro FX chain: compressor, EQ, de-esser, limiter, reverb, stereo width
- Text-to-speech with 100+ voices (Edge-TTS / F5-TTS)
- Procedural SFX generator (tones, sweeps, impacts, noise)
- AI music generation via MusicGen (AudioCraft)
- Audio ducking (auto-lower music under dialogue)

### Video (12 modules)
- FFmpeg stabilization (deshake/vidstab)
- Chromakey with green/blue/red screen + spill suppression
- Picture-in-picture overlay with position/scale
- 14 blend modes (multiply, screen, overlay, etc.)
- 34 xfade transitions (fade, wipe, slide, circle, pixelize, radial, zoom)
- 7 particle effect presets (confetti, sparkles, snow, rain, fire, smoke, bubbles)
- 6 animated title presets (fade, slide, typewriter, lower third, countdown, kinetic)
- Speed ramping with ease curves
- Scene detection via PySceneDetect
- Film grain, letterbox, vignette effects
- 15 built-in cinematic LUTs + external .cube/.3dl support
- Color correction (exposure, contrast, saturation, temperature, shadows, highlights)
- Color space conversion (sRGB, Rec.709, Rec.2020, DCI-P3)

### AI Tools (6 modules)
- AI upscaling: Lanczos (fast), Real-ESRGAN (balanced), Video2x (premium)
- Background removal via rembg (U2-Net)
- Face enhancement via GFPGAN
- Face swapping via InsightFace
- Neural style transfer
- Object/watermark removal (FFmpeg delogo + LaMA inpainting)
- Auto-thumbnail extraction with AI scoring

### Export & Batch (3 modules)
- Platform presets: YouTube, TikTok, Instagram, Twitter, LinkedIn, Podcast
- Batch processing with parallel execution
- Transcript export (SRT, VTT, plain text, timestamped)

### UI
- 6 dark themes (Cyberpunk, Midnight OLED, Catppuccin, GitHub Dark, Stealth, Ember)
- 6 main tabs with 24 sub-tabs
- Persistent processing banner with progress
- Single-job enforcement with UI lockout
- Auto-import results back to Premiere Pro timeline
