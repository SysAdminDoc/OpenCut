# OpenCut — Implementation Roadmap

**Version**: 2.0 Vision
**Created**: 2026-03-25
**Baseline**: v1.7.2 (738-line server.py, 6,850-line main.js, 175+ endpoints, 50+ core modules)

---

## Guiding Principles

1. **Never break what works** — Every phase ships a working product. No "rewrite everything then test."
2. **Incremental migration** — New code coexists with old. Feature flags gate rollout. Old paths removed only after new paths are proven.
3. **User-facing value first** — Each phase delivers visible improvements, not just internal refactors.
4. **Measure before optimizing** — Add telemetry/logging before assuming bottlenecks.

---

## Phase 0: Foundation & Safety Net

**Goal**: Get the project into a state where large-scale changes are safe.

### 0.1 — Test Coverage Baseline

**Current state**: 3 test files, ~100 lines total. No frontend tests. conftest.py has basic Flask test client + CSRF fixture + FFmpeg media generators.

| Task | Detail |
|------|--------|
| **Route smoke tests** | Auto-generate one test per POST route: valid CSRF, minimal payload, assert 200 or valid job_id. Target: all 175+ endpoints have at least a "doesn't crash" test. Use conftest.py's `client` and `csrf_token` fixtures. |
| **Core module unit tests** | For each `opencut/core/*.py` module, test the primary function with the generated test media from conftest.py (`generate_test_audio`, `generate_test_video`). Mock FFmpeg subprocess where needed for speed. |
| **ExtendScript mock harness** | Create `tests/jsx_mock.js` — a fake Premiere DOM (`app.project.rootItem`, `app.project.activeSequence`, `ProjectItem`, `TrackItem`). Load `host/index.jsx` into Node.js with the mock. Test: `ocApplySequenceCuts`, `ocBatchRenameProjectItems`, `ocAddSequenceMarkers`, `ocAddNativeCaptionTrack`. |
| **CI enforcement** | Update `.github/workflows/build.yml`: fail build if coverage < 60% (routes) or < 50% (core). Add `pytest-cov` to dev deps. |
| **Pre-commit hooks** | Add ruff format check + pytest smoke suite to `.pre-commit-config.yaml`. |

**Deliverable**: Coverage report. Every route callable without crash. Every core module's happy path tested.

### 0.2 — Structured Error Taxonomy

**Current state**: Routes return ad-hoc `{"error": "some string"}` or raise unhandled exceptions. No error codes. Toast messages mix "FFmpeg returned code 1" with "Something went wrong."

| Task | Detail |
|------|--------|
| **Error code enum** | Create `opencut/errors.py` with categorized error codes: `WHISPER_MODEL_MISSING`, `GPU_OOM`, `FFMPEG_TIMEOUT`, `FFMPEG_CODEC_UNSUPPORTED`, `FILE_NOT_FOUND`, `FILE_PERMISSION_DENIED`, `INVALID_PARAMETER`, `DEPENDENCY_MISSING`, `RATE_LIMITED`, `JOB_CANCELLED`, `SERVER_BUSY`. Each code maps to: (a) HTTP status, (b) user-facing message template, (c) recovery suggestion. |
| **Error response helper** | `def error_response(code, detail=None) -> tuple[dict, int]` — returns `{"success": false, "error_code": "GPU_OOM", "message": "Your GPU ran out of memory.", "suggestion": "Try reducing clip length or switch to CPU mode.", "detail": "CUDA error: out of memory"}` with correct HTTP status. |
| **Migrate routes** | Replace all bare `return {"error": str(e)}, 500` patterns across all 11 route files with `error_response()` calls. Grep shows ~180 instances of `"error"` in route files. |
| **Frontend error mapper** | In `main.js`, update `enhanceError()` to read `error_code` from API response and map to specific guidance. Fall back to `message` field for unknown codes. |

**Deliverable**: Every API error returns a structured, actionable response. Frontend shows recovery steps, not stack traces.

### 0.3 — Structured Logging

**Current state**: `logging.info(f"Processing {filepath}")` scattered inconsistently. Crash log writes to `~/.opencut/crash.log` on 500s only.

| Task | Detail |
|------|--------|
| **JSON log format** | Add `python-json-logger` to deps. Configure in `server.py` startup. Every log line includes: `timestamp`, `level`, `module`, `job_id` (from thread-local or context), `endpoint`, `duration_ms`. |
| **Job correlation** | Thread-local `job_id` set by `@async_job` decorator, read by logger. Every log line during a job is traceable. |
| **Log levels audit** | Grep all `logging.*` calls (~200). Ensure: DEBUG for verbose processing steps, INFO for job start/complete, WARNING for fallbacks/retries, ERROR for failures. Remove any that log secrets (API keys, tokens). |
| **Log viewer endpoint** | `GET /logs/tail?lines=100&level=ERROR&job_id=xxx` — returns filtered log lines as JSON array. Wire to a "Logs" card in Settings tab. |

**Deliverable**: Debugging a user-reported issue goes from "send me your crash.log" to "what's the job ID?"

---

## Phase 1: Backend Modernization

**Goal**: Migrate from Flask+threads to FastAPI+async without breaking the CEP/UXP panels.

### 1.1 — FastAPI Parallel Server

**Strategy**: Don't rewrite Flask. Stand up FastAPI alongside it. Proxy shared state. Migrate routes one blueprint at a time.

| Task | Detail |
|------|--------|
| **FastAPI scaffold** | Create `opencut/api/` directory. `app.py` with FastAPI instance, CORS middleware (same `null`/`file://` origins), CSRF middleware ported from `security.py`. Uvicorn entrypoint. |
| **Pydantic models** | For each route blueprint, create `opencut/api/models/<blueprint>.py` with request/response models. Replace all `safe_float()`/`safe_int()` hand-validation with Pydantic `Field(ge=0, le=100)` constraints. Generate from existing route code — each `request.json.get()` call maps to a model field. |
| **OpenAPI auto-generation** | Remove manual `/openapi.json` endpoint. FastAPI generates it automatically. Add response models to every route for full schema. |
| **Migrate system routes first** | Port `routes/system.py` (simplest, no jobs) to `api/routes/system.py` as FastAPI router. Verify: `/health`, `/info`, `/system/gpu`, dependency checks all work identically. Run old Flask + new FastAPI tests side-by-side. |
| **Migrate remaining routes** | One blueprint per PR: system → settings → search → nlp → timeline → jobs → captions → audio → video (largest last). Each PR: write FastAPI router, add Pydantic models, write tests, verify parity with Flask route. |
| **Port CSRF to middleware** | FastAPI middleware that checks `X-OpenCut-Token` header using same rotating token pool from `security.py`. |
| **Port rate limiting** | FastAPI dependency injection: `Depends(rate_limit("model_install", max_concurrent=1))`. |

**Deliverable**: All 175+ endpoints served by FastAPI. Flask code removed. Automatic API docs at `/docs`.

### 1.2 — Async Job System

**Current state**: `@async_job` spawns `threading.Thread`. Jobs stored in an in-memory dict with `threading.RLock`. No persistence across restarts. No priority. No retry.

| Task | Detail |
|------|--------|
| **Job persistence** | SQLite database at `~/.opencut/jobs.db`. Table: `jobs(id, type, status, progress, message, result_json, error, created_at, started_at, completed_at, endpoint, payload_json)`. On server restart, mark all `running` jobs as `interrupted`. |
| **Job retry** | `POST /jobs/retry/<job_id>` already exists in settings routes. Wire it to actually re-dispatch the original endpoint+payload from the DB. Add exponential backoff for auto-retry on transient errors (timeout, OOM). Max 3 retries. |
| **Priority queue** | Replace simple list queue with priority levels: `critical` (user-initiated), `normal` (batch), `background` (indexing). Use `heapq` or SQLite ordering. |
| **Concurrency control** | Config: `max_concurrent_jobs = 2` (1 GPU + 1 CPU). Job dispatcher checks GPU availability before scheduling GPU jobs. |
| **WebSocket job streaming** | FastAPI WebSocket endpoint `/ws/jobs/<job_id>`. Real-time push replaces SSE polling. Fallback to SSE for CEP panels that don't support WebSocket well. |
| **Job cleanup** | Cron-style cleanup: delete completed jobs older than 7 days from DB. Delete associated temp files. |

**Deliverable**: Jobs survive server restarts. Failed jobs retry automatically. Multiple jobs can run concurrently (CPU + GPU).

### 1.3 — Process Isolation for GPU Work

**Current state**: All AI models load into the same Python process. WhisperX + Demucs + RealESRGAN competing for GPU memory = OOM crashes.

| Task | Detail |
|------|--------|
| **Worker pool architecture** | Create `opencut/workers/` with a `WorkerManager` class. Workers are separate Python processes started via `multiprocessing` or `subprocess`. Each worker loads one model family (whisper, demucs, realesrgan, etc.). |
| **IPC protocol** | Workers communicate via localhost HTTP (each worker is a minimal FastAPI on a random port) or via `multiprocessing.Queue`. Job dispatcher routes to appropriate worker based on job type. |
| **GPU memory management** | Worker reports GPU memory usage on startup. Dispatcher won't schedule a GPU job if available VRAM < model's known requirement. Workers release GPU memory and exit after idle timeout (5 min). |
| **Graceful degradation** | If GPU worker OOMs, catch the error, report to user with specific guidance ("This model needs 4GB VRAM, you have 2GB available. Switching to CPU mode."), and optionally re-dispatch to CPU worker. |

**Deliverable**: No more OOM crashes from model conflicts. GPU utilization visible in UI.

---

## Phase 2: Frontend Architecture

**Goal**: Break the monolithic 6,850-line main.js and 3,100-line index.html into a maintainable, modular system.

### 2.1 — Build System

**Current state**: No build system. Raw HTML/CSS/JS served directly. No minification, no tree-shaking, no source maps.

| Task | Detail |
|------|--------|
| **Vite setup** | Create `extension/com.opencut.panel/vite.config.js`. Input: `client/src/main.js`. Output: `client/dist/`. CEP loads `dist/index.html`. Vite dev server not needed (CEP has its own Chromium). |
| **Entry point split** | `main.js` → `src/main.js` (50 lines: imports + init). Each feature area becomes a module: `src/tabs/cut.js`, `src/tabs/captions.js`, `src/tabs/audio.js`, etc. |
| **CSS modules or scoped styles** | Each tab module imports its own CSS. Vite bundles them. Global design tokens stay in `src/styles/tokens.css`. Component styles in `src/styles/components/`. |
| **HTML template system** | Replace 3,100-line monolithic HTML with JS-rendered templates. Each tab module exports a `render()` function that returns HTML string. Tab container calls `render()` on first visit (lazy). |
| **Source maps** | Enable in Vite config. CEP Chromium DevTools can now map errors to original source files. |
| **Hot reload for dev** | Vite watch mode + CEP reload shortcut. Edit a file, see changes in 2 seconds instead of manually reloading the panel. |

**Deliverable**: `npm run build` produces optimized bundle. `npm run dev` enables fast iteration. No behavior change for users.

### 2.2 — TypeScript Migration

**Strategy**: Incremental. Rename `.js` → `.ts` one module at a time. Start with API layer (highest value — types catch mismatched payloads).

| Task | Detail |
|------|--------|
| **tsconfig.json** | `strict: false` initially (allow implicit any). Target: ES2020 (CEP Chromium supports it). `allowJs: true` for gradual migration. |
| **API types** | Create `src/api/types.ts` from FastAPI's OpenAPI schema (auto-generated). Every `api()` call gets typed request/response. Script to regenerate types from `/openapi.json`. |
| **Core state types** | Type the global state: `currentJob: Job \| null`, `connected: boolean`, `csrfToken: string`, `sequenceInfo: SequenceInfo \| null`. |
| **Tab module types** | As each tab module is extracted (Phase 2.1), type its params and DOM references. |
| **Strict mode** | After all modules migrated, enable `strict: true`. Fix remaining ~200 implicit-any errors. |

**Deliverable**: Type-safe API calls. Autocomplete in IDE. Compile-time catch for payload mismatches.

### 2.3 — State Management

**Current state**: State scattered across 20+ closure variables (`currentJob`, `connected`, `csrfToken`, `lastTimelineCuts`, `sequenceInfo`, `footageIndex`, `beatMarkerTimes`, etc.) plus DOM element values plus localStorage.

| Task | Detail |
|------|--------|
| **Reactive store** | Create `src/store.ts` — a simple pub/sub store (no framework dependency). State tree with typed slices: `{ connection: {...}, job: {...}, media: {...}, settings: {...} }`. |
| **Subscribe pattern** | `store.subscribe("job.status", (newVal, oldVal) => updateJobUI())`. Tab modules subscribe to their relevant slices. No more manual DOM updates scattered across 50 functions. |
| **Persistence layer** | `store.persist("settings")` auto-syncs to localStorage. Replace all manual `localStorage.getItem`/`setItem` calls (grep shows ~30). |
| **DevTools integration** | `window.__OPENCUT_STORE__` exposes state for debugging. Log state transitions in dev mode. |

**Deliverable**: Single source of truth. Any component can react to any state change. Debugging = inspect the store.

### 2.4 — Component System

**Goal**: Each UI section is a self-contained component with its own template, styles, and logic.

| Task | Detail |
|------|--------|
| **Base component class** | `src/components/Component.ts` — `render()`, `mount(container)`, `destroy()`, `on(event, handler)`. No framework, just a pattern. |
| **Card component** | `CardComponent({ title, description, children })` — replaces the 100+ manual `<section class="card">` blocks in HTML. |
| **Form components** | `SliderField`, `SelectField`, `CheckboxField`, `FilePickerField` — each handles its own label, hint, validation, disabled state. Eliminates 200+ lines of manual form HTML. |
| **Tab components** | `TabContainer` manages tab switching, lazy rendering, and cleanup. Each tab is a component that registers its sub-tabs. |
| **Toast system** | `ToastManager` component. Queue-based, auto-dismiss, action buttons. Replaces the current scattered `showToast()` calls. |

**Deliverable**: Adding a new feature = create one component file. No touching index.html, no touching main.js init.

---

## Phase 3: UX Transformation

**Goal**: Turn OpenCut from a feature catalog into a workflow tool.

### 3.1 — Workflow Engine

**Current state**: Every feature is an independent card with its own form. No concept of chaining operations.

| Task | Detail |
|------|--------|
| **Workflow definition format** | JSON schema: `{ name: "Clean Interview", steps: [{ action: "silence/detect", params: {...} }, { action: "silence/remove" }, { action: "audio/normalize", params: {...} }] }`. Stored in `~/.opencut/workflows/`. |
| **Workflow runner** | Backend endpoint `POST /workflow/run` accepts workflow JSON + input file. Executes steps sequentially. Each step's output feeds the next step's input. Progress: `step 2/5 — Normalizing audio`. |
| **Workflow builder UI** | Drag-and-drop step builder in the existing Workflows section (Settings tab). Each step is a card with collapsible params. Add/remove/reorder steps. Preview estimated time. |
| **Preset workflows** | Ship 6 built-in workflows: (1) Clean Interview, (2) Podcast Polish, (3) Social Media Clip, (4) Music Video, (5) Documentary Rough Cut, (6) YouTube Upload Ready. |
| **One-click workflows** | Top of each main tab shows "Quick Actions" — the most common workflow for that context. Cut tab: "Clean Up" button. Audio tab: "Studio Polish" button. Captions tab: "Auto Subtitle" button. |

**Deliverable**: 80% of users complete their task with one click. Power users build custom workflows.

### 3.2 — Contextual Awareness

**Current state**: Panel shows all 40+ sub-tabs regardless of what's selected in Premiere.

| Task | Detail |
|------|--------|
| **Clip type detection** | ExtendScript queries active clip metadata: has audio? has video? duration? codec? frame rate? Track type? Send to panel via `evalScript`. |
| **Feature relevance scoring** | Each feature declares its relevance: `{ requires: ["audio"], suggests: ["talking_head"], irrelevant: ["image_only"] }`. Panel scores each feature against clip metadata. |
| **Smart tab ordering** | Sub-tabs reorder by relevance score. Irrelevant features move to "More" overflow. Talking-head clip: Silence, Captions, Denoise float to top. B-roll clip: Color, Speed, Transitions float to top. |
| **Empty state guidance** | When no clip is selected: "Select a clip in your timeline to see relevant tools." When clip type is detected: "Interview clip detected — try Clean Up for quick results." |

**Deliverable**: Panel feels like it understands what you're editing. Reduces cognitive load from 40+ options to 5-10 relevant ones.

### 3.3 — Preview Before Commit

**Current state**: Click "Detect Silence" → wait → cuts applied (or not). No visual feedback before commit.

| Task | Detail |
|------|--------|
| **Waveform preview for silence** | After silence detection, render waveform with colored regions (green = keep, red = remove). User can drag region boundaries before applying. Uses `POST /audio/waveform` data. Render in `<canvas>`. |
| **Caption preview overlay** | After caption generation, show scrollable text with timestamps. Click a line to seek in Premiere. Edit text inline before applying. |
| **Color grade A/B** | After LUT/color correction, show source vs. processed frame side-by-side. Use `POST /video/preview-frame` to generate comparison frames. |
| **Cut list review** | After any cut operation (silence, full edit, auto-edit), show a table: `Start → End | Duration | Action | Reason`. Checkboxes to include/exclude individual cuts. "Apply Selected" button. |
| **Undo integration** | Before applying any timeline modification, create a Premiere undo group via ExtendScript `app.project.activeSequence.undoGroupStart("OpenCut: Remove Silence")`. After, `undoGroupEnd()`. User can Cmd+Z to undo the entire batch. |

**Deliverable**: Users feel in control. No more "click and pray." Errors caught before they happen.

### 3.4 — Keyboard Shortcuts & Command Palette

**Current state**: Command palette exists (Ctrl+K) with ~30 commands. No customizable shortcuts. No vim-style composition.

| Task | Detail |
|------|--------|
| **Shortcut registry** | `src/shortcuts.ts` — map of action IDs to key combos. Default set: `Ctrl+Shift+S` (silence), `Ctrl+Shift+C` (captions), `Ctrl+Shift+N` (normalize), `Ctrl+Shift+D` (denoise). |
| **Customization UI** | Settings → Keyboard Shortcuts card. Table of all actions with editable key combos. Conflict detection. Reset to defaults. |
| **Shortcut persistence** | Store in localStorage `opencut_shortcuts`. Merge with defaults on load. |
| **Command palette enhancement** | Fuzzy search (not just startsWith). Show keyboard shortcut next to each command. Recently used commands float to top. "Run workflow..." sub-menu. |

**Deliverable**: Power users never touch the mouse for common operations.

---

## Phase 4: Installation & Reliability

**Goal**: Make OpenCut trivially installable and impossible to break.

### 4.1 — Dependency Resolution Overhaul

**Current state**: `safe_pip_install()` has a 3-tier fallback (system, --user, --target). Still fails with Errno 13 on locked directories. No progress feedback during install. No verification after install.

| Task | Detail |
|------|--------|
| **Isolated venv per feature group** | Create venvs at `~/.opencut/envs/whisper/`, `~/.opencut/envs/demucs/`, etc. Each feature group gets its own Python environment. No cross-contamination. No system pip. |
| **Install with progress** | `safe_pip_install()` returns a generator/callback with progress. Parse pip's `--progress-bar` output. Stream to frontend via SSE during install. Show: "Installing WhisperX... Downloading pytorch (340MB/1.2GB)". |
| **Post-install verification** | After pip install, attempt `import <package>` in a subprocess. If it fails, report the specific error (missing CUDA, wrong Python version, missing C compiler) with actionable guidance. |
| **Pre-built wheels** | Host pre-compiled wheels for Windows (the hardest platform) on a CDN/GitHub Releases. `safe_pip_install()` checks CDN first before falling back to PyPI. Eliminates "need Rust compiler" errors. |
| **Dependency dashboard enhancement** | Show per-feature: installed version, latest version, disk usage, GPU requirement, install/update/remove buttons. Not just "installed: yes/no". |

**Deliverable**: "Install WhisperX" works on the first click, every time, on every machine.

### 4.2 — Docker Deployment Option

| Task | Detail |
|------|--------|
| **Dockerfile** | Multi-stage build: (1) Python 3.12 + CUDA base, (2) pip install all deps, (3) copy OpenCut source, (4) bundle FFmpeg. Final image: ~8GB with all models, ~2GB without. |
| **docker-compose.yml** | `opencut-server` service with GPU passthrough (`deploy.resources.reservations.devices`). Volume mount for `~/.opencut/` (models, config, jobs DB). Port 5679 exposed. |
| **Model volume** | Separate named volume for AI models (`~/.opencut/models/`). Persists across container rebuilds. First-run downloads models on demand. |
| **Health check** | Docker HEALTHCHECK hits `/health` endpoint. Auto-restart on failure. |
| **One-liner install** | `docker run -d --gpus all -p 5679:5679 -v opencut-data:/root/.opencut opencut/server:latest` |

**Deliverable**: Zero-config deployment for users comfortable with Docker. Guaranteed reproducible environment.

### 4.3 — Health Monitoring & Crash Recovery

**Current state**: Health check is a simple `/health` ping. No GPU monitoring. No crash recovery. Server dies = user must manually restart.

| Task | Detail |
|------|--------|
| **System status endpoint** | `GET /system/status` returns: CPU usage, RAM usage, GPU VRAM used/total, disk space in output dir, active jobs count, queue depth, uptime, Python version, CUDA version. |
| **Status bar in panel** | Persistent footer bar: `● Connected | GPU: 2.1/8GB | Jobs: 1 running, 2 queued`. Updates every 5 seconds. Red indicators for: disconnected, GPU unavailable, disk full. |
| **Server auto-restart** | Panel detects disconnection → attempts `/health` with backoff → if server process died, re-launch it (via the existing VBS launcher or direct `subprocess`). Show "Reconnecting..." banner with progress. |
| **Job recovery** | On server restart, check `jobs.db` for interrupted jobs. Show banner: "1 job was interrupted. [Retry] [Dismiss]". Retry re-dispatches with original params. |
| **Crash telemetry (opt-in)** | On crash, collect: error code, endpoint, Python version, GPU model, OS version. No file contents or paths. Send to a telemetry endpoint (opt-in with clear toggle in Settings). |

**Deliverable**: The panel always recovers. Users never see "connection lost" without a path back.

---

## Phase 5: Performance & Scale

**Goal**: Handle large projects (1000+ clips, 4K+ footage, hour-long timelines) without choking.

### 5.1 — Lazy Tab Rendering

**Current state**: All 40+ sub-tabs exist in the DOM simultaneously. 100+ form controls initialized on page load.

| Task | Detail |
|------|--------|
| **Tab lifecycle** | `TabManager` creates tab DOM only on first visit. Destroys DOM after navigating away (or keeps the last 3 visited tabs cached). |
| **Measured impact** | Profile current memory usage vs. lazy. Expected: 150MB → 40MB DOM footprint. |
| **Skeleton loading** | Show lightweight placeholder (pulsing card outlines) during tab render. Feels instant even if render takes 50ms. |

### 5.2 — Response Streaming

**Current state**: Large responses (caption lists with 500+ segments, scene detection with 200+ scenes) are returned as one JSON blob.

| Task | Detail |
|------|--------|
| **Streaming JSON** | For endpoints that return large arrays, use NDJSON (newline-delimited JSON). Each line is one item. Frontend parses incrementally and updates UI progressively. |
| **Chunked caption loading** | Caption endpoints return segments in batches of 50. Frontend renders first batch immediately, loads rest in background. User sees results in <1s instead of waiting 5s for full response. |
| **Thumbnail streaming** | `POST /export/thumbnails` currently generates all then returns. Switch to: generate one, stream it, generate next. User sees thumbnails appearing one by one. |

### 5.3 — Background Indexing

**Current state**: Footage search requires manual "Index" button click. Index is stored in memory (lost on restart).

| Task | Detail |
|------|--------|
| **Auto-index on project open** | When panel connects and detects a Premiere project, auto-start background indexing of all project items. Low-priority job that yields to user-initiated work. |
| **Persistent index** | Store footage index in SQLite (`~/.opencut/footage_index.db`). Keyed by project path + item nodeId. Incremental: only re-index changed/new items. |
| **Search speed** | Full-text search with SQLite FTS5. Sub-100ms search across 10,000+ clips. |

### 5.4 — Parallel Processing

| Task | Detail |
|------|--------|
| **Batch parallelism** | When batch-processing multiple clips (batch export, batch color grade), process N clips in parallel (N = CPU cores or 2 for GPU jobs). Current: serial, one at a time. |
| **Pipeline parallelism** | In workflows, start the next step's download/preprocessing while the current step finishes. Example: while silence detection runs on CPU, start loading the Whisper model for captions on GPU. |

**Deliverable**: Large projects feel as responsive as small ones. Background work happens without user action.

---

## Phase 6: Ecosystem & Polish

**Goal**: Make OpenCut a platform, not just a tool.

### 6.1 — Plugin System

| Task | Detail |
|------|--------|
| **Plugin manifest** | `plugin.json`: `{ name, version, author, description, routes: [...], ui: { tab, icon, component } }`. Plugins are Python packages with FastAPI routers + optional frontend components. |
| **Plugin loader** | Server scans `~/.opencut/plugins/` on startup. Registers plugin routes under `/plugins/<name>/` prefix. Frontend loads plugin UI into a "Plugins" tab. |
| **Plugin marketplace** | GitHub-based registry. `GET /plugins/registry` fetches available plugins. One-click install from UI. |
| **Example plugins** | Ship 2 reference plugins: (1) "Timecode Watermark" — burns timecode overlay on exports. (2) "Clip Notes" — adds searchable notes to project items. |

### 6.2 — Localization

**Current state**: All 3,100+ lines of HTML contain hardcoded English strings.

| Task | Detail |
|------|--------|
| **i18n framework** | `initI18n()` already exists in main.js (currently a stub). Implement: load locale JSON from `client/locales/en.json`. `t("silence.detect_button")` returns localized string. |
| **String extraction** | Script to extract all user-visible strings from HTML (button text, labels, descriptions, hints, error messages) into `en.json`. ~800 strings estimated. |
| **Locale files** | Start with: English, Spanish, Japanese, Portuguese, Chinese (Simplified). These cover ~70% of Premiere's user base. |
| **RTL support** | CSS logical properties (`margin-inline-start` instead of `margin-left`). Test with Arabic locale. |

### 6.3 — Collaboration & Sharing

| Task | Detail |
|------|--------|
| **Preset export/import** | Export: serialize all settings for a feature (silence thresholds, caption styles, color presets) as a `.opencut` file (JSON with metadata). Import: drag-drop or file picker. |
| **Workflow sharing** | Workflows exported as `.opencut-workflow` files. Include all step definitions and default params. Importable via UI or command palette. |
| **Team presets** | Shared network folder for presets. Settings → Team Presets → point to a folder. All team members see the same presets. |

### 6.4 — Project Templates

| Task | Detail |
|------|--------|
| **Template definitions** | JSON files: `{ name: "YouTube Video", export: { format: "mp4", codec: "h264", resolution: "1080p", bitrate: "15M" }, captions: { style: "youtube-default" }, audio: { loudness: -14, normalize: true }, aspect: "16:9" }`. |
| **Template selection** | New project wizard or Settings → Active Template. All export/audio/caption defaults auto-configure. |
| **Built-in templates** | YouTube, Instagram Reels, TikTok, Podcast (audio-only), Cinema (ProRes 4K), Broadcast (EBU R128 loudness). |
| **Custom templates** | "Save Current Settings as Template" button. Names and saves all current config as a reusable template. |

---

## Phase 7: UXP Parity & Native Integration

**Goal**: UXP panel matches CEP feature-for-feature, then surpasses it with native APIs.

### 7.1 — Shared UI Core

**Current state**: CEP and UXP panels share zero code. 211KB HTML + 311KB JS (CEP) vs 46KB HTML + 68KB JS (UXP).

| Task | Detail |
|------|--------|
| **Shared component library** | `extension/shared/` directory with framework-agnostic components. Both CEP and UXP import from here. Build system outputs two bundles. |
| **Feature registry** | `features.json` defines every feature: id, label, description, endpoint, params schema, requires (audio/video/gpu). Both panels read this and auto-generate UI. Adding a feature = add one JSON entry + one backend route. |
| **UXP feature gap closure** | Port all 40+ CEP sub-tabs to UXP. Estimated: 30 are direct ports (same HTTP calls), 10 need UXP-specific adaptations (native file handling, timeline API). |

### 7.2 — Native UXP Timeline Access

**Current state**: CEP uses ExtendScript `evalScript()` for all timeline operations. Round-trip: JS → CEP → ExtendScript → Premiere → ExtendScript → CEP → JS. Slow, serialization-heavy, error-prone.

| Task | Detail |
|------|--------|
| **Direct timeline read** | UXP's `premierepro` module provides `app.project.activeSequence` directly. Read track items, markers, in/out points without ExtendScript. |
| **Direct timeline write** | Apply cuts, add markers, rename clips through native UXP API. No serialization. No evalScript latency. |
| **Performance comparison** | Benchmark: apply 100 cuts via ExtendScript vs UXP native. Expected: 10x faster (eliminate JSON serialization round-trips). |
| **Hybrid mode** | For Premiere 25.6+ features not yet in UXP API, fall back to ExtendScript via UXP's `script.evalScript()`. Feature-detect at runtime. |

### 7.3 — Premiere Menu Integration

| Task | Detail |
|------|--------|
| **Context menu items** | Right-click a clip in Premiere → "OpenCut: Remove Silence", "OpenCut: Add Captions", "OpenCut: Normalize Audio". UXP API supports custom menu items. |
| **Keyboard shortcut registration** | Register shortcuts in Premiere's shortcut system (not just panel-internal). `Ctrl+Alt+S` triggers silence removal from anywhere in Premiere. |
| **Panel auto-focus** | When a context menu action is triggered, panel opens and navigates to the relevant tab with the selected clip pre-loaded. |

---

## Execution Timeline

```
Phase 0: Foundation          ████████░░░░░░░░░░░░░░░░░░░░░░  (Weeks 1-4)
Phase 1: Backend             ░░░░████████████░░░░░░░░░░░░░░  (Weeks 3-10)
Phase 2: Frontend            ░░░░░░░░████████████░░░░░░░░░░  (Weeks 6-14)
Phase 3: UX Transform        ░░░░░░░░░░░░████████████░░░░░░  (Weeks 10-20)
Phase 4: Install/Reliability ░░░░░░░░░░░░░░░░████████░░░░░░  (Weeks 14-20)
Phase 5: Performance         ░░░░░░░░░░░░░░░░░░░░████████░░  (Weeks 18-24)
Phase 6: Ecosystem           ░░░░░░░░░░░░░░░░░░░░░░░░██████  (Weeks 22-28)
Phase 7: UXP Parity          ░░░░░░░░░░░░░░░░░░████████████  (Weeks 18-28)
```

**Overlap is intentional** — phases are parallelizable across developers:
- Backend dev works Phase 1 while frontend dev starts Phase 2
- UX work (Phase 3) can start as soon as component system (Phase 2.4) is ready
- Phase 7 (UXP) is independent and can run in parallel with everything

---

## Priority Matrix

If resources are constrained, do these first (highest user impact per effort):

| Priority | Item | Phase | Impact | Effort |
|----------|------|-------|--------|--------|
| Priority | Item | Phase | Impact | Effort | Status |
|----------|------|-------|--------|--------|--------|
| **P0** | Test coverage baseline | 0.1 | Enables everything else | Medium | DONE |
| **P0** | Structured error taxonomy | 0.2 | Fixes #1 user complaint (cryptic errors) | Medium | DONE |
| **P0** | Structured logging (JSON) | 0.3 | Debugging job failures | Medium | DONE |
| **P1** | Dependency resolution overhaul | 4.1 | Fixes #2 user complaint (install failures) | High | DONE |
| **P1** | One-click workflows | 3.1 | Biggest UX improvement possible | High | DONE |
| **P1** | Preview before commit | 3.3 | Removes fear of using the tool | High | DONE |
| **P2** | Vite build system | 2.1 | Enables all frontend improvements | Medium | DONE |
| **P2** | Health monitoring | 4.3 | Eliminates "is it working?" confusion | Medium | DONE |
| **P2** | Lazy tab rendering | 5.1 | Immediate performance win | Low | DONE |
| **P2** | Job persistence (SQLite) | 1.2 | Jobs survive restarts | Medium | DONE |
| **P2** | Keyboard shortcuts | 3.4 | Power user productivity | Low | DONE |
| **P2** | Docker deployment | 4.2 | Zero-config deployment | Medium | DONE |
| **P2** | i18n framework | 6.2 | Opens global market | High | DONE |
| **P2** | Project templates | 6.4 | Quick-start for new users | Low | DONE |
| **P2** | Preset export/import | 6.3 | Collaboration & sharing | Low | DONE |
| **P2** | Contextual awareness | 3.2 | Panel adapts to clip type | Medium | DONE |
| **P2** | Background indexing (SQLite FTS5) | 5.3 | Fast persistent footage search | Medium | DONE |
| **P2** | Response streaming (NDJSON) | 5.2 | Progressive large result delivery | Low | DONE |
| **P2** | Plugin system scaffold | 6.1 | Extensibility platform | High | DONE |
| **P2** | Multicam XML export | — | Premiere import of multicam edits | Low | DONE |
| **P2** | Parallel batch processing | 5.4 | Multi-clip ops in parallel | Medium | DONE |
| **P2** | Clip Notes example plugin | 6.1 | Second reference plugin | Low | DONE |
| **P2** | Frontend context integration | 3.2 | Guidance banner + tab highlighting | Low | DONE |
| **P2** | Route smoke tests | 0.1 | Every endpoint has a smoke test | Medium | DONE |
| **P2** | CI coverage enforcement | 0.1 | Builds fail below 50% coverage | Low | DONE |
| **P2** | Smart tab reordering | 3.2 | Sub-tabs reorder by relevance | Low | DONE |
| **P2** | Structured error migration | 0.2 | Routes use safe_error() | Medium | DONE |
| **P2** | __import__() → importlib | — | Security hardening | Low | DONE |
| **P2** | Frontend error code mapper | 0.2 | Actionable error guidance in UI | Medium | DONE |
| **P2** | Core module unit tests (15) | 0.1 | 15 most important modules covered | High | DONE |
| **P2** | i18n string extraction | 6.2 | ~200 elements with data-i18n | Medium | DONE |
| **P2** | CI on PRs + coverage gate | 0.1 | CI runs on PRs, not just tags | Low | DONE |
| **P2** | Version string unification | — | Single source of truth for version | Low | DONE |
| **P2** | Pre-commit hooks | 0.1 | Ruff + pytest on commit/push | Low | DONE |
| **P2** | Log levels audit | 0.3 | Correct DEBUG/INFO/WARNING usage | Low | DONE |
| **P2** | Core module unit tests batch 2 (28) | 0.1 | 28 more modules covered (135 tests) | High | DONE |
| **P2** | ExtendScript mock harness | 0.1 | JSX functions testable under Node.js | Medium | DONE |
| **P3** | FastAPI migration | 1.1 | Big effort, mostly internal benefit | Very High | PLANNED |
| **P3** | TypeScript migration | 2.2 | Developer productivity, not user-facing | High | SCAFFOLDED |
| **P3** | Process isolation (GPU) | 1.3 | Eliminates OOM crashes | Very High | PLANNED |
| **P3** | UXP parity | 7.1-7.3 | Modern Premiere integration | Very High | PLANNED |

---

## Success Metrics

| Metric | Current | Phase 2 Target | Phase 7 Target |
|--------|---------|----------------|----------------|
| Time to first useful action | ~2 min (find tab, configure, run) | 30s (one-click workflow) | 10s (context-aware suggestion) |
| Install success rate | ~60% (permission/CUDA errors) | 90% (isolated venvs) | 99% (Docker/pre-built) |
| Test coverage (backend) | <5% | 60% | 85% |
| Test coverage (frontend) | 0% | 30% | 60% |
| Memory usage (panel) | ~150MB | 60MB (lazy tabs) | 40MB (virtual DOM) |
| Error messages with recovery steps | ~5% | 80% | 100% |
| Features available in UXP | ~40% | 70% | 100% |
| Supported languages | 1 (English) | 1 | 6 |
