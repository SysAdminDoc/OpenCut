# OpenCut — Comprehensive Audit & Improvement Prompt

> **Purpose:** Paste this entire prompt into a fresh Claude Opus session with the OpenCut project loaded. It will drive a systematic, multi-pass audit of every file in the codebase.

---

## Context

You are auditing **OpenCut v1.9.16** — a Flask + CEP/UXP backend for Adobe Premiere Pro. It's a large codebase: **~82,500 lines** across 138 files (52K Python, 30K JS/HTML/CSS). The project has 867 tests (4 skipped), ruff lint is clean, and a prior audit (v1.9.17) fixed 3 bugs. Your job is to go **much deeper**.

**Read `CLAUDE.md` first** — it contains the full architecture, every gotcha, and the history of 35+ prior bug-fix batches. This is your reference bible. Then systematically work through the phases below.

---

## Phase 1: Backend Python — Route-Level Audit

Audit **every route blueprint** in `opencut/routes/`. For each file, read the full source and check:

### Files (18 total, ~11K lines):
`system.py` (2074L), `audio.py` (1557L), `video_core.py` (1396L), `captions.py` (1210L), `video_editing.py` (752L), `video_fx.py` (653L), `settings.py` (578L), `jobs_routes.py` (425L), `video_ai.py` (421L), `timeline.py` (422L), `video_specialty.py` (421L), `search.py` (267L), `workflow.py` (245L), `plugins.py` (174L), `deliverables.py` (168L), `nlp.py` (80L), `context.py` (106L), `__init__.py` (34L)

### Checklist per route file:
- [ ] Every POST route has `@require_csrf` decorator
- [ ] Every file-accepting route validates paths via `validate_filepath()` or `validate_path()`
- [ ] Every async route uses `@async_job("type")` — no manual `_new_job()`/`Thread` patterns
- [ ] All `on_progress` callbacks default `msg=""` (core modules call with 1 arg)
- [ ] `safe_float()`/`safe_int()` used for all numeric params from user input (not bare `int()`/`float()`)
- [ ] Route is in `_ALLOWED_QUEUE_ENDPOINTS` in jobs_routes.py if it's async
- [ ] Rate-limited routes use manual `rate_limit()`/`rate_limit_release()`, NOT the `@require_rate_limit` decorator (which releases immediately on async routes)
- [ ] Error responses use structured `{error, code, suggestion}` format from errors.py
- [ ] `TooManyJobsError` is caught around `_new_job()` calls
- [ ] Subprocess calls use `_sp` alias, not raw `subprocess`
- [ ] FFmpeg/ffprobe calls use `get_ffmpeg_path()`/`get_ffprobe_path()`, never bare `"ffmpeg"`
- [ ] All string params that go into FFmpeg filters are sanitized (no `;` injection)
- [ ] Allowlists are synced with actual module constants (e.g., style names, model names, blend modes)

---

## Phase 2: Core Module Audit

Audit **every core module** in `opencut/core/`. For each file, read the full source and check:

### Files (71 total, ~26K lines):
Read each `.py` file in `opencut/core/`. The major ones (300+ lines) are: `styled_captions.py`, `video_ai.py`, `silence.py`, `social_post.py`, `object_removal.py`, `multimodal_diarize.py`, `scene_detect.py`, `motion_graphics.py`, `animated_captions.py`, `voice_gen.py`, `auto_edit.py`, `style_transfer.py`, `speed_ramp.py`, `shorts_pipeline.py`, `captions.py`, `music_ai.py`, `music_gen.py`, `nlp_command.py`, `workflow.py`, `audio_suite.py`, `video_fx.py`, `lut_library.py`, `color_management.py`, `diarize.py`, `highlights.py`

### Checklist per core module:
- [ ] `ensure_package()` return value is checked before importing (must raise RuntimeError on failure)
- [ ] `cv2.VideoCapture`/`VideoWriter` have `isOpened()` checks and `try/finally` cleanup
- [ ] GPU tensor cleanup uses `.cpu()` → `del` → `torch.cuda.empty_cache()` in `finally`
- [ ] `run_ffmpeg()` usage is correct (returns `str`, raises `RuntimeError` — don't check `.returncode`)
- [ ] Consolidated helpers used (`run_ffmpeg`, `ensure_package`, `get_video_info` from `helpers.py`)
- [ ] No local copies of `_run_ffmpeg`/`_ensure_package`/`_get_video_info`
- [ ] Thread-safe access to global cached objects (double-checked locking)
- [ ] All `open()` calls have `encoding="utf-8"`
- [ ] Temp files cleaned in `finally` blocks, not just success paths
- [ ] Float/int coercion on all values from JSON before use in FFmpeg filters or math
- [ ] No division by zero possibilities (fps, duration, counts)

---

## Phase 3: Infrastructure Audit

Read and audit these core infrastructure files:

### Files:
- `opencut/server.py` (805L) — app factory, startup, system site-packages
- `opencut/jobs.py` (422L) — job state, async_job decorator, install route factory
- `opencut/security.py` (415L) — CSRF, path validation, safe_pip_install, safe_float/safe_int
- `opencut/helpers.py` (696L) — FFmpegCmd, run_ffmpeg, ensure_package, get_video_info, temp cleanup
- `opencut/errors.py` (284L) — error taxonomy, safe_error classification
- `opencut/job_store.py` (289L) — SQLite persistence, WAL mode, thread-local connections
- `opencut/workers.py` (154L) — WorkerPool priority queue
- `opencut/user_data.py` (276L) — thread-safe JSON file I/O with per-file locks
- `opencut/mcp_server.py` (558L) — MCP stdio server with 23 tools
- `opencut/cli.py` (1081L) — Click CLI with 16 commands
- `opencut/gpu.py` (160L) — GPU detection and caching

### Checklist:
- [ ] Race conditions in shared state (jobs dict, caches, locks)
- [ ] Resource leaks (threads, file handles, DB connections, subprocesses)
- [ ] Lock ordering (no deadlocks from nested locks)
- [ ] Error propagation (exceptions don't get silently swallowed)
- [ ] Timeout handling on all subprocess and network calls
- [ ] Graceful shutdown (cleanup of threads, connections, temp files)

---

## Phase 4: Frontend Audit

### CEP Panel (com.opencut.panel/):
- `main.js` (9731L) — Read thoroughly. Check for:
  - XSS via innerHTML (should use `esc()` helper or DOM APIs)
  - Null/undefined guards on `.split()`, `.toFixed()`, `.length`, property access
  - Event listener leaks (use delegation, not per-element)
  - Timer leaks (all intervals/timeouts in `cleanupTimers()`)
  - CSRF header is `X-OpenCut-Token` (not `X-CSRF-Token`)
  - `safeFixed()` used instead of raw `.toFixed()`
  - `api()` timeout handling
  - Race conditions in async callbacks (stale `selectedPath`, `currentJob`)

- `index.html` (3661L) — Check for:
  - Duplicate `class=` attributes (HTML parser silently drops the second)
  - All buttons have `type="button"`
  - `data-i18n` attributes match locale keys
  - Form elements have proper labels/ARIA

- `style.css` (8514L) — Check for:
  - Dead CSS rules (selectors matching nothing in HTML)
  - Inconsistent spacing (should follow 4px rhythm)
  - Missing interactive states (hover, focus-visible, disabled, active)
  - Responsive breakpoint gaps

### UXP Panel (com.opencut.uxp/):
- `main.js` (2658L) — Check that:
  - All endpoints use full Blueprint-prefixed paths (`/audio/denoise` not `/denoise`)
  - Param names match what routes expect (`filepath` not `file_path`)
  - Result field access matches actual API responses
  - CSRF header is `X-OpenCut-Token`

### ExtendScript (host/index.jsx):
- `index.jsx` (2300L) — Check that:
  - ES3 compliant (no let/const/arrow/template literals)
  - All `app.project` access guarded with null checks
  - MarkerCollection uses iterator pattern (not indexed access)
  - `try/catch` on all Premiere API calls
  - `Number()` coercion before arithmetic on values from JSON

---

## Phase 5: Test Coverage Audit

### Files (22 total, ~9K lines):
Read test files and check:
- [ ] Do tests cover all 18 route blueprints? (check `test_route_smoke.py`)
- [ ] Are error paths tested (400, 404, 429, 500)?
- [ ] Are CSRF enforcement tests present for all POST routes?
- [ ] Do integration tests use the `create_app()` factory (not module-level app)?
- [ ] Are there tests for edge cases documented in Gotchas?
- [ ] Mock paths match actual module locations after consolidation
- [ ] No tests that always pass regardless of implementation (assertions too weak)

---

## Phase 6: Feature Improvement Proposals

After completing the audit, propose concrete improvements:

1. **Missing validations** — Any user inputs that reach FFmpeg/subprocess without sanitization
2. **Missing error handling** — Code paths that would produce 500 instead of structured errors
3. **Performance** — Unnecessary work, missing caches, N+1 patterns
4. **Dead code** — Unreachable branches, unused imports, orphaned functions
5. **Consistency** — Patterns used in some files but not others (e.g., some routes validate, others don't)
6. **Security** — Any remaining injection vectors, SSRF, path traversal, or info disclosure

---

## Output Format

For each issue found, report:

```
### [SEVERITY: P0/P1/P2/P3] File: path/to/file.py (line N)

**Bug:** One-line description
**Impact:** What breaks or could go wrong
**Fix:**
```python
# OLD:
broken_code_here

# NEW:
fixed_code_here
```
```

Severity levels:
- **P0** — Crashes, data loss, security vulnerabilities
- **P1** — Functional bugs (wrong behavior, missing features)
- **P2** — Robustness (missing error handling, edge cases)
- **P3** — Code quality (dead code, inconsistencies, minor improvements)

**Work through each phase sequentially. Read the actual source files — do not guess or assume. Fix every real bug you find. At the end, provide a summary count of issues by severity.**
