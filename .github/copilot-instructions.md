# OpenCut — Copilot Instructions

## What this project is

OpenCut is a Python/Flask local HTTP server (port 5679) that powers an Adobe Premiere Pro automation panel. The CEP panel (`extension/com.opencut.panel/`) and UXP panel (`extension/com.opencut.uxp/`) communicate with the backend via REST. It ships as a PyInstaller single-executable with a Windows installer built by Inno Setup.

## Commands

```bash
# Dev server
python -m opencut.server                   # http://localhost:5679

# Install
pip install -e ".[dev]"                    # backend + test/lint tools
pip install -e ".[standard]"              # common AI deps (whisper, opencv, etc.)
pip install -e ".[ai]"                    # GPU-optional AI models (CPU onnxruntime)
pip install -e ".[ai-gpu]"               # GPU AI models (onnxruntime-gpu)

# Lint
ruff check --select E,F,I --ignore E501 opencut/
ruff format opencut/

# Tests — full suite
python -m pytest tests/ -v --tb=short

# Single test file
python -m pytest tests/test_route_smoke.py -v

# Single test
python -m pytest tests/test_route_smoke.py::TestSystemRoutes::test_health -v

# Integration tests (require real FFmpeg)
python -m pytest tests/ -m integration -v

# Version sync (always use this — never hand-edit version strings)
python scripts/sync_version.py --set X.Y.Z
python scripts/sync_version.py --check    # CI uses this

# Build executable
pyinstaller opencut_server.spec
```

## Architecture

```
opencut/
  server.py          Flask app factory — registers all blueprints, Sentry, temp_cleanup
  jobs.py            Module-level job dict + lock; async_job decorator; job registry
  helpers.py         FFmpegCmd builder, run_ffmpeg(), get_video_info(), deferred cleanup
  security.py        CSRF (rotating tokens, TTL), path validation, rate_limit primitives
  checks.py          check_X_available() one-liners — 40+ optional dep guards
  errors.py          OpenCutError taxonomy + error_response() helper
  workers.py         Priority-based thread pool (JobPriority enum: CRITICAL→BACKGROUND)
  config.py          OpenCutConfig dataclass — env-var-driven, single source of truth
  utils/config.py    CaptionConfig, SilenceConfig, etc. feature dataclasses

  core/              ~450 single-responsibility feature modules
  routes/            ~90 Flask blueprints
    wave_a/ – wave_h/    Recent feature batches added in waves
  export/            OTIO, Premiere XML, AAF, OTIOZ, SRT/VTT/ASS writers

extension/com.opencut.panel/   CEP panel (HTML/JS + ExtendScript ES3)
extension/com.opencut.uxp/     UXP panel (ES modules, Premiere 25.6+)
installer/src/OpenCut.Installer/  Windows WPF installer (C#, .NET 9)
tests/                         pytest suite; tests/fuzz/ for Atheris harnesses
scripts/                       sync_version.py, sbom.py
```

**Request flow:** CEP/UXP panel → `POST /silence` (or any route) → `@require_csrf` → `@async_job("silence")` → worker thread → job status polled via `GET /status/<job_id>` or SSE stream.

**Job state** lives in module-level globals in `opencut/jobs.py` (`jobs` dict + `job_lock`). Tests reset this via the `_isolate_global_state` autouse fixture in `tests/conftest.py`.

## Key Patterns

### Adding a new async route
```python
from opencut.security import require_csrf
from opencut.jobs import async_job

@bp.route("/my-feature", methods=["POST"])
@require_csrf
@async_job("my_feature")
def my_feature(job_id, filepath, data):
    ...
```
Then add the endpoint to `_ALLOWED_QUEUE_ENDPOINTS` in `routes/jobs_routes.py`.

### Adding an optional dependency
Add a `check_X_available()` entry in `opencut/checks.py`. Gate the import inside the function. Never raise — return a 503 `MISSING_DEPENDENCY` with an install hint if the dep is absent.

### FFmpeg subprocess
Always use `run_ffmpeg(cmd, job_id=job_id)` (not raw `subprocess`). The `job_id` parameter registers the Popen with the job subsystem so cancel can actually kill the child process.

### Dataclass results
Make them subscriptable via `__getitem__` + `keys()` so routes can `return jsonify(dict(result))` without a `.to_dict()`. See `core/neural_interp.InterpResult` as the canonical shape.

### Rate limiting
```python
@rate_limit_category("gpu_heavy" | "cpu_heavy" | "io_bound" | "light")
```
Use `@gpu_exclusive` on the inner worker body for GPU model loads.

### Deprecating a route
```python
@deprecated_route(remove_in="2.0.0", replacement="/new/path",
                  reason="...", sunset_date="2026-10-01")
```
This wires the OpenAPI spec, response headers, and server logs automatically.

### CSRF in tests
```python
resp = client.get("/health")
token = resp.get_json()["csrf_token"]
headers = {"X-OpenCut-Token": token, "Content-Type": "application/json"}
```
The `csrf_token` and `csrf_headers()` helpers in `tests/conftest.py` do this for you.

## Version Management

Version is defined in `opencut/__init__.py` and must stay in sync across 19 targets (manifests, README badge, CHANGELOG, etc.). **Always** use:
```bash
python scripts/sync_version.py --set X.Y.Z
```

Targets include: `pyproject.toml`, `extension/com.opencut.panel/CSXS/manifest.xml`, `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.uxp/main.js` (`VERSION` constant), `README.md` badge, `CHANGELOG.md`.

## Test Conventions

- `tests/test_route_smoke.py` — smoke harness for every route; use Flask test client (no real server, no subprocess, no GPU)
- `tests/fuzz/` — Atheris fuzz entry points
- Mark slow/real-FFmpeg tests with `@pytest.mark.integration` or `@pytest.mark.slow`
- The `_isolate_global_state` autouse fixture in `conftest.py` clears module-level job/queue state after each test — don't bypass it
- Install tests are skipped when `app.config["TESTING"] = True` (guarded by `should_skip_install_in_testing()` in `jobs.py`)

## Commit Style

Single-line title ≤ 70 chars. No emojis. No `Co-Authored-By` lines.

Version-bump commits must be made via `sync_version.py` and touch all 19 targets in one commit. Tag every release `vX.Y.Z`.

Do **not** commit `content_calendar.*` — those are marketing-local files.
