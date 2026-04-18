# Contributing to OpenCut

Thanks for your interest. OpenCut is an active project with ~450 core modules, ~90 route blueprints, and 1,200+ API routes. This guide gets you oriented fast.

## Ground rules

- **No emojis in commit messages or code.** Enforced.
- **No `Co-Authored-By: Claude` lines.** Enforced.
- **No `--no-verify` on commits.** Hooks exist for a reason.
- **Rewrite, don't tack on.** If a module grows > 600 lines, consider decomposing.
- **Version everything.** `scripts/sync_version.py` keeps 19 targets in sync — run `--check` in CI.

## Repository layout (sub-100-line tour)

```
opencut/
  __init__.py              version stamp + module doc
  server.py                Flask app factory, Sentry init, temp_cleanup boot
  jobs.py                  async_job decorator, job registry, SQLite persistence
  helpers.py               FFmpegCmd builder, run_ffmpeg, get_video_info,
                           check_disk_space (v1.3+), deferred temp cleanup worker
  security.py              CSRF, path validation, rate_limit primitives, safe_*
  checks.py                check_X_available() one-liners (40+ entries)
  errors.py                OpenCutError taxonomy + safe_error()
  utils/config.py          CaptionConfig, SilenceConfig, etc. dataclasses

  core/                    ~450 single-responsibility feature modules
  routes/                  ~90 Flask blueprints — wave_a/b/c/d/e/f for recent adds
  export/                  OTIO, Premiere XML, AAF, OTIOZ, SRT/VTT/ASS writers

extension/com.opencut.panel/   CEP (Adobe) panel — HTML/JS/ExtendScript
extension/com.opencut.uxp/     UXP panel (Premiere 25.6+)
installer/src/OpenCut.Installer/   Windows WPF installer (C#, .NET 9)
tests/                    pytest suite; tests/fuzz/ for Atheris harness
scripts/                  sync_version, sbom, misc dev helpers
```

## Setting up

```bash
git clone https://github.com/SysAdminDoc/OpenCut.git
cd OpenCut
python -m venv .venv
.venv/Scripts/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[ai]"            # backend + AI extras (CPU)
pre-commit install                # ruff + trailing-ws + yaml/json checks
python -m opencut.server           # starts at http://localhost:5679
```

CEP panel development:

```powershell
reg add "HKCU\Software\Adobe\CSXS.11" /v PlayerDebugMode /t REG_SZ /d 1 /f
```

Open Chrome at `http://localhost:7474` after launching Premiere with the panel visible.

## Patterns you'll use every day

- **New async route** → `@require_csrf` → `@async_job("job_type")` on the route; worker body receives `(job_id, filepath, data)`. Add the rule to `_ALLOWED_QUEUE_ENDPOINTS` in `jobs_routes.py`.
- **New optional dep** → add a `check_X_available()` entry in `opencut/checks.py`. Gate imports inside the function. Never hard-fail if the dep is missing — return a 503 `MISSING_DEPENDENCY` with an install hint.
- **Dataclass results** → make them subscriptable via `__getitem__` + `keys()` so routes can `return dict(result)` to Flask's `jsonify` without a `.to_dict()` detour. See `core/neural_interp.InterpResult` for the canonical shape.
- **FFmpeg subprocess** → use `run_ffmpeg(cmd, job_id=job_id)` (v1.24+). The `job_id` parameter is how cancel actually kills the child process. Without it, you get the legacy non-cancellable path.
- **Rate limiting** → apply `@rate_limit_category("gpu_heavy" | "cpu_heavy" | "io_bound" | "light")` on the route. Use `@gpu_exclusive` on the inner worker body for GPU model loads.
- **Deprecating a route** → wrap with `@deprecated_route(remove_in="2.0.0", replacement="/new/path", reason="...", sunset_date="2026-10-01")`. The OpenAPI spec, response headers, and server logs pick it up automatically.
- **Governance / auth changes** → update `SECURITY.md` + bump the supported-versions table.

## Commit style

Single-line title ≤ 70 chars, descriptive body. Example:

```
v1.24.0: subprocess tracking + disk monitor + request correlation

helpers.run_ffmpeg now accepts job_id=... and auto-registers the Popen
with the job subsystem, closing the v1.14.0 audit finding "158
untracked subprocess calls" that couldn't be interrupted on cancel.
...
```

Version-bump commits touch `opencut/__init__.py` + 16 other targets. Always run `python scripts/sync_version.py --set X.Y.Z` — don't hand-edit.

## Pull requests

- Run `python -m ruff check --select E,F,I --ignore E501 opencut/` before pushing. CI blocks on this.
- Ship a test for any new behaviour. `tests/test_route_smoke.py` has the smoke harness; `tests/fuzz/` has the Atheris entry points.
- Update `CHANGELOG.md` with a bullet under the relevant version.
- Update `CLAUDE.md` if your change introduces a new pattern / gotcha / file location that future sessions should know about. CLAUDE.md is deliberately verbose — it's the onboarding document for maintainers returning after months away.
- **Don't commit `content_calendar.*`** — those are marketing-local files and shouldn't land upstream.

## Release checklist

1. `python scripts/sync_version.py --set X.Y.Z` and verify with `--check`.
2. `python -m ruff check --select E,F,I --ignore E501 opencut/` → `All checks passed!`
3. `python -c "from opencut.server import create_app; create_app()"` → smoke boot.
4. `CHANGELOG.md` + `CLAUDE.md` + `ROADMAP-NEXT.md` updated.
5. `python scripts/sbom.py` to refresh the SBOM (optional but nice).
6. Tag: `git tag -a vX.Y.Z -m "OpenCut vX.Y.Z"` — GitHub Actions build workflow picks it up and publishes installers.

## Where to ask questions

- Bugs / security: [SECURITY.md](SECURITY.md).
- Architecture / "where does X live": read [`CLAUDE.md`](CLAUDE.md) — it documents every non-obvious invariant + gotcha in the codebase.
- Feature roadmap: [`ROADMAP.md`](ROADMAP.md) (long-range 302-feature plan) and [`ROADMAP-NEXT.md`](ROADMAP-NEXT.md) (active quarter plan).
