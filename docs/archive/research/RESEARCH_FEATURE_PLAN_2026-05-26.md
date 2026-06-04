# OpenCut — Research and Feature Plan (2026-05-26 Pass)

> **Companion file to** [`ROADMAP.md`](ROADMAP.md), [`ROADMAP-NEXT.md`](ROADMAP-NEXT.md), and [`RESEARCH_FEATURE_PLAN_2026-05-25.md`](RESEARCH_FEATURE_PLAN_2026-05-25.md) (closed; 22+ items shipped across 4 autonomous loops).
> **Scope of this pass:** evidence-grounded survey of areas the 2026-05-25 plan did **not** cover — performance, observability, crash recovery, plugin/skill extensibility, resource preflight, and trust signals. Where prior plans cover an item, I link rather than re-author.
> **Baseline:** HEAD `7796a20`. v1.32.0. 1,517 routes / 107 blueprints / 211 test files / 439 locale keys (297 consumers).
> **Method:** read 4 prior planning files, ran 3 parallel subagent surveys (perf, extensibility, onboarding), then **verified each cited claim against the source** before including it. 3 agent claims rejected as incorrect; corrections noted inline.

---

## Executive Summary

OpenCut at HEAD `7796a20` is the most surface-rich open-source Premiere automation backend in existence — and the 2026-05-25 plan closed every actionable governance/feature item the prior research identified. The shape that's working: aggressive shipping (~70 ships in 6 days), strict `@async_job` discipline, generated manifests, 7 release-smoke gates, and a now-coherent UXP Agent tab that surfaces the four major new backends.

The remaining headroom is **not** more model surfaces or more wave letters. It's the dimension the prior plans **did not measure**: what happens *after* a user hits Run — does the system tell them what's going on, recover when things fail, cache expensive work, and let third parties extend it? Today the answers are: partially, no, no, and barely. That's the highest-value direction for v1.33+.

### 10 highest-value opportunities (priority order)

| # | Title | Priority | Effort | Why first |
|---|-------|----------|--------|-----------|
| 1 | Content-addressable transcript cache (by audio-hash) | **P0** | M | Single biggest user-visible perf win. Whisper costs are 10–20 min/GB; today every re-run pays full price. |
| 2 | `missing_dependency()` includes the pip extra name | **P0** | S | One-line fix unblocks 30+ minute install-fumbling sessions per user. |
| 3 | GPU semaphore: default to 30s acquire-wait (not immediate 429) | **P0** | S | Concurrent AI calls return 429 *instantly* today — UX bug, not capacity guard. |
| 4 | Disk preflight wired into heavy routes | **P1** | M | `disk_monitor.preflight()` exists but no route calls it. Render-fills-disk → corrupted output. |
| 5 | Job resume for interrupted jobs (not just flag as interrupted) | **P1** | L | Today a 30-min job that crashes at minute 28 must restart from zero. |
| 6 | `GET /webhooks/event-types` discovery endpoint | **P1** | S | Webhook authors cannot enumerate event types without reading source. |
| 7 | Plugin job-registration API (`@plugin_job` decorator) | **P1** | M | Plugins can register routes but cannot start `@async_job`-style background work. |
| 8 | Third-party agent-skill loader (`~/.opencut/skills/<id>/SKILL.md`) | **P1** | M | Shipped in v4.95; user packages now load after schema and route-manifest validation. |
| 9 | Enriched job metadata: `peak_vram_mb`, `exit_reason`, `started_at` | **P2** | M | `job_store.py` schema is captured-but-not-queryable; debugging crashed jobs is guesswork. |
| 10 | Request-ID propagation into FFmpeg/subprocess stderr | **P2** | S | A 10-min job's logs cannot be walked end-to-end via the existing `request_id` filter. |

---

## Evidence Reviewed

### Local files & directories inspected
- HEAD `7796a20`; full `git log --since="14 days ago"` (168 commits surveyed).
- Planning corpus: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) §1–§12, [`ROADMAP.md`](ROADMAP.md) (tier tables), [`ROADMAP-NEXT.md`](ROADMAP-NEXT.md), [`RESEARCH_FEATURE_PLAN_2026-05-25.md`](RESEARCH_FEATURE_PLAN_2026-05-25.md) (Execution Status block), [`CHANGELOG.md`](CHANGELOG.md) `[Unreleased]`.
- Core modules: `opencut/server.py` (imports + `create_app`), `opencut/errors.py` (line 110–150), `opencut/job_store.py` (line 197–301), `opencut/jobs.py` (line 159–170), `opencut/core/gpu_semaphore.py` (line 24–107), `opencut/core/disk_monitor.py`, `opencut/core/request_correlation.py`, `opencut/core/captions.py` (line 274–806), `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py`, `opencut/core/plugin_marketplace.py`, `opencut/core/agent_skills.py`, `opencut/core/workflow.py`, `opencut/core/webhook_system.py`, `opencut/mcp_server.py` (line 700–767), `opencut/core/footage_index_db.py`, `opencut/core/footage_search.py`, `opencut/routes/captions.py` (line 1–63), `opencut/cli.py`.
- Generated manifests: `opencut/_generated/route_manifest.json` (1,517 / 107).
- Panel HTML: `extension/com.opencut.panel/client/index.html` (line 71, 3991–4007).
- Panel JS: `extension/com.opencut.panel/client/main.js` (line 1690, 10518–10580).
- Installer: `Install.bat` (line 1–25).
- Locale: `extension/com.opencut.panel/client/locales/en.json` (439 keys).
- Subagent reports: 3 parallel surveys (performance/observability, extensibility, onboarding).

### Claims that did NOT survive verification (corrected here)
The onboarding subagent made 3 mistakes; the perf subagent made 1. I caught them by direct file inspection and removed them from this plan:

1. **Wrong — "Heavy ML imports at module level break cold start."** Subagent claimed `routes/captions.py:41-59` eagerly imports `faster_whisper`. **Verified:** lines 40–58 import only `core.silence`, `core.zoom`, `export.premiere`, `export.srt`, `utils.config`, `utils.media`. Whisper is *correctly* lazy — `opencut/core/captions.py:516` (`import whisper`), `:629` (`from faster_whisper import WhisperModel`), `:806` (`import whisperx`) are all inside function bodies. **Cold-start is not the problem.**
2. **Wrong — "First-Run Wizard advertised but missing from main.js."** Subagent claimed only `<span id="workspaceClipStatus">` exists. **Verified:** `index.html:3991-4007` declares `wizardOverlay`; `main.js:1690-1691` binds it; `main.js:10518-10580` runs the open/close/Escape/Tab handlers. The wizard IS shipped.
3. **Wrong — "Install.bat does not auto-escalate on non-admin."** Subagent claimed silent exit. **Verified:** `Install.bat:19` runs `Start-Process … -Verb RunAs` which triggers UAC. The agent's "silent failure" framing was wrong; what *can* happen is the user dismissing UAC, but the cmd window stays open at `pause`.
4. **Partially wrong — "Telemetry exists at `telemetry_aptabase.py`."** **Verified:** there is no `opencut/core/telemetry_aptabase.py` file; the test file `test_telemetry_aptabase.py` exists, and `docs/TELEMETRY.md` describes opt-in Aptabase. The integration shape is correct but the file path the agent cited doesn't exist. (Search shows the integration is in `core/aptabase.py` per `tests/`.)

### Git history range
`a8f62c0` (Wave S, 2026-05-19) → `7796a20` (loop-4 wrap, 2026-05-25). 168 commits in 14 days. Pattern: 22+ feature/UI commits closed the 2026-05-25 plan; remainder is wave-N–S core stubs, audit batches 1–8, and infrastructure hardening (CI, manifests, registries).

### External research
- CEP→UXP migration timing: Adobe CEP EOL ~Sept 2026 per `docs/UXP_MIGRATION.md`.
- Whisper performance baselines: faster-whisper docs (`tiny` ~5×realtime CPU, `base` ~2×realtime, `large-v3` ~0.3×realtime); content-hash caching is the cited pattern across pyannote, Demucs, Whisper community tooling.
- Plugin sandboxing patterns: Obsidian (capability-declared manifests), VSCode (engines + activationEvents + contributes), Vite plugins (no sandbox; trust-on-install) — OpenCut's lock-file model is closer to VSCode's signed-marketplace approach but without a public registry yet.

### Areas not verified
- **Live Whisper benchmark numbers.** I cite Whisper costs in minutes/GB by family, but haven't run a fresh measurement on this machine. **Needs live validation** before final acceptance criteria are set.
- **`opencut/core/aptabase.py` file contents.** I confirmed `docs/TELEMETRY.md` exists and the integration is opt-in by design; the actual scrubbing code (subagent cited file:line references that don't match) deserves re-verification before any further telemetry recommendation.
- **GPU semaphore behavior under contention.** I verified `ACQUIRE_TIMEOUT=0` default at `gpu_semaphore.py:51`, but did not run a contention test to confirm the 429-flood claim. **Needs live validation.**

---

## Current Product Map

### Core workflows
Identical to the 2026-05-25 plan's §Current Product Map — 10 workflows. Refer to that file rather than re-stating.

### Platforms & distribution
- Windows installer (WPF recommended + Inno fallback), Windows ARM64 doc'd but not shipped (F101).
- macOS source launcher; notarization tooling in place, GitHub Secrets pending (F202).
- Linux source launcher + Flatpak/AppImage build wiring (F249); Flathub submission pending.
- Docker CPU + GPU.
- MCP server (curated 39 + opt-in 1,464 as of `ROADMAP.md` v4.90) + UXP MCP bridge `/mcp/*` (F146).

### Notable surfaces added since 2026-05-25
- `/agent/chat/*` (F143/F144) + UXP Agent tab.
- `/enhance/auto/*` (Q3) one-click pipeline.
- `/shorts/variants/*` (Q8) A/B variants.
- `/timeline/sequence-index/*` (Q7/F273) spreadsheet view.
- `/mcp/*` (F146) UXP bridge.
- F236 caption display-settings UI in UXP Captions tab.

---

## Feature Inventory (delta only — items present but **lacking what the 2026-05-25 plan promised next**)

For each item below: where it lives, what works, what was deferred.

| Feature surface | Maturity | What's missing |
|---|---|---|
| Transcription | **Complete (no cache)** | `core/captions.py` runs Whisper every call. No file-hash → JSON cache. Footage-search index (`core/footage_index_db.py`) is a search index, not a cache. |
| `missing_dependency()` error | **Partial** | `opencut/errors.py:144-150` produces "Install it from the Settings tab under Dependencies" — does NOT include the pip extra name. Users must trial-and-error `pip install opencut[…]`. |
| GPU semaphore | **Functional but UX-hostile** | `gpu_semaphore.py:51` defaults `ACQUIRE_TIMEOUT=0` → instant 429 on contention. The semaphore is a *limit*, not a *queue*. |
| Disk-space preflight | **Lib present, unwired** | `core/disk_monitor.py::preflight()` exists; no heavy route calls it. |
| Job interruption recovery | **Half-done** | `job_store.py::mark_interrupted()` flips status on startup; `get_interrupted_jobs()` lists them; **no resume path** — user must resubmit. |
| Webhook event catalogue | **Hard-coded** | `core/webhook_system.py` ships 8+ event types as constants; no `GET /webhooks/event-types` discovery endpoint. The dispatcher fires by string. |
| Plugin job registration | **Routes-only** | `core/plugins.py` lets a plugin register a Flask blueprint; no `@plugin_job` decorator analogous to `@async_job`. Plugins can't start background work. |
| Agent skills | **Built-in only** | `core/agent_skills.py` loads from `opencut/data/builtin_skills/<id>/`. No `~/.opencut/skills/` user-directory scan, no marketplace, no schema for 3rd-party. |
| Job metadata richness | **Sparse** | `job_store.py:197-200` records `started_at`/`completed_at` but not `peak_vram_mb`, `exit_reason`, `requested_extras`, or the runtime CPU/GPU split. |
| Request-ID propagation | **Python-only** | `core/request_correlation.py` adds `r-{uuid16}` to the OpenCut logger; subprocess stderr (FFmpeg, faster-whisper progress) flushes to logs without the ID. |
| First-Run Wizard | **Shipped** | `main.js:10518-10580` wires it; one-time persist via `/system/onboarding`. **(Subagent claimed missing; verified present.)** |

---

## Competitive and Ecosystem Research (deltas not in prior plans)

| Source | Capability not yet in OpenCut | Steal vs avoid |
|---|---|---|
| **Obsidian** (plugins) | Plugin author can register both UI tabs AND background work via the same manifest; capability scopes are enforced at runtime (network/file). | **Steal:** runtime capability enforcement, not just declaration. **Avoid:** Obsidian's "trust on first install" — OpenCut's lock-file model is better. |
| **VSCode** (extensions) | `engines.vscode` + `contributes.commands` + activation events. Extensions can declare *when* they load (don't pay cost until used). | **Steal:** lazy-activation hint in plugin manifest. **Avoid:** the marketplace's review backlog model. |
| **Aider / Cursor** (agents) | Snapshot-per-step with cheap rollback. Plan → preview → accept/reject per step. | **Steal:** already partially captured in F143 design; the *snapshot* portion (OTIO marker snapshot per executed step) is still pending. |
| **GitHub Actions** (caching) | `actions/cache@v3` content-addressable cache with hash-of-inputs keys; restore-keys for partial hits. | **Steal:** the input-hash + restore-key pattern for Whisper / Demucs / Depth Anything outputs. **Avoid:** the cloud dependency — OpenCut's cache must stay local. |
| **Frame.io** (delivery review) | Review portal already in OpenCut (F232 Headscale plan). Frame.io's advantage: per-clip retention curve, hover-scrub, frame-perfect timecode comments. | **Steal:** the hover-scrub UX. **Avoid:** Frame.io's pricing — keep local. |
| **Docker Buildx** (resource preflight) | Refuses builds when free disk < $required, shows what's needed. | **Steal:** the explicit "needs Xgb, have Ygb" preflight error. |
| **DaVinci Resolve** (panel UX) | Per-clip rating chip in viewer, persists across project sessions. | **Steal:** combine with the Q7 Sequence Index — ratings/tags already in the row schema. |
| **fly.io machine API** (job lifecycle) | Jobs have explicit states `created → started → running → stopped` with `exit_reason` enum. | **Steal:** the enum-shaped exit reason in `job_store.py`. |

---

## Highest-Value New Features

### N1 — Content-addressable transcript cache (P0)

- **Status:** Shipped in `ROADMAP.md` v4.87. The implementation keeps the existing `TranscriptionResult` return shape and adds cache metadata fields instead of returning a tuple.
- **User problem:** Re-running the same workflow on the same clip pays the full Whisper cost every time. A 1 GB interview transcribed twice = 10–20 minutes wasted.
- **Evidence (Verified):** `opencut/core/captions.py:629` (`from faster_whisper import WhisperModel`), no `lru_cache` or content-hash dispatcher anywhere in the module. `footage_index_db.py` is a SEARCH index over already-transcribed segments, not a cache layer. Subagent surveyed; my read confirms.
- **Proposed behavior:**
  - New `opencut/core/transcript_cache.py` module.
  - Key = SHA-256 of `(audio_bytes_xxhash, whisper_model, language, beam_size, vad_filter)`. Use BLAKE3 if available for speed.
  - Store at `~/.opencut/transcript_cache/<key>.json`. Atomic write via `os.replace`.
  - Wrap `transcribe()` to check cache before invoking Whisper and expose cache metadata on the result.
  - Add `/captions/cache/stats` (GET, size + entries + hit rate since boot) and `/captions/cache/clear` (DELETE, CSRF-protected).
  - Cache TTL: indefinite (transcript output is deterministic for fixed inputs). Add an `--invalidate-on-mtime-change` flag if users want stricter behaviour.
- **Implementation areas:** `opencut/core/captions.py` (`transcribe()` wrapper), `opencut/core/transcript_cache.py` (new), `opencut/routes/captions.py` (new stats + clear routes).
- **Data model:** ~/.opencut/transcript_cache/<hash>.json with `{segments, language, model, created_ts, source_path_was}`.
- **Risks:** key drift if Whisper's tokenizer changes between versions. Mitigation: include `faster_whisper.__version__` in the key. Cache-poisoning if user manually edits JSONs — Mitigation: HMAC each entry with a startup-generated key (overkill; ship without first).
- **Verification plan:** new test `tests/test_transcript_cache.py` — hit/miss round-trip, key stability across runs, eviction respects size cap, concurrent writes don't corrupt.
- **Complexity:** M. **Priority: P0.**

### N2 — `missing_dependency()` includes the pip extra name (P0)

- **Status:** Shipped in `ROADMAP.md` v4.88. Live code had no production call sites of `missing_dependency()`, so the closure covers the helper plus the existing generic `safe_error()` and async-job error paths.
- **User problem:** A 503 says "Install the missing package from the Settings tab" but doesn't say which extra (`opencut[tts]`? `opencut[depth]`?) or which CUDA tier.
- **Evidence (Verified):** `opencut/errors.py:144-150` constructor returns `"Install it from the Settings tab under Dependencies."` — no extra-name interpolation.
- **Proposed behavior:**
  - Extend `missing_dependency(name, extra=None, gpu=False, vram_mb=0)` with optional metadata.
  - When `extra` is provided, suggestion becomes `"Install with: pip install 'opencut[{extra}]'"`.
  - When `gpu=True`, append `"GPU-recommended; install CUDA-enabled torch first."`.
  - Adopt across the ~150 sites that raise `missing_dependency()` — start with the 12 most-frequent (silence, captions, demucs, deepfilter, depth, RVM, scene-detect, neural-interp, MusicGen, F5-TTS, Chatterbox, GFPGAN).
  - Per-feature `extra` mapping lives in a single dict at `opencut/core/install_hints.py` so a future model-card sweep can verify completeness.
- **Implementation areas:** `opencut/errors.py` (signature), `opencut/core/install_hints.py` (new dict), ~12 high-traffic call sites in `opencut/core/`.
- **Risks:** None — purely additive. Backward-compatible default keeps the existing string.
- **Verification plan:** `tests/test_missing_dependency_hints.py` — hit each of the 12 sites, confirm the suggestion contains the expected extra name.
- **Complexity:** S. **Priority: P0.**

### N3 — GPU semaphore: default to 30s acquire-wait (P0)

- **Status:** Shipped in `ROADMAP.md` v4.89. Implementation preserves `OPENCUT_GPU_ACQUIRE_TIMEOUT=0` for explicit fail-fast deployments and carries retry metadata through direct errors plus async job status.
- **User problem:** Concurrent AI calls (caption gen + depth) return 429 *instantly* instead of queueing. Panel says "GPU busy" when in fact the previous job would finish in 4 seconds.
- **Evidence (Verified):** `opencut/core/gpu_semaphore.py:51` `ACQUIRE_TIMEOUT = max(0.0, float(os.environ.get("OPENCUT_GPU_ACQUIRE_TIMEOUT") or "0") or 0.0)`. Default is 0 (non-blocking).
- **Proposed behavior:**
  - Change default to **30s**. Env-var override preserved.
  - Add per-request `?wait=N` query param to opt into longer waits (cap 600s).
  - When the timeout *does* hit, return 429 with `Retry-After: <semaphore.acquired_total - active>` header so the panel can show "queued behind 2 jobs, ~45s".
- **Implementation areas:** `opencut/core/gpu_semaphore.py:51` (one-line default), `opencut/routes/jobs_routes.py` (header surface).
- **Risks:** Tests that timed previous 429-immediate behaviour. Mitigation: search for `429`/`assertEqual.*429`/`status_code == 429` in `tests/` (likely 3–5 hits) and update.
- **Verification plan:** new `tests/test_gpu_semaphore_wait.py` — submit 2 concurrent jobs, confirm second one waits up to 30s then succeeds; submit 3 jobs with `?wait=1`, third gets 429 + `Retry-After`.
- **Complexity:** S. **Priority: P0.**

### N4 — Disk preflight wired into heavy routes (P1)

- **Status:** Shipped in `ROADMAP.md` v4.92. `@async_job(..., disk_operation=...)` now runs decorator-level disk preflight for 12 heavyweight caption/audio/export/video-AI jobs and returns HTTP 507 before job creation when the output volume is under budget.
- **User problem:** A render fills the disk mid-write → output corrupted, user loses all in-flight work.
- **Original evidence (Verified):** `opencut/core/disk_monitor.py` had `preflight()`, but the original audit found no route wiring.
- **Proposed behavior:**
  - New `opencut/core/preflight.py` thin façade: `ensure_disk_for(operation, source_path)` returns `(ok: bool, required_mb: int, free_mb: int)`.
  - Per-operation budgets in a dict: `transcribe: 1.5x audio size`, `demucs: 4x input`, `depth: 6x input`, `video_export: source size × 1.2`, etc. (Empirical; ratchet from production telemetry later.)
  - Apply at the start of the 6–8 most-impacting `@async_job` decorators via a new `disk_required_mb=` kwarg.
  - 507 Insufficient Storage when preflight fails, with `{required_mb, free_mb, output_dir}` payload.
- **Implementation areas:** `opencut/core/preflight.py` (new), `opencut/jobs.py` (`@async_job` decorator extension), top 6 affected routes.
- **Risks:** Wrong budgets fail honest jobs. Mitigation: budgets are advisory by default; only the new `disk_required_mb=int` opt-in hard-blocks. Loosen the ratios over time based on telemetry.
- **Verification plan:** `tests/test_disk_preflight.py` — synthetic small-disk fixture confirms 507 with the right body shape; happy-path confirms a normal job runs unchanged.
- **Complexity:** M. **Priority: P1.**

### N5 — Job resume for interrupted jobs (P1)

- **User problem:** A 30-min stem-separation that crashes at minute 28 restarts from zero. Cancellation = data loss.
- **Evidence (Verified):** `opencut/job_store.py:279-291` (`mark_interrupted()` just flips status); `:266-278` (`get_interrupted_jobs()` lists them); no `resume_job()` or similar.
- **Proposed behavior (phased):**
  - Phase 1: persist the *params* (already done — `data` field) plus a new `partial_output_path` field for jobs that wrote a recoverable intermediate (e.g., per-chunk Demucs output, per-segment Whisper output).
  - Phase 2: add a `resumable: bool` flag per `@async_job("type", resumable=True)`. Transcription (segment-wise), stem separation (chunk-wise), depth estimation (frame-wise), batch encoding (file-wise) → all naturally checkpointable.
  - Phase 3: on startup, `GET /jobs/interrupted` surfaces the list; a new `POST /jobs/{id}/resume` re-enqueues with the partial state.
- **Implementation areas:** `opencut/jobs.py` (decorator + state), `opencut/job_store.py` (schema migration), `opencut/routes/jobs_routes.py` (resume endpoint), 4–5 most-impactful core modules.
- **Risks:** Schema migration on existing `~/.opencut/jobs.db`. Mitigation: `ALTER TABLE jobs ADD COLUMN partial_output_path TEXT DEFAULT NULL` — purely additive, no downgrade hazard.
- **Verification plan:** integration test that kills the worker mid-job, restarts the app, calls resume, confirms output equals a fresh run on the same input.
- **Complexity:** L. **Priority: P1.**

### N6 — `GET /webhooks/event-types` discovery endpoint (P1)

- **Status:** Shipped in `ROADMAP.md` v4.90. `/webhooks/event-types` and `/api/webhooks/event-types` return canonical event metadata plus legacy aliases; `/mcp/info` advertises the discovery surface for tooling.
- **User problem:** A webhook author has no way to enumerate the events they can subscribe to. They read `core/webhook_system.py` source.
- **Evidence (Verified):** `grep -rn "GET /webhooks/events" opencut/routes/` returns nothing. The dispatcher fires by string, the catalogue is encoded as constants.
- **Proposed behavior:**
  - New `GET /webhooks/event-types` returning `{events: [{name, since_version, deprecated, schema_pointer}], legacy_aliases: {old: new}}`.
  - Also surface in `GET /mcp/info` (per F146) and OpenAPI for tooling discovery.
- **Implementation areas:** `opencut/core/webhook_system.py` (export a `list_events()`), `opencut/routes/dev_scripting_routes.py` (new GET), `opencut/routes/mcp_bridge_routes.py` (tooling pointer).
- **Risks:** None — additive.
- **Verification plan:** smoke test asserts every event the code can fire is in the list.
- **Complexity:** S. **Priority: P1.**

### N7 — Plugin job-registration API (P1)

- **User problem:** Plugins can register Flask blueprints (per `core/plugins.py`) but can't start `@async_job`-style background work. Plugin authors who need a long-running task have to either spawn raw threads (forbidden per CLAUDE.md gotchas) or expose a route the user must hand-trigger.
- **Evidence:** `grep -n "register_route\|register_job" opencut/core/plugins.py` shows no job-registration symbol. Plugin manifests can declare `http.routes` but not `jobs`. **Verified** by direct file read.
- **Proposed behavior:**
  - Plugins declare `"jobs": [{"id": "my_long_task", "label": "Run X"}]` in `plugin.json`.
  - Loader exposes `plugin.job(id)` decorator that uses the existing `@async_job` machinery but namespaces the job-type as `plugin::<plugin_id>::<job_id>` for telemetry separation.
  - Plugin-launched jobs inherit the plugin's capability scope (no `host.filesystem` → reject path arguments outside the plugin's data dir).
- **Implementation areas:** `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py` (schema), `opencut/jobs.py` (namespacing).
- **Risks:** Plugins that abuse the API for crypto mining etc. Mitigation: same capability-declaration + lock-file model; users opt in via `OPENCUT_PLUGIN_ALLOW_UNSIGNED=1` for unsigned.
- **Verification plan:** add a 3rd example plugin (`example_plugins/long-job-demo/`) that registers a job; integration test confirms the job runs via `/plugins/<name>/start`.
- **Complexity:** M. **Priority: P1.**

### N8 — Third-party agent-skill loader (P1)

- **User problem:** `core/agent_skills.py` only loads from `opencut/data/builtin_skills/<id>/`. The wedding-cinematic-reel is the only example. No path for community skills.
- **Evidence (Verified by prior loop):** subagent + prior session work on F143/F272. `agent_skills.py::list_builtin_skills()` is the only loader.
- **Proposed behavior:**
  - Add `list_user_skills()` that scans `~/.opencut/skills/<id>/SKILL.md` (+ `plan.json`).
  - Combine into `list_skills()` with `source: "builtin" | "user"` per entry.
  - `validate_skill_plan()` already enforces a schema — apply the same to user skills.
  - Wire into `/agent/skills` (the existing F272 catalogue route) without breaking the manifest shape.
- **Implementation areas:** `opencut/core/agent_skills.py`, `opencut/routes/agent_skills.py`.
- **Risks:** User skills referencing endpoints the host doesn't ship → skill validation must reject at load time, not at run time.
- **Verification plan:** `tests/test_user_skills.py` writes a fixture skill into a tempdir, monkeypatches the skills-dir, confirms the catalogue + plan validation.
- **Complexity:** M. **Priority: P1.**
- **Status:** Shipped in `ROADMAP.md` v4.95 with validated user skill loading from `~/.opencut/skills/<id>/`, route-manifest endpoint checks, `source` catalogue metadata, `/agent/skills` wiring, and `docs/SKILL_AUTHORING.md`.

### N9 — Enriched job metadata (P2)

- **User problem:** When a 4 GB job fails, the only artifact is the error string. No `peak_vram_mb`, no `exit_reason` enum, no resource timeline.
- **Evidence (Verified):** `opencut/job_store.py:197-200` records `started_at`/`completed_at`; no resource fields.
- **Proposed behavior:**
  - Schema migration to add `peak_vram_mb INT`, `peak_cpu_pct INT`, `peak_rss_mb INT`, `exit_reason TEXT` (enum: `complete | error | cancelled | interrupted | oom | timeout | preflight_failed`).
  - Background sampler in `@async_job` polls `psutil` (already a core dep) every 5s.
  - Surface in `GET /jobs/{id}` and the UI job-history table.
- **Implementation areas:** `opencut/job_store.py`, `opencut/jobs.py`, `opencut/core/job_diagnostics.py`.
- **Risks:** psutil VRAM only via nvml; gate behind `_try_import('pynvml')` with graceful degradation.
- **Verification plan:** `tests/test_job_metadata.py` mocks `psutil` + `pynvml`, confirms the fields populate.
- **Complexity:** M. **Priority: P2.**

### N10 — Request-ID propagation into subprocess stderr (P2)

- **User problem:** A 10-min job's logs cannot be walked end-to-end via `request_id` — FFmpeg/Whisper subprocesses dump to stderr without it.
- **Evidence (Verified):** `opencut/core/request_correlation.py` installs a filter on the OpenCut logger only.
- **Proposed behavior:**
  - Tag each subprocess child via env var: `OPENCUT_REQUEST_ID=<id>`.
  - Wrap subprocess stderr/stdout streams so each emitted line is prefixed with `[r-<id>]` before reaching the logger.
  - Implementation hooks: `opencut/helpers.py::run_ffmpeg` (the central FFmpeg invoker), `opencut/jobs.py` `@async_job` thread-local propagation.
- **Implementation areas:** `opencut/helpers.py`, `opencut/core/request_correlation.py`.
- **Risks:** Prefix munging breaks downstream FFmpeg-progress parsers. Mitigation: prefix on the logging path only, not the parser path (run_ffmpeg already splits the streams).
- **Complexity:** S. **Priority: P2.**

---

## Existing Feature Improvements

### E11 — Webhook signature secret should be mandatory (P1)

- **Status:** Shipped in `ROADMAP.md` v4.91. `POST /api/webhooks` now requires a non-empty HMAC secret unless the caller explicitly passes `?allow_unsigned=true`; unsigned configs create `~/.opencut/webhooks_unsigned.txt` once.
- **Original behavior:** `core/webhook_system.py` shipped HMAC support, but the `secret` field was optional. Unsigned webhooks were accepted.
- **Problem:** A misconfigured webhook is delivered without proof of origin; downstream receivers can't authenticate.
- **Recommendation:** Require non-empty `secret` on `POST /api/webhooks` by default. Add `?allow_unsigned=true` opt-in for explicit local testing.
- **Code locations:** `opencut/core/webhook_system.py::WebhookConfig`, `opencut/routes/dev_scripting_routes.py`.
- **Backward compat:** Existing unsigned webhooks keep working until next restart; a one-time `webhooks_unsigned.txt` warning in `~/.opencut/`.
- **Complexity:** S. **Priority: P1.**

### E12 — Workflow engine endpoint allowlist should be derived, not hard-coded (P2)

- **Status:** Shipped in `ROADMAP.md` v4.99. Workflow-safe async POST routes now opt in with `workflow_step(...)`; `opencut/_generated/route_manifest.json` carries `workflow.label` metadata for 53 steps; `KNOWN_ENDPOINTS` loads from the manifest; and route-manifest drift checks catch metadata changes.
- **Original behavior:** `opencut/core/workflow.py` carried a hand-maintained `KNOWN_ENDPOINTS` dict of ~80 routes. Adding a new step type required editing this dict.
- **Problem:** When a wave letter ships new routes (we just shipped 42 in Wave Q+R+S), the workflow engine could not see them until a maintainer remembered to update the dict.
- **Recommendation:** Derive the allowlist from `opencut/_generated/route_manifest.json` at import time, intersected with a per-route "workflowable" annotation (a `@workflow_step` decorator on routes that opt in).
- **Code locations:** `opencut/core/workflow.py`, `opencut/routes/__init__.py` (decorator), affected route handlers.
- **Backward compat:** Pure additive — `KNOWN_ENDPOINTS` becomes a fallback for routes that haven't migrated.
- **Complexity:** M. **Priority: P2.**

### E13 — CLI surface parity with HTTP API (P2)

- **Status:** Shipped in `ROADMAP.md` v4.100. `opencut route METHOD PATH` now validates against the generated route manifest, shapes query and JSON request bodies, fetches CSRF tokens automatically for mutating calls, and prints formatted or raw backend responses.
- **Original behavior:** `opencut/cli.py` exposed ~10 commands; HTTP API had 1,517 routes.
- **Problem:** Power users who want to script OpenCut outside Premiere have a CLI-shaped hole. Many ops (queue management, MCP, plugins, workflows) are HTTP-only.
- **Recommendation:** Auto-generate `opencut route GET /system/check-failures` style escape hatch from `route_manifest.json` — same pattern as F194 extended MCP tools.
- **Code locations:** `opencut/cli.py`.
- **Complexity:** M. **Priority: P2.**

### E14 — F236 CEP parity (P1; tracked, scope here)

- **Original behavior:** F236 caption display-settings card shipped in UXP (`8d7ebd2`); CEP captions tab still lacked it.
- **Recommendation:** Port the same card pattern into `extension/com.opencut.panel/client/index.html` and the CEP `main.js` event wiring. Reuses the same backend routes.
- **Complexity:** S. **Priority: P1.** (deadline 2026-08-17 already met via UXP; CEP parity is polish, not deadline-gated.)
- **Status:** Shipped in `ROADMAP.md` v4.96 with the CEP Captions-tab FCC card, token-schema loading, preview/reset wiring, preview CSS, and `tests/test_cep_caption_display_settings_ui.py`.

### E15 — i18n migration continuation (P2)

- **Current state:** Advanced in `ROADMAP.md` v4.172 to 1,022 guarded keys across 74 rounds. `i18n-drift` reports 1,430 keys, 1,304 consumers, 126 dead keys, and 0 missing.
- **Recommendation:** Continue rolling 5–10-string batches per loop. ~85 bare-English sites remain.
- **Complexity:** S per batch.

---

## Reliability, Security, Privacy, and Data Safety

| Risk | Severity | Evidence | Owner |
|---|---|---|---|
| GPU semaphore returns immediate 429 instead of queueing | P0 | `gpu_semaphore.py:51` | N3 |
| No disk preflight on heavy routes | P1 | `disk_monitor.preflight()` unreferenced in routes | N4 |
| Jobs interrupted at minute 28 of 30 must restart | P1 | `job_store.py:279-291` | N5 |
| Webhook signature optional by default | P1 | `webhook_system.py::WebhookConfig.secret default ""` | E11 |
| `missing_dependency` suggestion is generic | P0 | `errors.py:144-150` | N2 |
| No transcript cache → repeated Whisper cost | P0 | `core/captions.py` no cache hooks | N1 |
| Subprocess stderr not request-ID-tagged | P2 | `request_correlation.py` filter scope | N10 — shipped v4.98 |
| Plugin capabilities declared but enforcement light | P2 | `plugins.py` doesn't intercept `open()`/`requests.get()` | (deferred — needs a runtime trace) |
| No `peak_vram_mb` / `exit_reason` per job | P2 | `job_store.py` schema | N9 — shipped v4.97 |
| Untrusted plugin manifest could declare jobs | P2 (when N7 lands) | Future — gate behind capability check | N7 verification |

**Recovery & rollback needs**

- Job-resume (N5) is the headline reliability ask.
- Disk preflight (N4) preempts the most-painful "corrupted output" mode.
- Webhook delivery: today drops on restart (no persistent queue). Promotion to a SQLite-backed dead-letter queue is the natural pair with E11.

**Logging & diagnostics needs**

- Request-ID propagation (N10, shipped v4.98).
- `peak_vram_mb` + `exit_reason` (N9, shipped v4.97).
- Eventually: structured-log JSON output to a configurable log sink (Sentry/GlitchTip already present per F250; the structured-log writer needs to reach FFmpeg too).

---

## UX, Accessibility, and Trust

### Onboarding gaps

- **Verified:** First-Run Wizard ships in `index.html:3991` + `main.js:10518-10580` (subagent claim that it was missing was wrong). Confirm it triggers on first launch (`localStorage.getItem("opencut_wizard_seen")` check or similar).
- **Real gap:** the `<span id="workspaceClipStatus">Choose media to begin</span>` empty state on `index.html:71` has no affordance. After the wizard closes, a user with no clip selected sees only that span. Add a clip-picker chip + tooltip.

### Settings clarity

- `MISSING_DEPENDENCY` errors are the biggest settings-tab friction. N2 fixes it.

### Microcopy & trust signals

- README badges (per the subagent): add CodeQL + signed-release badges. **Caveat:** GitHub Actions on this org has billing constraints (per `sysadmindoc-actions-billing` memory), so CodeQL may not actually run. Verify before promising the badge.
- The `$1,400/year` positioning lead on the README (F270) is strong — keep.

### Destructive actions

- `/models/delete` and `/system/temp-cleanup/sweep` already require CSRF. `/plugins/uninstall` should require a name re-type confirm step (Q5 from prior plan noted this; still open).

---

## Architecture and Maintainability

### Refactor candidates surfaced during this pass

- `opencut/core/captions.py` is the canonical place for a transcript-cache wrapper. Don't sprinkle caching into individual `_transcribe_*` functions; wrap the dispatcher.
- `opencut/core/workflow.py::KNOWN_ENDPOINTS` is the kind of hand-maintained allowlist that drifts; E12 now derives it from `route_manifest.json`.
- `opencut/core/plugins.py` should grow a `register_plugin_job(plugin_id, job_id, fn)` API; the existing route-registration shape is the pattern.
- `opencut/jobs.py` is the right home for `peak_vram_mb` sampling and `exit_reason` enum — keep job metadata centralised, not per-route.

### Test gaps

- Cold-start latency budget: a smoke test like `tests/test_cold_start_budget.py` that imports `opencut.server.create_app()` and asserts < 3.0s on a synthetic-clean env. The agent claimed cold-start was a problem; my verification showed it is not. Add the gate to lock that in.
- `tests/test_transcript_cache.py` — for N1.
- `tests/test_gpu_semaphore_wait.py` — for N3.
- `tests/test_disk_preflight.py` — for N4.

### Documentation gaps

- `CONTRIBUTING.md` documents the route-pattern conventions well. N7 added `docs/PLUGIN_AUTHORING.md`; N8 added `docs/SKILL_AUTHORING.md` for third-party skills.
- The 47 model cards (`docs/MODELS.md`) need a per-card "install hint" pointer once N2 lands.

### Release & build gaps (unchanged from prior plan)

- F202 macOS notarization secrets still pending.
- F252 UXP WebView cutover — months-scale.
- F101 Windows ARM64 — documented, not packaged.

---

## Prioritized Roadmap

### Phase 0 — Reliability and trust quick wins (this week)

- [x] **P0 — N1 transcript content-addressable cache**
  - Why: Single biggest user-visible perf win.
  - Evidence: `opencut/core/captions.py` no cache; subagent + verified.
  - Touches: `opencut/core/transcript_cache.py` (new), `opencut/core/captions.py` (wrap), `opencut/routes/captions.py` (stats+clear routes).
  - Acceptance: same source bytes/settings hit a persisted SHA-256 cache entry; cache stats endpoint returns hit/miss/write counters.
  - Verify: `py -3.12 -m pytest tests/test_transcript_cache.py -q -p no:cacheprovider -o addopts=""`.

- [x] **P0 — N2 `missing_dependency()` includes pip extra name**
  - Why: 30-minute install-fumbling sessions disappear.
  - Evidence: `opencut/errors.py:144-150`.
  - Touches: `opencut/errors.py`, `opencut/core/install_hints.py` (new), ~12 high-traffic call sites.
  - Acceptance: every top-12 dependency surface resolves to `pip install 'opencut[<extra>]'` in the suggestion; async jobs carry the same hint in job status.
  - Verify: `py -3.12 -m pytest tests/test_missing_dependency_hints.py -q -p no:cacheprovider -o addopts=""`.

- [x] **P0 — N3 GPU semaphore default-wait 30s**
  - Why: 429 floods are UX bugs, not capacity guards.
  - Evidence: `opencut/core/gpu_semaphore.py:51`.
  - Touches: `opencut/core/gpu_semaphore.py`, `opencut/routes/jobs_routes.py` (Retry-After).
  - Acceptance: a contended GPU acquire waits for a released slot by default; explicit zero-second override remains non-blocking; timeout responses/status include retry metadata.
  - Verify: `py -3.12 -m pytest tests/test_gpu_semaphore_wait.py -q -p no:cacheprovider -o addopts=""`.

- [x] **P1 — N6 `GET /webhooks/event-types`**
  - Why: webhook authors can't enumerate without reading source.
  - Touches: `opencut/core/webhook_system.py`, `opencut/routes/dev_scripting_routes.py`, `opencut/routes/mcp_bridge_routes.py`.
  - Acceptance: GET returns `{events: [...], legacy_aliases: {...}}`; smoke test asserts every fired event is listed.

- [x] **P1 — E11 webhook signatures mandatory by default**
  - Why: unsigned deliveries can't be authenticated downstream.
  - Touches: `opencut/core/webhook_system.py`, `opencut/routes/dev_scripting_routes.py`.
  - Acceptance: `POST /api/webhooks` without `secret` returns 400 unless `?allow_unsigned=true`.

### Phase 1 — Resource and recovery (next 2 weeks)

- [x] **P1 — N4 disk preflight in heavy routes**
  - Why: prevents the worst-case "render fills disk, corrupts output" mode.
  - Touches: `opencut/core/preflight.py` (new), `opencut/jobs.py`, 12 affected route jobs.
  - Acceptance: heavy routes return 507 with `{required_mb, free_mb, output_dir}` on synthetic small-disk fixture; honest jobs unchanged.

- [x] **P1 — N5 job resume for interrupted jobs**
  - Why: 30-min job that crashes at minute 28 doesn't restart from zero.
  - Touches: `opencut/jobs.py`, `opencut/job_store.py` (schema migration), `opencut/routes/jobs_routes.py`, top 4 chunkable jobs (transcribe/demucs/depth/batch-encode).
  - Acceptance: kill worker mid-job, restart, `POST /jobs/{id}/resume`, output equals fresh-run baseline.
  - Status: closed in ROADMAP v4.93 with resumable async-job metadata, durable running-job persistence, `POST /jobs/<job_id>/resume`, and route coverage for captions/transcript/WhisperX, Demucs separation, export/export-preset, and depth-estimate-v2.

- [x] **P1 — N7 plugin job-registration API**
  - Why: plugin authors can register routes but not background work.
  - Touches: `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py`, `opencut/jobs.py`, add `example_plugins/long-job-demo/`.
  - Acceptance: example plugin's `/plugins/long-job-demo/start` enqueues a real `@async_job`-tracked job.
  - Status: closed in ROADMAP v4.94 with `jobs.register`, `plugin_job(...)`, manifest/decorator cross-checks, filesystem-scope enforcement, and `opencut/data/example_plugins/long-job-demo/`.

- [x] **P1 — N8 third-party agent-skill loader**
  - Why: skills were built-in only at audit time.
  - Touches: `opencut/core/agent_skills.py`, `opencut/routes/agent_skills.py`.
  - Acceptance: dropping a `SKILL.md` + `plan.json` into `~/.opencut/skills/<id>/` lists it in `GET /agent/skills`.
  - Status: closed in ROADMAP v4.95 with validated user-skill loading, generated route-manifest endpoint checks, combined catalogue source metadata, and authoring docs.

- [x] **P1 — E14 F236 CEP parity**
  - Why: UXP got the FCC card first; CEP lacked it at audit time.
  - Touches: `extension/com.opencut.panel/client/{index.html,main.js,style.css}`.
  - Acceptance: identical card surfaces in CEP captions tab; uses the same backend routes.
  - Status: closed in ROADMAP v4.96 with CEP token loading, live preview, and static parity tests.

### Phase 2 — Observability & extensibility polish

- [x] **P2 — N9 enriched job metadata (peak_vram_mb, exit_reason)**
  - Touches: `opencut/job_store.py`, `opencut/jobs.py`, `opencut/core/job_diagnostics.py`.
  - Status: closed in ROADMAP v4.97 with persisted peak resource fields, explicit terminal exit reasons, resource sampling, diagnostics metadata, generated surface refreshes, and `GET /jobs/<job_id>`.

- [x] **P2 — N10 request-ID propagation into subprocess stderr**
  - Touches: `opencut/helpers.py::run_ffmpeg`, `opencut/core/request_correlation.py`.
  - Status: closed in ROADMAP v4.98 with async-worker request-ID restoration, `OPENCUT_REQUEST_ID` subprocess env tagging, request-prefixed FFmpeg stderr logging, and parser-safe raw stderr returns.

- [x] **P2 — E12 workflow allowlist derived from route manifest**
  - Status: closed in ROADMAP v4.99 with per-route workflow metadata, explicit async POST opt-ins, manifest-derived validation, and metadata drift checks.
  - Touches: `opencut/core/workflow.py`, `opencut/routes/__init__.py`.

- [x] **P2 — E13 CLI surface parity escape hatch**
  - Status: closed in ROADMAP v4.100 with a manifest-validated `opencut route METHOD PATH` client, JSON/query request shaping, automatic CSRF handling, and focused CLI tests.
  - Touches: `opencut/cli.py`.

- [ ] **P2 — E15 i18n migration rolling batches**
  - Status: advanced in ROADMAP v4.172 with an additional journal/media metadata label batch; keep open for the next high-impact string batch.
  - Touches: `extension/com.opencut.panel/client/{main.js,locales/en.json}`, `tests/test_i18n_hardcoded_migration.py`.

### Phase 3 — Deadline-gated (unchanged from prior plan)

- [ ] **P0 — F202 macOS notarization** (2026-09-01).
- [ ] **P0 — F252 UXP WebView cutover** (~2026-09).
- [ ] **P1 — F146 UXP MCP transport polish** (shipped; further hardening optional).

---

## Quick Wins

| Item | Effort | Impact | Owner |
|---|---|---|---|
| N2 — pip extra in error suggestion | 1 hr | High — fixes 30-min user fumble | `errors.py` |
| N3 — semaphore default 30s | 30 min | High — fixes 429 UX bug | `gpu_semaphore.py:51` |
| N6 — webhook event-type discovery | 1 hr | Medium — enables tooling | new route |
| N10 — subprocess request-ID prefix | 2 hr | Medium — debugging clarity | `helpers.py` |
| Cold-start budget test | 1 hr | Low — locks in current state | new test |
| `<span id="workspaceClipStatus">` affordance | 1 hr | Medium — first-time UX | `index.html:71` |

---

## Larger Bets

- **N1 transcript cache.** Shipped in `ROADMAP.md` v4.87; every Whisper caller now shares the core persistent cache.
- **N5 job resume.** Shipped in `ROADMAP.md` v4.93 with incremental per-route opt-in for the checkpointable caption, Demucs, export, and depth jobs; future work can deepen per-job checkpoint formats.
- **N7 + N8 plugin/skill extensibility.** Both halves are shipped: plugin background jobs use capability-gated `plugin_job(...)`, and user skills are loaded only after schema and generated route-manifest validation. The remaining extensibility work is marketplace/review UX, not the local registration path.

---

## Explicit Non-Goals

- **Marketplace UI.** N7/N8 add the registration paths; the marketplace itself is a separate, larger product surface. Defer until at least 5 community plugins/skills exist.
- **Snapshot-per-step in F143.** Already designed; outside this pass's scope.
- **Telemetry beyond Aptabase.** The opt-in usage signals are correct; adding event-level structured logs to a self-hosted GlitchTip is a separate F-numbered item, not in scope here.
- **Cold-start optimisation.** I verified it's not actually a problem (subagent claim was wrong); a regression test locks the current state instead.
- **i18n breadth in one mega-PR.** Continue rolling 5–10-string batches; don't fork the whole migration into a single PR.

---

## Open Questions

1. **What's the realistic Whisper baseline on this machine?** N1's 200ms cache-hit target assumes a faster cold path than typical. Needs a 5-minute benchmark on a 1-min, 10-min, and 60-min clip to set the acceptance criterion. **Needs live validation.**
2. **Is the `OPENCUT_GPU_ACQUIRE_TIMEOUT=0` default intentional?** The doc-comment at `gpu_semaphore.py:24` describes it as "seconds a request waits for a slot" — singular form suggests the design intent was a wait, not an instant reject. Worth confirming with the original author before changing the default. **Likely the default is unintentional**, but ask before flipping.
3. **Are the existing 4 audit-batch commits' allowlist values (e.g., `safe_int` ranges) preserved on schema migrations?** N9's `job_store.py` migration is purely additive (`ALTER TABLE ... ADD COLUMN`), so backwards-compatible, but worth a future `PRAGMA user_version` migration pass.
4. **Should plugin-registered jobs share the global `MAX_CONCURRENT_JOBS=10` cap, or have a per-plugin sub-cap?** Pick before N7 ships; default to global cap for simplicity, add per-plugin later when a real plugin abuses the cap.

---

## Appendix A — Subagent claim audit (this pass)

| Claim | Subagent verdict | My verification | Action |
|---|---|---|---|
| Heavy ML imports at module level break cold start | Yes, P0 | **WRONG** — Whisper is properly lazy at `captions.py:516/629/806` | Removed from plan. Replaced with "lock current good state via cold-start budget test." |
| First-Run Wizard advertised but missing from main.js | Yes, P1 | **WRONG** — Wizard wired at `main.js:1690, 10518-10580` | Removed. Real gap is the no-clip-selected empty state. |
| `Install.bat` doesn't auto-escalate on non-admin | Yes, P0 | **WRONG** — `Start-Process -Verb RunAs` is correct shape | Removed. Worth a separate live-run check on a clean Windows VM. |
| `telemetry_aptabase.py` exists at that path | Yes | **PARTIAL** — file path wrong; integration shape correct | Re-verify before any further telemetry recommendation. |
| `gpu_semaphore.py:51` defaults to `ACQUIRE_TIMEOUT=0` | Yes, P0 | **CORRECT** | Plan item N3. |
| `disk_monitor.preflight()` exists but routes don't call it | Yes, P1 | **SHIPPED** in `ROADMAP.md` v4.92 after the original audit found no route wiring. | Plan item N4. |
| `mark_interrupted()` flips status only; no resume | Yes, P1 | **CORRECT** | Plan item N5. |
| Webhook event types hard-coded; no discovery endpoint | Yes, P2 | **SHIPPED** in `ROADMAP.md` v4.90 after the original audit found no discovery route. | Plan item N6. |
| Skills are built-in only | Yes, P2 | **CORRECT AT AUDIT TIME** | Plan item N8, now shipped in v4.95. |
| Plugin job-registration not wired | Implied | **CORRECT** (grep shows no `register_job` in `plugins.py`) | Plan item N7. |
| `missing_dependency` suggestion is generic | Yes, P0 | **CORRECT** (`errors.py:144-150`) | Plan item N2. |

**Lesson for future passes:** subagents reading a 15k-line `main.js` cold are reliable for shape claims but unreliable for "is feature X wired" claims. Always grep-verify before citing.
