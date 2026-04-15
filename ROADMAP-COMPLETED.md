# OpenCut — Completed Roadmap Items

## Completed Work (v1.0 - v1.9.26)

The original Phases 0-6 are complete. Summary of what shipped:

| Phase | What Shipped | Status |
|-------|-------------|--------|
| **0.1** Test coverage | 867 tests, 22 test files, JSX mock harness, CI enforcement | DONE |
| **0.2** Structured errors | `errors.py` with 13 error types, `safe_error()` auto-classification, frontend `enhanceError()` | DONE |
| **0.3** Structured logging | JSON file handler, `[job_id]` correlation, `/logs/tail` endpoint, log levels audited | DONE |
| **1.2** Job system | SQLite persistence, priority queue (WorkerPool), interrupted job recovery, job stats | DONE |
| **2.1** Build system | Vite config, package.json, tsconfig.json scaffolded | DONE |
| **3.1** Workflow engine | 6 built-in presets, server-side step execution, output chaining, cancellation | DONE |
| **3.2** Contextual awareness | 35-feature scoring, `classify_clip()`, guidance messages, smart tab reordering | DONE |
| **3.3** Preview before commit | Waveform preview, cut review panel, side-by-side color preview | DONE |
| **3.4** Keyboard shortcuts | 8 default bindings, configurable registry, reference card, command palette (30+ entries) | DONE |
| **4.1** Dependency resolution | 3-tier `safe_pip_install()`, `--target` fallback, post-install verification | DONE |
| **4.2** Docker | Multi-stage Dockerfile, docker-compose with GPU variant, `.dockerignore` | DONE |
| **4.3** Health monitoring | `/system/status`, status bar (CPU/RAM/GPU/jobs), exponential backoff reconnect | DONE |
| **5.1** Lazy tab rendering | 5 heavy tabs deferred, `_tabRendered` tracking | DONE |
| **5.2** Response streaming | NDJSON generators, batched/per-item/progress modes | DONE |
| **5.3** Background indexing | SQLite FTS5 at `~/.opencut/footage_index.db`, WAL mode, auto-index | DONE |
| **5.4** Parallel processing | ThreadPoolExecutor batch processing, GPU/CPU worker separation | DONE |
| **6.1** Plugin system | Plugin loader, manifest validation, 2 example plugins, dynamic blueprint registration | DONE |
| **6.2** i18n | `t()` function, `data-i18n` on ~200 elements, `en.json` with 417 keys | DONE |
| **6.3** Preset export/import | `.opencut-preset` JSON export/import, settings bundling | DONE |
| **6.4** Project templates | 6 built-in templates, save/apply, dropdown UI | DONE |
| **Competitive upgrades** | BiRefNet default, Whisper turbo, distil models, audio-separator, ClearerVoice, CodeFormer, InsightFace, ACE-Step, Chatterbox, AI LUT, ProPainter, SeamlessM4T, BasicVSR++, PySceneDetect | DONE |
| **35 bug-fix batches** | 600+ bugs fixed across 29 audit rounds, full codebase hardening | DONE |

**What remains from the original roadmap:**

| Item | Phase | Status | Notes |
|------|-------|--------|-------|
| FastAPI migration | 1.1 | PLANNED | Big effort, mostly internal benefit. Defer until Wave 3. |
| Process isolation (GPU) | 1.3 | PLANNED | Eliminates OOM crashes. Critical for heavy AI features. |
| TypeScript migration | 2.2 | SCAFFOLDED | tsconfig.json exists. Incremental migration ongoing. |
| UXP full parity | 7.1-7.3 | IN PROGRESS | UXP panel at ~85% parity. CEP end-of-life ~Sept 2026. |

---

