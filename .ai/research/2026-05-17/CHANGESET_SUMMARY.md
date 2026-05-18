# OpenCut — Changeset Summary (2026-05-17 research run)

**Run scope:** Autonomous deep research, memory consolidation, and roadmap planning. Read-only on existing code (no functional changes); created / appended documentation artefacts only.

---

## Pass 63 addendum (2026-05-18)

Pass 63 closed **F256** by adding UXP Transcript API host helpers to `extension/com.opencut.uxp/main.js`. The new bridge surface wraps `Transcript.querySupportedLanguages()`, resolves clip project items before `Transcript.hasTranscript()`, optionally calls `Transcript.exportToJSON()`, and exposes both helpers through `window.OpenCutUXPHost` for the WebView cutover.

Added `tests/test_uxp_transcript_api_integration.py` and registered it in release-smoke `pytest-fast`. Updated ROADMAP.md v4.66, CHANGELOG.md, PROJECT_CONTEXT.md, docs/UXP_MIGRATION.md, and the Pass-2 backlog/state files. Validation passed: focused F256 tests (`5 passed`), focused UXP/release-smoke slice (`37 passed`), `py_compile`, focused Ruff, `node --check`, and release-smoke `pytest-fast` (`642 passed`).

---

## Pass 64 addendum (2026-05-18)

Pass 64 closed **F257** by adding UXP Object Mask state helpers to `extension/com.opencut.uxp/main.js`. The bridge wraps `ObjectMaskUtils.hasObjectMask(projectOrSequence)`, supports active-sequence and project scopes, returns explicit unavailable/no-target responses, and exposes `getObjectMaskState(payload)` through `window.OpenCutUXPHost`.

Added `tests/test_uxp_object_mask_api_integration.py` and registered it in release-smoke `pytest-fast`. Updated ROADMAP.md v4.67, CHANGELOG.md, PROJECT_CONTEXT.md, docs/UXP_MIGRATION.md, and the Pass-2 backlog/state files. Validation passed: focused F257 tests (`5 passed`), focused UXP/release-smoke slice (`42 passed`), `py_compile`, focused Ruff, `node --check`, and release-smoke `pytest-fast` (`647 passed`).

---

## Pass 65 addendum (2026-05-18)

Pass 65 closed **F258** by adding UXP AAF export helpers to `extension/com.opencut.uxp/main.js`. The bridge wraps `ProjectConverter.exportAAF(sequence, filePath, aafExportOptions?)`, builds optional `AAFExportOptions` from payload settings, maps AIFF/WAV constants, validates output paths, and exposes `exportAafSequence(payload)` through `window.OpenCutUXPHost`.

Added `tests/test_uxp_aaf_export_integration.py` and registered it in release-smoke `pytest-fast`. Updated ROADMAP.md v4.68, CHANGELOG.md, PROJECT_CONTEXT.md, docs/UXP_MIGRATION.md, and the Pass-2 backlog/state files. Validation passed: focused F258 tests (`6 passed`), focused UXP/release-smoke slice (`48 passed`), `py_compile`, focused Ruff, `node --check`, and release-smoke `pytest-fast` (`653 passed`).

---

## Pass 66 addendum (2026-05-18)

Pass 66 closed **F260** by adding a generated UXP migration risk dashboard. `build_dashboard_manifest()` derives summary counts, risk counts, hybrid candidates, priority rows, and per-host-action rows from the F198 parity catalogue. `dump_uxp_migration_dashboard` writes both the repository generated artifact and bundled panel JSON.

The UXP Settings tab now loads `uxp-migration-dashboard.json`, summarizes direct UXP coverage, CEP fallbacks, and high-risk actions, and renders per-action status and replacement plans. Added `tests/test_uxp_migration_dashboard.py` and release-smoke registration. Validation passed: focused F260 tests (`7 passed`), focused UXP/release-smoke slice (`55 passed`), `py_compile`, focused Ruff, dashboard sync check, `node --check`, and release-smoke `pytest-fast` (`660 passed`).

---

## Pass 67 addendum (2026-05-18)

Pass 67 closed **F267** by adding a generated UXP Developer Tool smoke harness for the 14 direct-UXP host actions. `build_udt_harness_manifest()` derives action coverage from the F198 parity catalogue and attaches scenario payloads, fixture requirements, safety flags, expected result keys, and acceptable environment blockers. `dump_uxp_udt_harness` writes both the repository generated artifact and bundled panel JSON.

The UXP panel now loads `udt-smoke.js`, exposing `window.OpenCutUXPUdtHarness.run()` for safe-by-default UDT runs and `run({ includeMutating: true })` for disposable-project mutation/file-write coverage. Added `tests/test_uxp_udt_harness.py` and release-smoke registration. Validation passed: focused F267 tests (`6 passed`), focused UXP/release-smoke slice (`61 passed`), `py_compile`, focused Ruff, harness sync check, `node --check`, and release-smoke `pytest-fast` (`666 passed`).

---

## Pass 68 addendum (2026-05-18)

Pass 68 closed **F263** by adding `opencut.tools.pip_audit_extras`, a structured Python dependency audit wrapper that covers both `requirements.txt` and `pyproject[all]`. The wrapper builds temporary requirements files, isolates pip/pip-audit cache directories, emits per-target JSON, and reports allowed vs unallowed vulnerabilities so release smoke can fail only on undocumented advisories.

The pass also refreshed stale optional dependency pins so `[all]` resolves: `transnetv2-pytorch>=1.0.5,<2`, `auto-editor>=29.3,<30`, `opentimelineio>=0.17,<1`, `otio-aaf-adapter>=2.0,<3`, and `pyannote.audio>=4.0,<5`. AudioCraft/MusicGen and Resemble Enhance remain explicit Python 3.11 extras outside `[all]` because their published packages hard-pin older Torch stacks. Added `docs/PYTHON_ADVISORIES.md` for the BasicSR and Transformers advisories currently reported by `pyproject[all]`. Validation passed: focused F263 tests (`34 passed`), focused Ruff, `py_compile`, model-card/readiness checks, live `pip_audit_extras --json --extra all`, and release-smoke `--only pip-audit`.

---

## Pass 69 addendum (2026-05-18)

Pass 69 closed **F271** by extending the F100/F191 feature-readiness contract with per-feature hardware and minimum-VRAM metadata. `FeatureRecord` and `/system/feature-state` now include `hardware`, `requires_gpu`, and `minimum_vram_mb`; `dump_feature_readiness` copies model-card hardware strings and parses `>= N GB/MB VRAM` into structured megabytes.

The CEP panel helper now exposes `hardwareFor(featureId)` and annotates `data-feature-id` controls with `data-feature-hardware`, `data-feature-min-vram-mb`, and tooltip hardware summaries. The merge path also enriches hand-written stub records by `feature_id`, so roadmap/stub rows inherit generated model-card hardware metadata when available. Validation passed: focused F271 tests (`34 passed`), focused Ruff, `py_compile`, `node --check`, and feature-readiness sync check.

---

## Pass 70 addendum (2026-05-18)

Pass 70 closed **F272** by adding the first concrete built-in agent skill package. `opencut/data/builtin_skills/wedding-cinematic-reel/` now ships a SKILL.md front-matter manifest plus `plan.json` for a 240-second wedding reel workflow that chains color match, music beat markers, highlight extraction, beat-synced assembly, and review-master export planning.

`opencut.core.agent_skills` parses and validates built-in skill packages, while `GET /agent/skills` and `GET /agent/skills/<skill_id>` expose compact and full skill payloads to the agent surface. Generated artifacts were refreshed to 1,376 routes and 1,319 opt-in extended MCP tools. Validation passed: focused F272 tests (`4 passed`), focused route/skill tests (`17 passed`), focused Ruff, `py_compile`, generated route/MCP checks, roadmap lint, and release-smoke `pytest-fast` (`691 passed`).

---

## Pass 71 addendum (2026-05-18)

Pass 71 closed **F193** by replacing the legacy OpenAPI endpoint hand-table with dataclass-discovered response schema bindings. `opencut.openapi_registry` now discovers schema/core dataclasses with route metadata and feeds `opencut.openapi._ENDPOINT_SCHEMAS`; `opencut.openapi` resolves nested dataclasses, optional/union types, list/tuple payloads, `Path`, and computed response properties.

The pass added typed core-result coverage for audio-description drafts, audio-description generation, delivery transfer bundles, marker imports, eval dataset details, crash packets, project health, OCIO validation, review bundles, and C2PA sidecars. The extended MCP manifest still contains 1,319 tools and now reports 100 response-schema annotations. Validation passed: focused OpenAPI tests (`6 passed`), focused OpenAPI+MCP tests (`15 passed`), focused Ruff (`E,F,I`), `py_compile`, roadmap lint, extended MCP sync check, and release-smoke `pytest-fast` (`693 passed`).

---

## 1. Files created

### Repo root (1 file)

| Path | Purpose |
|---|---|
| `PROJECT_CONTEXT.md` | Canonical, cross-tool source of truth. Identity + current numbers + architecture + documentation map + question-routing table + hard constraints + active in-flight changes + Now/Next/Later cadence summary + biggest gaps + onboarding guide. ~280 lines. Replaces no existing file; supplements `CLAUDE.md` and `ROADMAP.md`. |

### `.ai/research/2026-05-17/` (10 files)

| Path | Purpose | Approx. size |
|---|---|---|
| `.ai/research/2026-05-17/STATE_OF_REPO.md` | Local repo reconnaissance: identity, live counts (1,359 routes / 101 blueprints / 523 core modules / 131 test files / 47 model cards), architecture diagram, wave + F-number ledger snapshot, dirty working tree breakdown, CI/build/release infrastructure, known infrastructure debt, security posture snapshot, what this audit changed | ~300 lines |
| `.ai/research/2026-05-17/MEMORY_CONSOLIDATION.md` | Inventory + reconciliation of every AI/agent instruction file, roadmap doc, changelog. Reconciliation logic (where files agree / disagree / duplicate / conflict). Open conflicts. Recommended changes to AGENTS.md + CLAUDE.md. What gets extracted to PROJECT_CONTEXT.md. Question-routing table. Gaps no doc owns. Action items | ~270 lines |
| `.ai/research/2026-05-17/SECURITY_AND_DEPENDENCY_REVIEW.md` | TL;DR action table. Per-dependency tables (core web / scientific / audio / ASR / generative / ML framework / tooling / frontend / FFmpeg). The torch ≥2.6 cascade analysis. Action items mapped to F-numbers. Repo-side hardening in the dirty working tree. CVE summary table | ~300 lines |
| `.ai/research/2026-05-17/COMPETITOR_MATRIX.md` | Direct Premiere extension competitors. OSS NLEs. Adjacent OSS automation. Commercial closed-source AI tools. **Agentic / chat-driven editing systems** (Underlord, FireRed, vibeframe, VideoAgent, ViMax, etc.). Mind-the-gap summary. Notable paywalls that exist because local-AI alternative isn't ready | ~260 lines |
| `.ai/research/2026-05-17/DATASET_MODEL_INTEGRATION_REVIEW.md` | Models the team has not yet considered (post-2026-05-16): video gen, TTS/voice/dub, restoration/upscaling/depth/tracking, standards/spec, agentic/MCP/tooling. Datasets relevant to evaluation harness. External APIs/integrations. Model card / licence hygiene. AI eval harness v2 extensions. Priority for AI/integration items | ~330 lines |
| `.ai/research/2026-05-17/FEATURE_BACKLOG.md` | Raw harvest of 70 new F-numbers F121-F190. Grouped: A) security/deps, B) caption/accessibility/standards, C) agentic/chat/MCP, D) AI capability gaps, E) real-time/streaming, F) model surface refreshes, G) eval/governance/docs, H) Premiere UXP gap reports for Adobe. Explicit declined items | ~190 lines |
| `.ai/research/2026-05-17/PRIORITIZATION_MATRIX.md` | Now / Next / Later / Under Consideration / Rejected tier placement with Impact/Effort/Risk scores. Sequencing summary v1.33-v1.43+. Capacity assumptions | ~210 lines |
| `.ai/research/2026-05-17/SOURCE_REGISTER.md` | Every local + external source cited. Local R-L01-41 (41 files/commands). External: R-P01-18 Premiere (18), R-M01-15 MCP/agents (15), R-C01-19 commercial products (19), R-A01-27 AI models (27), R-D01-60 dependency advisories (60), R-S01-08 community signal (8) | ~250 lines |
| `.ai/research/2026-05-17/RESEARCH_LOG.md` | Phases executed. Subagent prompts. Search strategies. Saturation tests per category. Failed searches / dead ends. Bias / assumptions to flag. Things this audit did NOT do. Continuation hints | ~230 lines |
| `.ai/research/2026-05-17/CHANGESET_SUMMARY.md` | This file | ~150 lines |

**Total new content:** ~2,770 lines of structured markdown across 11 files.

---

## 2. Files edited (minimal, additive)

| Path | Change | Lines added |
|---|---|---|
| `AGENTS.md` | Appended "Canonical Project Context" block (pointer to `PROJECT_CONTEXT.md` + `ROADMAP.md` + `CHANGELOG.md` + `.ai/research/2026-05-17/`). Pre-existing 9-line content preserved verbatim. | ~12 |
| `CLAUDE.md` | Inserted "Canonical Project Context" block immediately after the `# OpenCut - CLAUDE.md` title and before `## Tech Stack`. Pre-existing 1,509-line content preserved verbatim. Markdown-lint warnings on the heading are pre-existing throughout the file (tight heading format). **NOTE**: `CLAUDE.md`, `CODEX-CHANGELOG.md`, `CLAUDE-HANDOFF-PROMPT.md`, and `research.md` are listed in `.gitignore` (team convention: agent memory is local-only). The edit persists on disk for future agent sessions on this machine but will not appear in `git diff` or commits. The corresponding pointer in `AGENTS.md` is the *tracked* analogue. | ~12 |
| `ROADMAP.md` | Appended a new "**2026-05-17 v4.4 Autonomous Research Audit (delta)**" section between the v4.3 status note and the existing v4.3 Phase 0 heading. Includes: Phase 0 delta (what v4.3 missed or what changed since 2026-05-16), Phase 1 research coverage delta, Phase 2 F121-F190 tier summary, Phase 3 top three strategic moves the v4.3 audit understated, Phase 4 self-audit. References the 10 supporting artefacts in `.ai/research/2026-05-17/`. Existing v4.3 content (sections from "## 2026-05-16 v4.3 Autonomous Research Audit" through the v4.3 source appendix) preserved verbatim. | ~70 |

---

## 3. Files NOT changed (deliberately)

- `README.md` — marketing-soft route count (1,344) is stale vs manifest (1,359). Did not edit; the F099 manifest is the truth and is documented in `PROJECT_CONTEXT.md`. The README badge regeneration is a separate cleanup.
- `ROADMAP-NEXT.md` — predecessor to ROADMAP.md v4.3 / v4.4. Kept as archive with its source URLs and per-wave gotchas.
- `ROADMAP-COMPLETED.md` — high-level summary, stale relative to CHANGELOG.md. Kept; banner is on the F185 backlog.
- `CHANGELOG.md` — authoritative for ship dates. Not edited (no release shipped today).
- `MODERNIZATION.md` — module-by-module dep audit, baseline v1.9.18. Regeneration covered by F177 model cards sweep.
- `AUDIT.md`, `research.md` — v1.11.0 / v1.28.2 competitive audits. Kept as predecessor to ROADMAP.md v4.3 / v4.4.
- `features.md` — 402-feature aspirational catalogue. Banner pending F185 (Now tier).
- `CODEX-CHANGELOG.md` — Codex handoff snapshot. Kept.
- `CLAUDE-HANDOFF-PROMPT.md` — boilerplate prompt. Kept.
- `docs/*` — `MODELS.md`, `NODE_ADVISORIES.md`, `RESEARCH.md`, `ROADMAP.md`, `ROADMAP-COMPLETED.md`, `UXP_MIGRATION.md`, `WINDOWS_ARM64_PACKAGING.md`. Auto-generated or live artefacts; not edited.
- `.github/copilot-instructions.md` — should get the same canonical-context pointer eventually, but deferred. Out of scope for this run.
- **All Python source files** (`opencut/**/*.py`) — read-only research; no functional changes.
- **All test files** (`tests/**/*.py`) — read-only; no test changes.
- **The 7 dirty modified files in the working tree** (auth.py, security.py, helpers.py, user_data.py, routes/captions.py, routes/system.py, routes/timeline.py) — left exactly as found. Documented as **F138 commit-pending** in PRIORITIZATION_MATRIX.md and PROJECT_CONTEXT.md §8.

---

## 4. Files NOT created (and why)

- **`CONTINUE_FROM_HERE.md`** — not needed. This audit completed within session budget. Continuation hints are captured in `RESEARCH_LOG.md` §8.

---

## 5. Total impact

| Metric | Value |
|---|---|
| Files created | 11 |
| Files edited | 3 |
| Lines added across edited files | ~94 |
| Lines created in new files | ~2,770 |
| Source citations (this run alone) | 138 unique R-prefixed IDs |
| New F-numbers proposed | 70 (F121-F190) |
| Existing F-numbers status-updated | 0 (v4.3 ledger preserved verbatim) |
| Wave letters touched | 0 (wave-letter plan preserved; F180 calls for future re-tier) |
| Python / JSX / JS / CSS / config changes | 0 |
| Tests added or modified | 0 |
| Dependency pins changed | 0 (recommendations only — see SECURITY_AND_DEPENDENCY_REVIEW.md for action items) |
| Commits made | 0 (read-only research run; F138 recommends committing the dirty tree as a separate PR) |
| Pushes attempted | 0 (push to `SysAdminDoc/OpenCut` is blocked from this VM per persistent memory) |

---

## 6. Verification checklist (post-run)

A future session can verify this audit by running:

```bash
# All artefacts in place
ls Z:/repos/OpenCut/.ai/research/2026-05-17/
ls Z:/repos/OpenCut/PROJECT_CONTEXT.md

# Cross-tool pointers
grep -l "PROJECT_CONTEXT.md" Z:/repos/OpenCut/AGENTS.md Z:/repos/OpenCut/CLAUDE.md

# v4.4 section in ROADMAP.md
grep -n "v4.4 Autonomous Research Audit" Z:/repos/OpenCut/ROADMAP.md

# No accidental code touches
cd Z:/repos/OpenCut && git status --short
# Should show only the 7 pre-existing modified files (auth.py, security.py, helpers.py, user_data.py, captions.py, system.py, timeline.py)
# AND the new files / edits this run added:
#   AGENTS.md (modified)
#   CLAUDE.md (modified)
#   PROJECT_CONTEXT.md (untracked)
#   ROADMAP.md (modified)
#   .ai/research/2026-05-17/* (untracked)
```

---

## 7. What is now true that was not before

1. There is a single `PROJECT_CONTEXT.md` at the repo root that an agent or contributor can open first and route from.
2. The wave-letter vs F-number distinction is named.
3. The dirty working tree is documented as a coherent security batch (F138), not random in-flight noise.
4. Post-2026-05-16 model surfaces (daVinci-MagiHuman, LTX-2.3, etc.), standards (C2PA 2.3, IMSC 1.3, OCIO 2.5), and dependency exposures (Pillow CVEs, audiocraft cascade) are tracked.
5. The Now-tier security debt is explicit: Pillow 12.2, flask-cors 6.x, pydub-on-3.13, OpenTimelineIO-Plugins migration.
6. The Next-tier flagship is named: `/agent/chat` conductor + UXP MCP transport + StreamDiffusionV2 real-time preview.
7. ROADMAP.md self-cites the v4.4 audit; future v4.5 audits know to read this run.

---

## 2026-05-18 Pass 57 Addendum — F249 Linux Distribution Packaging

**Functional changes:** Added Flatpak/AppImage Linux release packaging for
`io.github.sysadmindoc.opencut`.

| Path | Change |
|---|---|
| `io.github.sysadmindoc.opencut.yml` | Flatpak manifest using Freedesktop Platform/Sdk 25.08 and the PyInstaller server bundle source. |
| `flathub.json` | x86_64-only architecture policy for the current binary release bundle path. |
| `packaging/linux/` | Desktop file, MetaInfo, Flatpak launcher, and AppImage `AppRun`. |
| `scripts/build_linux_packages.sh` | Builds AppDir/AppImage/Flatpak artifacts from `dist/OpenCut-Server`. |
| `.github/workflows/build.yml` | Installs Linux packaging tools, builds, archives, and uploads `.flatpak` / `.AppImage` artifacts on tag/manual release jobs. |
| `tests/test_linux_distribution_packaging.py` | Static contract tests for manifest, metadata, launcher, script, workflow, docs, and release-smoke wiring. |
| `docs/LINUX_DISTRIBUTION.md` | Documents Flatpak-first/AppImage-fallback packaging and the Flathub hosted-submission boundary. |
| `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md` | Synced roadmap/state/docs for F249 closure. |

**Validation:** focused packaging tests, Bash syntax check, focused Ruff,
`py_compile`, and reduced release smoke passed locally.

---

## 2026-05-18 Pass 58 Addendum — F250 Aptabase Opt-In Telemetry

**Functional changes:** Added disabled-by-default Aptabase telemetry as the
documented opt-in telemetry provider while preserving the legacy Plausible
route for older self-hosted deployments.

| Path | Change |
|---|---|
| `opencut/core/telemetry_aptabase.py` | Implements the Aptabase `/api/v0/events` batch contract with `App-Key`, region/self-host resolution, bounded queueing, app-key masking, and prop scrubbing. |
| `opencut/user_data.py` | Adds atomic persisted `telemetry_settings.json` helpers with Aptabase defaults. |
| `opencut/routes/wave_e_routes.py` | Adds `GET /telemetry/aptabase/info`, `GET/POST /telemetry/aptabase/settings`, and `POST /telemetry/aptabase/track` with CSRF on mutating calls. |
| `opencut/checks.py` | Adds the `check_aptabase_configured` readiness probe. |
| `tests/test_telemetry_aptabase.py` | Covers disabled defaults, settings validation, app-key masking, public self-host URL enforcement, sensitive-prop scrubbing, sync POST shape, and route CSRF behavior. |
| `scripts/release_smoke.py` | Adds the Aptabase telemetry test file to the focused release gate. |
| `docs/TELEMETRY.md` | Documents the opt-in posture, environment overrides, Aptabase SDK contract, privacy boundaries, and legacy Plausible status. |
| `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md` | Synced roadmap/state/docs for F250 closure. |

**Validation:** focused Aptabase tests, focused Ruff, `py_compile`, generated
route/MCP/API-alias/feature-readiness checks, and reduced release smoke passed
locally.

---

## 2026-05-18 Pass 59 Addendum — F252.1 Bolt UXP WebView Scaffold

**Functional changes:** Added the first safe F252 migration slice: a dormant
Bolt-shaped WebView scaffold that can be iterated beside the shipped UXP panel
without changing the live manifest entrypoint.

| Path | Change |
|---|---|
| `extension/com.opencut.uxp/bolt-webview/README.md` | Documents the dormant scaffold posture, current file map, and cutover sequence. |
| `extension/com.opencut.uxp/bolt-webview/uxp.config.ts` | Adds a least-privilege Bolt/WebView config template that inherits the live plugin identity, keeps `PPRO` minVersion 25.6, enables WebView UI/message bridge/local rendering, and preserves the loopback backend allowlist. |
| `extension/com.opencut.uxp/bolt-webview/src/api/*.ts` | Adds generic UXP helpers and Premiere host wrappers that return plain data across the WebView boundary. |
| `extension/com.opencut.uxp/bolt-webview/webview-ui/` | Adds the WebView shell plus `window.uxpHost.postMessage` bridge helpers for host calls and host-to-WebView callbacks. |
| `tests/test_uxp_webview_scaffold.py` | Pins scaffold files, config permissions, host API exports, bridge envelope tokens, and migration documentation. |
| `scripts/release_smoke.py` | Adds the scaffold guardrail test to the focused release gate. |
| `docs/UXP_MIGRATION.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md` | Synced roadmap/state/docs for F252.1 completion while leaving F252 open. |

**Validation:** focused F252.1 scaffold tests, focused release-smoke unit
coverage, focused Ruff, `py_compile`, and release-smoke `pytest-fast`
(`623 passed`) passed locally.

---

## 2026-05-18 Pass 60 Addendum — F252.2 UXP Host Action Dispatcher

**Functional changes:** Added the next bounded F252 migration slice: a live UXP
host-action dispatcher for the catalogued direct `ocXxx` actions, exposed for
the upcoming WebView bridge without switching the production manifest.

| Path | Change |
|---|---|
| `extension/com.opencut.uxp/main.js` | Adds `UXP_DIRECT_HOST_ACTIONS`, `CEP_FALLBACK_HOST_ACTIONS`, `PProBridge.executeHostAction()`, `PProBridge.hostActionStatus()`, marker/project/bin/playhead helper methods, and `window.OpenCutUXPHost`. |
| `tests/test_uxp_host_action_dispatch.py` | Pins the direct action map against `opencut/_generated/cep_uxp_parity.json`, verifies every direct action is dispatched, preserves explicit CEP fallback handling, and prevents the direct dispatcher from drifting back to `evalScript`/`CSInterface`. |
| `scripts/release_smoke.py` | Adds the dispatcher guardrail test to the focused release gate. |
| `docs/UXP_MIGRATION.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md` | Synced roadmap/state/docs for F252.2 dispatcher progress while leaving F252 open for UDT/live cutover. |

**Validation:** focused dispatcher tests, focused release-smoke unit coverage,
`py_compile`, focused Ruff, `node --check extension\com.opencut.uxp\main.js`,
and release-smoke `pytest-fast` (`627 passed`) passed locally.

---

## 2026-05-18 Pass 61 Addendum — F254 UXP Subsequence Range Integration

**Functional changes:** Closed F254 by replacing the placeholder export-range
API check with a UXP `Sequence.createSubsequence(ignoreTrackTargeting?)`
handoff.

| Path | Change |
|---|---|
| `extension/com.opencut.uxp/main.js` | Adds `_tickTimeFromSeconds()`, `_executeProjectActions()`, and `PProBridge.createSubsequenceFromRange()`; `exportSequenceRange()` now creates a subsequence before handing off to F255. |
| `tests/test_uxp_create_subsequence_integration.py` | Pins the beta package assumption, range action wiring, project transaction use, restore behavior, F255 encoder boundary, and release-smoke registration. |
| `scripts/release_smoke.py` | Adds the F254 guardrail test to the focused release gate. |
| `docs/UXP_MIGRATION.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md` | Synced roadmap/state/docs for F254 closure and F255 continuation. |

**Validation:** focused F254 tests, focused UXP/release-smoke unit coverage,
`py_compile`, focused Ruff, `node --check extension\com.opencut.uxp\main.js`,
and release-smoke `pytest-fast` (`632 passed`) passed locally.

---

## 2026-05-18 Pass 62 Addendum — F255 UXP EncoderManager Export Handoff

**Functional changes:** Closed F255 by completing the UXP encoder handoff for
range subsequences created by F254.

| Path | Change |
|---|---|
| `extension/com.opencut.uxp/main.js` | Adds `_encoderManager()`, `_encoderExportType()`, and `PProBridge.exportSubsequenceWithEncoder()`; `exportSequenceRange()` now calls EncoderManager instead of returning a pending F255 response. |
| `tests/test_uxp_encoder_manager_integration.py` | Pins the beta package assumption, EncoderManager API wiring, AME/immediate export type selection, output-path validation, and release-smoke registration. |
| `tests/test_uxp_create_subsequence_integration.py` | Updates the F254 handoff guardrail to expect the F255 encoder call. |
| `scripts/release_smoke.py` | Adds the F255 guardrail test to the focused release gate. |
| `docs/UXP_MIGRATION.md`, `CHANGELOG.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md` | Synced roadmap/state/docs for F255 closure and F256 continuation. |

**Validation:** focused F255/F254 tests, focused UXP/release-smoke unit
coverage, `py_compile`, focused Ruff, `node --check
extension\com.opencut.uxp\main.js`, and release-smoke `pytest-fast`
(`637 passed`) passed locally.

---

## 7.5 Pass 2 additions (same day, second autonomous research run)

Pass 2 added 5 new artefacts + extended 6 existing ones + appended ROADMAP.md v4.5 section + updated PROJECT_CONTEXT.md + wrote CONTINUE_FROM_HERE.md.

### Files created in Pass 2 (`.ai/research/2026-05-17/`)

| Path | Purpose | Approx lines |
|---|---|---|
| `ROUTE_READINESS_AUDIT.md` | F100 / F115 / MCP / OpenAPI coverage gaps; CEP-bound route catalogue; /api/* alias surface; proposes F191-F199 | 220 |
| `INSTALLER_AUDIT.md` | WPF .NET 9 + Inno + PyInstaller + Docker + Windows ARM64 (F101); CI walk; signing/notarisation deadlines; proposes F200-F207 | 200 |
| `TEST_COVERAGE_GAPS.md` | 12 specific gaps (OpenAPI validity, MCP consistency, JS unit tests, launcher smoke, WPF installer tests, ML perf benchmarks, fuzz extensions, race conditions, UXP contract, SBOM completeness); proposes F208-F219 | 230 |
| `FEATURES_RECONCILIATION.md` | features.md (402 entries) sample walk on 40 entries; status methodology; ~60% SHIPPED, ~27% UNCLEAR; proposes F220-F224 | 220 |
| `FEATURE_BACKLOG_ADDENDUM.md` | F191-F260 (+70 items) consolidated from the four new artefacts + 3 Pass-2 subagent briefs; tier summary with regulatory-deadline priority bumps (F202, F236) | 230 |
| `CONTINUE_FROM_HERE.md` | Pass-3 hand-off: quick wins / medium / large items; known limitations; suggested entry point | 250 |

### Files extended in Pass 2

| Path | What was added |
|---|---|
| `PRIORITIZATION_MATRIX.md` | §6.5 — Pass-2 tier deltas with two regulatory-deadline bumps to *Now* (F202, F236) |
| `SOURCE_REGISTER.md` | Pass-2 section — 14 new local IDs (R-L42-55) + 15 Premiere/UXP IDs (R-P19-33) + 30 Frame.io IDs (R-F01-30) + 27 niche AI / accessibility / standards IDs (R-N01-27) — ~85 new R-prefixed sources |
| `RESEARCH_LOG.md` | Pass-2 section — phases, subagent prompts, saturation tests, failed searches, bias, things-not-done, continuation hints |
| `CHANGESET_SUMMARY.md` | This addition (§7.5) |
| `ROADMAP.md` (root) | Appended v4.5 audit section: Phase 0 (what Pass 1 missed) + Phase 1 (research delta) + Phase 2 (F191-F260 tier summary) + Phase 3 (top 3 strategic moves Pass 1 understated: OTIO Marker review-bundle anchor, Bolt UXP WebView migration, Hybrid Plugins for 5 CEP-blocked features) + Phase 4 (self-audit) |
| `PROJECT_CONTEXT.md` (root) | §1 last-consolidated note ("two passes that day"); §3 wave/F-number table extended to F260; new §9.5 regulatory deadlines block (F202, F236); §10 expanded with 4 new biggest-gaps entries (features.md reconciliation 60% shipped, OTIO Marker anchor, WebView UI migration target, route-readiness coverage gap); §12 artefact list extended with Pass-2 files |

### Files NOT changed in Pass 2

- All Python source files — read-only research; no functional changes.
- All test files — no test additions.
- All build / installer configs — no changes.
- `AGENTS.md` / `CLAUDE.md` — already received the canonical-context pointer in Pass 1; no further edits in Pass 2.
- The 7 dirty modified files in working tree — promoted into the Pass-4 validated local checkpoint as F138.

### Cumulative impact (Pass 1 + Pass 2)

| Metric | Pass 1 | Pass 2 | Total |
|---|---:|---:|---:|
| Files created | 11 | 6 | 17 |
| Files edited | 3 | 6 (incl. 4 existing artefacts + ROADMAP + PROJECT_CONTEXT) | 9 unique files touched |
| Lines created in new files | ~2,770 | ~1,350 | ~4,120 |
| Lines added to edited files | ~94 | ~480 | ~574 |
| F-numbers newly proposed | F121-F190 (70) | F191-F260 (70) | F121-F260 (**140 new**) |
| Wave letters touched | 0 | 0 | 0 (F180 retire-wave-letters-into-F-numbers still deferred) |
| Source citations | 138 R-prefixed | +85 R-prefixed | ~225 unique R-prefixed |
| Subagents launched | 3 | 3 | 6 total |
| Python / JS / config changes | 0 | 0 | 0 |
| Tests added or modified | 0 | 0 | 0 |
| Dependency pins changed | 0 | 0 | 0 (recommendations only) |
| Commits made | 0 | 0 | 0 |
| Pushes attempted | 0 | 0 | 0 |

---

## 7.6 Pass 3 additions (same day, third autonomous research run)

Pass 3 added 4 new artefacts + updated PROJECT_CONTEXT.md (§9.4 live verification block + §10 expansions 12-13) + appended ROADMAP.md v4.6 + extended CHANGESET (this section) + CONTINUE_FROM_HERE.md.

### Files created in Pass 3 (`.ai/research/2026-05-17/`)

| Path | Purpose | Approx lines |
|---|---|---|
| `LIVE_VERIFICATION.md` | Executed CONTINUE_FROM_HERE §3.1 quick-wins. F099/F096/F093/F094/npm-audit live results. **Cross-platform launcher gap confirmed (F261)**. Side-channel: bootstrap warnings, FFmpeg path detection. JSX inventory (18 `ocXxx` functions). Pass-3 corrections to Pass 1/2. Proposes F261-F265 | 240 |
| `CEP_UXP_PARITY_MATRIX.md` | **Completes F198** (CEP-only route catalogue). All 18 JSX functions × `@adobe/premierepro@26.3.0-beta.67` typings; risk classification (14 low, 1 med, 1 different-mechanism, 2 truly CEP-only). Deep-dive on `ocAddNativeCaptionTrack` and `ocQeReflect`. Revised F252 (XL→L) + F253 (XL→L) effort estimates with sub-phases. Proposes F266-F267 | 220 |
| `AGENT_UX_RFC.md` | F143-F145 design space RFC. Adopts: Copilot Workspace editable-plan, Cursor checkpoint+rollback, Underlord post-turn self-review, Aider snapshot discipline, per-region accept/reject, Claude Code Skills format. Rejects: accept-all, auto-commit-before-preview, atomic multi-file apply. Endpoint shape, self-review sketch, Skills SDK layout. Proposes phasing within v1.36. Open questions for maintainer | 280 |
| `MARKET_POSITIONING.md` | OpenCut replaces ~$1,400/yr subscriptions ($720 AutoCut+AutoPod+Submagic + $288 Descript + $299-699 Topaz). Mister Horse Animation Composer ~900k installs proves free-shell + paid-packs model. Per-creator-type pain-point map. README copy proposal. Proposes F268-F272 | 200 |

### Files extended in Pass 3

| Path | What was added |
|---|---|
| `PROJECT_CONTEXT.md` (root) | §9.4 — live verification of governance gates + 2-function CEP residual; §10 expanded with 2 more biggest-gaps entries (#12 agent conductor convergence, #13 market positioning); §12 artefact list extended with Pass-3 files |
| `ROADMAP.md` (root) | Appended v4.6 section: Phase 0 live verification (4 gates pass, 1 ledger gap), Phase 1 CEP↔UXP parity (2 of 18 CEP-only), Phase 2 agent UX RFC, Phase 3 market positioning, Phase 4 F261-F272 deltas, Phase 5 top 3 strategic moves Pass 2 understated, Phase 6 self-audit |
| `CHANGESET_SUMMARY.md` | This section (§7.6) |
| `CONTINUE_FROM_HERE.md` | Pass-4 hand-off updated with what Pass 3 closed + remaining deferrals |

### Pass-3 subagents launched

1. **NLE plugin pricing market-fit signal** — Descript ARR + 78% transcription-only stat; OpusClip + Submagic + HeyGen pricing structure; **Topaz killed perpetual Oct 3, 2025** (strongest market signal of period); Aescripts top-sellers reconstruction; Mister Horse Animation Composer 900k-install model; SaaS↔OSS migration drivers; per-creator-type pain points
2. **Cursor/Copilot/IDE-agent patterns** — Cursor 2.0 Composer + checkpoints; Copilot Workspace editable-plan; Claude Code Skills open standard (agentskills.io); Aider git-commit-per-edit discipline; Cody 2-stage agentic chat; MCP Inspector; mapping IDE concepts to video editor; 3 patterns to copy + 3 to deliberately NOT copy

### F-numbers added in Pass 3

| F# | Title | Tier |
|---|---|---|
| F261 | Ship missing `OpenCut-Server.command` + `.sh` launchers (closes Wave I I1.4 ledger discrepancy) | **Now** |
| F262 | Fix uxp-api-notes.md sample-repo URL typo | Now |
| F263 | Re-run pip-audit on full `[all]` extras | Done in Pass 68 |
| F264 | Add `npm audit --json` machine-parseable CI assertion | Now |
| F265 | UDT test harness for all 18 `ocXxx` JSX functions | Later |
| F266 | Document 2-function CEP residual + drop-QE plan | Now |
| F267 | UDT test harness for 14 low-risk JSX→UXP ports | Done locally in Pass 67 |
| F268 | Adobe Exchange storefront listing | Next |
| F269 | Premium model-pack bundling format | Later |
| F270 | README "$1,400/yr" marketing copy refresh | Now |
| F271 | Per-feature VRAM requirement UI surface | Done in Pass 69 |
| F272 | Wedding-specific Skill (color match + beat sync + 4-min reel) | Done in Pass 70 |

### Cumulative impact (Pass 1 + Pass 2 + Pass 3)

| Metric | Pass 1 | Pass 2 | Pass 3 | Total |
|---|---:|---:|---:|---:|
| Files created in `.ai/research/2026-05-17/` | 10 | 6 | 4 | 20 |
| Root files created | 1 (PROJECT_CONTEXT) | 0 | 0 | 1 |
| Root files edited | 2 (AGENTS, ROADMAP) + 1 (CLAUDE gitignored) | 1 (ROADMAP) + 1 (PROJECT_CONTEXT) | 1 (ROADMAP) + 1 (PROJECT_CONTEXT) | 4 unique tracked + 1 gitignored |
| Lines created in new files | ~2,770 | ~1,350 | ~940 | ~5,060 |
| Lines added to edited root files | ~94 | ~480 | ~190 | ~764 |
| F-numbers newly proposed | F121-F190 (70) | F191-F260 (70) | F261-F272 (12) | **152 new** |
| Source citations (R-prefixed) | 138 | +85 | +20 | ~243 unique |
| Subagents launched | 3 | 3 | 2 | **8 total** |
| Live commands executed | 0 | 0 | 6 | 6 |
| Python / JS / config code changes | 0 | 0 | 0 | 0 |
| Tests added or modified | 0 | 0 | 0 | 0 |
| Dependency pins changed | 0 | 0 | 0 | 0 (recommendations only) |
| Commits made | 0 | 0 | 0 | 0 |
| Pushes attempted | 0 | 0 | 0 | 0 (auth blocked from this VM) |

---

## 8. Recommended next-session actions (in priority order)

1. **Commit the 7-file dirty hardening batch** as F138 (single PR: *"Harden auth loopback classification, UNC realpath, helper lifecycle, and safe_bool follow-up"*).
2. **Push the 25-commit backlog** to `SysAdminDoc/OpenCut` (auth needs to come from a machine with push permission — this VM is blocked).
3. **Run the `gh issue` seeder** (F182) — `python scripts/seed_github_issues.py` — to populate the contributor channel from `.github/issue-seeds.yml`.
4. **Take F121 + F122 + F126 + F130 + F133 + F135 + F137** as a single dependency-bump release (v1.33.0) — they are independent low-risk security fixes with no API impact.
5. **Schedule F123 (pydub) + F128 (FFmpeg filter regression suite)** as a paired pre-release task. These unblock F129 bundled FFmpeg bump and full Python 3.13 support.
6. **Open an RFC issue for F127 (Python 3.10 floor + Transformers v5)** — this is the highest-impact strategic decision in the audit and deserves a dedicated discussion thread before implementation.
7. **Open an RFC issue for F143 (`/agent/chat` conductor)** — the flagship Next-tier item; design space includes UI surface (CEP+UXP), LLM provider routing, timeline diff representation, and post-turn self-review semantics.

---

## 9. Pass 4 additions (same day, release-smoke validation)

Pass 4 converted the research checkpoint from "well documented" to "release-gate clean".

### Files extended in Pass 4

| Path | What was added |
|---|---|
| `PROJECT_CONTEXT.md` | Updated "last consolidated" to Pass 4, changed the F-number range to F001-F272, replaced the stale "uncommitted hardening" section with Pass-4 validation status, and added release-smoke PASS details. |
| `ROADMAP.md` | Added v4.7 status note and a release-smoke validation addendum. |
| `.ai/research/2026-05-17/LIVE_VERIFICATION.md` | Added §8 with first failed release-smoke run, Ruff cleanup actions, final PASS matrix, and targeted `119 passed` validation. |
| `.ai/research/2026-05-17/SOURCE_REGISTER.md` | Added Pass-4 local command evidence R-P4-L01 through R-P4-L09 and external source refresh R-P4-E01 through R-P4-E05. |
| `.ai/research/2026-05-17/RESEARCH_LOG.md` | Added Pass-4 phases, validation results, and saturation note. |
| `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md` | Updated header from Pass 3 to Pass 5 and added Pass-4 handoff state. |

### Code cleanup included in Pass 4

| Scope | Change |
|---|---|
| F138 hardening batch | Kept the security changes from the dirty working tree: loopback classification, UNC realpath guard, FFmpeg helper lifecycle, nested user-data writes, and safe_bool follow-up. |
| Release-smoke lint scope | Applied Ruff safe fixes for unused imports and import ordering in `opencut/` and `scripts/` so `ruff check opencut --select E,F,I --ignore E501,E402` passes. |

### Validation after Pass 4

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_route_manifest --check` | PASS |
| `python scripts/sync_version.py --check` | PASS |
| `python scripts/bootstrap_check.py` | PASS |
| `python -m pip_audit -r requirements-lock.txt` | PASS |
| `ruff check opencut --select E,F,I --ignore E501,E402` | PASS |
| Targeted pytest slice | PASS — `119 passed` |
| `python scripts/release_smoke.py --json` | PASS — release-smoke pytest-fast `232 passed` |

### Cumulative impact after Pass 4

| Metric | Total |
|---|---:|
| Research files in `.ai/research/2026-05-17/` | 20 |
| Root files created | 1 (`PROJECT_CONTEXT.md`) |
| F-numbers newly proposed this research date | 152 (F121-F272) |
| Local validation gates passed in final state | route manifest, version sync, bootstrap, model cards, license gate, roadmap lint, Ruff, pytest-fast, pip-audit, npm advisory allow-list, panel-source |
| Commits made by Pass 4 | 1 local checkpoint commit |

---

## 10. Pass 5 additions (same day, launcher/docs implementation)

Pass 5 converted three Pass-3 Now items from research findings into repository changes.

### Files added or edited in Pass 5

| Path | Change |
|---|---|
| `.gitattributes` | Added LF line-ending rules for POSIX launchers so shell scripts remain runnable from release/source checkouts. |
| `OpenCut-Server.command` | Added macOS double-click launcher that delegates to the POSIX shell launcher. |
| `OpenCut-Server.sh` | Added Linux/POSIX launcher with `OPENCUT_HOME`, bundled/system Python 3.9+ detection, bundled FFmpeg path support, bundled model environment variables, and `python -m opencut.server` startup. |
| `README.md` | Replaced the generic local-first lead with the quantified "$1,400/year" subscription-replacement lead and added macOS/Linux launch instructions. |
| `extension/com.opencut.uxp/uxp-api-notes.md` | Corrected the UXP sample repository URL to `AdobeDocs/uxp-premiere-pro-samples`. |
| `ROADMAP.md` | Added v4.8 status/addendum and marked F261/F262/F270 closed. |
| `PROJECT_CONTEXT.md` | Updated canonical context for Pass 5 and marked cross-platform launchers/README positioning current. |
| `.ai/research/2026-05-17/LIVE_VERIFICATION.md` | Updated the launcher gap to closed after Pass 5 and added a Pass-5 implementation note. |
| `.ai/research/2026-05-17/INSTALLER_AUDIT.md` | Updated launcher inventory to show the macOS/Linux files now exist. |
| `.ai/research/2026-05-17/CONTINUE_FROM_HERE.md` | Moved the handoff to Pass 6 and listed remaining immediate work. |

### Items closed in Pass 5

| F# | Result |
|---|---|
| F261 | Closed — macOS/Linux source launchers exist. |
| F262 | Closed — UXP docs URL typo fixed. |
| F270 | Closed — README lead and launcher instructions updated. |

### Validation after Pass 5

| Command | Result |
|---|---|
| `git diff --check` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 11 release-smoke steps green; pytest-fast reported `232 passed` |

---

## 12. Pass 7 additions (same day, F199 implementation)

Pass 7 closed F199 and corrected the earlier alias-count assumption.

### Files added or edited in Pass 7

| Path | Change |
|---|---|
| `opencut/tools/dump_api_aliases.py` | Added generator/checker for `/api` alias policy derived from the live route manifest. |
| `opencut/_generated/api_aliases.json` | New generated manifest: 233 total `/api/*` routes, 15 true aliases, 218 canonical `/api` routes. |
| `scripts/release_smoke.py` | Added `api-aliases` drift check to the release-smoke matrix. |
| `tests/test_api_aliases.py` | Added committed-vs-live manifest guard plus shape/policy checks. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md` | Marked F199 closed and corrected "233 alias pairs" wording. |

### Item closed in Pass 7

| F# | Result |
|---|---|
| F199 | Closed — generated `/api` alias policy manifest and drift checks. |

### Validation after Pass 7

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases, 218 canonical `/api` routes |
| `python -m pytest tests/test_api_aliases.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `python -m py_compile opencut/tools/dump_api_aliases.py scripts/release_smoke.py` | PASS |
| `ruff check opencut/tools/dump_api_aliases.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 12 release-smoke steps green; pytest-fast reported `236 passed` |

### Push status after Pass 5

`git push origin main` failed with HTTP 403:

```
remote: Permission to SysAdminDoc/OpenCut.git denied to MavenImaging.
fatal: unable to access 'https://github.com/SysAdminDoc/OpenCut.git/': The requested URL returned error: 403
```

The local commits are intact; pushing requires GitHub credentials with write access to `SysAdminDoc/OpenCut`.

---

## 11. Pass 6 additions (same day, F264/F266 implementation)

Pass 6 closed the remaining Pass-3 Now items.

### Files added or edited in Pass 6

| Path | Change |
|---|---|
| `extension/com.opencut.panel/scripts/check-advisories.mjs` | Added `--json` mode with stable `status`, `summary`, `allowed`, and `unwaived` fields. |
| `scripts/release_smoke.py` | Changed `npm-advisory` to invoke `check-advisories.mjs --json`, parse the machine-readable report, and fail on malformed output or unwaived advisories. |
| `docs/NODE_ADVISORIES.md` | Documented the JSON advisory-check command and release-smoke contract. |
| `docs/UXP_MIGRATION.md` | Added F266 section naming `ocAddNativeCaptionTrack` and `ocQeReflect`, plus the drop-QE rules. |
| `tests/test_release_smoke.py` | Added coverage for the JSON advisory contract and malformed-output failure. |
| `tests/test_node_advisories.py` | Added text guard for JSON output support in the Node checker. |
| `tests/test_uxp_migration_docs.py` | Added F266 documentation guard. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md` | Updated status to mark F264/F266 closed and identify F199/F179 as the next work. |

### Items closed in Pass 6

| F# | Result |
|---|---|
| F264 | Closed — release smoke now asserts machine-readable npm advisory JSON. |
| F266 | Closed — CEP residual and drop-QE plan documented. |

### Validation after Pass 6

| Command | Result |
|---|---|
| `node scripts/check-advisories.mjs --json` | PASS — one Vite advisory reported as allowed, zero unwaived |
| `python scripts/release_smoke.py --only npm-advisory --json` | PASS — `npm-advisory` reports `advisories on allow-list (1 allowed)` |
| `python -m pytest tests/test_release_smoke.py tests/test_node_advisories.py tests/test_uxp_migration_docs.py -q` | PASS — `20 passed` |
| `node --check extension/com.opencut.panel/scripts/check-advisories.mjs` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 11 release-smoke steps green; pytest-fast reported `232 passed` |

---

## 13. Pass 8 additions (same day, F191/F197 implementation)

Pass 8 closed the generated feature-readiness registry and registry-owned allowlist items.

### Files added or edited in Pass 8

| Path | Change |
|---|---|
| `opencut/tools/dump_feature_readiness.py` | Added generator/checker that scans route functions for public `checks.py` probes, joins endpoints to the live route manifest, and emits a deterministic readiness manifest. |
| `opencut/_generated/feature_readiness.json` | New generated manifest: 58 route-derived records across 67 direct route/check bindings. |
| `opencut/registry.py` | Loads generated readiness records, merges generated routes into curated records, exposes generated manifest metadata, and owns `NON_AI_CHECKS`. |
| `opencut/model_cards.py` | Imports `NON_AI_CHECKS` from `registry.py` so F115 and F191 share one allowlist. |
| `scripts/release_smoke.py` | Added `feature-readiness` drift check and included the generator tests in the release-gate pytest slice. |
| `tests/test_feature_readiness_generator.py` | Added committed-vs-live generator drift tests and route/probe binding assertions. |
| `tests/test_feature_registry.py` | Added generated-record merge coverage and shared allowlist assertions. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `ROUTE_READINESS_AUDIT.md`, `CONTINUE_FROM_HERE.md` | Marked F191/F197 closed and updated live counts/remaining gaps. |

### Items closed in Pass 8

| F# | Result |
|---|---|
| F191 | Closed — generated route/check readiness manifest loaded into `/system/feature-state`. |
| F197 | Closed — `NON_AI_CHECKS` now belongs to `registry.py`. |

### Validation after Pass 8

| Command | Result |
|---|---|
| `python -m opencut.tools.dump_feature_readiness` | PASS — 58 generated records / 67 route bindings |
| `python -m py_compile opencut/registry.py opencut/model_cards.py opencut/tools/dump_feature_readiness.py scripts/release_smoke.py` | PASS |
| `python -m pytest tests/test_feature_registry.py tests/test_feature_readiness_generator.py tests/test_model_cards.py tests/test_release_smoke.py -q` | PASS — `35 passed` |
| `python scripts/release_smoke.py --only feature-readiness --json` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `241 passed` |

---

## 14. Pass 9 additions (same day, F195 MCP curated tool expansion)

Pass 9 closed the missing post-Wave-M curated MCP tools item.

### Files added or edited in Pass 9

| Path | Change |
|---|---|
| `opencut/mcp_server.py` | Expanded `MCP_TOOLS` from 27 to 39 entries, added route mappings for 12 shipped routes, added Brand Kit and semantic-search action dispatch, and expanded MCP path validation for new scalar/array path keys. |
| `tests/test_mcp_server.py` | Added registration, route mapping, dispatch, special-action, and MCP path-validation coverage. |
| `scripts/release_smoke.py` | Added `tests/test_mcp_server.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `ROUTE_READINESS_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F195 closed, updated MCP tool counts from 27 to 39, and recorded the remaining route/tooling gaps. |

### Items closed in Pass 9

| F# | Result |
|---|---|
| F195 | Closed — curated MCP coverage now includes face reshape, skin retouch, smart upscale, ElevenLabs TTS, caption QC, review bundles, C2PA provenance, marker import, capability probe, Brand Kit, semantic search, and spectral match. |

### Validation after Pass 9

| Command | Result |
|---|---|
| `python -m py_compile opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py` | PASS |
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| `ruff check opencut/mcp_server.py scripts/release_smoke.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `246 passed` |

---

## 15. Pass 10 additions (same day, F202 macOS notarization tooling)

Pass 10 closed the repository-side macOS signing/notarization release path.

### Files added or edited in Pass 10

| Path | Change |
|---|---|
| `scripts/notarize_macos.sh` | Added macOS-only Developer ID certificate import, hardened-runtime Mach-O signing, `xcrun notarytool submit --wait`, and notarized ZIP creation. |
| `.github/workflows/build.yml` | Tagged/manual macOS release builds now call the notarization script and upload `OpenCut-Server-macOS.zip` to GitHub Releases. |
| `docs/MACOS_NOTARIZATION.md` | Documents required GitHub secrets, local commands, and Apple references. |
| `tests/test_macos_notarization.py` | Static release-gate coverage for notarytool usage, hardened runtime signing, workflow wiring, and documentation. |
| `scripts/release_smoke.py` | Added `tests/test_macos_notarization.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F202 repository-side tooling closed and documented the remaining secret/live-service limitation. |

### Items closed in Pass 10

| F# | Result |
|---|---|
| F202 | Closed locally — macOS release builds have Developer ID signing + notarytool submission wiring; live Apple acceptance requires configured secrets. |

### Validation after Pass 10

| Command | Result |
|---|---|
| `python -m pytest tests/test_macos_notarization.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_macos_notarization.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `C:\Program Files\Git\bin\bash.exe -n scripts/notarize_macos.sh` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `249 passed` |

---

## 16. Pass 11 additions (same day, F204 release SBOM attachment)

Pass 11 closed automatic SBOM attachment for release builds.

### Files added or edited in Pass 11

| Path | Change |
|---|---|
| `.github/workflows/build.yml` | Linux tagged/manual release builds now generate `dist/opencut-sbom.cyclonedx.json`, archive it as `OpenCut-SBOM-CycloneDX`, and upload it to GitHub Releases on tags. |
| `tests/test_release_sbom.py` | Added generator smoke coverage and workflow wiring assertions for SBOM generation/archive/upload. |
| `scripts/release_smoke.py` | Added `tests/test_release_sbom.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F204 closed and kept F219 as the deeper SBOM completeness gap. |

### Items closed in Pass 11

| F# | Result |
|---|---|
| F204 | Closed — release builds now attach the generated CycloneDX SBOM. |

### Validation after Pass 11

| Command | Result |
|---|---|
| `python -m pytest tests/test_release_sbom.py tests/test_release_smoke.py -q` | PASS — `14 passed` |
| `ruff check tests/test_release_sbom.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -c "import pathlib, yaml; yaml.safe_load(pathlib.Path('.github/workflows/build.yml').read_text())"` | PASS |
| `python scripts/sbom.py --format json --output dist/opencut-sbom.cyclonedx.json` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `251 passed` |

---

## 17. Pass 12 additions (same day, F205 attempt + F207 installer FFmpeg manifest)

Pass 12 attempted F205 and closed F207.

### Files added or edited in Pass 12

| Path | Change |
|---|---|
| `installer/src/OpenCut.Installer/Models/AppConstants.cs` | Added bundled FFmpeg/ffprobe version constants and the installer manifest filename. |
| `installer/src/OpenCut.Installer/Models/InstallConfig.cs` | Added `InstallerManifestPath` for `~/.opencut/installer.json`. |
| `installer/src/OpenCut.Installer/Services/InstallEngine.cs` | Added a manifest-writing step that emits app/install paths, installer kind, FFmpeg versions, and install timestamp. |
| `installer/src/OpenCut.Installer/GlobalUsings.cs` | Added `System.Text.Json` for manifest serialization. |
| `OpenCut.iss` | Added bundled FFmpeg/ffprobe version defines and an Inno post-install manifest writer. |
| `tests/test_ffmpeg_installer_manifest.py` | Added static coverage for WPF/Inno FFmpeg version manifest contracts. |
| `scripts/release_smoke.py` | Added `tests/test_ffmpeg_installer_manifest.py` to the release-gate pytest slice. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `INSTALLER_AUDIT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `CONTINUE_FROM_HERE.md` | Marked F207 closed and recorded the F205 coverage timeout blocker. |

### Items closed in Pass 12

| F# | Result |
|---|---|
| F207 | Closed — bundled FFmpeg version is now in `AppConstants.cs` and the installed `~/.opencut/installer.json` manifest for both WPF and Inno paths. |

### Validation after Pass 12

| Command | Result |
|---|---|
| `python -m pytest tests/test_ffmpeg_installer_manifest.py tests/test_release_smoke.py -q` | PASS — `15 passed` |
| `ruff check tests/test_ffmpeg_installer_manifest.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile scripts/release_smoke.py` | PASS |
| `.\ffmpeg\ffmpeg.exe -version` | PASS — reports `8.0.1-essentials_build-www.gyan.dev` |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast reported `254 passed` |
| `dotnet build installer/src/OpenCut.Installer/OpenCut.Installer.csproj --no-restore` | BLOCKED — no .NET SDK installed on this VM |

### F205 attempt

Installed missing local plugins with `python -m pip install pytest-cov pytest-xdist`, then attempted the CI-style coverage measurement with the floor disabled. The command timed out after 20 minutes and produced no `dist\coverage-f205.json`, so F205 remains open.

---

## 18. Pass 13 additions (same day, F208 OpenAPI contract gate)

Pass 13 closed F208.

### Files added or edited in Pass 13

| Path | Change |
|---|---|
| `opencut/openapi.py` | Converts Flask path parameters to OpenAPI `{param}` syntax, emits path parameter metadata, generates unique path-qualified operation IDs for aliased endpoints, and adds 400/403 response shapes for every mutating method. |
| `tests/test_openapi_contract.py` | New contract tests for `/openapi.json` route parity, path parameters, operation IDs, response schemas, and `/api/openapi.json` path-parameter syntax. |
| `scripts/release_smoke.py` | Added `tests/test_openapi_contract.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F208 closed and updated the next-pass handoff. |

### Items closed in Pass 13

| F# | Result |
|---|---|
| F208 | Closed — OpenAPI structural validity and route coverage are pinned in release smoke. |

### Validation after Pass 13

| Command | Result |
|---|---|
| `python -m pytest tests/test_openapi_contract.py tests/test_release_smoke.py -q` | PASS — `16 passed` |
| `ruff check opencut/openapi.py tests/test_openapi_contract.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/openapi.py scripts/release_smoke.py tests/test_openapi_contract.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `258 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F209, F218, and F219.

---

## 19. Pass 14 additions (same day, F209 MCP route consistency)

Pass 14 closed F209.

### Files added or edited in Pass 14

| Path | Change |
|---|---|
| `opencut/mcp_server.py` | Corrected `opencut_chat_edit` from planned `/agent/chat` to the shipped `POST /chat` route. |
| `tests/test_mcp_server.py` | Added a live Flask route-consistency test covering all 39 MCP tool routes plus dynamic action routes for music, style transfer, Brand Kit, semantic search, and job status. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F209 closed and updated the next-pass handoff. |

### Items closed in Pass 14

| F# | Result |
|---|---|
| F209 | Closed — curated MCP route drift now fails release smoke. |

### Validation after Pass 14

| Command | Result |
|---|---|
| `python -m pytest tests/test_mcp_server.py tests/test_release_smoke.py -q` | PASS — `18 passed` |
| `ruff check opencut/mcp_server.py tests/test_mcp_server.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/mcp_server.py tests/test_mcp_server.py` | PASS |
| live route-table probe | PASS — 39 MCP tools / 39 route mappings / 0 missing backend routes |
| `python scripts/release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `259 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F218 and F219.

---

## 20. Pass 15 additions (same day, F218 blueprint import-order stability)

Pass 15 closed F218.

### Files added or edited in Pass 15

| Path | Change |
|---|---|
| `tests/test_route_collisions.py` | Added `EXPECTED_CORE_BLUEPRINT_ORDER` and a test that pins the 99 core blueprints plus final `motion_design_api` alias registration order. |
| `scripts/release_smoke.py` | Added `tests/test_route_collisions.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F218 closed and updated the next-pass handoff. |

### Items closed in Pass 15

| F# | Result |
|---|---|
| F218 | Closed — deterministic built-in blueprint registration order now fails release smoke if it drifts unintentionally. |

### Validation after Pass 15

| Command | Result |
|---|---|
| `python -m pytest tests/test_route_collisions.py tests/test_release_smoke.py -q` | PASS — `19 passed` |
| `ruff check tests/test_route_collisions.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile tests/test_route_collisions.py scripts/release_smoke.py` | PASS |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `266 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now item is F219.

---

## 21. Pass 16 additions (same day, F219 SBOM completeness)

Pass 16 closed F219.

### Files added or edited in Pass 16

| Path | Change |
|---|---|
| `scripts/sbom.py` | Added unique dependency component assembly, model-card components, CycloneDX dependency graph output for JSON/XML, and SBOM CLI counts. |
| `tests/test_sbom_completeness.py` | Added the F219 completeness gate for declared dependencies, 47 model cards, unique `bom-ref` values, and dependency graph references. |
| `scripts/release_smoke.py` | Added `tests/test_sbom_completeness.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F219 closed and updated the next-pass handoff. |

### Items closed in Pass 16

| F# | Result |
|---|---|
| F219 | Closed — the release SBOM now fails release smoke if declared dependencies, model cards, or dependency graph entries drift. |

### Validation after Pass 16

| Command | Result |
|---|---|
| `python -m pytest tests/test_sbom_completeness.py tests/test_release_sbom.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| `ruff check scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile scripts/sbom.py tests/test_sbom_completeness.py scripts/release_smoke.py` | PASS |
| `python scripts/sbom.py --format json --output dist/opencut-sbom-f219.cyclonedx.json` and XML equivalent | PASS — 14 required components / 73 optional components / 47 model-card components |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `269 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F237, F240, F243, and F244; F236 was closed in Pass 17, while F251/F259 may need fresh external/API checks before implementation.

---

## 22. Pass 17 additions (same day, F236 FCC caption display-settings tokens)

Pass 17 closed F236.

### Files added or edited in Pass 17

| Path | Change |
|---|---|
| `opencut/core/caption_display_settings.py` | Added the canonical caption display setting token schema, normalization, preview CSS conversion, and ASS `force_style` conversion. |
| `opencut/routes/captions.py` | Added token/preview routes and wired `display_settings` into `/captions/burnin/file`. |
| `tests/test_caption_display_settings.py` | Added the F236 regression tests for FCC factors, token coverage, normalization, preview payloads, and routes. |
| `scripts/release_smoke.py` | Added `tests/test_caption_display_settings.py` to the `pytest-fast` release gate. |
| `opencut/_generated/route_manifest.json` | Regenerated after adding two routes; now 1,361 routes / 101 blueprints. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F236 closed, added FCC/eCFR source evidence, and updated route/test counts. |

### Items closed in Pass 17

| F# | Result |
|---|---|
| F236 | Closed — OpenCut now has a FCC-sourced caption display token contract with preview and burn-in integration. |

### Validation after Pass 17

| Command | Result |
|---|---|
| `python -m pytest tests/test_caption_display_settings.py tests/test_route_manifest.py tests/test_release_smoke.py -q` | PASS — `21 passed` |
| `ruff check opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/caption_display_settings.py opencut/routes/captions.py tests/test_caption_display_settings.py scripts/release_smoke.py` | PASS |
| `python -m opencut.tools.dump_route_manifest` | PASS — wrote 1,361 routes / 101 blueprints |
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases / 218 canonical `/api` routes |
| `python -m opencut.tools.dump_feature_readiness --check` | PASS — 58 generated records / 67 route bindings |
| `python scripts\release_smoke.py --json` | PASS — all 13 release-smoke steps green; pytest-fast `273 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F237, F240, F241, F243, and F244. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 23. Pass 18 additions (same day, F237 loudness standards registry)

Pass 18 closed F237.

### Files added or edited in Pass 18

| Path | Change |
|---|---|
| `opencut/core/loudness_standards.py` | Added canonical standards, source URL, preset, and platform target metadata for ITU BS.1770-5, EBU R128 v5.0, FFmpeg loudnorm, and loudness profiles. |
| `opencut/core/audio_suite.py` | Replaced the local preset literal with the canonical registry while preserving the historical `LOUDNESS_PRESETS` export. |
| `opencut/core/audio_analysis.py` | Imports the shared platform target map and keeps existing `broadcast = -24 LUFS` semantics alongside `ebu_broadcast = -23 LUFS`. |
| `opencut/core/broadcast_qc.py` | Added source-backed EBU R128 / ITU BS.1770-5 metadata to broadcast QC standards. |
| `opencut/routes/audio.py` | `/audio/loudness-presets` now returns standards metadata and correction notes; `/audio/normalize` returns target/source metadata. |
| `tests/test_loudness_standards.py` | Added F237 tests for current ITU/EBU facts, corrected preset targets, compatibility exports, platform targets, and route payload. |
| `scripts/release_smoke.py` | Added `tests/test_loudness_standards.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F237 closed and documented the corrected BS.1770-5 evidence. |

### Items closed in Pass 18

| F# | Result |
|---|---|
| F237 | Closed — loudness target metadata is now centralized and the roadmap now reflects that ITU-R BS.1770-5 is in force while BS.1770-4 is superseded. |

### Validation after Pass 18

| Command | Result |
|---|---|
| `python -m pytest tests/test_loudness_standards.py tests/test_release_smoke.py -q` | PASS — `17 passed` |
| focused compatibility/route pytest slice | PASS — `9 passed` |
| `ruff check opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/loudness_standards.py opencut/core/audio_suite.py opencut/core/audio_analysis.py opencut/core/broadcast_qc.py opencut/routes/audio.py tests/test_loudness_standards.py scripts/release_smoke.py` | PASS |
| `python scripts/release_smoke.py --json` | PASS — all 13 steps green; pytest-fast `278 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F240, F241, F243, and F244. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 24. Pass 19 additions (same day, F240 caption reading-speed profiles)

Pass 19 closed F240.

### Files added or edited in Pass 19

| Path | Change |
|---|---|
| `opencut/core/caption_reading_profiles.py` | Added source-backed reading-speed profiles for Netflix adult/children, BBC editorial, DCMP upper-level educational media, FCC qualitative timing, and YouTube advisory captions. |
| `opencut/core/caption_compliance.py` | Added per-call rule overrides so speed profiles can be applied without mutating the global standards table. |
| `opencut/core/caption_qc.py` | Added `reading_profile` overlay support, profile metadata in `QcResult.as_dict()`, and advisory-mode downgrades for the actual compliance violation names. |
| `opencut/routes/captions.py` | Added `GET /captions/qc/reading-profiles` and `reading_profile`/`profile`/`speed_profile` handling on `POST /captions/qc`. |
| `tests/test_caption_reading_profiles.py` | Added F240 regression tests for source facts, aliases, Netflix adult-vs-children CPS, BBC WPM warnings, and route payloads. |
| `scripts/release_smoke.py` | Added `tests/test_caption_reading_profiles.py` to the `pytest-fast` release gate. |
| `opencut/_generated/route_manifest.json` | Regenerated after adding one route; now 1,362 routes / 101 blueprints. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F240 closed and documented the corrected Netflix/FCC/YouTube source evidence. |

### Items closed in Pass 19

| F# | Result |
|---|---|
| F240 | Closed — caption QC now exposes source-backed reading-speed profiles and correctly distinguishes official hard caps from qualitative/advisory profile assumptions. |

### Validation after Pass 19

| Command | Result |
|---|---|
| `python -m pytest tests/test_caption_reading_profiles.py tests/test_caption_qc.py tests/test_analysis.py::TestCaptionCompliance -q --tb=short` | PASS — `31 passed` |
| `ruff check opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/caption_reading_profiles.py opencut/core/caption_compliance.py opencut/core/caption_qc.py opencut/routes/captions.py scripts/release_smoke.py tests/test_caption_reading_profiles.py` | PASS |
| `python -m opencut.tools.dump_route_manifest` | PASS — wrote 1,362 routes / 101 blueprints |
| `python -m opencut.tools.dump_route_manifest --check --quiet` | PASS — 1,362 routes / 101 blueprints |
| `python -m opencut.tools.dump_api_aliases --check` | PASS — 15 aliases / 218 canonical `/api` routes |
| `python -m opencut.tools.dump_feature_readiness --check` | PASS — 58 generated records / 67 route bindings |
| `python scripts\release_smoke.py --json` | PASS — all 13 steps green; pytest-fast `284 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F241, F243, and F244. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 25. Pass 20 additions (same day, F241 text-shaping gate)

Pass 20 closed F241.

### Files added or edited in Pass 20

| Path | Change |
|---|---|
| `opencut/tools/text_shaping_gate.py` | Added the machine-readable text-shaping gate for FFmpeg/libass HarfBuzz/FriBidi/ASS/subtitles support plus Pillow RAQM and optional Skia shaping reporting. |
| `scripts/release_smoke.py` | Added the `text-shaping` step and included `tests/test_text_shaping_gate.py` in `pytest-fast`. |
| `.github/workflows/build.yml` | Added a build-matrix text-shaping gate after standard dependency installation. |
| `tests/test_text_shaping_gate.py` | Added F241 regression tests for exact FFmpeg parsing, missing-HarfBuzz failure, strict Pillow promotion, release-smoke wiring, and workflow wiring. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F241 closed and documented the local FFmpeg/Pillow/Skia capability evidence. |

### Items closed in Pass 20

| F# | Result |
|---|---|
| F241 | Closed — release smoke and CI now fail when the FFmpeg/libass burn-in path lacks HarfBuzz/FriBidi shaping support, while Pillow RAQM and optional Skia shaping are surfaced as reportable capabilities with strict flags. |

### Validation after Pass 20

| Command | Result |
|---|---|
| `python -m opencut.tools.text_shaping_gate --json` | PASS — FFmpeg/libass hard gates OK; Pillow RAQM advisory warning; Skia skipped |
| `python -m pytest tests/test_text_shaping_gate.py tests/test_release_smoke.py -q --tb=short` | PASS — `17 passed` |
| `ruff check opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/tools/text_shaping_gate.py scripts/release_smoke.py tests/test_text_shaping_gate.py` | PASS |
| `python scripts\release_smoke.py --only text-shaping --json` | PASS — one advisory Pillow warning |
| `python scripts\release_smoke.py --json` | PASS — all 14 steps green; pytest-fast `289 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The next local-verifiable Now items are F243 and F244. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 26. Pass 21 additions (same day, F243 UTF-8 no-BOM SRT policy)

Pass 21 closed F243.

### Files added or edited in Pass 21

| Path | Change |
|---|---|
| `opencut/export/srt.py` | Added the explicit SRT encoding policy: UTF-8 without BOM by default, `legacy_windows_bom=True` / `encoding="utf-8-sig"` for old Windows players, and rejection of non-UTF-8 encodings. |
| `opencut/routes/captions.py` | Added `srt_legacy_bom` / `windows_legacy_bom` / `legacy_bom` request aliases and SRT response `srt_encoding` metadata for caption export routes. |
| `opencut/cli.py` | Added `--srt-legacy-bom` to `opencut captions` and `opencut full`. |
| `opencut/core/subtitle_shot_aware.py` | Reused the shared SRT writer for file export and accepted `legacy_windows_bom=True`. |
| `tests/test_srt_encoding.py` | Added byte-level regression tests for no-BOM default, opt-in BOM, encoding validation, route alias parsing, and shot-aware export behavior. |
| `scripts/release_smoke.py` | Added `tests/test_srt_encoding.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F243 closed and documented the encoding policy evidence. |

### Items closed in Pass 21

| F# | Result |
|---|---|
| F243 | Closed — newly exported SRT files are UTF-8 without BOM by default, with an explicit legacy Windows BOM opt-in at the Python API, route, CLI, and shot-aware file-export surfaces. |

### Validation after Pass 21

| Command | Result |
|---|---|
| `python -m pytest tests/test_srt_encoding.py tests/test_captions_regressions.py tests/test_core.py::TestSRTExport tests/test_subtitle_pro.py::TestShotAwareExport -q --tb=short` | PASS — `13 passed` |
| `ruff check opencut/export/srt.py opencut/routes/captions.py opencut/cli.py opencut/core/subtitle_shot_aware.py scripts/release_smoke.py tests/test_srt_encoding.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/export/srt.py opencut/routes/captions.py opencut/cli.py opencut/core/subtitle_shot_aware.py scripts/release_smoke.py tests/test_srt_encoding.py` | PASS |
| `python scripts\release_smoke.py --json` | PASS — all 14 steps green; pytest-fast `294 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. F244 is now closed in Pass 22; the remaining Now items are F205, F251, and F259. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 27. Pass 22 additions (same day, F244 Whisper confidence + human-review flags)

Pass 22 closed F244.

### Files added or edited in Pass 22

| Path | Change |
|---|---|
| `opencut/core/captions.py` | Added clamped word/segment confidence metadata, language confidence, Hindi/Arabic human-review rules, low-confidence review reasons, shared `caption_segment_to_dict`, and backend mappers for OpenAI Whisper, faster-whisper, and WhisperX. |
| `opencut/routes/captions.py` | Preserved/exposed review metadata through `/captions`, `/transcript`, `/transcript/export`, `/full`, `/interview-polish`, transcript cache reuse, summarize, chapters, and repeat-detect segment payloads. |
| `opencut/export/srt.py` | Included review metadata in JSON exports and made JSON export tolerant of namespace-style cached transcription results. |
| `opencut/polish_state.py` | Persisted/restored language confidence, segment confidence, review flags/reasons, speaker, and word confidence for interview-polish resume state. |
| `opencut/cli.py` | Added a review recommendation line to `opencut captions` when any segment is flagged. |
| `tests/test_caption_language_confidence.py` | Added F244 regression coverage for language flags, low-confidence reasons, JSON export, remap preservation, transcript route payloads, and edited-transcript export preservation. |
| `scripts/release_smoke.py` | Added `tests/test_caption_language_confidence.py` to the `pytest-fast` release gate. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Marked F244 closed and documented the validation evidence. |

### Items closed in Pass 22

| F# | Result |
|---|---|
| F244 | Closed — Whisper transcript segments now carry ASR confidence, language confidence, Hindi/Arabic review flags, and stable low-confidence review reason codes across route, cache, state, JSON export, edited export, and CLI surfaces. |

### Validation after Pass 22

| Command | Result |
|---|---|
| `python -m pytest tests/test_caption_language_confidence.py tests/test_captions_regressions.py tests/test_srt_encoding.py -q --tb=short` | PASS — `12 passed` |
| `ruff check opencut/core/captions.py opencut/routes/captions.py opencut/export/srt.py opencut/polish_state.py opencut/cli.py scripts/release_smoke.py tests/test_caption_language_confidence.py --select E,F,I --ignore E501,E402` | PASS |
| `python -m py_compile opencut/core/captions.py opencut/routes/captions.py opencut/export/srt.py opencut/polish_state.py opencut/cli.py scripts/release_smoke.py tests/test_caption_language_confidence.py` | PASS |
| `python scripts\release_smoke.py --json` | PASS — all 14 steps green; pytest-fast `300 passed` |

### Remaining immediate work

F205 remains open after the Pass 12 coverage timeout. The remaining Now items are F205, F251, and F259. F251 and F259 need fresh Adobe/UXP verification before implementation.

---

## 29. Pass 23 additions (same day, F205 interrupted coverage reattempt wrap-up)

Pass 23 wrapped up an interrupted F205 coverage reattempt. No application code changed, and no F-number was closed.

### Files added or edited in Pass 23

| Path | Change |
|---|---|
| `.ai/research/2026-05-17/F205_INTERRUPTED_COVERAGE_NOTE.md` | Added a durable note for the interrupted coverage command, partial JSON totals, SHA256, cleanup action, and the reason F205 remains open. |
| `ROADMAP.md`, `PROJECT_CONTEXT.md`, `FEATURE_BACKLOG_ADDENDUM.md`, `PRIORITIZATION_MATRIX.md`, `TEST_COVERAGE_GAPS.md`, `INSTALLER_AUDIT.md`, `SOURCE_REGISTER.md`, `RESEARCH_LOG.md`, `CONTINUE_FROM_HERE.md` | Updated current memory/state files so future agents do not mistake the partial coverage artifact for a valid floor-setting run. |
| `C:\Users\Xray\.codex\memories\extensions\ad_hoc\notes\2026-05-17T19-45-34-opencut-f205-wrapup.md` | Added the Codex memory checkpoint requested by the user. |

### F205 wrap-up facts

| Item | Result |
|---|---|
| Coverage command | Interrupted after 2,206.6 seconds (36m46s). |
| Partial JSON | Valid but incomplete: 126,421 statements, 65,890 covered, 60,531 missing, 52.1195% coverage across 670 files. |
| Artifact hash | `dist\coverage-f205.json` SHA256 `63DD45BF6C617BB05A7944911DEFF735A528F37F96CAD4CCC10F6E93CF59A6F9` (ignored, not committed). |
| Cleanup | Stopped leftover `python.exe -m pytest tests sidecar/tests -q`, then removed stale `.coverage` and `dist\coverage-f205.json` after recording the evidence. |
| Decision | F205 remains open; CI coverage floor remains `--cov-fail-under=50`. |

### Remaining immediate work

F205 should resume only where the full CI-style coverage command can finish. F251 and F259 remain open and need fresh Adobe/UXP verification before implementation.

---

## Pass 72 addendum (2026-05-18)

Pass 72 closed **F196** by making the registry/model-card/check relationship an enforced catalogue contract. `opencut/registry.py` now carries curated rows for all 47 model-card feature IDs, including the 16 surfaces the F191 route scanner could not infer through helper layers, and `/system/feature-state` now exposes 100 records total.

Added `opencut/catalog_contract.py` and `tests/test_catalog_contract.py`, and registered the test in release-smoke `pytest-fast`. The contract verifies public `check_*_available` triage, model-card-to-registry feature IDs, and matching hardware/GPU/VRAM metadata. Updated ROADMAP.md v4.75, CHANGELOG.md, PROJECT_CONTEXT.md, and Pass-2 state files. Validation passed: focused registry/model-card/catalog tests (`32 passed`), `py_compile`, focused Ruff, model-card/readiness sync checks, roadmap lint, and release-smoke `pytest-fast` (`698 passed`).

---

## Pass 73 addendum (2026-05-18)

Pass 73 closed **F206** by splitting pull-request CI from the full release matrix. Added `.github/workflows/pr-fast.yml`, a Linux-only pull-request workflow that installs Python 3.12, FFmpeg, `ruff`, `pytest`, and `opencut[standard]`, then runs the fast release-smoke subset while skipping release-only audit/panel/upstream-drift checks.

Renamed `.github/workflows/build.yml` to **Release Full** and removed its `pull_request` trigger while preserving push, tag, and manual dispatch behavior for the three-OS build/sign/package/SBOM/release path. Added `tests/test_ci_workflow_split.py` and registered it in `pytest-fast`. Validation passed: focused F206 tests (`4 passed`), `py_compile`, and focused Ruff.
