# Project Research and Feature Plan

**Project:** OpenCut
**Date:** 2026-06-06
**Baseline:** v1.32.0 -- 1,523 routes, 107 blueprints, 599 core modules, 8,800+ tests
**Research method:** Multi-agent deep research (20 agents across orientation, inventory, audit, landscape, and synthesis phases)

> This document complements `ROADMAP.md` (strategic what-to-build) and
> `RESEARCH_REPORT.md` (research cycle synthesis). It focuses on **new findings
> not already captured** in those files. Items already tracked in `ROADMAP.md`
> are referenced by ID (e.g., RA-01, F143) but not duplicated.

---

## Executive Summary

OpenCut is a remarkably ambitious MIT-licensed video-editing automation backend
for Adobe Premiere Pro, replacing ~$1,400/year of subscriptions with 1,523
local-first API routes, 744 async job endpoints, and 94 optional AI dependency
gates. The project is approximately 98% wired (only ~21 stub routes remain)
with a strong security posture, comprehensive job infrastructure, and an
emerging agentic editing layer.

**Strongest current shape:** The core editing pipeline (silence removal,
transcription, styled captions, audio processing, batch workflows) is
production-grade with deep test coverage. The tiered stub system
(working/503/501) is a smart pattern for managing the AI feature explosion
across Waves A-M without shipping broken functionality.

**Highest-value improvement direction:** Harden the security boundary (3 code
execution risks found), close UXP parity gaps (broad i18n parity, missing empty
states), and improve the expression/scripting sandbox before the Chat Conductor
agent (F143) makes these surfaces more exposed.

### Top 10 Opportunities (priority order)

1. **P0 -- Fix `torch.load(weights_only=False)` RCE** in `model_quantization.py:371` -- closed 2026-06-07
2. **P0 -- Replace `pickle.load()` with safe deserialization** in `semantic_video_search.py:139` -- closed 2026-06-07
3. **P0 -- Switch `os.startfile()` from blocklist to allowlist** in `system.py:784` -- closed 2026-06-07
4. **P1 -- UXP panel i18n parity** -- foundation loader plus Cut/Captions/FCC display/Audio/Video/Timeline static-shell slices shipped; broad parity remains open vs CEP's 1,190 attributes
5. **P1 -- Expression engine per-frame thread elimination** -- closed 2026-06-07
6. **P1 -- Scripting console code length limit** -- closed 2026-06-07
7. **P1 -- Security audit logging** -- closed 2026-06-07
8. **P2 -- CEP structured empty-state components** -- closed 2026-06-07
9. **P2 -- Multi-language locale files** -- i18n plumbed but only English shipped
10. **P2 -- Rate-limit decorator adoption** -- closed 2026-06-07

---

## Evidence Reviewed

### Local files and directories inspected
- `CLAUDE.md` (292 lines -- module-by-module developer reference)
- `PROJECT_CONTEXT.md` (full project context via agent)
- `MODERNIZATION.md` (dependency audit via agent)
- `ROADMAP.md` (448 lines -- v5.0 strategic roadmap)
- `ROADMAP-NEXT.md` (Wave A-K detail)
- `TODO.md` (compact execution queue)
- `CHANGELOG.md` (release log through v1.32.0)
- `COMPLETED.md` (shipped work summaries)
- `RESEARCH_REPORT.md` (research cycle synthesis)
- `README.md` (user-facing documentation)
- `opencut/security.py`, `opencut/jobs.py`, `opencut/helpers.py`, `opencut/server.py`
- `opencut/config.py`, `opencut/errors.py`, `opencut/checks.py`
- `opencut/routes/__init__.py` (95 blueprint registrations)
- `opencut/routes/audio.py`, `video_core.py`, `captions.py`, `system.py`
- `opencut/core/model_quantization.py`, `semantic_video_search.py`
- `opencut/core/expression_engine.py`, `scripting_console.py`
- `opencut/core/onboarding.py`, `context_awareness.py`, `batch_executor.py`
- `opencut/core/registry.py` (~55 hand-curated feature records)
- `extension/com.opencut.panel/client/index.html`, `main.js`, `style.css`
- `extension/com.opencut.uxp/index.html`, `main.js`, `style.css`
- `tests/conftest.py` + 223 test files
- `.github/workflows/build.yml`, `pr-fast.yml`, `adobe-premierepro-versions.yml`

### Git history range reviewed
- Last 80 commits (i18n migration batches 120-153, SSRF hardening, lockfile advisory, UXP Agent Tab)
- Contributor analysis since 2026-01-01

### External sources reviewed
- ROADMAP.md competitive landscape table (12 competitors with pricing)
- ROADMAP.md source register (200+ URLs in `.ai/research/2026-05-17/SOURCE_REGISTER.md`)
- Existing research artifacts in `.ai/research/2026-05-17/`

### Areas that could not be verified
- Live Premiere Pro panel behavior (no Adobe installation available)
- GPU memory behavior under load (no GPU available in research environment)
- macOS/Linux platform-specific behavior
- Docker container runtime behavior
- Social posting OAuth flows
- Actual FFmpeg 8.x filter availability

---

## Current Product Map

### Core workflows
1. **Silence/filler detection and removal** -- detect and cut silent/filler sections
2. **Auto-subtitling/captioning** -- multi-backend Whisper with styled/animated/karaoke/burn-in captions
3. **Caption translation** -- SRT-in/SRT-out via NLLB-200 and SeamlessM4T
4. **AI audio processing** -- stem separation, speech enhancement, TTS, AI music generation, ducking, loudness normalization
5. **AI video processing** -- upscaling, background removal, frame interpolation, style transfer, depth effects, face operations, scene detection
6. **Highlight extraction** -- LLM-powered highlight detection and summarization
7. **Auto-editing** -- motion/audio-based automatic cuts
8. **B-roll generation** -- AI-generated B-roll via Wan/CogVideoX/HunyuanVideo
9. **Social media posting** -- direct posting to YouTube/TikTok/Instagram
10. **Batch processing** -- parallel batch executor for multiple operations
11. **Custom workflow builder** -- user-defined multi-step workflow chains
12. **Review bundle** -- OTIO marker-based review bundles with LAN portal
13. **NLP command** -- natural language to API route mapping
14. **Chat editing assistant** -- multi-turn chat-based editing via LLM

### Existing features (by maturity)

**Most mature (deep routes, full tests, production-ready):**
- Audio processing (29 async jobs in `audio.py`)
- Captions/Transcription (15 async jobs in `captions.py`)
- Audio advanced suite (19 async jobs -- podcast, ADR, surround, speaker processing)
- Editing workflows (24 async jobs -- script alignment, publish queue, copilot)
- Documentary workflows (20 async jobs -- selects bin, conform, brand kit, montage)
- Color/MAM (19 async jobs -- color wheels, HSL, ACES pipeline, proxy generation)
- Audio production (14 async jobs -- declip, dehum, dereverb, M&E mix, dialogue premix)
- Video core (8 async jobs -- watermark, scene detection, export, merge, trim)

**Stub/incomplete (21 routes across Wave H and K):**
- Cloud gen-video (Hailuo/Seedance) -- 501 stubs, no implementation
- Advanced lip-sync (GaussianHeadTalk, FantasyTalking2) -- 501 stubs
- Video outpainting, face age transform, trailer generation -- 501 stubs
- FlashVSR, ROSE inpaint, Sammie rotoscope, OmniVoice, ReEzSynth -- 503 dep-check stubs

### User personas
1. **Video editors using Adobe Premiere Pro** -- primary audience (professional and prosumer)
2. **DaVinci Resolve users** -- secondary via Resolve scripting bridge
3. **MCP/AI tool clients** -- developers and AI agents (39 curated + 1,466 extended tools)
4. **Plugin/skill developers** -- extending via plugin manifest v1
5. **Self-hosters/Docker users** -- containerized deployment

### Platforms and distribution
- **Windows** (primary): WPF .NET 9 installer + .bat/.vbs launchers
- **macOS**: .command launcher (notarization pending -- F202)
- **Linux**: .sh launcher + Flatpak/AppImage
- **Docker**: Containerized with GPU compose variant
- **Distribution weakness**: Windows-only installer; no PyPI/Homebrew/winget/Snap (all planned in ROADMAP.md)

### Important integrations
- Adobe Premiere Pro (CEP 2019-25.5, UXP 25.6+)
- DaVinci Resolve (scripting bridge)
- MCP JSON-RPC (port 5681)
- LLM providers (Ollama, OpenAI, Anthropic, Gemini -- stdlib urllib)
- FFmpeg (primary media processing engine)
- YouTube/TikTok/Instagram (social posting)
- C2PA provenance (content authenticity)

### Storage
- SQLite WAL: `~/.opencut/jobs.db` (job persistence), `footage_index.db` (FTS5 search)
- File-based JSON: llm_settings, social_credentials, brand_kit, onboarding, auth, plugin locks
- Transcript cache: SHA-256-keyed under `~/.opencut/transcript_cache/`
- Generated manifests: `opencut/_generated/` (route manifest, model cards, API aliases, feature readiness)

---

## Feature Inventory

> Only features with notable findings are listed. The full 744-endpoint inventory
> is captured in the research agent output and the `registry.py` feature catalogue.

### Audio Processing
- **User value:** Core pipeline -- silence removal is the primary reason users install OpenCut
- **Entry point:** `/silence`, `/fillers`, `/audio/*` (29 async endpoints)
- **Code:** `opencut/routes/audio.py` (~2,175 lines), `opencut/core/silence.py`, `audio.py`, `audio_enhance.py`, `audio_pro.py`, `audio_duck.py`, `loudness_match.py`
- **Maturity:** Complete. Deepest test coverage of any domain.
- **Improvement:** Silero VAD v5->v6.2 upgrade is tracked (ROADMAP.md Now tier)

### Captions & Transcription
- **User value:** Second most-used feature set after silence removal
- **Entry point:** `/captions/*` (15 async endpoints)
- **Code:** `opencut/routes/captions.py` (~1,590 lines), `opencut/core/captions_enhanced.py`, `animated_captions.py`, `crisper_whisper.py`
- **Maturity:** Complete. FCC caption display-settings (F236) shipped.
- **Improvement:** Whisper large-v3-turbo evaluation is tracked (ROADMAP.md Now tier)

### Workflow Engine
- **User value:** Power-user automation -- chain multiple operations
- **Entry point:** `/workflow/*` (run, presets, save, delete)
- **Code:** `opencut/routes/workflow.py`, `opencut/core/workflow.py`
- **Maturity:** Complete for sequential chains. Missing: conditional steps, branching, error recovery with retry.
- **Improvement:** Conditional workflow steps tracked in ROADMAP.md Next tier

### Expression Engine
- **User value:** Keyframe expressions and scripted automation
- **Entry point:** Used internally by timeline tools
- **Code:** `opencut/core/expression_engine.py` (~700 lines)
- **Maturity:** Functional; per-frame thread churn closed 2026-06-07
- **Improvement:** Per-frame thread creation was closed by Cycle 86 with inline trace-deadline evaluation

### Plugin System
- **User value:** Extensibility for third-party developers
- **Entry point:** `/plugins/*` (list, install, uninstall)
- **Code:** `opencut/core/plugins.py`, `plugin_manifest.py`
- **Maturity:** Partial. Manifest v1 works, 3 example plugins exist, but no marketplace, discovery, or developer documentation beyond examples.
- **Improvement:** Plugin ecosystem items tracked in ROADMAP.md (Under Consideration)

### Onboarding
- **User value:** First-use experience
- **Entry point:** `/system/onboarding/*`
- **Code:** `opencut/core/onboarding.py` (~100 lines)
- **Maturity:** Complete but minimal. Simple state persistence. No guided tour, no feature discovery.
- **Improvement:** NEW finding -- onboarding could be much richer (not in ROADMAP.md)

### Context Awareness
- **User value:** Smart feature recommendations based on clip metadata
- **Entry point:** `/context/analyze`, `/context/features`
- **Code:** `opencut/core/context_awareness.py` (~300 lines), 40+ feature scoring rules
- **Maturity:** Complete. Well-structured scoring system by tab (cut, captions, audio, video, effects, export, ai).
- **Improvement:** Could feed into a smarter onboarding or "Quick Mode" UI (tracked in ROADMAP.md)

---

## Competitive and Ecosystem Research

> Competitor pricing and gap analysis is already comprehensive in ROADMAP.md
> (lines 362-391). This section adds observations not captured there.

### Descript ($24-65/mo)
- **Notable:** Underlord AI agent, text-based editing, Overdub voice cloning
- **Learn:** Their "Scenes" feature (auto-segmenting long-form into sections) is a simpler version of OpenCut's highlights + chapters that users find more intuitive. The AI agent has undo/checkpoint built into the UX -- OpenCut's Chat Conductor (F143) should prioritize this.
- **Avoid:** Text-based editing as primary UI (already rejected in ROADMAP.md)

### LosslessCut (41K stars, free)
- **Notable:** 7+ distribution platforms (Flathub, Snap, Homebrew, winget, Chocolatey, AUR, Microsoft Store)
- **Learn:** Distribution breadth is a competitive moat that OpenCut explicitly flags as a weakness. LosslessCut ships on every major platform with auto-update. The project should prioritize PyPI + Homebrew + winget as the three highest-ROI channels.
- **Avoid:** Scope creep into NLE territory -- LosslessCut stays focused on lossless operations

### Submagic (4M+ users, $12-40/mo)
- **Notable:** Animated viral caption styles are their entire product. Users pay $12-40/mo for what OpenCut already has in `animated_captions.py`.
- **Learn:** OpenCut has the backend capability but the UX for discovering and previewing caption styles may not be as polished. A caption style gallery/preview in the panel would make this feature more discoverable.
- **Avoid:** Nothing -- this is pure value-capture opportunity

### Gling ($10-50/mo)
- **Notable:** "Bad take detection" and "best take selection" are their differentiators
- **Learn:** OpenCut has `repeat_detect.py` (Jaccard similarity) and `best_take.py` exists as a core module. The gap isn't capability -- it's UX polish. A "review takes" mode in the Cut tab that highlights bad takes with one-click removal would match Gling's core value.
- **Avoid:** Nothing -- OpenCut has the building blocks

---

## Highest-Value New Features

> Features NOT already in ROADMAP.md. Each is grounded in evidence from the audit.

### NF-01: Caption Style Gallery and Preview
- **User problem:** OpenCut has animated captions (Submagic charges $12-40/mo for this) but users can't browse/preview styles without running the full pipeline
- **Evidence:** `animated_captions.py` exists with multiple style backends; Submagic has 4M users paying for just this feature; no preview endpoint exists
- **Proposed behavior:** GET `/captions/styles/gallery` returns available styles with thumbnail previews. Panel shows a visual grid of caption styles. User picks a style before running the pipeline.
- **Implementation:** New route in `captions.py`, static preview images in `opencut/data/caption_styles/`, gallery component in both CEP and UXP panels
- **Risks:** Preview generation for animated styles requires FFmpeg render of a sample clip
- **Verification:** Load gallery in panel, select a style, run caption pipeline, verify style applied
- **Complexity:** M
- **Priority:** P2

### NF-02: Take Review Mode
- **User problem:** OpenCut has `repeat_detect.py` and `best_take.py` but no integrated UX for reviewing takes (Gling charges $10-50/mo for this workflow)
- **Evidence:** `repeat_detect.py` detects repeated takes via Jaccard similarity; `best_take.py` exists as a core module; Cut tab has a cut review panel but not a take review panel
- **Proposed behavior:** After transcription, "Review Takes" button in Cut tab highlights detected repeated takes with audio confidence scores. User approves/rejects each take. "Keep best" auto-selects highest-confidence version.
- **Implementation:** New UI component in Cut tab, wiring to existing `repeat_detect` and `best_take` routes
- **Risks:** Accuracy depends on transcription quality; false positives on intentional repetition
- **Verification:** Load video with multiple takes, run take detection, verify UI shows takes with correct timestamps
- **Complexity:** M
- **Priority:** P2

### NF-03: Onboarding Tour with Context-Aware Feature Discovery
- **User problem:** First-use experience is a simple state flag (`onboarding.json`). New users see 8+ tabs with 50+ sub-tabs and no guidance on which features matter for their content type.
- **Evidence:** `onboarding.py` is 100 lines of state persistence; `context_awareness.py` scores 40+ features by clip metadata but this scoring isn't used in onboarding; competitors (CapCut, Descript) have guided tours
- **Proposed behavior:** On first launch, ask user their primary content type (podcast, YouTube, shorts, documentary, corporate). Use `context_awareness.py` scores to highlight the 5 most relevant features in a step-by-step tour with tooltips pointing to actual UI elements.
- **Implementation:** Extend `onboarding.py` with content-type selection; add tour step data to `opencut/data/onboarding_tours.json`; add tour rendering in both panels
- **Risks:** Tour steps become stale as UI changes; maintenance burden
- **Verification:** Fresh install, complete onboarding, verify highlighted features match content type
- **Complexity:** M
- **Priority:** P2

### NF-04: Structured Error Recovery Panel
- **User problem:** When operations fail, users see only a toast notification. No retry button, no "what went wrong" explanation, no recovery path.
- **Evidence:** Security audit found toasts are the only error surface in both panels; `errors.py` has a rich structured error taxonomy with codes and suggestions but the panel doesn't render the `suggestion` field prominently; UXP has 10+ empty-state components but 0 error-state components
- **Proposed behavior:** When a job fails, show an error panel (not just a toast) with: error message, structured suggestion from the backend, retry button with same parameters, link to relevant log entry, and "Report Issue" button (wiring to `issue_report.py`)
- **Implementation:** New `ErrorRecoveryPanel` component in both CEP and UXP; consume `suggestion` field from error JSON; add retry endpoint or client-side replay
- **Risks:** Some errors are not retryable (e.g., missing dependency)
- **Verification:** Trigger a known error (e.g., missing FFmpeg), verify error panel shows with suggestion and retry button
- **Complexity:** M
- **Priority:** P2

### NF-05: Security Audit Event Log
- **User problem:** Closed 2026-06-07. Security-relevant events now have a structured local trail for CSRF failures, path validation rejections, auth failures, and rate-limit rejections.
- **Evidence:** `opencut/security_audit.py` writes schema-tagged JSONL records, `security.py` records CSRF/path/rate-limit denials, `server.py` records remote auth-token denials, and `/system/audit-log` returns capped recent events.
- **Proposed behavior:** Shipped as `security_audit.jsonl`, configurable with `OPENCUT_SECURITY_AUDIT_LOG`; test apps keep the default sink disabled unless explicitly configured.
- **Implementation:** `opencut/security_audit.py`, hooks in `opencut/security.py` and `opencut/server.py`, read route in `opencut/routes/system.py`.
- **Risks:** Log file growth without rotation remains a follow-up if audit volume becomes high.
- **Verification:** `tests/test_security_audit.py` covers CSRF token redaction, path-traversal evidence, rate-limit denial records, remote auth denial records, and `/system/audit-log` reads.
- **Complexity:** M
- **Priority:** P1

---

## Existing Feature Improvements

### EI-01: Expression Engine -- Eliminate Per-Frame Thread Creation
- **Current behavior:** Closed 2026-06-07. `evaluate_expression()` now evaluates inline with trace-based deadline checks and restores any prior trace hook, so `evaluate_timeline()` no longer creates one worker thread per frame.
- **Problem:** Thread creation on Windows was ~10ms each, adding ~3 seconds overhead per 10-second timeline. Thread object churn stressed the GC.
- **Recommended change:** Shipped in Cycle 86 with inline caller-thread evaluation and deadline tracing.
- **Code locations:** `opencut/core/expression_engine.py` lines 670-710
- **Backward compatibility:** No API change -- internal optimization only
- **Verification:** `tests/test_motion_design.py::TestExpressionEngine::test_evaluate_timeline_does_not_spawn_per_frame_threads`; 30-second local benchmark evaluated 900 frames with no errors and held `threading.enumerate()` at 2 before and after
- **Complexity:** S
- **Priority:** P1

### EI-02: Rate-Limit Decorator Adoption
- **Current behavior:** Closed 2026-06-07. Async route locks now use worker-lifetime `async_job(rate_limit_key=...)`; the MCP bridge uses `rate_limit_slot()` for dynamic per-tool throttling; route modules no longer call the primitives directly.
- **Problem:** Manual pattern was error-prone -- forgetting the `finally` block could leak a rate-limit slot, and async routes needed the slot held until the worker finished rather than only while the Flask handler returned a `job_id`.
- **Recommended change:** Shipped in Cycle 91 with shared async-job rate-limit keys, dynamic slot context management, and a route primitive regression gate.
- **Code locations:** `opencut/jobs.py`, `opencut/security.py`, `opencut/routes/audio.py`, `video_ai.py`, `video_specialty.py`, `system.py`, `mcp_bridge_routes.py`
- **Backward compatibility:** No API change -- refactor only
- **Verification:** `rg -n "\brate_limit\(|\brate_limit_release\(" opencut/routes` returns no matches; `tests/test_async_job_rate_limit.py` covers wrapper rejection/release behavior and route static drift.
- **Complexity:** M (high file count, mechanical changes)
- **Priority:** P2

### EI-03: CEP Panel Empty-State Components
- **Current behavior:** Closed 2026-06-07. CEP empty hints now carry the shared `oc-empty-state` classes and Favorites renders a localized empty state instead of hiding the bar.
- **Problem:** When lists returned zero results (job history, search, favorites), CEP either used plain hints or removed the surface entirely. UXP's structured empty states were a clear UX improvement not fully ported back.
- **Recommended change:** Shipped in Cycle 92 with shared helper classes, localized Favorites empty-state copy, and static coverage for job history, search, favorites, workflow steps, and batch files.
- **Code locations:** `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, `tests/test_i18n_hardcoded_migration.py`
- **Backward compatibility:** Additive -- no existing behavior changes
- **Verification:** `tests/test_i18n_hardcoded_migration.py` verifies the shared helper, localized Favorites keys, and list-renderer usage. Panel Vitest, ESLint, and build verification also passed; rendered Browser/Playwright validation was unavailable in this thread.
- **Complexity:** S
- **Priority:** P2

### EI-04: UXP Panel i18n Parity
- **Current behavior:** CEP panel has 1,190 `data-i18n` attributes and a drift-lint test. UXP now has a local locale loader plus first-viewport, Cut-tab, Captions-tab, FCC display-settings, Audio-tab, Video-tab, Timeline-tab, Search-tab, Deliverables-tab, Agent-tab, Settings-tab, generated Settings status, shared runtime-toast coverage, Cut runtime-feedback coverage, Captions runtime-feedback coverage, Audio runtime-feedback coverage, and Video core/AI-effects/Shorts runtime-feedback coverage, but non-English locale files and deeper feature-specific dynamic strings remain open.
- **Problem:** UXP is "the future" (CLAUDE.md design principle #6), but its i18n coverage still trails the CEP panel's painstakingly migrated surface. When CEP EOL happens (~September 2026), UXP becomes the only panel, so broad localization parity remains urgent.
- **Recommended change:** Port the i18n loading system (`loadLocale`, `t()`, `applyI18nToDOM`) from CEP `main.js` to UXP `main.js`. Add `data-i18n` attributes to all UXP HTML strings. Share the same `locales/en.json` keys.
- **Code locations:** `extension/com.opencut.uxp/main.js`, `index.html`; reference `extension/com.opencut.panel/client/main.js` i18n module
- **Backward compatibility:** Additive -- English behavior unchanged
- **Verification:** Verify `data-i18n` attribute count in UXP matches CEP. Run the UXP i18n static guard and expand it toward the existing CEP drift depth.
- **Complexity:** L (9 tabs, many sub-panels, all strings)
- **Priority:** P1

### EI-05: Scripting Console Code Length Limit
- **Current behavior:** `scripting_console.py` accepts arbitrary-length code strings via `_clean_string()` which strips whitespace but has no length limit. The expression engine has a 2000-char `_MAX_EXPRESSION_LENGTH` but the scripting console does not.
- **Problem:** A multi-megabyte code string would be compiled and executed, consuming CPU and memory. The expression engine correctly limits this; the scripting console should too.
- **Recommended change:** Add `MAX_CODE_LENGTH = 102400` (100KB) constant. Reject longer inputs before compilation with a descriptive error.
- **Code locations:** `opencut/core/scripting_console.py`
- **Backward compatibility:** Only rejects pathologically large inputs
- **Verification:** Submit 200KB code string, verify 400 rejection. Submit 50KB code string, verify execution.
- **Complexity:** S
- **Priority:** P1

### EI-06: Cleanup Thread Lazy Initialization
- **Status:** Closed 2026-06-07 by deferring the `opencut-temp-cleanup` daemon until `_schedule_temp_cleanup()` is first called.
- **Prior behavior:** `helpers.py:135-137` started a daemon cleanup thread at module import time, meaning `import opencut.helpers` anywhere (tests, CLI tools) started a background thread as a side effect.
- **Problem:** Surprising side effect of importing a utility module. Tests and CLI tools that imported helpers previously got an unwanted background thread.
- **Implemented change:** `_ensure_cleanup_thread_started()` now starts the shared daemon once, after the first queued temp cleanup.
- **Code locations:** `opencut/helpers.py`, `tests/test_helpers_cleanup.py`
- **Backward compatibility:** Same behavior at runtime, cleaner in test/CLI contexts
- **Verification:** `tests/test_helpers_cleanup.py` imports `opencut.helpers` in a fresh interpreter, verifies no cleanup thread appears, then schedules cleanup and verifies the worker starts.
- **Complexity:** S
- **Priority:** P3

---

## Reliability, Security, Privacy, and Data Safety

### CRITICAL: Unsafe `torch.load(weights_only=False)` -- Arbitrary Code Execution

**Status:** Closed 2026-06-07 by switching model quantization to `weights_only=True`
and raising Torch-backed extras to `torch>=2.6` / `torchvision>=0.21`.

**File:** `opencut/core/model_quantization.py:371`
```python
model = torch.load(model_path, map_location="cpu", weights_only=False)
```
`weights_only=False` disables PyTorch's safe deserialization guard. A malicious `.pt` file loaded through the quantization route executes arbitrary Python. Other callsites (`face_swap.py:105`, `video_ai.py:881`) correctly use `weights_only=True`.

**Fix:** Change to `weights_only=True`. Handle `UnpicklingError` with a clear error.
**Note:** PyTorch CVE-2025-32434 (CVSS 9.3) is already tracked in ROADMAP.md, but this specific callsite is not mentioned.

### HIGH: Pickle Deserialization of Untrusted Cache Files

**Status:** Closed 2026-06-07 by replacing raw pickle CLIP caches with
compressed `.npz` files that store JSON metadata and load arrays with
`allow_pickle=False`.

**File:** `opencut/core/semantic_video_search.py:139`
```python
return pickle.load(f)  # CLIP embedding cache
```
Cache files at `~/.opencut/clip_cache/clip_*.pkl` use raw `pickle.load()`. If an attacker can write to `~/.opencut/` (shared system, malware, symlink attack), they achieve code execution when the cache is loaded. Cache key is derived from predictable file path + stat metadata.

**Fix:** Switch to `numpy.savez`/`numpy.load` for embedding arrays, or implement a `RestrictedUnpickler` that only allows numpy types.

### HIGH: `os.startfile()` Blocklist Incomplete

**Status:** Closed 2026-06-07 by replacing direct-open blocklisting with a safe
media/document extension allowlist while preserving reveal mode.

**File:** `opencut/routes/system.py:784`
The `/system/open-path` open mode blocks common executable extensions but misses dangerous Windows file types: `.msc`, `.cpl`, `.application`, `.appref-ms`, `.url`, `.library-ms`, `.settingcontent-ms` (CVE-2018-8414 vector).

**Fix:** Switch from blocklist to allowlist of known-safe media/document extensions (`.mp4`, `.mov`, `.avi`, `.mkv`, `.wav`, `.mp3`, `.jpg`, `.png`, `.pdf`, etc.).

### MEDIUM: Expression Engine AST Bypass Risk

**File:** `opencut/core/expression_engine.py:383-398`
AST checker only validates `ast.Name` nodes for banned function calls. It does not check `ast.Attribute` calls, so `some_object.exec(...)` or chained attribute access patterns could potentially bypass the name-based check. The `sys.settrace()` timeout also doesn't trigger during C-extension calls.

**Mitigating factor:** Local-only API. Low practical exploitability.

### MEDIUM: CORS `null` Origin Allowed

**File:** `opencut/config.py:69`
Default CORS origins include `"null"`, which is sent by sandboxed iframes and `data:` URIs. Combined with the CSRF token being available via GET `/health`, a sandboxed iframe could exfiltrate the CSRF token and make authenticated requests.

**Mitigating factor:** CSRF token required on mutations. Default localhost binding. Remote-bind requires auth token.

### LOW: `send_file()` Without Path Confinement

**File:** `opencut/routes/generative_routes.py:405`
Serves `frame_path` from `render_splat_frame()` without checking path confinement. Unlike the `/preview` route in `system.py` which validates via `is_path_within_any()`, this route serves whatever the render function returns.

### Missing Guardrails

- **No audit logging** for security events (covered in NF-05 above)
- **No code length limit** on scripting console (covered in EI-05 above)
- **Rate limiter map never shrinks** -- `security.py:589` `_rate_limits` dict only grows. Not exploitable with current hardcoded keys but the API accepts any string key. Low risk.

---

## UX, Accessibility, and Trust

### Strengths (above average for this project type)

- **Skip links:** Both panels have skip-to-main-content links
- **ARIA coverage:** CEP has 195 `aria-*` + 84 `role` attributes; UXP has 81 `aria-*`
- **aria-live regions:** Processing banners, status, toasts all use `aria-live`
- **Focus styles:** 187 `:focus`/`:focus-visible` rules in CEP CSS, 43 in UXP
- **Reduced motion:** 4 `prefers-reduced-motion` queries in CEP, 3 in UXP
- **i18n infrastructure:** 1,190 `data-i18n` attributes in CEP, drift-lint test; UXP shell loader shipped in Cycle 94, Cut-tab static coverage shipped in Cycle 95, Captions-tab static coverage shipped in Cycle 96, FCC display-settings static/dynamic coverage shipped in Cycle 97, Audio-tab static coverage shipped in Cycle 98, top/core Video-tab static coverage through Depth Effects shipped in Cycle 99, Video effects coverage through Style Transfer shipped in Cycle 100, remaining Shorts/Social Video coverage shipped in Cycle 101, Timeline-tab coverage shipped in Cycle 102, Search-tab coverage shipped in Cycle 103, Deliverables-tab coverage shipped in Cycle 104, Agent-tab coverage shipped in Cycle 105, Settings-tab static/generated-shell coverage shipped in Cycle 106, shared runtime-toast coverage shipped in Cycle 107, Cut runtime-feedback coverage shipped in Cycle 108, Captions runtime-feedback coverage shipped in Cycle 109, Audio runtime-feedback coverage shipped in Cycle 110, Video core runtime-feedback coverage shipped in Cycle 111, Video AI/effects runtime-feedback coverage shipped in Cycle 112, and Video Shorts runtime-feedback coverage shipped in Cycle 113
- **a11y regression tests:** `test_panel_a11y_invariants.py` guards toast a11y

### Gaps

1. **UXP i18n is only a partial foundation** -- the main tab static shells, shared runtime toasts, Cut runtime feedback, Captions runtime feedback, Audio runtime feedback, and Video core/AI-effects/Shorts runtime feedback now have first-pass coverage, but deeper dynamic surfaces and non-English locale files remain open (covered in EI-04)
2. **Only English locale shipped** -- i18n plumbed but `locales/en.json` is the only file
3. **CEP lacks structured empty-state components** -- closed 2026-06-07 with shared CEP empty-state classes and Favorites empty copy
4. **No structured error-recovery UI** -- errors surface via toasts only (covered in NF-04)
5. **No automated WCAG/contrast audit** in CI -- closed 2026-06-07 with a static CEP/UXP token-pair contrast audit wired into release smoke and PR Fast
6. **No system color scheme auto-detection in UXP** -- CEP has inline script for `prefers-color-scheme`, UXP does not
7. **Tab structure diverges** -- CEP has 8 tabs, UXP has 9 (Export vs Deliverables+Agent split). Tracked by `test_panel_tab_parity.py` but still a user-facing inconsistency.
8. **Onboarding is minimal** -- simple state flag, no guided tour (covered in NF-03)

---

## Architecture and Maintainability

### Module Boundaries

**95+ Flask blueprints** is an extreme count. Each blueprint adds startup cost, attack surface, and maintenance burden. The `assert_no_route_collisions()` check in `__init__.py` is a good guardrail but the sheer number makes auditing difficult.

**Recommendation:** No immediate action needed -- the wave-based organization is logical. But future waves should consider consolidating related routes rather than adding new blueprints.

### JobRegistry vs. Module Globals

`jobs.py` contains a `JobRegistry` class (line 95) for test isolation, but runtime uses module-level globals because "80+ route files already hold direct references." This split personality means test patches on `opencut.jobs.jobs` may not affect routes that captured the dict reference at import time.

**Recommendation:** Low priority. The current approach works; the class exists for test isolation.

### Code Duplication

- **Rate-limit pattern** repeated 25+ times vs 2 uses of the `require_rate_limit` decorator -- closed 2026-06-07 with worker-lifetime `async_job(rate_limit_key=...)`, dynamic `rate_limit_slot()`, and a route primitive regression gate
- **Filepath validation + 400 response** pattern repeated in every non-`@async_job` route. The `@async_job` decorator centralizes this well, but synchronous routes duplicate it.

### Test Gaps

- **54% coverage floor is low** for 8,700+ tests. The volume is impressive but large swaths of subprocess-heavy code (FFmpeg, Whisper) are untested because they need real binaries.
- **No frontend/browser tests in CI.** Panel a11y tests are static JS source analysis, not runtime DOM assertions.
- **No visual regression tests.** No screenshot comparison integration.
- **Minimal fuzz testing** -- only `tests/fuzz/test_parser_fuzz.py` (1 test function covering 5 parsers).

### Documentation Gaps

- **No user-facing documentation.** CONTRIBUTING.md and SECURITY.md shipped, but no user tutorials, getting-started guide, or feature documentation beyond README.md.
- **Plugin developer docs** are limited to example plugins. No guide for creating plugins, no API reference for plugin capabilities.
- **UXP API notes** exist (`uxp-api-notes.md`) but are internal dev notes, not published documentation.

---

## Prioritized Roadmap

> Items already in ROADMAP.md are not repeated. This roadmap covers NEW findings only.

### Phase 1: Security Hardening (P0, immediate)

- [x] P0 - **Fix `torch.load(weights_only=False)` RCE**
  - Why: Arbitrary code execution via malicious model file
  - Evidence: `opencut/core/model_quantization.py`; other callsites correctly use `weights_only=True`
  - Touches: `opencut/core/model_quantization.py`, `pyproject.toml`, dependency-surface tests
  - Acceptance: All `torch.load` calls use `weights_only=True`; Torch-backed extras require `torch>=2.6` / `torchvision>=0.21`
  - Verify: `rg -n "weights_only=False|torch>=2\\.0|torchvision>=0\\.15" opencut pyproject.toml tests` returns no matches

- [x] P0 - **Replace `pickle.load()` with safe deserialization**
  - Why: Cache poisoning leads to code execution
  - Evidence: `opencut/core/semantic_video_search.py:139,151`
  - Touches: `opencut/core/semantic_video_search.py`, `tests/test_object_intel.py`
  - Acceptance: CLIP cache uses `numpy.savez_compressed`/`numpy.load(..., allow_pickle=False)` with JSON metadata
  - Verify: `rg -n "pickle\.load|pickle\.dump" opencut`

- [x] P0 - **Switch `os.startfile()` from blocklist to allowlist**
  - Why: Blocklist misses dangerous Windows file types (.msc, .cpl, .settingcontent-ms, etc.)
  - Evidence: `opencut/routes/system.py`; CVE-2018-8414 vector via `.settingcontent-ms`
  - Touches: `opencut/routes/system.py`
  - Acceptance: `/system/open-path` uses allowlist of media/document extensions only
  - Verify: focused open-path tests reject `.msc`, `.cpl`, `.settingcontent-ms`, and `.url`

### Phase 2: Performance & Sandbox Hardening (P1)

- [x] P1 - **Expression engine: eliminate per-frame thread creation**
  - Why: 300 threads/10s of video, ~3s overhead on Windows
  - Evidence: closed 2026-06-07 in `opencut/core/expression_engine.py`; `evaluate_expression()` now uses inline trace-deadline evaluation instead of per-eval `threading.Thread`
  - Touches: `opencut/core/expression_engine.py`
  - Acceptance: `evaluate_timeline()` creates at most 1 worker thread
  - Verify: 30-second clip benchmark kept thread count constant at 2 before and after while evaluating 900 frames with no errors

- [x] P1 - **Scripting console: add code length limit**
  - Why: No cap on exec'd code size allows resource exhaustion
  - Evidence: `opencut/core/scripting_console.py`; expression engine has `_MAX_EXPRESSION_LENGTH` and the console now has `MAX_CODE_LENGTH_BYTES`
  - Touches: `opencut/core/scripting_console.py`, `opencut/routes/dev_scripting_routes.py`, `opencut/routes/workflow_dev_routes.py`, `tests/test_dev_scripting.py`, `tests/test_workflow_dev.py`
  - Acceptance: Code over 100 KiB rejected before compile/exec; scripting HTTP routes return 400 `CODE_TOO_LARGE`
  - Verify: Oversized core and route tests reject 200 KiB submitted scripts larger than `MAX_CODE_LENGTH_BYTES`

- [x] P1 - **Security audit event log**
  - Why: No structured trail for CSRF failures, path traversal attempts, auth failures
  - Evidence: closed 2026-06-07 with `opencut/security_audit.py`, security/server hooks, and `/system/audit-log`
  - Touches: `opencut/security_audit.py`, `opencut/security.py`, `opencut/server.py`, `opencut/routes/system.py`, `tests/test_security_audit.py`
  - Acceptance: Security rejections are logged to `security_audit.jsonl` in structured JSON without token values
  - Verify: focused tests trigger CSRF, path traversal, rate-limit, and remote auth denials, then read the audit log

- [x] P1 - **`send_file()` path confinement in generative_routes** -- closed 2026-06-07
  - Why: Serves arbitrary paths without confinement check
  - Evidence: `opencut/routes/generative_routes.py`; unlike `/preview` route which validates via `is_path_within_any()`
  - Touches: `opencut/routes/generative_routes.py`, `tests/test_generative_routes_security.py`
  - Acceptance: `send_file()` validates path is within tempdir or `~/.opencut`
  - Verify: Renderer-returned temp preview frames serve successfully; unconfined renderer-returned paths return 403

### Phase 3: UXP Parity (P1-P2)

- [ ] P1 - **UXP panel i18n infrastructure**
  - Why: CEP EOL ~September 2026; UXP needs full i18n parity before it becomes the only panel
  - Evidence: CEP has 1,190 `data-i18n` attributes; Cycle 94 added the UXP locale loader and first shell slice, Cycle 95 added Cut-tab coverage, Cycle 96 added Captions-tab coverage, Cycle 97 added FCC display-settings coverage, Cycle 98 added Audio-tab coverage, Cycle 99 added top/core Video-tab coverage through Depth Effects, Cycle 100 added Video effects coverage through Style Transfer, Cycle 101 completed the remaining Video Shorts/Social static shell, Cycle 102 added Timeline-tab static-shell coverage, Cycle 103 added Search-tab static-shell coverage, Cycle 104 added Deliverables-tab static-shell coverage, Cycle 105 added Agent-tab static-shell coverage, Cycle 106 added Settings-tab static/generated-shell coverage, Cycle 107 added shared runtime-toast coverage, Cycle 108 added Cut runtime-feedback coverage, Cycle 109 added Captions runtime-feedback coverage, Cycle 110 added Audio runtime-feedback coverage, Cycle 111 added Video core runtime-feedback coverage, Cycle 112 added Video AI/effects runtime-feedback coverage, Cycle 113 added Video Shorts runtime-feedback coverage, and broad UXP parity has crossed the >660 static-attribute target while remaining short of full locale parity
  - Touches: `extension/com.opencut.uxp/main.js`, `index.html`, `locales/en.json`, `tests/test_uxp_i18n.py`
  - Acceptance: UXP loads `locales/en.json`, `data-i18n` count > 660
  - Verify: UXP i18n guard passes, then expand toward full drift parity against the UXP panel

- [x] P2 - **CEP empty-state components**
  - Why: UXP has 10+ structured empty states; CEP originally relied on plain hints or hidden empty containers
  - Evidence: closed 2026-06-07 by routing CEP `buildEmptyHintMarkup()` through shared `oc-empty-state` classes and adding a visible Favorites empty state
  - Touches: `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/locales/en.json`, `tests/test_i18n_hardcoded_migration.py`
  - Acceptance: Job history, search, favorites, workflow steps, and batch files use shared empty-state markup when empty
  - Verify: Static migration coverage proves the shared helper and localized Favorites empty-state keys remain wired

- [ ] P2 - **Structured error recovery panel**
  - Why: Errors show only as toasts; no retry, no actionable guidance
  - Evidence: `errors.py` has rich `suggestion` field but panel doesn't render it prominently; 0 error-state components in either panel
  - Touches: Both panels' JS and HTML
  - Acceptance: Failed job shows error panel with suggestion text and retry button
  - Verify: Trigger job failure, verify error panel with suggestion and retry

### Phase 4: Developer Experience & Polish (P2-P3)

- [x] P2 - **Rate-limit decorator migration**
  - Why: Previously 25+ manual acquire/release patterns vs 2 using decorator; error-prone
  - Evidence: closed 2026-06-07 with worker-lifetime `async_job(rate_limit_key=...)`, dynamic `rate_limit_slot()`, and a release-smoke regression that scans route modules
  - Touches: `opencut/jobs.py`, `opencut/security.py`, async route modules, `tests/test_async_job_rate_limit.py`
  - Acceptance: Zero manual `rate_limit(`/`rate_limit_release(` calls remain in `opencut/routes`
  - Verify: `rg -n "\brate_limit\(|\brate_limit_release\(" opencut/routes` returns no matches

- [ ] P2 - **Caption style gallery/preview**
  - Why: Submagic charges $12-40/mo for animated captions; OpenCut has the backend but no style discovery UX
  - Evidence: `animated_captions.py` has multiple styles; no preview endpoint exists
  - Touches: `opencut/routes/captions.py`; both panels
  - Acceptance: Gallery endpoint returns styles with previews; panel shows visual grid
  - Verify: Load gallery in panel, select style, verify caption pipeline uses selection

- [ ] P2 - **Take review mode**
  - Why: Gling charges $10-50/mo for bad-take detection; OpenCut has `repeat_detect.py` and `best_take.py` but no integrated review UX
  - Evidence: `repeat_detect.py` exists with Jaccard similarity; `best_take.py` is a core module; Cut tab has cut review but not take review
  - Touches: Both panels' Cut tab; `opencut/routes/captions.py` (repeat-detect route)
  - Acceptance: "Review Takes" button in Cut tab shows detected takes with approve/reject
  - Verify: Load multi-take video, run take detection, verify UI highlights takes

- [ ] P2 - **Context-aware onboarding tour**
  - Why: First-use experience is a state flag; no guided discovery of features
  - Evidence: `onboarding.py` is 100 lines; `context_awareness.py` scores 40+ features but isn't used in onboarding
  - Touches: `opencut/core/onboarding.py`; both panels
  - Acceptance: First launch shows content-type selection then guided tour of top 5 features
  - Verify: Fresh install flow completes tour end-to-end

- [x] P3 - **Cleanup thread lazy initialization**
  - Why: `import opencut.helpers` starts a daemon thread as side effect
  - Evidence: closed 2026-06-07 in `opencut/helpers.py`; `_ensure_cleanup_thread_started()` starts `opencut-temp-cleanup` only after `_schedule_temp_cleanup()` queues work
  - Touches: `opencut/helpers.py`, `tests/test_helpers_cleanup.py`
  - Acceptance: Thread only starts on first `_schedule_temp_cleanup()` call
  - Verify: `tests/test_helpers_cleanup.py` covers fresh-interpreter import and first-schedule startup

- [x] P2 - **WCAG contrast audit in CI**
  - Why: No automated accessibility contrast checking; 195 ARIA attributes but no validation they're correct
  - Evidence: closed 2026-06-07 with `opencut.tools.contrast_audit` plus release-smoke `contrast-audit`
  - Touches: `opencut/tools/contrast_audit.py`, `scripts/release_smoke.py`, `tests/test_contrast_audit.py`, CEP/UXP panel token CSS
  - Acceptance: PR Fast runs release smoke without skipping `contrast-audit`, so WCAG AA token violations fail the PR gate
  - Verify: `tests/test_contrast_audit.py` includes a deliberately low-contrast fixture and current CEP/UXP token audit; `python -m opencut.tools.contrast_audit --json` audits 72 pairs with 0 failures

---

## Quick Wins

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | Fix `torch.load(weights_only=False)` | Shipped 2026-06-07 | Closed by safe quantization loads plus Torch/TorchVision advisory floors |
| 2 | Add code length limit to scripting console | Shipped 2026-06-07 | Prevents resource exhaustion |
| 3 | Switch `os.startfile()` to allowlist | Shipped 2026-06-07 | Closed extension-bypass vector |
| 4 | Fix `send_file()` path confinement | Shipped 2026-06-07 | Closed file-serve bypass with temp/`~/.opencut` confinement and 403 regression coverage |
| 5 | Replace `pickle.load` with numpy | Shipped 2026-06-07 | Closed cache-poisoning execution path |
| 6 | Defer cleanup thread start | Shipped 2026-06-07 | Cleaner test/CLI imports |
| 7 | Rate-limit decorator migration | Shipped 2026-06-07 | Route-level primitive calls are gone and async locks now fail fast before job creation |
| 8 | Add CEP empty-state components | Shipped 2026-06-07 | Shared CEP empty-state helper and visible Favorites empty state |

---

## Larger Bets

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | UXP panel i18n parity | L | Essential for CEP EOL transition |
| 2 | Expression engine thread elimination | Shipped | Performance for timeline expressions |
| 3 | Rate-limit decorator migration | Shipped 2026-06-07 | Worker-lifetime async locks plus route primitive regression coverage |
| 4 | Security audit event log | M | Incident investigation capability |
| 5 | Caption style gallery | M | Monetization parity with Submagic ($12-40/mo) |
| 6 | Take review mode | M | Monetization parity with Gling ($10-50/mo) |
| 7 | Context-aware onboarding tour | M | First-use experience transformation |
| 8 | WCAG contrast CI gate | Shipped 2026-06-07 | Automated accessibility enforcement |

---

## Explicit Non-Goals

| Item | Reason |
|------|--------|
| Rewriting Flask to FastAPI | Already in ROADMAP.md "Under Consideration" with correct reasoning (Flask works at current scale) |
| TypeScript panel migration | Already deferred in ROADMAP.md; incremental approach is correct |
| Reducing blueprint count | 95+ blueprints is unusual but the wave-based organization is logical and route collision detection works |
| Adding more stub routes | The project has 21 stubs already; shipping stubs without implementation creates maintenance debt |
| Mobile companion app | Already rejected in ROADMAP.md with correct reasoning (low ROI vs Premiere focus) |
| Cloud render offloading | Contradicts local-first design principle |
| Duplicate ROADMAP.md items | This document deliberately avoids re-listing the 100+ items already tracked there |

---

## Open Questions

1. **UXP i18n priority vs CEP i18n completion:** E15 is at batch 173/~160+ for CEP. Should UXP i18n (EI-04) start in parallel, or wait until E15 completes? The CEP EOL deadline (~September 2026) suggests parallel work is warranted.

2. **Expression engine scope:** Is `evaluate_timeline()` actually used in production, or is it a planned capability? If unused, the per-frame thread fix (EI-01) can be deprioritized. Verify with `grep -rn "evaluate_timeline" opencut/`.

3. **Scripting console exposure:** Is the scripting console route (`/scripting/*`) intended for end users or only for plugin/skill developers? If end-user-facing, the sandbox hardening (expression engine AST, code length limit) becomes P0.

4. **CLIP cache usage frequency:** How often is `semantic_video_search.py` used in practice? If rarely, the pickle fix (Phase 1) can be a simple removal of the cache rather than a migration to numpy format.
