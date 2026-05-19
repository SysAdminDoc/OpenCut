# OpenCut — Research Log

**Run date:** 2026-05-17
**Total elapsed:** ~25 minutes wall-clock from session start to last artefact written.
**Tools used:** Bash (`git log`, `git status`, `git diff`, `ls`, `wc -l`, `python -c`), Read, Glob, Grep, Write, Edit, Agent (general-purpose), TodoWrite.
**Subagents launched:** 3 in parallel.

---

## Pass 63 implementation note (2026-05-18)

F256 was closed from local evidence plus the unpacked `@adobe/premierepro@26.3.0-beta.67` typings. The beta package confirms `Transcript.querySupportedLanguages()`, `Transcript.hasTranscript(clipProjectItem)`, `Transcript.exportToJSON(clipProjectItem)`, and `ClipProjectItem.cast(projectItem)`. Implementation stayed inside `extension/com.opencut.uxp/main.js`, added static guardrails in `tests/test_uxp_transcript_api_integration.py`, and extended release-smoke `pytest-fast`.

Validation evidence for the pass: focused F256 tests (`5 passed`), focused UXP/release-smoke slice (`37 passed`), touched Python compile, focused Ruff, UXP JS syntax check, and release-smoke `pytest-fast` (`642 passed`).

---

## Pass 64 implementation note (2026-05-18)

F257 was closed from local evidence plus the unpacked `@adobe/premierepro@26.3.0-beta.67` typings. The beta package confirms `ObjectMaskUtils.hasObjectMask(projectOrSequence: Project | Sequence): boolean`. Implementation stayed inside `extension/com.opencut.uxp/main.js`, added static guardrails in `tests/test_uxp_object_mask_api_integration.py`, and extended release-smoke `pytest-fast`.

Validation evidence for the pass: focused F257 tests (`5 passed`), focused UXP/release-smoke slice (`42 passed`), touched Python compile, focused Ruff, UXP JS syntax check, and release-smoke `pytest-fast` (`647 passed`).

---

## Pass 65 implementation note (2026-05-18)

F258 was closed from local evidence plus the unpacked `@adobe/premierepro@26.3.0-beta.67` typings. The beta package confirms `ProjectConverter.exportAAF(sequence, filePath, aafExportOptions?)`, `AAFExportOptions`, and `Constants.AAFExportAudioFormat`. Implementation stayed inside `extension/com.opencut.uxp/main.js`, added static guardrails in `tests/test_uxp_aaf_export_integration.py`, and extended release-smoke `pytest-fast`.

Validation evidence for the pass: focused F258 tests (`6 passed`), focused UXP/release-smoke slice (`48 passed`), touched Python compile, focused Ruff, UXP JS syntax check, and release-smoke `pytest-fast` (`653 passed`).

---

## Pass 66 implementation note (2026-05-18)

F260 was closed from local F198 parity catalogue evidence. `build_dashboard_manifest()` derives the dashboard from `CEP_UXP_PARITY`, `dump_uxp_migration_dashboard` writes repository and panel artifacts, and the UXP Settings tab loads the bundled JSON to show direct UXP, fallback, high-risk, and per-action replacement-plan status.

Validation evidence for the pass: focused F260 tests (`7 passed`), focused UXP/release-smoke slice (`55 passed`), touched Python compile, focused Ruff, dashboard sync check, UXP JS syntax check, and release-smoke `pytest-fast` (`660 passed`).

---

## Pass 67 implementation note (2026-05-18)

F267 was closed from the local F198 parity catalogue and the already shipped F252.2 UXP dispatcher surface. The new `build_udt_harness_manifest()` emits one UDT scenario per direct-UXP host action, with payloads, fixture needs, mutation/file-write safety flags, expected result keys, and acceptable environment blockers. The generated JSON is committed both under `opencut/_generated/` and beside the UXP panel.

The panel-side `udt-smoke.js` exposes `window.OpenCutUXPUdtHarness`, defaulting to safe-by-default scenarios and requiring `includeMutating: true` for destructive/file-writing runs in a disposable Premiere project. Validation evidence for the pass: focused F267 tests (`6 passed`), focused UXP/release-smoke slice (`61 passed`), touched Python compile, focused Ruff, harness sync check, UXP JS syntax check, and release-smoke `pytest-fast` (`666 passed`).

---

## Pass 68 implementation note (2026-05-18)

F263 was closed from live dependency-resolution evidence. `requirements.txt` still audits clean, while the previous `pyproject[all]` gap exposed stale optional pins and Torch-stack conflicts. The pass added `opencut.tools.pip_audit_extras` so release smoke now audits `requirements.txt` and `pyproject[all]` through structured per-target JSON with isolated pip caches.

Implementation refreshed the `[all]` dependency set (`transnetv2-pytorch`, `auto-editor>=29.3`, `otio-aaf-adapter>=2.0`, `pyannote.audio>=4.0`) and kept AudioCraft/MusicGen plus Resemble Enhance as explicit Python 3.11-only extras because their published packages hard-pin older Torch stacks. `docs/PYTHON_ADVISORIES.md` records the two currently allowed live findings: BasicSR `CVE-2024-27763` and Transformers `CVE-2026-1839`. Validation evidence for the pass: focused F263 tests (`34 passed`), touched Python compile, focused Ruff, model-card/readiness sync checks, live `pip_audit_extras --json --extra all`, and release-smoke `--only pip-audit`.

---

## Pass 69 implementation note (2026-05-18)

F271 was closed by extending the existing feature-readiness surface rather than creating a separate VRAM registry. The generated readiness manifest already joins route probes to model cards, so the pass added hardware/minimum-VRAM fields there and lets `/system/feature-state` carry the metadata to panel controls.

Implementation details: `dump_feature_readiness` parses model-card hardware strings such as `gpu (>= 12 GB VRAM)`, `registry._merge_generated_records()` now enriches hand-written records by `feature_id` as well as check probe, and `feature-state.js` adds `hardwareFor(featureId)` plus hardware/min-VRAM data attributes and tooltips. Validation evidence for the pass: focused F271 tests (`34 passed`), touched Python compile, focused Ruff, feature-readiness sync check, and CEP helper syntax check.

---

## Pass 70 implementation note (2026-05-18)

F272 was closed by shipping the wedding workflow as a built-in agent skill instead of waiting for the full F145 runtime. The package lives at `opencut/data/builtin_skills/wedding-cinematic-reel/` and contains both SKILL.md instructions and a schema-versioned plan for color match, beat markers, highlight extraction, beat-synced assembly, and four-minute review-master export planning.

Implementation details: `opencut.core.agent_skills` provides the built-in skill loader and plan validator, `solver_agent_routes.py` exposes read-only `/agent/skills` catalogue routes, and generated route/MCP artifacts were refreshed to 1,376 routes and 1,319 opt-in route tools. Validation evidence for the pass: focused F272 tests (`4 passed`), focused route/skill tests (`17 passed`), touched Python compile, focused Ruff, generated route/MCP checks, roadmap lint, and release-smoke `pytest-fast` (`691 passed`).

---

## Pass 71 implementation note (2026-05-18)

F193 was closed by making OpenAPI response schemas dataclass-discovered rather than table-maintained inside `opencut.openapi`. The new `opencut.openapi_registry` carries the route metadata contract, while `opencut.schemas` and selected safe core result modules register their dataclasses for discovery.

Implementation details: the legacy `_ENDPOINT_SCHEMAS` compatibility map is now generated from registry discovery, `opencut.openapi` resolves nested dataclass fields and computed properties, and `mcp_extended_tools` reads the same registry map for response-schema annotations. Validation evidence for the pass: focused OpenAPI tests (`6 passed`), focused OpenAPI+MCP tests (`15 passed`), touched Python compile, focused Ruff (`E,F,I`), roadmap lint, extended MCP manifest sync check, and release-smoke `pytest-fast` (`693 passed`).

---

## 1. Phases executed

| Phase | What | Output |
|---|---|---|
| 1. Local reconnaissance | Listed repo root, read README + AGENTS + CLAUDE in chunks (CLAUDE.md is 1,509 lines — limited reads). Listed `opencut/routes/` (101 blueprints) + `opencut/core/` (523 files) + `tests/` (131 files). Inspected `pyproject.toml`, `requirements*.txt`, `scripts/`, `.github/`, `opencut/_generated/`. | `STATE_OF_REPO.md` |
| 2. Roadmap / changelog reconciliation | Read ROADMAP.md (sampled 200 + 200 + 500 lines through v4.3 audit), ROADMAP-NEXT.md (sampled 200 + 400 + 250 lines through Wave K), ROADMAP-COMPLETED.md, MODERNIZATION.md, AUDIT.md, research.md, features.md (first 100 lines), CHANGELOG.md (first 200 lines), CODEX-CHANGELOG.md. Cross-referenced against `git log --oneline -60`. | `MEMORY_CONSOLIDATION.md` + `STATE_OF_REPO.md` |
| 3a. SOTA AI models research (subagent) | Briefed with full known-integrated-model list and instruction to find only post-April 2026 additions. | Returned 30 high-signal items (daVinci-MagiHuman, LTX-2.3, Wan 2.7, MAGI-1, StreamDiffusionV2, DiffSynth, OmniVoice, IndexTTS2, VoxCPM2, Qwen3-TTS, Fish Speech S2 Pro, HunyuanVideo-Foley, Mimi codec, SAM 3/3.1, Depth Anything 3, Online Video Depth Anything, SeedVR/2.5, FlashVSR v1.1, MatAnyone 2, C2PA 2.3, IMSC 1.3, OCIO 2.5, VideoAgent, FireRed, vibeframe) with licences. |
| 3b. Premiere/UXP/competitor research (subagent) | Briefed with CEP EOL timeline + UXP API gaps known to OpenCut + ask for chat-agent UX patterns and DaVinci 21 release. | Returned: Premiere 26.0/26.0.1/26.0.2/26.2/26.2.2 release notes, UXP API state (stable list + confirmed missing: createCaptionTrack, createSubsequence, exportAsFinalCutProXML, startDrag, QE DOM), CEP EOL Sept 2026 confirmed, Bolt UXP WebView (Mar 2026), 3 PPro MCP competitors (all CEP-bound), DaVinci 21 features, agentic-editor UX patterns (Underlord post-turn self-review), top 10 community pain points. |
| 3c. Dependency / security advisories (subagent) | Briefed with full `pyproject.toml` pin set. | Returned: Pillow CVE-2026-40192/25990, flask-cors 4.x CVEs, pydub broken on 3.13, basicsr abandonment + functional_tensor break, audiocraft torch==2.1.0 cascade, OpenTimelineIO adapter split, Transformers v5, pyannote 4.x, PySceneDetect 0.7, esbuild GHSA-67mh-4wv8-2f99 (already mitigated), Vite 8, FFmpeg 8.1, ONNX Runtime 1.25 CVEs. |
| 4. Synthesis | Cross-referenced subagent outputs against existing F001-F120 ledger to identify which items the team had already considered and which are genuinely new. Continued F-numbering at F121 to avoid collision with ROADMAP.md v4.3. | `FEATURE_BACKLOG.md`, `PRIORITIZATION_MATRIX.md` |
| 5. Composition | Wrote artefacts in dependency order: STATE_OF_REPO → MEMORY_CONSOLIDATION → SECURITY → COMPETITOR → DATASET → FEATURE_BACKLOG → PRIORITIZATION → SOURCE_REGISTER → RESEARCH_LOG. Then PROJECT_CONTEXT.md at root, append ROADMAP.md v4.4 section, add canonical pointers to AGENTS.md + CLAUDE.md. | All artefacts |
| 6. Self-audit | This file + `CHANGESET_SUMMARY.md`. | — |

---

## 2. Subagent prompts (full text)

Recorded for reproducibility / next-session continuation.

### 2.1 SOTA AI models research

Brief: complete model list of what OpenCut already has, plus instruction to find only post-April-2026 additions and to flag licences. 1,200-word target.

### 2.2 Premiere/UXP/competitor research

Brief: Premiere 26.x release notes, UXP API status, CEP EOL, Bolt WebView, MCP servers, DaVinci 21, agent UX, community pain points. 1,500-word target.

### 2.3 Dependency / security advisories

Brief: full pyproject pin set + known issues (deep-translator, Vite/esbuild) + check Python 3.13 compatibility + basicsr abandonment + audiocraft pin + Transformers v5. 1,500-word target.

Full prompts are also in the live session transcript / agent JSONL files.

---

## 3. Search strategies the subagents used (paraphrased from their summaries)

| Strategy | Examples |
|---|---|
| Direct GitHub repo enumeration | `daVinci-MagiHuman`, `MAGI-1`, `Wan 2.7`, `DiffSynth-Studio`, `StreamDiffusionV2`, `OmniVoice`, `IndexTTS2`, `VoxCPM`, `SAM 3.1`, `Depth Anything 3`, `MatAnyone2` |
| Adobe documentation crawl | `helpx.adobe.com/premiere/desktop/whats-new/release-notes.html`, `developer.adobe.com/premiere-pro/uxp/ppro_reference/`, `AdobeDocs/uxp-premiere-pro-samples` |
| arXiv / paper search | StreamDiffusionV2 (2511.07399v2), Online Video Depth Anything (2510.09182), DriftSE (2604.24199v2) |
| Advisory DBs | GitHub Advisory DB (GHSA-67mh-4wv8-2f99), OSV.dev (PYSEC-2022-252), SentinelOne (CVE-2026-40192) |
| PyPI release pages | Pillow, flask-cors, faster-whisper, whisperx, pyannote.audio, scenedetect, audiocraft, MCP |
| Vendor release notes | Vite 8 announcement, FFmpeg 8.1 release, PyTorch 2.11 blog, Transformers v5 blog |
| Standards orgs | c2pa.org, w3.org/news, opencolorio.org |
| Community forums | creativeclouddeveloper.com forum threads, HN discussions, Reddit Premiere/FFmpeg searches |

---

## 4. Saturation tests performed

| Category | Saturation? | Evidence |
|---|---|---|
| AI video generation models (post-Apr 2026) | **Saturated** — 4 separate top-of-class items (daVinci-MagiHuman, LTX-2.3, Wan 2.7, MAGI-1) plus StreamDiffusionV2 covering the streaming axis. Further search returned closed-source (Sora 2, Veo 3.1, Kling 3.0) or research-only (MirageLSD). | Subagent A returned the same items consistently across multiple queries. |
| TTS / voice (post-Apr 2026) | **Saturated** — OmniVoice (600+ langs), IndexTTS2 (duration control), Fish Speech S2 Pro, VoxCPM2, Qwen3-TTS all surfaced. Skip-list (Voxtral) clearly distinct. | Same. |
| Restoration / depth / matting | **Saturated** — SAM 3.1, Depth Anything 3, SeedVR2.5, MatAnyone 2 surfaced; FlashVSR v1.1 confirmed as stable pin. No further candidates of merit. | Same. |
| Premiere UXP gaps | **Saturated** — creativeclouddeveloper.com forum thread is the canonical living catalogue. Adobe blog (Apr 2026) confirms Hybrid Plugins as the escape hatch. | Subagent B exhausted the API gap list. |
| Premiere 26.x release notes | **Saturated through 26.2.2 (May 2026)**. | Adobe release-notes page direct. |
| MCP servers for Premiere | **Saturated** — 3 active (all CEP-bound) + 1 adjacent (vibeframe). No fourth surfaced. | Subagent B's GitHub topic walk. |
| DaVinci Resolve 21 features | **Saturated** — Blackmagic PDF + multiple journalist write-ups confirm same feature list. | Subagent B + research.md preliminary list. |
| Dependency CVEs | **Saturated for top-level deps**. Transitive list could go deeper (e.g. ctranslate2 sub-deps) but the audit ID + recommended action would not change. | Subagent C tabled every pin. |
| Python 3.13 compat | **Saturated** — pydub is the lone blocker; everything else has either a wheel or a documented shim. | Subagent C + audioop-lts research. |
| Agentic editor UX | **Saturated** — Underlord, Captions/Mirage, Odysser, FireRed, vibeframe, VideoAgent, ViMax, Crayo, CrePal. Pattern (sidebar chat + timeline diff + self-review + skills) is consistent. | Subagent B. |

---

## 5. Failed searches / dead ends

- **Crayotter** (cited in research.md) — no current online presence. Likely typo / rename — best fit is **Crayo.ai** or **CrePal**. Treated as ambiguous in this run.
- **HappyHorse 1.0** licence — Alibaba announcement Apr 2026 but licence not yet confirmed in public sources. Marked conditional.
- **Wan 2.7 weights upload status** — announced as Apache 2.0 but the weights page status couldn't be confirmed in this run. Marked conditional in F165.
- **Fish Speech S2 Pro licence** — fishaudio/fish-speech repo historically Apache but version-specific licence couldn't be confirmed. Marked verify-licence in F171.
- **IndexTTS2 licence** — repo historically Apache; verify per version. Marked verify in M12 / F168.
- **HunyuanVideo-Foley licence** — Tencent; territory carve-outs likely (same posture as HunyuanVideo). Marked conditional.
- **PrismAudio weights** — research-only confirmed despite Apache code.
- **Tencent Hunyuan-1.5 territory carve-outs** — confirmed in research; some marketing materials elide this.

---

## 6. Bias / assumptions to flag

1. **Dependency upgrade aggression**: this audit recommends bumping every pinned dep with an open CVE or 2026 release. A conservative reading would defer Pillow 12 / flask-cors 6 to v1.34.0 to avoid bundling churn. The audit prioritises CVE closure over conservatism, justified by the *Now* tier's "release trust" framing in v4.3.
2. **Transformers v5 + Python 3.10 floor**: this is a strategic call. The audit recommends an RFC + decision in v1.33.0, implementation in v1.34.0. A more conservative project might delay another quarter.
3. **`/agent/chat` flagship**: the audit places F143 in *Next* as the highest-leverage item. This is a judgement call based on competitor signal (Underlord, Captions, FireRed). A maintainer focused on UXP migration might prefer to ship F146 (UXP MCP transport) first.
4. **Wave T+ items**: ROADMAP.md already names wave T (agent ecosystem, TTS fleet refresh, video diffusion modernisation) but does not yet have F-numbered tier placements for them. The audit deliberately re-numbered into F-series F121+ instead of letter-prefixed waves; F180 calls for a re-tier of the wave letters into F-numbers.
5. **Sept 2026 CEP EOL is treated as firm**. If Adobe extends the date, F146 / F160 timing relaxes — but it would be irresponsible to plan around that hope.

---

## 7. Things this audit did NOT do

- Did **not** read the full 1,509-line CLAUDE.md (the read-tool token limit forced sampling at 300 + 600-line chunks).
- Did **not** walk all 402 entries in `features.md` and mark each as `[shipped]` / `[planned]` / `[rejected]` — that's a 1-2 day cleanup pass and is on the backlog as F179.
- Did **not** re-test the regression suite (`scripts/release_smoke.py`, `python -m pytest`) — this is read-only research.
- Did **not** commit the 7-file dirty hardening batch — user authorisation required (recommendation noted in artefacts).
- Did **not** run `python -m opencut.tools.dump_route_manifest --check` — relied on the cached `route_manifest.json` showing 1,359 routes.
- Did **not** push the 25-commit backlog to `SysAdminDoc/OpenCut` — auth blocked from this VM per the [SwiftFloris git-auth memory pattern](C:\Users\Xray\.claude\projects\C--Users-Xray\memory\swiftfloris-git-auth.md) (same constraint).
- Did **not** generate a fresh SBOM via `scripts/sbom.py` — last F094 SBOM is current.
- Did **not** verify Wan 2.7 weights publication; deferred to per-integration-time gate.

---

## 8. Continuation hints for next session

If a next research session picks this up:

1. **Verify Wan 2.7 weights** are actually live on the model zoo before scheduling F165 implementation.
2. **Verify Fish Speech S2 Pro / IndexTTS2 / HunyuanVideo-Foley licences** before scheduling F171 / F168 / F172 implementation.
3. **Commit the 7-file dirty hardening batch** as a single PR titled "Harden auth loopback classification, UNC realpath, helper lifecycle, and safe_bool follow-up" — see SECURITY_AND_DEPENDENCY_REVIEW.md § 5.
4. **Run the `gh issue` seeder** to populate the contributor channel (F182).
5. **Walk `features.md` row-by-row** for the F179 reconciliation cleanup (1-2 days).
6. **Re-test bundled FFmpeg 8.1** under F128 before scheduling F129 installer-bundle bump.
7. **Issue Adobe gap reports F186-F190** — these are the unique high-leverage non-code items.

A blank `CONTINUE_FROM_HERE.md` was **not** written because this audit completed within session budget; hand-off content is captured in this RESEARCH_LOG.md and `CHANGESET_SUMMARY.md`.

---

## Pass 2 (2026-05-17 — second autonomous research run, same calendar day)

The user pasted the same autonomous-research prompt a second time. Pass 1 had completed all mandated artefacts; the prompt's "do not stop after one research pass" directive made Pass 2 a deeper investigation rather than a restart.

### Pass 2 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 2.1 | Deep local recon — full read of `opencut/checks.py` (105 functions, R-L42), `opencut/registry.py` (29 FeatureRecord entries, R-L43), `opencut/openapi.py` (30 schema-mapped endpoints, R-L44), `opencut/mcp_server.py` first 200 lines + full `MCP_TOOLS` (27 tools, R-L45), `route_manifest.json` per-blueprint walk (R-L46), `model_cards.json` full license breakdown (R-L47), CLAUDE.md lines 300-500 (R-L53), installer/src walk (2,326 lines C#, R-L48), `.github/workflows/build.yml` (R-L49), `.github/issue-seeds.yml` (15 seeded items, R-L50), features.md sample walk (40 entries across 12 categories, R-L52), tests/ listing + skip-marker scan (R-L54). | `ROUTE_READINESS_AUDIT.md`, `INSTALLER_AUDIT.md`, `TEST_COVERAGE_GAPS.md`, `FEATURES_RECONCILIATION.md` |
| Pass 2.2 | External research wave 2 — three parallel subagents on uncovered ground: (a) Frame.io / collab review platforms; (b) niche AI / accessibility / standards / packaging / telemetry / WebGPU; (c) Adobe UXP samples + `@adobe/premierepro` typings + Hybrid Plugins + post-CEP migration patterns. | Three subagent briefs incorporated into `FEATURE_BACKLOG_ADDENDUM.md` (F225-F260) |
| Pass 2.3 | Synthesis — added F191-F260 (70 new F-numbers) to `FEATURE_BACKLOG_ADDENDUM.md`. Bumped F202 (Apple notarisation) and F236 (FCC caption tokens) to *Now* tier because of regulatory deadlines (Homebrew Cask Sept 1 2026, FCC effective Aug 17 2026). Updated `PRIORITIZATION_MATRIX.md` §6.5 with Pass-2 tier deltas. | Updated existing artefacts |
| Pass 2.4 | Documentation update — appended ROADMAP v4.5 delta section, refreshed PROJECT_CONTEXT.md numbers + biggest-gaps list, wrote CONTINUE_FROM_HERE.md for Pass 3. | `ROADMAP.md`, `PROJECT_CONTEXT.md`, `CONTINUE_FROM_HERE.md` |

### Pass-2 subagent prompts

All three subagents were briefed with: (a) what Pass 1 had already surfaced (so they wouldn't duplicate), (b) explicit "go deeper into corners Pass 1 didn't reach" framing, (c) licence-flagging requirement, (d) ~1,500-word target. Full prompts preserved in the live session transcript / agent JSONL files.

### Pass-2 saturation tests

| Category | Saturation | Evidence |
|---|---|---|
| Frame.io / OSS review platforms | **Saturated** — Clapshot, FreeFrame, OpenFrame, OpenVidReview, video-review surface in repeated searches. OTIO Marker as the right interchange anchor became obvious across all sources. Closed-source landscape (Frame.io, Wipster/Memento, Vimeo Review, Iconik, Filestage, Krock, Picflow, MediaSilo, Replay) is well-mapped. | Frame.io subagent §2 |
| WCAG 3.0 draft + FCC + EBU + ITU-R deltas | **Saturated** | Niche-AI subagent §2 |
| RTL / CJK / Indic rendering gaps | **Saturated** — HarfBuzz is the universal answer; libass / Pillow / Skia all need explicit linking; ICU4X for CJK line breaking | Niche-AI subagent §3 |
| 2026 packaging deadlines | **Saturated** — Homebrew Cask Sept 1 2026, Windows cert validity drop March 2026, Apple notarisation requirements, winget/Chocolatey/Scoop hierarchy | Niche-AI subagent §5 |
| UXP API surface | **Saturated** — full `@adobe/premierepro@26.3.0-beta.67` typings walked; 5 truly CEP-blocked features identified; `createSubsequence` confirmed present (Pass 1 had inferred it missing); known macOS HTTP bug documented; CY2026 promise for `startDrag` tracked | UXP subagent §1-§4 + §10 |
| Hybrid Plugins | **Saturated** for current state — Adobe SDK location, C++ header surface, mac/win build conventions, Bolt UXP 1.3 template existence. Adopter case studies are still thin (Adobe only mentions OpenCV/TFLite/codecs as use cases). | UXP subagent §4 |

### Pass-2 failed searches / dead ends

- **Hailuo 2.3 / Seedance 2.0 closed-source quality benchmarks** — couldn't get current head-to-head vs LTX-2.3 / Wan 2.7. Closed APIs don't publish reproducible benchmarks. Treat as opaque comparisons.
- **Adobe's hard "CEP off date"** — confirmed Pass 1 finding: Sept 2026 is community-confirmed via Adobe staff in forum threads, **not** published in writing anywhere. Pass 2's UXP subagent §7 took the conservative interpretation: "ExtendScript may stop being patched" not "panels stop loading." Plan accordingly.
- **Wan 2.7 weights publication status** — Pass 1 conditional remains. Pass 2 didn't find a definitive "weights live on HF" confirmation. F165 remains gated.
- **WPF installer adoption stats** — couldn't determine what % of OpenCut users install via the WPF installer vs Inno Setup vs Docker vs source. Without telemetry (F250) this is unanswerable.
- **OpenCut user count / install base** — no telemetry, no published download counts. The decision to ship F202 (Apple notarisation) for "the Mac user base" cannot be sized empirically.

### Pass-2 bias / assumptions

1. **F202 + F236 priority bump** — both regulatory deadlines (Aug 17 + Sept 1 2026) take precedence over feature work even though they don't ship user-visible features. A maintainer might disagree if their actual user base is overwhelmingly Windows + non-US.
2. **OTIO Marker schema as review-bundle interchange** — assumes future NLE-side adoption of OTIO marker color/comment fields. The schema is standards-blessed but not universally consumed yet.
3. **WebView UI in UXP migration** — Pass 2 strongly recommends Bolt UXP WebView (March 2026) for the 8,500-line CEP main.js. A maintainer who has already invested in Spectrum UXP widgets may have different priorities.
4. **No retired-product check** — Pass 2 didn't verify that all 47 model cards still have a live upstream (e.g. is `realesrgan>=0.3,<1` still installable from PyPI today?). The dep-audit subagent flagged dead packages but didn't try a live `pip install` matrix.

### Pass-2 things explicitly NOT done

- Did **not** complete the full F179 features.md reconciliation — only a representative 40-entry sample walk. The remaining ~370 entries need a follow-up 1-2 day pass.
- Did **not** commit the dirty hardening batch (still 7 files modified, recommendation unchanged).
- Did **not** run `python -m opencut.tools.dump_route_manifest --check` to verify the cached 1,359-route figure against live `url_map`.
- Did **not** run `python scripts/release_smoke.py --json` end-to-end to validate the F098 runner.
- Did **not** check live PyPI for `pip install` matrix of pinned deps (would catch silent-dead packages).
- Did **not** read the full 1,509-line CLAUDE.md (Pass 1 sampled lines 1-300; Pass 2 sampled 300-500. Lines 500-1509 still unread; mostly v1.18.0 and earlier change history — lower-priority context for forward planning).
- Did **not** look at `tests/fuzz/test_parser_fuzz.py` source to validate the 5 documented fuzz targets.
- Did **not** read `extension/com.opencut.uxp/uxp-api-notes.md` — should be read in Pass 3 for the UXP migration plan.

### Pass-2 continuation hints for Pass 3

If a Pass 3 session picks this up:

1. **Run the F098 release-smoke matrix end-to-end** and capture which steps actually fail vs the v4.3 audit's claims.
2. **Live PyPI / npm matrix check** of every pinned dep — see what installs cleanly today on Python 3.12 / Python 3.13 / Node 22.
3. **Walk CLAUDE.md lines 500-1509** — extract any remaining patterns/gotchas not yet captured.
4. **Complete the full F179 features.md reconciliation** — walk all 410 entries; emit `features_status.md` companion file.
5. **Read `extension/com.opencut.uxp/uxp-api-notes.md`** — internal CEP-vs-UXP comparison the team maintains.
6. **Walk `tests/fuzz/test_parser_fuzz.py`** + audit which validators have fuzz coverage vs F215's 8 proposed targets.
7. **Inspect `opencut/preflight.py` and `opencut/workers.py`** — Pass 2 didn't read these.
8. **Inspect `opencut/journal.py`** — Operation Journal feature referenced in CLAUDE.md but not deeply understood.
9. **Inspect the actual `tests/conftest.py`** to understand the Flask test fixture pattern.
10. **Trace `ChatGPT-class agent UX patterns`** further — Underlord, FireRed, vibeframe got coverage; **Cursor's IDE-agent pattern** for video editing wasn't surveyed (could be a model for OpenCut's F143).

These hints are also captured in `CONTINUE_FROM_HERE.md` for visibility.

---

## Pass 4 (2026-05-17 — release-smoke validation and commit prep)

Pass 4 was a verification and consolidation pass, not a new feature-harvest wave. It closed the largest item left in the Pass-3 handoff: full `scripts/release_smoke.py --json`.

### Pass 4 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 4.1 | Re-ran the local governance gates directly: route manifest, version sync, bootstrap, pip-audit, raw npm audit, Ruff on touched files, and the targeted hardening pytest slice. | `LIVE_VERIFICATION.md` §8 |
| Pass 4.2 | Ran `python scripts/release_smoke.py --json`. First run failed only on Ruff; all other gates passed. | Captured in `LIVE_VERIFICATION.md` §8 |
| Pass 4.3 | Applied safe Ruff cleanup in the release-smoke lint scope (`opencut/`, `scripts/`): unused imports and import ordering only. | Code diff; release-smoke rerun |
| Pass 4.4 | Reran `python scripts/release_smoke.py --json`; it exited `0`. | `ROADMAP.md` v4.7, `PROJECT_CONTEXT.md` §9.4 |
| Pass 4.5 | Refreshed source evidence for official Adobe UXP/Hybrid Plugin docs, OTIO adapter split, Vite advisory, and `@adobe/premierepro` npm dist-tags. | `SOURCE_REGISTER.md` Pass 4 section |

### Pass 4 validation results

| Check | Result |
|---|---|
| Targeted hardening pytest slice | **PASS** — `119 passed` |
| Full release smoke | **PASS** — `python scripts/release_smoke.py --json` exit `0` |
| Release-smoke pytest-fast | **PASS** — `232 passed` |
| Release-smoke Ruff | **PASS** after safe Ruff fixes |
| pip-audit | **PASS** — no known vulnerabilities |
| npm advisory posture | release-smoke allow-list **PASS**; raw `npm audit --json` still shows the known moderate Vite `.map` advisory documented by F095 |

### Pass 4 things explicitly NOT done

- Did not run the full pytest suite outside the release-smoke fast gate.
- Did not run a Python 3.10 / 3.11 / 3.13 install matrix.
- Did not run Premiere UDT for JSX to UXP parity; this still needs a real host.
- Did not implement F261/F262/F270; those remain next-session implementation work.

### Pass 4 saturation note

The research corpus is now saturated enough for the user-requested roadmap operation. Remaining gaps are execution tasks (F261, F262, F270), strategic decisions (F127, F161, F200, F252), or multi-day audits (F179). Additional web searching would mostly duplicate the source classes already captured unless it focuses on one of those deferred items.

---

## Pass 9 (2026-05-17 — F195 MCP implementation)

Pass 9 was an implementation pass, not a new external research pass. It used local route sources only.

### Pass 9 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 9.1 | Inspected `opencut/mcp_server.py` and the shipped route modules for the 12 missing post-Wave-M surfaces. | Confirmed routes and payload shapes for F195. |
| Pass 9.2 | Added curated MCP tools, route mappings, multi-action dispatch for Brand Kit / semantic search, and path-validation coverage for new path keys. | `opencut/mcp_server.py` |
| Pass 9.3 | Added focused MCP tests and release-smoke gate coverage. | `tests/test_mcp_server.py`, `scripts/release_smoke.py` |
| Pass 9.4 | Updated roadmap/context/research artefacts with 27→39 MCP tool count and F195 closure. | `ROADMAP.md`, `PROJECT_CONTEXT.md`, `.ai/research/2026-05-17/*` |

### Pass 9 validation results

| Check | Result |
|---|---|
| Python compile on touched Python files | **PASS** |
| Focused MCP + release-smoke tests | **PASS** — `17 passed` |
| Ruff on touched files | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `246 passed` |

### Pass 9 saturation note

No new external searching was needed. The next research-heavy gaps are still F179 (features.md reconciliation), F202/F236 regulatory implementation detail, and the Python cross-version install matrix. The next local-verifiable implementation gaps are F208/F209/F218/F219/F204.

---

## Pass 10 (2026-05-17 — F202 macOS notarization tooling)

Pass 10 was an implementation pass with a small official-doc refresh against Apple notarization guidance.

### Pass 10 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 10.1 | Inspected the existing GitHub Actions build workflow and PyInstaller spec. | Confirmed macOS builds produce `dist/OpenCut-Server` but had no signing/notarization step. |
| Pass 10.2 | Checked Apple Developer docs for current notarization tooling. | Confirmed `notarytool` path and `altool` retirement. |
| Pass 10.3 | Added macOS signing/notarization script and wired it into tagged/manual macOS release builds. | `scripts/notarize_macos.sh`, `.github/workflows/build.yml` |
| Pass 10.4 | Added docs and static tests for required secrets, notarytool usage, hardened-runtime signing, and release upload wiring. | `docs/MACOS_NOTARIZATION.md`, `tests/test_macos_notarization.py` |

### Pass 10 validation results

| Check | Result |
|---|---|
| Focused notarization/release-smoke tests | **PASS** — `15 passed` |
| Ruff on touched Python files | **PASS** |
| Shell syntax check via Git Bash | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `249 passed` |

### Pass 10 saturation note

Repository-side F202 tooling is complete enough to run on a macOS GitHub Actions release runner once secrets exist. The remaining evidence gap is external and credential-bound: first live Apple notary acceptance with real Developer ID and App Store Connect API credentials.

---

## Pass 11 (2026-05-17 — F204 release SBOM attachment)

Pass 11 was a local release-plumbing implementation pass. No new external research was needed because `scripts/sbom.py` already existed and emitted CycloneDX 1.5 JSON.

### Pass 11 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 11.1 | Inspected `scripts/sbom.py` and the release workflow. | Confirmed F204 was missing only CI generation/upload wiring. |
| Pass 11.2 | Added Linux release steps to generate/archive/upload the SBOM. | `.github/workflows/build.yml` |
| Pass 11.3 | Added tests that run the generator and pin release workflow wiring. | `tests/test_release_sbom.py`, `scripts/release_smoke.py` |
| Pass 11.4 | Updated roadmap/research artifacts and preserved F219 as the deeper completeness test. | `ROADMAP.md`, `.ai/research/2026-05-17/*` |

### Pass 11 validation results

| Check | Result |
|---|---|
| Focused SBOM/release-smoke tests | **PASS** — `14 passed` |
| Ruff on touched Python files | **PASS** |
| Workflow YAML parse | **PASS** |
| Direct SBOM generation | **PASS** — generated CycloneDX 1.5 JSON |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `251 passed` |

### Pass 11 saturation note

F204 is complete as release plumbing. Pass 16 later closed F219, which enforces SBOM completeness against all declared dependencies, model cards, and dependency graph entries.

---

## Pass 12 (2026-05-17 — F205 attempt + F207 installer FFmpeg manifest)

Pass 12 began with F205 because it was next in the Now queue, but the measurement could not complete locally. The pass then moved to F207, which was local-verifiable.

### Pass 12 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 12.1 | Tried to run the CI coverage measurement; local env lacked `pytest-cov`/`pytest-xdist`. | Installed missing plugins with pip. |
| Pass 12.2 | Reran the CI-style coverage measurement with `--cov-fail-under=0`. | Timed out after 20 minutes; no coverage JSON. F205 remains open. |
| Pass 12.3 | Read WPF/Inno installer code and the bundled FFmpeg binaries. | Confirmed current bundled version is `8.0.1-essentials_build-www.gyan.dev`. |
| Pass 12.4 | Added FFmpeg version constants and WPF/Inno `installer.json` writers. | `AppConstants.cs`, `InstallEngine.cs`, `OpenCut.iss` |
| Pass 12.5 | Added static tests and release-smoke gate inclusion. | `tests/test_ffmpeg_installer_manifest.py`, `scripts/release_smoke.py` |

### Pass 12 validation results

| Check | Result |
|---|---|
| Bundled FFmpeg version probe | **PASS** — first line reports `8.0.1-essentials_build-www.gyan.dev` |
| Focused F207/release-smoke tests | **PASS** — `15 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for `scripts/release_smoke.py` | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `254 passed` |
| WPF .NET build | **BLOCKED** — no .NET SDK installed on this VM |

### Pass 12 saturation note

F207 is complete at repository/source level. F205 should resume on a machine or CI runner where the full coverage command can finish; do not infer a new threshold from partial local data.

---

## Pass 13 (2026-05-17 — F208 OpenAPI contract gate)

Pass 13 was a local implementation and verification pass. No new external research was needed; the work compared the live Flask `url_map` against the two in-repo OpenAPI generators.

### Pass 13 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 13.1 | Inspected the legacy `opencut/openapi.py` generator, `/api/openapi.json` generator, and existing OpenAPI tests. | Confirmed `/openapi.json` still emitted Flask `<param>` syntax and duplicate operation IDs for aliased endpoints. |
| Pass 13.2 | Added path conversion, path parameter metadata, unique path-qualified operation IDs, and mutating-method error responses to `opencut/openapi.py`. | `/status/<job_id>` now appears as `/status/{job_id}` with a required string path parameter. |
| Pass 13.3 | Added `tests/test_openapi_contract.py` and wired it into release smoke. | The test compares root OpenAPI operations to every live non-static Flask route/method and checks response/schema shape. |
| Pass 13.4 | Updated roadmap and research state files. | F208 marked closed; F205 remains open after the Pass 12 coverage timeout. |

### Pass 13 validation results

| Check | Result |
|---|---|
| Focused OpenAPI/release-smoke tests | **PASS** — `16 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `258 passed` |

### Pass 13 saturation note

F208 is complete for structural OpenAPI validity and route coverage. It does not close F192/F193, which remain the richer typed-schema expansion and introspection work for the ~1,250 mostly generic response schemas.

---

## Pass 14 (2026-05-17 — F209 MCP route consistency gate)

Pass 14 was a local implementation and verification pass. No new external research was needed; the work compared the MCP route table against the live Flask `url_map`.

### Pass 14 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 14.1 | Compared all `_TOOL_ROUTES` entries to live Flask operations. | Found one stale target: `opencut_chat_edit` pointed to planned `/agent/chat` instead of shipped `/chat`. |
| Pass 14.2 | Corrected the stale MCP route mapping. | `opencut_chat_edit` now dispatches to `POST /chat`. |
| Pass 14.3 | Added an MCP route-consistency test. | `tests/test_mcp_server.py` now checks all 39 tools, all default route mappings, and special action routes. |
| Pass 14.4 | Updated roadmap and research state files. | F209 marked closed; next local-verifiable Now items are F218/F219. |

### Pass 14 validation results

| Check | Result |
|---|---|
| Focused MCP/release-smoke tests | **PASS** — `18 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Live route-table probe | **PASS** — 39 MCP tools / 39 route mappings / 0 missing backend routes |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `259 passed` |

### Pass 14 saturation note

F209 is complete for curated MCP route drift. It does not generate new MCP tools from the full route catalogue; that remains F194/T1.5-style extended MCP surface work.

---

## Pass 15 (2026-05-17 — F218 blueprint import-order stability)

Pass 15 was a local test-hardening pass. No new external research was needed; the work pinned the existing explicit blueprint registration order.

### Pass 15 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 15.1 | Inspected `opencut/routes/__init__.py` and existing route-collision tests. | Confirmed explicit `get_core_blueprints()` order plus special `motion_design_api` registration. |
| Pass 15.2 | Added a blueprint order regression test. | `tests/test_route_collisions.py` now pins the 99 core blueprint names and final alias registration. |
| Pass 15.3 | Added the route-collision/import-order test file to release smoke. | `scripts/release_smoke.py` now includes `tests/test_route_collisions.py`. |
| Pass 15.4 | Updated roadmap and research state files. | F218 marked closed; F219 is the next local-verifiable Now item. |

### Pass 15 validation results

| Check | Result |
|---|---|
| Focused route-collision/release-smoke tests | **PASS** — `19 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `266 passed` |

### Pass 15 saturation note

F218 is complete for deterministic built-in blueprint order. It does not test third-party plugin registration order, which remains dynamic by design and should stay covered by plugin manifest validation rather than this core-order gate.

---

## Pass 16 (2026-05-17 — F219 SBOM completeness gate)

Pass 16 was a local implementation and verification pass. No new external research was needed; the work compared the release SBOM generator against the repo's declared dependency and model-card sources.

### Pass 16 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 16.1 | Inspected `scripts/sbom.py`, `tests/test_release_sbom.py`, `pyproject.toml`, `requirements.txt`, and `opencut/_generated/model_cards.json`. | Confirmed F204 only smoked SBOM shape/workflow upload; it did not assert dependency completeness, model-card coverage, or a `dependencies` graph. |
| Pass 16.2 | Extended `scripts/sbom.py`. | SBOM output now has unique declared Python dependency components, 47 model-card components, JSON/XML dependency graph output, and CLI component counts. |
| Pass 16.3 | Added `tests/test_sbom_completeness.py` and wired it into release smoke. | The test asserts every declared dependency is present, every committed model card is represented, `bom-ref` values are unique, and every component is in the dependency graph. |
| Pass 16.4 | Updated roadmap and research state files. | F219 marked closed; the remaining local-verifiable Now queue starts with F237/F240/F243/F244 after the F205 coverage timeout. |

### Pass 16 validation results

| Check | Result |
|---|---|
| Focused SBOM/release-smoke tests | **PASS** — `17 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| JSON/XML SBOM generation | **PASS** — 14 required components / 73 optional components / 47 model-card components |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `269 passed` |

### Pass 16 saturation note

F219 is complete for in-repo SBOM completeness against declared Python dependencies and OpenCut model cards. It does not claim installed transitive dependency capture; `scripts/sbom.py` remains a declaration-level SBOM generator, while installed-environment SBOMs should still use tools such as `cyclonedx-py` or `syft`.

---

## Pass 17 (2026-05-17 — F236 FCC caption display-settings tokens)

Pass 17 combined targeted external regulatory verification with local implementation. The regulatory source was refreshed because F236 depends on a current legal compliance date and rule text.

### Pass 17 external searches

| Query / source | Result |
|---|---|
| `FCC caption display settings readily accessible August 17 2026 rule closed captioning user settings font size color opacity edge background 2026` | Found the codified 47 CFR § 79.103(e) text, FCC 24-79 Report and Order, and Federal Register compliance-date notice. |
| 47 CFR § 79.103(e) | Confirmed the four readily-accessible factors: proximity, discoverability, previewability, consistency/persistence. |
| Federal Register FR Doc. 2025-02816 | Confirmed effective date February 21, 2025 and compliance date August 17, 2026 for 47 CFR 79.103(e). |
| FCC 24-79 PDF | Confirmed the display-setting surface includes presentation, color, opacity, size, font, caption background color/opacity, character edge attributes, and caption window color. |

### Pass 17 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 17.1 | Inspected caption burn-in, styled caption, caption route, and caption QC code. | Found existing style presets and burn-in `force_style` support but no canonical user-overridable display token contract. |
| Pass 17.2 | Added `opencut/core/caption_display_settings.py`. | New schema normalizes FCC-style font, size, text color/opacity, background color/opacity, edge, and window tokens; emits CSS preview values and ASS `force_style`. |
| Pass 17.3 | Added routes and burn-in integration. | `/captions/display-settings/tokens` exposes the schema; `/captions/display-settings/preview` returns normalized preview payloads; `/captions/burnin/file` accepts `display_settings`. |
| Pass 17.4 | Added `tests/test_caption_display_settings.py`, wired it into release smoke, and regenerated route manifest. | Route manifest now reports 1,361 routes / 101 blueprints. |
| Pass 17.5 | Updated roadmap and research state files. | F236 marked closed; next Now items are F237/F240/F241/F243/F244/F251/F259, with F205 still blocked by the coverage timeout. |

### Pass 17 validation results

| Check | Result |
|---|---|
| Focused caption-display/route-manifest/release-smoke tests | **PASS** — `21 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Route manifest regeneration | **PASS** — 1,361 routes / 101 blueprints |
| API alias drift check | **PASS** — 15 aliases / 218 canonical `/api` routes |
| Feature-readiness drift check | **PASS** — 58 generated records / 67 route bindings |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `273 passed` |

### Pass 17 saturation note

F236 is complete for the repository-side token contract and burn-in integration. It does not prove end-user UI discoverability in the Premiere panel; that remains downstream UI/persistence polish and should be tested with the eventual UXP/CEP settings surface.

---

## Pass 18 (2026-05-17 — F237 loudness standards registry)

Pass 18 combined targeted standards verification with a local implementation pass. The external lookup was required because the backlog text said to drop a "speculative" BS.1770-5 reference, but official ITU sources now show BS.1770-5 is real and in force.

### Pass 18 external searches

| Query / source | Result |
|---|---|
| `site:itu.int BS.1770-5 Algorithms to measure audio programme loudness and true-peak audio level` | Found the official ITU BS.1770-5 page and table of contents. |
| `site:itu.int R-REC-BS.1770 latest BS.1770-5` | Found the version listing confirming BS.1770-5 is Main/In force and BS.1770-4 is Superseded. |
| `site:tech.ebu.ch R 128 version 5.0 loudness recommendation PDF` | Found EBU R 128 Version 5.0 (November 2023), target -23 LUFS and maximum true peak descriptor. |
| FFmpeg loudnorm docs | Confirmed OpenCut's FFmpeg normalization path can target integrated loudness, LRA, and maximum true peak under EBU R128 normalization. |
| Spotify loudness guidance | Confirmed the source-backed -14 LUFS / -1 dBTP platform profile for Spotify. |

### Pass 18 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 18.1 | Inspected `audio_suite.py`, `audio_analysis.py`, `broadcast_qc.py`, `loudness_match.py`, and `/audio/loudness-presets`. | Found duplicated/scattered targets and one stale generic `broadcast` ambiguity. |
| Pass 18.2 | Added `opencut/core/loudness_standards.py`. | Central registry now records ITU BS.1770-5, EBU R128 v5.0, FFmpeg loudnorm, platform/profile targets, source URLs, and correction notes. |
| Pass 18.3 | Wired audio normalization, audio analysis, broadcast QC, and the loudness preset route to the registry. | Existing `LOUDNESS_PRESETS` import compatibility is preserved; `/audio/loudness-presets` now returns standards metadata. |
| Pass 18.4 | Added `tests/test_loudness_standards.py` and wired it into release smoke. | Tests cover current ITU/EBU facts, preset targets, compatibility export, platform target behavior, and route payload. |
| Pass 18.5 | Updated roadmap and research state files. | F237 marked closed; next Now items are F240/F241/F243/F244/F251/F259, with F205 still blocked by the coverage timeout. |

### Pass 18 validation results

| Check | Result |
|---|---|
| Focused loudness/release-smoke tests | **PASS** — `17 passed` |
| Focused compatibility/route slice | **PASS** — `9 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `278 passed` |

### Pass 18 saturation note

F237 is complete for source-backed standards metadata and preset API correction. It intentionally does not declare generic streaming/podcast targets as universal compliance standards; they remain editable profiles, while EBU R128 and ITU BS.1770 carry the standards metadata.

---

## Pass 19 (2026-05-17 — F240 caption reading-speed profiles)

Pass 19 combined targeted external standards verification with a local caption-QC implementation pass. The source check was necessary because the backlog shorthand bundled supported and unsupported numeric claims.

### Pass 19 external searches

| Query / source | Result |
|---|---|
| `Netflix Timed Text Style Guide English USA reading speed characters per second adult children 17 20` | Found the current Netflix English (USA) guide. It lists adult programs at 20 CPS and children's programs at 17 CPS, correcting the earlier "Netflix 17 cps" shorthand. |
| `BBC subtitle guidelines 160 180 words per minute reading speed subtitles` | Found an archived official BBC Subtitle Guidelines page, version 1.2.4a, recommending 160-180 WPM. |
| `DCMP Captioning Key presentation rate words per minute 160 180 captions` | Found DCMP Captioning Key presentation-rate guidance; upper-level educational media should not exceed 160 WPM. |
| `FCC caption quality best practices reading speed words per minute` + 47 CFR § 79.1 | Found qualitative FCC timing/readability requirements but no hard fixed WPM cap in the official rule text. |
| `YouTube subtitle guidelines reading speed words per minute 220 official captions` | Found YouTube official caption upload/timing help, but no official hard numeric reading-speed cap. |

### Pass 19 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 19.1 | Inspected `caption_compliance.py`, `caption_qc.py`, `/captions/qc`, and existing caption tests. | Confirmed QC already wraps the compliance checker and can accept a narrow speed-profile overlay without a parallel parser. |
| Pass 19.2 | Added `opencut/core/caption_reading_profiles.py`. | New registry records Netflix adult/children, BBC editorial, DCMP upper-level, FCC qualitative, and YouTube advisory profiles with source URLs, confidence, enforcement, and max CPS/WPM data. |
| Pass 19.3 | Wired profile overlays through `caption_compliance` and `caption_qc`. | `check_caption_compliance` now accepts per-call rule overrides; `qc_captions` accepts `reading_profile` and returns profile metadata; advisory mode now downgrades actual violation rule names. |
| Pass 19.4 | Added API route and tests. | `GET /captions/qc/reading-profiles` exposes the registry; `POST /captions/qc` accepts `reading_profile`, `profile`, and `speed_profile`; `tests/test_caption_reading_profiles.py` pins behavior and source corrections. |
| Pass 19.5 | Regenerated route manifest and updated roadmap/research state. | Route manifest now reports 1,362 routes / 101 blueprints; F240 marked closed; next local-verifiable Now queue starts with F241/F243/F244 while F205 remains blocked by long coverage runtime. |

### Pass 19 validation results

| Check | Result |
|---|---|
| Focused caption profile/QC/compliance tests | **PASS** — `31 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Route manifest regeneration/check | **PASS** — 1,362 routes / 101 blueprints |
| API alias drift check | **PASS** — 15 aliases / 218 canonical `/api` routes |
| Feature-readiness drift check | **PASS** — 58 generated records / 67 route bindings |
| Full release smoke | **PASS** — all 13 steps green; pytest-fast `284 passed` |

### Pass 19 saturation note

F240 is complete for repository-side source-backed reading-speed profiles and QC API exposure. It intentionally keeps FCC as qualitative and YouTube's 220 WPM profile as a low-confidence advisory because no official numeric cap was found in the checked sources.

---

## Pass 20 (2026-05-17 — F241 text-shaping gate)

Pass 20 was a local implementation and verification pass. No new external search was needed because the work closes a repository capability gate against local FFmpeg/Pillow/Skia surfaces already identified by the Pass-2 niche standards research.

### Pass 20 local probes

| Probe | Result |
|---|---|
| `opencut/core/caption_burnin.py` | Confirmed subtitle burn-in uses FFmpeg `ass` / `subtitles` filters, so libass is the hard release surface. |
| `opencut/core/styled_captions.py` | Confirmed styled overlays use Pillow by default and optional `skia-python` when installed. |
| bundled `ffmpeg/ffmpeg.exe -hide_banner -version` | Confirmed the bundled build exposes `--enable-libass`, `--enable-libharfbuzz`, and `--enable-libfribidi`. |
| bundled `ffmpeg/ffmpeg.exe -hide_banner -filters` | Confirmed exact `ass` and `subtitles` filters; also confirmed why substring parsing is unsafe (`greyedge assumption`). |
| Pillow feature probe | Local Pillow `12.2.0` lacks RAQM/HarfBuzz/FriBidi but has Freetype, so the default gate records an advisory instead of failing this dev environment. |

### Pass 20 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 20.1 | Added `opencut/tools/text_shaping_gate.py`. | New JSON/human CLI resolves FFmpeg, hard-checks libass/HarfBuzz/FriBidi/filter support, and reports Pillow RAQM plus optional Skia shaping status. |
| Pass 20.2 | Wired release smoke and CI. | `scripts/release_smoke.py` now has a `text-shaping` step; `.github/workflows/build.yml` runs the same gate after standard dependency install. |
| Pass 20.3 | Added `tests/test_text_shaping_gate.py`. | Tests cover exact FFmpeg parsing, missing-HarfBuzz failure, strict Pillow promotion, release-smoke wiring, and workflow wiring. |
| Pass 20.4 | Updated roadmap and research state files. | F241 marked closed; next local-verifiable Now queue is F243/F244 while F205 remains blocked by long coverage runtime. |

### Pass 20 validation results

| Check | Result |
|---|---|
| Text-shaping gate CLI | **PASS** — FFmpeg/libass hard gates OK; Pillow RAQM warning; Skia skipped |
| Focused text-shaping/release-smoke tests | **PASS** — `17 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 14 steps green; pytest-fast `289 passed` |

### Pass 20 saturation note

F241 is complete for the CI/release-gate layer. It does not rewrite the Pillow renderer to shape complex scripts; it exposes that limitation as a machine-readable warning by default and as a strict failure when packaging environments opt into `--require-pillow-raqm`.

---

## Pass 21 (2026-05-17 — F243 UTF-8 no-BOM SRT policy)

Pass 21 was a local implementation and verification pass focused on SRT writer behavior and route/CLI exposure.

### Pass 21 local probes

| Probe | Result |
|---|---|
| `opencut/export/srt.py` | Found the primary Whisper transcript SRT writer; it already wrote UTF-8 without BOM but had no explicit policy, validation, or opt-in legacy toggle. |
| `opencut/routes/captions.py` | Found four SRT export surfaces that should share the same toggle: `/captions`, `/transcript/export`, `/full`, and `/interview-polish`. |
| `opencut/cli.py` | Found two CLI SRT output paths: `captions` and `full`. |
| `opencut/core/subtitle_shot_aware.py` | Found a secondary SRT file writer used by shot-aware subtitle post-processing. |

### Pass 21 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 21.1 | Updated `opencut/export/srt.py`. | Added `srt_text_encoding`, `write_srt_text`, `has_utf8_bom`, and `legacy_windows_bom` support on `export_srt`. |
| Pass 21.2 | Wired routes and CLI. | SRT routes accept `srt_legacy_bom`, `windows_legacy_bom`, or `legacy_bom`; SRT responses report `srt_encoding`; CLI commands expose `--srt-legacy-bom`. |
| Pass 21.3 | Updated shot-aware file export. | `export_to_file(..., fmt="srt")` now uses the shared SRT writer and accepts `legacy_windows_bom=True`. |
| Pass 21.4 | Added `tests/test_srt_encoding.py` and release-smoke wiring. | Tests pin byte-level no-BOM default, opt-in BOM, encoding validation, route aliases, and shot-aware export behavior. |
| Pass 21.5 | Updated roadmap and research state files. | F243 marked closed; next local-verifiable Now queue is F244 while F205 remains blocked by long coverage runtime. |

### Pass 21 validation results

| Check | Result |
|---|---|
| Focused SRT encoding/caption-regression/core/subtitle export tests | **PASS** — `13 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 14 steps green; pytest-fast `294 passed` |

### Pass 21 saturation note

F243 is complete for repository SRT export behavior and user-facing toggles. Existing SRT readers continue accepting BOMmed input through `utf-8-sig`; this pass only controls newly written SRT bytes.

---

## Pass 22 (2026-05-17 — F244 Whisper confidence + human-review flags)

Pass 22 was a local implementation and verification pass. No new external search was needed because the Pass-2 niche standards research already identified Whisper Hindi/Arabic accuracy risk (`R-N17`), and the work was to wire that risk through existing local transcription/export surfaces.

### Pass 22 local probes

| Probe | Result |
|---|---|
| `opencut/core/captions.py` | Found `Word`, `CaptionSegment`, and `TranscriptionResult` lacked segment language/confidence/review metadata even though Whisper backends expose word probabilities and faster-whisper exposes language probability. |
| `opencut/routes/captions.py` | Found `/captions`, `/transcript`, `/transcript/export`, `/full`, `/interview-polish`, transcript cache reuse, chapters, summarize, and repeat-detect build their own segment dict shapes. |
| `opencut/export/srt.py` | Found JSON export was the right durable carrier for per-segment review metadata. |
| `opencut/polish_state.py` | Found interview-polish resume state would drop any new segment metadata unless explicitly persisted/restored. |
| `scripts/release_smoke.py` | Confirmed caption/release-gate test files are curated explicitly, so the new F244 test file needed to be added. |

### Pass 22 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 22.1 | Extended caption dataclasses and backend mappers. | `CaptionSegment` now carries `language`, `language_confidence`, segment `confidence`, `human_review_recommended`, and `review_reasons`; OpenAI Whisper, faster-whisper, and WhisperX populate the fields from word probability, avg-logprob/no-speech, and language probability where available. |
| Pass 22.2 | Added shared segment serialization. | `caption_segment_to_dict` is now used by route dicts, `transcribe_audio`, transcript cache payloads, JSON export, and repeat/chapter/summarize paths. |
| Pass 22.3 | Wired routes, cache/state, and CLI. | Caption/transcript/full/interview-polish responses expose review summaries; edited transcript export preserves metadata; interview-polish state preserves/restores it; `opencut captions` prints review recommendations. |
| Pass 22.4 | Added `tests/test_caption_language_confidence.py` and release-smoke wiring. | Tests cover Hindi/Arabic flags, low ASR/language confidence reasons, JSON export, remap preservation, transcript route payloads, edited transcript export, and release-smoke inclusion. |
| Pass 22.5 | Updated roadmap and research state files. | F244 marked closed; remaining Now queue is F205, F251, and F259, with F205 still blocked by long coverage runtime and F251/F259 needing fresh Adobe/UXP verification. |

### Pass 22 validation results

| Check | Result |
|---|---|
| Focused caption confidence/regression/SRT tests | **PASS** — `12 passed` |
| Ruff on touched Python files | **PASS** |
| Python compile for touched Python files | **PASS** |
| Full release smoke | **PASS** — all 14 steps green; pytest-fast `300 passed` |

### Pass 22 saturation note

F244 is complete for repository-side confidence metadata and human-review surfacing. The thresholds remain pragmatic review heuristics, not a statistical calibration claim; future ASR-evaluation work should tune them against the F176/F178 evaluation corpus once that corpus exists.

---

## Pass 23 (2026-05-17 — F205 interrupted coverage reattempt wrap-up)

Pass 23 was a wrap-up pass after the autonomous development loop started a second F205 coverage measurement and the session was interrupted before pytest completed. No external research was needed.

### Pass 23 local probes

| Probe | Result |
|---|---|
| Running Python/pytest processes | One leftover process remained: `python.exe -m pytest tests sidecar/tests -q`; it was stopped and a follow-up process list was empty. |
| `dist\coverage-f205.json` | Parsed as valid coverage.py 7.14.0 JSON, but pytest did not complete, so the data is partial. |
| Ignored artifact status | `.coverage` and `dist/` are ignored, so the partial coverage report was not committed; stale `.coverage` and `dist\coverage-f205.json` were removed after evidence capture. |

### Pass 23 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 23.1 | Checked for leftover coverage/test processes. | Stopped one leftover pytest process. |
| Pass 23.2 | Parsed the partial coverage JSON. | 126,421 statements, 65,890 covered, 60,531 missing, 52.1195% coverage across 670 files; SHA256 captured in the source register. |
| Pass 23.3 | Updated roadmap and memory/state files. | F205 remains open; F251/F259 remain the other Now items. |
| Pass 23.4 | Added a Codex memory checkpoint. | Wrote `C:\Users\Xray\.codex\memories\extensions\ad_hoc\notes\2026-05-17T19-45-34-opencut-f205-wrapup.md`. |

### Pass 23 saturation note

The partial 52.12% number is intentionally treated as unusable for policy. A complete F205 run still needs a runner where the CI-style pytest+coverage command exits cleanly; only then should `.github/workflows/build.yml` move above `--cov-fail-under=50`.

---

## Pass 82 (2026-05-19 — F205 completed coverage measurement)

Pass 82 resumed F205 now that the full pytest+coverage command could finish locally.

| Step | Action | Result |
|---|---|---|
| Pass 82.1 | Installed missing local coverage/xdist tooling. | `pytest-cov` 5.0.0, `pytest-xdist` 3.8.0, `coverage` 7.14.0, and `execnet` 2.1.2 were available for the run. |
| Pass 82.2 | Ran the CI-style coverage command with `--cov-fail-under=0`. | PASS — `8,540 passed`, `16 skipped`, 7 warnings in 132.73 seconds. |
| Pass 82.3 | Parsed `dist\coverage-f205.json`. | 131,130 statements, 70,935 covered lines, 60,195 missing lines, 54.09517272935255% coverage; SHA256 `C3044F261073964E868FED338B7B09114F0115DA16F6EAF0C34005146576F318`. |
| Pass 82.4 | Raised Release Full CI coverage policy. | `.github/workflows/build.yml` now uses `--cov-fail-under=54`. |
| Pass 82.5 | Updated roadmap and state files. | F205 marked closed; Pass 12 and Pass 23 attempts remain historical invalid/partial evidence. |

The completed measurement invalidates the older 75-80% estimate in the planning notes. The new floor uses the integer-safe measured baseline, not a speculative target.

---

## Pass 83 (2026-05-19 — F252.3 UDT result-capture validator)

Pass 83 advanced F252 after checking that the existing F267 UDT harness had no repository-side contract for validating pasted UDT results before a WebView manifest switch.

| Step | Action | Result |
|---|---|---|
| Pass 83.1 | Read the F252/F267 scaffold, harness, docs, and tests. | Confirmed the harness can run in UDT but no saved-result validator existed. |
| Pass 83.2 | Added result-template and validation primitives. | `opencut.core.uxp_udt_results` now validates raw or wrapped UDT harness JSON. |
| Pass 83.3 | Added CLI and tests. | `python -m opencut.tools.validate_uxp_udt_results` emits templates and strict JSON reports; `tests/test_uxp_udt_results.py` covers readiness and diagnostic cases. |
| Pass 83.4 | Updated roadmap and state files. | F252 now records F252.1/F252.2/F252.3 repository-side slices, while the real Premiere UDT capture and WebView cutover remain open. |

F252 remains external/tooling-locked until a disposable Premiere project can run `window.OpenCutUXPUdtHarness.run({ includeMutating: true })`, save the returned JSON, and pass `python -m opencut.tools.validate_uxp_udt_results <capture.json> --json` with `ready_for_webview_cutover=true`.

---

## Pass 57 (2026-05-18 — F249 Linux distribution packaging)

Pass 57 closed the Next-tier Linux packaging item after fresh official-source
verification against Flatpak, Flathub, and AppImage documentation.

### Pass 57 external checks

| Probe | Result |
|---|---|
| Flatpak current runtime tutorial | Freedesktop Platform/Sdk 25.08 is the current tutorial baseline. |
| Flatpak conventions | App ID, desktop file, MetaInfo, and icon names must align; GitHub-hosted IDs use `io.github`. |
| Flathub requirements | Hosted submissions need current hosted runtimes, no network during build, source/dependency manifests, and top-level manifest/flathub.json in the Flathub repo. |
| AppImage AppDir spec | AppDir requires root `AppRun`, desktop file, icon, and normally uses a `usr/` payload layout. |

### Pass 57 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 57.1 | Audited existing Linux release path. | Found PyInstaller Linux tarball release only; no Flatpak/AppImage metadata or workflow artifact. |
| Pass 57.2 | Added Linux desktop metadata and package builder. | Added `io.github.sysadmindoc.opencut` manifest, desktop/MetaInfo files, `flathub.json`, Flatpak/AppImage launchers, and `scripts/build_linux_packages.sh`. |
| Pass 57.3 | Wired CI/release and tests. | Linux tag/manual jobs build/archive/upload `.flatpak` and `.AppImage`; release smoke now includes `tests/test_linux_distribution_packaging.py`. |
| Pass 57.4 | Synced roadmap/state docs. | Marked F249 closed in ROADMAP/addendum/matrix/context/continue docs and added `docs/LINUX_DISTRIBUTION.md`. |

### Pass 57 saturation note

The repo now produces Flatpak release bundles from the PyInstaller output.
The Flathub hosted submission still needs a source-build manifest and generated
Python dependency manifests before it should be submitted to Flathub; this is
documented as a boundary, not silently claimed as complete hosted publication.

---

## Pass 58 (2026-05-18 — F250 Aptabase opt-in telemetry)

Pass 58 closed the Next-tier telemetry item after fresh verification against
Aptabase's SDK contract, Python client source, and privacy/product materials.

### Pass 58 external checks

| Probe | Result |
|---|---|
| Aptabase SDK contract | Events are POSTed as JSON batches to `/api/v0/events` with an `App-Key` header; batches are capped at 25 events and include timestamp, session ID, event name, system props, and props. |
| App-key host routing | `A-EU-*` maps to `https://eu.aptabase.com`, `A-US-*` maps to `https://us.aptabase.com`, and self-hosted `A-SH-*` requires a custom host. |
| Failure semantics | SDK guidance says clients should queue while offline and not crash an app because telemetry delivery failed. |
| Privacy positioning | Aptabase positions itself as privacy-first app analytics with no cookies and no long-term user-identifying IDs. |

### Pass 58 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 58.1 | Verified the telemetry contract and current provider assumptions. | Used Aptabase primary sources instead of extending the older Plausible default assumption. |
| Pass 58.2 | Added the local Aptabase client and settings storage. | Implemented stdlib-only background batching, opt-in settings, env overrides, region/self-host validation, masking, and sensitive-prop scrubbing. |
| Pass 58.3 | Wired routes, checks, docs, and release-smoke coverage. | Added `/telemetry/aptabase/*`, `check_aptabase_configured`, `docs/TELEMETRY.md`, README/CHANGELOG/ROADMAP updates, and `tests/test_telemetry_aptabase.py`. |
| Pass 58.4 | Regenerated generated artifacts and synced state files. | Route manifest now has 1,374 routes; opt-in extended MCP catalogue now has 1,317 route tools. |

### Pass 58 saturation note

F250 is complete for repository-side, disabled-by-default Aptabase telemetry.
It intentionally does not add UI controls yet; the route/settings contract and
docs give the UXP/CEP settings surfaces a stable integration target.

---

## Pass 59 (2026-05-18 — F252.1 Bolt UXP WebView scaffold)

Pass 59 advanced the F252 UXP migration by implementing the first bounded,
locally-verifiable sub-phase: a dormant Bolt-shaped WebView scaffold beside the
current shipped UXP panel.

### Pass 59 external checks

| Probe | Result |
|---|---|
| Bolt UXP repository | Confirms the project shape and UXP config pattern used for the scaffold. |
| Bolt UXP WebView UI announcement | Confirms WebView UI is the intended Bolt UXP path for richer UI surfaces without immediately rewriting the whole panel into native widgets. |
| Existing OpenCut parity matrix | Confirms the low-risk UXP/Premiere wrappers are a safe first scaffold layer while the CEP-only functions remain out of this slice. |

### Pass 59 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 59.1 | Verified the F252 sub-phase boundary. | Chose F252.1 scaffold work instead of claiming full F252 migration closure. |
| Pass 59.2 | Added the dormant Bolt/WebView tree. | Created `extension/com.opencut.uxp/bolt-webview/` with a least-privilege `uxp.config.ts`, host API wrappers, WebView shell, and message bridge helpers. |
| Pass 59.3 | Wired docs and release guardrails. | Updated `docs/UXP_MIGRATION.md`, ROADMAP/state files, added `tests/test_uxp_webview_scaffold.py` to release smoke, and verified the curated `pytest-fast` release gate (`623 passed`). |

### Pass 59 saturation note

F252.1 is complete as a repository-side scaffold and is covered by the local
release-smoke `pytest-fast` gate. F252 remains open until the live manifest is
switched after an in-Premiere UDT smoke pass and the UI/API migration is
validated against the real host.

---

## Pass 60 (2026-05-18 — F252.2 UXP host-action dispatcher)

Pass 60 advanced the next F252 slice by turning the parity catalogue into a
live UXP dispatcher surface that the current panel and the future WebView host
bridge can share.

### Pass 60 local checks

| Probe | Result |
|---|---|
| Parity catalogue | The dispatcher direct-action map matches the 14 `direct_uxp` entries in `opencut/_generated/cep_uxp_parity.json`. |
| CEP fallback boundary | `ocAddNativeCaptionTrack` and `ocQeReflect` remain explicit fallback responses instead of being silently marked as native UXP operations. |
| WebView bridge readiness | `window.OpenCutUXPHost` exposes `executeHostAction` and `getHostActionStatus` for the dormant WebView bridge. |

### Pass 60 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 60.1 | Converted the parity manifest into a UXP dispatcher contract. | Added `UXP_DIRECT_HOST_ACTIONS`, `CEP_FALLBACK_HOST_ACTIONS`, and `PProBridge.executeHostAction()`. |
| Pass 60.2 | Added live bridge helpers. | Added marker read/remove, project-bin enumeration, rename/bin/delete helpers, playhead dispatch, and guarded pending responses for UDT/F255-gated actions. |
| Pass 60.3 | Wired docs and release guardrails. | Added `tests/test_uxp_host_action_dispatch.py`, release-smoke registration, roadmap/state updates, focused test/Ruff/compile checks, `node --check`, and release-smoke `pytest-fast` validation (`627 passed`). |

### Pass 60 saturation note

F252.2 now has a stable dispatcher surface covered by the local release-smoke
`pytest-fast` gate, but F252 remains open. The next UXP work should either add
UDT coverage for these host actions or move into the explicit F254-F258 API
migrations.

---

## Pass 61 (2026-05-18 — F254 UXP createSubsequence integration)

Pass 61 closed F254 by wiring the verified beta `Sequence.createSubsequence`
API behind the export-range dispatcher.

### Pass 61 package checks

| Probe | Result |
|---|---|
| npm metadata | `@adobe/premierepro` still reports `latest=26.2.0` and `beta=26.3.0-beta.67`. |
| Beta typings | `premierepro.d.ts` exposes `Sequence.createSubsequence(ignoreTrackTargeting?)`, `TickTime.createWithSeconds`, `Sequence.createSetInPointAction`, `Sequence.createSetOutPointAction`, and `Project.executeTransaction`. |
| Boundary | The subsequence is now created in UXP; AME/export execution remains sequenced as F255. |

### Pass 61 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 61.1 | Refreshed the beta API assumptions. | Used npm metadata and unpacked typings to confirm F254 inputs before coding. |
| Pass 61.2 | Implemented the subsequence range handoff. | Added `createSubsequenceFromRange()`, TickTime conversion, project transaction helpers, range restoration, and F255 handoff metadata. |
| Pass 61.3 | Wired docs and release guardrails. | Added `tests/test_uxp_create_subsequence_integration.py`, release-smoke registration, roadmap/state updates, focused test/Ruff/compile checks, `node --check`, and release-smoke `pytest-fast` validation (`632 passed`). |

### Pass 61 saturation note

F254 is complete for repository-side UXP range-subsequence creation and is
covered by the local release-smoke `pytest-fast` gate. F255 is the immediate
follow-up for `EncoderManager.launchEncoder` / `startBatchEncode` and export
execution.

---

## Pass 62 (2026-05-18 — F255 UXP EncoderManager handoff)

Pass 62 closed F255 by routing F254 range subsequences through Premiere's UXP
EncoderManager APIs.

### Pass 62 package checks

| Probe | Result |
|---|---|
| Beta typings | `premierepro.d.ts` exposes `EncoderManager.getManager()`, `exportSequence(...)`, `launchEncoder()`, `startBatchEncode()`, `isAMEInstalled`, and `Constants.ExportType`. |
| Export boundary | `ocExportSequenceRange` now reaches encoder queue/immediate export wiring instead of stopping at an F255 pending response. |
| AME posture | Queued exports require AME when `isAMEInstalled` is false; immediate exports remain available through `Constants.ExportType.IMMEDIATELY`. |

### Pass 62 phases executed

| Phase | What | Output |
|---|---|---|
| Pass 62.1 | Confirmed EncoderManager API names from beta typings. | Verified the `launchEncoder`, `exportSequence`, and `startBatchEncode` signatures before coding. |
| Pass 62.2 | Implemented the encoder handoff. | Added manager lookup, export type selection, AME availability guard, AME launch, sequence export, and optional batch start. |
| Pass 62.3 | Wired docs and release guardrails. | Added `tests/test_uxp_encoder_manager_integration.py`, updated the F254 test, registered release smoke, and ran focused test/Ruff/compile/Node plus release-smoke `pytest-fast` validation (`637 passed`). |

### Pass 62 saturation note

F255 is complete for repository-side UXP EncoderManager dispatch and covered by
the local release-smoke `pytest-fast` gate. Pass 63 later closed F256
Transcript APIs, Pass 64 closed F257 Object Mask detection, and Pass 65 closed
F258 AAF export. Pass 66 closed the F260 migration risk dashboard that tracks
the remaining CEP fallback and UDT/hybrid work.

---

## Pass 72 implementation note (2026-05-18)

F196 was closed from local registry/model-card/check evidence. The route scanner
already covers direct route/check bindings, but 16 model-card surfaces were still
only represented in `model_cards.py` because their availability gates sit behind
core helpers or stubs. The pass added curated `FeatureRecord` rows for those
surfaces, backfilled older manual hardware metadata, and raised
`/system/feature-state` to 100 records.

The new `opencut.catalog_contract` validator makes the contract explicit:
every public `check_*_available` must be backed by a model card or the
registry-owned `NON_AI_CHECKS` allowlist, every model card must map to a
registry row, and the registry must expose matching hardware/GPU/VRAM metadata.
`tests/test_catalog_contract.py` is now in `pytest-fast`, so future drift fails
inside release smoke.

Validation evidence for the pass: focused registry/model-card/catalog tests
(`32 passed`), touched Python compile, focused Ruff, model-card/readiness sync
checks, roadmap lint, and release-smoke `pytest-fast` (`698 passed`) passed
locally.

---

## Pass 73 implementation note (2026-05-18)

F206 was closed by splitting the GitHub Actions paths. Pull requests now run a
new `.github/workflows/pr-fast.yml` workflow on Linux only, installing the
standard Python surface and running the fast release-smoke subset while skipping
release-only audit, panel, and upstream-version drift checks.

The previous `.github/workflows/build.yml` matrix is renamed Release Full and no
longer triggers on pull requests. It still runs the Windows, Linux, and macOS
matrix on pushes to `main`, version tags, and manual dispatch, preserving the
packaging, signing, SBOM, notarization, and release-upload path.

Validation evidence for the pass: focused F206 workflow tests (`4 passed`),
touched Python compile, and focused Ruff passed locally.

---

## Pass 74 implementation note (2026-05-18)

F210 was closed by extracting the panel-side utility seams into small
production-loaded modules that Vitest can import directly. The CEP panel now
loads `client/panel-utils.js` before `main.js`; `main.js` routes HTML escaping,
ExtendScript double-quoted string escaping, lazy DOM proxying, and command
palette section building through that helper. The UXP panel imports
`uxp-utils.js` for HTML escaping and safe DOM-id normalization.

The pass added `npm test` with Vitest 4.x, 8 unit tests across CEP and UXP
utilities, release-smoke `panel-unit`, and workflow wiring so PR Fast installs
panel dependencies and Release Full runs the panel unit suite.

Validation evidence for the pass: `npm test` (`8 passed`), npm advisory
allow-list check, focused F210 Python guard tests (`3 passed`), touched Python
compile, focused Ruff, release-smoke `panel-unit`, `npm run build:verify`,
`npm run build`, and the PR-fast release-smoke command (`pytest-fast` 705
passed plus `panel-unit`) passed locally.

---

## Pass 81 implementation note (2026-05-19)

F245-F248 were closed as local-first delivery-standard planning surfaces. Fresh
source checks confirmed the right boundary for this repo: OpenCut can assemble
operator command arrays and metadata/checklist plans for Netflix IMF/Dolby
Vision, DPP/broadcaster IMF, Dolby Vision Profile 5/8.1 review packages, and ADM
BW64 Atmos-master preparation, but it should not claim platform certification,
broadcaster acceptance, or licensed Dolby `.ec3`/DD+JOC encode output.

The pass added `opencut/core/delivery_standards.py`, two delivery-master routes,
`docs/DELIVERY_STANDARDS.md`, generated route/MCP artifacts, and
`tests/test_delivery_standards.py` in release smoke. The route manifest now
reports 1,381 routes and the opt-in extended MCP catalogue reports 1,325 route
tools.

Validation evidence for the pass: touched Python compile, focused
delivery-standard tests (`6 passed`), focused generated-surface tests (`20
passed`), and focused Ruff passed locally.
