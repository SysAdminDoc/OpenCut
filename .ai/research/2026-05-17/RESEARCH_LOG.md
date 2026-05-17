# OpenCut — Research Log

**Run date:** 2026-05-17
**Total elapsed:** ~25 minutes wall-clock from session start to last artefact written.
**Tools used:** Bash (`git log`, `git status`, `git diff`, `ls`, `wc -l`, `python -c`), Read, Glob, Grep, Write, Edit, Agent (general-purpose), TodoWrite.
**Subagents launched:** 3 in parallel.

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

F204 is complete as release plumbing. F219 remains the right place to enforce SBOM completeness against all declared dependencies and model cards; do not treat the F204 generator smoke test as that deeper coverage.

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
