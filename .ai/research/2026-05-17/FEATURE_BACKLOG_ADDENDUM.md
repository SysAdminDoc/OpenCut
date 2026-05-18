# OpenCut — Feature Backlog Addendum (Pass 2)

**Audit date:** 2026-05-17 (Pass 2)
**Continues:** `FEATURE_BACKLOG.md` (Pass 1) which proposed F121-F190.

This file adds F191-F260 from the four Pass-2 sources:
- Route readiness audit (`ROUTE_READINESS_AUDIT.md`) → F191-F199
- Installer audit (`INSTALLER_AUDIT.md`) → F200-F207
- Test coverage gaps (`TEST_COVERAGE_GAPS.md`) → F208-F219
- features.md sample reconciliation (`FEATURES_RECONCILIATION.md`) → F220-F224
- Frame.io / collab review subagent → F225-F234
- Niche AI / accessibility / standards subagent → F235-F250
- UXP Hybrid Plugins / migration subagent → F251-F260

Format: ID — title — what — source(s) — effort (S/M/L/XL) — fit (yes/conditional/no) — tier.

---

## A. Route / OpenAPI / MCP / registry (Pass-2 route audit)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F191 | Auto-derive `FeatureRecord` from check functions + route manifest | DONE in Pass 8 — generated catalogue covers 58 direct route/check records and 67 route bindings | ROUTE_READINESS_AUDIT §3 | M | Done |
| [x] F192 | Bulk OpenAPI response schemas for top 50 routes | Closed in Pass 41 — legacy `/openapi.json` typed schema map now covers 100 routes with reusable response envelopes and contract tests | ROUTE_READINESS_AUDIT §4 | M | Next |
| F193 | Introspection-based OpenAPI schema (replace hand-table) | Walk `core/*Result` dataclasses to generate schemas | ROUTE_READINESS_AUDIT §4 | M | Later |
| [x] F194 | Auto-generate extended MCP tools from manifest | Closed in Pass 43 — 1,307 opt-in `opencut_route_*` tools generated from route manifest + OpenAPI schemas while 39 curated tools remain default | ROUTE_READINESS_AUDIT §5 | L | Next |
| F195 | Add 12 missing MCP tools for shipped post-Wave-M routes | DONE in Pass 9 — curated MCP surface is now 39 tools, including face_reshape, skin_retouch, smart_upscale, elevenlabs_tts, caption_qc, review_bundle, c2pa_provenance, marker_import, capability_probe, brand_kit, semantic_search, and spectral_match | ROUTE_READINESS_AUDIT §5 | S | Done |
| F196 | Make `registry.py` primary; derive `model_cards.py` / `checks.py` from it | Single source of truth | ROUTE_READINESS_AUDIT §6 | L | Later |
| F197 | `NON_AI_CHECKS` allowlist in `registry.py` | DONE in Pass 8 — registry owns the allowlist; model_cards imports it | ROUTE_READINESS_AUDIT §6 | S | Done |
| [x] F198 | CEP-only route catalogue + UXP replacement plan | Closed in Pass 42 — code-owned 18-function CEP↔UXP catalogue + generated JSON pins the two true CEP-only calls and replacement plans | ROUTE_READINESS_AUDIT §7 + UXP subagent §10 | M | Next |
| F199 | Document `/api/*` alias policy + generate alias map | DONE in Pass 7 — live result is 15 aliases and 218 canonical `/api` routes | ROUTE_READINESS_AUDIT §8 | S | Done |

## B. Installer / packaging (Pass-2 installer audit)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| [x] F200 | Document WPF-vs-Inno installer policy (retire one or formalise both) | Closed in Pass 33 — WPF recommended, Inno deprecated-but-supported, lockstep tests added | INSTALLER_AUDIT §3 | S | Next |
| [x] F201 | Automate WPF installer build in CI | Closed in Pass 44 — Windows tag/manual CI builds and archives the recommended WPF installer before the Inno fallback | INSTALLER_AUDIT §7 | M | Next |
| F202 | Apple notarisation for macOS PyInstaller bundle | DONE in Pass 10 — macOS release workflow signs with Developer ID, submits `OpenCut-Server-macOS.zip` through `xcrun notarytool`, and documents required secrets; first live acceptance still needs configured GitHub secrets | INSTALLER_AUDIT §7 + niche AI §5 | M | Done |
| F203 | Authenticode code-signing for Windows installer + signing-cert renewal policy | Cert validity drops to 458 days March 2026; EV bypass for SmartScreen no longer auto-trusted | INSTALLER_AUDIT §7 + niche AI §5 | M | Next |
| F204 | Auto-attach SBOM (CycloneDX 1.5) to GitHub release | DONE in Pass 11 — Linux release job generates, archives, and uploads `dist/opencut-sbom.cyclonedx.json` | INSTALLER_AUDIT §7 | S | Done |
| F205 | Raise CI coverage floor from 50% to current actual | Still open — Pass 12 timed out after 20 minutes without output; Pass 23 reattempt was interrupted after 36m46s and produced only partial ignored coverage JSON (52.12%), which is invalid for setting a floor. Do not raise the floor until a complete measurement exists | INSTALLER_AUDIT §7 + TEST_COVERAGE §4 + F205_INTERRUPTED_COVERAGE_NOTE | S | Now |
| F206 | Split CI into PR-fast and release-full workflows | Every PR currently runs full 3-OS matrix | INSTALLER_AUDIT §7 | M | Later |
| F207 | Embed bundled FFmpeg version in `AppConstants.cs` + installer manifest | DONE in Pass 12 — current bundled build `8.0.1-essentials_build-www.gyan.dev` is in `AppConstants.cs` and WPF/Inno `~/.opencut/installer.json` manifests | INSTALLER_AUDIT §9 | S | Done |

## C. Testing (Pass-2 test gaps)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F208 | OpenAPI spec validity test (per-endpoint 200/400/403) | DONE in Pass 13 — `/openapi.json` now emits OpenAPI path params, unique operation IDs, mutating-method 400/403 responses, and a release-gate contract test | TEST_COVERAGE §3.2 | S | Done |
| F209 | MCP tool ↔ route consistency test | DONE in Pass 14 — fixed `opencut_chat_edit` to shipped `POST /chat` and added a live Flask route-consistency gate for all MCP default and special action routes | TEST_COVERAGE §3.3 | S | Done |
| F210 | Vitest unit tests for CEP/UXP utility functions | `esc()`, `escPath()`, command-palette indexer, lazy DOM proxy | TEST_COVERAGE §3.4 | M | Later |
| [x] F211 | Cross-platform launcher script smoke tests in CI | Closed in Pass 33 — 5 launcher entry points covered and wired into release smoke | TEST_COVERAGE §3.5 | S | Next |
| F212 | WPF installer xUnit test suite + headless install in Windows runner | Largest single coverage gap | TEST_COVERAGE §3.6 | XL | Later |
| [x] F213 | Inno Setup install/uninstall smoke in CI | Closed in Pass 37 with CI-only install/uninstall smoke script + workflow wiring + static guard tests | TEST_COVERAGE §3.6 | M | Next |
| [x] F214 | Extend F128 with ML + TTS perf benchmarks | Closed in Pass 40 — performance benchmark registry pins ASR/upscaler/compose/TTS throughput specs, backend matrix, opt-in execution gate, and wall-clock normalization primitive | TEST_COVERAGE §3.7 | M | Next |
| [x] F215 | Extend fuzz harness with 8 targets | Closed in Pass 38 — 13 total fuzz targets now cover path validation, OTIO parse, FCPXML parse, marker import, C2PA sidecars, plugin manifests, webhook HMAC signatures, and `safe_pip_install` package validation | TEST_COVERAGE §3.8 | M | Next |
| [x] F216 | Concurrent job-cancellation race test | Closed in Pass 39 — `_cancel_job()` terminates registered child processes and `tests/test_job_cancellation_race.py` covers the FFmpeg progress cancellation race | TEST_COVERAGE §3.9 | M | Next |
| [x] F217 | UXP backend-client contract test | Closed in Pass 34 — JS-side BackendClient static contract plus Flask health/status/CSRF runtime gates | TEST_COVERAGE §3.10 | S | Next |
| F218 | Import-order stability test for blueprint registration | DONE in Pass 15 — pins the 99-blueprint core order and final `motion_design_api` alias registration in release smoke | TEST_COVERAGE §3.11 | S | Done |
| F219 | SBOM completeness test | DONE in Pass 16 — CycloneDX now covers declared Python deps, 47 model cards, and a non-empty dependency graph in release smoke | TEST_COVERAGE §3.12 | S | Done |

## D. features.md UNCLEAR → F-number graduations

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F220 | AI Voice Conversion / RVC backend | features.md #1.5 / #78.5; not in F-ledger; RVC popularity explosion | FEATURES_RECONCILIATION §3 | M | Later |
| F221 | AI Auto-Color Grading (LLM-driven mood→LUT) | features.md #1.6; extends `color_match.py` + `lut_library.py` | FEATURES_RECONCILIATION §3 | M | Later |
| F222 | AI Pacing & Rhythm Analysis (genre templates) | features.md #1.12; cuts-per-minute, shot-duration distribution | FEATURES_RECONCILIATION §3 | M | Later |
| F223 | RTL / CJK / Bidi caption rendering validation suite | features.md #20.6; cf. niche AI §3 — HarfBuzz must be linked; libass issue #295 / #694 | FEATURES_RECONCILIATION §3 + niche AI §3 | M | Next |
| F224 | Deepfake / fake-video detector | features.md #27.3; adjacent to J2.6 SafeVision | FEATURES_RECONCILIATION §3 | L | Later |

## E. Collaboration / review-bundle extensions (Frame.io subagent)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F225 | Anchor F105 review bundles on OpenTimelineIO Marker schema | Apache-2 interchange; Premiere/Resolve/Final Cut round-trip free | Frame.io subagent §3 | M | Next |
| F226 | Add SVG drawing annotations to F105 bundles | Frame-accurate overlay per frame | Frame.io subagent §2 | M | Next |
| F227 | Add threaded comments + completion status to F105 bundles | Underlord-style review trust pattern | Frame.io subagent §2 | M | Next |
| F228 | Add voice-note attachments to F105 bundles | Match Clapshot / Vimeo Review | Frame.io subagent §2 | S | Later |
| F229 | EDL/OTIO export of review comments → Premiere markers | Round-trip; OpenVidReview's pattern | Frame.io subagent §2 | M | Next |
| F230 | HLS rendition in F105 bundles for browser scrubbing | No source download needed | Frame.io subagent §2 | M | Later |
| F231 | Local-LAN review portal (embedded Caddy + mDNS + HMAC URL) | No account server; share-link bearer | Frame.io subagent §5 | M | Next |
| F232 | Optional Headscale path for cross-site LAN review | Self-hosted Tailscale control plane | Frame.io subagent §5 | M | Later |
| F233 | Outbound HMAC-signed webhook + per-project Atom feed | Notification surface | Frame.io subagent §6 | M | Next |
| F234 | croc + rclone bundle in delivery menu | One-shot P2P + cloud-bucket transfer | Frame.io subagent §7 | S | Next |

## F. Standards / accessibility / packaging (niche-AI / accessibility / standards subagent)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F235 | WCAG 3.0-draft compatibility hooks (descriptive transcript, extended AD timing) | Future-proof caption renderer | Niche AI §2 | M | Later |
| F236 | FCC user-overridable caption style tokens | DONE in Pass 17 — canonical font/size/color/opacity/background/edge/window token schema, preview route, and burn-in integration | Niche AI §2 + R-P17-E01/R-P17-E02 | S | Done |
| F237 | EBU R128 v5.0 + BS.1770-5 correction | DONE in Pass 18 — source-backed loudness registry; ITU BS.1770-5 is in force, BS.1770-4 is superseded, EBU R128 v5.0 broadcast target is -23 LUFS / -1 dBTP | Niche AI §2 + R-P18-E01/R-P18-E04 | S | Done |
| F238 | ITU-R BT.1702 PSE checker with 360ms/334ms gap rule + Japan red-flash threshold | Hitachi Flicker Check reference implementation; extends J1.4 | Niche AI §2 | M | Next |
| F239 | Microsoft `ai-audio-descriptions` integration | Per-scene description + dialogue transcript; pair with IndexTTS2 (F168) | Niche AI §2 | M | Next |
| F240 | Per-target reading-speed profile | DONE in Pass 19 — source-backed QC profiles for Netflix adult 20 cps, Netflix children 17 cps, BBC 160-180 wpm, DCMP upper-level 160 wpm, FCC qualitative timing, and YouTube advisory 220 wpm with unsupported official numeric caps called out | Niche AI §2 + R-P19-E01/R-P19-E05 | S | Done |
| F241 | HarfBuzz-mandatory CI gate (libass + Pillow + Skia) | DONE in Pass 20 — `opencut.tools.text_shaping_gate` hard-fails missing FFmpeg/libass HarfBuzz/FriBidi/ASS/subtitles support, reports Pillow RAQM and optional Skia shaping status, and is wired into release smoke plus CI | Niche AI §3 + F223 + R-P20-L01/R-P20-L10 | S | Done |
| F242 | ICU4X-based CJK line breaking | Replace naive whitespace splitting | Niche AI §3 | M | Next |
| F243 | UTF-8 (no BOM) SRT writer + opt-in Windows-legacy BOM toggle | DONE in Pass 21 — primary SRT export defaults to UTF-8 without BOM, `legacy_windows_bom=True` / `--srt-legacy-bom` opts into `utf-8-sig`, and route aliases expose/report the encoding policy | Niche AI §3 + R-P21-L01/R-P21-L08 | S | Done |
| F244 | Language confidence per Whisper segment + Hindi/Arabic human-review flag | DONE in Pass 22 — Whisper segment metadata now carries ASR confidence, language confidence, Hindi/Arabic review flags, low-confidence review reasons, and survives transcript cache/state, JSON export, edited transcript export, and CLI output | Niche AI §3 + R-N17 + R-P22-L01/R-P22-L08 | S | Done |
| F245 | Netflix IMF builder macro (Bento4 + dovi_tool) | Apache-2 + open OSS chain | Niche AI §4 | L | Later |
| F246 | DPP IMF (BBC / ARD / EBU) preset | Public-broadcaster delivery | Niche AI §4 | L | Later |
| F247 | Dolby Vision Profile 5/8.1 OSS pipeline (dovi_tool + Shaka) | Profile 7 still painful; document constraints | Niche AI §4 | L | Later |
| F248 | ADM BWF (Audio Definition Model in BW64) Atmos master | EBU TR 045 — fully OSS up to final encode; .ec3 needs DEE (commercial) | Niche AI §4 | L | Later |
| F249 | Flatpak (Flathub) primary Linux distribution + AppImage fallback | Linux 2026 packaging winner | Niche AI §5 | M | Next |
| F250 | Aptabase as default opt-in telemetry (privacy-first desktop) | Replaces speculative Plausible default (F067 was Under Consideration) | Niche AI §6 | M | Next |

## G. UXP / Hybrid Plugins / Premiere 26.3+ (UXP subagent)

| F# | Title | What | Source | Effort | Tier |
|---|---|---|---|---|---|
| F251 | Track `@adobe/premierepro@beta` per-week diff via CI | Catch new APIs that close UXP gaps (e.g. `startDrag`, `createCaptionTrack`) the moment they ship | UXP subagent §3 | S | Now |
| F252 | UXP migration sequence: Bolt UXP scaffold + WebView UI for 3,210-line HTML | Plan Pass 1 spike F160a; this is the actionable implementation | UXP subagent §6 + §10 | XL | Next |
| F253 | UXP Hybrid Plugin (.uxpaddon) for file drag-out + QE-equivalent ops | C++ DLL/dylib bundled with mac x64/arm64 + win x64/arm64 (Bolt 1.3.0 has the template) | UXP subagent §4 + §10 | XL | Later |
| F254 | UXP `createSubsequence` integration (already exposed — correct Pass 1 claim) | Replace any CEP path that used in/out subsequence creation | UXP subagent §2 + §3 | S | Next |
| F255 | UXP `EncoderManager.launchEncoder` / `startBatchEncode` integration | New beta APIs; useful for AME-bound exports | UXP subagent §3 | M | Next |
| F256 | UXP `Transcript.hasTranscript` / `querySupportedLanguages` integration | New beta APIs; useful for caption-QC path | UXP subagent §3 | S | Next |
| F257 | UXP `ObjectMaskUtils.hasObjectMask` integration | Premiere 26 AI Object Mask detection | UXP subagent §3 | S | Next |
| F258 | UXP `ProjectConverter.exportAAF` migration (replaces CEP+ExtendScript path) | Maps to F104 (already shipped via FCP XML) but for AAF | UXP subagent §3 | M | Next |
| F259 | UXP HTTP-on-macOS workaround documentation + auto-HTTPS sidecar | Known 25.6.3 bug; affects MCP sidecar (F146) on Mac | UXP subagent §5 | S | Now |
| F260 | UXP migration risk dashboard: per-route CEP-vs-UXP-vs-Hybrid status | Generates from F198 catalogue; surfaces in panel Settings | UXP subagent §10 | M | Next |

---

## Sequencing summary

| Tier | Pass-2 F-numbers added | Total (cumulative) |
|---|---|---|
| **Now** | F205, F251, F259 | 3 |
| **Done locally after Pass 23 wrap-up** | F191, F195, F197, F199, F202, F204, F207, F208, F209, F218, F219, F236, F237, F240, F241, F243, F244 | 17 |
| **Next** | [x] F192, [x] F194, [x] F198, [x] F200, [x] F201, F203, [x] F211, [x] F213, [x] F214, [x] F215, [x] F216, [x] F217, F223, F225, F226, F227, F229, F231, F233, F234, F238, F239, F242, F249, F250, F252, F254, F255, F256, F257, F258, F260 | 32 |
| **Later** | F193, F196, F206, F210, F212, F220, F221, F222, F224, F228, F230, F232, F235, F245, F246, F247, F248, F253 | 18 |

**Total Pass-2 F-numbers: 70** (F191-F260).
**Total project F-numbers: 140** (F121-F260 from Pass 1 + Pass 2; plus the pre-existing F001-F120 from v4.3).

This addendum supplements `FEATURE_BACKLOG.md` (Pass 1 F121-F190) — both files are inputs to the updated `PRIORITIZATION_MATRIX.md`.
