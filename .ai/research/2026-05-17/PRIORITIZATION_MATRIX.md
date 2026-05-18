# OpenCut — Prioritization Matrix

**Audit date:** 2026-05-17
**Baseline:** v1.32.0 + F138 hardening tree validated in Pass 4 + ROADMAP.md v4.3-v4.7 *Now*-tier mostly shipped.
**Input:** `FEATURE_BACKLOG.md` F121-F190 (70 items) + ROADMAP.md unfinished F-numbers + Wave N-T plan.

Scores: **Impact 1-5**, **Effort 1-5** (1 = small, 5 = XL), **Risk 1-5** (1 = trivial, 5 = breaks many surfaces). **Tier** = Now / Next / Later / Under Consideration / Rejected.

> The v4.3 audit already tiered F093-F120 and shipped most of the *Now* tier in the past 30 days. This matrix continues the F-numbering at F121 and only re-states pre-F120 items where the audit changed their status.

---

## 1. Now — ship in next 1-2 releases (v1.33 — v1.34)

These remove known false-positive claims, security exposures, or trivial wins that unblock the *Next* tier.

| F# | Title | I | E | R | Notes |
|---|---|---|---|---|---|
| **F138** | Commit the 7-file in-flight hardening batch (UNC realpath + ipaddress.is_loopback + helper finally + nested user_data + safe_bool follow-up) | 4 | 1 | 1 | Already implemented in dirty tree; needs `git commit` + push. Closes two defence-in-depth holes on F112 / F069 / F070. |
| **F121** | Pillow 12.2 major bump + thumbnail/caption/watermark regression test | 5 | 2 | 2 | Closes CVE-2026-40192 / 25990. Re-run caption renderer fixtures. |
| **F122** | flask-cors 6.x bump | 5 | 1 | 1 | Closes CVE-2024-1681 / 6839 / 6844 / 6866 / 6221. |
| **F123** | pydub replacement / audioop-lts shim for Python 3.13 | 4 | 3 | 2 | Required for the Python 3.13 support claimed in `pyproject.toml`. Pick shim *or* migrate 4 callers. |
| **F126** | OpenTimelineIO-Plugins migration | 4 | 1 | 1 | One-line pin swap. F104/F103/F105 still depend on AAF adapter. |
| **F127a** | Transformers v5 / Python 3.10 floor RFC | 4 | 1 | 2 | Decision document — implementation deferred to F127b in *Next*. Cascades to F134/F136. |
| **F128** | FFmpeg filter-graph regression suite | 4 | 3 | 2 | Required gate before F129 bundled FFmpeg bump. |
| **F130** | Bundled OpenCV → 4.13 wheel refresh | 3 | 1 | 1 | Closes CVE-2025-1594 / 9951 / 23-49502 / 23-6605 / 26-22801 in bundled libs. |
| **F131** | esbuild ≥0.25 verification on every npm install | 3 | 1 | 1 | Already mitigated via F095; add a CI assertion. |
| **F133** | onnxruntime ≥1.25 bump | 3 | 1 | 1 | Closes CVE-2026-27904. |
| **F135** | whisperx 3.8.5 bump | 3 | 1 | 1 | Active maintenance; minor. |
| **F137** | MCP SDK pin to `>=1.26,<2` | 3 | 1 | 1 | Avoid pre-alpha v2 rewrite. |
| **F139** | Caption translation endpoint (`POST /captions/translate`) | 5 | 2 | 2 | NLLB-200, low effort, strong commercial parity. Caption-only path, not full dub. |
| **F140** | C2PA 2.3 sidecar bump | 4 | 3 | 2 | Upgrade F110 emitter to 2.3 spec (live-video provenance, plain-text/OGG/large-AVI manifests). |
| **F147** | Register `opencut-mcp-server` upstream in `modelcontextprotocol/servers` | 3 | 1 | 1 | Discoverability. |
| **F149** | Fill K3.5 — AI Slate ID | 4 | 3 | 2 | Florence-2 already installed; small new module + Premiere XMP write. DaVinci 21 parity. |
| **F162** | SAM 2 → SAM 3.1 promotion | 4 | 3 | 2 | New default for `object_removal.py` + downstream Cutie / DEVA. Apache-2 lineage. |
| **F163** | Depth Anything V2 / Depth Pro → Depth Anything 3 | 4 | 3 | 2 | Apache-2; spatially consistent geometry. Update `depth_effects.py` + `depth_depthpro.py`. |
| **F167** | OmniVoice fill of Wave H2.4 (Apache-2, 600+ langs) | 4 | 3 | 2 | Fills the long-tail-language TTS gap. |
| **F169** | Qwen3-TTS streaming backend | 3 | 2 | 1 | 97 ms latency — real-time preview path. |
| **F176** | Public eval dataset bundle (gated by env var) | 4 | 3 | 1 | Required to make F178 / F120 evaluation harness meaningful. |
| **F177** | model_cards.py 2026-Q2 sweep + regenerate `docs/MODELS.md` | 4 | 2 | 1 | Add cards for ~20 new model surfaces from this audit. |
| **F178** | AI eval harness v2 (VRAM peak, reference-dataset score, backend telemetry, cross-backend compare endpoint) | 4 | 2 | 1 | Small but enables future model claims to be measurable. |
| **F181** | Hermetic bootstrap on UV-style trampolines | 3 | 1 | 1 | Document UV-incompat *or* fix `scripts/bootstrap_check.py`. |
| **F182** | Run `gh issue` seeder against live repo (F097 / F117) | 3 | 1 | 1 | Currently `gh issue list` returns empty. Make the contributor channel real. |
| **F183** | Remove accidentally-committed log files; `.gitignore` them | 2 | 1 | 1 | `pt.log`, `build104.log`, `build35.log`, `pytest-*.log` in repo root. |
| **F185** | features.md aspirational marker banner | 2 | 1 | 1 | Three-line header. Stops new contributors from mistaking features.md for a commitment. |

**Now-tier total:** 27 items. Estimated effort: ~6 weeks at 1 maintainer, ~3 weeks at 2 maintainers.

---

## 2. Next — high leverage after Now (v1.35 — v1.40)

These build on the Now-tier substrate and deliver the user-visible features that the wave letters and competitor matrix flag as gaps.

| F# | Title | I | E | R | Notes |
|---|---|---|---|---|---|
| **F143** | `/agent/chat` conductor + timeline diff (the flagship) | 5 | 4 | 3 | Biggest competitive leverage. Underlord-style sidebar chat + visible timeline diff before commit. |
| **F144** | Post-turn self-review for `/agent/chat` | 4 | 2 | 2 | Underlord 2026 trust pattern. Must ship together with F143 or F143 erodes user trust. |
| **F145** | Skills SDK + MCP packaging | 5 | 4 | 3 | Reusable agent skills (`polish_interview`, `cut_youtube_short`, …) as Claude Code Skills + MCP tools. Distribution channel. |
| **F146** | UXP-native MCP transport (survive Sept 2026 CEP EOL) | 5 | 4 | 3 | Every competing PPro MCP today is CEP-bound. First UXP-MCP wins the post-EOL landscape. |
| **F158** | StreamDiffusionV2 real-time preview on existing LTX-2.3 / Wan / Open-Sora backends | 5 | 4 | 3 | Single biggest UX leap vs CapCut / Runway / Captions. |
| **F127b** | Transformers v5 + Python 3.10 floor implementation | 4 | 4 | 4 | Big cascade — unblocks F134, F136, many model surfaces. Schedule for v1.34.0 boundary. |
| **F124** | basicsr / gfpgan / realesrgan replacement (ONNX migration *or* shim) | 4 | 4 | 4 | Block on F127b; pick ONNX path. |
| **F125** | audiocraft torch isolation / replacement | 4 | 3 | 3 | Single biggest blocker on torch upgrades. Pick: fork, sidecar venv, or replace with transformers v5 native MusicGen. |
| **F129** | Bundled FFmpeg upgrade to 8.1 (after F128) | 4 | 2 | 2 | D3D12 H.264/AV1 hw encode (huge Windows win), Vulkan ProRes, drawvg vector overlays. |
| **F132** | Vite 8 / Rolldown upgrade | 3 | 2 | 2 | Builds 10-30× faster; pairs with Vite 8 esbuild fix. |
| **F134** | pyannote.audio 4.x bump (after F127b) | 3 | 2 | 2 | Community-1 model, 40% faster; breaking `Binarize.__call__` semantics. |
| **F136** | PySceneDetect 0.7 bump (after F127b) | 3 | 1 | 1 | Refactored video input API. |
| **F141** | IMSC Text Profile 1.3 caption emit | 4 | 3 | 2 | Netflix/OTT delivery requirement. |
| **F142** | OCIO 2.5 + ACES 2.0 built-in configs | 4 | 3 | 2 | Update F109 validator + LUT pipeline. |
| **F148** | Fill K3.4 — AI Face Age Transformer | 4 | 4 | 3 | DaVinci 21 parity. IP-Adapter + Cutie temporal tracking. |
| **F150** | Fill K3.3 — IntelliScript screenplay→sequence (`.fdx` / `.fountain` parser + WhisperX fuzzy match) | 4 | 4 | 3 | DaVinci 21 parity. |
| **F151** | Fill K2.19 — CineFocus rack focus (Depth Pro pipeline → DOF bokeh) | 4 | 3 | 2 | DaVinci 21 parity. Depth pipeline already in place. |
| **F152** | AI Eye Contact correction | 4 | 4 | 3 | MediaPipe face mesh + lightweight GAN; preview slider. |
| **F153** | AI Overdub / voice-correction conductor | 4 | 4 | 3 | Chatterbox + EchoMimic V3 + crossfade. Building blocks exist. |
| **F154** | Fill K3.2 — Auto-trailer / promo generator | 4 | 3 | 2 | Descript Underlord parity. |
| **F155** | VidMuse 2026 ckpt — video-to-music | 4 | 4 | 3 | No commercial editor ships local-free equivalent. |
| **F156** | OpusClip engagement heatmap + A/B variants | 4 | 3 | 2 | Extends Wave H virality scoring. |
| **F159** | Diffusion Templates plug-in framework | 3 | 3 | 2 | DiffSynth-Studio — user LoRAs on top of LTX-2.3 / Wan. |
| **F160a** | WebView UI UXP migration spike (Bolt UXP March 2026 pattern) | 4 | 3 | 3 | Spike only — full impl in *Later*. |
| **F164** | LTX-Video 1.x → 2.3 | 4 | 3 | 2 | Native 4K @ 50 FPS, 20 s, sync audio, portrait 9:16. |
| **F165** | Wan 2.1 → Wan 2.7 (gate on weights upload verification) | 3 | 3 | 3 | Conditional. |
| **F166** | daVinci-MagiHuman 15B backend | 4 | 4 | 3 | Third lip-sync backend; auto-recommended when ≥48GB GPU. |
| **F168** | IndexTTS2 — duration-controlled TTS | 4 | 3 | 2 | Critical for `dub_pipeline.py` to land translated audio fitting original cut. |
| **F170** | VoxCPM2 backend | 3 | 3 | 2 | Tokenizer-free TTS, 48 kHz. |
| **F171** | Fish Speech S2 Pro backend (gate on licence verify) | 3 | 3 | 2 | < 150 ms latency. |
| **F172** | HunyuanVideo-Foley backend (gate on licence) | 3 | 3 | 3 | Frame-accurate foley; no OpenCut equivalent. |
| **F174** | SeedVR / SeedVR2.5 backend in `upscale_hub.py` | 3 | 3 | 2 | Pair with FlashVSR. |
| [x] **F180** | Wave N-T planning ledger refresh through F-number lens | 3 | 3 | 1 | Closed in Pass 36 with `WAVE_N_T_F_NUMBER_LEDGER.md` and a roadmap/ledger drift test. |

**Next-tier total:** ~28 items. Estimated effort: ~6 months sequenced; 3 months parallelised across 3 maintainers.

---

## 3. Later — credible but dependent on stable foundations (v1.41+)

| F# | Title | I | E | R | Notes |
|---|---|---|---|---|---|
| **F157** | Motion Brush equivalent | 3 | 5 | 4 | Runway / CapCut feature. Depends on F158 streaming. |
| **F160b** | WebView UI UXP full implementation | 4 | 5 | 4 | Depends on F160a spike + Adobe UXP CSS variables for Premiere stabilising. |
| **F173** | Mimi codec audio I/O tier | 2 | 3 | 2 | Useful for browser preview + low-bandwidth review bundles. |
| **F175** | MagiCompiler inference acceleration | 3 | 2 | 2 | Plug-and-play; defer until daVinci-MagiHuman is in. |
| **F179** | features.md ↔ F-number reconciliation pass | 3 | 4 | 1 | Cleanup walk of 402 features. |
| **F184** | docs/ROADMAP.md mirror resolution | 2 | 1 | 1 | Pick canonical home. |

---

## 4. Under Consideration — needs an RFC

| F# | Title | Why "Under Consideration" |
|---|---|---|
| **F161** | UXP Hybrid Plugin sidecar RFC | Native C++ DLL/dylib inside the panel is powerful (OpenCV / TFLite / hardware SDKs) but adds packaging complexity. Worth an RFC; ship only if measurable latency win on common round-trips. |

(Plus the pre-F120 Under-Consideration set from ROADMAP.md: F039 cloud co-edit, F067 telemetry, F075 WebCodecs, F077 MLT/GStreamer backend, F087 FastAPI sidecar, F107 WebCodecs preview, F113 opt-in local telemetry — all unchanged.)

---

## 5. Rejected (carry forward from ROADMAP.md v4.3)

Unchanged:

- F043 native mobile editor
- F078 full NLE replacement UI
- F079 in-app marketplace with remote code execution
- F080 AGPL code import
- F081 no-license small-AI code reuse
- F082 mandatory cloud account for core workflows
- F083 silent model downloads
- F084 route count as quality metric
- F085 raw user-media upload to third-parties by default
- F086 Flask rewrite before readiness work

Newly explicit:

- **Voxtral TTS** — CC-BY-NC weights.
- **MatAnyone 2 in production** — NTU S-Lab 1.0 non-commercial.
- **HunyuanVideo / 1.5 default-on** — Tencent territory carve-outs.

---

## 6. Adobe gap reports (file, do not implement)

These are not OpenCut work but should be tracked in the same ledger.

| F# | UXP API to file with Adobe |
|---|---|
| F186 | `createCaptionTrack()` |
| F187 | `createSubsequence()` from in/out points |
| F188 | `exportAsFinalCutProXML()` / `exportAsProject()` |
| F189 | `startDrag()` for file drag-out (promised CY2026) |
| F190 | UXP QE-DOM replacements (ripple delete, effect-by-name, advanced trim) |

---

## 6.5 Pass-2 additions (F191-F260)

The second autonomous research pass on 2026-05-17 added 70 more F-numbers across four sources: route audit, installer audit, test coverage, features.md reconciliation, plus three subagents (Frame.io review, niche AI / accessibility / standards, UXP migration deep-dive). Full ledger in [`FEATURE_BACKLOG_ADDENDUM.md`](FEATURE_BACKLOG_ADDENDUM.md). Tier deltas:

**Now (priority bumps + new): 3 open items; F191/F195/F197/F199/F202/F204/F207/F208/F209/F218/F219/F236/F237/F240/F241/F243/F244 closed locally after Pass 23 wrap-up**

| F# | Title | Why priority |
|---|---|---|
| [x] F191 | Auto-derive `FeatureRecord` from check functions | Closed in Pass 8 with 58 generated records / 67 route bindings |
| [x] F195 | Add 12 missing MCP tools for post-Wave-M shipped routes | Closed in Pass 9; 27→39 curated MCP tools with dispatch/path-validation tests |
| [x] F197 | `NON_AI_CHECKS` allowlist in `registry.py` | Closed in Pass 8; registry now owns allowlist |
| [x] F199 | Document `/api/*` alias policy | Closed in Pass 7; 15 aliases + 218 canonical `/api` routes |
| [x] **F202** | **Apple notarisation for macOS PyInstaller bundle** | Closed locally in Pass 10 with Developer ID signing + notarytool release wiring; first live acceptance requires configured GitHub secrets |
| [x] F204 | Auto-attach SBOM to GitHub release | Closed in Pass 11 with Linux release generation, artifact archive, and tag release upload |
| F205 | Raise CI coverage floor from 50% to actual | Still open; Pass 12 timed out after 20 minutes and Pass 23 was interrupted after 36m46s with only partial 52.12% coverage JSON |
| [x] F207 | Embed bundled FFmpeg version in installer manifest | Closed in Pass 12 with WPF/Inno installer manifests |
| [x] F208 | OpenAPI spec validity test | Closed in Pass 13 with `/openapi.json` path-parameter normalization, unique operation IDs, mutating-method 400/403 responses, and release-smoke contract tests |
| [x] F209 | MCP tool ↔ route consistency test | Closed in Pass 14 with live Flask route checks for all MCP routes and special action dispatch paths |
| [x] F218 | Import-order stability test | Closed in Pass 15 with a pinned core blueprint order and release-smoke route-collision/import-order gate |
| [x] F219 | SBOM completeness test | Closed in Pass 16 with declared-dependency, model-card, and dependency-graph coverage in release smoke |
| [x] **F236** | **FCC user-overridable caption style tokens** | **Closed in Pass 17 with FCC-sourced token schema, preview route, and burn-in integration** |
| [x] **F237** | **EBU R128 v5.0 + BS.1770-5 correction** | **Closed in Pass 18 with source-backed loudness registry; ITU BS.1770-5 is in force and BS.1770-4 is superseded** |
| [x] F240 | Per-target reading-speed profile (Netflix/BBC/DCMP/FCC/YouTube) | Closed in Pass 19 with source-backed caption QC profile overlays |
| [x] F241 | HarfBuzz-mandatory CI gate | Closed in Pass 20 with a release-smoke/CI text-shaping gate for FFmpeg/libass HarfBuzz/FriBidi plus Pillow RAQM and optional Skia capability reporting |
| [x] F243 | UTF-8 (no BOM) SRT writer + opt-in legacy toggle | Closed in Pass 21 with UTF-8/no-BOM default SRT output plus opt-in `utf-8-sig` route/CLI/file-writer support |
| [x] F244 | Language confidence per Whisper segment | Closed in Pass 22 with ASR/language confidence metadata, Hindi/Arabic review flags, low-confidence review reasons, route/export/cache/state preservation, and release-smoke coverage |
| F251 | Track `@adobe/premierepro@beta` per-week diff in CI | Catch UXP-gap-closing APIs early |
| F259 | UXP HTTP-on-macOS workaround documentation | Known 25.6.3 bug affects MCP sidecar |

**Next (32 items):** [x] F192, [x] F194, [x] F198, [x] F200, [x] F201, [x] F203, [x] F211, [x] F213, [x] F214, [x] F215, [x] F216, [x] F217, [x] F223, F225, F226, F227, F229, F231, F233, F234, F238, F239, [x] F242, F249, F250, F252, F254, F255, F256, F257, F258, F260. Includes the **F143 conductor + F252 UXP/WebView migration + F225-F229 F105 review-bundle extensions** as parallel flagship tracks.

**Later (18 items):** F193, F196, F206, F210, F212, F220, F221, F222, F224, F228, F230, F232, F235, F245, F246, F247, F248, F253. Includes deferred OSS-quality items (Dolby Vision pipeline, ADM BWF Atmos, IMF builders) that wait on F215 fuzz + F218 import-order regression coverage.

**Two regulatory deadlines bumped to Now:**
- F202 Apple notarisation (Homebrew Cask mandatory Sept 1, 2026)
- F236 FCC caption style tokens (effective Aug 17, 2026; repository-side token contract closed in Pass 17)

Both should ship in v1.33 or earliest v1.34 to keep Mac users on Homebrew and US broadcast users compliant.

Adjacent standards correction: F237 closed in Pass 18 with the source-backed BS.1770-5 / EBU R128 v5.0 loudness registry.

---

## 7. Sequencing summary

```
v1.33.0 (Now batch 1, ~2 weeks)
  Security: F138 commit, F121 Pillow, F122 flask-cors, F126 OTIO-Plugins,
            F130 OpenCV, F131 esbuild check, F133 ORT, F137 MCP, F135 whisperx
  UX:       F139 caption translation, F140 C2PA 2.3, F147 MCP registry
  Eval:     F176 dataset bundle, F177 model cards sweep, F178 eval v2
  Cleanup:  F181 bootstrap fix, F182 issue seeder run, F183 logs, F185 banner

v1.34.0 (Now batch 2, ~2 weeks)
  Models:   F149 Slate ID, F162 SAM 3.1, F163 Depth Anything 3,
            F167 OmniVoice, F169 Qwen3-TTS
  Test:     F128 FFmpeg filter regression suite

v1.35.0 (RFC + cascade prep, ~2 weeks)
  Decision: F127a Transformers v5 / Python 3.10 RFC
  Spike:    F160a WebView UI spike

v1.36-v1.38 (Next flagship, ~6 weeks)
  Conductor: F143 /agent/chat, F144 self-review, F145 Skills SDK
  MCP:       F146 UXP MCP transport
  UXP:       (parity gap closure)

v1.39-v1.42 (Next breadth, ~3 months parallel)
  Models:    F148 face age, F150 IntelliScript, F151 CineFocus,
             F152 eye contact, F153 Overdub, F154 trailer, F155 VidMuse,
             F156 OpusClip variants, F158 StreamDiffusionV2,
             F159 Diffusion Templates, F164 LTX-2.3, F165 Wan 2.7,
             F166 daVinci-MagiHuman, F168 IndexTTS2, F170 VoxCPM2,
             F171 Fish Speech S2, F172 HunyuanVideo-Foley,
             F174 SeedVR2.5
  Infra:     F127b Transformers v5 + Py 3.10 floor, F124 basicsr replace,
             F125 audiocraft isolate, F129 FFmpeg 8.1, F132 Vite 8,
             F134 pyannote 4, F136 scenedetect 0.7
  Standards: F141 IMSC 1.3, F142 OCIO 2.5/ACES 2.0
  Wave:      [x] F180 Wave N-T re-tier

v1.43+ (Later)
  F157 Motion Brush, F160b WebView impl, F173 Mimi codec,
  F175 MagiCompiler, F179 features.md sweep, F184 docs/ROADMAP merge
```

---

## 8. Capacity assumptions

- 1-2 maintainers, mixed full-time / part-time.
- ~2 weeks per minor version (already the observed cadence: v1.25→v1.32 in ~one month).
- Major model integrations (F143, F146, F158) take 4-6 weeks each.
- Cascade upgrades (F127b → F124/F125/F134/F136) block on a single decision; once decided, they parallelise.

The sequencing above assumes the Now batch is taken **first** because it removes false-positive marketing and CVE exposure that erode trust faster than any model wave can add it.
