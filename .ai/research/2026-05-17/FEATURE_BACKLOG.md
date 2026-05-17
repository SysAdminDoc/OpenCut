# OpenCut — Feature Backlog (Raw Harvest)

**Audit date:** 2026-05-17
**Status:** Pre-prioritization. Items here flow into `PRIORITIZATION_MATRIX.md`. Item numbering continues the F-series from ROADMAP.md v4.3 (last shipped: F006-F120).

> Each item: ID — title — one-line "what" — source(s) — effort (S/M/L/XL) — fit (yes/conditional/no) — supersedes / complements / new — proposed tier.

---

## A. Security & dependency hardening (drop-in)

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F121 | Pillow 12.2 major bump + thumbnail regression test | Close CVE-2026-40192 (FITS GZIP bomb) + CVE-2026-25990 (PSD overflow); re-test caption renderer, thumbnail, watermark detection. | Pillow advisories | S | yes | n/a | **Now** |
| F122 | flask-cors 6.x bump | Close CVE-2024-1681 / 6839 / 6844 / 6866 / 6221 (unfixed in 4.x). | Debian DLA-4197-1 | S | yes | n/a | **Now** |
| F123 | pydub replacement / audioop-lts shim for Python 3.13 | Either vendor `audioop-lts` and alias to `audioop` before pydub loads, or migrate the four pydub callers to `soundfile` + `librosa`. | pydub PR #816 closed; audioop-lts PyPI | M | yes | replace | **Now** |
| F124 | basicsr / gfpgan / realesrgan replacement or shim | Either vendor the `functional_tensor → functional` shim or replace face/upscale stack with direct ONNX. | BasicSR PR #659, Real-ESRGAN #859 | L | yes | replace | **Next** |
| F125 | audiocraft torch isolation | Either fork to relax `torch==2.1.0`, or run audiocraft in its own venv, or drop in favour of transformers v5 native MusicGen wrappers. | audiocraft #423 | M | yes | replace | **Next** |
| F126 | OpenTimelineIO-Plugins migration | Switch pin from `opentimelineio` to `OpenTimelineIO-Plugins` so AAF adapter (now external) keeps working. | OTIO adapters split | S | yes | n/a | **Now** |
| F127 | Transformers v5 / Python 3.10 floor decision RFC | Decide: stay on 4.x (keep Python 3.9), or take v5 hop (drop 3.9, gain modern model support). | HF transformers v5 migration guide | S (RFC) + L (impl) | yes | n/a | **Now (RFC), Next (impl)** |
| F128 | FFmpeg filter-graph regression suite | Golden-output tests on every FFmpeg-driven core module so the installer-bundled FFmpeg can be safely bumped to 8.1 without silent breakage. | FFmpeg 8.1 release notes | M | yes | n/a | **Now** |
| F129 | Bundled FFmpeg upgrade to 8.1 | After F128 passes; gain D3D12 H.264/AV1 hw encode, Vulkan ProRes, JPEG-XS, drawvg vector overlays, EXIF parsing. | FFmpeg 8.1 | S | yes | n/a | **Next** |
| F130 | Bundled OpenCV → 4.13 wheel refresh | Get ffmpeg/libpng CVE fixes in the bundled wheel; pre-install system FFmpeg 8.1 on PATH to shadow bundled libs. | OpenCV-python #1212/#1186 | S | yes | n/a | **Now** |
| F131 | esbuild ≥0.25 verification on every npm install | Already mitigated via F095 `overrides`; add a CI assertion that `npm ls esbuild` resolves only ≥0.25. | GHSA-67mh-4wv8-2f99 | S | yes | n/a | **Now** |
| F132 | Vite 8 / Rolldown upgrade | Faster builds (10-30×), better .map handling. Defer to next CEP polish wave. | Vite 8 announce | M | yes | n/a | **Next** |
| F133 | onnxruntime ≥1.25 bump | Closes minimatch CVE-2026-27904 and OOB-read fixes. | ORT 1.25 release | S | yes | n/a | **Now** |
| F134 | pyannote.audio 4.x bump (conditional on F127) | Community-1 model, 40% faster; breaking `Binarize.__call__` returns strings; requires Py 3.10. | pyannote 4.0 changelog | S (after F127) | conditional | n/a | **Next** |
| F135 | whisperx 3.8.5 bump | Active maintenance, alignment + diarization fixes. | whisperx releases | S | yes | n/a | **Now** |
| F136 | PySceneDetect 0.7 bump (conditional on F127) | Refactored video input API, requires Py 3.10. | PySceneDetect 0.7 | S | conditional | n/a | **Next** |
| F137 | MCP SDK pin to `>=1.26,<2` | v2 is pre-alpha (FastMCP → McpServer rewrite). | MCP changelog | S | yes | n/a | **Now** |
| F138 | Commit the in-flight 7-file hardening batch | UNC realpath + ipaddress.is_loopback + helper finally-block + nested user_data dirs + safe_bool follow-up. Already implemented in the dirty tree. | dirty diff 2026-05-17 | S | yes | n/a | **Now (commit)** |

## B. Caption / accessibility / standards (drop-in)

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F139 | Caption translation endpoint | `POST /captions/translate` accepts SRT + src/tgt lang; runs NLLB-200 locally; returns translated SRT preserving timing. | research.md §1.5; Premiere 26 "Translate Captions" | S | yes | new | **Now** |
| F140 | C2PA 2.3 sidecar bump | Update F110 emitter to spec 2.3 (live-video provenance, plain-text/OGG/large-AVI manifests, finer-grained edit logs, cloud-anchored trust lists). | c2pa.org 2.3 launch | M | yes | upgrade | **Now** |
| F141 | IMSC Text Profile 1.3 emit | Add IMSC 1.3 as caption export format alongside WebVTT/SRT/ASS — Netflix/OTT delivery requirement. | W3C IMSC 1.3 CR (3 Apr 2026) | M | yes | new | **Next** |
| F142 | OCIO 2.5 + ACES 2.0 built-in configs | Bump F109 validator + LUT pipeline to OCIO 2.5; default-config to ACES 2.0 Output Transforms via OCIO views. | opencolorio.org; VFX Reference Platform 2026 | M | yes | upgrade | **Next** |

## C. Agentic / chat / MCP (the biggest competitive gap)

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F143 | `/agent/chat` conductor + timeline diff | User types intent → LLM produces JSON plan of OpenCut API calls → job-queued execution → **visible timeline diff before commit**. | Descript Underlord; Captions.ai; Crayotter; FireRed-OpenStoryline; research.md §1.2 | L | yes | new | **Next (flagship)** |
| F144 | Post-turn self-review for `/agent/chat` | Agent diffs executed plan vs original request; re-invokes itself on out-of-scope edits. Underlord's 2026 trust-fix pattern. | Underlord 2026 release notes | M (after F143) | yes | new | **Next** |
| F145 | Skills SDK + MCP packaging | Named reusable multi-step workflows (`polish_interview`, `cut_youtube_short`, `prep_podcast_episode`); ship as Claude Code Skills + MCP tools; cleanly versioned. | FireRed-OpenStoryline Style Skills | L | yes | new | **Next** |
| F146 | UXP-native MCP transport | Migrate MCP server's Premiere ops from CEP-bound calls to UXP messaging so it survives Sept 2026 CEP EOL. | Adobe CEP EOL; competitor MCP servers all CEP-bound | L | yes | upgrade | **Next** |
| F147 | Register `opencut-mcp-server` upstream | List in `modelcontextprotocol/servers` registry. | MCP registry | S | yes | new | **Now** |

## D. AI capability gaps (parity with DaVinci 21 / Descript / Runway)

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F148 | Fill K3.4 — AI Face Age Transformer | IP-Adapter + Cutie temporal tracking; age slider 0-1. | DaVinci 21 Face Age Transformer | L | yes | fill stub | **Next** |
| F149 | Fill K3.5 — AI Slate ID | Florence-2 VLM (already installed) reads clapperboard scene/take/camera; stamps into OTIO + Premiere XMP. | DaVinci 21 Slate ID | M | yes | fill stub | **Now** |
| F150 | Fill K3.3 — IntelliScript screenplay→sequence | Accept `.fdx` / `.fountain`; parse scene headings; WhisperX fuzzy-match transcript; auto-assemble. | DaVinci 21 IntelliScript; Fountain spec | L | yes | fill stub | **Next** |
| F151 | Fill K2.19 — CineFocus rack focus | Depth Pro pipeline drives DOF bokeh; keyframeable focal point + aperture. | DaVinci 21 CineFocus | M | yes | fill stub | **Next** |
| F152 | AI Eye Contact correction | MediaPipe face mesh + lightweight GAN gaze redirection; preview slider; per-frame temporal smoothing. | Descript Eye Contact; NVIDIA Maxine | L | yes | new | **Next** |
| F153 | AI Overdub / voice-correction conductor | Clone voice from surrounding audio (Chatterbox) → generate corrected segment → lip-sync (EchoMimic V3) → crossfade audio boundaries → composite video. | Descript Overdub; research.md §1.3 | L | yes | new | **Next** |
| F154 | Fill K3.2 — Auto-trailer / promo generator | LLM moment scoring → top-N extract → MusicGen ramp + title card via `declarative_compose.py` + CTA. | Descript Underlord trailer | M | yes | fill stub | **Next** |
| F155 | VidMuse 2026 ckpt — video-to-music | Pretrained VidMuse on HF → AudioCraft MusicGen decoder; mood/tempo/scene-transition matched score. | VidMuse CVPR'25; research.md §1.6 | L | yes | new | **Next** |
| F156 | OpusClip engagement heatmap + A/B variants | Generate multiple short versions per source with different hooks; predicted retention overlay; ranked output. | OpusClip ClipAnything; AUDIT.md §Opus Clip | M | yes | new | **Next** |
| F157 | Motion Brush equivalent | Paint motion onto stills / static regions; drives video generation. | Runway Motion Brush; CapCut | XL | conditional | new | **Later** |

## E. Real-time / streaming (the biggest UX gap)

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F158 | StreamDiffusionV2 real-time preview | Convert existing LTX-2.3 / Wan / Open-Sora backends into live-streaming generators (< 40 ms / frame target). | arXiv 2511.07399 v2 | L | yes | new | **Next** |
| F159 | Diffusion Templates plug-in framework | DiffSynth-Studio "Diffusion Templates" — user-supplied controllable LoRAs (camera, pose, style) on top of LTX-2.3 / Wan. | DiffSynth-Studio repo | M | yes | new | **Next** |
| F160 | WebView UI UXP migration spike | Adopt Bolt UXP WebView pattern (March 2026) to preserve 8500-line vanilla JS/CSS panel through CEP EOL. | Hyperbrew Bolt UXP March 2026 announcement | L (spike), XL (impl) | yes | upgrade | **Next (spike), Later (impl)** |
| F161 | UXP Hybrid Plugin sidecar RFC | Evaluate native C++ DLL / dylib inside the panel for OpenCV / TFLite / hardware SDK calls; could shave roundtrip latency. | Adobe UXP Hybrid Plugins (Apr 2026) | S (RFC) | conditional | new | **Under Consideration** |

## F. Model surface refreshes

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F162 | SAM 2 → SAM 3.1 promotion | New default for `object_removal.py` and downstream Cutie / DEVA. | Meta AI blog | M | yes | upgrade | **Now** |
| F163 | Depth Anything V2 / Depth Pro → Depth Anything 3 | New default for `depth_effects.py`; complement with Online Video Depth Anything for streaming. | ByteDance Seed repo | M | yes | upgrade | **Now** |
| F164 | LTX-Video 0.9.8 / 1.x → LTX-Video 2.3 | Native 4K @ 50 FPS, 20 s, synchronized audio, portrait 9:16, distilled variants. New routes `/generate/ltx/v23`, `/generate/ltx/desktop-handoff`. | LTX-2.3 release | M | yes | upgrade | **Next** |
| F165 | Wan 2.1 → Wan 2.7 (gate on weights upload) | Apache 2.0 weights; major motion + audio-sync upgrade. | WaveSpeed Wan 2.7 | M | conditional | upgrade | **Next** |
| F166 | daVinci-MagiHuman 15B backend | Add as third lip-sync backend; `recommended: true` when ≥48GB GPU available. | Sand.ai + SII-GAIR | L | yes | new | **Next** |
| F167 | OmniVoice fill of Wave H2.4 | Fill the OmniVoice stub with Apache-2 600+ language TTS. | k2-fsa/OmniVoice | M | yes | fill stub | **Now** |
| F168 | IndexTTS2 — duration-controlled TTS | Critical for dubbing-to-timecode in `dub_pipeline.py`. | IndexTTS2 repo | M | yes | new | **Next** |
| F169 | Qwen3-TTS streaming backend | 97 ms ultra-low-latency for real-time preview path. | QwenLM/Qwen3-TTS | S | yes | new | **Now** |
| F170 | VoxCPM2 backend | Tokenizer-free TTS, 30 langs, 48 kHz, voice design. | OpenBMB/VoxCPM | M | yes | new | **Next** |
| F171 | Fish Speech S2 Pro backend (gate on licence verify) | < 150 ms latency, 80+ langs, emotion tags. | fishaudio/fish-speech | M | conditional | new | **Next** |
| F172 | HunyuanVideo-Foley backend (gate on licence) | End-to-end Text-Video-to-Audio, 48 kHz, frame-accurate. | HunyuanVideo-Foley | M | conditional | new | **Next** |
| F173 | Mimi codec audio I/O tier | 12.5 Hz, 1.1 kbps streaming neural codec; useful for browser preview, low-bandwidth review bundles. | Kyutai/Mimi | M | yes | new | **Later** |
| F174 | SeedVR / SeedVR2.5 backend | Pair with FlashVSR in `upscale_hub.py`; one-step diffusion, arbitrary resolution. | ByteDance Seed | M | yes | new | **Next** |
| F175 | MagiCompiler inference acceleration | Plug-and-play speedups for OpenCut's diffusion paths. | Sand.ai MAGI-1 | S | yes | new | **Later** |

## G. Eval / governance / docs

| F# | Title | What | Source | Effort | Fit | S/C/N | Tier |
|---|---|---|---|---|---|---|---|
| F176 | Eval dataset bundle | Bundle public eval datasets (VBench / Spring / VFI-2024 / DAVIS-2017 / REDS / LRS3 / VoxCeleb 2 / LibriTTS / MUSDB18 / EBU SQAM / Netflix open content / C2PA test vectors / IMSC reference) under `opencut/data/eval/` with checksums + licence notes. Gated by `OPENCUT_DOWNLOAD_EVAL=1`. | Common ML benchmarks | M | yes | new | **Now** |
| F177 | model_cards.py 2026-Q2 sweep | Add cards for ~20 new model surfaces above; regenerate `docs/MODELS.md`. | F115 generator | S | yes | new | **Now** |
| F178 | AI eval harness v2 | Add per-model VRAM peak, reference-dataset score column, backend selection telemetry, cross-backend comparison endpoint. | F120 extension | S | yes | upgrade | **Now** |
| F179 | features.md ↔ F-number reconciliation pass | Walk all 402 features in `features.md`; mark each as `[shipped]` / `[planned-F###]` / `[planned-W###.##]` / `[rejected]`. | MEMORY_CONSOLIDATION.md §3 | L | yes | n/a | **Later (cleanup)** |
| F180 | Wave T-V planning ledger refresh | The wave letters (N-T) in ROADMAP.md predate the v4.3 audit. Re-tier each item against the F-number governance lens. | ROADMAP.md | M | yes | n/a | **Next** |
| F181 | Hermetic bootstrap on UV-style trampolines | F093 currently fails on `.venv` UV trampoline (cannot spawn child Python). Either document UV-incompatibility or fix the script. | dirty diff repo state | S | yes | fill | **Now** |
| F182 | gh issue seeder run | F097/F117 ready; nobody has executed the seeder against the live repo. Currently `gh issue list` returns empty. | F097 seeded labels + templates | S | yes | n/a | **Now** |
| F183 | Removing accidentally-committed log files | Repo root has `pt.log`, `build104.log`, `build35.log`, `pytest-*.log`. Add to `.gitignore` and remove from history. | repo scan | S | yes | n/a | **Now (cleanup)** |
| F184 | docs/ROADMAP.md mirror resolution | `docs/ROADMAP.md` + `docs/ROADMAP-COMPLETED.md` drift from repo root. Pick canonical home or make one a generation step. | MEMORY_CONSOLIDATION.md §3 | S | yes | n/a | **Later** |
| F185 | features.md banner / aspirational marker | Add top-of-file banner: "Aspirational catalogue; live decisions in ROADMAP.md. Not a ship promise." | MEMORY_CONSOLIDATION.md §3 | S | yes | n/a | **Now** |

## H. Premiere UXP gap-closing (file directly with Adobe)

These are not OpenCut code items — they are Adobe UXP API gap reports that OpenCut should file (or co-file with other panel maintainers) to unblock real workflows.

| F# | Title | What | Source |
|---|---|---|---|
| F186 | File Adobe ticket: UXP `createCaptionTrack()` API | No programmatic SRT/caption injection — blocks every caption tool's UXP path. | UXP forum thread Aug 2025-Apr 2026 |
| F187 | File Adobe ticket: UXP `createSubsequence()` from in/out | Missing. |  |
| F188 | File Adobe ticket: UXP `exportAsFinalCutProXML()` / `exportAsProject()` | Both missing; blocks UXP-only OTIO/FCP-XML round-trips. |  |
| F189 | File Adobe ticket: UXP `startDrag()` for file drag-out | Promised "before end of CY2026" as of April 29, 2026 — track. |  |
| F190 | File Adobe ticket: UXP QE DOM replacements | Ripple delete, effect-by-name, advanced trim — no UXP equivalent. |  |

## I. Items explicitly NOT in this backlog (declined / out of scope)

- **Full mobile editor** — F043 already rejected. Mobile companion review/ingest (F042) remains *Later*.
- **Full NLE replacement UI** — F078 already rejected.
- **Cloud-required collaboration** — F082 already rejected.
- **Voxtral TTS** — CC-BY-NC.
- **MatAnyone 2** in production — NTU S-Lab non-commercial (research eval only).
- **HunyuanVideo / 1.5 default-on** — Tencent licence territory carve-outs (track, do not default-on).
- **MirageLSD** — research preview only.
- **Sora 2 / Veo 3.1 / Kling 3.0 / Seedance 2.0** — closed source.
- **Native cloud render farm** — F040 *Later*; OpenCut's local-first philosophy.

---

## J. Counts

| Bucket | Count |
|---|---|
| New F-numbers proposed (F121-F190) | 70 |
| Now tier | 27 |
| Next tier | 28 |
| Later tier | 6 |
| Under Consideration | 1 |
| Adobe gap reports (file, not implement) | 5 |
| Already shipped (verify against dirty tree) | 1 (F138) |

These flow into `PRIORITIZATION_MATRIX.md` with explicit impact / effort / risk / dependency scoring.
