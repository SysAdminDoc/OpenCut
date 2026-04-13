# OpenCut — Implementation Roadmap

**Version**: 3.0
**Updated**: 2026-04-13
**Baseline**: v1.9.26 (254 routes, 68 core modules, 17 blueprints, 867 tests)
**Feature Plan**: 302 features across 62 categories (see `features.md`)

---

## Guiding Principles

1. **Never break what works** — Every wave ships a working product. No "rewrite everything then test."
2. **Incremental migration** — New code coexists with old. Feature flags gate rollout. Old paths removed only after new paths are proven.
3. **User-facing value first** — Each wave delivers visible improvements, not just internal refactors.
4. **Measure before optimizing** — Add telemetry/logging before assuming bottlenecks.
5. **Shared infrastructure first** — When multiple features need the same foundation (e.g., object tracking, spectral analysis), build the foundation once, then fan out.
6. **One new dependency per feature maximum** — Avoid dep explosion. Prefer extending existing deps (OpenCV, FFmpeg, Pillow) over adding new ones.

---

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

## Implementation Waves

Features are organized into 7 waves based on dependency chains, shared infrastructure, and priority. Each wave is independently shippable. Feature numbers reference `features.md`.

### Dependency Legend

| Symbol | Meaning |
|--------|---------|
| **FFmpeg** | Pure FFmpeg filter — no Python deps beyond subprocess |
| **Pillow** | Image composition — already installed |
| **OpenCV** | Computer vision — already installed (`opencv-python-headless`) |
| **Existing AI** | Uses models already in the codebase (Whisper, Demucs, face detection, etc.) |
| **New dep** | Requires a new pip dependency |
| **New model** | Requires downloading a new AI model (potentially large) |
| **Pipeline** | Orchestrates existing modules — no new deps |

---

## Wave 1: Quick Wins — No New Dependencies

**Goal**: Ship 40+ features using only existing FFmpeg filters, Pillow, NumPy, and current AI models. Maximum user value with minimum risk.

**Timeline**: 4-6 weeks
**New deps**: Zero
**New routes**: ~35

### 1A — FFmpeg Filter Features (14 features)

These are pure FFmpeg filter additions — each is a new route calling `run_ffmpeg()` with a new filter graph.

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 53.2 | Adaptive Deinterlacing | S | FFmpeg | `yadif`/`bwdif` filter. Auto-detect via `ffprobe` `field_order` or `idet` filter. |
| 52.1 | Lens Distortion Correction | M | FFmpeg | `lenscorrection` filter with k1/k2 coefficients. Ship camera profile JSON (source: lensfun). |
| 52.3 | Chromatic Aberration Removal | S | FFmpeg | `chromanr` filter or per-channel scale via `split`/`scale`/`merge`. |
| 53.5 | Frame Rate Conversion (Optical Flow) | M | FFmpeg | `minterpolate` filter for up/down conversion. Preset modes. |
| 44.1 | Timecode Burn-In Overlay | S | FFmpeg | `drawtext` with `%{pts\:hms}` or `timecode` option. Configurable position/font. |
| 45.2 | AV1 Encoding Support | M | FFmpeg | `libaom-av1` or `libsvtav1` encoder. Add to export presets and social platform presets. |
| 45.1 | ProRes Export on Windows | M | FFmpeg | `prores_ks` encoder. Profile selector (Proxy/LT/422/HQ/4444). |
| 32.1 | Hardware-Accelerated Encoding | M | FFmpeg | Detect NVENC/QSV/AMF. Add `h264_nvenc`/`hevc_nvenc` codec options in export. |
| 20.4 | Photosensitive Seizure Detection | S | FFmpeg | Frame-to-frame luminance delta analysis. Flag >3 flashes/sec per ITU-R BT.1702. |
| 38.1 | GIF / WebP / APNG Export | S | FFmpeg | `gif`/`libwebp_anim` output format. Palette optimization via `palettegen`/`paletteuse`. |
| 3.10 | Film Grain & Vignette (Enhanced) | S | FFmpeg | `noise` + `vignette` filters with presets (Super 8, 16mm, 35mm, VHS). |
| 25.1 | Dialogue De-Reverb | M | FFmpeg | `arnndn` or `afftdn` with speech-optimized profile. |
| 42.2 | Timelapse Deflicker | M | FFmpeg | `deflicker` filter or rolling-average luminance normalization per frame. |
| 30.3 | Freeze Frame Insert | S | FFmpeg | Extract frame at timestamp, generate still clip of configurable duration, splice into sequence. |

### 1B — Pillow/Canvas Overlay Features (10 features)

Image composition overlays using existing Pillow renderer.

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 61.1 | Composition Guide Overlay | S | Pillow | Rule-of-thirds, golden ratio, center cross, safe areas on preview frame. Display-only. |
| 36.1 | Platform Safe Zone Overlay | S | Pillow | TikTok/YouTube/Instagram UI element overlays on preview frame. JSON-driven zone definitions. |
| 34.1 | Scrolling Credits Generator | M | Pillow | Bottom-to-top scroll rendered as video via Pillow frame sequence + FFmpeg encode. |
| 34.3 | Lower Third Generator | M | Pillow | Name/title bar with configurable style presets. Burn into video at timestamp range. |
| 20.3 | Color Blind Simulation Preview | S | Pillow | Apply CVD color matrix (deuteranopia, protanopia, tritanopia) to preview frame. |
| 11.2 | Click & Keystroke Overlay | M | Pillow | Parse click/key logs → render ripple animations and key badges as overlay frames. |
| 11.3 | Callout & Annotation Generator | M | Pillow | Numbered callouts, spotlight boxes, blur regions, arrows at timestamps. |
| 18.2 | Retro VHS / CRT Effect | M | Pillow+FFmpeg | Scanlines, chroma shift, noise, tracking artifacts, date stamp. Preset chain. |
| 18.3 | Glitch Effect Pack | M | Pillow+FFmpeg | Datamosh, RGB shift, block displacement, scan distortion. Per-frame render. |
| 48.1 | Highlight Reel Auto-Assembly | M | Pipeline | Score clips by audio energy + motion → select top N → assemble with transitions + music. |

### 1C — Existing AI Extensions (10 features)

Features that extend already-installed AI models with new analysis modes.

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 55.3 | Profanity Bleep Automation | S | Existing AI | Whisper word timestamps + configurable word list → 1kHz tone or silence at flagged words. |
| 61.2 | Shot Type Auto-Classification | M | Existing AI | Face size relative to frame (MediaPipe) → ECU/CU/MCU/MS/WS classification per scene. |
| 29.1 | Shot Type Search & Tagging | M | Existing AI | Store shot type in footage index (FTS5). Enable search by shot type. |
| 56.4 | Room Tone Auto-Generation | M | NumPy | Analyze quiet segments → spectral envelope → shape white noise to match → fill cuts. |
| 61.3 | Intelligent Pacing Analysis | M | Existing AI | Scene detection cut points → mean/median/stddev shot lengths → genre benchmark comparison. |
| 28.1 | Black Frame / Frozen Frame Detection | S | FFmpeg+OpenCV | `blackdetect` filter + frame differencing for frozen frames. Report timestamps. |
| 28.2 | Audio Phase & Silence Gap Check | S | FFmpeg | `aphasemeter` + silence detection. Flag phase issues and unnatural gaps. |
| 4.8 | Best Take Selection | M | Existing AI | Per-take scoring: audio quality (SNR), face visibility, sharpness, duration. Rank takes. |
| 11.5 | Dead-Time Detection & Speed Ramp | S | Existing AI | Frame differencing (scene_detect) + silence detection → speed-ramp or cut dead time. |
| 52.4 | Lens Profile Auto-Detection | S | FFmpeg | Parse camera model from `ffprobe` metadata → look up in lensfun JSON database. |

### 1D — Split-Screen & Comparison (6 features)

New composite video modes using FFmpeg `overlay`/`hstack`/`vstack`.

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 57.1 | Split-Screen Layout Templates | M | FFmpeg | JSON layout definitions (cells with x/y/w/h %). Composite via `overlay` filter chain. |
| 57.2 | Reaction Video Template | M | FFmpeg | Main content + PiP webcam. Auto-sync via audio cross-correlation. Audio ducking. |
| 57.3 | Before/After Comparison Export | M | FFmpeg | `hstack`/`vstack`, animated wipe via `overlay` + keyframed crop. Label overlay. |
| 57.4 | Multi-Cam Grid View Export | M | FFmpeg | 2x2 to 4x4 grid. Optional active-speaker highlight border from diarization data. |
| 6.3 | Side-by-Side Before/After Preview | M | FFmpeg | Preview modal showing original vs processed frame. Slider wipe in panel. |
| 3.9 | Multi-Camera Audio Sync | M | FFmpeg+NumPy | Audio fingerprint cross-correlation for time offset detection. Multicam XML output. |

**Wave 1 Total: ~40 features, 0 new dependencies, ~35 new routes**

---

## Wave 2: Pipeline Orchestration — Chain Existing Modules

**Goal**: Build high-value composite workflows that chain existing modules into new products. These are the features that competitors charge monthly subscriptions for.

**Timeline**: 3-5 weeks (can overlap with Wave 1)
**New deps**: Zero (all existing)
**New routes**: ~20

### 2A — Content Repurposing Pipelines (5 features)

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 58.1 | Long-Form to Multi-Short Extraction | L | Pipeline | Transcribe → LLM highlights (N clips) → per-clip: trim + face-reframe 9:16 + burn captions + export. Folder of numbered shorts + metadata CSV. |
| 58.4 | Podcast Episode Bundle | M | Pipeline | Denoise + normalize → clean audio export → transcribe → chapters → highlight clips → audiogram → show notes → transcript. All outputs in timestamped folder. |
| 54.4 | AI Video Summary / Condensed Recap | M | Pipeline | Scene detect → transcript LLM analysis → engagement scoring → select top N shots → trim 3-5s each → assemble with crossfades. Configurable target duration. |
| 58.2 | Video-to-Blog-Post Generator | M | Pipeline | Transcribe → LLM structured article with section headings → extract key frames at section boundaries → assemble markdown + images folder. |
| 58.3 | Social Media Caption Generator | S | Pipeline | Per-exported-clip: extract transcript → LLM generates platform-optimized post caption (char limits, hashtags, tone). JSON output alongside each clip. |

### 2B — Advanced Workflow Presets (8 features)

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 53.3 | Old Footage Restoration Pipeline | L | Pipeline | Stabilize → deinterlace (53.2) → denoise (temporal) → upscale (Real-ESRGAN) → color restore → frame rate conversion. VHS/8mm/Early Digital presets. |
| 40.3 | Video Podcast to Audio-Only | S | FFmpeg | Extract audio track, normalize, denoise, export as podcast-ready MP3/WAV with ID3 tags. |
| 40.4 | Podcast Show Notes Generator | M | Pipeline | Transcribe → LLM: summary, key topics with timestamps, pull quotes, mentioned resources, chapter markers. Markdown/HTML output. |
| 12.3 | Auto Montage Builder | M | Pipeline | Score clips (audio energy + motion) → select top N → detect beats in music track → trim clips to beat intervals → concatenate with transitions. |
| 14.1 | Paper Edit / Script Sync | L | Pipeline | Import script text → fuzzy-match against transcript → generate organized clip assembly with confidence scores. |
| 4.1 | Watch Folder / Hot Folder | M | Pipeline | Monitor directory for new files → auto-run configured workflow → output to destination folder. Background polling with configurable interval. |
| 4.2 | Render Queue | M | Pipeline | Queue multiple export jobs with different settings. Sequential execution with progress tracking. Notification on batch completion. |
| 5.1 | Multi-Platform Batch Publish | L | Pipeline | Single source → batch export for YouTube + TikTok + Instagram + LinkedIn with per-platform reframe, caption style, loudness target, and metadata. |

### 2C — Composite Feature Enhancements (4 features)

| # | Feature | Effort | Dep | Detail |
|---|---------|--------|-----|--------|
| 24.1 | Shot-Change-Aware Subtitle Timing | M | Pipeline | Scene detection (existing) → post-process captions: split at cut boundaries with minimum gap. Integrate into caption generation pipeline. |
| 16.1 | Beat-Synced Auto-Edit | L | Pipeline | Detect beats (existing librosa) → scene detect → align cuts to nearest beat → assemble. Music video editing automation. |
| 36.4 | Vertical-First Intelligent Reframe | M | Pipeline | Saliency detection + face tracking → auto-crop to 9:16 with smooth path. Better than center-crop for non-face content. |
| 30.1 | Ripple Trim / Gap Close | M | ExtendScript | After cut application, auto-close gaps by rippling subsequent clips. ExtendScript `removeEmptyTrackItems()`. |

**Wave 2 Total: ~17 features, 0 new dependencies, ~20 new routes**

---

## Wave 3: Architecture & Infrastructure

**Goal**: Complete the remaining architectural work that enables heavy AI features in Waves 4-7. These are not user-facing but are prerequisites for scale.

**Timeline**: 6-10 weeks (runs in parallel with Waves 1-2)
**Dependencies**: Internal refactoring

### 3A — Process Isolation for GPU Workers (P0)

The single most important infrastructure change. Every AI feature in Waves 4-7 benefits from this.

| Task | Detail |
|------|--------|
| **Worker pool architecture** | `opencut/workers/` with `WorkerManager`. Workers are separate Python processes per model family (whisper, demucs, realesrgan, depth, generation). |
| **IPC protocol** | Workers communicate via localhost HTTP (minimal Flask on random port) or `multiprocessing.Queue`. Job dispatcher routes by type. |
| **GPU memory management** | Worker reports VRAM on startup. Dispatcher checks available VRAM against model's known requirement before scheduling. Workers exit after 5-min idle to free VRAM. |
| **Graceful degradation** | GPU OOM → specific guidance ("Model needs 4GB VRAM, you have 2GB. Switching to CPU.") → optional CPU re-dispatch. |
| **Model registry** | `models.json` mapping model name → VRAM requirement, download size, expected load time. UI shows this info. |

**Deliverable**: No more OOM crashes from model conflicts. GPU utilization visible in status bar.

### 3B — UXP Full Parity & CEP Migration (P0)

CEP end-of-life is approximately September 2026. UXP must be production-ready before then.

| Task | Detail |
|------|--------|
| **Shared component library** | `extension/shared/` with framework-agnostic components. Both CEP and UXP import from here. Build system outputs two bundles. |
| **Feature registry** | `features.json` defines every feature: id, label, endpoint, params schema, requires. Both panels auto-generate UI from this. Adding a feature = one JSON entry + one backend route. |
| **UXP feature gap closure** | Port remaining ~15% of CEP features to UXP. Mostly: workflow builder, full settings panel, plugin UI. |
| **Native UXP timeline access** | Replace ExtendScript `evalScript()` with direct `premierepro` UXP module for timeline read/write. 10x faster. |
| **Premiere menu integration** | Right-click → "OpenCut: Remove Silence" / "Add Captions" / "Normalize Audio" via UXP API. |
| **CEP deprecation plan** | Mark CEP panel as "legacy" in docs. Freeze CEP feature additions. All new features UXP-only after Wave 3. |

**Deliverable**: UXP panel at 100% parity. CEP can be removed when Adobe enforces it.

### 3C — FastAPI Migration (P3 — Deferred)

Low priority. Flask works fine at current scale. Migrate only if:
- Request validation boilerplate becomes unmanageable (>300 routes)
- WebSocket needs outgrow the current `websockets` library
- Auto-generated OpenAPI docs become essential for plugin developers

If triggered, migrate one blueprint at a time (system → settings → search → nlp → timeline → jobs → captions → audio → video). Pydantic models replace `safe_float()`/`safe_int()` hand-validation.

### 3D — TypeScript Migration (P3 — Incremental)

Continue incremental migration as files are touched. Priority order:
1. API layer (`src/api/types.ts` from OpenAPI schema)
2. Store/state management
3. Tab modules as they're refactored for new features

No dedicated sprint. Piggyback on feature work.

---

## Wave 4: New Feature Domains — Moderate Dependencies

**Goal**: Add new feature domains that require 1-2 new dependencies each but significantly expand OpenCut's capability.

**Timeline**: 6-8 weeks (after Wave 1, can overlap with Wave 3)
**New deps**: 4-6 new pip packages
**New routes**: ~30

### 4A — Privacy & Content Redaction (5 features)

Shared infrastructure: object detection framework, tracking pipeline, audio masking.

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 55.1 | License Plate Detection & Blur | M | `paddleocr` or YOLO plate model | Detect plates per frame → track with IoU → Gaussian blur on tracked regions. |
| 55.3 | Profanity Bleep Automation | S | None (done in Wave 1) | — |
| 55.2 | OCR-Based PII Redaction | L | `paddleocr` (shared with 55.1) | OCR → regex PII patterns (SSN, phone, email, CC) → NER for names → track text regions → blur. |
| 55.4 | Document & Screen Redaction | M | OpenCV (existing) | Edge detection → perspective transform → classify as screen/document/whiteboard → blur surface. |
| 55.5 | Audio Speaker Anonymization | M | Existing (pedalboard) | Diarize → target speaker segments → pitch shift + formant shift or TTS resynthesis. |

**New dependency**: `paddleocr` (or reuse existing Tesseract if sufficient). One dep serves 55.1 + 55.2.

### 4B — Camera & Lens Correction (3 remaining features)

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 52.2 | Rolling Shutter Correction | L | `gyroflow` CLI (subprocess) | Integrate Gyroflow as subprocess with lens profiles. Parse gyro metadata from GoPro/DJI. |
| 13.4 | LOG / Camera Profile Pipeline | M | None | Auto-detect LOG profile from ffprobe metadata → apply bundled technical LUT (free Sony/Canon/Panasonic LUTs). |
| 43.4 | Color Space Auto-Detection | M | None | Read `color_primaries`/`transfer_characteristics` from ffprobe → auto-apply correct input transform. |

**New dependency**: `gyroflow` CLI (optional, subprocess only — not a pip package).

### 4C — Spectral Audio Editing (4 features)

Shared infrastructure: STFT analysis/resynthesis pipeline.

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 56.4 | Room Tone Auto-Generation | M | None (done in Wave 1) | — |
| 56.3 | AI Environmental Noise Classifier | M | `tensorflow-lite` or `onnxruntime` (existing) | YAMNet model (521 sound classes, TFLite). Classify → selective removal via spectral masking. |
| 56.2 | Spectral Repair / Frequency Removal | M | `librosa` (existing) | STFT → identify persistent spectral peaks (hum/buzz) → attenuate → inverse STFT. Auto-detect mode. |
| 56.1 | Visual Spectrogram Editor | L | `librosa` (existing) | FFmpeg `showspectrumpic` or librosa STFT → zoomable canvas in panel → brush tool mask → inverse STFT reconstruction. |

**New dependency**: None if using `onnxruntime` (already installed) for YAMNet. Otherwise `tflite-runtime` (lightweight).

### 4D — Proxy & Media Management (4 features)

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 60.1 | Auto Proxy Generation | L | None | Detect clips >1080p → FFmpeg scale to target res + CRF 28 → store in `~/.opencut/proxies/` with manifest. Background job. |
| 60.2 | Proxy-to-Full-Res Swap on Export | S | None | Query timeline clip paths via ExtendScript → check against proxy manifest → verify originals exist → report. |
| 60.3 | Media Relinking Assistant | M | None | ExtendScript: enumerate offline items. Python: recursive search by filename + size matching. Batch relink UI. |
| 60.4 | Duplicate Media Detection | M | None | File size grouping → partial hash (first+last 64KB) → full hash for matches. Optional pHash for content matches. |

**New dependency**: None.

### 4E — Pro Color Science — First Pass (4 features)

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 13.1 | Real-Time Color Scopes | L | FFmpeg+Pillow | FFmpeg `waveform`, `vectorscope`, `histogram` filters render scope images. Display as image grid in panel. |
| 13.5 | Film Stock Emulation | M | None | Custom 3D LUTs per stock (Kodak/Fuji) + grain overlay + gate weave + halation via blend. Preset package. |
| 13.4 | LOG Camera Profile Pipeline | M | None (listed in 4B) | — |
| 43.1 | ACES Color Pipeline | L | None | ACES IDT/ODT via FFmpeg `colorspace` + `lut3d`. Bundled ACES LUTs (free from AMPAS). |

**New dependency**: None (FFmpeg + bundled LUT files).

**Wave 4 Total: ~18 features (excluding duplicates from Wave 1), 1-2 new deps, ~30 new routes**

---

## Wave 5: AI Dubbing & Voice Translation

**Goal**: Build the end-to-end AI dubbing pipeline — the single highest-value new AI capability. This is what ElevenLabs, HeyGen, and Rask.ai charge $50-100/month for.

**Timeline**: 4-6 weeks (after Wave 3A process isolation is ready)
**Prerequisite**: Wave 3A (GPU process isolation) — dubbing loads multiple large models sequentially
**New deps**: Minimal (leverages existing Chatterbox, Whisper, Demucs, SeamlessM4T)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 62.1 | End-to-End AI Dubbing Pipeline | XL | Transcribe → translate (SeamlessM4T) → voice-clone TTS (Chatterbox) with duration constraints → stem-separate original (Demucs, remove dialogue, keep music/SFX) → mix dubbed dialogue + original music/SFX → export. |
| 62.2 | Isochronous Translation | L | LLM-assisted translation constrained by segment duration. Iterate: translate → estimate TTS duration from syllable count → if too long, ask LLM to rephrase shorter → if too short, expand. Target +-10% of original. |
| 62.3 | Multi-Language Audio Track Management | M | FFmpeg `-map` to mux multiple audio streams with language metadata. Panel UI: track list with language dropdown, add/remove, default flag. Export multi-track MKV/MP4 or per-language files. |
| 62.4 | Emotion-Preserving Voice Translation | L | Extract prosody (F0 contour via librosa, RMS energy, speaking rate) from original → generate TTS with neutral prosody → transfer original prosody shape to dubbed audio via WORLD vocoder or pitch manipulation. |

**Workflow chain**: The dubbing pipeline calls 5 existing modules in sequence. The key new code is the orchestrator (`core/dubbing.py`) and the isochronous translation loop (`core/isochron_translate.py`).

**New dependency**: Potentially `pyworld` for vocoder-based prosody transfer (62.4). Everything else is already installed.

**Wave 5 Total: 4 features, 0-1 new deps, ~8 new routes**

---

## Wave 6: Advanced Professional Features

**Goal**: Deep features for professional editors, colorists, and post-production specialists. These differentiate OpenCut from casual tools.

**Timeline**: 8-12 weeks (can be worked on in parallel tracks)
**New deps**: 2-4

### 6A — Composition & Framing Intelligence (3 features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 61.4 | Saliency-Guided Auto-Crop | M | Face regions (high weight) + motion regions (frame diff) + text regions (OCR) + high-contrast edges → weighted heat map → place crop to maximize saliency. |
| 13.2 | Three-Way Color Wheels | L | SVG color wheel widgets in panel → map wheel positions to FFmpeg `colorbalance` filter values (lift/gamma/gain). Preview via frame extraction. |
| 13.3 | HSL Qualifier / Secondary Correction | L | OpenCV HSV range masking with feathered edges → apply corrections to masked region only → composite. Preview matte in panel. |

### 6B — Pre-Production Tools (4 features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 59.4 | Script-to-Rough-Cut Assembly | XL | Batch transcribe all footage → fuzzy-match transcript segments against script text → rank matches by similarity + audio quality + face visibility → assemble best take per segment as OTIO/Premiere XML. |
| 59.2 | Shot List Generator from Screenplay | M | Parse screenplay format (INT./EXT., ACTION, DIALOGUE) → LLM suggests shot count and camera angles per scene → export as CSV. |
| 59.1 | AI Storyboard Generation from Script | L | Parse script into shots → generate one image per shot via Stable Diffusion or external API → layout as storyboard grid with descriptions → export PDF. |
| 59.3 | Mood Board Generator from Footage | M | Extract keyframes → k-means color clustering → style tags (warm/cold, contrast, saturation) → suggest matching LUTs → compile as visual reference image. |

### 6C — Video Repair & Restoration (3 remaining features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 53.1 | Corrupted File Recovery | M | Detect corruption type (missing moov, truncated stream). For missing moov: untrunc algorithm with reference file. For truncated: `ffmpeg -err_detect ignore_err` salvage. Report recovery stats. |
| 53.4 | SDR-to-HDR Upconversion | L | FFmpeg `zscale` (bt709 → bt2020) + inverse tone mapping. Apply PQ/HLG transfer function. Embed ST.2086 metadata. |
| 13.6 | Power Windows with Tracking | L | Shape masks (circle, rect, polygon) in panel → track via MediaPipe (face) or SAM2 (object) → apply corrections inside/outside mask via per-frame FFmpeg filter. |

### 6D — Forensic & Legal (3 features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 35.1 | Selective Redaction Tool | M | Click-to-select regions in preview → track across frames → blur/pixelate/black. Export redaction log for audit trail. |
| 35.2 | Chain of Custody Metadata | S | SHA-256 hash of original + all operations applied + timestamps → embed as metadata or export as sidecar JSON. |
| 35.3 | Forensic Enhancement | M | Stabilize + denoise + sharpen + contrast stretch + frame interpolation for low-quality surveillance footage. |

### 6E — Accessibility & Compliance (3 features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 20.1 | Caption Compliance Checker | M | Parse captions → check against rulesets (Netflix <=42 CPL, FCC <=32 CPL, BBC <=160 WPM, min duration, CPS). Flag violations with auto-fix suggestions. |
| 20.2 | Audio Description Track Generator | L | Detect dialogue pauses (existing VAD) → extract key frames during pauses → describe via LLM vision → TTS synthesis → mix into gaps → export as AD track. |
| 27.1 | C2PA Content Credentials | M | Embed Content Authenticity Initiative metadata (origin, edit history, AI disclosure). `c2pa-python` library. |

**Wave 6 Total: ~16 features, 2-3 new deps, ~25 new routes**

---

## Wave 7: AI Generation, 360, & Emerging Tech

**Goal**: Forward-looking AI capabilities and niche professional features. These are differentiators, not table-stakes.

**Timeline**: Ongoing (8-16 weeks, lowest priority)
**New deps**: Several (heavy AI models)
**Prerequisite**: Wave 3A (GPU process isolation) essential for multiple large models

### 7A — AI Video Generation & Synthesis (5 features)

| # | Feature | Effort | New Dep | Detail |
|---|---------|--------|---------|--------|
| 54.2 | Image-to-Video Animation | L | `diffusers` (existing) | SVD or CogVideoX with image conditioning → 2-6s clip from still image + motion prompt. |
| 54.5 | AI Background Replacement | L | `diffusers` (existing) | RVM foreground extraction + Stable Diffusion background from text prompt → composite. |
| 54.1 | AI Outpainting / Frame Extension | L | `diffusers` (existing) | Extend canvas to target aspect ratio → inpaint borders via ProPainter or SD. Keyframe-based for temporal consistency. |
| 54.3 | AI Scene Extension | XL | `diffusers` (existing) | Feed last N frames to video prediction model → generate continuation. Best for static scenes. |
| 21.1 | Multimodal Timeline Copilot | XL | LLM API (existing) | Chat interface backed by multimodal AI that sees video + audio + transcript. Navigate, select, edit via natural language. |

### 7B — 360 / VR / Immersive (4 features)

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 51.2 | Equirectangular to Flat Projection | M | FFmpeg `v360` filter. Keyframeable yaw/pitch/roll for virtual camera paths. |
| 51.3 | FOV Region Extraction from 360 | M | Face detection in equirectangular space → per-speaker flat extraction with smooth tracking → multicam XML. |
| 51.1 | 360 Video Stabilization | L | Parse gyro metadata (GoPro GPMF, Insta360) → apply inverse rotation via FFmpeg `v360`. |
| 51.4 | Spatial Audio Alignment | L | Map speaker positions from face detection → route mono dialogue to correct ambisonic channel. First-order ambisonics output. |

### 7C — Niche Professional Features

| # | Feature | Effort | Detail |
|---|---------|--------|--------|
| 41.1 | DJI Telemetry Data Overlay | M | Parse DJI SRT files → render altitude, speed, GPS, battery as configurable overlay. |
| 42.1 | Image Sequence Import & Assembly | M | Import folder of images (TIFF, EXR, DPX, PNG) → assemble as video with configurable FPS and transitions. |
| 39.1 | Elgato Stream Deck Integration | M | WebSocket/HTTP listener for Stream Deck commands → map buttons to OpenCut operations. Plugin for Stream Deck SDK. |
| 12.1 | Gaming Highlight / Kill Detection | L | Multi-signal fusion: audio peaks + motion intensity + optional OCR on kill feed → score segments → extract top clips. |
| 33.1 | Lecture Recording Auto-Split | M | Scene detection + chapter generation → split lecture by topic → generate per-topic clips with title cards. |
| 46.1 | Multi-Step Autonomous Editing Agent | XL | LLM plans editing steps from high-level instruction → executes via OpenCut API → iterates on result quality. Full agent loop with human review checkpoints. |

**Wave 7 Total: ~15 features, 0-2 new deps (most already installed), ~20 new routes**

---

## Implementation Order & Dependencies

```
Wave 1 (Quick Wins)          |=============================|
Wave 2 (Pipelines)           |=======================|
Wave 3A (GPU Isolation)           |========================|
Wave 3B (UXP Parity)              |=====================|
Wave 4 (New Domains)                   |========================|
Wave 5 (AI Dubbing)                         |================|
Wave 6 (Pro Features)                            |===========================|
Wave 7 (Emerging)                                      |=========================>
                              Wk 1    Wk 6    Wk 12   Wk 18   Wk 24   Wk 30+
```

**Critical path**: Wave 3A (GPU isolation) must land before Waves 5 and 7A (heavy AI features).

**Parallel tracks**:
- Wave 1 + Wave 2 can run simultaneously (different developers or even same developer — no conflicts)
- Wave 3A + Wave 3B are independent
- Wave 4 can start as soon as Wave 1 is done (shares no code)
- Wave 6 features are independent of each other (can be cherry-picked)

---

## Route Growth Projection

| Milestone | Routes | Core Modules | Tests (est.) |
|-----------|--------|-------------|-------------|
| Current (v1.9.26) | 254 | 68 | 867 |
| After Wave 1 | ~290 | ~78 | ~1,050 |
| After Wave 2 | ~310 | ~85 | ~1,200 |
| After Wave 4 | ~340 | ~95 | ~1,400 |
| After Wave 5 | ~348 | ~99 | ~1,500 |
| After Wave 6 | ~373 | ~110 | ~1,700 |
| After Wave 7 | ~393 | ~120 | ~1,900 |

---

## Priority Matrix (Updated)

### P0 — Critical (Do First)

| # | Feature | Wave | Effort | Why Critical |
|---|---------|------|--------|-------------|
| 3A | GPU Process Isolation | 3 | XL | Prerequisite for all heavy AI features. Eliminates OOM crashes. |
| 3B | UXP Full Parity | 3 | XL | CEP end-of-life ~Sept 2026. Must be ready before then. |
| 32.1 | Hardware-Accelerated Encoding | 1 | M | Users with GPUs expect NVENC/QSV. Every other tool has this. |
| 58.1 | Long-Form to Multi-Short Extraction | 2 | L | $228/year competitor (Opus Clip). Highest-value pipeline. |

### P1 — High Impact (Next Priority)

| # | Feature | Wave | Effort | Why High Impact |
|---|---------|------|--------|----------------|
| 62.1 | End-to-End AI Dubbing | 5 | XL | $50-100/month competitor category. Uses all existing modules. |
| 57.1 | Split-Screen Templates | 1 | M | CapCut/iMovie table-stakes. Massive content category. |
| 55.1 | License Plate Blur | 4 | M | Privacy law compliance. Every content creator needs this. |
| 55.3 | Profanity Bleep | 1 | S | Broadcast requirement. Trivial to build. |
| 53.2 | Adaptive Deinterlacing | 1 | S | Every NLE has this. Legacy footage is common. |
| 52.1 | Lens Distortion Correction | 1 | M | Standard camera correction. lensfun database is free. |
| 56.4 | Room Tone Auto-Generation | 1 | M | iZotope RX feature. Makes silence removal sound professional. |
| 60.1 | Auto Proxy Generation | 4 | L | Premiere/Resolve/FCPX all have this. 4K editing prerequisite. |
| 61.2 | Shot Type Classification | 1 | M | Enables intelligent editing decisions and footage search. |
| 45.2 | AV1 Encoding | 1 | M | Modern codec with 30-50% bitrate savings. YouTube prefers it. |
| 45.1 | ProRes Export (Windows) | 1 | M | Professional delivery format. Resolve offers this on Windows. |
| 13.1 | Real-Time Color Scopes | 6 | L | Every colorist needs scopes. Color tools are blind without them. |
| 59.4 | Script-to-Rough-Cut | 6 | XL | Biggest time saver in post-production. Avid ScriptSync competitor. |
| 20.1 | Caption Compliance Checker | 6 | M | Netflix/FCC/BBC requirements. Prevents platform rejection. |
| 24.1 | Shot-Change-Aware Subtitle Timing | 2 | M | Broadcast QC standard. Simple post-processing. |

### P2 — Valuable (Scheduled)

All remaining Wave 1-6 features not listed above (~60 features).

### P3 — Future (Backlog)

All Wave 7 features + FastAPI migration + TypeScript + niche items (~40 features).

---

## Success Metrics (Updated)

| Metric | v1.9.26 (Current) | After Waves 1-2 | After Waves 3-5 | After Waves 6-7 |
|--------|--------------------|-----------------|-----------------|-----------------|
| API routes | 254 | ~310 | ~348 | ~393 |
| Core modules | 68 | ~85 | ~99 | ~120 |
| Tests | 867 | ~1,200 | ~1,500 | ~1,900 |
| Time to first useful action | ~30s (workflow) | ~15s (pipeline) | ~10s (context + agent) | ~5s (copilot) |
| Install success rate | ~90% | ~92% | ~95% (isolation) | ~99% (Docker) |
| Competitor features covered | ~60% | ~75% | ~85% | ~95% |
| Features available in UXP | ~85% | ~90% | 100% | 100% |
| New deps added | 0 | 0 | 1-2 | 4-6 |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| CEP deprecation before UXP ready | High | Wave 3B is P0. Start immediately. Freeze CEP feature additions. |
| GPU process isolation complexity | High | Start with simple subprocess model. Upgrade to full worker pool later. Ship incremental improvements. |
| AI model download sizes | Medium | Models are optional. Clear size warnings in UI. Pre-download in installer. Offer cloud API fallback where possible. |
| Too many features → quality regression | High | Every new feature gets a smoke test before merge. Ruff lint on CI. No feature without a test. |
| Dependency conflicts from new packages | Medium | One new dep per feature max. Pin versions. Test in isolated venv before adding to `pyproject.toml`. |
| Scope creep from 302-feature plan | Medium | Waves are independently shippable. Only commit to one wave at a time. Review and reprioritize between waves. |

---

*This roadmap should be reviewed at the start of each wave and reprioritized based on user feedback, competitive landscape changes, and lessons learned from the previous wave.*
