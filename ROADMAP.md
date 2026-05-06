# OpenCut — Implementation Roadmap

**Version**: 4.0
**Updated**: 2026-05-09
**Baseline**: v1.32.0 (1,275 routes, 99 blueprints, 460+ core modules, 7,551 tests, light theme + premium UX shipped)
**Feature Plan**: 302 features across 62 categories (see `features.md`)

> **⚡ Active work** lives in [ROADMAP-NEXT.md](ROADMAP-NEXT.md) (Waves A–K, mostly shipped through v1.28.x)
> and the wave sections in this file (L through S). Wave R (v1.52→v1.55) is the most recent committed plan.
> Wave S (v1.56→v1.58) below is the post-May-2026 OSS research pass.
> Shipped history is archived in [ROADMAP-COMPLETED.md](ROADMAP-COMPLETED.md).

---

## Guiding Principles

1. **Never break what works** — Every wave ships a working product. No "rewrite everything then test."
2. **Incremental migration** — New code coexists with old. Feature flags gate rollout. Old paths removed only after new paths are proven.
3. **User-facing value first** — Each wave delivers visible improvements, not just internal refactors.
4. **Measure before optimizing** — Add telemetry/logging before assuming bottlenecks.
5. **Shared infrastructure first** — When multiple features need the same foundation (e.g., object tracking, spectral analysis), build the foundation once, then fan out.
6. **One new dependency per feature maximum** — Avoid dep explosion. Prefer extending existing deps (OpenCV, FFmpeg, Pillow) over adding new ones.

---

> Completed work (v1.0 - v1.9.26) moved to ROADMAP-COMPLETED.md.

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

---

## Research & Strategic Gaps (Auto-Generated Analysis)

**Auditor**: Principal Systems Architect analysis
**Date**: 2026-04-14
**Baseline**: v1.14.0 (1,088 routes, 408 core modules, 83 blueprints, 87 test files, 6,925 tests)
**Method**: Full codebase scan, security audit, architecture bottleneck analysis, test/CI pipeline review

> **Context**: This roadmap was authored at v1.9.26 (254 routes, 68 modules). The codebase has since grown **4.3x in routes** and **6x in modules**. The Wave 1-7 structure and growth projections are now obsolete — the "After Wave 7" target of ~393 routes was surpassed at v1.10.5. This analysis identifies the gaps that the rapid feature expansion has opened.

---

### HIGH Priority — Blocking Issues

- **GPU process isolation is still unimplemented (Wave 3A).** This was marked P0 and remains the single most critical infrastructure gap. `MAX_CONCURRENT_JOBS = 10` in `opencut/jobs.py:42` allows 10 simultaneous ML model loads into VRAM. PyTorch models (Demucs, Real-ESRGAN, InsightFace, SAM2, CLIP, etc.) each consume 500MB-4GB VRAM. Concurrent loads **will** OOM on consumer GPUs. No memory reservation, no model-aware scheduling, no graceful degradation path exists. **Every AI feature added since v1.10 has widened this gap.** The 408-module codebase now has 40+ modules that load GPU models — 6x more than when Wave 3A was planned.
  - *Recommended action*: Implement a GPU memory budget system immediately. At minimum: reduce `MAX_CONCURRENT_JOBS` to 3 for GPU-tagged routes, add a `@gpu_exclusive` decorator that serializes GPU model access behind a semaphore, and report VRAM usage in `/system/status`.

- **Rate limiting covers 4% of async routes.** Security audit found 597 async route handlers but only 23 rate-limit calls. The `require_rate_limit()` decorator exists and works, but was only applied to model-install and a handful of AI routes. All 574 unprotected async routes accept concurrent requests limited only by `MAX_CONCURRENT_JOBS=10`. A single client can trivially exhaust all 10 job slots with expensive operations (batch rendering, video processing, ML inference), starving other requests.
  - *Recommended action*: Introduce rate-limit categories (`gpu_heavy`, `cpu_heavy`, `io_bound`, `light`) and apply to all async routes. GPU-heavy operations should share a pool of 2-3 concurrent slots. CPU-heavy should cap at 4-6.

- **Test coverage is broad but shallow.** 87 test files exist with 6,925 test functions, but the architecture audit reveals 97% of the 408 core modules lack dedicated behavioral tests — they're only exercised indirectly through route smoke tests. The smoke tests in `test_route_smoke.py` use broad status code assertions like `assert resp.status_code in (200, 400, 429)` which pass regardless of whether the feature works correctly. CI enforces only 50% line coverage (`--cov-fail-under=50` in `build.yml`), which is insufficient for a codebase of this size and complexity.
  - *Recommended action*: Raise CI coverage threshold to 65% (target 80% over 2 sprints). Add schema validation for route responses (JSON structure, not just "is JSON"). Prioritize integration tests for the 40 GPU-model-loading modules — these are the highest-risk code paths with the least coverage.

- **Roadmap growth projections are 3x out of date.** The "Route Growth Projection" table estimates 393 routes after all 7 waves. Actual count is 1,088 — a 2.8x overshoot. The "Success Metrics" table, "Completed Work" section, and wave feature lists don't reflect v1.10-v1.14 additions (categories 63-77, 155 new core modules, 20 new route blueprints). The roadmap should be rebased to reflect current reality so it can be trusted for planning.
  - *Recommended action*: Rebase all tables to v1.14.0 actuals. Mark Wave 1-2 features that were implemented in v1.10-v1.14 as DONE. Update dependency legend with new module families. Revise success metrics to reflect 1,088-route baseline.

---

### MEDIUM Priority — Technical Debt & Infrastructure

- **`helpers.py` is a god module (350 imports).** Every core module and most route files import from `opencut/helpers.py`. It contains FFmpeg execution, video probing, output path logic, temp file cleanup, package installation, and progress utilities — responsibilities that span 6+ concerns. This makes it a merge conflict magnet, impossible to test in isolation, and a startup bottleneck (every import chain pulls in the entire module).
  - *Recommended action*: Decompose into `helpers/ffmpeg.py`, `helpers/video_probe.py`, `helpers/paths.py`, `helpers/cleanup.py`, `helpers/packages.py`. Re-export from `helpers/__init__.py` for backward compat. Do this incrementally during feature work, not as a dedicated refactor sprint.

- **UXP migration has 5 months remaining.** CEP end-of-life is approximately September 2026. The roadmap states UXP is at ~85% feature parity (Wave 3B). The UXP panel (`extension/com.opencut.uxp/`) has 7 tabs vs. CEP's 8, and the UXP main.js is 1,523 lines vs. CEP's 7,730 — indicating significant feature gaps in the frontend. No UXP-specific tests exist in CI. The CEP panel continues to receive features (v1.14.0 version bumps touch CEP files), violating the roadmap's "freeze CEP feature additions" directive.
  - *Recommended action*: Audit UXP vs. CEP parity at the feature level (not tab level). Add UXP smoke test to CI. Enforce CEP freeze — new frontend features go to UXP only.

- **No type checking in CI.** 523 Python files with no mypy or pyright enforcement. Type errors (None where str expected, dict where dataclass expected, wrong callback signature) are caught at runtime — if at all. The `on_progress` callback pattern is already documented in CLAUDE.md as a gotcha (core modules call with 1 arg, routes define closures with 2 args), which is exactly the class of bug static typing catches.
  - *Recommended action*: Add `mypy --ignore-missing-imports opencut/` to CI. Start with `--no-strict` and fix errors incrementally. Target: 0 type errors in `opencut/core/` within 2 sprints.

- **Untracked subprocesses can orphan on cancel.** The `@async_job` decorator registers the job's main thread for cancellation, and `_register_job_process()` tracks Popen handles. But 158 subprocess calls across core modules call `subprocess.run()` directly — these finish synchronously within the job thread but can't be interrupted mid-execution. If a user cancels a job while FFmpeg is mid-render (a 30-minute operation), the FFmpeg process runs to completion even though the job is marked cancelled. The process exit code is then silently discarded.
  - *Recommended action*: Wrap long-running subprocess calls in a pattern that checks `job_cancelled` flag and sends SIGTERM to the child process. Alternatively, refactor `run_ffmpeg()` in helpers.py to accept a `job_id` parameter and auto-register the Popen for cancellation.

- **No security scanning in CI pipeline.** The `build.yml` workflow runs ruff lint and pytest but has no security tooling: no bandit (Python security linter), no CodeQL (GitHub's code scanning), no dependabot/Snyk (dependency vulnerability scanning), no SBOM generation. For a project that executes FFmpeg subprocesses, runs `pip install` at runtime via `safe_pip_install()`, and loads ML models from external sources, this is a meaningful gap.
  - *Recommended action*: Add `bandit -r opencut/ -ll` to CI (catches high-confidence security issues). Enable GitHub Dependabot for dependency alerts (zero-effort, just add `dependabot.yml`). Add CodeQL for deeper analysis.

- **Temp file accumulation under load.** 93 modules create temp files via `tempfile.mkstemp()` or `NamedTemporaryFile()`. The deferred cleanup mechanism (`_schedule_temp_cleanup()` in helpers.py) uses a 5-second delay with 3 retries. Under concurrent load (10 video processing jobs), this means hundreds of multi-GB temp files (intermediate FFmpeg outputs, extracted frames, model outputs) can accumulate before cleanup fires. No disk quota, no max-temp-size check, no cleanup-on-startup sweep.
  - *Recommended action*: Add a startup sweep of `tempfile.gettempdir()` for stale `opencut_*` temp files. Add a periodic (60s) background cleanup for files older than 10 minutes. Log temp disk usage in `/system/status`.

- **25+ tests use `time.sleep()` creating flaky CI.** Tests in `test_batch_executor.py`, `test_batch_parallel.py`, `test_boolean_coercion.py`, `test_integration_ffmpeg.py`, and `test_preview_realtime.py` contain sleeps ranging from 10ms to 500ms. These are timing-dependent and will intermittently fail on slow CI runners, Windows VMs, or under load. Additionally, `test_solver_agent.py` uses `random.seed(42)` but other tests don't seed, introducing non-determinism.
  - *Recommended action*: Replace `time.sleep()` in tests with event-based synchronization (threading.Event, condition variables). For async result tests, poll with timeout rather than fixed sleep. Audit and seed all random usage.

---

### LOW Priority — Future Investment

- **No auto-generated API documentation.** With 1,088 routes across 83 blueprints, there is no OpenAPI/Swagger spec, no auto-generated endpoint catalog, and no machine-readable API schema. Plugin developers and external integrators must read route source code. The roadmap's Wave 3C notes FastAPI migration (which brings auto-generated OpenAPI) but defers it. The original trigger — "if >300 routes" — was passed long ago.
  - *Recommended action*: Generate an OpenAPI spec from Flask routes using `flask-smorest` or `apispec` without migrating to FastAPI. Serve Swagger UI at `/api/docs` for development mode only. This is a 1-day effort that unlocks plugin ecosystem development.

- **Blueprint registration is sequential and eager.** `register_blueprints()` in `routes/__init__.py` performs 83 sequential `import` statements at app startup. Each import may trigger module-level initialization (cache setup, constant computation, availability checks). Measured impact is 2-5 seconds on startup — not a production issue but noticeable during development when the server auto-restarts on file changes.
  - *Recommended action*: No immediate action needed. If dev-cycle time becomes a complaint, implement lazy blueprint registration (register on first request to URL prefix).

- **No performance regression detection.** No benchmarks, no load tests, no response-time tracking in CI. With 1,088 routes and 408 modules, a single change to `helpers.py` or `jobs.py` could degrade performance across hundreds of endpoints with no visibility.
  - *Recommended action*: Add a simple benchmark suite (10 representative endpoints, measure p50/p95 response time) that runs in CI and fails on >20% regression. Use `pytest-benchmark` or custom timing.

- **Missing production governance files.** No `SECURITY.md` (vulnerability disclosure process), no `CODE_OF_CONDUCT.md`, no `CONTRIBUTING.md` with architecture guide, no SBOM (software bill of materials). For an open-source project with 408 modules and ML model downloads, these are expected by enterprise adopters.
  - *Recommended action*: Add `SECURITY.md` with disclosure process and supported-versions table. Generate SBOM from `pyproject.toml` deps.

- **FastAPI migration trigger has been reached.** The roadmap defers FastAPI migration until ">300 routes" with the rationale that validation boilerplate would become unmanageable. Current state: 1,088 routes, 879 mutation endpoints, manual `safe_float()`/`safe_int()`/`safe_bool()` validation in every handler. Pydantic models would eliminate ~60% of per-route validation boilerplate and provide automatic request/response schema generation.
  - *Recommended action*: This remains low priority because Flask works and migration risk is high with 83 blueprints. However, the original deferral rationale no longer holds. If a major refactor is planned (e.g., helpers.py decomposition), consider migrating 1-2 blueprints to FastAPI as a proof-of-concept to measure the cost/benefit.

---

### Summary Matrix

| Finding | Priority | Effort | Impact | Status |
|---------|----------|--------|--------|--------|
| GPU process isolation (Wave 3A) | HIGH | XL | Eliminates OOM crashes | Not started |
| Rate limiting expansion | HIGH | M | Prevents DoS / resource exhaustion | 4% coverage |
| Test depth & coverage threshold | HIGH | L | Catches regressions before release | 50% threshold |
| Roadmap rebase to v1.14.0 | HIGH | S | Accurate planning | Stale since v1.9.26 |
| helpers.py decomposition | MEDIUM | M | Reduces coupling, merge conflicts | 350 imports |
| UXP full parity (Wave 3B) | MEDIUM | L | CEP EOL Sept 2026 | ~85% parity |
| Type checking in CI | MEDIUM | M | Catches type bugs statically | Not started |
| Subprocess cancellation | MEDIUM | M | Clean job cancel behavior | 158 untracked calls |
| Security scanning in CI | MEDIUM | S | Catches vulnerabilities | Not started |
| Temp file disk management | MEDIUM | S | Prevents disk exhaustion | No quota |
| Flaky test elimination | MEDIUM | S | Reliable CI | 25+ sleep-based tests |
| Auto-generated API docs | LOW | S | Enables plugin ecosystem | No spec exists |
| Performance benchmarks in CI | LOW | M | Detects regressions | Not started |
| Production governance files | LOW | S | Enterprise readiness | Missing |
| FastAPI migration evaluation | LOW | XL | Reduces boilerplate at scale | Deferred |

## Open-Source Research (Round 2)

### Related OSS Projects
- **hetpatel-11/Adobe_Premiere_Pro_MCP** — https://github.com/hetpatel-11/Adobe_Premiere_Pro_MCP — MCP server bridging AI assistants (Claude/Codex) to Premiere Pro's scripting engine; CEP panel + experimental UXP plugin; covers project, ingest, sequence, timeline, transitions, effects, keyframes, metadata, export
- **sebinside/PremiereRemote** — https://github.com/sebinside/PremiereRemote — local HTTP/WebSocket server inside Premiere that lets external tools trigger ExtendScript; foundation for AutoHotkey and AI integrations
- **cameron-astor/jumpcut** — https://github.com/cameron-astor/jumpcut — jumpcut plugin that uses an external pyinstaller binary for waveform analysis (because CEP can't read audio); exact architectural model OpenCut uses
- **Adobe-CEP/Samples** — https://github.com/Adobe-CEP/Samples — official reference implementations of CEP extensions across Adobe apps
- **adobe-cep organization** — https://github.com/adobe-cep — all official samples, CEP debug and packaging tooling
- **SLNimesh/pro-console** — https://github.com/SLNimesh/pro-console — fx-console-style quick effect browser for Premiere; good UX reference

### Features to Borrow
- MCP server bridge so OpenCut can be driven by any MCP-speaking AI client (Premiere Pro MCP)
- Local HTTP + WebSocket trigger surface so AutoHotkey, Stream Deck, shell scripts can call OpenCut actions (PremiereRemote)
- External-binary pattern for CPU-heavy work (silence detection, OCR, transcription) — keep CEP panel thin, ship signed helper EXE (jumpcut)
- UXP migration plan: track which operations work in UXP today, feature-flag them in the extension, keep CEP fallback until parity (Premiere MCP tracks both)
- Signed-bundle distribution with ZXP/ZIP + installer, and auto-update via version manifest (Adobe CEP packaging)
- "Effect console" quick palette: Ctrl-Space to fuzzy-search effects and presets (pro-console)
- Natural-language command-line inside the panel that calls the MCP server locally ("trim last 5 seconds of selected", "add 30-frame dissolve") (Premiere MCP)
- ExtendScript library of composable helpers: importClips, insertAtPlayhead, applyEffect with sane defaults (PremiereRemote has a foundation)
- Debug mode detector that warns the user once and writes PlayerDebugMode reg key only on user confirm (CEP standard)
- Ingest workflow: watch folder, auto-import, tag by filename pattern, route to sequence bins (community extensions)

### Patterns & Architectures Worth Studying
- CEP panel as UI + local HTTP server for out-of-process work — keeps ExtendScript minimal, pushes heavy lifting to Python/Node (PremiereRemote, jumpcut)
- MCP protocol as the AI ↔ app boundary, avoiding custom chat UI and letting users choose their client (Premiere MCP)
- External-helper-binary pattern: ship a PyInstaller EXE alongside the .zxp and call it from ExtendScript via file.exec; remember the Windows fork-bomb requires multiprocessing.freeze_support() (project rule)
- UXP-CEP dual-ship: feature detection on load, choose the best API per operation (Premiere MCP)
- Audio-waveform out-of-band fetch: call ffmpeg to decode, analyze in helper, return JSON of silence/markers (jumpcut)

## Implementation Deep Dive (Round 3)

### Reference Implementations to Study
- **Adobe-CEP/Samples/PProPanel** — https://github.com/Adobe-CEP/Samples/tree/master/PProPanel — the only Adobe-maintained reference covering the full Premiere Pro ExtendScript surface (sequence, clip, marker, metadata, import); every new panel action should start from the matching method here
- **Adobe-CEP/CEP-Resources CEP 10.x Cookbook** — https://github.com/Adobe-CEP/CEP-Resources/blob/master/CEP_10.x/Documentation/CEP%2010.0%20HTML%20Extension%20Cookbook.md — manifest.xml schema, Node enablement flag, debug `.debug` file format, CSXS event round-trip
- **Breakthrough/PySceneDetect** — https://github.com/Breakthrough/PySceneDetect — BSD-3 scene-cut library; `ContentDetector`, `AdaptiveDetector`, `ThresholdDetector`; timecode output directly compatible with Premiere sequence markers
- **geerlingguy/final-cut-it-out** — https://github.com/geerlingguy/final-cut-it-out — silencedetect -> XML edit pattern; adapt the parse-silence-ranges step for Premiere .prproj / sequence-marker output
- **mifi/lossless-cut** — https://github.com/mifi/lossless-cut — GPLv2 reference for FFmpeg-driven lossless cut planning + silence/black-scene detection UX; great source for silence threshold/default values to preseed
- **hyperbrew/bolt-cep** — https://github.com/hyperbrew/bolt-cep — modern React/Vue/Svelte CEP scaffold with TypeScript and hot reload; consider adopting for future panel iterations to shed jQuery
- **alphacep/vosk-api** — https://github.com/alphacep/vosk-api — offline STT; small (50MB) and large (1.8GB) English models; use for "cut silence by filler-word" beyond raw silencedetect
- **openai/whisper** (or faster-whisper ctranslate2) — https://github.com/SYSTRAN/faster-whisper — GPU-optional transcript-driven cut planning; produces word-level timestamps needed for caption-accurate trims

### Known Pitfalls from Similar Projects
- **evalScript is async with a string-only result** — https://github.com/Adobe-CEP/CEP-Resources/blob/master/CEP_10.x/Documentation/CEP%2010.0%20HTML%20Extension%20Cookbook.md — results come back as a string later, not synchronously; JSON-encode all payloads both directions and time out at 30s
- **Node integration disabled by default** — Cookbook above — add `<CEFCommandLine><Parameter>--enable-nodejs</Parameter></CEFCommandLine>` to manifest.xml or `require('child_process')` throws; also must sign/self-sign to load unsigned extensions (`PlayerDebugMode=1`)
- **ExtendScript is ES3** — https://community.adobe.com/t5/after-effects-discussions/how-extendscript-cep-actually-works-on-the-back-end/m-p/14354578 — no `const`, no arrow functions, no Promise; transpile from TS with @types/extendscript or ship ES3 hand-written
- **Vulcan async queue reorders calls** — same community post — concurrent evalScript calls from the panel can return out-of-order; queue them client-side with a single in-flight request or you get dropped-frame metadata
- **CEP is deprecated, UXP is the future** — CEP Cookbook — Premiere hasn't shipped UXP panels yet but is on the roadmap; keep the ExtendScript layer thin so a UXP port is feasible
- **PySceneDetect memory with HEVC** — https://github.com/Breakthrough/PySceneDetect/issues?q=hevc — HEVC decode via OpenCV may balloon RAM on long files; use `--backend pyav` and stream rather than load
- **silencedetect dB threshold false positives** — https://dev.to/dak425/automatically-trim-silence-from-video-with-ffmpeg-and-python-2kol — `-35dB` is too aggressive for quiet rooms; start at `-30dB, d=400ms` and tune per-clip
- **Flask single-threaded by default** — https://flask.palletsprojects.com/en/stable/deploying/ — Python backend must run under `waitress` or `gunicorn` on Windows or long renders block the panel's next request

### Library Integration Checklist
- **CEP 12** target (manifest `<RequiredRuntime>` CSXS 12.0) — covers Premiere 2024/2025/2026; older panels stay on CSXS 10; gotcha: signed extensions required unless user sets `PlayerDebugMode=1` (HKCU `Software\Adobe\CSXS.12`)
- **CSInterface.js** (pinned version from `Adobe-CEP/CEP-Resources/CEP_12.x/CSInterface.js`) — vendor rather than npm to guarantee compatibility; gotcha: don't mix CSInterface versions between panels in the same extension
- **Flask 3.0+** + **waitress 3.0** for Windows service mode; listen on `127.0.0.1:0` (OS-assigned port) and pass the port to the panel via `CSEvent` — hardcoded ports conflict with other users' panels
- **PySceneDetect 0.6.4+** — `pip install scenedetect[opencv]`; use `detect('in.mp4', AdaptiveDetector(adaptive_threshold=3.0), show_progress=False)` and emit `(start, end)` as `FrameTimecode.get_timecode()` strings; gotcha: Windows path backslashes pass through OK but Premiere `importFile` wants forward slashes
- **FFmpeg 7.1 static builds** — https://www.gyan.dev/ffmpeg/builds/ — bundle in `server/bin/ffmpeg.exe`; set `FFMPEG_BINARY` env var for PySceneDetect and moviepy; gotcha: HEVC hardware decode via `-hwaccel d3d11va` only works when Premiere isn't currently holding the decoder
- **moviepy 2.1+** — for silence-based trimming with audio; gotcha: moviepy 2.x broke the 1.x API (`VideoFileClip.subclip` -> `subclipped`); pin exactly
- **Vosk 0.3.45** + English small model (50MB) — offline STT; gotcha: streaming API buffers must be 16kHz mono PCM — transcode via `ffmpeg -ar 16000 -ac 1 -f s16le` before feeding
- **faster-whisper 1.0+** with CTranslate2 — for higher-accuracy transcript cuts; gotcha: CPU int8 is ~3x slower than GPU fp16 but doesn't require CUDA DLLs — default to CPU, let power users enable CUDA
- **PyInstaller 6.10** with the `multiprocessing.freeze_support()` guard (see user global instructions) — every Python subprocess spawned by the panel must short-circuit on `sys.frozen` or you get the Windows fork-bomb

---

# Wave L — Agent Interface + Creative Intelligence (v1.33.0 → v1.35.0)

**Updated**: 2026-04-17  
**Baseline**: v1.32.0 (1,335 routes, ~360 core modules, 7,551+ tests)  
**Research pass**: April 2026 competitive audit + fresh GitHub OSS survey (see §L-OSS below)

This wave synthesises:
1. Gaps identified in `research.md` (§1–§8, April 2026)
2. Newly discovered OSS tools not yet referenced in any prior OpenCut roadmap document
3. Promotion of five Wave K stubs to full implementation

All guiding principles from ROADMAP-NEXT.md apply: never break what works, one new required dep per feature maximum, permissive licences only, match existing patterns (`check_X_available()`, `@async_job`, queue allowlist).

---

## Wave L1 — UX & Agent Surface (v1.33.0)

**Goal**: Make OpenCut scriptable, agent-native, and catch the remaining UX gaps that competing tools exploit in daily use.  
**New required deps**: `mcp[cli]` (MCP server), `transformers` already present (caption translate reuses it), `flask-sse` or stdlib `queue.Queue` (SSE)  
**New routes**: ~22

### Tier 1 — Immediate User Impact

| # | Feature | Route(s) | Module | Dep | Effort | Licence |
|---|---------|----------|--------|-----|--------|---------|
| L1.1 | **SSE job progress streaming** — server-sent events endpoint for any async job; replace polling `GET /jobs/{id}/status` with a streaming `text/event-stream` feed. Emits `{pct, msg, eta_s}`. Improves perceived performance for long encodes / AI inference. | `GET /jobs/{id}/progress/stream` | `core/job_sse.py` | stdlib `queue.Queue` + Flask `Response(stream_with_context)` | S | — |
| L1.2 | **Caption translation** — translate existing `.srt`/`.vtt` captions (from any pipeline) to any of 200 languages via NLLB-200 (1.3B, Apache-2). Closes the Captions.ai and SubMagic multilingual gap. `POST` accepts `{ path, src_lang, tgt_lang[] }`, returns translated SRT paths. | `POST /captions/translate`, `GET /captions/translate/languages` | `core/caption_translate.py` | `transformers` (already installed, NLLB-200 weights ~5 GB; lazy-load) | M | Apache-2 (NLLB-200), MIT (sentencepiece) |
| L1.3 | **MCP server interface** — expose all 1,335 OpenCut routes as MCP tools so any MCP-speaking AI client (Claude Desktop, Cline, Cursor, Codex) can drive OpenCut without a custom chat UI. Auto-generates tool schemas from Flask route docstrings. `opencut/mcp_server.py` is the entry point; `python -m opencut.mcp_server` launches it. Closes the Crayotter / Premiere-MCP agent-native gap. | `opencut/mcp_server.py` (standalone process, exposes stdio MCP transport) | `opencut/mcp_server.py` | `mcp[cli]` ≥1.0 (MIT) | M | MIT |
| L1.4 | **Upscaling hub dispatcher** — single smart route `POST /video/upscale/smart` picks the best available upscaler (RealESRGAN → BSRGAN fallback → ffmpeg lanczos emergency) based on content type (`face`, `natural`, `animation`) detected from the first 8 frames via CLIP-IQA+. UX consolidation; no new model downloads required. | `POST /video/upscale/smart`, `GET /video/upscale/smart/info` | `core/upscale_hub.py` | Pipeline (existing: RealESRGAN, BSRGAN, CLIP-IQA+, FFmpeg) | S | — |
| L1.5 | **ElevenLabs TTS cloud backend** — cloud fallback for users whose GPU can't run local TTS (MaskGCT/CosyVoice2/F5-TTS). `POST /audio/tts/elevenlabs` proxies the v1 `/text-to-speech` API. Requires user-supplied API key stored in `opencut/config.json` (never in source). Surfaces key prompt in `/audio/tts/backends` 503 hint. | `POST /audio/tts/elevenlabs`, `GET /audio/tts/elevenlabs/voices` | `core/tts_elevenlabs.py` | `elevenlabs` SDK ≥1.0 (Apache-2) | S | Apache-2 |
| L1.6 | **AI face reshaper** — apply facial geometry corrections (jaw slim, eye enlarge, nose reduce, chin lift) using MediaPipe face mesh + thin-plate-spline (TPS) warp. Processes each frame independently; Cutie temporal propagation keeps the mask consistent. DaVinci Resolve 21 ships this as a premium AI feature. | `POST /video/face/reshape`, `GET /video/face/reshape/info` | `core/face_reshape.py` | `mediapipe` ≥0.10 (Apache-2) — new dep | L | Apache-2 |
| L1.7 | **AI blemish / skin retouching** — GFPGAN-guided skin inpainting (bilateral filter + frequency separation) to suppress blemishes, even skin tone, reduce under-eye circles. Works per-frame with Cutie mask; no face geometry deformation. Optional strength slider 0.0–1.0. | `POST /video/face/retouch`, `GET /video/face/retouch/info` | `core/skin_retouch.py` | `gfpgan` (already in OpenCut for face restore) — no new dep | M | Apache-2 |
| L1.8 | **Job history panel** — persist completed/failed async jobs to SQLite (`opencut/data/jobs.db`), queryable by time range, route, status, media path. Powers a "recent operations" panel in the UI. Routes return `{jobs: [...], total, pages}`. | `GET /jobs/history`, `GET /jobs/history/{job_id}`, `DELETE /jobs/history/{job_id}` | `core/job_history.py` | `sqlalchemy` (already present) — no new dep | S | — |
| L1.9 | **Bulk clip operations** — apply a route (silence removal, stabilise, denoise, upscale, caption) to a folder of clips in one API call. Job fan-out with per-clip SSE progress and a summary result. Closes the CapCut Batch Export gap. | `POST /clips/bulk/process`, `GET /clips/bulk/status/{batch_id}` | `core/bulk_processor.py` | Pipeline + L1.1 SSE — no new dep | M | — |

---

## Wave L2 — New AI Engines (v1.34.0)

**Goal**: Integrate five OSS models discovered in the April 2026 research pass that are **not referenced in any previous OpenCut roadmap document**. All ship as 503 stubs behind `check_X_available()` with clear install instructions; promoted to Tier 1 once stabilised.  
**New required deps**: `framepack`, `acestep`, `sparktts` (one each, all Apache-2)  
**New routes**: ~18

### OSS Discoveries — Not in Any Prior Roadmap

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| L2.1 | **FramePack image-to-video** (NeurIPS 2025, lllyasviel) — next-frame-prediction video diffusion that reframes the T2V problem as sequential frame conditioning. Generates up to 60-second video from a single image + prompt on **6 GB VRAM** — far below LTX-Video (12 GB) or Open-Sora (24 GB). Apache-2, 13B parameter model, ships an optimised inference server. Adds a fourth T2V backend to the existing dispatcher (`/generate/backends`). Different architecture → different aesthetic; creator toolbox benefit. | `POST /generate/framepack`, `GET /generate/framepack/info` | `core/gen_video_framepack.py` | `framepack` (Apache-2) — new dep | L | Apache-2 | First VRAM-frugal T2V backend; democratises video gen on consumer GPUs |
| L2.2 | **ACE-Step full-song music generation** (ACE Studio + StepFun, January 2026) — Apache-2 foundation model for full-length music generation (up to 4 min) with lyric alignment, voice cloning, stem separation, lyric editing, and repainting. 3.5B model; generates 1 min of music in 4.7 s on RTX 3090. Supersedes the Wave K `AudioGen` SFX-only route for music-creation workflows. Ships `POST /audio/music/acestep` (text+lyrics→song) and `POST /audio/music/acestep/edit` (lyric editing via flow-edit). | `POST /audio/music/acestep`, `POST /audio/music/acestep/edit`, `POST /audio/music/acestep/extend`, `GET /audio/music/acestep/info` | `core/music_acestep.py` | `acestep` ≥0.1 (Apache-2) — new dep | L | Apache-2 | First full-song-with-lyrics generator in OpenCut; closes Suno/Udio gap locally |
| L2.3 | **Spark-TTS voice synthesis** (SparkAudio, March 2026, Apache-2) — SOTA zero-shot TTS with natural prosody and voice cloning from a 3-second reference clip. ONNX-compatible, runs on CPU without CUDA. Benchmarks show comparable or better MOS to ElevenLabs v2 on English speech. Sixth TTS backend; positioned as the default for users without GPU. | `POST /audio/tts/spark`, `GET /audio/tts/spark/voices`, `GET /audio/tts/spark/info` | `core/tts_sparktts.py` | `sparktts` ≥0.1 (Apache-2) — new dep | M | Apache-2 | Best CPU-native zero-shot TTS now available; replaces ElevenLabs cloud dependency for offline users |
| L2.4 | **VidMuse video-to-music** (CVPR 2025, MIT) — generate background music that is semantically and rhythmically synchronised to video content (scene cuts, motion energy, visual tempo). Unlike MusicGen (text-only) and ACE-Step (lyric-focused), VidMuse reads the video directly to compose a matching score. `POST /audio/music/vidmuse` returns a stereo audio file matched to the input clip duration. | `POST /audio/music/vidmuse`, `GET /audio/music/vidmuse/info` | `core/music_vidmuse.py` | VidMuse (MIT) — new dep | L | MIT | Only local tool that composes music *from* the video itself rather than text prompts |
| L2.5 | **Chat-based editing agent** — natural-language edit interface inside the panel: "remove all silences", "add a zoom punch on the word 'launch'", "translate captions to Spanish and French". Routes each instruction to the correct OpenCut API via a lightweight intent router (function-calling LLM against the 1,335 route schemas). Supports local (Ollama + llama-3.1) and cloud (OpenAI / Anthropic) backends, user-configurable. MCP server (L1.3) provides the tool definitions. | `POST /agent/chat`, `GET /agent/chat/history`, `DELETE /agent/chat/history` | `core/agent_chat.py` | `openai` ≥1.0 SDK (MIT) for cloud mode; Ollama REST (no pip dep) for local — no new required dep if user has Ollama | L | MIT / Apache-2 (user-configurable backend) | Wave L flagship: positions OpenCut as agent-native rather than just API-rich |
| L2.6 | **Moonshine real-time ASR** (Useful Sensors / Moonshine AI, 2025 — MIT for English models) — Whisper-compatible API with 10× faster inference on CPU, specifically optimised for streaming and edge devices. Adds `moonshine` as a fourth STT backend in `/captions/backends`. English-only MIT licence; multilingual models use a community licence (non-commercial) and are gated separately. Use case: live caption preview without GPU. | `POST /audio/transcribe/moonshine`, `GET /audio/transcribe/moonshine/info` | `core/asr_moonshine.py` | `moonshine` ≥0.1 (MIT for English) — new dep | M | MIT (English model) | Fastest CPU ASR available; enables real-time caption preview on low-spec machines |

---

## Wave L3 — Stub Promotions (v1.35.0)

**Goal**: Promote five Wave K Tier-3 stubs to full working implementations. All supporting backends (EchoMimic V3, CosyVoice2, SEA-RAFT, Depth Pro, LTX-2) shipped in Wave K — the only gap is the orchestration layer.  
**New required deps**: None (all backends already installed via Wave K)  
**New routes**: ~14

| # | Feature | Route(s) | Module | Builds on | Effort | Notes |
|---|---------|----------|--------|-----------|--------|-------|
| L3.1 | **Full local video dubbing pipeline** (K3.1 promotion) — end-to-end: WhisperX STT → NLLB-200 translate (L1.2 reuse) → CosyVoice2 zero-shot voice clone → EchoMimic V3 lip sync → FFmpeg composite with original audio replace. Private, free, local. HeyGen charges per-minute; this is the OSS alternative. | `POST /dub/pipeline`, `GET /dub/pipeline/backends`, `GET /dub/pipeline/status/{job_id}` | `core/dub_pipeline.py` | K2.4 CosyVoice2 + K2.5 EchoMimic V3 + L1.2 caption translate | L | Priority: unblocks avatar pipeline and multilingual content creation |
| L3.2 | **Auto trailer / promo generator** (K3.2 promotion) — LLM-scored moment extraction (emotion arc + virality score + face presence) → top-N clip selection → MusicGen ramp + title card (declarative_compose) + CTA overlay + auto-paced cut rhythm. All component pieces in OpenCut from prior waves; the conductor (pipeline module) is the only gap. Descript Underlord charges for this. | `POST /generate/trailer`, `POST /generate/promo`, `GET /generate/trailer/presets` | `core/trailer_gen.py` | K1.3 virality + A2.5 emotion arc + Wave I declarative_compose + K2.18 audio-reactive | M | Wire existing outputs; conductor logic is the effort |
| L3.3 | **Sports / genre-agnostic highlights** (K3.8 promotion) — optical flow velocity peak (SEA-RAFT) + YAMNet crowd energy + laughter/cheer detection + face-count burst combine into a per-frame excitement score. Top-N segments extracted with 2-second padding. Works for sports, concerts, events, gaming clips — not just talking-head. OpusClip ClipAnything charges per clip. | `POST /analyze/highlights/sports`, `GET /analyze/highlights/genres`, `GET /analyze/highlights/info` | `core/highlights_sports.py` | K2.9 SEA-RAFT + H1.4 YAMNet (or AudioGen CLAP) + A2.5 emotion arc | M | `--genre` param gates which scoring functions activate |
| L3.4 | **EchoMimic V3 talking head** (K2.5 promotion from "stub" to "tested + recommended") — full integration test suite, half-body mode, reference-audio conditioning, and promotion to `recommended: true` in `/lipsync/backends`. Unblocks L3.1 dub pipeline. | `POST /lipsync/echomimic`, `GET /lipsync/echomimic/presets`, `GET /lipsync/echomimic/info` | `core/lipsync_echomimic.py` (extend existing stub) | K2.5 EchoMimic V3 stub | M | Integration tests required before `recommended: true` flag |
| L3.5 | **AI CineFocus rack focus** (K2.19 promotion) — depth-of-field bokeh using Depth Pro metric depth: keyframeable focal point, aperture shape, f-number slider, rack-focus animation (focus-pull from background to foreground over N frames). DaVinci Resolve 21 CineFocus requires a paid licence. OpenCut ships free. | `POST /video/cinefocus/render`, `POST /video/cinefocus/preview`, `GET /video/cinefocus/presets` | `core/cinefocus.py` | K2.13 Depth Pro + FFmpeg boxblur | M | Expose `focal_x`, `focal_y` as keyframeable float params |

---

## L-OSS: New OSS Ecosystem Survey (April 2026)

Tools discovered in the April 2026 research pass that are not mentioned in any prior roadmap wave. Listed with licence, stars tier, and the OpenCut feature each enables.

### AI Video & Image Generation

| Tool | Org | Licence | Status | Relevance |
|------|-----|---------|--------|-----------|
| **FramePack** | lllyasviel | Apache-2 | NeurIPS 2025 — production weights on HuggingFace | L2.1 — fourth T2V backend, 6 GB GPU viable |
| **FLUX.1 Kontext** (dev variant) | Black Forest Labs | Apache-2 (dev only) | June 2025 — instruction-guided image/frame editing | Evaluate for Wave M: `POST /video/frame-edit/kontext` — edit a single frame then propagate via TokenFlow |
| **Wan 2.2** | Alibaba Wan-Video | Apache-2 | Q2 2026 — follow-up to Wan 2.1 with improved motion quality | Evaluate: upgrade existing C4 Wan2.1 backend when stable |

### Audio & Music Generation

| Tool | Org | Licence | Status | Relevance |
|------|-----|---------|--------|-----------|
| **ACE-Step** (v1.5, Jan 2026) | ACE Studio + StepFun | Apache-2 | Production — 3.5B, RTX 3090 in 4.7 s/min | L2.2 — full-song music with lyrics, voice clone, lyric edit |
| **Spark-TTS** | SparkAudio | Apache-2 | March 2026 — ONNX-ready, CPU-native | L2.3 — best CPU-native zero-shot TTS; replaces cloud ElevenLabs for offline use |
| **VidMuse** | — (CVPR 2025) | MIT | Research release | L2.4 — video-conditioned music composition |

### Speech Recognition

| Tool | Org | Licence | Status | Relevance |
|------|-----|---------|--------|-----------|
| **Moonshine** (English) | Useful Sensors / Moonshine AI | MIT (English models) | Stable — C++/Python/Android/iOS | L2.6 — 10× faster than Whisper on CPU; real-time caption preview |

### Competing OSS Editors & Automation Tools

| Tool | Licence | Key Differentiator | Gap vs OpenCut |
|------|---------|-------------------|----------------|
| **[mifi/lossless-cut](https://github.com/mifi/losslesscut)** | GPLv2 | FFmpeg-first lossless cut planning + smart UX | No AI features; OpenCut laps it on AI but its silence-threshold UX defaults are worth borrowing |
| **[WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor)** | MIT | CLI silence removal with sub-second previews | OpenCut bulk processor (L1.9) closes the gap; auto-editor's `--motion` mode is a reference for sports highlights (L3.3) |
| **[Crayotter](https://github.com/idwts/Crayotter)** | MIT | Multimodal agentic video editor (GPT-4o + FFMPEG) | OpenCut chat agent (L2.5) closes this gap with local model support |
| **[hetpatel-11/Adobe_Premiere_Pro_MCP](https://github.com/hetpatel-11/Adobe_Premiere_Pro_MCP)** | MIT | MCP server bridging Claude/Codex → Premiere scripting | OpenCut MCP server (L1.3) closes this gap but targets the backend layer (1,335 routes) not just Premiere's scripting layer |
| **[FireRed/OpenStoryline](https://github.com/FireRedTeam/FireRed-OpenStoryline)** | Apache-2 | Style Skills — LLM-generated motion/FX presets | Evaluate for Wave M: "Style Skills" tab in the effects panel |
| **[sebinside/PremiereRemote](https://github.com/sebinside/PremiereRemote)** | MIT | WebSocket trigger surface for Premiere from external tools | OpenCut MCP server (L1.3) + chat agent (L2.5) cover the use case more broadly |

---

## L-Competitive: Gap Closure Matrix

| Competitor Feature | OpenCut Gap (pre-L) | Wave L closes it |
|---|---|---|
| Captions.ai — multilingual auto-captions | No translation route | L1.2 caption translate (NLLB-200, 200 languages) |
| Captions.ai — AI face retouch | No skin retouching | L1.7 blemish removal + L1.6 face reshaper |
| OpusClip — sports highlights | K3.8 stub unshipped | L3.3 promotion |
| HeyGen — video dubbing pipeline | K3.1 stub unshipped | L3.1 promotion |
| Descript Underlord — trailer generator | K3.2 stub unshipped | L3.2 promotion |
| DaVinci Resolve 21 — CineFocus | K2.19 stub unshipped | L3.5 promotion |
| Suno v5.5 / Udio — full song gen | No local equivalent | L2.2 ACE-Step (Apache-2, local) |
| ElevenLabs — SOTA TTS | Cloud-gated | L2.3 Spark-TTS (CPU-native Apache-2) |
| Runway Gen-4.5 — agent / API surface | 1,335 routes, no MCP | L1.3 MCP server (any AI client can drive OpenCut) |
| Premiere Pro 26 — AI chat editing | No chat interface | L2.5 chat agent (local + cloud backend) |
| Topaz Video AI — batch processing | No batch API | L1.9 bulk clip operations |

---

## Wave L Gotchas

- **SSE + Flask dev server (L1.1)** — Flask's built-in server buffers responses; SSE requires `threaded=True` and `Response(stream_with_context(...))`. Under waitress/gunicorn use `text/event-stream` with explicit `X-Accel-Buffering: no` header. Test with `curl -N` before wiring the panel.
- **NLLB-200 1.3B cold-load time (L1.2)** — ~8 s on first request. Lazy-load and keep resident in a module-level singleton guarded by a `threading.Lock`. Do not reload per request. Memory footprint: ~3 GB. Warn users with < 8 GB RAM in the 503 install hint.
- **MCP server port conflict (L1.3)** — MCP over stdio is the safe default (no port). HTTP transport is optional but requires a free port; document in the install hint. Never auto-start MCP server at extension load; user must explicitly launch `python -m opencut.mcp_server`.
- **FramePack sequential inference (L2.1)** — FramePack generates frames one-at-a-time, not in batch. Progress callback should emit per-frame percentage. Cap output at 240 frames (10 s at 24fps) by default to bound inference time; expose `--max_frames` as an advanced param.
- **ACE-Step Windows torch.compile (L2.2)** — `--torch_compile true` requires `triton-windows` on Windows; surface this in the 503 install hint rather than hard-requiring it. Default to `--torch_compile false` on Windows. `--cpu_offload true --overlapped_decode true` should be the Windows default to fit within 8 GB VRAM.
- **Spark-TTS ONNX path (L2.3)** — The PyPI `sparktts` package is the Python reference; the ONNX runtime path (via `onnxruntime`) is 3–5× faster on CPU. Detect `onnxruntime` at runtime and prefer it; surface conversion instructions if missing.
- **VidMuse GPU memory (L2.4)** — VidMuse requires 24 GB VRAM at full resolution. Add `--resize_input 512` as a default to bring it within 12 GB; document the quality trade-off in the 503 hint.
- **Chat agent hallucinated routes (L2.5)** — the intent router must validate every candidate route against the live `/routes/list` endpoint before calling it. Never let the LLM construct routes from memory; always look them up. Rate-limit at 5 concurrent chat requests to prevent runaway job queues.
- **Moonshine multilingual gate (L2.6)** — English models are MIT; non-English models are Moonshine Community Licence (non-commercial, registration required). Gate them separately: `check_moonshine_en_available()` vs `check_moonshine_multilingual_available()`. Never auto-download non-English models silently.
- **EchoMimic V3 backend priority (L3.4)** — when promoting to `recommended: true`, do not silently redirect existing MuseTalk/LatentSync requests. Keep the old backends available; only set the new `recommended` flag so the panel defaults to it for new sessions.
- **Bulk processor job cancellation (L1.9)** — individual jobs within a batch can fail without cancelling the whole batch. Return `{completed: N, failed: M, skipped: K}` in the summary. Expose `POST /clips/bulk/cancel/{batch_id}` to stop in-flight batches.

---

## Wave L Shipping Cadence

| Release | Target | Key deliverables |
|---------|--------|-----------------|
| v1.33.0 | 2026-Q3 | L1.1 SSE streaming, L1.2 caption translate, L1.3 MCP server, L1.4 upscaling hub, L1.5 ElevenLabs TTS, L1.6 face reshaper, L1.7 skin retouch, L1.8 job history, L1.9 bulk ops |
| v1.34.0 | 2026-Q3 | L2.1 FramePack, L2.2 ACE-Step, L2.3 Spark-TTS, L2.4 VidMuse, L2.5 chat agent, L2.6 Moonshine ASR |
| v1.35.0 | 2026-Q4 | L3.1 dub pipeline, L3.2 trailer gen, L3.3 sports highlights, L3.4 EchoMimic V3, L3.5 CineFocus |

---

## Wave L: Not Adopted / Deferred

- **FLUX.1 Kontext pro variant** — commercial licence; dev variant (Apache-2) is viable but requires testing on single-frame edit + TokenFlow propagation workflow. Defer to Wave M once that pipeline is validated.
- **Moonshine multilingual models** — Moonshine Community Licence requires registration for revenue >$1M; safe for OpenCut but requires explicit user opt-in and a separate `check_moonshine_multilingual_available()` gate. Defer multilingual support to Wave M.
- **ReEzSynth Ebsynth style propagation** — research.md §8 Tier 4. Licence unclear at time of survey; revisit when clarified.
- **FlashVSR real-time VSR** — CVPR 2026 weights pending public release. Add to Wave M watch list.
- **STAR temporal coherence post-processor** — research-only weights at time of survey. Revisit for Wave M.
- **Sammie-Roto-2 rotoscoping** — research-only weights. Cutie (K2.7) covers the tracking use case under MIT. Revisit if licence clarifies.
- **HappyHorse 1.0 T2V** — licence TBD at time of survey. Revisit for Wave M.
- **GaussianHeadTalk / FantasyTalking2** — early research releases; EchoMimic V3 (K2.5) is more mature and production-ready. Revisit after L3.4 ships.
- **Digital twin / AI avatar pipeline** — requires stable L3.1 dub + L3.4 EchoMimic + brand-kit identity data. Planned for Wave M as a first-class pipeline once all component pieces are hardened in Wave L.
- **Plugin marketplace / hub** — architectural dependency on stable MCP server (L1.3) and job history (L1.8). Defer to Wave M once those foundations are proven.

---

## Wave L Sources

- **FramePack** — [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) (NeurIPS 2025, Apache-2)
- **ACE-Step** — [ace-step/ACE-Step](https://github.com/ace-step/ACE-Step) v1.5 (Jan 2026, Apache-2); [Technical Report arXiv:2506.00045](https://arxiv.org/abs/2506.00045)
- **Spark-TTS** — [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS) (Mar 2026, Apache-2)
- **VidMuse** — CVPR 2025, MIT licence (weights on HuggingFace)
- **Moonshine** — [moonshine-ai/moonshine](https://github.com/moonshine-ai/moonshine) (MIT for English, Moonshine Community Licence for multilingual)
- **FLUX.1 Kontext** — Black Forest Labs (June 2025, Apache-2 dev variant, commercial pro variant)
- **MCP SDK** — [modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) (MIT)
- **NLLB-200** — Meta AI (Apache-2); [research paper](https://arxiv.org/abs/2207.04672)
- **Crayotter** — [idwts/Crayotter](https://github.com/idwts/Crayotter) (MIT, agentic video editor reference)
- **Premiere Pro MCP** — [hetpatel-11/Adobe_Premiere_Pro_MCP](https://github.com/hetpatel-11/Adobe_Premiere_Pro_MCP) (MIT, MCP architecture reference)
- **OpenCut `research.md`** — April 2026 competitive audit, §1–§9, Wave L priorities §8
- **OpenCut `ROADMAP-NEXT.md`** — Wave K stubs (K2.5, K2.19, K3.1, K3.2, K3.8) promoted in L3

---

# Wave M — Audio Intelligence + Video Model Upgrades (v1.36.0 → v1.38.0)

**Updated**: 2026-06-15
**Baseline**: v1.35.0 (post-Wave L; MCP server, job history, FramePack, ACE-Step, Spark-TTS, Moonshine English, dub pipeline, trailer gen, EchoMimic V3 all shipped)
**Research pass**: June 2026 GitHub OSS survey — Chatterbox, Kokoro, DiffRhythm, Wan2.2 family

This wave synthesises:
1. Four new OSS tools confirmed in this research pass not yet referenced in any prior roadmap document (Chatterbox TTS, Kokoro, DiffRhythm v1.2, Wan2.2 family)
2. Features deferred from Wave L's "Not Adopted" list that now have their required dependencies hardened (Moonshine multilingual, FLUX.1 Kontext, plugin marketplace, digital twin pipeline)
3. A video model upgrade path: Wan2.1 K3.7 VACE stub → Wan2.2 full family (T2V, S2V, Animate)

All guiding principles from prior waves apply: never break what works, one new required dep per feature maximum, permissive licences only, `check_X_available()` guard + `@async_job` + queue allowlist.

---

## Wave M1 — Audio Intelligence Expansion (v1.36.0)

**Goal**: Dramatically improve the TTS and music generation ecosystem with three newly confirmed permissive-licence models, and promote the deferred Moonshine multilingual gate.
**New required deps**: `chatterbox-tts` (MIT), `kokoro` ≥0.9.4 (Apache-2), DiffRhythm inference script + `requirements.txt` (Apache-2), `espeak-ng` system dep (shared by Kokoro + DiffRhythm)
**New routes**: ~16

### OSS Discoveries — New Audio Models

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| M1.1 | **Chatterbox TTS** (Resemble AI, MIT) — Chatterbox-Turbo (350M) is the fastest emotional open-source TTS available: paralinguistic tags `[laugh]`, `[chuckle]`, `[cough]` baked into the model architecture, zero-shot voice cloning from a 10-second clip, and a multilingual variant (500M, 23 languages) that benchmarks ahead of ElevenLabs Turbo v2.5 and Cartesia Sonic 3 in independent Podonos evaluation. Built-in Perth perceptual watermarking. Ships as two backends: `chatterbox-turbo` (English, GPU preferred) and `chatterbox-multilingual` (23 langs, gated behind a second `check_chatterbox_multilingual_available()` guard). Supersedes the ElevenLabs cloud fallback (L1.5) for users who can run local. | `POST /audio/tts/chatterbox`, `POST /audio/tts/chatterbox/multilingual`, `GET /audio/tts/chatterbox/voices`, `GET /audio/tts/chatterbox/info` | `core/tts_chatterbox.py` | `chatterbox-tts` ≥0.1 (MIT) — new dep | M | MIT | Only open-source TTS with native paralinguistic emotion tags; beats ElevenLabs in naturalness benchmarks; zero-shot cloning |
| M1.2 | **Kokoro ultralight TTS** (hexgrad, Apache-2) — 82M parameter TTS, CPU-only (`pip install kokoro`), 24 kHz output, 9 languages (US/UK English, Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin). No CUDA required whatsoever. Fastest possible TTS for low-spec machines and CI-pipeline preview renders. Ships as the new last-resort TTS fallback below Spark-TTS in the priority chain: Chatterbox Turbo → Spark-TTS → Kokoro → (error). Requires `espeak-ng` system package for G2P OOD fallback. | `POST /audio/tts/kokoro`, `GET /audio/tts/kokoro/voices`, `GET /audio/tts/kokoro/info` | `core/tts_kokoro.py` | `kokoro` ≥0.9.4 + `misaki` + `espeak-ng` system dep (Apache-2) — new dep | S | Apache-2 | 82M params, CPU-only, `pip install kokoro`; adds a "works on any machine" fallback tier for TTS |
| M1.3 | **DiffRhythm full-song generation** (ASLP-lab, Apache-2) — first diffusion-based full-length song generator; base model outputs 1m35s, full model outputs up to 4m45s. Accepts LRC lyrics + optional audio style reference OR text style prompt (e.g., "Jazzy Nightclub Vibe", "Pop Emotional Piano"). DiffRhythm v1.2 adds song editing, continuation, and instrumental-only mode. Complements ACE-Step (L2.2): ACE-Step excels at lyric editing and stem separation; DiffRhythm excels at composing complete songs end-to-end from a text brief. 8 GB VRAM minimum; `--chunked` flag reduces peak to 8 GB. Requires `espeak-ng` for lyrics phonemisation (same system dep as M1.2). Windows: must set `PHONEMIZER_ESPEAK_LIBRARY` and `PHONEMIZER_ESPEAK_PATH` env vars. | `POST /audio/music/diffrhythm`, `POST /audio/music/diffrhythm/extend`, `GET /audio/music/diffrhythm/styles`, `GET /audio/music/diffrhythm/info` | `core/music_diffrhythm.py` | DiffRhythm git clone + `requirements.txt` (Apache-2) — new dep | L | Apache-2 | End-to-end song composer from lyrics + style text; up to 285 s; complements ACE-Step |
| M1.4 | **Moonshine multilingual ASR** — deferred from Wave L "Not Adopted" list. Moonshine Community Licence is non-commercial; acceptable for OpenCut (personal/hobbyist use) with explicit user acknowledgment. Ships behind `check_moonshine_multilingual_available()` guard that also checks for a stored acknowledgment timestamp in `opencut/config.json` (`moonshine_multilingual_ack`). Presents a one-time "Non-commercial use only" consent prompt on first activation. Adds 99-language transcription to the ASR backend chain. | `POST /audio/transcribe/moonshine/multilingual`, `GET /audio/transcribe/moonshine/multilingual/info` | `core/asr_moonshine_multilingual.py` | `moonshine` multilingual weights (Community Licence) — new dep with user opt-in | M | Moonshine Community Licence (non-commercial; gated) | Closes 99-language caption gap for non-English creators on CPU hardware; no GPU required |

---

## Wave M2 — Video Model Upgrades (v1.37.0)

**Goal**: Upgrade the Wan2.1 VACE stub (K3.7) to the full Wan2.2 family, add speech-driven talking-head video (S2V) and character animation/replacement (Animate), and integrate FLUX.1 Kontext for AI image editing across video frames.
**New required deps**: Wan2.2 package (Apache-2, upgrades existing Wan2.1 dep), `diffusers` ≥0.29 already present (FLUX Kontext uses diffusers pipeline)
**New routes**: ~14

### OSS Discoveries — Video Model Upgrades

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| M2.1 | **Wan2.2 T2V/I2V/TI2V upgrade** (Wan-Video, Apache-2) — replaces the Wan2.1 K3.7 VACE stub with the full Wan2.2 model family. Key improvements over Wan2.1: MoE architecture (larger effective capacity at same compute), +65.6% image training data / +83.2% video training data, cinematic aesthetics labels (lighting, composition, colour tone), and the TI2V-5B consumer model (720P@24fps on a 4090). Integrated into ComfyUI and Diffusers. Routes: create new `/generate/wan2.2` family (T2V, I2V, TI2V) with a backwards-compat alias from `/generate/wan2.1`. `TI2V-5B` is the default for consumer deployments; A14B models surface a VRAM check and require `--offload_model True` flag. | `POST /generate/wan2.2/t2v`, `POST /generate/wan2.2/i2v`, `POST /generate/wan2.2/ti2v`, `GET /generate/wan2.2/info` | `core/gen_video_wan22.py` | Wan2.2 (`git clone Wan-Video/Wan2.2` + `requirements.txt`, Apache-2) — upgrades existing Wan2.1 dep | L | Apache-2 | Best open-source T2V; MoE architecture, cinematic aesthetics, 720P@24fps on consumer GPUs (TI2V-5B) |
| M2.2 | **Wan2.2-S2V speech-to-video** (Wan-Video, Apache-2, Aug 2025) — 14B model that generates a talking-head video from an audio clip + reference portrait image. Given any audio recording (no TTS needed — accepts real voice files), the model generates synchronized lip movements, upper-body motion, and natural facial expression. Optional CosyVoice2 integration (already in OpenCut) enables full text→speech→video mode. Primary use case: narration video without filming — write a script, clone a voice (M1.1 Chatterbox), generate the video. Requires 80 GB VRAM for single-GPU, multi-GPU via FSDP, or `--offload_model True` for consumer cards. | `POST /generate/wan2.2/s2v`, `GET /generate/wan2.2/s2v/info` | `core/gen_video_wan22_s2v.py` | Wan2.2-S2V-14B weights + `requirements_s2v.txt` (Apache-2) — CosyVoice2 optional | L | Apache-2 | Script + voice clone → talking-head video; closes HeyGen/Synthesia gap locally |
| M2.3 | **Wan2.2-Animate character animation and replacement** (Wan-Video, Apache-2, Sep 2025) — unified model for two workflows: (a) animate a still character photo to match motions from a reference video, (b) replace the character in a video with a different appearance while preserving all movements and expressions. Replicates holistic body movement and facial expression simultaneously. Ships as a complement to EchoMimic V3 (L3.4): EchoMimic excels at portrait-only lip-sync from audio; Animate excels at full-body motion transfer and character swap. | `POST /generate/wan2.2/animate`, `GET /generate/wan2.2/animate/info` | `core/gen_video_wan22_animate.py` | Wan2.2-Animate-14B weights (Apache-2) — additional Wan2.2 variant | L | Apache-2 | Full-body character animation + replacement; closes Adobe Character Animator gap locally |
| M2.4 | **FLUX.1 Kontext-dev image editing** (Black Forest Labs, Apache-2 dev variant) — deferred from Wave L "Not Adopted" list. Context-aware image-to-image editing: accepts an image + natural language instruction and returns the edited result. Primary workflow: apply per-frame AI edits (object removal, style transfer, subject replacement, background swap) and propagate changes across a video clip using TokenFlow (already in OpenCut as a Wave K dep). The dev variant is Apache-2; the pro variant is commercial and is NOT used. Pre-flight check: FLUX Kontext-dev weights are ~24 GB — `GET /image/edit/kontext/info` returns `{size_gb: 24, downloaded: bool}` and the install route confirms before downloading. | `POST /image/edit/kontext`, `POST /video/edit/kontext`, `GET /image/edit/kontext/info` | `core/image_edit_kontext.py` | `diffusers` ≥0.29 (already present) + FLUX Kontext-dev weights via HuggingFace (~24 GB) | M | Apache-2 (dev variant only) | Per-frame AI editing propagated to video via TokenFlow; closes Runway/Pika AI object-edit gap |

---

## Wave M3 — Platform Maturity (v1.38.0)

**Goal**: Ship two major platform-level features that were deferred from Wave L because they depended on components that were not yet stable. Both dependencies are now hardened in Waves L and M.
**New required deps**: None — chains existing (L1.3 MCP, L1.8 job history, L3.1 dub, L3.4 EchoMimic, M1.1 Chatterbox, M2.2 S2V)
**New routes**: ~12

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| M3.1 | **Plugin marketplace / hub** — deferred from Wave L. Now that MCP server (L1.3) and job history (L1.8) are production-hardened, build a plugin registry: `GET /plugins/list` returns a manifest of installed + available community plugins; `POST /plugins/install` downloads, validates SHA-256 manifest, and registers the plugin; each plugin exposes new routes via MCP tool schemas validated against the `/routes/list` contract. Ships a built-in plugin SDK (`opencut/plugin_sdk.py`) with a plugin template, route validator, and schema generator. Initial curated set: 5 first-party plugins (batch caption translate, bulk denoise, VidMuse auto-score, Chatterbox narration, export preset packs). Community plugins require explicit user consent. | `GET /plugins/list`, `POST /plugins/install`, `DELETE /plugins/{id}`, `GET /plugins/{id}/schema`, `GET /plugins/{id}/routes` | `core/plugin_manager.py`, `opencut/plugin_sdk.py` | Chains L1.3 MCP + L1.8 job history — no new dep | L | — | First truly extensible OpenCut architecture; community-driven feature growth post-Wave M |
| M3.2 | **Digital twin / AI avatar pipeline** — deferred from Wave L. Full end-to-end localisation pipeline: (1) clone voice from a 10-second reference clip using Chatterbox TTS (M1.1), (2) generate narration audio from the script, (3) generate lip-sync talking-head video using Wan2.2-S2V (M2.2) or EchoMimic V3 (L3.4), (4) translate and dub the output into target languages using the dub pipeline (L3.1), (5) composite the avatar onto original footage. Exposed as a single `POST /pipeline/digital_twin` endpoint accepting `{script, voice_ref_path, face_ref_path, target_langs[]}` and returning a completed per-language dubbed video package. Each stage is independently skipable if pre-existing assets are provided. | `POST /pipeline/digital_twin`, `GET /pipeline/digital_twin/info`, `GET /pipeline/digital_twin/stages` | `core/pipeline_digital_twin.py` | Chains M1.1 Chatterbox, M2.2 S2V, L3.1 dub, L3.4 EchoMimic — no new dep | XL | Combined (MIT + Apache-2) | Complete localisation pipeline — script-in, multilingual-dubbed-video-out; closes CapCut AI dubbing gap |

---

## Wave M: M-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave M Role |
|------|-----|---------|----------|-------------|
| Chatterbox TTS | Resemble AI | MIT | TTS | M1.1 — Emotional TTS + voice clone |
| Chatterbox-Multilingual | Resemble AI | MIT | TTS | M1.1 — 23-language variant |
| Kokoro-82M | hexgrad | Apache-2 | TTS | M1.2 — 82M CPU-only fallback TTS |
| DiffRhythm v1.2 | ASLP-lab | Apache-2 | Music Gen | M1.3 — End-to-end song generation |
| DiffRhythm+ / DiffRhythm 2 | ASLP-lab | Apache-2 | Music Gen | Watch list — not yet stable; Wave N |
| Moonshine multilingual | Moonshine AI | Community | ASR | M1.4 — Gated multilingual ASR |
| Wan2.2 T2V/I2V/TI2V | Wan-Video | Apache-2 | T2V | M2.1 — Upgrades Wan2.1 K3.7 stub |
| Wan2.2-S2V-14B | Wan-Video | Apache-2 | Talking Head | M2.2 — Speech-to-video generation |
| Wan2.2-Animate-14B | Wan-Video | Apache-2 | Char Anim | M2.3 — Character animation/replacement |
| FLUX.1 Kontext-dev | Black Forest Labs | Apache-2 | Image Edit | M2.4 — Per-frame AI editing |
| Fish-Speech v1.5 | Fish Audio | FARL | TTS | NOT ADOPTED — non-commercial licence |
| LightX2V | ModelTC | Apache-2 | Inference Acc | Watch list — Wan2.2 acceleration; Wave N |
| FastVideo | hao-ai-lab | Apache-2 | Inference Acc | Watch list — distilled Wan2.2; Wave N |

---

## Wave M: Competitive Gap Closure

| Gap | Competitor | Wave M Feature | Closes? |
|-----|-----------|---------------|---------|
| Emotional/paralinguistic TTS local | ElevenLabs | M1.1 Chatterbox | Y — free, local, MIT; beats ElevenLabs in benchmarks |
| Full-length song from lyrics text | Suno, Udio | M1.3 DiffRhythm | Y — up to 4m45s, text or audio reference |
| Audio-driven talking-head video | HeyGen, Synthesia | M2.2 Wan2.2-S2V | Y — local, any audio recording as input |
| Character replacement + animation | Adobe Character Animator | M2.3 Wan2.2-Animate | Y — full-body motion + expression transfer |
| AI object/subject editing in video | Runway Gen-3, Pika 2.0 | M2.4 FLUX Kontext | Y — per-frame editing propagated via TokenFlow |
| Plugin / extension ecosystem | Premiere Pro extensions | M3.1 Plugin marketplace | Y — MCP-native SDK, community plugins |
| End-to-end script-to-dubbed-video | CapCut AI dubbing, HeyGen | M3.2 Digital twin | Y — script + face ref + voice ref → multilingual video |

---

## Wave M Gotchas

- **Chatterbox on Windows**: Resemble AI developed on Debian 11. Windows users will need `espeak-ng` for phonemisation fallback (same as DiffRhythm). Document `PHONEMIZER_ESPEAK_LIBRARY` and `PHONEMIZER_ESPEAK_PATH` in the M1.1 503 install hint.
- **DiffRhythm espeak-ng env vars**: `check_diffrhythm_available()` must validate both env vars are set on Windows before allowing the model to load. Fail with a clear 503 hint that includes the MSI download URL and the two env var names.
- **Wan2.2 A14B models**: Both T2V-A14B and I2V-A14B require 80 GB VRAM for single-GPU inference. Gate these variants with `check_wan22_highvram_available()` (checks `torch.cuda.mem_get_info()` for 76+ GB free). Surface a 503 that recommends TI2V-5B for consumer cards.
- **Wan2.2-S2V CosyVoice2 dep**: S2V has an optional CosyVoice2 dependency (`requirements_s2v.txt`) for the text→speech→video mode. Keep this optional — S2V works perfectly well with a pre-recorded audio file. Install hint should clarify the two modes.
- **FLUX Kontext-dev weight size**: ~24 GB download. Add a size-check in `check_kontext_available()` that returns `{available: false, reason: "weights_not_downloaded", size_gb: 24}` when absent. The `POST /image/edit/kontext` 503 should include an explicit download confirmation step, not a silent auto-download.
- **Moonshine multilingual licence gate**: Never auto-start the multilingual model without a stored acknowledgment. The `check_moonshine_multilingual_available()` guard must verify `opencut/config.json["moonshine_multilingual_ack"]` exists and is a valid ISO 8601 timestamp before returning `available: true`.
- **Fish-speech is NOT viable**: Fish Audio Research License explicitly requires a separate commercial license for any deployment beyond personal research. Do not adopt regardless of model quality. Chatterbox (MIT) or Kokoro (Apache-2) are the correct alternatives.
- **DiffRhythm 2 / DiffRhythm+**: Both are newer variants with separate papers (arXiv:2507.12890 for DiffRhythm+) but were still maturing at the time of this research pass. Integrate DiffRhythm v1.2-base and v1.2-full in M1.3; revisit newer variants for Wave N.
- **Digital twin pipeline (M3.2) VRAM budget**: Pipeline chains Chatterbox (350M, ~1 GB VRAM) → S2V (14B, ~28 GB offloaded) → dub (CosyVoice2, ~4 GB) → EchoMimic (if chosen, ~6 GB). Total peak per-stage, never concurrent. Coordinate offloading via the existing `ModelRegistry` to avoid OOM.

---

## Wave M Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.36.0 | 2026-Q4 | M1.1 Chatterbox TTS, M1.2 Kokoro, M1.3 DiffRhythm, M1.4 Moonshine multilingual |
| v1.37.0 | 2026-Q4 | M2.1 Wan2.2 upgrade, M2.2 S2V, M2.3 Animate, M2.4 FLUX Kontext |
| v1.38.0 | 2027-Q1 | M3.1 Plugin marketplace, M3.2 Digital twin pipeline |

---

## Wave M: Not Adopted / Deferred

- **Fish-speech** (Fish Audio Research License) — non-commercial only; Fish Audio requires a separate commercial license for any production deployment. NOT viable. Use Chatterbox (MIT) or Kokoro (Apache-2) instead.
- **DiffRhythm+ / DiffRhythm 2** — Apache-2 but newer variants still maturing at time of survey. DiffRhythm v1.2 is the stable target. Monitor for Wave N.
- **Wan2.2-T2V-A14B / I2V-A14B consumer** — 80 GB VRAM requirement makes these impractical on consumer hardware. TI2V-5B (M2.1) covers the consumer use case. Defer A14B consumer offload optimisation to Wave N.
- **LightX2V acceleration** (ModelTC, Apache-2) — step-distillation and sparse-attention wrappers for Wan2.2; not yet production-stable. Watch list for Wave N once upstream inference stabilises.
- **FastVideo distilled Wan2.2** (hao-ai-lab, Apache-2) — distilled Wan models with sparse attention; same stability concern as LightX2V. Wave N watch list.
- **FLUX.1 Kontext-pro** — commercial licence. Dev variant (Apache-2) is the only viable path. Do not use pro variant.
- **Kokoro.js** — web-only JavaScript variant; not relevant to the Python/Flask backend.
- **HunyuanVideo** (Tencent) — Tencent Community License prohibits commercial use for >100M MAU and is geo-restricted (excludes EU, UK, South Korea). Already excluded in Wave K. Confirmed NOT viable.

---

## Wave M Sources

- **Chatterbox** — [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) (MIT, May 2025); Podonos evaluations: [vs ElevenLabs Turbo v2.5](https://podonos.com/resembleai/chatterbox-turbo-vs-elevenlabs-turbo), [vs Cartesia Sonic 3](https://podonos.com/resembleai/chatterbox-turbo-vs-cartesia-sonic3)
- **Kokoro** — [hexgrad/kokoro](https://github.com/hexgrad/kokoro) (Apache-2); [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M); 82M params, CPU-only
- **DiffRhythm** — [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) (Apache-2, March 2025); [arXiv:2503.01183](https://arxiv.org/abs/2503.01183); DiffRhythm+ [arXiv:2507.12890](https://arxiv.org/abs/2507.12890); v1.2 released May 2025
- **Wan2.2** — [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) (Apache-2, July 2025); [arXiv:2503.20314](https://arxiv.org/abs/2503.20314); S2V added Aug 2025; Animate added Sep 2025
- **FLUX.1 Kontext** — [black-forest-labs/flux](https://github.com/black-forest-labs/flux) (Apache-2 dev variant, June 2025); `model_licenses/LICENSE-FLUX1-dev` confirmed Apache-2
- **Fish-speech** — [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) — confirmed Fish Audio Research License (non-commercial); NOT adopted
- **Wave L Not Adopted list** — Moonshine multilingual, FLUX.1 Kontext, plugin marketplace, digital twin pipeline all promoted to Wave M
- **Wave K stubs** — K3.7 Wan2.1 VACE upgraded to Wan2.2 full family in M2.1

---

# Wave N — Acceleration + Scene Intelligence + Content Understanding (v1.39.0 → v1.41.0)

**Updated**: 2026-04-24
**Baseline**: v1.38.0 (post-Wave M; FastVideo/LightX2V not yet integrated; Wan2.2 S2V, Animate, DiffRhythm, Chatterbox, Kokoro, FLUX Kontext, plugin marketplace, digital twin pipeline all shipped)
**Research pass**: April 2026 GitHub OSS survey — FastVideo, LightX2V, SAM2, Depth-Anything-V2, CogVideoX, Qwen2.5-VL, CSM

This wave synthesises:
1. Two inference-acceleration frameworks (FastVideo, LightX2V) that make the Wave M Wan2.2 models viable on mid-range consumer hardware by trading quality-neutral for speed gains
2. Two scene-intelligence primitives (SAM2 video segmentation, Depth-Anything-V2 depth maps) that unlock a new class of compositor effects not possible without per-frame spatial understanding
3. Three content-intelligence and model-expansion features (CogVideoX-5B, Qwen2.5-VL smart timeline, DiffRhythm+ upgrade) that close remaining gaps in the creative toolbox
4. Deferred items from Wave M watch list that have now stabilised (DiffRhythm 2, LightX2V Wan2.2 I2V A14B 4-step)

---

## Wave N1 — Inference Acceleration (v1.39.0)

**Goal**: Make all Wan2.2 endpoints from Wave M usable on mid-range consumer hardware without quality degradation. FastVideo adds sparse distillation to the T2V/TI2V path; LightX2V adds quantization + 4-step step-distillation to the I2V A14B path.
**New required deps**: `fastvideo` ≥0.1 (Apache-2), `lightx2v` (Apache-2)
**New routes**: ~10

### OSS Discoveries — Inference Acceleration

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| N1.1 | **FastVideo sparse distillation** (hao-ai-lab, Apache-2, April 2025) — unified post-training and real-time inference framework for accelerated video generation. Ships distilled `FastWan2.1-T2V-1.3B` and `FastWan2.2-TI2V-5B` models that achieve **>50× denoising speedup** over baseline Wan2.2 via sparse-attention distillation. Demonstrated 5-second 1080P video in 4.5 s on a single 4090. `pip install fastvideo`; supports Windows, Linux, macOS. Integration: add `?engine=fast` query flag on `POST /generate/wan2.2/t2v` and `POST /generate/wan2.2/ti2v` that swaps the `VideoGenerator` backend to `FastVideo/FastWan2.2-TI2V-5B-Diffusers`. Quality parity for most prompts; falls back to baseline Wan2.2 if fast model is not downloaded. | `POST /generate/wan2.2/t2v?engine=fast`, `POST /generate/wan2.2/ti2v?engine=fast`, `GET /generate/wan2.2/fast/info` | `core/gen_video_fastvideo.py` | `fastvideo` ≥0.1 (Apache-2) — new dep | M | Apache-2 | >50× speedup on existing Wan2.2 routes; enables real-time 1080P generation on single 4090 |
| N1.2 | **LightX2V quantization + step distillation** (ModelTC, Apache-2, April 2026) — lightweight video generation inference framework offering FP8/INT8 quantization, 4-step step-distilled models (Wan2.2-I2V-A14B in 4 steps instead of 50), sparse attention (≈1.5× additional speedup vs FP8 alone), and disaggregated deployment for multi-GPU. The latest `Wan2.2-I2V-A14B-4step-720p` weights (April 20, 2026) are trained on a high-quality 720P dataset with low-noise algorithm for better fine-grained detail. Primary impact: Wan2.2 I2V A14B, which previously required 80 GB VRAM at full precision, becomes usable on 24 GB cards at 4-step FP8. Integration: `GET /generate/wan2.2/i2v` gains `?quant=fp8&steps=4` params; `check_lightx2v_available()` validates the LightX2V package and distilled weights. | `GET /generate/wan2.2/i2v?quant=fp8&steps=4`, `GET /generate/wan2.2/i2v/quantization/info`, `GET /generate/wan2.2/i2v/backends` | `core/gen_video_lightx2v.py` | `lightx2v` (Apache-2) — new dep; `Wan2.2-I2V-A14B-4step-720p` weights via HuggingFace | M | Apache-2 | A14B I2V on 24 GB cards via 4-step FP8; up to 42× acceleration combined with CFG distillation |

---

## Wave N2 — Scene Intelligence (v1.40.0)

**Goal**: Add per-frame spatial understanding to OpenCut. SAM2 enables pixel-perfect video masking and object tracking from user prompts; Depth-Anything-V2 generates per-frame depth maps that feed into depth-of-field, parallax, and smart-reframe effects.
**New required deps**: `sam2` ≥0.4 (Apache-2), `depth_anything_v2` (Apache-2)
**New routes**: ~14

### OSS Discoveries — Scene Intelligence Primitives

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| N2.1 | **SAM 2.1 video object segmentation** (Meta FAIR, Apache-2, ECCV 2024) — foundation model for promptable visual segmentation in images and videos. Accepts click, box, or mask prompts on any frame and propagates the segmentation mask throughout the entire video in real time. SAM 2.1 (Sep 2024) is the improved checkpoint suite: 4 sizes (Tiny 38M@91FPS → Large 224M@39FPS on A100). Practical workflows: (a) AI background removal / green-screen replacement without a studio, (b) per-object colour grade, (c) selective blur/mosaic, (d) AI rotoscoping for motion graphics compositing. Renders exported masks as alpha-channel video, matted video, or COCO-format JSON for downstream compositing. Installation: `git clone facebookresearch/sam2 && pip install -e .`; Windows: WSL recommended for CUDA kernel compilation. | `POST /video/segment/sam2`, `POST /video/segment/sam2/propagate`, `GET /video/segment/sam2/info` | `core/segment_sam2.py` | `sam2` ≥0.4 (Apache-2) — new dep; checkpoints: sam2.1_hiera_small (46M, recommended default) | L | Apache-2 | Per-frame object tracking + mask propagation; closes Adobe After Effects Roto Brush gap locally |
| N2.2 | **Depth-Anything-V2 depth maps** (DepthAnything, Apache-2, NeurIPS 2024) — monocular depth estimation foundation model; accepts a single frame (or video), produces a per-pixel depth map. Available in 4 sizes (Small 24M → Large 335M); all run on GPU or CPU. Primary workflows in OpenCut: (a) upgrades AI CineFocus rack-focus (L3.5) with a second depth engine alongside Depth Pro — Depth-Anything-V2 is faster (real-time on GPU) while Depth Pro is more metric-accurate; (b) **parallax video effect**: separate foreground/background layers by depth and apply independent motion to each, creating a simulated camera-movement 2.5D effect; (c) **smart vertical-to-horizontal reframe**: depth-guided subject isolation for platform-aware cropping. `POST /video/depth/estimate` returns a float32 depth map video; `POST /video/depth/parallax` renders the 2.5D effect directly. | `POST /video/depth/estimate`, `POST /video/depth/parallax`, `POST /video/cinefocus/render?engine=depth_anything`, `GET /video/depth/info` | `core/depth_anything_v2.py` | `depth_anything_v2` (Apache-2) — new dep; Small model 24M params, CPU-capable | M | Apache-2 | Real-time depth maps; enables parallax 2.5D, smart reframe, depth-guided compositing |
| N2.3 | **SAM2 + depth compositor pipeline** — wires N2.1 and N2.2 together into a single `POST /video/compose/depth_segment` endpoint: (1) SAM2 segments the subject(s), (2) Depth-Anything-V2 estimates depth, (3) compositor combines masks + depth to produce a layered composite with configurable per-layer effects (colour grade, blur, motion parallax, replace background). This closes the biggest remaining gap between OpenCut and professional compositing tools (Adobe After Effects, DaVinci Fusion) for single-subject video editing. No new deps beyond N2.1 + N2.2. | `POST /video/compose/depth_segment`, `GET /video/compose/depth_segment/info` | `core/compose_depth_segment.py` | Chains N2.1 SAM2 + N2.2 Depth-Anything-V2 — no new dep | M | Combined (Apache-2) | End-to-end compositing pipeline; closes After Effects Roto + Depth gap locally |

---

## Wave N3 — Content Intelligence + Model Expansion (v1.41.0)

**Goal**: Add a second T2V model family for wider hardware coverage (CogVideoX-5B on RTX 3060), add VLM-powered smart timeline analysis (Qwen2.5-VL), and promote DiffRhythm+ (now stable) as an upgrade to the Wave M DiffRhythm v1.2 backend.
**New required deps**: `diffusers` ≥0.32 (already present for FLUX Kontext), `qwen_vl_utils` (Apache-2)
**New routes**: ~16

### OSS Discoveries — Content Intelligence + Model Expansion

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| N3.1 | **CogVideoX-5B T2V + I2V** (THUDM/Zhipu AI, Apache-2, Aug 2024 / updated Mar 2025) — second T2V family alongside Wan2.2. Key differentiators: (a) runs on **RTX 3060 (12 GB VRAM)** — far more accessible than Wan2.2 TI2V-5B (16 GB) or A14B (80 GB); CogVideoX-2B even runs on GTX 1080TI; (b) CogVideoX1.5-5B generates **10-second videos** at higher resolution with I2V support at any resolution; (c) DDIM Inverse support enables non-destructive video editing (invert → edit → re-generate); (d) LoRA fine-tuning on a single 4090 via `cogvideox-factory`; (e) available via diffusers (no new framework). Ships as a second backend in `/generate/backends` alongside Wan2.2; `POST /generate/cogvideox` for T2V, `POST /generate/cogvideox/i2v` for I2V, `POST /generate/cogvideox/invert` for DDIM inversion. Aesthetic: more cinematic/stylised vs Wan2.2's photorealistic. | `POST /generate/cogvideox`, `POST /generate/cogvideox/i2v`, `POST /generate/cogvideox/invert`, `GET /generate/cogvideox/info` | `core/gen_video_cogvideox.py` | `diffusers` ≥0.32 (already present) + CogVideoX-5B weights via HuggingFace (~18 GB) | L | Apache-2 | Second T2V aesthetic; accessible on 12 GB GPUs; 10s videos; DDIM inversion for video editing |
| N3.2 | **Qwen2.5-VL smart timeline analysis** (Alibaba/QwenLM, Apache-2, Sep 2024) — vision-language model optimised for visual understanding; accepts video clips and natural-language questions, returns structured analysis. In OpenCut: powers a `POST /analyze/video/vl` endpoint that answers questions like "describe each scene in this video", "identify all text visible on screen", "list all products shown", "rate the visual quality of each clip". Returns structured JSON with timestamped answers. Feeds two downstream features: (a) AI-assisted search across a clip library by natural-language query (`POST /library/search/vl`); (b) auto-chapter generation from semantic scene descriptions (complements L3.2 trailer gen). Models: Qwen2.5-VL-7B recommended; Qwen2.5-VL-3B for low-spec machines. Available via `pip install transformers qwen_vl_utils`. | `POST /analyze/video/vl`, `POST /library/search/vl`, `POST /analyze/video/chapters`, `GET /analyze/video/vl/info` | `core/analyze_vl_qwen.py` | `qwen_vl_utils` (Apache-2) + `transformers` (already present) + Qwen2.5-VL-7B weights (~16 GB) | M | Apache-2 | VLM-powered content understanding; closes Descript AI Scenes gap; enables natural-language clip search |
| N3.3 | **DiffRhythm+ music upgrade** (ASLP-lab, Apache-2, July 2025) — improved version of Wave M's DiffRhythm v1.2 backend. Key improvements in DiffRhythm+ (`arXiv:2507.12890`): better style control fidelity, improved voice/instrument separation in generated songs, stronger adherence to LRC lyric timing, and support for longer compositions. Ships as a drop-in upgrade to `core/music_diffrhythm.py`: add `DiffRhythm+` as a second variant in `GET /audio/music/diffrhythm/info` alongside `v1.2`; route `POST /audio/music/diffrhythm` accepts `?model=v1.2` or `?model=plus` (default: whichever is downloaded, preferring `plus`). Backwards compatible — no route changes. | `POST /audio/music/diffrhythm?model=plus`, `GET /audio/music/diffrhythm/models` | `core/music_diffrhythm.py` (extend existing) | DiffRhythm+ weights via HuggingFace (Apache-2) — no new pip dep | M | Apache-2 | Drop-in music quality upgrade for Wave M users; better lyric timing and style fidelity |
| N3.4 | **Sesame CSM-1B conversational speech** (Sesame AI Labs, Apache-2, March 2025) — 1B-parameter speech generation model that produces contextually-aware conversation audio. Unlike Chatterbox (single utterance + emotion) and Kokoro (TTS from text), CSM accepts a conversation context (prior speaker audio + text segments) and generates the next utterance in a natural dialogue style with consistent speaker identity. Primary use case: generating realistic multi-speaker dialogue for explainer videos, podcasts, and AI avatar conversations. Native Transformers support as of v4.52.1 (`from transformers import CsmForConditionalGeneration`). Gated: requires accepting Meta Llama Community License for the Llama-3.2-1B backbone — add `csm_llama_ack` timestamp to `config.json`; consent shown on first activation. English-only. | `POST /audio/speech/csm`, `GET /audio/speech/csm/context`, `GET /audio/speech/csm/info` | `core/tts_csm.py` | `transformers` ≥4.52.1 (already present) + CSM-1B weights + Llama-3.2-1B weights (both gated via HuggingFace, requires Meta Community License accept) | M | Apache-2 (code + CSM weights); Meta Llama Community License (Llama-3.2-1B backbone, gated) | Only local model producing contextual multi-speaker dialogue audio; closes ElevenLabs Conversations gap |

---

## Wave N: N-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave N Role |
|------|-----|---------|----------|-------------|
| SAM 2.1 | Meta FAIR | Apache-2 | Video Seg | N2.1 — Video object segmentation + tracking |
| Depth-Anything-V2 | DepthAnything | Apache-2 | Depth Est | N2.2 — Monocular depth maps for compositing |
| FastVideo | hao-ai-lab | Apache-2 | Inference Acc | N1.1 — >50× speedup for Wan2.2 T2V/TI2V |
| LightX2V | ModelTC | Apache-2 | Inference Acc | N1.2 — 4-step FP8 Wan2.2 I2V A14B |
| CogVideoX-5B | THUDM/Zhipu AI | Apache-2 | T2V | N3.1 — Second T2V family; 12 GB GPU; 10s videos |
| CogVideoX-2B | THUDM/Zhipu AI | Apache-2 | T2V | N3.1 — Ultra-low-VRAM fallback for older GPUs |
| Qwen2.5-VL | QwenLM/Alibaba | Apache-2 | VLM | N3.2 — Smart timeline analysis + clip search |
| DiffRhythm+ | ASLP-lab | Apache-2 | Music Gen | N3.3 — Quality upgrade for Wave M DiffRhythm |
| CSM-1B | Sesame AI Labs | Apache-2 + Meta Llama CL | Speech | N3.4 — Contextual multi-speaker dialogue TTS |
| AudioCraft (MusicGen) | Meta/facebookresearch | Code MIT, weights CC-BY-NC | Music Gen | NOT ADOPTED — weights non-commercial |
| SUPIR | SupPixel Pty Ltd | Custom non-commercial | Upscaling | NOT ADOPTED — non-commercial weights |
| CodeFormer | S-Lab NTU | S-Lab License (non-commercial) | Face Restore | NOT ADOPTED — non-commercial only |
| Moshi | Kyutai Labs | Multiple licence subdirs | Speech | Watch list — licence structure requires per-component audit; Wave O |

---

## Wave N: Competitive Gap Closure

| Gap | Competitor | Wave N Feature | Closes? |
|-----|-----------|---------------|---------|
| Video generation on mid-range GPUs (12 GB) | Runway Gen-4, Pika 2.0 (cloud-only) | N3.1 CogVideoX-5B | Y — local T2V on RTX 3060 |
| Video generation acceleration (consumer) | Sora, Luma Dream Machine (cloud) | N1.1 FastVideo | Y — >50× speedup on 4090 |
| A14B I2V on consumer 24 GB cards | — | N1.2 LightX2V 4-step FP8 | Y — 80 GB → 24 GB via quantization |
| Per-frame video masking (rotoscoping) | After Effects Roto Brush | N2.1 SAM2 | Y — prompt-driven, propagates entire video |
| Depth-of-field + parallax compositor | DaVinci Fusion, AE Camera Lens Blur | N2.2 + N2.3 | Y — real-time depth maps + layered composite |
| Natural-language clip search | Frame.io AI, Descript AI Scenes | N3.2 Qwen2.5-VL | Y — semantic video search across clip library |
| Contextual multi-speaker dialogue audio | ElevenLabs Conversations, Synthesia | N3.4 CSM-1B | Y — local, context-aware, free |

---

## Wave N Gotchas

- **SAM2 CUDA kernel on Windows**: SAM2 requires compiling a custom CUDA extension (`pip install -e .` in the cloned repo). On Windows, WSL2 with Ubuntu is the path of least resistance. Without WSL, some post-processing features (connected-component labels) may fail; the mask prediction itself still works. Document this in the N2.1 503 install hint with a fallback message.
- **SAM2 model size**: `sam2.1_hiera_small` (46M, 85 FPS) is the recommended default — good accuracy/speed balance. `sam2.1_hiera_large` (224M, 40 FPS) is better for complex occlusion cases. Gate size selection via `?model=small|base|large` param; default `small`.
- **FastVideo VSA kernel**: The Video Sparse Attention kernel requires a build step (`uv pip install fastvideo` handles it on supported platforms). If VSA fails to compile, FastVideo falls back to FlashAttention which is still faster than baseline but not 50×. Document the `FASTVIDEO_ATTENTION_BACKEND` env var.
- **LightX2V Windows support**: LightX2V was developed primarily for Linux and server deployments. On Windows, the quantization kernels may require manual CUDA toolkit alignment. Note WSL2 as the recommended path for N1.2 on Windows hosts; flag this in the `check_lightx2v_available()` output with a `{windows_note: true}` field.
- **CogVideoX weight size**: CogVideoX-5B is ~18 GB; CogVideoX1.5-5B is ~20 GB; CogVideoX-2B is ~6 GB. Follow the same pre-flight size check pattern as FLUX Kontext (M2.4): return `{available: false, reason: "weights_not_downloaded", size_gb: N}` and require explicit download confirmation.
- **CogVideoX prompt length**: The model is trained on long, detailed prompts. Add a `POST /generate/cogvideox/enhance_prompt` step (using GLM-4 or local LLM via Ollama) analogous to CogVideoX's own `convert_demo.py` before generating. Short prompts produce noticeably weaker results.
- **Qwen2.5-VL VRAM vs CPU**: Qwen2.5-VL-7B is the quality-recommended model (~16 GB weights) but Qwen2.5-VL-3B (~8 GB) runs comfortably on a 12 GB card. Add a `?model=7b|3b` param; default `3b` to ensure it works on the widest hardware range.
- **CSM Llama dependency gate**: CSM requires Llama-3.2-1B weights which are gated on HuggingFace (requires acceptance of Meta Llama Community License). `check_csm_available()` must verify both the CSM weights AND the Llama-3.2-1B weights are present locally before returning `available: true`. The consent prompt should display the Meta Llama Community License URL. Store `csm_llama_ack` timestamp in `config.json` alongside `moonshine_multilingual_ack`.
- **AudioCraft / MusicGen weights are CC-BY-NC 4.0**: The code is MIT but the model weights are non-commercial. Do not adopt regardless of audio quality. ACE-Step (L2.2) and DiffRhythm (M1.3) are the correct alternatives for music generation.
- **DiffRhythm+ maturity**: DiffRhythm+ paper (`arXiv:2507.12890`) was published July 2025. Confirm production-stable weights are available on HuggingFace before shipping N3.3; if not, fall back to v1.2 and leave plus as a `?model=plus` stub returning 503 with install hint.

---

## Wave N Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.39.0 | 2027-Q1 | N1.1 FastVideo, N1.2 LightX2V |
| v1.40.0 | 2027-Q2 | N2.1 SAM2, N2.2 Depth-Anything-V2, N2.3 depth compositor pipeline |
| v1.41.0 | 2027-Q2 | N3.1 CogVideoX-5B, N3.2 Qwen2.5-VL, N3.3 DiffRhythm+, N3.4 CSM-1B |

---

## Wave N: Not Adopted / Deferred

- **AudioCraft / MusicGen / AudioGen** (Meta, facebookresearch/audiocraft) — Code: MIT, but model weights: CC-BY-NC 4.0 (non-commercial). OpenCut cannot ship CC-BY-NC default-on features for commercial workflows. ACE-Step (L2.2) is the correct alternative for music generation; no gap remains.
- **SUPIR image restoration** (SupPixel Pty Ltd) — Custom SUPIR licence, non-commercial only; commercial use requires written permission from SupPixel. NOT viable. The Wave L1.4 upscaling hub (RealESRGAN + BSRGAN) covers the upscaling use case under permissive licences.
- **CodeFormer face restoration** (S-Lab NTU) — S-Lab License 1.0, explicitly non-commercial. NOT viable. GFPGAN (already in OpenCut, L1.7) covers the face restoration use case.
- **Moshi** (Kyutai Labs) — Multiple licence files across subdirectories; per-component licence audit required before adoption. Watch list for Wave O pending audit result.
- **Sesame CSM multilingual** — CSM-1B is English-only; the model has some non-English capacity from training data contamination but is not officially supported. Multilingual speech generation covered by Chatterbox-Multilingual (M1.1).
- **CogVideoX-2B GTX 1080TI ultra-low-spec mode** — GTX 1080TI (11 GB) can technically run CogVideoX-2B but with very slow inference (~20+ minutes per clip). Adding a dedicated ultra-low-spec mode adds complexity for marginal value. Default to CogVideoX-5B (12 GB / RTX 3060+) and document 2B as a community-supported variant.
- **Wan2.2 A14B single-card full-precision** — 80 GB VRAM requirement makes single-card consumer deployment infeasible. LightX2V N1.2 FP8 4-step is the practical path for A14B on 24 GB. Wave O may revisit with INT4 GGUF quantization.
- **LTX-2 audio-video generation** (Lightricks) — LightX2V supports LTX-2 (Jan 2026). LTX-2 licence TBD at time of this survey. Monitor for Wave O.

---

## Wave N Sources

- **SAM 2 / SAM 2.1** — [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (Apache-2, July 2024; SAM 2.1 Sep 2024); [arXiv:2408.00714](https://arxiv.org/abs/2408.00714); 4 model sizes 38M–224M
- **Depth-Anything-V2** — [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2, June 2024); [NeurIPS 2024](https://arxiv.org/abs/2406.09414); Small 24M → Large 335M
- **FastVideo** — [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo) (Apache-2, April 2025); `FastWan2.2-TI2V-5B-Diffusers` distilled model; `pip install fastvideo`; Windows supported; real-time 1080P demo March 2026
- **LightX2V** — [ModelTC/lightx2v](https://github.com/ModelTC/lightx2v) (Apache-2); `Wan2.2-I2V-A14B-4step-720p` weights April 20, 2026; FP8 + NVFP4 quantization; combined 42× acceleration
- **CogVideo / CogVideoX** — [THUDM/CogVideo](https://github.com/THUDM/CogVideo) (Apache-2, Aug 2024); CogVideoX-5B on RTX 3060; CogVideoX1.5-5B for 10s videos; diffusers native; [arXiv:2408.06072](https://arxiv.org/abs/2408.06072)
- **Qwen2.5-VL** — [QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) (Apache-2, Sep 2024); 3B and 7B consumer models; `pip install transformers qwen_vl_utils`
- **DiffRhythm+** — [ASLP-lab/DiffRhythm](https://github.com/ASLP-lab/DiffRhythm) (Apache-2); DiffRhythm+ [arXiv:2507.12890](https://arxiv.org/abs/2507.12890) (July 2025)
- **Sesame CSM-1B** — [SesameAILabs/csm](https://github.com/SesameAILabs/csm) (Apache-2, March 2025); HuggingFace native as of `transformers` v4.52.1; requires Llama-3.2-1B backbone (Meta Community License, gated)
- **AudioCraft** — [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) (code MIT, weights CC-BY-NC 4.0); NOT adopted — weights non-commercial
- **SUPIR** — [Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR) — Custom SUPIR licence, non-commercial; NOT adopted
- **CodeFormer** — [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) — S-Lab License 1.0, non-commercial; NOT adopted

---

# Wave O — TTS Expansion + Fast Video + AI Music Creation (v1.42.0 → v1.44.0)

**Updated**: 2026-04-24
**Baseline**: v1.41.0 (post-Wave N; CogVideoX-5B, SAM2, Depth-Anything-V2, Qwen2.5-VL, FastVideo, LightX2V, DiffRhythm+, CSM-1B all shipped)
**Research pass**: April 2026 GitHub OSS survey — Dia, Parler-TTS, LTX-Video, YuE, LTX-2, HunyuanVideo, F5-TTS, Mochi-1, Stable Audio Open

This wave covers three distinct domains:
1. Two new TTS models that fill gaps in the voice synthesis suite: Dia delivers production-quality multi-speaker scripted dialogue with nonverbal sounds; Parler-TTS adds natural-language voice description as a new interaction paradigm distinct from voice-cloning (Chatterbox) and preset voices (Kokoro).
2. LTX-Video (LTXV 0.9.8) as a fourth T2V engine — the fastest DiT-based model available under a fully permissive Apache-2 licence, with up to 60-second video, multi-keyframe conditioning, and I2V support.
3. YuE — the first fully open-source (Apache-2) lyrics-to-full-song model capable of generating complete songs with vocals and accompaniment, filling the only remaining major audio creation gap in OpenCut.

---

## Wave O1 — Enhanced TTS Voice Suite (v1.42.0)

**Goal**: Add two TTS models that cover usage patterns not addressed by Chatterbox (audio-prompt voice clone), Kokoro (style-preset TTS), or CSM-1B (dialogue context): (a) scripted multi-speaker dialogue with nonverbal sounds for podcast/explainer workflows; (b) natural-language voice description for any-voice TTS without a reference audio clip.
**New required deps**: `dia` (Apache-2), `parler_tts` ≥0.1 (Apache-2)
**New routes**: ~12

### OSS Discoveries — TTS Voice Suite

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| O1.1 | **Dia 1.6B dialogue TTS** (Nari Labs, Apache-2, April 2025) — 1.6B parameter model that generates fully-scripted multi-speaker dialogue from a transcript using `[S1]`/`[S2]` speaker tags. Unique capabilities: (a) nonverbal generation `(laughs)`, `(coughs)`, `(sighs)`, `(applause)`, etc. — first local model to do this convincingly; (b) voice cloning from 5–10s reference audio; (c) native HuggingFace Transformers support (`DiaForConditionalGeneration`); (d) 4.4 GB VRAM on bfloat16. English only. Dia2 (the successor) also released Nov 2025 on GitHub. Use case: scripted podcast creation, explainer video narration with two presenters, training video VO. `POST /audio/speech/dia` accepts array of `{speaker: "S1"|"S2", text: "...", nonverbals: ["(laughs)"]}` entries and returns a single rendered audio file. | `POST /audio/speech/dia`, `POST /audio/speech/dia/preview`, `GET /audio/speech/dia/info` | `core/tts_dia.py` | `dia` (Apache-2) — new dep; `Dia-1.6B-0626` weights via HuggingFace (~3.2 GB) | M | Apache-2 | Two-speaker scripted dialogue TTS with nonverbal sounds; closes Descript Overdub gap locally |
| O1.2 | **Parler-TTS natural language voice description** (HuggingFace, Apache-2, Aug 2024) — TTS model that generates speech matching a free-text voice description: "A female speaker delivers animated speech at a moderate pace with very clear audio." Both code and weights are fully permissive Apache-2, trained on 45k hours of audiobook data. Ships Mini (880M, ~2 GB) and Large (2.3B, ~5 GB) variants. 34 named speaker presets for consistent cross-generation voice identity. SDPA + Flash Attention 2 for fast inference; `torch.compile` compatible. Use case: creating a branded AI narrator by describing the desired voice in plain English, without needing a reference audio clip. `POST /audio/speech/parler` accepts `{text: "...", description: "..."}` and returns audio. `GET /audio/speech/parler/speakers` returns the 34 named preset speakers. | `POST /audio/speech/parler`, `GET /audio/speech/parler/speakers`, `GET /audio/speech/parler/info` | `core/tts_parler.py` | `parler_tts` ≥0.1 (Apache-2) — new dep; `parler-tts-mini-v1` weights (~2 GB) | S | Apache-2 | Natural language voice description TTS; closes ElevenLabs voice design gap without API keys |

---

## Wave O2 — LTX-Video: Fast Multi-Keyframe Generation (v1.43.0)

**Goal**: Add LTX-Video (LTXV 0.9.8) as a fourth T2V/I2V/V2V engine. LTX-Video is the fastest production-ready DiT-based video model under a fully permissive Apache-2 licence, with unique multi-keyframe conditioning that enables storyboard-to-video workflows not possible with Wan2.2 or CogVideoX.
**New required deps**: None — LTX-Video uses `diffusers` (already present)
**New routes**: ~14

### OSS Discoveries — Fast Multi-Keyframe Video

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| O2.1 | **LTX-Video LTXV 0.9.8 T2V/I2V** (Lightricks, Apache-2, July 2025 for 0.9.8) — fastest Apache-2-licenced DiT video generation model. Key capabilities: (a) **up to 60 seconds** of video with LTXV-13B (the distilled long-form variant); (b) very low latency — "real-time" for short clips on a 4090; (c) native diffusers support (`pip install -e .`); (d) T2V, I2V, video extension forward/backward, video-to-video. Ships as `/generate/ltxv/t2v` and `/generate/ltxv/i2v` following the same pattern as existing Wan2.2 routes. Backend toggleable via `GET /generate/backends`. | `POST /generate/ltxv/t2v`, `POST /generate/ltxv/i2v`, `GET /generate/ltxv/info` | `core/gen_video_ltxv.py` | `diffusers` ≥0.32 (already present) + LTXV-2B or LTXV-13B weights (~6 GB / ~25 GB) | M | Apache-2 | Fastest Apache-2 video model; low latency previews before committing to Wan2.2 generation |
| O2.2 | **LTX-Video multi-keyframe storyboard-to-video** — unique LTXV capability not present in any other Wave model: multi-keyframe conditioning accepts 2+ reference images as anchor frames and generates the video that flows between/through them. Use case: the user uploads a storyboard (sequence of key images) and LTX-Video generates a coherent video that matches each frame at the designated timecode. `POST /generate/ltxv/keyframes` accepts `[{time_sec: N, image_b64: "..."}]` and returns a video whose frames match each keyframe at the specified timestamp. Builds on LTX-Video's IC-LoRA control model support. | `POST /generate/ltxv/keyframes`, `GET /generate/ltxv/keyframes/info` | `core/gen_video_ltxv.py` (extend) | `diffusers` + IC-LoRA control model weights via HuggingFace (Apache-2); no new pip dep | M | Apache-2 | Storyboard-to-video workflow; closes Runway Act-One / Kling keyframe gap |
| O2.3 | **LTX-Video LoRA fine-tuning pipeline** — integrates with the [LTX-Video-Trainer](https://github.com/Lightricks/LTX-Video-Trainer) to expose a `POST /train/ltxv/lora` endpoint that fine-tunes LTXV on a user-provided set of reference videos (brand style, character consistency, motion style). Same pattern as CogVideoX LoRA (N3.1) but for LTXV's aesthetics. Requires the user to provide 10–30 reference clips; training runs locally in the background as an async job. | `POST /train/ltxv/lora`, `GET /train/ltxv/lora/status`, `GET /train/ltxv/lora/list` | `core/train_ltxv_lora.py` | `ltxv_trainer` via `pip install git+https://github.com/Lightricks/LTX-Video-Trainer` (Apache-2) — new dep | L | Apache-2 | Brand-style video LoRA training on consumer hardware; closes Runway Explore style fine-tuning gap |
| O2.4 | **LTX-Video video extension** — LTXV supports forward and backward temporal extension: given an existing video clip, generate N more seconds in the forward direction (extend the story) or backward direction (generate a prequel). Exposed as `POST /generate/ltxv/extend` with `{video_b64: "...", direction: "forward"|"backward", duration_sec: N}`. Use case: a short social clip extended into a long-form edit, or a generated clip extended at both ends for B-roll material. | `POST /generate/ltxv/extend`, `GET /generate/ltxv/extend/info` | `core/gen_video_ltxv.py` (extend) | diffusers (already present) | S | Apache-2 | Video temporal extension; closes Runway Extend Clip gap |

---

## Wave O3 — AI Music Creation + Platform Hardening (v1.44.0)

**Goal**: Add YuE lyrics-to-full-song generation (the only missing music creation primitive now that ACE-Step covers instrumental and DiffRhythm covers sync-to-video); add GGUF quantization support as a cross-cutting platform improvement to make ultra-large models accessible on sub-16 GB hardware.
**New required deps**: `yue-inference` (Apache-2, optional heavy dep), `llama-cpp-python` ≥0.3 (MIT)
**New routes**: ~10

### OSS Discoveries — Music Creation + Platform Hardening

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| O3.1 | **YuE lyrics-to-full-song** (HKUST/M-A-P, Apache-2, January 2025) — open-source foundation model for lyrics2song: takes a genre-tag prompt + structured lyrics (with `[verse]`/`[chorus]`/`[bridge]` section labels) and generates a complete multi-minute song with a vocal track and a backing accompaniment track. Key capabilities: (a) multilingual — English, Mandarin, Cantonese, Japanese, Korean; (b) style transfer via ICL (dual-track mode: provide reference vocal + instrumental; model writes a new song in the same style); (c) LoRA fine-tuning for custom artist styles (June 2025); (d) incremental generation — complete a song session-by-session (verse first, then chorus). Hardware: 2 sessions (verse + chorus) fits in 24 GB VRAM. S1 model: YuE-s1-7B-anneal-en-cot (7B LM); S2 model: YuE-s2-1B-general (1B decoder). RT: 30s audio takes ~360s on RTX 4090. Ships as `POST /audio/music/yue`; integrates with OpenCut's job queue for long-running generation. | `POST /audio/music/yue`, `POST /audio/music/yue/icl`, `GET /audio/music/yue/info` | `core/music_yue.py` | YuE inference requirements: `transformers`, `flash-attn`, `xcodec-mini`; weights auto-downloaded from HuggingFace (~28 GB total for s1+s2) | L | Apache-2 | Only local model generating complete vocal+accompaniment songs from lyrics; closes Suno/Udio gap entirely |
| O3.2 | **GGUF quantization engine** — integrate `llama-cpp-python` as a CPU/low-VRAM fallback inference engine for any model that has a published GGUF checkpoint on HuggingFace. Primary targets: (a) Qwen2.5-VL (N3.2) — GGUF-Q4_K_M reduces from 16 GB to ~5 GB; (b) YuE-s1-7B GGUF Q4_K_M (community released, when available); (c) future LLM components. Adds `?quant=gguf_q4` param to routes that have GGUF-compatible backends; `check_gguf_available()` validates the `llama-cpp-python` package and the specific model GGUF file. Enables CPU-only operation for text-based AI features on machines without discrete GPU. | `GET /system/quantization/gguf/info`, `GET /system/quantization/gguf/models`, `POST /system/quantization/gguf/download` | `core/gguf_backend.py` | `llama-cpp-python` ≥0.3 (MIT) — new dep; GGUF model files via HuggingFace | M | MIT | CPU/low-VRAM fallback for all LLM/VLM components; enables OpenCut on laptops and integrated-graphics machines |
| O3.3 | **Multi-GPU task scheduler** — route large model inference tasks (Wan2.2 A14B, YuE 7B+1B, Mochi-1, CogVideoX1.5-5B 20 GB) to a multi-GPU pool automatically. Detects available GPU topology via `torch.cuda.device_count()` and tensor-parallel hints, maps models to device groups, and serialises requests through the existing `@async_job` queue to prevent GPU contention. Exposes `GET /system/gpu/topology` and `GET /system/gpu/allocation` for the UXP panel's resource monitor. Required prerequisite for future 70B+ model support and for running YuE full-song (4+ sessions requiring 80 GB) on dual A100 workstations. | `GET /system/gpu/topology`, `GET /system/gpu/allocation`, `POST /system/gpu/prefer` | `core/gpu_scheduler.py` | `torch` (already present); no new dep | M | N/A (platform code) | Enables large model use on multi-GPU workstations; prerequisite for 70B+ future models |

---

## Wave O: O-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave O Role |
|------|-----|---------|----------|-------------|
| Dia 1.6B | Nari Labs | Apache-2 | Dialogue TTS | O1.1 — Two-speaker scripted dialogue with nonverbals |
| Parler-TTS | HuggingFace | Apache-2 | TTS | O1.2 — Natural language voice description synthesis |
| LTX-Video 0.9.8 | Lightricks | Apache-2 | T2V/I2V | O2.1–O2.4 — Fast multi-keyframe; up to 60s; LoRA |
| YuE | HKUST/M-A-P | Apache-2 | Music Gen | O3.1 — Lyrics-to-full-song; vocal + backing track |
| LTX-2 | Lightricks | LTX-2 Community Licence | T2V (audio+video) | NOT ADOPTED — $10M revenue threshold for commercial |
| HunyuanVideo | Tencent-Hunyuan | Tencent Hunyuan Community Licence | T2V | NOT ADOPTED — EU/UK/South Korea excluded; geographic restrictions |
| F5-TTS | SWivid/X-LANCE | MIT code + CC-BY-NC weights | TTS | NOT ADOPTED — weights non-commercial (Emilia dataset) |
| Mochi-1 | Genmo | Apache-2 | T2V | Watch list — Apache-2 ✓ but single-GPU requires 60 GB VRAM; Wave P |
| Stable Audio Open | StabilityAI | Code MIT; model weights SA-CL (non-commercial) | Music Gen | NOT ADOPTED — weights non-commercial |
| Dia2 | Nari Labs | TBD | Dialogue TTS | Watch list — Dia2 released Nov 2025; licence audit required for Wave P |

---

## Wave O: Competitive Gap Closure

| Gap | Competitor | Wave O Feature | Closes? |
|-----|-----------|---------------|---------|
| Two-speaker scripted dialogue audio | Descript Overdub, ElevenLabs Studio | O1.1 Dia 1.6B | Y — nonverbal sounds + voice clone locally |
| Natural language voice design (no reference audio) | ElevenLabs Voice Design | O1.2 Parler-TTS | Y — free-text describes any voice |
| Fast preview T2V before committing to full render | — | O2.1 LTX-Video T2V | Y — near-real-time previews on 4090 |
| Storyboard-to-video (multi keyframe) | Runway Act-One, Kling Keyframe | O2.2 LTX-Video keyframes | Y — image sequence → coherent video |
| Video temporal extension | Runway Extend Clip | O2.4 LTX-Video extend | Y — forward + backward extension locally |
| Lyrics-to-full-song with vocals | Suno AI, Udio | O3.1 YuE | Y — open-weights, commercial OK, Apache-2 |
| CPU/laptop GPU model inference | Cloud-only TTS/VLM services | O3.2 GGUF engine | Y — Q4_K_M reduces 16 GB → 5 GB for VLM features |

---

## Wave O Gotchas

- **Dia English-only**: Dia 1.6B and Dia2 are trained primarily on English. Other languages may produce incorrect pronunciation or degraded quality. Document this in the `GET /audio/speech/dia/info` response as `languages: ["en"]` and link to Chatterbox-Multilingual (M1.1) for non-English use.
- **Dia nonverbal vocabulary**: Only the listed nonverbals produce reliable output; using arbitrary descriptors causes audio artifacts. Hard-code the validated list in `tts_dia.py` and expose it via `GET /audio/speech/dia/nonverbals`. Reject or warn on unlisted tags.
- **Parler-TTS voice consistency**: Random-prompt mode produces a different voice every generation; use a named speaker from the 34 presets to get consistent cross-generation identity. Document this in the UI with a speaker picker.
- **Parler-TTS prompt engineering**: "very clear audio" produces the highest quality output; "very noisy audio" produces noise. Expose a quality slider in the UI that appends the appropriate phrase automatically rather than expecting users to know this.
- **LTX-Video LTXV-13B weight size**: 25 GB weight file. Follow the same pre-flight pattern as FLUX Kontext: `{available: false, reason: "weights_not_downloaded", size_gb: 25}`. Default to LTXV-2B (~6 GB) for standard clips; LTXV-13B only for 60s long-form jobs.
- **LTX-2 NOT adopted**: LTX-2 (January 2026) is the successor to LTX-Video and adds synchronized audio+video generation, but ships under a custom community licence that requires a paid commercial agreement for entities with ≥$10M revenue. OpenCut cannot bundle LTX-2 for commercial use without guaranteeing every user meets the licence terms. Monitor for an Apache-2 re-release or equivalent permissive re-licensing before Wave P.
- **HunyuanVideo geographic restriction**: The Tencent Hunyuan Community Licence explicitly excludes EU, UK, and South Korea from the Territory. OpenCut ships globally including to EU/EEA users; bundling HunyuanVideo would mean European users are unintentionally unlicensed. Do not adopt under any circumstances until Tencent changes the licence.
- **F5-TTS CC-BY-NC weights**: The MIT code licence covers the inference framework, but the pre-trained weights were trained on the Emilia dataset under CC-BY-NC 4.0. The weights themselves are therefore non-commercial. Do not adopt. Note: if a user brings their own F5-TTS weights trained on permissively-licensed data, OpenCut can support the F5-TTS inference engine as a community plugin (via the Wave M plugin marketplace) without bundling the non-commercial weights.
- **YuE VRAM**: Full 4-session song (verse + pre-chorus + chorus + outro) requires 80 GB+ VRAM for parallel session generation. `check_yue_available()` must detect VRAM and cap `--run_n_segments` to 2 for cards with <24 GB, with a user notification. For multi-GPU setups (O3.3), YuE can be tensor-paralleled across 2× 40 GB cards.
- **YuE generation latency**: 30s of audio takes ~360s on an RTX 4090. For a full 4-session song that's ~24 minutes. Always run `POST /audio/music/yue` as a fully async background job with an SSE status stream. Disable the "Cancel" button mid-session to avoid partial-write corruption in the YuE xcodec_mini decoder.
- **GGUF weight availability**: Not all model architectures have community GGUF checkpoints yet. `core/gguf_backend.py` must handle a missing GGUF gracefully: return `{gguf_available: false, reason: "no_gguf_checkpoint", alternatives: ["download_full_weights"]}` rather than a 503. Track GGUF checkpoint status per model in a `gguf_manifest.json`.
- **Stable Audio Open weights**: The Stability AI `stable-audio-open-1.0` model code is MIT but the weights are under the Stability AI Community Licence which prohibits commercial use. Do not adopt. ACE-Step (L2.2) covers the ambient/sfx music generation use case under Apache-2.

---

## Wave O Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.42.0 | 2027-Q3 | O1.1 Dia, O1.2 Parler-TTS |
| v1.43.0 | 2027-Q3 | O2.1 LTX-Video T2V/I2V, O2.2 multi-keyframe, O2.3 LoRA, O2.4 video extend |
| v1.44.0 | 2027-Q4 | O3.1 YuE, O3.2 GGUF engine, O3.3 multi-GPU scheduler |

---

## Wave O: Not Adopted / Deferred

- **LTX-2** (Lightricks, Jan 2026) — Custom LTX-2 Community Licence; entities with annual revenues ≥$10M must obtain a separate paid commercial licence from Lightricks. OpenCut cannot verify revenue thresholds for individual users; this creates an unmanageable compliance burden. Monitor for Apache-2 re-release or equivalent. LTX-Video (0.9.8, Apache-2) adopted in its place.
- **HunyuanVideo** (Tencent-Hunyuan, Dec 2024) — Tencent Hunyuan Community Licence explicitly excludes the EU, UK, and South Korea. OpenCut cannot ship a model restricted by user geography. Hard pass unless Tencent releases under Apache-2.
- **F5-TTS** (SWivid/X-LANCE, 2024) — MIT inference code, CC-BY-NC 4.0 model weights (Emilia training data). The pre-trained weights cannot be used commercially. The F5-TTS architecture can be supported as a plugin inference engine for user-supplied custom weights via the Wave M plugin marketplace — document this in the plugin SDK.
- **Mochi-1** (Genmo, Nov 2024) — Apache-2 ✓ but requires approximately 60 GB VRAM for single-GPU inference in this repository. ComfyUI can reduce this to ~20 GB but that requires a different integration path. Watch list for Wave P pending a consumer-VRAM-optimised inference implementation (target: <16 GB with quantization).
- **Stable Audio Open** (StabilityAI, 2024) — Stability AI Community Licence (non-commercial for model weights). NOT viable. ACE-Step (L2.2) + DiffRhythm (M1.3) + YuE (O3.1) collectively cover all music generation use cases.
- **Dia2** (Nari Labs, Nov 2025) — Successor to Dia with unspecified improvements. Licence not yet audited at time of this survey. Dia 1.6B adopted; Dia2 added to watch list for Wave P after licence confirmation.
- **YuE full-song on consumer hardware** — 4+ session full song requires 80 GB VRAM. Deferred pending GGUF Q4 community release for YuE-s1-7B (GitHub issue open: `multimodal-art-projection/YuE#467`). Once GGUF is available, O3.2 GGUF engine can serve as the fallback.

---

## Wave O Sources

- **Dia 1.6B** — [nari-labs/dia](https://github.com/nari-labs/dia) (Apache-2, April 2025); Dia2: [nari-labs/dia2](https://github.com/nari-labs/dia2) (Nov 2025); HF: `nari-labs/Dia-1.6B-0626`; HF Transformers native (`DiaForConditionalGeneration`)
- **Parler-TTS** — [huggingface/parler-tts](https://github.com/huggingface/parler-tts) (Apache-2, Aug 2024); Mini 880M + Large 2.3B; [arXiv:2402.01912](https://arxiv.org/abs/2402.01912) (Lyth & King, Stability AI); `pip install git+https://github.com/huggingface/parler-tts.git`
- **LTX-Video** — [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) (Apache-2); LTXV 0.9.8 (July 2025): up to 60s, LTXV-13B; [arXiv:2501.00103](https://arxiv.org/abs/2501.00103); diffusers native; LTX-Video-Trainer for LoRA
- **LTX-2** — [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) (LTX-2 Community Licence, Jan 2026); audio+video; 4K/50FPS; NOT adopted — commercial revenue threshold restriction
- **YuE** — [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) (Apache-2, Jan 2025; Apache-2 relicensing confirmed Jan 30 2025); [arXiv:2503.08638](https://arxiv.org/abs/2503.08638); YuE-s1-7B + YuE-s2-1B; multilingual; LoRA fine-tuning (June 2025)
- **HunyuanVideo** — [Tencent-Hunyuan/HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo) (Tencent Hunyuan Community Licence, Dec 2024); geographic restrictions — NOT adopted
- **F5-TTS** — [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) (MIT code, CC-BY-NC 4.0 weights); [arXiv:2410.06885](https://arxiv.org/abs/2410.06885); weights non-commercial — NOT adopted
- **Mochi-1** — [genmoai/mochi](https://github.com/genmoai/mochi) (Apache-2, Nov 2024); 10B AsymmDiT; 60 GB VRAM single-GPU; watch list
- **Stable Audio Open** — (Stability AI, 2024); code MIT, weights Stability AI Community Licence (non-commercial) — NOT adopted

---

# Wave P — Identity-Consistent Video + SOTA T2I + Multimodal Intelligence (v1.45.0 → v1.47.0)

**Updated**: 2026-04-24
**Baseline**: v1.44.0 (post-Wave O; Dia, Parler-TTS, LTX-Video, YuE, GGUF engine, multi-GPU scheduler all shipped)
**Research pass**: April 2026 GitHub OSS survey — ConsisID, Allegro, HiDream-I1/E1, CogView4, Open-Sora 2.0, Qwen2.5-Omni, DepthCrafter, VoiceCraft, Open-Sora-Plan

This wave covers three distinct capabilities:
1. Video generation with **identity consistency** (a person's face held constant across the entire video from a single reference photo) and a new **lowest-VRAM T2V** option (9.3 GB) that unlocks consumer-grade GPUs for the first time.
2. A **SOTA text-to-image upgrade** — HiDream-I1 (17B Sparse DiT) and its companion instruction-based editor HiDream-E1, plus CogView4-6B as the first bilingual (English + Chinese) T2I model in the stack.
3. **Multimodal video intelligence** — Qwen2.5-Omni processes video frames + audio simultaneously and generates spoken narration audio as output, enabling automated video commentary and analysis workflows; and Open-Sora 2.0 as an 11B high-quality T2V completing the video generation tier.

---

## Wave P1 — Identity-Consistent Video + Efficient T2V (v1.45.0)

**Goal**: Add two video generation models addressing gaps not covered by any prior wave: (a) identity-preserving T2V where a specific person (from a reference face photo) appears consistently throughout the generated video — critical for brand ambassador, tutorial, and social content workflows; (b) a lightweight T2V at 9.3 GB VRAM, the lowest of any model in the stack, making T2V accessible to users with 12 GB cards for the first time.
**New required deps**: None — both models use `diffusers` (already present) ≥0.33
**New routes**: ~14

### OSS Discoveries — Identity-Consistent + Efficient T2V

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| P1.1 | **ConsisID identity-preserving T2V** (PKU-YuanGroup, Apache-2, CVPR 2025 Highlight) — tuning-free DiT-based identity-preserving text-to-video model built on CogVideoX-5B infrastructure. Accepts a **face reference image** (a photo of any person) and generates a video in which that person's face, identity, and appearance remain consistent across every frame throughout the clip. Uses a frequency decomposition approach to decouple identity features from motion, so the model generates natural dynamic motion while locking the face. Ships as `POST /generate/consisid/t2v` with `{prompt: "...", face_image_b64: "..."}`. Hardware: ~18 GB VRAM (same as CogVideoX-5B); diffusers 0.33.0+ native. Use case: create a video where a specific person is the subject (brand spokesperson, presenter, social avatar, training character) without any actor or green screen. | `POST /generate/consisid/t2v`, `POST /generate/consisid/preview`, `GET /generate/consisid/info` | `core/gen_video_consisid.py` | `diffusers` ≥0.33 (already present) + ConsisID weights (~18 GB) via HuggingFace | M | Apache-2 | Identity-preserving T2V; closes Runway "reference person" gap locally |
| P1.2 | **Allegro lightweight T2V + TI2V** (rhymes-ai, Apache-2, Oct 2024) — 2.8B DiT T2V model; the lowest-VRAM production T2V in the stack at 9.3 GB with `--enable_cpu_offload` (vs ~18 GB for Wan2.2/CogVideoX and ~6 GB for LTX-Video). Generates 6-second, 720×1280 video at 15 FPS. Allegro-TI2V variant adds first-frame + optional last-frame image conditioning for first-and-last-frame interpolation workflows. Both variants are in diffusers 0.32.0+. A Presto fine-tune (rhymes-ai) extends clips to longer durations. Use case: users with 12 GB VRAM cards can run T2V for the first time; first-and-last frame interpolation is unique in the stack. Ships as `/generate/allegro/t2v` and `/generate/allegro/ti2v`. | `POST /generate/allegro/t2v`, `POST /generate/allegro/ti2v`, `GET /generate/allegro/info` | `core/gen_video_allegro.py` | `diffusers` ≥0.32 (already present) + Allegro-T2V and Allegro-TI2V weights (~5 GB each) via HuggingFace | M | Apache-2 | Lowest-VRAM T2V (9.3 GB); first-and-last-frame interpolation; unlocks 12 GB GPU users |

---

## Wave P2 — SOTA Text-to-Image + Instruction Image Editing (v1.46.0)

**Goal**: Elevate the text-to-image tier with models that exceed FLUX.1-dev quality. HiDream-I1 is the new SOTA T2I at 17B parameters; HiDream-E1 adds natural-language instruction-based editing ("make the car red", "remove the person"). CogView4-6B adds the first Chinese-language T2I capability in the stack.
**New required deps**: HiDream-I1 requires `meta-llama/Meta-Llama-3.1-8B-Instruct` (same Meta Community Licence handling as CSM-1B — user opt-in gate)
**New routes**: ~16

### OSS Discoveries — SOTA T2I + Editing

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| P2.1 | **HiDream-I1 SOTA text-to-image** (HiDream.ai, MIT, April 2025) — 17B Sparse Diffusion Transformer text-to-image model that achieves the highest scores on DPG-Bench (85.89 overall), GenEval (0.83), and HPSv2.1 (33.82) among all open models — surpassing FLUX.1-dev, SD3-Medium, DALL-E 3, and Janus-Pro-7B. Ships in three inference-speed variants: Full (50 steps), Dev (28 steps), Fast (16 steps). Diffusers `HiDreamImagePipeline`. Requires `meta-llama/Meta-Llama-3.1-8B-Instruct` (~16 GB) as the text encoder backbone — same gate as CSM-1B; expose via the existing `llama_ack` user consent mechanism in `opencut/config.json`. Combined weight footprint: ~33 GB (17B DiT + Llama-3.1-8B). Use Fast variant as the default; Full for highest quality. | `POST /image/generate/hidream`, `GET /image/generate/hidream/info`, `GET /image/generate/hidream/variants` | `core/t2i_hidream.py` | `diffusers` ≥0.32 (already present) + HiDream-I1 weights (~17 GB) + `meta-llama/Meta-Llama-3.1-8B-Instruct` (Meta Community Licence, gated HF; user opt-in) | M | MIT (code + HiDream weights); Meta Community Licence (Llama backbone, gated) | SOTA T2I quality; closes Midjourney generation quality gap; surpasses FLUX.1-dev on all benchmarks |
| P2.2 | **HiDream-E1 instruction image editing** (HiDream.ai, MIT, July 2025) — instruction-based image editing companion to HiDream-I1: accepts an image + a natural language edit instruction and returns the edited image. Capabilities: style transfer, object addition/removal, color change, attribute modification, background swap. Reuses the HiDream-I1 weights + Llama backbone; no additional weight download once I1 is installed. Ships as `POST /image/edit/hidream` with `{image_b64: "...", instruction: "make the car bright red"}`. Use case: per-frame AI editing on video stills (combined with TokenFlow for video propagation, same pipeline as FLUX Kontext P2 from Wave M). | `POST /image/edit/hidream`, `POST /image/edit/hidream/batch`, `GET /image/edit/hidream/info` | `core/img_edit_hidream.py` | Same as P2.1 (no additional deps) + HiDream-E1 weights (~17 GB, separate model from I1) | S | MIT | Natural language image editing; complements FLUX Kontext (Wave M2.4); enables instruction-based per-frame editing workflow |
| P2.3 | **CogView4-6B bilingual text-to-image** (ZhipuAI/THUDM, Apache-2, March 2025) — 6B parameter text-to-image DiT with native bilingual support (English + simplified Chinese input). At 13 GB VRAM with CPU offload and int4 text encoder, it is the lightest full-quality T2I in the stack. Ships `CogView4Pipeline` in diffusers. Competitive with FLUX.1-dev on DPG-Bench (85.13 vs 83.79) and strong Chinese-language text accuracy. Unique value over HiDream-I1: (a) no gated Llama dependency, (b) 13 GB VRAM instead of 33 GB, (c) Chinese-language input. Fine-tunable with CogKit or finetrainers on a single 4090. | `POST /image/generate/cogview4`, `GET /image/generate/cogview4/info` | `core/t2i_cogview4.py` | `diffusers` ≥0.32 (already present) + CogView4-6B weights (~12 GB) | S | Apache-2 | Bilingual T2I (English + Chinese); 13 GB VRAM; no gated deps; competes with FLUX.1-dev quality |

---

## Wave P3 — Multimodal Video Intelligence + Open-Sora 2.0 (v1.47.0)

**Goal**: Add Qwen2.5-Omni for the first model that can both understand video (watching + listening) and generate audio narration as output simultaneously; and add Open-Sora 2.0 as the highest-quality Apache-2 T2V for creators who prioritise quality over hardware constraints.
**New required deps**: `transformers` update for Qwen2.5-Omni `Qwen2_5OmniModel` support
**New routes**: ~10

### OSS Discoveries — Multimodal Intelligence + Quality T2V

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| P3.1 | **Qwen2.5-Omni multimodal video narrator** (Alibaba Cloud, Apache-2, March 2025) — end-to-end multimodal model that accepts any combination of text, audio, images, and video clips as input and generates both text and natural speech audio as output. Unique OpenCut integration: `POST /analyze/video/narrate` accepts a video clip and a narration style (e.g. "documentary", "sports commentary", "educational explainer") and returns (a) a written commentary script, (b) a generated speech audio file voiced by Qwen2.5-Omni's Talker module. Also enables `POST /analyze/video/qa` for free-form questions about video content ("what is happening in this clip?" → spoken + written answer). Makes OpenCut the first local video editor with an AI that can watch a clip and narrate it. | `POST /analyze/video/narrate`, `POST /analyze/video/qa`, `GET /analyze/video/omni/info` | `core/multimodal_omni.py` | `transformers` update for `Qwen2_5OmniModel` + `Qwen2.5-Omni-7B` weights (~14 GB, Apache-2) via HuggingFace | M | Apache-2 | First model that watches+listens AND narrates video; unique in video editor ecosystem; closes Adobe Firefly audio-to-video description gap |
| P3.2 | **Open-Sora 2.0 high-quality T2V** (hpcaitech, Apache-2, March 2025) — 11B T2V model (the largest Apache-2 T2V in the stack) that benchmarks on-par with HunyuanVideo 11B and Step-Video 30B on VBench + Human Preference. Generates 5-second 720×1280 video. Training code and all checkpoints are fully open-source ($200K training cost documented). Ships as `/generate/opensora2/t2v` — the quality-tier T2V for users with >18 GB VRAM who want the best possible output. Also includes Open-Sora 1.3 (1B) as a lightweight variant for users with <12 GB VRAM; both models share the same backend. | `POST /generate/opensora2/t2v`, `POST /generate/opensora2/t2v/1b`, `GET /generate/opensora2/info` | `core/gen_video_opensora2.py` | `diffusers` or Open-Sora native inference code (Apache-2); 11B weights (~22 GB); 1B weights (~3 GB) | L | Apache-2 | Highest-quality Apache-2 T2V available; equals HunyuanVideo quality without geographic licence restrictions |
| P3.3 | **UXP panel v1.0 milestone** — complete the CEP → UXP transition targeting full parity on all Wave L through P features in the UXP panel. CEP reaches end-of-life in September 2026 with Adobe Premiere Pro; all users will need the UXP panel before that date. Milestone checklist: (a) all async job routes render progress in the UXP job queue panel; (b) GPU topology + GGUF model status exposed in UXP resource monitor; (c) all video preview/export buttons functional; (d) all audio/TTS playback controls in UXP. CEP panel moves to "legacy support" status (security fixes only) after this release. | `— (panel-only work, no new backend routes)` | UXP panel codebase (`panel-uxp/`) | No new Python deps | L | N/A (platform code) | CEP EOL Sept 2026; all users must migrate; this is the last major UXP gap-fill before EOL |

---

## Wave P: P-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave P Role |
|------|-----|---------|----------|-------------|
| ConsisID | PKU-YuanGroup | Apache-2 | Identity T2V | P1.1 — Face reference → consistent-identity T2V |
| Allegro | rhymes-ai | Apache-2 | T2V + TI2V | P1.2 — 9.3 GB VRAM T2V; first+last frame conditioning |
| HiDream-I1 | HiDream.ai | MIT | T2I | P2.1 — SOTA 17B T2I; surpasses FLUX.1-dev on all benchmarks |
| HiDream-E1 | HiDream.ai | MIT | Image Editing | P2.2 — Instruction-based image editing companion to I1 |
| CogView4-6B | ZhipuAI/THUDM | Apache-2 | T2I | P2.3 — Bilingual (EN+ZH) T2I; 13 GB VRAM; no gated deps |
| Qwen2.5-Omni | Alibaba Cloud | Apache-2 | Multimodal | P3.1 — Video understanding + audio narration generation |
| Open-Sora 2.0 | hpcaitech | Apache-2 | T2V | P3.2 — 11B SOTA quality T2V; equals HunyuanVideo quality |
| DepthCrafter | Tencent | Custom (non-commercial) | Depth Estimation | NOT ADOPTED — academic/research/education only |
| VoiceCraft | jasonppy | CC BY-NC-SA 4.0 code + Coqui model | TTS/Speech Edit | NOT ADOPTED — both code and weights non-commercial |
| Open-Sora-Plan | PKU-YuanGroup | MIT | T2V | NOT ADOPTED — overlapping with Open-Sora 2.0 + ConsisID |
| Mochi-1 | Genmo | Apache-2 | T2V | Watch list (carry-over from Wave O) — 60 GB VRAM |

---

## Wave P: Competitive Gap Closure

| Gap | Competitor | Wave P Feature | Closes? |
|-----|-----------|---------------|---------|
| Person-consistent video from reference photo | Runway Act-One, Kling AI Face Reference | P1.1 ConsisID | Y — face photo → identity-locked T2V locally |
| First-and-last-frame video interpolation | Runway, Kling Interpolation | P1.2 Allegro TI2V | Y — first + last frame → coherent transition video |
| T2V for 12 GB VRAM consumer GPUs | Cloud-only services | P1.2 Allegro (9.3 GB) | Y — RTX 3060 Ti / 4060 Ti now capable of T2V |
| SOTA T2I quality beyond FLUX.1 | Midjourney V7, Ideogram 3 | P2.1 HiDream-I1 | Y — best GenEval + DPG scores of any open model |
| Natural language image editing | Adobe Firefly, Photoshop Generative Fill | P2.2 HiDream-E1 | Y — instruction-based editing of any image locally |
| Chinese-language text-to-image | Cloud services (Tongyi Wanxiang) | P2.3 CogView4-6B | Y — native Chinese T2I; first in OpenCut stack |
| Video QA + spoken narration generation | OpenAI o1 Vision API | P3.1 Qwen2.5-Omni | Y — watch video + generate narration audio locally |
| SOTA T2V quality without HunyuanVideo licence | HunyuanVideo (restricted) | P3.2 Open-Sora 2.0 | Y — equal quality; Apache-2; no geographic restrictions |

---

## Wave P Gotchas

- **ConsisID face quality dependency**: ConsisID performs best with a high-quality, front-facing face photograph at 512×512 or larger. Low-resolution or side-profile inputs reduce identity consistency significantly. Add a pre-flight face detection check using the existing SAM2 (N2.1) face detection or a lightweight `face_recognition` check; reject inputs with no detected face rather than silently producing poor results.
- **ConsisID single-person only**: The current model is trained for single-identity generation. Multi-person identity preservation is not yet reliable. Document this clearly in `GET /generate/consisid/info` response as `max_identities: 1`. Monitor ConsisID for multi-identity updates in Wave Q.
- **Allegro 15 FPS output**: Allegro generates at 15 FPS natively. For 30 FPS delivery, use EMA-VFI (a frame interpolation model) as a post-processing step. Add an `--interpolate_fps: 30` option to `/generate/allegro/t2v` that pipelines EMA-VFI automatically. Note: EMA-VFI licence must be checked before bundling — it is MIT, so this is safe.
- **HiDream-I1 Llama-3.1 gate**: Same situation as CSM-1B (Wave N3.3). The Meta Llama 3.1 Community Licence requires a HuggingFace access token and `meta-llama/Meta-Llama-3.1-8B-Instruct` model access. Use the same `llama_ack` mechanism already in `opencut/config.json` — if the user has already accepted for CSM-1B, reuse that acceptance for HiDream-I1. If not accepted, `GET /image/generate/hidream/info` returns `{available: false, reason: "llama_gate_required", gate_url: "https://hf.co/meta-llama/Meta-Llama-3.1-8B-Instruct"}`.
- **HiDream-I1 weight size**: 17B Sparse DiT (~17 GB) + Llama-3.1-8B (~16 GB) = ~33 GB combined. If the user already has Llama-3.1-8B installed for CSM-1B or YuE, avoid re-downloading: detect via `HF_HOME` scan for `meta-llama/Meta-Llama-3.1-8B-Instruct`. In the pre-flight check, state `{llama_already_cached: true, total_new_download_gb: 17}`.
- **CogView4 prompt style**: CogView4 was trained on long synthetic descriptions; short prompts produce mediocre results. Add an automatic prompt expansion pass using Qwen2.5-VL (N3.2, already in the stack) before passing to CogView4; expose as an optional `?expand_prompt=true` parameter.
- **Qwen2.5-Omni audio output rate**: The Talker module generates audio in a streaming fashion — expose it via the existing SSE streaming infrastructure so the first audio chunk plays in the UXP panel before generation is complete. This gives a "real-time narration" feel for short clips.
- **Open-Sora 2.0 11B VRAM**: 11B T2V requires approximately 22 GB VRAM. This is the highest hardware requirement of any Wave P model. Gate behind `check_opensora2_available()` that returns `{available: false, reason: "insufficient_vram", required_gb: 22, detected_gb: N}` for cards below 20 GB. Open-Sora 1.3 (1B) automatically becomes the default fallback.
- **DepthCrafter non-commercial**: DepthCrafter (Tencent) provides temporal depth estimation for video sequences — highly useful for depth-based video compositing. However, its licence explicitly restricts use to academic, research, and education purposes. Do not adopt. For depth estimation on static frames, Depth-Anything-V2 (Wave N2.1, Apache-2) already covers the use case. Monitor for a permissive re-release.

---

## Wave P Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.45.0 | 2027-Q4 | P1.1 ConsisID, P1.2 Allegro T2V+TI2V |
| v1.46.0 | 2028-Q1 | P2.1 HiDream-I1, P2.2 HiDream-E1, P2.3 CogView4-6B |
| v1.47.0 | 2028-Q1 | P3.1 Qwen2.5-Omni, P3.2 Open-Sora 2.0, P3.3 UXP panel v1.0 |

---

## Wave P: Not Adopted / Deferred

- **DepthCrafter** (Tencent, 2024) — custom Tencent licence; explicitly non-commercial ("only for academic, research and education purposes"). NOT viable for OpenCut. The depth-estimation use case is covered by Depth-Anything-V2 (Wave N2.1, Apache-2) for static frames. Monitor for a permissive re-licence.
- **VoiceCraft** (jasonppy, 2024) — CC BY-NC-SA 4.0 code AND Coqui Public Model Licence weights; both code and model are non-commercial. NOT adopted. The zero-shot speech editing use case (edit a spoken word in an existing audio clip without re-recording) has no current Apache-2 equivalent in the stack; add to watch list for Wave Q.
- **Open-Sora-Plan** (PKU-YuanGroup, 2024) — MIT ✓ for code, but the feature set is now superseded by Open-Sora 2.0 (better quality, same licence category) and ConsisID (identity-preserving variant, same lab). Not adopted to avoid a redundant video generation backend.
- **Mochi-1** (Genmo, Nov 2024) — Apache-2 ✓ (carry-over from Wave O watch list). 60 GB single-GPU VRAM requirement remains a hard blocker. With the O3.3 multi-GPU scheduler in place, evaluate for Wave Q using 2× A100 or 4× 3090 multi-GPU inference.
- **Step-Video** (Kuaishou, 2025) — Apache-2 for code; 30B model. Could not confirm a clean Apache-2 checkpoint for the full model weights. Monitor for HuggingFace release with confirmed permissive weights licence.
- **SkyReels** (ByteDance, 2025) — repository not found at anticipated locations under `bytedance/` org. Monitor; ByteDance has a pattern of releasing strong T2V models with restrictive commercial licences.
- **Zero-shot speech editing** (in-audio word replacement) — VoiceCraft is the best-known OSS approach (non-commercial ❌). Examine `audiocraft/audiogen` (already excluded for weights) and community re-training efforts. The use case is high-value for podcast editing but has no Apache-2/MIT viable model in 2026. Reserve as Wave Q P0 if a permissive model emerges.

---

## Wave P Sources

- **ConsisID** — [PKU-YuanGroup/ConsisID](https://github.com/PKU-YuanGroup/ConsisID) (Apache-2, Nov 2024); CVPR 2025 Highlight; [arXiv:2411.17440](https://arxiv.org/abs/2411.17440); diffusers 0.33.0+; Windows one-click installer by community
- **Allegro** — [rhymes-ai/Allegro](https://github.com/rhymes-ai/Allegro) (Apache-2, Oct 2024); 2.8B DiT; Allegro-TI2V for first/last-frame conditioning; [arXiv:2410.15458](https://arxiv.org/abs/2410.15458); diffusers 0.32.0+
- **HiDream-I1** — [HiDream-ai/HiDream-I1](https://github.com/HiDream-ai/HiDream-I1) (MIT, April 2025); 17B Sparse DiT; [arXiv:2505.22705](https://arxiv.org/abs/2505.22705); Full/Dev/Fast variants; requires `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated)
- **HiDream-E1** — [HiDream-ai/HiDream-E1](https://github.com/HiDream-ai/HiDream-E1) (MIT, July 2025); instruction-based image editing; companion to HiDream-I1; HuggingFace Space demo available
- **CogView4-6B** — [THUDM/CogView4](https://github.com/THUDM/CogView4) (Apache-2, March 2025); 6B parameters; bilingual English + Chinese; `CogView4Pipeline` in diffusers; 13 GB VRAM with int4 text encoder; CogKit fine-tuning support
- **Qwen2.5-Omni** — [QwenLM/Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni) (Apache-2, March 2025); Thinker + Talker dual-module architecture; video + audio + text in; text + audio out; `Qwen2_5OmniModel` in transformers
- **Open-Sora 2.0** — [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) (Apache-2); v2.0 (11B, March 2025); [arXiv:2503.09642](https://arxiv.org/abs/2503.09642); 5s 720×1280; on-par with HunyuanVideo 11B + Step-Video 30B; Open-Sora 1.3 (1B) for consumer hardware
- **DepthCrafter** — [Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter) (Custom non-commercial, 2024); temporal video depth estimation; NOT adopted — academic/research/education only
- **VoiceCraft** — [jasonppy/VoiceCraft](https://github.com/jasonppy/VoiceCraft) (CC BY-NC-SA 4.0 code + Coqui model licence, 2024); zero-shot speech editing and TTS; NOT adopted — non-commercial

---

# Wave Q — Video Compositing Suite + Voice Generation Upgrade + Infinite Video (v1.48.0 → v1.51.0)

**Updated**: 2026-04-24
**Baseline**: v1.47.0 (post-Wave P; UXP panel v1.0, Open-Sora 2.0, Qwen2.5-Omni, CogView4-6B, HiDream-I1/E1, ConsisID, Allegro all shipped)
**Research pass**: April 2026 GitHub OSS survey — VACE, CosyVoice 2.0, MaskGCT (Amphion), Vevo (Amphion), OmniGen2, SkyReels V2, SkyReels V3, IndexTTS2, FireRedTTS, Stable Virtual Camera

This wave introduces three capability clusters:
1. **VACE all-in-one video compositing** — the single most impactful video editing feature missing from OpenCut: compose, move, swap, reference, expand, and animate elements inside an existing video using a mask + prompt. Uses Wan2.1-VACE backend. Closes Adobe Firefly's video compositing feature gap.
2. **Voice generation upgrade** — replaces and supplements the Wave L/M/N TTS tier: CosyVoice 2.0 adds 9-language + 18 Chinese dialect zero-shot voice cloning at 150 ms streaming latency; MaskGCT from the Amphion toolkit adds a parallel (non-autoregressive) TTS path for the fastest inference speed of any model in the stack.
3. **Multi-reference image synthesis + infinite-length video** — OmniGen2 closes the "combine multiple reference images into one coherent output" gap (Kling/Runway's "actor swap" workflow); SkyReels V2 closes the infinite-length temporal video generation gap using Diffusion Forcing autoregressive architecture.

---

## Wave Q1 — VACE All-in-One Video Compositing (v1.48.0)

**Goal**: Expose the full Wan2.1-VACE compositing capability as a first-class feature in OpenCut: mask-based inpainting, subject replacement, scene expansion, reference-object insertion, depth-based re-animation, and background swap. One model handles all six compositing workflows.
**New required deps**: `wan` (Wan2.1 native inference — git-installable, Apache-2; or diffusers if VACE is integrated there)
**New routes**: ~18

### OSS Discoveries — Video Compositing

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| Q1.1 | **VACE all-in-one video compositing** (ali-vilab/VACE, Apache-2, ICCV 2025) — a single model (Wan2.1-VACE-14B or 1.3B) that performs all of: (a) **V2V** — style + appearance editing of an existing video via text prompt; (b) **MV2V** (masked V2V) — mask + prompt to edit a specific region while keeping the rest intact; (c) **R2V** (reference-to-video) — insert a specific object or person from a reference image into a video. Named editing workflows all built on these three modes: **Move-Anything** (mask subject → new position), **Swap-Anything** (mask subject → new subject from text/image), **Reference-Anything** (insert a reference image object into a video), **Expand-Anything** (extend video spatial canvas outward), **Animate-Anything** (give motion to a still object in-place). Ship as a compositing tool in the OpenCut timeline: user paints a mask on a clip, picks a workflow, enters a prompt or reference image, and VACE renders a new video segment. VACE-Wan2.1-1.3B (480×832) as default; Wan2.1-VACE-14B (720×1280) as quality mode. | `POST /compose/vace/v2v`, `POST /compose/vace/mv2v`, `POST /compose/vace/r2v`, `GET /compose/vace/preprocess/{task}`, `GET /compose/vace/info` | `core/video_compose_vace.py` + `core/vace_preprocess.py` | `pip install wan@git+https://github.com/Wan-Video/Wan2.1` (Apache-2) + VACE-Wan2.1-1.3B weights (~5 GB) + optional Wan2.1-VACE-14B (~28 GB) | L | Apache-2 | Closes entire Adobe Firefly video compositing gap locally; most compositing workflows require no extra model beyond one VACE checkpoint |
| Q1.2 | **VACE preprocessor toolkit** — VACE's V2V/MV2V tasks require mask + depth/flow/edge preprocessors. Wrap `VACE-Annotators` (ali-vilab, Apache-2) as a pre-flight step: auto-detect the task type and run the appropriate annotator (depth map for `depth` task, optical flow for `motion` task, edge map for `edge` task) before calling the VACE inference backend. Expose as a `preprocess` step in the compositing UI so the user sees a preview of the extracted map before sending to VACE. | `POST /compose/vace/preprocess/depth`, `POST /compose/vace/preprocess/flow`, `POST /compose/vace/preprocess/edge` | `core/vace_preprocess.py` | VACE-Annotators (~2 GB) — auto-downloaded from HuggingFace `ali-vilab/VACE-Annotators` | S | Apache-2 | Required preprocessing for VACE depth/flow/edge modes; without this, those modes require manual map generation |

---

## Wave Q2 — Multilingual Voice Generation Suite (v1.49.0)

**Goal**: Upgrade the TTS/voice-cloning tier with two high-quality models that together cover 9 languages + 18 Chinese dialects (CosyVoice 2.0) and the fastest parallel TTS inference path of any model in the stack (MaskGCT).
**New required deps**: CosyVoice 2.0 requires `WeTextProcessing`; MaskGCT requires `phonemizer` + `espeak-ng`
**New routes**: ~12

### OSS Discoveries — Voice Generation

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| Q2.1 | **CosyVoice 2.0 / Fun-CosyVoice 3.0 multilingual TTS** (FunAudioLLM/CosyVoice, Apache-2, December 2024 / December 2025) — 0.5B multilingual TTS model covering 9 languages (Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian) and 18+ Chinese dialects/accents (Cantonese, Sichuan, Shanghainese, Minnan, and more) with zero-shot voice cloning. Key features over existing Wave TTS models: (a) bi-streaming: text-in streaming + audio-out streaming with 150 ms end-to-end latency — ship as an SSE streaming endpoint so the first audio chunk plays before generation completes; (b) instruction-based generation: `{text: "Hello.", instruction: "speak with a warm, friendly smile"}` controls prosody, speed, volume, emotion and dialect; (c) zero-shot voice cloning with a 3-second reference audio sample. Fun-CosyVoice 3.0 (December 2025) extends with GRPO reinforcement learning training and improved content accuracy and speaker similarity scores. | `POST /tts/cosyvoice`, `POST /tts/cosyvoice/stream` (SSE), `POST /tts/cosyvoice/clone`, `GET /tts/cosyvoice/info` | `core/tts_cosyvoice.py` | CosyVoice2 repo (`pip install -r requirements.txt`) + CosyVoice2-0.5B weights (~1 GB) via HuggingFace `FunAudioLLM/CosyVoice2-0.5B` | M | Apache-2 | 9-language + 18-dialect coverage; fastest TTS streaming (150ms); unique Chinese dialect control; closes multilingual dubbing gap |
| Q2.2 | **MaskGCT zero-shot parallel TTS** (open-mmlab/Amphion, MIT, October 2024) — fully non-autoregressive TTS model from the Amphion toolkit. Trained on 100K hours of in-the-wild speech (the Emilia dataset). In benchmarks outperforms SOTA autoregressive TTS systems (including VALL-E 2, VoiceLM) on naturalness, speaker similarity, and intelligibility at significantly faster inference speed — parallel generation means inference time scales with sequence length but not linearly. Zero-shot: provide a 3-second reference audio and synthesize new text in the same voice + style. Ships as `POST /tts/maskgct` with `{text: "...", reference_audio_b64: "..."}`. Complements CosyVoice 2.0: CosyVoice is used for multilingual + streaming; MaskGCT is used for the fastest single-request batch TTS. | `POST /tts/maskgct`, `POST /tts/maskgct/clone`, `GET /tts/maskgct/info` | `core/tts_maskgct.py` | Amphion repo (`pip install amphion` or clone + install) + MaskGCT weights (~5 GB) via HuggingFace `amphion/maskgct` | M | MIT (Amphion toolkit) | Fastest parallel TTS; 100K hours training; outperforms autoregressive systems on all three key metrics |

---

## Wave Q3 — Multi-Reference Image Generation + Infinite-Length Video (v1.50.0 → v1.51.0)

**Goal**: Add OmniGen2 for multi-reference in-context generation (combine 2–4 reference images of different people/objects into one coherent output — the "actor swap" / "product placement" workflow); and SkyReels V2 for true infinite-length video generation using Diffusion Forcing architecture (generate 30-second+ videos without temporal drift).
**New required deps**: OmniGen2 requires `qwen-vl-utils`; SkyReels V2 diffusers integration
**New routes**: ~14

### OSS Discoveries — Multi-Reference + Infinite Video

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| Q3.1 | **OmniGen2 multi-reference in-context image generation** (VectorSpaceLab/OmniGen2, Apache-2, June 2025) — multimodal generation model built on Qwen-VL-2.5 with two decoding pathways for text and image. Key capability for OpenCut: **in-context generation** — accepts 2–4 reference images (different people, different objects, different scenes) plus a text description, and generates a single coherent output that places all referenced subjects in the described scene. Example: `[photo of person A] + [photo of person B] + "Person A and Person B shaking hands in an office"` → one realistic image with both people. Also provides SOTA instruction-based image editing ("instruction-guided editing" mode). Fine-tunable on custom data. GGUF/TeaCache/TaylorSeer acceleration supported. | `POST /image/generate/omnigen2`, `POST /image/edit/omnigen2`, `GET /image/generate/omnigen2/info` | `core/t2i_omnigen2.py` | `pip install omnigen2` (or clone + install) + OmniGen2 weights (~13 GB via HuggingFace `OmniGen2/OmniGen2`) + `qwen-vl-utils` | M | Apache-2 | SOTA multi-reference in-context generation; closes Kling/Runway "two actors in one scene" workflow gap entirely locally |
| Q3.2 | **SkyReels V2 infinite-length T2V + I2V** (SkyworkAI/SkyReels-V2, Skywork Community Licence — commercial allowed, April 2025) — 14B DiT video generation model implementing the **Diffusion Forcing** architecture for autoregressive infinite-length video generation. Unlike fixed-clip models (5-10s), SkyReels V2 generates clips of arbitrary length by overlapping temporal windows (17-frame overlap, `ar_step=5` mode). Generates 720P at 24 FPS. Ships as both T2V and I2V (image-to-video). Built on Wan2.1 VAE; diffusers `SkyReelsV2DiffusionForcingPipeline` available. Use case: generate 30-second+ B-roll sequences from a single text prompt; cinematic long-form video. Consumer option: 1.3B variant for 540P generation. | `POST /generate/skyreels2/t2v`, `POST /generate/skyreels2/i2v`, `POST /generate/skyreels2/t2v/infinite`, `GET /generate/skyreels2/info` | `core/gen_video_skyreels2.py` | `diffusers` (SkyReelsV2DiffusionForcingPipeline) + SkyReels-V2-DF-14B-720P weights (~28 GB) or 1.3B-540P (~5 GB) | L | Skywork Community Licence (commercial use allowed) | First infinite-length T2V in the stack; closes Adobe Stock long B-roll generation gap; 30-second+ coherent clips |
| Q3.3 | **SkyReels V3 talking avatar** (SkyworkAI/SkyReels-V3, Skywork Community Licence — commercial allowed, January 2026) — 19B A2V (audio-to-video) model generating a lifelike talking avatar: input is a portrait image + audio track (up to 200 seconds) + optional text prompt describing expression/scene; output is a video of that person speaking with accurate lip sync and natural head movement. Built for long-form content: news reports, training videos, dubbing, virtual spokespersons. Supports Chinese, English, Korean, singing, and fast dialogue. Also includes V3-R2V (Reference-to-Video): 1–4 reference images → video preserving all subjects. | `POST /generate/skyreels3/avatar`, `POST /generate/skyreels3/r2v`, `GET /generate/skyreels3/info` | `core/gen_video_skyreels3.py` | SkyReels V3 repo + SkyReels-V3-A2V-19B weights (~38 GB) + SkyReels-V3-R2V-14B (~28 GB); `--offload` required for consumer hardware | L | Skywork Community Licence (commercial use allowed; gated HF download) | First high-quality talking avatar model in the stack; closes Adobe Podcast video production gap; 200-second audio support |

---

## Wave Q: Q-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave Q Role |
|------|-----|---------|----------|-------------|
| VACE (Wan2.1-VACE) | ali-vilab / Wan-AI | Apache-2 | Video Compositing | Q1.1 — all-in-one V2V/MV2V/R2V compositing suite |
| VACE-Annotators | ali-vilab | Apache-2 | Preprocessors | Q1.2 — depth/flow/edge preprocessors for VACE tasks |
| CosyVoice 2.0 / Fun-CosyVoice 3.0 | FunAudioLLM | Apache-2 | TTS | Q2.1 — 9-language + 18-dialect multilingual streaming TTS |
| MaskGCT | open-mmlab/Amphion | MIT | TTS | Q2.2 — zero-shot parallel (non-AR) TTS; fastest inference |
| OmniGen2 | VectorSpaceLab | Apache-2 | T2I + Editing | Q3.1 — multi-reference in-context image generation; SOTA instruction editing |
| SkyReels V2 | SkyworkAI | Skywork CL (commercial ✓) | T2V + I2V | Q3.2 — infinite-length T2V/I2V via Diffusion Forcing; 720P 24 FPS |
| SkyReels V3 | SkyworkAI | Skywork CL (commercial ✓, gated) | A2V + R2V | Q3.3 — talking avatar (200s audio); multi-reference video synthesis |
| IndexTTS2 | bilibili / IndexTeam | Custom (contact for commercial) | TTS | NOT ADOPTED — commercial contact required; no clear Apache-2/MIT grant |
| FireRedTTS | FireRedTeam | MPL 2.0 | TTS | NOT ADOPTED — MPL 2.0 file-level copyleft; complex licence situation |
| Stable Virtual Camera | Stability AI | Gated (custom, HF auth required) | Novel View Synthesis | NOT ADOPTED — gated HF access; Stability AI custom licence (likely non-commercial) |
| Vevo / Vevo2 | open-mmlab/Amphion | MIT | Voice Conversion | WATCH LIST — MIT ✓; voice timbre + style + emotion conversion; defer to Wave R if voice conversion demand grows |

---

## Wave Q: Competitive Gap Closure

| Gap | Competitor | Wave Q Feature | Closes? |
|-----|-----------|---------------|---------|
| Video region compositing (mask + edit) | Adobe Firefly (Generative Fill for video), RunwayML Gen-3 inpainting | Q1.1 VACE MV2V | Y — mask any region, edit with text prompt, preserve rest of video |
| Move a specific object to a new position in video | Runway "Move Brush" | Q1.1 VACE Move-Anything | Y — mask subject, prompt new position |
| Replace object/person in existing video | Runway "Swap Brush", Pika Effects | Q1.1 VACE Swap-Anything | Y — mask region, swap with any text/image |
| Insert a reference-image object into a video | Adobe After Effects compositing | Q1.1 VACE Reference-Anything | Y — reference image + mask zone → coherent insertion |
| Chinese / multilingual TTS with dialect control | ElevenLabs multilingual, 11Labs, Kling | Q2.1 CosyVoice 2.0 | Y — 9 languages + 18 Chinese dialects; locally |
| Sub-200ms TTS streaming latency | ElevenLabs streaming | Q2.1 CosyVoice SSE streaming | Y — 150 ms first-chunk SSE streaming |
| Non-autoregressive parallel TTS (fastest batch) | Commercial batch TTS APIs | Q2.2 MaskGCT | Y — parallel non-AR; fastest per-request inference |
| Multi-person image composition (2–4 reference subjects) | Midjourney "omni reference", Kling "multi-subject", Runway | Q3.1 OmniGen2 in-context | Y — 2–4 reference images → coherent composite scene |
| Infinite-length video (>10s, no drift) | Runway Gen-3 Extend, Kling Continue | Q3.2 SkyReels V2 Diffusion Forcing | Y — 30–60s+ video with temporal coherence, 720P 24 FPS |
| Talking avatar from portrait + audio (long-form) | HeyGen, Synthesia, Runway Act-Two | Q3.3 SkyReels V3 A2V | Y — 200-second audio → lifelike portrait animation; local inference |

---

## Wave Q Gotchas

- **VACE mask format**: VACE's MV2V tasks require a binary mask video (same temporal length as the input video, white = edit region, black = preserve). Add a `POST /compose/vace/mask/from_frame` helper that accepts a single-frame mask image and replicates it across N frames; expose a mask-painter layer in the UXP compositing panel that writes to this format.
- **VACE task vs. mode**: VACE's pipeline distinguishes "task" (what preprocessing to run: `depth`, `inpainting`, `flow`, `edge`, `reference`, etc.) from the inference "base" (wan vs. ltx). Always use `wan` as the base for production quality. VACE-LTX-Video-0.9 uses RAIL-M licence — do not ship it; always use the Apache-2 `VACE-Wan2.1-1.3B` or `Wan2.1-VACE-14B` variants.
- **CosyVoice 2.0 WeTextProcessing dep**: CosyVoice requires `WeTextProcessing` (a Chinese text normalization library) for Chinese input. It pulls in additional C++ extensions. If `WeTextProcessing` fails to install on Windows (common), fall back to `--use_tn=False` mode which skips normalization (slightly lower Chinese punctuation accuracy). Document the fallback in `core/tts_cosyvoice.py`.
- **CosyVoice streaming vs. batch**: The SSE streaming endpoint (`/tts/cosyvoice/stream`) requires the CosyVoice bi-streaming mode and a persistent async generator. Use FastAPI's `StreamingResponse` with an `asyncio` generator wrapping the CosyVoice streaming API. Test that the SSE connection is properly closed when the generator completes.
- **MaskGCT espeak-ng on Windows**: MaskGCT (via Amphion) uses `phonemizer` which requires `espeak-ng` as a system dependency. On Windows, `espeak-ng` is not installable via pip — it requires a separate installer from `espeak-ng.github.io`. Add a pre-flight check: `shutil.which("espeak-ng")`. If absent, return `{available: false, reason: "espeak_ng_required", install_url: "https://espeak-ng.github.io/espeak-ng/"}` instead of crashing.
- **OmniGen2 flash-attention optional**: OmniGen2 explicitly states it works without `flash-attn` (as of June 23, 2025 update). Do not add `flash-attn` as a hard requirement — leave it as an optional performance upgrade. This matters on Windows where `flash-attn` compilation is difficult.
- **OmniGen2 in-context vs. editing**: OmniGen2 uses different prompt formats for in-context generation vs. instruction editing. In-context: images are injected inline as `<img>` tokens in the prompt. Instruction editing: prompt describes the edit in plain text. Add a `mode` parameter to `POST /image/generate/omnigen2`: `mode: "t2i" | "in_context" | "edit"`.
- **SkyReels V2 Skywork CL**: Unlike Apache-2, the Skywork Community License requires that you read and comply with the Skywork Model Community License Agreement (PDF at `github.com/SkyworkAI/Skywork/`). Specifically: no misuse for illegal activities, no bypassing safety reviews for internet services. These are behavioural restrictions, not commercial restrictions. Document in a `SKYWORK_LICENCE_NOTICE.txt` shipped with OpenCut.
- **SkyReels V2 infinite generation memory**: Long video generation (60+ seconds) accumulates VRAM for the window context. With `overlap_history=17` and `offload=True`, generation is feasible on a 24 GB card but slow. Add a `max_duration_seconds` parameter that caps at 30s for 24 GB cards and 60s for 40 GB+ cards, auto-detected from `check_skyreels2_available()`.
- **SkyReels V3 A2V gated download**: SkyReels-V3 weights require HuggingFace authentication (`huggingface-cli login`). Use the same HF token mechanism as CSM-1B / HiDream-I1 / Open-Sora 2.0 — the user's `HF_TOKEN` env var or the `opencut/config.json` `hf_token` field. The Skywork licence does not restrict commercial use; the gate is for usage tracking only.
- **SkyReels V3 A2V single portrait limitation**: The A2V model (talking avatar) takes a single portrait image as input. Multi-person A2V is not yet supported. Document as `max_subjects: 1` in the info endpoint. For the R2V model, 1–4 reference images are supported (same pattern as OmniGen2 in-context).

---

## Wave Q Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.48.0 | 2028-Q2 | Q1.1 VACE V2V/MV2V/R2V compositing, Q1.2 VACE Annotators preprocessors |
| v1.49.0 | 2028-Q2 | Q2.1 CosyVoice 2.0 + streaming, Q2.2 MaskGCT parallel TTS |
| v1.50.0 | 2028-Q3 | Q3.1 OmniGen2 multi-reference in-context, Q3.2 SkyReels V2 infinite T2V |
| v1.51.0 | 2028-Q3 | Q3.3 SkyReels V3 talking avatar + multi-reference video synthesis |

---

## Wave Q: Not Adopted / Deferred

- **IndexTTS2** (index-tts/index-tts, bilibili/IndexTeam, September 2025) — Zero-shot autoregressive TTS with emotion control and precise duration control. However, the README explicitly states "Please contact the authors for more detailed information. For commercial usage and cooperation, please contact [indexspeech@bilibili.com]." No clear Apache-2/MIT grant in the licence file. NOT adopted until a clean permissive licence is confirmed. Monitor for a licence update.
- **FireRedTTS / FireRedTTS-1S** (FireRedTeam/FireRedTTS, Mozilla Public License 2.0) — Streamable foundation TTS with flow-matching decoder. MPL 2.0 is technically a commercial-use-allowed licence, but it is file-level copyleft: any modifications to MPL-licensed files must remain MPL-licensed. Since OpenCut would use it as a dependency without modifying its source files, this is technically fine; however, MPL-2.0 integration requires a legal review before shipping as part of a commercial product. Defer to Wave R pending a legal confirmation pass. The TTS space is now well-covered by CosyVoice (Apache-2) and MaskGCT (MIT).
- **Stable Virtual Camera / SEVA** (Stability-AI/stable-virtual-camera, March 2025) — Generalist novel view synthesis model; 1.3B. Weights are gated (requires HuggingFace login + form submission to `stabilityai/stable-virtual-camera`). Licence is not Apache-2 — Stability AI Community Licence is non-commercial for weights. NOT adopted. The novel view synthesis use case (generate camera fly-around from static images) has no current Apache-2 open model; monitor for a permissive alternative.
- **Vevo / Vevo2** (open-mmlab/Amphion, MIT, 2025/2026) — Zero-shot voice conversion: timbre, style, accent, and emotion transfer from a reference audio clip. MIT ✓ and high quality, but voice conversion (changing one person's voice to sound like another) is already partially covered by CosyVoice 2.0 zero-shot cloning in Q2.1. Voice conversion is a niche use case within OpenCut's primary workflow. Add to WATCH LIST — if user demand for voice timbre transfer grows post-Wave Q, adopt Vevo2 as a Wave R add-on.
- **SkyCaptioner-V1** (SkyworkAI, Skywork CL) — Video captioning model shipped alongside SkyReels V2. Generates detailed text descriptions of video clips. Functionality already partly covered by Qwen2.5-VL (Wave N3.2) and Qwen2.5-Omni (Wave P3.1). Evaluate in Wave R if more targeted video captioning (structured caption format for T2V prompt generation) is needed.
- **EzAudio** — Apache-2 text-to-audio; ICCV 2025. Defer to Wave R; text-to-audio (foley effects) is an unaddressed gap that merits its own Wave R feature slot with proper research into 2025–2026 SOTA options.
- **Step-Video** (Kuaishou/KwaiVGI) — 30B T2V; Apache-2 for code. Still cannot confirm the weight licence for the full 30B checkpoint is clearly permissive. Monitor HuggingFace for a clean Apache-2 full-model release. If confirmed, Step-Video would join Open-Sora 2.0 and SkyReels V2 as the highest-quality T2V tier.

---

## Wave Q Sources

- **VACE** — [ali-vilab/VACE](https://github.com/ali-vilab/VACE) (Apache-2, March 2025); ICCV 2025; Wan2.1-VACE-1.3B (480×832, Apache-2) + Wan2.1-VACE-14B (720×1280, Apache-2) at HuggingFace `Wan-AI/Wan2.1-VACE-1.3B` and `Wan-AI/Wan2.1-VACE-14B`; VACE-Annotators for preprocessing
- **CosyVoice 2.0** — [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) (Apache-2, Dec 2024); `CosyVoice2-0.5B` at HuggingFace `FunAudioLLM/CosyVoice2-0.5B`; bi-streaming 150ms latency; Fun-CosyVoice3-0.5B-2512 (Dec 2025) is current best version
- **MaskGCT** — [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) (MIT); `amphion/maskgct` on HuggingFace; [arXiv:2409.00750](https://arxiv.org/abs/2409.00750); fully non-autoregressive TTS; 100K hours Emilia training data
- **OmniGen2** — [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) (Apache-2, June 2025); [arXiv:2506.18871](https://arxiv.org/abs/2506.18871); `OmniGen2/OmniGen2` on HuggingFace; TeaCache/TaylorSeer acceleration; ComfyUI official support
- **SkyReels V2** — [SkyworkAI/SkyReels-V2](https://github.com/SkyworkAI/SkyReels-V2) (Skywork Community Licence — commercial allowed, April 2025); [arXiv:2504.13074](https://arxiv.org/abs/2504.13074); Diffusion Forcing architecture for infinite-length video; diffusers `SkyReelsV2DiffusionForcingPipeline`; 14B (720P) + 1.3B (540P) variants
- **SkyReels V3** — [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3) (Skywork Community Licence — commercial allowed, January 2026); [arXiv:2601.17323](https://arxiv.org/abs/2601.17323); SkyReels-V3-R2V-14B (1–4 reference image → video) + SkyReels-V3-V2V-14B (video extension) + SkyReels-V3-A2V-19B (talking avatar, 200s audio)
- **IndexTTS2** — [index-tts/index-tts](https://github.com/index-tts/index-tts) (bilibili); [arXiv:2506.21619](https://arxiv.org/abs/2506.21619); Sept 2025; contact required for commercial use — NOT adopted
- **FireRedTTS-1S** — [FireRedTeam/FireRedTTS](https://github.com/FireRedTeam/FireRedTTS) (MPL 2.0); [arXiv:2503.20499](https://arxiv.org/abs/2503.20499); streamable TTS with flow-matching — NOT adopted (MPL requires legal review)
- **Stable Virtual Camera** — [Stability-AI/stable-virtual-camera](https://github.com/Stability-AI/stable-virtual-camera); gated HF weights; custom Stability AI licence — NOT adopted (non-commercial weights)
- **Vevo** — [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) (MIT); voice conversion framework — WATCH LIST
- **Vevo2** — [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) (MIT, March 2026); unified speech + singing voice generation — WATCH LIST

---

## Wave R — Foley Engine, Lip Sync, Camera Control & HPC T2V (v1.52.0 → v1.55.0)

**Baseline**: v1.51.0 (post-Wave Q)
**Goal**: Close the last major audio gap (foley / sound-effects synthesis), add audio-driven
lip sync for talking-head and dubbing workflows, bolt trajectory-based camera control onto
existing Wan/CogVideoX video models, and surface Mochi-1 (10 B, consumer-GPU) plus
Step-Video-T2V-Turbo (30 B, HPC) as high-fidelity long-video T2V options.

---

### Wave R Feature Table

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| R1.1 | EzAudio T2A sound-effects generation | `POST /generate/ezaudio` | `services/ezaudio_service.py` | `OpenSound/EzAudio` (s3_xl) | M | MIT | First dedicated foley/SFX engine in the stack; fills gap deferred since Wave L |
| R1.2 | EzAudio audio inpainting (mask-region edit) | `POST /edit/audio/inpaint` | `services/ezaudio_service.py` | EzAudio `s3_xl` | S | MIT | Replace a 5 s clip mid-track without re-generating the whole file |
| R1.3 | EzAudio ControlNet reference-audio conditioning | `POST /generate/ezaudio_controlnet` | `services/ezaudio_service.py` | EzAudio ControlNet ckpt (energy model) | M | MIT | Match energy/timbre of an existing reference; critical for sound-design consistency |
| R2.1 | MuseTalk 1.5 audio-driven lip sync | `POST /animate/lip_sync` | `services/musetalk_service.py` | `TMElyralab/MuseTalk` (MIT code / CreativeML-OpenRAIL-M weights) | L | MIT / CreativeML-OpenRAIL-M | Real-time 30 fps+ lip sync; enables AI-avatar dubbing and multi-language post-sync |
| R2.2 | VideoX-Fun camera-control I2V (trajectory) | `POST /generate/videox_fun/camera` | `services/videox_fun_service.py` | `aigc-apps/VideoX-Fun`, Wan2.1-Fun-14B-Control | L | Apache-2 | Pan / zoom / orbit trajectories on Wan2.1-Fun; plugs the camera-path gap |
| R2.3 | VideoX-Fun structural control I2V (Canny / Depth / Pose / MLSD) | `POST /generate/videox_fun/control` | `services/videox_fun_service.py` | VideoX-Fun control ckpts | M | Apache-2 | ControlNet-style motion conditioning; complements VACE (Wave Q1) with lighter-weight control signals |
| R3.1 | Mochi-1 consumer T2V | `POST /generate/mochi` | `services/mochi_service.py` | `genmo/mochi-1-preview` (Apache-2) | L | Apache-2 | 10 B; best open motion fidelity; `--cpu_offload` path for 16 GB VRAM; LoRA fine-tune support |
| R3.2 | Step-Video-T2V-Turbo HPC T2V | `POST /generate/stepvideo` | `services/stepvideo_service.py` | `stepfun-ai/stepvideo-t2v-turbo` (MIT) | XL | MIT | 30 B; 204 frames (≈ 6.8 s @ 30 fps); bilingual EN+ZH; 10–15 step turbo; gates behind `STEPVIDEO_ENABLED` |
| R3.3 | Step-Video-Ti2V (image-to-video companion) | `POST /generate/stepvideo/i2v` | `services/stepvideo_service.py` | `stepfun-ai/stepvideo-ti2v` (MIT) | M | MIT | Same 30 B engine, image-conditioned; reuses service scaffolding from R3.2 |

---

### Wave R OSS Survey

| Project | Repo | Licence | Est. Stars | VRAM | Status |
|---------|------|---------|------------|------|--------|
| EzAudio | haidog-yaqub/EzAudio | **MIT ✓** | ~800 | ~6–8 GB | Interspeech 2025 oral; diffusers-compatible |
| MuseTalk 1.5 | TMElyralab/MuseTalk | **MIT code / CreativeML-OpenRAIL-M weights** | ~10 K | ~4 GB | v1.5 March 2025; training code open April 2025 |
| VideoX-Fun (Wan2.1-Fun) | aigc-apps/VideoX-Fun | **Apache-2 ✓** | ~2 K | 12–24 GB | Camera control + Wan2.2 support Oct 2025 |
| Mochi-1 | genmoai/mochi | **Apache-2 ✓** | ~5 K | ~30 GB (20 GB w/ offload) | Nov 2024; ComfyUI consumer-GPU support Nov 2024 |
| Step-Video-T2V-Turbo | stepfun-ai/Step-Video-T2V | **MIT ✓** (code + weights) | ~2 K | ~80 GB (4× 80 GB GPU) | Feb 2025; DiffSynth-Studio quantization path available |
| Step-Video-Ti2V | stepfun-ai/Step-Video-Ti2V | **MIT ✓** | ~500 | ~80 GB (4× 80 GB GPU) | March 2025; I2V companion to T2V |
| AudioGen (Meta) | facebookresearch/audiocraft | CC-BY-NC-4.0 ❌ | ~21 K | — | Not adopted |
| AudioLDM2 | haoheliu/AudioLDM2 | CC-BY-NC-SA-4.0 ❌ | ~3 K | — | Not adopted |
| Stable Audio Open | Stability-AI/stable-audio-tools | Stability AI non-commercial ❌ | ~4 K | — | Not adopted |

---

### Wave R Competitive Gap Matrix

| Capability | Pre-Wave R | Post-Wave R |
|-----------|-----------|-------------|
| Text-to-sound effects / foley | None | EzAudio (MIT, DiT, 44.1 kHz) |
| Audio inpainting (mid-clip edit) | None | EzAudio mask-based inpainting |
| Reference-audio conditioning | None | EzAudio ControlNet |
| Talking head / lip sync (real-time) | SkyReels V3 A2V (Wave Q3.3, offline) | + MuseTalk 1.5 (30 fps+ real-time, multi-language) |
| Camera-trajectory video generation | None | VideoX-Fun Wan2.1-Fun camera control |
| Structural control I2V (pose, depth) | VACE (Wave Q1) | + VideoX-Fun Canny / Depth / Pose / MLSD modes |
| High-motion-fidelity consumer T2V | Open-Sora 2.0, SkyReels V2 | + Mochi-1 (10 B, `--cpu_offload`, LoRA) |
| 30 B long-video HPC T2V | None | Step-Video-T2V-Turbo (204 frames, bilingual) |
| I2V from still image at HPC tier | None | Step-Video-Ti2V (same 30 B engine, image-conditioned) |

---

### Wave R Gotchas

- **EzAudio ControlNet checkpoint**: separate from the base `s3_xl` model (`OpenSound/EzAudio-ControlNet`); both must be downloaded in `check_ezaudio_available()`.
- **MuseTalk CreativeML-OpenRAIL-M weights**: commercial use allowed; ship `MUSETALK_LICENCE_NOTICE.txt` alongside weights. The licence prohibits generating misleading content, synthesizing the voice/face of a real person without consent, and creating harmful deepfakes. Add a consent acknowledgment checkbox to the lip-sync UI panel.
- **MuseTalk face region 256×256**: optimized for portrait-orientation close-up faces; wide-angle crowd shots degrade quality. Recommend face-crop preprocessing (`mediapipe` or `InsightFace`) before passing to MuseTalk.
- **MuseTalk WhisperTiny dependency**: requires `openai/whisper` (MIT) — already installed if CSM-1B (Wave N) is active. Reuse the existing `whisper-tiny` download path.
- **VideoX-Fun Wan2.1-Fun camera ckpt**: requires `Wan2.1-Fun-14B-Control` (not the base Wan2.1 already in stack from Wave M). Separate download; reuse existing `WAN_MODEL_DIR` parent directory for storage locality.
- **Mochi-1 VRAM**: 10 B model needs ~30 GB unquantized; `--cpu_offload` reduces to ~20 GB GPU + system RAM but increases generation time (~5 min per clip). Expose a "quality mode" vs. "memory mode" toggle in the UI.
- **Mochi-1 diffusers integration**: uses `MochiPipeline` (diffusers ≥ 0.31); compatible with the diffusers version required by other pipeline models already in stack.
- **Step-Video-T2V Linux-only text encoder**: the `step_llm` text encoder uses CUDA kernels that only compile on Linux (sm_80 / sm_86 / sm_90 required). Gate behind `STEPVIDEO_ENABLED=1` env flag; show a Linux-only warning in the UI when running on Windows or macOS.
- **Step-Video-T2V quantization**: DiffSynth-Studio supports int8 quantization, reducing VRAM to approximately 2× 40 GB (two A100-40 GB). Add `STEPVIDEO_QUANTIZE=int8` env option with a note that int8 reduces output quality slightly.
- **Step-Video-Ti2V reuse**: Step-Video-Ti2V shares the same DiT and VAE as T2V; the image condition is injected as an extra token. The R3.2 service setup covers R3.3 with minimal extra scaffolding.

---

### Wave R Shipping Cadence

| Version | Deliverable |
|---------|-------------|
| v1.52.0 | R1: EzAudio T2A + inpainting + ControlNet endpoints; `check_ezaudio_available()` guard; 44.1 kHz output |
| v1.53.0 | R2a: MuseTalk 1.5 lip sync endpoint; consent UI; `MUSETALK_LICENCE_NOTICE.txt` |
| v1.54.0 | R2b: VideoX-Fun camera-control I2V + control-signal I2V (Canny/Depth/Pose/MLSD) endpoints |
| v1.55.0 | R3: Mochi-1 T2V (`--cpu_offload` default on < 24 GB cards) + Step-Video-T2V-Turbo + Ti2V (`STEPVIDEO_ENABLED`, Linux-only) |

---

### Wave R Not Adopted

| Project | Reason |
|---------|--------|
| AudioGen (Meta / AudioCraft) | CC-BY-NC-4.0 ❌ |
| AudioLDM2 (HKUST) | CC-BY-NC-SA-4.0 ❌ |
| Stable Audio Open (Stability AI) | Stability AI non-commercial licence ❌ |
| DepthCrafter (Tencent) | Academic/research only ❌ (re-checked from Wave P survey) |

---

## Wave R Sources

- **EzAudio** — [haidog-yaqub/EzAudio](https://github.com/haidog-yaqub/EzAudio) (MIT); [arXiv:2409.10819](https://arxiv.org/abs/2409.10819); `OpenSound/EzAudio` on HuggingFace; `OpenSound/EzAudio-ControlNet` for reference-audio variant; Interspeech 2025 oral
- **MuseTalk 1.5** — [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) (MIT code / CreativeML-OpenRAIL-M weights); [arXiv:2410.10122](https://arxiv.org/abs/2410.10122); `TMElyralab/MuseTalk` on HuggingFace; v1.5 released March 2025; training code open April 2025
- **VideoX-Fun** — [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) (Apache-2); `alibaba-pai/Wan2.1-Fun-14B-Control` on HuggingFace; camera control + control-signal I2V; Wan2.2 + VACE support added Oct 2025
- **Mochi-1** — [genmoai/mochi](https://github.com/genmoai/mochi) (Apache-2); [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview) on HuggingFace; `--cpu_offload` consumer path; ComfyUI support Nov 2024; LoRA fine-tuning Nov 2024
- **Step-Video-T2V** — [stepfun-ai/Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V) (MIT); [arXiv:2502.10248](https://arxiv.org/abs/2502.10248); `stepfun-ai/stepvideo-t2v` + `stepfun-ai/stepvideo-t2v-turbo` on HuggingFace; DiffSynth-Studio quantization path; released Feb 2025
- **Step-Video-Ti2V** — [stepfun-ai/Step-Video-Ti2V](https://github.com/stepfun-ai/Step-Video-Ti2V) (MIT); image-to-video companion; released March 2025



# Wave S — Video Relighting, Next-Gen ASR, Vision-Language Modernization, FFmpeg 8 + UXP EOL Cutover (v1.56.0 → v1.58.0)

**Updated**: 2026-05-09
**Baseline**: v1.55.0 (post-Wave R; EzAudio + MuseTalk 1.5 + VideoX-Fun + Mochi-1 + Step-Video shipped)
**Research pass**: May 2026 OSS survey — SeedVR2 (ICLR 2026), Light-A-Video (ICCV 2025), DiffusionRenderer (NVIDIA Toronto AI Lab), Qwen3-VL (Sept 2025), InternVL3 (April 2025), Parakeet TDT 0.6B v2 (NVIDIA), Canary-1B-Flash (NVIDIA), FFmpeg 8.0 "Huffman" (Aug 2025), Adobe UXP changelog (Premiere 2026), HeartMuLa (Apache-2 music), face_reaging (FRAN reimplementation, MIT)

This wave closes four distinct competitive-parity and platform-modernization gaps that have accumulated since Wave L was authored a year ago:
1. **Video relighting** — DaVinci Resolve 21's flagship "Relight" / CineFocus tool has no equivalent in OpenCut. Light-A-Video (training-free) + IC-Light V2 (per-frame) + DiffusionRenderer (physically-grounded) collectively close this gap.
2. **One-step video super-resolution** — SeedVR2 supersedes the FlashVSR/Real-ESRGAN tier on quality and parity with commercial VSR (Topaz Video AI).
3. **Vision-language model refresh** — Qwen2.5-VL (N3.2) is now a generation behind Qwen3-VL and InternVL3, both Apache-2; refresh keeps OpenCut's video understanding tier competitive.
4. **Infrastructure modernization** — FFmpeg 8.0 native Whisper filter + Vulkan AV1/VP9/ProRes-RAW encoders + the UXP panel v1.0 final cutover before September 2026 CEP EOL.

---

## Wave S1 — Video Relighting Suite (v1.56.0)

**Goal**: Add the relighting capability that DaVinci Resolve 21 ships as a flagship paid AI feature. Three complementary engines: (a) per-frame image relighting via IC-Light V2 for FLUX (already partly available via Wave M2 FLUX integration); (b) temporally consistent training-free video relighting via Light-A-Video; (c) physically grounded inverse + forward rendering via NVIDIA DiffusionRenderer for full scene relight (replace lighting environment, not just colour-grade).
**New required deps**: None — all three reuse `diffusers` ≥0.32 (already present from Wave M); `cogvideox-5b` already present from Wave N.
**New routes**: ~12

### OSS Discoveries — Video Relighting

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| S1.1 | **IC-Light V2 per-frame relight** (lllyasviel/IC-Light, Apache-2, FLUX variant 2025) — Per-frame relighting on stills or per-frame on video. Two modes: text-conditioned ("studio softbox light from camera-left, warm 3200K") and background-conditioned (composite into a new HDR scene; lighting follows the new background). Reuses FLUX.1 weights already present from Wave M2.4 (FLUX Kontext). Use as a pre-filter before TokenFlow temporal propagation, or as the per-frame engine inside Light-A-Video (S1.2). | `POST /relight/iclight/text`, `POST /relight/iclight/background`, `GET /relight/iclight/info` | `core/relight_iclight.py` | `diffusers` (already present) + IC-Light V2 LoRA weights (~3 GB) via HuggingFace `lllyasviel/ic-light` | M | Apache-2 (code + LoRAs) | Closes "AI Relight" / "CineFocus" gap (DaVinci Resolve 21); per-frame; FLUX-quality |
| S1.2 | **Light-A-Video training-free video relighting** (bcmi/Light-A-Video, ICCV 2025, MIT) — Training-free zero-shot video relighting framework that combines a per-frame relighting model (IC-Light from S1.1) with a video diffusion model (CogVideoX-5B already present from Wave N3.3) using two innovations: Consistent Light Attention (CLA) to stabilise lighting across frames and Progressive Light Fusion (PLF) for natural transitions. The first temporally coherent video relighter in the open ecosystem. Use case: change lighting on existing footage ("make this clip look like golden-hour" / "make this clip look like a moonlit night") without retraining or per-clip LoRA. | `POST /relight/video/light_a_video`, `GET /relight/video/info` | `core/relight_video_lav.py` | IC-Light V2 weights (S1.1) + CogVideoX-5B weights (Wave N) — no new heavy deps | L | MIT | First training-free temporally coherent video relighting; closes DaVinci 21 Relight gap end-to-end |
| S1.3 | **DiffusionRenderer inverse + forward rendering** (nv-tlabs/diffusionrenderer, Apache-2, NVIDIA Toronto AI Lab 2025) — Generalist neural inverse renderer that decomposes a video into intrinsic G-buffers (albedo, normal, depth, roughness, metallic) and a forward renderer that re-renders the scene under arbitrary new lighting (HDR environment map, point lights, area lights). Unlike Light-A-Video (S1.2), which is a 2D pixel-space recoloring approach, DiffusionRenderer is physically grounded — supports environment-map relighting, material edits, and view-coherent multi-clip relighting. Heavy compute (~24 GB VRAM) — gates behind a `quality` flag; default to S1.2 for consumer hardware. | `POST /relight/video/diffrenderer`, `POST /relight/video/diffrenderer/decompose`, `POST /relight/video/diffrenderer/relight`, `GET /relight/video/diffrenderer/info` | `core/relight_diffrenderer.py` | `nv-tlabs/diffusionrenderer` (Apache-2) + DiffusionRenderer weights (~12 GB) via HuggingFace | XL | Apache-2 | Physically grounded relight; HDR env-map support; view-consistent multi-clip relighting; future-proof |

---

## Wave S2 — One-Step VSR + Next-Gen ASR (v1.57.0)

**Goal**: Replace the diffusion-VSR tier (currently FlashVSR + Real-ESRGAN for the smart upscaling hub from Wave L2) with SeedVR2's one-step diffusion approach (~10× faster at equal quality). Add NVIDIA Parakeet TDT and Canary-1B-Flash to the ASR fleet — both are now SOTA on English benchmarks and complement the existing Whisper Large-v3 stack with ultra-low-latency streaming and ultra-fast batch transcription.
**New required deps**: `nemo_toolkit[asr]` (Apache-2) for Parakeet/Canary; SeedVR2 reuses `diffusers`.
**New routes**: ~10

### OSS Discoveries — VSR + ASR

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| S2.1 | **SeedVR2 one-step diffusion video super-resolution** (ByteDance-Seed/SeedVR, ICLR 2026, Apache-2) — Single-step diffusion VSR that achieves comparable quality to multi-step methods (FlashVSR, VEnhancer) at ~10× the throughput. Two checkpoints: SeedVR2-3B (consumer GPU, 12 GB VRAM) and SeedVR2-7B (24 GB VRAM, higher fidelity). Both Apache-2 with open weights. Becomes the new default backend in `upscale_hub` (Wave L2.2): SeedVR2 → FlashVSR → Real-ESRGAN → Lanczos. Add a `quality` content hint that routes to SeedVR2-7B; `fast` continues to use SeedVR2-3B. | `POST /video/upscale/seedvr2`, `POST /video/upscale/smart` (default backend swap), `GET /video/upscale/seedvr2/info` | `core/upscale_seedvr2.py` (+ `core/upscale_hub.py` registration update) | `diffusers` (already present) + SeedVR2-3B (~6 GB) and SeedVR2-7B (~14 GB) weights via HuggingFace `ByteDance-Seed/SeedVR2-3B` and `SeedVR2-7B` | M | Apache-2 | One-step VSR; supersedes FlashVSR on speed and quality; closes Topaz Video AI commercial gap |
| S2.2 | **NVIDIA Parakeet TDT 0.6B v2 streaming ASR** (NVIDIA NeMo, CC-BY-4.0 model + Apache-2 NeMo toolkit) — Transducer-based streaming ASR with sub-200 ms first-chunk latency at 0.6B parameters; outperforms Whisper Large-v3 on English while running 4× faster on CPU and 10× faster on consumer GPU. Use case: live captioning during a recording session, real-time subtitle preview in the UXP panel. Ships alongside Whisper (not replacing it — Whisper remains for multilingual and translation; Parakeet is the English-streaming default). | `POST /audio/asr/parakeet`, `POST /audio/asr/parakeet/stream` (SSE), `GET /audio/asr/parakeet/info` | `core/asr_parakeet.py` | `nemo_toolkit[asr]` (Apache-2, ~500 MB) + Parakeet TDT 0.6B v2 weights (~600 MB CC-BY-4.0) via HuggingFace `nvidia/parakeet-tdt-0.6b-v2` | M | CC-BY-4.0 (model) + Apache-2 (NeMo) — both commercial-OK | Streaming ASR with sub-200 ms latency; closes ElevenLabs/AssemblyAI streaming-API gap; live preview in UXP panel |
| S2.3 | **NVIDIA Canary-1B-Flash batch ASR** (NVIDIA NeMo, CC-BY-4.0) — Batch-optimised English+multilingual ASR at RTFx 1000+ (transcribes 1 hour of audio in <4 seconds on an RTX 4090). Use case: bulk-transcribe an entire footage library overnight; fast retroactive caption generation across long-form content (podcasts, lectures). Complements Whisper (multilingual + translation) and Parakeet (streaming). | `POST /audio/asr/canary`, `POST /audio/asr/canary/batch`, `GET /audio/asr/canary/info` | `core/asr_canary.py` | `nemo_toolkit[asr]` (already added by S2.2) + Canary-1B-Flash weights (~1 GB) via HuggingFace `nvidia/canary-1b-flash` | S | CC-BY-4.0 + Apache-2 | RTFx 1000+ batch ASR; bulk library transcription overnight |
| S2.4 | **FFmpeg 8.0 native Whisper filter integration** (FFmpeg 8.0 "Huffman", Aug 2025, LGPL/GPL — code only; ggml-whisper bundled separately under MIT) — FFmpeg 8.0 ships a native `whisper` filter built on whisper.cpp/ggml. Replace the Wave A subtitle pipeline's external Whisper invocation with the native FFmpeg filter where available; fallback to existing whisper.cpp path. Eliminates one subprocess hop, simplifies the FFmpeg-Python wiring, and enables on-the-fly subtitle burn-in during transcode. Also adopt FFmpeg 8.0's Vulkan AV1/VP9/ProRes-RAW encoders for hardware acceleration on cross-vendor GPUs (replacing NVENC-only fast path). | `core/transcribe.py` (refactor) + `core/encode_vulkan.py` (new) + `GET /system/ffmpeg/info` (extend) | `core/transcribe.py`, `core/encode_vulkan.py` | FFmpeg 8.0+ binary (bundle in Windows installer; document Linux/macOS upgrade path); no Python deps | M | LGPL (FFmpeg core) — already in stack | Eliminates Whisper subprocess hop; cross-vendor Vulkan AV1 encode; modernises the codec pipeline |

---

## Wave S3 — Vision-Language Modernization + Face Re-Aging (v1.58.0)

**Goal**: Refresh the multimodal video understanding tier. Qwen2.5-VL (N3.2) is now a generation behind — Qwen3-VL extends to 256K-token context (up from 32K) and adds documented two-hour video analysis with frame-level timestamp recall. InternVL3 lands as a parallel option (different lineage, different fine-tuning, different bias profile). Add face age transformation (de-aging / age progression) — DaVinci Resolve 21 ships this as "Face Age Transformer"; the FRAN-based open-source `face_reaging` reimplementation provides a clean MIT path.
**New required deps**: None — all reuse `transformers` (already present).
**New routes**: ~10

### OSS Discoveries — VLM Refresh + Face Tools

| # | Feature | Route(s) | Module | Dep | Effort | Licence | Why it matters |
|---|---------|----------|--------|-----|--------|---------|----------------|
| S3.1 | **Qwen3-VL multimodal upgrade** (QwenLM/Qwen3-VL, Apache-2, Sept 2025; tech report Nov 2025 arXiv:2511.21631) — Replaces Qwen2.5-VL (Wave N3.2) as the default VLM. Native 256K-token context (extensible to 1M with YaRN), Interleaved-MRoPE for long-horizon video modelling, DeepStack image-language alignment, frame-level timestamp recall in videos up to 2 hours long. Matches or beats Gemini 2.5 Pro and GPT-5 on MathVista, MathVision, DocVQA, VideoMME. Available in 4B, 8B, 32B, 72B, and 235B-A22B (MoE) sizes; default to 8B for consumer GPU with 32B available as a `--quality high` opt-in. | `POST /analyze/video/qwen3vl` (replaces `qwen25vl`), `POST /analyze/video/timestamps`, `GET /analyze/video/qwen3vl/info` | `core/multimodal_qwen3vl.py` (replaces `multimodal_qwen25vl.py`; preserve old route as deprecated alias for one wave) | `transformers` ≥4.45 (update from current pin) + Qwen3-VL-8B-Instruct weights (~16 GB) via HuggingFace `Qwen/Qwen3-VL-8B-Instruct`; optional 32B (~64 GB) | M | Apache-2 | 256K-token context (8× current); 2-hour video analysis; frame-timestamp recall; closes Gemini 2.5 Pro video gap |
| S3.2 | **InternVL3 alternative VLM** (OpenGVLab/InternVL, Apache-2, April 2025) — Parallel multimodal LLM for users who want a non-Alibaba option (different training data, different bias profile). Variable Visual Position Encoding (V2PE) for long video sequences; native multimodal pretraining (not post-hoc adaptation). Available in 1B, 2B, 8B, 14B, 38B, 78B sizes. Ship as an opt-in alternative to Qwen3-VL — same `/analyze/video` route surface, switchable via `model: "qwen3vl" \| "internvl3"`. | `POST /analyze/video/internvl3`, `GET /analyze/video/internvl3/info` | `core/multimodal_internvl3.py` | `transformers` ≥4.45 (already added by S3.1) + InternVL3-8B weights (~16 GB) via HuggingFace `OpenGVLab/InternVL3-8B` | S | Apache-2 (code + weights) | Vendor diversity for VLM tier; users can pick Qwen3-VL or InternVL3 based on bias / language preferences |
| S3.3 | **face_reaging (FRAN reimplementation) face age transformation** (timroelofs123/face_reaging, MIT) — Open implementation of Disney Research's "Production-Ready Face Re-Aging for Visual Effects" (FRAN, SIGGRAPH 2022). U-Net architecture trained on synthetic age-paired data; takes a video + target age delta (-30 to +30 years) and outputs the re-aged subject with preserved identity. Combines with MediaPipe face detection (already in stack from Wave L3.3) for per-frame face crop + composition. Ships as `POST /video/face/reage` with `{video_path, target_age_delta: int, strength: 0..1}`. | `POST /video/face/reage`, `GET /video/face/reage/info` | `core/face_reage.py` | `face_reaging` (MIT, ~50 MB) + pretrained FRAN weights (~150 MB) | M | MIT | Closes DaVinci 21 "Face Age Transformer" gap; production VFX-quality age progression / regression |
| S3.4 | **HeartMuLa music generation** (HeartMuLa/heartlib, Apache-2, 2025/2026) — Family of open music foundation models (text-to-music, high-fidelity neural music codec, lyric transcription) with multilingual lyric conditioning. Complements ACE-Step (Wave L2.2), DiffRhythm (Wave M1.3), and YuE (Wave O3.1) — HeartMuLa's strength is lyric-aligned generation with precise word-level timing, useful for music-video sync workflows. Ship as an alternate engine inside the existing `/music/generate` dispatcher (Wave L); model selection via `engine: "ace-step" \| "diffrhythm" \| "yue" \| "heartmula"`. | `POST /music/generate/heartmula`, `GET /music/generate/heartmula/info` | `core/music_heartmula.py` | HeartMuLa weights (~5 GB) via HuggingFace `HeartMuLa/heartlib`; `transformers` ≥4.45 (added by S3.1) | M | Apache-2 | Lyric-aligned music generation with word-level timing; complements existing music engines for music-video sync |
| S3.5 | **UXP panel v1.0 final EOL cutover** — Adobe Premiere 2026 (April 2026) made UXP the standard with CEP slated for full removal in Premiere 2027 (~September 2026 cutoff). The Wave P3.3 milestone covered Wave L–P parity in UXP; this milestone covers Q + R + S parity (VACE, CosyVoice, MaskGCT, OmniGen2, SkyReels, EzAudio, MuseTalk, VideoX-Fun, Mochi-1, Step-Video, Light-A-Video, SeedVR2, Parakeet, Qwen3-VL, face_reage, HeartMuLa) and flips the default panel installer to UXP. CEP panel moves to **deprecated** status — security fixes only, no new features. Removed from the installer entirely once Premiere 2027 ships. | `— (panel-only, no new backend routes)` | `panel-uxp/` (Q/R/S feature wiring) + installer `bin/install-panel.ps1` (default flip) | None | L | N/A | CEP EOL <12 months out; this is the last Wave that ships any new CEP-side panel UI; UXP becomes default |

---

## Wave S: S-OSS Ecosystem Survey

| Tool | Org | Licence | Category | Wave S Role |
|------|-----|---------|----------|-------------|
| IC-Light V2 (FLUX) | lllyasviel | Apache-2 | Image Relighting | S1.1 — Per-frame text/background relighting |
| Light-A-Video | bcmi (Beihang) | MIT | Video Relighting | S1.2 — Training-free temporally coherent video relighter |
| DiffusionRenderer | NVIDIA Toronto AI Lab | Apache-2 | Inverse + Forward Render | S1.3 — Physically grounded video relighting |
| SeedVR2 | ByteDance-Seed | Apache-2 | Video Super-Resolution | S2.1 — One-step diffusion VSR (3B + 7B) |
| Parakeet TDT 0.6B v2 | NVIDIA NeMo | CC-BY-4.0 (model) + Apache-2 (toolkit) | ASR (streaming) | S2.2 — Sub-200 ms streaming ASR |
| Canary-1B-Flash | NVIDIA NeMo | CC-BY-4.0 + Apache-2 | ASR (batch) | S2.3 — RTFx 1000+ batch ASR |
| FFmpeg 8.0 "Huffman" | FFmpeg | LGPL/GPL | Codec / Filter | S2.4 — Native Whisper filter + Vulkan AV1/VP9 |
| Qwen3-VL | Alibaba Qwen | Apache-2 | Vision-Language | S3.1 — 256K context, 2-hour video analysis |
| InternVL3 | OpenGVLab | Apache-2 | Vision-Language | S3.2 — Alternative VLM lineage |
| face_reaging (FRAN) | timroelofs123 | MIT | Face VFX | S3.3 — Face age transformation (Disney FRAN) |
| HeartMuLa | HeartMuLa | Apache-2 | Music Generation | S3.4 — Lyric-aligned music with word-level timing |
| UXP (Adobe) | Adobe | N/A (platform) | Panel Migration | S3.5 — Final CEP→UXP cutover before Premiere 2027 |
| MyTimeMachine | Toronto / SIGGRAPH 2025 | Academic (unclear) | Face De-Aging | NOT ADOPTED — licence uncertain; revisit if MIT/Apache release confirmed |
| Wan 2.5 / Wan 2.6 | Alibaba | Closed (API-only) | T2V | NOT ADOPTED — closed weights; no local inference path |
| RelightVid | aleafy | Unclear | Video Relighting | NOT ADOPTED — Light-A-Video (S1.2) covers same use case under MIT |
| RelightMaster | ICLR 2026 | Paper-only at time of survey | Video Relighting | WATCH LIST — pending code release with permissive licence |

---

## Wave S: Competitive Gap Closure

| Gap | Competitor | Wave S Feature | Closes? |
|-----|-----------|---------------|---------|
| Re-light a clip after the fact (text or background) | DaVinci Resolve 21 "Relight" / CineFocus, Adobe Firefly Relight | S1.1 IC-Light V2 + S1.2 Light-A-Video | Y — text-prompt relight + background-conditioned + temporally coherent |
| Physically grounded video relighting (HDR env map) | Disney VFX house tools (no commercial parity) | S1.3 DiffusionRenderer | Y — first commodity neural inverse renderer in an editor |
| Topaz-quality video super-resolution | Topaz Video AI ($300/yr) | S2.1 SeedVR2 | Y — Apache-2; 10× faster than diffusion-VSR baselines |
| Sub-200 ms streaming ASR (live captions) | ElevenLabs streaming, AssemblyAI realtime | S2.2 Parakeet TDT | Y — local streaming with English SOTA quality |
| Bulk-transcribe a 100-hour library overnight | AssemblyAI batch API | S2.3 Canary-1B-Flash | Y — RTFx 1000+ on a single 4090 |
| Hardware-accelerated AV1 encode on AMD/Intel | NVENC-only fast paths | S2.4 FFmpeg 8.0 Vulkan AV1 | Y — cross-vendor Vulkan AV1 encoder |
| 2-hour video analysis with frame-timestamp recall | Gemini 2.5 Pro Video, GPT-5 Vision | S3.1 Qwen3-VL | Y — 256K context; documented 2-hour analysis |
| Face age transformation (de-aging / progression) | DaVinci Resolve 21 "Face Age Transformer" | S3.3 face_reaging (FRAN) | Y — production VFX-quality age delta with identity preservation |
| Lyric-aligned music with word-level timing | Suno, Udio | S3.4 HeartMuLa | Y — local Apache-2 music with lyric timing |
| Premiere 2027 forward compatibility (CEP EOL) | n/a (platform mandate) | S3.5 UXP v1.0 final | Y — fully UXP-default before CEP removal |

---

## Wave S Gotchas

- **IC-Light V2 FLUX dep**: IC-Light V2 LoRAs target FLUX.1-dev. The Wave M2.4 FLUX Kontext integration already pins FLUX.1-dev — do not bump to FLUX.1-schnell or FLUX.1-pro without re-validating IC-Light compatibility. Pin FLUX.1-dev SHA in `requirements.txt` lock alongside the IC-Light LoRA pin.
- **Light-A-Video CogVideoX coupling**: Light-A-Video uses CogVideoX-5B as the temporal video diffusion backbone. CogVideoX-5B is already shipped in Wave N3.3, but its weights live in `~/.opencut/models/cogvideox-5b/`. Reuse the existing path; do not re-download. If a user has not installed CogVideoX-5B, `GET /relight/video/info` returns `{available: false, reason: "cogvideox_5b_not_installed", install_route: "/system/models/install/cogvideox-5b"}`.
- **DiffusionRenderer VRAM**: 24 GB minimum for the full pipeline. On consumer GPUs <24 GB, gate the route behind `quality: "extreme"` and document the requirement in `/info`. Always recommend Light-A-Video (S1.2) as the consumer default — falling back automatically when VRAM probe fails.
- **DiffusionRenderer HDR input format**: DiffusionRenderer accepts `.exr` and `.hdr` environment maps. Add a small helper `POST /relight/diffrenderer/hdr/upload` that validates the HDR file is RGB float32 and not a tonemapped `.png`. Reject sRGB inputs with a clear error.
- **SeedVR2 vs FlashVSR fallback**: Update `core/upscale_hub.py` (Wave L2.2 dispatcher) to register SeedVR2 as the new default. Keep FlashVSR + Real-ESRGAN as fallbacks for users who explicitly select them or who have <12 GB VRAM. Document the auto-selection table in `GET /video/upscale/smart/info`.
- **Parakeet/Canary CC-BY-4.0**: The model weights are CC-BY-4.0, which requires attribution but allows commercial use. Add a one-line attribution to the OpenCut "About" panel ("Includes Parakeet TDT 0.6B v2 and Canary-1B-Flash by NVIDIA, licensed under CC-BY-4.0"). The NeMo toolkit itself is Apache-2.
- **NeMo Windows install**: `nemo_toolkit[asr]` requires `pynini` for text normalization, which only ships pre-built wheels for Linux. On Windows, fall back to `nemo_toolkit[asr_no_pynini]` and skip text normalization features (numbers stay as digits). Document in `core/asr_parakeet.py` install gate.
- **FFmpeg 8.0 binary distribution**: FFmpeg 8.0 ships in late 2025 as binaries via gyan.dev and BtbN; the Windows installer must bundle 8.0 (current bundled is 7.x). Update `bin/install-ffmpeg.ps1` to download FFmpeg 8.0 + ggml-whisper model files. Test the `--enable-whisper` build flag is present in the bundled binary; if absent, fall back to subprocess Whisper.
- **FFmpeg 8.0 Vulkan kernel availability**: Vulkan AV1 encode requires a Vulkan 1.3 driver (NVIDIA 535+, AMD 24.x+, Intel ARC drivers). On older drivers, fall back to NVENC/AMF/QSV. Probe via `vulkaninfo --summary` in `core/encode_vulkan.py`.
- **Qwen3-VL transformers version**: Qwen3-VL requires `transformers>=4.45.0` with the `Qwen3VLForConditionalGeneration` class. The current pin in `pyproject.toml` is older — bump as part of Wave S1.1 to avoid a surprise mid-wave migration. Validate that no other model (Qwen2.5-VL old code path, EchoMimic, Wan2.2) breaks on the upgrade — run the full multimodal test suite before merging.
- **Qwen2.5-VL backward compat**: Keep `POST /analyze/video/qwen25vl` route alive as a deprecated alias for one full wave (S1.x → T1.x). Returns the same response shape but logs a deprecation warning and a `Sunset: <date>` HTTP header. Migration guide in `docs/UPGRADE_QWEN3VL.md`.
- **InternVL3 vs Qwen3-VL prompt format**: The two models use slightly different chat templates. Implement a small adapter in `core/multimodal_dispatcher.py` that translates between OpenCut's canonical message format and each model's native template. Test both engines against the same evaluation suite to confirm semantic equivalence.
- **face_reaging dependency**: The repository is small (~50 MB) but pulls in `face-alignment` which has a torch dependency. Reuse the existing torch install (already present from Wave N).
- **face_reaging strength clamping**: Strength values >1.0 produce uncanny artifacts (eye distortion, hairline shift). Clamp to 0..1 in the route handler and document in `/info`.
- **HeartMuLa lyric format**: HeartMuLa expects time-aligned lyrics in `[mm:ss.cc] line` LRC format. Add a small helper that converts plain text → LRC by splitting on punctuation and distributing evenly across the requested duration; expose as `POST /music/generate/heartmula/lyrics_to_lrc`.
- **UXP panel CEP feature parity audit**: Before flipping UXP to default in S3.5, run the parity audit (every CEP route → corresponding UXP renderer + UI control). Track in `panel-uxp/PARITY_AUDIT.md`. Specifically validate: GPU topology display, GGUF model status, all Wave Q/R/S new routes, MCP server status, async job queue. Any feature that fails parity stays CEP-only and the UXP panel shows a "fall back to CEP for this feature" link.

---

## Wave S Shipping Cadence

| Release | ETA | Features |
|---------|-----|---------|
| v1.56.0 | 2028-Q4 | S1.1 IC-Light V2 per-frame, S1.2 Light-A-Video training-free video relighting, S1.3 DiffusionRenderer (gated) |
| v1.57.0 | 2029-Q1 | S2.1 SeedVR2 one-step VSR (default backend), S2.2 Parakeet TDT streaming, S2.3 Canary-1B-Flash batch, S2.4 FFmpeg 8.0 + Vulkan AV1 |
| v1.58.0 | 2029-Q1 | S3.1 Qwen3-VL upgrade, S3.2 InternVL3 alternative, S3.3 face_reaging, S3.4 HeartMuLa, S3.5 UXP v1.0 final cutover |

---

## Wave S: Not Adopted / Deferred

- **MyTimeMachine** (SIGGRAPH 2025) — Personalised facial age transformation (50-selfie reference). Strong quality but academic licence is unclear at time of this survey; the GitHub repo carries no `LICENSE` file. NOT adopted until a permissive (MIT/Apache-2/BSD) licence is published. Watch list. face_reaging (S3.3) covers the use case under MIT.
- **Wan 2.5 / Wan 2.6** (Alibaba, Sept 2025 / Dec 2025) — Closed-weight cloud-only release; no downloadable inference path. NOT adoptable as a local engine. Wan 2.1 + Wan 2.2 (already in stack from Wave M / N) remain the supported Wan tier. Monitor for an open-weights re-release.
- **RelightVid** (aleafy) — Temporal-consistent video relighting; functionality overlaps with Light-A-Video (S1.2). NOT adopted to avoid two-engine maintenance burden in the same niche.
- **RelightMaster** (ICLR 2026 paper) — Multi-plane light images for precise video relighting. Code release pending at survey time. WATCH LIST — re-evaluate for Wave T once code drops with a permissive licence.
- **HunyuanVideo 1.5** — Tencent released a 1.5 update (Q1 2026) but the geographic licence restrictions from Wave O still apply (EU/UK/SK excluded). Hard pass; remains on the rejected list.
- **Stable Audio 2.5 / Stable Audio Open community models** — Stability AI Community Licence remains non-commercial. NOT adopted; HeartMuLa (S3.4) covers the music gap.
- **AI CineFocus / aperture simulation** — DaVinci Resolve 21's depth-of-field synthesis effect. The optical-flow-based focus simulation is implementable but the "click to refocus" UX requires a depth model + a synthetic-bokeh renderer. Wave R already shipped Mochi-1 and Wave N shipped DepthAnythingV2; revisit in Wave T as a higher-level UX layer on top of those building blocks rather than a new model integration.
- **AI Slate ID / IntelliSearch** — DaVinci 21 AI search + clapperboard reading. Already shipped in Wave K (`/slate/identify`, `/search/ai`). Confirmed parity exists; no Wave S work needed.
- **Captions.ai AI Twin / generative actors** — Closed-source synthetic actor library; OpenCut already provides the building blocks (SkyReels V3 A2V + ConsisID + MuseTalk). The "AI Twin" UX is a panel-side wizard, not a new engine — schedule as a Wave T panel feature, not a model integration.
- **Submagic Magic Clips / Opus Clip viral scoring** — Closed-source viral-moment scoring. Wave M sports highlights (`/video/highlights/sports`) covers the engine; the viral-scoring model itself is proprietary. A heuristic open-source equivalent (audio energy + scene change + caption sentiment) is feasible but defers to Wave T as a higher-level pipeline rather than a new model.

---

## Wave S Sources

- **IC-Light V2** — [lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light) (Apache-2); FLUX variant 2025; ComfyUI nodes by kijai; HuggingFace `lllyasviel/ic-light`
- **Light-A-Video** — [bcmi/Light-A-Video](https://github.com/bcmi/Light-A-Video) (MIT, ICCV 2025); [arXiv:2502.08590](https://arxiv.org/abs/2502.08590); project page `bujiazi.github.io/light-a-video.github.io`
- **DiffusionRenderer** — [nv-tlabs/diffusionrenderer](https://github.com/nv-tlabs/diffusionrenderer) (Apache-2, NVIDIA Toronto AI Lab 2025); project page `research.nvidia.com/labs/toronto-ai/DiffusionRenderer`
- **SeedVR2** — [ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR) (Apache-2, ICLR 2026); HuggingFace `ByteDance-Seed/SeedVR2-3B` and `ByteDance-Seed/SeedVR2-7B`
- **Parakeet TDT 0.6B v2** — [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo) (Apache-2 toolkit); HuggingFace `nvidia/parakeet-tdt-0.6b-v2` (CC-BY-4.0 model)
- **Canary-1B-Flash** — [NVIDIA/NeMo](https://github.com/NVIDIA/NeMo); HuggingFace `nvidia/canary-1b-flash` (CC-BY-4.0); RTFx 1000+ benchmarks documented on the HF model card
- **FFmpeg 8.0 "Huffman"** — [FFmpeg release notes](https://ffmpeg.org/index.html#pr8.0); August 2025; native Whisper filter (`af_whisper`), Vulkan AV1 encoder, Vulkan VP9 / ProRes-RAW acceleration, VAAPI VVC decode; [Phoronix coverage](https://www.phoronix.com/news/FFmpeg-8.0-Released)
- **Qwen3-VL** — [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (Apache-2, Sept 2025); [arXiv:2511.21631](https://arxiv.org/abs/2511.21631) tech report Nov 2025; HuggingFace `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-32B-Instruct`, `Qwen/Qwen3-VL-235B-A22B-Instruct`
- **InternVL3** — [OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL) (Apache-2); [InternVL3 blog](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/); HuggingFace `OpenGVLab/InternVL3-8B`, `OpenGVLab/InternVL3-78B`
- **face_reaging (FRAN reimplementation)** — [timroelofs123/face_reaging](https://github.com/timroelofs123/face_reaging) (MIT); upstream paper: Disney Research "Production-Ready Face Re-Aging for Visual Effects" (FRAN, SIGGRAPH 2022)
- **HeartMuLa** — [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) (Apache-2, 2025/2026); music foundation model family
- **Adobe UXP for Premiere 2026** — [developer.adobe.com/premiere-pro/uxp/](https://developer.adobe.com/premiere-pro/uxp/) — UXP standard since Premiere 2026 (April 2026); CEP scheduled for full removal in Premiere 2027 (~September 2026 timeline); migration guide `developer.adobe.com/premiere-pro/uxp/guides/cep-migration/`
- **DaVinci Resolve 21** — [What's New](https://www.blackmagicdesign.com/products/davinciresolve/whatsnew); 2026; flagship "Relight" / CineFocus, Face Age Transformer, IntelliSearch, Magic Mask v2 — used as a competitive parity reference for S1, S3.3, and the Wave S Competitive Gap Closure table
- **NVIDIA NeMo ASR comparison** — [Best open-source STT 2026 — Northflank](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks); [Gladia 2026 STT benchmarks](https://www.gladia.io/blog/best-open-source-speech-to-text-models)
- **Awesome AI Video Editing 2026** — [GagnDeep/awesome-best-ai-tools-for-video-editors-2026](https://github.com/GagnDeep/awesome-best-ai-tools-for-video-editors-2026); [awesome-ai-tools/curated-ai-image-video](https://github.com/awesome-ai-tools/curated-ai-image-video) — surveyed as the discovery surface for OSS ecosystem state

---
