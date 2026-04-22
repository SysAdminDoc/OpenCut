# OpenCut — Next-Wave Roadmap (2026-Q2 → 2026-Q4)

**Version**: 1.7
**Created**: 2026-04-17 (updated 2026-04-17 after v1.24.0 ship)
**Baseline**: v1.24.0 (1,241 routes, 460 core modules, 7,689+ tests)
**Source**: Synthesised from an OSS survey of LosslessCut, auto-editor, editly,
Descript, Shotcut/MLT, Olive, OpenShot, Kdenlive, OpenTimelineIO, WhisperX,
PyAV, and a 2024–2026 scan of new SOTA AI video models (see research notes
under *Sources* at the bottom).

> **Scope**: This document extends [ROADMAP.md](ROADMAP.md) — the
> original Wave 1–7 plan. Anything already covered there is not repeated
> here; this is **only the incremental work discovered after the v1.16.3
> cross-project research pass**.

---

## Guiding Principles (carried forward)

1. **Never break what works.** Every wave ships independently.
2. **One new required dependency per feature maximum.** Prefer optional
   pip extras with graceful degradation (`@async_job` + `checks.py`).
3. **Permissive licences only.** Apache-2, MIT, BSD, LGPL are fine.
   CC-BY-NC, research-only, or unclear licences are deferred until the
   author clarifies.
4. **Match existing patterns** — `InterpResult` / `ComposeResult` /
   `MEMixResult` / `PremixResult` style: subscriptable dataclass, single
   `run()` entry point, `on_progress(pct, msg="")` callback default-arg,
   queue allowlist entry, `check_X_available()` guard.
5. **Frontend parity last.** CEP panel first, UXP second, CLI/MCP
   third — but never on the same PR as the backend addition.

---

## v1.17.0 — Shipped (2026-04-17)

| # | Feature | Core Module | Routes | OSS Source |
|---|---------|-------------|--------|------------|
| 17.1 | **Neural Frame Interpolation** — RIFE-NCNN-Vulkan CLI with FFmpeg `minterpolate` fallback. 3-pass doubling cap (8×). `InterpResult` dataclass. | `core/neural_interp.py` | `GET /video/interpolate/backends`, `POST /video/interpolate/neural` | [rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan) |
| 17.2 | **Declarative JSON Video Composition** — editly-inspired, 18 xfade transitions, 4 clip types, background audio ducking, drawtext captions. `ComposeResult` dataclass. | `core/declarative_compose.py` | `GET /compose/schema`, `POST /compose/validate`, `POST /compose/render` | [editly](https://github.com/mifi/editly) |

**Status**: Merged. Queue allowlist updated. Lint clean. Blueprint
registered (`enhancement_bp`). Version synced to v1.17.0.

## v1.18.0 — Shipped (2026-04-17)

First production batch from Wave A + Wave D. All 10 features built with graceful degradation via `check_X_available()` — routes return 503 `MISSING_DEPENDENCY` with install hints when optional backends are absent.

| # | Feature | Module | Routes | Source |
|---|---------|--------|--------|--------|
| 18.1 / A1.3 | **F5-TTS zero-shot voice clone** | `core/tts_f5.py` | `POST /audio/tts/f5`, `GET /audio/tts/f5/models` | [F5-TTS](https://github.com/SWivid/F5-TTS) |
| 18.2 / A1.4 | **WhisperX `--diarize` exposure** | `core/captions.py` (extended), `utils/config.py::CaptionConfig` | — (inside existing `/captions` routes via new config flags) | [WhisperX](https://github.com/m-bain/whisperX) + [pyannote](https://github.com/pyannote/pyannote-audio) |
| 18.3 / A1.5 | **BeatNet downbeat detection** | `core/beats_beatnet.py` | `POST /audio/beats/beatnet` | [BeatNet](https://github.com/mjhydri/BeatNet) |
| 18.4 / A2.1 | **Scene-detect auto-dispatcher** (TransNetV2 → PySceneDetect → FFmpeg) | `core/scene_detect.py::detect_scenes_auto` | `POST /video/scenes/auto` | [TransNetV2](https://github.com/soCzech/TransNetV2) |
| 18.5 / A2.4 | **CLIP-IQA+ clip quality scoring** | `core/clip_quality.py` | `POST /video/quality/score`, `POST /video/quality/rank` | [CLIP-IQA](https://github.com/IceClear/CLIP-IQA) |
| 18.6 / A2.5 | **HSEmotion emotion arc** | `core/emotion_arc.py` | `POST /video/emotion/arc` | [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) |
| 18.7 / A4.1 | **ab-av1 VMAF-target encode** | `core/ab_av1.py` | `POST /video/encode/vmaf-target`, `GET /video/encode/vmaf-target/info` | [ab-av1](https://github.com/alexheretic/ab-av1) |
| 18.8 / A5.1 | **OTIO AAF export** (Avid interchange) | `export/otio_export.py::export_aaf` | `POST /timeline/export/aaf` | [otio-aaf-adapter](https://github.com/OpenTimelineIO/otio-aaf-adapter) |
| 18.9 / A5.2 | **OTIOZ bundle export** (portable handoff) | `export/otio_export.py::export_otioz` | `POST /timeline/export/otioz` | OpenTimelineIO built-in |
| 18.10 / D1.2 | **Broadcast compliance profiles** (EBU-TT-D, YouTube-broadcast, accessibility) | `core/caption_compliance.py` (extended) | `GET /captions/compliance/standards` | [SubtitleEdit rulesets](https://github.com/SubtitleEdit/subtitle-edit) |
| 18.11 / D6.1 | **Event moment finder** (wedding / ceremony) | `core/event_moments.py` | `POST /events/moments` | [clip-detector](https://github.com/davidmigloz/clip-detector) + YAMNet stub |

**Status**: Merged. 1,192 total routes (+15 vs v1.17.0). Lint clean on all new files. 8 new `check_*_available()` entries in `opencut/checks.py`. Queue allowlist extended.

## v1.19.0 — Shipped (2026-04-17)

Second batch closes the remaining Wave A items plus Wave D2 restoration pack and the D3.2 webhook auto-emit hook. All 7 core modules ship with graceful degradation; every restoration module has either an ONNX path via a user-supplied checkpoint env var, or a lighter FFmpeg fallback.

| # | Feature | Module | Routes | Source |
|---|---------|--------|--------|--------|
| 19.1 / A2.3 | **BiRefNet still/keyframe matte** | `core/matte_birefnet.py` | `POST /video/matte/birefnet` | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) |
| 19.2 / A3.1 | **Karaoke captions (6 libass presets)** | `core/captions_karaoke_adv.py` | `POST /captions/karaoke-adv/render`, `GET /captions/karaoke-adv/presets` | [libass](https://github.com/libass/libass) + [PyonFX](https://github.com/CoffeeStraw/PyonFX) hook |
| 19.3 / A4.2 | **SVT-AV1-PSY encoder (3 presets)** | `core/svtav1_psy.py` | `POST /video/encode/svtav1-psy`, `GET /video/encode/svtav1-psy/info` | [SVT-AV1-PSY](https://github.com/gianni-rosato/svt-av1-psy) |
| 19.4 / D2.1 | **DDColor B&W colorisation** | `core/colorize_ddcolor.py` | `POST /video/restore/colorize` | [DDColor](https://github.com/piddnad/DDColor) |
| 19.5 / D2.2 | **VRT / RVRT unified restoration** | `core/restore_vrt.py` | `POST /video/restore/vrt` | [VRT](https://github.com/JingyunLiang/VRT) |
| 19.6 / D2.3 | **Neural deflicker (+ FFmpeg fallback)** | `core/deflicker_neural.py` | `POST /video/restore/deflicker`, `GET /video/restore/backends` | [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) |
| 19.7 / D3.2 | **Webhook auto-emit on job completion** | `jobs.py::_emit_job_webhook`, existing `core/webhook_system.py::fire_event` | `GET /webhooks/events` | built-in |

## v1.20.0 — Shipped (2026-04-17)

Clears the remaining Wave D items identified in the initial research pass. Two items (D1.1 audio description, D6.2 auto-quiz) became *LLM enrichments* of existing modules rather than new-from-scratch duplicates — `core/audio_description.py` already had a complete template-based pipeline and `core/quiz_overlay.py` already had a TF-IDF generator.

| # | Feature | Module | Routes | Source |
|---|---------|--------|--------|--------|
| 20.1 / D3.1 | **Semantic OTIO timeline diff** | `export/otio_diff.py` | `POST /timeline/otio-diff` | [OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) |
| 20.2 / D4.1 | **Objective quality metrics (VMAF/SSIM/PSNR)** | `core/quality_metrics.py` | `POST /video/quality/compare`, `POST /video/quality/batch-compare`, `GET /video/quality/backends` | [libvmaf](https://github.com/Netflix/vmaf) + FFmpeg filters |
| 20.3 / D5.2 | **Sentry / GlitchTip optional observability** | `server.py::_init_sentry_if_configured` | `GET /observability/status` | [GlitchTip](https://glitchtip.com/) / [Sentry](https://sentry.io) |
| 20.4 / D1.1 | **LLM-enriched audio description** (extends v1.15-era `audio_description.py`) | `core/audio_description.py::describe_scene_llm` | — (inside existing `/audio/audio-description` routes) | `core/llm.py` + template fallback |
| 20.5 / D6.2 | **LLM-enriched auto-quiz** (extends v1.15-era `quiz_overlay.py`) | `core/quiz_overlay.py::generate_quiz_questions_llm` | — (inside existing `/api/education/quiz-*` routes) | `core/llm.py` + TF-IDF fallback |

**Status**: Merged. 1,207 total routes (+5 vs v1.19.1). Lint clean. 4 new `check_*_available()` entries. Queue allowlist extended for the three async Wave C routes.

### Already-shipped Wave A items detected during the v1.18.0 pass (skipped, noted for reference)

- **A1.1 BS-RoFormer** — already available via the `backend="audio-separator"` path of `POST /audio/separate` (models include `bs_roformer`, `mel_band_roformer`, `scnet`, `mdx23c`).
- **A1.2 Chatterbox TTS** — already shipped in `core/voice_gen.py::chatterbox_generate()`.
- **A2.2 Depth Anything V2** — already shipped in `core/depth_effects.py` (uses `depth-anything/Depth-Anything-V2-*-hf` HF repos).

### Was in research shortlist but already implemented
- `faster-whisper + WhisperX` word alignment — present in
  `core/captions.py::_transcribe_whisperx`.
- `DeepFilterNet` Studio Sound — present in `core/audio_pro.py` and
  `routes/audio.py::/audio/pro/deepfilter`.
- `OpenTimelineIO` export — present in `opencut/export/otio_export.py`.
- `EBU R128` two-pass loudnorm — present in `core/loudness_match.py`
  (+ `/audio/loudness-match` and batch variant).
- `LosslessCut` GOP-aware smart cut — present in `core/smart_render.py`
  (stream-copy unchanged GOPs, re-encode only the edges).
- `Descript-style` transcript editing — present in
  `core/transcript_edit.py` + `core/transcript_timeline_edit.py` (+
  `routes/transcript_edit_routes.py`).

---

## Wave A — Next 2 Weeks (Small / Medium Effort, High ROI)

Targeted for v1.18.0 → v1.19.0. All items are permissive-license, one pip
dep each, degrade gracefully when the dep is absent.

> **Shipped in v1.18.0** (2026-04-17): A1.3 F5-TTS, A1.4 WhisperX diarise,
> A1.5 BeatNet, A2.1 TransNetV2 promotion, A2.4 CLIP-IQA+, A2.5 HSEmotion,
> A4.1 ab-av1, A5.1 OTIO AAF, A5.2 OTIOZ, D1.2 broadcast compliance
> profiles, D6.1 event moments. The sub-tables below show remaining
> Wave A items — A3.1 karaoke captions, A4.2 SVT-AV1-PSY are still open.
> A1.1 / A1.2 / A2.2 / A2.3 were discovered already present during the
> v1.18.0 build (BS-RoFormer via audio-separator, Chatterbox via voice_gen,
> Depth-Anything-V2 via depth_effects, BiRefNet still open).

### A1 — Audio & Voice

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| A1.1 | **BS-RoFormer stem separation** (band-split rotary transformer, beats Demucs v4 by ~1 dB SDR on SDX leaderboard). Backend option for existing `/audio/separate`. | `core/stems_bsroformer.py` (backend of `core/stem_remix.py`) | [BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) | MIT | S | Best-in-class stem quality without replacing Demucs. |
| A1.2 | **Chatterbox voice cloning** — Resemble AI's 2025 OSS TTS, beats ElevenLabs on MOS, emotion-exaggeration slider. Add as a provider in existing voice_speech_routes. | `core/tts_chatterbox.py` | [chatterbox](https://github.com/resemble-ai/chatterbox) | MIT | S | Best OSS voice clone available; drop-in for existing TTS pipeline. |
| A1.3 | **F5-TTS zero-shot clone** — flow-matching, 15 s reference, faster than XTTS-v2. Alternative provider alongside Chatterbox. | `core/tts_f5.py` | [F5-TTS](https://github.com/SWivid/F5-TTS) | MIT | S | Faster inference path for live / preview use. |
| A1.4 | **WhisperX `--diarize` exposure** — existing `whisperx` backend already supports diarisation; expose the HF-token-gated flag through `CaptionConfig`. | `core/captions.py` (flag add) | [WhisperX](https://github.com/m-bain/whisperX) | BSD | S | Already installed; zero new deps. |
| A1.5 | **BeatNet downbeat detection** — CRNN + particle filter, beats librosa/madmom on downbeats. New backend for existing `/audio/beats`. | `core/beats_beatnet.py` | [BeatNet](https://github.com/mjhydri/BeatNet) | MIT | S | Music-video auto-cut-to-beat accuracy. |

### A2 — Video Intelligence

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| A2.1 | **TransNetV2 shot-boundary** — already referenced in `CLAUDE.md`; verify `check_transnetv2_available()` gate is wired and add as preferred backend over FFmpeg scene threshold. | `core/scene_detect.py` (promote TransNetV2 to default when installed) | [TransNetV2](https://github.com/soCzech/TransNetV2) | MIT | S | Already checked; promotion only. |
| A2.2 | **Depth Anything V2** — parallax / 3D-photo FX from a single still. Alternative backend for existing depth routes. | `core/depth_v2.py` | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2 | S | Cleanest depth on static scenes; enables "Ken Burns on steroids". |
| A2.3 | **BiRefNet matte** — CVPR'24, dominates DIS/HRSOD leaderboards. Use for stills / keyframes where RVM temporal noise is acceptable loss for edge quality. | `core/matte_birefnet.py` | [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | MIT | S | Higher edge quality than RVM on thumbnails / title-card art. |
| A2.4 | **CLIP-IQA+ clip quality scoring** — zero-shot sharpness / aesthetic / exposure. Rank best takes, auto-reject shaky / OOF footage. | `core/clip_quality.py` | [CLIP-IQA](https://github.com/IceClear/CLIP-IQA) | Apache-2 | S | Enables automated "best take" picker. |
| A2.5 | **HSEmotion per-frame arc** — emotion timeline graph over a clip for engagement analysis. | `core/emotion_arc.py` | [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) | Apache-2 | S | Feeds existing `emotion_timeline.py`; export to PNG arc. |

### A3 — Caption & Typography

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| A3.1 | **libass + pyonfx karaoke captions** — expose `\kf`, `\t`, `\move` advanced ASS tags; pyonfx for per-syllable FX. Extends `core/animated_captions.py`. | `core/captions_karaoke_adv.py` | [libass](https://github.com/libass/libass), [pyonfx](https://github.com/CoffeeStraw/PyonFX) | ISC / LGPL | S | Aegisub-grade karaoke without Aegisub. |

### A4 — Encoding & Delivery

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| A4.1 | **ab-av1 VMAF-target encode** — Netflix per-title lite: "give me target VMAF 95, you pick the CRF". | `core/ab_av1.py` | [ab-av1](https://github.com/alexheretic/ab-av1) | MIT | S | One-click "quality target" preset in `export_presets.json`. |
| A4.2 | **SVT-AV1-PSY** — perceptually tuned AV1 fork used by AV1 enthusiast community. Drop-in binary. | `core/av1_export.py` (backend) | [SVT-AV1-PSY](https://github.com/gianni-rosato/svt-av1-psy) | BSD | S | Better visual quality at same bitrate; no API change. |

### A5 — Interchange

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| A5.1 | **OTIO AAF adapter** — `pip install otio-aaf-adapter`; enables Avid Media Composer round-trip through existing `export/otio_export.py`. | `export/aaf_export.py` wrapper | [otio-aaf-adapter](https://github.com/OpenTimelineIO/otio-aaf-adapter) | Apache-2 | S | Unlocks Avid workflows; 1-line install. |
| A5.2 | **OTIOZ bundles** — zipped timeline + media; add `.otioz` as an export format option. | `export/otioz_export.py` | OpenTimelineIO built-in | Apache-2 | S | Portable project handoff. |

**Wave A total**: ~13 features, ~13 new pip deps (all optional extras),
~14 new routes, zero breaking changes.

---

## Wave B — Next Quarter (Medium / Large Effort, Moderate Risk)

Targeted for v1.20.0 → v1.22.0.

### B1 — AI Lip-Sync & Talking Video

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| B1.1 | **LatentSync 1.6** diffusion lip-sync | [LatentSync](https://github.com/bytedance/LatentSync) | Apache-2 | M | ~5 GB model; requires GPU. Unlocks dubbing/ADR without reshoots. |
| B1.2 | **MuseTalk 1.5** real-time latent-space inpainting (30 fps @ 256²) | [MuseTalk](https://github.com/TMElyralab/MuseTalk) | MIT | M | Faster than LatentSync for live preview; ships pip + checkpoint. |

Ship **one** of these — pick LatentSync for quality, MuseTalk for speed.
Build behind a feature flag until feedback clarifies the trade-off.

### B2 — Pro Color Pipeline

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| B2.1 | **OpenColorIO 2.4 + ACES 1.3** | [OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | BSD | M | `PyOpenColorIO` pip. Replaces ad-hoc LUT handling with film-grade pipeline. |
| B2.2 | **colour-science scopes** — CIECAM02-UCS, proper vectorscope math (not RGB histograms) | [colour-science](https://github.com/colour-science/colour) | BSD | S | Ships `scipy`/`numpy` deps already present. |

### B3 — AI Video Generation (Real-Time-ish)

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| B3.1 | **LTX-Video 0.9.5** — 2B DiT, real-time on RTX 4090 (2 s output per 2 s compute) | [LTX-Video](https://github.com/Lightricks/LTX-Video) | Apache-2 | M | Only OSS gen model fast enough for editor loop. |
| B3.2 | **CogVideoX-5B + CogVideoX-Fun ControlNet** | [CogVideo](https://github.com/THUDM/CogVideo) | Apache-2 | M | ControlNet-for-video exists now; edits existing footage. |

### B4 — Diarisation & Scene Understanding

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| B4.1 | **NeMo Sortformer / MSDD** end-to-end diarisation | [NeMo](https://github.com/NVIDIA/NeMo) | Apache-2 | M | Handles overlapped speech better than pyannote; pip-heavy. |
| B4.2 | **pyannote 3.3 + diart** streaming diarisation | [diart](https://github.com/juanmc2005/diart) | MIT | S | Real-time speaker labels during recording. |
| B4.3 | **InternVideo2-small** highlight auto-picker (300 M) | [InternVideo2](https://github.com/OpenGVLab/InternVideo2) | Apache-2 | M | Feeds existing `highlights.py` LLM scorer with visual features. |

### B5 — Delivery

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| B5.1 | **libvvenc / libvvdec (VVC / H.266)** via FFmpeg 7.1+ | [vvenc](https://github.com/fraunhoferhhi/vvenc) | BSD-3 | S | 30 % smaller than HEVC at equal VMAF. Ships with FFmpeg rebuild. |
| B5.2 | **Shaka Packager** HLS / DASH + CENC DRM | [Shaka Packager](https://github.com/shaka-project/shaka-packager) | MIT | M | Proper adaptive-streaming delivery. CLI subprocess. |
| B5.3 | **aiortc WebRTC browser preview** | [aiortc](https://github.com/aiortc/aiortc) | BSD | M | Live preview in CEP/UXP panel without CEP-specific hacks. |
| B5.4 | **SRT streaming output** via `ffmpeg -f mpegts srt://` | [srt](https://github.com/Haivision/srt) | MPL-2 | S | Low-latency contribution feed. |

**Wave B total**: ~11 features, ~5 new pip deps, ~15 new routes.

---

## Wave C — 6-Month Horizon (Large / Extra-Large, Heavy Deps or Research)

Targeted for v1.23.0+. All of these are aspirational — schedule only
after usage metrics confirm demand.

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| C1 | **MLT headless alternate render backend** — Kdenlive's GPU-OpenGL pipeline, XML timeline, deterministic renders | [MLT](https://github.com/mltframework/mlt) | LGPL / GPL | L | Alternative to FFmpeg concat for complex timelines. CLI binary. |
| C2 | **VGGT (Meta, CVPR'25)** feed-forward 3D from video — enables true 2D→3D dolly / parallax | [VGGT](https://github.com/facebookresearch/vggt) | Apache-2 | L | Heavy GPU; niche but wow-factor. |
| C3 | **gsplat** Gaussian Splatting B-roll pipeline | [gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2 | L | Photogrammetric B-roll from phone footage. CUDA lib. |
| C4 | **Wan 2.1 (Alibaba, 14B)** text-to-video — beats Sora on VBench | [Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache-2 | L | 80 GB VRAM for 720p. Power-user path only. |
| C5 | **ProRes on Windows via native encoder** — once FFmpeg ships official `prores_ks` on Windows builds, re-verify our existing implementation and update presets. | FFmpeg built-in | LGPL | S | No new code — follow-up check. |

### Rejected or wait

- **MASt3R**, **MatAnyone** — non-commercial research licences. Revisit
  if the authors relicense.
- **HunyuanVideo** — 80 GB VRAM excludes 99 % of users. Not worth
  shipping as optional unless the distilled variant arrives.
- **SadTalker, LivePortrait, MODNet, Bark, Coqui XTTS-v2** —
  superseded by Chatterbox / F5-TTS / BiRefNet / MuseTalk or the upstream
  project is abandoned.
- **Sora, Gen-3, Veo, Kling** — closed APIs. No OSS equivalent worth
  tracking beyond LTX-Video / Wan 2.1 / CogVideoX.
- **PerceptualVMAF as a codec** — it's a metric, not a codec. Already
  usable through `ab-av1` (Wave A4.1).

---

## Wave D — Breadth Pass (Next 2 Weeks, parallelisable with Wave A)

Discovered during the niche / verticals / accessibility / developer
survey. All are small enough to ship alongside Wave A without delaying
it.

### D1 — Accessibility & Compliance

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D1.1 | **Audio-description auto-generation** — scene-tag + LLM + TTS pipeline that targets silence gaps (reuses existing Whisper gap detection + Piper TTS). | `core/audio_description_ai.py` | [audio-describe](https://github.com/openai-community/audio-describe) | MIT | M | Unlocks blind-viewer market; **no current OSS editor ships this turnkey**. |
| D1.2 | **Broadcast compliance validator (full rulesets)** — port Netflix / BBC / EBU-TT-D / FCC rule JSON into the existing `caption_compliance.py`, fail-fast before export. | `core/caption_compliance.py` (extend) | [SubtitleEdit rules](https://github.com/SubtitleEdit/subtitle-edit) (GPL-3 — port the rule JSON, not the code) | GPL-3 data, new Python code MIT | S | Prevents client rejection on broadcast deliverables. |

### D2 — Archive & Restoration

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D2.1 | **DDColor AI colorisation** — dual-decoder, beats DeOldify on LPIPS / FID. | `core/colorize_ddcolor.py` | [DDColor](https://github.com/piddnad/DDColor) | Apache-2 | M | DeOldify is stale; DDColor is the 2024 SOTA. |
| D2.2 | **VRT / RVRT unified restoration** — single transformer for denoise + deblur + super-res in one pass. | `core/restore_vrt.py` | [VRT](https://github.com/JingyunLiang/VRT) | Apache-2 | M | One model replaces three current filters. |
| D2.3 | **All-In-One-Deflicker** — CVPR 2023 neural deflicker, active 2024. | `core/deflicker_neural.py` | [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) | MIT | S | Better than FFmpeg `deflicker` on timelapse and old footage. |

### D3 — Collaboration & Review

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D3.1 | **OpenTimelineIO diff/merge** — semantic timeline diff, git-style merge. | `export/otio_diff.py` | [otio-diff](https://github.com/OpenTimelineIO/otio-diff) | Apache-2 | M | True "git for video" for XML/OTIO timelines. |
| D3.2 | **Webhook emitter on export** — OTIO + FastAPI-style webhook pattern (inside existing Flask); POST to Discord / Slack on job completion. | `core/webhook_emit.py` (extend existing `webhook_system.py` if present) | pattern only | MIT | S | Plugs into existing review loops; no new dep. |

### D4 — Dev SDK & Testing

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D4.1 | **ffmpeg-quality-metrics golden tests** — VMAF / SSIM / PSNR harness for render regression, wired into CI. | `tests/quality_harness.py` + `.github/workflows/quality.yml` | [ffmpeg-quality-metrics](https://github.com/slhck/ffmpeg-quality-metrics) | MIT | S | Catches silent visual regressions the lint can't see. |
| D4.2 | **Atheris fuzz tests on parsers** — SRT, VTT, OTIO, FCP XML, ASS parsers. | `tests/fuzz/` | [Atheris](https://github.com/google/atheris) | Apache-2 | S | Cheap defence against malformed-input RCE / DoS. |

### D5 — Real-Time & Observability

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D5.1 | **ffmpeg.wasm thumbnail/waveform** — move cheap panel operations to client-side WebAssembly; kill backend round-trips. | `extension/com.opencut.uxp/wasm/` | [ffmpeg.wasm](https://github.com/ffmpegwasm/ffmpeg.wasm) | LGPL-2.1 | S | Instant thumbnails during timeline scrub; frees backend for heavy work. |
| D5.2 | **GlitchTip crash aggregation** — Sentry-protocol OSS self-host. Drop-in Sentry SDK. | `opencut/obs.py` | [GlitchTip](https://glitchtip.com/) | MIT | S | Render crash visibility across production installs. |
| D5.3 | **Plausible render-time telemetry** — self-host; custom events per route. | (config only) | [Plausible](https://github.com/plausible/analytics) | AGPL-3 | S | Surface slow-endpoint regressions. |

### D6 — Niche Vertical Quick-Wins

| # | Feature | Module (new) | OSS Source | Licence | Effort | Why |
|---|---------|--------------|------------|---------|--------|-----|
| D6.1 | **Wedding event-moment finder** — audio-energy + ML spike detector for "first kiss / first dance / ring exchange" timestamps. Plugs into existing `silence.py` / `highlights.py` infrastructure. | `core/event_moments.py` | [clip-detector](https://github.com/davidmigloz/clip-detector) | MIT | S | Zero-effort "highlight my wedding" feature. |
| D6.2 | **Auto-quiz generation** for lecture captures — Whisper transcript + Claude structured-output → WebVTT cue quiz overlays. | `core/edu_quiz.py` | pattern (pieces already present) | MIT | S | Unlocks the education / training market. |
| D6.3 | **Gemini 2.0 Video Understanding** — free-tier frame-level scene description as optional LLM provider. | `core/llm.py` (provider add) | closed weights, free API | — | S | Broadens existing LLM abstraction to 4th provider; no pip dep. |

---

## Wave E — Strategic Larger Bets (Next Quarter)

Parallel track with Wave B. These are the high-leverage L/XL items from
the niche pass.

| # | Feature | OSS Source | Licence | Effort | Notes |
|---|---------|------------|---------|--------|-------|
| E1 | **OpenFX (OFX) plugin host** — load standard `.ofx` VFX plugins as FFmpeg filter_complex wrappers. Unlocks the DaVinci / Nuke / Resolve plugin ecosystem. | [OpenFX](https://github.com/ofxa/openfx) | BSD | XL | C shim + Python bridge. Highest-leverage dev-platform bet. |
| E2 | **VapourSynth frameserver** — Python-scriptable frameserver with massive filter library. OpenCut emits `.vpy`, VS serves frames to FFmpeg. | [VapourSynth](https://github.com/vapoursynth/vapoursynth) | LGPL-2.1 | M | Opens the entire VS plugin catalogue. |
| E3 | **LTX-Video-Agent MCP tool server** — turn OpenCut routes into tools for Lightricks' agentic shot-planner SDK. | [LTX-Video](https://github.com/Lightricks/LTX-Video) | Apache-2 | L | Agentic editing is the emerging paradigm; arrive early. |
| E4 | **WhisperX voice-command grammar** — live mic → grammar parser (`"cut here"`, `"slip 4 frames"`, `"mark"`) → existing timeline ops. | [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) + custom grammar | MIT | M | Hands-free editing for accessibility + speed. |
| E5 | **Flamenco 3 render farm** — Blender Foundation's 2024 render farm, battle-hardened. OpenCut emits FFmpeg jobs. | [Flamenco](https://flamenco.blender.org/) | GPL-3 | M | Multi-machine render without rolling our own. |
| E6 | **OBS WebSocket v5 bridge** — live OBS scene → OpenCut ingestion, auto-clip via Twitch chat triggers. | [obs-websocket](https://github.com/obsproject/obs-websocket) | GPL-2 | M | Gaming vertical unlock without a separate product. |
| E7 | **Looper frame-accurate review sidecar** — Docker sidecar; frame-accurate WS review UI linked from panel. | [Looper](https://github.com/looper-dev/looper) | AGPL-3 | L | Frame.io parity without their pricing. |
| E8 | **SignDiff overlay rendering** — diffusion-based sign-language video from text; PIP overlay in composition JSON. | [SignDiff](https://github.com/SignDiff/Pretrained-Models) | Apache-2 | L | Niche but **zero** current editors ship it. |
| E9 | **RunPod serverless render** — template-based GPU job submission. Not OSS but template pairs with existing infra. | [runpod-python](https://github.com/runpod/runpod-python) | MIT client | S–M | Burst GPU capacity when local is saturated. |

---

## Infrastructure Carry-Over (from existing ROADMAP.md)

These items from the original Wave 3 / "Research & Strategic Gaps"
section remain open and are **not** replaced by this document — they
should ship in parallel with Wave A/B work:

1. **GPU process isolation (Wave 3A, P0)** — still unimplemented as of
   v1.17.0. Every Wave B entry above assumes this lands first.
   Mitigation path: `@gpu_exclusive` decorator + `MAX_CONCURRENT_GPU_JOBS = 3`
   semaphore.
2. **Rate-limit expansion** — 4 % async-route coverage. Add category
   decorators (`gpu_heavy` / `cpu_heavy` / `io_bound` / `light`).
3. **Subprocess cancellation** — 158 `subprocess.run()` calls that can't
   be interrupted mid-execution.
4. **Type-check CI** — `mypy --ignore-missing-imports opencut/` with
   0-error target in `opencut/core/` within 2 sprints.
5. **Security scanning CI** — `bandit -r opencut/ -ll`, Dependabot.
6. **UXP full parity** — CEP EOL ~Sept 2026.

---

## Shipping Cadence

| Version | Target | Wave | Expected Date |
|---------|--------|------|---------------|
| v1.17.0 | Neural interp + declarative compose | — | **Shipped 2026-04-17** |
| v1.18.0 | F5-TTS, WhisperX diarise, BeatNet, scenes-auto, CLIP-IQA+, HSEmotion, ab-av1, AAF/OTIOZ, compliance profiles, event moments | A + D | **Shipped 2026-04-17** |
| v1.19.0 | BiRefNet matte, karaoke captions, SVT-AV1-PSY, DDColor, VRT/RVRT, neural deflicker, webhook auto-emit | A + D | **Shipped 2026-04-17** |
| v1.19.1 | Hardening audit on v1.17-v1.19 — matte_birefnet, event_moments, tts_f5, neural_interp, AAF/OTIOZ async | — | **Shipped 2026-04-17** |
| v1.20.0 | OTIO diff, VMAF/SSIM/PSNR harness, Sentry, LLM audio description, LLM auto-quiz | D | **Shipped 2026-04-17** |
| v1.21.0 | VVC/H.266, SRT streaming, colour-science scopes, voice-command grammar, Atheris fuzz harness | B + D + E | **Shipped 2026-04-17** |
| v1.22.0 | Shaka Packager HLS/DASH/CENC, OBS WebSocket v5 bridge, RunPod serverless, Plausible telemetry | B + D + E | **Shipped 2026-04-17** |
| v1.23.0 | Wide-net infrastructure: OpenAPI 3.1 + Swagger UI, GPU semaphore (Wave 3A MVP), rate-limit categories, temp-file startup sweep | infra | **Shipped 2026-04-17** |
| v1.24.0 | Wide-net infra round 2: run_ffmpeg(job_id), disk monitor, request-ID middleware, deprecation registry, SECURITY.md + CONTRIBUTING.md + SBOM generator | infra | **Shipped 2026-04-17** |
| v1.19.1 | Wave D3–D6 (collab, dev SDK, obs, verticals) | D | 2026-05 |
| v1.20.0 | Wave B1 (lip-sync) + B2 (OCIO/ACES) + GPU isolation MVP | B | 2026-06 |
| v1.21.0 | Wave B3 (LTX-Video) + B4 (diarisation) + Wave E4 (voice-command grammar) | B + E | 2026-07 |
| v1.22.0 | Wave B5 (delivery: VVC, Shaka, WebRTC, SRT) + Wave E2 (VapourSynth) + E6 (OBS) | B + E | 2026-08 |
| v1.23.0+ | Wave C cherry-picks + Wave E1 (OFX) / E3 (LTX agent) / E5 (Flamenco) based on usage signal | C + E | 2026-Q4 |
| v1.28.0 | Wave K Tier 1: AudioSeal, Brand Kit, Podcast Suite, batch reframe, star rating, subtitle QA, profanity censor, spectral match, Lottie import, semantic search | K | 2026-Q4 |
| v1.28.x | Wave K Tier 2 stubs: GPT-SoVITS, Amphion/Vevo2, CosyVoice2, EchoMimic V3, TokenFlow, Cutie, DEVA, SEA-RAFT, DiffBIR, Gyroflow, NAFNet, Depth Pro, DepthFlow, AudioGen SFX, Open-Sora v2, LTX-2 A+V, audio-reactive FX, CineFocus | K | rolling 2026-Q4 |
| v1.29.0 | Wave K Tier 3: dub pipeline, trailer gen, IntelliScript, face age, slate ID, outpainting, VACE editing, sports highlights | K | 2027-Q1 |

---

## Wave H — Commercial Parity & Content-Creator Polish (v1.25.0, 2026-04-19)

Cross-project research pass against commercial editors (Opus Clip,
Descript, CapCut, ScreenStudio, Runway Gen-3, DaVinci 19+ Magic Mask,
Adobe Firefly Video) and GitHub projects that landed *after* the
April-2026 survey (FlashVSR, ROSE, Sammie-Roto-2, OmniVoice, ReEzSynth,
VidMuse, VideoAgent, ViMax, Hailuo 2.3, Seedance 2.0, GaussianHeadTalk,
FantasyTalking2). All three tiers in this wave ship together as v1.25.0
"shipped scaffolding" — Tier 1 lands as fully-working backend + panel
additions; Tier 2/3 AI-model features land as `check_X_available()`-
gated stubs returning 503 `MISSING_DEPENDENCY` with install hints,
matching the v1.18–1.20 pattern. Frontend wiring trails by one release.

### Tier 1 — High ROI, Small Effort (content-creator polish)

| # | Feature | Module (new) | Routes | OSS / Product Source |
|---|---------|--------------|--------|----------------------|
| H1.1 | **Virality / hook score 0–100** — multimodal: transcript sentiment × audio energy peaks × visual salience. Ranks short-form clip candidates before shorts_pipeline picks one. | `core/virality_score.py` | `POST /analyze/virality`, `POST /analyze/virality/rank` | Opus Clip pattern |
| H1.2 | **Cursor-event auto-zoom** — detect mouse-click timestamps from a screen-recording metadata side-car (or OpenCV diff-based cursor detection) and emit timeline-aligned zoom keyframes. Extends `auto_zoom.py`. | `core/cursor_zoom.py` | `POST /video/cursor-zoom` | ScreenStudio / Screen.Studio |
| H1.3 | **Eye-gaze correction** — MediaPipe face-mesh keypoint rotation to fake camera gaze for teleprompter reads. | `core/eye_contact.py` | `POST /video/eye-contact` | Descript Eye Contact |
| H1.4 | **In-panel changelog toast** — panel fetches GitHub releases on startup, shows unseen release notes. | `client/main.js` (+ `POST /system/changelog/mark-seen`) | `GET /system/changelog/latest`, `POST /system/changelog/mark-seen` | Every polished CEP extension |
| H1.5 | **"Send log" / crash-to-issue** — panel button posts filtered `/logs/tail` + `crash.log` excerpt into a pre-filled GitHub issue URL. | `client/main.js` (+ `GET /system/issue-report/bundle`) | `GET /system/issue-report/bundle` | Bolt CEP convention |
| H1.6 | **Demo-footage bundle** — 10–30 s public-domain sample ships under `opencut/data/demo/sample.mp4`; "Try on demo" button in every tab pre-fills `filepath`. | `opencut/data/demo/` + `GET /system/demo/sample` | `GET /system/demo/sample`, `GET /system/demo/list` | Kapwing / AEJuice |
| H1.7 | **Preset sharing via GitHub Gist** — export/import workflow presets, LUT configs, and favorites through Gist URLs. Pure stdlib `urllib`. | `core/gist_sync.py` | `POST /settings/gist/push`, `POST /settings/gist/pull`, `GET /settings/gist/info` | Community pattern |
| H1.8 | **First-run onboarding wizard** — 5-step panel tour (Connect → Pick clip → Cut → Caption → Export). Skippable, remembered per profile. | `client/onboarding.js` + `client/onboarding.css` | `GET /settings/onboarding`, `POST /settings/onboarding` | FCP, Premiere built-in tours |

### Tier 2 — High Impact, Medium Effort (new AI surfaces)

| # | Feature | Module (stub) | Routes | Source |
|---|---------|--------------|--------|--------|
| H2.1 | **FlashVSR** — streaming diffusion VSR (CVPR'26). Preview-grade 4K via locality-constrained sparse attention. | `core/upscale_flashvsr.py` | `POST /video/upscale/flashvsr`, `GET /video/upscale/flashvsr/info` | [OpenImagingLab/FlashVSR](https://github.com/OpenImagingLab/FlashVSR) |
| H2.2 | **ROSE** — video inpainting that preserves shadows/reflections (the "remove object but keep shadow" problem). | `core/inpaint_rose.py` | `POST /video/inpaint/rose`, `GET /video/inpaint/rose/info` | [rose2025-inpaint.github.io](https://rose2025-inpaint.github.io/) |
| H2.3 | **Sammie-Roto-2** — AI rotoscoping with VideoMaMa segmentation + in/out markers (v2.3 Mar 2026). Temporal complement to BiRefNet. | `core/matte_sammie.py` | `POST /video/matte/sammie`, `GET /video/matte/sammie/info` | [Zarxrax/Sammie-Roto-2](https://github.com/Zarxrax/Sammie-Roto-2) |
| H2.4 | **OmniVoice** — zero-shot TTS with 600+ languages. New backend alongside F5-TTS / Chatterbox for long-tail languages. | `core/tts_omnivoice.py` | `POST /audio/tts/omnivoice`, `GET /audio/tts/omnivoice/models` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) |
| H2.5 | **ReEzSynth** — flicker-free Ebsynth successor (bidirectional NNF + temporal propagation). | `core/style_reezsynth.py` | `POST /video/style/reezsynth`, `GET /video/style/reezsynth/info` | [FuouM/ReEzSynth](https://github.com/FuouM/ReEzSynth) |
| H2.6 | **VidMuse** — video-to-music generation (CVPR'25) with long-short-term modeling. Pairs with existing MusicGen. | `core/music_vidmuse.py` | `POST /audio/music/vidmuse`, `GET /audio/music/vidmuse/info` | [vidmuse.github.io](https://vidmuse.github.io/) |
| H2.7 | **BridgeTalk async JSX bridge** — replace panel polling with CSXS events emitted from JSX for cut-review / batch-rename / sequence-introspection ops. | `host/index.jsx` (extend) + `client/main.js` event listener | — (event plumbing only) | Adobe CEP docs |
| H2.8 | **QE API reflection probe** — call `qe.reflect.methods` at startup, surface the result through `GET /system/qe-reflect`. Unlocks undocumented Premiere 2025+ APIs. | `host/index.jsx::ocQeReflect` + `routes/system.py` | `GET /system/qe-reflect` | vakago-tools.com |

### Tier 3 — Strategic Bets (stub + research note)

| # | Feature | Module (stub) | Routes | Source |
|---|---------|--------------|--------|--------|
| H3.1 | **VideoAgent / ViMax** — agentic LLM-routed search across indexed footage + auto-storyboard from a script. | `core/video_agent.py` | `POST /agent/search-footage`, `POST /agent/storyboard` | [HKUDS/VideoAgent](https://github.com/HKUDS/VideoAgent), [HKUDS/ViMax](https://github.com/HKUDS/ViMax) |
| H3.2 | **Hailuo 2.3 / Seedance 2.0** — commercial gen-video backends (closed-weights, HTTP API). Alternative to LTX-Video / Wan 2.1 for higher quality at the cost of cloud dependency. | `core/gen_video_cloud.py` | `POST /generate/cloud/submit`, `GET /generate/cloud/status/<id>`, `GET /generate/cloud/backends` | hailuo-02.com, seed.bytedance.com |
| H3.3 | **GaussianHeadTalk / FantasyTalking2** — wobble-free talking-head alternatives to LatentSync/MuseTalk for higher-end dubbing. | `core/lipsync_advanced.py` | `POST /lipsync/gaussian`, `POST /lipsync/fantasy2`, `GET /lipsync/advanced/backends` | WACV/AAAI 2026 |
| H3.4 | **Magnetic-timeline snap UI** — FCP-inspired gap-closing snap for the cut review panel (drag cuts across sequence boundaries without gaps). | `client/main.js` (cut review panel) | — (frontend only) | FCP |
| H3.5 | **WebView UI UXP migration path** — adopt Bolt UXP's WebView pattern to share the CEP codebase post-CEP-EOL (Sept 2026). Research spike only; no code lands in v1.25.0. | — | — | [Bolt UXP WebView](https://blog.developer.adobe.com/en/publish/2026/03/introducing-webview-ui-in-bolt-uxp-build-richer-adobe-plugins-faster/) |

### Tier 1 ships **fully working** in v1.25.0. Tier 2 ships as stubs + `check_X_available()` guards returning 503 `MISSING_DEPENDENCY` with install hints. Tier 3 lands as route scaffolding with a single "not yet implemented" response body + a TODO comment naming the upstream reference; promoted to Tier 2 once a user files a feature request or the upstream licence clarifies.

**Wave H total**: 21 new routes (Tier 1 + Tier 2), ~8 new stub routes (Tier 3), ~14 new `check_*_available()` entries, zero new *required* pip deps, 1 new blueprint (`wave_h_bp`).

### Wave H gotchas
- **Gist sharing writes to public gists by default** — `/settings/gist/push` requires an explicit `private=True` flag to target a secret gist, and requires `GITHUB_TOKEN` env for authenticated push. Unauthenticated push uses anonymous gists (IP-rate-limited by GitHub).
- **Virality score is heuristic** — no ML model; a simple weighted blend of audio-energy peaks (from existing `silence.py`), transcript sentiment (via `core/llm.py` if available, falls back to keyword lexicon), and visual salience (optical-flow magnitude). Results are ranked 0–100 but the absolute number is not comparable across video types.
- **Cursor-zoom metadata parsing** — accepts either a ScreenStudio / Screen.Studio sidecar JSON (`{clicks: [{t, x, y}]}`), an OBS-WebSocket recording log, or a frame-diff fallback (slower, OpenCV-based). Never trust client-supplied coordinates; clamp to `[0, width] × [0, height]`.
- **Eye-contact shader** — the MediaPipe face-mesh keypoint rotation fakes gaze at the cost of introducing a small warp around the eye region. The module returns a `warp_factor` between 0 and 1 so frontend previews can show the user a "before/after" slider rather than commit irreversibly.
- **Demo footage is bundled only in installer builds** — PyInstaller spec adds `opencut/data/demo/sample.mp4`; pip-installed dev installs rely on a post-install `opencut-server --download-demo` flag that pulls from a GitHub release asset.
- **Onboarding wizard persists per-profile** — stored as `onboarding_seen: true` in `~/.opencut/onboarding.json`; deleting the file re-triggers the tour. Don't use localStorage — it doesn't survive panel reinstalls.
- **Issue-report bundle scrubs filepaths** — `/system/issue-report/bundle` redacts any path under `$HOME` to `~/.../<basename>`. Never let a user email raw crash.log to a bug tracker that could include private directory structures.
- **BridgeTalk event names are namespaced** — all events use the `com.opencut.<event>` prefix. Panel listens via `CSInterface.addEventListener` in `main.js`. JSX emits via `new CSXSEvent(...)` (ES3-safe; no template literals).
- **Tier 3 routes always return 501** — `ROUTE_STUBBED` error code. Frontend treats 501 as "coming soon" (greyed-out with tooltip), never as a failed call.

---

## Wave I — Panel Polish & Agentic Assembly (v1.26.0, target 2026-05)

Cross-project research pass against
[ayushozha/AdobePremiereProMCP](https://github.com/ayushozha/AdobePremiereProMCP)
— an MCP server for Premiere Pro (Go + Rust + TypeScript + Python
polyglot, WebSocket-in-panel transport, ~907 generated tools). Their
architecture isn't worth adopting wholesale (polyglot overhead for
negligible gain), but four small polish items and one strategic
capability are worth porting. All items are additive; no breaking
changes.

### Tier 1 — small polish (high ROI, 1-2 days each)

| # | Feature | Module (new) | Routes | Source |
|---|---------|--------------|--------|--------|
| I1.1 | **Live panel stats widget** — uptime, command count, avg response time (p50/p95), error count, active SSE / WS clients, last-error text. Renders as a new card on the Settings tab. | `core/panel_stats.py` + `client/main.js` stats card | `GET /system/stats` | AdobePremiereProMCP CEP panel |
| I1.2 | **Lazy-loaded JSX chunks** — split `host/index.jsx` into `host/core.jsx` (media scan + ping + marker ops, eager) and `host/domain.jsx` (color / audio / transitions / captions, loaded on first call via `$.evalFile`). Target: trim cold-panel-open time. | `host/core.jsx`, `host/domain.jsx` (new) + `client/main.js` lazy-load helper | — (JSX loader change) | AdobePremiereProMCP lazy-load pattern |
| I1.3 | **WebSocket heartbeat pings (15 s)** — active ping/pong from panel to `/ws` so dead sockets are detected before the next user action. Extends the existing `wsDisconnect()` reconnect loop. | `core/websocket_server.py` + `client/main.js` heartbeat timer | — (ws plumbing only) | AdobePremiereProMCP panel.js |
| I1.4 | **Cross-platform launchers** — add `OpenCut-Server.command` (macOS) and `OpenCut-Server.sh` (Linux) to match the existing `OpenCut-Server.bat` / `OpenCut-Launcher.vbs`. Keeps tarball installs turnkey on all three OSes. | `OpenCut-Server.command`, `OpenCut-Server.sh` (new, repo root) | — (scripts only) | AdobePremiereProMCP launchers |

### Tier 2 — strategic capability (M effort)

| # | Feature | Module (new) | Routes | Source |
|---|---------|--------------|--------|--------|
| I2.1 | **Script → EDL → native Premiere sequence** in one call. Chains: whisper transcribe (or raw script text) → LLM scene split → `footage_search.py` to match shots against an indexed media library → `multicam_xml.py` to emit FCP XML → host JSX import. Returns the new sequence's nodeId. Single POST replaces what currently takes 4-5 sequential jobs. | `core/script_to_sequence.py` | `POST /timeline/assemble-from-script`, `POST /timeline/assemble-from-script/preview` | AdobePremiereProMCP `ExecuteEDL` RPC |

### Not adopted (deliberate)

- **Polyglot stack (Go + Rust + TS + Python)** — huge dependency burden for no user-visible benefit. OpenCut's single-process Flask ships as one exe via PyInstaller; we keep that.
- **WebSocket-in-panel server** — they invert the normal CEP pattern (panel = server, external MCP client connects in on port 9801). Fine for their "MCP client drives Premiere" use case but breaks OpenCut's install-and-forget UX.
- **Auto-generated tool stubs** — their own README disagrees with itself (907 vs 1,060 tools) suggesting heavy use of boilerplate generators. OpenCut's 1,275 routes are hand-written and tested.
- **gRPC between internal services** — Flask + SSE + NDJSON streaming is enough. No cross-language boundary to bridge.

### Wave I gotchas (anticipated)

- **Stats widget cardinality** — don't track per-route p99 for every one of 1,275 routes (that's a memory leak waiting to happen). Aggregate at category level (`audio/*`, `video/*`, `captions/*`, `system/*`, `settings/*`) and keep a rolling window of the last 5 000 completed jobs.
- **Lazy JSX chunk loader must be idempotent** — `$.evalFile(path)` called twice loads the script twice, which on ES3 redefines every top-level `function`. Track a `window._ocLoadedJSXChunks` Set on the panel side and short-circuit.
- **Heartbeat pings must be cheap** — the server-side handler must be O(1) per ping. Don't touch the job store, don't run DB queries, don't acquire `job_lock`.
- **macOS `.command` file perms** — must be committed executable (`chmod +x`) AND have a Gatekeeper-friendly `#!/bin/sh` shebang. Without +x macOS refuses to double-click-execute.
- **`assemble-from-script` LLM cost** — the scene-split step is a single LLM call per ~4000-word chunk; cap at 8 chunks per request (≈32k-word script) or the backend will hit any rate limit on free Anthropic / OpenAI tiers.
- **Media-library index must exist first** — `/timeline/assemble-from-script` requires a pre-built `core/footage_index_db.py` index; return 400 `MISSING_INDEX` with a hint to call `POST /search/index` first if the index is empty.

**Wave I total**: 4 new routes (Tier 1: `/system/stats`; Tier 2: `/timeline/assemble-from-script` + preview), 2 new core modules, 2 new launcher scripts, 1 JSX file split, zero new *required* pip deps.

---

## Wave J — Depth & Differentiation (v1.27.0, target 2026-Q3)

Three-angle research pass (April 2026): OSS Premiere / NLE extensions on
GitHub + Adobe Exchange, niche new AI releases Q1-Q2 2026 outside the
mainstream, and creator-adjacent tool UX patterns (podcast, streaming,
screen recording, DIT, MAM, client review). Twenty items survive the
licence + "actually novel" filter; grouped into three tiers matching
the Wave H pattern (Tier 1 fully working, Tier 2 stubs, Tier 3 research
scaffolding).

### Tier 1 — Small-effort depth (fully working, ≤1 week each)

| # | Feature | Module (new) | Routes | OSS Source | Licence |
|---|---------|--------------|--------|------------|---------|
| J1.1 | **Burned-in subtitle extraction** — PaddleOCR walk over frames, locate + OCR + mask the subtitle region, emit SRT. Opens an entire archival / repurposing workflow lane (foreign-language redubs, broadcast-to-web reformats). | `core/caption_ocr.py` | `POST /captions/extract-burned-in`, `GET /captions/extract-burned-in/info` | [timminator/VideOCR](https://github.com/timminator/VideOCR), [SWHL/RapidVideOCR](https://github.com/SWHL/RapidVideOCR) | MIT / Apache-2 |
| J1.2 | **EDL → CDL colour metadata passthrough** — parse an EDL, extract Color Decision List values, emit as a `.cdl` file + an OTIO sidecar for DaVinci round-trip. Tiny code, high value for colourists. | `core/cdl_bridge.py` | `POST /timeline/export/cdl`, `POST /timeline/import/cdl` | [walter-arrighetti/edl2cdl](https://github.com/walter-arrighetti/edl2cdl) | MIT |
| J1.3 | **Semantic keyframe extraction** — CLIP embeddings + clustering pick N representative frames per clip for thumbnails / previews / summaries. Pairs with the existing virality score so the top-ranked clips get the smartest thumbnails. | `core/keyframes_semantic.py` | `POST /video/keyframes/semantic`, `POST /video/keyframes/ranked` | [keplerlab/katna](https://github.com/keplerlab/katna) | Apache-2 |
| J1.4 | **PSE hue-flash detector extension** — extend the existing ITU-R BT.1702 flash detector to catch rapid hue changes (red→blue) that don't register on luminance delta. Accessibility win for seizure-prone viewers. | `core/pse_flash.py` (extend) | — (enhances existing `/video/pse/check`) | ITU-R BT.1702 + custom | — |
| J1.5 | **Video fingerprinting / duplicate detection** — perceptual hash over clip segments; finds duplicate shots across an ingested library. Pure-stdlib pHash-style implementation (NOT the GPLv3 `pHash` lib — we roll our own to stay MIT). | `core/video_fingerprint.py` | `POST /video/fingerprint`, `POST /search/duplicates` | [Light1Knight/video-fingerprinting-system](https://github.com/Light1Knight/video-fingerprinting-system) pattern | MIT (reimplementation) |

### Tier 2 — New AI surfaces (503 MISSING_DEPENDENCY stubs)

Same pattern as Wave H Tier 2: ship `check_X_available()` guards, 503 with install hints, full wiring lands in later releases once each upstream pins a stable Python entry point.

| # | Feature | Module (stub) | Routes | OSS Source |
|---|---------|---------------|--------|------------|
| J2.1 | **DCVC-RT real-time neural codec** (Microsoft, CVPR'25) — 21% bitrate saving vs H.266 at 100+ fps 1080p, real-time 4K on modern GPUs. Replaces H.264 for proxy / preview generation. | `core/codec_dcvc.py` | `POST /video/encode/dcvc`, `POST /video/decode/dcvc`, `GET /video/encode/dcvc/info` | [microsoft/DCVC](https://github.com/microsoft/DCVC) |
| J2.2 | **GIMM-VFI** (NeurIPS'24) — generalisable implicit motion modelling for arbitrary-timestep frame interpolation. Beats RIFE on fast-action / ghosting. Slots alongside existing `/video/interpolate/neural`. | `core/interp_gimm.py` | `POST /video/interpolate/gimm`, `GET /video/interpolate/gimm/info` | [GSeanCDAT/GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) |
| J2.3 | **PerVFI / EMA-VFI** — asymmetric synergistic blending + hybrid CNN+Transformer frame interpolation. Tier of backends for `/video/interpolate/*` alongside RIFE and GIMM. | `core/interp_pervfi.py` | `POST /video/interpolate/pervfi` | [mulns/PerVFI](https://github.com/mulns/PerVFI), [MCG-NJU/EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) |
| J2.4 | **ITMLUT inverse tonemapping** (CVMP'23) — SDR→HDR via 3D-LUT; very fast inference. Archival upgrade path for old web video. | `core/hdr_itmlut.py` | `POST /video/tone-map/inverse`, `GET /video/tone-map/backends` | [AndreGuo/ITMLUT](https://github.com/AndreGuo/ITMLUT) |
| J2.5 | **FoleyCrafter** — realistic Foley + SFX generation from silent video. Complements existing MusicGen (which handles music, not ambience). | `core/foley_crafter.py` | `POST /audio/foley/generate`, `GET /audio/foley/info` | FoleyCrafter (permissive fork) |
| J2.6 | **SafeVision / NudeNet** content moderation — NSFW + violence frame-level detection, auto-blur pipeline. Fills the enterprise / platform-compliance gap OpenCut doesn't currently address. | `core/content_moderation.py` | `POST /video/content/scan`, `POST /video/content/blur`, `GET /video/content/info` | [im-syn/SafeVision](https://github.com/im-syn/SafeVision), NudeNet pattern |
| J2.7 | **Advanced frame interpolation aggregator** — unified `/video/interpolate` route that dispatches across RIFE (shipped) + GIMM (J2.2) + PerVFI (J2.3) + FFmpeg minterpolate fallback based on availability + user preference. | `core/neural_interp.py` (extend) | `GET /video/interpolate/backends` (extend existing) | — (dispatcher over J2.2/J2.3) |

### Tier 3 — Strategic UX patterns (scaffolding + research notes)

These are patterns from the research pass that deserve a documented landing spot but don't warrant a code stub — they're UX investments or architecture decisions that pay off across multiple releases.

| # | Feature | OSS / Product Source | Notes |
|---|---------|----------------------|-------|
| J3.1 | **Scene-aware auto-ducking** — LLM-decided dialogue-vs-music submix routing. Existing `/audio/duck` does amplitude-based ducking only; extend to a scene-tag-aware router that knows "this is dialogue over music bed, dip the bed 12 dB". | Hindenburg, Auphonic (closed products) | New route `/audio/auto-duck-scene`; relies on existing LLM + transcript infra. **L effort; no stub in Wave J.** |
| J3.2 | **Multi-pass caption review gate** — per-segment approve / flag / lock before export, with broadcast-compliance auto-check layered on top of `caption_compliance.py`. Differentiates from Premiere's native caption flow. | Rev, Glocap, Aegisub (pattern) | New route `/captions/review-gate`; panel card extension. **M effort; no stub in Wave J.** |
| J3.3 | **Node-based colour graph UI** — SVG canvas on the panel, nodes = colour ops (lift/gamma/gain, LUT apply, curves), edges = pipeline. Lets users wire grades visually instead of through a linear filter chain. Attracts colourists. | DaVinci Resolve colour page | New panel card + `POST /video/color-node-graph/apply`; graph schema in core. **L effort; no stub in Wave J.** |
| J3.4 | **Client-review feedback loop** — export a watermarked password-gated preview, collect frame-locked comments via a lightweight web view, reimport as timeline markers. Closes the post-production → client → revision loop without Frame.io subscription. | Frame.io, Wipster, Vimeo Review (pattern only) | New blueprint + static HTML review site; persist comments in `~/.opencut/reviews/<session_id>.json`. **L effort; design spike only.** |
| J3.5 | **De-subtitling (burned-in subtitle removal)** — detect burned-in subtitle regions via OCR confidence (reuse J1.1) then inpaint via existing ProPainter / ROSE. Inverse of J1.1 — produces a clean base video for re-localisation. | Glocap (pattern) | New route `/video/de-subtitle`; chains J1.1 + existing inpainting infra. **M effort; schedule after J1.1 lands.** |
| J3.6 | **Multi-language audio package delivery** — extend `/delivery/export` to mux N audio streams (dialogue per language) + 1 master subtitle stream into a single MKV or H.264/H.265 container. Single output file instead of N separate files for N languages. | Broadcast delivery conventions | Extension to existing delivery routes; **S effort; schedule when J1.1 archival lane lands.** |

### Not adopted (with rationale)

Documented explicitly so future research passes don't re-surface these:

- **pHash (perceptual hash library)** — GPLv3 licence contaminates OpenCut's MIT promise. We implement our own MIT-licensed pHash-style fingerprint in J1.5 instead.
- **C-MET (CVPR 2026 emotion edit)** — research-only licence. Revisit if authors relicense.
- **EmoMUNIT (voice emotion transfer)** — niche, low user demand, lab-quality voice artefacts. Skip unless a user files a feature request.
- **MyFrame / FreeFrame** (self-hosted Frame.io clones) — shipping these well is a business, not a feature. J3.4 captures the narrow client-review slice that matters to OpenCut users.
- **Timeline-as-code / Cursorful-style markdown editing** — overlaps with existing workflow presets; niche audience; Git-friendly diff benefit doesn't outweigh implementation cost.
- **StreamDeck webhook integration** — fine as a user-supplied plugin calling existing routes; not core.
- **Recordly / Kap / ShareX / general screen recorders** — out of scope. OpenCut edits screen recordings; it doesn't capture them.
- **Hypothesis.is / Milanote-style note-taking panels** — overlaps with Operation Journal already shipped.
- **ai-typography / atokern** — font-level work is a different product. Skip.
- **pHash (again, GPLv3)** — still no.

### Wave J gotchas (anticipated)

- **PaddleOCR GPU footprint** — J1.1 needs ~2 GB of PaddleOCR models per language pack. Download lazily per-language via a new `/captions/extract-burned-in/install?lang=<iso>` endpoint rather than front-loading every language on startup.
- **J1.3 CLIP embeddings already cached** — `footage_index_db.py` (Wave 1.9.0) already caches CLIP embeddings for footage search. Re-use those — don't recompute on every keyframe request.
- **DCVC-RT decoder must match encoder** — bitstreams are NOT interchangeable with H.264/HEVC. Any clip encoded with J2.1 requires J2.1 for decode. Flag this in the delivery preset so users don't hand off a DCVC proxy to a client who can't play it back.
- **GIMM-VFI + PerVFI share a CUDA-heavy runtime** — ship them under a single `opencut[interp-neural]` pip extra so users don't double-install torch.
- **J1.4 PSE hue detector must not flag brand colour flips** — branded motion graphics (logo reveal red→blue) triggers the naive detector. Gate the hue-delta check on a per-region basis (detect foreground vs background first) or expose a `pse_hue_sensitivity` knob.
- **J2.6 content-moderation scores are not decisions** — return `score` + `category` and let the user / platform apply the policy. Never hard-block export on a content-moderation flag — that's someone else's compliance team's call.
- **Node-based colour graph JSON schema needs versioning** — users will save graphs and expect them to keep loading in v1.28.0+. Pin the schema from day 1 with a `"version": 1` field and a migration path.
- **J3.4 client-review feedback URL must be opt-in shareable** — default it to localhost-only with a "generate public link" button that exposes a reverse-proxy endpoint. Don't make the panel silently open a public port.
- **J1.1 OCR + J3.5 de-subtitling must chain safely** — if J1.1 extraction fails (OCR confidence too low), J3.5 should not blindly inpaint whatever rectangles J1.1 guessed. Propagate a confidence threshold + abort flag.

**Wave J total**: 15 new routes (J1.1-J1.5 + J2.1-J2.7), 9 new core modules, 9 new `check_*_available()` entries, zero new *required* pip deps, 1 new blueprint (`wave_j_bp`).

### Wave J shipping cadence

| Phase | Items | Target |
|-------|-------|--------|
| v1.26.0 (Wave I) | Panel polish + script-to-sequence | 2026-05 |
| v1.26.x (Wave H Tier 2 fills) | Wire FlashVSR / ROSE / Sammie / OmniVoice / ReEzSynth / VidMuse | rolling through 2026-Q2 |
| **v1.27.0 (Wave J Tier 1)** | **J1.1 VideOCR, J1.2 edl2cdl, J1.3 katna, J1.4 PSE hue, J1.5 fingerprint** | **2026-Q3** |
| v1.27.x (Wave J Tier 2 stubs) | DCVC-RT, GIMM-VFI, PerVFI, ITMLUT, FoleyCrafter, SafeVision, interp aggregator | 2026-Q3 |
| v1.28.0 (Wave J Tier 3 rollout) | J3.1-J3.6 UX patterns rolled in progressively | 2026-Q4 |

---

---

## Wave K -- Completeness Pass & First-Mover Gaps (v1.28.0, target 2026-Q4)

Four-angle research pass (May 2026): OSS tools (Gyroflow, Kdenlive, SubtitleEdit, VapourSynth),
AI models 2024-2026 (AudioSeal, Amphion/Vevo2, GPT-SoVITS, TokenFlow, Cutie, DEVA, SEA-RAFT,
EchoMimic V3, CosyVoice2, DiffBIR, Apple Depth Pro, NAFNet, Open-Sora v2, LTX-2, DepthFlow,
Gyroflow), and commercial feature analysis (CapCut 2026, Descript Underlord, OpusClip, Runway
Gen-4.5, Adobe Premiere 2026, DaVinci Resolve 21, HeyGen, ElevenLabs, Suno v5.5). 27 items
survive the licence + novelty filter across three tiers.

### Tier 1 -- High ROI, Zero/Minimal ML (fully working)

| # | Feature | Module (new) | Routes | Source | Licence |
|---|---------|--------------|--------|--------|---------|
| K1.1 | **AudioSeal AI-content watermark** -- imperceptible audio watermark embeds provenance into all AI-generated audio. `pip install audioseal`. No other local editor ships this; legally significant for AI output. | `core/audio_watermark.py` | `POST /audio/watermark/embed`, `POST /audio/watermark/detect`, `GET /audio/watermark/info` | [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal) | MIT |
| K1.2 | **Brand Kit system** -- logo, hex palette, fonts, intro/outro clip, watermark position stored in `~/.opencut/brand_kit.json`. Auto-inject via `brand_kit=true` flag on compose routes. Zero ML; pure UX. CapCut/OpusClip ship this; no OSS editor does. | `core/brand_kit.py` | `GET /settings/brand-kit`, `POST /settings/brand-kit`, `POST /settings/brand-kit/preview`, `DELETE /settings/brand-kit` | CapCut / OpusClip pattern | -- |
| K1.3 | **Podcast Suite** -- chains existing pieces: transcript -> auto-chapters, LLM show-notes, audiogram renderer (waveform + pull-quote card). Single conductor route returns chapter VTT + show-notes markdown + audiogram path. | `core/podcast_suite.py` | `POST /audio/podcast/suite`, `POST /audio/podcast/audiogram`, `POST /audio/podcast/show-notes` | Descript / Headliner pattern | -- |
| K1.4 | **Multi-ratio batch reframe** -- one call produces 16:9 + 9:16 + 1:1 + 4:5 + 4:3 crops via existing `smart_reframe.py`. Returns zip with ratio-named filenames. CapCut/OpusClip charge per export. | `core/batch_reframe.py` | `POST /video/reframe/batch`, `GET /video/reframe/batch/presets` | CapCut / OpusClip pattern | -- |
| K1.5 | **Star rating + clip tagging** -- good/neutral/rejected + 1-5 stars + free-form tags per clip in `~/.opencut/clip_db.json`. DaVinci / FCP ship this for dailies culling; OpenCut has no rating system. | `core/clip_rating.py` | `POST /clips/rate`, `POST /clips/tag`, `GET /clips/search`, `DELETE /clips/tag` | DaVinci / FCP pattern | -- |
| K1.6 | **Subtitle QA validator** -- CPS check, min/max gap, overlap detection, max line length across entire SRT/VTT/ASS. Four built-in profiles (Netflix, BBC, YouTube, EBU-TT-D). Extends `caption_compliance.py`. | `core/subtitle_qa.py` | `POST /captions/qa/validate`, `GET /captions/qa/profiles` | SubtitleEdit rule patterns (GPL-3 data reimplemented MIT) | MIT |
| K1.7 | **Bulk profanity censor** -- Whisper word timestamps -> beep tone via FFmpeg `aevalsrc`. Modes: bleep / silence / mute_speaker. Custom word list via JSON. | `core/profanity_censor.py` | `POST /audio/censor/profanity`, `GET /audio/censor/wordlists` | Premiere / Descript pattern | -- |
| K1.8 | **EQ / Level Spectral Matcher** -- FFT (scipy) measures reference clip spectral curve, computes FIR correction filter, applies to target. "Make this interview sound like that reference mic." DaVinci Fairlight charges ~$295 for this. Pure Python, zero new GPU deps. | `core/spectral_match.py` | `POST /audio/spectral-match`, `POST /audio/spectral-match/preview` | DaVinci Fairlight pattern | -- |
| K1.9 | **Lottie animation import** -- render `.json` / `.lottie` as a video clip with alpha via `lottie-python` (MIT). Output: WEBM/MOV with alpha for compositing. DaVinci 21 ships native Lottie; no OSS editor does. | `core/lottie_import.py` | `POST /video/lottie/render`, `GET /video/lottie/info` | [lottie-python](https://github.com/LottieFiles/lottie-python) | MIT |
| K1.10 | **AI semantic media search** -- unified CLIP visual + CLAP audio + Whisper transcript index over the media library. "Find shots with a person laughing outdoors." Extends existing `footage_index_db.py`. Adobe Premiere 2026 ships this; no OSS editor does. | `core/semantic_search.py` | `POST /search/semantic`, `POST /search/index`, `GET /search/index/status` | CLIP + [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) + WhisperX | Apache-2 / MIT |

### Tier 2 -- New AI Surfaces (503 MISSING_DEPENDENCY stubs + check_X_available guards)

| # | Feature | Module (stub) | Routes | OSS Source | Licence |
|---|---------|---------------|--------|------------|---------|
| K2.1 | **GPT-SoVITS voice cloning** -- 5-second few-shot clone + TTS. 44k stars, REST API-ready. Superior cloning fidelity on short reference audio. Fourth TTS backend in the dispatcher. | `core/tts_gptsovits.py` | `POST /audio/tts/gpt-sovits`, `GET /audio/tts/gpt-sovits/voices` | [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) | MIT |
| K2.2 | **Amphion MaskGCT SOTA TTS** -- outperforms ElevenLabs on MOS. Fifth TTS backend. | `core/tts_amphion.py` | `POST /audio/tts/amphion`, `GET /audio/tts/amphion/models` | [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) | MIT |
| K2.3 | **Vevo2 singing voice conversion** -- Amphion Vevo2: convert speech/TTS into a singing performance with pitch conditioning. First singing capability in OpenCut. Shares Amphion install with K2.2. | `core/singing_vevo2.py` | `POST /audio/sing/vevo2`, `GET /audio/sing/vevo2/info` | [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) Vevo2 | MIT |
| K2.4 | **CosyVoice2 streaming TTS** -- Alibaba, 150ms latency, zero-shot voice clone. Best streaming TTS for real-time preview. Apache-2. | `core/tts_cosyvoice2.py` | `POST /audio/tts/cosyvoice2`, `GET /audio/tts/cosyvoice2/voices` | [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | Apache-2 |
| K2.5 | **EchoMimic V3 talking head** -- audio-driven portrait + half-body + gesture animation (AAAI 2025/CVPR 2025/AAAI 2026). Apache-2, production-ready. Promote to `recommended: true` in `/lipsync/backends` over existing stubs. | `core/lipsync_echomimic.py` | `POST /lipsync/echomimic`, `GET /lipsync/echomimic/info` | [antgroup/echomimic](https://github.com/antgroup/echomimic) | Apache-2 |
| K2.6 | **TokenFlow training-free video style edit** -- ICLR 2024, MIT. Apply diffusion style to real footage without training. "Restyle this clip as watercolour." No commercial editor ships a local free equivalent. | `core/style_tokenflow.py` | `POST /video/style/tokenflow`, `GET /video/style/tokenflow/info` | [omerbt/TokenFlow](https://github.com/omerbt/TokenFlow) | MIT |
| K2.7 | **Cutie persistent video object tracking** -- CVPR 2024, MIT. Track a segmented object across the full video with temporal memory. Pass SAM2 mask from frame 0; Cutie propagates. Enables "remove object from entire video" without per-frame annotation. | `core/track_cutie.py` | `POST /video/track/cutie`, `GET /video/track/cutie/info` | [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie) | MIT |
| K2.8 | **DEVA open-vocabulary video tracking** -- ICCV 2023, MIT. Text-prompted: "track all cars" or "track the person in the blue shirt." Grounded-SAM + temporal propagation. Unique vs SAM2 click prompting. | `core/track_deva.py` | `POST /video/track/deva`, `GET /video/track/deva/info` | [hkchengrex/Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) | MIT |
| K2.9 | **SEA-RAFT optical flow** -- ECCV 2024, BSD-3. 2.3x faster than RAFT, SOTA on Spring benchmark. Feeds motion blur synthesis, motion trails, improved interpolation. Drop-in for any current RAFT call. | `core/flow_searaft.py` | `POST /video/flow/searaft`, `GET /video/flow/backends` | [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) | BSD-3 |
| K2.10 | **DiffBIR blind unified restoration** -- ECCV 2024, Apache-2. Diffusion prior handles blur + noise + JPEG artifacts + low-res in one pass. Fills the non-face general content restoration gap that VRT/RVRT misses on severely degraded footage. | `core/restore_diffbir.py` | `POST /video/restore/diffbir`, `GET /video/restore/diffbir/info` | [XPixelGroup/DiffBIR](https://github.com/XPixelGroup/DiffBIR) | Apache-2 |
| K2.11 | **Gyroflow IMU stabilization** -- Apache-2 CLI. Gyroscope/IMU warp stab from GoPro/DJI/Sony metadata sidecar. Far superior to vidstab for action-cam footage. Lens profile DB, horizon lock, STmap export, Sony IBIS. Subprocess call to `gyroflow` binary. | `core/stabilize_gyroflow.py` | `POST /video/stabilize/gyroflow`, `GET /video/stabilize/gyroflow/info`, `GET /video/stabilize/gyroflow/lens-profiles` | [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow) | Apache-2 |
| K2.12 | **AI motion deblur** -- NAFNet (ECCV 2022, Apache-2) for motion blur; MIMO-UNet as lightweight fallback. Zero deblur capability in OpenCut today. DaVinci Resolve 21 ships this as a premium AI feature. | `core/deblur_motion.py` | `POST /video/restore/deblur-motion`, `GET /video/restore/deblur-motion/backends` | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), [chosj95/MIMO-UNet](https://github.com/chosj95/MIMO-UNet) | Apache-2 |
| K2.13 | **Apple Depth Pro metric depth** -- MIT, zero-shot metric depth with absolute scale. Faster + more accurate than Depth Anything V2 on single-frame depth. New backend for existing depth routes; enables accurate parallax and cinefocus without calibration. | `core/depth_depthpro.py` | `POST /video/depth/depthpro`, `GET /video/depth/backends` | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) | MIT |
| K2.14 | **DepthFlow parallax-from-stills** -- convert a single still into a parallax-motion video using depth-based 2.5D warp. Creates motion from one still (Ken Burns on steroids). CLI subprocess. | `core/depth_flow.py` | `POST /video/depth-flow/generate`, `GET /video/depth-flow/info` | [BrokenSource/DepthFlow](https://github.com/BrokenSource/DepthFlow) | MIT-adjacent |
| K2.15 | **Text-to-SFX (AudioCraft AudioGen)** -- generate SFX from text prompt ("footsteps on gravel", "thunderstorm"). Code MIT; weights CC-BY-NC (download instructions in 503 hint, not bundled). No other local editor ships this. | `core/sfx_audiogen.py` | `POST /audio/sfx/generate`, `GET /audio/sfx/info` | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) AudioGen | MIT code / CC-BY-NC weights |
| K2.16 | **Open-Sora v2 T2V backend** -- Apache-2, different DiT architecture from CogVideoX/LTX-Video. Third OSS T2V option in the B-roll dispatcher. | `core/gen_video_opensora.py` | `POST /generate/opensora`, `GET /generate/opensora/info` | [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) | Apache-2 |
| K2.17 | **LTX-Video 0.9.8 + LTX-2 audio+video joint** -- LTX-2 is the first model to generate audio and video simultaneously. Upgrade existing LTX-Video backend; add audio+video joint generation route. No other local tool ships synchronized A+V generation. | `core/gen_video_ltx.py` (extend) | `POST /generate/ltx/v2`, `GET /generate/ltx/backends` | [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) | Apache-2 |
| K2.18 | **Audio-driven visual FX system** -- BeatNet beat timestamps + frequency band analysis drive zoom pulse, chromatic aberration, colour saturation, shake, strobe keyframes. Reactive presets ("boom", "bass-drop", "snare"). No OSS editor exposes this as a system. | `core/audio_reactive_fx.py` | `POST /video/audio-reactive/render`, `GET /video/audio-reactive/presets` | DaVinci Fairlight pattern (existing BeatNet + FFmpeg filter_complex) | -- |
| K2.19 | **AI CineFocus rack focus** -- depth map (Depth Pro or Depth Anything V2) drives depth-of-field bokeh: keyframeable focal point, aperture shape, f-number. "Rack focus foreground to background over 30 frames." DaVinci 21 CineFocus charges licence; OpenCut ships free. | `core/cinefocus.py` | `POST /video/cinefocus/render`, `POST /video/cinefocus/preview`, `GET /video/cinefocus/info` | DaVinci 21 pattern (Depth Pro + FFmpeg boxblur + depth mask) | -- |

### Tier 3 -- Strategic Pipelines (route scaffolding + research notes)

| # | Feature | Module (stub) | Routes | Source | Notes |
|---|---------|---------------|--------|--------|-------|
| K3.1 | **Full local video dubbing pipeline** -- WhisperX STT -> NLLB-200 translate -> CosyVoice2/GPT-SoVITS voice clone -> EchoMimic V3 lip sync -> composite. HeyGen charges per-minute; OpenCut: private, free, local. | `core/dub_pipeline.py` | `POST /dub/pipeline`, `GET /dub/pipeline/status/<job_id>` | HeyGen pattern | **L effort; schedule after K2.4 + K2.5 fill.** |
| K3.2 | **Auto trailer/promo generator** -- LLM moment scoring -> top-N extract -> MusicGen ramp + title card (declarative_compose) + CTA. All pieces in OpenCut; conductor is the gap. Descript Underlord ships this. | `core/trailer_gen.py` | `POST /generate/trailer`, `POST /generate/promo` | Descript Underlord pattern | **M effort.** |
| K3.3 | **IntelliScript .fdx / Fountain import** -- extend Wave I script-to-sequence (I2.1) to accept Final Draft `.fdx` and Fountain `.fountain` files. Parse scene headings + WhisperX fuzzy-match transcript -> auto-assemble edit order. DaVinci 21 IntelliScript charges licence. | `core/screenplay_parser.py` | `POST /timeline/assemble-from-screenplay` (extends I2.1) | [Fountain spec](https://fountain.io/syntax) (MIT) | **M effort; builds on I2.1.** |
| K3.4 | **AI Face Age Transformer** -- age slider on a face in video via IP-Adapter + Cutie temporal tracking. DaVinci 21 ships this. No OSS equivalent at video level yet. | `core/face_age_transform.py` | `POST /video/face/age-transform`, `GET /video/face/age-transform/info` | DaVinci 21 pattern | **L effort; confirm weights licence before promoting to Tier 2.** |
| K3.5 | **AI Slate ID** -- Florence-2 VLM (already installed) reads clapperboard scene/take/camera from clip-head frames. Stamps metadata into OTIO + Premiere XMP. DaVinci 21 ships this. | `core/slate_id.py` | `POST /video/slate/identify`, `GET /video/slate/identify/info` | DaVinci 21 pattern (Florence-2 already in OpenCut) | **M effort; Florence-2 already installed.** |
| K3.6 | **Video outpainting** -- expand frame borders via diffusion to change aspect ratio (generate content at edges). Wan2.1 VACE or LTX-2 inpainting conditioned on existing frame content. Runway charges per-second. | `core/outpaint_video.py` | `POST /video/outpaint`, `GET /video/outpaint/info` | Runway Gen-4 pattern | **L effort; depends on K2.17 or K3.7.** |
| K3.7 | **Wan2.1 VACE video editing** -- existing C4 stub covers T2V; VACE adds editing of existing footage via video conditioning (background change, re-light, modify action). Different inference path from T2V. | `core/gen_video_wan_vace.py` | `POST /generate/wan/vace`, `GET /generate/wan/vace/info` | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) VACE | **L effort; extends C4 stub.** |
| K3.8 | **Sports/genre-agnostic highlights** -- optical flow velocity + YAMNet crowd energy + laughter detection + face-count peak. Works for sports, concerts, events -- not just talking-head clips. OpusClip ClipAnything charges per clip. | `core/highlights_sports.py` | `POST /analyze/highlights/sports`, `GET /analyze/highlights/genres` | OpusClip ClipAnything pattern | **M effort.** |

### Not adopted (Wave K)

- **VoiceCraft** (CC-BY-NC-SA) -- in-place speech word editing. Revisit if relicensed.
- **SeamlessExpressive** (CC-BY-NC) -- CosyVoice2 (K2.4) covers the use case under Apache-2.
- **Co-Tracker3** (Meta CC-BY-NC) -- DEVA (K2.8) covers open-vocabulary tracking under MIT.
- **SUPIR** (non-commercial) -- DiffBIR (K2.10) covers blind restoration under Apache-2.
- **HunyuanVideo** (Tencent non-commercial) -- camera motion video gen blocked by licence.
- **Hallo2** (S-Lab mixed licence) -- EchoMimic V3 (K2.5) is Apache-2 and production-ready. Skip.
- **LivePortrait** (MIT code / non-commercial weights) -- no practical value without distributable weights.
- **MuseTalk weights** (non-commercial) -- EchoMimic V3 supersedes.
- **BSRGAN** -- DiffBIR (K2.10) covers the same degradation space more comprehensively.
- **UniMatch** -- SEA-RAFT + Depth Pro cover flow and depth better individually.
- **Bark** -- AudioGen (K2.15) provides better text control for SFX. Skip.
- **ChatTTS** (AGPL-3) -- licence contaminates MIT promise. Monitor for MIT alternative.
- **DAC neural codec** -- no user-visible feature until a future neural-audio-editing wave.
- **AudioCraft JASCO** (MIT code / CC-BY-NC weights) -- surfaced via K2.15 AudioGen route.

### Wave K gotchas (anticipated)

- **AudioSeal latency (K1.1)** -- embed runs >1x realtime on CPU; wire as post-export background job, never synchronous on the export path.
- **Brand Kit opt-out (K1.2)** -- must be explicit `brand_kit=true` per render. Never auto-apply to client footage without consent.
- **GPT-SoVITS server (K2.1)** -- ships its own inference server (port 9880). OpenCut wraps it as a subprocess sidecar. Check server health before routing; surface install instructions when absent.
- **Amphion + Vevo2 shared checkpoint (K2.2/K2.3)** -- one `check_amphion_available()` guard covers both. Don't require two separate downloads.
- **EchoMimic V3 backend priority (K2.5)** -- when available, `/lipsync/backends` sets echomimic to `recommended: true`. Don't silently redirect from MuseTalk/LatentSync; let the user choose.
- **SEA-RAFT resolution cap (K2.9)** -- cap input to 1080p and use downsample-process-upsample unless user explicitly requests 4K flow.
- **DiffBIR inference time (K2.10)** -- expose `tile_size` (default 512) and `fast_mode=true` (4-step DPM-Solver++ vs 50 DDIM) to manage 30-60 s per-frame cost.
- **Gyroflow binary (K2.11)** -- not on PyPI. `check_gyroflow_available()` fetches pre-built binary from gyroflow GitHub releases for the detected platform.
- **DepthFlow headless (K2.14)** -- uses ModernGL for GPU rendering; needs virtual framebuffer (Xvfb) on headless Linux. Document in 503 install hint.
- **AudioGen weights (K2.15)** -- CC-BY-NC cannot be bundled. `check_audiogen_available()` detects presence and surfaces the download URL. Never auto-download CC-BY-NC weights silently.
- **LTX-2 A+V joint (K2.17)** -- different inference call from existing LTX T2V path. New route `/generate/ltx/v2` keeps old `/generate/ltx` backward-compatible.
- **CineFocus bokeh (K2.19)** -- pre-compute depth map for entire clip in batch before rendering blur sequence. Expose `focal_z_start`, `focal_z_end`, `focal_frame_start`, `focal_frame_end` params.
- **Dub pipeline translation (K3.1)** -- NLLB-200 runs locally (MIT). Never route translation through a cloud LLM unless user explicitly selects an API provider.

**Wave K total**: 34 new routes (Tier 1 + Tier 2 + Tier 3 scaffolding), 27 new core modules,
20 new `check_*_available()` entries, 0 new *required* pip deps, 1 new blueprint (`wave_k_bp`).

**Ten features where OpenCut will be first to ship locally (no OSS NLE equivalent):**
1. **K1.1 AudioSeal** -- AI audio provenance watermarking on all generated output
2. **K1.2 Brand Kit** -- project-identity injection into every render
3. **K1.8 EQ Spectral Matcher** -- FFT-based EQ matching (DaVinci Fairlight charges ~$295)
4. **K2.3 Vevo2 Singing VC** -- singing voice synthesis from text + reference voice
5. **K2.6 TokenFlow style edit** -- training-free diffusion restyle of real footage locally
6. **K2.8 DEVA open-vocab tracking** -- text-prompted "track all cars in this video"
7. **K2.11 Gyroflow** -- IMU/gyroscope warp stabilization (GoPro/DJI/Sony grade)
8. **K2.15 Text-to-SFX** -- "footsteps on gravel" -> audio, running locally
9. **K2.17 LTX-2 A+V joint** -- synchronized audio+video generation in one pass
10. **K3.1 Local dubbing pipeline** -- private, free, local alternative to HeyGen

### Wave K shipping cadence

| Phase | Items | Target |
|-------|-------|--------|
| v1.28.0 (Wave K Tier 1) | K1.1-K1.10 content-creator polish | 2026-Q4 |
| v1.28.x (Wave K Tier 2 stubs) | K2.1-K2.19 AI backends | rolling 2026-Q4 |
| v1.29.0 (Wave K Tier 3 rollout) | K3.1-K3.8 pipeline orchestrators | 2027-Q1 |


## Sources (OSS survey, April 2026)

- **Editors surveyed**: LosslessCut, auto-editor, editly, Descript,
  Shotcut/MLT, Olive Editor, OpenShot, Kdenlive.
- **Interchange**: OpenTimelineIO + otio-aaf-adapter, OTIOZ bundles.
- **ASR/TTS**: WhisperX, faster-whisper (both integrated), Chatterbox,
  F5-TTS, IndexTTS, NeMo Sortformer, diart.
- **Source separation**: BS-RoFormer vs Demucs.
- **Matting**: BiRefNet (2024), MatAnyone (2025, research-only).
- **Scene/video understanding**: TransNetV2, InternVideo2, VideoMAE v2.
- **Lip-sync**: LatentSync 1.6, MuseTalk 1.5.
- **Caption typography**: libass + pyonfx.
- **Encoding**: vvenc (VVC), SVT-AV1-PSY, ab-av1.
- **Delivery**: Shaka Packager, aiortc, SRT.
- **Generation**: LTX-Video (2 B, real-time), CogVideoX-Fun, Wan 2.1.
- **3D**: MASt3R (research-only), VGGT, Depth Anything V2, gsplat.
- **Color**: OpenColorIO 2.4 + ACES 1.3, colour-science.
- **Analysis**: CLIP-IQA+, HSEmotion, BeatNet.
- **Accessibility**: audio-describe, SubtitleEdit compliance rules,
  SignDiff.
- **Restoration**: DDColor, VRT, All-In-One-Deflicker.
- **Collaboration**: otio-diff, Looper, OBS-websocket v5.
- **Dev SDK**: OpenFX, VapourSynth, ffmpeg-quality-metrics, Atheris.
- **Observability**: GlitchTip, Plausible.
- **Real-time**: ffmpeg.wasm, MediaPipe Tasks Web, aiortc.
- **Render farm**: Flamenco 3, RunPod-python.
- **Agents**: LTX-Video-Agent, whisper-timestamped voice grammar.
- **Verticals**: clip-detector (wedding moments), OBS bridge (gaming),
  Gemini 2.0 scene description.

Revisit this list every 6 months. AI video space moves quickly — a new
model on par with Chatterbox or LTX-Video can appear between major
releases.

---

## Sources (Wave H addendum, April 2026)

- **Commercial products**: Opus Clip, Reap, Vidyo.ai, SubMagic,
  Riverside Magic Clips, Descript (Storyboard / Eye Contact / Studio
  Sound / Rooms / Underlord), Runway ML Gen-3 (Act-One, Motion Brush,
  Multi-Motion), CapCut + CapCut Pro desktop, Adobe Firefly Video
  (Generative Extend, Enhance Speech), DaVinci Resolve 19+
  (Magic Mask, UltraNR, Voice Isolation, AI Smart Reframe), FCP / Motion,
  Kapwing, Veed.io, Clipchamp, Motion Array / AEJuice / Envato,
  ScreenStudio / Screen.Studio / Loom / Tella.
- **Post-April-2026 GitHub projects**: FlashVSR (CVPR'26 VSR), STAR
  (ICCV'25 spatial-temporal augmentation), ROSE (video inpainting with
  shadows), FloED (flow-guided inpainting), VideoVanish, Hailuo 2.3,
  Seedance 2.0, GaussianHeadTalk (WACV'26), FantasyTalking2 (AAAI'26),
  VASA-3D (NeurIPS'25), OmniVoice (k2-fsa), Voice-Pro (WebUI),
  ViDubb (video dubbing), SpectroStream (neural codec, Aug'25),
  Sammie-Roto-2 (v2.3 Mar'26), Cutie Roto, RotoForge-AI,
  ReEzSynth (bidirectional Ebsynth), VidMuse (CVPR'25 video-to-music),
  VideoAgent, ViMax (script-to-video agentic).
- **CEP/UXP ecosystem**: bolt-cep (Hyper Brew), Bolt UXP WebView UI,
  Adobe UXP Premiere Pro Samples, Adobe CEP Samples (PProPanel),
  SoundBuddy Studio, jumpcut, vakago-tools QE API documentation.
- **MCP servers for NLEs (Wave I source)**:
  [ayushozha/AdobePremiereProMCP](https://github.com/ayushozha/AdobePremiereProMCP)
  — polyglot Go + Rust + TS + Python MCP server exposing ~907
  Premiere tools over a WebSocket-in-panel transport on port 9801.
  Architecture not adopted; four polish patterns (live stats widget,
  lazy JSX chunking, WS heartbeat, cross-platform launchers) +
  one strategic route (script-to-sequence `ExecuteEDL` equivalent)
  promoted into Wave I.

## Sources (Wave J addendum — April 2026 three-angle research pass)

- **Premiere / NLE OSS extensions**: [timminator/VideOCR](https://github.com/timminator/VideOCR),
  [SWHL/RapidVideOCR](https://github.com/SWHL/RapidVideOCR),
  [roybaer/burnt-in-subtitle-extractor](https://github.com/roybaer/burnt-in-subtitle-extractor),
  [URUWorks/TeroSubtitler](https://github.com/URUWorks/TeroSubtitler),
  [walter-arrighetti/edl2cdl](https://github.com/walter-arrighetti/edl2cdl),
  [KyleTryon/MyFrame](https://github.com/KyleTryon/MyFrame),
  [Techiebutler/freeframe](https://github.com/Techiebutler/freeframe),
  [ros-dorian/TwitchHack](https://github.com/ros-dorian/TwitchHack),
  [furmonenko/easy-markers-twitch](https://github.com/furmonenko/easy-markers-twitch),
  [JaINTP/OBS-Highlight-Display](https://github.com/JaINTP/OBS-Highlight-Display),
  [wulkano/Kap](https://github.com/wulkano/Kap),
  [ShareX/ShareX](https://github.com/ShareX/ShareX),
  [phiresky/ripgrep-all](https://github.com/phiresky/ripgrep-all).
- **Q1-Q2 2026 niche AI (adopted into Tier 1/2)**:
  [microsoft/DCVC](https://github.com/microsoft/DCVC) (DCVC-RT real-time neural codec),
  [GSeanCDAT/GIMM-VFI](https://github.com/GSeanCDAT/GIMM-VFI) (NeurIPS'24 frame interpolation),
  [mulns/PerVFI](https://github.com/mulns/PerVFI) (perception-oriented VFI),
  [MCG-NJU/EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) (CNN+Transformer VFI),
  [AndreGuo/ITMLUT](https://github.com/AndreGuo/ITMLUT) (SDR→HDR 3D-LUT),
  FoleyCrafter (Stable Audio-class SFX generation),
  [keplerlab/katna](https://github.com/keplerlab/katna) (CLIP semantic keyframes),
  [im-syn/SafeVision](https://github.com/im-syn/SafeVision) (content moderation),
  [fcakyon/content-moderation-deep-learning](https://github.com/fcakyon/content-moderation-deep-learning),
  [Light1Knight/video-fingerprinting-system](https://github.com/Light1Knight/video-fingerprinting-system) (pattern reimplemented MIT).
- **Q1-Q2 2026 niche AI (evaluated, not adopted)**:
  DiffuEraser, COCOCO, ViDeNN, UDVD, EmoMUNIT, C-MET (research-only),
  GMR (motion retargeting — out of scope for an NLE extension),
  ai-typography, atokern (out of scope), pHash (GPLv3).
- **Creator-adjacent UX patterns surveyed**:
  Happy Scribe, Rev, Aegisub, SubtitleEdit, Oto, Riverside,
  Hindenburg Journalist Pro, Auphonic, Descript Overdub, Twitch Studio,
  StreamElements, BOOMR, OBS + StreamDeck integrations, ScreenStudio,
  Cap.so, Tella, Loom, Glocap, DaVinci Resolve (colour page + DIT),
  Blackmagic LUT packs, Natural Light, Telestream Vantage,
  Frame.io, Wipster, Vimeo Review, Shotgun, CantemoPortal,
  Edpuzzle, Hypothesis.is, Milanote, VLC bookmarks,
  Cursorful, Marp.

## Sources (Wave K addendum -- May 2026 four-angle research pass)

- **OSS video tools surveyed**:
  [gyroflow/gyroflow](https://github.com/gyroflow/gyroflow) (IMU stabilization, Apache-2),
  [mifi/losslesscut](https://github.com/mifi/losslesscut),
  [WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor),
  [Kdenlive](https://invent.kde.org/multimedia/kdenlive),
  [Shotcut/MLT](https://github.com/mltframework/mlt),
  [SubtitleEdit](https://github.com/SubtitleEdit/subtitle-edit),
  [VapourSynth](https://github.com/vapoursynth/vapoursynth),
  [vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

- **AI models adopted (Wave K)**:
  [facebookresearch/audioseal](https://github.com/facebookresearch/audioseal) (MIT),
  [open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) MaskGCT + Vevo2 (MIT),
  [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) (MIT),
  [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) CosyVoice2 (Apache-2),
  [antgroup/echomimic](https://github.com/antgroup/echomimic) V3 (Apache-2),
  [omerbt/TokenFlow](https://github.com/omerbt/TokenFlow) (MIT),
  [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie) (MIT),
  [hkchengrex/Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA) (MIT),
  [princeton-vl/SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) (BSD-3),
  [XPixelGroup/DiffBIR](https://github.com/XPixelGroup/DiffBIR) (Apache-2),
  [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (Apache-2),
  [chosj95/MIMO-UNet](https://github.com/chosj95/MIMO-UNet) (Apache-2),
  [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) (MIT),
  [BrokenSource/DepthFlow](https://github.com/BrokenSource/DepthFlow),
  [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) AudioGen (MIT code / CC-BY-NC weights),
  [hpcaitech/Open-Sora](https://github.com/hpcaitech/Open-Sora) v2 (Apache-2),
  [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) 0.9.8 + LTX-2 (Apache-2),
  [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) (Apache-2),
  [LottieFiles/lottie-python](https://github.com/LottieFiles/lottie-python) (MIT).

- **AI models evaluated, not adopted (Wave K)**:
  VoiceCraft (CC-BY-NC-SA), SeamlessExpressive (CC-BY-NC), Co-Tracker3 (CC-BY-NC),
  SUPIR (non-commercial), HunyuanVideo (Tencent non-commercial), Hallo2 (S-Lab mixed),
  LivePortrait weights (non-commercial), MuseTalk weights (non-commercial),
  BSRGAN, UniMatch, Bark, ChatTTS (AGPL-3), DAC, AudioCraft JASCO (CC-BY-NC weights).

- **Commercial products surveyed**:
  CapCut 2026 (batch export, brand kit, script-to-video), Descript Underlord
  (trailer gen, eye contact, studio sound, podcast suite), OpusClip (ClipAnything sports
  highlights, brand kit), Runway ML Gen-4.5 + Act-Two (video outpainting, camera conditioning),
  Adobe Premiere Pro 2026 (AI Media Intelligence semantic search, generative frame extension,
  profanity censor), DaVinci Resolve 21 (CineFocus, motion deblur, face age transformer,
  IntelliScript, slate ID, audio-reactive FX, spectral match, Lottie import, star rating),
  Topaz Video AI (motion deblur, AI upscale), HeyGen (dubbing pipeline), ElevenLabs (TTS,
  text-to-SFX), Pika Labs (video outpainting), Suno v5.5 (singing voice synthesis, stem export).
