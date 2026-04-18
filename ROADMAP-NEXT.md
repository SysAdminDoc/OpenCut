# OpenCut — Next-Wave Roadmap (2026-Q2 → 2026-Q4)

**Version**: 1.5
**Created**: 2026-04-17 (updated 2026-04-17 after v1.22.0 ship)
**Baseline**: v1.22.0 (1,228 routes, 451 core modules, 7,689+ tests)
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
| 20.1 / D3.1 | **Semantic OTIO timeline diff** | `export/otio_diff.py` | `POST /timeline/diff` | [OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) |
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
| v1.19.1 | Wave D3–D6 (collab, dev SDK, obs, verticals) | D | 2026-05 |
| v1.20.0 | Wave B1 (lip-sync) + B2 (OCIO/ACES) + GPU isolation MVP | B | 2026-06 |
| v1.21.0 | Wave B3 (LTX-Video) + B4 (diarisation) + Wave E4 (voice-command grammar) | B + E | 2026-07 |
| v1.22.0 | Wave B5 (delivery: VVC, Shaka, WebRTC, SRT) + Wave E2 (VapourSynth) + E6 (OBS) | B + E | 2026-08 |
| v1.23.0+ | Wave C cherry-picks + Wave E1 (OFX) / E3 (LTX agent) / E5 (Flamenco) based on usage signal | C + E | 2026-Q4 |

---

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
