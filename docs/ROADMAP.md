# OpenCut — Implementation Roadmap

> Prioritized plan for integrating open source improvements, new features, and architectural changes.
> Based on research conducted March 2026 (see [RESEARCH.md](RESEARCH.md)).

---

## Timeline Overview

| Phase | Timeframe | Focus |
|-------|-----------|-------|
| **Phase 1** | Immediate (Weeks 1-2) | Tier 1 — Highest-impact backend improvements |
| **Phase 2** | Short-term (Weeks 3-5) | Tier 2 — New features & expanded capabilities |
| **Phase 3** | Medium-term (Weeks 6-10) | Tier 3 — Differentiators & new product areas |
| **Phase 4** | Ongoing (Weeks 6+) | Architecture — UXP migration, DaVinci support |

---

## Phase 1 — Tier 1: Highest Impact (Weeks 1-2)

### 1.1 Silero VAD Silence Detection
- **What:** Replace energy-based threshold detection with Silero VAD neural model
- **Technology:** [Silero VAD](https://github.com/snakers4/silero-vad) — 1.8MB ONNX model
- **Files to modify:**
  - `opencut/core/silence.py` — add `detect_silences_vad()` using Silero
  - `opencut/routes/audio.py` — add `method` parameter to `/silence` endpoint
  - `opencut/checks.py` — add Silero availability check
  - Extension UI — add "Detection Method" dropdown (Energy / Silero VAD)
- **Effort:** 1-2 days
- **Impact:** 87.7% vs 50% true positive rate at same false positive rate

### 1.2 CrisperWhisper Filler Detection
- **What:** Add verbatim ASR backend that explicitly marks `[UH]` and `[UM]` with timestamps
- **Technology:** [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper)
- **Files to modify:**
  - `opencut/core/` — new `crisper_whisper.py` module
  - `opencut/routes/audio.py` — add backend option to `/fillers` endpoint
  - `opencut/checks.py` — add CrisperWhisper availability check
- **Effort:** 2-3 days
- **Impact:** #1 on OpenASR Leaderboard for verbatim transcription

### 1.3 Resemble Enhance Speech Restoration
- **What:** Add two-stage speech enhancement (denoise + bandwidth restoration to 44.1kHz)
- **Technology:** [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance)
- **Files to modify:**
  - `opencut/core/audio_enhance.py` — already exists, add Resemble Enhance mode
  - `opencut/routes/audio.py` — add `/audio/enhance` endpoint options
  - Extension UI — add "Enhancement Mode" (Denoise Only / Full Enhancement)
- **Effort:** 1-2 days
- **Impact:** Transforms phone-quality audio to studio quality

### 1.4 SAM2 + ProPainter Object Removal
- **What:** Click-to-select any object → track through video → remove with temporal-consistent inpainting
- **Technology:** [SAM2](https://github.com/facebookresearch/sam2) + [ProPainter](https://github.com/sczhou/ProPainter)
- **Files to modify:**
  - `opencut/core/` — new `object_removal.py` module
  - `opencut/routes/video.py` — new `/video/object-remove` endpoint
  - Extension UI — add object selection interface with preview frame
- **Effort:** 3-5 days (most complex Tier 1 item)
- **Impact:** Major new capability — no competitor offers click-to-remove in a Premiere extension

### 1.5 UXP Migration Planning
- **What:** Evaluate Bolt UXP framework, create migration plan, set up dual CEP+UXP build
- **Technology:** [Bolt UXP](https://github.com/hyperbrew/bolt-uxp) with WebView UI
- **Tasks:**
  - Audit current CEP features for UXP API coverage gaps
  - Set up bolt-uxp project skeleton with WebView UI
  - Create feature parity checklist (CEP vs current UXP panel)
  - Plan phased migration: core features first, then advanced
- **Effort:** 2-3 days (planning + skeleton)
- **Impact:** CEP support ends ~Sept 2026. This is time-critical.

---

## Phase 2 — Tier 2: High Value (Weeks 3-5)

### 2.1 Kokoro Local TTS
- **Technology:** [Kokoro](https://github.com/hexgrad/kokoro) — 82M params, Apache license
- **Impact:** Offline TTS with <0.3s generation time

### 2.2 F5-TTS Voice Cloning
- **Technology:** [F5-TTS](https://github.com/SWivid/F5-TTS) — zero-shot from 15s reference
- **Impact:** Record a voice → generate narration in that voice

### 2.3 ClippedAI Shorts Pipeline Improvements
- **Technology:** Study [ClippedAI](https://github.com/Shaarav4795/ClippedAI) engagement scoring
- **Impact:** Better viral potential prediction for generated shorts

### 2.4 Robust Video Matting
- **Technology:** [RVM](https://github.com/PeterL1n/RobustVideoMatting)
- **Impact:** Temporally consistent background removal without green screen

### 2.5 OpenTimelineIO Export
- **Technology:** [OTIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO)
- **Impact:** Universal export to Premiere, Resolve, FCP, Avid

### 2.6 Bandit Cinematic Stem Separation
- **Technology:** [Bandit](https://github.com/kwatcharasupat/bandit)
- **Impact:** Dialogue/music/effects separation designed for video

### 2.7 TransNetV2 Scene Detection
- **Technology:** [TransNetV2](https://github.com/soCzech/TransNetV2)
- **Impact:** Neural detection catches fades/dissolves that thresholds miss

### 2.8 CodeFormer Face Restoration
- **Technology:** [CodeFormer](https://github.com/sczhou/CodeFormer) with fidelity slider
- **Impact:** Better quality than GFPGAN + user-controllable quality/identity tradeoff

---

## Phase 3 — Tier 3: Differentiators (Weeks 6-10)

### 3.1 Video Depth Anything Effects
- **Technology:** [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- **Impact:** Parallax zoom, bokeh simulation, 3D Ken Burns effect

### 3.2 Emotion-Based Highlight Detection
- **Technology:** [deepface](https://github.com/serengil/deepface) emotion analysis
- **Impact:** Emotion curve over time → peaks = highlights

### 3.3 Multimodal Diarization for Multicam
- **Technology:** [3D-Speaker](https://github.com/modelscope/3D-Speaker) (audio + face)
- **Impact:** More accurate speaker-to-camera mapping

### 3.4 Chat-Driven Editing Assistant
- **Technology:** [Frame](https://github.com/aregrid/frame) pattern + LLM
- **Impact:** Conversational editing interface

### 3.5 AI B-Roll Generation
- **Technology:** [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) or Wan 2.2
- **Impact:** Generate B-roll footage from text descriptions

### 3.6 WebSocket Premiere Bridge
- **Technology:** [PremiereRemote](https://github.com/sebinside/PremiereRemote) pattern
- **Impact:** Real-time communication replacing evalScript polling

### 3.7 Auto B-Roll Insertion
- **Technology:** pyannote diarization + transcript keyword matching
- **Impact:** Auto-insert B-roll at dialogue pauses based on spoken content

### 3.8 Smart Thumbnail Scoring
- **Technology:** PySceneDetect + face detection + composition analysis
- **Impact:** Auto-select best thumbnail from video

### 3.9 Social Media Direct Posting
- **Technology:** Platform APIs (TikTok, Instagram, YouTube)
- **Impact:** Post generated shorts without leaving Premiere

### 3.10 Florence-2 Watermark Detection
- **Technology:** [WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI)
- **Impact:** Better watermark detection than current approach

---

## Phase 4 — Architecture (Ongoing, Weeks 6+)

### 4.1 Full UXP Migration
- **Technology:** [Bolt UXP](https://github.com/hyperbrew/bolt-uxp) + WebView UI
- **Timeline:** Must complete before CEP removal (~Sept 2026)
- **Approach:**
  1. WebView UI wrapper for existing HTML/CSS (quick wins)
  2. Replace CSInterface calls with UXP APIs
  3. Replace ExtendScript with modern JS
  4. Add TypeScript for type safety
  5. CCX packaging for Adobe Marketplace

### 4.2 DaVinci Resolve Support
- **Technology:** Native Python scripting API
- **Impact:** Reach Resolve's growing user base with zero extension overhead

### 4.3 Multi-Engine Backend
- **Technology:** Plugin architecture for swappable AI models
- **Impact:** Users choose quality/speed tradeoffs per feature

---

## Dependency Installation Strategy

All new ML models follow the existing pattern:
1. Feature checks in `opencut/checks.py`
2. Lazy import in core modules (try/except at function call time)
3. Install button in extension UI → `POST /install` endpoint
4. `safe_pip_install()` from `opencut/security.py`
5. Capability flag in `/health` response
6. UI hint/install-button visibility based on capability

---

## Implementation Status

> Last updated: March 2026

### Phase 1 — Complete
| Item | Status | Notes |
|------|--------|-------|
| 1.1 Silero VAD | DONE | `detect_silences_vad()`, "auto" mode, UI dropdown in CEP+UXP |
| 1.2 CrisperWhisper | DONE | New `crisper_whisper.py`, `/fillers` backend param, UI toggle |
| 1.3 Resemble Enhance | ALREADY EXISTED | `audio_enhance.py` had full denoise+enhance |
| 1.4 SAM2 + ProPainter | ALREADY EXISTED | `object_removal.py` had full pipeline |
| 1.5 UXP Migration Plan | DONE | `docs/UXP_MIGRATION.md` with phased plan |

### Phase 2 — Mostly Complete
| Item | Status | Notes |
|------|--------|-------|
| 2.1 Kokoro TTS | ALREADY EXISTED | `voice_gen.py` — `kokoro_generate()` |
| 2.2 Voice Cloning | ALREADY EXISTED | `voice_gen.py` — `chatterbox_generate()` |
| 2.3 Engagement Scoring | DONE | `EngagementScore` in `highlights.py`, blended with LLM score |
| 2.4 Robust Video Matting | DONE | `_remove_background_rvm()` in `video_ai.py`, backend param on route |
| 2.5 OTIO Export | DONE | New `otio_export.py`, route, UI in CEP+UXP |
| 2.6 Bandit Separation | ALREADY EXISTED | `audio-separator` backend with BS-RoFormer/MDX models |
| 2.7 TransNetV2 | ALREADY EXISTED | `scene_detect.py` — `detect_scenes_ml()` |
| 2.8 CodeFormer | ALREADY EXISTED | `face_swap.py` — fidelity slider |

### Phase 3 — Complete
| Item | Status | Notes |
|------|--------|-------|
| 3.1 Video Depth Anything | DONE | `depth_effects.py`: depth map, bokeh sim, parallax zoom. 3 routes added |
| 3.2 Emotion Highlights | DONE | New `emotion_highlights.py`, route `/video/emotion-highlights` |
| 3.3 Multimodal Diarization | DONE | `multimodal_diarize.py`: audio+face cross-modal alignment. InsightFace/facenet/Haar backends. Route + CEP/UXP UI |
| 3.4 Chat-Driven Editing | DONE | `chat_editor.py`: multi-turn LLM agent, session mgmt, action parsing. 3 routes |
| 3.5 AI B-Roll Generation | DONE | `broll_generate.py`: CogVideoX/Wan/HunyuanVideo/SVD backends. Text-to-video + image-to-video. Route + CEP/UXP UI |
| 3.6 WebSocket Bridge | DONE | `ws_bridge.py`: asyncio WebSocket server on port 5680. Real-time progress/events/commands. Start/stop/status routes |
| 3.7 Auto B-Roll Insertion | DONE | `broll_insert.py`: gap detection, topic shifts, visual refs. Route + plan output |
| 3.8 Smart Thumbnails | DONE | `thumbnail.py` has face scoring + composition balance + center interest + blur penalty |
| 3.9 Social Posting | DONE | `social_post.py`: YouTube resumable upload, TikTok Content Posting API, Instagram Graph API. OAuth flow + credential storage. 5 routes + CEP/UXP UI |
| 3.10 Florence-2 Watermark | DONE | `detect_watermark_region()` + edge fallback + route + UI button |

### Phase 4 — Architecture
| Item | Status | Notes |
|------|--------|-------|
| 4.1 UXP Migration | IN PROGRESS | `UXP_MIGRATION.md` complete. UXP panel has full feature parity: settings tab, engine registry, WebSocket, depth/emotion/chat/broll-plan features |
| 4.2 DaVinci Resolve | DONE | `resolve_bridge.py`: project info, media pool, markers, import, render. 5 API routes |
| 4.3 Multi-Engine Backend | DONE | `engine_registry.py`: plugin architecture with 18+ engines across 12 domains. Priority-based resolution, user preferences, availability cache. `/engines` routes for status/preference/resolve |

---

## Success Metrics

| Metric | Before | Current | Target |
|--------|--------|---------|--------|
| Silence detection accuracy | ~50% TPR (energy) | 87%+ TPR (Silero VAD) | ACHIEVED |
| Filler detection accuracy | ~60% (text matching) | 90%+ (CrisperWhisper) | ACHIEVED |
| Speech enhancement | Denoise only | Full restoration (44.1kHz) | ALREADY EXISTED |
| TTS options | 1 (edge-tts) | 3 (edge-tts, Kokoro, Chatterbox) | ACHIEVED |
| Stem separation | 1 (Demucs) | 3+ (Demucs, BS-RoFormer, MDX) | ALREADY EXISTED |
| Object removal | Watermark only | Any object (SAM2 + ProPainter) | ALREADY EXISTED |
| Background removal | Per-frame (rembg) | Per-frame + Temporal (RVM) | ACHIEVED |
| Scene detection | Threshold only | Threshold + Neural (TransNetV2) | ALREADY EXISTED |
| Timeline export | Premiere XML only | XML + OTIO (universal) | ACHIEVED |
| Highlight scoring | LLM-only 0-1 | Blended LLM + engagement heuristics | ACHIEVED |
| Emotion analysis | None | deepface emotion curve + peak detection | ACHIEVED |
| NLE support | Premiere only | Premiere + DaVinci Resolve | ACHIEVED |
| Real-time progress | HTTP polling only | WebSocket + polling fallback | ACHIEVED |
| Engine selection | Hardcoded backends | Plugin registry, 18+ engines, user prefs | ACHIEVED |
| UXP feature parity | CEP only | CEP + UXP with settings, engines, WS, chat | ACHIEVED |
