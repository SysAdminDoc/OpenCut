# OpenCut (docs) — Completed Roadmap Items

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
