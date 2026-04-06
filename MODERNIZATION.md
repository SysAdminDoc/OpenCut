# OpenCut — Module Modernization Plan

**Created**: 2026-04-06
**Baseline**: v1.9.18 (68 core modules, 35 external dependencies, 15+ hardcoded models)
**Last Audit**: Batches 27-29 (150 bugs fixed across 52 files)

This document tracks the current state of every external dependency and AI model used in OpenCut, identifies upgrades and alternatives, and prioritizes modernization work.

---

## Dependency Inventory

### 68 Core Modules by Domain

#### Audio Processing (11 modules)
| Module | Functions | External Deps | Notes |
|--------|-----------|---------------|-------|
| `audio.py` | extract_audio_pcm/wav, analyze_energy, find_emphasis_points | None | FFmpeg only, stable |
| `audio_suite.py` | denoise, isolate, measure/normalize loudness, detect_beats, ducking, effects | None | FFmpeg only, 12 effect presets |
| `audio_pro.py` | apply_pedalboard_effect, deepfilter_denoise | pedalboard, deepfilternet, torch | Pro audio chain |
| `audio_duck.py` | sidechain_duck, mix_with_duck, auto_duck_video, mix_audio_tracks | None | FFmpeg sidechaincompress |
| `audio_enhance.py` | enhance_speech, enhance_speech_clearvoice | torch, torchaudio, resemble_enhance, clearvoice | Two enhancement backends |
| `loudness_match.py` | measure/normalize_to_lufs, batch_loudness_match | None | FFmpeg loudnorm two-pass |
| `music_gen.py` | generate_tone/sfx/silence, concatenate_audio | None | FFmpeg lavfi synthesis |
| `music_ai.py` | generate_music, melody-conditioned, continue, ace_step | audiocraft, soundfile, torch, ace_step | MusicGen + ACE-Step |
| `voice_gen.py` | edge_tts, kokoro, chatterbox TTS | edge_tts, kokoro, chatterbox, torch, soundfile | 3 TTS backends |
| `silence.py` | detect_silences (FFmpeg + Silero VAD), speed_up_silences | torch (VAD path) | Dual detection engine |
| `speed_ramp.py` | change_speed, reverse_video, speed_ramp | None | FFmpeg setpts/atempo |

#### Video Processing (18 modules)
| Module | Functions | External Deps | Notes |
|--------|-----------|---------------|-------|
| `video_fx.py` | stabilize, chromakey, LUT, vignette, grain, letterbox | None | FFmpeg only |
| `video_ai.py` | upscale (ESRGAN), rembg, RVM matting, interpolate, denoise | realesrgan, basicsr, rembg, cv2, torch | Multi-AI video |
| `upscale_pro.py` | upscale_realesrgan, SeedVR2 | torch, cv2, diffusers | Premium upscaling |
| `chromakey.py` | HSV chromakey compositing | cv2, numpy | OpenCV-based |
| `style_transfer.py` | Neural style transfer, arbitrary AdaIN | cv2 | OpenCV DNN .t7 models |
| `particles.py` | Particle overlays (confetti, sparkles, snow) | cv2, PIL, numpy | 8 presets |
| `transitions_3d.py` | Video transitions (FFmpeg xfade + ModernGL) | moderngl (optional) | |
| `color_management.py` | Colorspace convert, color correct, OCIO | PyOpenColorIO (optional) | |
| `color_match.py` | YCbCr histogram matching | cv2, numpy | |
| `export_presets.py` | Platform export profiles | None | 13 platforms |
| `motion_graphics.py` | Animated titles, kinetic typography | manim (optional) | Falls back to FFmpeg drawtext |
| `lut_library.py` | LUT management, AI LUT generation | PIL, numpy | |
| `face_reframe.py` | MediaPipe face-tracking auto-framing | mediapipe, cv2, numpy | |
| `face_tools.py` | Face detection, blur, tracking | mediapipe, cv2 | |
| `face_swap.py` | Face swap + enhancement (InsightFace, GFPGAN) | insightface, onnxruntime, cv2, torch, gfpgan | |
| `object_removal.py` | SAM2 masks + ProPainter/LaMA inpainting | torch, cv2, numpy | |
| `depth_effects.py` | Depth map, bokeh, parallax via Depth Anything V2 | torch, transformers, cv2, numpy | |
| `social_post.py` | YouTube/TikTok/Instagram direct posting | None (stdlib urllib) | |

#### AI/ML — Transcription, Diarization, Scene Detection (7 modules)
| Module | Functions | External Deps | Notes |
|--------|-----------|---------------|-------|
| `captions.py` | Multi-backend transcription | faster_whisper, whisperx, whisper, torch | 3 Whisper backends |
| `crisper_whisper.py` | Filler word detection | transformers, torch, faster_whisper | CrisperWhisper model |
| `diarize.py` | Speaker diarization | pyannote.audio, torch | pyannote 3.1 |
| `multimodal_diarize.py` | Audio+visual speaker diarization | insightface, facenet_pytorch, cv2, torch | |
| `scene_detect.py` | 3-backend scene detection | transnetv2, scenedetect, torch | FFmpeg + TransNetV2 + PySceneDetect |
| `emotion_highlights.py` | Facial expression emotion detection | deepface, cv2 | |
| `auto_edit.py` | Motion/audio-based auto-editing | auto-editor (CLI) | Subprocess invocation |

#### Captions/Subtitles (5 modules)
| Module | Functions | External Deps | Notes |
|--------|-----------|---------------|-------|
| `captions_enhanced.py` | WhisperX alignment, NLLB/SeamlessM4T translation | whisperx, ctranslate2, sentencepiece, pysubs2, transformers | |
| `caption_burnin.py` | Burn subtitles into video | None | FFmpeg ass/subtitles |
| `styled_captions.py` | 18-style caption overlay renderer | PIL (Pillow) | Pillow + FFmpeg pipe |
| `animated_captions.py` | CapCut-style word-by-word animation | cv2, PIL, numpy | |
| `fillers.py` | Filler word detection/removal | None | Pure Python |

#### LLM Integration (4 modules)
| Module | Functions | External Deps | Notes |
|--------|-----------|---------------|-------|
| `llm.py` | Ollama/OpenAI/Anthropic abstraction | None (stdlib urllib) | Zero pip deps |
| `chat_editor.py` | Multi-turn chat editing assistant | None (uses llm.py) | |
| `nlp_command.py` | Natural language to API route mapping | None (uses llm.py) | 19 command patterns |
| `highlights.py` | LLM-powered highlight extraction | None (uses llm.py) | |

#### Utilities (13 modules)
All pure Python, no external dependencies. Includes: zoom.py, auto_zoom.py (cv2), repeat_detect.py, chapter_gen.py, footage_search.py, footage_index_db.py, deliverables.py, multicam.py, multicam_xml.py, workflow.py, streaming.py, context_awareness.py, plugins.py, batch_executor.py, batch_process.py, engine_registry.py, broll_insert.py, broll_generate.py (diffusers, torch), thumbnail.py (cv2), resolve_bridge.py, ws_bridge.py.

---

## Hardcoded Models

| Model Reference | File | Version | Check For Updates |
|----------------|------|---------|-------------------|
| `"facebook/musicgen-small/medium/large/melody"` | music_ai.py | MusicGen v1 | Stable Audio Open as alternative |
| `"facebook/seamless-m4t-v2-large"` | captions_enhanced.py | v2-large | Newer checkpoints |
| `"nyrahealth/CrisperWhisper"` | crisper_whisper.py | Current | Newer filler detection models |
| `"pyannote/speaker-diarization-3.1"` | diarize.py | 3.1 | Check for 3.2+ |
| `"LiheYoung/depth-anything-v2-small"` | depth_effects.py | v2-small | Depth Anything V3, Depth Pro |
| `"PeterL1n/RobustVideoMatting"` | video_ai.py | RVM v1 | Check for v2 |
| `"snakers4/silero-vad"` | silence.py | v4 (assumed) | v5 with ONNX support |
| `"RealESRGAN_x4plus"` | video_ai.py | x4plus | 4x-UltraSharp, SwinIR |
| `"JustFrederik/nllb-200-distilled-600M-ct2-float16"` | captions_enhanced.py | 600M | Deprecate for SeamlessM4T |
| `"vggface2"` (facenet pretrained) | multimodal_diarize.py | vggface2 | InsightFace alternatives |
| OpenCV .t7 style transfer URLs | style_transfer.py | Stanford 2017 | Modern arbitrary style transfer |
| Anthropic API version `"2023-06-01"` | llm.py | 2023-06-01 | Latest API version |
| `"MossFormer2_SE_48K"` / `"FRCRN_SE_16K"` | audio_enhance.py | Current | Verify latest ClearerVoice models |
| `"buffalo_l"` / `"inswapper_128.onnx"` | face_swap.py | Current | InsightFace updates |
| HunyuanVideo / CogVideoX / SVD | broll_generate.py | Various | Wan 2.1 as primary |

---

## Modernization Priorities

### TIER 1 -- Critical Updates (Breakage Risk or Major Gaps)

| # | Issue | Module | Action | Status |
|---|-------|--------|--------|--------|
| 1 | auto-editor v30 rewritten in Nim; pip package frozen at v29.3.1 | `auto_edit.py` | Detect native Nim binary, pip fallback | TODO |
| 2 | Anthropic API version stale (`2023-06-01`) | `llm.py` | Update version string, add latest model IDs | TODO |
| 3 | CTranslate2 in maintenance mode | `captions_enhanced.py` | Make SeamlessM4T primary, NLLB fallback | TODO |
| 4 | FFmpeg minterpolate slow + artifacting | `video_ai.py` | Add RIFE neural interpolation as `method="rife"` | TODO |
| 5 | B-roll gen backends outdated (SVD superseded) | `broll_generate.py` | Add Wan 2.1 as primary, remove SVD | TODO |

### TIER 2 -- Significant Improvements Available

| # | Improvement | Module | Action | Status |
|---|-------------|--------|--------|--------|
| 6 | Silero VAD v5 (2x faster, ONNX) | `silence.py` | Upgrade torch.hub model, add ONNX path | TODO |
| 7 | Kokoro v1.0 ONNX model available | `voice_gen.py` | Verify version, test ONNX inference | TODO |
| 8 | DeepFilterNet3 improvements | `audio_pro.py` | Verify v0.5.x installed | TODO |
| 9 | CodeFormer > GFPGAN for degraded faces | `face_swap.py` | Wire CodeFormer as default enhancer | TODO |
| 10 | Depth Anything checkpoint verification | `depth_effects.py` | Verify latest HuggingFace checkpoints | TODO |
| 11 | Two-stage scene detection pipeline | `scene_detect.py` | Add `method="hybrid"` (PySceneDetect + TransNetV2) | TODO |
| 12 | .t7 style transfer models are 2017-era | `style_transfer.py` | Research AesPA-Net / InST for temporal consistency | TODO |
| 13 | Stable Audio Open vs MusicGen | `music_ai.py` | Add as backend option | TODO |
| 14 | AV1 export preset (40% smaller files) | `export_presets.py` | Add SVT-AV1 + NVENC AV1 presets | TODO |
| 15 | Gyroflow integration for camera stabilization | `video_fx.py` | Research integration path | TODO |
| 16 | Caption rendering performance (Pillow bottleneck) | `styled_captions.py` | Research skia-python or FFmpeg drawvg (Cairo) | TODO |

### TIER 3 -- Verify & Maintain (Working Well)

| # | Module | Action | Status |
|---|--------|--------|--------|
| 17 | `captions.py` | Verify faster-whisper>=1.1 compat, test turbo-ct2 model | TODO |
| 18 | `diarize.py` | Verify pyannote 3.1 still latest | TODO |
| 19 | `audio_enhance.py` | Verify Resemble Enhance + ClearerVoice current | TODO |
| 20 | `video_ai.py` (upscale) | Verify Real-ESRGAN x4plus current, research 4x-UltraSharp | TODO |
| 21 | `object_removal.py` | Verify SAM2 + ProPainter current | TODO |
| 22-35 | Remaining modules | No external dep changes needed | OK |

---

## External Library Versions (Reference)

| Library | Current in OpenCut | Latest Available (Q1 2026) | Notes |
|---------|-------------------|---------------------------|-------|
| FFmpeg | 7.x (assumed) | **8.1 "Hoare"** | D3D12 accel, Vulkan ProRes, drawvg filter, AV1 improvements |
| PyTorch | 2.x | **2.11** | torch.compile mature, Intel Arc support |
| faster-whisper | >=1.1 | 1.1.x | Turbo + distil model support |
| pedalboard | 0.9.x | **0.9.22** | VST3/AU plugin hosting, WindowedSinc32 resampler |
| ONNX Runtime | 1.x | **1.24.4** | CUDA 12.x required, RTX EP, auto EP selection |
| auto-editor | 29.3.1 (pip) | **30.0.0 (Nim)** | Python pip package frozen |
| Kokoro TTS | 0.x | **v1.0** | ONNX model, 210x realtime, 54 voices |
| DeepFilterNet | 0.4.x | **0.5.x (DFNet3)** | PESQ 3.5-4.0+ |
| torchaudio | 2.x | **2.10 (maintenance)** | Migrating to TorchCodec |
| Wan 2.1 | Not in project | **Wan 2.1** | Best open-source text/image-to-video, Apache 2.0 |

---

## Execution Phases

### Phase 1: Quick Wins (1-2 days each)
1. Silero VAD v5 upgrade
2. DeepFilterNet3 verification
3. Kokoro v1.0 verification
4. Depth Anything checkpoint verification
5. AV1 export preset addition
6. Anthropic API version update

### Phase 2: Medium Effort (3-5 days each)
1. auto-editor Nim binary migration
2. Wan 2.1 B-roll backend
3. RIFE frame interpolation
4. Two-stage scene detection
5. SeamlessM4T as primary translation
6. CodeFormer wiring

### Phase 3: Larger Projects (1-2 weeks each)
1. Hardware-accelerated encoding (NVENC/QSV/AV1)
2. Caption rendering performance (Skia/Cairo)
3. Modern style transfer (temporal consistency)
4. FFmpeg 8.x feature adoption

---

*Update this document when completing any modernization task. Mark status as DONE with date.*
