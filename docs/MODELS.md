# OpenCut model + license cards

Generated from `opencut/model_cards.py`. **Do not hand-edit** — regenerate with `python -m opencut.tools.dump_model_cards`.

Total optional AI/model surfaces: **47**. Each row carries license, hardware, install hint, privacy posture, and (where relevant) an advisory note. Backends not listed here are infrastructure guards on the explicit `NON_AI_CHECKS` allowlist.

## Audio

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [AudioCraft / MusicGen](https://github.com/facebookresearch/audiocraft) | MIT | gpu (>= 8 GB VRAM) | local-only | `pip install audiocraft` |
| [BeatNet beat/downbeat tracker](https://github.com/mjhydri/BeatNet) | MIT | cpu | local-only | `pip install BeatNet` |
| [Demucs (htdemucs / hdemucs)](https://github.com/facebookresearch/demucs) | MIT | cpu/gpu | local-only | `pip install demucs` |
| [Edge TTS (Microsoft cloud voices)](https://github.com/rany2/edge-tts) | MIT (client) | cpu | cloud — text is sent to Microsoft's Speech API | `pip install edge-tts` |
| [ElevenLabs cloud TTS](https://github.com/elevenlabs/elevenlabs-python) | proprietary client SDK; cloud service | cpu (client) | cloud — text is sent to ElevenLabs | `pip install elevenlabs + ELEVENLABS_API_KEY` |
| [F5-TTS (zero-shot voice clone)](https://github.com/SWivid/F5-TTS) | MIT | gpu (>= 6 GB VRAM) | local-only | `pip install f5-tts` |
| [OmniVoice TTS](https://github.com/k2-fsa/OmniVoice) | Apache-2.0 | gpu | local-only | `pip install omnivoice` |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | MIT | gpu | local-only | `pip install resemble-enhance` |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | cpu | local-only | `pip install silero-vad` |
| [VidMuse video-to-music](https://vidmuse.github.io/) | Apache-2.0 | gpu | local-only | `pip install vidmuse (stub — roadmap H2.6)` |
| [WhisperX voice-command grammar](https://github.com/linto-ai/whisper-timestamped) | MIT | cpu/gpu | local-only | `pip install whisperx + faster-whisper` |

**Advisory notes**:
- *Edge TTS (Microsoft cloud voices)* — Sends synthesis text to Microsoft over HTTPS. Disable for offline-only deployments.
- *ElevenLabs cloud TTS* — Requires an ElevenLabs account. Synthesised audio + reference voices may be retained per their TOS.
- *VidMuse video-to-music* — Roadmap stub.

## Captions

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) | Apache-2.0 | gpu | local-only | `pip install ctranslate2 + crisper-whisper` |
| [Multimodal diarisation pipeline](https://github.com/m-bain/whisperX) | MIT | gpu | local-only (HF token used for weights download only) | `pip install whisperx + pyannote.audio (HF token required)` |
| [pyonfx ASS karaoke](https://github.com/CoffeeStraw/PyonFX) | LGPL-3.0 | cpu | local-only | `pip install pyonfx` |

**Advisory notes**:
- *Multimodal diarisation pipeline* — pyannote checkpoints are gated by Hugging Face acceptance — set HUGGINGFACE_HUB_TOKEN.
- *pyonfx ASS karaoke* — LGPL — using as a library inside a proprietary distribution requires legal review.

## Editing

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [auto-editor](https://github.com/WyattBlue/auto-editor) | Unlicense (public domain) | cpu | local-only | `pip install auto-editor` |
| [B-roll suggestion pipeline](https://github.com/openai/whisper) | MIT (orchestration; underlying providers vary) | cpu/gpu | local + optional cloud (LLM dependency) | `pip install transformers torch` |
| [OpenCV / face-tracker auto-zoom](https://github.com/opencv/opencv-python) | Apache-2.0 | cpu | local-only | `pip install opencv-python` |

## Generation

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [Cloud gen-video (Hailuo / Seedance)](https://hailuo-02.com) | proprietary cloud service | cpu (client) | cloud — prompts + reference frames leave the machine | `API key required — opt-in only` |

**Advisory notes**:
- *Cloud gen-video (Hailuo / Seedance)* — Disabled by default. Set HAILUO_API_KEY / SEEDANCE_API_KEY to enable.

## Lipsync

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [GaussianHeadTalk / FantasyTalking2](https://wacv.thecvf.com/) | Apache-2.0 | gpu (>= 12 GB VRAM) | local-only | `pip install gaussian-head-talk + fantasytalking2 (stubs)` |

**Advisory notes**:
- *GaussianHeadTalk / FantasyTalking2* — Roadmap stub (H3.3).

## Llm

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [Local LLM provider (any of: Ollama, vLLM, OpenAI-compatible)](https://github.com/ollama/ollama) | varies per backend | gpu (recommended) | local-only when using Ollama / vLLM; cloud otherwise | `install Ollama or set OPENAI_API_KEY` |
| [Ollama backend](https://github.com/ollama/ollama) | MIT | cpu/gpu | local-only | `install Ollama from https://ollama.ai` |
| [VideoAgent / ViMax agentic search](https://github.com/HKUDS/VideoAgent) | MIT | cpu/gpu | local + optional cloud (depends on configured LLM) | `pip install videoagent (stub — roadmap H3.1)` |
| [Virality / hook scorer](https://github.com/SysAdminDoc/OpenCut) | MIT (orchestration) | cpu | local + optional cloud LLM | `ships with core; optional LLM dependency` |

**Advisory notes**:
- *Local LLM provider (any of: Ollama, vLLM, OpenAI-compatible)* — Configure OPENCUT_LLM_PROVIDER=ollama to force local; otherwise cloud providers can be hit.
- *VideoAgent / ViMax agentic search* — Roadmap stub. LLM dependency may route through a cloud provider.
- *Virality / hook scorer* — Falls back to a keyword lexicon when no LLM provider is configured.

## Video

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) | MIT | gpu | local-only | `pip install all-in-one-deflicker` |
| [CLIP-IQA+](https://github.com/IceClear/CLIP-IQA) | Apache-2.0 | gpu | local-only | `pip install pyiqa` |
| [colour-science (CIE / vectorscope math)](https://github.com/colour-science/colour) | BSD-3-Clause | cpu | local-only | `pip install colour-science` |
| [DDColor](https://github.com/piddnad/DDColor) | Apache-2.0 | gpu | local-only | `pip install ddcolor` |
| [DeepFace](https://github.com/serengil/deepface) | MIT | cpu/gpu | local-only | `pip install deepface` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2.0 | gpu | local-only | `pip install transformers torch` |
| [Face reshape / beauty filter](https://github.com/google-ai-edge/mediapipe) | MIT | cpu/gpu | local-only | `pip install mediapipe + opencv-python` |
| [FlashVSR (streaming VSR)](https://github.com/OpenImagingLab/FlashVSR) | Apache-2.0 | gpu (>= 12 GB VRAM) | local-only | `pip install flashvsr (stub — roadmap H2.1)` |
| [HF upscale model hub](https://huggingface.co/models?other=image-super-resolution) | varies per model | gpu | local-only (model downloads only) | `pip install diffusers transformers` |
| [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) | Apache-2.0 | cpu/gpu | local-only | `pip install hsemotion` |
| [LaMa inpainting watermark removal](https://github.com/advimman/lama) | Apache-2.0 | gpu | local-only | `pip install simple-lama-inpainting` |
| [MediaPipe (face mesh, hands, pose)](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | cpu/gpu | local-only | `pip install mediapipe` |
| [ProPainter](https://github.com/sczhou/ProPainter) | NTU S-Lab License 1.0 | gpu (>= 12 GB VRAM) | local-only | `pip install propainter` |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | BSD-3-Clause | cpu | local-only | `pip install scenedetect` |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | gpu | local-only | `pip install realesrgan` |
| [ReEzSynth (Ebsynth successor)](https://github.com/FuouM/ReEzSynth) | Apache-2.0 | gpu | local-only | `pip install reezsynth (stub — roadmap H2.5)` |
| [rembg (U^2-Net family)](https://github.com/danielgatis/rembg) | MIT | cpu/gpu | local-only | `pip install rembg` |
| [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) | GPL-3.0 | gpu | local-only | `pip install robust-video-matting` |
| [ROSE shadow-aware inpaint](https://rose2025-inpaint.github.io/) | Apache-2.0 | gpu | local-only | `pip install rose-inpaint (stub — roadmap H2.2)` |
| [Sammie-Roto-2 (VideoMaMa)](https://github.com/Zarxrax/Sammie-Roto-2) | MIT | gpu | local-only | `pip install sammie-roto (stub — roadmap H2.3)` |
| [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) | Apache-2.0 | gpu | local-only | `pip install sam2` |
| [Skin retouch (bilateral / GAN)](https://github.com/opencv/opencv-python) | MIT | cpu/gpu | local-only | `pip install opencv-python (optional onnx model)` |
| [TransNetV2 shot-boundary](https://github.com/soCzech/TransNetV2) | MIT | gpu | local-only | `pip install transnetv2` |
| [VRT / RVRT video restoration](https://github.com/JingyunLiang/VRT) | Apache-2.0 | gpu (>= 16 GB VRAM for VRT) | local-only | `pip install vrt-restoration` |

**Advisory notes**:
- *FlashVSR (streaming VSR)* — Roadmap stub; check returns False until the implementation lands.
- *ProPainter* — Custom S-Lab license — non-commercial use only. Verify before shipping in paid distributions.
- *ReEzSynth (Ebsynth successor)* — Roadmap stub.
- *ROSE shadow-aware inpaint* — Roadmap stub.
- *Robust Video Matting (RVM)* — GPL-3.0 — bundling the model with proprietary distributions requires legal review.
- *Sammie-Roto-2 (VideoMaMa)* — Roadmap stub.
- *HF upscale model hub* — Each Hugging Face model has its own license — check before redistributing weights.

---

## Excluded infrastructure checks

These `opencut.checks` functions do not gate a model/AI dependency and are deliberately excluded from this surface:

- `check_aaf_adapter_available`
- `check_ab_av1_available`
- `check_atheris_available`
- `check_birefnet_available`
- `check_changelog_feed_available`
- `check_color_match_available`
- `check_cursor_zoom_available`
- `check_declarative_compose_available`
- `check_demo_bundle_available`
- `check_deprecation_registry_available`
- `check_disk_monitor_available`
- `check_event_moments_available`
- `check_footage_search_available`
- `check_gist_sync_available`
- `check_gpu_semaphore_available`
- `check_issue_report_available`
- `check_loudness_match_available`
- `check_neural_interp_available`
- `check_obs_bridge_available`
- `check_onboarding_available`
- `check_openapi_available`
- `check_otio_available`
- `check_otio_diff_available`
- `check_pedalboard_available`
- `check_quality_metrics_available`
- `check_rate_limit_categories_available`
- `check_request_correlation_available`
- `check_resolve_available`
- `check_rife_cli_available`
- `check_runpod_available`
- `check_sentry_available`
- `check_shaka_available`
- `check_social_post_available`
- `check_srt_available`
- `check_svtav1_psy_available`
- `check_temp_cleanup_available`
- `check_vmaf_available`
- `check_vvc_available`
- `check_websocket_available`

