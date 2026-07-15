# OpenCut model + license cards

Generated from `opencut/model_cards.py`. **Do not hand-edit** — regenerate with `python -m opencut.tools.dump_model_cards`.

Total optional AI/model surfaces: **67**. Each row carries license, hardware, install hint, privacy posture, and (where relevant) an advisory note. Backends not listed here are infrastructure guards on the explicit `NON_AI_CHECKS` allowlist.

## Audio

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [ACE-Step (full-song music with lyrics)](https://github.com/ACEStudio/ACE-Step) | Apache-2.0 | gpu (>= 8 GB VRAM) | local-only | `pip install acestep` |
| [AudioCraft / MusicGen](https://github.com/facebookresearch/audiocraft) | MIT | gpu (>= 8 GB VRAM) | local-only | `pip install "opencut[music]" (Python 3.11; Torch 2.1 stack)` |
| [BeatNet beat/downbeat tracker](https://github.com/mjhydri/BeatNet) | MIT | cpu | local-only | `pip install BeatNet` |
| [Chatterbox TTS (emotional voice clone)](https://github.com/resemble-ai/chatterbox) | MIT | gpu | local-only | `pip install chatterbox-tts` |
| [Demucs (htdemucs / hdemucs)](https://github.com/facebookresearch/demucs) | MIT | cpu/gpu | local-only | `pip install demucs` |
| [DiffRhythm (diffusion full-song generator)](https://github.com/ASLP-lab/DiffRhythm) | Apache-2.0 | gpu (>= 8 GB VRAM) | local-only | `git clone https://github.com/ASLP-lab/DiffRhythm && pip install -r DiffRhythm/requirements.txt` |
| [Edge TTS (Microsoft cloud voices)](https://github.com/rany2/edge-tts) | MIT (client) | cpu | cloud — text is sent to Microsoft's Speech API | `pip install edge-tts` |
| [ElevenLabs cloud TTS](https://github.com/elevenlabs/elevenlabs-python) | proprietary client SDK; cloud service | cpu (client) | cloud — text is sent to ElevenLabs | `pip install elevenlabs + ELEVENLABS_API_KEY` |
| [F5-TTS (zero-shot voice clone)](https://github.com/SWivid/F5-TTS) | MIT | gpu (>= 6 GB VRAM) | local-only | `pip install f5-tts` |
| [Kokoro TTS (82M, CPU-only)](https://github.com/hexgrad/kokoro) | Apache-2.0 | cpu | local-only | `pip install kokoro (needs espeak-ng for some languages)` |
| [OmniVoice TTS](https://github.com/k2-fsa/OmniVoice) | Apache-2.0 | gpu | local-only | `pip install omnivoice` |
| [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) | MIT | gpu | local-only | `pip install "opencut[enhance]" (Python 3.11; Torch 2.1 stack)` |
| [Silero VAD](https://github.com/snakers4/silero-vad) | MIT | cpu | local-only | `pip install silero-vad` |
| [Spark-TTS (CPU-native zero-shot)](https://github.com/SparkAudio/Spark-TTS) | Apache-2.0 | cpu | local-only | `pip install sparktts` |
| [VidMuse video-to-music](https://vidmuse.github.io/) | Apache-2.0 | gpu | local-only | `pip install vidmuse (placeholder; see docs/MODELS.md)` |
| [WhisperX voice-command grammar](https://github.com/linto-ai/whisper-timestamped) | MIT | cpu/gpu | local-only | `pip install whisperx + faster-whisper` |

**Advisory notes**:
- *Edge TTS (Microsoft cloud voices)* — Sends synthesis text to Microsoft over HTTPS. Disable for offline-only deployments.
- *ElevenLabs cloud TTS* — Requires an ElevenLabs account. Synthesised audio + reference voices may be retained per their TOS.
- *VidMuse video-to-music* — Readiness placeholder; check returns False until the implementation lands.

## Captions

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) | Apache-2.0 | gpu | local-only | `pip install ctranslate2 + crisper-whisper` |
| [Moonshine ASR (CPU-optimized STT)](https://github.com/usefulsensors/moonshine) | MIT (English models) | cpu | local-only | `pip install moonshine` |
| [Multimodal diarisation pipeline](https://github.com/m-bain/whisperX) | MIT | gpu | local-only (HF token used for weights download only) | `pip install whisperx + pyannote.audio (HF token required)` |
| [NLLB-200 distilled caption translation](https://huggingface.co/facebook/nllb-200-distilled-600M) | CC-BY-NC-4.0 | cpu/gpu | local-only (model downloads only) | `pip install ctranslate2 sentencepiece huggingface-hub` |
| [NVIDIA NeMo ASR (Parakeet / Canary)](https://github.com/NVIDIA/NeMo) | Apache-2.0 (toolkit; model terms vary) | cpu/gpu | local-only (model downloads only) | `pip install "nemo_toolkit[asr]"` |
| [NVIDIA Sortformer diarization](https://github.com/NVIDIA/NeMo) | Apache-2.0 (NeMo toolkit; model terms vary) | gpu | local-only (model downloads only) | `pip install "nemo_toolkit[asr]"` |
| [pyannote speaker diarization](https://huggingface.co/pyannote/speaker-diarization-community-1) | CC-BY-4.0 (community-1; legacy model terms vary) | cpu/gpu | local-only (HF token used for gated weight download only) | `pip install pyannote.audio` |
| [pyonfx ASS karaoke](https://github.com/CoffeeStraw/PyonFX) | LGPL-3.0 | cpu | local-only | `pip install pyonfx` |
| [SeamlessM4T v2 caption translation](https://huggingface.co/facebook/seamless-m4t-v2-large) | CC-BY-NC-4.0 | gpu (recommended) | local-only (model downloads only) | `pip install transformers torch sentencepiece` |

**Advisory notes**:
- *pyannote speaker diarization* — The community-1 pipeline is the default; legacy pyannote 3.1 remains a fallback and may require accepting gated Hugging Face terms.
- *Moonshine ASR (CPU-optimized STT)* — Multilingual models use a community (non-commercial) license and are gated separately.
- *Multimodal diarisation pipeline* — pyannote checkpoints are gated by Hugging Face acceptance — set HUGGINGFACE_HUB_TOKEN.
- *NVIDIA NeMo ASR (Parakeet / Canary)* — Parakeet and Canary are selectable ASR engines; verify each downloaded model card before redistributing weights.
- *NLLB-200 distilled caption translation* — Non-commercial model license: never auto-selected by /captions/translate and not bundled with OpenCut.
- *NLLB-200 distilled caption translation* — Operators must pass accept_restricted_license=true before download/use.
- *pyonfx ASS karaoke* — LGPL — using as a library inside a proprietary distribution requires legal review.
- *SeamlessM4T v2 caption translation* — Non-commercial model license: never auto-selected by /captions/translate and not bundled with OpenCut.
- *SeamlessM4T v2 caption translation* — Operators must pass accept_restricted_license=true before download/use.
- *NVIDIA Sortformer diarization* — Sortformer is an optional NeMo-gated diarization engine; verify the selected checkpoint license before redistribution.

## Editing

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [auto-editor](https://github.com/WyattBlue/auto-editor) | Unlicense (public domain) | cpu | local-only | `Native v30+ binary: https://github.com/WyattBlue/auto-editor/releases — or legacy pip: pip install auto-editor` |
| [B-roll suggestion pipeline](https://github.com/openai/whisper) | MIT (orchestration; underlying providers vary) | cpu/gpu | local + optional cloud (LLM dependency) | `pip install transformers torch` |
| [OpenCV / face-tracker auto-zoom](https://github.com/opencv/opencv-python) | Apache-2.0 | cpu | local-only | `pip install opencv-python` |

## Generation

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [Cloud gen-video (Hailuo / Seedance)](https://hailuo-02.com) | proprietary cloud service | cpu (client) | cloud — prompts + reference frames leave the machine | `API key required — opt-in only` |
| [FLUX.1 Kontext-dev (image editing)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Apache-2.0 (dev variant) | gpu | local-only | `pip install diffusers>=0.29 torch transformers accelerate` |
| [FramePack (image-to-video diffusion)](https://github.com/lllyasviel/FramePack) | Apache-2.0 | gpu (>= 6 GB VRAM) | local-only | `pip install framepack` |

**Advisory notes**:
- *Cloud gen-video (Hailuo / Seedance)* — Disabled by default. Set HAILUO_API_KEY / SEEDANCE_API_KEY to enable.
- *FLUX.1 Kontext-dev (image editing)* — Kontext-dev weights (~24 GB) download on first use; the pro variant is commercial and is not used.

## Lipsync

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [GaussianHeadTalk / FantasyTalking2](https://wacv.thecvf.com/) | Apache-2.0 | gpu (>= 12 GB VRAM) | local-only | `pip install gaussian-head-talk + fantasytalking2 (stubs)` |
| [LatentSync diffusion lip-sync](https://github.com/bytedance/LatentSync) | Apache-2.0 (code; checkpoint license unconfirmed) | gpu (>= 6 GB VRAM) | local-only (model downloads only) | `pip install diffusers torch torchvision` |

**Advisory notes**:
- *LatentSync diffusion lip-sync* — The engine is opt-in and not auto-selected until the checkpoint license is confirmed.
- *GaussianHeadTalk / FantasyTalking2* — Readiness placeholder; see docs/MODELS.md for install and license status.

## Llm

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [Local LLM provider (any of: Ollama, vLLM, OpenAI-compatible)](https://github.com/ollama/ollama) | varies per backend | gpu (recommended) | local-only when using Ollama / vLLM; cloud otherwise | `install Ollama or set OPENAI_API_KEY` |
| [Ollama backend](https://github.com/ollama/ollama) | MIT | cpu/gpu | local-only | `install Ollama from https://ollama.ai` |
| [VideoAgent / ViMax agentic search](https://github.com/HKUDS/VideoAgent) | MIT | cpu/gpu | local + optional cloud (depends on configured LLM) | `pip install videoagent (placeholder; see docs/MODELS.md)` |
| [Virality / hook scorer](https://github.com/SysAdminDoc/OpenCut) | MIT (orchestration) | cpu | local + optional cloud LLM | `ships with core; optional LLM dependency` |

**Advisory notes**:
- *Local LLM provider (any of: Ollama, vLLM, OpenAI-compatible)* — Configure OPENCUT_LLM_PROVIDER=ollama to force local; otherwise cloud providers can be hit.
- *VideoAgent / ViMax agentic search* — Readiness placeholder. LLM dependency may route through a cloud provider.
- *Virality / hook scorer* — Falls back to a keyword lexicon when no LLM provider is configured.

## Video

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [All-In-One-Deflicker](https://github.com/ChenyangLEI/All-In-One-Deflicker) | MIT | gpu | local-only | `pip install all-in-one-deflicker` |
| [AutoShot gradual-transition scene detector](https://github.com/wentaozhu/AutoShot) | Custom (verify upstream before redistributing weights) | cpu/gpu | local-only | `git clone https://github.com/wentaozhu/AutoShot && pip install -e AutoShot` |
| [CLIP-IQA+](https://github.com/IceClear/CLIP-IQA) | Apache-2.0 | gpu | local-only | `pip install pyiqa` |
| [colour-science (CIE / vectorscope math)](https://github.com/colour-science/colour) | BSD-3-Clause | cpu | local-only | `pip install colour-science` |
| [DDColor](https://github.com/piddnad/DDColor) | Apache-2.0 | gpu | local-only | `pip install ddcolor` |
| [DeepFace](https://github.com/serengil/deepface) | MIT | cpu/gpu | local-only | `pip install deepface` |
| [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) | Apache-2.0 | gpu | local-only | `pip install transformers torch` |
| [Face reshape / beauty filter](https://github.com/google-ai-edge/mediapipe) | MIT | cpu/gpu | local-only | `pip install mediapipe + opencv-python` |
| [FlashVSR (streaming VSR)](https://github.com/OpenImagingLab/FlashVSR) | Apache-2.0 | gpu (>= 12 GB VRAM) | local-only | `pip install flashvsr (placeholder; see docs/MODELS.md)` |
| [HF upscale model hub](https://huggingface.co/models?other=image-super-resolution) | varies per model | gpu | local-only (model downloads only) | `pip install diffusers transformers` |
| [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) | Apache-2.0 | cpu/gpu | local-only | `pip install hsemotion` |
| [IC-Light v1 relight](https://github.com/lllyasviel/IC-Light) | Apache-2.0 | gpu (>= 4 GB VRAM) | local-only (model downloads only) | `pip install diffusers>=0.32 torch` |
| [LaMa inpainting watermark removal](https://github.com/advimman/lama) | Apache-2.0 | gpu | local-only | `pip install simple-lama-inpainting` |
| [MediaPipe (face mesh, hands, pose)](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | cpu/gpu | local-only | `pip install mediapipe` |
| [MediaPipe visual multicam cues](https://github.com/google-ai-edge/mediapipe) | Apache-2.0 | cpu/gpu | local-only | `pip install mediapipe opencv-python-headless` |
| [ProPainter](https://github.com/sczhou/ProPainter) | NTU S-Lab License 1.0 | gpu (>= 12 GB VRAM) | local-only | `pip install propainter` |
| [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) | BSD-3-Clause | cpu | local-only | `pip install scenedetect` |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause | gpu | local-only | `pip install realesrgan` |
| [ReEzSynth (Ebsynth successor)](https://github.com/FuouM/ReEzSynth) | Apache-2.0 | gpu | local-only | `pip install reezsynth (placeholder; see docs/MODELS.md)` |
| [rembg (U^2-Net family)](https://github.com/danielgatis/rembg) | MIT | cpu/gpu | local-only | `pip install rembg` |
| [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) | GPL-3.0 | gpu | local-only | `pip install robust-video-matting` |
| [ROSE shadow-aware inpaint](https://rose2025-inpaint.github.io/) | Apache-2.0 | gpu | local-only | `pip install rose-inpaint (placeholder; see docs/MODELS.md)` |
| [SAM 3 text-prompted video segmentation](https://github.com/facebookresearch/sam3) | Custom (SAM License, commercial-permissive) | gpu | local-only (model downloads only) | `pip install sam3 torch` |
| [Sammie-Roto-2 (VideoMaMa)](https://github.com/Zarxrax/Sammie-Roto-2) | MIT | gpu | local-only | `pip install sammie-roto (placeholder; see docs/MODELS.md)` |
| [SeedVR2 one-step diffusion VSR](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) | Apache-2.0 | gpu (>= 8 GB VRAM) | local-only (model downloads only) | `pip install diffusers torch  # download seedvr2-3b via model manager` |
| [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/sam2) | Apache-2.0 | gpu | local-only | `pip install sam2` |
| [Skin retouch (bilateral / GAN)](https://github.com/opencv/opencv-python) | MIT | cpu/gpu | local-only | `pip install opencv-python (optional onnx model)` |
| [TransNetV2 shot-boundary](https://github.com/allenday/transnetv2_pytorch) | MIT | gpu | local-only | `pip install transnetv2-pytorch` |
| [VRT / RVRT video restoration](https://github.com/JingyunLiang/VRT) | Apache-2.0 | gpu (>= 16 GB VRAM for VRT) | local-only | `pip install vrt-restoration` |

**Advisory notes**:
- *AutoShot gradual-transition scene detector* — AutoShot is preferred over TransNetV2 when installed; keep it opt-in until the upstream license and weights are reviewed for release bundling.
- *FlashVSR (streaming VSR)* — Readiness placeholder; check returns False until the implementation lands.
- *IC-Light v1 relight* — OpenCut targets IC-Light v1 only; IC-Light v2 is not registered because the local module documents non-commercial/release concerns.
- *ProPainter* — Custom S-Lab license — non-commercial use only. Verify before shipping in paid distributions.
- *ReEzSynth (Ebsynth successor)* — Readiness placeholder; check returns False until the implementation lands.
- *ROSE shadow-aware inpaint* — Readiness placeholder; check returns False until the implementation lands.
- *Robust Video Matting (RVM)* — GPL-3.0 — bundling the model with proprietary distributions requires legal review.
- *SAM 3 text-prompted video segmentation* — SAM 3 is used for text-prompted object removal and falls back to SAM2/CLIP when absent.
- *Sammie-Roto-2 (VideoMaMa)* — Readiness placeholder; check returns False until the implementation lands.
- *HF upscale model hub* — Each Hugging Face model has its own license — check before redistributing weights.

## Vision

| Backend | License | Hardware | Privacy | Install |
|---|---|---|---|---|
| [pytesseract / EasyOCR text extraction](https://github.com/tesseract-ocr/tesseract) | Apache-2.0 | cpu/gpu | local-only | `pip install pytesseract easyocr  # plus system Tesseract for pytesseract` |

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

