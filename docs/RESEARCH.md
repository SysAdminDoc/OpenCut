# OpenCut — Open Source Feature Research

> Comprehensive audit of open source projects that can improve, expand, or replace OpenCut features.
> Research conducted March 2026 across 6 domains using parallel web research agents.

---

## 1. Silence Removal & Filler Detection

### Current Implementation
Energy-based threshold detection with configurable dB threshold, min duration, padding before/after.

### Open Source Landscape

| Project | URL | Stars | Key Features |
|---------|-----|-------|-------------|
| **auto-editor** | [github.com/WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor) | ~4,100 | Multi-method editing (audio+motion+combined), exports Premiere XML, clip sequences, Nim rewrite |
| **Silero VAD** | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) | ~8,600 | ML-based VAD, 1.8MB ONNX, <1ms/chunk, 87.7% TPR vs 50% for energy-based at 5% FPR |
| **CrisperWhisper** | [github.com/nyrahealth/CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) | ~931 | Verbatim ASR with `[UH]`/`[UM]` markers, #1 on OpenASR Leaderboard |
| **jumpcutter** | [github.com/carykh/jumpcutter](https://github.com/carykh/jumpcutter) | ~5,000+ | Speed up silent parts instead of cutting |
| **unsilence** | [github.com/lagmoellertim/unsilence](https://github.com/lagmoellertim/unsilence) | ~592 | Differential speed: audible at 1x, silent at 6x |
| **pyannote.audio** | [github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio) | ~6,000+ | Speaker-aware VAD, overlapped speech detection, diarization |
| **PodcastFillers** | [podcastfillers.github.io](https://podcastfillers.github.io/) | — | 35K annotated fillers dataset, VAD→ASR→classifier pipeline |

### Key Insight
**None of the popular silence removal tools use ML-based VAD.** All rely on energy thresholds. Silero VAD would be a genuine competitive differentiator.

### Recommendations
1. **Replace energy-based detection with Silero VAD** — 2MB model, dramatic accuracy improvement
2. **Add CrisperWhisper for filler detection** — purpose-built for verbatim transcription with filler markers
3. **Add motion-based editing** — like auto-editor's `--edit motion` for screen recordings
4. **Add combined mode** — `audio AND motion` or `audio OR motion` logic
5. **Speaker-aware detection via pyannote** — don't cut when one speaker is silent but another speaks

---

## 2. Captions & Transcription

### Current Implementation
faster-whisper backend, styled caption overlay, SRT/VTT export, word-level timestamps, WhisperX alignment.

### Open Source Landscape

| Project | URL | Key Features |
|---------|-----|-------------|
| **faster-whisper** | [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 backend, 4x faster, int8/float16 quantization |
| **WhisperX** | [github.com/m-bain/whisperX](https://github.com/m-bain/whisperX) | Word-level alignment via wav2vec2, speaker diarization |
| **insanely-fast-whisper** | [github.com/Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper) | Flash Attention 2 + batched inference, 3-4x faster than faster-whisper |
| **distil-whisper** | [huggingface.co/distil-whisper](https://huggingface.co/distil-whisper) | 6x faster, 49% smaller, within 1% WER |
| **stable-ts** | [github.com/jianfch/stable-ts](https://github.com/jianfch/stable-ts) | Stabilized timestamps, reduces hallucination |
| **Moonshine** | [github.com/usefulsensors/moonshine](https://github.com/usefulsensors/moonshine) | Tiny efficient ASR for edge devices |
| **SeamlessM4T v2** | [github.com/facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication) | End-to-end speech-to-text translation, 100 languages |
| **NLLB-200** | Meta AI | 200 languages, multiple model sizes |
| **Opus-MT** | [github.com/Helsinki-NLP/Opus-MT](https://github.com/Helsinki-NLP/Opus-MT) | 1000+ language pairs, lightweight MarianMT models |
| **Argos Translate** | [github.com/argosopentech/argos-translate](https://github.com/argosopentech/argos-translate) | Fully offline translation |
| **pysubs2** | [github.com/tkarabela/pysubs2](https://github.com/tkarabela/pysubs2) | Programmatic ASS/SRT manipulation |

### Recommendations
1. **Add insanely-fast-whisper backend** — 10-15x speedup on powerful GPUs
2. **Add distil-whisper for draft mode** — quick preview while editing
3. **Integrate stable-ts timestamp stabilization** — reduce jitter
4. **Expand animated caption presets** — bounce, typewriter, glow, shake, gradient text
5. **Add real-time transcription preview** — stream results live

---

## 3. Audio Processing

### Current Implementation
noisereduce denoising, DeepFilterNet, Demucs stem separation, edge-tts, beat detection via librosa, Pedalboard effects.

### Open Source Landscape

| Project | URL | Key Features |
|---------|-----|-------------|
| **Resemble Enhance** | [github.com/resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance) | Denoise + enhance (bandwidth restoration to 44.1kHz) |
| **DeepFilterNet** | [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | Full-band 48kHz, ultra-low latency, pre-compiled binaries |
| **RNNoise** | [github.com/xiph/rnnoise](https://github.com/xiph/rnnoise) | Ultra-lightweight C library, runs on Raspberry Pi |
| **VoiceFixer** | [github.com/haoheliu/voicefixer](https://github.com/haoheliu/voicefixer) | Handles clipping, reverb, extreme upsampling (2kHz→44.1kHz) |
| **HTDemucs v4** | [github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs) | Hybrid Transformer, htdemucs_ft fine-tuned variant, 6-stem |
| **Bandit** | [github.com/kwatcharasupat/bandit](https://github.com/kwatcharasupat/bandit) | Cinematic dialogue/music/effects separation (video-specific) |
| **BS-RoFormer** | [github.com/lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer) | ByteDance SOTA, highest separation scores |
| **MDX-Net** | [github.com/kuielab/sdx23](https://github.com/kuielab/sdx23) | Best for clean vocal isolation |
| **Kokoro** | [github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro) | 82M params, <0.3s, Apache license, runs on CPU |
| **F5-TTS** | [github.com/SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) | Zero-shot voice cloning from 15s reference |
| **Bark** | [github.com/suno-ai/bark](https://github.com/suno-ai/bark) | Expressive speech with laughter, emotion, non-verbal |
| **Piper** | [github.com/rhasspy/piper](https://github.com/rhasspy/piper) | Fastest local TTS, runs on Raspberry Pi |
| **BeatNet** | [github.com/mjhydri/BeatNet](https://github.com/mjhydri/BeatNet) | Real-time streaming beat tracking |
| **madmom** | [github.com/CPJKU/madmom](https://github.com/CPJKU/madmom) | RNN-based, ranked #1 for beat tracking |
| **AudioCraft** | [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | AudioGen (text→SFX) + MusicGen (text→music) |
| **Stable Audio** | [github.com/Stability-AI/stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | 47s stereo, royalty-free training data |

### Recommendations
1. **Add Resemble Enhance "Speech Enhancement" mode** — restores bandwidth + removes distortion
2. **Add Bandit for cinematic separation** — dialogue/music/effects stems for video
3. **Add Kokoro as local TTS** — no internet, 82M params, Apache license
4. **Add F5-TTS voice cloning** — record 15s → clone voice for narration
5. **Add BeatNet real-time beat visualization** — live beat tracking during preview
6. **Use ffmpeg `sidechaincompress`** — simplify audio ducking implementation

---

## 4. Video AI & Effects

### Current Implementation
Real-ESRGAN upscaling, rembg background removal, RIFE interpolation, GFPGAN face restoration, basic video denoising, FFmpeg-based scene detection.

### Open Source Landscape

| Project | URL | Key Features |
|---------|-----|-------------|
| **SAM2** | [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Click-to-segment any object, streaming memory, tracks through occlusions |
| **ProPainter** | [github.com/sczhou/ProPainter](https://github.com/sczhou/ProPainter) | ICCV 2023, SOTA video inpainting, dual-domain propagation |
| **VideoPainter** | [github.com/TencentARC/VideoPainter](https://github.com/TencentARC/VideoPainter) | SIGGRAPH 2025, any-length video inpainting |
| **Video Depth Anything** | [github.com/DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) | CVPR 2025, consistent depth for super-long videos |
| **CodeFormer** | [github.com/sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | NeurIPS 2022, controllable fidelity slider, better than GFPGAN |
| **FaceFusion** | [github.com/facefusion/facefusion](https://github.com/facefusion/facefusion) | 26K+ stars, face swap + lip-sync + enhancement |
| **Robust Video Matting** | [github.com/PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | Temporally coherent matting, no green screen |
| **Video2X** | [github.com/k4yt3x/video2x](https://github.com/k4yt3x/video2x) | Multi-engine upscaling (ESRGAN, Waifu2x, Anime4K, SRMD) |
| **WatermarkRemover-AI** | [github.com/D-Ogi/WatermarkRemover-AI](https://github.com/D-Ogi/WatermarkRemover-AI) | Florence-2 detection + LaMa inpainting |
| **TransNetV2** | [github.com/soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) | Neural shot boundary detection, catches gradual transitions |
| **FastDVDnet** | [github.com/m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet) | Fast video denoising without optical flow |
| **Google AutoFlip** | Part of MediaPipe | Three-phase auto-reframe pipeline |
| **LUTforge-AI** | [github.com/veedy-dev/lutforge-ai](https://github.com/veedy-dev/lutforge-ai) | AI LUT generation via MKL color transfer |
| **AnimateDiff** | [github.com/guoyww/AnimateDiff](https://github.com/guoyww/AnimateDiff) | Video-to-video AI editing with ControlNet |
| **HunyuanVideo** | [github.com/Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo) | 13B params, beats Runway Gen-3 |

### Recommendations
1. **Add SAM2 + ProPainter** — click-to-select → track → remove any object
2. **Switch face restoration to CodeFormer** — controllable fidelity slider
3. **Add Robust Video Matting** — temporally consistent background removal
4. **Add Video Depth Anything** — depth-based parallax, bokeh simulation
5. **Upgrade watermark removal** — Florence-2 detection + ProPainter inpainting
6. **Add TransNetV2** — catch fades/dissolves that threshold detection misses

---

## 5. Editing Automation & Workflows

### Current Implementation
Workflow presets, custom workflow builder, multi-step job chains, shorts pipeline, NLP command.

### Open Source Landscape

| Project | URL | Key Features |
|---------|-----|-------------|
| **ClippedAI** | [github.com/Shaarav4795/ClippedAI](https://github.com/Shaarav4795/ClippedAI) | Open-source OpusClip, engagement scoring, auto 9:16, animated subtitles |
| **OpenShorts** | [github.com/mutonby/openshorts](https://github.com/mutonby/openshorts) | Gemini 2.0 Flash for viral detection, face tracking, 30+ language dubbing, social posting |
| **ShortGPT** | [github.com/RayVentura/ShortGPT](https://github.com/RayVentura/ShortGPT) | Separate engines for shorts, long-form, translation |
| **Frame** | [github.com/aregrid/frame](https://github.com/aregrid/frame) | Chat-driven video editing (Cursor-like pattern) |
| **LAVE** | [arxiv.org/abs/2402.10294](https://arxiv.org/abs/2402.10294) | LLM plan-and-execute agent for editing commands |
| **OpenTimelineIO** | [github.com/AcademySoftwareFoundation/OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) | Universal timeline interchange (FCP XML, AAF, EDL) |
| **Remotion** | [github.com/remotion-dev/remotion](https://github.com/remotion-dev/remotion) | Programmatic video from React templates |
| **editly** | [github.com/mifi/editly](https://github.com/mifi/editly) | Declarative video editing via JSON |
| **3D-Speaker** | [github.com/modelscope/3D-Speaker](https://github.com/modelscope/3D-Speaker) | Multimodal (audio+face) speaker diarization |
| **deepface** | [github.com/serengil/deepface](https://github.com/serengil/deepface) | Face emotion/age/gender analysis |

### Recommendations
1. **Study ClippedAI engagement scoring** — improve our shorts pipeline
2. **Add OTIO export** — universal timeline interchange
3. **Add emotion-based highlight detection** — deepface emotion curve → peaks = highlights
4. **Add multimodal diarization** — face + voice for multicam (3D-Speaker)
5. **Add conditional workflow steps** — "if loudness < -20 LUFS, normalize; else skip"

---

## 6. Extension Architecture

### Current Implementation
CEP panel (CSXS 9+) with ExtendScript, UXP panel (Premiere 25.6+), Python backend on localhost.

### Open Source Landscape

| Project | URL | Key Features |
|---------|-----|-------------|
| **Bolt CEP** | [github.com/hyperbrew/bolt-cep](https://github.com/hyperbrew/bolt-cep) | Svelte/React/Vue + Vite + TypeScript, type-safe evalTS() |
| **Bolt UXP** | [github.com/hyperbrew/bolt-uxp](https://github.com/hyperbrew/bolt-uxp) | Same for UXP, WebView UI (March 2026), CCX packaging |
| **PremiereRemote** | [github.com/sebinside/PremiereRemote](https://github.com/sebinside/PremiereRemote) | HTTP + WebSocket server inside Premiere |
| **Premiere Pro MCP** | [github.com/leancoderkavy/premiere-pro-mcp](https://github.com/leancoderkavy/premiere-pro-mcp) | 269 tools, file-based IPC, QE DOM coverage |
| **Pymiere** | [github.com/qmasingarbe/pymiere](https://github.com/qmasingarbe/pymiere) | Python-native Premiere API mirror |
| **OpenFX** | [github.com/AcademySoftwareFoundation/openfx](https://github.com/AcademySoftwareFoundation/openfx) | Universal visual effects plugin standard |

### Critical Timeline
- **CEP support ends ~September 2026** — ~6 months remaining
- **UXP is GA in Premiere 25.6** — official as of late 2025
- **Bolt UXP WebView UI** — March 2026, enables full browser rendering inside UXP
- Adobe calls migration a "reconstruction, not a migration"

### Recommendations
1. **Plan CEP → UXP migration immediately** — use Bolt UXP framework
2. **Adopt WebView UI pattern** — embed existing HTML/CSS inside UXP WebView
3. **Add WebSocket communication** — replace polling (PremiereRemote pattern)
4. **Add TypeScript** — prevents null-reference bugs
5. **Support DaVinci Resolve** — Python scripting API, no extension needed
