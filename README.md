# OpenCut

![Version](https://img.shields.io/badge/version-1.0.0--beta-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Premiere Pro](https://img.shields.io/badge/Premiere%20Pro-2023+-9999FF?logo=adobepremierepro&logoColor=white)

> A free, open-source Premiere Pro extension that brings AI-powered video editing automation, caption generation, audio processing, and visual effects -- all running locally on your machine. This project aims to replace all of the leading paid Premiere extensions and products on the market.

## Quick Start

### Prerequisites

- **Adobe Premiere Pro** 2023 or later
- **Python 3.9+** ([python.org](https://python.org/downloads))
- **FFmpeg** ([ffmpeg.org](https://ffmpeg.org/download.html)) -- must be on PATH
- **Windows 10/11** (macOS/Linux: server works, CEP panel is Windows-only)

### Installation

```
git clone https://github.com/opencut/opencut.git
cd opencut
```

**Option A -- Double-click installer (recommended):**

Run `Install.bat` as Administrator. It handles everything: FFmpeg check, Python deps, CEP extension deployment, registry keys, and launcher creation.

**Option B -- Manual setup:**

```bash
pip install -e ".[all]"
python -m opencut.server
```

Then copy `extension/com.opencut.panel` to your Premiere Pro CEP extensions folder and enable unsigned extensions via the registry key `PlayerDebugMode = 1`.

**Option C -- Cross-platform Python installer:**

```bash
python install.py
```

### Launch

1. Start the backend: run `Start-OpenCut.bat` or `python -m opencut.server`
2. In Premiere Pro: **Window > Extensions > OpenCut**
3. Select a clip in your timeline and use the panel

## Features

OpenCut v1.0.0 includes **34 processing modules**, **116 API routes**, and **6 panel tabs** covering every major video editing automation task.

### Captions & Transcription

| Feature | Description | Engine |
|---------|-------------|--------|
| Transcription | Speech-to-text with word-level timestamps | WhisperX / faster-whisper |
| 19 Caption Styles | YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, etc. | Pillow renderer |
| Animated Captions | CapCut-style word-by-word pop, fade, bounce, glow, highlight | Pillow + OpenCV |
| Caption Burn-in | Hard-burn styled captions directly into video | FFmpeg drawtext |
| Speaker Diarization | Identify who's speaking for podcasts | pyannote.audio |
| Filler Word Detection | Auto-detect and remove um, uh, like, you know | WhisperX |
| Translation | Translate captions to 50+ languages | deep-translator |

### Audio Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| Stem Separation | Isolate vocals, drums, bass, other | Demucs (htdemucs) |
| Noise Reduction | AI noise removal + spectral gating | noisereduce / RNNoise |
| Normalization | Loudness targeting (LUFS) | FFmpeg loudnorm |
| Beat Detection | BPM analysis and beat markers | librosa |
| Audio Ducking | Auto-lower music under dialogue | FFmpeg sidechaincompress |
| Pro FX Chain | Compressor, EQ, de-esser, limiter, reverb, stereo width | Pedalboard (Spotify) |
| TTS Voice Generation | Text-to-speech with 100+ voices | Edge-TTS / F5-TTS |
| SFX Generator | Procedural tones, sweeps, impacts | NumPy synthesis |
| AI Music Generation | Text-to-music from prompts | MusicGen (AudioCraft) |

### Video Effects & Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| Stabilization | Deshake / vidstab | FFmpeg |
| Chromakey | Green/blue screen removal + compositing | OpenCV HSV |
| Picture-in-Picture | Overlay PiP with position/scale controls | FFmpeg overlay |
| Blend Modes | 14 blend modes (multiply, screen, overlay, etc.) | FFmpeg blend |
| 34 Transitions | Crossfade, wipe, slide, circle, pixelize, radial, zoom | FFmpeg xfade |
| Particle Effects | Confetti, sparkles, snow, rain, fire, smoke, bubbles | Pillow renderer |
| Title Cards | Animated titles with 6 presets (fade, slide, typewriter, lower third) | FFmpeg drawtext |
| Speed Ramping | Time remapping with ease-in/out curves | FFmpeg setpts |
| Scene Detection | Auto-detect cuts and scene changes | PySceneDetect |
| Film Grain | Adjustable noise overlay | FFmpeg noise |
| Letterbox | Cinema aspect ratio bars | FFmpeg pad |
| Vignette | Adjustable corner darkening | FFmpeg vignette |
| LUT Library | 15 built-in cinematic LUTs + external .cube/.3dl support | FFmpeg lut3d |

### AI & ML Tools

| Feature | Description | Engine |
|---------|-------------|--------|
| AI Upscaling | 3 tiers: Lanczos, Real-ESRGAN, Video2x | FFmpeg / Real-ESRGAN |
| Background Removal | Remove video backgrounds | rembg (U2-Net) |
| Face Enhancement | Restore/upscale faces | GFPGAN |
| Face Swap | Replace faces with reference image | InsightFace |
| Style Transfer | Neural artistic style transfer | PyTorch models |
| Object/Watermark Removal | Remove logos via delogo or LaMA AI inpainting | FFmpeg / LaMA |
| Color Correction | Exposure, contrast, saturation, temperature, lift/gamma/gain | FFmpeg eq/colorbalance |
| Color Space Conversion | sRGB, Rec.709, Rec.2020, DCI-P3 | FFmpeg colorspace |
| Auto Thumbnails | AI-scored frame extraction for thumbnails | Florence-2 / OpenCV |

### Export & Batch

| Feature | Description | Engine |
|---------|-------------|--------|
| Platform Presets | YouTube, TikTok, Instagram, Twitter, LinkedIn, Podcast | FFmpeg encode |
| Batch Processing | Process multiple clips in parallel | ThreadPool |
| Transcript Export | SRT, VTT, plain text, timestamped | Built-in |

## Architecture

```
+-----------------------+     HTTP/JSON      +-----------------------+
|   Premiere Pro CEP    | <================> |   OpenCut Server      |
|   Panel (HTML/JS)     |   localhost:5679   |   (Python/Flask)      |
|                       |                    |                       |
|  6 tabs, 24 sub-tabs  |                    |  34 core modules      |
|  Dark theme, 6 color  |                    |  116 API routes       |
|  schemes              |                    |  Async job queue      |
+-----------------------+                    +-----------+-----------+
                                                         |
                                             +-----------+-----------+
                                             |           |           |
                                          +--+--+   +----+---+  +---+----+
                                          |FFmpeg|   |Whisper |  |PyTorch |
                                          |OpenCV|   |Demucs  |  |Models  |
                                          +-----+   +--------+  +--------+
```

The server runs entirely on your local machine. No data leaves your computer. No API keys needed for core features.

## Dependency Tiers

OpenCut uses a tiered dependency model. Only the core tier is required -- everything else is optional and auto-detected at runtime.

### Core (required, ~5MB)

```
flask, flask-cors, click, rich
```

### Standard (recommended, ~200MB)

```
pip install opencut[standard]
```

Adds: `faster-whisper`, `opencv-python-headless`, `Pillow`, `numpy`, `librosa`, `pydub`, `noisereduce`, `deep-translator`

### Full (everything, ~2-5GB depending on GPU)

```
pip install opencut[all]
```

Adds all standard deps plus: `whisperx`, `demucs`, `pyannote.audio`, `pedalboard`, `edge-tts`, `realesrgan`, `rembg`, `gfpgan`, `insightface`, `audiocraft`, `simple-lama-inpainting`

### GPU Acceleration

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCUT_PORT` | `5679` | Server port |
| `OPENCUT_HOST` | `127.0.0.1` | Bind address |
| `OPENCUT_OUTPUT_DIR` | Source file dir | Default output directory |
| `WHISPER_MODELS_DIR` | `~/.cache` | Whisper model cache |

### Extension Themes

6 dark themes switchable from the gear icon: Cyberpunk, Midnight OLED, Catppuccin Mocha, GitHub Dark, Stealth, Ember.

## FAQ / Troubleshooting

**Q: The panel says "Server offline"**
A: Make sure the backend is running. Check that port 5679 is not blocked.

**Q: Transcription is slow**
A: Install CUDA-enabled PyTorch and use `faster-whisper` with a GPU. The `tiny` model is fastest.

**Q: I get "module not found" errors for AI features**
A: Most AI features are optional. Install them individually or use `pip install opencut[all]`.

**Q: Can I use this without Premiere Pro?**
A: Yes. The server runs standalone with a REST API. Call any route with curl or build your own frontend.

**Q: Does this send data to the cloud?**
A: No. Everything runs locally. No telemetry, no API keys for core features. Edge-TTS requires internet.

## Contributing

Issues and PRs welcome. The codebase:

```
opencut/
  core/           # 34 processing modules (one per feature area)
  server.py       # Flask server with 116 routes
  cli.py          # CLI entry point
extension/
  com.opencut.panel/
    client/       # CEP panel (index.html, main.js, style.css)
    CSXS/         # Extension manifest
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with FFmpeg, Whisper, Demucs, PyTorch, and many other open-source projects.
