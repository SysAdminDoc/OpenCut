# OpenCut

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Premiere Pro](https://img.shields.io/badge/Premiere%20Pro-2022+-9999FF?logo=adobepremierepro&logoColor=white)
![Status](https://img.shields.io/badge/status-active-success)

> Open-source AI-powered video editing automation for Adobe Premiere Pro. Remove silences, generate styled captions, denoise audio, clone voices, apply cinematic color grades, upscale footage, batch-process entire folders, and export for any platform — all from a single panel inside Premiere.

![Screenshot](screenshot.png)

---

## Quick Start

### One-Line Install (Windows)

```powershell
irm https://raw.githubusercontent.com/SysAdminDoc/OpenCut/refs/heads/main/Install.ps1 | iex
```

### Manual Install

```bash
git clone https://github.com/opencut/opencut.git
cd opencut
pip install -e ".[all]"
```

Then install the Premiere Pro panel:

```bat
Install.bat
```

Restart Premiere Pro and open the panel via **Window > Extensions > OpenCut**.

### Prerequisites

- Windows 10/11
- Python 3.9 or newer
- Adobe Premiere Pro 2022 or newer
- FFmpeg on PATH ([download](https://ffmpeg.org/download.html))
- NVIDIA GPU recommended for AI features (CUDA 11.8+)

---

## Features

OpenCut is organized into 8 tabs with 35 sub-tabs, powered by 81 server endpoints and 16 core processing modules.

### Cut — Intelligent Editing

| Feature | Description |
|---------|-------------|
| Silence Detection | Find and remove dead air with configurable threshold (dB), minimum duration, and padding |
| Filler Word Removal | Detect "um", "uh", "like", "you know" and custom words using Whisper transcription |
| Full Analysis | Combined silence + filler detection in a single pass |

### Captions — Subtitle Generation

| Feature | Description |
|---------|-------------|
| Whisper Transcription | Word-level timestamps via faster-whisper, whisperx, or openai-whisper |
| Styled Captions | Animated caption overlays with 12 preset styles (Hormozi, Ali Abdaal, MrBeast, Neon Pop, Typewriter, etc.) |
| Subtitle Export | SRT, VTT, ASS formats with speaker labels |
| Transcript Export | Plain text, timestamped, SRT, VTT, ASS, JSON |
| Emoji Mapping | Auto-insert contextual emoji into caption text |
| Speaker Diarization | Multi-speaker identification via pyannote.audio |

### Audio Suite — Professional Audio Processing

| Feature | Description |
|---------|-------------|
| AI Denoise | DeepFilterNet-powered noise reduction (light / moderate / aggressive) |
| Vocal Isolation | Demucs stem separation — extract vocals, drums, bass, or other |
| Loudness Normalization | EBU R128 targeting with presets for broadcast (-23 LUFS), streaming (-14), podcast (-16), YouTube (-13) |
| Parametric EQ | 8 presets: Voice Clarity, Warm Vocals, Bass Boost, Podcast Standard, De-Ess, Telephone, and more |
| Beat Detection | BPM analysis and beat-marker placement on the Premiere timeline |
| Audio Effects | Reverb, echo, chorus, flanger, phaser, distortion, lo-fi, radio, underwater, telephone — 12 creative effects |
| Audio Ducking | Auto-duck music under speech with configurable threshold, reduction amount, and attack/release |
| Crossfade | 6 crossfade types between clips on the timeline |

### Voice Lab — AI Voice Synthesis

| Feature | Description |
|---------|-------------|
| Text-to-Speech | Qwen3-TTS with 14 preset voices (narrators, characters, accents) |
| Voice Cloning | Clone any voice from a reference audio sample + transcript |
| Voice Profiles | Save and manage reusable voice configurations |
| Voice Design | Describe a voice in natural language and generate it |
| Dialogue Replace | Replace spoken words in-place while preserving timing |

### Video Intelligence — AI-Powered Visual Processing

| Feature | Description |
|---------|-------------|
| Color Grading | 8 cinematic LUT presets (Cinematic, Vintage, Teal & Orange, Film Noir, Bleach Bypass, etc.) with intensity control |
| Chroma Key | GPU-accelerated green/blue screen removal with spill suppression and edge refinement |
| AI Background Removal | rembg-powered subject isolation with transparent or solid-color backgrounds |
| AI Slow Motion | RIFE frame interpolation — 2x, 4x, or 8x slow motion from standard footage |
| AI Upscaling | Real-ESRGAN super-resolution — upscale to 2x or 4x (720p to 4K) |
| Auto-Reframe | Intelligent crop for aspect ratio conversion (16:9, 9:16, 4:5, 1:1, 4:3, 21:9) with face detection |
| Scene Detection | Content-aware scene boundary detection with configurable threshold |
| Speed Ramping | 6 presets: Smooth Slow, Impact Hit, Timelapse, Pulse, Dramatic Entry, Rewind |

### Export & Publish — Multi-Platform Delivery

| Feature | Description |
|---------|-------------|
| Platform Presets | One-click export for YouTube (1080p/4K/Shorts), TikTok, Instagram (Reels/Feed/Square), Twitter/X, LinkedIn, Podcast |
| Custom Render | Full codec control — H.264, H.265, VP9, ProRes with CRF or bitrate mode |
| Thumbnail Extraction | Auto best-frame, specific timestamp, multi-frame, or 3x3 contact sheet |
| Subtitle Burn-In | Hardcode SRT/VTT/ASS subtitles with font, color, position, and outline styling |
| Watermark | Text or image overlay with position, opacity, and scale controls |
| GIF Export | Two-pass palette-optimized GIF with bayer dithering |
| Audio Extraction | MP3, AAC, FLAC, WAV, Opus with optional loudness normalization |

### Batch & Workflow — Automation at Scale

| Feature | Description |
|---------|-------------|
| Batch Queue | Process entire folders through any workflow with progress tracking and cancellation |
| Workflow Chains | Chain multiple operations sequentially — output of each step feeds the next |
| 6 Workflow Presets | YouTube Ready, Podcast Clean, Social Vertical, Archive Master, Quick GIF, Full Clean & Grade |
| Watch Folder | Monitor a directory and auto-process new files as they appear |
| Media Inspector | Full ffprobe analysis — codecs, streams, color metadata, bitrates, chapters |
| Folder Scanner | Recursive media file discovery with size and format metadata |

### Settings

| Feature | Description |
|---------|-------------|
| Whisper Backend | Choose between faster-whisper, whisperx, or openai-whisper; configure model size |
| GPU Status | CUDA availability and VRAM detection |
| Server Health | Backend connection status and version info |

---

## How It Works

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Adobe Premiere Pro                            │
│                                                                      │
│  ┌────────────────────────────┐     ┌─────────────────────────────┐  │
│  │     CEP Panel (HTML/JS)    │────>│    ExtendScript (JSX)       │  │
│  │                            │     │                             │  │
│  │  8 tabs, 35 sub-tabs       │     │  Timeline manipulation      │  │
│  │  Dark theme UI              │     │  Marker placement           │  │
│  │  Real-time status updates   │<────│  Clip import/export         │  │
│  └─────────────┬──────────────┘     └─────────────────────────────┘  │
│                │ HTTP (localhost:5679)                                │
└────────────────┼─────────────────────────────────────────────────────┘
                 │
  ┌──────────────▼──────────────┐
  │    Flask Backend Server      │
  │    81 REST API endpoints     │
  │                              │
  │  ┌────────────────────────┐  │
  │  │   16 Core Modules      │  │
  │  │                        │  │
  │  │  audio.py        190L  │  │     ┌─────────────────────────┐
  │  │  audio_suite.py 1408L  │──────>│  AI Models               │
  │  │  batch.py        681L  │  │     │                         │
  │  │  captions.py     367L  │  │     │  Whisper (speech-to-text)│
  │  │  color.py        334L  │  │     │  DeepFilterNet (denoise) │
  │  │  diarize.py      218L  │  │     │  Demucs (stem split)     │
  │  │  export.py      1072L  │  │     │  Qwen3-TTS (voice synth) │
  │  │  fillers.py      323L  │  │     │  rembg (bg removal)      │
  │  │  reframe.py      545L  │  │     │  RIFE (frame interp)     │
  │  │  scene_detect.py 308L  │  │     │  Real-ESRGAN (upscale)   │
  │  │  silence.py      260L  │  │     │  pyannote (diarization)  │
  │  │  speed_ramp.py   695L  │  │     └─────────────────────────┘
  │  │  styled_captions 930L  │  │
  │  │  video_fx.py    1211L  │  │     ┌─────────────────────────┐
  │  │  voice_lab.py    874L  │──────>│  FFmpeg                  │
  │  │  zoom.py         111L  │  │     │                         │
  │  └────────────────────────┘  │     │  Encoding / Decoding     │
  │                              │     │  Filtering / Muxing      │
  │  Async job queue + progress  │     │  Analysis / Probing      │
  │  Background thread workers   │     └─────────────────────────┘
  └──────────────────────────────┘
```

The panel sends HTTP requests to a local Flask server running on port 5679. The server delegates work to 16 specialized core modules which call AI models and FFmpeg for processing. Results are sent back and applied to the Premiere Pro timeline via ExtendScript (JSX). Long-running operations use an async job queue with real-time progress streaming.

---

## Platform Export Presets

| Platform | Resolution | Aspect | Codec | Audio | Max Duration |
|----------|-----------|--------|-------|-------|-------------|
| YouTube 1080p | 1920x1080 | 16:9 | H.264 CRF 18 | AAC 192k | None |
| YouTube 4K | 3840x2160 | 16:9 | H.264 CRF 16 | AAC 256k | None |
| YouTube Shorts | 1080x1920 | 9:16 | H.264 CRF 20 | AAC 128k | 60s |
| TikTok | 1080x1920 | 9:16 | H.264 CRF 20 | AAC 128k | 10m |
| Instagram Reels | 1080x1920 | 9:16 | H.264 CRF 20 | AAC 128k | 90s |
| Instagram Feed | 1080x1350 | 4:5 | H.264 CRF 20 | AAC 128k | 60s |
| Instagram Square | 1080x1080 | 1:1 | H.264 CRF 20 | AAC 128k | 60s |
| Twitter/X | 1280x720 | 16:9 | H.264 CRF 22 | AAC 128k | 2m 20s |
| LinkedIn | 1920x1080 | 16:9 | H.264 CRF 20 | AAC 192k | 10m |
| Podcast Video | 1920x1080 | 16:9 | H.264 CRF 23 | AAC 192k | None |
| Podcast Audio | Audio-only | — | — | MP3 192k | None |

---

## Workflow Presets

| Preset | Steps | Description |
|--------|-------|-------------|
| YouTube Ready | Denoise > Normalize (-14 LUFS) > YouTube 1080p Export | Quick publish pipeline |
| Podcast Clean | Denoise > Silence Remove > Normalize (-16 LUFS) > MP3 Export | Podcast post-production |
| Social Vertical | Reframe 9:16 > Cinematic Color > TikTok Export | Long-form to short-form conversion |
| Archive Master | Light Denoise > ProRes/FLAC Export | Lossless archival |
| Quick GIF | Auto Thumbnail > 5s GIF @ 480px | Preview asset generation |
| Full Clean & Grade | Denoise > Voice EQ > Normalize > Cinematic Color > YouTube 1080p | Complete production pipeline |

---

## Configuration

### Optional Dependencies

OpenCut installs only core dependencies by default. AI features are installed on-demand via the Settings tab or manually:

```bash
# Captions (pick one)
pip install "opencut[captions]"       # faster-whisper (recommended)
pip install "opencut[captions-full]"  # whisperx (best word-level timestamps)

# Speaker diarization
pip install "opencut[diarize]"

# AI audio (denoise + stem separation)
pip install "opencut[audio-ai]"

# AI video (background removal, upscaling)
pip install "opencut[video-ai]"

# Everything
pip install "opencut[all]"
```

### Voice Lab (Qwen3-TTS)

The Voice Lab auto-downloads the Qwen3-TTS model (~2 GB) on first use. Requires an NVIDIA GPU with at least 4 GB VRAM.

### Server Configuration

The backend defaults to `localhost:5679`. Override with:

```bash
opencut --host 0.0.0.0 --port 8080
```

Or run directly:

```bash
python -m opencut --port 5679
```

---

## Project Structure

```
opencut/
├── opencut/
│   ├── __init__.py              # Package metadata (version)
│   ├── __main__.py              # Entry point
│   ├── cli.py                   # CLI interface (click)
│   ├── server.py                # Flask server, 81 endpoints (4,602 lines)
│   ├── core/
│   │   ├── audio.py             # Basic audio analysis
│   │   ├── audio_suite.py       # Denoise, EQ, normalize, isolate, effects, ducking
│   │   ├── batch.py             # Batch queue, workflow chains, watch folder, inspector
│   │   ├── captions.py          # Whisper transcription + word timestamps
│   │   ├── color.py             # LUT generation + color grading
│   │   ├── diarize.py           # Speaker diarization (pyannote)
│   │   ├── export.py            # Platform presets, render, thumbnail, GIF, watermark
│   │   ├── fillers.py           # Filler word detection
│   │   ├── reframe.py           # Aspect ratio conversion + face detection
│   │   ├── scene_detect.py      # Scene boundary detection
│   │   ├── silence.py           # Silence detection
│   │   ├── speed_ramp.py        # Speed ramping presets
│   │   ├── styled_captions.py   # 12 animated caption styles
│   │   ├── video_fx.py          # Chroma key, bg remove, slow-mo, upscale
│   │   ├── voice_lab.py         # TTS, cloning, voice design, dialogue replace
│   │   └── zoom.py              # Ken Burns / zoom effects
│   ├── export/
│   │   ├── premiere.py          # Premiere Pro XML generation
│   │   └── srt.py               # SRT file handling
│   └── utils/
│       ├── config.py            # Configuration management
│       └── media.py             # Media file utilities
├── extension/
│   └── com.opencut.panel/
│       ├── CSXS/manifest.xml    # CEP extension manifest
│       ├── client/
│       │   ├── index.html       # Panel UI (1,437 lines)
│       │   ├── main.js          # Panel logic (2,170 lines)
│       │   ├── style.css        # Dark theme styles (762 lines)
│       │   └── CSInterface.js   # Adobe CEP library
│       └── host/
│           └── index.jsx        # ExtendScript bridge to Premiere
├── build/
│   ├── build.ps1                # Windows build script
│   ├── installer.iss            # Inno Setup installer config
│   └── opencut.spec             # PyInstaller spec
├── Install.bat                  # One-click installer
├── Install.ps1                  # PowerShell installer
├── Uninstall.bat                # Clean uninstall
├── pyproject.toml               # Package config
├── requirements.txt             # Core dependencies
└── tests/
    └── test_core.py             # Test suite
```

---

## API Reference

The server exposes 81 REST endpoints organized by feature area. All long-running operations return a `job_id` for async status polling.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health check |
| GET | `/info` | Server version, capabilities, GPU status |
| GET | `/status/<job_id>` | Poll async job progress |
| POST | `/cancel/<job_id>` | Cancel a running job |
| GET | `/jobs` | List all jobs |
| GET | `/stream/<job_id>` | SSE progress stream |

### Cut & Analysis (6 endpoints)

`/silence`, `/fillers`, `/full`, `/captions`, `/transcript`, `/transcript/export`

### Audio Suite (15 endpoints)

`/audio/denoise`, `/audio/isolate`, `/audio/normalize`, `/audio/measure`, `/audio/beats`, `/audio/effects`, `/audio/effects/apply`, `/audio/eq`, `/audio/eq-presets`, `/audio/duck`, `/audio/ducking-keyframes`, `/audio/crossfade`, `/audio/crossfade-types`, `/audio/stems`, `/audio/loudness-presets`

### Voice Lab (10 endpoints)

`/voice/check`, `/voice/install`, `/voice/generate`, `/voice/replace`, `/voice/speakers`, `/voice/profiles` (GET/POST/DELETE), `/voice/unload`, `/voice/preview/<path>`

### Video Intelligence (15 endpoints)

`/video/scenes`, `/video/speed-presets`, `/video/speed-ramp`, `/video/lut`, `/video/lut-validate`, `/video/chroma-key`, `/video/bg-check`, `/video/bg-install`, `/video/bg-remove`, `/video/slowmo-check`, `/video/slowmo-install`, `/video/slowmo`, `/video/upscale-check`, `/video/upscale-install`, `/video/upscale`, `/video/reframe-presets`, `/video/reframe`, `/video/fx-capabilities`

### Export & Publish (8 endpoints)

`/export/platform-presets`, `/export/render`, `/export/thumbnail`, `/export/burn-subs`, `/export/watermark`, `/export/gif`, `/export/audio-extract`

### Batch & Workflow (12 endpoints)

`/batch/capabilities`, `/batch/inspect`, `/batch/scan`, `/batch/workflow-presets`, `/batch/start`, `/batch/status/<id>`, `/batch/cancel/<id>`, `/batch/jobs`, `/batch/watch/start`, `/batch/watch/stop/<id>`, `/batch/watch/status/<id>`, `/batch/watch/list`

---

## FAQ / Troubleshooting

**The panel shows "Cannot connect to server"**
Make sure the backend is running. Open a terminal and run `python -m opencut`. The panel connects to `localhost:5679` by default.

**Whisper / captions aren't working**
Install a Whisper backend: `pip install faster-whisper`. You can also install from the Settings tab inside the panel.

**AI features are slow**
Most AI operations (denoise, upscale, slow-motion, TTS) benefit significantly from an NVIDIA GPU. CPU fallback works but is 5-20x slower depending on the operation.

**Voice Lab says "model not found"**
Qwen3-TTS downloads automatically on first use (~2 GB). Ensure you have internet access and sufficient disk space.

**Background removal / upscaling won't start**
These features require separate model downloads. Use the install buttons in the Video tab to download rembg, RIFE, or Real-ESRGAN models.

**FFmpeg not found**
Install FFmpeg and ensure it's on your system PATH. Download from [ffmpeg.org](https://ffmpeg.org/download.html) or install via `choco install ffmpeg` / `winget install ffmpeg`.

**Panel doesn't appear in Premiere Pro**
Run `Install.bat` as Administrator, then restart Premiere Pro. The panel should appear under **Window > Extensions > OpenCut**. If not, enable unsigned extensions by setting `PlayerDebugMode=1` in the CEP debug registry key (the installer does this automatically).

---

## Development

```bash
# Clone and install in dev mode
git clone https://github.com/opencut/opencut.git
cd opencut
pip install -e ".[dev,all]"

# Run tests
pytest tests/

# Start server in debug mode
python -m opencut --debug

# Lint
ruff check opencut/
```

### Build Installer (Windows)

```powershell
cd build
.\build.ps1
```

This creates a standalone executable via PyInstaller and an Inno Setup installer.

---

## Stats

| Metric | Value |
|--------|-------|
| Total source lines | 18,499 |
| Server endpoints | 81 |
| Core modules | 16 |
| Panel tabs | 8 |
| Panel sub-tabs | 35 |
| Action buttons | 36 |
| AI models supported | 8 |
| Platform export presets | 11 |
| Workflow presets | 6 |
| Audio effects | 12 |
| Caption styles | 12 |
| Color grade presets | 8 |

---

## Contributing

Issues and pull requests are welcome. Please open an issue first for major changes.

Areas where contributions would be especially valuable: DaVinci Resolve support, additional AI models, localization, and cross-platform testing.

---

## License

[MIT](LICENSE)

---

Built with FFmpeg, Whisper, PyTorch, and a lot of Python.
