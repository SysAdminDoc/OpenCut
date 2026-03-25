# OpenCut

![Version](https://img.shields.io/badge/version-1.7.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-0078D4)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Premiere Pro](https://img.shields.io/badge/Premiere%20Pro-2023+-9999FF?logo=adobepremierepro&logoColor=white)
![Routes](https://img.shields.io/badge/API%20Routes-142-orange)

> A free, open-source Premiere Pro extension that brings AI-powered video editing automation, caption generation, audio processing, and visual effects -- all running locally on your machine. No subscriptions, no cloud, no API keys. Replaces the need for paid Premiere extensions.

## Quick Start

### Prerequisites

- **Adobe Premiere Pro** 2023 or later
- **Windows 10/11** (macOS/Linux: server works, CEP panel is Windows-only)

### Installation

**Option A -- Installer exe (recommended):**

Download `OpenCut-Setup-1.7.2.exe` from [Releases](https://github.com/SysAdminDoc/OpenCut/releases) and run it. The installer handles everything: server exe, FFmpeg, CEP extension deployment, registry keys, desktop shortcut, and optional Whisper model download. **No Python or FFmpeg needed.**

**Option B -- From source (requires Python 3.9+ and [FFmpeg](https://ffmpeg.org/download.html) on PATH):**

```bash
git clone https://github.com/SysAdminDoc/OpenCut.git
cd OpenCut
pip install -e ".[all]"
python -m opencut.server
```

Then copy `extension/com.opencut.panel` to your Premiere Pro CEP extensions folder and enable unsigned extensions via the registry key `PlayerDebugMode = 1`.

**Option C -- Double-click installer from source (requires Python 3.9+):**

Clone the repo and run `Install.bat` as Administrator. It handles FFmpeg check, Python deps, CEP extension deployment, registry keys, and launcher creation.

### Launch

1. Start the backend: run `Start-OpenCut.bat`, the exe, or `python -m opencut.server`
2. In Premiere Pro: **Window > Extensions > OpenCut**
3. Select a clip and use the panel

---

## Features

OpenCut v1.5.0 includes **142 API routes**, **7 panel tabs** with **50+ sub-tabs**, and covers every major video editing automation task — now with ~20 new capabilities including Timeline Integration, AI-powered NLP commands, footage search, post-production deliverables, and a UXP panel for Premiere Pro 25.6+.

### Cut & Clean

| Feature | Description | Engine |
|---------|-------------|--------|
| Silence Removal | Detect and remove silent segments with adjustable threshold | FFmpeg / VAD |
| Filler Word Detection | Auto-detect and cut um, uh, like, you know, so, actually | WhisperX |
| Waveform Preview | Visual waveform with draggable threshold line synced to slider | FFmpeg PCM + Canvas |
| Trim Tool | Set in/out points to extract a clip portion (stream copy or re-encode) | FFmpeg |
| Full Pipeline | Combined silence + filler removal in one pass | Multi-stage |
| Repeated Take Detection | Automatically detects when speakers restart a sentence or fumble a phrase. Removes the earlier attempt using transcript similarity analysis (Jaccard overlap) | WhisperX |

### Captions & Transcription

| Feature | Description | Engine |
|---------|-------------|--------|
| Transcription | Speech-to-text with word-level timestamps | WhisperX / faster-whisper |
| 19 Caption Styles | YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, etc. | Pillow renderer |
| Animated Captions | CapCut-style word-by-word pop, fade, bounce, glow, highlight (7 presets) | Pillow + OpenCV |
| Caption Burn-in | Hard-burn styled captions directly into video | FFmpeg drawtext |
| Speaker Diarization | Identify who's speaking for podcasts | pyannote.audio |
| Filler Word Detection | Auto-detect and remove filler words | WhisperX |
| Translation | Translate captions to 50+ languages | deep-translator |
| Karaoke Mode | Word-by-word highlight sync for lyrics/captions | Pillow renderer |
| Transcript Editor | Edit segments in-panel with undo/redo and search | Built-in |
| YouTube Chapter Generation | Analyze transcript with local (Ollama) or cloud (OpenAI, Anthropic) LLM to identify topic changes. Outputs ready-to-paste chapter timestamps for YouTube descriptions | Ollama / OpenAI / Anthropic |
| SRT to Native Captions | Import any SRT file as a native Premiere Pro caption track | ExtendScript |

### Audio Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| Stem Separation | Isolate vocals, drums, bass, other | Demucs (htdemucs) |
| Noise Reduction | AI noise removal + spectral gating | noisereduce / RNNoise |
| Normalization | Loudness targeting (LUFS) with waveform preview | FFmpeg loudnorm |
| Beat Detection | BPM analysis and beat markers | librosa |
| Audio Ducking | Auto-lower music under dialogue | FFmpeg sidechaincompress |
| Pro FX Chain | Compressor, EQ, de-esser, limiter, reverb, stereo width | Pedalboard (Spotify) |
| TTS Voice Generation | Text-to-speech with 100+ voices | Edge-TTS / F5-TTS |
| SFX Generator | Procedural tones, sweeps, impacts, noise | NumPy synthesis |
| AI Music Generation | Text-to-music from prompts | MusicGen (AudioCraft) |
| Audio Preview Player | Floating player to preview generated audio before importing | Built-in |
| Loudness Match | FFmpeg two-pass LUFS normalization across multiple clips. Ensures consistent audio levels throughout a sequence | FFmpeg loudnorm |
| Beat Markers | Export detected beat timestamps as Premiere Pro sequence markers for manual snap-to-beat editing | librosa + ExtendScript |

### Video Effects & Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| Stabilization | Deshake / vidstab with smoothing and zoom controls | FFmpeg |
| Chromakey | Green/blue/red screen removal + spill suppression | OpenCV HSV |
| Picture-in-Picture | Overlay PiP with position/scale controls | FFmpeg overlay |
| Blend Modes | 14 modes (multiply, screen, overlay, etc.) | FFmpeg blend |
| 34 Transitions | Crossfade, wipe, slide, circle, pixelize, radial, zoom | FFmpeg xfade |
| Particle Effects | 7 presets: confetti, sparkles, snow, rain, fire, smoke, bubbles | Pillow renderer |
| Animated Titles | 6 presets: fade, slide, typewriter, lower third, countdown, kinetic | FFmpeg drawtext |
| Speed Ramping | Time remapping with ease-in/out curves, reverse, slow-mo | FFmpeg setpts |
| Scene Detection | Auto-detect cuts and scene changes | PySceneDetect |
| Film Grain / Vignette | Adjustable film look overlays | FFmpeg noise/vignette |
| Letterbox | Cinematic aspect ratio bars (2.39:1, 2:1, 1.85:1, 4:3, 1:1) | FFmpeg pad |
| LUT Library | 15 built-in cinematic LUTs + external .cube/.3dl support | FFmpeg lut3d |
| Color Correction | Exposure, contrast, saturation, temperature, shadows, highlights | FFmpeg eq/colorbalance |
| Color Space Conversion | sRGB, Rec.709, Rec.2020, DCI-P3 | FFmpeg colorspace |
| Video Reframe | Resize/crop for TikTok, Shorts, Reels, Instagram, or custom dims | FFmpeg scale/crop/pad |
| Auto-Crop Detect | Smart reframe anchor using cropdetect for talking-head content | FFmpeg cropdetect |
| Merge / Concatenate | Join multiple clips (fast stream copy or re-encoded) | FFmpeg concat |
| Side-by-Side Preview | Before/after frame comparison modal for effects | FFmpeg + base64 |
| Watermark Removal | Remove logos via delogo or LaMA AI inpainting | FFmpeg / LaMA |
| Color Match | Match the color profile of one clip to a reference clip using YCbCr histogram matching | OpenCV |

### AI & ML Tools

| Feature | Description | Engine |
|---------|-------------|--------|
| AI Upscaling | 3 tiers: Lanczos (fast), Real-ESRGAN (balanced), Video2x (premium) | FFmpeg / Real-ESRGAN |
| Background Removal | Remove video backgrounds | rembg (U2-Net) |
| Face Enhancement | Restore/upscale faces | GFPGAN |
| Face Swap | Replace faces with reference image | InsightFace |
| Style Transfer | Neural artistic style transfer | PyTorch models |
| Auto Thumbnails | AI-scored frame extraction for thumbnails | OpenCV scoring |
| Auto Zoom Keyframes | Face-detected push-in zoom for talking-head content. Returns keyframes for the Premiere Pro Motion effect | OpenCV + ExtendScript |
| Multicam Auto-Switching | Speaker diarization → camera cut list. Maps speakers to track indices and generates multicam edits | pyannote.audio + ExtendScript |

### Export & Batch

| Feature | Description | Engine |
|---------|-------------|--------|
| Platform Presets | YouTube, TikTok, Instagram, Twitter/X, LinkedIn, Podcast, Snapchat Story, Facebook Reel, Facebook Post, YouTube Long Form, Pinterest Video, Podcast MP3 | FFmpeg encode |
| Social Quick Export | One-click export optimized per platform's format and limits | FFmpeg |
| Batch Processing | Process multiple clips in parallel | ThreadPool |
| Transcript Export | SRT, VTT, ASS, plain text, timestamped | Built-in |
| Auto Thumbnails | AI-scored thumbnail candidates from video | OpenCV |

### Panel UX

| Feature | Description |
|---------|-------------|
| Command Palette | Ctrl+K fuzzy search across all 28+ operations with keyboard navigation |
| Clip Preview | Thumbnail + duration/resolution metadata when selecting a clip |
| Recent Clips | Dropdown of last 10 used clips, persisted across sessions |
| Favorites Bar | Pin frequently-used operations as quick-access chips |
| Waveform Preview | Visual waveform on Silence, Denoise, and Normalize tabs |
| First-Run Wizard | 3-step onboarding overlay for new users |
| Output File Browser | Browse recent outputs with Import-to-Premiere button |
| Custom Workflows | Chain operations into reusable named workflows |
| Collapsible Cards | Click headers to collapse/expand dense form sections |
| Right-Click Menu | Quick-action context menu on clip selector |
| Parameter Tooltips | Hover info on every range slider |
| Job Time Estimates | Estimated processing time based on historical data |
| Per-Operation Presets | Save/load settings per operation |
| Settings Import/Export | Bundle all settings as JSON for backup or sharing |
| Server Health Monitor | Auto-reconnect banner when backend goes offline |
| 6 Dark Themes | Cyberpunk, Midnight OLED, Catppuccin Mocha, GitHub Dark, Stealth, Ember |
| Premiere Theme Sync | Auto-detect Premiere's UI brightness |
| Toast Notifications | Non-intrusive slide-in alerts for job completion |

### Backend Infrastructure

| Feature | Description |
|---------|-------------|
| Async Job System | Background processing with SSE streaming and polling fallback |
| Job Queue | Sequential job processing with queue management |
| FFmpeg Progress | Real-time percentage from `-progress pipe:1` parsing |
| Cancel + Kill | Job cancellation terminates running FFmpeg subprocesses |
| Output Deduplication | Auto-increment suffix prevents overwriting previous outputs |
| Temp Cleanup | Stale preview files cleaned up on server startup |
| Dependency Dashboard | Grid view of all 24 optional deps with install status |
| GPU Auto-Detection | Recommend optimal settings based on detected GPU VRAM |
| AI Model Manager | View/delete downloaded models to free disk space |

---

## Timeline Integration

OpenCut v1.5.0 adds direct write-back to the active Premiere Pro sequence via ExtendScript:

| Feature | Description |
|---------|-------------|
| Apply Cuts | Remove silences, repeated takes, or custom ranges directly from the timeline |
| Beat Markers | Add detected beats as sequence markers at the click of a button |
| Multicam Cuts | Apply speaker-driven cuts directly to the open sequence |
| Clip Keyframes | Write scale/position keyframes to Motion effect for auto-zoom |
| Batch Rename | Rename project panel clips with find/replace patterns |
| Smart Bins | Auto-sort project items into bins by rule (name, type, duration) |
| Export from Markers | Batch-export clip ranges defined by sequence markers |
| Native Captions | Import SRT as a native Premiere Pro caption track |

---

## Search & AI Commands

### Footage Search

Index your entire media library by spoken content. Search across all clips by keyword or phrase. Results include clip path, timestamp, and matched text snippet.

### Natural Language Commands

Type commands in plain English — OpenCut maps them to API routes using keyword matching or LLM analysis:
- *"remove silence from this clip"* → `/captions/silence`
- *"add captions in Spanish"* → `/captions/transcribe` with language=Spanish
- *"generate YouTube chapters"* → `/captions/chapters`

### Post-Production Deliverables

Generate professional post-production documents from sequence data:
- **VFX Sheet** — Shot list with effects for the VFX supervisor
- **ADR List** — Dialogue list for re-recording sessions
- **Music Cue Sheet** — Music usage log for licensing/clearance
- **Asset List** — Complete media inventory with usage counts

---

## UXP Panel

A parallel panel (`com.opencut.uxp`) is available for Premiere Pro 25.6+ using Adobe's modern UXP platform:

- **Modern JavaScript** — ES modules, async/await, native `fetch()`
- **Same Python backend** — Connects to the same local server on port 5679
- **Auto port detection** — Scans ports 5679–5689 automatically
- **7 tabs** — Cut & Clean, Captions, Audio, Video, Timeline, Search, Deliverables
- **Timeline API** — Uses the `premierepro` UXP module for direct sequence access (where available)

> **Note:** The CEP panel (`com.opencut.panel`) remains the primary panel and supports Premiere Pro 2019+. The UXP panel is the future-facing option for 25.6+.

---

## Architecture

```
+-----------------------+     HTTP/JSON      +-----------------------+
|   Premiere Pro CEP    | <================> |   OpenCut Server      |
|   Panel (HTML/JS)     |   localhost:5679   |   (Python/Flask)      |
|                       |                    |                       |
|  6 tabs, 43 sub-tabs  |                    |  142 API routes       |
|  Dark theme, 6 color  |                    |  Async job queue      |
|  schemes              |                    |  SSE + polling        |
+-----------------------+                    +-----------+-----------+
                                                         |
                                             +-----------+-----------+
                                             |           |           |
                                          +--+--+   +----+---+  +---+----+
                                          |FFmpeg|   |Whisper |  |PyTorch |
                                          |OpenCV|   |Demucs  |  |Models  |
                                          +------+   +--------+  +--------+
```

Everything runs locally. No data leaves your machine. No API keys needed for core features.

## Dependency Tiers

Only the core tier is required -- everything else is optional and auto-detected at runtime.

### Core (required, ~5MB)

```
flask, flask-cors, click, rich
```

### Standard (recommended, ~200MB)

```
pip install opencut[standard]
```

Adds: `faster-whisper`, `opencv-python-headless`, `Pillow`, `numpy`, `librosa`, `pydub`, `noisereduce`, `deep-translator`, `scenedetect`

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

## CLI Usage

```bash
# Generate YouTube chapters
opencut chapters interview.mp4 --provider ollama --model llama3

# Detect repeated takes
opencut repeat-detect recording.mp4 --threshold 0.6

# Search footage library
opencut search index *.mp4
opencut search query "camera lens focal length"

# Match colors to reference
opencut color-match source.mp4 reference.mp4

# Normalize loudness
opencut loudness-match clip1.mp4 clip2.mp4 --target-lufs -14

# Generate post-production documents
opencut deliverables --sequence-json sequence.json --type all

# Natural language editing
opencut nlp "remove silence and add captions in Spanish" --file video.mp4
```

## FAQ

**Q: The panel says "Server offline"**
A: Make sure the backend is running. Check that port 5679 is not blocked. The panel will auto-reconnect when the server comes back.

**Q: Transcription is slow**
A: Install CUDA-enabled PyTorch and use `faster-whisper` with a GPU. The `tiny` model is fastest.

**Q: I get "module not found" errors for AI features**
A: Most AI features are optional. Install them individually or use `pip install opencut[all]`. Check the Dependency Dashboard in Settings to see what's installed.

**Q: Can I use this without Premiere Pro?**
A: Yes. The server runs standalone with a REST API. Call any route with curl or build your own frontend.

**Q: Does this send data to the cloud?**
A: No. Everything runs locally. No telemetry, no API keys for core features. Edge-TTS requires internet for voice synthesis.

**Q: How do I update?**
A: `git pull` and restart the server. Or download the latest exe from [Releases](https://github.com/SysAdminDoc/OpenCut/releases).

## Contributing

Issues and PRs welcome. The codebase:

```
opencut/
  server.py       # Flask server (7500+ lines, 142 routes)
  core/           # Processing modules
  utils/          # Media probing, config
  export/         # Premiere XML, SRT, VTT exporters
  cli.py          # CLI entry point
extension/
  com.opencut.panel/
    client/       # CEP panel (index.html, main.js, style.css)
    host/         # ExtendScript (index.jsx)
    CSXS/         # Extension manifest
```

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with FFmpeg, Whisper, Demucs, PyTorch, and many other open-source projects.
