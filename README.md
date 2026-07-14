# OpenCut

![Version](https://img.shields.io/badge/version-1.33.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-0078D4)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Premiere Pro](https://img.shields.io/badge/Premiere%20Pro-2019+-9999FF?logo=adobepremierepro&logoColor=white)
![Routes](https://img.shields.io/badge/API%20Routes-1546-orange)
![Tests](https://img.shields.io/badge/Tests-11100+-brightgreen)

> Route count is generated from `opencut/_generated/route_manifest.json` and
> reflects **shipped** routes only — each route is tagged
> `implemented` / `dependency-gated` / `stub`, and strategic 501 stubs are
> excluded from the advertised count. Run
> `python -m opencut.tools.dump_route_manifest --check` to verify it is in
> sync with the live Flask app, and `GET /system/route-readiness` for the live
> shipped count and stub list.

> **OpenCut replaces ~$1,400/year of video-editing subscriptions** with a free, MIT-licensed Premiere Pro extension. Unlimited silence-cut direct to timeline, stem separation, voice cloning, animated captions, local LLM highlights, and multi-platform social export -- all running on your own machine with no subscriptions, no usage caps, no cloud upload, and no API keys required for core features.

---

## Project planning

- [ROADMAP.md](ROADMAP.md) is the only active open-work tracker.
- [CHANGELOG.md](CHANGELOG.md) records shipped release-facing changes.
- [RESEARCH.md](RESEARCH.md) summarizes the current research direction and points to archived research inputs.
- [docs/RESEARCH_COMPETITIVE_TEARDOWN_2026-06-10.md](docs/RESEARCH_COMPETITIVE_TEARDOWN_2026-06-10.md) preserves the June 2026 competitive teardown evidence.

---

## Naming & distribution

**Decision (2026-06-14):** the product keeps the name **OpenCut**, but the
package-manager distribution token is **`opencut-ppro`**.

There is an unrelated, popular browser-based editor also called *OpenCut*
([github.com/OpenCut-app/OpenCut](https://github.com/OpenCut-app/OpenCut),
~48K stars). To keep search, PyPI, Homebrew, and winget unambiguous — and to
stay distinct from it when it relaunches — this project distributes under a
qualified token: the **`-ppro`** suffix marks the Adobe **Premiere Pro**
integration that uniquely identifies this tool. So:

- **PyPI:** `pip install opencut-ppro[standard]` (distribution name `opencut-ppro`)
- **Homebrew / winget:** the same `opencut-ppro` token (see ROADMAP).
- **Unchanged:** the product name *OpenCut*, the Python import package
  (`import opencut`), the CLI commands (`opencut`, `opencut-server`,
  `opencut-mcp-server`), and the CEP/UXP extension IDs
  (`com.opencut.panel`, `com.opencut.uxp`).

The `opencut-ppro` name is reserved-by-decision here and in `pyproject.toml`;
the actual PyPI publish (via trusted publishing) is tracked in the roadmap and
is the one step that needs the maintainer's PyPI account.

---

## Quick Start

### Prerequisites

- **Adobe Premiere Pro** 2019 or later (CEP panel) / 25.6+ (UXP panel)
- **Windows 10/11, macOS, or Linux** (Windows installer, macOS/Linux source launchers, and Linux Flatpak/AppImage release packaging)
- **Python 3.11+** for source installs; the Windows installer bundles its own runtime
- **FFmpeg** must be installed and on your system PATH for all video/audio processing. The Windows installer handles this automatically; source installs require a manual install (see below)

### Installation

**Option A -- Windows Installer (recommended for Windows):**

Download the latest Windows installer from [Releases](https://github.com/SysAdminDoc/OpenCut/releases) and run it. Release artifacts are named `OpenCut-Setup-<version>.exe` and handle everything: server, FFmpeg, CEP extension, registry, and optional model downloads. No Python needed.

**Option B -- Install.bat (recommended for Windows source installs):**

This is the easiest way to install from source on Windows. It handles FFmpeg, Python dependencies, the CEP extension, registry keys, and launcher creation in one step.

```
git clone https://github.com/SysAdminDoc/OpenCut.git
cd OpenCut
```

Then right-click `Install.bat` and select **Run as Administrator**. The installer will:
1. Check for and install FFmpeg (via winget) if missing
2. Verify Python 3.9+ is on your PATH
3. Install all Python dependencies (Flask, click, rich, etc.)
4. Install the OpenCut Python package
5. Optionally install Whisper for caption generation
6. Copy the CEP extension to Adobe's extensions folder
7. Set the `PlayerDebugMode` registry key so unsigned extensions load
8. Create `Start-OpenCut.bat` on your desktop to launch the backend server

After the installer finishes, you are ready to use OpenCut -- see **Launch** below.

**Option C -- From source (manual):**

If you prefer manual control or are on macOS/Linux, follow these steps:

1. **Install FFmpeg** and make sure it is on your system PATH:
   - Windows: `winget install Gyan.FFmpeg` or download from https://ffmpeg.org/download.html and add the `bin` folder to your PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) or equivalent for your distro
   - Verify: `ffmpeg -version` should print version info

2. **Clone and install the Python package:**
   ```bash
   git clone https://github.com/SysAdminDoc/OpenCut.git
   cd OpenCut
   pip install -e ".[all]"
   ```
   If `pip` is not found, use `python -m pip install -e ".[all]"` or `py -3.12 -m pip install -e ".[all]"` on Windows. If you get permission errors, add `--user` to the pip command.

3. **Install the CEP extension** (for the Premiere Pro panel):
   - Windows: copy the folder `extension/com.opencut.panel` to `%APPDATA%\Adobe\CEP\extensions\` (create the `extensions` folder if it does not exist)
   - macOS: copy to `~/Library/Application Support/Adobe/CEP/extensions/`
   - Then set the debug registry key so Premiere loads unsigned extensions:
     - Windows: open `regedit`, navigate to `HKCU\Software\Adobe\CSXS.18` (create the key if needed), add a String value `PlayerDebugMode` set to `1`. Repeat for CSXS.17, .16, etc. to cover your Premiere version.
     - macOS: `defaults write com.adobe.CSXS.18 PlayerDebugMode 1`

4. **Start the server:** `python -m opencut.server`

Add `torch-stack` or narrower feature extras only when you need Torch-backed
backends such as WhisperX diarized captioning, Demucs separation, pyannote
diarization, RealESRGAN/GFPGAN restoration, or TransNetV2 scene detection.
The source default keeps the audited convenience extra away from current
Torch/Transformers advisory exposure.

**Option D -- Docker:**

```bash
docker compose up opencut-server
docker compose --profile gpu up opencut-server-gpu
```

Docker publishes the HTTP API on port 5679 and persists data under
`/home/opencut/.opencut`. It does not publish the optional WebSocket 5680 or
MCP 5681 sidecars by default; add a dedicated service/profile before exposing
those opt-in processes from a container.

**Option E -- Linux desktop package:**

Tagged Linux releases build Flatpak and AppImage artifacts from the PyInstaller server bundle. See `docs/LINUX_DISTRIBUTION.md` for the `io.github.sysadmindoc.opencut` Flatpak/AppImage contract and local package-build command.

### Launch

1. **Start the backend server** -- this must be running before the panel can connect:
   - Windows (installer): double-click `Start-OpenCut.bat` on your desktop or in the OpenCut folder
   - Windows (source): open a terminal in the OpenCut folder and run `python -m opencut.server`
   - macOS: run `./OpenCut-Server.command` or `python -m opencut.server`
   - Linux: run `./OpenCut-Server.sh` or `python -m opencut.server`
   - You should see output like `Running on http://127.0.0.1:5679` -- **keep this terminal window open**
2. Open Premiere Pro and go to **Window > Extensions > OpenCut**
3. The panel should show a green connection indicator. Select a clip and start editing.

### Troubleshooting

**Panel says "Server offline" or won't connect:**
The OpenCut panel connects to a backend server running on `http://127.0.0.1:5679`. If the panel cannot connect:
- Make sure the backend server is running (step 1 above). You need a terminal window open with `python -m opencut.server` running.
- Check the terminal for errors. Common issues: port 5679 already in use (change with `OPENCUT_PORT=5680`), missing Python packages (re-run `pip install -e ".[all]"`).
- Check that your firewall is not blocking localhost connections on port 5679.
- The "Live Updates Bridge" toggle in the Settings tab is an optional WebSocket feature for streaming progress updates. It is **not** required for the panel to work -- do not confuse it with the main server connection.

**"Module not found" errors for AI features:**
Most AI features are optional dependencies. Open the **Settings** tab in the panel and scroll to the **Dependency Dashboard** -- it shows every optional package with its install status and the exact pip command to install missing ones. Or install all audited extras at once: `pip install -e ".[all]"`.

**FFmpeg not found:**
The server needs FFmpeg on your PATH. Run `ffmpeg -version` in a terminal to verify. If not found: Windows users can run `winget install Gyan.FFmpeg`, macOS users `brew install ffmpeg`, Linux users `sudo apt install ffmpeg`. After installing, restart your terminal and the OpenCut server.

**Python not found or wrong version:**
OpenCut requires Python 3.11 or later. Run `python --version` to check. On Windows, if Python is installed but not on PATH, use `py -3.12 -m opencut.server` instead, or re-run the Python installer and check "Add Python to PATH".

---

## What OpenCut adds beyond Premiere 26

Premiere 26.x (Jan 2026) ships Object Mask, Generative Extend, Media Intelligence, and 90+ Film Impact transitions natively. OpenCut complements those with capabilities Adobe does not bundle:

| Capability | OpenCut | Premiere 26.x Native |
|-----------|---------|---------------------|
| Silence-cut to timeline | Full ripple-delete with cut review panel | Pause-mute only (no ripple delete) |
| Stem separation | Demucs / BS-RoFormer / MDX-Net (unlimited) | Not available |
| Voice cloning | Chatterbox TTS (15-sec training, local) | Not available |
| Filler word detection | CrisperWhisper verbatim markers + custom words | Not available |
| Animated captions | 55 styles, word-by-word pop/fade/bounce/glow | Basic captions, cloud translation |
| LLM highlights | Ollama (local, no API key), engagement scoring | Not available |
| Social export | Direct YouTube/TikTok/Instagram OAuth upload | Not available |
| Cross-NLE export | OTIO to Resolve, FCP, Avid | Not available |
| Repeated take detection | Jaccard-overlap transcript similarity | Not available |
| Cut review panel | Human-in-loop approve/reject before applying | Not available |

**No usage caps.** Unlike CapCut Pro ($240/yr, metered), Descript ($288-600/yr, credit burn), and Submagic ($468/yr, 40 videos/mo), OpenCut processes unlimited content for $0.

---

## Feature Overview

OpenCut v1.33.1 includes **1,546 shipped API routes** (implemented or dependency-gated; 6 strategic 501 stubs are tracked separately and excluded), **8 panel tabs** with **50+ sub-tabs**, and covers every major video editing automation task.

### Cut & Clean

| Feature | Description | Engine |
|---------|-------------|--------|
| Silence Removal | Detect and remove silent segments with adjustable threshold or AI neural detection | FFmpeg / Silero VAD |
| Silero VAD Mode | ML-based voice activity detection -- 87% TPR vs 50% for energy thresholds. Auto-fallback if PyTorch unavailable | Silero VAD (ONNX) |
| Filler Word Detection | Detect and cut "um", "uh", "like", "you know" + custom words. Two backends: standard Whisper or CrisperWhisper (verbatim mode) | faster-whisper / CrisperWhisper |
| CrisperWhisper Mode | Modified Whisper that transcribes verbatim with `[UH]`/`[UM]` markers -- #1 on OpenASR Leaderboard for filler detection | HuggingFace transformers |
| Waveform Preview | Visual waveform with draggable threshold line synced to slider | FFmpeg PCM + Canvas |
| Trim Tool | Set in/out points to extract a clip portion (stream copy or re-encode) | FFmpeg |
| Full Pipeline | Combined silence + filler removal + captions + zoom in one pass | Multi-stage |
| Repeated Take Detection | Detect when speakers restart a sentence using transcript similarity (Jaccard overlap) | WhisperX |
| Auto-Edit | Motion-based and audio-based automated rough cuts | auto-editor |
| Cut Review Panel | Review and approve/reject individual cuts before applying to timeline | Built-in |

### Captions & Transcription

| Feature | Description | Engine |
|---------|-------------|--------|
| Transcription | Speech-to-text with word-level timestamps | faster-whisper / WhisperX |
| 55 Caption Styles | YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, and more | Pillow renderer |
| Animated Captions | CapCut-style word-by-word pop, fade, bounce, glow, highlight (7 presets) | Pillow + OpenCV |
| Caption Burn-in | Hard-burn styled captions directly into video | FFmpeg drawtext / ASS |
| Speaker Diarization | Identify who's speaking for podcasts/interviews | pyannote.audio |
| Multimodal Diarization | Audio + face cross-modal speaker mapping for multi-camera setups | InsightFace / facenet + pyannote |
| Translation | Translate captions to 50+ languages | NLLB / SeamlessM4T |
| Karaoke Mode | Word-by-word highlight sync for lyrics/captions | Pillow renderer |
| Transcript Editor | Edit segments in-panel with undo/redo and search | Built-in |
| YouTube Chapters | LLM-powered topic change detection for chapter timestamps | Ollama / OpenAI / Anthropic |
| SRT to Native Captions | Import any SRT file as a native Premiere Pro caption track | ExtendScript |

### Audio Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| Stem Separation | Isolate vocals, drums, bass, guitar, piano. Multiple backends and models | Demucs / BS-RoFormer / MDX-Net |
| Noise Reduction | AI noise removal + spectral gating | noisereduce / DeepFilterNet |
| Speech Enhancement | Full speech restoration: denoise + bandwidth extension to 44.1kHz studio quality | Resemble Enhance |
| Normalization | Loudness targeting (LUFS) with broadcast standards | FFmpeg loudnorm |
| Loudness Match | Batch normalize multiple clips to consistent LUFS | FFmpeg two-pass |
| Beat Detection | BPM analysis and beat marker timestamps | librosa |
| Beat Markers | Export beats as Premiere Pro sequence markers for snap-to-beat editing | librosa + ExtendScript |
| Audio Ducking | Auto-lower music under dialogue | FFmpeg sidechaincompress |
| Pro FX Chain | Compressor, EQ, de-esser, limiter, reverb, stereo width, and more | Pedalboard (Spotify) |
| TTS Voice Generation | Text-to-speech with 100+ voices. Three engines: cloud, local fast, and voice cloning | Edge-TTS / Kokoro / Chatterbox |
| Voice Cloning | Record 15 seconds of voice, generate narration in that voice | Chatterbox TTS |
| SFX Generator | Procedural tones, sweeps, impacts, noise | NumPy synthesis |
| AI Music Generation | Generate background music from text prompts | MusicGen (AudioCraft) |

### Video Effects & Processing

| Feature | Description | Engine |
|---------|-------------|--------|
| AI Upscaling | 3 tiers: Lanczos (fast), Real-ESRGAN (balanced), Video2x (premium) | FFmpeg / Real-ESRGAN |
| Background Removal | Per-frame (rembg, 5 models) or temporal (Robust Video Matting, 2 models) | rembg / RVM |
| Robust Video Matting | Recurrent neural network for temporally consistent matting -- no green screen needed | torch hub (RVM) |
| Depth Effects | Depth map estimation, depth-of-field bokeh simulation, 3D Ken Burns parallax zoom | Depth Anything V2 |
| Frame Interpolation | Slow motion / frame rate conversion | FFmpeg / RIFE |
| Face Enhancement | Restore and upscale faces with controllable fidelity slider | GFPGAN / CodeFormer |
| Face Swap | Replace faces using a reference image | InsightFace |
| Face Blur | Privacy-aware face detection and blur/pixelate | MediaPipe / YOLO |
| Style Transfer | Neural artistic style transfer for video | PyTorch models |
| Stabilization | Deshake / vidstab with smoothing and zoom controls | FFmpeg |
| Chromakey | Green/blue/red screen removal + spill suppression | OpenCV HSV |
| Picture-in-Picture | Overlay PiP with position/scale controls | FFmpeg overlay |
| Blend Modes | 14 modes (multiply, screen, overlay, etc.) | FFmpeg blend |
| 34 Transitions | Crossfade, wipe, slide, circle, pixelize, radial, zoom | FFmpeg xfade |
| Particle Effects | 7 presets: confetti, sparkles, snow, rain, fire, smoke, bubbles | Pillow renderer |
| Animated Titles | 6 presets: fade, slide, typewriter, lower third, countdown, kinetic | FFmpeg drawtext |
| Speed Ramping | Time remapping with ease-in/out curves, reverse, slow-mo | FFmpeg setpts |
| Scene Detection | Auto-detect cuts and scene changes (threshold + neural) | PySceneDetect / TransNetV2 |
| Film Grain / Vignette | Adjustable film look overlays | FFmpeg noise/vignette |
| Letterbox | Cinematic aspect ratio bars (2.39:1, 2:1, 1.85:1, 4:3, 1:1) | FFmpeg pad |
| LUT Library | 15 built-in cinematic LUTs + external .cube/.3dl support | FFmpeg lut3d |
| Color Correction | Exposure, contrast, saturation, temperature, shadows, highlights | FFmpeg eq/colorbalance |
| Color Match | Match color profile of one clip to a reference using YCbCr histogram matching | OpenCV |
| Video Reframe | Resize/crop for TikTok, Shorts, Reels, Instagram, or custom dimensions (max 7680px) | FFmpeg scale/crop/pad |
| Face-Tracked Reframe | Auto-crop vertical video centered on the active speaker's face | MediaPipe + FFmpeg |
| Merge / Concatenate | Join multiple clips (fast stream copy or re-encoded) | FFmpeg concat |
| Watermark Removal | Auto-detect (Florence-2) or manual region + LaMA AI inpainting or FFmpeg delogo | Florence-2 / LaMA / FFmpeg |
| Object Removal | Click-to-select any object, track through video, inpaint with temporal consistency | SAM2 + ProPainter |
| Video Denoising | AI temporal denoising for noisy footage | FFmpeg / neural models |

### Highlight & Shorts Generation

| Feature | Description | Engine |
|---------|-------------|--------|
| AI Highlight Extraction | LLM analyzes transcript to find viral/engaging moments with engagement scoring | Ollama / OpenAI / Anthropic |
| Engagement Scoring | Multi-dimensional scoring: hook strength, emotional intensity, pacing, quotability | Text heuristics + LLM |
| Emotion-Based Highlights | Facial emotion analysis across frames, detect emotional peaks as highlights | deepface + OpenCV |
| Shorts Pipeline | One-click: transcribe, highlight, trim, face-reframe, caption burn-in, export | Multi-stage |
| Smart Thumbnails | AI-scored frame extraction with face detection, composition balance, and sharpness analysis | OpenCV |
| Auto Watermark Detection | Florence-2 vision-language model auto-locates watermarks, logos, and text overlays | Florence-2 / edge fallback |
| AI B-Roll Planning | Transcript analysis to identify B-roll insertion points (dialogue gaps, topic shifts, visual references) | Text analysis |
| AI B-Roll Generation | Text-to-video B-roll from prompts using 4 backends | CogVideoX / Wan 2.2 / HunyuanVideo / SVD |

### Timeline Integration

| Feature | Description |
|---------|-------------|
| Apply Cuts to Timeline | Remove silences, repeated takes, or custom ranges directly in the active sequence |
| Beat Markers | Add detected beats as Premiere Pro sequence markers |
| Multicam Auto-Switch | Speaker diarization to multicam cut list, applied directly to sequence |
| Clip Keyframes | Write scale/position keyframes for auto-zoom effects |
| Batch Rename | Rename project panel clips with find/replace patterns |
| Smart Bins | Auto-sort project items into bins by rule (name, type, duration) |
| Export from Markers | Batch-export clip ranges defined by sequence markers |
| SRT to Native Captions | Import SRT as a native Premiere Pro caption track |
| OTIO Export | Export edits as OpenTimelineIO for DaVinci Resolve, Final Cut Pro, Avid, and any OTIO-compatible editor |

### Search & AI Commands

| Feature | Description |
|---------|-------------|
| Footage Search | Index your entire media library by spoken content, search across all clips by keyword |
| FTS5 Database Search | SQLite full-text search index with auto-indexing and cleanup |
| Natural Language Commands | Type in English -- OpenCut maps commands to API routes via keyword matching or LLM |
| Chat Editor | Multi-turn LLM-powered editing assistant with session memory and action parsing |
| Post-Production Deliverables | Generate VFX sheets, ADR lists, music cue sheets, and asset inventories from sequence data |

### Export & Batch

| Feature | Description | Engine |
|---------|-------------|--------|
| 13 Platform Presets | YouTube, TikTok, Instagram, Twitter/X, LinkedIn, Podcast, Snapchat, Facebook, Pinterest | FFmpeg encode |
| Batch Processing | Process multiple clips in parallel with GPU-aware concurrency | ThreadPool |
| Transcript Export | SRT, VTT, ASS, plain text, timestamped | Built-in |
| OTIO Timeline Export | Universal timeline interchange for Premiere, Resolve, FCP, Avid | OpenTimelineIO |
| Social Media Upload | Direct posting to YouTube, TikTok, and Instagram with OAuth | Platform APIs |

### DaVinci Resolve Integration

| Feature | Description |
|---------|-------------|
| Resolve Bridge | Python scripting API bridge with auto-reconnect |
| Media Pool Access | List, import, and organize media pool clips |
| Timeline Info | Read timeline structure, add markers |
| Render Queue | Start renders programmatically |

---

## Panel UX

| Feature | Description |
|---------|-------------|
| Command Palette | Ctrl+K fuzzy search across all 30+ operations |
| Clip Preview | Thumbnail + duration/resolution/size when selecting a clip |
| Recent Clips | Dropdown of last 10 used clips, persisted across sessions |
| Auto Media Discovery | Periodic project media scan + visibility/focus refresh + post-job re-scan |
| Favorites Bar | Pin frequently-used operations as quick-access chips |
| First-Run Wizard | 3-step onboarding overlay for new users |
| Output File Browser | Browse recent outputs with Import-to-Premiere button |
| Custom Workflows | Chain operations into reusable named workflows (6 built-in) |
| Project Templates | YouTube, Shorts, TikTok/Reels, Podcast, Cinema, Broadcast |
| Collapsible Cards | Click headers to collapse/expand dense form sections |
| Cut Review Panel | Review and approve/reject individual cuts before applying |
| Right-Click Context Menu | Quick-action context menu on clip selector |
| Job Time Estimates | Estimated processing time based on historical data |
| Per-Operation Presets | Save/load settings per operation |
| Settings Import/Export | Bundle all settings as JSON for backup or sharing |
| Server Health Monitor | Auto-reconnect with exponential backoff when backend goes offline |
| Studio Graphite UI | Single premium theme tuned for the CEP editing workspace |
| Workspace Polish | Premium shell, command palette, history, and output surfaces |
| Toast Notifications | Non-intrusive slide-in alerts for job completion |
| Keyboard Shortcuts | Configurable shortcuts with reference card (Ctrl+Shift+S for silence, etc.) |
| Quick Action Buttons | One-click workflows on Cut, Captions, Audio, and Video tabs |
| Status Bar | Live system health, GPU usage, uptime, and job count |
| i18n | Internationalization system with extensible locale files (417 keys) |
| Responsive Layout | 4 breakpoints for compact panels (800px, 480px, 440px, 380px) |
| Context-Aware Guidance | Clip-specific recommendations and smart tab reordering |
| Engine Preferences | Per-domain AI engine selection with quality/speed labels |
| WebSocket Real-Time | Live job progress via WebSocket with SSE/poll fallback |

---

## UXP Panel (Premiere Pro 25.6+)

A modern panel (`com.opencut.uxp`) using Adobe's UXP platform:

- **Modern JavaScript** -- ES modules, async/await, native `fetch()`
- **Same Python backend** -- Connects to the same local server on port 5679
- **Auto port detection** -- Scans ports 5679-5689 automatically
- **9 tabs** -- Cut & Clean, Captions, Audio, Video, Timeline, Search, Deliverables, Agent, Settings
- **Direct Premiere API** -- Uses the `premierepro` UXP module for sequence access
- **Project media discovery** -- Scans project items via UXP API with datalist autocomplete
- **OTIO export** -- Export timeline edits in universal format from within the panel
- **Connection-aware UI** -- Buttons disable when server is offline, re-enable on reconnect
- **Near-complete feature parity** with CEP panel including depth effects, emotion highlights, B-roll, chat editor, social upload, engine preferences, and WebSocket bridge

> **Premiere 26 compatibility:** Both CEP (`[13.0, 99.9]` host range) and UXP (`minVersion 25.6`) panels cover Premiere 26.x. CEP support in Premiere Pro ends approximately September 2026. The UXP panel is the future-facing option. See `docs/UXP_MIGRATION.md` for the migration plan.

---

## Architecture

```
+-----------------------+     HTTP/JSON      +-----------------------+
|   Premiere Pro CEP    | <================> |   OpenCut Server      |
|   Panel (HTML/JS)     |   localhost:5679   |   (Python/Flask)      |
|                       |                    |                       |
|  8 tabs, 50+ sub-tabs |   WebSocket:5680   |  1,546 shipped routes |
|  Studio Graphite, i18n| <~~~~~~~~~~~~~~~>  |  602 core modules     |
|  Keyboard shortcuts   |   SSE streaming    |  107 route blueprints |
+-----------+-----------+                    +-----------+-----------+
            |                                            |
+-----------+-----------+                    +-----------+-----------+
|   Premiere Pro UXP    |                    |           |           |
|   Panel (ES modules)  |                 +--+--+   +----+---+  +---+----+
|   9 tabs, modern JS   |                 |FFmpeg|   |Whisper |  |PyTorch |
+-----------------------+                 |OpenCV|   |Demucs  |  |Models  |
            |                             +------+   +--------+  +--------+
+-----------+-----------+                            |
|  DaVinci Resolve      |                 +----------+----------+
|  (Python scripting)   |                 | Engine Registry      |
+-----------------------+                 | 18+ engines, 12      |
                                          | domains, swappable   |
                                          +---------------------+
```

Everything runs locally. No data leaves your machine. No API keys needed for core features.

---

## AI Models & Backends

OpenCut supports multiple backends per feature via the **Engine Registry**, letting you choose speed vs. quality:

| Feature | Fast | Balanced | Best Quality |
|---------|------|----------|-------------|
| Silence Detection | FFmpeg energy threshold | Silero VAD (auto) | Silero VAD |
| Filler Detection | Whisper + text matching | Whisper + text matching | CrisperWhisper verbatim |
| Transcription | faster-whisper tiny | faster-whisper base | WhisperX medium |
| Stem Separation | Demucs htdemucs | BS-RoFormer | Demucs htdemucs_ft |
| Denoising | noisereduce | DeepFilterNet | Resemble Enhance |
| TTS | Edge-TTS (cloud) | Kokoro (local, 82M) | Chatterbox (voice clone) |
| Background Removal | rembg U2Net | rembg BiRefNet | Robust Video Matting |
| Face Restoration | GFPGAN | CodeFormer (0.5 fidelity) | CodeFormer (0.7 fidelity) |
| Scene Detection | FFmpeg threshold | PySceneDetect | TransNetV2 (neural) |
| Watermark Detection | Edge/corner fallback | Florence-2 base | Florence-2 base |
| Upscaling | FFmpeg Lanczos | Real-ESRGAN | Real-ESRGAN anime |
| Highlights | Keyword heuristics | LLM + engagement scoring | LLM + emotion analysis |
| Depth Effects | -- | Depth Anything V2 | Depth Anything V2 |
| B-Roll Generation | SVD (image-to-video) | Wan 2.2 | CogVideoX |
| Speaker Diarization | pyannote.audio | pyannote + face clustering | Multimodal diarization |

Engine preferences are configurable per-domain in the Settings tab and persist across sessions.

---

## Dependency Tiers

Only the core tier is required -- everything else is optional and auto-detected at runtime.

### Core (required, ~5MB)

```
flask, flask-cors, click, rich
```

### Standard (recommended, ~200MB)

```bash
pip install opencut-ppro[standard]
```

Adds: `faster-whisper`, `opencv-python-headless`, `Pillow`, `numpy`, `librosa`, `noisereduce`, `scenedetect`

### Audited convenience install (non-Torch optional stack)

```bash
pip install opencut-ppro[all]
```

Adds all standard deps plus: `pedalboard`, `edge-tts`, `rembg`, `insightface`, `onnxruntime`, `auto-editor`, `opentimelineio`, and `otio-aaf-adapter`. Torch/Transformers-backed packages stay out of this release-audited lane until their advisory and resolver posture is clean.

For the larger Torch-backed stack, install the explicit extra or narrower feature extras:

```bash
pip install opencut-ppro[all,torch-stack]
pip install opencut-ppro[captions-whisperx]  # WhisperX diarized captions
```

`torch-stack` includes WhisperX, Demucs, RealESRGAN/GFPGAN, pyannote.audio, TransNetV2, PyTorch, torchvision, and Transformers. AudioCraft/MusicGen and Resemble Enhance remain separate Python 3.11 installs (`opencut-ppro[music]`, `opencut-ppro[enhance]`) because they hard-pin older Torch stacks.

### GPU Acceleration

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

GPU-heavy routes have built-in rate limiting (one GPU job at a time) and cancellation support.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCUT_PORT` | `5679` | Server port |
| `OPENCUT_HOST` | `127.0.0.1` | Bind address |
| `OPENCUT_OUTPUT_DIR` | Source file dir | Default output directory |
| `WHISPER_MODELS_DIR` | `~/.cache` | Whisper model cache |
| `OPENCUT_C2PA_SIGNING_KEY` | unset | Ed25519 private-key PEM or path used to sign local C2PA sidecars |
| `OPENCUT_C2PA_C2PATOOL` | `c2patool` on PATH | Optional C2PA Tool executable used for embedded MP4/JPEG/PNG credentials |
| `OPENCUT_C2PA_SIGNING_CERT` | unset | Optional certificate path/content forwarded to C2PA Tool for operator trust chains |

## CLI Usage

```bash
# Silence removal with Silero VAD
opencut silence video.mp4 --method vad

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

# Call any generated backend route from scripts
opencut route GET /system/check-failures
opencut route POST /queue/add --data '{"endpoint":"/captions","payload":{"filepath":"C:/clip.mp4"}}'
```

---

## Backend Infrastructure

| Feature | Description |
|---------|-------------|
| Async Job System | Background processing with SSE streaming, WebSocket, and polling fallback |
| Job Queue | Sequential job processing with queue management and cancellation |
| GPU Rate Limiting | One GPU-heavy job at a time with 429 responses when busy |
| FFmpeg Progress | Real-time percentage from `-progress pipe:1` parsing |
| Cancel + Kill | Job cancellation terminates running FFmpeg subprocesses |
| Output Deduplication | Auto-increment suffix prevents overwriting previous outputs |
| Temp Cleanup | Stale preview files cleaned up on server startup |
| Dependency Dashboard | Grid view of all optional deps with install status and one-click install |
| GPU Auto-Detection | Recommend optimal settings based on detected GPU VRAM |
| AI Model Manager | View/delete downloaded models to free disk space |
| Engine Registry | Swappable AI backends per feature domain (18+ engines, 12 domains) |
| Job Persistence | SQLite-backed job history survives server restarts |
| Job Recovery | Detects and reports interrupted jobs on startup |
| Structured Errors | Error taxonomy with machine-readable codes and recovery suggestions |
| JSON Structured Logging | File handler outputs JSON logs with job-ID correlation |
| Workflow Engine | Chain multi-step processing with cancellation between steps |
| Plugin System | Example plugins (timecode watermark, clip notes) with hot-reload |
| Docker Support | Multi-stage Dockerfile + docker-compose with GPU variant |
| Log Viewer | Filtered log tail endpoint for real-time debugging |
| CSRF Protection | Token-based cross-site request forgery protection |
| Rate Limiting | Per-endpoint rate limits to prevent abuse |
| Path Validation | Input sanitization with realpath + prefix whitelist to prevent path traversal |
| WebSocket Bridge | Real-time bidirectional communication on port 5680 |
| Social Media Integration | OAuth-based upload to YouTube, TikTok, Instagram |

---

## Security

- **No `shell=True`** in any subprocess call
- **No `eval`/`exec`/`pickle`** anywhere in the codebase
- **CSRF protection** on all POST/DELETE routes via `X-OpenCut-Token` header
- **Path traversal prevention** via `validate_filepath()` with realpath + prefix whitelist
- **HTML sanitization** via `esc()` for all dynamic innerHTML content
- **Input bounds** on all numeric parameters (dimensions, durations, counts)
- **GPU rate limiting** prevents resource exhaustion from concurrent heavy jobs
- **ASS subtitle injection prevention** strips override sequences from caption text
- **Social OAuth credentials** stored locally at `~/.opencut/social_credentials.json`

---

## Testing

```bash
# Verify the selected Python has runtime metadata and test tooling
py -3.12 scripts/bootstrap_check.py --metadata-only --dev

# Repair a stale repo virtualenv before running tests from it
py -3.12 -m venv --clear .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -q

# Run smoke tests only
python -m pytest tests/test_route_smoke.py -q

# Run core module tests
python -m pytest tests/test_core_modules.py tests/test_core_modules_batch2.py -q

# Run ExtendScript mock tests
node tests/jsx_mock.js

# CEP panel checks from Windows UNC/HGFS checkouts
cd extension/com.opencut.panel
npm run audit:check:win -- --json
npm run audit:esbuild:win -- --json
npm run build:verify:win

# Pre-commit hooks (auto-runs ruff lint/format on commit, pytest smoke on push)
pre-commit install
pre-commit install --hook-type pre-push
```

11,100+ estimated tests across 279 root test files covering route smoke tests, core module unit tests, feature integration tests, plugin tests, and ExtendScript mock harness.

---

## FAQ

**Q: The panel says "Server offline" or "Server disconnected"**
A: See the [Troubleshooting](#troubleshooting) section above. In short: the backend server must be running (`python -m opencut.server` or `Start-OpenCut.bat`), keep its terminal window open, and make sure port 5679 is not blocked by a firewall. The panel auto-reconnects with exponential backoff when the server comes back.

**Q: Transcription is slow**
A: Install CUDA-enabled PyTorch and use `faster-whisper` with a GPU. The `tiny` model is fastest. For batch processing, `insanely-fast-whisper` on GPU offers 10-15x speedup.

**Q: I get "module not found" errors for AI features**
A: See the [Troubleshooting](#troubleshooting) section above. Most AI features are optional. Use `pip install -e ".[all]"` (source) or `pip install opencut-ppro[all]` (PyPI) to install all audited extras. For Torch-backed features add `torch-stack`: `pip install opencut-ppro[all,torch-stack]`. The Dependency Dashboard in the Settings tab shows every optional package and the pip command to install it.

**Q: Can I use this without Premiere Pro?**
A: Yes. The server runs standalone with a REST API. Call any route with curl, use the CLI, or build your own frontend. DaVinci Resolve is also supported via the Resolve Bridge.

**Q: Does this send data to the cloud?**
A: No by default. Everything core runs locally, and fresh installs emit no telemetry. Optional Aptabase telemetry is available only after explicit opt-in; see [docs/TELEMETRY.md](docs/TELEMETRY.md). Edge-TTS requires internet for voice synthesis; LLM features can use local Ollama or cloud providers. Social media upload is opt-in and requires explicit OAuth connection.

**Q: Can I export edits to DaVinci Resolve or Final Cut Pro?**
A: Yes. Use the OTIO (OpenTimelineIO) export in the Timeline tab. OTIO files can be imported into Resolve, FCP, Avid, and any OTIO-compatible editor. Resolve also has a direct Python scripting bridge.

**Q: How do I update?**
A: `git pull` and restart the server. Or download the latest exe from [Releases](https://github.com/SysAdminDoc/OpenCut/releases).

**Q: What's the difference between the CEP and UXP panels?**
A: The CEP panel supports Premiere Pro 2019+ and has full feature coverage. The UXP panel is for Premiere Pro 25.6+ using Adobe's modern platform. Both connect to the same backend. CEP will be deprecated by Adobe around September 2026.

**Q: How do I choose between AI backends?**
A: Go to Settings > AI Engine Preferences. Each feature domain (silence, transcription, TTS, etc.) has a dropdown showing available backends with quality and speed ratings. The default "Auto" mode picks the highest-priority available engine.

---

## Project Structure

```
opencut/
  server.py          # Flask app factory + startup
  core/              # 602 processing modules (silence, captions, audio, video, AI, VR, dubbing)
  routes/            # 105 route modules / 107 manifest blueprints
  export/            # Premiere XML, SRT, VTT, ASS, OTIO exporters
  utils/             # Media probing, config dataclasses
  checks.py          # Dependency availability checks
  errors.py          # Structured error taxonomy
  jobs.py            # Async job system with cancellation
  job_store.py       # SQLite job persistence
  gpu.py             # GPU context manager for VRAM cleanup
  security.py        # CSRF, rate limiting, path validation
  helpers.py         # FFmpeg progress parsing, output dedup
  cli.py             # CLI entry point
extension/
  com.opencut.panel/
    client/          # CEP panel (index.html, main.js ~16,661 lines, style.css ~17,960 lines)
    host/            # ExtendScript host (index.jsx ~2,736 lines)
    CSXS/            # Extension manifest
  com.opencut.uxp/
    main.js          # UXP panel (~7,175 lines)
    index.html       # UXP panel UI
    style.css        # UXP dark theme
tests/               # pytest test suite (11,100+ estimated tests, 279 root test files)
RESEARCH.md          # Current consolidated research conclusions
ROADMAP.md           # Active open-work tracker
docs/
  RESEARCH_COMPETITIVE_TEARDOWN_2026-06-10.md # June 2026 competitive teardown
  UXP_MIGRATION.md   # CEP to UXP migration plan
installer/           # WPF installer (C# .NET 9) + legacy Inno Setup
scripts/             # Build and utility scripts
```

## Cost Comparison (June 2026)

| Tool | Price | Local/Cloud | Auto-Captions | Stem Separation | Voice Clone | Silence Cut |
|------|-------|-------------|---------------|----------------|-------------|-------------|
| **OpenCut** | **$0** (MIT) | Local | Whisper (unlimited) | Demucs (unlimited) | Chatterbox (unlimited) | Yes |
| CapCut Pro | $19.99/mo ($240/yr) | Cloud | Paywalled (was free) | No | No | Limited |
| Descript | $24-50/mo ($288-600/yr) | Cloud | 1-12h/mo quota | No | Overdub (metered) | Yes |
| Submagic | $39/mo ($468/yr) | Cloud | 40 videos/mo | No | No | No |
| AutoCut | $16/mo ($192/yr) | Local | Whisper-based | No | No | Yes |
| FireCut | $20/mo ($240/yr) | Local | Yes | No | No | Yes |

**Annual cost of paid alternatives: $192-$600/yr.** OpenCut runs locally, has no usage caps, and ships every feature above for $0 under MIT.

## Contributing

Issues and PRs welcome. See `CLAUDE.md` for codebase patterns and conventions.

Key patterns:
- **Backend:** Lazy imports for optional deps, structured errors, async job system, capability flags in `/health`, TooManyJobsError handling, GPU rate limiting
- **CEP panel:** `_on()` helper for null-safe event binding, `_setHint()` for capability hints, `esc()` for HTML sanitization, `escPath()` for ExtendScript strings, `api()` callback wrapper for all HTTP calls
- **UXP panel:** `PProBridge` for Premiere API, `BackendClient` for HTTP, `JobPoller` with completion hooks, `textContent` for XSS-safe DOM updates
- **ExtendScript:** ES3 only, JSON.stringify returns, heavy try/catch, poll loops for imports

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with FFmpeg, Whisper, Demucs, PyTorch, Silero VAD, SAM2, ProPainter, Florence-2, deepface, Depth Anything, OpenTimelineIO, and many other open-source projects.
