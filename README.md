# <img width="48" height="48" alt="ChatGPT Image Feb 3, 2026, 03_15_28 PM-48x48" src="https://github.com/user-attachments/assets/19d60483-f302-4843-8e9f-c656835711e3" /> OpenCut


**Open-source video editing automation for Adobe Premiere Pro.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Premiere Pro 2023+](https://img.shields.io/badge/Premiere%20Pro-2023%2B-9999FF.svg)](https://www.adobe.com/products/premiere.html)

Remove silences, generate captions, auto-zoom on emphasis, strip filler words, render styled caption overlays, and automate podcast multicam switching — all from a **panel inside Premiere Pro** or the **command line**. Non-destructive editing via Premiere Pro XML or direct timeline integration.

> Free, open-source alternative to [AutoCut](https://www.autocut.com), [FireCut](https://firecut.co), and [TimeBolt](https://www.timebolt.io).

<img width="1352" height="496" alt="2026-02-03 14_19_32-Adobe Premiere - C__Users_--_Documents_Adobe_Premiere Pro_26 0_dds prproj _" src="https://github.com/user-attachments/assets/c3cccb31-26ce-442b-862f-a34aea55e52f" />

---

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Command Line Usage](#command-line-usage)
- [Presets](#presets)
- [Whisper Model Guide](#whisper-model-guide)
- [Architecture](#architecture)
- [Manual Installation](#manual-installation)
- [Uninstall](#uninstall)
- [Contributing](#contributing)
- [License](#license)
- [Credits](#credits)

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Windows 10/11** | macOS/Linux CLI works, but the CEP panel is Windows-focused |
| **Python 3.9+** | [Download](https://www.python.org/downloads/) — check **"Add Python to PATH"** |
| **Premiere Pro 2023+** | Version 23.0 or newer for the panel; CLI works without Premiere |

### Install

```
1. Download or clone this repository
2. Double-click Install.bat
3. Click "Yes" if prompted by UAC
```

The installer handles everything automatically:

- Installs **FFmpeg** via winget (or chocolatey as fallback)
- Installs all **Python dependencies** in editable mode
- Optionally installs **Whisper** for caption generation (with multiple fallback strategies if Rust/tokenizers fail)
- Copies the **CEP extension** into Premiere Pro's extensions directory
- Sets the **PlayerDebugMode** registry key for unsigned extensions (CSXS 7–12)
- Creates a **desktop shortcut** and `Start-OpenCut.bat` launcher

### Usage

1. **Double-click `Start-OpenCut.bat`** (or the desktop shortcut) to start the backend server
2. Open **Premiere Pro**
3. Go to **Window → Extensions → OpenCut**
4. Select a clip in the Project panel or drop a file into the OpenCut panel
5. Choose your settings and click a feature button (e.g., **Remove Silences**)
6. When processing completes, click **Edit in Timeline** to apply directly, or **Import XML** for a new sequence

---

## Features

### Silence Removal

Detects and removes silent sections from your footage using FFmpeg's `silencedetect` filter. Configurable threshold, minimum duration, and padding give you precise control over how aggressively cuts are made.

- Adjustable noise threshold (dB), minimum silence length, and before/after padding
- Five built-in presets for common workflows (YouTube, podcast, interview, etc.)
- Exports FCP 7 XML for Premiere Pro import, or applies edits directly to the timeline via ExtendScript
- Dry-run mode to preview what would be cut before committing

### Caption Generation

AI-powered transcription using OpenAI's Whisper models. Supports three backends with automatic detection and graceful fallback:

| Backend | Install | Speed | Notes |
|---------|---------|-------|-------|
| **faster-whisper** | `pip install faster-whisper` | Fastest | Recommended default; CTranslate2-based |
| **openai-whisper** | `pip install openai-whisper` | Medium | Official reference implementation |
| **whisperx** | `pip install whisperx` | Fast | Best word-level timestamp alignment |

- Model sizes from `tiny` (39 MB) to `large-v3` (1.5 GB) — choose your speed/accuracy tradeoff
- Word-level timestamps for precise subtitle alignment
- Auto language detection or explicit language selection
- Exports to **SRT**, **VTT**, or **JSON**
- On-demand Whisper installation from the Premiere panel (no terminal required) with 4 fallback strategies for tokenizers/Rust issues

### Styled Caption Overlays

Render animated caption overlays as transparent video files (QuickTime Animation with alpha) that layer directly on top of your footage in Premiere.

Six visual presets are included:

| Preset | Style |
|--------|-------|
| **YouTube Bold** | Impact font, yellow highlight, thick black stroke |
| **Clean Modern** | Arial, cyan highlight, semi-transparent background box |
| **Neon Pop** | Arial Black, green highlight, purple neon glow |
| **Minimal** | Thin stroke, subtle shadow, understated |
| **Boxed** | Dark background panel, yellow highlight, no stroke |
| **Cinematic** | Georgia serif, warm tones, letterbox-friendly positioning |

Each preset supports word-by-word highlight animation and automatic action word emphasis (detected via audio energy peaks and keyword matching). Overlays are rendered with Pillow and composited into transparent MOV files via FFmpeg.

### Filler Word Removal

Detects and removes 40+ filler words and phrases from your recordings using Whisper's word-level timestamps. Includes hesitation sounds (`um`, `uh`, `er`, `ah`, `hm`) and verbal fillers (`like`, `so`, `basically`, `you know`, `I mean`, `kind of`, `sort of`, etc.).

- **Context-aware filtering** — distinguishes safe fillers (always remove) from context-dependent ones (only remove at sentence boundaries or in filler clusters)
- **Custom word support** — add your own filler words or phrases
- **Integrates with silence removal** — filler-word gaps are merged into the silence removal pass for seamless edits

### Speaker Diarization (Podcast Multicam)

Identifies who is speaking when using [pyannote.audio](https://github.com/pyannote/pyannote-audio), then generates automatic camera switch events for multicam podcast editing.

- Auto-detect or specify the number of speakers
- Maps speakers to camera angles for multicam sequences
- Per-speaker duration analysis
- Minimum segment duration to prevent rapid switching
- Requires a free [HuggingFace token](https://huggingface.co/settings/tokens) for model access

### Auto Zoom

Analyzes audio energy to find emphasis points and generates zoom keyframes that add visual punch to talking-head content.

- Configurable max zoom scale, in/out duration, and minimum interval between zooms
- Energy threshold controls sensitivity
- Keyframes export into the Premiere XML timeline

### Edit in Timeline

Instead of the traditional XML import workflow (which can trigger Premiere's "Locate Media" dialog if file paths don't match exactly), OpenCut can build the edited sequence directly inside Premiere using ExtendScript. This creates a new sequence with your speech segments placed on the timeline, referencing media already in your project — no relinking needed.

### Additional Panel Features

- **Waveform visualization** — audio waveform data for timeline scrubbing previews
- **Audio preview** — generate and play back preview clips of specific time ranges
- **Job management** — long-running tasks run in background threads with real-time progress via SSE streaming; jobs can be cancelled
- **Capabilities detection** — the panel auto-detects which optional backends (Whisper, pyannote) are installed and shows/hides features accordingly

---

## Command Line Usage

The CLI works independently of Premiere Pro — useful for batch processing, scripting, and headless servers.

```bash
# Remove silences (exports Premiere XML)
opencut silence video.mp4

# Use a preset
opencut silence video.mp4 --preset aggressive

# Custom silence settings
opencut silence video.mp4 -t -25 -d 0.3 --padding-before 0.08

# Generate captions
opencut captions video.mp4 --model small --language en

# Full pipeline (silence + captions + zoom)
opencut full video.mp4 --preset youtube

# Podcast workflow (silence + diarization)
opencut podcast video.mp4 --speakers 2

# Show media info (duration, codec, resolution, etc.)
opencut info video.mp4

# Dry run — analyze without exporting
opencut silence video.mp4 --dry-run

# Custom output path and sequence name
opencut silence video.mp4 -o "~/Desktop/my_edit.xml" --name "Final Cut"
```

---

## Presets

| Preset | Threshold | Min Silence | Padding | Best For |
|--------|-----------|-------------|---------|----------|
| `default` | -30 dB | 0.5s | 0.1s | General use |
| `aggressive` | -25 dB | 0.3s | 0.05s | Fast-paced YouTube, shorts |
| `conservative` | -40 dB | 1.0s | 0.2s | Interviews, quiet rooms |
| `podcast` | -35 dB | 0.75s | 0.15s | Conversations, 2-speaker default |
| `youtube` | -28 dB | 0.4s | 0.08s | YouTube with small Whisper model + 1.2x zoom |

Use presets from the CLI with `--preset <name>` or select them in the Premiere panel dropdown.

---

## Whisper Model Guide

| Model | Download | Speed | Accuracy | VRAM |
|-------|----------|-------|----------|------|
| `tiny` | 39 MB | Fastest | Basic | ~1 GB |
| `base` | 74 MB | Fast | Good *(default)* | ~1 GB |
| `small` | 244 MB | Medium | Very good | ~2 GB |
| `medium` | 769 MB | Slow | Excellent | ~5 GB |
| `large-v3` | 1.5 GB | Slowest | Best | ~10 GB |
| `turbo` | 809 MB | Fast | Near-best | ~6 GB |

Whisper runs on CPU if no GPU is available (slower but works). For `faster-whisper`, a CUDA-compatible GPU with the appropriate cuDNN libraries provides the best performance.

---

## Architecture

```
Install.bat / Install.ps1           ← Double-click to install everything
Start-OpenCut.bat                   ← Launches the backend server
Uninstall.bat                       ← Clean removal

opencut/                             Python Package
├── server.py                        Flask API on localhost:5679
├── cli.py                           Click-based CLI with Rich console output
├── __main__.py                      python -m opencut entry point
├── core/
│   ├── silence.py                   FFmpeg silencedetect → speech segments
│   ├── captions.py                  Whisper transcription (3 backends)
│   ├── styled_captions.py           Pillow + FFmpeg caption overlay renderer
│   ├── fillers.py                   Filler word detection + context-aware removal
│   ├── diarize.py                   pyannote speaker diarization → camera switches
│   ├── zoom.py                      Audio energy → zoom keyframes
│   └── audio.py                     PCM extraction, RMS energy analysis
├── export/
│   ├── premiere.py                  FCP 7 XML timeline generation
│   └── srt.py                       SRT / VTT / JSON subtitle export
└── utils/
    ├── config.py                    Dataclass configs + 5 presets
    └── media.py                     ffprobe metadata wrapper

extension/com.opencut.panel/         CEP Extension (Premiere Pro Panel)
├── CSXS/manifest.xml                Extension manifest (Premiere 2019+ / v13–99)
├── client/
│   ├── index.html                   Panel UI (dark theme)
│   ├── main.js                      Backend communication, job polling, SSE
│   ├── style.css                    Dark theme matching Premiere's UI
│   └── CSInterface.js               Adobe CEP library
└── host/
    └── index.jsx                    ExtendScript (ES3) — file browse, XML import,
                                     direct timeline editing, caption/overlay import
```

### How It Works

```
┌─────────────────────────────┐
│  Premiere Pro                │
│  ┌───────────────────────┐  │
│  │  OpenCut Panel (CEP)  │  │  HTTP / JSON
│  │  index.html + main.js │──────────────────┐
│  └───────────────────────┘  │               │
│           │                  │               ▼
│           │ ExtendScript     │     ┌──────────────────┐
│           ▼                  │     │  Flask Backend    │
│  ┌───────────────────────┐  │     │  localhost:5679   │
│  │  host/index.jsx       │  │     │                   │
│  │  - Import XML         │  │     │  ├─ FFmpeg        │
│  │  - Edit in Timeline   │  │     │  ├─ Whisper       │
│  │  - Import captions    │  │     │  ├─ pyannote      │
│  │  - Import overlay     │  │     │  └─ Pillow        │
│  └───────────────────────┘  │     └──────────────────┘
└─────────────────────────────┘              │
                                             ▼
                                    ┌──────────────────┐
                                    │  Output Files     │
                                    │  ├─ .xml (timeline)│
                                    │  ├─ .srt / .vtt   │
                                    │  └─ .mov (overlay) │
                                    └──────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/capabilities` | GET | Detect installed backends (Whisper, pyannote) |
| `/info` | POST | Media file metadata via ffprobe |
| `/silence` | POST | Silence detection + XML/timeline export |
| `/captions` | POST | Whisper transcription → SRT/VTT/JSON |
| `/styled-captions` | POST | Rendered caption overlay → transparent MOV |
| `/caption-styles` | GET | List available caption style presets |
| `/fillers` | POST | Filler word detection + removal |
| `/full` | POST | Combined pipeline (silence + captions + zoom) |
| `/install-whisper` | POST | On-demand Whisper installation with fallbacks |
| `/waveform` | POST | Audio waveform data for visualization |
| `/preview-audio` | POST | Generate audio preview for a time range |
| `/regenerate` | POST | Re-process with different settings |
| `/export-video` | POST | Render edited video directly |
| `/status/<job_id>` | GET | Job progress polling |
| `/stream/<job_id>` | GET | SSE real-time progress stream |
| `/cancel/<job_id>` | POST | Cancel a running job |
| `/jobs` | GET | List all active/completed jobs |

---

## Manual Installation

If you prefer not to use the one-click installer:

```bash
# 1. Install FFmpeg
winget install Gyan.FFmpeg

# 2. Clone and install the Python package
git clone https://github.com/opencut/opencut.git
cd opencut
pip install -e .

# 3. (Optional) Install Whisper for captions
pip install faster-whisper

# 4. (Optional) Install pyannote for speaker diarization
pip install pyannote.audio

# 5. Copy the CEP extension to Adobe's extensions folder
xcopy /E /I "extension\com.opencut.panel" "%APPDATA%\Adobe\CEP\extensions\com.opencut.panel"

# 6. Enable unsigned extensions (PowerShell as admin)
# Repeat for each CSXS version your Premiere uses (typically CSXS.11 or CSXS.12)
New-Item -Path "HKCU:\Software\Adobe\CSXS.12" -Force
Set-ItemProperty -Path "HKCU:\Software\Adobe\CSXS.12" -Name "PlayerDebugMode" -Value "1"

# 7. Start the backend server
python -m opencut.server

# 8. Open Premiere Pro → Window → Extensions → OpenCut
```

---

## Uninstall

Double-click `Uninstall.bat`, or run:

```powershell
.\Install.ps1 -Uninstall
```

This removes the CEP extension, registry keys, desktop shortcut, and launcher. Your Python packages are left intact (uninstall with `pip uninstall opencut` if desired).

---

## Contributing

```bash
# Clone the repo
git clone https://github.com/opencut/opencut.git
cd opencut

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check opencut/
```

### Project Notes

- The Flask server uses **CORS** because CEP panels run with a `null` origin
- Long-running tasks execute in **background threads** with progress via SSE streaming or polling
- **FCP 7 XML** format is used instead of FCPXML for maximum Premiere compatibility
- ExtendScript files **must use ES3 syntax** — no `let`, `const`, arrow functions, or template literals
- Whisper backend auto-detection order: `whisperx` → `faster-whisper` → `openai-whisper`
- Logs are written to `~/.opencut/server.log`

---

## License

[MIT](LICENSE)

---

## Credits

Built on the shoulders of:

- [FFmpeg](https://ffmpeg.org) — media processing backbone
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2-accelerated Whisper inference
- [whisperx](https://github.com/m-bain/whisperX) — word-level timestamp alignment
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — speaker diarization
- [Rich](https://github.com/Textualize/rich) — beautiful CLI output
- [auto-editor](https://github.com/WyattBlue/auto-editor) — inspiration and XML format reference
