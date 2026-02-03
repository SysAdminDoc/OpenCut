# OpenCut

**Open source video editing automation for Premiere Pro.**

Remove silences, generate captions, automate podcast multicam switching, and add dynamic zoom. Works as a **panel inside Premiere Pro** or from the **command line**. Exports Premiere Pro XML for non-destructive editing.

> Open-source alternative to [AutoCut](https://www.autocut.com), [FireCut](https://firecut.co), and [TimeBolt](https://www.timebolt.io).

---

## One-Click Install

### Prerequisites

- **Windows 10/11**
- **Python 3.9+** ([download](https://www.python.org/downloads/) -- check "Add Python to PATH")
- **Premiere Pro 2023+** (v23.0 or newer)

### Install

1. Download/clone this repository
2. **Double-click `Install.bat`**
3. Click "Yes" if prompted by UAC

That's it. The installer handles everything:

- Installs FFmpeg (via winget)
- Installs all Python dependencies
- Installs the CEP extension into Premiere Pro
- Sets the registry key for unsigned extensions
- Creates a desktop shortcut to launch the backend

### Use in Premiere Pro

1. **Double-click `Start-OpenCut.bat`** (or the desktop shortcut) -- starts the backend server
2. Open **Premiere Pro**
3. Go to **Window > Extensions > OpenCut**
4. Select a clip or drop a file into the panel
5. Choose your settings and click **Remove Silences**
6. Click **Import XML into Premiere** when done

The generated XML creates a new sequence in your project with all silences removed.

---

## Features

| Feature | Status | How It Works |
|---------|--------|-------------|
| **Silence Removal** | Ready | FFmpeg detects silence, exports XML with only speech segments |
| **Premiere Pro Panel** | Ready | CEP extension with dark UI, talks to Python backend |
| **One-Click Installer** | Ready | PowerShell script handles all prerequisites |
| **Caption Generation** | Ready | Whisper AI transcription to SRT/VTT |
| **Auto Zoom** | Ready | Audio energy analysis for zoom keyframes |
| **Podcast Multicam** | Beta | pyannote speaker diarization for camera switching |
| **CLI** | Ready | Full command-line interface for automation |

---

## Command Line Usage

The CLI works independently of Premiere Pro:

```bash
# Remove silences (exports Premiere XML)
opencut silence video.mp4

# Use a preset
opencut silence video.mp4 --preset aggressive

# Custom settings
opencut silence video.mp4 -t -25 -d 0.3 --padding-before 0.08

# Generate captions
opencut captions video.mp4 --model small --language en

# Full pipeline (silence + zoom + captions)
opencut full video.mp4 --preset youtube

# Show media info
opencut info video.mp4
```

---

## Presets

| Preset | Threshold | Min Silence | Padding | Best For |
|--------|-----------|-------------|---------|----------|
| `default` | -30 dB | 0.5s | 0.1s | General use |
| `aggressive` | -25 dB | 0.3s | 0.05s | Fast-paced YouTube |
| `conservative` | -40 dB | 1.0s | 0.2s | Interviews, quiet content |
| `podcast` | -35 dB | 0.75s | 0.15s | Podcasts, conversations |
| `youtube` | -28 dB | 0.4s | 0.08s | YouTube with captions + zoom |

---

## Architecture

```
Install.bat / Install.ps1       <-- Double-click to install everything
Start-OpenCut.bat               <-- Launches the backend server

opencut/                         Python Package
  server.py                      Flask API (localhost:5679)
  cli.py                         Command-line interface
  core/
    silence.py                   FFmpeg silence detection
    captions.py                  Whisper transcription
    diarize.py                   pyannote speaker diarization
    zoom.py                      Audio energy zoom keyframes
    audio.py                     PCM extraction, RMS analysis
  export/
    premiere.py                  FCP 7 XML generation
    srt.py                       SRT/VTT/JSON subtitle export
  utils/
    media.py                     ffprobe metadata
    config.py                    Settings + presets

extension/com.opencut.panel/     CEP Extension (Premiere Pro Panel)
  CSXS/manifest.xml              Extension manifest
  client/
    index.html                   Panel UI
    main.js                      Backend communication + job management
    style.css                    Dark theme matching Premiere
  host/
    index.jsx                    ExtendScript (file browse, XML import)
```

### Data Flow

```
Premiere Pro Panel (CEP)
  |  HTTP requests to localhost:5679
  v
Flask Backend Server
  |  Processes media using:
  |--- FFmpeg (silence detection)
  |--- Whisper (speech-to-text)
  |--- pyannote (speaker ID)
  v
Generated Files
  |--- video_opencut.xml  --> Import into Premiere Pro
  |--- video.srt          --> Subtitle file
```

---

## Manual Installation

If you prefer not to use the installer:

```bash
# 1. Install FFmpeg
winget install Gyan.FFmpeg

# 2. Install Python dependencies
pip install -e .

# 3. (Optional) Install Whisper for captions
pip install faster-whisper

# 4. Copy extension to Adobe folder
xcopy /E /I "extension\com.opencut.panel" "%APPDATA%\Adobe\CEP\extensions\com.opencut.panel"

# 5. Enable unsigned extensions (run in PowerShell as admin)
New-Item -Path "HKCU:\Software\Adobe\CSXS.12" -Force
Set-ItemProperty -Path "HKCU:\Software\Adobe\CSXS.12" -Name "PlayerDebugMode" -Value "1"

# 6. Start the backend
python -m opencut.server

# 7. Open Premiere Pro > Window > Extensions > OpenCut
```

---

## Uninstall

Double-click `Uninstall.bat`, or run:

```powershell
.\Install.ps1 -Uninstall
```

---

## Whisper Model Guide

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| `tiny` | 39M | Fastest | Basic | ~1 GB |
| `base` | 74M | Fast | Good (default) | ~1 GB |
| `small` | 244M | Medium | Very Good | ~2 GB |
| `medium` | 769M | Slow | Excellent | ~5 GB |
| `large-v3` | 1.5G | Slowest | Best | ~10 GB |
| `turbo` | 809M | Fast | Near-best | ~6 GB |

---

## Contributing

```bash
git clone https://github.com/opencut/opencut.git
cd opencut
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT

---

## Credits

Built on: [FFmpeg](https://ffmpeg.org), [OpenAI Whisper](https://github.com/openai/whisper), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [pyannote.audio](https://github.com/pyannote/pyannote-audio), [auto-editor](https://github.com/WyattBlue/auto-editor) (inspiration + XML reference)
