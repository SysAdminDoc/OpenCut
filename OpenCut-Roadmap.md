# OpenCut Feature Roadmap — Swiss Army Knife for Premiere Pro

## Vision

Transform OpenCut from a silence/filler removal tool into a comprehensive AI-powered editing suite inside Premiere Pro — rivaling CapCut, Opus Clip, and Descript while running 100% locally with open-source models. No subscriptions, no cloud dependency, no data leaving the user's machine.

---

## Architecture Overview

```
Premiere Pro (CEP Panel)
    |
    |-- index.html (Tabbed GUI)
    |-- main.js (API calls + JSX bridge)
    |-- index.jsx (ExtendScript: timeline, markers, XML import)
    |
    v
Flask Server (Python backend)
    |
    |-- /api/cut          Phase 1: Silence & Filler Removal
    |-- /api/captions      Phase 2: Advanced Captions
    |-- /api/audio         Phase 3: Audio Suite
    |-- /api/voice         Phase 4: Voice Lab
    |-- /api/video         Phase 5: Video Intelligence
    |-- /api/reframe       Phase 6: Auto-Reframe
    |-- /api/export        Phase 7: Export & Publish
    |
    v
AI Model Manager (lazy-loads models on first use)
    |-- Whisper (transcription)          ~1.5 GB
    |-- DeepFilterNet (noise reduction)  ~10 MB
    |-- Demucs (stem separation)         ~300 MB
    |-- Qwen3-TTS (voice synthesis)      ~3.4 GB
    |-- MediaPipe (face/pose detection)  ~10 MB
    |-- rembg / BiRefNet (bg removal)    ~170 MB
    |-- RIFE (frame interpolation)       ~110 MB
    |-- Real-ESRGAN (upscaling)          ~65 MB
    |-- PySceneDetect (scene cuts)       ~1 MB
    |-- librosa (beat detection)         ~1 MB
```

### Model Management Strategy

Models are NOT bundled with the installer. On first use of a feature, OpenCut downloads the required model to `%LOCALAPPDATA%\OpenCut\models\` and caches it permanently. A "Model Manager" tab in Settings lets users pre-download, delete, or update models. This keeps the installer under 50 MB while supporting 5+ GB of AI models.

```
%LOCALAPPDATA%\OpenCut\
    opencut-server.exe
    models/
        whisper-base/           ~140 MB (default)
        whisper-large-v3/       ~3 GB (optional upgrade)
        deepfilternet3/         ~10 MB
        demucs-htdemucs/        ~300 MB
        qwen3-tts-0.6b/         ~1.2 GB
        qwen3-tts-1.7b/         ~3.4 GB (optional upgrade)
        mediapipe-face/         ~10 MB
        rembg-u2net/            ~170 MB
        rife-v4.26/             ~110 MB
        realesrgan-x4plus/      ~65 MB
    voices/                     User's cloned voice profiles
    presets/                    Speed ramp, caption style presets
    cache/                      Temporary processing files
```

---

## GUI Layout — Tab-Based Navigation

The panel uses a **horizontal tab bar** at the top (icons + labels) with each tab revealing its own card-based settings area. This scales cleanly from 5 features to 25+ without overwhelming the user.

```
+------------------------------------------+
|  [Logo] OpenCut           [?] [Settings] |
|  ● Connected              v0.5.0         |
+------------------------------------------+
|  [ Cut ] [ Captions ] [ Audio ] [ Voice ]|
|  [ Video ] [ Reframe ] [ Export ]        |
+------------------------------------------+
|                                          |
|  (Active tab content area)               |
|                                          |
|  +------------------------------------+ |
|  |  Card: Primary Feature Controls    | |
|  +------------------------------------+ |
|  +------------------------------------+ |
|  |  Card: Advanced Options            | |
|  +------------------------------------+ |
|                                          |
|  [========= Progress Bar =========]     |
|  Processing: 45% - Analyzing audio...   |
|                                          |
|  [  Run  ]                              |
+------------------------------------------+
```

### Tab Icons (SVG inline, matching current design)

| Tab | Icon | Color When Active |
|-----|------|-------------------|
| Cut | Scissors | Blue |
| Captions | Speech bubble | Cyan |
| Audio | Waveform | Green |
| Voice | Microphone | Purple |
| Video | Film frame | Orange |
| Reframe | Crop/resize | Pink |
| Export | Upload/share | Teal |

---

## Phase 1 — Core Editing (CURRENT — v0.4.0)

**Status: COMPLETE**

### Tab: Cut

| Feature | Tech | Status |
|---------|------|--------|
| Silence detection & removal | FFmpeg audio analysis | Done |
| Filler word detection | Whisper transcription + word matching | Done |
| Adjustable silence threshold | dB slider | Done |
| Min silence duration | ms slider | Done |
| Padding controls | Lead/trail ms | Done |
| FCP XML generation | Python XML builder | Done |
| Full pipeline (silence + fillers + captions) | Combined endpoint | Done |

---

## Phase 2 — Advanced Captions (v0.5.0)

**Estimated effort: 2-3 weeks**

Upgrades the existing caption system to match CapCut's most popular feature — word-by-word highlighting with animated styles.

### Tab: Captions

```
+------------------------------------+
|  Caption Style                     |
|  [ Minimal v ]  [Preview]         |
|                                    |
|  Highlight Mode                    |
|  ( ) None  (x) Word  ( ) Line    |
|                                    |
|  Highlight Color  [#FFD700]       |
|  Text Color       [#FFFFFF]       |
|  Background       [#000000 50%]   |
|                                    |
|  Font Size    [--====------] 48px |
|  Position     ( ) Top  (x) Bottom |
|  Max Words/Line   [--==--------] 4|
|                                    |
|  +-----Advanced----+              |
|  | Animation: Fade In  v          |
|  | Outline: 2px black             |
|  | Shadow: On                     |
|  | Emoji Keywords: On             |
|  +------------------+             |
|                                    |
|  [  Generate Captions  ]          |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| Word-by-word highlighting | Whisper word timestamps -> ASS subtitle per-word styling with `\kf` tags | Whisper `word_timestamps=True` |
| Caption style presets | JSON preset files (font, colors, animations, position) | Built-in |
| 15+ built-in styles | Glow, Minimal, Bold, Outline, Karaoke, Neon, etc. | FFmpeg drawtext + ASS |
| Custom style editor | UI for font/color/size/position | Panel UI |
| Auto-emoji insertion | Keyword -> emoji mapping on common words | spaCy / keyword dict |
| Bilingual subtitles | Dual-line with translation | Whisper + translation API |
| SRT/VTT/ASS export | Standard subtitle file export | Built-in formatters |
| Transcript editor | Edit words in-panel before generating | Panel textarea |

### Technical: Word-Level Highlighting

```python
# Whisper returns word-level timestamps
result = model.transcribe(audio, word_timestamps=True)
# Each segment contains:
# segment.words = [
#   {"word": "Hello", "start": 0.0, "end": 0.4},
#   {"word": "world", "start": 0.5, "end": 0.9},
# ]

# Generate ASS subtitle with per-word highlight timing
# Using \kf (karaoke fill) tags for smooth highlight sweep
# or \1c (primary color) with \t (animation) for color transitions
```

### Caption Style Preset Format

```json
{
    "name": "Neon Glow",
    "fontFamily": "Impact",
    "fontSize": 52,
    "primaryColor": "#FFFFFF",
    "highlightColor": "#00FF88",
    "outlineColor": "#000000",
    "outlineWidth": 3,
    "shadowColor": "#00FF8844",
    "shadowDepth": 4,
    "position": "bottom",
    "maxWordsPerLine": 4,
    "animation": "scale-pop",
    "background": "none"
}
```

---

## Phase 3 — Audio Suite (v0.6.0)

**Estimated effort: 3-4 weeks**

A complete audio toolkit — every feature creators currently leave Premiere for.

### Tab: Audio

```
+------------------------------------+
|  NOISE REDUCTION                   |
|  Intensity  [--======----] Strong |
|  [x] Preserve speech detail       |
|  [  Clean Audio  ]                |
+------------------------------------+
|  VOICE ISOLATION                   |
|  Separate vocals from background   |
|  [x] Export stems separately      |
|  [  Isolate Voice  ]             |
+------------------------------------+
|  LOUDNESS                          |
|  Target  [ YouTube -14 LUFS  v ]  |
|  [x] True peak limiting (-1 dB)  |
|  [  Normalize  ]                  |
+------------------------------------+
|  BEAT SYNC                         |
|  [  Detect Beats  ]              |
|  Found: 127 BPM, 48 beats        |
|  [x] Add as Premiere markers     |
|  [x] Auto-cut on beats           |
+------------------------------------+
|  AUDIO DUCKING                     |
|  Music duck level  [-12 dB v]    |
|  Fade time  [--==--------] 200ms |
|  [  Apply Ducking  ]             |
+------------------------------------+
|  VOICE EFFECTS                     |
|  Effect  [ None           v ]    |
|    Robot | Chipmunk | Deep |      |
|    Echo | Reverb | Lo-Fi |        |
|  [  Preview  ] [  Apply  ]       |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| AI noise reduction | Deep learning speech enhancement | DeepFilterNet3 (`pip install deepfilternet`) |
| Voice isolation | Stem separation (vocals/drums/bass/other) | Demucs v4 (`pip install demucs`) |
| Background music separation | 4-stem or 2-stem separation | Demucs htdemucs model |
| Loudness normalization | LUFS analysis + normalization | FFmpeg `loudnorm` filter |
| Platform presets | YouTube -14, Podcast -16, Broadcast -23 LUFS | FFmpeg loudnorm with targets |
| Beat detection | Tempo & beat position analysis | librosa `beat_track()` |
| Beat markers | Export beats as Premiere markers via XML | FCP XML marker generation |
| Auto-cut on beats | Cut timeline at beat positions | Beat timestamps -> cut list |
| Audio ducking | Auto-lower music during speech | Silero VAD + FFmpeg volume keyframes |
| Voice effects | Pitch shift, echo, reverb, distortion | pedalboard (Spotify) or FFmpeg filters |
| Audio extraction | Extract audio from video | FFmpeg `-vn` |

### Technical: Noise Reduction Pipeline

```python
from df.enhance import enhance, init_df_model, load_audio, save_audio

model, df_state, _ = init_df_model()
audio, sr = load_audio("noisy.wav", sr=df_state.sr())
enhanced = enhance(model, df_state, audio)
save_audio("clean.wav", enhanced, sr)
```

### Technical: Voice Isolation

```python
import demucs.separate

# Separate into vocals + accompaniment
demucs.separate.main([
    "--two-stems", "vocals",
    "-n", "htdemucs",
    "input.wav"
])
# Output: separated/htdemucs/input/vocals.wav
#         separated/htdemucs/input/no_vocals.wav
```

### Technical: Beat Detection

```python
import librosa

y, sr = librosa.load("audio.wav")
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
# beat_times = [0.5, 0.97, 1.44, 1.91, ...] (seconds)
# tempo = 127.3 BPM
```

---

## Phase 4 — Voice Lab (v0.7.0)

**Estimated effort: 4-6 weeks**

The most differentiated feature set — local voice cloning and text-to-speech powered by Qwen3-TTS. No cloud API needed, no per-character costs, unlimited generation.

### Tab: Voice

```
+------------------------------------+
|  TEXT TO SPEECH                     |
|  Voice  [ Sarah (English)    v ]  |
|  +--------------------------------+|
|  | Type or paste your script here ||
|  | ...                            ||
|  +--------------------------------+|
|  Speed  [--====------] 1.0x      |
|  Emotion [ Neutral         v ]   |
|    Calm | Cheerful | Dramatic     |
|  [  Generate Speech  ]           |
|  [>  Preview  ]  [Import to PPro]|
+------------------------------------+
|  VOICE CLONING                     |
|  Record or upload 3-15s of voice   |
|  [  Record  ] or [  Upload WAV  ]|
|                                    |
|  Saved Voices:                     |
|  [x] My Voice          [Delete]  |
|  [ ] Client - Sarah    [Delete]  |
|  [ ] Narrator - Deep   [Delete]  |
|                                    |
|  [  Clone from Selection  ]      |
|  (Uses selected audio in PPro)    |
+------------------------------------+
|  VOICE REPLACE                     |
|  Replace spoken words in video     |
|  Original: "We had forty users"   |
|  Replace:  "We had four thousand" |
|  [  Generate Replacement  ]      |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| Text-to-speech (17 voices) | Local TTS with preset speakers | Qwen3-TTS-12Hz-0.6B-Instruct |
| 10 language support | EN, CN, JP, KR, DE, FR, RU, PT, ES, IT | Qwen3-TTS multilingual |
| Voice cloning (3s sample) | Clone from audio file or recording | Qwen3-TTS-12Hz-1.7B-Base |
| Emotion/tone control | Natural language prompting for voice style | Qwen3-TTS voice design |
| Voice design by description | "A warm female voice with slight British accent" | Qwen3-TTS-12Hz-1.7B-VD |
| Voice profiles library | Save/load cloned voice embeddings | Local JSON + speaker embeddings |
| Word replacement | Re-synthesize specific words in cloned voice | Qwen3-TTS clone + splice |
| Speed/pitch control | Adjust speaking rate and pitch | Model parameters + FFmpeg |
| Multi-voice dialogue | Multiple voices in one script | Voice bank + dialogue parser |
| Audio preview | In-panel playback before importing | HTML5 Audio element |

### Technical: Qwen3-TTS Integration

```python
import torch
from qwen_tts import Qwen3TTSModel

# Load model (lazy, cached after first load)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Instruct",
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16,
)

# Text-to-speech with preset voice
wavs, sr = model.generate(
    text="Welcome to our product demo.",
    speaker="Chelsie",
)

# Voice cloning from 3-second sample
wavs, sr = model.generate_voice_clone(
    text="This is my cloned voice speaking new words.",
    ref_audio="voice_sample.wav",
    ref_text="The original words spoken in the sample.",
)
```

### Technical: Voicebox Integration (Alternative/Complementary)

```python
# Voicebox (github.com/jamiepine/voicebox) provides a full
# studio UI, but we integrate its backend directly:
# - Uses Qwen3-TTS under the hood
# - Adds multi-track timeline editing
# - Voice bank management
# We wrap the same Qwen3-TTS models but with our own API
```

### GPU vs CPU Handling

```python
# Auto-detect GPU and adjust model size
import torch

def get_voice_model():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram >= 8:
            return "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Best quality
        else:
            return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"   # Good quality
    else:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"       # CPU fallback (slower)
```

---

## Phase 5 — Video Intelligence (v0.8.0)

**Estimated effort: 4-5 weeks**

AI-powered video analysis and enhancement features.

### Tab: Video

```
+------------------------------------+
|  SCENE DETECTION                   |
|  Sensitivity  [--====------] Med  |
|  [x] Add as Premiere markers     |
|  [x] Auto-generate chapter list  |
|  [  Detect Scenes  ]             |
|  Found: 23 scenes                 |
+------------------------------------+
|  SPEED RAMPING                     |
|  Preset  [ Montage       v ]     |
|    Hero | Bullet | Smooth |       |
|    Montage | Dramatic | Custom    |
|  Speed range: 0.25x - 4.0x       |
|  [  Apply Speed Ramp  ]          |
+------------------------------------+
|  BACKGROUND REMOVAL                |
|  Model  [ BiRefNet (Best)  v ]   |
|  [x] Replace with: [ Blur   v ] |
|    Blur | Solid Color | Image     |
|  [  Remove Background  ]         |
|  ~2 min/30s video @ 1080p        |
+------------------------------------+
|  SLOW MOTION (Optical Flow)       |
|  Target: [ 0.25x (4x frames) v ]|
|  Quality [ High            v ]   |
|  [  Generate Slow Motion  ]      |
|  Requires NVIDIA GPU              |
+------------------------------------+
|  UPSCALING                         |
|  Scale  ( ) 2x  (x) 4x          |
|  Model  [ Real-ESRGAN     v ]   |
|  [  Upscale Video  ]             |
|  ~5 min/30s video @ 720p->4K     |
+------------------------------------+
|  CHROMA KEY                        |
|  Key Color  [ Green        v ]   |
|  Tolerance  [--====------]       |
|  Edge Feather [--==--------]     |
|  [  Apply Chroma Key  ]          |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| Scene detection | Visual + audio scene boundary detection | PySceneDetect (`pip install scenedetect[opencv]`) |
| Chapter markers | Scene boundaries -> Premiere markers + YouTube chapters | PySceneDetect + FCP XML |
| Speed ramp presets | Predefined velocity curves applied via FFmpeg | FFmpeg `setpts` + JSON presets |
| Custom speed curves | User-adjustable Bezier curve for speed | Panel Canvas + FFmpeg |
| Background removal (image) | AI segmentation + alpha matte | rembg (`pip install rembg`) |
| Background removal (video) | Frame-by-frame with temporal consistency | rembg + SAM2 tracking |
| Background replacement | Replace with blur, color, or image | FFmpeg composite + rembg masks |
| Optical flow slow-motion | AI frame interpolation for smooth slo-mo | RIFE v4.26 (`pip install`) |
| Video upscaling | AI super-resolution 2x or 4x | Real-ESRGAN (`pip install realesrgan`) |
| Chroma key / green screen | Color-based keying with edge refinement | FFmpeg `chromakey` + OpenCV |
| LUT application | Apply .cube LUT files | FFmpeg `lut3d` filter |

### Technical: Scene Detection

```python
from scenedetect import detect, ContentDetector, AdaptiveDetector

scene_list = detect("input.mp4", ContentDetector(threshold=27.0))
# scene_list = [(0.0, 5.2), (5.2, 12.8), (12.8, 18.4), ...]

# Or adaptive (better for gradual transitions):
scene_list = detect("input.mp4", AdaptiveDetector())
```

### Speed Ramp Preset Format

```json
{
    "name": "Hero Moment",
    "description": "Dramatic slowdown then snap back to speed",
    "keyframes": [
        {"position": 0.0,  "speed": 1.0},
        {"position": 0.3,  "speed": 1.0},
        {"position": 0.4,  "speed": 0.2, "easing": "ease-out"},
        {"position": 0.6,  "speed": 0.2},
        {"position": 0.7,  "speed": 2.0, "easing": "ease-in"},
        {"position": 1.0,  "speed": 1.0, "easing": "ease-out"}
    ]
}
```

### Technical: Background Removal

```python
from rembg import remove, new_session
import cv2

session = new_session("birefnet-general")  # Best quality model

# Single image
with open("frame.png", "rb") as f:
    output = remove(f.read(), session=session)

# Video pipeline (frame-by-frame)
cap = cv2.VideoCapture("input.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = remove(frame, session=session)
    # Write result with alpha channel
```

### Technical: RIFE Slow Motion

```python
# Using RIFE for 4x frame interpolation (0.25x speed)
import subprocess

subprocess.run([
    "python", "inference_video.py",
    "--exp=2",           # 2^2 = 4x frames
    "--video=input.mp4",
    "--output=slowmo.mp4",
    "--scale=1.0",       # 0.5 for 4K to save memory
])
```

---

## Phase 6 — Auto-Reframe (v0.9.0)

**Estimated effort: 3-4 weeks**

The #1 most-requested creator feature — automatically convert landscape video to portrait for TikTok/Reels/Shorts with intelligent subject tracking.

### Tab: Reframe

```
+------------------------------------+
|  AUTO-REFRAME                      |
|  Target  [ 9:16 Portrait    v ]  |
|    9:16 | 1:1 | 4:5 | 4:3 | Custom|
|                                    |
|  Tracking  [ Face Priority  v ]  |
|    Face Priority | Center |       |
|    Motion Follow | Manual         |
|                                    |
|  Smoothness  [--======----] High |
|  (Camera movement smoothing)      |
|                                    |
|  Padding  [ Blur Background v ]  |
|    Blur | Black | Solid Color     |
|                                    |
|  [x] Split-screen for 2+ people  |
|  [x] Auto-zoom on active speaker |
|                                    |
|  [  Reframe Video  ]             |
|  ~1 min/30s video                 |
+------------------------------------+
|  MANUAL KEYFRAMES                  |
|  Override auto-tracking at specific|
|  timecodes                         |
|  [  Add Keyframe at Current  ]   |
|  00:15  Center (auto)             |
|  00:32  Left 30% (manual)  [x]   |
|  01:05  Center (auto)             |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| Face detection + tracking | Real-time face position per frame | MediaPipe Face Detection |
| Person detection | Full body tracking for wide shots | MediaPipe Pose / Holistic |
| Smooth camera path | Bezier interpolation of crop positions | scipy interpolation + smoothing |
| Multi-person handling | Split-screen or priority tracking | MediaPipe multi-face + logic |
| Active speaker detection | Lip movement analysis to find who's talking | MediaPipe FaceMesh landmarks |
| Letterbox/pillarbox padding | Blur, color, or extended background | FFmpeg `pad` + `boxblur` |
| Aspect ratio presets | 9:16, 1:1, 4:5, 4:3, custom | FFmpeg `crop` + `scale` |
| Manual keyframe override | User-specified crop positions at timecodes | JSON keyframe list |
| Batch reframe | Process multiple clips at once | Queue system |

### Technical: Auto-Reframe Pipeline

```python
import mediapipe as mp
import cv2
import numpy as np
from scipy.signal import savgol_filter

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)

def get_face_positions(video_path):
    """Extract face center positions for every frame."""
    cap = cv2.VideoCapture(video_path)
    positions = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(rgb)
        if results.detections:
            # Use highest-confidence face
            det = max(results.detections, key=lambda d: d.score[0])
            bbox = det.location_data.relative_bounding_box
            cx = bbox.xmin + bbox.width / 2
            cy = bbox.ymin + bbox.height / 2
            positions.append((cx, cy))
        else:
            positions.append(None)
    return positions

def smooth_camera_path(positions, window=31):
    """Apply Savitzky-Golay filter for smooth camera movement."""
    xs = [p[0] if p else 0.5 for p in positions]
    ys = [p[1] if p else 0.5 for p in positions]
    # Interpolate missing positions
    # ... (fill gaps)
    xs_smooth = savgol_filter(xs, window, 3)
    ys_smooth = savgol_filter(ys, window, 3)
    return list(zip(xs_smooth, ys_smooth))

# pyautoflip provides a higher-level API:
from pyautoflip import reframe_video
reframe_video(
    input_path="landscape.mp4",
    output_path="portrait.mp4",
    target_aspect_ratio="9:16",
    motion_threshold=0.3,
    padding_method="blur"
)
```

---

## Phase 7 — Export & Publish (v1.0.0)

**Estimated effort: 2-3 weeks**

Platform-optimized export, transcript generation, and publishing workflows.

### Tab: Export

```
+------------------------------------+
|  TRANSCRIPT                        |
|  Format  [ Timestamped     v ]   |
|    Plain Text | Timestamped |     |
|    Blog Post | Show Notes |       |
|    YouTube Chapters               |
|  [x] Include speaker labels      |
|  [  Generate Transcript  ]       |
+------------------------------------+
|  YOUTUBE CHAPTERS                  |
|  Auto-detect topic changes         |
|  Min chapter length  [ 30s  v ]  |
|  [  Generate Chapters  ]         |
|  00:00 Introduction               |
|  01:24 Setup and Installation      |
|  03:45 Advanced Configuration      |
|  [  Copy to Clipboard  ]         |
+------------------------------------+
|  PLATFORM PRESETS                  |
|  Quick export settings for:        |
|  [YouTube] [TikTok] [Instagram]  |
|  [Twitter] [LinkedIn] [Podcast]  |
|                                    |
|  Selected: YouTube                 |
|  Resolution: 1920x1080             |
|  Bitrate: 10 Mbps                  |
|  Audio: AAC 320kbps, -14 LUFS     |
|  [  Apply to Premiere Export  ]   |
+------------------------------------+
|  B-ROLL SUGGESTIONS                |
|  AI-detected moments for B-roll:   |
|  00:45 "the dashboard shows..."   |
|  02:12 "our growth last quarter"  |
|  [  Add as Markers  ]            |
+------------------------------------+
```

### Features

| Feature | Implementation | Open-Source Tech |
|---------|---------------|-----------------|
| Transcript export (6 formats) | Whisper transcription -> formatted output | Whisper + text formatting |
| YouTube chapters | Scene detection + topic analysis | PySceneDetect + keyword grouping |
| Speaker diarization | Label who's speaking when | pyannote.audio or whisperx |
| Platform export presets | Optimal settings per platform | JSON presets -> Premiere AME |
| B-roll insertion points | Transcript analysis for visual moments | NLP keyword detection (spaCy) |
| Timeline markers | Add markers at detected points | FCP XML / ExtendScript |
| Batch processing | Queue multiple videos | Threading + progress tracking |
| Project templates | Save/load full OpenCut configurations | JSON preset files |

---

## Settings & Model Manager

### Settings Tab (gear icon)

```
+------------------------------------+
|  GENERAL                           |
|  Server Port  [ 5000 ]           |
|  [x] Auto-launch with Premiere   |
|  [x] Check for updates           |
|  Theme  [ Dark (Default)    v ]  |
+------------------------------------+
|  PERFORMANCE                       |
|  GPU  [ Auto-detect        v ]   |
|    Auto | NVIDIA CUDA | CPU Only  |
|  Max threads  [ 8           v ]  |
|  [x] Low-memory mode (< 8GB)    |
+------------------------------------+
|  MODEL MANAGER                     |
|  +------------------------------+ |
|  | Model          Size  Status  | |
|  | Whisper base   140MB  Ready  | |
|  | Whisper large  3GB  [Get]   | |
|  | DeepFilterNet  10MB   Ready  | |
|  | Demucs         300MB [Get]  | |
|  | Qwen3-TTS 0.6B 1.2GB [Get] | |
|  | Qwen3-TTS 1.7B 3.4GB [Get] | |
|  | MediaPipe      10MB   Ready  | |
|  | rembg          170MB [Get]  | |
|  | RIFE           110MB [Get]  | |
|  | Real-ESRGAN    65MB  [Get]  | |
|  +------------------------------+ |
|  Storage used: 450 MB / 8.3 GB   |
|  Models: %LOCALAPPDATA%\OpenCut\  |
+------------------------------------+
|  VOICE LIBRARY                     |
|  Manage saved voice profiles       |
|  +------------------------------+ |
|  | My Voice        EN    3.2s   | |
|  | Client Sarah    EN    8.1s   | |
|  +------------------------------+ |
|  [  Import Voice  ]              |
+------------------------------------+
```

---

## Python Dependencies by Phase

### Phase 1 (Current)
```
flask, flask-cors
faster-whisper (or openai-whisper)
rich (logging)
ffmpeg-python (or subprocess calls)
```

### Phase 2 (Captions)
```
# No new heavy deps — uses existing Whisper + FFmpeg
pysubs2          # ASS/SRT subtitle manipulation
Pillow           # Caption image rendering (alternative to FFmpeg drawtext)
```

### Phase 3 (Audio)
```
deepfilternet    # AI noise reduction (~10 MB model)
demucs           # Stem separation (~300 MB model)
librosa          # Beat detection, audio analysis
soundfile        # Audio I/O
pedalboard       # Audio effects (Spotify's library)
pyloudnorm       # LUFS loudness measurement
```

### Phase 4 (Voice)
```
qwen-tts         # Qwen3-TTS (0.6B: ~1.2 GB, 1.7B: ~3.4 GB)
torch            # PyTorch (required for Qwen3-TTS)
torchaudio       # Audio processing for TTS
sounddevice      # Microphone recording for voice cloning
```

### Phase 5 (Video)
```
scenedetect[opencv]  # Scene detection
rembg                # Background removal (~170 MB model)
# RIFE — cloned from GitHub, not pip (model files ~110 MB)
# Real-ESRGAN — pip install realesrgan (~65 MB model)
opencv-python        # Video frame processing
```

### Phase 6 (Reframe)
```
mediapipe        # Face/pose detection (~10 MB)
pyautoflip       # High-level reframing API
scipy            # Camera path smoothing
numpy            # Array operations
```

### Phase 7 (Export)
```
spacy            # NLP for B-roll detection, keyword extraction
# pyannote.audio or whisperx — speaker diarization
```

---

## Server API Endpoints (Full)

```
Phase 1 (Current):
  POST /api/detect-silences     Analyze audio for silence segments
  POST /api/detect-fillers      Find filler words via transcription
  POST /api/generate-xml        Generate FCP XML edit list
  POST /api/process             Full pipeline (silence + fillers + captions)

Phase 2 (Captions):
  POST /api/captions/generate   Generate word-level captions
  POST /api/captions/styles     List available caption presets
  POST /api/captions/preview    Render a preview frame with caption
  POST /api/captions/export     Export SRT/VTT/ASS file

Phase 3 (Audio):
  POST /api/audio/denoise       Run noise reduction
  POST /api/audio/isolate       Separate vocals from background
  POST /api/audio/normalize     LUFS normalization
  POST /api/audio/beats         Detect beats and tempo
  POST /api/audio/duck          Generate ducking keyframes
  POST /api/audio/effects       Apply voice effects

Phase 4 (Voice):
  GET  /api/voice/speakers      List available TTS voices
  POST /api/voice/generate      Text-to-speech generation
  POST /api/voice/clone         Create voice profile from audio
  POST /api/voice/replace       Replace words in audio
  GET  /api/voice/profiles      List saved voice profiles
  DELETE /api/voice/profiles/:id  Delete a voice profile

Phase 5 (Video):
  POST /api/video/scenes        Detect scene boundaries
  POST /api/video/speed-ramp    Apply speed ramp preset
  POST /api/video/bg-remove     Remove background from video
  POST /api/video/slowmo        Generate slow-motion via RIFE
  POST /api/video/upscale       AI upscale video
  POST /api/video/chromakey     Apply chroma key

Phase 6 (Reframe):
  POST /api/reframe/analyze     Analyze video for face positions
  POST /api/reframe/generate    Generate reframed video
  POST /api/reframe/preview     Preview reframe at specific timecode

Phase 7 (Export):
  POST /api/export/transcript   Generate formatted transcript
  POST /api/export/chapters     Auto-generate YouTube chapters
  POST /api/export/broll        Detect B-roll insertion points
  POST /api/export/markers      Generate Premiere markers XML

System:
  GET  /api/health              Server health check
  GET  /api/models              List installed models + status
  POST /api/models/download     Download a model
  DELETE /api/models/:name      Delete a model
  GET  /api/gpu                 GPU detection and VRAM info
```

---

## Release Timeline

| Phase | Version | Features | Est. Time | Installer Size |
|-------|---------|----------|-----------|----------------|
| 1 | v0.4.0 | Cut, Silence, Fillers, Basic Captions | **DONE** | ~50 MB |
| 2 | v0.5.0 | Advanced Captions, Word Highlighting, Styles | 2-3 weeks | ~50 MB |
| 3 | v0.6.0 | Audio Suite (denoise, isolate, beats, duck) | 3-4 weeks | ~50 MB |
| 4 | v0.7.0 | Voice Lab (TTS, cloning, voice design) | 4-6 weeks | ~50 MB |
| 5 | v0.8.0 | Video Intelligence (scenes, BG remove, slo-mo) | 4-5 weeks | ~50 MB |
| 6 | v0.9.0 | Auto-Reframe (face tracking, multi-platform) | 3-4 weeks | ~50 MB |
| 7 | v1.0.0 | Export Suite (transcripts, chapters, presets) | 2-3 weeks | ~50 MB |

**Total development: ~6-8 months to feature-complete v1.0.0**

Note: Installer stays small (~50 MB) because AI models are downloaded on-demand. Total model storage if user installs everything: ~5-8 GB.

---

## Hardware Requirements

| Tier | GPU | RAM | Experience |
|------|-----|-----|------------|
| Minimum | None (CPU) | 8 GB | Cut, Captions, Audio, Export work well. Voice/Video are slow. |
| Recommended | NVIDIA GTX 1660+ (6GB VRAM) | 16 GB | All features work. Voice gen ~5s. BG removal ~3 fps. |
| Optimal | NVIDIA RTX 3060+ (8GB+ VRAM) | 32 GB | Everything fast. Voice gen <2s. BG removal ~8 fps. Slo-mo real-time. |

---

## Open-Source Model Reference

| Model | License | Size | GPU Required | Used For |
|-------|---------|------|-------------|----------|
| Whisper (OpenAI) | MIT | 140MB-3GB | No (faster w/ GPU) | Transcription, word timestamps |
| DeepFilterNet3 | MIT | ~10 MB | No | Noise reduction |
| Demucs v4 (Meta) | MIT | ~300 MB | No (faster w/ GPU) | Stem separation, voice isolation |
| Qwen3-TTS (Alibaba) | Apache 2.0 | 1.2-3.4 GB | Recommended | TTS, voice cloning, voice design |
| MediaPipe (Google) | Apache 2.0 | ~10 MB | No | Face detection, pose tracking |
| rembg / BiRefNet | MIT | ~170 MB | No (faster w/ GPU) | Background removal |
| RIFE v4.26 | MIT | ~110 MB | Yes (NVIDIA) | Frame interpolation, slo-mo |
| Real-ESRGAN | BSD-3 | ~65 MB | Yes (NVIDIA) | Video/image upscaling |
| PySceneDetect | BSD-3 | ~1 MB | No | Scene boundary detection |
| librosa | ISC | ~1 MB | No | Beat detection, audio analysis |
| pedalboard (Spotify) | GPL-3.0 | ~5 MB | No | Audio effects |
| pyautoflip | MIT | ~1 MB | No | Video reframing |
| spaCy | MIT | ~50 MB | No | NLP, keyword extraction |

All models: **Apache 2.0, MIT, BSD, or ISC licensed** — safe for commercial use (except pedalboard GPL-3 which only applies if distributing modified source).

---

## Competitive Positioning

| Feature | CapCut | Descript | Opus Clip | OpenCut |
|---------|--------|---------|-----------|---------|
| Silence removal | Yes | Yes | Yes | **Yes** |
| Filler word removal | Yes | Yes | No | **Yes** |
| Auto captions | Yes | Yes | Yes | **Yes** |
| Word highlighting | Yes | No | No | **Phase 2** |
| Noise reduction | Yes | Yes | No | **Phase 3** |
| Voice isolation | No | Yes | No | **Phase 3** |
| Beat sync | Yes | No | No | **Phase 3** |
| Text-to-speech | Yes (cloud) | No | No | **Phase 4 (local)** |
| Voice cloning | Yes (cloud) | Yes (cloud) | No | **Phase 4 (local)** |
| Background removal | Yes | No | No | **Phase 5** |
| Slow motion (AI) | Yes | No | No | **Phase 5** |
| Auto-reframe | Yes | No | Yes | **Phase 6** |
| YouTube chapters | No | Yes | Yes | **Phase 7** |
| Runs inside Premiere | No | No | No | **Yes** |
| 100% local/offline | No | No | No | **Yes** |
| One-time purchase | No ($7.99/mo) | No ($24/mo) | No ($19/mo) | **Yes** |
| Open source | No | No | No | **Yes** |
