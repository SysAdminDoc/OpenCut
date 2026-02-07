# OpenCut v0.5.0+ Build Plan
## Swiss Army Knife Expansion

---

## Current State (v0.4.0)

### Backend (server.py - 2195 lines)
- 20+ Flask endpoints: silence, fillers, captions, styled-captions, full pipeline
- Audio Suite endpoints: /audio/denoise, /audio/isolate, /audio/normalize, /audio/measure, /audio/beats, /audio/effects
- Video endpoints: /video/scenes, /video/speed-presets
- System: /system/gpu, SSE streaming, job management, install-whisper

### Core Modules
- `silence.py` - Speech detection, edit summaries
- `fillers.py` - Filler word detection/removal (um, uh, like, etc.)
- `captions.py` - Whisper transcription (3 backends: faster-whisper, openai-whisper, whisperx)
- `styled_captions.py` - 6 caption styles, word-by-word highlight, action word detection, Pillow+FFmpeg rendering
- `zoom.py` - Auto-zoom events based on audio energy
- `audio.py` - PCM/WAV extraction utilities
- `audio_suite.py` - Noise reduction, voice isolation, loudness normalization, beat detection, audio effects (718 lines)
- `scene_detect.py` - Scene boundary detection, chapter marker generation, speed ramp presets (309 lines)
- `diarize.py` - Speaker diarization (pyannote)

### Frontend (CEP Panel)
- 4 tabs: Silence, Fillers, Captions, Full Edit
- Premium dark theme with design tokens
- Auto-import to Premiere Pro, SSE job streaming, port scanning
- 299-line HTML, 1058-line CSS, 843-line JS

### What's Missing
The backend has Audio Suite + Scene Detection + GPU endpoints fully wired, but the **frontend only exposes 4 tabs**. All audio/video/export features have NO GUI.

---

## PHASE 1: Core Editing (v0.4.0) - COMPLETE

Already built:
- Silence detection & removal with presets
- Filler word detection & removal (12 filler types + custom)
- Basic captions (SRT/VTT/JSON export via Whisper)
- Styled caption overlays (6 styles, word-by-word highlight, action words)
- Full pipeline (silence + zoom + captions + fillers combined)
- Auto-import into Premiere Pro
- Job system with SSE streaming and polling fallback
- Inno Setup installer with PyInstaller single-exe

---

## PHASE 2: GUI Overhaul + Advanced Captions (v0.5.0)

**Goal**: Transform the 4-tab panel into a 7-tab professional toolkit and expand captions.

### 2A: GUI Architecture Overhaul
- New horizontal icon tab bar: Cut | Captions | Audio | Video | Export | Settings
- "Cut" tab consolidates: Silence removal, Filler removal, Full pipeline
- Collapsible card sections within each tab
- Smooth tab transitions
- Persistent file selection across all tabs
- Updated version badge and header

### 2B: Advanced Captions
- **9 new caption styles** (15 total): Glow, Karaoke, Outline, Gradient, Typewriter, Bounce, Comic, Subtitle, News Ticker
- **ASS subtitle export** with word-by-word karaoke timing (\kf tags)
- **Custom style editor**: font picker, color pickers (text/highlight/action/stroke), size slider, position control
- **Auto-emoji insertion**: keyword->emoji mapping (laugh->laughing emoji, fire->fire emoji, etc.)
- **Transcript editor panel**: view/edit transcribed text, re-export captions
- **Caption preview**: live preview of style in the panel with sample text

### Files Modified
- `extension/com.opencut.panel/client/index.html` - Complete rewrite (new tab structure)
- `extension/com.opencut.panel/client/style.css` - Expanded for new tabs/components
- `extension/com.opencut.panel/client/main.js` - Complete rewrite (tab routing, new API calls)
- `opencut/core/styled_captions.py` - Add 9 new styles
- `opencut/export/srt.py` - Add ASS export function
- `opencut/server.py` - Add /captions/custom-style and /captions/transcript-edit endpoints

---

## PHASE 3: Audio Suite Panel (v0.6.0)

**Goal**: Wire existing audio backend to the new Audio tab GUI.

### Features
- **Noise Reduction**: Method selector (afftdn/highpass+lowpass), strength slider, one-click denoise
- **Voice Isolation**: Single-button voice emphasis with bandpass filtering
- **Loudness Normalization**: Platform presets (YouTube -14 LUFS, Podcast -16, Broadcast -23, Spotify -14, Apple -16), custom LUFS input, loudness meter display
- **Beat Detection**: BPM display, beat visualization, export beat markers to Premiere timeline
- **Audio Ducking**: VAD-based automatic music ducking during speech
- **Audio Effects**: Reverb, echo, pitch shift, bass boost, treble boost, telephone, radio, slow-mo voice
- **Loudness Meter**: Real-time LUFS readout before/after processing

### Files Modified
- `extension/.../index.html` - Add Audio tab panels
- `extension/.../style.css` - Audio-specific UI components (meters, waveform)
- `extension/.../main.js` - Audio API integration, beat visualization
- `opencut/server.py` - Add /audio/duck endpoint

---

## PHASE 4: Video Intelligence Panel (v0.7.0)

**Goal**: Wire existing video backend + add new features to Video tab GUI.

### Features
- **Scene Detection**: Sensitivity slider, scene list with timestamps, one-click scene markers
- **Chapter Markers**: Auto-generate YouTube chapters from scenes, copy-to-clipboard
- **Speed Ramp Presets**: Ramp In, Ramp Out, Pulse, Heartbeat, Smooth Slow-Mo - applied via FCP XML
- **Custom Speed Curves**: Visual Bezier curve editor for custom velocity profiles
- **Auto-Reframe**: Aspect ratio presets (9:16, 1:1, 4:5), face tracking crop via FFmpeg cropdetect + drawbox
- **LUT Application**: Load .cube/.3dl LUT files, apply color grading via FFmpeg lut3d

### New Backend
- `opencut/core/speed_ramp.py` - Speed curve generation, XML speed keyframes
- `opencut/core/reframe.py` - Aspect ratio conversion with face-tracking crop
- `opencut/server.py` - Add /video/speed-ramp, /video/reframe, /video/lut endpoints

### Files Modified
- `extension/.../index.html` - Add Video tab panels
- `extension/.../style.css` - Speed curve editor, scene list, LUT preview
- `extension/.../main.js` - Video API integration, curve editor widget

---

## PHASE 5: Export & Publish Panel (v0.8.0)

**Goal**: Comprehensive export tools for multi-platform publishing.

### Features
- **Transcript Export**: 6 formats - Plain text, timestamped text, blog post, show notes, YouTube description, social media clips
- **YouTube Chapters**: Scene-based chapter generation with topic labels, copy-to-clipboard
- **Platform Presets**: One-click export profiles for YouTube, TikTok, Instagram Reels, Twitter/X, LinkedIn, Podcast (audio-only)
- **B-Roll Insertion Points**: NLP keyword detection marking where B-roll footage should go
- **Batch Processing**: Queue multiple files for sequential processing
- **Project Templates**: Save/load processing configurations as reusable presets

### New Backend
- `opencut/core/transcript_export.py` - Multi-format transcript generation
- `opencut/core/broll_detect.py` - NLP keyword extraction for B-roll markers
- `opencut/server.py` - Add /export/* endpoints

### Files Modified
- `extension/.../index.html` - Add Export tab panels
- `extension/.../style.css` - Export cards, platform icons, batch queue
- `extension/.../main.js` - Export API integration, clipboard, batch queue

---

## PHASE 6: Settings, Polish & Ship (v1.0.0)

**Goal**: Settings tab, final polish, installer update, documentation.

### Features
- **Settings Tab**: Default model selection, output directory, auto-import toggle, theme customization
- **Model Manager**: View installed models, download/delete on-demand, storage usage display
- **Keyboard Shortcuts**: Expanded shortcut system for power users
- **Onboarding**: First-run welcome screen with feature tour
- **Error Recovery**: Improved error messages, retry buttons, diagnostic info
- **Performance**: Lazy-load tab contents, debounced API calls, memory cleanup

### Build Updates
- Updated Inno Setup installer for v1.0.0
- Updated PyInstaller spec
- Updated README with full feature list
- CHANGELOG.md
- GitHub release automation

### Files Modified
- `extension/.../index.html` - Add Settings tab
- `extension/.../style.css` - Settings components, model manager
- `extension/.../main.js` - Settings persistence, model management
- `build/installer.iss` - Version bump
- `build/opencut.spec` - Include new modules
- `README.md` - Complete rewrite
- `pyproject.toml` - Version bump, new dependencies

---

## Architecture Overview

```
extension/com.opencut.panel/client/
  index.html          # 7-tab layout: Cut | Captions | Audio | Video | Export | Settings
  style.css           # Premium dark theme with all component styles
  main.js             # Tab routing, API integration, UI controllers

opencut/
  server.py           # Flask server - all endpoints
  core/
    silence.py        # Phase 1 (done)
    fillers.py        # Phase 1 (done)
    captions.py       # Phase 1 (done)
    styled_captions.py # Phase 1 (done) + Phase 2 (new styles)
    zoom.py           # Phase 1 (done)
    audio.py          # Phase 1 (done)
    audio_suite.py    # Phase 3 (backend done, needs ducking)
    scene_detect.py   # Phase 4 (backend done)
    diarize.py        # Phase 1 (done)
    speed_ramp.py     # Phase 4 (new)
    reframe.py        # Phase 4 (new)
    transcript_export.py  # Phase 5 (new)
    broll_detect.py   # Phase 5 (new)
  export/
    premiere.py       # Phase 1 (done)
    srt.py            # Phase 1 (done) + Phase 2 (ASS export)
  utils/
    config.py         # Phase 1 (done)
    media.py          # Phase 1 (done)
```

## Tab → Endpoint Mapping

| Tab | Features | Backend Endpoints |
|-----|----------|-------------------|
| Cut | Silence, Fillers, Full Pipeline | /silence, /fillers, /full |
| Captions | Styled overlay, SRT/VTT/ASS, Custom styles, Transcript editor | /styled-captions, /captions, /caption-styles |
| Audio | Denoise, Isolate, Normalize, Beats, Effects, Ducking | /audio/* |
| Video | Scenes, Chapters, Speed Ramp, Reframe, LUT | /video/* |
| Export | Transcript, YouTube chapters, Platform presets, B-roll, Batch | /export/* |
| Settings | Models, Preferences, Shortcuts, About | /system/*, /health |

## Build Order

Each phase builds incrementally on the previous:
1. Phase 2 creates the new GUI shell that all subsequent phases plug into
2. Phase 3-5 each add one tab's worth of functionality
3. Phase 6 adds settings and polish for v1.0.0 release

Total estimated new/modified code: ~8,000-12,000 lines across all phases.
