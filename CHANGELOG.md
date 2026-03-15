# Changelog

## v1.1.0 (2026-03-15)

### New Features
- **Preset Save/Load**: Save and load all panel settings as named presets (persisted to `~/.opencut/user_presets.json`)
- **AI Model Manager**: View all downloaded AI models (HuggingFace, Torch, Whisper) with sizes, delete unused models to free space
- **GPU Auto-Detection & Recommendation**: Automatically recommend optimal model sizes and settings based on detected GPU VRAM
- **Job Queue System**: Queue multiple jobs for sequential processing instead of waiting for each to finish
- **Enhanced Job History**: Re-run previous jobs directly from history, toast notifications on completion
- **Keyboard Shortcuts**: Enter to run active operation, 1-6 to switch tabs, Escape to cancel
- **Enhanced Drag-and-Drop**: Drop files anywhere on the panel, not just the drop zone
- **Toast Notifications**: Non-intrusive slide-in notifications for job completion/errors
- **Transcript Search**: Search and navigate through transcript segments with highlight
- **Social Platform Export Presets**: Quick export presets for YouTube Shorts, TikTok, Instagram Reel/Story/Post, Twitter/X, LinkedIn
- **Premiere Theme Sync**: Detect Premiere Pro's UI brightness via CSInterface
- **Universal Auto-Import**: Smart ExtendScript import function that routes files to appropriate project bins

### Bug Fixes (v1.0.0 audit)
- Fixed 6 broken install routes using `subprocess.run` instead of `_sp.run`
- Fixed SSE race condition (dict copy + status read now inside `job_lock`)
- Fixed VBS launcher path quoting for `C:\Program Files\OpenCut`
- Replaced `eval()` with `JSON.parse()` in ExtendScript
- Fixed wrong `/cut/silence` endpoint in social_ready workflow
- Fixed `jobStarting` guard and `no_input` flag in all applicable routes
- Fixed temp file leaks in audio_separate and watermark routes
- Added `user-select: text` override for CEP Chromium inputs

## v1.0.0-beta (2026-02-09)

First public beta release with full feature set.

### Core (34 modules, 116 routes)
- Silence detection and removal with configurable thresholds
- Filler word detection (um, uh, like, you know, so, actually)
- Speaker diarization via pyannote.audio
- Job queue with async processing and progress polling

### Captions (5 modules)
- WhisperX / faster-whisper transcription with word-level timestamps
- 19 caption styles (YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, etc.)
- 7 animated caption presets (pop, fade, slide, typewriter, bounce, glow, highlight)
- Caption burn-in with style selection
- Translation to 50+ languages

### Audio (8 modules)
- Stem separation via Demucs (vocals, drums, bass, other)
- AI noise reduction + spectral gating
- Loudness normalization (LUFS targeting)
- Beat detection and BPM analysis
- Pro FX chain: compressor, EQ, de-esser, limiter, reverb, stereo width
- Text-to-speech with 100+ voices (Edge-TTS / F5-TTS)
- Procedural SFX generator (tones, sweeps, impacts, noise)
- AI music generation via MusicGen (AudioCraft)
- Audio ducking (auto-lower music under dialogue)

### Video (12 modules)
- FFmpeg stabilization (deshake/vidstab)
- Chromakey with green/blue/red screen + spill suppression
- Picture-in-picture overlay with position/scale
- 14 blend modes (multiply, screen, overlay, etc.)
- 34 xfade transitions (fade, wipe, slide, circle, pixelize, radial, zoom)
- 7 particle effect presets (confetti, sparkles, snow, rain, fire, smoke, bubbles)
- 6 animated title presets (fade, slide, typewriter, lower third, countdown, kinetic)
- Speed ramping with ease curves
- Scene detection via PySceneDetect
- Film grain, letterbox, vignette effects
- 15 built-in cinematic LUTs + external .cube/.3dl support
- Color correction (exposure, contrast, saturation, temperature, shadows, highlights)
- Color space conversion (sRGB, Rec.709, Rec.2020, DCI-P3)

### AI Tools (6 modules)
- AI upscaling: Lanczos (fast), Real-ESRGAN (balanced), Video2x (premium)
- Background removal via rembg (U2-Net)
- Face enhancement via GFPGAN
- Face swapping via InsightFace
- Neural style transfer
- Object/watermark removal (FFmpeg delogo + LaMA inpainting)
- Auto-thumbnail extraction with AI scoring

### Export & Batch (3 modules)
- Platform presets: YouTube, TikTok, Instagram, Twitter, LinkedIn, Podcast
- Batch processing with parallel execution
- Transcript export (SRT, VTT, plain text, timestamped)

### UI
- 6 dark themes (Cyberpunk, Midnight OLED, Catppuccin, GitHub Dark, Stealth, Ember)
- 6 main tabs with 24 sub-tabs
- Persistent processing banner with progress
- Single-job enforcement with UI lockout
- Auto-import results back to Premiere Pro timeline
