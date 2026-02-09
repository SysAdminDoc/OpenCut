# Changelog

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
