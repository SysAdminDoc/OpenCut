# OpenCut — Feature Expansion Plan

**Baseline**: v1.10.4 (254 routes, 68 core modules, 8 panel tabs, 867 tests)
**Created**: 2026-04-12
**Scope**: New features, expansion features, and improvements not yet in the codebase or ROADMAP.md

Items already completed or tracked in ROADMAP.md are excluded. This document focuses exclusively on what's **not yet planned or implemented**.

**Updated**: 2026-04-14 — 377 features across 77 categories

---

## Legend

| Symbol | Meaning |
|--------|---------|
| **P0** | Critical — competitive table-stakes, users expect this |
| **P1** | High — significant differentiator or major usability win |
| **P2** | Medium — valuable but not urgent |
| **P3** | Nice-to-have — future expansion or niche use case |
| **S** | Small effort (<1 day) |
| **M** | Medium effort (1-3 days) |
| **L** | Large effort (3-7 days) |
| **XL** | Extra large (1-2 weeks+) |

---

## 1. AI-Powered Editing Features

### 1.1 Transcript-Based Editing (P0 / XL)
Edit video by editing text — the single most-requested feature in modern video editing (Descript's core). Select and delete words from the transcript to cut the corresponding video. Rearrange paragraphs to rearrange clips. Copy-paste sections to duplicate.
- **Why**: Descript built a $700M+ company on this one feature. Every AI editing SaaS now copies it. OpenCut has transcription but no text-as-timeline editing.
- **Implementation**: Use existing WhisperX word-level timestamps to build a text↔timeline bidirectional map. Panel renders editable transcript where selections map to time ranges. Delete text → generate cut list → apply via existing `ocApplySequenceCuts`. Rearrange text → export as new sequence via OTIO.
- **References**: Descript, Simon Says, Recut, Kapwing

### 1.2 AI Eye Contact Correction (P1 / L)
Adjust the speaker's gaze direction so they appear to look directly into the camera lens. Critical for webcam recordings, interviews, and talking-head content where the speaker looks at their screen instead of the camera.
- **Why**: Descript, CapCut, and NVIDIA Broadcast all offer this. Common pain point for creators.
- **Implementation**: Use MediaPipe face mesh (already a dependency via face_reframe) to detect eye landmarks, then apply gaze redirection via a lightweight GAN model. Process per-frame with temporal smoothing.
- **References**: Descript Eye Contact, NVIDIA Maxine Eye Contact, GazeRedirect

### 1.3 AI Overdub / Voice Correction (P1 / L)
Fix spoken mistakes by typing the correct words — the AI speaks them in the original speaker's cloned voice and seamlessly replaces the audio+video. No reshoots needed.
- **Why**: Combines OpenCut's existing voice cloning (Chatterbox) with lip sync to create Descript's Overdub equivalent. The pieces exist but aren't connected.
- **Implementation**: (1) Clone speaker voice from surrounding audio via Chatterbox. (2) Generate corrected audio segment. (3) Apply lip sync (see 1.4) to match new audio. (4) Crossfade audio boundaries. (5) Composite video.
- **References**: Descript Overdub, ElevenLabs Dubbing

### 1.4 AI Lip Sync (P1 / L)
Replace mouth movements in video to match new audio — enables dubbing, voice correction, and translation with matching lip movements.
- **Why**: Essential for dubbing/localization workflows. Multiple open-source options now production-ready. Pairs with voice cloning and translation features already in OpenCut.
- **Implementation**: Add MuseTalk (diffusion-based, sharper results, 30+ FPS on GPU) as primary backend, video-retalking as fallback (preserves everything except mouth region). Both take video + audio → output video with synced lips.
- **References**: MuseTalk, video-retalking, Wav2Lip, SadTalker

### 1.5 AI Voice Conversion / RVC (P2 / L)
Convert a speaker's voice to sound like a different person while preserving speech content, timing, and emotion. Useful for voice anonymization, dubbing with consistent character voices, or replacing a scratch track narrator.
- **Why**: RVC has exploded in popularity (voice covers, dubbing). OpenCut has TTS and voice cloning but no voice-to-voice conversion.
- **Implementation**: Integrate RVC (Retrieval-based Voice Conversion) — load a target voice model, process source audio, output converted audio. Models are small (~50MB each) and inference is fast.
- **References**: RVC, So-VITS-SVC, OpenVoice, Applio

### 1.6 AI Auto-Color Grading (P2 / M)
Automatically apply a cinematic color grade based on a reference image, film name, or mood keyword ("warm sunset", "teal and orange", "horror"). Goes beyond LUT application by analyzing the source footage and adapting the grade.
- **Why**: LUT library exists but requires manual selection. Auto-grading lets creators describe the look they want in words.
- **Implementation**: (1) Keyword/mood → select from expanded LUT library + custom parameter adjustments. (2) Reference image → existing color_match.py histogram matching. (3) LLM interpretation → parse "make it look like Blade Runner 2049" into specific color correction parameters via llm.py.
- **References**: DaVinci Resolve AI color, LUTforge-AI, Colourlab.ai

### 1.7 AI Content Moderation / Compliance Scanner (P2 / M)
Scan video for NSFW content, profanity in speech, violence, flashing lights (epilepsy risk), and other compliance issues. Generate a timestamped report with severity levels.
- **Why**: Creators publishing to YouTube/TikTok need to know if content will be demonetized or age-restricted. Broadcast editors need FCC compliance.
- **Implementation**: (1) Profanity: flag words in existing transcript. (2) Visual: run frames through NSFW classifier (NudeNet or similar, small model). (3) Flashing: FFmpeg `flashdetect` filter. (4) Loudness: existing LUFS measurement against broadcast standards. Output: timestamped compliance report.
- **References**: Google Video Intelligence API (design reference), NudeNet, Hive Moderation

### 1.8 AI Scene Description / Alt-Text (P3 / M)
Generate natural-language descriptions of each scene for accessibility, metadata enrichment, or AI-assisted editing. Uses a vision-language model to describe what's happening in each shot.
- **Why**: Accessibility compliance (WCAG), better footage search, richer metadata for the footage index.
- **Implementation**: Run scene detection (existing), extract key frame per scene, feed to Florence-2 (already a dependency for watermark detection) or a local LLaVA model for captioning. Store descriptions in footage index DB.
- **References**: Florence-2, LLaVA, CogVLM

### 1.9 AI Video Summarization (P3 / M)
Generate a text summary or visual summary (highlight reel) of a long video. Takes a 60-minute interview and produces a 5-paragraph written summary or a 3-minute highlight compilation.
- **Why**: Pre-production workflow — quickly understand what's in raw footage before editing. Also useful for generating video descriptions.
- **Implementation**: Text summary: transcribe → chunk → LLM summarize (existing llm.py). Visual summary: transcribe → LLM identify key moments → extract clips → concatenate. Both leverage existing modules.
- **References**: VideoLLaMA, Claude vision, Gemini video understanding

### 1.10 AI Talking Head / Avatar Generation (P3 / XL)
Generate a talking head video from a still photo + audio. Useful for creating presenter videos, explainer content, or placeholder footage during pre-production.
- **Why**: Growing demand for AI presenters. Combines with TTS for fully synthetic video narration.
- **Implementation**: Integrate SadTalker (image+audio→video) or Linly-Talker (full pipeline). Both are open-source and run locally.
- **References**: SadTalker, Linly-Talker, LivePortrait, HeyGen (commercial reference)

### 1.11 AI B-Roll Suggestion & Insertion (P1 / L)
Analyze the transcript to detect talking-head segments that would benefit from B-roll cutaways, suggest relevant footage from the project media pool or stock libraries, and auto-insert at natural pause points. Transforms static interviews into dynamic visual content.
- **Why**: Every interview and talking-head video needs B-roll. Deciding where to cut and what to show is the most time-consuming part of post-production. AI transcript analysis ("when she mentions the factory...") combined with footage index search automates this entirely.
- **Implementation**: (1) LLM analyzes transcript to identify visual concepts and B-roll cue points (existing llm.py). (2) Search footage index (existing footage_search) for matching clips by semantic similarity. (3) If no local footage matches: suggest stock search queries (ties to 19.1). (4) Generate cut list with B-roll placements at natural phrase boundaries. (5) Apply via timeline insertion.
- **References**: Descript B-roll, Pictory AI, Adobe Sensei B-roll suggestions

### 1.12 AI Pacing & Rhythm Analysis (P2 / M)
Analyze the editing rhythm of the current timeline — cut frequency, average shot duration, pacing curves — and compare against genre templates (fast-paced trailer, conversational interview, meditative documentary). Flag sections that are too slow or too fast for the target style and suggest specific tightening or breathing room.
- **Why**: Pacing is invisible to novice editors but critical to audience retention. YouTube analytics consistently show drop-off points correlated with pacing lulls. An automated pacing analyzer catches these before publishing.
- **Implementation**: Extract cut points from timeline (existing ExtendScript). Calculate: cuts per minute, shot duration distribution, pacing curve (rolling average of shot duration over 30-second windows). Compare against genre profiles (stored as JSON templates). LLM generates specific suggestions ("shots 14-18 average 12 seconds — consider tightening to 5-7 for this genre"). Render pacing curve as overlay on waveform timeline.
- **References**: YouTube audience retention analytics, Every Frame a Painting analysis, EditStock pacing tools

---

## 2. Advanced Audio Features

### 2.1 Podcast Production Suite (P1 / L)
One-click podcast polish: auto-level all speakers to consistent loudness, remove crosstalk/overlap, apply per-speaker EQ profiles, add configurable intro/outro, insert chapter markers, and export with podcast-standard loudness (-16 LUFS mono / -19 LUFS stereo).
- **Why**: OpenCut has all the individual audio tools but no integrated podcast workflow. AutoPod and Descript dominate this niche.
- **Implementation**: Combine existing modules: diarize speakers → per-speaker loudness_match → per-speaker audio_pro (EQ/compress) → concatenate with intro/outro → chapter_gen → export. New workflow template + dedicated UI card.
- **References**: AutoPod, Descript, Auphonic, Riverside.fm

### 2.2 Audio Restoration Toolkit (P2 / M)
Specialized tools beyond generic denoising: **declip** (fix clipped/distorted audio), **decrackle** (remove vinyl/recording crackle), **dehum** (remove 50/60Hz electrical hum and harmonics), **dewind** (remove wind noise from outdoor recordings), **dereverb** (reduce room echo/reverb).
- **Why**: Each artifact type needs a specific approach. Generic denoising handles broadband noise but not these targeted issues. Pro audio tools (iZotope RX) charge $400+ for this.
- **Implementation**: declip: FFmpeg `adeclip` filter. dehum: FFmpeg `bandreject` at 50/60Hz + harmonics. decrackle: FFmpeg `arnndn` with crackle model. dewind: high-pass filter + spectral gating. dereverb: requires a model (WPE algorithm or existing DeepFilterNet).
- **References**: iZotope RX (design reference), FFmpeg audio filters, SpeechBrain

### 2.3 Real-Time Voice Conversion for Narration (P2 / L)
Record narration directly through OpenCut with real-time voice conversion applied — speak in your voice, output in a selected voice model. Useful for creators who want professional narration without hiring a voice actor.
- **Why**: RVC runs at real-time speeds on GPU. Combining with OpenCut's recording pipeline eliminates the need for post-processing.
- **Implementation**: Capture mic input → RVC inference in real-time → write converted audio. Requires a simple audio capture route + RVC integration from feature 1.5.
- **References**: RVC, Applio, voice.ai

### 2.4 AI Sound Effects Generation from Video (P2 / M)
Analyze video content and automatically generate matching sound effects — footsteps, door closes, ambient sounds, impacts. Automated foley.
- **Why**: Foley work is expensive and time-consuming. AI SFX generation is emerging but not yet in any Premiere extension.
- **Implementation**: (1) Scene detection + frame analysis via Florence-2 to identify on-screen actions. (2) Map actions to SFX prompts. (3) Generate SFX via AudioLDM2 or AudioGen (AudioCraft, already a dependency). (4) Time-align to video events.
- **References**: AudioLDM2, AudioGen (AudioCraft), ElevenLabs Sound Effects, Stability Audio

### 2.5 Spatial Audio / Immersive Audio (P3 / M)
Convert stereo audio to spatial formats: binaural (headphones), 5.1 surround, 7.1, or Dolby Atmos-compatible output. Position audio sources in 3D space based on speaker position in video.
- **Why**: YouTube, Apple, and Spotify all support spatial audio. Growing demand for immersive content.
- **Implementation**: Use FFmpeg's `sofalizer` filter for binaural rendering, `pan` filter for surround upmix. Advanced: use speaker diarization face positions to auto-pan speakers in surround field.
- **References**: FFmpeg sofalizer, Atmos Production Suite, DearVR

### 2.6 Audio Spectrum Analyzer & Loudness Meter (P2 / S)
Real-time frequency spectrum visualization and LUFS/True Peak metering displayed in the panel during playback or after processing. Shows whether audio meets platform standards.
- **Why**: Creators need to verify audio meets platform requirements (YouTube: -14 LUFS, Spotify: -14 LUFS, broadcast: -24 LUFS). Currently must use external tools.
- **Implementation**: FFmpeg `ebur128` filter outputs real-time LUFS data. Parse and display as a meter widget. Spectrum: FFmpeg `showfreqs` filter or `astats` for quick analysis. Render as canvas visualization in panel.
- **References**: Youlean Loudness Meter, FFmpeg ebur128

### 2.7 Stem Remix / Stem Effects (P3 / M)
After stem separation, apply different effects to each stem independently and recombine. Example: add reverb to vocals only, compress drums, EQ the bass, keep other stems dry. Full mix control.
- **Why**: OpenCut separates stems but only exports them. The natural next step is to process and remix them.
- **Implementation**: Separate → apply per-stem audio_pro effects chain → mix stems with volume/pan controls per stem → export final mix. UI: multi-track mixer card with per-stem sliders and FX toggles.
- **References**: iZotope Music Rebalance, LALAL.AI, stems.ai

### 2.8 Audio Fingerprinting & Copyright Detection (P1 / M)
Scan project audio against a local database of known copyrighted music to flag potential copyright strikes before uploading to YouTube/TikTok. Also detect background music bleeding into production audio (coffee shop ambiance with recognizable songs, radio playing during interviews).
- **Why**: Copyright strikes demonetize or block videos. Creators often don't realize copyrighted music leaked into their production audio. Detecting before upload prevents strikes and manual re-edits. YouTube Content ID catches it after upload — catching it before is strictly better.
- **Implementation**: Audio fingerprinting via Chromaprint/acoustid (open-source Shazam equivalent). Generate fingerprint of all audio tracks. Compare against a user-curated local DB of flagged tracks (common stock music, popular songs). Advanced: run Demucs stem separation → isolate background music → fingerprint the music stem separately to catch buried music.
- **References**: Chromaprint/acoustid, Audible Magic (design reference), YouTube Content ID

---

## 3. Advanced Video Features

### 3.1 Motion Tracking & Object Annotation (P1 / L)
Track any object through a video and attach overlays (text labels, arrows, blur regions, bounding boxes, graphics). Click to select, auto-track through the clip. Essential for tutorials, sports analysis, and privacy.
- **Why**: OpenCut has face tracking (MediaPipe) and object segmentation (SAM2) but no general-purpose motion tracking with overlay attachment. This is a top request for educational/tutorial creators.
- **Implementation**: SAM2 click-to-select → track through frames → export track data as (x,y,w,h) per frame → overlay graphics/text/blur at tracked positions via FFmpeg drawtext with per-frame coordinates or Pillow compositing.
- **References**: After Effects motion tracking, DaVinci Resolve tracker, SAM2

### 3.2 AI Sky Replacement (P2 / M)
Detect the sky region in outdoor footage and replace it with a different sky — sunset, cloudy, starry night, aurora, or a custom image/video. Maintains foreground lighting consistency.
- **Why**: Popular in photography (Photoshop, Luminar) but rare in video editing tools. Visually dramatic with relatively simple implementation.
- **Implementation**: Use Depth Anything V2 (already a dependency) to segment sky (far-depth region) or a dedicated sky segmentation model. Replace sky pixels with source sky. Apply color temperature adjustment to foreground to match new sky lighting.
- **References**: Luminar AI Sky Replacement, Photoshop Sky Replacement

### 3.3 AI Relighting (P2 / L)
Change the lighting direction, intensity, color temperature, or add/remove light sources in existing footage. Useful for fixing badly-lit interviews or creating dramatic lighting effects.
- **Why**: Depth Anything V2 (already integrated) provides the depth map needed for relighting. This is a natural extension of existing depth features.
- **Implementation**: Depth map + normal estimation → relight using new light position/color. IC-Light (Stability AI) is an open-source relighting model. Alternatively, use depth map to create a simple diffuse shading pass.
- **References**: IC-Light, DaVinci Resolve lighting effects, Relight (research)

### 3.4 360 Video Support (P3 / L)
Import, process, and export equirectangular 360 video. Apply stabilization, reframing (extract flat crops from 360), and basic effects. Export as standard flat video or 360.
- **Why**: GoPro Max, Insta360, Ricoh Theta are popular. Shotcut already has 360 filters. Niche but growing.
- **Implementation**: FFmpeg `v360` filter handles projection conversion, reframing, and stabilization. Add 360-aware crop tool that lets user click-drag a viewport window on equirectangular preview. Export flat crops or full 360.
- **References**: Shotcut 360 filters, FFmpeg v360, Insta360 Studio

### 3.5 AI Deinterlacing (P3 / S)
High-quality AI-based deinterlacing for legacy/broadcast footage. Better than FFmpeg's `yadif` for archival content.
- **Why**: Anyone working with tape-based footage, broadcast archives, or retro content needs deinterlacing. AI methods (QTGMC equivalent) are far superior to traditional approaches.
- **Implementation**: FFmpeg `bwdif` filter is already excellent for most cases. For premium quality, integrate `nnedi3` via FFmpeg filter or vapoursynth. Add as a toggle in video processing.
- **References**: QTGMC, nnedi3, FFmpeg bwdif

### 3.6 HDR/SDR Tone Mapping (P2 / M)
Convert HDR (HLG, PQ/HDR10, Dolby Vision) footage to SDR for web delivery, or apply inverse tone mapping to uplift SDR footage to HDR. Include HDR metadata handling.
- **Why**: iPhone and modern cameras shoot HDR by default. Creators need to deliver SDR for most platforms but preserve HDR for YouTube HDR uploads. Currently no tools in OpenCut handle this.
- **Implementation**: FFmpeg `zscale` + `tonemap` filters for HDR→SDR. `libplacebo` for high-quality tone mapping with custom curves. Add HDR detection in media probe, auto-suggest tone mapping when HDR source detected.
- **References**: FFmpeg tonemap, libplacebo, HandBrake HDR

### 3.7 Lens Distortion Correction (P3 / S)
Automatically detect and correct barrel/pincushion distortion from wide-angle lenses, action cameras, and drones. Straighten horizon lines.
- **Why**: GoPro and drone footage always needs distortion correction. Simple to implement with FFmpeg.
- **Implementation**: FFmpeg `lenscorrection` filter with k1/k2 parameters. Add presets for common cameras (GoPro, DJI). Auto-detect via EXIF metadata if available. Add horizon leveling via FFmpeg `rotate` filter.
- **References**: FFmpeg lenscorrection, GoPro lens profiles, Lightroom lens corrections

### 3.8 Pika-Style Object Effects (P3 / XL)
Apply physics-simulated effects to selected objects in video — squish, melt, inflate, explode, dissolve, crystallize. Select an object, pick an effect, render.
- **Why**: Viral creative effect from Pika Labs. Unique differentiator if done locally.
- **Implementation**: SAM2 (existing) for object selection → mask extraction → apply effect via specialized diffusion model or procedural animation. Heavy R&D, likely requires a custom model or adaptation of AnimateDiff + ControlNet.
- **References**: Pika Pikaffects, AnimateDiff, ComfyUI workflows

### 3.9 Multi-Camera Audio Sync (P1 / M)
Automatically synchronize multiple camera angles by matching their audio waveforms. Essential for multicam shoots where cameras weren't timecode-synced.
- **Why**: OpenCut has multicam switching via diarization but no way to sync unsynchronized cameras first. This is a prerequisite for proper multicam workflows.
- **Implementation**: Extract audio from each angle → compute cross-correlation to find time offset → report offsets for user to align in timeline, or auto-apply via ExtendScript multi-clip alignment. FFmpeg `arealtime` + numpy cross-correlation.
- **References**: PluralEyes, Premiere Pro multicam sync, FFmpeg cross-correlation

### 3.10 Video Comparison / Diff Tool (P2 / S)
Side-by-side, overlay, or wipe comparison between two versions of a clip (before/after processing, two takes, two color grades). Generates a comparison frame or video.
- **Why**: Currently no way to visually compare processing results. Essential for color grading, denoising, upscaling QC.
- **Implementation**: FFmpeg `hstack`, `blend`, or custom `xfade` for side-by-side/overlay/wipe. Generate comparison frames or full comparison videos. Add toggle in panel for A/B comparison of any processed output.
- **References**: DaVinci Resolve split-screen wipe, Topaz comparison view

### 3.11 Corrupted Video Repair & Recovery (P2 / M)
Attempt to recover and repair corrupted video files: fix broken headers, reconstruct missing moov atoms, recover data from partially-written files (camera died mid-record, power loss, SD card corruption). Diagnose the corruption type and salvage as much footage as possible.
- **Why**: Corrupted files from camera crashes and SD card failures are a common and devastating problem. Professional recovery tools charge $50-200 per file. FFmpeg can often recover the data but requires arcane command knowledge.
- **Implementation**: Detect corruption type: (1) Missing moov atom: use `untrunc` with a reference file from the same camera, or FFmpeg `-movflags faststart` remux. (2) Broken container: rewrap with `ffmpeg -err_detect ignore_err -i input -c copy output`. (3) Partial file: attempt to demux recoverable streams. Report: recovered duration vs. expected, frame drop locations, severity assessment.
- **References**: untrunc (GitHub), FFmpeg error recovery, Stellar Video Repair (commercial reference)

### 3.12 Rolling Shutter Correction (P2 / M)
Fix the wobble/jello effect from CMOS rolling shutter sensors in handheld footage, especially visible on phones, drones, and action cameras during fast lateral motion or vibration.
- **Why**: Rolling shutter is the most common artifact in smartphone and drone footage. Premiere's Warp Stabilizer partially addresses it, but a dedicated correction tool produces cleaner results, especially when gyroscope data is available.
- **Implementation**: FFmpeg `dejudder` filter for simple cases. For full correction: parse gyroscope data from camera metadata (DJI SRT files, GoPro GPMF telemetry) to compute exact rolling shutter parameters. Without gyro data: estimate via optical flow between scan lines. Apply inverse warp via OpenCV remap. Combine with existing stabilization as a pre-processing step.
- **References**: GyroFlow, GoPro ReelSteady, Premiere Warp Stabilizer rolling shutter option

### 3.13 Morph Cut / Smooth Jump Cut (P1 / L)
Seamlessly blend between two portions of the same talking-head shot to hide a jump cut. Morphs the speaker's face and body between the two frames using optical flow interpolation, creating an invisible edit that looks like continuous footage.
- **Why**: Jump cuts from removing filler words, pauses, or bad takes are visually jarring in interviews and vlogs. Premiere's Morph Cut exists but is notoriously slow and often produces artifacts. A fast, AI-powered version would be a killer feature for the silence/filler removal workflow.
- **Implementation**: (1) Detect face region in exit/entry frames (MediaPipe, existing). (2) Compute dense optical flow between exit and entry frames. (3) Generate N interpolated frames using RIFE (existing) or flow-based warping. (4) Crossfade non-face background regions. (5) Insert interpolated frames at the cut point. Process only the 0.5-1 second transition zone. Auto-apply after silence/filler removal as optional post-process.
- **References**: Premiere Pro Morph Cut, RIFE interpolation, FILM (Frame Interpolation for Large Motion)

---

## 4. Workflow & Editing Automation

### 4.1 Watch Folder / Hot Folder (P1 / M)
Monitor a designated folder — automatically process any new file dropped into it using a specified workflow. Ideal for ingest pipelines, batch interview processing, or automated social media content creation.
- **Why**: Broadcast and production workflows rely on watch folders. Pairs perfectly with the existing workflow engine.
- **Implementation**: `watchdog` Python library monitors a folder for new files. On new file: validate media → run configured workflow → output to destination folder. Add UI in Settings for configuring watch folders with workflow assignment.
- **References**: Adobe Media Encoder watch folders, FFmpeg watch scripts, Handbrake queue

### 4.2 Render Queue (P1 / M)
Queue multiple export jobs with different settings — export the same project as YouTube 1080p, TikTok 9:16, Instagram Square, and Podcast audio-only, then walk away. Progress shown per item.
- **Why**: Creators routinely export the same content in 3-5 formats. Currently must run each export manually. Major time-saver.
- **Implementation**: New `/export/queue` endpoint that accepts an array of export configs. Process sequentially or in parallel (CPU exports). Reuse existing batch_executor infrastructure. Panel UI: render queue card showing all pending exports with progress bars.
- **References**: Adobe Media Encoder, DaVinci Resolve render queue, Handbrake queue

### 4.3 Conditional Workflow Steps (P2 / M)
Add conditions to workflow steps: "if loudness < -20 LUFS, normalize", "if duration > 60s, detect scenes", "if has_face, apply face-tracked reframe". Skip unnecessary steps automatically.
- **Why**: Current workflows run every step unconditionally. Wastes time on steps that don't apply. Makes workflows less reusable across different input types.
- **Implementation**: Extend workflow JSON schema with `condition` field per step: `{ "if": "loudness_lufs < -20" }`, `{ "if": "has_video && duration > 60" }`. Workflow runner evaluates conditions against clip metadata (from `/context/analyze`) before each step.
- **References**: GitHub Actions conditionals, n8n workflow conditions

### 4.4 Undo Stack / Operation History (P2 / M)
Track every operation performed in a session with full undo/redo. Panel-level undo (not just Premiere's undo) that can reverse any OpenCut action.
- **Why**: Users fear making irreversible changes. The existing cut review panel addresses this for cuts but not for audio processing, color grading, exports, etc.
- **Implementation**: Each operation saves: input file, output file, parameters, timestamp. Undo = revert to previous output file. Store in session-scoped list. Display as scrollable history timeline in panel. Limit to last 20 operations to manage disk space.
- **References**: Photoshop History panel, DaVinci Resolve undo history

### 4.5 EDL / AAF Import & Export (P2 / M)
Import and export Edit Decision Lists (EDL/CMX3600) and Advanced Authoring Format (AAF) files. Industry-standard interchange formats used by colorists, audio mixers, and VFX houses.
- **Why**: OTIO export exists but EDL/AAF are still the industry standard for color and audio post. Colorists (DaVinci Resolve) and audio mixers (Pro Tools) expect EDL/AAF.
- **Implementation**: OpenTimelineIO (already a dependency) supports EDL and AAF adapters. Enable OTIO's CMX3600 adapter for EDL, write AAF via OTIO's AAF adapter. Add import endpoint that parses EDL → applies cuts to timeline.
- **References**: OpenTimelineIO adapters, Premiere Pro EDL export

### 4.6 Project Archival / Package (P3 / M)
Package the current project state: collect all source media, outputs, workflows, presets, and metadata into a single portable archive. Useful for backup, handoff to another editor, or cold storage.
- **Why**: Premiere's Project Manager does this but it's clunky. A one-click archive that includes OpenCut settings and workflows is valuable.
- **Implementation**: Enumerate all files referenced in recent jobs + current project media → copy to archive directory → include OpenCut settings/workflows JSON → compress to .zip. Add restore endpoint that unpacks and re-links.
- **References**: Premiere Project Manager, DaVinci Resolve archive

### 4.7 Script/Storyboard Integration (P2 / L)
Import a script (text/PDF/FDX) or storyboard (images) and use it to guide the editing process. Match script lines to transcript segments, flag missing coverage, suggest B-roll insertion points based on script action lines.
- **Why**: Narrative and corporate video editors work from scripts. Bridging script→footage is manual and tedious. No competing tool does this well.
- **Implementation**: Parse script (PDF text extraction or .fdx XML). Use LLM to align script lines with transcript segments (semantic similarity). Highlight unmatched script sections as "missing coverage". Suggest B-roll from existing footage index or AI generation.
- **References**: Movie Magic Screenwriter, WriterSolo, Celtx

### 4.8 Best Take Selection (P1 / M)
When multiple takes of the same line exist (detected via existing repeat_detect), automatically score each take on delivery quality (clarity, energy, timing, emotion) and recommend the best one.
- **Why**: AutoCut offers this. OpenCut detects repeated takes but doesn't help choose between them. Natural extension.
- **Implementation**: For each detected repeat group: (1) Audio analysis — loudness consistency, speech rate, clarity score. (2) Optional: emotion analysis via deepface (existing). (3) LLM scoring of transcript quality. (4) Rank and highlight best take. User clicks to keep winner and remove others.
- **References**: AutoCut Best Take, Descript's take management

---

## 5. Export, Distribution & Social

### 5.1 Multi-Platform Batch Publish (P1 / L)
Export and publish to multiple platforms simultaneously: YouTube + TikTok + Instagram + Twitter/X in one click. Each platform gets its optimized format (aspect ratio, duration limits, codec, metadata). Queue with per-platform progress.
- **Why**: Social media upload exists for individual platforms. Batch publishing across all platforms is what creators actually need. Save hours of manual uploading.
- **Implementation**: Extend existing social_post.py with batch mode. Per-platform: auto-reframe to target aspect ratio (existing), apply platform preset (existing), upload via OAuth (existing). New UI: publish checklist with per-platform toggles, title/description per platform, schedule time.
- **References**: Buffer, Hootsuite, Later, CapCut publish flow

### 5.2 AI SEO Optimizer (P2 / M)
Generate optimized titles, descriptions, tags, and hashtags for each platform based on video content analysis (transcript + visual analysis). A/B test title variants.
- **Why**: SEO is critical for discoverability. LLM can generate platform-optimized metadata from transcript.
- **Implementation**: Transcribe → LLM generates: (1) 5 title options ranked by click-worthiness, (2) platform-specific descriptions with keywords, (3) relevant tags/hashtags, (4) optimal posting time suggestions. Use existing llm.py.
- **References**: TubeBuddy, VidIQ, Opus Clip metadata generation

### 5.3 Broadcast QC / Standards Checker (P2 / M)
Verify exported files meet broadcast or platform standards: audio levels (EBU R128 / ATSC A/85), color gamut (Rec.709 / Rec.2020), safe areas (title safe / action safe), resolution, codec compliance, closed caption presence.
- **Why**: Broadcast delivery has strict QC requirements. Failing QC means re-export and missed deadlines. Automated checking prevents rejected deliverables.
- **Implementation**: FFmpeg `ebur128` for audio, `signalstats` for video levels, custom checks for resolution/codec/container. Generate pass/fail report with specific violations. Add as post-export automatic check.
- **References**: Telestream Vidchecker, FFmpeg QC scripts, Netflix delivery specs

### 5.4 Thumbnail A/B Generator (P3 / M)
Generate multiple thumbnail variants for the same video — different frames, text overlays, compositions, color treatments — and package them for A/B testing on YouTube.
- **Why**: Thumbnails drive 90% of click-through rate on YouTube. Generating variants for testing is high-value.
- **Implementation**: Extend existing thumbnail.py: (1) Score top 10 candidate frames (existing). (2) For each, generate variants: original, color-boosted, with text overlay, with face zoom. (3) LLM generates click-worthy text for overlay. (4) Export as image grid + individual files.
- **References**: TubeBuddy A/B testing, Canva thumbnail templates

### 5.5 Export Profiles with Watermark Toggle (P3 / S)
Toggle a visible watermark on/off per export — draft exports get a watermark, final exports don't. Watermark can be text (DRAFT / REVIEW / client name) or image (logo). Position and opacity configurable.
- **Why**: Common request for review workflows. Currently must manually add/remove watermark for each export.
- **Implementation**: Add `watermark` option to export presets: `{ enabled, type: "text"|"image", content, position, opacity }`. Apply via FFmpeg `drawtext` or `overlay` filter during export. Toggle in export UI.
- **References**: Frame.io watermarked reviews, Vimeo review pages

### 5.6 Embedded Soft Subtitles in Container (P2 / S)
Embed subtitle tracks as soft subtitles inside MP4/MKV containers — viewers toggle them on/off in their player. Support multiple language tracks in a single file. Currently OpenCut only does hard-burned captions or sidecar SRT/VTT files.
- **Why**: Soft subtitles are the standard for streaming platforms, Plex/Jellyfin media servers, and professional delivery. They're toggleable, don't degrade video quality, and support multiple languages in one file without re-encoding.
- **Implementation**: FFmpeg `-c:s mov_text` for MP4 (iTunes/QuickTime compatible) or `-c:s srt`/`-c:s ass` for MKV. Accept SRT/ASS input from existing caption pipeline. Map multiple language tracks: `-map 0:v -map 0:a -map 1:s -metadata:s:0 language=eng -metadata:s:1 language=spa`. Add language tag selector per track in export settings.
- **References**: FFmpeg subtitle muxing, MKV subtitle spec, MP4 mov_text format

---

## 6. Panel UX & Interface

### 6.1 Drag-and-Drop File Handling (P1 / S)
Drop media files directly onto the panel to set them as the active clip. Drop onto specific operation cards to immediately process (e.g., drop a file onto the "Remove Silence" card to process it). Also support dropping LUT files, SRT files, and preset files.
- **Why**: Natural interaction pattern missing from the panel. Reduces clicks.
- **Implementation**: CEP: `document.addEventListener("dragover"/"drop")` with file path extraction. Map drop zones to operations. UXP: `entrypoints.setup({ panels: { opencut: { dragHandlers } } })`.
- **References**: CapCut drop-to-edit, DaVinci Resolve media pool drag

### 6.2 Workspace Layouts (P2 / M)
Save and restore panel layout configurations — which tabs are visible, card collapse states, sidebar width, favorites bar items. Switch between presets like "Audio Post", "Color Grade", "Social Export".
- **Why**: Different editing phases need different tools visible. Kdenlive has this. Reduces clutter.
- **Implementation**: Serialize panel state (tab order, collapsed cards, sidebar state, favorites) to JSON. Save to user_data. Add layout switcher dropdown in panel header. Ship 4 built-in layouts: Assembly, Audio, Color, Delivery.
- **References**: Kdenlive workspace layouts, DaVinci Resolve pages

### 6.3 Side-by-Side Before/After Preview (P1 / M)
For any processing operation (color correction, upscaling, denoising, style transfer), show a before/after comparison. Wipe, split-screen, or toggle modes. Preview a single frame before committing to full processing.
- **Why**: Users need to validate results before spending processing time. Essential for color and quality operations. Currently must process the entire clip to see results.
- **Implementation**: New `/preview/frame` endpoint: extract frame at timestamp, apply operation to single frame, return both original and processed as base64 images. Panel renders with interactive wipe slider (canvas).
- **References**: Topaz Photo AI comparison, Lightroom before/after

### 6.4 Interactive Waveform Timeline (P2 / L)
Full waveform visualization with interactive controls: zoom, scroll, click-to-seek, drag-to-select regions, overlay silence/filler/scene markers. Replaces the current simple waveform preview with a mini-timeline.
- **Why**: Waveform preview exists but isn't interactive. A proper waveform timeline makes cut points visible and adjustable.
- **Implementation**: Generate waveform data via FFmpeg PCM extraction (existing). Render in `<canvas>` with zoom/scroll. Overlay colored regions from silence detection, filler detection, scene detection results. Click to seek in Premiere via ExtendScript.
- **References**: Audacity waveform, Descript waveform editor, Adobe Audition

### 6.5 Mini Player / Preview Window (P2 / L)
In-panel video preview window that shows the output of processing operations. Preview a processed frame, play back a short segment, or compare with the original. No need to switch back to Premiere to verify results.
- **Why**: Currently must import results into Premiere to see them. In-panel preview saves a round-trip for every operation.
- **Implementation**: Use `<video>` element to play output files. For preview frames, render as `<img>`. Add playback controls (play/pause, seek, speed). Frame extraction via FFmpeg for still previews.
- **References**: CapCut preview, After Effects layer preview

### 6.6 Right-Click Custom Actions in Premiere (P1 / M)
Add OpenCut actions to Premiere Pro's right-click context menu on clips: "Remove Silence", "Add Captions", "Normalize Audio", "Upscale", "Remove Background". Clicking opens the panel focused on that operation with the clip pre-loaded.
- **Why**: Reduces the workflow from: select clip → open panel → find tab → find card → configure → run. Context menu: right-click → action → run.
- **Implementation**: UXP: `entrypoints.setup({ commands: { removeSilence: { ... } } })` with menu registration. CEP: ExtendScript `app.project.activeSequence` context menus are limited but keyboard shortcuts can be registered via Premiere's shortcut system.
- **References**: AutoCut Premiere integration, Excalibur command palette

### 6.7 Quick Preview Thumbnails for Operations (P3 / S)
Show a thumbnail preview of what each video effect will look like before applying it — LUT preview grid, transition preview animations, style transfer preview on current frame, particle effect preview.
- **Why**: Choosing between 15 LUTs or 34 transitions without seeing them is guesswork. Visual previews make selection instant.
- **Implementation**: For LUTs: extract one frame → apply all LUTs → display as thumbnail grid. For transitions: render 2-second preview clips with first/last frames. For styles: apply each to one frame. Cache previews per clip.
- **References**: DaVinci Resolve LUT browser, Final Cut Pro transition previews

### 6.8 Dark / Light / System Theme Toggle (P2 / S)
Switch between dark, light, and system-matching themes for the panel UI. Currently the panel is dark-only, which clashes with Premiere's optional light theme and causes eye strain for some users in bright environments.
- **Why**: Premiere Pro supports light and dark workspace themes. OpenCut's panel should match. Accessibility concern for users who need high contrast or light backgrounds. Simple to implement with CSS custom properties.
- **Implementation**: CSS custom properties (design tokens already partially in place). Create `themes/light.css` overriding `--bg-primary`, `--text-primary`, `--card-bg`, `--border-color`, etc. Add theme toggle in Settings (dark/light/system). `prefers-color-scheme` media query for system-matching. Persist preference in localStorage. CEP: works natively. UXP: use Spectrum theme tokens.
- **References**: Premiere Pro theme matching, VS Code themes, Adobe Spectrum design system

---

## 7. Backend & Infrastructure

### 7.1 FastAPI Migration (P0 / XL)
Migrate from Flask to FastAPI for async support, automatic OpenAPI docs, Pydantic validation, WebSocket native support, and dependency injection. Already planned in ROADMAP Phase 1.1 but not started.
- **Why**: Flask's synchronous nature limits concurrency. Pydantic replaces ~200 manual `safe_float()`/`safe_int()` calls. Auto-generated OpenAPI replaces manual spec maintenance.
- **Action**: This is already fully designed in ROADMAP.md Phase 1.1. Execute the plan as written.

### 7.2 Process Isolation for GPU Workers (P0 / XL)
Run AI models in separate Python processes to prevent OOM crashes from model conflicts. Already designed in ROADMAP Phase 1.3 but not started.
- **Why**: WhisperX + Demucs + Real-ESRGAN in the same process = guaranteed OOM on <12GB VRAM GPUs. This is the #1 cause of crashes for GPU users.
- **Action**: This is already fully designed in ROADMAP.md Phase 1.3. Execute the plan as written.

### 7.3 Auto-Update Mechanism (P2 / M)
Check for new versions on startup, notify user, and offer one-click update. For pip installs: `pip install --upgrade opencut`. For exe installs: download new installer, prompt to run.
- **Why**: Users on old versions miss bug fixes and features. No current update notification mechanism.
- **Implementation**: On startup, hit GitHub Releases API for latest version. Compare with current `__version__`. If newer: show banner in panel with changelog summary + update button. Update button triggers appropriate update method (pip/exe/git pull).
- **References**: VS Code update mechanism, npm update-notifier

### 7.4 Model Quantization / Optimization (P2 / M)
Offer INT8 and INT4 quantized versions of heavy AI models for users with limited VRAM. Trade small quality reduction for major VRAM savings.
- **Why**: Real-ESRGAN, Depth Anything, SAM2 each need 2-4GB VRAM. Quantized versions run in 1-2GB. Opens AI features to GTX 1060/1070 users.
- **Implementation**: Use ONNX Runtime INT8 quantization (already a dependency). For PyTorch models: `torch.quantization.quantize_dynamic()`. Store quantized variants alongside full models. Engine registry chooses based on available VRAM.
- **References**: ONNX Runtime quantization, bitsandbytes, GPTQ

### 7.5 Remote Processing / Render Node (P3 / XL)
Offload heavy processing to a remote machine (home server, cloud GPU, office workstation). Send job + source file to remote OpenCut server, receive results back.
- **Why**: Laptop users can offload GPU work to a desktop. Small studios can share one GPU workstation.
- **Implementation**: New `/remote/submit` endpoint on the receiving server. Sending server: upload source file via HTTP multipart, submit job config, poll for completion, download result. Authentication via shared secret or API key.
- **References**: DaVinci Resolve render nodes, Adobe Team Projects

### 7.6 Webhook / API Notifications (P3 / S)
Send notifications to external services when jobs complete: Slack message, Discord webhook, email, or custom HTTP webhook. Useful for leaving processing overnight.
- **Why**: Long-running jobs (upscaling, batch processing) benefit from notifications. "Your video is ready" on your phone.
- **Implementation**: Add `notification` settings with webhook URL. On job completion: POST job result summary to webhook URL. Add Slack/Discord webhook presets. Simple HTTP POST with JSON payload.
- **References**: GitHub Actions notifications, Zapier webhooks

### 7.7 MCP Server Expansion (P2 / M)
Expand the MCP server from 23 tools to cover all major operations. Enable AI assistants (Claude, GPT) to fully control OpenCut: "clean up this interview, remove silence, add captions, normalize audio, and export for YouTube."
- **Why**: MCP is the emerging standard for AI tool integration. Full MCP coverage makes OpenCut the most AI-controllable video editor.
- **Implementation**: Map each major API route to an MCP tool. Add compound tools: `opencut_clean_interview`, `opencut_prepare_for_youtube`, `opencut_podcast_polish`. Add resource endpoints for listing available operations, presets, and workflows.
- **References**: Premiere Pro MCP (269 tools as reference), Claude MCP spec

---

## 8. Collaboration & Professional

### 8.1 Review & Approval Links (P2 / L)
Generate a shareable review link for processed video. Reviewer can watch, leave timestamped comments, approve or request changes. Comments sync back to the panel.
- **Why**: Client review is a critical workflow step. Frame.io charges $180/year for this. A basic local version would be valuable.
- **Implementation**: Serve processed video via built-in HTTP server. Generate unique URL with token-based access. Simple web page with video player + comment form (timestamp captured on click). Comments stored in SQLite, displayed in panel.
- **References**: Frame.io, Wipster, Vimeo Review

### 8.2 Team Presets via Shared Folder (P2 / S)
Point OpenCut to a shared network folder (Dropbox, Google Drive, NAS) for presets, workflows, and LUTs. Team members automatically see the same presets.
- **Why**: Consistency across a team. Export/import exists but is manual. A shared folder is automatic.
- **Implementation**: Settings → Team folder path. On startup, scan folder for `.opencut-preset`, `.opencut-workflow`, `.cube` files. Merge with local presets. Watch for changes. Already have preset export/import — just need the shared folder scanner.
- **References**: DaVinci Resolve shared databases, Premiere Pro team projects

### 8.3 Timestamped Project Notes (P3 / S)
Add notes to specific timestamps in the video — "Fix audio at 2:34", "Color is off here", "Client wants this cut shorter". Notes persist across sessions and export as a PDF/text report.
- **Why**: Clip Notes plugin exists as an example but isn't integrated into the main workflow. First-class note support is more useful.
- **Implementation**: Extend existing clip-notes plugin concept into core: notes stored in SQLite with timestamp, text, priority, status (open/resolved). Display as markers on waveform timeline (feature 6.4). Export as PDF/CSV.
- **References**: Frame.io annotations, Premiere Pro markers with comments

---

## 9. Platform & Ecosystem Expansion

### 9.1 DaVinci Resolve Deep Integration (P1 / L)
Expand from basic Resolve bridge to full feature parity with Premiere: media pool management, timeline editing, color grade application, audio processing, render queue control. Dedicated Resolve panel (Python-based GUI via PySide/tkinter).
- **Why**: Resolve is free and growing rapidly. Deep integration doubles OpenCut's addressable market. Basic bridge exists but only covers read operations.
- **Implementation**: Extend resolve_bridge.py with write operations: add markers, apply cuts, import media, set render settings. Build a lightweight Python GUI panel (PySide6 or web-based localhost UI) since Resolve doesn't support CEP/UXP.
- **References**: DaVinci Resolve scripting API, Pymiere (design pattern reference)

### 9.2 Standalone Web UI (P2 / XL)
A full-featured web interface at `localhost:5679` that works without any NLE. Complete editing capabilities: upload media, process, preview, download. For users who don't have Premiere Pro or Resolve.
- **Why**: FAQ already says "Can I use this without Premiere Pro? Yes." But the experience is curl/CLI only. A web UI makes OpenCut accessible to everyone.
- **Implementation**: React/Svelte SPA served by the backend. Reuse all API routes. Add: file upload, video player, operation cards matching panel design, job monitoring. Start minimal (upload → process → download) and expand.
- **References**: Kapwing, Canva Video, Runway ML web interface

### 9.3 After Effects Extension (P3 / XL)
CEP/UXP panel for Adobe After Effects. Subset of features relevant to motion graphics and VFX: background removal, upscaling, style transfer, object removal, depth effects.
- **Why**: Same CEP architecture, same backend API. Extension manifest change + AE-specific ExtendScript for timeline integration.
- **Implementation**: Fork CEP panel manifest to target After Effects. Add AE-specific ExtendScript (comp items instead of sequences). Subset the UI to AE-relevant features.
- **References**: Existing CEP panel as base, AE ExtendScript API

### 9.4 Final Cut Pro Export (P3 / M)
Direct FCPXML export alongside existing OTIO. While OTIO theoretically supports FCP, native FCPXML generation is more reliable and feature-complete.
- **Why**: Final Cut Pro users on Mac represent a large segment. OTIO→FCPXML conversion can lose metadata.
- **Implementation**: Generate FCPXML 1.11 directly from cut/edit data. Include: clip references, cuts, markers, captions. FCP can import FCPXML natively.
- **References**: FCPXML spec, DaVinci Resolve FCPXML export

### 9.5 Plugin Marketplace (P3 / L)
Community plugin registry — browse, install, update, and rate plugins from within the panel. Plugin authors publish via GitHub with a manifest. OpenCut aggregates into a searchable catalog.
- **Why**: Plugin system exists but discovery is manual (copy folders). A marketplace drives ecosystem growth.
- **Implementation**: GitHub-based registry (JSON file listing plugins with GitHub URLs). Panel fetches registry, displays cards with install/update buttons. Each plugin is a GitHub repo following the plugin manifest spec. Rate/review stored centrally.
- **References**: VS Code marketplace, Obsidian plugin browser, OBS plugin directory

---

## 10. Model & AI Infrastructure

### 10.1 Model Download Manager with Resume (P1 / M)
Background download queue for AI models with resume-on-failure, progress tracking, bandwidth throttling, and disk space estimation before download. Currently models download inline during first use, blocking the operation.
- **Why**: First-time use of any AI feature triggers a blocking download (sometimes 2-5GB). Users think the app crashed. Background pre-download with progress is critical.
- **Implementation**: Download manager service with queue. Pre-scan available models on first run, show "Recommended Downloads" based on GPU. Resume via HTTP Range headers. Store download state for recovery. Display progress in AI Model Manager panel.
- **References**: Ollama model management, HuggingFace Hub download, Stable Diffusion WebUI model manager

### 10.2 ONNX Runtime Everywhere (P2 / L)
Convert all PyTorch model inference paths to ONNX where possible. ONNX Runtime is faster, uses less memory, doesn't require full PyTorch, and supports DirectML (AMD GPUs) and TensorRT (NVIDIA optimization).
- **Why**: Many users have AMD GPUs where PyTorch CUDA doesn't work. ONNX with DirectML enables AMD GPU acceleration. Also reduces install size (no PyTorch needed for some features).
- **Implementation**: Export key models to ONNX: Silero VAD (done), style transfer (done), face detection, depth estimation. Add ONNX inference path alongside PyTorch in each module. Engine registry auto-selects ONNX when available.
- **References**: ONNX Runtime DirectML, Silero VAD ONNX (already done)

### 10.3 AMD GPU Support via ROCm / DirectML (P2 / L)
First-class support for AMD GPUs via ROCm (Linux) and DirectML (Windows). Currently all GPU features require NVIDIA CUDA.
- **Why**: ~30% of discrete GPU users have AMD. They're completely locked out of GPU acceleration. DirectML via ONNX Runtime is the simplest path on Windows.
- **Implementation**: Detect GPU vendor in gpu.py. For AMD on Windows: use ONNX Runtime with DirectML EP. For AMD on Linux: document ROCm PyTorch install. Update engine registry to report AMD compatibility per engine.
- **References**: ONNX Runtime DirectML, PyTorch ROCm, AMD GPUOpen

### 10.4 Apple Silicon / MPS Acceleration (P3 / M)
Optimize for Apple Silicon Macs using PyTorch MPS (Metal Performance Shaders) backend. Detect M1/M2/M3/M4 and route to MPS instead of CPU.
- **Why**: Growing Mac user base with powerful Apple Silicon GPUs. PyTorch MPS is now stable.
- **Implementation**: Detect Apple Silicon in gpu.py via `torch.backends.mps.is_available()`. Update all model loading paths to accept `device="mps"`. Test all AI features on MPS (some operations may need CPU fallback).
- **References**: PyTorch MPS backend, MLX (Apple's ML framework)

---

## Priority Summary

### P0 — Do These First
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 7.1 | FastAPI Migration | XL | Backend |
| 7.2 | Process Isolation for GPU Workers | XL | Backend |
| 1.1 | Transcript-Based Editing | XL | AI Editing |

### P1 — High Impact
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 1.2 | AI Eye Contact Correction | L | AI Editing |
| 1.3 | AI Overdub / Voice Correction | L | AI Editing |
| 1.4 | AI Lip Sync | L | AI Editing |
| 2.1 | Podcast Production Suite | L | Audio |
| 3.1 | Motion Tracking & Object Annotation | L | Video |
| 3.9 | Multi-Camera Audio Sync | M | Video |
| 4.1 | Watch Folder / Hot Folder | M | Workflow |
| 4.2 | Render Queue | M | Workflow |
| 4.8 | Best Take Selection | M | Workflow |
| 5.1 | Multi-Platform Batch Publish | L | Distribution |
| 6.1 | Drag-and-Drop File Handling | S | UX |
| 6.3 | Side-by-Side Before/After Preview | M | UX |
| 6.6 | Right-Click Custom Actions in Premiere | M | UX |
| 9.1 | DaVinci Resolve Deep Integration | L | Platform |
| 10.1 | Model Download Manager with Resume | M | Infrastructure |

### P2 — Valuable
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 1.5 | AI Voice Conversion / RVC | L | AI Editing |
| 1.6 | AI Auto-Color Grading | M | AI Editing |
| 1.7 | AI Content Moderation | M | AI Editing |
| 2.2 | Audio Restoration Toolkit | M | Audio |
| 2.3 | Real-Time Voice Conversion | L | Audio |
| 2.4 | AI Sound Effects from Video | M | Audio |
| 2.6 | Audio Spectrum Analyzer & Loudness Meter | S | Audio |
| 3.2 | AI Sky Replacement | M | Video |
| 3.3 | AI Relighting | L | Video |
| 3.6 | HDR/SDR Tone Mapping | M | Video |
| 3.10 | Video Comparison / Diff Tool | S | Video |
| 4.3 | Conditional Workflow Steps | M | Workflow |
| 4.4 | Undo Stack / Operation History | M | Workflow |
| 4.5 | EDL / AAF Import & Export | M | Workflow |
| 4.7 | Script/Storyboard Integration | L | Workflow |
| 5.2 | AI SEO Optimizer | M | Distribution |
| 5.3 | Broadcast QC / Standards Checker | M | Distribution |
| 6.2 | Workspace Layouts | M | UX |
| 6.4 | Interactive Waveform Timeline | L | UX |
| 6.5 | Mini Player / Preview Window | L | UX |
| 7.3 | Auto-Update Mechanism | M | Backend |
| 7.4 | Model Quantization / Optimization | M | Backend |
| 7.7 | MCP Server Expansion | M | Backend |
| 8.1 | Review & Approval Links | L | Collaboration |
| 8.2 | Team Presets via Shared Folder | S | Collaboration |
| 9.2 | Standalone Web UI | XL | Platform |
| 10.2 | ONNX Runtime Everywhere | L | Infrastructure |
| 10.3 | AMD GPU Support | L | Infrastructure |

### P3 — Future Expansion
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 1.8 | AI Scene Description / Alt-Text | M | AI Editing |
| 1.9 | AI Video Summarization | M | AI Editing |
| 1.10 | AI Talking Head / Avatar Generation | XL | AI Editing |
| 2.5 | Spatial Audio / Immersive Audio | M | Audio |
| 2.7 | Stem Remix / Stem Effects | M | Audio |
| 3.4 | 360 Video Support | L | Video |
| 3.5 | AI Deinterlacing | S | Video |
| 3.7 | Lens Distortion Correction | S | Video |
| 3.8 | Pika-Style Object Effects | XL | Video |
| 4.6 | Project Archival / Package | M | Workflow |
| 5.4 | Thumbnail A/B Generator | M | Distribution |
| 5.5 | Export Profiles with Watermark Toggle | S | Distribution |
| 7.5 | Remote Processing / Render Node | XL | Backend |
| 7.6 | Webhook / API Notifications | S | Backend |
| 8.3 | Timestamped Project Notes | S | Collaboration |
| 9.3 | After Effects Extension | XL | Platform |
| 9.4 | Final Cut Pro Export | M | Platform |
| 9.5 | Plugin Marketplace | L | Platform |
| 10.4 | Apple Silicon / MPS Acceleration | M | Infrastructure |

---

## 11. Screen Recording & Tutorial Post-Production

### 11.1 Auto Zoom-to-Cursor (P1 / M)
Detect mouse clicks in screen recordings and automatically generate smooth zoom-in keyframes centered on the cursor position, then zoom back out after the action. Transforms flat screen recordings into dynamic tutorials.
- **Why**: Loom, Tella, and ScreenPal all do this natively. Screen recording tutorials are a massive content category and OpenCut's editing tools don't address this workflow at all.
- **Implementation**: Analyze cursor movement data (if available from OBS/Loom metadata) or detect cursor position via template matching on each frame. Generate zoom keyframes at click points. Apply via FFmpeg `zoompan` filter or existing auto_zoom module.
- **References**: Loom auto-zoom, Tella, Screen Studio (Mac)

### 11.2 Click & Keystroke Overlay (P2 / M)
Render visual indicators for mouse clicks (ripple animation, highlight ring) and keystrokes (on-screen keyboard badge showing "Ctrl+S") burned into the video. Essential for software tutorials.
- **Why**: Every tutorial creator needs this. Currently requires separate tools like KeyCastr (Mac) or PointerFocus (Windows) during recording. Post-recording overlay is more flexible.
- **Implementation**: Parse click/keystroke log files from OBS plugins or detect via frame analysis. Render click ripple as animated overlay (Pillow/FFmpeg). Keystroke badges as drawtext overlay with background box. Configurable style, size, position.
- **References**: PointerFocus, KeyCastr, ScreenPal annotations

### 11.3 Callout & Annotation Generator (P2 / M)
Insert numbered step callouts, spotlight boxes (dim everything except a region), blur/redact regions, and arrow annotations at specific timestamps. GUI-driven annotation tool for tutorials and training videos.
- **Why**: Common need for training/onboarding videos. Currently requires After Effects or Camtasia. Bringing this to Premiere via OpenCut fills a gap.
- **Implementation**: Define annotation data as JSON (type, region, timestamp range, text). Render via FFmpeg drawbox/drawtext + Pillow for complex shapes. Panel UI: click on preview frame to place annotations, set duration.
- **References**: Camtasia annotations, Snagit, CloudApp

### 11.4 Screenshot-to-Video with Ken Burns (P3 / M)
Import a folder of screenshots or images and auto-generate a video with intelligent pan-and-zoom (Ken Burns) that focuses on regions of interest (UI elements, text, faces). Auto-detect focal points.
- **Why**: Product demos, changelogs, and presentations often start as screenshots. Auto-animating them saves hours of manual keyframing.
- **Implementation**: Detect regions of interest via edge detection, text detection (OCR), or saliency map. Generate pan-zoom keyframe paths that visit each ROI. Apply via FFmpeg `zoompan` with interpolated motion. Configurable per-image duration and transition.
- **References**: Ken Burns effect in iMovie, Apple Photos, Remotion

### 11.5 Dead-Time Detection & Speed Ramp (P2 / S)
Detect segments in screen recordings where nothing changes (no mouse movement, no keyboard activity, no screen changes) and either cut them or speed-ramp through them. Like silence removal but for visual inactivity.
- **Why**: Screen recordings have long idle periods (loading screens, waiting for builds, reading). Removing dead time makes tutorials 30-50% shorter.
- **Implementation**: Frame differencing (existing scene_detect logic) to find static segments. Combine with audio silence detection (existing). Speed-ramp or cut segments below activity threshold. Reuse speed_ramp.py and silence.py infrastructure.
- **References**: TimeBolt, Recut, auto-editor `--edit motion`

---

## 12. Gaming & Streaming

### 12.1 Highlight / Kill Clip Detection (P2 / L)
Analyze gaming footage to automatically detect highlight moments: audio spikes (reactions), sudden visual changes (kill feeds, score popups, achievement banners), and chat activity spikes. Mark or extract these as clips.
- **Why**: Medal.tv and Outplayed built entire products around this. Huge gaming creator audience that currently has no Premiere-based solution.
- **Implementation**: Multi-signal fusion: (1) Audio peak detection (existing). (2) Motion/visual change intensity (frame differencing). (3) Optional: OCR on kill feed regions. Score each segment, extract top N as clips. Reuse highlights scoring infrastructure.
- **References**: Medal.tv, Outplayed, NVIDIA ShadowPlay highlights

### 12.2 Chat Replay Overlay (P3 / M)
Import Twitch/YouTube chat logs and render them as an animated scrolling overlay synced to video timestamps. Configurable style, font, colors, position, and opacity.
- **Why**: Stream VODs without chat feel incomplete. Chat adds context and entertainment. Currently requires browser extensions or manual compositing.
- **Implementation**: Parse chat log formats (Twitch IRC log, YouTube chat JSON). Render as scrolling text overlay via Pillow frame compositor or FFmpeg drawtext with timestamp-based filtering. Style presets matching Twitch/YouTube native chat appearance.
- **References**: Chatty (Twitch), TwitchDownloader chat render, StreamElements

### 12.3 Auto Montage Builder (P2 / M)
Given a folder of gaming clips, automatically score each by excitement level (audio energy, motion intensity, duration) and assemble a montage with beat-synced cuts to a chosen music track.
- **Why**: Gaming montages are the most popular gaming content format. Automating the selection + assembly + beat-sync saves hours.
- **Implementation**: (1) Score clips via audio energy + motion analysis. (2) Select top N clips. (3) Detect beats in music track (existing librosa). (4) Trim clips to beat intervals. (5) Concatenate with transitions. All existing modules, just needs orchestration.
- **References**: Medal.tv montage, GoPro Quik auto-edit

### 12.4 Multi-POV Sync (P3 / M)
Synchronize multiple players' recordings of the same gaming session using audio fingerprinting (shared game audio) or timestamp correlation. Output as a multi-cam project for switching between perspectives.
- **Why**: Esports content and collaborative gaming videos need multi-POV editing. Extends the multi-cam audio sync feature (3.9) for gaming-specific use cases.
- **Implementation**: Extract game audio from each recording → cross-correlate to find time offsets (same as 3.9). Generate multicam XML for Premiere. Add UI for selecting "main" perspective.
- **References**: PluralEyes, Premiere multicam, Restream multi-track

---

## 13. Pro Color Science

### 13.1 Real-Time Color Scopes (P1 / L)
Waveform monitor, vectorscope, RGB parade, and histogram displayed in the panel. Update per-frame or on selected frame. Essential for broadcast-legal levels (IRE 0-100), skin tone validation, and exposure checking.
- **Why**: Every colorist needs scopes. DaVinci Resolve, Premiere's Lumetri, and FCPX all have them. OpenCut's color tools are blind without feedback.
- **Implementation**: FFmpeg `waveform`, `vectorscope`, `histogram`, and `signalstats` filters render scope images. Extract frame → apply scope filter → return as base64 image. Update on frame change or on-demand. Render in panel as image grid.
- **References**: DaVinci Resolve scopes, Premiere Lumetri scopes, ScopeBox

### 13.2 Three-Way Color Wheels (P2 / L)
Lift/Gamma/Gain color wheels with numeric precision inputs and an Offset wheel. The standard colorist control surface. Adjustments map to FFmpeg `colorbalance` and `eq` filters.
- **Why**: Standard color correction interface that every NLE and grading app provides. OpenCut's color correction is slider-based only (exposure, contrast, saturation). Wheels are more intuitive for color work.
- **Implementation**: Panel UI: three SVG color wheel widgets with click-drag. Map wheel positions to FFmpeg `colorbalance` filter values (rs, gs, bs, rm, gm, bm, rh, gh, bh). Preview via single-frame extraction.
- **References**: DaVinci Resolve primary wheels, Lumetri color wheels

### 13.3 HSL Qualifier / Secondary Color Correction (P2 / L)
Isolate a specific color range (hue, saturation, luminance) with feathered edges and apply corrections only to that range. Essential for tasks like "make the sky bluer" or "warm up skin tones" without affecting the rest of the image.
- **Why**: Industry-standard secondary correction tool. OpenCut only has global color adjustments. HSL qualification enables targeted, professional color work.
- **Implementation**: FFmpeg `colorkey` or `chromakey` for basic isolation. For proper soft-key: extract frame → OpenCV HSV range masking with feathered edges → apply corrections to masked region → composite. Preview the matte in panel.
- **References**: DaVinci Resolve qualifier, Lumetri HSL secondary

### 13.4 LOG / Camera Profile Pipeline (P1 / M)
Auto-detect camera LOG profiles (S-Log3, C-Log3, V-Log, ARRI LogC, BRAW) from metadata and apply the correct Input Display Transform (IDT). Support LUT stacking: technical LUT (LOG→Rec.709) + creative LUT with order control.
- **Why**: Modern cameras shoot LOG for maximum dynamic range. Without the correct LUT, footage looks flat and washed out. Auto-detection eliminates guesswork.
- **Implementation**: Read codec/color_primaries/transfer_characteristics from FFmpeg probe output. Map to known LOG profiles. Apply corresponding technical LUT from a bundled library (free Sony, Canon, Panasonic LUTs). Stack with user creative LUT via FFmpeg `lut3d` chain.
- **References**: DaVinci Resolve color management, Premiere Lumetri input LUT

### 13.5 Film Stock Emulation (P2 / M)
Apply photochemical film characteristics: specific Kodak/Fuji stock color response curves, organic film grain with gate weave, halation (light bloom on highlights), and color fade. More authentic than generic LUTs.
- **Why**: FilmConvert and Dehancer charge $100-150 for this. OpenCut could offer it free with open-source grain textures and color science.
- **Implementation**: Color response: custom 3D LUTs per film stock (generate from published sensitometry data). Grain: existing grain filter + gate weave via FFmpeg `overlay` with random sub-pixel offset. Halation: extract highlights → gaussian blur → blend screen mode. Package as "Film Emulation" presets.
- **References**: FilmConvert, Dehancer, GrainFactory (free grain textures)

### 13.6 Power Windows with Tracking (P2 / L)
Shape-based masks (circle, rectangle, gradient, custom polygon) that follow subjects via motion tracking. Apply corrections only inside/outside the mask. Core tool for localized grading: brighten a face, desaturate a background, add vignette centered on a moving subject.
- **Why**: DaVinci Resolve's power windows are a cornerstone of professional grading. Combined with existing face/object tracking, this enables pro-level localized work.
- **Implementation**: Define mask shapes in panel (SVG-based editor). Track mask position via MediaPipe (face) or SAM2 (object). Apply mask to FFmpeg filter graph via `drawbox`, `colorkey`, or Pillow alpha compositing. Per-frame mask position from tracking data.
- **References**: DaVinci Resolve power windows, After Effects masks

---

## 14. Documentary & Interview Workflow

### 14.1 Paper Edit from Transcript (P1 / L)
Display the full interview transcript as a text document. Editor highlights passages, drags them into a desired order in a "paper edit" column, and OpenCut auto-assembles the sequence from those text selections. The fundamental documentary editing workflow.
- **Why**: This is how documentaries are edited. Extends transcript-based editing (1.1) with a dedicated two-column UI optimized for long-form interview selection. Descript and Simon Says offer this.
- **Implementation**: Full transcript in left column, paper edit (ordered selections) in right column. Each selection maps to timecodes. "Assemble" button generates cut list → applies to timeline via existing `ocApplySequenceCuts` or exports OTIO. Support for adding notes per selection.
- **References**: Descript, Simon Says paper edit, Avid ScriptSync

### 14.2 Selects Bin with Ratings & Tags (P2 / M)
Rate clips with 1-5 stars and apply custom tags ("emotional", "B-roll", "expert-quote", "establishing", "reaction"). Filter and search across all project media by rating and tags. Persist ratings in the footage index database.
- **Why**: Standard documentary/corporate workflow for managing hundreds of clips. Premiere's labels are limited. A dedicated selects system integrated with the footage search database is more powerful.
- **Implementation**: Extend footage_index_db.py with `rating` and `tags` columns. Add UI card for rating/tagging current clip. Filter/search UI in the Search tab. Export selects list as CSV/JSON. Integrate with existing Smart Bins feature.
- **References**: Premiere label colors, Resolve smart bins, Silverstack

### 14.3 String-Out Reel Generator (P2 / M)
Auto-assemble all clips matching a filter (tag, rating, date range, speaker) into a continuous reel for review. Generates a playable sequence of all "selects" for a given topic before fine-cutting begins.
- **Why**: Standard documentary review step. After tagging selects (14.2), editors need to watch all selects for a topic back-to-back. Automates what's currently a manual bin→timeline drag.
- **Implementation**: Query footage_index_db for matching clips. Concatenate via FFmpeg concat (fast stream copy). Or generate timeline XML/OTIO for in-Premiere review. Add chapter markers per clip for navigation.
- **References**: Avid string-out, Resolve smart bins → timeline

### 14.4 Archival Footage Auto-Conform (P3 / M)
Detect mixed frame rates (24/25/29.97/30), aspect ratios (4:3/16:9/2.35:1), interlacing, and color spaces from archival/mixed sources. Auto-conform each clip to project settings with appropriate letterboxing/pillarboxing, frame rate conversion, and deinterlacing.
- **Why**: Documentaries mix modern footage with archival material. Manual conforming is tedious and error-prone. Auto-detection and correction streamlines ingest.
- **Implementation**: FFmpeg probe each clip for frame rate, resolution, interlacing, colorspace. Compare to project target settings. Auto-apply: deinterlace (bwdif), frame rate conversion (existing interpolation), scale with letterbox/pillarbox (existing reframe). Report non-conformant clips in a dashboard.
- **References**: DaVinci Resolve auto-conform, Premiere interpret footage

---

## 15. Corporate & Brand Video

### 15.1 Brand Kit Enforcement (P2 / M)
Define an approved brand profile: hex color palette, font families, logo assets, and style guidelines. OpenCut warns when off-brand elements are detected (wrong colors in captions, unapproved fonts, missing logo watermark) and offers auto-correction.
- **Why**: Corporate teams need consistency across videos. Brand guidelines are typically a PDF that editors must manually reference. Machine-enforced guidelines prevent mistakes.
- **Implementation**: Brand kit JSON config: `{ colors: ["#FF5500", ...], fonts: ["Inter", ...], logo: "logo.png", rules: { require_logo: true, require_lower_third_font: "Inter" } }`. Check caption styles, overlay colors, and fonts against kit. Warn in panel with "Fix" button.
- **References**: Canva Brand Kit, Visme brand management

### 15.2 Auto Lower-Thirds from Data Source (P1 / M)
Import a CSV, Excel, or JSON file with name/title/company data and batch-generate lower-third title cards from a chosen template. Auto-insert at speaker change timestamps (from diarization).
- **Why**: Corporate videos, conferences, and webinars need lower-thirds for every speaker. Creating them manually is tedious. Pairing with existing diarization automates the entire process.
- **Implementation**: Parse data source (CSV with columns: name, title, organization). Apply to lower-third template (FFmpeg drawtext or Pillow renderer with background box). Place at diarization speaker-change timestamps. Export as overlay video or burn into source.
- **References**: Premiere Essential Graphics templates, AutoCut lower-thirds

### 15.3 Template-Based Video Assembly (P2 / L)
Pre-built video templates with placeholder zones: intro sequence, talking-head section, B-roll zone, lower-third zone, call-to-action end screen. User drops media into zones and OpenCut assembles the final video.
- **Why**: Corporate teams produce repetitive content (weekly updates, product demos, testimonials). Templates make non-editors productive and ensure consistency.
- **Implementation**: Template JSON: ordered sections with type (video/image/text), duration, transition, and placeholder ID. Panel UI shows template preview with drop zones. User assigns media per zone. Assembly via FFmpeg concat + overlay chain.
- **References**: Canva Video templates, Biteable, InVideo, Remotion

---

## 16. Music Video & Audio-Reactive

### 16.1 Beat-Synced Auto-Cuts (P1 / M)
Automatically place cuts on detected beats or bar boundaries. Given multiple clips and a music track, assemble an edit where every cut lands exactly on a beat. Configurable density (every beat, every 2nd beat, every bar).
- **Why**: OpenCut detects beats (librosa) and adds markers, but doesn't use them to drive editing decisions. Closing this loop enables one-click music video assembly.
- **Implementation**: Detect beats (existing). Assign clips to beat intervals (round-robin, random, or scored by motion energy). Trim clips to beat duration. Concatenate with optional transitions on cuts. Output as timeline or rendered video.
- **References**: GoPro Quik, CapCut auto-beat-sync, Adobe Premiere beat edit

### 16.2 Lip Sync Verification (P2 / M)
Compare audio waveform timing to detected mouth movements and flag segments where they drift beyond a configurable threshold (e.g., >2 frames). Essential for music videos, dubbed content, and ADR verification.
- **Why**: Lip sync drift is invisible until the final render. Catching it early saves re-editing. No current tool automates this check.
- **Implementation**: Extract mouth region movement via MediaPipe face mesh (existing). Compute mouth-open energy curve. Cross-correlate with audio energy envelope. Flag segments where correlation drops below threshold or offset exceeds N frames.
- **References**: Premiere manual sync, Pro Tools VocAlign concept

### 16.3 Rhythm-Driven Auto-Effects (P2 / M)
Tie visual effect intensity to audio amplitude or beat triggers: zoom pulse on each beat, color flash on drops, shake on bass hits, brightness pump on snare. Keyframe-free audio-reactive effects.
- **Why**: Music video editors manually keyframe effects to beats. Automating this with audio analysis creates professional results in seconds.
- **Implementation**: Map audio features (beat times, amplitude envelope, spectral bands) to effect parameters (zoom scale, brightness, RGB shift, shake offset). Apply via FFmpeg filter expressions with per-frame parameter injection. Configurable mapping: which audio feature → which effect → intensity curve.
- **References**: After Effects audio-to-keyframe, CapCut beat effects

### 16.4 Audio-Reactive Visualizers (P3 / L)
Generate music visualization overlays: waveform bars, circular spectrum, particle field, frequency waterfall. Driven by the audio track, rendered as a transparent overlay layer.
- **Why**: Music-focused content (Spotify canvases, podcast audiograms, music promo clips) needs visualizers. Currently requires After Effects templates.
- **Implementation**: FFmpeg `showwaves`, `showspectrum`, `showfreqs` filters for basic visualizers. For advanced (circular, particles): Python + Pillow per-frame renderer driven by librosa spectral analysis. Render as transparent overlay → composite.
- **References**: FFmpeg showwaves, Wavacity, Renderforest audio visualizer

---

## 17. VFX & Compositing

### 17.1 Planar Tracking (P2 / XL)
Track a flat surface (screen, sign, billboard, wall) through camera movement and perspective changes. Output corner-pin tracking data for inserting replacement content that sticks perfectly to the surface.
- **Why**: Screen replacements, sign replacements, and tracked inserts are bread-and-butter VFX tasks. Currently requires After Effects or Mocha. Local planar tracking would be a major capability addition.
- **Implementation**: OpenCV `findHomography` + feature matching (ORB/SIFT) between frames. Track 4 corner points. Apply perspective warp to replacement image/video per frame. Composite via Pillow or FFmpeg overlay with perspective transform.
- **References**: Mocha (Boris FX), After Effects corner pin, OpenCV homography

### 17.2 3D Camera Solver (P3 / XL)
Solve camera motion from footage to determine 3D camera path. Enables adding text or objects that appear locked to the 3D scene (ground-plane text, floating labels, HUD overlays).
- **Why**: High-end VFX capability. Complex to implement fully, but even a basic ground-plane solve opens many creative possibilities. COLMAP is open-source and battle-tested.
- **Implementation**: Extract frames → COLMAP for sparse reconstruction + camera solve → export camera data. For basic use: track ground plane points, compute plane normal + camera transform, render 3D text/objects via simple projection math.
- **References**: COLMAP, Blender camera tracking, After Effects 3D camera tracker

### 17.3 Clean Plate Generation (P2 / L)
Generate a "clean plate" — a single static frame with all moving objects removed — from a video clip. Useful as a background for compositing, green screen replacement base, or object removal.
- **Why**: Manual clean plate creation takes hours. AI inpainting across multiple frames can auto-generate one. Extends existing object removal (SAM2 + ProPainter).
- **Implementation**: Analyze N frames → median composite (keeps static elements, removes moving objects). For remaining gaps: apply LaMA inpainting (existing dependency). Output single clean frame or short clean plate loop.
- **References**: After Effects median composite, Nuke clean plate, ProPainter

### 17.4 Advanced Stabilization Modes (P2 / M)
Expand stabilization beyond the current deshake/vidstab with: (1) Perspective-aware stabilization (corrects rotation + perspective, not just translation). (2) No-motion lock (complete lockdown for tripod-like result). (3) AI-fill edges (instead of cropping, inpaint the black edges from stabilization).
- **Why**: Current stabilization is basic FFmpeg vidstab. Premiere's Warp Stabilizer and DaVinci Resolve's stabilizer offer multiple modes. Edge fill via AI is a compelling differentiator.
- **Implementation**: Perspective: FFmpeg `vidstabtransform` with `optzoom=2` for perspective. No-motion: zero smoothing factor. AI edge fill: detect black edges → LaMA/ProPainter inpaint per frame. Combine with existing stabilization infrastructure.
- **References**: Premiere Warp Stabilizer, DaVinci Resolve stabilizer, GyroFlow (perspective)

---

## 18. Creative Effects Pack

### 18.1 Cinemagraph Creator (P2 / M)
Create cinemagraphs: still images with a small region of continuous motion (a waterfall flows while everything else is frozen, hair blows in the wind on an otherwise static portrait). User paints a motion mask, OpenCut loops the motion region seamlessly.
- **Why**: Cinemagraphs are high-engagement social media content. Currently requires Photoshop or Flixel ($200/year). A simple mask-paint + loop tool would be unique.
- **Implementation**: Extract reference frame as base. User paints motion mask in panel (canvas drawing tool). Loop: blend masked region from video over static base, with crossfade at loop point for seamless playback. Export as GIF, MP4, or APNG.
- **References**: Flixel, Plotagraph, Photoshop Timeline

### 18.2 Retro / Analog Effects Pack (P1 / S)
One-click presets for vintage looks: **VHS** (tracking lines, chroma bleed, date stamp, scan lines), **Super 8** (gate weave, heavy grain, light leaks, rounded corners), **Film Damage** (scratches, dust, splice marks, color fade), **Old TV** (static, horizontal hold, vertical roll).
- **Why**: Retro effects are evergreen content trends. Currently OpenCut has grain + vignette but nothing that fully simulates specific analog media. Easy to implement with FFmpeg filters, high visual impact.
- **Implementation**: VHS: FFmpeg `noise` + `chromashift` + `drawtext` (date stamp) + custom scan line overlay. Super 8: `geq` for vignette + `noise` + random `translate` for gate weave. Film damage: overlay transparent PNG sequences of scratches/dust. Package as one-click presets.
- **References**: Red Giant Universe VHS, FilmConvert, Premiere VHS presets

### 18.3 Glitch Effects Engine (P2 / M)
Parameterized digital glitch effects: datamosh (P-frame manipulation), RGB channel split/offset, scan-line displacement, pixel sorting, block corruption, and digital noise bursts. Keyframeable intensity for build-up → hit → recover patterns.
- **Why**: Glitch is a major aesthetic in music videos, trailers, and social content. Currently no glitch effects in OpenCut.
- **Implementation**: RGB split: FFmpeg `rgbashift` filter. Scan displacement: FFmpeg `scroll` + `displace`. Pixel sort: Python per-frame pixel sorting (numpy). Datamosh: remove I-frames from encoded video (requires raw bitstream manipulation). Block corruption: random block offset via numpy. Package as configurable preset with intensity/frequency controls.
- **References**: Data Glitch (AE plugin), FFmpeg rgbashift, pixel-sorting algorithms

### 18.4 Hyperlapse Stabilization (P2 / M)
Create smooth hyperlapses from handheld walking footage: sample every Nth frame, apply aggressive warp stabilization, and output smooth accelerated video. Goes beyond simple speed-up by handling the extreme camera shake of walking footage.
- **Why**: GoPro and phone footage from walking can be turned into professional hyperlapses with the right stabilization. Microsoft Hyperlapse set the standard but was discontinued.
- **Implementation**: (1) Sample frames at target speed factor. (2) Apply multi-pass vidstab with high smoothing (existing). (3) Optional: AI edge fill for stabilization crop (see 17.4). (4) Export. Simple pipeline of existing tools.
- **References**: Microsoft Hyperlapse (discontinued), GoPro TimeWarp, Instagram Hyperlapse

### 18.5 Tilt-Shift Miniature Effect (P3 / S)
Simulate a miniature model look: adjustable focal band position and width with gaussian blur gradient above and below, plus saturation boost and optional time-lapse speed-up. Makes real-world footage look like stop-motion miniatures.
- **Why**: Iconic visual effect that's easy to implement and popular for city/landscape content.
- **Implementation**: FFmpeg `tiltshift` filter (if available) or custom: apply strong gaussian blur via `boxblur`, mask with a gradient (sharp in center band, blurry above/below), boost saturation via `eq`. All standard FFmpeg operations.
- **References**: Instagram tilt-shift, iMovie miniature effect, Lensbaby

### 18.6 Light Leaks & Lens Flares (P3 / S)
Overlay library of organic light leak animations (warm amber wash, rainbow prism, film edge fog) and parameterized lens flares (position, color, intensity, element count). Keyframeable for dynamic reveals.
- **Why**: Stock light leak overlays cost $20-50. Bundling a set of free, high-quality overlays adds production value with zero effort. Simple blend mode compositing.
- **Implementation**: Bundle 8-12 transparent PNG sequences / short video clips of light leaks. Composite via FFmpeg `overlay` with blend mode (screen, add). Lens flare: procedural generation via Pillow (concentric circles, anamorphic streaks) positioned at user-specified point.
- **References**: Video Copilot Optical Flares, RocketStock free light leaks

### 18.7 Transition Sound Effects Pack (P2 / S)
Bundle a library of transition sound effects — whooshes, impacts, risers, swooshes, stingers, and hits — that auto-apply when using video transitions. Each transition type maps to a matching SFX category. Configurable volume, style (cinematic, corporate, playful), and override per transition instance.
- **Why**: Transitions without SFX feel incomplete. Editors manually layer sound effects on every transition, which is repetitive for videos with 50+ cuts. Auto-matching SFX to transition type eliminates this tedium. CapCut does this natively and it's one of its most-loved features.
- **Implementation**: Bundle 30-40 royalty-free SFX organized by category (whoosh_fast, whoosh_slow, impact_heavy, impact_light, riser_3s, stinger_dramatic, etc.). Map each transition type to a default SFX category (crossfade→none, wipe→whoosh, zoom→impact, slide→swoosh). On transition application, auto-insert matched SFX at transition point via FFmpeg amerge. UI: SFX browser card with preview, per-transition override selector, master volume slider.
- **References**: CapCut transition SFX, Premiere Essential Sound library, Artlist SFX packs

---

## 19. Stock Media & Asset Integration

### 19.1 In-Panel Stock Media Search (P1 / L)
Search free stock libraries (Pexels, Pixabay, Unsplash) directly from the OpenCut panel. Preview results, download, and insert into the Premiere project — all without leaving the editor.
- **Why**: Every editing session involves searching for B-roll, backgrounds, or music. Alt-tabbing to a browser breaks flow. Storyblocks and Artlist plugins for Premiere are subscription-only.
- **Implementation**: Pexels and Pixabay both have free APIs (API key required, free tier generous). Add Search tab card: query → display thumbnail grid → click to download to project media folder → import to Premiere via ExtendScript. Include both video and photo results.
- **References**: Storyblocks Premiere plugin, Pexels API, Pixabay API

### 19.2 Free Sound Effects Library (P2 / M)
Search and download sound effects from Freesound.org directly from the panel. Filter by category, duration, license. Preview in-panel, download, and import to project.
- **Why**: Freesound has 500,000+ Creative Commons sounds. Searching for SFX is a constant need. In-panel access eliminates the browser round-trip.
- **Implementation**: Freesound API (free with API key). Search → filter → preview (audio player widget) → download → import. Cache downloaded files in `~/.opencut/sfx/` for reuse. Show license attribution requirements.
- **References**: Freesound API, Epidemic Sound plugin (commercial reference)

### 19.3 Google Fonts for Captions (P2 / S)
Browse and use Google Fonts directly in caption styling. Preview fonts on sample text, download on demand, and use for styled caption overlays. Currently limited to system-installed fonts.
- **Why**: Caption styling is limited by installed fonts. Google Fonts offers 1,500+ free fonts. Downloading and installing fonts manually is friction.
- **Implementation**: Fetch Google Fonts API catalog (free, no key needed). Display font browser card with preview text. Download `.ttf` on selection to `~/.opencut/fonts/`. Pass font path to styled_captions.py renderer. Cache downloaded fonts.
- **References**: Google Fonts API, Canva font browser

### 19.4 License Tracking & Attribution Export (P3 / S)
Automatically log every stock asset used in a project (source, license type, attribution requirements). Export as a text/CSV attribution document for legal compliance.
- **Why**: CC-licensed assets require attribution. Tracking which assets were used across a project is manual and error-prone. Automated tracking prevents legal issues.
- **Implementation**: On each stock asset download (19.1, 19.2), record: filename, source URL, license type, attribution text, download date. Store in project-scoped SQLite. Export endpoint generates formatted attribution document.
- **References**: Storyblocks license tracking, Shutterstock Editor attribution

---

## 20. Accessibility & Compliance

### 20.1 Caption Compliance Checker (P1 / M)
Validate captions against accessibility standards: reading speed (characters per second — CPS), maximum characters per line (CPL), minimum display duration, simultaneous caption limit, positioning (not over key visuals). Report violations with fix suggestions.
- **Why**: FCC, WCAG, Netflix, and BBC all have specific caption standards. Violating them means accessibility failures, legal risk, or platform rejection. No current tool in OpenCut checks this.
- **Implementation**: Parse generated captions and check against configurable rulesets: Netflix (≤42 CPL, ≤200ms min duration, ≤20 CPS), FCC (≤32 CPL), BBC (≤160 WPM). Flag violations with line-by-line report. Auto-fix: re-segment captions to comply.
- **References**: Netflix Timed Text Style Guide, BBC Subtitling Guidelines, W3C WCAG

### 20.2 Audio Description Track Generator (P2 / L)
AI-generate a spoken audio description track for visually impaired viewers. Analyze video scenes, generate descriptions of key visual actions during dialogue pauses, synthesize as speech, and mix into an alternate audio track.
- **Why**: Legal requirement for broadcast in many jurisdictions. WCAG Level AA requires audio descriptions. Currently requires expensive human describers.
- **Implementation**: (1) Detect dialogue pauses via existing silence/VAD detection. (2) Extract key frames during pauses. (3) Describe via Florence-2 or LLM vision (see 1.8). (4) Generate speech via TTS (existing). (5) Mix description audio into gaps via audio ducking (existing). Output as separate AD track or mixed audio.
- **References**: WCAG audio description guidelines, YouDescribe, Amazon AD service

### 20.3 Color Blind Simulation Preview (P3 / S)
Preview the video through color vision deficiency simulations: deuteranopia (red-green, most common), protanopia (red-green), tritanopia (blue-yellow), and achromatopsia (total). Verify that color-coded information is still distinguishable.
- **Why**: ~8% of men have color vision deficiency. Videos relying on color alone (red/green indicators, color-coded graphics) are inaccessible. Quick preview catches these issues.
- **Implementation**: FFmpeg `colorchannelmixer` filter can simulate CVD by adjusting color matrix. Apply to preview frame and display in panel alongside original. Pillow also has CVD simulation via matrix transform.
- **References**: Photoshop soft proofing, Chrome DevTools CVD emulation, Coblis simulator

### 20.4 Photosensitive Seizure Detection (P1 / S)
Scan video for rapid flashing, extreme luminance changes, and saturated red flashes that could trigger photosensitive seizures. Flag violations against the ITU-R BT.1702 (Harding test) guidelines.
- **Why**: Legal liability in broadcast, YouTube demonetization risk, and genuine safety concern. The Harding test is standard for broadcast compliance. Simple to implement.
- **Implementation**: Analyze frame-to-frame luminance delta. Flag segments where: >3 flashes per second, luminance change >20 cd/m², or saturated red alternating frames. FFmpeg `flashdetect` filter or manual frame analysis. Report timestamps + severity.
- **References**: ITU-R BT.1702, Harding FPA, W3C WCAG 2.3.1

---

## 21. Emerging AI & Future Tech

### 21.1 Multimodal Timeline Copilot (P1 / L)
Chat with the timeline using natural language backed by multimodal AI that understands video, audio, and transcript simultaneously. "Find the part where she talks about pricing", "Show me the most emotional moment", "Cut everything before the first question." The AI navigates, selects, and edits.
- **Why**: Extends existing NLP command system (keyword matching) and chat editor (text-only) with actual video understanding. This is where the industry is heading — Frame (chat-driven editing), LAVE (LLM agent for editing).
- **Implementation**: Extend existing chat_editor.py with multimodal context: current transcript, scene descriptions (from 1.8), emotion scores (from existing deepface), and frame thumbnails. LLM generates API calls from conversational requests. Execute via existing route infrastructure.
- **References**: Frame (chat-driven editor), Descript AI actions, Adobe Firefly in Premiere

### 21.2 AI Frame Extension / Outpainting (P2 / L)
Extend video frames beyond their original boundaries — widen a 4:3 clip to 16:9 by AI-generating the missing edges, or extend a clip by generating additional frames at the end (temporal outpainting).
- **Why**: Format conversion without cropping. Extend too-short clips. Fill stabilization edges. Generative outpainting models are now production-quality.
- **Implementation**: Spatial outpainting: extract frames → run through image outpainting model (Stable Diffusion inpainting with mask, or dedicated outpainting model) → composite back into video. Temporal: use video generation model (Wan 2.1, existing) conditioned on last frame to generate continuation frames.
- **References**: Runway infinite canvas, Stable Diffusion outpainting, Pika extend

### 21.3 AI-Powered Rough Cut Assembly (P1 / XL)
Given raw footage and a creative brief (or script), AI analyzes all clips, selects the best takes, orders them narratively, adds music and captions, and outputs a rough cut. Human editor refines from there.
- **Why**: The holy grail of AI editing automation. Combines every existing OpenCut feature (transcription, highlight detection, scene detection, caption generation, audio processing) into an autonomous agent. Opus Clip and Vizard do this for short-form; this extends to long-form.
- **Implementation**: Multi-stage agent pipeline: (1) Transcribe + diarize all footage. (2) LLM matches transcript to brief/script. (3) Score and select best clips per section. (4) Arrange in narrative order. (5) Apply audio processing. (6) Generate captions. (7) Add music. (8) Output as OTIO/XML for import. Each step uses existing modules.
- **References**: Opus Clip, Vizard AI, Adobe Sensei, LAVE research

### 21.4 Real-Time AI Processing Pipeline (P3 / XL)
Apply AI effects (style transfer, background removal, face enhancement, color grading) in real-time during playback preview. Not for final render — for instant feedback while adjusting parameters.
- **Why**: Current workflow is: adjust parameters → process entire clip → review → repeat. Real-time preview at reduced resolution eliminates the processing loop.
- **Implementation**: Run AI models on single preview-resolution frames at 10-15 FPS. Use ONNX Runtime with GPU for speed. Display in mini-player (feature 6.5). Queue full-resolution render as background job when user is satisfied with preview parameters.
- **References**: DaVinci Resolve real-time processing, Topaz Video AI preview mode

### 21.5 Programmatic Video from Data (P2 / L)
Generate videos programmatically from data and templates — batch-create personalized videos (name, stats, charts) from a CSV/database, automated report videos from metrics, or dynamic social content from RSS feeds.
- **Why**: Remotion proved this market. Corporate teams need hundreds of personalized video variants. Marketing teams need data-driven content. No Premiere extension offers this.
- **Implementation**: Define video template (JSON with sections, text fields, image slots, data bindings). Bind data source (CSV/JSON/API). Render per-row: fill template → generate video via FFmpeg drawtext/overlay pipeline. Batch execute via existing batch_executor.
- **References**: Remotion, Creatomate, Shotstack, Synthesia

### 21.6 3D Gaussian Splat Viewer (P3 / XL)
Import 3D Gaussian splat captures (.ply) and render virtual camera moves through captured real-world 3D scenes. Export as video for use as B-roll or establishing shots. Enables "impossible camera moves" through real locations captured with a phone.
- **Why**: Gaussian splatting is the fastest-growing 3D capture format. Phone apps can now capture splats. Rendering them as video in a video editing tool bridges 3D capture and traditional editing.
- **Implementation**: Use gsplat or nerfstudio Python libraries to load .ply, define camera path (keyframed position/rotation), render frames, export as video. Requires GPU but models are small. Panel UI: define start/end camera positions, interpolation path.
- **References**: Luma AI, Polycam, gsplat, nerfstudio

### 21.7 AI Storyboard Generator from Script (P2 / L)
Given a script or shot list, generate a visual storyboard using AI image generation — one frame per shot description. Directors and producers can pre-visualize their project before shooting, then use the storyboard as editing reference during post-production.
- **Why**: Pre-visualization is standard in film production but requires a storyboard artist or expensive software like FrameForge. AI image generation from shot descriptions produces usable storyboards in minutes. Useful for pitch decks, pre-production planning, client approvals, and editing reference.
- **Implementation**: Parse script/shot list → LLM extracts shot descriptions with framing, action, and mood (existing llm.py). Generate image per shot using Stable Diffusion XL (via AUTOMATIC1111/ComfyUI API or local diffusers pipeline). Assemble as storyboard grid (Pillow) with shot number, description text, and duration. Export as PDF storyboard, image grid, or video animatic with duration per shot and temp music.
- **References**: Storyboarder (open-source Wonder Unit), Boords, Katalist AI storyboard, Midjourney pre-viz workflows

---

## 22. Developer & Scripting

### 22.1 Python Scripting Console (P2 / M)
In-panel Python REPL that exposes OpenCut's core modules. Run ad-hoc processing scripts: `from opencut.core.silence import detect_silences; detect_silences("clip.mp4", threshold=-30)`. For power users who need custom processing beyond the UI.
- **Why**: Opens OpenCut to infinite customization without requiring plugin development. Quick experiments and one-off processing.
- **Implementation**: New `/scripting/execute` endpoint (authenticated, local-only). Accept Python code string, execute in sandboxed scope with core modules pre-imported, return output. Panel UI: code editor with syntax highlighting (CodeMirror) and output console.
- **References**: DaVinci Resolve Python console, Blender scripting workspace

### 22.2 Macro Recording & Playback (P2 / M)
Record a sequence of panel actions (select clip → remove silence → normalize audio → add captions) as a replayable macro. Like workflows but created by demonstration rather than manual configuration.
- **Why**: Bridges the gap between manual one-off operations and configured workflows. "Do what I just did, but on all these clips."
- **Implementation**: Intercept API calls during recording session. Store as ordered list of endpoint + parameters. Playback: execute each call sequentially, substituting the target file. Panel UI: record/stop/play buttons. Save macros as named .opencut-macro files.
- **References**: Photoshop Actions, Excel macro recorder

### 22.3 Custom FFmpeg Filter Chain Builder (P3 / L)
Visual node-based editor for building custom FFmpeg filter chains. Drag filter nodes, connect inputs/outputs, preview result. Export as reusable preset. For users who know FFmpeg but want a GUI.
- **Why**: OpenCut wraps FFmpeg with high-level operations, but power users sometimes need specific filter combinations. A visual builder is more accessible than writing filter_complex strings.
- **Implementation**: Panel UI: node editor (canvas-based). Each node = one FFmpeg filter with exposed parameters. Connections = filter chain. Generate `-filter_complex` string from graph. Preview on single frame. Save as preset.
- **References**: FFmpeg filter documentation, ComfyUI (node-based AI), Natron (node compositor)

---

## 23. Media Asset Management

### 23.1 Proxy Generation Pipeline (P1 / L)
Automatically generate low-resolution proxy files (ProRes Proxy, H.264 quarter-res) on media ingest. Edit with responsive proxies, automatically relink to full-resolution originals at export time. Manages proxy storage and cleanup.
- **Why**: 4K/6K/8K footage is standard but kills editing performance. Premiere has proxy support but it's manual. Automated proxy generation on import makes high-res editing seamless.
- **Implementation**: On media import or watch folder ingest, generate proxy via FFmpeg (scale to 1/4 res, H.264 fast encode). Store in `~/.opencut/proxies/` with source file hash for relinking. Export pipeline auto-swaps proxy paths to originals. Add proxy status indicator in panel.
- **References**: DaVinci Resolve proxy workflow, Premiere Pro proxy, Kyno transcoding

### 23.2 AI Metadata Enrichment (P1 / L)
Automatically tag imported clips with searchable metadata: face identity (clustering across project), objects detected, shot type (close-up/medium/wide/aerial), indoor/outdoor, day/night, dominant colors, and spoken content (via transcription). All indexed in the footage search database.
- **Why**: Manual logging is the bottleneck in post-production. AI can do in seconds what takes hours. Every MAM system (CatDV, iconik, Frame.io) is adding AI tagging.
- **Implementation**: On import: (1) Run scene detection (existing). (2) Extract key frame per scene. (3) Florence-2 (existing dep) for object/scene captioning. (4) Face clustering via InsightFace (existing dep) → assign "Person A/B/C" tags (user renames later). (5) Store all tags in footage_index_db. UI: tag filter in Search tab.
- **References**: iconik MAM, Google Video Intelligence, AWS Rekognition

### 23.3 Duplicate / Near-Duplicate Detection (P2 / M)
Identify duplicate or near-identical clips across the project using perceptual hashing (visual fingerprinting). Detect exact copies, re-encoded versions, and trimmed variants. Flag redundant media to free storage.
- **Why**: Projects accumulate duplicate files across ingest rounds, copied folders, and multiple exports. Finding and removing duplicates saves storage and reduces confusion.
- **Implementation**: Compute perceptual hash (pHash or dHash) of first frame + middle frame + last frame per clip. Compare hashes across all indexed media. Flag matches above similarity threshold (>90%). UI: duplicates report with keep/delete actions.
- **References**: Gemini (Mac duplicate finder), VideoHash, imagededup

### 23.4 Structured Ingest Workflow (P2 / M)
Define rules-based ingest templates: auto-rename files by pattern (date_camera_reel_###), verify checksums (MD5/SHA256), apply metadata presets, sort into folder structure by camera/date/type, and generate a detailed ingest report.
- **Why**: Production environments need repeatable, auditable ingest. Manual file management leads to lost footage and naming chaos.
- **Implementation**: Ingest config JSON: rename pattern, folder structure template, metadata defaults, checksum verification toggle. Process: scan source folder → validate files → rename → copy to structured destination → verify checksum → index. Report: CSV/JSON with all file mappings.
- **References**: Silverstack, Kyno, Hedge, Shotput Pro

### 23.5 Storage Tiering & Auto-Archive (P3 / M)
Automatically move unused or older project media from fast local storage to cheaper cold storage (external drive, NAS, or cloud). Restore on demand when a project is reopened. Track what's archived and where.
- **Why**: AI models + source media + proxies + outputs consume hundreds of GB. Automatic tiering keeps active projects on fast storage and archives the rest.
- **Implementation**: Track file access times. After configurable idle period (30/60/90 days), move to archive path (user-configured). Replace with stub file in original location. On access attempt, prompt to restore. Store archive manifest in SQLite.
- **References**: iconik MAM tiering, LTO archival workflows, AWS Glacier

---

## 24. Professional Subtitling & Localization

### 24.1 Shot-Change-Aware Subtitle Timing (P1 / M)
Automatically detect scene cuts and snap subtitle in/out points so no subtitle spans a cut boundary. Enforce minimum gap between subtitle and cut (configurable, typically 2 frames). Improves readability and meets broadcast standards.
- **Why**: Subtitles that span cuts are jarring and fail broadcast QC. Netflix, BBC, and every broadcaster requires shot-change-aware timing. No current tool in OpenCut does this.
- **Implementation**: Run scene detection (existing) to get cut timestamps. Post-process generated captions: if a subtitle spans a cut, split it at the cut point with appropriate gap. Integrate into caption generation pipeline as an automatic post-processing step.
- **References**: Netflix Timed Text Style Guide, Subtitle Edit shot-change feature, EBU STL spec

### 24.2 Multi-Language Simultaneous Editing (P2 / L)
Edit subtitles for multiple languages side-by-side in a tabbed or split view. Timing is locked across all languages — translators only change text, not timing. Sync changes propagate to all language tracks.
- **Why**: Localization workflows require maintaining subtitles in 5-20+ languages simultaneously. Editing each language file separately is error-prone (timing drifts between versions).
- **Implementation**: Store subtitles as: base timing track + per-language text arrays. Panel UI: language tabs with shared timeline ruler. Edit text in any language, timing auto-syncs. Export per-language SRT/VTT/ASS files. Import existing translations to align with timing.
- **References**: CaptionHub, Subtitle Edit multi-file, Amara

### 24.3 Broadcast Closed Caption Export (P2 / M)
Export to broadcast closed-caption formats: CEA-608, CEA-708, EBU-TT, TTML, IMSC1, and WebVTT with positioning. These are required for TV broadcast, OTT platforms, and FCC compliance — SRT alone is insufficient.
- **Why**: SRT/VTT export exists but broadcast delivery requires CEA-608/708 embedded in the video stream or EBU-TT sidecar files. Without these, content can't air on television.
- **Implementation**: Generate EBU-TT XML and TTML from existing caption data. For CEA-608/708: embed in MP4 via FFmpeg `-c:s eia_608` or use `ccextractor` library for conversion. Add format selector in export options.
- **References**: CaptionHub, 3Play Media, ccextractor, FFmpeg closed captions

### 24.4 SDH / HoH Formatting (P2 / S)
Apply Subtitles for the Deaf and Hard-of-Hearing conventions automatically: add speaker identification (`[JOHN]:`), sound effect descriptions (`[door slams]`, `[phone rings]`), music notation (`♪ gentle piano ♪`), and positional placement for off-screen speakers.
- **Why**: Legal accessibility requirement in many jurisdictions. Manual SDH formatting is tedious. AI can detect non-speech sounds and auto-format.
- **Implementation**: Speaker labels: from existing diarization. Sound effects: detect via AudioSet classifier or simple energy-based non-speech detection. Music: detect via existing stem separation (vocals removed = music). Format using SDH conventions and insert into caption track.
- **References**: DCMP captioning guidelines, Netflix SDH spec, BBC subtitling guidelines

### 24.5 Subtitle Styling Per-Region Positioning (P3 / M)
Position subtitles dynamically per-frame to avoid covering important visual information. Detect faces, text, logos, and action in the frame, then place the subtitle in the least obstructive position.
- **Why**: Standard bottom-center positioning covers lower-third graphics, faces in interview setups, and on-screen text. Dynamic positioning is a premium feature in professional subtitling.
- **Implementation**: Analyze each frame's lower-third region for content (face detection, text/OCR, brightness). If obstructed, move subtitle to top or side. Apply positioning via ASS subtitle format (which supports per-line positioning). Preview in panel.
- **References**: Netflix forced narrative placement, professional subtitle positioning rules

---

## 25. Audio Post-Production (Film/TV)

### 25.1 ADR Cueing System (P2 / L)
Mark dialogue lines needing replacement (poor audio, script changes, noise), generate ADR cue sheets with timecode, original line text, character name, and scene context. Provide a record-and-sync workflow for re-recording replacement dialogue.
- **Why**: ADR is a standard film/TV post-production step. Currently requires Pro Tools or specialized ADR software. Integrating with OpenCut's existing transcription and TTS creates a unique workflow.
- **Implementation**: UI: mark transcript segments as "needs ADR". Generate cue sheet (PDF/CSV with timecode, line, character). Record mode: play guide audio, record replacement, auto-sync to original timecode. Optional: use TTS/voice clone as temp ADR fill until live recording is done.
- **References**: Pro Tools ADR, Nuendo ADR, EdiPrompt

### 25.2 Room Tone Matching & Generation (P1 / M)
Analyze production audio to extract the ambient room tone profile, then generate matching fill to smooth edits, patch dialogue gaps, and extend backgrounds seamlessly. Every cut point needs room tone to sound natural.
- **Why**: Every audio editor manually records or synthesizes room tone. AI extraction from existing audio is faster and more accurate. Critical for professional-sounding dialogue edits.
- **Implementation**: Extract room tone: find segments with lowest speech energy (existing silence detection), extract 5-10 second sample, loop seamlessly (crossfade loop points). Generate: use spectral analysis to synthesize matching ambient profile via filtered noise. Apply as fill layer between dialogue segments.
- **References**: iZotope Ambience Match, Pro Tools strip silence + tone fill, Fairlight room tone

### 25.3 M&E Mix Export (P2 / M)
Generate a Music & Effects stem (everything minus dialogue) automatically from the project audio. Essential for international versioning — the M&E mix is re-combined with foreign-language dubbed dialogue.
- **Why**: International distribution requires M&E mixes. Creating them manually means re-exporting every non-dialogue element. Using existing stem separation to remove dialogue and keep everything else is the AI shortcut.
- **Implementation**: Run Demucs/BS-RoFormer stem separation (existing) to isolate vocals. Invert: subtract vocals from original mix to get M&E. Or: if multi-track source, mute dialogue tracks and export. UI: one-click M&E export in deliverables section.
- **References**: Netflix delivery specs, Dolby deliverables, Pro Tools stem export

### 25.4 Automated Dialogue Premix (P2 / M)
Apply per-speaker processing chains automatically: de-ess sibilance, apply EQ curves matched to mic type, compress dynamic range, and level-match all dialogue clips to target LUFS. Produces a consistent dialogue premix from raw production audio.
- **Why**: Dialogue premixing is hours of manual work per project. Per-speaker processing using diarization data + existing audio_pro chain automates the tedious parts.
- **Implementation**: Diarize speakers (existing). Per speaker: analyze frequency profile → auto-select EQ preset (remove room resonance, add presence). Apply de-esser (existing pedalboard). Compress to target dynamic range. Level-match to target LUFS (existing). Chain via workflow engine.
- **References**: iZotope Dialogue Match, Premiere auto-duck + essential sound, Fairlight auto-level

### 25.5 Surround Sound Panning & Upmix (P3 / L)
Provide per-clip spatial panning to 5.1 or 7.1 surround sound fields. Auto-upmix stereo sources to surround. Monitor with stereo downmix preview for compatibility. Export as multichannel audio or channel-split WAV files.
- **Why**: Surround delivery is required for streaming platforms (Netflix, Disney+, Apple TV+) and theatrical. Basic upmix opens the door to surround workflows.
- **Implementation**: FFmpeg `pan` filter for surround routing. Upmix: center channel = mono sum, surrounds = phase-shifted ambient content. Panning: define per-clip pan position in surround field. Export via FFmpeg multichannel encoding (AC-3, E-AC-3, AAC multichannel).
- **References**: FFmpeg surround, DaVinci Resolve Fairlight, Nuendo surround panner

---

## 26. Motion Design & Animation

### 26.1 Kinetic Typography Engine (P1 / L)
Animate text per-character, per-word, or per-line with preset easing curves: bounce, elastic, typewriter, wave, cascade, spiral, explode, assemble. Extends beyond current animated captions to full motion graphics.
- **Why**: Animated titles exist in OpenCut but are limited to FFmpeg drawtext presets. A proper kinetic typography engine enables professional motion graphics without After Effects.
- **Implementation**: Define animation data: per-element transform keyframes (position, scale, rotation, opacity) with easing functions. Render via Pillow or Skia (existing optional dep) frame-by-frame. Compose over video via FFmpeg pipe. Ship 20+ animation presets. UI: preview animation on selected text.
- **References**: After Effects text animators, CapCut text effects, Apple Motion behaviors

### 26.2 Data-Driven Animation (P2 / L)
Bind motion graphics properties (bar heights, counter values, labels, colors) to CSV/JSON data sources. Auto-generate animated charts, tickers, scoreboards, and infographics from data. Batch-render personalized variants.
- **Why**: Corporate reports, sports content, financial summaries, and social media stats all need animated data visualization. Currently requires After Effects + scripting. Pairs with programmatic video (21.5).
- **Implementation**: Template JSON defines visual elements with data binding expressions: `{ type: "bar", height: "${data.revenue}", label: "${data.quarter}" }`. Load CSV rows, substitute values, render animated transitions between keyframes. Output as video overlay or standalone.
- **References**: Remotion, Cavalry, After Effects expressions, D3.js (design reference)

### 26.3 Shape Layer Animation (P3 / L)
Animate vector shapes: path morphing between keyframes (circle transforms into star), stroke drawing animation (line draws itself), fill color transitions, and compound shape operations (boolean union/subtract animated).
- **Why**: Logo reveals, icon animations, and abstract motion design are huge on social media. Currently only possible in After Effects, Motion, or Cavalry.
- **Implementation**: Define shapes as SVG paths. Interpolate between path keyframes (point-count must match or use smart path matching). Render via Skia or Pillow. Stroke animation: animate `dashoffset` to simulate drawing. Export as transparent overlay.
- **References**: After Effects shape layers, Lottie animations, Rive

### 26.4 Expression Scripting for Animation (P3 / M)
Expose a lightweight scripting layer that can drive any animatable property based on time, audio amplitude, or other parameters. Example: `scale = 1 + sin(time * 3) * 0.1` makes an element pulse. Like After Effects expressions but in Python.
- **Why**: Bridges the gap between manual keyframing and fully programmatic animation. Power users can create complex behaviors with simple expressions.
- **Implementation**: Per-property expression field (Python eval with sandboxed globals: `time`, `frame`, `audio_amplitude`, `beat`, `random`, `sin`, `cos`, `lerp`). Evaluate per frame during render. Apply to: position, scale, rotation, opacity, color, any numeric parameter.
- **References**: After Effects expressions, Cavalry expressions, CSS animations

---

## 27. AI Safety & Content Authenticity

### 27.1 C2PA Content Credentials Embedding (P2 / M)
Embed Content Credentials metadata (C2PA standard) into exported video files, recording the full edit history: which AI tools were used (upscaling, voice cloning, inpainting), what modifications were made, and the original source provenance.
- **Why**: C2PA is becoming the industry standard for content authenticity (adopted by Adobe, BBC, Google, Microsoft). Social platforms are starting to flag AI-modified content without provenance data. Being C2PA-compliant makes OpenCut output trusted.
- **Implementation**: Use `c2pa-python` library to create and embed C2PA manifests. Record each AI operation as a C2PA action assertion. Embed in MP4 container on export. Add toggle: "Embed Content Credentials" in export settings.
- **References**: C2PA specification, Adobe Content Authenticity Initiative, c2pa-python

### 27.2 Invisible AI Watermarking (P2 / M)
Apply imperceptible watermarks to AI-generated or AI-modified frames (upscaled, face-swapped, inpainted, style-transferred) so synthetic content is machine-detectable without visible artifacts. Survives compression and re-encoding.
- **Why**: AI-generated content regulation is increasing globally. Self-watermarking demonstrates responsible AI use and may become legally required. Google's SynthID and Meta's Stable Signature set the standard.
- **Implementation**: Frequency-domain watermarking: embed a binary string in DCT coefficients of AI-processed frames. Lightweight encoder (NumPy/OpenCV) runs during any AI processing step. Decoder extracts watermark for verification. Survives JPEG compression and mild re-encoding.
- **References**: SynthID (Google), Stable Signature (Meta), StegaStamp

### 27.3 Deepfake Detection & Confidence Scoring (P3 / L)
Analyze clips for signs of AI manipulation: face swap artifacts, lip sync inconsistencies, temporal flickering in AI-generated regions, and GAN fingerprints. Output per-segment confidence scores and flag suspected synthetic content.
- **Why**: Editors receiving third-party footage need to verify authenticity. Broadcasters need to validate source material. Self-check before publishing prevents accidental deepfake distribution.
- **Implementation**: Face region analysis: check for blending boundaries, inconsistent lighting between face and neck, temporal jitter in face region vs. stable background. Use lightweight CNN classifier trained on deepfake detection datasets. Report: timestamped confidence scores with visual heatmap.
- **References**: FaceForensics++, Microsoft Video Authenticator, Sensity AI

### 27.4 Provenance Chain & Audit Trail (P3 / S)
Generate a signed JSON manifest listing every operation applied to a video file: source file hash, each processing step (with parameters), AI models used, timestamps, and output file hash. Exportable for compliance documentation.
- **Why**: Broadcast compliance, legal proceedings, and brand safety increasingly require full edit provenance. Lightweight to implement since job history already exists in SQLite.
- **Implementation**: On export, query jobs.db for all operations applied to the source file. Generate JSON manifest: `{ source_hash, operations: [{endpoint, params, model, timestamp}], output_hash, opencut_version }`. Sign with HMAC using server key. Include as sidecar file or embed in MP4 metadata.
- **References**: C2PA provenance, blockchain-based media verification

---

## 28. Automated QA / QC

### 28.1 Black Frame & Frozen Frame Detection (P1 / S)
Scan video for segments of solid black frames or identical frozen frames exceeding a configurable duration threshold. Flag with timestamps for review. Catches dead air, failed renders, and signal dropouts.
- **Why**: Standard broadcast QC check. Black/frozen frames fail every delivery spec. Easy to implement and high value for catching errors before delivery.
- **Implementation**: FFmpeg `blackdetect` filter for black frames. Frozen: frame differencing (SSIM or pixel MSE between consecutive frames) below threshold. Report: timestamps, durations, frame numbers. Add as automatic post-export check.
- **References**: Telestream Vidchecker, Baton QC, FFmpeg blackdetect

### 28.2 Audio Phase & Silence Gap Check (P1 / S)
Detect out-of-phase stereo audio (which cancels on mono playback) and unexpected silence gaps in the audio track. Phase issues are invisible until content plays on a mono speaker (phones, some TVs).
- **Why**: Phase-inverted audio is a common recording mistake that's inaudible on headphones but destroys the audio on mono playback. Silence gaps indicate missing audio or bad edits.
- **Implementation**: Phase: FFmpeg `aphasemeter` filter, flag segments where phase < -0.5. Silence gaps: reuse existing silence detection with adjusted parameters (detect silence > 2 seconds in final mix). Report: timestamped list of issues.
- **References**: FFmpeg aphasemeter, Nuendo phase correlation meter, broadcast QC standards

### 28.3 Color Bars & Slate Detection (P2 / S)
Automatically locate and mark leader elements: SMPTE color bars, countdown leaders, slates/clapper boards, and tone signals at the beginning of raw footage. Auto-trim or mark for removal.
- **Why**: Raw camera footage and tape captures start with bars/tone/slate. Auto-detecting them saves manual trimming. Slate OCR can extract scene/take metadata.
- **Implementation**: Color bars: detect SMPTE pattern via histogram analysis of first 60 seconds. Slate: detect high-contrast text card via frame analysis. Tone: detect 1kHz sine in audio (FFmpeg `astats`). Mark boundaries for auto-trim. Optional: OCR slate text for metadata.
- **References**: DaVinci Resolve auto-detect leader, Telestream Vidchecker

### 28.4 Dropout & Glitch Detection (P2 / M)
Identify digital dropout artifacts: single-frame glitches, macroblocking, timecode breaks, and bitstream errors across the full program. Common in ingested tape footage, recovered files, and corrupted recordings.
- **Why**: Dropouts are invisible in scrubbing but obvious during playback. Automated detection catches them before delivery when they can still be fixed (re-ingest, patch, interpolate).
- **Implementation**: Analyze frame-to-frame SSIM/PSNR: sudden drops indicate glitches. Detect macroblocking via edge density analysis. Timecode breaks: parse timecode track for non-sequential jumps. FFmpeg `decoding_error` flag catches bitstream issues. Report: severity + timestamp + thumbnail.
- **References**: Baton QC, CerifyNow, FFmpeg error detection

### 28.5 Comprehensive QC Report Generator (P2 / M)
Run all QC checks (black frames, frozen frames, audio phase, silence gaps, loudness, color levels, dropouts) in a single pass and generate a professional QC report with pass/fail per check, screenshots of failures, and an overall delivery verdict.
- **Why**: Individual QC checks are useful but a comprehensive automated report is what production houses actually need. Run once before delivery, get a go/no-go assessment.
- **Implementation**: Orchestrate all QC modules in sequence. Aggregate results into structured report JSON. Generate PDF/HTML report with: summary table, per-check detail with frame screenshots, audio level graphs, and overall pass/fail. Configurable rulesets per delivery standard (broadcast, Netflix, YouTube).
- **References**: Telestream Vidchecker reports, Netflix delivery specs, EBU QC guidelines

---

## 29. Video Intelligence & Analytics

### 29.1 Shot Type Auto-Classification (P1 / M)
Classify each shot as close-up, medium shot, wide shot, extreme close-up, over-the-shoulder, aerial/drone, POV, or insert shot. Tag in footage index for searchable filtering ("show me all wide shots").
- **Why**: Shot type is fundamental metadata for editing decisions. Auto-classification eliminates manual logging. Enables intelligent features like "alternate between close-up and wide for interviews."
- **Implementation**: Extract key frame per scene (existing scene detection). Run through lightweight image classifier (fine-tuned MobileNet or CLIP zero-shot: "a close-up shot of a person"). Store classification in footage_index_db. UI: filter by shot type in Search tab.
- **References**: Google Video Intelligence, Clarifai, Shotdeck (shot type database)

### 29.2 On-Screen Text / OCR Extraction (P2 / M)
Detect and extract visible text from video frames: signs, lower-third graphics, burned-in captions, whiteboards, documents, presentation slides. Index as searchable metadata with frame timestamps.
- **Why**: Text visible in video is valuable metadata that's invisible to audio-only search. Enables searching footage by on-screen content ("find the frame where the slide says Q4 revenue").
- **Implementation**: Sample frames at 1/second. Run OCR via PaddleOCR (lightweight, pip installable) or Tesseract. Filter: deduplicate repeated text across consecutive frames. Index in footage_index_db with frame timestamps. UI: searchable text results with frame thumbnails.
- **References**: PaddleOCR, Google Vision API (design reference), EasyOCR

### 29.3 Face Tagging & Recognition Across Project (P2 / L)
Detect and cluster faces across all project media. Assign consistent identity labels ("Person A", "Person B") that persist across clips. User names the clusters once, then face identity is searchable everywhere.
- **Why**: "Show me all clips of the CEO" or "find every frame with Person B" are common needs. InsightFace (already a dependency) supports face recognition.
- **Implementation**: On media index: extract faces per scene key frame via InsightFace (existing). Compute face embeddings. Cluster embeddings (DBSCAN or Chinese Whispers). Store cluster assignments in footage_index_db. UI: face gallery with rename capability. Search by face identity.
- **References**: Apple Photos face recognition, Google Photos, InsightFace

### 29.4 Emotion & Energy Timeline (P2 / M)
Generate a visual timeline overlay showing emotional intensity, audience energy, and engagement potential across the video. Combine facial expression analysis, audio energy, speech pace, and transcript sentiment into a composite "energy curve."
- **Why**: Identifies the most engaging and least engaging segments at a glance. Directly supports highlight extraction and pacing decisions. Extends existing emotion detection from individual clips to timeline-wide visualization.
- **Implementation**: Combine existing signals: (1) deepface emotion scores per scene. (2) Audio RMS energy envelope. (3) Speech rate from transcript word timestamps. (4) LLM sentiment from transcript chunks. Normalize and composite into single energy curve. Render as color-coded timeline overlay (green=high energy, yellow=medium, red=low).
- **References**: YouTube audience retention graph, Opus Clip engagement scoring

---

## 30. Advanced Timeline Operations

### 30.1 Smart Fit-to-Fill (P2 / M)
Automatically speed-ramp a source clip to exactly fill a target timeline gap while preserving natural motion cadence. If a 10-second clip needs to fill an 8-second gap, intelligently accelerate with smooth easing rather than uniform speed change.
- **Why**: Common editing task that's currently manual (calculate speed factor, apply, check). Automating it with intelligent easing produces better results than uniform speed change.
- **Implementation**: Calculate required speed factor. If minor (<15% change): uniform speed via FFmpeg `setpts`. If larger: apply ease-in/ease-out speed curve (existing speed_ramp infrastructure). If extreme: use RIFE frame interpolation (existing) to smooth slow-motion or frame blending for speed-up.
- **References**: Avid fit-to-fill, Final Cut Pro fit-to-fill, Premiere rate stretch

### 30.2 Through-Edit Cleanup (P2 / S)
Scan the timeline for adjacent cuts from the same source clip with continuous timecode (accidental double-cuts, incomplete trims). Offer to merge them into single clips, removing unnecessary edit points that clutter the timeline.
- **Why**: Timeline accumulates unnecessary edit points during rough cutting. Cleaning them improves performance and readability. Currently a manual process.
- **Implementation**: Via ExtendScript: iterate track items, detect adjacent items where `source.path` matches and `outPoint_A ≈ inPoint_B` (within 1 frame). Offer merge: extend first item's outPoint and remove second item. Report: list of merge-able edits.
- **References**: Avid "Add Edit" / remove edit point, Premiere through-edit removal

### 30.3 Adjustment Layer Presets (P2 / M)
Apply saved correction stacks (LUT + denoise + vignette + color correction) as non-destructive "adjustment layers" spanning selected timeline ranges. Change the preset and all clips under it update. Enable per-scene or per-act grading.
- **Why**: Applying corrections per-clip is repetitive. Adjustment layers (Premiere supports them natively) with OpenCut-managed presets enable scene-wide grading changes in one click.
- **Implementation**: Generate a transparent video overlay with the correction baked in (FFmpeg filter chain applied to transparent/black source). Import as clip above target range. Or: via ExtendScript, create adjustment layer in Premiere and apply Lumetri preset. OpenCut manages the preset library.
- **References**: Premiere Pro adjustment layers, DaVinci Resolve group grades

### 30.4 Nested Sequence Auto-Collapse (P3 / M)
Detect repeated clip groups in the timeline (same sequence of clips appearing in multiple places) and offer to convert them into nested sequences / compound clips. Reduces timeline complexity and enables group editing.
- **Why**: Complex edits with repeated elements (recurring intro, bumper, lower-third sequences) clutter the timeline. Auto-detection and nesting is a productivity feature.
- **Implementation**: Via ExtendScript: scan track items for recurring patterns (same source clips in same order with similar durations). Offer to nest: create new sequence from pattern, replace instances with nested reference. UI: pattern report with "Nest" button.
- **References**: Premiere nested sequences, FCPX compound clips, Resolve compound clips

### 30.5 Ripple Edit Automation (P2 / S)
After any cut operation (silence removal, filler removal, repeated take removal), automatically ripple-close all gaps in the timeline. Currently, cut operations may leave gaps that need manual closing.
- **Why**: The last step of any cut operation is closing the gaps. Automating this completes the workflow end-to-end. Simple ExtendScript operation.
- **Implementation**: After applying cuts via `ocApplySequenceCuts`, iterate all tracks and find gaps (track items with space between them). Close each gap by shifting subsequent items earlier. Respect locked tracks. Add as optional "auto-ripple" toggle in cut operations.
- **References**: Premiere ripple delete, Resolve ripple trim

---

## 31. Live Production & Streaming Post-Processing

### 31.1 Multi-Track ISO Recording Ingest (P2 / M)
Import multi-camera ISO recordings (one file per source from OBS, vMix, ATEM) and auto-sync into a multicam sequence using embedded timecode, filename timestamps, or audio waveform matching. Auto-create multitrack timeline with each source pre-routed.
- **Why**: Live events and streams are increasingly recorded as multi-track ISOs. Importing and syncing them is manual and tedious. Automated ingest streamlines post-event editing.
- **Implementation**: Detect multi-track recordings (OBS .mkv with multiple video streams, or separate files per camera). Extract audio → cross-correlate for sync offsets (existing 3.9 infrastructure). Generate multicam XML/OTIO. Import as synchronized multi-angle sequence.
- **References**: OBS multi-track recording, vMix ISO, ATEM ISO recording

### 31.2 Instant Replay Builder (P2 / M)
Flag moments during review playback and auto-generate replay clips with configurable slow-motion ramp (existing speed_ramp), replay overlay graphics ("REPLAY" text + border), and transition in/out. Built for sports, esports, and event recap edits.
- **Why**: Replay generation is a core sports/esports editing task. All the pieces exist (speed ramp, text overlay, transitions) but aren't assembled into a replay workflow.
- **Implementation**: Mark moments in review (timestamp + label). Per replay: extract ±5 seconds around mark → apply slow-motion ramp at marked point → overlay "REPLAY" graphic → add transition dissolve → export clip. Batch: generate all replays from a session in one pass.
- **References**: vMix instant replay, Slomo.tv, NewTek 3Play

### 31.3 Stream Recording Auto-Chaptering (P2 / S)
Analyze long-form stream recordings (2-8 hours) and auto-generate chapter markers at natural break points: scene changes, topic shifts (via transcript analysis), title screen appearances, and extended silence/BRB periods.
- **Why**: Stream VODs are unwieldy without chapters. Automated chaptering makes them navigable for viewers and editors. Combines existing scene detection + chapter generation.
- **Implementation**: Run scene detection (existing) for visual breaks. Run chapter_gen (existing) on transcript for topic changes. Detect BRB/intermission screens via template matching or extended static frame detection. Merge all signals into chapter list. Export as YouTube chapter format or timeline markers.
- **References**: YouTube auto-chapters, Twitch VOD chapters, StreamLadder

### 31.4 Stream Highlight Reel Generator (P3 / M)
Automatically generate a highlight reel from a long stream recording: detect high-energy moments (chat spikes, audio peaks, viewer count peaks if available), extract clips, and assemble with transitions and music. One-click "stream recap" for social media.
- **Why**: Streamers need to cut highlights from 4-8 hour streams for YouTube/TikTok. Automates the most tedious part of being a content creator. Combines existing highlight detection with gaming clip detection (12.1).
- **Implementation**: Multi-signal scoring: audio energy spikes + transcript engagement keywords + visual excitement (motion intensity) + optional chat log analysis. Extract top N segments. Assemble with transitions and optional background music. Face-reframe to vertical for shorts. Use existing shorts pipeline infrastructure.
- **References**: Medal.tv stream highlights, Opus Clip for streams, Eklipse

---

## 32. Performance & Rendering Optimization

### 32.1 Hardware-Accelerated Encoding (P0 / M)
Auto-detect available GPU encoders (NVIDIA NVENC, Intel QSV, AMD AMF, Apple VideoToolbox) and route exports to the fastest available hardware encoder. Falls back to software encoding when unavailable. Support H.264, H.265/HEVC, and AV1 hardware encoding.
- **Why**: GPU encoding is 5-20x faster than software encoding. OpenCut currently uses software encoding for all exports. This is the single biggest performance win for export times.
- **Implementation**: Probe available encoders via `ffmpeg -encoders | grep nvenc/qsv/amf/videotoolbox`. Auto-select: NVENC > QSV > AMF > software. Map to FFmpeg codec options: `-c:v h264_nvenc`, `-c:v hevc_qsv`, etc. Add quality preset selection (speed/balanced/quality). Expose in export settings with auto-detect default.
- **References**: FFmpeg hardware encoding, HandBrake hardware presets, DaVinci Resolve GPU encoding

### 32.2 Smart Render / Partial Re-Encode (P2 / L)
Re-encode only the timeline segments that changed since the last render. Copy untouched GOP-aligned sections bitstream-exact. Reduces re-export time from minutes to seconds for small changes.
- **Why**: Changing one caption or adjusting one color grade shouldn't require re-encoding the entire video. Smart render is 10-100x faster for iterative exports.
- **Implementation**: Track which segments were modified since last export (via job history + file modification times). Use FFmpeg segment muxer to export only changed segments. Concatenate unchanged segments (stream copy) with re-encoded segments. Requires careful GOP alignment at cut points.
- **References**: DaVinci Resolve smart render, Vegas Pro smart render, TMPGEnc smart render

### 32.3 Background Render Queue with Priority (P1 / M)
Process multiple export jobs in the background while the user continues editing. Jobs have configurable priority (urgent exports jump the queue). Pause, resume, reorder, and cancel individual render jobs. Desktop notification on completion.
- **Why**: Export blocking is the biggest workflow interruption. Background rendering with priority lets users stay productive. Extends existing job system with dedicated export queue UI.
- **Implementation**: Extend existing batch_executor with a dedicated render queue. Lower process priority (`nice`/priority class) for background renders. Panel UI: render queue card showing all pending/active/completed exports with progress bars. System tray notification via desktop-notifier or win10toast.
- **References**: Adobe Media Encoder queue, DaVinci Resolve render queue

### 32.4 GPU Memory Management Dashboard (P2 / M)
Display real-time VRAM allocation: which AI models are loaded, how much memory each uses, total available, and controls to manually unload idle models. Visual bar chart of VRAM usage by model.
- **Why**: GPU OOM is the most common AI feature failure. Visibility into what's consuming VRAM lets users make informed decisions (unload Whisper before running Real-ESRGAN).
- **Implementation**: Track loaded models in a registry (extend existing gpu.py). Query `torch.cuda.memory_allocated()` per model context. Panel UI: bar chart showing per-model VRAM + available headroom. "Unload" button per model. Auto-unload recommendation when a new job would exceed available VRAM.
- **References**: NVIDIA nvidia-smi, Stable Diffusion WebUI VRAM display, ComfyUI model manager

### 32.5 Render Cache with Dependency Tracking (P3 / L)
Cache rendered effects, previews, and intermediate results. Auto-invalidate only the specific cache segments affected by upstream parameter changes. Re-render only what changed.
- **Why**: Repeatedly processing the same clip with minor parameter tweaks re-runs the entire pipeline. Caching intermediate results (transcription, scene detection, stem separation) avoids redundant work.
- **Implementation**: Content-addressable cache at `~/.opencut/cache/`. Key: hash(input_file + operation + parameters). Store intermediate results (transcripts, scene lists, separated stems). On parameter change: only re-run the changed step and downstream. TTL-based eviction for disk management.
- **References**: Make/Bazel dependency tracking, DaVinci Resolve render cache

---

## 33. Education & E-Learning

### 33.1 Slide Change Detection (P2 / M)
Detect slide transitions in screen recordings of presentations. Auto-generate chapter markers at each slide change, extract slide images as thumbnails, and optionally split the recording into per-slide segments.
- **Why**: Massive e-learning and corporate training content is recorded presentations. Auto-detecting slides enables: table of contents, per-slide navigation, and slide-specific annotations.
- **Implementation**: Frame differencing (existing scene detection logic) tuned for high-similarity thresholds (slides change entirely, not gradually). Extract slide frame as image. Generate chapter markers with OCR'd slide title (using 29.2 OCR). Split option: segment recording at slide boundaries.
- **References**: Panopto slide detection, Kaltura lecture capture, Microsoft Stream

### 33.2 Picture-in-Picture Lecture Processing (P2 / M)
Process picture-in-picture lecture recordings: detect and separate the speaker camera feed from the screen share content. Enable independent processing: crop/zoom on the speaker, or extract the screen content at full resolution.
- **Why**: Most lecture recordings combine a small speaker camera with a large screen share. Processing them independently (face tracking on speaker, OCR on screen) unlocks better results.
- **Implementation**: Detect PiP regions via contour analysis (small rectangle of camera feed in corner). Extract both regions. Process independently: speaker → face_reframe for auto-tracking zoom. Screen → slide detection + OCR. Optionally reconstruct as side-by-side layout.
- **References**: Zoom recording processing, OBS dual-source recording

### 33.3 Auto-Quiz Overlay Generator (P3 / M)
Generate quiz/comprehension check overlays at key points in educational videos. AI analyzes transcript to formulate multiple-choice questions, inserts them as pause-screen overlays at appropriate intervals.
- **Why**: Interactive quizzes increase engagement and retention in e-learning. Currently requires specialized authoring tools. LLM can generate relevant questions from transcript.
- **Implementation**: Chunk transcript by topic (existing chapter_gen logic). LLM generates 1-2 quiz questions per chunk (existing llm.py). Render question + multiple-choice options as full-screen overlay card (Pillow). Insert at chapter boundaries. For interactive: export as HTML5 video with clickable overlays.
- **References**: Edpuzzle, H5P interactive video, Articulate quiz overlays

---

## 34. Text & Graphics Overlays

### 34.1 Scrolling Credits Generator (P1 / M)
Generate professional end credits from a structured data source (text file or CSV with role/name pairs). Configurable: scroll speed, font, alignment (centered/two-column), background, music bed timing. Auto-calculate duration from content length.
- **Why**: End credits are needed for every production but tedious to build manually. A generator from structured data produces broadcast-quality credits in seconds.
- **Implementation**: Parse credit data (role: name pairs, section headers). Render as scrolling text image sequence via Pillow/Skia: large height image, scroll via FFmpeg `scroll` filter. Two-column layout: role left-aligned, name right-aligned. Auto-duration: content_height / scroll_speed. Export as overlay or standalone.
- **References**: Premiere Pro credits template, After Effects credit roll, DaVinci Resolve scroll

### 34.2 End Screen / CTA Template (P2 / M)
Generate YouTube-style end screens with subscribe button animation, video recommendation cards (using thumbnail from other videos), and custom call-to-action overlays. Templates match platform specifications (YouTube end screen safe zones).
- **Why**: Every YouTube video needs an end screen. Creating them manually each time is repetitive. Templates with variable slots (video thumbnail, channel icon, CTA text) automate this.
- **Implementation**: Template SVG/JSON with placeholder slots. User fills: CTA text, thumbnail image, channel icon. Render animated end screen (subscribe button pulse, card slide-in) via Pillow frame sequence. Duration: 5-20 seconds (YouTube spec). Export as overlay.
- **References**: YouTube end screen spec, Canva end screen templates, TubeBuddy

### 34.3 News Ticker / Crawl Overlay (P3 / M)
Generate broadcast-style scrolling text tickers (breaking news crawl, sports scores, stock prices) from a text file or RSS feed. Configurable: speed, font, background color, direction, position.
- **Why**: Broadcast, event, and news-style content needs tickers. Simple FFmpeg drawtext with scroll but needs a nice UI for configuration.
- **Implementation**: FFmpeg `drawtext` with `x='mod(n*speed, text_w+w)'` for horizontal scroll. Read content from text file, JSON, or RSS URL. Style: background bar color, text color, font, speed. Render as transparent overlay. Loop: repeat content until video duration is filled.
- **References**: OBS news ticker, vMix lower thirds, broadcast graphics systems

### 34.4 Countdown / Timer Overlay (P3 / S)
Generate countdown timers (event starts in 3:00:00), elapsed timers (workout timer), and stopwatch overlays. Configurable: format (HH:MM:SS, MM:SS, SS), font, size, position, color, background.
- **Why**: Event streams, fitness videos, cooking videos, and time-lapse content need visible timers. Simple but commonly requested.
- **Implementation**: FFmpeg `drawtext` with `text='%{eif\\:DURATION-t\\:d}'` for countdown or `text='%{pts\\:hms}'` for elapsed. Configurable start time, format, and style. Render as transparent overlay.
- **References**: OBS countdown timer, StreamElements timer

---

## 35. Forensic & Legal Video

### 35.1 Selective Redaction Tool (P1 / M)
Blur, pixelate, or black-out specific regions to protect privacy: faces, license plates, addresses, documents, screen content. Redaction follows tracked objects through the video. Supports irreversible (destructive) redaction for legal compliance.
- **Why**: Privacy laws (GDPR, CCPA), legal proceedings, journalism, and body-cam footage all require redaction. OpenCut has face blur but no general-purpose redaction with tracking.
- **Implementation**: SAM2 (existing) or manual region selection → track through video → apply blur/pixelate/black to tracked region per frame. Destructive mode: re-encode so original pixels are permanently removed (not just overlaid). Add redaction log for audit trail. Support: face, license plate (via YOLO/ALPR), custom region.
- **References**: Adobe Premiere redaction, Axon Evidence, YouTube face blur

### 35.2 Metadata Preservation & Stripping Controls (P2 / S)
Control which metadata is preserved or stripped during export: EXIF, XMP, GPS location, camera serial number, creation date, author. Strip all for privacy, preserve all for legal, or selective.
- **Why**: GPS data in video reveals filming locations. Camera serials can identify the device. Legal contexts need full metadata; privacy contexts need it removed.
- **Implementation**: FFmpeg `-map_metadata -1` strips all metadata. For selective: use `-metadata` flags to preserve specific fields. UI: metadata viewer showing current file's metadata, with per-field keep/strip toggles. Presets: "Strip All", "Preserve All", "Legal (keep timestamps, strip GPS)".
- **References**: ExifTool, FFmpeg metadata handling, GDPR data minimization

### 35.3 Evidence Chain-of-Custody Log (P3 / M)
Generate a chain-of-custody document for video evidence: original file hash (SHA-256), every processing step with parameters, output file hash, timestamps, operator identity. Suitable for legal proceedings.
- **Why**: Court-admissible video evidence requires documented chain of custody. Every modification must be recorded and verifiable. Extends provenance chain (27.4) for legal contexts.
- **Implementation**: On designated "evidence" projects: hash input file, log every operation with full parameters, hash output file, record operator (configurable name). Generate signed PDF report with all entries. Include integrity verification instructions for opposing counsel.
- **References**: Digital evidence guidelines (SWGDE), forensic video analysis standards

---

## 36. Social Media Optimization

### 36.1 Platform Safe Zone Overlay (P1 / S)
Display platform-specific safe zones on the preview: YouTube end screen click areas, TikTok UI overlay zones (username, description, buttons), Instagram caption area, Twitter/X aspect crop zones. Ensure important content isn't covered by platform UI.
- **Why**: Every platform overlays UI elements on the video. Content placed in those zones is invisible. Safe zone guides prevent this common mistake.
- **Implementation**: Define safe zone templates per platform as SVG overlays (TikTok: bottom 20% and right 15% blocked, YouTube: bottom 20% for end screen). Display as semi-transparent overlay on preview frame. Toggle per-platform. Update templates as platforms change.
- **References**: TikTok safe zone guides, YouTube end screen spec, Instagram design templates

### 36.2 AI Engagement Prediction (P2 / M)
Analyze a video before publishing and predict engagement metrics: estimated retention curve, hook strength score (first 3 seconds), virality potential, and content category classification. Suggest improvements.
- **Why**: Creators want to know if their content will perform before posting. LLM analysis of transcript + visual content can provide actionable feedback.
- **Implementation**: Analyze: (1) Hook strength: first 3 seconds energy/novelty via LLM. (2) Retention prediction: engagement curve from audio energy + topic changes. (3) Category: LLM classifies content type. (4) Suggestions: "Add a stronger hook", "Cut 30 seconds from the middle", "Thumbnail needs more contrast". Use existing llm.py.
- **References**: Opus Clip Virality Score, TubeBuddy SEO score, Spotter analytics

### 36.3 Auto-Hashtag & Caption Generator (P2 / S)
Generate platform-optimized post captions, hashtags, and descriptions from video content analysis. Different output for each platform (TikTok: trend hashtags, YouTube: keyword-rich description, Instagram: emoji-heavy caption).
- **Why**: Writing post copy for each platform is repetitive. LLM generates platform-appropriate text from transcript. Pairs with multi-platform publish (5.1).
- **Implementation**: Transcribe → LLM generates per-platform: (1) TikTok: short caption + 5-8 trending hashtags. (2) YouTube: keyword-rich description + chapters + tags. (3) Instagram: engaging caption + 30 hashtags. (4) Twitter/X: concise hook + 2-3 hashtags. Use existing llm.py with platform-specific prompts.
- **References**: Opus Clip auto-captions, TubeBuddy tag suggestions, Later caption generator

### 36.4 Vertical-First Intelligent Reframe (P1 / M)
Intelligently reframe 16:9 content to 9:16 using multi-subject tracking, dynamic cropping that follows the action, and split-screen mode for wide shots. Goes beyond simple center-crop to create genuinely good vertical content.
- **Why**: OpenCut has face-tracked reframe but it only handles single-face tracking. Multi-subject scenes, wide shots, and action sequences need smarter reframing that tracks the overall action, not just one face.
- **Implementation**: Multi-signal tracking: face positions (existing), motion center-of-mass, saliency map (content-aware crop point). Smooth transitions between tracked subjects. Split-screen mode: when subjects are far apart, show both in vertical split. Apply via FFmpeg crop with per-frame coordinates.
- **References**: CapCut auto-reframe, OpusClip smart reframe, AutoFlip (Google)

---

## 37. Quality of Life & UX Polish

### 37.1 Guided Feature Walkthroughs (P2 / M)
In-context tutorial overlays that walk new users through each feature step-by-step: "Step 1: Select a clip. Step 2: Adjust the threshold slider. Step 3: Click Detect." Triggered on first use of each feature or via help button.
- **Why**: OpenCut has 50+ sub-tabs and hundreds of options. The first-run wizard covers basics but individual features need guidance too. Reduces support burden.
- **Implementation**: Define walkthrough steps per feature as JSON: `{ selector, tooltip_text, highlight_region }`. Render as overlay spotlights with instruction text. Track completed walkthroughs in localStorage. Add "Show Tutorial" button per card. Dismiss permanently per feature.
- **References**: Intercom product tours, Shepherd.js, Premiere Pro "Learn" panel

### 37.2 Session State Restore (P2 / S)
Remember panel state across sessions: which tab was active, which sub-tabs were open, scroll position, last used clip, form field values, and collapse states. On relaunch, restore to exactly where the user left off.
- **Why**: Closing and reopening the panel loses all context. Users must re-navigate to their working tab every time. Simple localStorage persistence solves this.
- **Implementation**: On panel close: serialize active tab, sub-tab, scroll positions, and form values to localStorage. On panel open: restore from localStorage. Debounce serialization to avoid performance impact. Add "Reset Panel" option to clear saved state.
- **References**: VS Code workspace restore, browser session restore

### 37.3 Voice Commands (P3 / L)
Hands-free operation via voice commands: "Remove silence", "Add captions", "Normalize audio", "Export for YouTube." Uses browser Web Speech API or Whisper for recognition. Useful when hands are on physical editing controllers.
- **Why**: Niche but valuable for multi-tasking editors and accessibility. Web Speech API makes it zero-cost to implement in the CEP panel's Chromium environment.
- **Implementation**: CEP: `webkitSpeechRecognition` API (available in CEP Chromium). Map recognized phrases to command palette actions (existing). Activation: push-to-talk button or wake word. Display recognized text in command palette. Confidence threshold to avoid false triggers.
- **References**: Siri/Alexa voice control, Premiere Pro voice typing (beta), Dragon NaturallySpeaking

### 37.4 Feature Usage Analytics Dashboard (P3 / S)
Track which features are used most, processing times per operation, and error rates. Display as a dashboard in Settings. Helps the user understand their workflow and helps development prioritize improvements.
- **Why**: Understanding usage patterns optimizes the product and the user's workflow. "You spend 40% of processing time on captions — consider the fast Whisper model."
- **Implementation**: Log each API call with: endpoint, duration, success/failure, timestamp. Store in SQLite (extend job_store). Dashboard card in Settings: top 10 features by usage, average processing time, error rate over last 30 days. All local, no telemetry.
- **References**: VS Code telemetry viewer, Premiere Pro performance monitor

### 37.5 Offline Documentation Viewer (P3 / S)
Bundled documentation accessible from within the panel: feature descriptions, parameter explanations, keyboard shortcut reference, troubleshooting guide, and FAQ. No internet required.
- **Why**: Users need help but may not have internet or want to leave the panel. Bundled docs with search make the tool self-documenting.
- **Implementation**: Generate HTML documentation from existing README, DEVELOPMENT.md, and inline parameter descriptions. Bundle as static HTML in extension. Panel: help button per card opens relevant doc section. Global search across all docs.
- **References**: Premiere Pro Help panel, VS Code documentation, Blender manual

---

## 38. Multi-Format Delivery

### 38.1 GIF Export with Optimization (P2 / S)
Export video clips or segments as optimized GIFs: configurable resolution, frame rate, color palette, dithering, and loop count. Auto-optimize file size with quality threshold.
- **Why**: GIFs are still the primary animated image format for social media, messaging, and documentation. FFmpeg handles GIF generation but the optimization pipeline (palette generation, dithering) needs configuration.
- **Implementation**: FFmpeg two-pass: `palettegen` for optimal 256-color palette, then `paletteuse` with dithering. Add parameters: max width, fps (10-15 for GIF), max file size target, loop count. Preview estimated file size. UI: GIF export card in Export tab.
- **References**: FFmpeg GIF optimization, Giphy, ScreenToGif

### 38.2 Animated WebP / APNG Export (P2 / S)
Export as animated WebP (smaller than GIF, supports transparency, 24-bit color) or APNG (PNG-quality animation). Modern alternatives to GIF with better quality and compression.
- **Why**: WebP is supported by all modern browsers and messaging apps. APNG is supported by all browsers. Both are superior to GIF in quality and file size.
- **Implementation**: FFmpeg `-c:v libwebp_anim` for WebP, `-f apng` for APNG. Same optimization pipeline as GIF but with wider color gamut. Add alongside GIF export in the export card.
- **References**: FFmpeg WebP encoding, Google WebP spec, APNG spec

### 38.3 HLS / DASH Streaming Package (P3 / M)
Export as adaptive bitrate streaming packages: HLS (Apple) or DASH (universal). Generate multi-quality renditions (240p, 480p, 720p, 1080p) with manifest files. Ready for direct upload to CDN or self-hosting.
- **Why**: Self-hosted video playback requires adaptive streaming. HLS/DASH packages are the standard for web video players. Eliminates the need for a separate transcoding service.
- **Implementation**: FFmpeg `hls` muxer with multi-quality encoding: generate 4 quality variants in one pass via `split` + multiple output streams. Generate `.m3u8` manifest (HLS) or `.mpd` manifest (DASH). Package as a zip for upload. Add HLS/DASH preset in export options.
- **References**: FFmpeg HLS muxer, Cloudflare Stream, Video.js player

### 38.4 Podcast RSS Feed Generator (P3 / S)
Generate a podcast-standard RSS feed from processed audio episodes. Include: title, description, chapter markers, artwork, duration, file size. Produces a ready-to-submit feed for Apple Podcasts, Spotify, etc.
- **Why**: Podcast creators using OpenCut for audio processing need to publish. Generating the RSS feed from processed audio closes the production-to-publication loop.
- **Implementation**: Standard RSS 2.0 XML with iTunes podcast extensions. Pull metadata from user input (title, description) + auto-generated (duration, file size, chapters from existing chapter_gen). Validate against Apple Podcasts spec. Export as .xml file.
- **References**: Apple Podcasts RSS spec, Spotify for Podcasters, Podcast Index

---

## 39. Hardware Integration & Control Surfaces

### 39.1 Elgato Stream Deck Integration (P1 / M)
Map OpenCut operations to Stream Deck buttons: one-press silence removal, caption generation, export, workflow execution. Dynamic button labels showing operation status. Multi-action buttons for chained workflows. Folder pages for organized OpenCut command groups.
- **Why**: Stream Deck is the de facto control surface for content creators — over 2 million units sold. Deep integration makes OpenCut's 50+ operations accessible without navigating the panel. AutoCut and other competing tools already offer Stream Deck plugins.
- **Implementation**: Stream Deck SDK plugin (Node.js). Each button maps to an OpenCut API endpoint via HTTP POST. Dynamic labels: poll job status and update button text/icon in real-time. Multi-action: chain multiple API calls per button press. Ship preconfigured profiles for common workflows (Interview, Music Video, Social Content, Podcast).
- **References**: Stream Deck SDK, AutoCut Stream Deck plugin, Premiere Pro Stream Deck plugin

### 39.2 MIDI Controller Mapping (P2 / M)
Map MIDI knobs, faders, and buttons to OpenCut parameters: slider values (threshold, intensity, LUFS target), transport controls, and operation triggers. Any MIDI controller becomes an OpenCut control surface for hands-on adjustment.
- **Why**: Many editors already have MIDI controllers (Behringer X-Touch, Korg nanoKONTROL) for audio work. Mapping physical controls to OpenCut parameters enables eyes-on-screen adjustment of thresholds and effects without hunting for sliders in the panel.
- **Implementation**: Python `mido` library for MIDI input. Map: CC messages → parameter changes (POST to API with new value), note-on → operation triggers. UI: MIDI learn mode (move a physical control, click a panel parameter to bind). Store mappings as JSON config in `~/.opencut/midi_map.json`. Background MIDI listener thread.
- **References**: DaVinci Resolve MIDI mapping, Premiere Pro control surface support, Behringer X-Touch

### 39.3 Shuttle / Jog Wheel Support (P3 / M)
Support hardware jog/shuttle wheels (Contour ShuttlePRO, Loupedeck) for frame-accurate navigation and scrubbing through video preview. Map wheel rotation to timeline seek, buttons to OpenCut commands.
- **Why**: Jog wheels are standard equipment in professional editing suites. Precise scrubbing for finding exact cut points and reviewing effects frame-by-frame is faster with physical controls than with mouse dragging.
- **Implementation**: HID device input via Python `hid` library or platform-specific driver APIs. Map: jog rotation → seek forward/backward by frames, shuttle position → playback speed multiplier, buttons → configurable OpenCut actions. Send seek commands to Premiere via ExtendScript `app.project.activeSequence.setPlayerPosition()`.
- **References**: Contour ShuttlePRO SDK, Loupedeck plugin API, Premiere Pro control surface

### 39.4 Touch Screen & Pen Tablet Optimization (P3 / M)
Optimize panel UI for touch and pen input: larger tap targets, gesture support (pinch-to-zoom on waveform, swipe between tabs), pressure-sensitive drawing for manual mask creation (redaction regions, object selection), and stylus-friendly controls.
- **Why**: Surface Pro, iPad (via Sidecar), and Wacom tablets are common in editing setups. Touch-optimized UI removes the mouse bottleneck for spatial operations like mask drawing, region selection, and waveform navigation.
- **Implementation**: CSS touch target sizing (min 44px per WCAG). Add pointer event handlers alongside mouse events for unified input. Pinch-zoom via touch gesture detection on canvas elements. Pen pressure API (`pointerEvent.pressure`) for variable-width mask drawing. Test with Chrome DevTools touch simulation and real Wacom tablet.
- **References**: Apple Human Interface Guidelines (touch), WCAG touch target sizes, Wacom plugin SDK

---

## 40. Podcast Visual Production

### 40.1 Audiogram Generator for Social Media (P1 / M)
Generate animated audiogram videos — waveform/bars animation synced to audio with styled captions overlaid — for promoting podcast episodes on social media. Configurable: waveform style (bars, wave, circular), brand colors, background image, episode artwork, and caption style.
- **Why**: Audiograms are the #1 format for promoting podcast episodes on Instagram, Twitter/X, and LinkedIn. Headliner and Wavve charge $15-25/month for this. OpenCut has all the pieces (waveform generation, captions, audio processing) but doesn't assemble them into this specific highly-demanded format.
- **Implementation**: Input: audio clip + episode art + optional captions. Generate waveform animation (FFmpeg `showwaves`/`showspectrum` or custom Pillow renderer driven by audio amplitude data). Composite: background image/color + animated waveform + episode art + animated caption overlay. Export as square (1:1 for Instagram) or vertical (9:16 for Stories/Reels). Ship 5+ style presets matching popular audiogram aesthetics.
- **References**: Headliner, Wavve, Descript audiograms, Spotify Canvas

### 40.2 Multi-Speaker Layout Engine (P2 / L)
Automatically detect multiple speakers in a video podcast recording and render them in configurable layouts: side-by-side, grid (2x2, 3x2), active-speaker spotlight (enlarge the current speaker, shrink others), or picture-in-picture. Driven by existing diarization data for dynamic switching.
- **Why**: Video podcasts recorded on Riverside, Squadcast, or Zoom produce separate tracks per speaker. Assembling them into a dynamic, professional layout is manual and tedious. An automated layout engine driven by speaker activity produces polished results in minutes.
- **Implementation**: Input: multiple video tracks (one per speaker) + diarization data (existing). Layout templates define grid positions per speaker count (2, 3, 4, 5, 6). Active-speaker mode: use diarization timestamps to determine who's speaking → enlarge that speaker's window with smooth animation. Render via FFmpeg complex filter graph (`overlay` + `scale` + `pad` per speaker per frame). UI: layout template selector with live preview.
- **References**: Riverside.fm layouts, Squadcast recording, StreamYard layout engine

### 40.3 Video Podcast to Audio-Only Extraction (P1 / S)
One-click extraction of the audio track from a video podcast with automatic podcast-standard processing applied: normalize to -16 LUFS mono, apply noise reduction, strip video, embed chapter markers from existing chapter detection, and export as MP3/AAC with ID3 metadata and episode artwork.
- **Why**: Most video podcasts are published as both video (YouTube) and audio-only (Spotify/Apple Podcasts). Extracting and optimizing the audio is a common manual step that every podcaster repeats weekly. Combining extraction with podcast-standard processing saves a workflow.
- **Implementation**: FFmpeg `-vn` to strip video + existing audio pipeline: loudness normalization to -16 LUFS mono (existing), optional noise reduction (existing), chapter markers from chapter_gen (existing). Export as MP3 with ID3 tags (title, artist, album art from episode image) via `mutagen` library or FFmpeg metadata. Add as one-click preset in export options.
- **References**: Spotify podcast specs, Apple Podcasts audio requirements, Descript publish flow

### 40.4 AI Show Notes & Transcript Summary (P2 / M)
Generate formatted podcast show notes from the full transcript: episode summary, key topics with timestamps, guest bios extracted from introductions, links/resources mentioned, notable pull quotes, and chapter markers. Publish-ready markdown or HTML output.
- **Why**: Writing show notes is the most tedious podcast post-production task after editing itself. LLM analysis of the transcript can extract all key information and format it for publication in seconds. Every podcast hosting platform (Buzzsprout, Transistor, Podbean) requires show notes.
- **Implementation**: Transcribe (existing) → LLM generates structured show notes (existing llm.py): 2-3 sentence summary, bullet-point key topics with timestamps, notable quotes with attribution, mentioned resources/links (detected via NER or LLM extraction), auto-generated chapter markers. Output as markdown, HTML, or plain text. UI: editable preview before export with per-section regeneration.
- **References**: Descript show notes, Castmagic, Podium AI, Deciphr AI

---

## 41. Drone & Aerial Post-Production

### 41.1 Telemetry Data Overlay from SRT/CSV (P1 / M)
Parse DJI SRT subtitle files or telemetry CSV logs and render flight data as a configurable on-screen overlay: altitude, speed, GPS coordinates, distance from home point, battery level, gimbal angle, and camera settings (ISO/shutter/aperture). Customizable position, font, gauge style, and field selection.
- **Why**: DJI drones automatically record telemetry as SRT sidecar files with every recording. Overlaying this data is a common requirement for professional drone work, real estate flyovers, adventure content, and regulatory documentation. Currently requires third-party tools like the discontinued DashWare or manual FFmpeg filter chains.
- **Implementation**: Parse DJI SRT format (timestamp + `[latitude: X] [longitude: Y] [altitude: Z] [speed: V]` pattern). Render per-frame overlay via FFmpeg `drawtext` with per-frame variable substitution from parsed data, or Pillow compositing for styled gauge widgets. UI: data field selector (checkboxes per telemetry field), position presets (corners, center-bottom), font/size/color/background config. Also support generic CSV telemetry for non-DJI sources (GoPro GPMF, Insta360).
- **References**: DashWare (Microsoft, discontinued), DJI SRT format spec, Race Render, Garmin VIRB Edit

### 41.2 Flight Path Map Overlay (P2 / M)
Extract GPS coordinates from telemetry data and render an animated map overlay showing the drone's flight path as a growing line, current position indicator, and optional POI markers. Map tiles from OpenStreetMap or satellite imagery cached locally.
- **Why**: Flight path visualization adds geographic context to aerial footage — viewers see where the drone is relative to landmarks, roads, and terrain. Popular for travel vlogs, real estate tours, and adventure content. Also useful for regulatory flight documentation.
- **Implementation**: Extract GPS waypoints from DJI SRT or GPX file. Render map using Python `staticmap` library with OpenStreetMap tiles (free, cached locally). Animate flight path: draw progressively longer polyline synced to video timestamp. Current position: animated marker dot. Composite as corner PiP overlay via FFmpeg. Support satellite tile sources for more visual maps.
- **References**: Litchi flight planning map, AirData UAV track visualization, Google Earth Studio

### 41.3 Automated Aerial Hyperlapse (P2 / M)
Create stabilized hyperlapses from drone footage: intelligent frame sampling at regular GPS distance intervals (not just time intervals), aggressive perspective-aware stabilization, and smooth speed ramping. Compensates for the variable flight speed and wind-induced wobble of manual drone flight.
- **Why**: Drone hyperlapses shot in manual flight mode are jittery and inconsistent. GPS-interval sampling ensures uniform spatial pacing regardless of wind-induced speed variations. Combined with aggressive stabilization, this produces results matching DJI's built-in Hyperlapse mode from any manually-flown footage.
- **Implementation**: Parse GPS from DJI SRT → compute cumulative GPS distance → sample frames at equal distance intervals (e.g., every 5 meters). Apply multi-pass vidstab with high smoothing (existing). Optional RIFE interpolation (existing) to fill dropped frames for smoother motion. Speed ramp via existing speed_ramp.py with ease-in/out curves at start/end. Edge fill via inpainting (ties to 17.4 stabilization edge fill).
- **References**: DJI Hyperlapse mode, Microsoft Hyperlapse (discontinued), LRTimelapse

### 41.4 ND Filter Simulation for Drone Footage (P3 / S)
Simulate the motion blur effect of an ND filter on drone footage shot at too-fast a shutter speed. Adds natural per-frame directional motion blur based on inter-frame motion vectors, softening the staccato look of high-shutter-speed aerial footage.
- **Why**: Proper drone cinematography uses ND filters for a 180-degree shutter angle (1/50s at 24fps). Many drone operators forget or don't carry ND filters, resulting in hyper-sharp, stroboscopic footage. Simulated motion blur fixes this in post, rescuing otherwise unusable footage.
- **Implementation**: Compute optical flow between consecutive frames (OpenCV `calcOpticalFlowFarneback`). Apply directional blur per-pixel along flow vectors with intensity proportional to desired shutter angle simulation. FFmpeg `minterpolate` with `mi_mode=blend` achieves a simple version; custom per-pixel blur via NumPy/OpenCV is more accurate but slower.
- **References**: RSMB (RE:Vision Effects), ReelSteady motion blur, FFmpeg minterpolate blend

---

## 42. Timelapse & Image Sequence

### 42.1 Image Sequence Import & Assembly (P1 / M)
Import folders of numbered image files (TIFF, EXR, DPX, JPEG, PNG, RAW) as video clips with configurable frame rate. Essential for timelapse photography, VFX plates received from compositors, stop-motion animation, and scientific imaging workflows.
- **Why**: Image sequences are the standard interchange format for VFX pipelines, the output format for timelapse photography, and the native format for stop-motion animation. FFmpeg handles this natively but the command is non-obvious. A proper UI with preview makes it accessible.
- **Implementation**: Detect image sequences in folder (numbered filenames with common prefix, sorted numerically). FFmpeg `image2` demuxer: `-framerate 24 -i "img_%04d.tif"`. UI: folder picker → auto-detect sequence pattern → frame rate selector → start/end frame trim → preview first/last/middle frames. Support all FFmpeg-compatible image formats including EXR (float HDR). Output as ProRes/H.264/H.265 with configurable quality.
- **References**: DaVinci Resolve image sequence import, After Effects image sequence, FFmpeg image2 demuxer

### 42.2 Timelapse Deflicker (P1 / M)
Remove brightness flickering from timelapse sequences caused by mechanical aperture micro-variations (even in full manual mode), auto-exposure hunting, or natural light changes between frames. Smooths the luminance curve across the entire sequence for flicker-free playback.
- **Why**: Flickering is the #1 quality issue in timelapse photography and ruins otherwise stunning sequences. LRTimelapse charges $150 for deflicker alone. A free, integrated, high-quality deflicker makes OpenCut the go-to timelapse processing tool.
- **Implementation**: Analyze per-frame average luminance (and optionally per-region luminance for gradient flicker). Apply temporal smoothing filter: rolling average or polynomial curve fit to the luminance timeline. Adjust each frame's exposure to match the smoothed curve via FFmpeg `eq` filter with per-frame brightness parameter. FFmpeg `deflicker` filter handles simple cases. Custom NumPy implementation for full control. UI: before/after preview, smoothing window size slider, luminance curve visualization.
- **References**: LRTimelapse, FFmpeg deflicker filter, GBTimelapse, Sequence (timelapse app)

### 42.3 Holy Grail (Day-to-Night) Timelapse Processing (P2 / L)
Process "holy grail" timelapses that transition from day to night (or vice versa): automatically ramp exposure compensation, white balance, and brightness across the sequence to create a smooth transition without flicker despite massive lighting changes spanning 8+ exposure stops.
- **Why**: Holy grail timelapses are the most dramatic and technically challenging timelapse format. The exposure must ramp smoothly across 8+ stops as daylight fades. LRTimelapse is the only mainstream tool that handles this. An automated version would be a standout niche feature.
- **Implementation**: Analyze per-frame exposure metadata (EXIF) or compute from pixel histogram analysis. Fit smooth exposure ramp curve (polynomial or spline through keyframes). Apply per-frame correction: exposure compensation + white balance shift (daylight 5500K → tungsten 3200K → blue hour 8000K). Combine with deflicker (42.2) for final luminance smoothing. Advanced: allow user-placed keyframes for manual exposure/WB override at key transition points.
- **References**: LRTimelapse Holy Grail wizard, GBTimelapse, Lightroom timelapse processing

### 42.4 Star Trail Compositing (P3 / M)
Composite a folder of long-exposure night sky images into star trail photographs or animated star trail videos: lighten blend mode stacking (keep brightest pixels per position), gap filling between frames, airplane/satellite streak removal, and animated progressive buildup video.
- **Why**: Star trail photography is popular among landscape photographers but compositing hundreds of images is tedious. StarStaX is free but limited. A full pipeline including gap-filling, streak removal, and animation export is more valuable.
- **Implementation**: Lighten-mode stack: per-pixel maximum across all frames (NumPy, very fast for in-memory processing). Gap filling: interpolate missing trail segments between frames with synthetic arcs. Airplane removal: detect straight bright lines appearing in only 1-2 consecutive frames (outlier detection). Video animation: progressive stack (frame 1, frames 1-2, frames 1-3, ...) exported as video showing trails growing. FFmpeg `tblend=lighten` for simple streaming cases.
- **References**: StarStaX, Sequator, Advanced Stacker PLUS

### 42.5 Construction / Long-Duration Timelapse Builder (P3 / M)
Assemble long-duration timelapses from photos taken over weeks or months (construction cameras, plant growth experiments, weather stations): normalize framing across camera position shifts, auto-crop to common visible region, apply deflicker across extreme exposure variations between sessions, and handle missing frames/days gracefully.
- **Why**: Construction companies, nature photographers, and scientific researchers need to assemble timelapses from thousands of images captured over extended periods. Camera position drifts between maintenance sessions, lighting varies wildly across days, and gaps exist from equipment downtime. Specialized processing handles all of these challenges.
- **Implementation**: Feature-matching alignment (OpenCV ORB/SIFT) to compensate for camera shifts between sessions → compute homography per session → warp to common frame. Auto-crop to largest common visible region. Deflicker (42.2) with very wide temporal window. Missing frame handling: repeat previous frame or RIFE interpolation between flanking frames. Date/time overlay from EXIF. Export as video with configurable playback speed (1 frame per hour, per day, etc.).
- **References**: Brinno timelapse cameras, Afidus construction timelapse, Chronolapse

---

## 43. Color Management Pipeline

### 43.1 ACES Color Pipeline (P1 / L)
Implement the Academy Color Encoding System (ACES) workflow: Input Device Transforms (IDTs) per camera, processing in ACEScg linear color space, and Output Display Transforms (ODTs) for each delivery target (Rec.709, sRGB, DCI-P3, HDR10). The industry standard for consistent, future-proof color management across mixed-camera projects.
- **Why**: ACES is required by major studios (Netflix, Disney, most VFX houses) and ensures color consistency when mixing footage from different cameras with different color science. DaVinci Resolve has full ACES support. Adding this to OpenCut positions it for professional workflows and proper VFX interchange.
- **Implementation**: FFmpeg `lut3d` chain with ACES LUTs: IDT (camera-specific LOG→ACES) → processing in ACES color space → ODT (ACES→display target). Use the free ACES OCIO config for transform definitions. OpenColorIO (OCIO) Python bindings for advanced transforms. Bundle IDTs for common cameras (Sony S-Log3, Canon C-Log3, Panasonic V-Log, ARRI LogC, RED Log3G10, Blackmagic Film Gen5). UI: source camera selector → auto-apply IDT, output format selector → auto-apply ODT.
- **References**: ACES documentation (oscars.org), OpenColorIO, DaVinci Resolve ACES color management

### 43.2 Wide Gamut Workflow (Rec.2020 / DCI-P3) (P2 / M)
Support wide color gamut processing throughout the entire pipeline: detect Rec.2020 or DCI-P3 source footage from metadata, process in the native gamut without clipping to Rec.709, and deliver in the appropriate gamut for each output target with proper gamut mapping.
- **Why**: Modern cameras (iPhone 15+, Sony a7 series, RED, ARRI) shoot in wide gamut by default. Processing in Rec.709 clips the most vivid captured colors permanently. Proper gamut handling preserves the full color range and only maps to smaller gamuts at the final export stage.
- **Implementation**: Detect source gamut from FFmpeg probe (`color_primaries`, `color_trc`, `color_space` metadata). Process internally in source gamut (don't convert until export stage). On export: gamut map to target space via FFmpeg `zscale` with perceptual intent. Warn user when Rec.2020 source is being delivered to Rec.709 (show which colors will clip). UI: optional gamut scope visualization showing in-gamut vs. out-of-gamut colors.
- **References**: FFmpeg zscale gamut mapping, libplacebo, DaVinci Resolve color management

### 43.3 Display Calibration Verification (P3 / M)
Display reference test patterns (SMPTE color bars, grayscale ramp, gamut boundary colors, skin tone patches) to help users verify their monitor is calibrated correctly before making color decisions. Include a guided verification walkthrough with expected appearance descriptions.
- **Why**: Color grading decisions made on an uncalibrated display are meaningless. Many creators edit on uncalibrated consumer monitors. Even a basic verification check with guidance prevents the worst color mistakes and prompts users toward calibration.
- **Implementation**: Generate test pattern images (Pillow/NumPy): SMPTE RP 219 color bars, 0-100% grayscale ramp in 5% steps, Rec.709 gamut boundary colors, Macbeth ColorChecker reference patches, skin tone reference. Display in panel with guidance text explaining what each pattern should look like. Guided check: "If these two gray patches look different, your display needs calibration." Export patterns as reference images.
- **References**: DisplayCAL, Calman, SMPTE RP 219 test patterns

### 43.4 Color Space Auto-Detection & Conversion (P1 / M)
Automatically detect the color space, transfer function (gamma/log curve), and gamut of imported footage from container metadata and pixel histogram analysis. Auto-convert mismatched clips to the project's working color space during processing. Flag clips with missing or unknown metadata for manual identification.
- **Why**: Mixed-source projects (iPhone + Sony + GoPro + drone + archival) inevitably mix different color spaces. Without detection and conversion, clips look inconsistent from the start. Auto-detection and conversion to a unified working space ensures a consistent baseline before grading.
- **Implementation**: FFmpeg probe extracts: `color_primaries` (bt709/bt2020/unknown), `color_trc` (srgb/smpte2084/arib-std-b67/bt709), `color_space` (bt709/bt2020nc). Map to known profiles. For unknown/absent metadata: analyze pixel distribution (LOG footage has a distinctive compressed histogram shape, HDR has extended range). Convert via FFmpeg `colorspace` filter or `zscale`. Store detected profile in footage_index_db. Batch convert all project clips to working space on ingest.
- **References**: FFmpeg colorspace filter, MediaInfo metadata, DaVinci Resolve color space detect

---

## 44. Timecode & Professional Sync

### 44.1 Timecode Burn-In Overlay (P1 / S)
Burn visible timecode into video as a text overlay — SMPTE timecode (HH:MM:SS:FF), source timecode from camera metadata, or sequence elapsed time. Essential for offline review copies, client approval rounds, and frame-accurate post-production communication.
- **Why**: Every review copy in professional post-production needs visible timecode for precise frame-accurate notes. "At 01:23:45:12, the color is too warm" is standard professional communication. Currently requires manual FFmpeg commands or After Effects precomps.
- **Implementation**: FFmpeg `drawtext` with `text='%{pts\:hms}'` for elapsed time, or parse source timecode from container metadata (`timecode` stream) for original camera TC. Configurable: position (any corner, center-top, center-bottom), font size, background box opacity and color, TC format (HH:MM:SS:FF / HH:MM:SS.mmm / frame count / feet+frames for film), text color. Add as checkbox toggle in export settings: "Burn-in timecode for review."
- **References**: Premiere Pro timecode overlay effect, DaVinci Resolve burn-in, FFmpeg drawtext timecode

### 44.2 LTC / VITC Timecode Extraction (P2 / M)
Extract Longitudinal Time Code (LTC) from audio tracks or Vertical Interval Time Code (VITC) from the video signal in tape-captured footage. Parse the timecode and use it for synchronization, logging, and conforming workflows.
- **Why**: Production audio recorders (Sound Devices MixPre, Zoom F-series) embed LTC on a dedicated audio track for sync with camera footage. Tape-based archival footage encodes VITC in the vertical blanking interval. Extracting this timecode is essential for syncing dual-system audio and conforming archival material.
- **Implementation**: LTC: decode audio track using `ltcdump` (open-source LTC decoder command-line tool) or Python `ltc` library. Extract timecode values per audio frame, interpolate to video frame rate. VITC: analyze first 20 scan lines of each frame for the binary timecode pattern (applicable to interlaced tape captures only). Store extracted timecode as metadata associated with the clip. Use for multi-source sync (44.3).
- **References**: ltcdump (x42.github.io), Sound Devices LTC documentation, SMPTE 12M timecode standard

### 44.3 Timecode-Based Multi-Source Sync (P2 / M)
Synchronize multiple video and audio sources using matching timecodes from jam-synced devices. More precise than audio waveform sync (3.9) when professional timecode is available — guaranteed frame-accurate alignment with zero drift.
- **Why**: Professional multi-camera shoots use jam-synced timecode generators (Tentacle Sync, Deity TC-1) for frame-accurate sync across all devices. Audio waveform sync works but is approximate (±1-2 frames). Timecode sync is mathematically exact. This extends the multi-cam sync feature for productions using professional timecode workflows.
- **Implementation**: Extract timecode from each source: container metadata, LTC audio track (44.2), or filename timestamp pattern. Find common timecode range across all sources. Compute frame-accurate offset per source relative to the earliest common timecode. Apply offsets to align all sources. Generate multicam XML/OTIO with frame-accurate alignment. UI: timecode source selector per clip (embedded / LTC track / manual entry / filename).
- **References**: Tentacle Sync, PluralEyes, DaVinci Resolve auto-sync by timecode, Premiere Pro merge clips

### 44.4 Drop-Frame / Non-Drop-Frame Conversion & Display (P3 / S)
Detect and handle drop-frame (29.97fps, 59.94fps) vs. non-drop-frame timecode correctly throughout the application. Display timecode with correct notation (semicolons for DF, colons for NDF). Convert between DF and NDF for mixed-format broadcast projects.
- **Why**: Incorrect timecode handling causes cumulative frame drift in broadcast workflows. A 1-hour NTSC program drifts by 3.6 seconds if DF/NDF is confused. This is a fundamental professional requirement that's easy to get wrong and catastrophic when it happens.
- **Implementation**: Detect frame rate from FFmpeg probe. For 29.97/59.94fps: default to drop-frame display (HH:MM:SS;FF with semicolons). Provide NDF toggle. Conversion: implement SMPTE 12M drop-frame algorithm (skip frame numbers 0 and 1 at each minute except every 10th minute). Apply consistently throughout: timecode display, burn-in overlay, sync calculations, EDL/AAF export.
- **References**: SMPTE 12M standard, Premiere Pro timecode display options, FFmpeg frame rate handling

---

## 45. Advanced Encoding & Delivery Codecs

### 45.1 ProRes Export on Windows (P1 / M)
Export Apple ProRes (Proxy, LT, 422, 422 HQ, 4444, 4444 XQ) on Windows — not natively supported by Apple's tools but fully achievable via FFmpeg's production-quality ProRes encoder. Essential for delivery to Mac-based post-production facilities and broadcast.
- **Why**: ProRes is the de facto delivery and intermediate codec for Mac-based post houses, broadcasters, and VFX pipelines worldwide. Windows users currently can't export ProRes without expensive third-party tools like Telestream Switch. FFmpeg's `prores_ks` encoder is high-quality and free.
- **Implementation**: FFmpeg `-c:v prores_ks` encoder with `-profile:v` selector (0=Proxy, 1=LT, 2=422, 3=422HQ, 4=4444, 5=4444XQ). Add ProRes profiles to the export preset system with auto-calculated bitrate targets. Warn about alpha channel support (4444/4444XQ profiles only). Auto-select profile based on use case: Proxy for review, 422 HQ for delivery, 4444 for VFX with transparency.
- **References**: FFmpeg prores_ks encoder, Apple ProRes White Paper, DaVinci Resolve ProRes export

### 45.2 AV1 Encoding Support (P1 / M)
Export using the AV1 codec via SVT-AV1 (fastest), libaom (reference quality), or hardware encoders (NVENC AV1, Intel QSV AV1). AV1 delivers ~30% better compression than H.265 at equivalent quality with zero licensing fees. Essential for YouTube (which prioritizes AV1 content) and next-generation streaming.
- **Why**: YouTube serves AV1 to capable devices, giving AV1 uploads better visual quality at lower bandwidth. No royalty fees unlike H.265/HEVC. NVIDIA RTX 40-series and Intel Arc GPUs support hardware AV1 encoding at high speed. AV1 is the present and future of web video compression.
- **Implementation**: FFmpeg `-c:v libsvtav1` (fastest software encoder) or `-c:v av1_nvenc`/`-c:v av1_qsv` (hardware). CRF quality control. Add AV1 presets: "YouTube AV1" (CRF 28, SVT-AV1 preset 8), "Archive AV1" (CRF 20, preset 4), "Fast AV1 (GPU)" (hardware NVENC/QSV). Auto-detect AV1 hardware encoder availability and prefer it when present. Expose in export settings alongside H.264 and H.265.
- **References**: SVT-AV1, Alliance for Open Media spec, YouTube AV1 support, FFmpeg AV1 encoding

### 45.3 DNxHR / DNxHD Export (P2 / M)
Export Avid DNxHR (resolution-independent) and DNxHD (HD-locked) intermediate codecs. Standard delivery format for Avid Media Composer-based facilities, broadcast houses, and many post-production workflows.
- **Why**: DNxHR/DNxHD is the Avid ecosystem's standard intermediate codec, used by thousands of broadcast and post-production facilities worldwide. Along with ProRes (45.1), DNx covers the two major intermediate codec families that professionals require.
- **Implementation**: FFmpeg `-c:v dnxhd` with `-profile:v dnxhr_hq` (or dnxhr_sq, dnxhr_hqx, dnxhr_444). Standard container: MXF (broadcast) or MOV (general). Auto-select profile based on resolution and bit depth. Add to export presets alongside ProRes profiles. Bundle with MXF container support (45.5) for complete broadcast delivery.
- **References**: Avid DNxHR codec specifications, FFmpeg DNxHD/HR encoding, DaVinci Resolve DNx export

### 45.4 Lossless Intermediate Codec Pipeline (P2 / M)
Use lossless or near-lossless intermediate codecs (FFV1, HuffYUV, UT Video) for internal processing pipelines where quality preservation across multiple processing stages is critical. Ensures zero generational loss between sequential operations.
- **Why**: Multi-step processing chains (denoise → upscale → color grade → stabilize) re-encode at each stage. Even high-quality lossy codecs accumulate visible artifacts across 4-5 generations. Lossless intermediates preserve full quality through the entire pipeline, with only the final export using a lossy delivery codec.
- **Implementation**: Configurable intermediate codec in pipeline/workflow settings: FFV1 (best compression ratio, moderate speed), HuffYUV (fastest encode, largest files), or UT Video (Windows-native, good balance). Apply automatically during multi-step workflow processing. Final workflow step: encode to user-selected delivery codec. UI: pipeline quality mode selector (Speed: H.264 intermediates, Balanced: ProRes, Maximum: FFV1 lossless).
- **References**: FFV1 (Library of Congress digital preservation codec), FFmpeg lossless codecs, Lagarith

### 45.5 MXF Container Support (P2 / M)
Import and export Material Exchange Format (MXF) files — the broadcast industry standard container used by professional cameras (Sony XDCAM, Panasonic P2, ARRI ALEXA, RED) and all broadcast infrastructure. Handle OP1a, OP-Atom, and AS-11 DPP container variants.
- **Why**: Broadcast facilities worldwide work in MXF. Professional cameras output MXF. Post houses and broadcasters require MXF delivery. Without MXF support, OpenCut cannot participate in broadcast delivery workflows. FFmpeg supports MXF natively.
- **Implementation**: Import: FFmpeg reads all MXF variants natively — detect and preserve embedded timecode, descriptive metadata, and multiple audio tracks. Export: FFmpeg MXF muxer with OP1a (single interleaved file, most common) or OP-Atom (separate essence files, Avid-compatible) modes. AS-11 compliance: validate mandatory metadata fields for UK DPP broadcast delivery spec. Add MXF presets in export options alongside MP4/MOV.
- **References**: SMPTE MXF specification, FFmpeg MXF muxer, AS-11 DPP delivery spec, AMWA MXF tools

---

## 46. AI Editing Agent & Intelligence

### 46.1 Multi-Step Autonomous Editing Agent (P1 / XL)
An autonomous AI agent that plans and executes multi-step editing workflows from high-level natural language instructions. "Take this hour of interview footage, find the 5 most insightful moments, cut them into a 3-minute video with captions, background music, and a branded intro" — the agent generates an execution plan, runs each step via OpenCut's API, validates results at each stage, and delivers the final output with a change summary.
- **Why**: The existing NLP command system handles single operations. The chat editor handles conversational back-and-forth. Neither can plan and execute an autonomous multi-step workflow. This is the ultimate productivity feature — describe what you want, walk away, come back to a rough cut. Extends the existing AI rough cut concept (21.3) with a proper agent framework.
- **Implementation**: LLM agent framework (existing llm.py) with tool-use capability where available tools are OpenCut API endpoints. Agent loop: (1) Parse user request into goal. (2) Generate execution plan (ordered list of API calls with data dependencies). (3) Execute plan step-by-step, feeding each step's output to the next. (4) Self-validate results at each step (check audio levels meet targets, verify captions generated successfully, confirm export completed). (5) Error recovery: if a step fails, diagnose cause and either retry with adjusted parameters or adapt the plan. (6) Report results with full change summary.
- **References**: LAVE (LLM Agent for Video Editing), Devin (autonomous coding agent), Claude tool use

### 46.2 Context-Aware Command Suggestions (P2 / M)
Proactively suggest relevant operations based on the current editing context: clip type, recent actions, timeline state, and detected quality issues. "This clip has low audio — normalize?" "You've applied captions — add animated highlights?" Displayed as non-intrusive suggestion chips in the panel that the user can accept with one click or dismiss.
- **Why**: OpenCut has 50+ features across 8 tabs, but users often don't know which operation to use next or even that a feature exists. Context-aware suggestions guide the workflow naturally and surface useful features at exactly the right moment.
- **Implementation**: Context signals: clip metadata (audio levels, duration, codec, resolution), recent API calls (last 5 operations), timeline state (number of cuts, effects applied, captions present), detected issues (low audio, no captions, inconsistent color, unstabilized footage). Rule engine + LLM ranking: generate 2-3 relevant suggestions per context state. Display as dismissible pill-shaped suggestion chips below the active operation card. Track dismissals to suppress unwanted suggestions. Learn from user accept/dismiss patterns.
- **References**: GitHub Copilot inline suggestions, VS Code IntelliSense, Adobe Sensei suggestions

### 46.3 AI Project Organization & Bin Structuring (P2 / M)
Analyze all project media and automatically organize into logical bins: A-roll vs. B-roll, grouped by speaker identity, by location, by scene, by shot type, and by capture date. Create a sensible folder structure in the Premiere project panel based on AI analysis of the footage content.
- **Why**: Large projects with 100+ clips become unmanageable without organization. Manual sorting takes hours at the start of every project. AI classification (using existing shot type detection, face recognition, scene analysis, and metadata) can auto-organize the entire project in seconds.
- **Implementation**: Analyze all project items via existing modules: (1) Shot type classification (29.1 infrastructure). (2) Face clustering for speaker identification (existing InsightFace). (3) Scene/location grouping via visual similarity clustering (CLIP embeddings). (4) Date grouping from file metadata. Generate proposed bin structure. Create bins in Premiere via ExtendScript `app.project.rootItem.createBin()`. Move items to appropriate bins. UI: preview proposed organization before applying, with drag-to-adjust and manual override.
- **References**: Kyno auto-organization, iconik AI tagging, DaVinci Resolve smart bins

### 46.4 Natural Language Batch Operations (P2 / M)
Execute batch operations described in natural language: "Normalize all interview clips to -16 LUFS", "Add captions to every clip tagged 'talking-head'", "Apply the warm sunset LUT to everything in the B-roll bin", "Export all clips shorter than 60 seconds as vertical shorts." Operates across filtered sets of clips, not just the currently active selection.
- **Why**: Batch operations exist but are configured through UI forms one clip at a time. Natural language lets users express complex batch intents concisely. Combines the NLP command system with batch processing and footage search for maximum power.
- **Implementation**: Parse natural language with LLM (existing llm.py) into structured query: (1) Filter criteria (clip selection rules). (2) Operation to apply. (3) Parameters. Map filter criteria to footage_search queries (existing). Map operations to API endpoints. Preview: show matched clips count and planned operation before executing. Execute as batch job with per-clip progress tracking. Confirm destructive operations.
- **References**: Descript AI actions, Excel natural language queries, Frame chat-driven editing

---

## 47. Batch Operations & Mass Processing

### 47.1 Batch Transcode with Preset Matrix (P1 / M)
Transcode multiple files simultaneously with a matrix of output presets: select 20 source clips × 3 delivery presets = 60 output files queued and processed. Per-file progress tracking, priority ordering (finish urgent deliverables first), and automatic output folder organization by preset name.
- **Why**: Media ingest and delivery workflows routinely require transcoding dozens or hundreds of files to multiple output formats simultaneously. Current batch processing handles one operation at a time with manual configuration. A dedicated transcode queue with preset matrix handles the real-world production workflow.
- **Implementation**: Input: file list (folder scan, drag-drop, or project media selection) + preset list (multiple selected from export presets). Generate job matrix (files × presets). Execute via batch_executor with configurable parallelism (1 worker for GPU-heavy presets, N workers for CPU/hardware-encoded presets). Output folder organized by preset name automatically. Progress: per-file and overall dashboard. Resume capability: track completed items, skip on restart.
- **References**: Adobe Media Encoder, HandBrake queue, Apple Compressor, FFmpeg batch scripts

### 47.2 Batch Metadata Editor (P2 / M)
View and edit metadata fields (title, author, copyright, description, creation date, GPS, custom tags) across multiple files simultaneously in a spreadsheet-style grid. Bulk operations: apply templates, search/replace across fields, strip specific metadata types, import from CSV.
- **Why**: Large projects and delivery workflows need consistent metadata across all output files. Manual per-file metadata editing is impractical at scale. Broadcast delivery specs require specific metadata fields populated correctly across every deliverable.
- **Implementation**: Read metadata via FFmpeg probe (`format_tags`, `stream_tags`). Display as editable spreadsheet grid in panel (file per row, metadata field per column). Bulk operations: fill column with value, find/replace across field, clear field, apply metadata template. Write metadata via FFmpeg `-metadata key=value` flags (stream copy to avoid re-encoding). Export metadata inventory as CSV. Import metadata updates from CSV/spreadsheet.
- **References**: ExifTool GUI, Adobe Bridge metadata editor, Kyno metadata panel

### 47.3 Batch Watermark Application (P2 / S)
Apply a watermark (image logo or text) to multiple files in batch with consistent positioning, scaling, opacity, and margin. Essential for generating review copies, branded deliverables, and stock footage with consistent branding.
- **Why**: Generating review copies or branded deliverables across dozens of files requires consistent watermarking. One-at-a-time application is impractical for projects with many deliverables. Complements the per-export watermark toggle (5.5) with bulk processing of existing files.
- **Implementation**: Input: file list + watermark config (image path or text string, position preset, scale relative to frame, opacity, margin from edges). Process via FFmpeg `overlay` filter (image watermark) or `drawtext` filter (text watermark) per file. Batch execute with per-file progress. Presets: "Draft Review" (center, large, 40% opacity), "Brand Logo" (corner, small, 85% opacity), "Stock Preview" (tiled diagonal pattern). Stream copy audio to avoid unnecessary audio re-encoding.
- **References**: Visual Watermark app, FFmpeg overlay filter, Premiere Pro export with graphic

### 47.4 Batch Thumbnail Extraction (P2 / S)
Extract representative thumbnails from multiple video files: at specific timestamp offsets, at scene-detected key moments, or at the AI-scored "most visually interesting" frame. Export as organized image files for media catalogs, contact sheets, and web galleries.
- **Why**: Building media catalogs, content management databases, web galleries, and client deliverable indexes all require thumbnail images from every video file. Manual per-file screenshot extraction is impractical at scale.
- **Implementation**: Per file: extract frame(s) via FFmpeg `-ss [time] -vframes 1`. Extraction modes: (1) Fixed timestamp (e.g., 10% into clip duration). (2) Scene detection key frames (existing scene_detect) → first key frame per scene. (3) AI-scored best frame (existing thumbnail scoring from thumbnail.py). Output: organized by source filename, configurable resolution and format (JPEG/PNG). Optional: generate contact sheet grid image (Pillow montage of all thumbnails) as single-page visual overview.
- **References**: FFmpeg thumbnail extraction, contact sheet tools, DaVinci Resolve stills export

### 47.5 Batch Frame Rate & Resolution Conforming (P2 / M)
Conform a folder of mixed-format source files to a single target specification: frame rate, resolution, pixel aspect ratio, codec, and audio sample rate. Essential for preparing mixed-source material from multiple cameras and devices for consistent editing or delivery.
- **Why**: Multi-camera shoots, user-generated content collections, and archival restoration projects routinely contain clips with mixed formats. Manually conforming each clip to project settings wastes hours. Automated batch conforming handles the entire ingest pipeline in one operation.
- **Implementation**: Scan input folder → FFmpeg probe each file for frame rate, resolution, PAR, codec, audio sample rate. Compare against user-defined target spec. Generate per-file conform plan (list which conversions are needed). Execute: frame rate conversion via FFmpeg `fps` filter or RIFE interpolation (existing, for quality), resolution via scale filter (lanczos for downscale, existing AI upscale option for upscale), audio resample via `aresample`. Report: conforming summary with per-file changes applied, flagging any clips that required major format changes.
- **References**: DaVinci Resolve optimized media, Premiere Pro ingest settings, EditReady

---

## 48. Event & Ceremony Video

### 48.1 Multi-Camera Ceremony Auto-Edit (P2 / L)
Given synchronized multi-camera recordings of an event (wedding, conference, concert, panel discussion), automatically select the best camera angle at each moment: wide shot for establishing context, close-up on the active speaker, reaction shots during key moments, cutaway to audience during applause or laughter.
- **Why**: Multi-camera event editing is the most tedious form of post-production — hours of manually switching between 2-5 camera angles for a 2-hour ceremony. Automating angle selection based on audio activity, face detection, and motion analysis produces a competent first cut in minutes that the editor then refines.
- **Implementation**: Sync cameras (existing 3.9 or 44.3 timecode sync). Analyze each angle simultaneously: (1) Active speaker detection via diarization timestamps + face position tracking. (2) Motion intensity analysis for visual action. (3) Audio energy peaks for applause/laughter/music. Selection rules: default to wide establishing shot, switch to close-up on speaker changes (hold minimum 3-5 seconds per angle), cut to audience/reaction on applause peaks. Generate multicam XML with programmed angle switches. User fine-tunes in Premiere's native multicam viewer.
- **References**: Premiere multicam workflow, AutoCut multicam, Mevo auto-switch camera

### 48.2 Guest Message Compilation (P3 / M)
Import a folder of individual guest video messages (wedding wishes, birthday greetings, farewell messages, testimonials), auto-trim leading/trailing silence from each, normalize audio levels across all clips to consistent loudness, optionally generate name lower-thirds from filenames, and concatenate into a seamless compilation with transitions between messages.
- **Why**: Compilations of guest video messages are a staple of weddings, retirement parties, milestone celebrations, and corporate testimonials. Each clip needs silence trimming, audio leveling, and often a name card. Manual processing of 20-50 individual clips takes hours of repetitive work.
- **Implementation**: For each clip in folder: silence detection + head/tail trim (existing) → loudness normalization to common LUFS (existing) → optional lower-third from filename parsing (e.g., "John_Smith.mp4" → "John Smith" text overlay). Concatenate all with configurable transition (existing 34 transitions). Optional: sort alphabetically, randomize, or by clip duration. Add optional background music with auto-duck under speech (existing). Export as single compiled video.
- **References**: Tribute.co, Kudoboard, iMovie Magic Movie

### 48.3 Photo + Video Montage Builder (P2 / M)
Assemble a mixed montage from photos and video clips: apply Ken Burns pan-zoom animation to photos with intelligent focus on faces and subjects, interleave with video clips, sync all cuts to background music beats, and add configurable transitions between items. The classic "slideshow with videos" format used at every celebration.
- **Why**: Every event video includes a photo montage section. Every memorial, anniversary, graduation, and wedding celebration needs one. Currently requires manual work in After Effects or iMovie. Automating with music sync and intelligent face-aware Ken Burns animation makes it a one-click operation.
- **Implementation**: Import photo/video folder → detect item type. Photos: detect faces and subjects for Ken Burns focus points (existing face detection + content saliency), apply animated pan-zoom with smooth easing. Videos: trim to highlight or use full duration. Ordering: user-specified, date-sorted (from EXIF), or shuffled. Music sync: detect beats (existing librosa) → align cuts to beat or bar boundaries (extends 16.1 beat-synced cuts). Transitions: auto-apply between all items with variety. Export as standalone video or Premiere sequence XML.
- **References**: iMovie Magic Movie, Google Photos memories, Apple Photos montage, GoPro Quik

### 48.4 Event Recap Reel Generator (P2 / M)
From a multi-hour event recording (conference, gala, festival, workshop), automatically extract the best 3-5 minute highlight reel: keynote excerpts, speaker introductions, audience reactions, crowd atmosphere, and venue establishing shots. Score segments by energy, visual variety, and content relevance to produce a compelling recap.
- **Why**: Every event organizer needs a recap video for social media, future marketing, and stakeholder reporting. Watching 4-8 hours of raw footage to find 3-5 minutes of highlights is the primary bottleneck. AI scoring and extraction automates the selection process.
- **Implementation**: Run full analysis pipeline on event recording: scene detection (existing) → shot type classification (29.1 for wide/close/crowd) → audio energy scoring (existing) → transcript analysis for key quotes and memorable moments (LLM via llm.py) → face/crowd density detection. Score each segment across multiple dimensions: energy × visual variety × content relevance × shot type diversity. Select top segments with variety constraints (avoid consecutive identical shot types). Assemble with transitions and optional background music bed. Export at target duration with music fade.
- **References**: Opus Clip for events, Magisto (Vimeo), Lumen5, Wibbitz

---

## 49. Version Control & Edit History

### 49.1 Edit Decision Snapshots (P2 / M)
Save named snapshots of the current editing state: all cut decisions, effects applied, captions generated, and processing parameters at that point in time. Restore any snapshot to roll back to a previous edit state. Compare snapshots to see what changed between edit rounds.
- **Why**: "I liked the previous version better" is the most common client and director feedback. Currently, the only undo is Premiere's linear undo stack, which is lost on session close and cannot be named or compared. Named snapshots let editors save milestones, explore creative alternatives, and revert at any time.
- **Implementation**: Snapshot captures: serialized timeline state (via ExtendScript — clip positions, effects, markers), OpenCut job history relevant to current project (from jobs.db), list of generated output files with content hashes, and all processing parameters used. Store in `~/.opencut/snapshots/<project_hash>/<snapshot_name>.json`. Restore: re-apply serialized timeline state via ExtendScript, re-link generated files. UI: snapshot list with name, date, thumbnail preview, and visual indicator of difference from current state.
- **References**: Git commits (conceptual model), Photoshop History snapshots, DaVinci Resolve timeline versions

### 49.2 Timeline Diff / Comparison View (P3 / L)
Compare two edit snapshots side-by-side: highlight added clips, removed clips, repositioned clips, changed effects, modified audio levels, and new/deleted markers. Visual timeline diff showing exactly what changed between any two edit versions.
- **Why**: Across multiple revision rounds with client feedback, tracking what actually changed between versions is difficult and error-prone. A visual diff between edit snapshots gives editors and clients a clear, verifiable picture of all modifications.
- **Implementation**: Serialize timeline state from two snapshots into comparable data structures (ordered list of clips with source path, in/out points, timeline position, and applied effects). Diff algorithm: identify added, removed, repositioned, and modified clips using source+position matching. Visual render: timeline diagram with color-coded changes (green=added, red=removed, yellow=modified parameters, blue=repositioned). Export diff as text report, visual image, or HTML page.
- **References**: Git diff visualization, DaVinci Resolve timeline comparison, Frame.io version compare

### 49.3 Branching Edit Workflows (P3 / L)
Create named edit branches (analogous to git branches) to explore different creative directions without losing the original: "director's cut", "social media version", "client revision 2", "festival submission." Switch between branches independently. Optionally merge non-conflicting changes from one branch into another.
- **Why**: Complex projects often require multiple parallel versions (director's cut vs. broadcast cut vs. social cut vs. international version). Managing these as duplicate Premiere project files is error-prone and wastes storage. Branch-based editing keeps parallel versions organized under one project and allows shared improvements to flow between versions.
- **Implementation**: Branch = named chain of edit snapshots diverging from a common point. Create branch from any snapshot. Switch branches: restore the corresponding snapshot state. Track divergence point for potential merge. Simple merge strategy: apply non-conflicting changes (new clips, added effects) from source branch. Conflict resolution: flag clips modified in both branches for manual review. UI: branch graph visualization showing divergence points and optional merge arrows.
- **References**: Git branching model (conceptual), DaVinci Resolve timeline versions, Avid bin-based versioning

### 49.4 Change Annotations & Revision Notes (P3 / S)
Attach explanatory notes to specific edit changes: "Shortened intro per client call 4/10", "Replaced music track to avoid copyright claim", "Added lower-third for new panelist Dr. Kim." Annotations persist with snapshots and export as a structured revision history document.
- **Why**: Edit revisions without documentation lead to confusion about why changes were made, especially across team handoffs or long projects. Annotated revision history creates accountability, context for future editors, and a paper trail for client communication.
- **Implementation**: Annotation data structure: text note + associated change reference (clip ID, effect modification, etc.) + timestamp + author name. Store alongside snapshot data in the snapshot JSON. Display as annotation layer in timeline diff view (49.2). Export as formatted revision history document (markdown or PDF): chronological list of changes grouped by revision round, with reasons and author attribution.
- **References**: Frame.io comment threads, Google Docs suggestion mode, code review annotations (GitHub PR)

---

## 50. Third-Party Integrations

### 50.1 Frame.io Review Integration (P2 / L)
Upload processed video to Frame.io for client review, receive timestamped review comments back into the OpenCut panel, and apply requested changes. Two-way sync between the editing workflow and Frame.io's review/approval process.
- **Why**: Frame.io is the industry standard for video review and approval (acquired by Adobe, 1M+ professional users). Integrating review feedback directly into the editing panel eliminates the manual step of reading Frame.io comments separately and manually finding the corresponding timeline positions.
- **Implementation**: Frame.io v2 REST API: upload via multipart with progress tracking. Poll for new comments on uploaded assets. Display comments as timeline markers with text popover and commenter name. Map Frame.io comment timestamp → Premiere timeline position for one-click navigation. Two-way: mark comments as "resolved" from panel after addressing. OAuth2 authentication flow. Respect API rate limits.
- **References**: Frame.io API v2, Premiere Pro Frame.io panel (Adobe-native), Frame.io developer docs

### 50.2 Notion / Project Management Sync (P3 / M)
Sync project status, deliverables checklist, and revision notes with Notion databases (or other PM tools like Airtable, Monday.com, Asana) via their APIs. Auto-update task status as exports complete and revision notes as edits are finalized.
- **Why**: Video projects exist within larger production pipelines managed in project management tools. Manual status updates are forgotten and become stale. Auto-sync keeps the entire team informed of editing progress without any extra effort from the editor.
- **Implementation**: Notion API integration: map OpenCut project → Notion database page. Auto-update fields: editing status (in progress/review/delivered), current version number, last export date, deliverables list with download links if hosted. Webhook trigger on job completion events. Configurable field mapping via Settings UI. Also support generic webhook POST payload for any PM tool not directly integrated.
- **References**: Notion API, Frame.io project management integration (design pattern), Zapier templates

### 50.3 Slack / Discord Notification Bot (P2 / M)
Send rich notifications to Slack channels or Discord servers when key editing events occur: export completed (with thumbnail and download link), review copy uploaded, long-running job finished, error encountered requiring attention. Keep the production team informed without manual updates.
- **Why**: Team coordination for video production happens in Slack and Discord. Push notifications eliminate "is the render done yet?" messages and keep producers, clients, and collaborators informed in real-time without the editor manually updating anyone.
- **Implementation**: Slack: incoming webhook with Block Kit message formatting (thumbnail image, title, status badge, duration, file size, download link). Discord: webhook with rich embed (similar fields). Configurable event triggers: job_complete, export_ready, error_occurred, review_uploaded, batch_complete. Per-channel routing (exports→#deliveries, errors→#tech-support). Settings UI: webhook URL input, event toggle checkboxes, test notification button.
- **References**: Slack Block Kit builder, Discord webhook embeds, GitHub-Slack integration (design pattern)

### 50.4 Zapier / Make / n8n Webhook Actions (P3 / M)
Expose OpenCut events as outbound webhook triggers and accept inbound webhooks as operation triggers. Enables connecting OpenCut to 5000+ apps via automation platforms: "When a file appears in Google Drive → download → process through OpenCut → upload result to YouTube."
- **Why**: Automation platforms (Zapier, Make, n8n) connect everything. Exposing OpenCut as a webhook-capable service unlocks unlimited integration possibilities without building and maintaining each individual integration.
- **Implementation**: Outbound: POST structured event data (job type, status, result summary, output file paths) to configurable webhook URLs on key events. Inbound: accept POST requests at `/api/webhook/trigger` with operation name + file URL, download the file, execute the specified operation with provided parameters, POST result back to callback URL. Authentication: API key in `X-OpenCut-Key` header. Rate limiting. UI: webhook configuration panel with URL, event selection, and test button.
- **References**: Zapier webhook triggers, n8n HTTP Request node, Make webhook module

### 50.5 Google Drive / Dropbox / S3 Cloud Storage Integration (P2 / M)
Import media files directly from cloud storage services: browse Google Drive, Dropbox, or AWS S3 buckets from within the OpenCut panel, select files, and download to local processing folder. Also: auto-upload completed exports back to configured cloud destinations.
- **Why**: Remote teams and distributed productions store source footage in cloud storage. Currently must manually download files through a browser before processing, then manually re-upload results. Direct cloud integration eliminates both manual steps and closes the loop for remote production workflows.
- **Implementation**: OAuth2 authentication for Google Drive and Dropbox APIs. AWS credentials (access key + secret) for S3. Browse UI: folder tree + file list with thumbnails + size/date info. Download with progress to `~/.opencut/ingest/` or project media folder. Upload: after export, optionally upload result to configured cloud destination folder with matching directory structure. Track sync status to avoid re-downloading unchanged files.
- **References**: Google Drive API v3, Dropbox API v2, AWS S3 boto3 SDK, Frame.io cloud workflow

---

## 51. 360° / VR / Immersive Video

### 51.1 360° Video Stabilization (P2 / L)
Stabilize 360° equirectangular footage using gyroscope metadata (GoPro, Insta360) or visual feature tracking. Compensate for camera rotation without introducing black borders (wrap-around projection). Output stabilized equirectangular or rectilinear.
- **Why**: Every 360 camera produces shaky footage. DaVinci Resolve and GoPro Player have built-in 360 stabilization. OpenCut's existing stabilizer assumes flat projection and produces corrupted output on equirectangular footage.
- **Implementation**: Parse gyroscope data from MP4 GPMF (GoPro) or Insta360 metadata. Apply inverse rotation in equirectangular space via FFmpeg `v360` filter or OpenCV remap. For metadata-free footage: detect features across equirectangular seam → estimate rotation per frame → apply inverse. Output as stabilized equirectangular MP4.
- **References**: GoPro ReelSteady 360, Insta360 Studio, DaVinci Resolve 360 stabilizer

### 51.2 Equirectangular to Flat Projection Conversion (P2 / M)
Extract flat (rectilinear) views from 360° equirectangular footage at any viewing angle and field of view. Keyframe the virtual camera direction over time to create cinematic pans through 360 content. Export as standard flat video.
- **Why**: Most 360 footage is consumed as flat video on social media. Manually extracting views requires After Effects or specialized tools. Automating virtual camera paths turns raw 360 captures into cinematic flat content.
- **Implementation**: FFmpeg `v360` filter: `v360=e:flat:yaw=X:pitch=Y:roll=Z:h_fov=90:w_fov=120`. Keyframe yaw/pitch/roll over time via filter timeline. Panel UI: interactive 360 preview (Three.js equirectangular viewer) where user clicks to set viewing direction at timestamps. Generate smooth interpolated camera path between keyframes.
- **References**: Insta360 Studio reframe, GoPro OverCapture, Adobe Premiere VR

### 51.3 FOV Region Extraction from 360 (P2 / M)
Auto-detect regions of interest in 360° footage (faces, motion, audio direction) and extract multiple flat video clips — one per subject or area of interest. Enables shooting 360 and extracting multiple traditional camera angles in post.
- **Why**: A single 360 camera can replace 3-4 traditional cameras at events, interviews, and meetings. Auto-extraction of per-subject views saves hours of manual reframing. Insta360's "FlowState" does this for action sports but not for dialogue/interview scenarios.
- **Implementation**: Detect faces in equirectangular space via face_reframe.py (adapted for spherical coordinates). Cluster faces by identity (InsightFace embeddings). Generate per-speaker flat extraction with smooth tracking. Combine with audio diarization (existing) to auto-switch between speakers. Output: one flat video per detected subject + a multicam XML for Premiere.
- **References**: Insta360 FlowState, Kandao QooCam Studio, Facebook 360 Director

### 51.4 Spatial Audio Alignment for VR (P3 / L)
Map audio sources to spatial positions in 360° video. Convert stereo/mono dialogue to spatialized ambisonic audio where sound follows the speaker's position in the 360 frame. Export as first-order ambisonics (YouTube VR format).
- **Why**: 360 video without spatial audio is disorienting — sound stays fixed while the visual world rotates. YouTube VR and Meta Quest require ambisonic audio for proper 360 playback. Creating spatial audio currently requires Pro Tools or Facebook 360 Spatial Workstation (discontinued).
- **Implementation**: Detect speaker positions from face detection in equirectangular frames. Map screen position to azimuth/elevation angles. Apply FFmpeg `aphasemeter` and custom spatial panning to route mono dialogue to the correct ambisonic channel. Output as first-order ambisonics (4-channel WAV in ACN/SN3D format). Mux with equirectangular video in YouTube's metadata format.
- **References**: YouTube VR spatial audio spec, Facebook 360 Spatial Workstation, Reaper ambisonics

---

## 52. Camera & Lens Correction

### 52.1 Lens Distortion Correction (P1 / M)
Correct barrel distortion, pincushion distortion, and mustache distortion from wide-angle and fisheye lenses. Auto-detect camera model from EXIF/metadata and apply the matching correction profile. Manual mode with adjustable distortion coefficients.
- **Why**: Wide-angle action cameras (GoPro, DJI), drone cameras, and budget lenses all introduce visible distortion. Premiere's lens correction is basic and requires manual lookup. Auto-detection from metadata eliminates guesswork.
- **Implementation**: FFmpeg `lenscorrection` filter with k1/k2 coefficients. Build a camera profile database (JSON) mapping camera model → distortion coefficients (source: lensfun open-source database, 1000+ profiles). Read camera model from ffprobe metadata `com.apple.quicktime.model` or EXIF. Manual mode: distortion sliders with preview. OpenCV `undistort()` for more complex profiles.
- **References**: lensfun project (open-source lens database), Premiere lens distortion removal, GoPro lens profiles

### 52.2 Rolling Shutter Correction (P2 / L)
Fix rolling shutter (jello effect) caused by CMOS sensors scanning top-to-bottom during fast motion or camera shake. Analyze inter-frame motion to estimate per-row time offset and straighten warped frames.
- **Why**: Every phone, drone, and mirrorless camera has a CMOS sensor with visible rolling shutter on fast pans or vibrations. DaVinci Resolve, Premiere, and FCPX all offer rolling shutter correction. It's a table-stakes fix for action and drone footage.
- **Implementation**: Estimate global motion (optical flow between frames), then estimate per-row deviation from global motion as the rolling shutter warp. Correct by shifting each row backward in time using the motion estimate. FFmpeg `deshake` has limited RS correction. For better results: OpenCV optical flow → row-by-row warp → output via VideoWriter. Alternatively: `gyroflow` open-source project can be integrated as a subprocess with its own lens profiles.
- **References**: Gyroflow (open-source, Rust), DaVinci Resolve RS correction, After Effects rolling shutter repair

### 52.3 Chromatic Aberration Removal (P3 / S)
Remove color fringing (red/cyan or blue/yellow edges) caused by lens chromatic aberration. Detect fringe colors at high-contrast edges and shift color channels to realign.
- **Why**: Cheap lenses and wide-angle shots show visible purple/green fringing. Every photo editor (Lightroom, Photoshop) has one-click CA removal. Video tools lag behind.
- **Implementation**: FFmpeg `chromanr` filter for simple reduction. For precise correction: separate R/G/B channels, scale R and B channels by sub-pixel amounts (0.998-1.002) relative to G to realign. Detect optimal scale factors by minimizing edge color variance. Apply per-frame or globally.
- **References**: Lightroom CA removal, DxO PureRAW, lensfun CA profiles

### 52.4 Lens Profile Auto-Detection from Metadata (P2 / S)
Read camera body, lens model, focal length, and aperture from video file metadata (EXIF, QuickTime atoms, MKV tags). Auto-select the matching correction profile for distortion, vignetting, and chromatic aberration. Display detected camera info in the panel.
- **Why**: Manual lens selection from a dropdown of hundreds of lenses is tedious and error-prone. Metadata-driven auto-detection is instant and always correct when metadata is present.
- **Implementation**: Parse `ffprobe -show_format -show_streams` output for camera-related tags (varies by container: QuickTime `com.apple.quicktime.make`/`model`, MKV `ENCODER`, MP4 `handler_name`). Look up in lensfun database or custom JSON mapping. Display in clip preview card. Auto-apply corrections when a profile match is found.
- **References**: lensfun auto-detect, Adobe Camera Raw profiles, DxO auto-detect

---

## 53. Video Repair & Restoration

### 53.1 Corrupted File Recovery (P1 / M)
Attempt to recover playable video from corrupted or truncated files: fix broken container headers, recover partial recordings (camera crash, power loss, SD card corruption), and reconstruct missing moov atoms for MP4 files.
- **Why**: "My recording was cut short" and "the file won't play" are extremely common. recover_mp4 and untrunc exist but require technical knowledge. A one-click recovery tool in the panel would save countless lost recordings.
- **Implementation**: Detect corruption type: missing moov atom (MP4), truncated stream, broken headers. For missing moov: generate moov from a reference file shot with the same camera settings (untrunc algorithm). For truncated streams: re-mux with `ffmpeg -err_detect ignore_err` to salvage playable frames. Report: recovered duration, frame count, data loss percentage.
- **References**: untrunc (open-source), recover_mp4, Stellar Repair for Video

### 53.2 Adaptive Deinterlacing (P1 / S)
Convert interlaced footage (1080i, 480i from broadcast, legacy cameras, screen captures) to progressive with intelligent field detection. Auto-detect interlaced content from metadata or field analysis. Apply yadif (fast), bwdif (quality), or QTGMC (maximum quality) deinterlacing.
- **Why**: Legacy footage (tapes, broadcast recordings, security cameras) is interlaced. Playing interlaced content on modern displays shows visible combing artifacts. Every NLE and player has deinterlacing. OpenCut processes video without detecting or handling interlaced input.
- **Implementation**: Detect interlacing via `ffprobe` `field_order` tag or FFmpeg `idet` filter (statistical field analysis). Apply appropriate deinterlacer: `yadif=1` (fast, double framerate), `bwdif` (better quality), or `yadif=0` (same framerate). Auto-detect top-field-first vs bottom-field-first. Display interlacing status in clip metadata preview.
- **References**: FFmpeg yadif/bwdif filters, AviSynth QTGMC, HandBrake deinterlace

### 53.3 Old Footage Restoration Pipeline (P2 / L)
One-click restoration pipeline for degraded footage: stabilize → deinterlace → denoise (temporal) → upscale (Real-ESRGAN) → color restore (auto white balance + desaturation fix) → frame rate conversion. Optimized preset chains for VHS, 8mm film, and early digital.
- **Why**: Film scanners, VHS digitizers, and old digital cameras produce footage that needs multiple correction steps in a specific order. A dedicated restoration pipeline chains existing modules intelligently. Topaz Video AI charges $299 for this.
- **Implementation**: Orchestrate existing modules: stabilize_video → deinterlace (53.2) → denoise (FFmpeg nlmeans or BasicVSR++) → upscale (Real-ESRGAN with video model) → auto white balance (analyze frame, apply colorbalance correction) → frame interpolation (RIFE for 24→60fps). Preset chains: "VHS" (deinterlace + denoise heavy + upscale 2x + color fix), "8mm Film" (stabilize + gate weave removal + grain management + upscale 4x), "Early Digital" (denoise + upscale + sharpen).
- **References**: Topaz Video AI, DaVinci Resolve restoration tools, PixelTools

### 53.4 SDR-to-HDR Upconversion (P3 / L)
Convert standard dynamic range (SDR / Rec.709) video to high dynamic range (HDR10 / HLG) using AI-based inverse tone mapping. Expand the dynamic range, add specular highlights, and apply PQ or HLG transfer function. Output as HDR10 MP4 with proper SMPTE ST.2086 metadata.
- **Why**: HDR displays are now standard on phones, TVs, and monitors. Clients increasingly request HDR deliverables from SDR source footage. Manual HDR grading is a specialized skill. AI upconversion provides a reasonable automatic result.
- **Implementation**: Use ITM (inverse tone mapping) via FFmpeg `zscale` for color space conversion (bt709 → bt2020) + `tonemap` filter (inverse). For AI-based: train or use existing deep learning ITM model (Microsoft's HDRnet, Samsung's AI HDR). Apply PQ (HDR10) or HLG transfer function. Embed ST.2086 metadata (MaxCLL, MaxFALL) via FFmpeg `-metadata:s:v` or `x265 --master-display` params. Validate output with HDR10+ Analyzer.
- **References**: FFmpeg zscale + tonemap, Microsoft HDRnet, Samsung AI HDR, HandBrake HDR passthrough

### 53.5 Frame Rate Conversion with Optical Flow (P2 / M)
Convert video frame rate using optical flow interpolation for smooth, artifact-free results. Convert 24fps to 60fps (slow motion), 60fps to 24fps (cinematic), or any arbitrary rate. Superior to simple frame duplication or dropping.
- **Why**: Frame rate mismatches are constant in multi-camera shoots, stock footage integration, and deliverable specs. Basic frame dropping/duplication causes judder. RIFE already exists in OpenCut for frame interpolation but isn't exposed as a simple frame rate conversion tool.
- **Implementation**: Calculate target frame timing. Use RIFE (existing) for interpolation when upconverting. Use frame blending (`tblend` filter) or motion-compensated temporal interpolation (`minterpolate` FFmpeg filter) for smoother downconversion. Preset modes: "Smooth" (optical flow), "Cinematic" (3:2 pulldown removal for 29.97→23.976), "Sport" (high framerate).
- **References**: FFmpeg minterpolate, RIFE, DaVinci Resolve optical flow retiming, Twixtor

---

## 54. AI Video Generation & Synthesis

### 54.1 AI Outpainting / Frame Extension (P2 / L)
Extend the frame beyond its original boundaries using generative AI. Widen a 16:9 clip to ultrawide 21:9, or extend a vertical video to fill 16:9 without cropping — the AI generates the missing border content that seamlessly blends with the original footage.
- **Why**: Aspect ratio mismatches are constant (vertical phone footage in horizontal timelines, 4:3 archive footage in 16:9 projects). Current options are black bars, blur-fill backgrounds, or cropping. Generative outpainting fills the frame with contextually appropriate content. Adobe Firefly and Runway offer this.
- **Implementation**: Extract per-frame or per-keyframe. Extend canvas to target aspect ratio. Inpaint the extended borders using existing ProPainter (temporal coherence) or Stable Diffusion inpainting. For temporal consistency: outpaint keyframes at intervals, interpolate fill between them. Apply as FFmpeg overlay composite of original centered on extended canvas.
- **References**: Adobe Generative Expand, Runway Gen-2 extend, Stable Diffusion outpainting

### 54.2 Image-to-Video Animation (P2 / L)
Animate a still image into a short video clip using AI motion generation. Add camera motion (zoom, pan, orbit), subject motion (hair, clothing, environmental elements), or both. More sophisticated than Ken Burns — generates actual motion synthesis.
- **Why**: Sora, Kling, and Runway's image-to-video can turn any photo into a cinematic 4-second clip. Documentary and history content relies on animating archival photos. OpenCut has Ken Burns (11.4) but not AI-driven motion synthesis.
- **Implementation**: Use Stable Video Diffusion (SVD, already in OpenCut for B-roll), Wan 2.1, or CogVideoX with image conditioning. Input: still image + optional motion prompt ("slow zoom in with parallax", "hair blowing in wind"). Generate 2-6 second clip. Integrate as new option in existing B-roll generation route. Panel UI: image upload + motion prompt + duration.
- **References**: Stable Video Diffusion, Kling, Sora, Runway Gen-3

### 54.3 AI Scene Extension / Duration Stretch (P3 / XL)
Extend a clip's duration by generating new frames that continue the scene beyond its original end point. The AI predicts what happens next based on the visual context. Useful for extending reaction shots, establishing shots, or holding on a moment longer.
- **Why**: "I need this clip to be 2 seconds longer" is a constant editing problem. Currently the only options are speed-ramp (changes timing), freeze frame (static), or looping (obvious). Generative extension creates novel frames that look like a natural continuation.
- **Implementation**: Feed the last N frames to a video prediction model (SVD, CogVideoX) as conditioning. Generate continuation frames. Blend the transition with cross-fade or feature-matched compositing. This is bleeding-edge — quality may be inconsistent. Best for relatively static scenes (landscapes, establishing shots, reactions).
- **References**: Sora scene extension, Runway Gen-3 extend, Google Lumiere

### 54.4 AI Video Summary / Condensed Recap (P2 / M)
Generate a visually condensed video summary: automatically select the most important shots, trim them to their essential content, and assemble a 30-60 second recap of a longer video. Not just text summary — actual visual output.
- **Why**: Every long-form video needs a teaser/trailer. Auto-generating a visual summary from a 30-minute video saves hours of manual selection. Opus Clip and Pictory do this for social clips. A general-purpose visual summarizer is more versatile.
- **Implementation**: Combine existing modules: scene detection (shot boundaries) → transcript analysis (important moments via LLM) → engagement scoring (highlights) → select top N shots by score → trim each to 3-5s around peak moment → assemble with crossfade transitions → add background music (optional). Output as a self-contained summary clip. Configurable target duration.
- **References**: Opus Clip, Pictory, Google Video Intelligence summarization

### 54.5 Background Replacement with AI Generation (P2 / L)
Remove the background from a video subject (existing rembg/RVM) and replace it with an AI-generated environment from a text prompt. "Put me on a beach", "office background", "space station". Temporally consistent background generation.
- **Why**: Virtual backgrounds are mainstream (Zoom, Teams, Google Meet). Video background replacement is the next step — used for music videos, product demos, and creative content. Combining existing background removal with generative fill creates a powerful one-click effect.
- **Implementation**: (1) Extract foreground mask via RVM (existing). (2) Generate background image from text prompt via Stable Diffusion (add as optional dep). (3) For static backgrounds: single generation + overlay. For dynamic backgrounds: generate via SVD image-to-video + overlay foreground. (4) Feather mask edges, apply color matching between foreground and generated background for coherence.
- **References**: Runway background replacement, CapCut AI background, Unscreen

---

## 55. Privacy & Content Redaction

### 55.1 License Plate Detection & Blur (P1 / M)
Automatically detect vehicle license plates in video frames and blur, pixelate, or fill them. Track plates across frames for consistent redaction. Handles multiple plates, angles, and international plate formats.
- **Why**: Privacy regulations (GDPR, CCPA) require blurring plates in published content. Real estate, dashcam, vlog, and documentary creators need this constantly. Currently requires manual masking frame-by-frame or expensive tools like Adobe After Effects tracking.
- **Implementation**: Use PaddleOCR or OpenALPR (open-source) for plate detection, or YOLO with a plate-trained model. Detect plate bounding boxes per frame → track with IoU matching or SORT tracker → apply Gaussian blur to tracked regions. Export redaction metadata as JSON for audit trail. Panel UI: preview detected plates, toggle individual plates on/off.
- **References**: OpenALPR, PaddleOCR, Google Cloud Video Intelligence, YouTube blur tool

### 55.2 OCR-Based PII Redaction (P2 / L)
Detect and redact personally identifiable information visible in video frames: phone numbers, email addresses, Social Security numbers, credit card numbers, addresses, and names on documents. Uses OCR + regex pattern matching + named entity recognition.
- **Why**: Training videos, screen recordings, and documentary footage frequently contain visible PII that must be redacted before distribution. Manual redaction is slow and error-prone. Automated detection catches things humans miss.
- **Implementation**: Extract frames at intervals → OCR via PaddleOCR or Tesseract → regex pattern matching for structured PII (SSN: `\d{3}-\d{2}-\d{4}`, phone: various formats, email, credit card Luhn check) → NER via spaCy for names/addresses → track detected text regions across frames → apply blur mask. Report: list of all detected PII with timestamps for review before applying.
- **References**: Amazon Macie (AWS PII detection), Microsoft Presidio, Google DLP API

### 55.3 Profanity Bleep Automation (P1 / S)
Detect profanity in the audio transcript and automatically insert censor bleeps (1kHz tone) or silence over flagged words. Configurable profanity word list with severity levels. Optional: also blur the speaker's mouth during bleeped segments.
- **Why**: Broadcast, corporate, and family-friendly content requires profanity removal. Currently requires manual listening and editing. Whisper transcription already produces word-level timestamps — adding bleep generation is straightforward.
- **Implementation**: Transcribe with word-level timestamps (existing). Match words against configurable profanity list (ship with default English list, user can customize). Generate bleep audio: 1kHz sine tone at matching volume level, or silence. Mix bleep over original audio at flagged word timestamps using FFmpeg `amerge` with time-based mixing. Optional: detect mouth region at bleep timestamps and apply blur (existing face detection).
- **References**: YouTube auto-bleep (manual), TV broadcast bleep standards, Cleanfeed

### 55.4 Document & Screen Redaction in Video (P2 / M)
Detect documents, screens, monitors, and whiteboards in video frames. Highlight regions containing text and offer one-click blur/pixelate of the entire document area or specific text regions within it.
- **Why**: Conference recordings, office tours, vlogs, and training videos frequently capture whiteboards, monitor screens, and paper documents with confidential information. Detecting and redacting these surfaces is faster than finding individual text strings.
- **Implementation**: Detect rectangular surfaces via edge detection + perspective transform (document scanner algorithm). Classify as screen/document/whiteboard via aspect ratio + color analysis. Offer options: blur entire surface, OCR + selective text redaction (55.2), or manual region selection on detected surface. Track surface across frames with homography estimation.
- **References**: Microsoft Lens document detection, Google ML Kit document scanner, CamScanner

### 55.5 Audio Redaction & Speaker Anonymization (P3 / M)
Redact specific speaker voices from audio: replace a speaker's voice with a different synthetic voice while preserving the spoken content, or bleep entire speaker segments. Used for witness protection, anonymous interviews, and confidential source protection.
- **Why**: Documentary, journalism, and legal video frequently require voice anonymization. Current methods (pitch shift, voice disguise) are easily reversible. Proper anonymization requires voice conversion or resynthesis.
- **Implementation**: Diarize to identify target speaker segments (existing pyannote). For voice replacement: transcribe target segments → resynthesize with a different TTS voice (existing Kokoro/edge-tts) at matching timing. For full redaction: replace speaker audio with silence or tone. Ensure prosody (emphasis, pacing) is preserved in resynthesis for natural sound. For basic anonymization: pitch shift + formant shift via pedalboard.
- **References**: Witness protection voice disguise standards, RealTimeVoiceChanger, so-vits-svc

---

## 56. Spectral Audio Editing

### 56.1 Visual Spectrogram Editor (P2 / L)
Display audio as a scrollable spectrogram in the panel. Allow users to visually identify and select unwanted sounds (coughs, phone rings, background music bleed, HVAC hum) by painting on the spectrogram with a brush tool, then remove or attenuate the selected time-frequency region.
- **Why**: iZotope RX ($399-$1,299) is the industry standard for spectral editing. It's the most precise way to remove specific sounds without affecting surrounding audio. No free alternative exists as a video editing plugin. Offering even basic spectral editing in OpenCut would be unique.
- **Implementation**: Generate spectrogram via FFmpeg `showspectrumpic` or librosa STFT. Display as zoomable canvas in panel. Brush tool creates a time-frequency mask. Apply mask via inverse STFT: zero or attenuate masked bins, reconstruct waveform. Preview before applying. Export cleaned audio and remix into video.
- **References**: iZotope RX, Audacity spectrogram view, Adobe Audition spectral editor

### 56.2 Spectral Repair / Frequency Removal (P2 / M)
Remove specific frequency ranges or isolated sounds from audio without affecting the rest of the spectrum. Target: persistent hum (50/60Hz), HVAC drone (specific frequency band), ringing phones, and other narrowband interference.
- **Why**: FFmpeg's `highpass`/`lowpass`/`bandreject` filters are blunt instruments that affect everything in the frequency range. Spectral repair targets only the unwanted sound, preserving speech and music that overlap in frequency. The difference between "remove 60Hz hum" and "remove 60Hz hum without affecting the bass in the speaker's voice."
- **Implementation**: STFT analysis → identify target frequency bins (user-specified or auto-detected via peak analysis) → attenuate only those bins where the unwanted signal is dominant (spectral gating: if bin energy exceeds neighbors by threshold, it's interference) → inverse STFT. Auto-detect mode: find spectral peaks that persist across time (hum, buzz, drone) and offer one-click removal.
- **References**: iZotope RX De-hum, Audacity noise profile, Cedar DNS

### 56.3 AI Environmental Noise Classifier & Removal (P1 / M)
Classify and selectively remove specific environmental sounds: traffic, wind, rain, air conditioning, keyboard typing, construction, dogs barking, sirens. Unlike generic noise reduction which treats all non-speech as noise, this identifies and removes specific sound categories while preserving others (e.g., remove traffic but keep birdsong).
- **Why**: Generic noise reduction (existing noisereduce, DeepFilterNet) treats all non-speech uniformly. But sometimes you want to keep certain ambient sounds (nature, room tone) while removing specific unwanted ones (a passing truck, a phone ringing). Selective removal is more natural than blanket denoising.
- **Implementation**: Audio classification via YAMNet (Google, TFLite, 521 sound classes) or PANNs (pretrained audio neural networks). Classify sound events per time segment. User selects which sound classes to remove. Apply source separation (Demucs or BandIt) conditioned on the target class, or spectral masking guided by classifier confidence per time-frequency bin. Lighter alternative: detect noise event timestamps + crossfade to silence during those events.
- **References**: YAMNet (Google), PANNs, iZotope RX Dialogue Isolate, Adobe Podcast Enhance

### 56.4 Room Tone Auto-Generation & Matching (P1 / M)
Analyze a clip's ambient room tone profile and generate matching synthetic room tone to fill gaps left by silence removal, cut transitions, and audio edits. Prevents the jarring "dead silence" that makes edits obvious.
- **Why**: Every silence removal and cut creates an unnatural gap where room tone disappears. Professional audio editors always fill cuts with matching room tone. Currently OpenCut cuts to true silence, which sounds unnatural. Auto-filling with matched room tone is the industry-standard practice.
- **Implementation**: Analyze quiet segments of the clip (where speech is absent) to build a room tone profile: spectral envelope + amplitude characteristics. Synthesize matching room tone via spectral shaping of white noise (match the spectral envelope) or loop a clean room tone sample. Auto-insert synthesized room tone at all cut points and silence-removed segments. Crossfade 50ms at boundaries. Store per-clip room tone profile for reuse.
- **References**: iZotope RX Room Tone Match, Premiere Pro Auto-Fill, Pro Tools Strip Silence + fill

---

## 57. Split-Screen & Multi-View Templates

### 57.1 Split-Screen Layout Templates (P1 / M)
Preset and custom split-screen layouts: side-by-side (2-up), grid (2x2, 3x3), picture-in-picture variants, diagonal split, L-shaped, and asymmetric layouts. Each cell maps to a different source clip with independent scale, position, and crop controls.
- **Why**: Reaction videos, comparison content, tutorials, and multi-cam shows all need split-screen. Premiere's built-in split screen requires manual effect stacking. CapCut and iMovie have one-click split-screen templates. This is a highly requested missing feature class.
- **Implementation**: Define layouts as JSON: array of cells with x/y/w/h percentages + optional border style. Composite via FFmpeg `overlay` filter chain or Pillow frame-by-frame composition. Each cell: scale source to fit, apply crop mode (fill/fit/stretch), optional border/gap between cells. Panel UI: visual template picker + drag-to-assign clips to cells. Export as single composite video.
- **References**: CapCut split-screen, iMovie split-screen, Premiere multi-camera monitor

### 57.2 Reaction Video Template (P1 / M)
One-click reaction video layout: main content fills most of the frame, webcam/reaction footage in a corner or side panel with configurable size, position, rounded corners, and border. Auto-sync reaction audio with content audio.
- **Why**: Reaction videos are one of the largest YouTube categories. The layout is always the same pattern but requires manual compositing in Premiere. A dedicated template with auto-sync saves significant time.
- **Implementation**: Extend 57.1 with reaction-specific presets: corner PiP (4 positions), side panel (left/right), and bottom bar. Auto-sync via audio cross-correlation between reaction mic and content audio (time offset detection). Apply offset, composite layout, mix audio with ducking (lower content during reactor speech). Panel UI: assign "content" and "reaction" clips, pick layout, adjust size.
- **References**: StreamYard, OBS scenes, CapCut reaction template

### 57.3 Before/After Comparison Output (P2 / M)
Generate a side-by-side or wipe comparison video showing original vs. processed footage. Configurable split mode: vertical wipe (draggable line), horizontal wipe, side-by-side, and alternating (A-B-A-B with label overlay). Export as a single video for showcasing processing results.
- **Why**: Colorists, VFX artists, and restoration specialists need to show clients before/after comparisons. The panel has a preview modal (existing) but no way to export a comparison as a deliverable video. Social media posts showing processing results drive engagement and demonstrate value.
- **Implementation**: FFmpeg `hstack`/`vstack` for side-by-side. For wipe: `overlay` with dynamically positioned crop. For animated wipe: keyframe the wipe position over time. Add text labels ("Before" / "After") via drawtext. Alternating mode: intercut segments with label overlay. Panel UI: select original + processed files, pick comparison mode, export.
- **References**: Topaz comparison export, Lightroom before/after, DaVinci Resolve split-screen wipe

### 57.4 Multi-Cam Grid View Export (P2 / M)
Compose all multi-camera angles into a single grid view video (2x2, 3x3, up to 4x4) with optional active-speaker highlight border. Useful for review, logging, and selecting angles before fine-cut editing.
- **Why**: Multicam shoots produce 2-16 camera angles. Reviewing them individually is slow. A grid view showing all angles simultaneously — with the active speaker highlighted — lets editors make angle selection decisions quickly. Premiere's multi-camera monitor does this for playback but can't export it.
- **Implementation**: Extend 57.1 grid layouts. Add per-cell audio level meter overlay (FFmpeg `showvolume` or computed from PCM peaks). If diarization data exists (from existing multicam module), highlight the active speaker cell with a colored border that follows the cut list. Export as single composite video with optional per-cell timecode burn-in.
- **References**: Premiere multi-camera monitor, DaVinci Resolve multi-view, Blackmagic MultiView

---

## 58. Content Repurposing Pipeline

### 58.1 Long-Form to Multi-Short Auto-Extraction (P1 / L)
Analyze a long-form video (podcast, interview, lecture, vlog) and automatically extract multiple self-contained short clips optimized for different platforms. Each clip is individually scored, titled, trimmed, reframed to vertical, captioned, and exported — ready to post.
- **Why**: Every content creator with long-form video needs to extract 5-15 shorts per episode. Opus Clip ($228/year), Vizard, and Munch built entire SaaS businesses around this single workflow. OpenCut has all the individual pieces (highlights, reframe, captions, export) but no unified pipeline that chains them for mass extraction.
- **Implementation**: Orchestrate existing modules: transcribe → LLM extract N highlight segments (existing highlights.py, configurable count) → for each segment: trim (existing) → face-reframe to 9:16 (existing) → burn captions (existing) → export with platform preset (existing). Add: per-clip title generation via LLM, engagement score in filename, batch progress tracking. Output: folder of numbered shorts + metadata CSV (title, score, duration, topics). Panel UI: long-form file input, platform selector, clip count slider, review/edit extracted clips before export.
- **References**: Opus Clip, Vizard, Munch, Pictory, Descript clips

### 58.2 Video-to-Blog-Post Generator (P2 / M)
Generate a publish-ready blog post from video content: structured article with headings derived from topic changes, embedded key frame screenshots at relevant moments, pull quotes from the transcript, and SEO metadata (title, description, tags).
- **Why**: Repurposing video content as blog posts is a standard content strategy. Writing blog posts from transcripts is tedious manual work. LLM-powered generation with auto-extracted screenshots creates a near-complete draft in seconds.
- **Implementation**: Transcribe (existing) → LLM generates structured article with section headings, body text, and embedded image markers at key timestamps → extract frames at marked timestamps (existing preview-frame) → assemble as markdown or HTML with `![screenshot](image_N.jpg)` references → generate SEO metadata (title, meta description, keywords) via LLM. Output: markdown file + images folder, or single HTML file with embedded base64 images.
- **References**: Castmagic, Descript blog export, Pictory blog-to-video (reverse direction)

### 58.3 Social Media Caption Generator (P2 / S)
Generate platform-optimized social media post captions for each exported clip: engaging hook, relevant hashtags, call-to-action, and emoji usage calibrated per platform (LinkedIn formal, TikTok casual, Instagram visual, Twitter concise).
- **Why**: Creating the social post to accompany each video clip is a separate creative task. LLM-generated captions with platform-specific tone and formatting save time and improve consistency.
- **Implementation**: For each exported clip: extract transcript → LLM generates post caption with platform-specific formatting (character limits, hashtag conventions, emoji style). Per-platform templates: Twitter (≤280 chars, 3-5 hashtags), Instagram (hook + body + 30 hashtags in comment), LinkedIn (professional tone, no emoji), TikTok (trendy, heavy emoji). Output as JSON or text file alongside each exported clip.
- **References**: Hootsuite AI caption, Later, Buffer AI assistant

### 58.4 Podcast Episode to Multi-Platform Bundle (P1 / M)
One-click pipeline that takes a podcast recording and generates the complete distribution bundle: cleaned audio file (normalized, noise-reduced), multiple short video clips with captions, audiogram video for social, show notes (40.3), YouTube chapter markers, transcript, and RSS-compatible metadata.
- **Why**: Podcast production involves 6-8 separate post-production outputs per episode. Automating the entire bundle from a single recording eliminates hours of repetitive work. No single tool does all of these — podcasters cobble together 4-5 separate services.
- **Implementation**: Chain existing modules: denoise + normalize audio → export clean audio (MP3/WAV) → transcribe → generate chapters (existing) → extract highlights for short clips → reframe + caption each clip → generate audiogram (40.1) → generate show notes (40.3) → export transcript (SRT + TXT) → bundle all outputs in a timestamped folder. Panel UI: single upload + "Generate Podcast Bundle" button + progress for each output.
- **References**: Descript podcast workflow, Riverside export suite, Castmagic

### 58.5 Cross-Platform Content Calendar Export (P3 / M)
After extracting multiple clips from a source video, generate a content calendar spreadsheet suggesting optimal posting times and which clip goes to which platform on which day. Export as CSV, iCal, or direct integration with scheduling tools.
- **Why**: Extracting 10 clips is only half the battle — deciding when and where to post each one is the other half. A suggested calendar based on platform best practices (TikTok: 2-3 posts/day, YouTube Shorts: daily, LinkedIn: 1-2/week) helps creators maintain consistent posting schedules.
- **Implementation**: Take the list of extracted clips with scores and topics. Apply platform posting frequency best practices (configurable). Distribute clips across a calendar: highest-scored clips on peak days, space similar topics apart, respect per-platform daily limits. Export as CSV (date, platform, clip filename, caption, status) or iCal (.ics with event per post). Optional: generate Notion database entries via API (50.2).
- **References**: Later content calendar, Hootsuite planner, Buffer scheduling

---

## 59. Storyboard & Pre-Production

### 59.1 AI Storyboard Generation from Script (P2 / L)
Generate a visual storyboard from a written script or shot list: AI creates frame compositions for each scene description, laying out shots with camera angles, subject placement, and mood. Output as a printable storyboard PDF or panel of images.
- **Why**: Pre-production storyboarding is time-consuming and requires drawing skills. AI image generation can create quick visual representations of shot descriptions for planning and client communication. Saves hiring a storyboard artist for rough planning.
- **Implementation**: Parse script into individual shots/scenes (LLM extraction of scene descriptions). Generate one image per shot via Stable Diffusion or DALL-E API with prompts derived from scene description + camera angle + mood. Layout as storyboard grid: 3-4 images per row with scene number, description, dialogue, and camera direction text. Export as PDF (reportlab) or image grid (Pillow).
- **References**: Storyboarder (open-source, Wonder Unit), Boords, FrameForge

### 59.2 Shot List Generator from Screenplay (P2 / M)
Parse a screenplay (Final Draft .fdx, Fountain .fountain, or plain text) and generate a structured shot list: extract scene headings, action descriptions, dialogue blocks, and suggest camera angles (wide for establishing, close-up for dialogue, OTS for conversation). Export as spreadsheet for the shooting day.
- **Why**: Converting a screenplay into a shooting shot list is manual production assistant work. Auto-extraction + AI-suggested camera angles provides a starting point that's 80% complete, saving hours of pre-production planning.
- **Implementation**: Parse screenplay format: detect INT./EXT. scene headings (regex), ACTION blocks, CHARACTER names, DIALOGUE. For each scene: LLM suggests shot count and camera angles based on content (dialogue scenes → shot/reverse-shot pattern, action scenes → wide + close-ups). Output as CSV/JSON: scene number, shot number, description, camera angle, lens suggestion, notes. UI: editable table in panel before export.
- **References**: Final Draft shot list, StudioBinder, Celtx

### 59.3 Mood Board Generator from Footage (P3 / M)
Analyze existing footage or reference images and generate a mood board: extract dominant color palettes, identify visual styles (warm/cold, high/low contrast, saturated/muted), suggest LUTs that match, and compile as a visual reference document.
- **Why**: Mood boards establish the visual direction for a project. Generating one from existing footage or references helps communicate the intended look to clients, colorists, and team members. Auto-extraction of color palettes and style attributes eliminates manual sampling.
- **Implementation**: Extract keyframes from footage (existing scene detection). Analyze each frame: dominant colors via k-means clustering on pixels, brightness distribution (histogram), contrast ratio, saturation level. Generate mood board image: grid of keyframes + color palette swatches + style tags (Pillow composition). Suggest matching LUTs from the library by comparing histogram profiles. Export as PNG, PDF, or HTML.
- **References**: Coolors palette generator, Adobe Color, Pinterest mood boards

### 59.4 Script-to-Rough-Cut Assembly (P1 / XL)
Given a script and a folder of raw footage, automatically match footage to script segments and assemble a rough cut. Uses transcript matching (speech-to-text matched against script text), visual content analysis, and filename/metadata hints to find the best take for each scripted segment.
- **Why**: Rough cut assembly from raw footage is the most time-consuming part of post-production. A editor might spend 2-3 days selecting takes and assembling a first pass from a 4-hour shoot. Automated rough assembly provides a starting point that reduces this to review and refinement.
- **Implementation**: (1) Transcribe all raw footage (existing batch transcription). (2) Parse the script into segments. (3) For each script segment: fuzzy-match transcript text against script text (difflib.SequenceMatcher or LLM) → rank matching footage clips by similarity score + audio quality + face visibility. (4) Select best take per segment. (5) Assemble via OTIO or Premiere XML — ordered sequence matching script flow. (6) Output: rough cut timeline + match confidence report for review.
- **References**: ScriptSync (Avid), Descript's script-based editing, Simon Says assembly

---

## 60. Proxy & Media Management

### 60.1 Auto Proxy Generation (P1 / L)
Automatically generate lightweight proxy files for all project media: downsample 4K/8K footage to 720p or 1080p quarter-res proxies for smooth editing on lower-powered machines. Track the proxy↔original relationship. Auto-swap to full resolution on export.
- **Why**: 4K+ footage is standard but chokes Premiere on mid-range hardware. Premiere has built-in proxy workflows but they require manual setup per clip. Auto-proxy at ingest time — detect high-res footage and generate proxies in the background — eliminates the setup friction entirely.
- **Implementation**: Detect clips exceeding a resolution threshold (configurable, default >1080p). Generate proxy via FFmpeg: scale to target res + fast codec (H.264 CRF 28 or ProRes Proxy). Store proxy alongside original or in `~/.opencut/proxies/` with naming convention (`filename_proxy_720p.mp4`). Maintain manifest mapping proxy → original path + resolution. Background job with low priority. On export: route to original files. Panel UI: proxy status indicator per clip, "Generate Proxies" batch button.
- **References**: Premiere Pro proxy workflow, DaVinci Resolve optimized media, FCPX proxy media

### 60.2 Proxy-to-Full-Res Swap on Export (P1 / S)
Before final export, automatically detect if any clips in the timeline are using proxy files and swap them to their original full-resolution sources. Verify all originals are accessible and report any missing media before rendering begins.
- **Why**: The most common proxy workflow error is accidentally exporting with proxies still active — delivering a low-resolution final product. Auto-detection and swap eliminates this risk entirely.
- **Implementation**: Query Premiere timeline via ExtendScript: get all clip source paths. Check each against the proxy manifest. If any match a proxy filename pattern, resolve to the original path. Verify original exists and is readable. Report: list of clips with proxy status (using proxy / using original / original missing). If originals are missing: block export and show error with file paths. If all good: confirm "Exporting at full resolution" in panel.
- **References**: Premiere "Toggle Proxies" button, DaVinci Resolve render at full resolution

### 60.3 Media Relinking Assistant (P2 / M)
When project media goes offline (moved, renamed, external drive disconnected), help relink by searching for matching files: match by filename, file size, duration, codec, and creation date. Search user-specified directories and common media locations. Batch relink multiple missing clips at once.
- **Why**: "Media offline" is one of the most frustrating Premiere errors, especially when projects move between drives or machines. Premiere's relinking UI searches one file at a time. Batch relinking with smart matching is faster and more reliable.
- **Implementation**: ExtendScript: enumerate offline project items (items where mediaPath doesn't exist). For each: extract filename, expected size, duration from project item metadata. Python: search specified directories recursively for files matching by name (exact → fuzzy) and size (±1%). Rank candidates by match confidence. UI: table of offline items with suggested matches and confidence scores. "Relink All" applies matches above threshold, "Review" for low-confidence matches.
- **References**: Premiere relink media, DaVinci Resolve media management, Kyno media manager

### 60.4 Duplicate Media Detection (P2 / M)
Scan project media for duplicate files: exact duplicates (hash match), near-duplicates (same content, different encoding), and derivative duplicates (same source at different resolutions). Report disk space wasted and offer to consolidate.
- **Why**: Projects accumulate duplicate media over time: re-imported files, proxy/original pairs, multiple transcodes of the same source, and test exports. Identifying and cleaning duplicates can reclaim significant disk space.
- **Implementation**: Phase 1: Fast duplicate check via file size grouping → partial hash (first 64KB + last 64KB) → full hash for size-matched candidates. Phase 2: Content comparison for near-duplicates via perceptual hash (pHash) on extracted keyframes — catches same content at different resolutions/codecs. Report: groups of duplicates with sizes, paths, and which to keep. UI: review duplicate groups, select keeper, remove/archive others.
- **References**: rmlint (open-source), DupeGuru, Beyond Compare

---

## 61. Composition & Framing Intelligence

### 61.1 Rule-of-Thirds & Composition Guide Overlay (P2 / S)
Overlay composition guides on the preview frame: rule-of-thirds grid, golden ratio spiral, diagonal lines, center cross, and custom safe areas. Helps evaluate framing during reframe, crop, and PiP operations.
- **Why**: Every camera and photo editor has composition overlays. OpenCut's reframe and crop tools operate blind — users can't see composition guides while adjusting. A simple overlay makes framing decisions faster and more deliberate.
- **Implementation**: Draw guide lines on extracted preview frames via Pillow `ImageDraw` (thin semi-transparent lines). Guides are display-only — not burned into output. Toggle between guide types. For reframe operations: show guides on the crop preview so users can align subjects to thirds/golden points. Include broadcast safe areas (title safe 80%, action safe 90%).
- **References**: Camera viewfinder overlays, Lightroom crop overlay, Premiere safe margins

### 61.2 Shot Type Auto-Classification (P1 / M)
Automatically classify each shot in a video by type: extreme wide, wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, insert/cutaway, aerial, POV. Uses face size relative to frame, subject count, and scene content analysis.
- **Why**: Shot type metadata enables intelligent editing decisions: cut patterns should vary by shot type (don't cut from close-up to close-up), pacing analysis needs shot variety metrics, and search/organization benefits from shot type tags. Google Video Intelligence API offers this but costs money.
- **Implementation**: Detect faces (existing MediaPipe/InsightFace). Classify by face-to-frame ratio: face >50% of frame height = ECU, 30-50% = CU, 15-30% = MCU, 5-15% = MS, <5% = WS. No face detected: classify by scene complexity (edge density, depth variance). Multiple faces: two-shot, group shot. Apply to scene detection results — each scene gets a shot type tag. Store in footage index for search.
- **References**: Google Video Intelligence shot detection, Clarifai, FilmGrab shot classification

### 61.3 Intelligent Pacing Analysis (P2 / M)
Analyze the edit rhythm and pacing of a video: average shot length, shot length variance, pacing curve (fast/slow segments over time), and comparison against genre benchmarks (music video: 1.5s avg, documentary: 8s avg, commercial: 2s avg). Flag pacing anomalies.
- **Why**: Pacing is one of the hardest things to evaluate while editing — you lose perspective after watching a sequence repeatedly. An objective pacing analysis with visual rhythm map and genre benchmarks gives editors an external perspective on their edit.
- **Implementation**: Use scene detection (existing) to get cut points. Calculate: mean shot length, median, std dev, longest/shortest shots, shots-per-minute curve over time. Compare against configurable genre benchmarks (stored as JSON presets). Generate pacing visualization: horizontal timeline with shot lengths as bars (tall = long, short = quick). Color-code: green = within genre range, yellow = slightly off, red = significant deviation. Flag: "3 consecutive shots over 15s at 8:30-9:15 — consider tightening" or "rapid-fire cuts at 2:00-2:15 may cause viewer fatigue."
- **References**: Every Frame a Painting (editing analysis), EditBench, Premiere sequence statistics

### 61.4 Saliency-Guided Auto-Crop (P2 / M)
When cropping or reframing video, use visual saliency detection (where the viewer's eye is naturally drawn) to position the crop window optimally. Combines face detection, text detection, motion regions, high-contrast areas, and semantic importance into a unified saliency map.
- **Why**: OpenCut's existing reframe centers on detected faces — but many clips don't have faces (product shots, landscapes, screen recordings, graphics). Saliency-based framing handles any content type. Google's AutoFlip and Meta's AI crop use this approach.
- **Implementation**: Generate saliency map per frame: face regions (high weight, existing) + motion regions (frame differencing, existing) + text regions (OCR, high weight) + high-contrast edges + center bias. Weighted combination produces heat map. Place crop window to maximize saliency coverage. Smooth crop path over time (existing face_reframe smoothing). Fallback: center crop when saliency is uniform.
- **References**: Google AutoFlip, DeepGaze (saliency model), Meta Smart Crop, OpenCV saliency module

---

## 62. AI Dubbing & Voice Translation

### 62.1 End-to-End AI Dubbing Pipeline (P1 / XL)
Translate a video's dialogue into another language with AI-generated dubbed audio that matches the original speaker's voice, timing, and emotion. Full pipeline: transcribe → translate → voice-clone TTS in target language → lip-sync adjustment → audio mix. One-click dubbing into any of 50+ languages.
- **Why**: ElevenLabs, HeyGen, and Rask.ai charge $49-$99/month for AI dubbing. The global content market demands multi-language versions. OpenCut has every individual component (transcription, translation, voice cloning via Chatterbox, lip sync in roadmap) — chaining them into a unified dubbing pipeline makes it free and local.
- **Implementation**: Pipeline: (1) Transcribe with word timestamps (existing). (2) Translate segments to target language (existing deep-translator/NLLB/SeamlessM4T). (3) Clone original speaker's voice via Chatterbox (existing) or Kokoro with voice sample. (4) Generate dubbed audio per segment with timing constraints (match original duration ±10%). (5) Mix dubbed audio: remove original dialogue via stem separation (existing Demucs), retain music/SFX, overlay dubbed dialogue. (6) Optional: apply lip-sync correction (1.4) to match new audio. (7) Export final dubbed video.
- **References**: ElevenLabs Dubbing, HeyGen video translate, Rask.ai, Meta SeamlessM4T

### 62.2 Isochronous Translation (P1 / L)
Translate dialogue such that the translated text fits within the same time duration as the original speech. The translated version is spoken at natural speed, not sped up or slowed down — achieved by adjusting the translation itself (shorter phrasing, synonym selection) rather than the audio speed.
- **Why**: Simple translation + TTS produces audio segments of different lengths than the original, causing sync issues. Dubbing studios solve this with "isochronous translation" — a translation that naturally fills the same time window. LLM-assisted translation can be constrained by duration, dramatically improving dubbing quality.
- **Implementation**: For each dialogue segment: (1) Measure original speech duration. (2) Translate to target language. (3) Estimate target TTS duration from character/syllable count. (4) If too long: ask LLM to rephrase shorter ("Say the same thing in fewer words, aim for N syllables"). (5) If too short: ask LLM to expand ("Elaborate slightly to fill N seconds"). (6) Iterate until within ±10% of original duration. (7) Generate TTS at natural speed. This replaces the common hack of speeding up/slowing down dubbed audio.
- **References**: Netflix dubbing guidelines, ZOO Digital isochronous translation, ElevenLabs timing control

### 62.3 Multi-Language Audio Track Management (P2 / M)
Manage multiple language audio tracks within a single video file or project: add, remove, label, and set default language tracks. Export with embedded multi-language audio (MKV, MP4 with multiple audio streams) or as separate per-language files for platform upload.
- **Why**: International distribution requires multiple audio tracks per video. YouTube, Netflix, and most streaming platforms support multi-language audio. Managing these tracks is currently a manual FFmpeg operation. A panel-integrated manager simplifies the workflow.
- **Implementation**: FFmpeg `-map` to mux multiple audio streams with language metadata (`-metadata:s:a:0 language=eng`, `-metadata:s:a:1 language=spa`). Panel UI: audio track list with language dropdown (ISO 639-1 codes), default flag, and add/remove buttons. Import: add audio file as new language track. Export: single multi-track container (MKV preferred, MP4 for compatibility) or per-language separate files. Display active track count in clip metadata.
- **References**: FFmpeg multi-stream muxing, MKVToolNix, YouTube multi-audio upload, Netflix audio manifest

### 62.4 Voice Translation with Emotion Preservation (P2 / L)
Analyze the emotional prosody (pitch contour, speaking rate, emphasis, pauses) of the original speaker and transfer those characteristics to the dubbed audio in the target language. Preserve the emotional arc — excitement, sadness, anger, emphasis — across the language boundary.
- **Why**: Flat, monotone dubbing is the primary complaint about AI-generated voiceover. The technical accuracy of translation means nothing if the emotional delivery doesn't match. Preserving prosody makes dubbed content feel authentic rather than robotic.
- **Implementation**: (1) Extract prosody features from original audio: F0 (pitch) contour via librosa, speaking rate per segment, RMS energy envelope, pause locations and durations. (2) Generate dubbed TTS with neutral prosody. (3) Transfer prosody: modify F0 contour of TTS output to match original's pitch shape (relative, not absolute — account for language pitch differences), adjust speaking rate per segment, match energy envelope. Use WORLD vocoder or Praat for pitch manipulation. (4) Validate: compare prosody similarity metrics before/after transfer.
- **References**: ElevenLabs emotion control, Meta Expressive TTS, Microsoft VALL-E X, Coqui TTS prosody transfer

---

## Expanded Priority Summary

### P0 — Critical (4 total)
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 7.1 | FastAPI Migration | XL | Backend |
| 7.2 | Process Isolation for GPU Workers | XL | Backend |
| 1.1 | Transcript-Based Editing | XL | AI Editing |
| 32.1 | Hardware-Accelerated Encoding | M | Performance |

### P1 — High Impact (75 total)
All original P1 items plus:
| # | Feature | Effort | Category |
|---|---------|--------|----------|
| 1.11 | AI B-Roll Suggestion & Insertion | L | AI Editing |
| 2.8 | Audio Fingerprinting & Copyright Detection | M | Audio |
| 3.13 | Morph Cut / Smooth Jump Cut | L | Video |
| 23.1 | Proxy Generation Pipeline | L | Media Management |
| 23.2 | AI Metadata Enrichment | L | Media Management |
| 24.1 | Shot-Change-Aware Subtitle Timing | M | Subtitling |
| 25.2 | Room Tone Matching & Generation | M | Audio Post |
| 26.1 | Kinetic Typography Engine | L | Motion Design |
| 28.1 | Black Frame & Frozen Frame Detection | S | QA/QC |
| 28.2 | Audio Phase & Silence Gap Check | S | QA/QC |
| 29.1 | Shot Type Auto-Classification | M | Analytics |
| 32.3 | Background Render Queue | M | Performance |
| 34.1 | Scrolling Credits Generator | M | Graphics |
| 35.1 | Selective Redaction Tool | M | Forensic |
| 36.1 | Platform Safe Zone Overlay | S | Social |
| 36.4 | Vertical-First Intelligent Reframe | M | Social |
| 39.1 | Elgato Stream Deck Integration | M | Hardware |
| 40.1 | Audiogram Generator for Social Media | M | Podcast |
| 40.3 | Video Podcast to Audio-Only Extraction | S | Podcast |
| 41.1 | Telemetry Data Overlay from SRT/CSV | M | Drone |
| 42.1 | Image Sequence Import & Assembly | M | Timelapse |
| 42.2 | Timelapse Deflicker | M | Timelapse |
| 43.1 | ACES Color Pipeline | L | Color Mgmt |
| 43.4 | Color Space Auto-Detection & Conversion | M | Color Mgmt |
| 44.1 | Timecode Burn-In Overlay | S | Timecode |
| 45.1 | ProRes Export on Windows | M | Encoding |
| 45.2 | AV1 Encoding Support | M | Encoding |
| 46.1 | Multi-Step Autonomous Editing Agent | XL | AI Agent |
| 47.1 | Batch Transcode with Preset Matrix | M | Batch Ops |
| 52.1 | Lens Distortion Correction | M | Camera Correction |
| 53.1 | Corrupted File Recovery | M | Video Repair |
| 53.2 | Adaptive Deinterlacing | S | Video Repair |
| 55.1 | License Plate Detection & Blur | M | Privacy |
| 55.3 | Profanity Bleep Automation | S | Privacy |
| 56.3 | AI Environmental Noise Classifier | M | Spectral Audio |
| 56.4 | Room Tone Auto-Generation & Matching | M | Spectral Audio |
| 57.1 | Split-Screen Layout Templates | M | Multi-View |
| 57.2 | Reaction Video Template | M | Multi-View |
| 58.1 | Long-Form to Multi-Short Extraction | L | Repurposing |
| 58.4 | Podcast Episode to Multi-Platform Bundle | M | Repurposing |
| 59.4 | Script-to-Rough-Cut Assembly | XL | Pre-Production |
| 60.1 | Auto Proxy Generation | L | Media Mgmt |
| 60.2 | Proxy-to-Full-Res Swap on Export | S | Media Mgmt |
| 61.2 | Shot Type Auto-Classification | M | Composition |
| 62.1 | End-to-End AI Dubbing Pipeline | XL | Dubbing |
| 62.2 | Isochronous Translation | L | Dubbing |

### P2 — Valuable (155 total)
All original P2 items plus new sections' P2 features (1.12, 3.11, 3.12, 5.6, 6.8, 18.7, 21.7, 39.2, 40.2, 40.4, 41.2, 41.3, 42.3, 43.2, 44.2, 44.3, 45.3, 45.4, 45.5, 46.2, 46.3, 46.4, 47.2, 47.3, 47.4, 47.5, 48.1, 48.3, 48.4, 49.1, 50.1, 50.3, 50.5, 51.1, 51.2, 51.3, 52.2, 52.4, 53.3, 53.5, 54.1, 54.2, 54.4, 54.5, 55.2, 55.4, 56.1, 56.2, 57.3, 57.4, 58.2, 58.3, 59.1, 59.2, 60.3, 60.4, 61.1, 61.3, 61.4, 62.3, 62.4).

### P3 — Future (72 total)
All original P3 items plus new sections' P3 features (39.3, 39.4, 41.4, 42.4, 42.5, 43.3, 44.4, 48.2, 49.2, 49.3, 49.4, 50.2, 50.4, 51.4, 52.3, 53.4, 54.3, 55.5, 58.5, 59.3).

---

## Competitive Gap Analysis

Features that every major competitor has but OpenCut lacks:

| Feature | Who Has It | OpenCut Status |
|---------|-----------|----------------|
| Transcript-based editing | Descript, Kapwing, CapCut | **Missing** (1.1) |
| Eye contact correction | Descript, CapCut, NVIDIA | **Missing** (1.2) |
| Lip sync for dubbing | ElevenLabs, CapCut, HeyGen | **Missing** (1.4) |
| Hardware GPU encoding | Handbrake, Media Encoder, Resolve | **Missing** (32.1) |
| Watch folder automation | Media Encoder, Handbrake | **Missing** (4.1) |
| Render queue | Media Encoder, Resolve, Handbrake | **Missing** (4.2) |
| Before/after preview | Topaz, Lightroom, every photo editor | **Missing** (6.3) |
| Multi-cam audio sync | PluralEyes, Premiere, Resolve | **Missing** (3.9) |
| Best take selection | AutoCut, Descript | **Missing** (4.8) |
| Context menu integration | AutoCut, Excalibur | **Missing** (6.6) |
| Model download manager | Ollama, SD WebUI, ComfyUI | **Missing** (10.1) |
| Color scopes | DaVinci Resolve, Premiere, FCPX | **Missing** (13.1) |
| Paper edit | Descript, Simon Says, Avid | **Missing** (14.1) |
| Beat-synced editing | CapCut, GoPro Quik | **Missing** (16.1) |
| Stock media search | Storyblocks plugin, Envato plugin | **Missing** (19.1) |
| Caption compliance | Netflix QC, broadcast tools | **Missing** (20.1) |
| Seizure detection | Broadcast QC tools, Harding FPA | **Missing** (20.4) |
| Retro/VHS effects | CapCut, Premiere presets | **Missing** (18.2) |
| Lower-thirds from data | AutoPod, Riverside | **Missing** (15.2) |
| Auto zoom-to-cursor | Loom, Tella, Screen Studio | **Missing** (11.1) |
| AI rough cut assembly | Opus Clip, Vizard, CapCut | **Missing** (21.3) |
| Proxy workflow | Premiere, Resolve, FCPX | **Missing** (23.1) |
| Shot type classification | Clarifai, Video Intelligence | **Missing** (29.1) |
| Black/frozen frame QC | Vidchecker, Baton, every QC tool | **Missing** (28.1) |
| Selective redaction | Premiere, Axon Evidence | **Missing** (35.1) |
| Platform safe zones | CapCut, Canva Video | **Missing** (36.1) |
| Smart vertical reframe | CapCut, OpusClip, AutoFlip | **Missing** (36.4) |
| GIF export | ScreenToGif, every video tool | **Missing** (38.1) |
| Kinetic typography | After Effects, CapCut, Motion | **Missing** (26.1) |
| Scrolling credits | Premiere, Resolve, FCPX | **Missing** (34.1) |
| Morph cut / smooth jump cut | Premiere Pro, FCPX | **Missing** (3.13) |
| AI B-roll suggestion | Descript, Pictory AI | **Missing** (1.11) |
| Audio copyright detection | YouTube Content ID, Audible Magic | **Missing** (2.8) |
| Stream Deck integration | AutoCut, Premiere plugins | **Missing** (39.1) |
| Audiogram generator | Headliner, Wavve, Descript | **Missing** (40.1) |
| DJI telemetry overlay | DashWare, Race Render | **Missing** (41.1) |
| Timelapse deflicker | LRTimelapse, GBTimelapse | **Missing** (42.2) |
| ACES color management | DaVinci Resolve, Baselight | **Missing** (43.1) |
| Timecode burn-in | Premiere, Resolve, every NLE | **Missing** (44.1) |
| ProRes export (Windows) | Resolve, Telestream Switch | **Missing** (45.1) |
| AV1 encoding | HandBrake, FFmpeg, Resolve | **Missing** (45.2) |
| Batch transcode queue | Media Encoder, HandBrake, Compressor | **Missing** (47.1) |
| Image sequence import | Resolve, After Effects, every NLE | **Missing** (42.1) |
| AI editing agent | Descript AI, Runway, Adobe Firefly | **Missing** (46.1) |
| Lens distortion correction | DaVinci Resolve, Premiere, lensfun | **Missing** (52.1) |
| Rolling shutter correction | DaVinci Resolve, Premiere, Gyroflow | **Missing** (52.2) |
| Deinterlacing | HandBrake, Resolve, every NLE | **Missing** (53.2) |
| Corrupted file recovery | Stellar, untrunc | **Missing** (53.1) |
| Split-screen templates | CapCut, iMovie, Premiere | **Missing** (57.1) |
| Reaction video layout | StreamYard, CapCut | **Missing** (57.2) |
| License plate blur | YouTube, Premiere, Resolve | **Missing** (55.1) |
| Profanity bleep | Broadcast tools, Cleanfeed | **Missing** (55.3) |
| Spectral audio editing | iZotope RX, Audacity, Adobe Audition | **Missing** (56.1) |
| Room tone fill | iZotope RX, Premiere Auto-Fill | **Missing** (56.4) |
| Long-form to shorts pipeline | Opus Clip, Vizard, Munch | **Missing** (58.1) |
| Auto proxy generation | Premiere, Resolve, FCPX | **Missing** (60.1) |
| AI dubbing | ElevenLabs, HeyGen, Rask.ai | **Missing** (62.1) |
| Saliency-guided crop | Google AutoFlip, Meta Smart Crop | **Missing** (61.4) |
| Pacing analysis | EditBench, ScreenLight | **Missing** (61.3) |

---

## Feature Count Summary

| Section | Category | Features |
|---------|----------|----------|
| 1 | AI-Powered Editing | 12 |
| 2 | Advanced Audio | 8 |
| 3 | Advanced Video | 13 |
| 4 | Workflow & Automation | 8 |
| 5 | Export & Distribution | 6 |
| 6 | Panel UX & Interface | 8 |
| 7 | Backend & Infrastructure | 7 |
| 8 | Collaboration | 3 |
| 9 | Platform Expansion | 5 |
| 10 | Model & AI Infrastructure | 4 |
| 11 | Screen Recording & Tutorial | 5 |
| 12 | Gaming & Streaming | 4 |
| 13 | Pro Color Science | 6 |
| 14 | Documentary & Interview | 4 |
| 15 | Corporate & Brand Video | 3 |
| 16 | Music Video & Audio-Reactive | 4 |
| 17 | VFX & Compositing | 4 |
| 18 | Creative Effects Pack | 7 |
| 19 | Stock Media & Assets | 4 |
| 20 | Accessibility & Compliance | 4 |
| 21 | Emerging AI & Future Tech | 7 |
| 22 | Developer & Scripting | 3 |
| 23 | Media Asset Management | 5 |
| 24 | Professional Subtitling & Localization | 5 |
| 25 | Audio Post-Production (Film/TV) | 5 |
| 26 | Motion Design & Animation | 4 |
| 27 | AI Safety & Content Authenticity | 4 |
| 28 | Automated QA / QC | 5 |
| 29 | Video Intelligence & Analytics | 4 |
| 30 | Advanced Timeline Operations | 5 |
| 31 | Live Production & Streaming | 4 |
| 32 | Performance & Rendering | 5 |
| 33 | Education & E-Learning | 3 |
| 34 | Text & Graphics Overlays | 4 |
| 35 | Forensic & Legal Video | 3 |
| 36 | Social Media Optimization | 4 |
| 37 | Quality of Life & UX Polish | 5 |
| 38 | Multi-Format Delivery | 4 |
| 39 | Hardware Integration & Control Surfaces | 4 |
| 40 | Podcast Visual Production | 4 |
| 41 | Drone & Aerial Post-Production | 4 |
| 42 | Timelapse & Image Sequence | 5 |
| 43 | Color Management Pipeline | 4 |
| 44 | Timecode & Professional Sync | 4 |
| 45 | Advanced Encoding & Delivery Codecs | 5 |
| 46 | AI Editing Agent & Intelligence | 4 |
| 47 | Batch Operations & Mass Processing | 5 |
| 48 | Event & Ceremony Video | 4 |
| 49 | Version Control & Edit History | 4 |
| 50 | Third-Party Integrations | 5 |
| 51 | 360° / VR / Immersive Video | 4 |
| 52 | Camera & Lens Correction | 4 |
| 53 | Video Repair & Restoration | 5 |
| 54 | AI Video Generation & Synthesis | 5 |
| 55 | Privacy & Content Redaction | 5 |
| 56 | Spectral Audio Editing | 4 |
| 57 | Split-Screen & Multi-View | 4 |
| 58 | Content Repurposing Pipeline | 5 |
| 59 | Storyboard & Pre-Production | 4 |
| 60 | Proxy & Media Management | 4 |
| 61 | Composition & Framing Intelligence | 4 |
| 62 | AI Dubbing & Voice Translation | 4 |
| 63 | Competitive Media Enhancement | 5 |
| 64 | Social & Content Optimization | 5 |
| 65 | Next-Gen AI Features | 5 |
| 66 | Advanced VFX & Motion | 6 |
| 67 | UX Intelligence & Discovery | 4 |
| 68 | AI Timeline Intelligence | 5 |
| 69 | Advanced Object AI | 5 |
| 70 | Delivery & Mastering | 5 |
| 71 | AI Content Generation | 5 |
| 72 | Pipeline Intelligence | 5 |
| **Total** | | **352** |

---

## 63. Competitive Media Enhancement

### 63.1 AI Enhanced Speech Restoration (P0 / M)
Restore badly recorded dialogue — not just denoise, but reconstruct clarity, extend bandwidth, and normalize. Uses Resemble Enhance or DeepFilterNet with FFmpeg fallback. Three modes: denoise_only, enhance, full.
- **Why**: Adobe ships this natively. Podcast/interview creators need studio-quality from phone recordings.
- **Module**: `enhanced_speech.py` | **Routes**: `/audio/enhance-speech`, `/audio/enhance-speech/preview`

### 63.2 One-Click Enhance Pipeline (P1 / M)
Single button: auto-detect issues and apply upscale + denoise + color correct + stabilize. Three presets: fast, balanced, quality. Skips steps that aren't needed.
- **Why**: CapCut's most popular feature. Reduces 5-step workflow to one click.
- **Module**: `one_click_enhance.py` | **Route**: `/video/one-click-enhance`

### 63.3 Low-Light Video Enhancement (P1 / M)
Beyond denoising — actual light recovery for dark footage using FFmpeg curves for shadow lift + midtone boost + highlight protection, followed by detail recovery via unsharp mask and optional nlmeans denoise.
- **Why**: Topaz Nyx model does this at AI level. Essential for indoor/night footage.
- **Module**: `low_light.py` | **Routes**: `/video/enhance-low-light`, `/video/enhance-low-light/preview`

### 63.4 AI Scene Edit Detection (P1 / S)
Detect cuts in pre-edited footage (from clients, stock). Uses FFmpeg scdet filter + blackdetect + validation. Classifies as hard_cut, dissolve, or fade.
- **Why**: Adobe added this. Essential for reconforming received edits.
- **Module**: `scene_edit_detect.py` | **Route**: `/video/detect-edits`

### 63.5 AI Face Restoration (P2 / L)
Restore and enhance faces in video. Per-frame face detection via MediaPipe/retinaface, selective sharpening + denoising targeted at face regions.
- **Why**: Old/compressed footage needs face enhancement without affecting backgrounds.
- **Module**: `face_restore.py` | **Routes**: `/video/face-restore`, `/video/face-restore/detect`, `/video/face-restore/preview`

## 64. Social & Content Optimization

### 64.1 Engagement/Retention Prediction (P0 / M)
Predict where viewers will drop off. Analyzes hook strength, pacing, audio energy, visual variety. Generates per-5-second retention curve with drop-off point detection.
- **Why**: Opus Clip and YouTube analytics prove this drives editing decisions.
- **Module**: `engagement_predict.py` | **Route**: `/content/predict-engagement`

### 64.2 Animated Caption Style Library (P0 / M)
55+ animated word-by-word caption styles across 10 categories. Includes TikTok Bold, YouTube Clean, Karaoke Highlight, Neon Glow, etc. FFmpeg drawtext filter chains for each animation type.
- **Why**: CapCut and Captions.ai dominate here. Creators demand styled captions.
- **Module**: `caption_styles.py` | **Routes**: `/content/caption-styles` (GET/POST), `/content/caption-styles/preview`

### 64.3 AI Hook Generator (P1 / M)
Generate and insert a compelling opening hook in the first 3 seconds. Five hook types: question, statistic, bold_claim, teaser, quote. LLM integration with heuristic fallback.
- **Why**: Opus Clip popularized this. First 3 seconds determine viewer retention.
- **Module**: `hook_generator.py` | **Routes**: `/content/hook/generate`, `/content/hook/text-only`

### 64.4 A/B Variant Generator (P1 / M)
Generate 3-5 versions of the same clip with different hooks, caption styles, pacing, thumbnails. Test on platforms to optimize engagement.
- **Why**: Opus Clip popularized this. Data-driven content optimization.
- **Module**: `ab_variant.py` | **Route**: `/content/ab-variants`

### 64.5 Essential Graphics Caption Output (P1 / M)
Export captions as Premiere Essential Graphics (XML/JSON) instead of burned-in overlays. Supports Premiere XML, SRT, JSON for CEP panel, After Effects JSON.
- **Why**: AutoCut does this. Enables editor to reposition/restyle in Premiere.
- **Module**: `essential_graphics.py` | **Routes**: `/content/essential-graphics/export`, `/content/essential-graphics/premiere-xml`

## 65. Next-Gen AI Features

### 65.1 Video LLM Integration (P1 / L)
Use multimodal LLMs for direct video understanding: "find the funniest moment", "what's happening at 2:34". Supports OpenAI/Anthropic vision APIs and local LLM backends.
- **Why**: Goes beyond transcript search to visual understanding.
- **Module**: `video_llm.py` | **Routes**: `/ai/video-llm/query`, `/ai/video-llm/find-moment`

### 65.2 AI Music Remix / Duration Fit (P1 / M)
Automatically adjust background music duration to match video length. Three modes: smart (beat-aware looping/cutting), stretch (atempo), fade. Optional librosa for beat detection.
- **Why**: DaVinci Resolve added this. Music never fits video length perfectly.
- **Module**: `music_remix.py` | **Routes**: `/ai/music/remix`, `/ai/music/fit-duration`

### 65.3 Audio Category Tagging (P2 / M)
Auto-classify timeline audio as speech/music/SFX/ambience/silence using FFmpeg astats + optional spectral analysis. Enables smart ducking and stem-aware processing.
- **Why**: Adobe Sensei does this. Foundation for intelligent audio workflows.
- **Module**: `audio_category.py` | **Routes**: `/ai/audio/classify`, `/ai/audio/classify-timeline`

### 65.4 AI Color Match Between Shots (P2 / M)
Auto-match color/exposure between different shots for consistent look. Uses FFmpeg signalstats for analysis, eq + colorbalance filters for matching. Supports batch matching.
- **Why**: Adobe Auto Tone does this. Essential for multi-camera/multi-day shoots.
- **Module**: `color_match_shots.py` | **Routes**: `/ai/color-match`, `/ai/color-match/batch`, `/ai/color-match/analyze`

### 65.5 Consistent Character Generation (P2 / L)
Generate the same character across multiple AI-generated shots. Creates character profiles from reference frames using face/body embeddings. InsightFace + CLIP backends.
- **Why**: Runway Frames + IP-Adapter enable this. Critical for AI-generated narratives.
- **Module**: `character_consistency.py` | **Routes**: `/ai/character/create`, `/ai/character/list`, `/ai/character/generate`

## 66. Advanced VFX & Motion

### 66.1 Generative Extend (P0 / L)
AI extends clips beyond their recorded length. Three-tier fallback: video generation model → optical flow extrapolation → freeze-frame with Ken Burns. Audio extension via room tone loop.
- **Why**: Adobe Firefly does this in Premiere. Game-changer for short clips.
- **Module**: `generative_extend.py` | **Route**: `/video/generative-extend`

### 66.2 Green-Screen-Free Background Replacement (P1 / L)
AI matting without green screen. Three segmentation backends: SAM2, rembg, MediaPipe. Supports image, video, color, blur, and transparent backgrounds. Edge refinement + temporal smoothing.
- **Why**: Descript and CapCut both do this. Removes need for physical green screen.
- **Module**: `greenscreen_free.py` | **Routes**: `/video/replace-background`, `/video/replace-background/preview`

### 66.3 Motion Brush / Cinemagraph+ (P2 / M)
Paint where motion should happen on a still image/video. 8 motion directions with strength control. Still images get Ken Burns animation; videos get per-region overlay transforms.
- **Why**: Runway popularized this. Extends existing cinemagraph feature.
- **Module**: `motion_brush.py` | **Routes**: `/video/motion-brush`, `/video/motion-brush/preview`

### 66.4 Body/Pose-Driven Effects (P2 / L)
Track body keypoints via MediaPipe Pose, apply effects that follow body parts. Six effect types: glow, trail, highlight, blur_except, neon_outline, particle_follow.
- **Why**: CapCut does this. Popular for dance/fitness content.
- **Module**: `body_effects.py` | **Routes**: `/video/body-effects`, `/video/body-keypoints`

### 66.5 Full-Body Motion Transfer (P2 / L)
Transfer motion from a reference video to a target person. Pipeline: extract poses → AI model (AnimateAnyone/MimicMotion) → stick-figure fallback → assemble via FFmpeg.
- **Why**: AnimateAnyone 2 / MimicMotion enable this. Opens creative possibilities.
- **Module**: `motion_transfer.py` | **Routes**: `/ai/motion-transfer`, `/ai/motion-transfer/extract-poses`

### 66.6 AI Foley Sound Generation (P2 / L)
Generate sound effects from video content. Detects motion events via frame differencing, classifies into 10 foley categories, synthesizes audio via PCM synthesis, mixes with original.
- **Why**: HunyuanVideo-Foley and similar models make this feasible. Saves hours of manual SFX work.
- **Module**: `foley_gen.py` | **Routes**: `/audio/generate-foley`, `/video/detect-events`

## 67. UX Intelligence & Discovery

### 67.1 Command Palette / Spotlight Search (P0 / M)
Cmd+K instant search across all 327 features. 215 curated feature entries with fuzzy matching, recent tracking, and categorized results. Type 3 letters, hit Enter.
- **Why**: With 327 features, users need instant discovery. Every modern tool has this.
- **Module**: `command_palette.py` | **Routes**: `/ux/search`, `/ux/search/record`, `/ux/recents`, `/ux/feature-index`

### 67.2 AI Contextual Suggestions (P0 / M)
Analyze the current clip and suggest relevant operations. 17+ rules covering loudness, captions, stabilization, upscaling, denoising, content-type-specific features. Suppresses after dismissal.
- **Why**: Surfaces the right feature at the right moment. Users discover features they didn't know existed.
- **Module**: `contextual_suggest.py` | **Route**: `/ux/suggest`

### 67.3 Smart Defaults Engine (P2 / M)
Auto-detect clip characteristics (interview, music video, screencast, drone, etc.) and pre-fill optimal parameters. 10 content types, 10+ operation types with content-aware defaults.
- **Why**: Eliminates parameter guessing. Novices get expert settings automatically.
- **Module**: `smart_defaults.py` | **Route**: `/ux/smart-defaults`

### 67.4 Unified Preview System (P1 / M)
Before/after preview for ANY processing operation. Extract one frame, apply the operation, show side-by-side comparison. 8 supported operations.
- **Why**: Users currently process entire clips blind. Preview eliminates guess-and-check.
- **Module**: `preview_frame.py` | **Route**: `/ux/preview`

---

## 68. AI Timeline Intelligence

### 68.1 Holistic Timeline Quality Analysis (P1 / M)
Analyze entire timeline for color consistency, audio level consistency, pacing, and continuity issues. Generates overall quality score (0-100) with actionable recommendations.
- **Why**: No tool analyzes timeline holistically. Catches issues before delivery.
- **Module**: `timeline_quality.py` | **Route**: `/timeline/quality`

### 68.2 Per-Segment Engagement Scoring (P1 / M)
Divide video into segments and score each for visual interest, audio engagement, pacing effectiveness, and content diversity. Generates engagement curve.
- **Why**: Extends engagement prediction to full timeline analysis.
- **Module**: `timeline_score.py` | **Route**: `/timeline/score`

### 68.3 AI Narrative Clip Chaining (P1 / L)
Given multiple clips, analyze content and order them for story flow. 8 narrative styles (documentary, vlog, commercial, etc.). Suggests transitions and generates assembly cut list.
- **Why**: Automates rough cut assembly from unordered clips.
- **Module**: `clip_narrative.py` | **Route**: `/timeline/narrative`

### 68.4 Natural Language Color Grading (P1 / M)
Describe desired color grade in words ("warm sunset", "noir", "cyberpunk"). 32 named intents mapped to FFmpeg filter chains. LLM fallback for custom descriptions.
- **Why**: Makes color grading accessible to non-colorists.
- **Module**: `ai_color_intent.py` | **Routes**: `/video/color-intent`, `/video/color-intents`, `/video/color-intent/preview`

### 68.5 End-to-End Auto-Dubbing Pipeline (P0 / L)
Full dubbing pipeline: transcribe → translate → voice clone → generate audio → lip sync → composite. Supports 47 languages with weighted stage progress.
- **Why**: Closes the biggest gap vs ElevenLabs/HeyGen dubbing workflows.
- **Module**: `auto_dub_pipeline.py` | **Routes**: `/audio/auto-dub`, `/audio/auto-dub/languages`

## 69. Advanced Object AI

### 69.1 Text-Driven Video Segmentation (P1 / L)
Describe what to segment in natural language ("the red car", "the person in blue"). CLIP embeddings find matching regions, SAM2 refines masks, tracks across frames.
- **Why**: SAMWISE-style natural language segmentation is the next frontier.
- **Module**: `text_segment.py` | **Routes**: `/video/text-segment`, `/video/text-segment/preview`

### 69.2 Physics-Aware Object Removal (P1 / L)
Remove objects AND their shadows and reflections. Detects shadow direction, reflection surfaces, removes all together, inpaints with temporal consistency.
- **Why**: Netflix VOID model demonstrates this is now expected quality.
- **Module**: `physics_remove.py` | **Routes**: `/video/physics-remove`, `/video/physics-remove/detect-shadow`

### 69.3 Object Tracking with Graphic Overlays (P1 / M)
Track objects and attach overlays that follow them: text labels, arrows, blur, highlight, crosshair, spotlight, and 4 more. JSON track data export.
- **Why**: Essential for tutorials, sports analysis, and privacy workflows.
- **Module**: `object_track_overlay.py` | **Routes**: `/video/track-overlay`, `/video/overlay-types`

### 69.4 Semantic Video Search (P2 / M)
CLIP-based visual search across project clips. Search by text ("person laughing") or image similarity. Cached embeddings for fast re-query.
- **Why**: Goes beyond transcript search to visual content discovery.
- **Module**: `semantic_video_search.py` | **Routes**: `/search/semantic`, `/search/semantic/index`

### 69.5 Multi-Subject Intelligent Reframe (P1 / M)
Detect all subjects (faces, objects, text), score importance, choose optimal crop. Split-screen fallback when subjects are distant. 9 preset aspect ratios.
- **Why**: CapCut AI Reframe 2.0 tracks multiple subjects. Essential for shorts.
- **Module**: `ai_reframe_multi.py` | **Routes**: `/video/reframe-multi`, `/video/aspect-ratios`

## 70. Delivery & Mastering

### 70.1 DCP Export (P1 / L)
Digital Cinema Package for theatrical projection. JPEG2000 MXF video, 24-bit PCM audio MXF, CPL/PKL/ASSETMAP/VOLINDEX XML. 2K/4K DCI Flat/Scope.
- **Why**: Opens theatrical delivery market. No Premiere extension does this.
- **Module**: `dcp_export.py` | **Route**: `/export/dcp`

### 70.2 IMF Package (P2 / L)
Interoperable Master Format per SMPTE ST 2067. MXF OP1a wrapping, CPL/OPL/PKL, multiple audio languages, 4 profile variants.
- **Why**: Netflix, Disney+, and major streamers require IMF delivery.
- **Module**: `imf_package.py` | **Route**: `/export/imf`

### 70.3 Delivery Validation Suite (P0 / M)
Validate exports against platform specs: video, audio, container, subtitles. 7 built-in specs (Netflix, YouTube, Broadcast EBU, DCP, Apple TV+, Amazon, Theatrical). Pass/fail per check.
- **Why**: Failed delivery QC wastes days. Automated validation prevents rejection.
- **Module**: `delivery_validate.py` | **Routes**: `/delivery/validate`, `/delivery/specs`

### 70.4 Multi-Format Simultaneous Render (P1 / M)
Render to multiple formats in one pass with priority ordering. GPU renders sequential, CPU parallel. Per-render progress and cancellation.
- **Why**: Adobe Media Encoder's core value proposition. Must-have for batch delivery.
- **Module**: `multi_render.py` | **Routes**: `/render/multi`, `/render/multi/cancel`

### 70.5 Delivery Spec Profile Manager (P2 / S)
Define, import/export, compare, and auto-suggest delivery specs. 7 built-in profiles with full requirement definitions.
- **Why**: Teams need consistent delivery standards across projects.
- **Module**: `delivery_spec.py` | **Routes**: `/delivery/specs`, `/delivery/specs/compare`

## 71. AI Content Generation

### 71.1 Voice-to-Avatar Video Generation (P2 / L)
Record audio, generate lip-synced talking head video. 5 avatar styles (realistic, cartoon, silhouette, minimal, sketch). SadTalker/LivePortrait backends.
- **Why**: CapCut voice-to-avatar is a top feature. HeyGen built a company on this.
- **Module**: `voice_avatar.py` | **Routes**: `/ai/voice-avatar`, `/ai/voice-avatar/styles`

### 71.2 ML Thumbnail CTR Prediction (P1 / M)
Analyze thumbnails for CTR-relevant features: face presence, text readability, color vibrancy, composition, emotion. Score 0-100 with improvement suggestions. 5 platform weight profiles.
- **Why**: Thumbnails drive 90% of click-through rate. Data-driven optimization.
- **Module**: `ctr_predict.py` | **Routes**: `/content/predict-ctr`, `/content/compare-thumbnails`

### 71.3 AI B-Roll Generation from Text (P2 / L)
Generate B-roll clips from text descriptions. 4-tier fallback: API video gen → local model → AI image + Ken Burns → placeholder. Batch generation support.
- **Why**: Descript and Adobe are adding AI B-roll generation. Future table-stakes.
- **Module**: `broll_ai_gen.py` | **Routes**: `/ai/generate-broll`, `/ai/generate-broll/batch`

### 71.4 Auto Chapter Artwork Generation (P2 / M)
Generate styled chapter title cards. 5 card styles (minimal, bold, gradient, cinematic, split). Auto-title from transcript, brand kit integration.
- **Why**: Professional chapters need visual markers. Automates repetitive design.
- **Module**: `auto_chapter_art.py` | **Routes**: `/content/chapter-art`, `/content/chapter-art/styles`

### 71.5 Animated Video Intro Generation (P2 / M)
Generate animated intros from brand kit. 5 styles (logo_reveal, text_sweep, particles, minimal_fade, kinetic). Music bed integration, configurable duration.
- **Why**: Every video needs an intro. Templates save hours of motion design.
- **Module**: `ai_intro_gen.py` | **Routes**: `/video/generate-intro`, `/video/intro-styles`

## 72. Pipeline Intelligence

### 72.1 Pipeline Health Monitoring (P1 / M)
Track success/failure rates, processing times, error frequency, resource utilization. Per-component health scores (0-100) with trend detection and alerts.
- **Why**: Production pipelines need observability. Catches degradation early.
- **Module**: `pipeline_health.py` | **Route**: `/pipeline/health`

### 72.2 Scheduled/Recurring Jobs (P1 / M)
Cron-like job scheduling with full 5-field cron expression support. Persistent JSON storage, missed job detection, run history tracking.
- **Why**: Batch processing needs automation. Watch folders + schedules = hands-free.
- **Module**: `scheduled_jobs.py` | **Routes**: `/pipeline/schedules` (CRUD)

### 72.3 Smart Content Routing (P1 / M)
Auto-detect content type (10 types: interview, vlog, tutorial, etc.) and suggest optimal workflow with pre-configured parameters. Confidence scoring.
- **Why**: New users don't know which workflow to use. Smart routing guides them.
- **Module**: `smart_route.py` | **Routes**: `/video/classify-content`, `/video/suggest-workflow`

### 72.4 Processing Time Estimation (P2 / M)
Estimate processing time for any operation based on input characteristics, hardware, and historical data. 17 operation baselines with learning from actual runs.
- **Why**: Users need to know "how long will this take?" before committing.
- **Module**: `process_estimate.py` | **Routes**: `/pipeline/estimate`, `/pipeline/estimate/batch`

### 72.5 Resource Monitoring (P2 / M)
Real-time CPU/GPU/RAM/disk monitoring. GPU via nvidia-smi, ring-buffer history, availability checks before job start. Temperature monitoring.
- **Why**: GPU OOM is the #1 crash cause. Visibility prevents failures.
- **Module**: `resource_monitor.py` | **Routes**: `/pipeline/resources`, `/pipeline/resources/gpu`

## 73. AI Collaboration & Review

### 73.1 Frame-Accurate Review Comments (P1 / M)
Add timestamped, frame-accurate review comments with annotation types (text, rect, circle, arrow). Thread support, status tracking (open/resolved/wontfix). Export to JSON/CSV, import from Frame.io format. Persistent per-project storage.
- **Why**: Frame.io built a $1.3B company on review workflows. Collaborative review is table-stakes for teams.
- **Module**: `review_comments.py` | **Routes**: `/review/comments` (CRUD), `/review/comments/export`

### 73.2 Version Comparison (P1 / M)
Frame-by-frame comparison of two video renders. 4 modes: side_by_side, overlay_diff, flicker, swipe. SSIM/PSNR per frame, diff heatmaps, audio loudness comparison. Overall similarity score 0-100.
- **Why**: Every revision cycle needs visual diff. No Premiere extension offers automated video comparison.
- **Module**: `version_compare.py` | **Route**: `/version/compare`

### 73.3 Approval Workflow Pipeline (P1 / M)
Multi-stage approval: draft → internal_review → client_review → approved → final. Configurable required approvers per stage, auto-advance, deadline tracking with overdue detection.
- **Why**: Enterprise teams need structured approval. Replaces email chains and spreadsheet tracking.
- **Module**: `approval_workflow.py` | **Routes**: `/approval/status`, `/approval/advance`, `/approval/create`

### 73.4 Edit History & Audit Log (P2 / M)
Immutable JSONL audit log of all operations. Undo stack, diff between history points, replay capability, usage statistics. Append-only storage per project.
- **Why**: Compliance and reproducibility. "What changed and when?" is unanswerable without audit trails.
- **Module**: `edit_history.py` | **Route**: `/edit-history/export`

### 73.5 Shared Preset Library (P2 / M)
Team-shared presets across 5 categories (color grades, audio chains, export profiles, workflows, caption styles). Rating, duplicate detection, import/export as .opencut-preset files, merge conflict resolution.
- **Why**: Teams need consistent look and settings. Shared presets eliminate per-editor variation.
- **Module**: `shared_presets.py` | **Routes**: `/presets/shared` (GET/POST)

## 74. Advanced Timeline Automation

### 74.1 AI Rough Cut from Script (P0 / XL)
Given a script/brief + media files, AI matches transcript to script sections, scores candidates, selects best takes, generates EDL. Modes: strict, loose, highlight. LLM with keyword fallback.
- **Why**: The holy grail of AI editing. Opus Clip, Vizard, and Adobe Sensei are converging here.
- **Module**: `auto_rough_cut.py` | **Route**: `/timeline/rough-cut`

### 74.2 Multi-Track Audio Auto-Mix (P1 / L)
Analyze dialogue/music/SFX/ambience tracks, auto-duck, level-match, generate gain keyframes. Ducking profiles: podcast, film, music_video. Optional mixed-down audio output.
- **Why**: Auto-mixing saves hours per episode. DaVinci Resolve Fairlight has auto-leveling; Premiere doesn't.
- **Module**: `auto_mix.py` | **Routes**: `/timeline/auto-mix`, `/timeline/auto-mix/preview`

### 74.3 Smart Trim Point Detection (P1 / M)
Find optimal in/out points: skip pre-roll/slate, detect first speech, find natural ending. Modes: tight, broadcast, social (hook-first). Batch processing.
- **Why**: Manual trim is the most repetitive editing task. Smart detection eliminates guesswork.
- **Module**: `smart_trim.py` | **Routes**: `/timeline/smart-trim`, `/timeline/smart-trim/batch`

### 74.4 Batch Timeline Operations (P1 / M)
6 batch ops: color_grade, speed, normalize, transition, crop, caption. Operation chaining (pipeline), dry-run preview mode, per-clip progress.
- **Why**: Applying the same operation to 50 clips manually is tedious. Batch ops are essential for scale.
- **Module**: `batch_timeline_ops.py` | **Routes**: `/timeline/batch-ops`, `/timeline/batch-ops/preview`

### 74.5 Template-Based Assembly (P1 / M)
Assemble video from slot-based templates. 4 built-in: youtube_video, podcast_video, tutorial, social_clip. Auto-trim to fit durations, transitions between segments. EDL/OTIO export.
- **Why**: Recurring content uses identical structure. Templates make assembly instant.
- **Module**: `template_assembly_adv.py` | **Routes**: `/timeline/assemble`, `/timeline/templates`, `/timeline/templates/validate`

## 75. AI Sound Design & Music

### 75.1 AI Sound Design from Video (P1 / L)
Analyze video frames for motion events, map to 12 SFX categories (impact, whoosh, ambient, etc.), synthesize via PCM engines, mix to timeline. Automatic foley generation.
- **Why**: HunyuanVideo-Foley demonstrates this is feasible. Eliminates manual SFX placement.
- **Module**: `sound_design_ai.py` | **Routes**: `/audio/sound-design`, `/audio/sfx-categories`

### 75.2 Procedural Ambient Soundscape Generator (P2 / M)
Generate ambient audio from 7 presets (forest, ocean, city, rain, office, space, cafe). Multi-layer PCM synthesis, configurable intensity/duration/seed. Crossfade looping.
- **Why**: Every video needs ambient fill. Procedural generation is cheaper and more flexible than sample libraries.
- **Module**: `ambient_generator.py` | **Routes**: `/audio/ambient/generate`, `/audio/ambient/presets`

### 75.3 Music Mood Morph (P2 / M)
Transform music mood over time. 6 transforms: brighten, darken, energize, calm, build, drop. Keyframeable mood curves with FFmpeg EQ/atempo/compand filters.
- **Why**: Music rarely matches the emotional arc of the edit. Mood morphing adapts music to story beats.
- **Module**: `music_mood_morph.py` | **Route**: `/audio/mood-morph`

### 75.4 Beat-Synced Auto-Edit (P1 / L)
Detect beats, align video cuts to music rhythm. 6 modes: every_beat, every_bar, accent_only, custom. Energy matching pairs clip intensity to beat strength. Full assembly output.
- **Why**: Music video editing is 90% beat-matching. CapCut's auto-beat sync is a top feature.
- **Module**: `beat_sync_edit.py` | **Routes**: `/audio/beat-sync`, `/audio/beat-detect`

### 75.5 Stem Remix & Mashup (P2 / M)
Per-stem effects (volume, pan, reverb, delay, pitch shift, reverse, mute) with 7 presets: acapella, instrumental, karaoke, lo_fi, nightcore, slowed_reverb, drum_emphasis.
- **Why**: Stem separation exists; creative remixing makes it useful beyond simple isolation.
- **Module**: `stem_remix.py` | **Routes**: `/audio/stem-remix`, `/audio/remix-presets`, `/audio/stem-remix/preview`

## 76. Real-Time AI Preview

### 76.1 Live AI Effect Preview (P1 / M)
Preview any of 10 effects at 480p on a single frame for instant feedback. LRU cache (100 entries / 500MB). Effect chain support. Returns base64 JPEG or temp file.
- **Why**: Current workflow is blind: adjust → process whole clip → review. Live preview eliminates the loop.
- **Module**: `live_preview.py` | **Route**: `/preview/live`

### 76.2 GPU-Accelerated Preview Pipeline (P1 / L)
Preview render queue with GPU detection (nvidia-smi/torch.cuda) and CPU fallback. Batch preview at N timestamps for scrubbing. Effect chain pipeline.
- **Why**: GPU acceleration makes preview viable at interactive framerates.
- **Module**: `gpu_preview_pipeline.py` | **Routes**: `/preview/pipeline`, `/preview/scrub`, `/preview/pipeline/status`

### 76.3 A/B Comparison Generator (P1 / M)
6 comparison modes: side_by_side, overlay_blend, wipe_horizontal, wipe_vertical, split_diagonal, checkerboard. Quality metrics: SSIM, PSNR, MSE, color delta.
- **Why**: Every processing operation needs before/after validation. Export-ready comparisons.
- **Module**: `ab_compare.py` | **Routes**: `/preview/compare`, `/preview/compare/metrics`

### 76.4 Real-Time Video Scopes (P2 / M)
5 scope types: waveform, vectorscope, histogram, parade, false_color. 4 presets (colorist, exposure, broadcast, full). Broadcast legal range violation detection.
- **Why**: DaVinci Resolve's scopes are best-in-class. Bringing scopes to Premiere via OpenCut fills a major gap.
- **Module**: `realtime_scopes.py` | **Routes**: `/preview/scopes`, `/preview/scopes/presets`

### 76.5 Preview Render Cache (P2 / M)
Disk-persistent LRU cache in `~/.opencut/preview_cache/`. Configurable size/TTL, background cleanup, invalidation by file/effect/flush, cache warming. Thread-safe.
- **Why**: Re-computing identical previews wastes GPU cycles. Caching makes scrubbing instant.
- **Module**: `preview_cache.py` | **Routes**: `/preview/cache/stats`, `/preview/cache/warm`, `/preview/cache` (DELETE)

## 77. Cloud & Distribution

### 77.1 Cloud/Remote Render Dispatch (P1 / L)
Define render nodes, health check via HTTP, dispatch jobs with least-loaded balancing, retry on failure (max 2), fallback to local. Batch dispatch, persistent node config.
- **Why**: Render time is the biggest bottleneck. Offloading to faster/additional machines scales linearly.
- **Module**: `cloud_render.py` | **Routes**: `/cloud/render`, `/cloud/nodes` (CRUD)

### 77.2 Multi-Platform Auto-Publish (P1 / L)
Generate export packages for 10 platforms (YouTube, TikTok, Instagram variants, Twitter/X, LinkedIn, Facebook, Vimeo, Podcast RSS). Per-platform validation, thumbnail, metadata limits. Batch export with manifest.
- **Why**: Every creator publishes to 3-5 platforms. Manual re-export and metadata entry is hours of work per video.
- **Module**: `platform_publish.py` | **Routes**: `/publish/prepare`, `/publish/platforms`, `/publish/validate`

### 77.3 Content Fingerprint & Duplicate Detection (P2 / M)
Perceptual hash (pHash) + audio fingerprint. Hamming distance comparison, similarity 0-100. SQLite index for batch search across project library.
- **Why**: Duplicate detection saves storage. Fingerprinting enables content identification and rights management.
- **Module**: `content_fingerprint.py` | **Routes**: `/fingerprint/generate`, `/fingerprint/search`

### 77.4 Render Farm Management (P1 / L)
Split renders into segments, dispatch to multiple nodes, collect and concatenate. 3 strategies: equal_duration, scene_based, chapter_based. GPU routing, fault tolerance.
- **Why**: A 1-hour render on 4 machines takes ~15 minutes. Segment-based farm rendering is standard in VFX.
- **Module**: `render_farm.py` | **Routes**: `/farm/render`, `/farm/status`

### 77.5 Distribution Analytics (P2 / M)
Track publish records, manual metrics entry (views, likes, CTR, watch time). Cross-platform aggregation, top performers, content type analysis, growth trends. CSV/JSON export.
- **Why**: Understanding what works across platforms guides content strategy. No video editor includes post-publish analytics.
- **Module**: `distribution_analytics.py` | **Routes**: `/analytics/record`, `/analytics/report`

---

*This document should be reviewed and reprioritized quarterly as the competitive landscape and user feedback evolve.*
