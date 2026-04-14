# OpenCut v1.11.0 — Competitive Audit & Improvement Plan

**Date**: 2026-04-14
**Scope**: Competitive analysis, technology gaps, new feature opportunities, UX improvements
**Current state**: 302 features, 62 categories, 932 routes, 335 core modules, 67 blueprints, 5,127 tests

---

## Executive Summary

OpenCut's 302-feature breadth is unmatched for an open-source Premiere Pro extension. However, several competitors have shipped breakthrough features in 2025-2026 that OpenCut either lacks entirely or implements with older technology. The biggest gaps are:

1. **Real-time AI preview** — competitors show AI effects live; OpenCut requires full processing first
2. **Next-gen AI models** — OpenCut references 2023-2024 models when 2025-2026 successors exist
3. **Unified conversational editing** — Descript/Frame prove chat-driven editing is the future UX
4. **Feature discoverability** — 302 features across 50+ sub-tabs is overwhelming without a command palette + AI suggestions
5. **Collaborative workflows** — Frame.io integration, cloud render, team features are table-stakes for pros

---

## Part 1: Competitive Analysis

### Descript (Primary Competitor)

**What they do better:**
- Transcript-based editing is native and seamless (not bolted on)
- Overdub voice correction with 30-second voice clone setup
- Eye Contact correction with real-time preview
- "Underlord" AI agent handles multi-step editing from natural language
- Filler word removal is instant, visual, and reversible
- Built-in screen recording + editing in one tool
- Collaboration: real-time multi-user editing on shared projects
- AI Actions: "make this punchier", "remove all ums", "add B-roll" via chat

**New in 2025-2026:**
- AI-powered scene-aware B-roll insertion from stock library
- Automatic speaker-matched lower thirds from transcript
- "Regenerate" — re-record any line with AI voice clone + lip sync in one click
- Green screen replacement without actual green screen (AI matting)
- Integrated podcast distribution to Spotify/Apple/YouTube

**Pricing**: Free tier (1hr/mo transcription), $24/mo Creator, $33/mo Business

**OpenCut gaps to close:**
- [ ] One-click "Regenerate" workflow (voice clone + lip sync + composite)
- [ ] Real-time Overdub preview (hear the cloned voice before committing)
- [ ] Multi-user collaboration (even basic shared presets/workflows)

---

### CapCut (Desktop + Mobile)

**What they do better:**
- Instant AI effects with real-time preview (no processing wait)
- Motion tracking overlays are drag-and-drop simple
- Auto-captions with 50+ animated styles, one-click apply
- Beat-sync editing: drop clips + music, auto-cut to beats
- Massive free template/effect library (10,000+ templates)
- Integrated stock media (licensed from Pexels/Pixabay)
- Body-aware effects (object follows body parts)
- Mobile-first design that also works on desktop
- Transition sound effects auto-applied with video transitions

**New in 2025-2026:**
- AI Reframe 2.0: multi-subject tracking with intelligent crop decisions
- AI Video Enhance: one-click upscale + denoise + color correct
- Auto-edit from script: paste text, AI assembles matching clips
- Voice-to-avatar: record audio, AI generates talking avatar video
- AI Eraser: remove any object from video in real-time
- Long-to-Shorts AI: automatic highlights + vertical reframe + captions

**Pricing**: Free with watermark, $7.99/mo Pro, $13.99/mo Business

**OpenCut gaps to close:**
- [ ] Template marketplace / community templates
- [ ] Animated caption style library (50+ styles, not just positioning)
- [ ] One-click "enhance" that chains upscale + denoise + color
- [ ] Real-time AI effect previews (even at reduced resolution)

---

### Runway ML

**What they do better:**
- Gen-3 Alpha: 10-second video generation from text/image at 1080p
- Act-One: facial expression transfer to generated characters
- Universal video-to-video style transfer in real-time
- Inpainting/outpainting on video with temporal consistency
- Multi-modal editing: describe what you want in any frame
- Training custom models on your footage (fine-tuning)

**New in 2025-2026:**
- Gen-3 Alpha Turbo: 4x faster generation
- Frames: consistent character generation across scenes
- Motion Brush: paint motion onto static images
- Advanced camera controls in generated video (dolly, pan, zoom)
- API for programmatic video generation at scale
- Audio-reactive video generation synced to music

**Pricing**: $12/mo Standard (625 credits), $28/mo Pro, $76/mo Unlimited

**OpenCut gaps to close:**
- [ ] Video generation integration (Wan 2.1 is decent but Gen-3 quality is higher)
- [ ] Motion Brush equivalent (paint where motion should happen)
- [ ] Consistent character/scene generation across shots
- [ ] Video-to-video style transfer with temporal coherence

---

### AutoCut / AutoPod (Direct Premiere Extension Competitors)

**What they do better:**
- AutoCut: purpose-built, ultra-fast silence removal directly in Premiere timeline
- AutoPod: podcast multicam switching by speaker detection natively in timeline
- Both modify the Premiere timeline directly (not export-based workflow)
- Stream Deck integration out of the box
- Simpler UI — focused tools, not 302 features

**New in 2025-2026:**
- AutoCut 3.0: AI-powered best take selection
- Captions directly in Premiere Essential Graphics (not sidecar)
- Premiere native timeline manipulation (not XML import/export)

**Pricing**: AutoCut $14.99/mo, AutoPod $29/mo

**OpenCut advantage**: OpenCut already surpasses both in feature breadth. The gap is in native timeline integration speed and simplicity.

**OpenCut gaps to close:**
- [ ] Faster timeline operations via direct ExtendScript (avoid export/reimport)
- [ ] Premiere Essential Graphics caption insertion (not just overlay burn-in)
- [ ] Purpose-built "Quick Mode" UI for common operations

---

### Opus Clip / Vizard (AI Short-Form Repurposing)

**What they do better:**
- One-URL input: paste a YouTube link, get 10 viral shorts back
- AI Virality Score predicts which clips will perform best
- Dynamic captions with animated word highlights per platform
- Multi-platform export with platform-specific safe zones
- A/B testing: generate multiple versions with different hooks

**New in 2025-2026:**
- AI B-Roll: auto-insert relevant stock footage between talking segments
- Clip Chains: stitch multiple viral moments into a cohesive narrative
- Engagement heatmaps: predict exactly where viewers will drop off
- Auto-hook generation: AI writes and inserts a hook in the first 3 seconds

**Pricing**: Free (10 clips/mo), $19/mo Starter, $49/mo Pro

**OpenCut gaps to close:**
- [ ] Virality/engagement scoring with predicted retention curve
- [ ] Auto-hook generation (AI writes + inserts opening hook)
- [ ] A/B variant generation (multiple versions of same clip)
- [ ] Engagement heatmap prediction

---

### Topaz Video AI

**What they do better:**
- Real-time preview of upscaling/denoising before processing
- Multiple AI models per task (Proteus, Artemis, Gaia for upscale)
- Frame interpolation up to 120fps with near-zero artifacts
- Dedicated grain management (add natural grain after denoising)
- Batch processing with per-file model selection

**New in 2025-2026:**
- Nyx: low-light enhancement model (beyond denoising)
- 8K upscaling with temporal consistency
- Interlace-to-progressive with AI reconstruction (not just deinterlace)

**Pricing**: $199 perpetual (or $99/year updates)

**OpenCut gaps to close:**
- [ ] Low-light enhancement (not just denoising — actual light recovery)
- [ ] Multiple upscaling model options with quality/speed tradeoffs
- [ ] Real-time upscale preview at reduced resolution

---

### Adobe Firefly / Sensei (Built Into Premiere)

**What they do better:**
- Native integration — no extension needed
- Generative Extend: AI extends clips beyond their recorded length
- Enhanced Speech: AI restores badly recorded dialogue
- Text-Based Editing: native transcript editing in timeline
- Auto Tone: automatic color matching between shots
- Audio Category Tagging: AI identifies music/speech/SFX/ambience
- Captions workflow integrated into Essential Graphics

**New in 2025-2026:**
- Firefly Video: text-to-video B-roll generation in-app
- Object-Aware Reframe: intelligent crop that avoids cutting subjects
- AI Scene Edit Detection: auto-detect cuts in pre-edited footage
- AI Music Generation: generate royalty-free background music
- Generative Remove: erase objects from video natively

**Pricing**: Included with Creative Cloud ($22.99/mo)

**Critical note**: Adobe is building AI features directly into Premiere. OpenCut must offer features Adobe *won't* build (open-source models, privacy-first, no subscription, model choice, extensibility) or do them *better/faster*.

---

### DaVinci Resolve (Free NLE)

**What they do better:**
- Color: ACES pipeline, power windows, HSL qualifiers, node-based grading
- Fairlight: full DAW-quality audio post-production
- Fusion: node-based VFX compositing integrated into timeline
- Free version covers 95% of professional needs
- IntelliTrack: AI-powered object tracking in Fusion
- DaVinci Neural Engine: AI face refinement, object removal, upscaling
- Collaboration: multi-user project sharing with bin locking

**New in 2025-2026:**
- AI Dialogue Leveler: automatic per-speaker normalization
- AI Music Remix: adjust music duration while preserving feel
- IntelliTrack 2.0: faster, more accurate object tracking
- USD/3D scene import into Fusion
- Cloud collaboration for remote teams

**Pricing**: Free / $295 one-time for Studio

**OpenCut positioning**: Resolve is a full NLE competitor, not an extension. OpenCut's advantage is being embedded *inside* Premiere's workflow. Focus on features that augment Premiere rather than replicate Resolve.

---

### Emerging Competitors (2025-2026 Startups)

| Tool | Focus | Notable Innovation |
|------|-------|-------------------|
| **Captions.ai** | Auto-captions + social editing | Animated word-by-word captions with 200+ styles, AI avatars |
| **Veed.io** | Browser-based editing | AI eye contact, background removal, auto-subtitles all in browser |
| **Kapwing** | Collaborative browser editor | Real-time multi-user editing, AI tools, brand kit enforcement |
| **Gling** | YouTube editing automation | Silence + filler + bad take removal specifically for YouTube creators |
| **Wisecut** | AI auto-editing | Complete auto-edit from raw footage using AI pacing decisions |
| **Riverside.fm** | Podcast/interview recording | Studio-quality local recording + AI post-production built in |

---

## Part 2: Technology Gaps — Models to Upgrade

OpenCut references several AI models that now have superior successors. Upgrading these would improve quality, speed, or both.

### Video AI Models

| Current in OpenCut | Successor (2025-2026) | Improvement | Integration Effort |
|----|----|----|-----|
| Real-ESRGAN (upscale) | **SUPIR** / **Upscale-A-Video** | Temporal consistency, fewer artifacts, handles compression better | M — drop-in model swap |
| RIFE (frame interp) | **RIFE 4.26+** / **AMT** (Adaptive Multi-scale Temporal) | Better handling of large motion, fewer ghosting artifacts | S — version bump |
| SAM2 (segmentation) | **SAM 2.1** / **Grounded SAM 2** | Better text-prompted segmentation, improved tracking persistence | M — API compatible |
| Depth Anything V2 | **Depth Pro** (Apple) / **Metric3D v2** | Metric depth (actual distances), better edge accuracy | M — output format change |
| ProPainter (inpainting) | **Flow-Guided Video Completion** / **E2FGVI v2** | Better temporal coherence, faster, handles complex motion | M |
| vidstab (stabilization) | **Deep Video Stabilization** / **FuSta** | AI path planning, no crop needed, handles parallax | L — new pipeline |
| DeepFilterNet (denoise audio) | **Resemble Enhance** / **UNIVERSE++** | Handles music+speech, super-resolution for audio, reverb removal | M |
| Demucs v4 (stems) | **BS-RoFormer** / **Bandit v2** | Cleaner vocal isolation, 6-stem separation, better on live recordings | S — already optional |
| faster-whisper (ASR) | **Whisper v3 turbo** / **Canary** (NVIDIA) / **Parakeet** | 3-5x faster, better accuracy on accents, native word timestamps | S — config change |
| Chatterbox (TTS) | **F5-TTS** / **MaskGCT** / **CosyVoice 2** | Zero-shot voice cloning from 3-second sample, emotional control | M |
| AudioCraft (music gen) | **Stable Audio 2.0** / **Udio API** | Higher quality, longer generations, better stereo, style control | M |
| AnimateDiff (video gen) | **Wan 2.1** (already integrated) / **CogVideoX** / **Hunyuan Video** | 1080p output, longer clips, better motion coherence | Already partially done |
| MediaPipe (face mesh) | **MediaPipe 0.10.20+** / **InsightFace 2D/3D** | More landmarks, better occlusion handling, faster on CPU | S |
| Florence-2 (vision) | **Florence-2 v2** / **InternVL 2.5** / **Qwen2-VL** | Better scene understanding, grounded descriptions, video input | M |

### Audio AI Models

| Category | Best Available (2026) | Why It Matters |
|----|----|-----|
| Speech Enhancement | **Resemble Enhance** | Super-resolution for low-bitrate audio + denoising in one pass |
| Voice Conversion | **RVC v2** / **Seed-VC** | Real-time voice conversion with better timbre preservation |
| Speaker Diarization | **pyannote 3.3+** / **NeMo MSDD** | Multi-scale diarization, better overlap handling |
| Sound Classification | **BEATs** / **Audio Spectrogram Transformer** | Classify any environmental sound for auto-SFX/SDH tagging |
| Audio Restoration | **VoiceFixer 2** / **Apollo** | Fix clipping, reverb, noise, and codec artifacts in one model |
| Music Separation | **Band-Split RNN (BS-RoFormer)** | 6 stems (vocals, drums, bass, guitar, piano, other) |

### Emerging Tech Not Yet In Any Video Editor

| Technology | What It Enables | Maturity |
|----|----|-----|
| **Video LLMs** (Qwen2-VL, InternVL) | "Find the moment where she laughs" — direct video understanding | Production-ready |
| **4D Gaussian Splatting** | Capture + replay dynamic 3D scenes from phone video | Research → early production |
| **Neural Video Compression** | 30-50% better compression than AV1 with AI-decoded frames | Research |
| **ControlNet for Video** | Precise control over generated video (pose, depth, edge guidance) | Production-ready |
| **IP-Adapter** | Generate consistent characters/objects across multiple scenes | Production-ready |
| **AnimateAnyone 2** / **MimicMotion** | Full-body motion transfer from reference video | Production-ready |
| **SonicSim** / **SoundSpaces** | Spatially-aware audio rendering based on room geometry | Research |
| **MusicLM** / **MusicGen 2** | Generate soundtrack that matches video mood/pacing | Production-ready |

---

## Part 3: New Features to Add (Not in features.md)

These are features that emerged from competitive analysis that aren't covered by any of the 302 existing features:

### P0 — Must Add

| # | Feature | Rationale |
|---|---------|-----------|
| N1 | **Command Palette / Spotlight Search** | With 302 features, users need Cmd+K instant search. Every modern tool has this (VS Code, Figma, Notion). Search features by name, description, or intent. |
| N2 | **AI Enhanced Speech** (Adobe-style) | Restore badly recorded dialogue — not just denoise, but reconstruct clarity. Adobe ships this natively now. Use Resemble Enhance or similar. |
| N3 | **Generative Extend** | AI extends clips beyond their recorded length. Adobe Firefly does this in Premiere. Use Wan 2.1 or CogVideoX conditioned on last frames. |
| N4 | **Engagement/Retention Prediction** | Predict where viewers will drop off. Opus Clip and YouTube analytics prove this drives editing decisions. Use LLM + audio energy + pacing analysis. |
| N5 | **Animated Caption Style Library** | 50+ animated word-by-word caption styles (not just positioning). CapCut and Captions.ai dominate here. Package as JSON animation presets. |

### P1 — High Impact

| # | Feature | Rationale |
|---|---------|-----------|
| N6 | **AI Hook Generator** | Generate and insert a compelling opening hook in the first 3 seconds. Opus Clip does this. LLM writes hook text, TTS + caption overlay applies it. |
| N7 | **A/B Variant Generator** | Generate 3-5 versions of the same clip with different hooks, captions, thumbnails. Test on platforms. Opus Clip popularized this. |
| N8 | **Video LLM Integration** | Use Qwen2-VL or InternVL for direct video understanding: "find the funniest moment", "what's happening at 2:34". Goes beyond transcript search. |
| N9 | **One-Click Enhance Pipeline** | Single button: upscale + denoise + color correct + stabilize. CapCut's most popular feature. Chain existing modules with smart defaults. |
| N10 | **AI Music Remix / Duration Fit** | Automatically adjust background music duration to match video length while preserving feel. DaVinci Resolve added this. Use audio time-stretching + beat-aware cutting. |
| N11 | **Essential Graphics Caption Output** | Insert captions as Premiere Essential Graphics (MOGRT) instead of burned-in overlays. AutoCut does this. Enables editor to reposition/restyle in Premiere. |
| N12 | **Green-Screen-Free Background Replacement** | AI matting without green screen (Descript, CapCut both do this). Already have SAM2/rembg — needs real-time preview + background replacement compositor. |
| N13 | **Low-Light Enhancement** | Beyond denoising — actual light recovery for dark footage. Topaz Nyx model does this. Use specialized low-light model or exposure fusion. |
| N14 | **AI Scene Edit Detection** | Detect cuts in pre-edited footage (received from clients, stock footage). Adobe added this. FFmpeg `scdet` filter + validation. |

### P2 — Valuable Additions

| # | Feature | Rationale |
|---|---------|-----------|
| N15 | **Audio Category Tagging** | Auto-classify timeline audio as speech/music/SFX/ambience/silence. Adobe Sensei does this. Enables smart ducking, stem-aware processing. |
| N16 | **Consistent Character Generation** | Generate the same character across multiple AI-generated shots. Runway Frames + IP-Adapter enable this. |
| N17 | **Motion Brush / Cinemagraph+** | Paint where motion should happen on a still image/video. Runway popularized this. Extends existing cinemagraph feature. |
| N18 | **Body/Pose-Driven Effects** | Track body keypoints, apply effects that follow body parts. CapCut does this. MediaPipe Pose → effect anchoring. |
| N19 | **Full-Body Motion Transfer** | Transfer motion from a reference video to a target person. AnimateAnyone 2 / MimicMotion enable this. |
| N20 | **AI Color Match Between Shots** | Auto-match color/exposure between different shots for consistent look. Adobe Auto Tone does this. Histogram + color transfer + LLM-guided adjustment. |

---

## Part 4: UX Overhaul Recommendations

The single biggest challenge for OpenCut isn't features — it's making 302 features discoverable and usable. Here's the prioritized UX improvement plan:

### Tier 1: Critical UX Infrastructure

#### 1. Command Palette (Cmd+K / Ctrl+K)
**Priority**: P0 — this alone transforms the experience

Every modern creative tool has this. Type any keyword to find and execute any of 302 features instantly. No tab navigation needed.

**Implementation:**
- Overlay modal triggered by keyboard shortcut or persistent search icon
- Fuzzy-match search across feature names, descriptions, and aliases
- Show recent commands, favorites, and contextual suggestions
- Execute directly from results (opens the right tab + card pre-configured)
- Group results: Features | Workflows | Settings | Help

**Why it matters:** Users currently navigate 8 tabs → sub-tabs → scroll to find features. Command palette reduces any operation to: open palette → type 3 letters → Enter.

#### 2. AI-Powered Contextual Suggestions
**Priority**: P0

Analyze the current clip and suggest relevant operations as dismissible chips above the active panel.

**Implementation:**
- On clip selection: analyze audio levels, resolution, stability, faces, content type
- Generate 2-3 relevant suggestions: "Audio is quiet (-28 LUFS) — Normalize?" "Detected 3 speakers — Diarize?"
- One-click to execute with smart defaults
- Learn from accept/dismiss patterns
- Suppress after 2 dismissals of same suggestion type

**Why it matters:** Surfaces the right feature at the right moment. Users discover features they didn't know existed.

#### 3. Task-Based Navigation (Not Feature-Based)
**Priority**: P1

Reorganize the primary navigation around user goals, not tool categories:

```
Current (feature-based):     Proposed (task-based):
[Cut] [Audio] [Video]...     [Clean Up] [Caption] [Publish]
                              [Create] [Analyze] [Settings]
```

**Proposed task-based tabs:**
- **Clean Up**: Silence removal, filler removal, denoise, normalize, stabilize, deinterlace
- **Caption & Text**: Auto-captions, translations, lower thirds, titles, credits, OCR
- **Transform**: Reframe, upscale, speed, crop, rotate, lens correction, HDR
- **Create**: AI generation, effects, transitions, music, avatars, storyboards
- **Analyze**: Scopes, QC checks, loudness meter, pacing, shot classification
- **Publish**: Export presets, social media, batch render, thumbnails, SEO
- **Automate**: Workflows, watch folders, macros, agent, batch operations
- **Settings**: Preferences, AI models, themes, integrations, hardware

**Why it matters:** Users think "I need to clean up this interview" not "I need the audio tab, then the silence sub-tab." Task-based navigation matches mental models.

### Tier 2: Interaction Improvements

#### 4. Progressive Disclosure
**Priority**: P1

Every operation card shows a "Simple" view by default with 2-3 controls. An "Advanced" toggle reveals full parameter set.

**Implementation:**
- Default: input file + 1-2 key parameters + "Run" button
- Advanced toggle: all parameters, presets, output options
- Remember user preference per-card (localStorage)
- "Smart defaults" auto-fill based on clip analysis

**Why it matters:** Novice users see a clean, approachable interface. Power users get full control. Neither group is overwhelmed or constrained.

#### 5. Unified Preview System
**Priority**: P1

Before/after preview for ANY processing operation. Extract one frame, apply the operation, show side-by-side or wipe comparison.

**Implementation:**
- New `/preview/frame` endpoint: process single frame, return base64
- Panel: interactive wipe slider (canvas-based)
- Toggle: original | processed | side-by-side | overlay diff
- "Apply to clip" button after preview approval
- Cache previews per operation + parameters

**Why it matters:** Users currently process entire clips blind, then review. Preview eliminates the guess-and-check loop that wastes hours. Topaz and Lightroom prove this is the expected experience.

#### 6. Favorites & Quick Access Bar
**Priority**: P1

Persistent favorites bar at the top of the panel. Pin most-used operations for one-click access from any tab.

**Implementation:**
- Right-click any operation card → "Add to Favorites"
- Favorites bar: horizontal scrollable strip below tab bar
- Drag to reorder, right-click to remove
- Pre-populate with 5 most-used operations after 1 week
- Badge: show "NEW" on recently added features for 7 days

#### 7. Workflow Wizard (Guided Flows)
**Priority**: P2

Step-by-step guided flows for common multi-step workflows:

- **"Interview Cleanup"**: Select clip → Denoise → Normalize → Remove silence → Remove fillers → Add captions → Export
- **"Social Content"**: Select clip → Generate shorts → Add captions → Reframe 9:16 → Export per platform
- **"Podcast Publish"**: Select audio → Clean → Level → Add chapters → Export + RSS

**Implementation:**
- Wizard UI: progress steps at top, current step's operation card in center
- Each step pre-configured with workflow-appropriate defaults
- Skip/back buttons, preview between steps
- Save custom wizard flows
- "Run All Remaining" for experienced users

### Tier 3: Polish & Delight

#### 8. Operation Chaining / Pipeline Builder
**Priority**: P2

Visual pipeline builder: drag operations into a sequence, connect them, run as a batch. More intuitive than JSON workflow editing.

**Implementation:**
- Node-based visual editor (simplified, not ComfyUI-complex)
- Drag operation nodes from a sidebar palette
- Connect output → input between nodes
- Preview at any node in the chain
- Save as reusable workflow preset

#### 9. Inline Help & Context
**Priority**: P2

- Hover any parameter label → tooltip with explanation + recommended range
- "?" icon per card → opens documentation panel with examples
- First-use spotlight: highlight and explain new features on first encounter
- Error messages include "Fix" buttons with one-click resolution

#### 10. Notification & Progress Center
**Priority**: P2

Centralized notification center for all background operations:

- Slide-out panel showing all active/completed/failed jobs
- Per-job: progress bar, elapsed time, estimated remaining
- Click completed job → preview result
- Desktop notification on completion (opt-in)
- Sound notification (subtle chime) on completion (opt-in)

#### 11. Smart Defaults Engine
**Priority**: P2

Auto-detect clip characteristics and pre-fill optimal parameters:

- Interview detected (single speaker, static camera) → optimize for speech
- Music video detected (multiple cuts, music dominant) → optimize for visual effects
- Screen recording detected (static, text-heavy) → optimize for zoom/annotations
- Drone footage detected (GPS metadata, wide lens) → suggest stabilize + ND simulation

#### 12. Keyboard Navigation (Tab + Arrow Keys)
**Priority**: P3

Full keyboard navigation within the panel for accessibility:
- Tab through operation cards
- Arrow keys within cards (parameters)
- Enter to execute
- Escape to cancel/close
- `/` to open command palette

---

## Part 5: Prioritized Implementation Roadmap

### Wave 1: UX Foundation (1-2 weeks)
| Item | Type | Impact |
|------|------|--------|
| Command Palette | UX | Transforms feature discovery |
| AI Contextual Suggestions | UX | Surfaces features at right moment |
| Preview System (`/preview/frame`) | UX | Eliminates blind processing |
| Favorites Bar | UX | Quick access to frequent operations |
| Progressive Disclosure (Simple/Advanced) | UX | Reduces overwhelm |

### Wave 2: Competitive Gap Closers (2-3 weeks)
| Item | Type | Impact |
|------|------|--------|
| AI Enhanced Speech (Resemble Enhance) | Feature | Matches Adobe |
| Animated Caption Library (50+ styles) | Feature | Matches CapCut/Captions.ai |
| One-Click Enhance Pipeline | Feature | CapCut's #1 feature |
| Engagement/Retention Prediction | Feature | Matches Opus Clip |
| Essential Graphics Caption Output | Feature | Matches AutoCut |
| Low-Light Enhancement | Feature | Matches Topaz Nyx |

### Wave 3: Model Upgrades (1-2 weeks)
| Item | Type | Impact |
|------|------|--------|
| Upgrade to Whisper v3 turbo | Model | 3-5x faster transcription |
| Upgrade to RIFE 4.26+ | Model | Better frame interpolation |
| Upgrade to SAM 2.1 / Grounded SAM 2 | Model | Text-prompted segmentation |
| Add F5-TTS / CosyVoice 2 option | Model | Better voice cloning |
| Add Resemble Enhance option | Model | Audio super-resolution |
| Add BS-RoFormer for 6-stem separation | Model | Cleaner stem separation |

### Wave 4: Next-Gen Features (2-4 weeks)
| Item | Type | Impact |
|------|------|--------|
| Video LLM Integration (Qwen2-VL) | Feature | Direct video understanding |
| Generative Extend | Feature | Matches Adobe Firefly |
| AI Hook Generator | Feature | Matches Opus Clip |
| A/B Variant Generator | Feature | Matches Opus Clip |
| AI Music Duration Fit | Feature | Matches DaVinci Resolve |
| Green-Screen-Free BG Replacement | Feature | Matches Descript/CapCut |

### Wave 5: UX Polish (1-2 weeks)
| Item | Type | Impact |
|------|------|--------|
| Task-Based Navigation Reorganization | UX | Mental model alignment |
| Workflow Wizards | UX | Guided multi-step operations |
| Smart Defaults Engine | UX | Auto-optimal parameters |
| Notification Center | UX | Background job management |
| Inline Help & Tooltips | UX | Self-documenting interface |

---

## Part 6: Strategic Positioning

### Where OpenCut Wins (Lean Into These)
1. **Privacy-first**: All processing local. No cloud. No subscription. No data leaving the machine.
2. **Model choice**: User picks their AI backend. Not locked to one vendor's model.
3. **Extensibility**: Plugin system, API-first, MCP server. Build anything on top.
4. **Professional features**: ACES, MXF, EDL/AAF, timecode, broadcast QC — features the consumer tools don't touch.
5. **Open source**: Inspect, modify, self-host, contribute. No vendor lock-in.

### Where OpenCut Should NOT Compete
1. **Cloud-native editing** (Kapwing, Veed.io) — stay local-first
2. **Mobile editing** (CapCut mobile) — desktop/NLE extension is the lane
3. **Social network features** (CapCut's community templates) — focus on tools, not social
4. **Real-time collaborative editing** (Frame.io, Descript teams) — too complex for an extension

### Differentiation Strategy
Position OpenCut as: **"The open-source AI editing engine that professionals trust."**

- Consumer tools (CapCut, Opus Clip) prioritize ease → OpenCut prioritizes *control and quality*
- Adobe prioritizes lock-in → OpenCut prioritizes *openness and choice*
- SaaS tools require internet → OpenCut works *entirely offline*
- Paid tools cost $20-200/mo → OpenCut is *free forever*

---

## Appendix: Feature Gap Matrix

Features that competitors have and OpenCut currently lacks entirely:

| Feature | Descript | CapCut | Runway | Adobe | OpenCut |
|---------|---------|--------|--------|-------|---------|
| Command Palette | Yes | No | Yes | No | **MISSING** |
| Real-time AI Preview | Yes | Yes | Yes | Yes | **MISSING** |
| Animated Caption Styles (50+) | Yes | Yes | No | Yes | **MISSING** |
| Engagement/Retention Scoring | No | No | No | No | **MISSING** |
| Generative Video Extend | No | No | Yes | Yes | **MISSING** |
| AI Enhanced Speech | Yes | No | No | Yes | **MISSING** |
| Auto Hook Generation | No | No | No | No | **MISSING** |
| A/B Variant Testing | No | No | No | No | **MISSING** |
| Video LLM Understanding | No | No | Yes | Partial | **MISSING** |
| Essential Graphics Output | No | No | No | Native | **MISSING** |
| Template Marketplace | No | Yes | No | Yes | **MISSING** |
| Low-Light Enhancement | No | Yes | No | No | **MISSING** |
| AI Color Match (Shot-to-Shot) | No | No | No | Yes | **MISSING** |
| Music Duration Remixing | No | No | No | No | **MISSING** |
| Body Keypoint Effects | No | Yes | No | No | **MISSING** |
