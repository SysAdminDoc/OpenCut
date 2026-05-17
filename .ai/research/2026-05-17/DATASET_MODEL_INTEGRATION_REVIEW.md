# OpenCut — Dataset, Model & Integration Review

**Audit date:** 2026-05-17
**Scope:** Datasets, AI models, external APIs, benchmarks, and evaluation harness items relevant to OpenCut. Focused on **what changed in 2026-Q2** that the existing wave/F-number plans have not yet absorbed.

---

## 1. Why this file is not thin

OpenCut is heavily AI-centric: 47 model cards (`docs/MODELS.md`), 35+ optional AI extras gated by `check_X_available()`, an AI evaluation harness (F120) that records latency + quality + environment per feature, and a `core/llm.py` abstraction over Ollama / OpenAI / Anthropic / Gemini. So this review goes deep.

---

## 2. Models the team has *not* yet considered (post-2026-05-16 research pass)

### 2.1 Video generation / editing

| # | Model | Date | Licence | Why it matters | Action |
|---|---|---|---|---|---|
| M1 | **daVinci-MagiHuman 15B** (Sand.ai + SII-GAIR) | Mar–Apr 2026 | **Apache 2.0** | Joint denoising of video + audio + lip-sync in one stream; 5 s clip in 2 s on one H100; 7 langs incl. EN/JP/KR/DE/FR/Mandarin. Beats Ovi 1.1 80% in human eval. | **Supersedes MuseTalk / LatentSync / EchoMimic V3** for talking-head when ≥48 GB GPU is available. Add as new backend in `lipsync_*` family; mark `recommended: true` when present. |
| M2 | **LTX-Video 2.3 (22B)** + Lightricks LTX Desktop | 5 Mar 2026 | **Apache 2.0** | Native 4K @ 50 FPS, 20 s clips, synchronized audio, portrait 9:16, distilled variants. | **Supersedes** current LTX-Video (1.x / 0.9.8) in `core/gen_video_ltx.py`. New routes: `/generate/ltx/v23`, `/generate/ltx/desktop-handoff`. |
| M3 | **Wan 2.7** | Apr 2026 | **Apache 2.0** (weights — *verify before integration*) | Major motion + audio-sync upgrade over Wan 2.2. (Wan 2.5/2.6 are closed API only.) | **Supersedes** Wan 2.1 in `core/gen_video_wan_vace.py` (and the C4 stub). |
| M4 | **MAGI-1 + MagiCompiler** (Sand.ai) | active 2026 | **Apache 2.0** | Autoregressive video generation at scale; MagiCompiler plug-and-play inference/training speedups. | **Complements** existing generators; MagiCompiler can accelerate OpenCut's diffusion paths generically. |
| M5 | **HunyuanVideo-1.5 (8.3B)** | Nov 2025 + ongoing 2026 | **Tencent Hunyuan Community License** — *not* OSI; territory carve-outs (EU/UK/KR) | Runs on consumer GPUs, better motion coherence. | **Conditional** — only ship if user is outside EU/UK/KR or accepts the licence on download. Add a `licence_warning` flag in model cards. |
| M6 | **HappyHorse 1.0** (Alibaba) | Apr 2026 | TBD — *do not adopt until licence confirmed* | 1080p native + integrated audio; higher quality than Open-Sora v2. | Monitor; promote to backlog when licence clarifies. |
| M7 | **Open-Sora v2** | (already roadmapped K2.16) | Apache-2 | Third OSS T2V option in the B-roll dispatcher. | Fill K2.16 stub. |
| M8 | **DiffSynth-Studio "Diffusion Templates"** | 28 Apr 2026 | **Apache 2.0** | Plugin framework that drastically lowers training cost of controllable generators (camera, pose, style) on top of LTX-2.3 / Wan. | **Net new** — opens user-supplied custom-control LoRAs to all OpenCut video models. New module `core/diffusion_templates.py`. |
| M9 | **StreamDiffusionV2** | Feb 2026 + ongoing | **MIT** | Training-free framework to convert any efficient video model into a live-streaming generator (< 40 ms / frame target). | **Net new — biggest UX leap available.** Enables real-time editor preview that competitors (CapCut / Runway / Captions) charge for. New `core/streaming_diffusion.py`. |
| M10 | **MirageLSD** (Decart AI) | Mar 2026 | Research preview | 24 FPS live diffusion at <40 ms/frame — first live-stream diffusion model. | Inspirational; not yet shippable. Watch. |

### 2.2 TTS / voice / dub

| # | Model | Date | Licence | Why it matters | Action |
|---|---|---|---|---|---|
| M11 | **OmniVoice** (k2-fsa) | 7 Apr 2026 | **Apache 2.0** | 600+ languages zero-shot TTS, diffusion-LM, 40× RT. | **Net new** — fills the long-tail language gap that F5/CosyVoice2/Edge-TTS miss. Add `core/tts_omnivoice.py`. (Wave H2.4 stub exists — fill it now with this checkpoint.) |
| M12 | **IndexTTS2** | 2026 | Apache-2 *(verify)* | Industrial zero-shot TTS with **explicit duration control** (critical for dubbing-to-timecode) + disentangled emotion/timbre. | **Net new** — duration control is the missing piece for `dub_pipeline.py` to land translated audio that fits original cuts without re-timing. |
| M13 | **Qwen3-TTS (0.6B / 1.7B)** | Jan 2026 + ongoing | **Apache 2.0** | 97 ms ultra-low-latency streaming, 10 langs + dialect voices, voice clone + design. | **Complements** existing TTS stack; great for real-time preview. |
| M14 | **VoxCPM2** (OpenBMB) | 11 Apr 2026 | **Apache 2.0** | Tokenizer-free TTS, 2B, 30 langs, 48 kHz, voice design + cloning. | **Complements** F5-TTS / Chatterbox — cleaner architecture. |
| M15 | **Fish Speech S2 Pro (4B + 400M Dual-AR)** | 9-10 Mar 2026 | Apache-2 *(verify)* | < 150 ms latency, 80+ langs, natural-language emotion tags, multi-speaker single-pass. | **Supersedes** older Fish Speech (if currently in OpenCut); rivals F5-TTS. |
| M16 | **HunyuanVideo-Foley** (Tencent) | Aug 2025 + active 2026 | Tencent licence (verify) | End-to-end Text-Video-to-Audio, 48 kHz, frame-accurate footsteps/glass/etc. on 100k-hour curated dataset. | **Net new foley capability** — fills the gap between AudioGen (text → SFX) and MusicGen (text → music). New `core/foley_hunyuan.py` if licence permits. |
| M17 | **Mimi codec** (Kyutai) | continued 2026 | **Apache 2.0** | 12.5 Hz, 1.1 kbps streaming neural codec; basis of Moshi, Sesame CSM, VoXtream, LFM2-Audio. | **Net new audio I/O tier** — useful for browser preview, low-bandwidth review bundle exports. |
| M18 | **DriftSE** (speech enhancement) | Interspeech 2026 | Research; code TBD | One-step generative speech enhancement beating multi-step diffusion baselines. | Watch for code; potential **supersede** for DeepFilterNet / Resemble Enhance. |
| M19 | **PrismAudio** (FunAudioLLM) | ICLR 2026, Feb 2026 | Apache-2 code, **research-only weights** | 518M V2A, 9 s audio in 0.63 s, decomposed CoT over semantic/temporal/aesthetic/spatial. | Watch; not shippable yet (weights). |
| M20 | **Mistral Voxtral TTS (4B)** | 26 Mar 2026 | **CC-BY-NC 4.0** | 3-second voice cloning, 9 langs, beats ElevenLabs Flash v2.5. | **Skip until relicensed.** Track. |

### 2.3 Restoration, upscaling, depth, tracking

| # | Model | Date | Licence | Why it matters | Action |
|---|---|---|---|---|---|
| M21 | **SAM 3 / SAM 3.1** (Meta) | ICLR 2026; 3.1 on 27 Mar 2026 | Apache-2 (Meta SAM precedent) | Text + exemplar prompts, concept-aware tracking, real-time multiplexing. | **Supersedes SAM 2** in `core/object_removal.py` and downstream Cutie / DEVA tracking surfaces. |
| M22 | **Depth Anything 3** (ByteDance Seed) | ICLR 2026 (paper Nov 2025), active 2026 | Apache-2 *(per repo)* | Spatially consistent geometry from arbitrary views, w/ or w/o known camera poses. | **Supersedes Depth Anything V2 + Apple Depth Pro** for multi-view/parallax workflows. Update `core/depth_effects.py` + `core/depth_depthpro.py`. |
| M23 | **Online Video Depth Anything** | late 2025 + 2026 | Apache-2 | Temporally consistent depth with low memory — important for editor-loop preview. | **Complements** Depth Anything 3 for streaming. |
| M24 | **MatAnyone 2** | Mar 2026 | **NTU S-Lab 1.0** (research, non-commercial) | CVPR 2026 Highlight. Mask-propagation video matting w/ learned quality evaluator; pair with SAM 3 for prompted init. | **Cannot ship commercially** — keep for research/eval only. RVM remains the production matte. |
| M25 | **SeedVR2 / SeedVR v2.5** (ByteDance Seed) | ICLR 2026; v2.5 in 2026 | **Apache 2.0** | One-step diffusion video restoration; arbitrary resolution; head-to-head with FlashVSR. | **Complements** FlashVSR — already roadmapped in Wave S; ship as second backend in `upscale_hub.py`. |
| M26 | **FlashVSR v1.1** | Nov 2025 stable; CVPR 2026 | **Apache 2.0** | 17 FPS @ 768×1408 on A100, ~12× faster than prior one-step diffusion VSR. | OpenCut already integrated (`core/upscale_flashvsr.py`); pin to **v1.1** checkpoint. |

### 2.4 Standards / spec

| # | Spec | Date | Status | Action |
|---|---|---|---|---|
| M27 | **C2PA Content Credentials 2.3** | 2026 | Open spec, GA | F110 currently emits 2.x sidecars. **Bump to 2.3.** Adds: live-video provenance, plain-text/OGG/large-AVI manifests, finer-grained edit logs, cloud-anchored trust lists. New F145. |
| M28 | **IMSC Text Profile 1.3** | 3 Apr 2026 Candidate Recommendation | W3C | Text-only IMSC profile advancing to W3C Rec; hard requirement for Netflix/OTT delivery. **Net new emit format** alongside WebVTT/SRT/ASS. New F146. |
| M29 | **OpenColorIO 2.5 + ACES 2.0 built-in configs** | Sep 2025; in VFX Reference Platform 2026 | BSD-3, GA | ACES 2.0 Output Transforms shipped as OCIO views; Vulkan GPU renderer. F109 currently uses OCIO 2.4 / ACES 1.3. **Bump.** New F147. |

### 2.5 Agentic / MCP / tooling

| # | Project | Date | Licence | Action |
|---|---|---|---|---|
| M30 | **VideoAgent** (HKUDS) | 2026 | MIT/Apache *(verify)* | 0.95 workflow-composition success across LLM backbones. **Reference architecture** for OpenCut's planned chat conductor (F131). |
| M31 | **FireRed-OpenStoryline** (Xiaohongshu) | active 2026 | Apache-2 | LLM planner + MoviePy/FFmpeg + Style Skills + **Claude Code Skills** + MCP integration. Reference for F132 Skills SDK. |
| M32 | **vibeframe** | v0.104.3, May 2026 | MIT | CLI-first, multi-provider model bus, MCP-ready. Reference for F131/F132 chat conductor + skills. |

---

## 3. Datasets relevant to evaluation harness (F120)

OpenCut's `ai_eval_harness.py` records latency + quality + environment per feature. The harness currently has its own sample media. The following public datasets would strengthen evaluation:

| Dataset | Use | Licence |
|---|---|---|
| **VBench** (Tsinghua) | Video generation benchmark — covers motion coherence, subject consistency, scene consistency, overall quality | Open |
| **Spring** | Optical flow benchmark — for SEA-RAFT / GIMM-VFI evaluation | CC-BY 4.0 |
| **VFI-2024** | Frame interpolation benchmark — for RIFE / PerVFI / GIMM-VFI | Open |
| **DAVIS-2017** | Video object segmentation — for Cutie / DEVA / SAM 3.1 | Open |
| **DEVIS / MOSE** | Multi-object segmentation | Open |
| **REDS** | Video restoration (denoise / deblur / SR) — for VRT / RVRT / DiffBIR / SeedVR2 / NAFNet | Open |
| **YouTube-VOS** | Video object segmentation | Open |
| **Vimeo-90k septuplet** | Frame interpolation + VSR | Open |
| **LRS3-TED** | Talking-head / lip-sync — for daVinci-MagiHuman / EchoMimic / MuseTalk | Open for research |
| **VoxCeleb 2** | Speaker / voice clone | CC-BY |
| **LibriTTS** | TTS quality | CC-BY |
| **AudioCaps** | Text-to-audio | Open |
| **MUSDB18** | Stem separation — already standard for Demucs / BS-RoFormer | Open |
| **EBU SQAM** | Audio quality | Open |
| **EBU R128 reference set** | Loudness | Open |
| **Netflix open content** (e.g. Meridian, Cosmos Laundromat) | End-to-end delivery QC test | CC |
| **C2PA test vectors** | Provenance sidecar validation | Open |
| **IMSC reference content** | Caption rendering compliance | W3C |

**Action:** F120 harness already exists. Add **dataset-bundle download manifest** under `opencut/data/eval/` with checksum + licence per dataset. New small F: **F148 — public evaluation dataset bundle**, gated by `OPENCUT_DOWNLOAD_EVAL=1`.

---

## 4. External APIs / integrations

### 4.1 Already in OpenCut
- **Ollama** (local LLM) — `core/llm.py`
- **OpenAI** — `core/llm.py`
- **Anthropic** — `core/llm.py` (API version `2023-06-01` per MODERNIZATION.md — should bump to `2023-10-01` minimum, or align with current Anthropic SDK behaviour)
- **Gemini** — added per MODERNIZATION.md 2026-04-06 update
- **YouTube / TikTok / Instagram** — `core/social_post.py` OAuth direct posting
- **Frame.io** — none; reviewed and rejected in favour of local F088 review bundles
- **ElevenLabs** — `core/tts_elevenlabs.py` (Wave L)
- **HuggingFace Hub** — model download via `transformers` / `diffusers` / direct
- **GitHub Releases** — `core/changelog_feed.py` (F1.4) for self-update notifications
- **GitHub Gists** — `core/gist_sync.py` (F1.7) for preset sync
- **DaVinci Resolve** — `core/resolve_bridge.py` Python scripting

### 4.2 New since last review

| API / integration | Why interesting | Action |
|---|---|---|
| **Descript API** (open beta 2026-05-14) | Could expose Underlord-equivalent edits as an external connector; closed-source | Watch only; do not depend |
| **Adobe Premiere UXP Hybrid Plugins** (Apr 2026, Premiere 26.2+) | Native C++ DLL / dylib inference inside the panel; could host OpenCV / TFLite / hardware SDKs without Python sidecar | **Strategic** — evaluate whether a "OpenCut UXP Hybrid Plugin" sidecar could shave end-to-end latency on common Premiere-to-AI roundtrips. New F: **F149 — UXP Hybrid Plugin sidecar RFC**. |
| **a16z agentic-video editing trends** | Reference for `/agent/chat` UX patterns | Reference only |
| **MCP server registry** (`modelcontextprotocol/servers`) | OpenCut MCP could be listed | New F: **F150 — register `opencut-mcp-server` upstream**. |

---

## 5. Model card / licence hygiene

F115 ships 47 model cards. New model surfaces from this audit should each get a card:

- daVinci-MagiHuman (Apache-2, 15B, GPU heavy) — high priority
- LTX-Video 2.3 (Apache-2, 22B, GPU heavy)
- Wan 2.7 (Apache-2 *pending weights upload verification*)
- HunyuanVideo-1.5 (Tencent Community Licence — *territory restrictions, flag*)
- OmniVoice (Apache-2)
- IndexTTS2 (verify licence)
- VoxCPM2 (Apache-2)
- Qwen3-TTS (Apache-2)
- Fish Speech S2 Pro (verify licence)
- HunyuanVideo-Foley (Tencent licence — verify)
- Mimi codec (Apache-2)
- SAM 3 / SAM 3.1 (Apache-2)
- Depth Anything 3 (Apache-2 per repo)
- Online Video Depth Anything (Apache-2)
- SeedVR / SeedVR2.5 (Apache-2)
- StreamDiffusionV2 (MIT)
- DiffSynth-Studio (Apache-2)
- MatAnyone 2 (**NTU S-Lab 1.0 — research-only**, mark accordingly)

**Action:** new F: **F151 — extend model_cards.py with 2026-Q2 model surfaces** + regenerate `docs/MODELS.md`. The F115 dump script (`opencut/tools/dump_model_cards.py --check`) will catch drift in CI.

### Explicit skip list (do not ship; track for future relicense)

- **Mistral Voxtral TTS** — CC-BY-NC 4.0
- **MatAnyone 2** — NTU S-Lab 1.0 (non-commercial)
- **PrismAudio** — research-only weights
- **HunyuanVideo / HunyuanVideo-1.5** — Tencent territory carve-outs (EU/UK/KR)
- **Wan 2.5 / 2.6** — closed API
- **MirageLSD** — research preview; weights status unclear
- **Seedance 2.0**, **Kling 3.0**, **Veo 3.1**, **Sora 2** — closed source
- **VoiceCraft**, **SeamlessExpressive**, **Co-Tracker3**, **SUPIR**, **Hallo2**, **LivePortrait** (weights), **MuseTalk** (weights), **AudioCraft JASCO** (weights), **ChatTTS** (AGPL-3), **DAC** (no user-visible feature yet) — all listed in ROADMAP-NEXT.md Wave K rejects; status unchanged.

---

## 6. AI evaluation harness (F120) — recommended additions

F120 currently records latency + quality + environment per call, stores under `~/.opencut/ai_eval/<feature_id>.json` (capped 200 entries).

Recommended extensions:

1. **Per-model VRAM peak** capture via `torch.cuda.max_memory_allocated()` — useful for F063 cost estimator.
2. **Reference dataset score** column (when input matches a known eval clip) using golden outputs.
3. **Backend selection telemetry** — when `upscale_hub.py` dispatches Lanczos because FlashVSR is unavailable, record both the choice and the reason.
4. **Cross-backend comparison** report endpoint: `GET /system/ai-eval/compare?feature_id=tts&inputs=N` — runs same N inputs through every available backend, returns aligned latency/quality.

These are small. Bundle as a single follow-up F: **F152 — AI eval harness v2**.

---

## 7. Recommended priority for AI/integration items

Mapped into the new F-number plan for `PRIORITIZATION_MATRIX.md`:

**Now (high impact, low-medium effort)**
- F133 caption translation (NLLB-200, low effort, high commercial parity)
- F135 slate ID (Florence-2 already installed; small fill of K3.5)
- F145 C2PA 2.3 bump (small)
- F146 IMSC 1.3 emit (small)
- F147 OCIO 2.5 / ACES 2.0 bump (medium)
- F148 eval dataset bundle (small, gates future model claims)
- F151 model_cards.py 2026-Q2 sweep (small)
- F152 AI eval harness v2 (small)

**Next (medium effort, strong leverage)**
- F131 `/agent/chat` conductor (largest leverage; flagship)
- F132 Skills SDK + MCP packaging
- F138 StreamDiffusionV2 real-time preview (biggest UX leap)
- F134 face age transformer (DaVinci parity)
- F137 CineFocus (DaVinci parity; depth pipeline ready)
- F139 eye contact correction
- F140 Overdub conductor (voice clone + lip-sync + crossfade)
- F141 trailer / promo generator
- F142 VidMuse video-to-music
- F143 OpusClip engagement heatmap + A/B variants

**Later (large effort or dependent)**
- F136 IntelliScript screenplay→sequence (depends on caption QC + transcript fidelity)
- F144 Motion Brush
- F149 UXP Hybrid Plugin sidecar RFC
- M1 daVinci-MagiHuman integration (GPU-heavy; gate on Wave R/T plan)
- M2 LTX-Video 2.3 integration (supersede existing LTX)
- M3 Wan 2.7 integration (gate on weights publication)

**Under consideration**
- F150 register MCP server upstream

**Rejected / paused (licence or competing pri)**
- Voxtral, MatAnyone 2 (commercial), HunyuanVideo-1.5 default-on (territory)

---

## 8. What this file is *not*

- Not a re-litigation of the existing 47 model cards — see `docs/MODELS.md`.
- Not a re-listing of Wave A-K AI items — see ROADMAP-NEXT.md and ROADMAP.md.
- Not a benchmark table for the existing eval harness — see `~/.opencut/ai_eval/`.

Everything above is **delta** vs the team's 2026-05-16 research run.
