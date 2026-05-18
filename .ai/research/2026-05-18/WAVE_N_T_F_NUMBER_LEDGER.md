# OpenCut Wave N-T F-Number Governance Ledger

**Date:** 2026-05-18
**Closes:** F180
**Source scope:** `ROADMAP.md` Wave N, O, P, Q, R, S, and T feature tables as of roadmap v4.39.

F180 does not move every model-surface item into the F-number backlog. It makes the boundary explicit:

- **Wave IDs** remain the implementation owner for individual model surfaces, route families, and backend modules.
- **F-numbers** own cross-cutting governance: security, release trust, packaging, UXP migration, tests, documentation, dependency floors, eval claims, and licensing gates.
- Existing F-numbers win over duplicates. A wave row that is already covered by F143-F178, F251/F252/F260, or another governance item must not get a second F-number.
- A wave row with only model integration work stays wave-only, but its implementation must still satisfy the shared model-card, eval, license, optional-download, and release-smoke gates.

## Summary

| Bucket | Disposition |
|---|---|
| Existing F-number coverage | Reuse the existing F-number when a wave item is a direct implementation of a previously tiered governance or parity item. |
| Wave-only model surface | Keep the wave ID as the implementation backlog key; no new F-number is needed. |
| Wave-only with governance gate | Keep the wave ID, but require the existing shared gates (`F176`, `F177`, `F178`, release smoke, license gate, UXP parity checks) before the wave row can be called shipped. |
| Superseded by later wave row | Do not implement the older row as written; carry useful context into the newer wave row. |

## Canonical Disposition Table

| Wave ID | Feature | F-number disposition | F-tier after F180 | Rationale |
|---|---|---|---|---|
| N1.1 | FastVideo sparse distillation | Wave-only with `F178` eval gate | Next model surface | Backend acceleration for Wan routes; not a separate governance item. |
| N1.2 | LightX2V quantization + step distillation | Wave-only with `F178` eval gate | Next model surface | Quantized inference path; prove speed/quality via eval telemetry before marking shipped. |
| N2.1 | SAM 2.1 video object segmentation | Covered by `F162` | Now model upgrade | F162 promotes the segmentation default beyond this original SAM 2.1 row. |
| N2.2 | Depth-Anything-V2 depth maps | Covered by `F163` | Now model upgrade | F163 supersedes the V2 target with Depth Anything 3. |
| N2.3 | SAM2 + depth compositor pipeline | Wave-only; depends on `F162` and `F163` | Next model surface | Composition route is product work once the two upgraded primitives land. |
| N3.1 | CogVideoX-5B T2V + I2V | Wave-only with `F177`/`F178` gates | Next model surface | Second video-generation backend; no extra governance need beyond model card and eval. |
| N3.2 | Qwen2.5-VL smart timeline analysis | Superseded by S3.1 | Later only if compatibility alias is needed | Qwen3-VL is the newer default target; keep this as migration context only. |
| N3.3 | DiffRhythm+ music upgrade | Wave-only with `F178` eval gate | Next model surface | Drop-in music quality upgrade; no new F-number. |
| N3.4 | Sesame CSM-1B conversational speech | Wave-only with license/consent gate | Next model surface | Gated Llama-backbone acceptance is an implementation DoD, not a new F-number. |
| O1.1 | Dia 1.6B dialogue TTS | Wave-only; overlaps T2.3 | Next model surface | Preserve as a candidate engine inside the later dialogue-TTS dispatcher. |
| O1.2 | Parler-TTS natural language voice description | Wave-only | Next model surface | Voice-description TTS backend; model-card/eval gates suffice. |
| O2.1 | LTX-Video LTXV 0.9.8 T2V/I2V | Covered by `F164` | Next model upgrade | F164 is the canonical LTX upgrade track and should absorb this older LTXV row. |
| O2.2 | LTX-Video multi-keyframe storyboard-to-video | Covered by `F164` | Next model upgrade | Implement as part of the LTX upgrade capability surface. |
| O2.3 | LTX-Video LoRA fine-tuning pipeline | Wave-only with training safety gate | Later model surface | Fine-tuning is heavier than the core F164 generation upgrade. |
| O2.4 | LTX-Video video extension | Covered by `F164` | Next model upgrade | Extension is a direct LTX route capability. |
| O3.1 | YuE lyrics-to-full-song | Wave-only with model-card/eval gates | Next model surface | Song generation surface; no extra governance item. |
| O3.2 | GGUF quantization engine | Wave-only platform surface | Next model surface | Infrastructure feature, but implementation remains scoped to Wave O3.2 unless it changes global dependency policy. |
| O3.3 | Multi-GPU task scheduler | Wave-only platform surface | Next model surface | Scheduling feature belongs to the wave plan; release-smoke coverage is required when implemented. |
| P1.1 | ConsisID identity-preserving T2V | Wave-only with model-card/eval gates | Next model surface | Identity video backend; no separate F-number. |
| P1.2 | Allegro lightweight T2V + TI2V | Wave-only with model-card/eval gates | Next model surface | Lightweight video-generation backend; no separate F-number. |
| P2.1 | HiDream-I1 SOTA text-to-image | Wave-only with Llama license gate | Next model surface | Reuse the existing consent pattern; do not duplicate it as a new F-number. |
| P2.2 | HiDream-E1 instruction image editing | Wave-only with Llama license gate | Next model surface | Same gated-family handling as P2.1. |
| P2.3 | CogView4-6B bilingual text-to-image | Wave-only with model-card/eval gates | Next model surface | Bilingual T2I backend; no separate F-number. |
| P3.1 | Qwen2.5-Omni multimodal video narrator | Superseded by S3.1 unless narration-specific use remains | Later compatibility surface | Qwen3-VL becomes the newer VLM backbone; preserve narration semantics if needed. |
| P3.2 | Open-Sora 2.0 high-quality T2V | Wave-only with model-card/eval gates | Next model surface | Video-generation backend; no separate F-number. |
| P3.3 | UXP panel v1.0 milestone | Covered by `F146`, `F252`, and `F260` | Next UXP migration | UXP migration and risk tracking are already F-numbered. |
| Q1.1 | VACE all-in-one video compositing | Wave-only with `F178` eval gate | Next model surface | Compositing backend; route/eval gates cover the risk. |
| Q1.2 | VACE preprocessor toolkit | Wave-only; supports Q1.1 | Next model surface | Companion preprocessing module, not a governance task. |
| Q2.1 | CosyVoice 2.0 / Fun-CosyVoice 3.0 multilingual TTS | Wave-only with model-card/eval gates | Next model surface | Multilingual TTS surface; no separate F-number. |
| Q2.2 | MaskGCT zero-shot parallel TTS | Wave-only with model-card/eval gates | Next model surface | TTS backend; no separate F-number. |
| Q3.1 | OmniGen2 multi-reference in-context image generation | Wave-only with model-card/eval gates | Next model surface | Image-generation backend; no separate F-number. |
| Q3.2 | SkyReels V2 infinite-length T2V + I2V | Wave-only with model-card/eval gates | Next model surface | Video-generation backend; no separate F-number. |
| Q3.3 | SkyReels V3 talking avatar | Wave-only with model-card/eval gates | Next model surface | Avatar backend; no separate F-number. |
| R1.1 | EzAudio T2A sound-effects generation | Wave-only with model-card/eval gates | Next model surface | Foley/SFX backend; no separate F-number. |
| R1.2 | EzAudio audio inpainting (mask-region edit) | Wave-only; supports R1.1 | Next model surface | Companion route within the same EzAudio service. |
| R1.3 | EzAudio ControlNet reference-audio conditioning | Wave-only; supports R1.1 | Next model surface | Companion route within the same EzAudio service. |
| R2.1 | MuseTalk 1.5 audio-driven lip sync | Wave-only; feeds `F153` | Next model surface | Lip-sync engine can support the later overdub conductor but remains a model route. |
| R2.2 | VideoX-Fun camera-control I2V (trajectory) | Wave-only with model-card/eval gates | Next model surface | Camera-control video backend; no separate F-number. |
| R2.3 | VideoX-Fun structural control I2V (Canny / Depth / Pose / MLSD) | Wave-only with model-card/eval gates | Next model surface | Control-signal backend; no separate F-number. |
| R3.1 | Mochi-1 consumer T2V | Wave-only with model-card/eval gates | Next model surface | Consumer video backend; no separate F-number. |
| R3.2 | Step-Video-T2V-Turbo HPC T2V | Wave-only with explicit hardware gate | Later model surface | HPC-only backend must stay opt-in and capability-gated. |
| R3.3 | Step-Video-Ti2V (image-to-video companion) | Wave-only with R3.2 hardware gate | Later model surface | Companion route to the same HPC engine. |
| S1.1 | IC-Light V2 per-frame relight | Wave-only with model-card/eval gates | Next model surface | Relighting primitive; no separate F-number. |
| S1.2 | Light-A-Video training-free video relighting | Wave-only with model-card/eval gates | Next model surface | Video relighting route; no separate F-number. |
| S1.3 | DiffusionRenderer inverse + forward rendering | Wave-only with model-card/eval gates | Later model surface | Rendering research route; no separate F-number. |
| S2.1 | SeedVR2 one-step diffusion video super-resolution | Covered by `F174` | Next model upgrade | F174 is the canonical SeedVR/SeedVR2.5 upscale track. |
| S2.2 | NVIDIA Parakeet TDT 0.6B v2 streaming ASR | Wave-only with model-card/eval gates | Next model surface | ASR backend; no separate F-number. |
| S2.3 | NVIDIA Canary-1B-Flash batch ASR | Wave-only with model-card/eval gates | Next model surface | ASR backend; no separate F-number. |
| S2.4 | FFmpeg 8.0 native Whisper filter integration | Adjacent to `F129`; keep wave-only until FFmpeg bump lands | Later integration surface | Native filter adoption depends on the bundled FFmpeg upgrade gate. |
| S3.1 | Qwen3-VL multimodal upgrade | Wave-only; may depend on `F127b` | Next model surface | VLM upgrade should use the shared dependency-floor process if Transformers changes. |
| S3.2 | InternVL3 alternative VLM | Wave-only with model-card/eval gates | Next model surface | Optional vendor-diversity backend; no separate F-number. |
| S3.3 | face_reaging (FRAN reimplementation) face age transformation | Covered by `F148` | Next model upgrade | F148 is the canonical AI Face Age Transformer track. |
| S3.4 | HeartMuLa music generation | Wave-only with model-card/eval gates | Next model surface | Music backend; no separate F-number. |
| S3.5 | UXP panel v1.0 final EOL cutover | Covered by `F146`, `F252`, and `F260` | Next UXP migration | UXP default/cutover work belongs to the existing migration F-number set. |
| T1.1 | Skill registry + skill manifests | Covered by `F145` | Next flagship | F145 is the canonical Skills SDK + MCP packaging item. |
| T1.2 | Multi-pipeline orchestrator | Covered by `F143` and `F145` | Next flagship | Orchestration should ship with the conductor/skills track. |
| T1.3 | Style Skill library | Covered by `F145` | Next flagship | Style skills are part of the Skills SDK surface. |
| T1.4 | Live agent stats overlay (UXP panel) | Covered by `F143` track | Next flagship | Trust/trace UI should be implemented with the conductor route. |
| T1.5 | MCP skill exposure | Covered by `F145` | Next flagship | MCP skill exposure is explicitly part of the Skills SDK packaging work. |
| T2.1 | VoxCPM2 voice design + cloning | Covered by `F170` | Next model upgrade | F170 is the canonical VoxCPM2 backend item. |
| T2.2 | OmniVoice 600+ language zero-shot cloning | Covered by `F167` | Now model upgrade | F167 is the canonical OmniVoice item. |
| T2.3 | Dia + Higgs Audio V2 multi-speaker dialogue | Wave-only; feeds `F153` | Next model surface | Dialogue TTS can support overdub but remains a model-route implementation. |
| T2.4 | NeuTTS Air on-device TTS | Wave-only with license verification | Later model surface | Licence is not firm enough for an F-numbered priority. |
| T2.5 | Qwen3-TTS 3-second voice clone | Covered by `F169` | Now model upgrade | F169 is the canonical Qwen3-TTS item. |
| T2.6 | LongCat Adaptive Projection Guidance TTS | Wave-only with license verification | Later model surface | Keep on watch list until licence is confirmed. |
| T2.7 | uTalk AI positional audio panning | Wave-only with route/eval gates | Next model surface | Pure audio workflow; no new F-number. |
| T2.8 | AV-CASS audio-visual cinematic source separation | Wave-only with license verification | Later model surface | Research-stage dependency; keep gated. |
| T2.9 | Music Source Restoration BS-RoFormer + HiFi++ GAN | Wave-only with license verification | Later model surface | Restoration backend; wait for licence and weights clarity. |
| T3.1 | UniVidX unified video diffusion | Wave-only with license verification | Later model surface | Paper/code availability needs verification before implementation. |
| T3.2 | Any-to-Bokeh one-step video refocus | Covered by `F151` | Next model upgrade | F151 is the canonical CineFocus/rack-focus parity item. |
| T3.3 | Mono4DGS-HDR HDR upconversion from monocular video | Wave-only with model-card/eval gates | Later model surface | HDR/novel-view backend; no separate F-number. |
| T3.4 | Premiere Pro-style Generative Extend | Wave-only with model-card/eval gates | Next model surface | Commercial parity workflow built from wave backends; no separate F-number. |
| T3.5 | AV1 nano-restore filter | Wave-only with model-card/eval gates | Later model surface | Codec-restoration backend; no separate F-number. |
| T3.6 | DiffusionAsShader 3D-aware controllable video diffusion | Wave-only with license verification | Later model surface | Art-directable video backend; keep gated on licence confirmation. |

## Implementation Rules For Future Wave Work

1. When a wave item above says **Covered by F###**, update that F-number when implementing the wave item; do not create a duplicate checklist row.
2. When a wave item says **Wave-only**, use the wave ID in commit messages, tests, model cards, and changelog text.
3. Every model-surface implementation must update model cards (`F177`), eval/readiness metadata (`F178`), optional-download or consent gates, and release-smoke coverage appropriate to the route.
4. UXP-panel work must update the UXP migration artifacts (`F146`, `F252`, `F260`) instead of creating a parallel wave-only UI ledger.
5. If a future research pass materially changes a wave item's licence, dependency floor, or release trust profile, add or reuse an F-number for that cross-cutting risk before implementation.

