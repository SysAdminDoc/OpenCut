# OpenCut — Competitor & Adjacent-Project Matrix

**Audit date:** 2026-05-17
**Baseline:** OpenCut v1.32.0
**Sources:** Premiere/UXP/competitor research run (subagent), `research.md`, `AUDIT.md`, ROADMAP.md v4.3 (live competitive analysis), Adobe / Blackmagic / Descript / CapCut / Submagic / OpusClip / Captions.ai / HeyGen / Topaz / Runway / Pika / Kapwing / VEED / Filmora release pages, May 2026.

---

## 1. Direct Premiere extension competitors (panels / sidecars)

| Project | License | Distribution | Surface | Notable in 2026 | OpenCut delta |
|---|---|---|---|---|---|
| **AutoCut** | Commercial $14.99/mo | CEP panel | Silence removal, podcast multicam, best-take selection (3.0) | AutoCut 3.0 added AI best-take selection | OpenCut already exceeds in breadth + free; AutoCut wins on **timeline-native speed** (direct ExtendScript writes vs OpenCut's export/reimport-and-marker model). |
| **AutoPod** | Commercial $29/mo | CEP panel | Podcast multicam by speaker, audio leveling | Premiere Essential Graphics caption insertion (2026) | OpenCut has multicam via `multicam_xml.py` but lacks **native Essential Graphics caption insertion** — only burn-in or sidecar SRT. |
| **AdobePremiereProMCP** (ayushozha) | MIT | Standalone MCP, polyglot (Go + Rust + TS + Py) | **1,060 Premiere RPC tools** over stdio JSON-RPC | v0.1.0 Mar 2026; active | OpenCut shipped its own MCP sidecar in v1.30.0 (27 tools). AdobePremiereProMCP is **CEP-bound** — will break at CEP EOL (Sept 2026). OpenCut's MCP server has the same CEP dependency for Premiere-side ops but the OpenCut Python backend is independent. |
| **premiere-pro-mcp (ppmcp)** (leancoderkavy) | MIT | stdio + HTTP/SSE | 269 tools across 28 modules | Active; ships remote at `premiere-pro-mcp.fly.dev`; supports PPro 2020–2025+ | Same CEP-bound EOL risk. Strategic: ship an MCP-over-UXP transport before competitors do. |
| **Adobe_Premiere_Pro_MCP** (hetpatel-11) | MIT | stdio | 97 tools (43 live-validated) | March 2026 validation sweep | Smallest; reference implementation. |
| **PremiereRemote** (sebinside) | MIT | Local HTTP | ExtendScript bridge | 80 stars, active May 2026 | Pure ExtendScript bridge, no AI features — OpenCut superset. |
| **pymiere** | GPL-3 | Python | Premiere automation API | Active maintenance | GPL contaminates MIT integration; reference only. |

**Strategic implication:** every "MCP for Premiere" competitor today is CEP-bound. The first project to ship a UXP-native MCP (or a CEP+UXP bilingual one) wins the post-Sept-2026 landscape. OpenCut's MCP sidecar is already standalone Python — pivot the Premiere ops to UXP messaging and that bridge survives EOL.

---

## 2. Open-source NLEs (full editors)

| Project | License | Activity | Direction | Lesson |
|---|---|---|---|---|
| **Shotcut** (mltframework) | GPL-3 | Active (13,953 ⭐) | MLT-based desktop editor | Mature OSS wins on dependable export/proxy/device workflow. |
| **Kdenlive** | GPL-3 | Active (5,047 ⭐) | KDE + MLT | Polished timeline/effects, subtitle and proxy depth. |
| **OpenShot** | GPL-3 | Active (5,765 ⭐) | Python + libopenshot | Hundreds of open issues — stability gap is OpenCut's opportunity. |
| **Olive Editor** | GPL-2 | Slow (9,027 ⭐, last push Dec 2024) | Node + compositor | Stalled rewrite — cautionary tale on scope. |
| **Flowblade** | GPL-3 | Active (3,051 ⭐) | Linux-focused MLT | Stable timeline editing for power users. |
| **Pitivi** | LGPL-2.1 | Slow (239 ⭐) | GStreamer | Pipeline correctness lessons. |
| **Blender VSE** | GPL-3 | Active (18,411 ⭐) | Blender video sequence editor | Power users tolerate complexity if scripting + colour are strong. |
| **Natron** | GPL-2 | Slow (5,365 ⭐) | Node/VFX | High maintenance burden. |
| **LosslessCut** | GPL-2 | Active (40,461 ⭐) | FFmpeg lossless ops | Fast, safe ops dominate a category. **Top issues**: overwrite warning, CSV import, ARM64 packaging, segment colour, cursor ergonomics — all addressable for OpenCut F102/F101/F103/F105/F119. |
| **OpenMontage** | AGPL-3 | Active (3,747 ⭐) | Multi-pipeline production | Architecture reference only — AGPL blocks code reuse. |
| **AutoClip** (zhouxiaoka) | MIT | Active (5,230 ⭐) | Highlight clipping / auto-montage | Demand validation. |

OpenCut explicitly **rejected** the full-NLE replacement angle (ROADMAP F078). The companion-to-Premiere role is the OpenCut moat.

---

## 3. Adjacent OSS automation (programmatic video)

| Project | License | Activity | Where it overlaps OpenCut |
|---|---|---|---|
| **auto-editor** | MIT | Active (4,286 ⭐, Nim rewrite at v30) | Silence/speech-driven cuts. OpenCut already wraps the Nim binary via `core/auto_edit.py`. |
| **editly** | MIT | Active (5,409 ⭐) | Scriptable video composition. OpenCut's `declarative_compose.py` is editly-inspired. |
| **Remotion** | MIT | Very active (47,048 ⭐) | Programmatic video rendering with strong docs + template ecosystem. Reference for OpenCut's future template/preset format. |
| **MoviePy** | MIT | Active (14,613 ⭐) | Python video composition. OpenCut has the equivalent capability through FFmpeg builders. |
| **OpenTimelineIO** | Apache-2 | Active (1,863 ⭐) | Interchange. OpenCut exports + imports + diffs OTIO. **Adapter package split** in 0.17 — see SECURITY review. |
| **MLT** | LGPL-2.1 | Active (1,781 ⭐) | Headless render engine. OpenCut explicitly evaluated and rejected MLT backend (Wave C C1) for now. |
| **GStreamer** | LGPL-2.1 | Active (3,167 ⭐) | Pipeline architecture reference. |
| **FFmpeg** | LGPL-2.1 / GPL-2 | Active (60,167 ⭐) | Core dependency. 8.1 in March 2026. |
| **PySceneDetect** | BSD-3 | Active (4,812 ⭐) | Scene detection. OpenCut wraps it. 0.7 released May 2026. |
| **VapourSynth** | LGPL-2.1 | Active (2,023 ⭐) | Scripted video processing. OpenCut Wave E2 stub. |
| **Bolt UXP** | MIT | Active (166 ⭐) | UXP build system + **WebView UI (March 2026)** — critical for CEP→UXP migration of 8500-line vanilla JS panel. |
| **Bolt CEP** | MIT | Active (482 ⭐) | CEP build system reference. |

---

## 4. Commercial closed-source AI video tools

| Product | Pricing | What they ship | OpenCut gap |
|---|---|---|---|
| **Descript** + **Underlord** AI agent | $24/mo Creator | Transcript editing native, Overdub voice clone, Eye Contact, Studio Sound, AI "Underlord" agent that does multi-step edits from chat. **2026**: post-turn self-review (agent diffs its output vs request before commit). | OpenCut has the building blocks (transcript, voice clone, lip-sync stubs) but no **conductor chat agent with self-verification**. F125-class work. |
| **Captions.ai** (Mirage) | $24+/mo | Chat-driven editing, eye contact correction, AI avatar (Spokesperson), 30+ language dub | OpenCut has dub pipeline shipped (Wave M v1.30.0) but lacks chat-driven conductor + eye-contact module. |
| **Adobe Premiere 26.x** | CC subscription | Generative Extend, Media Intelligence (semantic search), AI Object Mask, Enhanced Speech, Frame.io Drive, Sequence Index panel, Color Mode workspace, native ARM64 | Premiere's UXP **caption-track API gap** is a blocker for everyone, not just OpenCut. **Generative Extend** is competition for OpenCut's outpaint stub. |
| **DaVinci Resolve 21** | Free + Studio $295 | AI IntelliSearch, Face Age Transformer, Face Reshaper, Blemish Removal, CineFocus, IntelliScript (screenplay → timeline), Slate ID, AI Magic Mask renderable as external matte, AI Motion Deblur, AI Speech Generator | OpenCut has stubs for most (K2.19 CineFocus, K3.4 Face Age, K3.5 Slate ID, K3.3 IntelliScript) but **no AI Face Reshaper, no AI Blemish Removal, no AI Speech Generator equivalent yet**. Wave L shipped face reshape (`face_reshape.py`) and skin retouch (`skin_retouch.py`) — these are real, not stubs. ✓ |
| **CapCut** (desktop + mobile) | Free / $7.99 Pro | Massive template library, AI Reframe 2.0 multi-subject, AI Video Enhance one-click, voice-to-avatar, AI Eraser, Long-to-Shorts, integrated stock media | OpenCut lacks **template marketplace** (F033 Later), animated caption style library is thinner. |
| **Submagic** | $16-25/mo | Captions, hooks, B-roll, zooms, sound effects, templates | OpenCut Wave H shipped virality scoring, cursor-zoom; brand kit shipped K1.2; auto-hook + animated caption styles are gaps. |
| **OpusClip** | $19-49/mo | URL-in → 10 viral shorts out, virality score, A/B variants, engagement heatmap | OpenCut shipped virality (Wave H), sports highlights (Wave M). **A/B variants, engagement heatmap, auto-hook generation** are gaps. |
| **HeyGen** (Avatar IV) | $30-100/mo | Photoreal talking-head from photo + voice, 60+ langs dub | OpenCut Wave M shipped EchoMimic V3 lip-sync; **no full avatar-creation pipeline** (face 3D recon + voice clone + lip-sync composite). research.md §1.8 LOW priority due to licence + GPU. |
| **ElevenLabs** | $5-330/mo | Flash v2.5 TTS (75ms latency), 3000+ voices, 32 langs, IVC, Dubbing | OpenCut Wave L shipped ElevenLabs as optional cloud backend. ✓ |
| **Runway ML** | $12-76/mo | Gen-4 / Act-Two T2V, video outpainting, Motion Brush, multi-modal editing | OpenCut has T2V stubs (Wave K LTX-2, Open-Sora, Wan VACE) but lacks **Motion Brush equivalent** and **multi-modal in-frame editing**. |
| **Pika Labs 2.0** | $10-58/mo | Video outpainting, Pika Effects, Lip Sync | OpenCut has outpaint stub (K3.6); shipped lip-sync (Wave M EchoMimic V3). |
| **Topaz Video AI** | $199 perpetual | Multi-model upscale (Proteus/Iris/Nyx/Rhea/Gaia/Artemis/Theia), motion deblur, frame interp to 120fps, low-light Nyx | OpenCut shipped Upscale Hub dispatcher in Wave L; has stubs for motion deblur (K2.12). **No Topaz-quality low-light enhancement** beyond denoising. |
| **Veed / Kapwing** | $12-30/mo | Browser editing, captions, avatars, brand kit, collaboration | OpenCut deliberately offline — does not compete on collaboration. |
| **Filmora** | $50-80/yr | AI translation, music, thumbnails, masking, templates | Consumer-tier; OpenCut targets professional/Premiere users. |

---

## 5. Agentic / chat-driven editing systems

This is the **emerging** category and the highest-leverage gap.

| Project | Licence | UX | Approach |
|---|---|---|---|
| **Descript Underlord** | Commercial | Sidebar chat in Descript | Multi-step edits, post-turn self-review (added 2026) |
| **FireRed-OpenStoryline** (Xiaohongshu) | Apache-2 | CLI + web UI + **MCP + Claude Code Skills** | LLM planner orchestrates MoviePy/FFmpeg; Style Skills library; AI transitions (Apr 2026), ASR rough-cut (Mar 2026) |
| **vibeframe** (Vericontext) | MIT | CLI-first | Brief → STORYBOARD.md/DESIGN.md → MP4; multi-provider model bus (OpenAI/Anthropic/Google/Runway/Kling/FAL/ElevenLabs); v0.104.3 May 2026 |
| **VideoAgent** (HKUDS) | MIT/Apache | All-in-one agent | 0.95 workflow-composition success across LLM backbones; video understanding + editing + remaking |
| **ViMax** (HKUDS) | MIT/Apache | Script-to-video | Companion to VideoAgent; script → timeline |
| **Crayo.ai** | Commercial | Template-driven | Short-form template browser, not chat agent |
| **CrePal** | Commercial | "AI Video Creation Agent" | Marketing leader for agent-driven creation |
| **Odysser** | Commercial | Chat-agent UI | Captions / B-roll / brand styling for talking-head |

**OpenCut position:** the F101+ governance layer, MCP sidecar (v1.30.0), and 1,344 routes are the **foundation** for an agent layer. What's missing is the **conductor**:
1. A `/agent/chat` endpoint that accepts free-form text + timeline state.
2. An LLM-driven plan generator that maps intent → sequence of OpenCut API calls.
3. A **visible timeline diff** before commit (Underlord pattern).
4. A **post-turn self-review** (Underlord 2026 pattern) that diffs the executed plan against the request and re-invokes itself on out-of-scope edits.
5. A **Skills library** (FireRed pattern) — named reusable multi-step workflows (e.g. `polish_interview`, `cut_youtube_short`, `prep_podcast_episode`).

This is the single largest competitive gap and the highest-leverage roadmap item not yet in any wave or F number.

---

## 6. Mind-the-gap summary

| Capability | OpenCut state | Commercial parity | New F# proposal |
|---|---|---|---|
| MCP for Premiere | ✓ v1.30.0 (27 tools) | 1,060-tool competitor (CEP-bound; will break Sept 2026) | F130 — UXP MCP transport (post-CEP-EOL survival) |
| Chat-driven editing agent | ✗ | Descript, Captions.ai, FireRed, vibeframe | **F131** — `/agent/chat` conductor with timeline diff + post-turn self-review |
| Reusable agent skills library | ✗ (workflow presets exist but not LLM-skill format) | FireRed OpenStoryline | **F132** — Skills SDK + claude.ai Skill manifests + MCP tool packaging |
| Caption translation (standalone) | ✗ (full dubbing shipped) | Adobe Premiere 26, Captions.ai | **F133** — `/captions/translate` NLLB-200 SRT-in → SRT-out |
| AI Face Reshaper | ✓ Wave L (`face_reshape.py`) | DaVinci 21 | shipped |
| AI Blemish Removal | ✓ Wave L (`skin_retouch.py`) | DaVinci 21 | shipped |
| AI Face Age Transformer | stub K3.4 | DaVinci 21 | F134 — fill K3.4 with Cutie + IP-Adapter pipeline |
| AI Slate ID | stub K3.5 | DaVinci 21 | F135 — fill K3.5 (Florence-2 already installed) |
| IntelliScript (screenplay→sequence) | stub K3.3 + Wave I I2.1 | DaVinci 21 | F136 — fill K3.3 with FDX/Fountain parser |
| CineFocus rack focus | stub K2.19 | DaVinci 21 | F137 — fill K2.19 with Depth Pro pipeline |
| Real-time editor-loop preview | ✗ | CapCut, Runway, Captions | **F138** — StreamDiffusionV2 + Diffusion Templates integration (huge UX leap) |
| AI Eye Contact correction | ✗ (research.md §1.3) | Descript, Captions, NVIDIA Broadcast | F139 — MediaPipe face mesh + lightweight GAN |
| AI Overdub (voice replace) | ✗ (research.md §1.4) | Descript | F140 — voice-clone + lip-sync + audio crossfade conductor |
| Auto-trailer / promo generator | stub K3.2 | Descript Underlord | F141 — fill K3.2 |
| Video-to-music generation | ✗ (VidMuse partially shipped) | None commercial | F142 — VidMuse 2026 ckpt + AudioCraft MusicGen decoder |
| Engagement heatmap / A/B variants | ✗ | OpusClip | F143 — shorts pipeline A/B with virality re-ranking |
| Motion Brush (paint motion onto stills) | ✗ | Runway, CapCut | F144 — likely L-effort, defer until F138 lands |
| Template marketplace (curated, sandboxed) | F033 Later | CapCut massive library | hold |
| Mobile companion review/ingest | F042 Later | every commercial | hold |
| Multi-user collaboration | F088 Next (review bundles), F039 Under Consideration (live co-edit) | every commercial | F088 path is OpenCut-aligned |
| C2PA 2.3 export | ✓ F110 (sidecar) — needs upgrade to 2.3 spec | Adobe Premiere, Frame.io | **F145** — bump F110 to C2PA 2.3 (live-video provenance, plain-text/OGG/large-AVI manifests) |
| IMSC Text Profile 1.3 captions | ✗ | Netflix/OTT delivery | **F146** — IMSC 1.3 emit in captions QC pipeline (alongside WebVTT/SRT) |
| OCIO 2.5 + ACES 2.0 | F109 OCIO validator shipped | DaVinci, Resolve | **F147** — bump OCIO validator + LUT pipeline to OCIO 2.5 / ACES 2.0 |

---

## 7. Notable commercial paywalls that exist *because* the local-AI alternative isn't ready

- **HeyGen avatar** ($30-100/mo) — only because local avatar creation is GPU + licence-heavy.
- **DaVinci Studio CineFocus / IntelliScript / Face Age** — all stubs in OpenCut waiting for someone to land them.
- **OpusClip / Submagic** ($19-49/mo) — for hooks + virality scoring (Wave H shipped) and B-roll insertion (`broll_insert.py` exists, conductor doesn't).
- **Frame.io** ($15/mo+ per user) — for client review. OpenCut F088 + F105 portable review bundle is the local-first answer.
- **Descript Overdub** — voice-clone + lip-sync conductor; OpenCut has every piece.
