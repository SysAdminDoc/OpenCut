# Research — OpenCut

**Consolidated**: 2026-06-10 (refresh)
**Baseline**: v1.32.0 working tree — 1,539 routes (manifest verified), 602 core modules, CEP + UXP panels, DaVinci bridge, MCP server, MIT. Bundled FFmpeg 8.0.1-essentials.
**Methodology**: Full repo walk, verification of the 2026-06-09 pass against the working tree (several ROADMAP items closed by commits 7521f61/b7434f6/81a8a85/88bdf62 on 2026-06-10), dependency/lock audit, and ~30 external sources (Adobe platform, FFmpeg, CVE databases, competitor releases, model licensing). Actionable items live in ROADMAP.md ("Research-Driven Additions").

---

## Executive Summary

OpenCut is a local-first Python/Flask video-editing automation server driving Premiere Pro (CEP + UXP) and DaVinci Resolve. It is feature-saturated (profanity bleep, SeedVR2 upscaling, Parakeet ASR, lip-sync dubbing, generative extend, paper-edit/rough-cut are all already implemented) and the June 2026 hardening pass just closed the worst security/correctness audit items in the working tree. The dominant gap is now **shipping and visibility, not engineering**: the latest GitHub release is **v1.25.1 (2026-04-20)** while the tree is at v1.32.0 with an unreleased security pass. Top remaining opportunities, in priority order:

1. **Cut release v1.33.0** — 7 minor versions plus arbitrary-file-overwrite fixes (commit 88bdf62) are sitting unshipped (Verified).
2. **Brand/namespace decision** — carried from the prior pass, still unactioned and still gating all distribution work; opencut.app (48K stars) is mid-rewrite and will re-dominate news cycles when it relaunches (Verified).
3. **FFmpeg 8.1 upgrade** — D3D12 H.264/AV1 encoding (vendor-agnostic Windows HW encode) and Vulkan ProRes; `hw_accel.py` knows only nvenc/qsv/amf/videotoolbox (Verified).
4. **SAM 3 engine** — text-prompted ("concept") segmentation + tracking, commercial-permissive license; upgrades the SAM2 click-to-track object-removal flow (Verified license; integration Likely M-L effort).
5. **Multicam visual cues** — Resolve 20 SmartSwitch and Wraith use lip movement + shot type; `core/multicam.py` is diarization-only (Verified).
6. **Caption template breadth** — 6 animation presets vs Submagic 35+/CapCut 50+, while CapCut just paywalled auto-captions entirely (Verified).
7. **CapCut-refugee positioning** — CapCut Pro doubled to $19.99/mo (Jan 2026) with free-tier degradation; zero-feature-work distribution window (Verified).

## Product Map

- **Core workflows**: (1) silence/filler/repeat-take cutting → Premiere timeline write-back; (2) transcription → styled/animated captions → SRT/native tracks; (3) audio cleanup/stems/loudness/TTS/voice-clone; (4) AI video FX (upscale incl. SeedVR2, matte, depth, face, object removal); (5) shorts/highlights pipeline → social upload/scheduling; (6) OTIO/FCPXML interchange.
- **Personas**: solo YouTuber, podcaster, social-clip editor, Premiere/Resolve power users automating repetitive passes locally.
- **Platforms/distribution**: Windows installer (primary), macOS/Linux source launchers, Docker (+GPU), Flatpak/AppImage scripts. No PyPI/Homebrew/winget/Snap. GitHub: 20 stars, 0 open issues, releases stalled at v1.25.1.
- **Integrations/data flows**: panels ⇄ localhost:5679 HTTP/JSON (+ opt-in WS 5680, MCP 5681); ExtendScript/UXP bridge for sequence write-back; SQLite (jobs, FTS5 footage index) under `~/.opencut`; optional LLM (Ollama/OpenAI/Anthropic) and OAuth social upload.

## Competitive Landscape

- **Adobe Premiere 26 (rebranded from "Premiere Pro", Jan 2026)** — now ships native on-device Object Mask, Generative Extend, Media Intelligence search, and 90+ Film Impact transitions. *Learn*: Adobe is commoditizing several OpenCut differentiators; reposition README around what Adobe will never bundle free (silence-cut→timeline, stems, TTS/voice clone, local LLM agent, social pipeline). *Avoid*: head-to-head investment in features Premiere now bundles (generative extend, basic object masking).
- **opencut.app (48K stars)** — announced a ground-up TypeScript rewrite (May 2026) with plugin system, headless rendering, and MCP server; contributions paused. *Learn*: its rewrite will own another news cycle; the name decision can't keep slipping. *Avoid*: a browser-NLE rewrite (mission creep).
- **Descript Underlord** — 2026 added a public API, MCP-triggered automation, and an LLM model picker (incl. Claude Sonnet 4.5). *Learn*: OpenCut's MCP server + chat agent is the right architecture; expose a model picker and promote agent-driven workflows. *Avoid*: text-first editing as primary UI.
- **AutoCut / FireCut / Phantom Editor (Wraith)** — the paid Premiere-plugin set; user complaints center on pricing, regional affordability, setup friction, licensing/support failures. *Learn*: free + local + no-license is the durable wedge; Wraith's multicam quality bar (visual cues) is the feature to match. *Avoid*: nothing structural.
- **DaVinci Resolve 20** — AI IntelliScript (script→rough cut), Multicam SmartSwitch (audio + lip movement + shot type), vertical-res Smart Reframe. *Learn*: visual-cue multicam switching; OpenCut has `auto_rough_cut.py`/`paper_edit.py` but multicam ignores video. *Avoid*: competing as an NLE.
- **CapCut / Submagic** — CapCut Pro doubled to $19.99/mo (Jan 2026), free tier cut to 720p/watermarks, auto-captions paywalled; Submagic still leads template count (35+). *Learn*: capture switchers with a comparison table + template-pack expansion. *Avoid*: cloud-upload workflows.
- **LosslessCut (41K stars)** — unchanged lesson: distribution breadth (Flathub/Snap/Homebrew/winget) beats feature count for adoption.

## Security, Privacy, and Reliability

- **Fixed in working tree, NOT yet released** (Verified, commits of 2026-06-10/11): arbitrary-file-overwrite via unvalidated `output_path` sinks in 8 route modules (88bdf62); FFmpeg filter/concat escaping incl. Windows drive-colon and 32KB argv limit (b7434f6); transcription-timeout caller pinning, trailing-silence misclassification, stem-remix level loss (81a8a85); CEP cut-review `t()` shadow that broke the review panel entirely, job-done UI lock, Escape/Enter hijacks (7521f61); Docker remote-bind startup fix (5d5ee5d); UXP cancel-button loading-state recovery (8787ce6). **Shipping these is the single highest-leverage security action.**
- **Still open from existing ROADMAP** (spot-verified): CORS null-origin + unauthenticated CSRF token via /health; MCP HTTP transport auth; `sys.path[0]` insertion; drawtext escaping copies remain in 5 modules (`caption_styles.py:365`, `click_overlay.py`, `news_ticker.py`, `quiz_overlay.py`, `telemetry_overlay.py` — down from 10).
- **Dependency posture** (Verified against `requirements-lock.txt`/`pyproject.toml`): Flask 3.1.3 covers CVE-2026-27205; Werkzeug 3.1.7 covers CVE-2026-21860 and CVE-2026-27199; pyannote.audio `>=4.0` (Community-1, CC-BY-4.0) already pinned; Pillow `>=12.2`; torch `>=2.6`; standalone depth/model-loading extras require `transformers>=5.3`. **Documented exception**: the explicit `torch-stack` lane keeps `transformers>=4.30` because WhisperX 3.8.x requires `huggingface-hub<1.0.0`; `pyproject[all]` excludes that lane and the exception is advisory-gated. **Watch**: onnxruntime 1.26 (in development) hardens multiple OOB/overflow ops (Attention, MaxPoolGrad, SVM/TreeEnsemble, RNN); floor is `>=1.25,<2` across 15+ importing modules.
- **Compliance**: FCC caption display settings (47 CFR §79.103(e)) compliance date confirmed **2026-08-17** — the existing P1 "UXP FCC card non-functional" item should land before then.
- **Recovery/rollback**: unchanged from prior pass — SQLite WAL job persistence, startup interruption recovery, dry-run/backup contracts on destructive routes. No new gaps found.

## Architecture Assessment

- **Release process is the weakest boundary** (Verified): `CHANGELOG.md` carries a populated `[Unreleased]` section, version strings sit at 1.32.0, `build.yml` supports tag-triggered releases — yet no tag has shipped since v1.25.1 (April). This is process, not tooling; consider a release-cadence rule (every N merged fixes or 2 weeks, whichever first).
- **Tracked-doc integrity** (Resolved after refresh): README planning links now point at tracked root files, installer artifacts are described by the release-version filename pattern, and `ROADMAP.md` is the only active open-work tracker. Local ignored agent notes were corrected to point at `ROADMAP.md` / `RESEARCH.md` instead of the removed project-context file.
- **Refactor candidates** (carried, still valid): `client/style.css` (~18K lines, eight stacked re-skins — existing P2 item); `client/main.js` (~16.6K lines) unbundled despite `vite.config.js`.
- **Engine currency** (Verified): Parakeet TDT (`core/asr_parakeet.py`), Whisper turbo/distil (`security.py:35-40`), SeedVR2 (`core/upscale_seedvr2.py`), FlashVSR, pyannote 4.0 are all present — the prior pass's "engine gaps" are mostly closed. Remaining genuine gaps: SAM 3 (text-prompted segmentation; repo has SAM2 only) and FFmpeg 8.1 D3D12/Vulkan encoders (`core/hw_accel.py:70-78` lists only nvenc/qsv/amf/videotoolbox; bundled binary is 8.0.1-essentials).
- **Test/docs**: coverage ~54% un-gated; function-scoped app fixture slowness (existing P3). Issue templates now exist (`.github/ISSUE_TEMPLATE/`) — prior-pass gap closed.

## Rejected Ideas

| Idea | Reason | Source |
|---|---|---|
| Standalone browser/web NLE vs opencut.app | Mission creep; moat is NLE automation | opencut.app rewrite announcement |
| Full rebrand away from "OpenCut" | Qualified distribution name + tagline suffices | prior pass, upheld |
| SeedVR2 backend addition | **Already implemented** (`core/upscale_seedvr2.py`) | this pass, code scan |
| Parakeet ASR engine addition | **Already implemented** (`core/asr_parakeet.py`) | this pass, code scan |
| Profanity bleep/censor (AutoCut parity) | **Already implemented** (`core/profanity_bleep.py`, `profanity_censor.py`) | this pass, code scan |
| IntelliScript-style script→rough-cut | Substantially covered by `auto_rough_cut.py`/`paper_edit.py`; revisit only after multicam visual cues land | Resolve 20 / code scan |
| Depth Anything 3 Large as depth engine | Large weights CC-BY-NC-4.0 (smaller variants need per-size license verification before any work) | github.com/ByteDance-Seed/Depth-Anything-3 |
| Competing with Premiere 26 native Generative Extend/Object Mask | Adobe bundles them free with the host app; keep OpenCut's versions as fallbacks, don't invest further | Adobe Jan 2026 release |
| UXP Hybrid (C++) addon now | Validator/scaffold exists (RA-12); no perf case yet justifies native code | UXP changelog v26.2 |
| FastAPI migration / Electron wrapper / mobile app | Unchanged: overhead without benefit at localhost scale | prior pass |
| Hard Sept-2026 CEP cutover deadline | Adobe states multi-year runway; CEP 12 is last major but still serviced | Adobe community/tech blog |

## Sources

Adobe platform
- https://blog.developer.adobe.com/en/publish/2025/12/uxp-arrives-in-premiere-a-new-era-for-plugin-development
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://community.adobe.com/t5/premiere-pro-discussions/is-cep-still-supported/td-p/14093791
- https://blog.adobe.com/en/publish/2026/01/20/new-ai-powered-video-editing-tools-premiere-major-motion-design-upgrades-after-effects
- https://petapixel.com/2026/01/20/rebranded-adobe-premiere-26-arrives-with-one-click-object-tracking/

FFmpeg / engines / models
- https://linuxiac.com/ffmpeg-8-1-brings-vulkan-compute-codecs-and-new-decoder-support/
- https://9to5linux.com/ffmpeg-8-1-hoare-multimedia-framework-brings-d3d12-h-264-av1-encoding
- https://blog.roboflow.com/what-is-sam3/
- https://www.marktechpost.com/2025/11/20/meta-ai-releases-segment-anything-model-3-sam-3-for-promptable-concept-segmentation-in-images-and-videos/
- https://github.com/ByteDance-Seed/Depth-Anything-3
- https://github.com/IceClear/SeedVR2
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://github.com/SYSTRAN/faster-whisper

Security
- https://github.com/advisories/GHSA-69w3-r845-3855 (CVE-2026-1839)
- https://pluto.security/blog/unauthenticated-remote-code-execution-in-huggingface-transformers-via-config-injection/ (CVE-2026-4372)
- https://www.sentinelone.com/vulnerability-database/cve-2026-27205/ (Flask)
- https://www.sentinelone.com/vulnerability-database/cve-2026-21860/ (Werkzeug)
- https://github.com/microsoft/onnxruntime/releases
- https://www.fcc.gov/consumer-governmental-affairs/commission-announces-effective-date-closed-captioning-display-settings-rule

Competitors / market
- https://github.com/OpenCut-app/OpenCut
- https://opencut.app/roadmap
- https://www.descript.com/underlord
- https://descript.canny.io/changelog
- https://www.autocut.com/en/pricing/
- https://www.producthunt.com/products/firecut-ai/reviews
- https://phantomeditor.video/blog/best-ai-plugins-adobe-premiere-pro-2026
- https://www.cined.com/davinci-resolve-20-released-with-handful-of-ai-assisted-features/
- https://www.eesel.ai/blog/capcut-pricing
- https://github.com/SysAdminDoc/OpenCut (releases/stars state)

## Open Questions

1. **Brand decision** (carried, blocks distribution): will the maintainer ship under a de-conflicted distribution name before opencut.app's rewrite relaunches?
2. **UXP caption-track write API** (carried): still no documented create/import caption-track API in the Premiere UXP changelog through v26.2 — blocks full F252 caption roundtrip.
3. **Release cadence intent**: is the release stall (v1.25.1 vs v1.32.0) deliberate (waiting on macOS notarization F202?) or drift? Determines whether the v1.33.0 item ships immediately or waits on external gates.
4. **Depth Anything 3 small/base weight licenses**: per-size verification needed before any DA3 work (Large is CC-BY-NC).
