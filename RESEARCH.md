# Research — OpenCut

_Last pass: 2026-06-14 (second pass, same day). Supersedes the earlier 2026-06-14 pass;
its conclusions remain valid and are carried forward below. Items that shipped between the
2026-06-13 and 2026-06-14 passes (CI dep floors, PyInstaller pin, Python 3.13 matrix,
local-only mode, auto-editor v30, SAM 3, SigLIP visual-search registry, caption font-fallback,
multicam audio+visual, SQLite WAL hardening) are recorded in CHANGELOG/git, not repeated here._

## Executive Summary
OpenCut is a local-first Python/Flask automation server (1,539 registered routes, 107
blueprints, 605 core modules, ~10,200 tests) plus Adobe Premiere CEP/UXP panels. Its moat is
breadth-with-locality: no commercial or OSS competitor covers the same automation surface while
running entirely on-device, and the accessibility surface (photosensitive flash check to
ITU-R BT.1702-3, Microsoft-pattern audio-description drafting, RTL/CJK caption fallback) is
already more complete than most paid tools. The codebase is healthy; correctness/dependency
findings from prior passes are fixed. The highest-value direction now is **converting the large
strategic-stub surface (145+ `NotImplementedError` core stubs, ~22 routes returning HTTP 501)
into shipped capability where the license is clean and the stub already exists** — starting with
the three areas where OpenCut has a built pipeline but a stubbed final stage: ASR (Whisper works,
but the faster/more-accurate Parakeet/Canary stubs are dark), dubbing (full transcribe→translate→
TTS pipeline ships, but visual lip-sync is stubbed), and relighting (IC-Light stub is mis-aimed
at a non-commercial model).

Top opportunities, in priority order:
1. Floor-pin transitive web deps in `pyproject.toml` (werkzeug/jinja2/urllib3/requests) — the lockfile is safe, the resolver lane is not. *(carried)*
2. Make the bundled-FFmpeg refresh assert the June-2026 security patch level, not just an 8.1.x version string. *(carried)*
3. Surface a capability/stub manifest so the route-count badge and panels stop presenting Tier-3 501 stubs as shipped. *(carried)*
4. Wire Parakeet TDT 0.6B v3 + Canary-1B-Flash ASR engines — stubs exist, CC-BY-4.0 weights, beat Whisper on WER and CPU speed. **(new)**
5. Wire LatentSync (Apache-2.0) lip-sync onto the existing dub pipeline → local end-to-end dubbing-with-lip-sync, the exact feature Rask/HeyGen/Sync.so paywall. **(new)**
6. Beat-synced auto-edit, SeedVR2 upscaling, Depth Anything 3 depth, AutoShot scene detection, local multimodal footage search. *(carried)*
7. Re-aim the IC-Light relight stub at v1 (Apache-2.0); v2 is non-commercial with weights never publicly released. **(new)**
8. Carry the existing roadmap (brand disambiguation, FFmpeg 8.1 encoders, Premiere 26 smoke, PyPI/Homebrew/winget, Vite migration, the two eval/exec sandbox hardenings, jsonify→error_response migration).

## Product Map
- Core workflows: Premiere/Resolve sequence automation; silence/filler/repeat-take cutting; transcription + caption styling/burn-in/import; audio cleanup/stems/TTS/dubbing/voice-clone; shorts/highlight generation; object/background removal and matting; depth/relight/restore; OTIO/FCPXML/AAF/EDL interchange; MCP/CLI/REST automation; footage search; multi-platform publish; watch-folder + scheduled jobs.
- Personas: solo creators replacing subscription plugins; podcast/interview editors inside Premiere; privacy-sensitive local teams; power users scripting via CLI/MCP/REST.
- Platforms/distribution: Windows installer (PyInstaller), Python 3.11–3.13 server, CEP (Premiere 2019+), UXP (Premiere 25.6+), Docker/GPU, macOS/Linux launchers, Flatpak/AppImage. No PyPI/Homebrew/winget presence yet (roadmapped, gated on brand decision).
- Data flows: panel → localhost Flask :5679 (HTTP/JSON), WebSocket :5680, MCP :5681; SQLite job + footage stores; plugins under `~/.opencut/plugins`; optional social OAuth and cloud LLM/TTS providers (all blockable via local-only mode).

## Competitive Landscape
**Adobe Premiere Pro 26 / Firefly Creative Agent** — Native on-device Object Mask, Generative Extend, Media Intelligence NL search across visuals + transcripts + described sounds. Creative Agent (beta, announced Apr 2026) is a conversational orchestration layer, but the announcement does not specify concrete Premiere editing actions — treat video-specific agentic capability as unproven. Learn: on-device multimodal *search* is table-stakes and is OpenCut's clearest parity gap (it indexes speech + CLIP frames, not objects/OCR/sound events). Avoid: rebuilding native object masking / generative extend the host already ships. (helpx.adobe.com, news.adobe.com)

**Rask AI / HeyGen / sync.so** — End-to-end dubbing (transcribe→translate→voice-clone→**lip-sync**) across 130+ languages; Rask paywalls lip-sync behind its $120/mo tier, sync.so charges $0.04–0.133/sec. Learn: OpenCut already has the full dub pipeline (`ai_dubbing.py`, `dub_pipeline.py`, `isochronous_translate.py`, `multilang_audio.py`) and voice cloning — only the visual lip-sync stage is stubbed (`lipsync_advanced.py`, `lipsync_echomimic.py`). Wiring an Apache-2.0 lip-sync model converts a paywalled flagship feature into a local/MIT one. (heygen.com, sync.so)

**Descript Underlord v2** — Agent that narrates reasoning, self-corrects mid-edit, offers a user-swappable model picker. Learn: explainable/interruptible agent UX and model-choice-as-UI; OpenCut has the ops + MCP + `agent_chat.py`/`autonomous_agent.py` but no first-class model picker or narrated plan. Avoid: metered-credit framing. (descript.canny.io)

**Opus Clip** — Generative B-roll over visual gaps, normalized AI virality score 0–100, import-by-link, per-platform AI cover/thumbnail. Learn: a normalized 0–100 virality score is cheap to add over OpenCut's existing engagement scoring; URL ingest is roadmapped. (opusclip.canny.io)

**Submagic / Klap / Veed 3.0 / Eddie AI** — Auto hook-titles, emoji/SFX auto-insert, music-sync, semantic clip selection, AI dubbing; Eddie does ingest→logging→rough-cut assembly into Premiere/Resolve. Learn: semantic highlight selection and rough-cut stringout assembly remain the biggest pro-workflow holes; emoji/SFX captioning largely already shipped (`animated_captions.py`, `social_captions.py`). (makeshorts.ai, redsharknews.com)

**OSS NLEs (OpenShot 3.5, Kdenlive, Shotcut)** — Absorbing local AI (transcription, object tracking, smart trim) with cloud disablable. Threat: locality alone is no longer a differentiator; OpenCut must lead on automation depth + Premiere integration. (resource.digen.ai)

Industry consensus (a16z, TechRadar 2026): tools "automate tasks but don't support editorial decision-making" and lose narrative coherence over 60–90s — the named unfilled niche, but research-grade and hard to scope into an acceptance criterion.

## Security, Privacy, and Reliability
**Transitive web-dep floors missing from `pyproject.toml`** (Verified): `dependencies` pins flask/flask-cors/waitress but not their transitive deps. `requirements-lock.txt` is current and safe (urllib3 2.7.0, Werkzeug 3.1.7, requests 2.33.0, Jinja2 3.1.6, certifi 2026.2.25), but `pip install opencut[all]` without the lock resolves transitively and can pull vulnerable versions. 2026 advisories: urllib3 CVE-2026-21441 / CVE-2025-66471 / CVE-2025-66418 (decompression-bomb DoS; fixed 2.6.3), Werkzeug CVE-2026-21860 / CVE-2025-66221 (`safe_join` Windows device-name DoS; fixed 3.1.5), Jinja2 CVE-2025-27516 (sandbox breakout; fixed 3.1.6), requests CVE-2026-25645 (fixed 2.33.0). Werkzeug/Jinja2 are always present via flask. Fix: add floor pins so the resolver lane matches the lockfile. Confidence: Verified (`pyproject.toml:35-43`).

**Bundled FFmpeg security provenance** (Verified gap): `ffmpeg/ffmpeg.exe` is bundled; the existing roadmap item bumps it to 8.1.x for D3D12VA/Vulkan encoders. June-2026's automated audit disclosed ~21 FFmpeg zero-days (CVE-2026-6385 confirmed CVSS 6.5; CVE-2026-39210…39218 reserved), several heap/stack overflows reachable via crafted media — the first untrusted-input path a media tool hits. Fixes landed as post-release master commits, so an 8.1.x *release tag* may predate them. The bump must assert a security patch level, not just a version string. Confidence: Verified for the gap; Needs-live-validation for which build clears all IDs.

**Strategic-stub surface vs. advertised capability** (Verified): 145+ `NotImplementedError` stubs in `opencut/core/` and ~22 routes returning HTTP 501 (`wave_h_routes.py`, `wave_k_routes.py`). Honest at the route level (docstrings + structured 501), but the README route-count badge (1,539) and feature framing include them. Credibility risk as distribution widens; needs a capability manifest distinguishing implemented / dependency-gated / Tier-3-stub.

**License hygiene on stubbed model targets** (Verified): some stubs name models whose *weights* are not MIT-distribution-safe even where the surrounding code is permissive. Concrete: `relight_iclight.py` is titled "IC-Light **V2**" — IC-Light v2 is non-commercial and its full weights were never publicly released (HF Space demo only), whereas IC-Light **v1 is Apache-2.0**. `musetalk_service.py` advertises `CreativeML-OpenRAIL-M` weights (use-based restrictions, not pure permissive). Each model promotion must pin the licence at the *model-card* level, not the repo code licence. This generalizes the existing MatAnyone (S-Lab non-commercial) caveat.

**Existing guardrails that hold** (Verified, re-confirmed): no `shell=True`, no `eval`/`exec`/`pickle` in shipped paths except the two sandboxed consoles already roadmapped (`expression_engine.py`, `scripting_console.py`); `torch.load(weights_only=True)`; CSRF gate on mutating routes; `validate_filepath()` realpath/prefix allowlist; lazy optional imports with `/health` capability flags; fresh installs emit no telemetry.

## Architecture Assessment
- **Capability manifest boundary**: add `/capabilities` (or extend `/health`) plus a generator tagging each route implemented/gated/stub; consume it in CEP+UXP to grey out stubbed actions and in the README badge to count only shipped routes. Touches `opencut/tools/dump_route_manifest.py`, route registration, both panels.
- **Engine-registry extensions (respect the existing pattern)**: ASR engines Parakeet/Canary (`asr_parakeet.py`, `asr_canary.py` stubs; no entry in transcription registry yet), lip-sync (`lipsync_advanced.py`/`lipsync_echomimic.py` stubs feeding the live dub pipeline), SeedVR2 upscaling, Depth Anything 3 depth, AutoShot scene detection, IC-Light v1 relight. Each is a registry entry, not a new subsystem.
- **Dub → lip-sync seam**: the dub pipeline (`dub_pipeline.py`, `auto_dub_pipeline.py`, `isochronous_translate.py`) produces translated/voice-cloned audio but stops before re-animating the mouth. A lip-sync engine slots after audio render as an optional final stage with a preview gate.
- **Footage search modalities**: `semantic_video_search.py` is speech + CLIP-frame; object-detection, OCR, and audio-event indexes would close the Premiere-26 Media-Intelligence gap locally.
- **Beat-synced assembly**: OpenCut detects beats (`librosa`) and writes Premiere markers but has no cut-to-beat assembler; add a route mapping beat timestamps → ripple cuts on the active sequence.
- **Refactor candidates** (carried): CEP `style.css` token consolidation and `main.js` modularization should follow the roadmapped Vite migration; do not start before it.
- **Testing gaps**: pyproject-resolver floor assertion (not just lockfile), FFmpeg security-patch-level canary, capability-manifest/stub-route parity test, engine-registry availability tests for each promoted model, dub+lip-sync fixture, beat-assembly fixture.
- **Category coverage**: accessibility (photosensitive flash check + audio-description drafting already ship — *not* gaps), i18n (existing items), observability (JSON logging + crash.log + request correlation ship), distribution (existing items), security (extended here), plugin ecosystem (defer to brand/release stability), mobile/multi-user SaaS (architectural misfits — rejected), offline/resilience (local-only mode ships).

## Rejected Ideas
- Photosensitive-epilepsy / Harding-style flash checker (external research lead): rejected — already shipped at `/accessibility/flash-detect` against ITU-R BT.1702-3, WCAG 2.2 SC 2.3.1, and PEAT thresholds (`docs/PHOTOSENSITIVE_FLASH_CHECK.md`).
- Audio-description automation (EAA driver): rejected — already shipped (`audio_description.py`, `/audio/description/microsoft-draft` + `/generate`, following Microsoft's `ai-audio-descriptions` pattern).
- Auto-emoji / SFX caption injection (Submagic): rejected — emoji handling already present in `animated_captions.py` / `social_captions.py`.
- BiRefNet matting as "missing": rejected — `matte_birefnet.py` already integrates the MIT BiRefNet front-end.
- MatAnyone / IC-Light v2 / REAL Video Enhancer (AGPL) / RIFE non-commercial weights as default engines: rejected as defaults — non-permissive licences conflict with MIT distribution; viable only as opt-in user-installed engines.
- distil-whisper / whisper-large-v3-turbo: rejected — `turbo` already offered (`panel_ux.py:615`); Parakeet/Canary (below) are the meaningful upgrade.
- Agentic multi-step editing as new work: rejected — `agent_chat.py`, `autonomous_agent.py`, `timeline_copilot.py`, `paper_edit.py` already implement it.
- AI avatar generation, narrative-coherence/story-structure agent, scale multicam/active-speaker switching, AV1 presets, watch-folder, multi-platform publish, C2PA, AAF/OTIOZ/EDL export, voice-clone consent: rejected — outside focus or already implemented (re-confirmed present).
- FlashVSR upscaling as a separate item: rejected for now — `upscale_flashvsr.py` stub exists but the weight licence is unconfirmed and SeedVR2 (Apache-2.0) already covers the upscaling-engine slot; revisit if FlashVSR ships a permissive licence.
- Re-adding already-roadmapped work: brand/namespace, FFmpeg 8.1 encoders, Premiere 26 smoke, PyPI/Homebrew/winget, Vite migration, sandbox hardening, jsonify→error_response migration, beat-sync, SeedVR2, DA3, AutoShot, multimodal footage search, import-by-link, capability manifest, dep floors, FFmpeg provenance.

## Sources
Adobe / Premiere 26
- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/search-for-media-using-ai-powered-media-intelligence.html
- https://news.adobe.com/news/2026/04/adobe-new-creative-agent

Dubbing / lip-sync
- https://github.com/bytedance/LatentSync
- https://lipsync.com/blog/open-source-lip-sync
- https://www.heygen.com/blog/heygen-vs-elevenlabs-vs-rask-ai-vs-dubverse
- https://sync.so/docs/models/lipsync

ASR / diarization
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://www.emergentmind.com/topics/sortformer-model

Short-form competitors
- https://opusclip.canny.io/changelog
- https://www.saasworthy.com/product/submagic.co/pricing
- https://www.redsharknews.com/eddie-ai-nab-2026-ai-video-editing-rough-cut
- https://descript.canny.io/changelog/announcingunderlord-v2

Models / capabilities
- https://github.com/lllyasviel/IC-Light
- https://github.com/IceClear/SeedVR2 · https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- https://github.com/ByteDance-Seed/depth-anything-3 · https://huggingface.co/depth-anything/DA3-SMALL
- https://github.com/wentaozhu/AutoShot
- https://github.com/pq-yang/MatAnyone

Security advisories
- https://thehackernews.com/2026/06/ai-agent-uncovers-21-zero-days-in.html
- https://github.com/advisories/GHSA-q22x-99q7-fr6w (FFmpeg CVE-2026-6385)
- https://github.com/advisories/GHSA-38jv-5279-wg99 (urllib3 CVE-2026-21441)
- https://www.sentinelone.com/vulnerability-database/cve-2026-21860/ (Werkzeug)
- https://github.com/advisories/GHSA-cpwx-vrp4-4pq7 (Jinja2 CVE-2025-27516)

Accessibility / distribution
- https://www.itu.int/rec/R-REC-BT.1702-3-202311-I
- https://www.getsubly.com/post/eaa-compliance-requirements-for-media
- https://docs.pypi.org/trusted-publishers/

## Open Questions
1. Should Tier-3 501 stub routes be excluded from the advertised route-count badge, or kept and re-labelled "planned" in the manifest? (Blocks the capability-manifest item's acceptance.)
2. Is the bundled-FFmpeg refresh allowed to use a post-8.1 master/snapshot build to clear the June-2026 CVEs, or must it stay on a tagged release even if the tag predates the fixes?
3. For OpenCut's MIT distribution, must every *default* engine carry a permissive weight licence (excluding LatentSync if its checkpoint licence proves restrictive, MuseTalk's OpenRAIL-M, IC-Light v2), with non-permissive models offered only as opt-in user installs? (Blocks the lip-sync/relight engine-default choice.)
4. (Carried) Distribution namespace decision before PyPI/winget/Homebrew/SEO.
5. (Carried) Live Premiere UDT evidence timeline for the UXP WebView cutover and Premiere 26 smoke pass.
