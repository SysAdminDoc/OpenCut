# Research — OpenCut

_Last pass: 2026-06-14. Supersedes the 2026-06-13 pass; items that pass shipped between
the two passes (CI dep floors, PyInstaller pin, Python 3.13 matrix, local-only mode,
auto-editor v30, SAM 3, SigLIP visual-search registry, caption font-fallback) are recorded
in CHANGELOG/git, not repeated here._

## Executive Summary
OpenCut is a local-first Python/Flask automation server (1,539 registered routes, 107
blueprints, 602 core modules, ~10,200 tests) plus Adobe Premiere CEP/UXP panels. Its moat is
breadth-with-locality: no commercial or OSS competitor covers the same automation surface
while running entirely on-device. The codebase is healthy — the prior pass's correctness and
dependency findings are already fixed. The highest-value direction now is **converting a large
strategic-stub surface (145 `NotImplementedError` core stubs, ~22 routes returning HTTP 501)
into either shipped capability or honest capability signalling**, plus a thin layer of
transitive-dependency and FFmpeg-provenance hardening that the lockfile already does but the
`pyproject` resolver floors do not.

Top opportunities, in priority order:
1. Floor-pin transitive web deps in `pyproject.toml` (werkzeug/jinja2/urllib3/requests) — the lockfile is safe, the resolver lane is not.
2. Make the bundled-FFmpeg refresh include the June-2026 security commits, not just an 8.1.x version string.
3. Surface a machine-readable capability/stub manifest so the route-count badge and panels stop presenting Tier-3 501 stubs as shipped features.
4. Beat-synced auto-edit (cut footage to detected beats) — OpenCut already detects beats and writes markers but cannot assemble to them; competitors ship it.
5. Wire SeedVR2 (Apache-2.0) as an upscaling engine — its stub exists, it beats Real-ESRGAN, and the license is clean.
6. Upgrade depth backend to Depth Anything 3 Small (Apache-2.0) over V2.
7. Local multimodal footage search (object/OCR/sound-event index) to match Premiere 26 Media Intelligence with on-device privacy.
8. Carry the existing roadmap (brand disambiguation, FFmpeg 8.1 encoders, Premiere 26 smoke, PyPI/Homebrew/winget, Vite migration, the two eval/exec sandbox hardenings, jsonify→error_response migration).

## Product Map
- Core workflows: Premiere/Resolve sequence automation; silence/filler/repeat-take cutting; transcription + caption styling/burn-in/import; audio cleanup/stems/TTS/dubbing/voice-clone; shorts/highlight generation; object/background removal and matting; depth/relight/restore; OTIO/FCPXML/AAF/EDL interchange; MCP/CLI/REST automation; footage search; multi-platform publish; watch-folder + scheduled jobs.
- Personas: solo creators replacing subscription plugins; podcast/interview editors inside Premiere; privacy-sensitive local teams; power users scripting via CLI/MCP/REST.
- Platforms/distribution: Windows installer (PyInstaller), Python 3.11–3.13 server, CEP (Premiere 2019+), UXP (Premiere 25.6+), Docker/GPU, macOS/Linux launchers, Flatpak/AppImage. No PyPI/Homebrew/winget presence yet (roadmapped, gated on brand decision).
- Data flows: panel → localhost Flask :5679 (HTTP/JSON), WebSocket :5680, MCP :5681; SQLite job + footage stores; plugins under `~/.opencut/plugins`; optional social OAuth and cloud LLM/TTS providers (all blockable via local-only mode).

## Competitive Landscape
**Adobe Premiere Pro 26** — Native on-device Object Mask (20× faster tracking), Generative Extend, and Media Intelligence NL search across visuals + transcripts + described sounds. Learn: on-device multimodal *search* is now table-stakes and is OpenCut's clearest parity gap (it indexes speech + CLIP frames, not objects/OCR/sound events). Avoid: rebuilding native object masking or generative frame extension the host already ships under the panel. (winbuzzer.com, timesofai.com)

**Descript Underlord v2** — Agent that plans, narrates its reasoning, self-corrects mid-edit, uses ~20% fewer credits, offers a user-swappable model picker (Anthropic/OpenAI/Google), and PDF/deck→video. Learn: explainable/interruptible agent UX and model-choice-as-UI; OpenCut has the ops + MCP + `agent_chat.py`/`autonomous_agent.py` but no first-class model picker or narrated plan. Avoid: a metered-credit framing — lead with MIT/no-credits. (descript.canny.io)

**Eddie AI** — AI footage logging into structured bins/sequences, "Dirty Multicam", BRAW/R3D/Sony codecs, rough-cut "stringouts"; integrates with Premiere/Resolve/FCP. Learn: ingest→logging→rough-cut-assembly is the biggest pro-workflow hole given OpenCut's Premiere panel. Avoid: nothing — this is a clean adjacency, but it is L/XL. (redsharknews.com, cined.com)

**Opus Clip** — Generative B-roll to cover visual gaps, AI virality score 0–100, import-by-link (Zoom/Apple Podcasts/Medal — no re-upload), per-platform AI cover/thumbnail. Learn: URL ingest connectors and a normalized virality score; OpenCut has B-roll + engagement scoring + smart thumbnails but no link ingest and no normalized 0–100 score. (opusclip.canny.io, coldiq.com)

**Submagic / Klap / Veed 3.0** — Auto hook-titles, emoji/SFX auto-insert, auto music-sync, in-app scheduling + per-platform metadata; intent-based (semantic) clip selection; AI dubbing 50+ langs with voice-matched cloning + lip-sync. Learn: lip-sync for dubbed audio and deeper semantic highlight selection. OpenCut has animated captions, social upload, voice cloning, and `scheduled_jobs.py`, but no lip-sync and weaker semantic clipping. (toolsforhumans.ai, makeshorts.ai, veed.io)

**aescripts ecosystem (PremiereGPT, Automation Blocks, Automated Video Editing)** — Active-speaker multicam switching, user-authored automation macros inside the panel, beat-synced auto-edit. Learn: beat-synced assembly and user macros are proven, local-runnable Premiere automations OpenCut lacks. (aescripts.com)

**OSS NLEs (OpenShot 3.5, Kdenlive, Shotcut)** — Now absorbing local AI (transcription, object tracking, smart trim) with cloud fully disablable. Threat: locality alone is no longer a differentiator; OpenCut must lead on automation depth + Premiere integration. (resource.digen.ai)

Industry consensus (a16z, TechRadar 2026 roundups): tools "automate tasks but don't support editorial decision-making" and lose narrative coherence over 60–90s. The story-structure / narrative-coherence tier is the named unfilled niche — but it is research-grade and hard to scope.

## Security, Privacy, and Reliability
**Transitive web-dep floors missing from `pyproject.toml`** (Verified): `dependencies` pins flask/flask-cors/waitress but not their transitive deps. `requirements-lock.txt` is current and safe (urllib3 2.7.0, Werkzeug 3.1.7, requests 2.33.0, Jinja2 3.1.6, certifi 2026.2.25), but `pip install opencut[all]` without the lock resolves transitively and can pull vulnerable versions. Relevant 2026 advisories: urllib3 CVE-2026-21441 / CVE-2025-66471 / CVE-2025-66418 (decompression-bomb DoS; fixed 2.6.3), Werkzeug CVE-2026-21860 / CVE-2025-66221 (`safe_join` Windows device-name DoS; fixed 3.1.5), Jinja2 CVE-2025-27516 (sandbox breakout; fixed 3.1.6), requests CVE-2026-25645 (fixed 2.33.0). Werkzeug/Jinja2 are always present (via flask, a core dep). Fix: add floor pins to `pyproject` so the resolver lane matches the lockfile. Confidence: Verified (absence confirmed in `pyproject.toml:35-43`).

**Bundled FFmpeg security provenance** (Verified gap): `ffmpeg/ffmpeg.exe` is bundled; the existing roadmap item bumps it to 8.1.x for D3D12VA/Vulkan encoders. In June 2026 an automated audit disclosed ~21 FFmpeg zero-days (CVE-2026-6385 confirmed CVSS 6.5; CVE-2026-39210…39218 reserved), several heap/stack overflows reachable via crafted media — exactly the untrusted-input path a media tool hits first. Fixes landed as post-release master commits, so an 8.1.x *release tag* may predate them. The version bump must therefore assert a security patch level, not just a version string. Confidence: Verified for the bump gap; Needs-live-validation for which build clears all IDs (several CVE pages are still NVD-RESERVED). (thehackernews.com, advisories GHSA-q22x-99q7-fr6w)

**Strategic-stub surface vs. advertised capability** (Verified): 145 `NotImplementedError` stubs in `opencut/core/` and ~22 routes returning HTTP 501 (`wave_h_routes.py`, `wave_k_routes.py`). They are honest at the route level (docstrings + structured 501), but the README route-count badge (1,539, generated from `route_manifest.json`) and feature framing include them. This is a credibility risk as distribution widens. Need a capability manifest that distinguishes implemented / dependency-gated / Tier-3-stub so docs and panels can signal accurately.

**Existing guardrails that hold** (Verified, re-confirmed): no `shell=True`, no `eval`/`exec`/`pickle` in shipped paths except the two sandboxed consoles already roadmapped (`expression_engine.py`, `scripting_console.py`); `torch.load(weights_only=True)`; CSRF gate on mutating routes; `validate_filepath()` realpath/prefix allowlist; `safe_bool` on flags; lazy optional imports with `/health` capability flags; fresh installs emit no telemetry.

## Architecture Assessment
- **Capability manifest boundary**: add a `/capabilities` (or extend `/health`) surface and a generator that tags each route implemented/gated/stub; consume it in CEP+UXP to grey out stubbed actions and in the README badge to count only shipped routes. Touches `opencut/tools/dump_route_manifest.py`, route registration, both panels.
- **Engine-registry extensions (respect the existing pattern)**: SeedVR2 upscaling (`upscale_seedvr2.py` stub exists), Depth Anything 3 depth backend (`cinefocus.py` defaults to DA2-Small), AutoShot scene detection (`scene_detect.py` registry is threshold + TransNetV2). Each is a registry entry, not a new subsystem.
- **Beat-synced assembly**: OpenCut detects beats (`librosa`) and writes Premiere markers but has no cut-to-beat assembler; add a route that maps beat timestamps → ripple cuts / clip boundaries on the active sequence.
- **Footage search modalities**: `semantic_video_search.py` is speech + CLIP-frame; object-detection, on-screen-text (OCR), and audio-event indexes would close the Premiere-26 Media-Intelligence gap locally.
- **Refactor candidates** (unchanged from prior pass, still valid): CEP `style.css` token consolidation and `main.js` modularization should follow the roadmapped Vite migration; do not start before it.
- **Testing gaps**: pyproject-resolver floor assertion (not just lockfile), FFmpeg security-patch-level canary, capability-manifest/stub-route parity test, SeedVR2/DA3 engine-registry availability tests, beat-assembly fixture.
- **Category coverage**: accessibility/i18n (existing items), observability (JSON logging + crash.log + request correlation already ship), distribution (existing items), security (extended here), plugin ecosystem (defer to brand/release stability), mobile/multi-user SaaS (architectural misfits — rejected), offline/resilience (local-only mode ships).

## Rejected Ideas
- "ProPainter not wired" (Explore subagent lead): rejected — `object_removal.py:175 inpaint_video_propainter` is implemented; it is gated on a manual `~/.opencut/models/propainter` clone, not missing. (At most a docs/install-path nicety.)
- Add whisper-large-v3-turbo transcription: rejected — already offered (`panel_ux.py:615` lists `turbo`).
- Add distil-whisper: rejected — `turbo` already covers the speed need; marginal quality delta for a new model surface.
- Pin `cryptography` ≥46.0.7: rejected as a hard item — not in the lockfile or core/optional tree; only reachable via the explicit `torch-stack`/`diarize` lanes. Revisit only if it enters the audited `[all]` resolution.
- AI avatar generation (HeyGen Avatar IV/V): rejected — outside the Premiere-automation focus; generation, not editing.
- Narrative-coherence / story-structure agent: rejected as a roadmap item — real unmet niche but research-grade and unscopable into an acceptance criterion today; revisit when an upstream model makes it tractable.
- Agentic multi-step editing mode: rejected — `agent_chat.py`, `autonomous_agent.py`, `timeline_copilot.py`, `paper_edit.py` already implement conversational/transcript-driven editing.
- Scale multicam / active-speaker switching as new work: rejected — `multicam.py` + `multicam_visual.py` (audio+visual mode shipped 2026-06) already map speakers to tracks dynamically with no camera-count limit.
- AV1/SVT-AV1 presets, watch-folder, multi-platform batch publish, C2PA provenance, AAF/OTIOZ/EDL export, voice-clone consent: rejected — all already implemented (see prior pass; re-confirmed present).
- MatAnyone video matting: noted but not roadmapped as default — NTU S-Lab non-commercial license conflicts with MIT distribution; only viable as an opt-in user-installed engine.
- Re-adding already-roadmapped work: brand/namespace, FFmpeg 8.1 encoders, Premiere 26 smoke, PyPI/Homebrew/winget, Vite migration, expression_engine/scripting_console sandbox hardening, jsonify→error_response migration.

## Sources
Adobe / Premiere 26
- https://winbuzzer.com/2026/01/20/adobe-launches-premiere-pro-26-with-ai-powered-object-mask-feature-xcxwbn/
- https://www.timesofai.com/news/adobe-ai-object-mask-premiere-pro-after-effects-26/
- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/search-for-media-using-ai-powered-media-intelligence.html

Competitors / ecosystem
- https://descript.canny.io/changelog/announcingunderlord-v2
- https://www.redsharknews.com/eddie-ai-nab-2026-ai-video-editing-rough-cut
- https://www.cined.com/eddie-ai-updates-released-visit-their-booth-at-nab-2026/
- https://opusclip.canny.io/changelog
- https://coldiq.com/tools/opus-clip
- https://www.toolsforhumans.ai/ai-tools/submagic
- https://www.makeshorts.ai/blog/best-ai-shorts-generators-2026
- https://www.veed.io/tools/ai-avatar
- https://aescripts.com/automated-video-editing-for-premiere-pro/
- https://aescripts.com/premieregpt-podcast-bundle/
- https://aescripts.com/automation-blocks-for-premiere-pro/
- https://a16z.com/its-time-for-agentic-video-editing/
- https://resource.digen.ai/open-source-ai-video-editor-2026/

Security advisories
- https://thehackernews.com/2026/06/ai-agent-uncovers-21-zero-days-in.html
- https://gist.github.com/cla7aye15I4nd/f9a7700240afe7ae8171ee65682e890f
- https://github.com/advisories/GHSA-q22x-99q7-fr6w (FFmpeg CVE-2026-6385)
- https://github.com/advisories/GHSA-38jv-5279-wg99 (urllib3 CVE-2026-21441)
- https://github.com/advisories/GHSA-hgf8-39gv-g3f2 (Werkzeug CVE-2025-66221)
- https://www.sentinelone.com/vulnerability-database/cve-2026-21860/ (Werkzeug)
- https://github.com/advisories/GHSA-cpwx-vrp4-4pq7 (Jinja2 CVE-2025-27516)

Models / capabilities
- https://github.com/IceClear/SeedVR2 · https://huggingface.co/ByteDance-Seed/SeedVR2-3B
- https://github.com/ByteDance-Seed/depth-anything-3 · https://huggingface.co/depth-anything/DA3-SMALL
- https://github.com/wentaozhu/AutoShot
- https://github.com/lixiaowen-xw/DiffuEraser
- https://huggingface.co/ResembleAI/chatterbox
- https://www.ffmpeg.org/ · https://www.phoronix.com/news/FFmpeg-8.0-Released

## Open Questions
1. Should Tier-3 501 stub routes be excluded from the advertised route-count badge, or kept and re-labelled "planned" in the manifest? (Blocks the capability-manifest item's acceptance.)
2. Is the bundled-FFmpeg refresh allowed to use a post-8.1 master/snapshot build to clear the June-2026 CVEs, or must it stay on a tagged release even if the tag predates the fixes?
3. For OpenCut's distribution license (MIT), are non-commercial model weights (MatAnyone, DA3-Large, Depth Pro) acceptable as opt-in user-installed engines, or must every default engine be commercial-clean?
4. (Carried) Distribution namespace decision before PyPI/winget/Homebrew/SEO.
5. (Carried) Live Premiere UDT evidence timeline for the UXP WebView cutover and Premiere 26 smoke pass.
