# Research — OpenCut

## Executive Summary
Verified: OpenCut v1.33.1 (`opencut-ppro`) is a local-first Premiere Pro automation system, not a standalone NLE — a Python 3.11+/Flask service with CEP and UXP panels, FFmpeg/model pipelines, and REST/MCP/CLI surfaces, plus 1,546 routes, 275 test files, generated capability manifests, recovery tooling, and local installers. Its strongest shape is breadth backed by unusually explicit privacy, readiness, provenance, and rollback controls; its highest-value direction remains making that breadth execution-truthful and hostile-input-safe rather than adding another feature wave. This cycle's newly verified priorities, in order: close the unguarded SSRF/unbounded-download in `url_ingest.py` (the one network fetch path that bypasses the project's own `validate_public_http_url`); raise the Torch floor above the 2026 `torch.load(weights_only=True)` RCE and gate downloaded weights; then the already-tracked archive/readiness/Click work. Beyond fixes, three local-achievable parity features are now well-evidenced: semantic media search (Premiere 26 Media Intelligence), script/transcript→timeline assembly with alt-takes (Resolve 20 IntelliScript), and a Parakeet-v3 + Whisper-turbo hybrid ASR router. Standards currency work (MCP 2026-07-28 spec, C2PA 2.4 AI-disclosure, IMSC 1.3 Rec) is timely and low-risk.

## Product Map
- Core workflows: transcript/silence/filler editing; captions, translation, and broadcast delivery; audio/video cleanup and generation; timeline write-back, smart rendering, review, multi-platform export.
- User personas: Premiere editors automating repetitive first-cut work; privacy-sensitive creators (medical/legal/NDA footage); post-production operators running local batches; technical users integrating through CLI, REST, MCP, or plugins.
- Platforms and distribution: Python 3.11–3.13 on Windows/macOS/Linux; CEP and Premiere 25.6+ UXP panels; bundled FFmpeg 8.1.2 on Windows; WPF/Inno Setup installers; local-only release policy, with signed/notarized/package-manager publishing held in `Roadmap_Blocked.md`.
- Key integrations and data flows: panel/CLI/MCP request → authenticated localhost Flask route → async job/journal → FFmpeg or optional model adapter → staged output/timeline action → local history, diagnostics, review bundle, or export.

## Competitive Landscape
- **Adobe Premiere Pro 26 (Jan 2026)** shipped on-device **Object Mask** (marketed as privacy-preserving, no training on user data), **Media Intelligence** natural-language search over visuals/transcripts/metadata, and 20× faster shape masks. Learn: local AI is now a premium, not a compromise — match Media Intelligence with a local CLIP/embedding index; expose supported host actions directly. Avoid: cloud/Firefly-gated generative extend and claiming host parity without UDT evidence.
- **Descript Underlord (2026)** is the closest conceptual competitor: an agentic sidebar that does jump cuts, filler removal, captions, Studio Sound, eye-contact correction, 20+‑language dubbing, and exports a timeline to Premiere via XML — all cloud-based. Learn: an offline agentic transcript editor writing back to Premiere directly undercuts the cloud model. Avoid: autonomous editorial decisions without human review.
- **DaVinci Resolve 20 (2025→2026) IntelliScript** builds a full timeline from a script by matching transcribed takes, placing alt-takes on extra tracks; also AI Dialogue Matcher and Multicam SmartSwitch by active speaker. Learn: script→timeline assembly is squarely in OpenCut's transcript wheelhouse and is a marquee write-back feature achievable with local diarization/alignment.
- **Submagic / OpusClip (2026)** lead on single-pass animated captions (48+ languages), silence/filler removal, contextual **B-roll insertion from a stock library**, and auto-zoom. Learn: B-roll suggestion and auto-reframe are the most-marketed gaps; keep human review and honest output state. Avoid: default cloud storage and usage caps.
- **CapCut (2026)** runs AI features cloud-only (offline = basic filters) and pre-uploads content at import for recommendations. Learn: this is a concrete, citable privacy failure mode OpenCut's local pipeline inverts — lean on it in positioning.
- **LosslessCut / auto-editor / Kdenlive / Shotcut** show mature media tools compete on undo, proxy/export reliability, packaging, and hostile-input handling as much as features. Learn: treat imported project/archive/URL data as untrusted; keep local execution resumable and boringly reliable.
- **WhisperX / faster-whisper / NVIDIA Parakeet** — 2026 local ASR frontier: Parakeet TDT 0.6B v3 beats Whisper large-v3 on the Open ASR Leaderboard (~6.3% vs 7.4% avg WER) at ~3,000× realtime but covers only 25 EU languages; whisper-large-v3-turbo covers 99 languages at ~216× realtime GPU. Learn: ship both and auto-route by detected language.

## Security, Privacy, and Reliability
- **Verified (new, P0):** `opencut/core/url_ingest.py::ingest_url()` → `_fetch_direct()` (lines 135–192) validates only the URL scheme, then calls `urllib.request.urlopen(url)` directly. It never calls `opencut/core/url_safety.py::validate_public_http_url` (which the webhook/model paths use), never checks `require_network_allowed`/local-only mode, follows redirects by default, and streams the body with no size ceiling. `POST /search/ingest {"url":"http://127.0.0.1:.../"}` or `http://169.254.169.254/...` is fetched by the server; the on-disk extension is taken from the attacker URL and never content-verified. This is a server-side request forgery + unbounded-download + local-only bypass on one surface.
- **Verified (new, P1):** `validate_public_http_url` (`url_safety.py:17`) explicitly does not resolve DNS ("accept as-is") and the caller (`webhook_system.py` `_send_payload`) re-resolves and follows redirects — a TOCTOU/DNS-rebinding and redirect-into-private-range window that a hostname-only pre-check cannot close. The fix should become a shared connect-time resolved-IP guard used by both webhooks and URL ingest.
- **Verified (new, P1):** All four `torch.load` sites (`face_swap.py:105`, `model_quantization.py:373`, `video_ai.py:881`) rely on `weights_only=True` as the safety mechanism, but that is exactly what CVE-2026-24747 (GHSA-63cw-57p8-fm3p, CVSS 9.8, torch ≤ 2.9.1) bypasses via a crafted checkpoint. `pyproject.toml` floors `torch>=2.6` with no ceiling excluding vulnerable releases, and no picklescan gate exists on downloaded weights. Confidence: Likely — confirm the GHSA range before pinning.
- **Verified (P2):** `/health` (`opencut/routes/system.py:374–401`) is a CSRF-exempt GET that returns `csrf_token` to any Origin not in the two-value denylist `{null, file://}`. In the local single-user model this is low-blast-radius, but it is a denylist where an allowlist (same-origin/no-Origin) is cheap and correct.
- **Verified (P2):** `opencut/helpers.py::check_disk_space` (lines 187–193) returns `{"ok": True, "free_bytes": 0}` on any `shutil.disk_usage` exception, so callers gating on `ok` proceed on a genuinely full/inaccessible local volume, converting a clean 507 into a mid-render FFmpeg failure with a half-written output. Fail open only for un-probeable network drives.
- **Verified (P2/P3):** `jobs.py` admits up to `OPENCUT_MAX_CONCURRENT_JOBS` (max 100) as "running" onto a fixed 10-thread worker pool; raising the env var silently over-admits. Clamp effective concurrency to the pool's `max_workers`.
- **Still open from prior cycle:** `requirements-lock.txt` pins `click==8.3.1` (CVE-2026-7246, fixed in 8.3.3); `project_archive.py`/`lottie_import.py` unbounded ZIP extraction; `registry.py::resolved_state()` marking 52 `NotImplementedError` adapters "available" once their dependency imports. These remain the correctly-tracked top fixes.
- **Verified good:** bundled `ffmpeg/ffmpeg.exe` is 8.1.2 (newer than the June 2026 8.0.3 security line); `torch.load` already uses `weights_only=True`; Werkzeug/Jinja2 floors already patch the 2026 Windows `safe_join` device-name CVEs (CVE-2026-21860/27199, CVE-2025-66221) and the Jinja2 sandbox CVE. Recovery foundations (job resume/checkpoints, journals, disk preflight, atomic writes, support bundles, fuzz targets) already exist.

## Architecture Assessment
- **URL/network fetch trust is duplicated and inconsistent.** `url_ingest.py`, `webhook_system.py`, and `model_manager` each make their own SSRF decision (or none). Extract one connect-time guard that resolves the host, rejects private/link-local/loopback IPs, re-validates every redirect hop, and enforces byte ceilings; route all outbound fetches through it.
- **Model-weight loading has no supply-chain boundary.** Downloaded checkpoints are trusted to `torch.load`; there is no picklescan pass and no floor above the 2026 RCE. Treat any file under the model cache as untrusted input.
- `opencut/registry.py`, `feature_readiness.py`, `dump_feature_readiness.py` still need an implementation-status field distinct from dependency/hardware probes (prior P1). 52 `opencut/core/` modules carry terminal `NotImplementedError` (gen-video: OpenSora/SkyReels/LTX/cloud; ASR: canary/parakeet; face/relight/TTS/music); route layer exposes ~6 honest HTTP 501 stubs (`wave_h`, `wave_k`). Catalogue growth is unjustified until this readiness defect is fixed.
- **Panels are unrendered in automation.** 275 Python tests are strong; the JS side is 2 Node utility test files with no jsdom/Playwright, so the 18,495-line CEP `style.css` and 17,460-line CEP `main.js` (plus 8,191-line UXP `main.js`) are unverified for DOM/ARIA/focus/theme/state behavior (prior P1 covers this).
- **i18n stops at the panels.** CEP and UXP ship English + Spanish only (`extension/*/locales/{en,es}.json`); the Python/CLI backend has no i18n framework and English-only error strings, and there is no RTL (`dir="rtl"`) support anywhere. Embedded DE/FR/JA/PT labels appear in `en.json` text without locale files.
- **Provenance and MCP surfaces are one spec revision behind.** The repo embeds signed C2PA provenance but should target C2PA 2.4 (Apr 2026: `c2pa.ai-disclosure` assertion, durable/soft-binding credentials). The MCP server should target the 2026-07-28 revision (stateless core, Multi-Round-Trip Requests replacing elicitation, Tasks extension for long renders, cache metadata on resources).
- `broadcast_cc.py`/`broadcast_caption.py` still duplicate EBU-TT/TTML/IMSC exporters tested for XML shape, not IMSC 1.3 (now a W3C Rec) conformance (prior P2). Panel monolith decomposition remains a later XL after rendered contracts land.
- Coverage decision: security, accessibility, i18n/l10n, testing, docs, plugin ingestion, and a few high-fit features have retained work. Observability/offline already have proportionate local diagnostics, journals, retries, checkpoints, and support bundles. Distribution is credential/signing-gated in `Roadmap_Blocked.md`. Mobile and cloud multi-user editing conflict with the desktop local-first Premiere boundary.

## Rejected Ideas
- Local generative video synthesis to match Firefly/Sora/OpenSora on-device (source: gen-video stubs, Submagic B-roll gen): rejected — a ~0.6–2B on-device model cannot compete on true synthesis; position OpenCut as provenance-honest local automation instead. Match locally only where small models win (ASR, masking, reframe, silence/filler, semantic search).
- Cloud-first collaboration / hosted data plane (Descript Underlord, OpusClip, CapCut): rejected — review bundles and optional handoffs preserve the local-first privacy advantage that is the product's whole wedge.
- Full standalone NLE parity (Kdenlive/Shotcut/OpenShot): rejected — Premiere is the host; interchange/export surfaces already exist.
- Mobile editor (commercial creator suites): rejected — Premiere/UXP + local FFmpeg are desktop workflows.
- More generative model adapters before honest readiness (current AI research): rejected — worsens the verified `resolved_state()` defect; classify the 52 stubs first.
- New generic TTML/EBU or smart/lossless-cut features (LosslessCut/auto-editor): rejected — two caption implementations and `smart_render.py` keyframe snapping already ship; conformance/consolidation is the gap.
- Immediate OpenCV 5 / Rich 14 / major-version churn: rejected — supported ranges are stable and no user-facing/security gap is verified.
- New CI/release workflows or package-manager publishing: rejected — repo policy is local-build only; PyPI/Homebrew/winget/notarization already live in `Roadmap_Blocked.md`.
- Prometheus/OpenTelemetry metrics export, crash-report send path: rejected for now — local-only diagnostics/journal/security-audit are proportionate to a single-user desktop tool; a network telemetry surface conflicts with the privacy narrative.
- Eye-contact correction and full multilingual voice dubbing (Descript): under consideration, not scheduled — high model weight and quality risk on-device; revisit only if a competitive small model lands.

## Sources
Repository and OSS:
- https://github.com/SysAdminDoc/OpenCut
- https://github.com/mifi/lossless-cut
- https://github.com/WyattBlue/auto-editor
- https://github.com/m-bain/whisperX
- https://github.com/PixarAnimationStudios/OpenTimelineIO/releases

Commercial products and 2026 releases:
- https://blog.adobe.com/en/publish/2026/01/20/new-ai-powered-video-editing-tools-premiere-major-motion-design-upgrades-after-effects
- https://helpx.adobe.com/premiere/desktop/add-video-effects/work-with-masks/object-masking.html
- https://www.descript.com/underlord
- https://www.engadget.com/apps/davinci-resolve-20s-latest-ai-feature-can-create-an-entire-timeline-based-on-a-script-120009351.html
- https://www.ngram.com/blog/opus-clip-vs-submagic
- https://www.capcut.com/clause/privacy-policy

MCP, provenance, captions standards:
- https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/
- https://stacktr.ee/blog/mcp-2026-spec-changes
- https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- https://www.w3.org/news/2026/imsc-text-profile-1-3-is-now-a-w3c-recommendation/
- https://tech.ebu.ch/publications/tech3380
- https://www.mux.com/blog/captions-on-the-web-and-fcc-compliance

ASR models 2026:
- https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks
- https://whispernotes.app/blog/parakeet-v3-default-mac-model
- https://www.arunbaby.com/speech-tech/0073-whisper-vs-parakeet-asr-decision/

Security advisories 2026:
- https://github.com/advisories/GHSA-63cw-57p8-fm3p
- https://www.sentinelone.com/vulnerability-database/cve-2026-53875/
- https://app.opencve.io/cve/?product=werkzeug&vendor=palletsprojects
- https://nvd.nist.gov/vuln/detail/CVE-2026-7246
- https://peps.python.org/pep-0803/

Community signal (local-first / anti-subscription):
- https://larryjordan.com/articles/adobe-raises-prices-and-introduces-new-creative-cloud-plans/
- https://fstoppers.com/software/real-reason-photographers-are-leaving-adobe-901527
- https://www.hunton.com/privacy-and-cybersecurity-law-blog/data-protection-authorities-globally-highlight-privacy-issues-in-ai-image-generation

## Open Questions
- Confirm the exact affected-version range and fixed release for CVE-2026-24747 (GHSA-63cw-57p8-fm3p) before setting the Torch floor — the fix may be a 2.9.x patch rather than a new minor.
- No other questions block prioritization. Live Premiere/UDT acceptance and signing/notarization remain external blockers in `Roadmap_Blocked.md`, not research questions.
