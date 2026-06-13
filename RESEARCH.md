# Research — OpenCut

## Executive Summary
OpenCut is a local-first Python/Flask automation server plus Adobe Premiere CEP/UXP panel replacing ~$1,400/year of video-editing subscriptions with 1,539 API routes, 602 core modules, and ~10,200 tests. Its strongest shape is unmatched breadth with local control — no competitor (commercial or OSS) covers the same surface area. The highest-value direction is making that breadth shippable: closing the distribution gap (release-channel lag, missing package-manager presence), hardening CI/dependency advisory posture, and finalizing the CEP-to-UXP migration before Adobe's September 2026 CEP deprecation.

Top opportunities, in priority order:
1. Restore public release-channel parity for the current 1.33.x code line.
2. Add a hard local-only/privacy mode across cloud-capable surfaces.
3. Pin PyInstaller >=6.0 in CI and add Python 3.13 to the test matrix.
4. Raise Flask/Waitress dependency floors for recent CVEs.
5. Add native auto-editor v30 compatibility canary.
6. Add Premiere caption export preflight and recovery.
7. Publish to Homebrew and winget after brand decision and code signing.
8. Keep existing roadmap items for brand disambiguation, UXP live evidence, Vite migration, FFmpeg 8.1, SAM 3, multicam visual cues, Premiere 26 positioning, PyPI publishing, and visual-search engines.

## Product Map
- Core workflows: Premiere sequence automation, silence/filler/repeat-take cutting, transcription/caption styling/import, audio cleanup/stems/TTS/dubbing/voice-clone, shorts/highlight generation, object removal/intelligence, OTIO/FCPXML/AAF/EDL interchange, MCP/CLI/API automation, stock footage search, multi-platform publish, watch-folder processing, webhook integrations.
- User personas: solo creators replacing subscription plugins, podcast/video editors working inside Premiere, privacy-sensitive local teams, power users scripting repeatable edits via CLI/MCP/API.
- Platforms and distribution: Windows installer (PyInstaller), Python 3.11+ server, CEP panel (Premiere 2019+), UXP panel (Premiere 25.6+), Docker/GPU, macOS/Linux source launchers, Flatpak/AppImage. Verified gap: public GitHub Releases show v1.25.1 as latest while internal code is v1.33.1.
- Key integrations: Adobe panel → localhost Flask API on port 5679, WebSocket sidecar (5680), MCP sidecar (5681), SQLite job/footage stores, plugin routes under `~/.opencut/plugins`, optional social OAuth/upload, optional cloud LLM/TTS/stock providers.

## Competitive Landscape
**Adobe Premiere 26 / UXP** — Bundles on-device Object Mask, Generative Extend, Media Intelligence, and 90+ Film Impact transitions. UXP replaces CEP by ~September 2026. Learn: on-device AI expectations are rising; live UXP packaging evidence is table-stakes. Avoid: recreating native object masking or generative extend features already shipping in the host.

**Descript** ($24-50/mo) — Full editor with Underlord AI co-editor, transcript-based editing, Overdub voice cloning, public API (open beta 2026), MCP integration. Learn: API/MCP play validates OpenCut's server-first architecture; agentic "AI co-pilot" editing is the new marketing paradigm. Avoid: making text editing the sole primary workflow.

**Phantom Wraith** (Free+$19/mo) — 8-camera multicam with conversation-pattern analysis, Dopple style copy-paste, 14+ tools including silence removal and transition assistant. Learn: 8-cam support and conversation-aware switching exceed AutoPod's 4-cam limit. Avoid: metered processing.

**AutoPod/FireCut/AutoCut** ($7-34/mo) — Premiere plugins proving demand for silence, captions, podcast, and multicam automation. AutoCut uniquely integrates Storyblocks B-roll injection. FireCut caps at 25 hrs/mo. OpenCut competitive lever: unlimited local processing vs. credit/time caps.

**CapCut Pro** ($19.99/mo) — Credit-gated AI features (avatars, voice cloning, text-to-video). Community frustration with paywalling previously free features. Learn: credit limitations are universally disliked. Avoid: any credit-gated model.

**Opus Clip** ($15-29/mo) — AI virality prediction, clip scoring. Data hostage: projects deleted 3 days after cancellation. Learn: clip scoring/virality angles are unique. Avoid: data-hostage policies.

**Gling** ($20-100/mo) — AI bad-take detection beyond silence removal. Learn: take detection is a strong selling point. OpenCut already has repeat_detect.py implementing this.

**OSS ecosystem** — auto-editor v29.3.1 (CLI-only silence editing), LosslessCut v3.69.0 (lossless trimmer with HTTP API), OpenTimelineIO (interchange standard). No OSS competitor matches OpenCut's breadth (1,539 routes vs. single-purpose tools).

## Security, Privacy, and Reliability
**Release-channel lag** (Verified): `gh release list` shows v1.25.1 as latest public release vs. v1.33.1 internal. Already roadmapped.

**Cloud surface gap** (Verified): `opencut/core/llm.py`, `video_llm.py`, `voice_overdub.py`, `social_post.py`, `stock_search.py`, `telemetry_aptabase.py` can contact external services. No single `OPENCUT_LOCAL_ONLY` kill switch. Already roadmapped.

**CI PyInstaller unpinned** (Verified): `.github/workflows/build.yml:59` runs `pip install pyinstaller` without version pin. CVE-2025-59042 (CVSS 7.0) is a local privilege escalation during PyInstaller bootstrap on Linux/macOS — affects Flatpak/AppImage build artifacts. Fixed in PyInstaller 6.0.

**Flask floor gap** (Verified): `pyproject.toml` pins `flask>=3.0,<4`. CVE-2026-27205 (Vary:Cookie info disclosure) fixed in Flask 3.1.3. Low severity for localhost listener but keeps `pip audit` clean.

**Waitress floor gap** (Verified): `pyproject.toml` pins `waitress>=3.0,<4`. CVE-2024-49768/49769 (request smuggling, socket exhaustion) fixed in Waitress 3.0.1. The >=3.0 floor permits the vulnerable 3.0.0 release.

**Python 3.13 untested** (Verified): CI matrix runs only Python 3.12. Flask 3.1.3 and core deps support 3.13; ML deps may lag. No test-matrix entry to catch breakage early.

**Existing guardrails that reduce priority**: All `torch.load` calls use `weights_only=True`; Transformers standalone floor at >=5.3 covers CVE-2026-4372; Pillow >=12.2 covers CVE-2026-42308/25990/40192; `np.load` calls disable pickle; global CSRF gate on all mutating routes; path validation on all file-accepting routes; `safe_bool` on all boolean flags.

**Node advisory posture**: Vite 5.4.21 pin with GHSA-4w7w-66w2-5vf9. Already roadmapped for HGFS-safe migration.

## Architecture Assessment
- **Privacy policy boundary needed**: one server-side local-only mode gating all external calls. Already roadmapped.
- **Release-channel boundary needed**: release smoke should distinguish internal version sync from public artifact availability. Already roadmapped.
- **CI security hygiene**: pin PyInstaller >=6.0, add Python 3.13 matrix entry, raise Flask/Waitress floors. New findings from this pass.
- **Distribution gap**: no Homebrew tap or winget manifest. After brand decision and code signing, these are the two highest-leverage package-manager channels (macOS and Windows respectively).
- **Visual search engine registry**: `opencut/core/semantic_video_search.py` fixed to CLIP ViT-B/32. Already roadmapped.
- **Refactor candidates**: CEP `style.css` token consolidation and `main.js` modularization should follow the roadmapped Vite upgrade.
- **Testing gaps**: Python 3.13 compat, public release parity assertion, local-only no-network mode, native auto-editor v30 fixtures, Premiere caption handoff mapping, visual-search cache invalidation, caption font-fallback rasters.
- **Category coverage**: accessibility (existing items), i18n (existing items), observability (JSON logging + crash.log + request correlation already ship), distribution (extended here), security (extended here), plugin ecosystem (wait on brand/release stability), mobile/multi-user SaaS (rejected as architectural misfits), offline/resilience (local-only mode).

## Rejected Ideas
- Fix version-string drift across files: rejected because `scripts/sync_version.py --check` passes at v1.33.1; the gap is public release-channel lag.
- Add C2PA provenance routes: rejected because `core/c2pa_sidecar.py`, `core/c2pa_embed.py`, `/provenance/c2pa`, and `/video/c2pa/*` already exist.
- Add AAF/OTIOZ/EDL export from scratch: rejected because interchange routes and tests already exist.
- Delete Edge-TTS: rejected because it's useful as an explicit cloud engine; fix is local-only fail-closed policy.
- Build a standalone browser NLE: rejected because it conflicts with the Premiere automation moat and duplicates OpenCut-app (48K stars).
- Build mobile or multi-user SaaS: rejected because it conflicts with local single-user desktop architecture.
- Add Remotion as a hard dependency: rejected per HN/community licensing concerns for commercial/free OSS.
- Launch plugin marketplace before brand/distribution: rejected as premature.
- Add ONNX Runtime memory-mapped .ort model loading: rejected as premature — cold-start time is not a user-reported pain point, and model loading is lazy/one-time per server session. Reconsider if startup latency becomes a complaint after broader distribution.
- Add agentic multi-step editing mode: rejected because `core/agent_chat.py`, `core/autonomous_agent.py`, `core/timeline_copilot.py`, and `core/paper_edit.py` already implement transcript-driven and conversational editing workflows.
- Add multi-platform batch export: rejected because `core/multi_publish.py`, `core/platform_publish.py`, and `core/podcast_bundle.py` already cover this.
- Add watch-folder automation: rejected because `core/watch_folder.py` and `core/scheduled_jobs.py` already wire this.
- Add AV1 export presets: rejected because `core/av1_export.py`, `core/svtav1_psy.py`, and 25 modules already reference AV1/SVT-AV1.
- Add voice-cloning consent workflow: rejected because 13 modules already implement consent checks for voice cloning.
- Scale multicam beyond 4 cameras: rejected because `core/multicam.py` has no camera-count limit — it maps speakers to tracks dynamically.
- Re-add already-roadmapped work: brand/namespace, FFmpeg 8.1, SAM 3, multicam visual cues, Premiere 26 audit, onnxruntime floor, PyPI trusted publishing, Vite migration, local-only mode, auto-editor v30, caption export preflight, visual-search engines, CJK/RTL caption fixtures.

## Sources
Adobe platform and Premiere behavior
- https://blog.developer.adobe.com/en/publish/2025/12/uxp-arrives-in-premiere-a-new-era-for-plugin-development
- https://developer.adobe.com/premiere-pro/uxp/
- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/search-for-media-using-ai-powered-media-intelligence.html
- https://helpx.adobe.com/premiere/desktop/add-video-effects/work-with-masks/object-masking.html
- https://helpx.adobe.com/premiere/desktop/titles-and-graphics/transcript-captions/export-captions.html

Commercial competitors
- https://www.autopod.fm/
- https://firecut.ai/changelog
- https://www.autocut.com/en/blogs/autopod-vs-autocut-alternative/
- https://phantomeditor.video/blog/best-autopod-alternative-2026-multicam-editing-premiere-pro
- https://www.descript.com/api
- https://help.descript.com/hc/en-us/articles/46056322186509-Use-Descript-with-an-AI-assistant-MCP
- https://help.opus.pro/docs/article/import-to-adobe-premiere
- https://www.capcut.com/resource/capcut-standard-vs-pro

OSS, awesome lists, and community signal
- https://github.com/SysAdminDoc/OpenCut/releases
- https://github.com/WyattBlue/auto-editor/releases
- https://github.com/mifi/lossless-cut
- https://github.com/OpenCut-app/OpenCut
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://github.com/ad-si/awesome-video-production
- https://documents.blackmagicdesign.com/SupportNotes/DaVinci_Resolve_20_New_Features_Guide.pdf

Dependencies, security, and model ecosystem
- https://www.ffmpeg.org/
- https://vite.dev/releases
- https://github.com/advisories/GHSA-4w7w-66w2-5vf9
- https://github.com/advisories/GHSA-53q9-r3pm-6pq6 (PyTorch CVE-2025-32434)
- https://github.com/advisories/GHSA-p2xp-xx3r-mffc (PyInstaller CVE-2025-59042)
- https://github.com/microsoft/onnxruntime/releases
- https://huggingface.co/blog/siglip2
- https://ai.meta.com/research/sam3/
- https://docs.pypi.org/trusted-publishers/
- https://www.reddit.com/r/premiere/comments/1s672iq/premiere_pro26_captions_and_export_settings/?tl=en

## Open Questions
1. Should a current v1.33.x GitHub release ship before F202 macOS notarization is live, or is the release lag intentional until macOS acceptance is complete?
2. What distribution namespace should replace or qualify "OpenCut" before PyPI, winget, Homebrew, and SEO work proceed?
3. When will live Premiere UDT evidence be available for the UXP WebView cutover and Premiere 26 smoke pass?
4. Should local-only mode hide cloud controls, hard-fail cloud route calls, or do both?
5. Which visual-search model should become the default after benchmarking: current CLIP, OpenCLIP, or SigLIP 2?
6. Is Windows code signing ($216-575/yr OV/EV certificate) budgeted for winget and SmartScreen reputation?
