# Research — OpenCut

Date: 2026-09-25. Replaces all prior research.

## Executive Summary

OpenCut is a local-first automation and review layer for Adobe Premiere Pro, delivered through CEP and UXP panels backed by a Flask service, CLI, and MCP surface (`README.md`, `opencut/server.py:398`, `opencut/_generated/route_manifest.json`). Its strongest current shape is the unusually explicit accounting of routes, feature readiness, host parity, model licensing, and media provenance under `opencut/_generated/`; its weakest shape is the released Windows runtime, where optional native packages and broad health probing can still make the process disappear before diagnostics are available (`opencut/server.py:289`, `opencut/routes/system.py:260`, https://github.com/SysAdminDoc/OpenCut/discussions/10). The highest-value direction is to make runtime, dependency, and interchange boundaries deterministic, then use current Premiere APIs and focused editor workflows instead of adding another broad feature layer.

Top opportunities, in priority order:

1. Replace the obsolete Frame.io V2 integration with a V4 contract covering Adobe IMS OAuth, resumable uploads, version identity, signed webhooks, and offline-safe retry (`opencut/core/frameio_integration.py:18`, https://next.developer.frame.io/platform/v4/docs/quick-start).
2. Keep `/health` free of optional native imports and move capability discovery into crash-contained workers (`opencut/routes/system.py:260`, `opencut/routes/system.py:478`, https://github.com/SysAdminDoc/OpenCut/discussions/10).
3. Give the bundled runtime an ABI-specific optional-package store shared by the WPF installer and runtime installer, with migration from the current flat directory (`opencut/server.py:289`, `opencut/security.py:460`, `installer/src/OpenCut.Installer/Services/DependencyInstaller.cs`).
4. Route all model downloads through one acquisition boundary; 33 core modules call `from_pretrained` directly despite the controls in `opencut/core/model_safety.py` (https://github.com/advisories/GHSA-fv5v-hfxp-5379).
5. Turn Premiere and FFmpeg security state into data-driven runtime policy rather than stale prose or version assumptions (`opencut/core/ffmpeg_provenance.py:64`, https://helpx.adobe.com/security/products/premiere_pro/apsb26-157.html, https://ffmpeg.org/download.html).
6. Correct the obsolete Adobe transition deadline and qualify UXP 26.5 while preserving the 25.6 baseline (`opencut/_generated/adobe_premierepro_versions.json`, https://blog.developer.adobe.com/en/publish/2026/09/investing-in-the-future-of-creative-cloud-extensibility-uxp-comes-to-our-flagship-applications, https://developer.adobe.com/premiere-pro/uxp/changelog/).
7. Use one rational time domain for edit plans and issue semantic fidelity receipts for OTIO, FCPXML, and AAF (`opencut/core/sequence_index.py:138`, `opencut/core/auto_edit.py:299`, https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues).
8. Replace card-per-tool layouts with task groups and reserve borders and pills for real state; the static panels declare 96 CEP cards and 66 UXP cards (`extension/com.opencut.panel/client/index.html:227`, `extension/com.opencut.uxp/index.html:149`).
9. Exercise real loading, empty, error, permission, offline, and confirmation paths instead of treating injected test markup as product coverage (`extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:982`, `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:2548`).
10. Convert transcript selections into ranged markers and marker-backed selects without retranscription, using the host transcript JSON already exposed by UXP (`extension/com.opencut.uxp/main.js:1507`, https://community.adobe.com/feature-requests-730/feature-request-convert-transcripts-into-sequence-markers-1327693).

## Product Map

### Core workflows

- Turn transcripts, silence, scenes, scripts, and editorial briefs into reviewable cut plans, then apply them through Premiere host actions (`opencut/core/transcript_timeline_edit.py`, `opencut/core/auto_edit.py`, `opencut/core/paper_edit.py`, `opencut/core/autonomous_agent.py`).
- Analyze and repair captions, dialogue, music, color, framing, motion, and damaged media through optional local or remote engines (`opencut/routes/caption_analysis_routes.py`, `opencut/core/audio_enhance.py`, `opencut/core/motion_tracking.py`).
- Index media, transcripts, shots, and metadata for project and federated search (`opencut/core/semantic_search.py`, `opencut/core/federated_media_index.py`, `opencut/core/sequence_index.py`).
- Export media, captions, review bundles, edit decisions, and provenance evidence (`opencut/core/delivery_validate.py`, `opencut/core/review_bundle.py`, `opencut/export/otio_export.py`, `opencut/core/c2pa_sidecar.py`).
- Expose the same backend through CEP, UXP, CLI, curated MCP, and generated API surfaces (`extension/com.opencut.panel/client/index.html`, `extension/com.opencut.uxp/index.html`, `pyproject.toml:255`, `opencut/mcp_server.py:2199`).

### User personas

- Premiere editors handling interview, documentary, social, podcast, and high-volume delivery work (`extension/com.opencut.panel/client/index.html:33`, `extension/com.opencut.uxp/index.html:70`).
- Assistant editors and reviewers who need transcript search, marker exchange, versioned comments, relink, and deterministic handoff (`opencut/core/review_comments.py`, `opencut/core/review_links.py`, `opencut/core/content_fingerprint.py`).
- Technical operators who automate repeatable work through CLI, MCP, queues, and local integrations while retaining human review (`opencut/cli.py`, `opencut/mcp_server.py`, `opencut/routes/jobs_routes.py`).

### Platforms and distribution

- CEP targets Premiere 2019 and later; UXP targets Premiere 25.6 and later, with two host actions still CEP-only and one partial UXP action (`extension/com.opencut.panel/CSXS/manifest.xml`, `extension/com.opencut.uxp/manifest.json`, `opencut/_generated/cep_uxp_parity.json`).
- Python 3.11 through 3.14 is supported for source installs; the Windows installer is a self-contained .NET 10 `win-x64` application that installs a PyInstaller server (`pyproject.toml:22`, `installer/src/OpenCut.Installer/OpenCut.Installer.csproj:5`, `opencut_server.spec`).
- Docker is CPU-only and does not contain CUDA, NVENC, or a GPU runtime (`Dockerfile:14`). UXP marketplace signing remains unavailable without an Adobe publisher identity (`README.md:589`, `docs/INSTALLER_POLICY.md`).

### Key integrations and data flows

- Panels call the loopback Flask service; host mutations cross CEP ExtendScript or UXP APIs and are tracked in generated parity and command manifests (`opencut/server.py`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/main.js`, `opencut/_generated/cep_uxp_parity.json`).
- Media and model work can involve FFmpeg, PyTorch, ONNX Runtime, Hugging Face, Whisper-family backends, and optional system packages (`pyproject.toml`, `requirements-build.txt`, `opencut/core/model_safety.py`, `opencut/core/ffmpeg_provenance.py`).
- Review and delivery integrations include Frame.io, cloud storage, webhooks, OTIO/OTIOZ, FCPXML, AAF, subtitles, C2PA, and local review bundles (`opencut/core/frameio_integration.py`, `opencut/core/review_bundle.py`, `opencut/export/`, `opencut/core/caption_interchange.py`).
- The product is local-first and loopback-only by default; remote binding requires explicit enablement and token authentication (`SECURITY.md:72`).

## Competitive Landscape

| Product or project | What it does well | Learn for OpenCut | Intentionally avoid | Source |
|---|---|---|---|---|
| Adobe Premiere Pro 26.5 | Native transcription, Paper Edit, semantic project search, markers, and host C2PA access | Detect and orchestrate host capabilities before running duplicate backend work | Rebuilding a second NLE or a weaker Paper Edit | https://helpx.adobe.com/premiere/desktop/whats-new/release-notes.html |
| DaVinci Resolve 21 | Local media analysis, timeline comparison, shared project state, and explicit change acceptance | Pair local search with visible collaboration state and reviewable diffs | Importing Resolve's broad finishing surface into a Premiere assistant | https://www.blackmagicdesign.com/products/davinciresolve/collaboration |
| Descript | Transcript-first editing with preview, retry, keep, and revert around generated changes | Make AI edits transactional and preserve the original by default | Hiding model uncertainty behind a one-click rewrite | https://help.descript.com/script-editing/fix-take |
| Frame.io V4 | Stable asset/version identity, timeline-linked comments, resumable upload, and signed event delivery | Use V4 identity as the cloud review boundary and make retries idempotent | Continuing the V2 token and endpoint model | https://next.developer.frame.io/platform/v4/docs/quick-start |
| CapCut | Fast transcript cleanup, filler removal, smart search, and accessible social workflows | Keep common transcript actions immediate, compact, and easy to audition | Cloud-first assumptions and irreversible automatic cleanup | https://www.capcut.com/tools/video-transcript-editing |
| VEED OpenEdit | Text-directed editing and API-oriented subtitle delivery | Keep intent-driven work exportable and observable | A hosted editor that competes with Premiere for timeline ownership | https://www.veed.io/tools/openedit |
| Runway Edit Studio and Agent | Iterative natural-language edits with previews and bounded retry | Offer alternatives and a visible plan before host mutation | Open-ended chat as the primary editing interface | https://help.runwayml.com/hc/en-us/articles/51683104370451-Creating-with-Edit-Studio |
| Kdenlive and Shotcut | Mature proxy, timeline, subtitle, and broad-format workflows backed by MLT | Test ripple, retime, VFR, proxy, and subtitle boundaries as first-class correctness cases | Matching every desktop-editor control | https://kdenlive.org/news/releases/ |
| LosslessCut | Focused, fast remux and keyframe-aware cutting with explicit smart-cut limits | Explain when a cut is exact, keyframe-bound, re-encoded, or experimental | Calling a fast path lossless when boundaries require re-encoding | https://github.com/mifi/lossless-cut/blob/master/issues.md |
| auto-editor | Composable analysis expressions, rational timestamps, reusable caches, and deterministic CLI work | Make edit rules inspectable, repeatable, and cache-aware | Removing every pause without editorial rhythm controls | https://github.com/WyattBlue/auto-editor/releases |
| Subtitle Edit | Deep subtitle repair, waveform review, format coverage, and pluggable speech recognition | Treat caption confidence, timing, OCR, and typesetting as review work | Turning caption generation into an unreviewed terminal action | https://github.com/SubtitleEdit/subtitleedit/releases |
| PySceneDetect | Narrow detector interfaces, reusable stats, and VFR-focused test improvements | Keep analysis engines replaceable and benchmark detector changes | Coupling scene detection to one model or one timeline format | https://www.scenedetect.com/changelog/ |
| OpenTimelineIO | A typed interchange model with adapters, metadata preservation, and active failure reports | Test semantic preservation, not just parse success | Making a pre-release adapter version the only supported path | https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases |

## Reported Issues

- **Issue #7, fixed on main but unreleased.** OpenCut v1.55.1 reported an installed GPU as unusable because physical detection was conflated with executable provider support. Commits `276114b`, `8863aa2`, and `1ff072e` repair provider and architecture handling in `opencut/gpu.py`, but v1.55.1 remains the latest published installer (https://github.com/SysAdminDoc/OpenCut/issues/7, https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.55.1). A new roadmap row would duplicate the existing release ledger and F424.
- **Issue #8, fixed on main but unreleased.** The released server omitted generated manifests, imported foreign Python 3.12 packages into bundled Python 3.13, allowed duplicate server ownership, and lost native crash evidence. Main addresses those reported causes in `opencut_server.spec`, `opencut/server.py`, `opencut/pid.py`, and `opencut/core/workflow.py` (https://github.com/SysAdminDoc/OpenCut/issues/8). Installed-artifact execution still needs the F424 smoke matrix.
- **Discussion #9, duplicate evidence.** Hiding Python 3.14 made the bundled Python 3.13 server healthy; restoring it reproduced an access violation. This is strong A/B evidence for issue #8, not a separate feature (https://github.com/SysAdminDoc/OpenCut/discussions/9).
- **Discussion #10, unresolved.** `WinError 206` while loading Torch DLLs traces to the remaining shared package directory, installer/runtime disagreement, and optional native imports during `/health`; F447 and F448 address those boundaries (`opencut/server.py:289`, `opencut/routes/system.py:260`, `installer/src/OpenCut.Installer/Services/DependencyInstaller.cs`, https://github.com/SysAdminDoc/OpenCut/discussions/10). The reporter's DLL-registration theory needs live validation.
- **Closed reports not re-proposed.** Issue #6's installer `NullReferenceException` shipped in v1.55.1; issue #5's CEP CSRF bootstrap was repaired; issues #1 and #2 are superseded by version automation and dependency diagnostics (https://github.com/SysAdminDoc/OpenCut/issues/6, https://github.com/SysAdminDoc/OpenCut/issues/5). Discussion #4 says the product never worked but supplies no environment or reproducible symptom, so it supports the installation cluster but no independent item (https://github.com/SysAdminDoc/OpenCut/discussions/4).
- **No tracker feature request has demonstrated demand.** The open tracker contains the two released-artifact bugs above, no open pull requests, and no actionable enhancement thread as of 2026-09-25 (https://github.com/SysAdminDoc/OpenCut/issues, https://github.com/SysAdminDoc/OpenCut/pulls).

## Security, Privacy, and Reliability

- **Verified, P0:** `/health` calls `_build_capabilities()`, which imports optional native stacks. Python exceptions are caught, but a native process abort cannot be converted into a response (`opencut/routes/system.py:260`, `opencut/routes/system.py:478`). Liveness must not execute Torch, ONNX Runtime, TensorFlow, or other crash-prone probes.
- **Verified, P0:** frozen startup appends the flat `~/.opencut/packages` directory, while runtime and WPF installation can select a different system Python from the bundled interpreter (`opencut/server.py:289`, `opencut/security.py:366`, `opencut/security.py:460`, `opencut/routes/system_whisper_routes.py`, `installer/src/OpenCut.Installer/Services/DependencyInstaller.cs`). Package ownership needs an ABI-keyed target, smoke import, migration, and quarantine path.
- **Verified, P0:** `opencut/core/model_safety.py` provides revision, path, and download controls, but 33 modules under `opencut/core/` call `from_pretrained` directly. The 2026-08-17 Transformers shard path-traversal advisory has no confirmed fixed version at the research cutoff, so dependency pinning alone is not a control (https://github.com/advisories/GHSA-fv5v-hfxp-5379).
- **Verified, P0:** Adobe APSB26-157 marks Premiere 26.3.2 and earlier and 25.6.5 and earlier affected by CVE-2026-84395; OpenCut records host versions but does not classify them against an advisory table (`opencut/tools/adobe_premierepro_versions.py`, `opencut/_generated/adobe_premierepro_versions.json`, https://helpx.adobe.com/security/products/premiere_pro/apsb26-157.html).
- **Verified, P0:** FFmpeg 8.1.3 was published on 2026-09-21, contradicting `RELEASE_LANE_OPEN = False` and the statement that 8.1.3 was never published (`opencut/core/ffmpeg_provenance.py:64`, https://ffmpeg.org/download.html). The lane must remain closed until every tracked fix is mapped to the 8.1.3 tag or a verified backport.
- **Verified, P0:** Frame.io retail accounts moved to V4 on 2026-06-01, but OpenCut hardcodes `https://api.frame.io/v2` (`opencut/core/frameio_integration.py:18`, https://help.frame.io/en/articles/9859849-adobe-premiere-frame-io-v4-comments-panel-overview). V4 OAuth, account identity, signatures, and resumable uploads need recorded contract tests.
- **Verified strength:** loopback binding, CSRF, host validation, SSRF checks, token-gated remote access, signed plugin registries, artifact verification, and crash logging already exist (`SECURITY.md`, `opencut/security.py`, `opencut/core/plugins.py`, `opencut/core/workflow.py`). New work should extend these boundaries rather than create parallel policy.
- **Recovery requirement:** every migration above must preserve the last known-good package store or integration state, expose a support-bundle receipt, and allow rollback without deleting user data (`opencut/user_data.py`, `opencut/core/workflow.py`, `opencut/core/version_compare.py`).

## Architecture Assessment

- **Time is not one domain.** `opencut/core/auto_edit.py:299`, `opencut/core/iso_ingest.py`, `opencut/core/multi_pov.py`, `opencut/core/multicam_xml.py`, and `opencut/core/script_to_roughcut.py` mix float seconds with rounded or integer frame rates; `opencut/core/sequence_index.py:138` explicitly does not handle drop-frame. A shared rational time type and boundary adapters should precede more timeline automation.
- **Interchange tests prove syntax more often than editorial meaning.** F418 covers rendered media bytes and F428 covers ORI review annotations, but OTIO, FCPXML, and AAF also need receipts for clip identity, rational ranges, transitions, retimes, reverse effects, enabled state, markers, links, and unknown metadata (`opencut/export/otio_export.py`, `opencut/core/fcpxml_export.py`, `opencut/core/edl_aaf.py`, https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues).
- **The panel hierarchy is still over-segmented.** Static markup contains 96 CEP card containers and 66 UXP cards, plus 316 and 149 buttons respectively. The existing compact-radius checks do not establish useful grouping (`extension/com.opencut.panel/client/index.html:227`, `extension/com.opencut.uxp/index.html:149`, `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:1018`). F455 should consolidate representative workflows before another surface is added.
- **State coverage partly tests a fixture instead of the product.** The six-state accessibility test injects loading, empty, error, permission, and confirmation nodes; the production-boundary helper covers offline plus limited empty/error indicators (`extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:982`, `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:2548`). F456 should drive real requests and host responses in both panels.
- **The host contract is stale.** The generated Adobe snapshot still ends at 26.3 even though Premiere 26.5.1 and UXP 26.5 are published; current APIs include clip transcription, language-pack checks, C2PA, media management, and work-area access (`opencut/_generated/adobe_premierepro_versions.json`, https://developer.adobe.com/premiere-pro/uxp/changelog/). Capability detection must preserve 25.6 through 26.4 behavior.
- **Documentation lacks a fact boundary.** README install commands use the wrong distribution name, model cards advertise nonexistent extras, UXP domain guidance disagrees with the live manifest, ARM64 guidance references a nonexistent workflow, and contributor counts trail generated inventory (`README.md:401`, `docs/MCP_SERVER.md:36`, `opencut/model_cards.py:105`, `docs/UXP_MACOS_HTTP.md:78`, `docs/WINDOWS_ARM64_PACKAGING.md:42`, `CONTRIBUTING.md:3`). F458 should compare public claims with `opencut/project_facts.py`, manifests, packaging policy, and generated counts.
- **Generated accounting is a strength with one gap.** The route manifest records 1,593 routes, 1,564 shipped routes, 29 strategic stubs, and 107 blueprints, but only 310 shipped routes have a direct first-party surface (`opencut/_generated/route_manifest.json`). F411, F412, and F414 already own readiness, control gating, and command discovery; no duplicate breadth item is warranted.
- **Category coverage:** security, observability, testing, documentation, packaging, resilience, cloud collaboration, migration, and upgrade strategy are addressed by F424 and F446 through F458. Accessibility is part of F455 and F456; multilingual caption quality is already F423. Plugin trust is already implemented and should be re-audited through F406. A mobile client conflicts with the Premiere desktop host boundary. A separate hosted multi-user system would duplicate Frame.io and the existing review model. Offline behavior is required where local model caches and Frame.io retry state cross a network boundary.

## Rejected Ideas

- **Build a full standalone NLE:** OpenCut's differentiator is Premiere automation through CEP/UXP, not timeline ownership (`README.md`, https://helpx.adobe.com/premiere/desktop/whats-new/release-notes.html).
- **Clone Premiere Paper Edit:** OpenCut already has `opencut/core/paper_edit.py`, while Premiere 26.5 now ships the native workflow. Invest in cross-project search, marker/select handoff, receipts, and repeatable batch work instead (https://community.adobe.com/announcements-732/now-in-beta-paper-edit-to-text-based-editing-1627992).
- **Make generic chat the primary UI:** editor research and current commercial patterns support plan, preview, alternatives, accept, and revert rather than unconstrained conversation (`opencut/core/autonomous_agent.py`, https://research.adobe.com/publication/videodiff-human-ai-video-co-creation-with-alternatives/).
- **Replace Flask with UXP Hybrid now:** Hybrid requires Premiere 26.2+, platform binaries, CCX packaging, and macOS notarization, with no measured reliability gain for OpenCut yet (https://developer.adobe.com/premiere-pro/uxp/plugins/hybrid-plugins/).
- **Add generic C2PA 2.4 or IMSC 1.3 projects:** current code already implements the relevant provenance and caption concepts in `opencut/core/c2pa_sidecar.py`, `opencut/core/c2pa_embed.py`, `opencut/core/caption_interchange.py`, and `opencut/core/caption_compliance.py`; conformance belongs in F418, F423, and F429 (https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html, https://www.w3.org/TR/ttml-imsc1.3/).
- **Add a mobile companion:** host actions, local media, and the current security model are desktop-bound, while review access is already served by bundles and cloud integrations (`SECURITY.md`, `opencut/core/review_bundle.py`).
- **Build a hosted multi-user review service:** Frame.io V4 and OpenCut's existing review model cover the credible collaboration need without adding account, tenancy, moderation, and retention systems (`opencut/core/review_links.py`, `opencut/core/review_comments.py`, https://help.frame.io/en/articles/9859849-adobe-premiere-frame-io-v4-comments-panel-overview).
- **Hard-pin OpenTimelineIO 0.18 as the only path:** 0.18.0 and 0.18.1 are still marked pre-release, so preserve new fields behind compatibility tests without removing the stable path (https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases).
- **Add SAM 2 immediately:** interactive masking is relevant, but current evidence does not justify its model size, cold start, VRAM, or packaging cost without a benchmark against `opencut/core/motion_tracking.py` and `opencut/core/motion_brush.py` (https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/).

## Sources

### Repository and tracker

- https://github.com/SysAdminDoc/OpenCut
- https://github.com/SysAdminDoc/OpenCut/issues/7
- https://github.com/SysAdminDoc/OpenCut/issues/8
- https://github.com/SysAdminDoc/OpenCut/issues/6
- https://github.com/SysAdminDoc/OpenCut/issues/5
- https://github.com/SysAdminDoc/OpenCut/discussions/9
- https://github.com/SysAdminDoc/OpenCut/discussions/10
- https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.55.1

### Open-source and adjacent projects

- https://kdenlive.org/news/releases/
- https://invent.kde.org/multimedia/kdenlive/-/issues
- https://www.shotcut.org/blog/
- https://github.com/mltframework/shotcut/issues
- https://github.com/mifi/lossless-cut/releases
- https://github.com/mifi/lossless-cut/blob/master/issues.md
- https://github.com/WyattBlue/auto-editor/releases
- https://auto-editor.com/
- https://github.com/SubtitleEdit/subtitleedit/releases
- https://www.nikse.dk/subtitleedit/help
- https://www.scenedetect.com/changelog/
- https://www.scenedetect.com/docs/latest/api/detectors.html
- https://github.com/OpenShot/openshot-qt/releases
- https://jliljebl.github.io/flowblade/webpage/
- https://www.mltframework.org/
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues
- https://lf-aswf.atlassian.net/wiki/spaces/PRWG/pages/605814827/OTIO%2B2D-Annotations%2BInterchange%2Bspecification
- https://github.com/OpenAssetIO/OpenAssetIO
- https://github.com/ascmitc/mhl-specification

### Awesome lists

- https://github.com/ad-si/awesome-video-production
- https://github.com/Supersynergy/awesome-ai-video-editing

### Commercial products and platform APIs

- https://helpx.adobe.com/premiere/desktop/whats-new/release-notes.html
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://developer.adobe.com/premiere-pro/uxp/ppro-reference/
- https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/overview/
- https://developer.adobe.com/premiere-pro/uxp/plugins/hybrid-plugins/
- https://blog.developer.adobe.com/en/publish/2026/09/investing-in-the-future-of-creative-cloud-extensibility-uxp-comes-to-our-flagship-applications
- https://www.blackmagicdesign.com/products/davinciresolve/whatsnew
- https://www.blackmagicdesign.com/products/davinciresolve/collaboration
- https://www.capcut.com/tools/desktop-ai-power
- https://www.capcut.com/tools/video-transcript-editing
- https://feedback.descript.com/changelog
- https://help.descript.com/script-editing/fix-take
- https://changelog.veed.io/
- https://www.veed.io/tools/openedit
- https://help.runwayml.com/hc/en-us/articles/51683104370451-Creating-with-Edit-Studio
- https://help.runwayml.com/hc/en-us/articles/51601639579667-Creating-with-Runway-Agent
- https://next.developer.frame.io/platform/v4/docs/quick-start
- https://next.developer.frame.io/platform/docs/guides/webhooks
- https://next.developer.frame.io/platform/docs/guides/uploading-to-frame-io/how-local-remote-uploads-work
- https://help.frame.io/en/articles/9859849-adobe-premiere-frame-io-v4-comments-panel-overview

### Standards

- https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- https://spec.c2pa.org/specifications/specifications/2.4/security/Security_Considerations.html
- https://c2pa.org/conformance/
- https://www.w3.org/TR/ttml-imsc1.3/
- https://www.w3.org/TR/webvtt1/
- https://www.w3.org/TR/WCAG22/

### Research and community signal

- https://research.adobe.com/publication/videodiff-human-ai-video-co-creation-with-alternatives/
- https://research.adobe.com/publication/chunkyedit-text-first-video-interview-editing-via-chunking/
- https://research.adobe.com/publication/b-script-transcript-based-b-roll-video-editing-with-recommendations/
- https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/
- https://arxiv.org/abs/2109.07809
- https://news.ycombinator.com/item?id=41174996
- https://www.reddit.com/r/editors/comments/q419v1/
- https://www.reddit.com/r/editors/comments/1vd48jr/aug_2026_open_source_tools_devs/
- https://stackoverflow.com/questions/tagged/adobe-premiere?tab=Active
- https://community.adobe.com/feature-requests-730/feature-request-convert-transcripts-into-sequence-markers-1327693
- https://community.adobe.com/feature-requests-730/use-a-marker-to-select-in-and-out-on-the-sequence-or-clip-a-feature-request-1328954
- https://community.adobe.com/announcements-732/search-markers-across-your-entire-project-1549479

### Dependencies and advisories

- https://helpx.adobe.com/security/products/premiere_pro/apsb26-157.html
- https://helpx.adobe.com/sg/security/products/premiere_pro/apsb26-76.html
- https://ffmpeg.org/download.html
- https://ffmpeg.org/security.html
- https://raw.githubusercontent.com/FFmpeg/FFmpeg/release/8.1/Changelog
- https://github.com/advisories/GHSA-575m-jfmw-q76c
- https://pyinstaller.org/en/latest/CHANGES.html
- https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr
- https://github.com/pytorch/pytorch/releases
- https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p
- https://github.com/huggingface/transformers/releases
- https://github.com/advisories/GHSA-fv5v-hfxp-5379
- https://flask.palletsprojects.com/en/stable/changes/
- https://github.com/pallets/werkzeug/security/advisories/GHSA-87hc-h4r5-73f7

## Open Questions

- **Needs live validation:** Can current main reproduce Discussion #10 in a disposable Windows profile across no-system-Python, Python 3.12, Python 3.13, and Python 3.14 PATH states, with each supported Torch build (https://github.com/SysAdminDoc/OpenCut/discussions/10)?
- **Needs live validation:** Which direct UXP actions pass captured UXP Developer Tool tests on Premiere 25.6, 26.2, 26.3, and 26.5? F386 in `Roadmap_Blocked.md` already tracks the required live-host authority.
- **Needs credentials or recorded fixtures:** Which Frame.io V4 account and webhook capabilities are available to the maintainer's Adobe organization? F446 can proceed with recorded contracts, but final OAuth and webhook verification needs an eligible account (https://next.developer.frame.io/platform/v4/docs/quick-start).
