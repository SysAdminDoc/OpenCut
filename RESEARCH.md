# Research — OpenCut

Date: 2026-08-23 — replaces all prior research.

Confidence labels used below:

- **Verified**: confirmed from repository state, a first-party specification, a release artifact, or a directly inspected tracker record.
- **Likely**: supported by several credible sources, but the exact OpenCut behavior still needs a fixture or host run.
- **Needs live validation**: the risk is verified, but the platform-specific result must be measured in a built artifact or Premiere host.

## Executive Summary

OpenCut v1.55.0 is a local-first Premiere automation system with a large Flask media backend, CEP and UXP panels, reviewable timeline mutations, delivery tooling, interchange, and agent-facing MCP operations. At commit `dcba7022`, its generated facts report 1,593 routes, 1,564 shipped routes, 29 explicit stubs, and 107 blueprints; the published source and Windows release both identify v1.55.0 (`opencut/_generated/project_facts.json`, `opencut/_generated/route_manifest.json`, [v1.55.0](https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.55.0)). **Verified.** Breadth is no longer the useful competitive target. The highest-value direction is provable trust: secure every decoder and model ingress path, expose only actions that can run, verify every produced artifact, and make complex edits inspectable before they touch a timeline.

| Order | Opportunity | Tier | Impact | Effort | Risk and novelty | Evidence |
|---:|---|---|---:|---|---|---|
| 1 | Remove vulnerable FFmpeg copies embedded in release dependencies | Now, P0 | 5 | L | Compatibility risk; security parity | `requirements-release-lock.txt:867`, [opencv-python PR 1255](https://github.com/opencv/opencv-python/pull/1255) |
| 2 | Harden Hugging Face downloads and remote model code | Now, P0 | 5 | M | Resolver and model-compatibility risk; security parity | `requirements-release-lock.txt:459`, `opencut/core/captions_enhanced.py:395`, [huggingface-hub 1.26.0](https://github.com/huggingface/huggingface_hub/releases/tag/v1.26.0) |
| 3 | Make resolved readiness authoritative from registry through every surface | Now, P0 | 5 | L | State-migration risk; leapfrog trust | `opencut/registry.py:148`, `opencut/mcp_extended_tools.py:138`, `extension/com.opencut.panel/client/feature-state.js:224` |
| 4 | Replace literal documentation assertions with source-derived conformance | Now, P1 | 4 | M | Low implementation risk; trust parity | `docs/RELEASE_PROVENANCE.md:32`, `docs/PYTHON_ADVISORIES.md:16`, `tests/test_release_provenance_attestation.py:31` |
| 5 | Repair realtime, indexing, and persisted-settings contracts | Now, P1 | 4 | L | Migration risk; product correctness | `opencut/routes/search.py:71`, `opencut/routes/system_realtime_routes.py:91`, `opencut/user_data.py:815` |
| 6 | Apply a common post-output validation receipt to every media job | Next, P1 | 5 | L | Runtime cost; leapfrog reliability | `opencut/core/smart_render.py:430`, `opencut/core/declarative_compose.py:493` |
| 7 | Evaluate agent-authored edits with executable fixture briefs | Next, P1 | 4 | M | Creative judges can overfit; leapfrog testing | `opencut/core/ai_eval_harness.py:1`, [EditDuet](https://arxiv.org/abs/2509.10761) |
| 8 | Complete transcript and caption trust workflows | Next, P1 | 4 | L | Language false positives; professional parity | `opencut/core/captions.py:1097`, `opencut/core/caption_qc.py:220`, [FCC caption quality](https://docs.fcc.gov/public/attachments/FCC-14-12A1_Rcd.pdf) |
| 9 | Carry unresolved review comments across cut versions | Later, P2 | 4 | M | Incorrect remapping; leapfrog review continuity | `opencut/core/review_links.py:35`, [review retiming request](https://www.reddit.com/r/editors/comments/q419v1) |
| 10 | Add resource-governed visual search and evidence-ranked audio repair | Later, P2 | 3 | L | Workstation contention and metric bias; measured parity | `opencut/core/federated_media_index.py:1184`, [DNSMOS](https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/README.md) |

## Product Map

### Core workflows

- Analyze local media, transcribe speech, detect silence or filler words, propose edits, and stage reviewable changes before Premiere write-back (`README.md:339`, `opencut/core/transcript_edit.py`, `opencut/core/cut_review.py`). **Verified.**
- Build captions, audio repairs, reframes, multicam decisions, highlights, and delivery variants through durable background jobs (`opencut/routes/jobs_routes.py`, `opencut/core/captions.py`, `opencut/core/multicam.py`). **Verified.**
- Review versions, comments, drawings, comparisons, and portable review bundles without requiring a hosted account (`opencut/core/review_links.py`, `opencut/core/review_bundle.py`, `opencut/core/review_portal.py`). **Verified.**
- Export media and timelines through FFmpeg, OTIO, AAF, MLT, FCP XML, caption sidecars, broadcast checks, and C2PA 2.4 provenance (`README.md:327`, `opencut/core/delivery_validate.py`, `opencut/core/c2pa_sidecar.py`). **Verified.**
- Search indexed media and expose controlled operations through the panels, CLI, REST, and MCP (`opencut/core/federated_media_index.py`, `opencut/cli.py`, `opencut/mcp_server.py`). **Verified.**

### User personas

- Premiere editors who want repetitive work automated without surrendering the final cut to an opaque service (`README.md:1`, `opencut/core/cut_review.py`). **Verified.**
- Privacy-sensitive creators and production teams that prefer local models, local project data, portable review bundles, and explicit cloud boundaries (`README.md:22`, `opencut/registry.py`). **Verified.**
- Technical operators who need batch, CLI, REST, or MCP access in addition to panel controls (`README.md:18`, `docs/MCP_SERVER.md`). **Verified.**

### Platforms and distribution

- Windows 10/11, macOS, and Linux are declared. Windows has a WPF installer; source installs serve macOS and Linux; Docker, Flatpak, and AppImage lanes exist (`README.md:68`, `installer/src/OpenCut.Installer/OpenCut.Installer.csproj`, `Dockerfile`, `io.github.sysadmindoc.opencut.yml`). **Verified.**
- CEP supports Premiere Pro 2019 and later. UXP targets Premiere 25.6 and later, with API typings pinned to 26.3 (`extension/com.opencut.panel/CSXS/manifest.xml`, `extension/com.opencut.uxp/manifest.json`, `extension/com.opencut.panel/package.json`). **Verified.**
- Adobe has not published an exact CEP removal version. UXP should remain the primary migration target, while CEP stays a tested fallback until Adobe publishes a firm cutoff ([Adobe UXP changelog](https://developer.adobe.com/premiere-pro/uxp/changelog/), [CEP transition notice](https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md)). **Verified for the published record; Needs live validation for host-specific parity.**

### Key integrations and data flows

- Both panels call a localhost Flask backend. Host mutations cross CEP ExtendScript or Premiere UXP actions; media processing uses external FFmpeg plus Python-native media libraries (`CLAUDE.md:31`, `extension/com.opencut.panel/host/index.jsx`, `extension/com.opencut.uxp/main.js`). **Verified.**
- Generated route, readiness, OpenAPI, MCP, parity, and project-fact manifests are intended to bind code to public claims, but several consumers use declared state rather than resolved runtime state (`opencut/_generated/`, `opencut/registry.py:168`). **Verified.**
- Project state is distributed across transcripts, JSON and SQLite indexes, caption sidecars, review records, OTIO timelines, and C2PA manifests. Stable identity and one clear migration contract are therefore more valuable than another isolated feature (`opencut/core/transcript_cache.py`, `opencut/core/footage_index_db.py`, `opencut/core/caption_roundtrip.py`, `opencut/core/review_links.py`). **Verified.**

## Competitive Landscape

### Adobe Premiere Pro, DaVinci Resolve, and Final Cut Pro

These products pair editing fundamentals with local search, transcription, masking, multicam assistance, and increasingly visible content provenance ([Premiere features](https://www.adobe.com/products/premiere/features.html), [Resolve](https://www.blackmagicdesign.com/products/davinciresolve), [Final Cut release notes](https://support.apple.com/en-us/102825)). **Verified.** OpenCut should learn from their tight coupling between analysis, source context, undo, and final output. It should avoid competing on raw feature count or copying metered generative functions that weaken its local-first position.

### AutoCut, FireCut, TimeBolt, AutoPod, and Excalibur

Focused plugins win by turning one painful edit into a short, previewable flow ([AutoCut](https://www.autocut.com/en/), [FireCut](https://firecut.ai/), [TimeBolt](https://www.timebolt.io/), [AutoPod](https://www.autopod.fm/), [Excalibur](https://knightsoftheeditingtable.com/excalibur)). **Verified.** Their strongest lesson is narrow intent, visible parameters, and a review step. Community reports about dense cut lists, rigid audio assumptions, and host-version breakage support OpenCut's existing non-destructive review posture, not more one-click mutation. **Likely.**

### Descript and Riverside

Transcript-first editing makes timing, speaker context, and immediate audition part of the editing surface ([Descript video editor](https://www.descript.com/video-editor), [Riverside editor](https://riverside.fm/video-editor)). **Verified.** OpenCut already has transcript-to-timeline mappings and a separate waveform view; it should combine them into one timing workbench. It should avoid cloud-only project state and credit-dependent iteration.

### CapCut, OpusClip, VEED, Gling, and Wisecut

These tools make social repurposing approachable through presets, automatic captions, silence removal, and ranked clips ([CapCut editor](https://www.capcut.com/tools/online-video-editor), [OpusClip](https://www.opus.pro/), [VEED](https://www.veed.io/tools/ai-video-editor), [Gling](https://www.gling.ai/), [Wisecut](https://www.wisecut.video/)). **Verified.** OpenCut should explain why a moment or edit was selected and let editors preview every mutation. It should avoid opaque virality scores, credit pressure, and template overload.

### Frame.io and Kitsu

Versioned review, annotations, comment history, comparison, and approvals are table stakes for professional review ([Frame.io review](https://frame.io/review-and-approval), [Kitsu review](https://kitsu.cg-wire.com/review/)). **Verified.** OpenCut already covers much of the static bundle and portal flow. The remaining high-value gap is carrying unresolved comments safely across a recut. A hosted production-management clone would conflict with the product's local and portable design.

### auto-editor and LosslessCut

Both projects show demand for fast, deterministic timeline automation and smart copy paths ([auto-editor](https://github.com/WyattBlue/auto-editor), [LosslessCut](https://github.com/mifi/lossless-cut)). **Verified.** Their issue patterns reinforce explicit copy versus re-encode plans, golden interchange fixtures, and post-output inspection. OpenCut should avoid implying frame accuracy when keyframe or adapter limits prevent it.

### Subtitle Edit and Aegisub

These tools make waveform timing, image-subtitle OCR, language repair, typesetting, and automation extensibility first-class ([Subtitle Edit releases](https://github.com/SubtitleEdit/subtitleedit/releases), [Aegisub](https://github.com/TypesettingTools/Aegisub)). **Verified.** OpenCut should borrow their timing ergonomics, accessibility audit depth, and bitmap-subtitle ingest. It should not become a general subtitle editor disconnected from Premiere.

### Kdenlive, Shotcut, OpenShot, Flowblade, and Olive

Open NLE trackers repeatedly concentrate on preview versus render differences, undo reliability, audio discontinuities, large-project load, and format edge cases ([Kdenlive releases](https://kdenlive.org/news/releases/), [Shotcut](https://github.com/mltframework/shotcut), [OpenShot](https://github.com/OpenShot/openshot-qt), [Flowblade](https://github.com/jliljebl/flowblade), [Olive](https://github.com/olive-editor/olive)). **Likely.** The lesson is to spend roadmap capacity on verified outputs and recovery, not to add a second generic timeline.

### Remotion, editly, Motion Canvas, and Revideo

Code-driven video systems favor deterministic compositions, reusable assets, content-addressed caching, and render fixtures ([Remotion](https://github.com/remotion-dev/remotion), [editly](https://github.com/mifi/editly), [Motion Canvas](https://github.com/motion-canvas/motion-canvas), [Revideo](https://github.com/redotvideo/revideo)). **Verified.** OpenCut should borrow fixture-driven output checks and partial reuse. A general node editor would add another surface without improving Premiere work.

### Agentic editing projects and research

Young projects such as premiere-agent and current editing research use boundary validation, evidence retrieval, and Editor/Critic loops rather than trusting a single generated plan ([premiere-agent](https://github.com/Kemerd/premiere-agent), [EditDuet](https://arxiv.org/abs/2509.10761), [MEDit-Bench](https://arxiv.org/abs/2607.25300)). **Likely.** OpenCut's advantage should be executable conformance against real media and timeline invariants. Unsupervised full-edit mutation is not recommended.

## Reported Issues

- The GitHub tracker had zero open issues and zero open pull requests on 2026-08-23. No tracker item therefore requires a new roadmap entry ([repository](https://github.com/SysAdminDoc/OpenCut/issues), [pull requests](https://github.com/SysAdminDoc/OpenCut/pulls)). **Verified.**
- Discussions 3 and 4 had no comments and contained no actionable defect or feature request on 2026-08-23 ([discussion 3](https://github.com/SysAdminDoc/OpenCut/discussions/3), [discussion 4](https://github.com/SysAdminDoc/OpenCut/discussions/4)). **Verified.**
- Closed issue 5 documented the CEP CSRF bootstrap failure; the fix is present and the issue closed on 2026-08-22. Closed issues 1 and 2 concern UXP typing drift and panel dependency/bootstrap behavior already represented in shipped code and history ([issue 5](https://github.com/SysAdminDoc/OpenCut/issues/5), [issue 1](https://github.com/SysAdminDoc/OpenCut/issues/1), [issue 2](https://github.com/SysAdminDoc/OpenCut/issues/2)). **Verified.**
- The local blocked-work ledger contains stale release and queue statements even though publication and queue classification have landed (`Roadmap_Blocked.md:29`, `Roadmap_Blocked.md:104`, `opencut/routes/jobs_routes.py:184`). This is documentation drift, not an open external report. **Verified.**

## Security, Privacy, and Reliability

| Finding | Assessment | Required guardrail |
|---|---|---|
| OpenCV 5.0.0.93 is locked at `requirements-release-lock.txt:867`; its 5.x Linux wheel recipe pins FFmpeg 8.1.1 while the upstream fix for CVE-2026-8461 remains in PR 1255. OpenCut's external FFmpeg gate does not attest embedded copies. | **Verified on Linux; Needs live validation on Windows and macOS.** [OpenCV release 93](https://github.com/opencv/opencv-python/releases/tag/93), [PR 1255](https://github.com/opencv/opencv-python/pull/1255) | Inventory linked media libraries in every artifact and fail release when any decoder copy misses the project advisory matrix. |
| `huggingface-hub==1.24.0` is locked, while 1.26.0 fixes CVE-2026-15717 path handling. OpenCut calls `snapshot_download(..., local_dir=...)` and has eight `trust_remote_code=True` call sites (`requirements-release-lock.txt:459`, `opencut/core/captions_enhanced.py:395`, `opencut/core/object_removal.py:484`). | **Verified.** [1.26.0 release](https://github.com/huggingface/huggingface_hub/releases/tag/v1.26.0) | Raise the secure floor, test hostile repository filenames, pin immutable revisions, prefer safetensors, and allow remote code only for reviewed model IDs and revisions. |
| Public security and dependency prose contradicts executable policy. The release guide accepts FFmpeg 8.1.1 and pins 8.1.2, while `ffmpeg_provenance.py` closes the release lane; Python advisory prose waives transformer findings absent from `ALLOWED_ADVISORIES` (`docs/RELEASE_PROVENANCE.md:32`, `opencut/core/ffmpeg_provenance.py:59`, `docs/PYTHON_ADVISORIES.md:16`, `opencut/tools/pip_audit_extras.py:57`). | **Verified.** | Generate or compare documentation facts against executable policy. Do not pin stale prose in tests. |
| PyInstaller 6.20.0 is below the 6.22.1 security fix, although OpenCut's canonical onedir artifact is not exposed to the modern onefile path. ONNX Runtime 1.27.0 and PyAV 18.0.0 trail releases with media and model hardening; general and release locks disagree on Werkzeug (`requirements-build.txt:3`, `requirements-release-lock.txt:64`, `:841`, `:1381`, `requirements-lock.txt:30`). | **Verified for version deltas; Needs live validation for provider compatibility.** [PyInstaller advisory](https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr), [ONNX Runtime 1.29.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0), [PyAV 18.1.0](https://github.com/PyAV-Org/PyAV/releases/tag/v18.1.0) | Upgrade through a CPU, GPU, frozen-artifact, and media-fixture matrix. Keep the artifact onedir assertion and pin the Docker base by digest. |
| A static count found 1,817 broad `Exception` or `BaseException` catches across 457 Python files and 165 empty JavaScript or JSX catches. Some are deliberate compatibility probes, but there is no ratchet that distinguishes reviewed suppression from silent loss (`opencut/checks.py`, `extension/com.opencut.panel/client/main.js`). | **Verified.** | Add explicit suppression reasons, structured failure evidence, and a no-growth gate. Burn down job, bridge, file, and model-loading paths first. |
| Smart render stages, probes, decodes, and atomically promotes output, but many producers stop at process success or file existence. Declarative compose explicitly says post-render durations “trust spec” (`opencut/core/smart_render.py:430`, `opencut/core/declarative_compose.py:493`). | **Verified.** | Introduce one output contract and evidence receipt for stream layout, duration, geometry, timestamps, sample decode, and atomic publication. |
| Native keyring is the intended secret store, but alternate or plaintext backends are not rejected by a release or runtime policy (`opencut/core/secret_store.py`, [keyring documentation](https://keyring.readthedocs.io/en/latest/index.html)). | **Likely.** | Fail closed for production credentials when the resolved backend is not an approved OS store; keep C2PA production keys external. |

Recovery should follow the pattern already proven in `opencut/core/smart_render.py` and `opencut/core/guarded_download.py`: write beside the destination, validate, fsync, promote atomically, preserve the prior destination, and return machine-readable evidence. **Verified.**

## Architecture Assessment

| Boundary | Finding | Recommendation and dependency |
|---|---|---|
| Readiness registry | Eighteen generated feature records disagree with terminal-stub resolution; queue admission, extended MCP, and OpenAPI do not share one resolved per-adapter state (`opencut/registry.py:148`, `opencut/_generated/feature_readiness.json`, `opencut/mcp_extended_tools.py:138`). **Verified.** | Make `resolved_state()` authoritative first. Then update queue, MCP, OpenAPI, generated manifests, and install guidance. |
| Panel availability | CEP loads a feature-state helper but production markup has no `data-feature-id` bindings; UXP does not consume `/system/feature-state`; UXP reconnect enables controls without restoring semantic prerequisites (`extension/com.opencut.panel/client/feature-state.js:224`, `extension/com.opencut.uxp/main.js:6819`). **Verified.** | Bind every dependency-sensitive control, share the state reducer, and prove disable plus re-enable behavior in rendered tests. |
| Command discovery | The backend catalog has 225 entries but only 43 runnable routes. The other 182 are accepted as `missing_route`, and direct-surface accounting counts the backend catalog as a user surface (`opencut/core/command_palette.py:400`, `tests/test_ux_intelligence.py:68`, `opencut/tools/dump_route_manifest.py:390`). **Verified.** | Generate one live catalog with invocation contracts. Move aspirational entries out of user search and count only literal consumer bindings as direct surface. |
| Realtime bridge | CEP hardcodes port 5680, UXP consumes the port returned by `/ws/start`, start returns before bind completion, and status discards the last bind error (`extension/com.opencut.panel/client/main.js:5562`, `extension/com.opencut.uxp/main.js:7615`, `opencut/routes/system_realtime_routes.py:91`). **Verified.** | Return success only after bind, propagate the selected port and last error, and exercise collision plus restart fixtures. |
| Media indexes | Visible folder indexing is nonrecursive, capped at 100, and retranscribes unchanged files. JSON, SQLite, and federated stores expose different search, status, and clear behavior; the clear route ignores a failed result (`opencut/routes/search.py:71`, `opencut/core/footage_search.py:314`, `opencut/core/footage_index_db.py`). **Verified.** | Route the visible flow through incremental indexing, add a recursive preflight, and unify stores behind one versioned adapter contract. |
| Persisted settings | Footage-index settings are stored but not consumed; visible UXP indexing hardcodes the base model; chapter and multicam execution bypass saved defaults; color-profile persistence has no production consumer (`opencut/user_data.py:815`, `:860`, `:875`, `extension/com.opencut.uxp/main.js:5956`). **Verified.** | Give every persisted key one consumer and one conformance test, or remove it with a migration. |
| Evaluation | The evaluation system measures individual model latency and caller-supplied quality. It does not execute an editorial brief and inspect the resulting timeline (`opencut/core/ai_eval_harness.py:1`, `opencut/core/eval_datasets.py:56`). **Verified.** | Add media fixtures, fixture briefs, deterministic invariants, and fault injection before any preference judge. |
| Transcript and caption trust | Repetition loops are flagged, but dropped windows, regressing word times, abnormal coverage gaps, and caption completeness against speech are not gated (`opencut/core/captions.py:1097`, `opencut/core/caption_qc.py:220`). **Verified.** | Add backend-independent ASR integrity checks, then expose them in a combined transcript, waveform, shot, and confidence workbench. |
| Review interchange | Comments are version-bound and review bundles export ordinary OTIO markers plus SVG drawings. There is no confidence-aware carry-forward or ORI annotation exchange (`opencut/core/review_links.py:35`, `opencut/core/review_bundle.py:368`). **Verified.** | Add carry-forward first. Add ORI import/export as an optional adapter while the specification matures. |
| Host and runtime migration | OpenCut pins Premiere API 26.3, while OTIO 0.18.1 documents Python only through 3.12 and OpenCut claims 3.11 through 3.14 (`extension/com.opencut.panel/package.json`, `pyproject.toml:22`, [OTIO 0.18.1](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases/tag/v0.18.1)). **Verified for declared support; Needs live validation on Python 3.13 and 3.14.** | Keep host and interpreter matrices executable. Disable unsupported capabilities with an honest reason instead of relying on package resolution. |

The 2026-08-22 rendered CEP and UXP pass found no remaining visual defect across its 72-case theme, width, motion, and forced-color matrix, so another generic polish item is not justified. Installer, packaging, plugin-runtime, MCP-depth, core-wide, CLI-depth, streaming-load, and live-Premiere audits remain recorded under F406 rather than being duplicated here (`ROADMAP.md:220`). **Verified.**

Accessibility, i18n, offline resilience, multi-user review, packaging, migration, plugins, and mobile were all considered. Caption completeness and review carry-forward need new work. i18n and rendered accessibility gates are green; durable jobs and local operation already cover the primary offline case. Packaging and plugin inspection remain F406. A separate mobile product would not fit a Premiere-hosted desktop tool (`scripts/i18n_lint.py`, `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs`, `opencut/core/workflow.py`, `ROADMAP.md:220`). **Verified.**

## Rejected Ideas

- A generic standalone NLE or mobile editor. OpenCut's differentiator is deep Premiere automation, local processing, and interoperable outputs; Kdenlive, Shotcut, OpenShot, and the unrelated [OpenCut web editor](https://github.com/OpenCut-app/OpenCut) already occupy generic editing. **Verified.**
- A hosted Frame.io or Kitsu clone. Comment carry-forward and portable annotation interchange fit the existing local review bundles; accounts, hosting, and production tracking do not ([Frame.io](https://frame.io/review-and-approval), [Kitsu](https://kitsu.cg-wire.com/review/)). **Verified.**
- One-prompt unsupervised full edits. Editor reports and agentic-editing research favor bounded operations, dry runs, evidence, and approval ([editor discussion](https://www.reddit.com/r/editing/comments/1td336d/has_ai_actually_replaced_video_editing_workflows/), [EditDuet](https://arxiv.org/abs/2509.10761)). **Likely.**
- A mandatory large video model or generative-video replacement lane. VRAM, runtime trust, visual inconsistency, rights, and packaging costs conflict with dependable local operation ([InternVideo3](https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo3/README.md), [OmniEdit-Bench](https://arxiv.org/abs/2608.05049)). **Likely.**
- Direct adoption of Temporal, LangGraph, OpenCue, or a general node editor. OpenCut already has checkpoints, approval gates, cancellation, and durable jobs; borrowing partial-execution semantics is cheaper than adding an operations platform ([Temporal](https://docs.temporal.io/), [LangGraph breakpoints](https://langchain-ai.github.io/langgraph/concepts/breakpoints/), [OpenCue](https://github.com/AcademySoftwareFoundation/OpenCue)). **Verified.**
- Another default denoising model. OpenCut already includes several repair engines; evidence shows denoising can reduce ASR quality, so selection and audition are the unmet work ([ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio), [When Denoising Hinders](https://arxiv.org/abs/2603.04710)). **Verified for available engines; Likely for per-source ranking.**
- More C2PA fields or a second MCP implementation. C2PA 2.4 and MCP 2026-07-28 behavior are already present with conformance tests (`opencut/core/c2pa_sidecar.py:34`, `tests/test_mcp_protocol_conformance.py`, [C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html), [MCP changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)). **Verified.**
- Another IMSC 1.3 implementation. OpenCut already uses the final 2026-05-21 profile designator and independent reference validators (`opencut/core/caption_interchange.py:28`, `opencut/core/standards_validators.py:107`, `tests/test_standards_validators.py:75`, [IMSC 1.3](https://www.w3.org/TR/ttml-imsc1.3/)). **Verified.**
- Implementing all 29 model stubs. They are speculative integration entries, and several upstream models carry high hardware or maintenance cost. Truthful readiness and exposure must land before any individual adapter is reconsidered (`opencut/_generated/route_manifest.json`, `opencut/registry.py:148`). **Verified.**
- A blind NumPy 2.5 or librosa 1.0 upgrade. NumPy 2.5 drops the Python 3.11 lane and librosa 1.0 is a breaking migration ([NumPy 2.5.2](https://github.com/numpy/numpy/releases/tag/v2.5.2), [librosa 1.0.0](https://github.com/librosa/librosa/releases/tag/1.0.0)). **Verified.**
- Declaring a fixed CEP retirement release. Adobe has not published one; a date invented by OpenCut would mislead maintainers ([CEP transition notice](https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md)). **Verified.**

## Sources

### Repository and tracker

- <https://github.com/SysAdminDoc/OpenCut>
- <https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.55.0>
- <https://github.com/SysAdminDoc/OpenCut/issues/5>
- <https://github.com/SysAdminDoc/OpenCut/discussions/3>
- <https://github.com/SysAdminDoc/OpenCut/discussions/4>

### Direct and adjacent open-source projects

- <https://github.com/WyattBlue/auto-editor>
- <https://github.com/mifi/lossless-cut>
- <https://github.com/mltframework/shotcut>
- <https://kdenlive.org/news/releases/>
- <https://github.com/SubtitleEdit/subtitleedit/releases>
- <https://github.com/TypesettingTools/Aegisub>
- <https://github.com/m-bain/whisperX/issues>
- <https://github.com/SYSTRAN/faster-whisper/issues>
- <https://github.com/remotion-dev/remotion>
- <https://github.com/modelscope/ClearerVoice-Studio>
- <https://github.com/OpenAssetIO/OpenAssetIO>
- <https://github.com/ascmitc/mhl-specification>
- <https://github.com/AcademySoftwareFoundation/OpenCue>
- <https://kitsu.cg-wire.com/review/>
- <https://github.com/Supersynergy/awesome-ai-video-editing>
- <https://github.com/brandonhimpfen/awesome-audiovisual>

### Commercial products

- <https://www.adobe.com/products/premiere/features.html>
- <https://www.blackmagicdesign.com/products/davinciresolve>
- <https://support.apple.com/en-us/102825>
- <https://www.descript.com/video-editor>
- <https://riverside.fm/video-editor>
- <https://www.capcut.com/tools/online-video-editor>
- <https://www.opus.pro/>
- <https://frame.io/review-and-approval>
- <https://www.autocut.com/en/>
- <https://firecut.ai/>
- <https://www.timebolt.io/>
- <https://knightsoftheeditingtable.com/excalibur>

### Standards and platform APIs

- <https://developer.adobe.com/premiere-pro/uxp/changelog/>
- <https://github.com/adobe/premierepro-types/releases/tag/v26.3.0>
- <https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md>
- <https://opentimelineio.readthedocs.io/en/latest/tutorials/feature-matrix.html>
- <https://lf-aswf.atlassian.net/wiki/spaces/PRWG/pages/605814827/OTIO%2B2D-Annotations%2BInterchange%2Bspecification>
- <https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html>
- <https://modelcontextprotocol.io/specification/2026-07-28/changelog>
- <https://www.w3.org/TR/webvtt1/>
- <https://www.w3.org/TR/ttml-imsc1.3/>
- <https://www.w3.org/TR/WCAG22/>
- <https://www.section508.gov/create/captions-transcripts/>
- <https://docs.fcc.gov/public/attachments/FCC-14-12A1_Rcd.pdf>
- <https://ffmpeg.org/security.html>
- <https://slsa.dev/spec/v1.2/>

### Dependencies and security

- <https://github.com/opencv/opencv-python/releases/tag/93>
- <https://github.com/opencv/opencv-python/pull/1255>
- <https://nvd.nist.gov/vuln/detail/CVE-2026-8461>
- <https://github.com/huggingface/huggingface_hub/releases/tag/v1.26.0>
- <https://nvd.nist.gov/vuln/detail/CVE-2026-15717>
- <https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr>
- <https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0>
- <https://github.com/PyAV-Org/PyAV/releases/tag/v18.1.0>
- <https://devguide.python.org/versions/>
- <https://keyring.readthedocs.io/en/latest/index.html>

### Research and community signals

- <https://arxiv.org/abs/2509.10761>
- <https://arxiv.org/abs/2607.25300>
- <https://arxiv.org/abs/2608.05049>
- <https://arxiv.org/abs/2603.04710>
- <https://arxiv.org/abs/2506.18883>
- <https://arxiv.org/abs/2606.15320>
- <https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo3/README.md>
- <https://www.reddit.com/r/AdobePremiere/comments/1vqw9mb/media_intelligence_analysis/>
- <https://www.reddit.com/r/premiere/comments/1qpqbzz/media_intelligence_analysis_question/>
- <https://www.reddit.com/r/editing/comments/1td336d/has_ai_actually_replaced_video_editing_workflows/>
- <https://www.reddit.com/r/editors/comments/q419v1>

## Open Questions

- Which patched OpenCV 5 wheel or controlled source build passes OpenCut's Windows, macOS, and Linux media corpus while proving that every embedded FFmpeg copy clears CVE-2026-8461? Linux exposure is verified; the other wheel payloads need artifact inspection. **Needs live validation.**
- Does OpenTimelineIO 0.18.1 behave correctly on Python 3.13 and 3.14 despite upstream documenting support only through Python 3.12? Until the matrix runs, OpenCut should report that capability as unverified rather than broken (`pyproject.toml:22`, [OTIO 0.18.1](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases/tag/v0.18.1)). **Needs live validation.**
- Live Premiere acceptance for host mutations remains under F386 in `Roadmap_Blocked.md`; it does not block the headless and static work prioritized here. **Verified.**
