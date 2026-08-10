# Research — OpenCut

Date: 2026-08-08 — replaces all prior research.

## Executive Summary

OpenCut is a local-first Premiere Pro automation bridge rather than a general-purpose nonlinear editor. The current v1.47.0 tree combines a Flask loopback service, CLI, MCP server, CEP and UXP panels, durable background jobs, SQLite/FTS5 media indexes, FFmpeg processing, optional local AI adapters, interchange exports, review artifacts, and Windows/Linux distribution paths. The generated route inventory contains 1,583 route rows, of which 1,558 are counted as shipped; 73 feature-readiness records report 52 available and 21 dependency-gated states. The panel exposes eight tabs and more than 50 sub-tabs. This breadth is valuable, but it makes contract consistency and host-version drift more important than adding another isolated model adapter.

The strongest net-new opportunities are:

1. **P1 — Make library auto-indexing real and durable.** `/search/auto-index` reports files as queued but currently does not create a job or dispatch indexing work. This is a concrete contract failure in a core workflow and has a bounded fix.
2. **P1 — Map the existing durable job engine to MCP Tasks.** OpenCut already has persistence, progress, cancellation, recovery metadata, and status routes. The modern MCP protocol now has a task lifecycle, but OpenCut advertises no task extension and exposes ad hoc `job_id` polling instead.
3. **P1 — Add a versioned Adobe UXP API compatibility matrix.** The current updater tracks package versions and dist-tags, not API signatures or behavioral changes. Premiere 26.3 changed synchronous behavior and added APIs while UXP remains version-coupled to the host.
4. **P2 — Build a federated backend multimodal index.** Text FTS5, legacy JSON search, and per-request visual sidecars are separate surfaces. A configured-root index would make cross-project visual, transcript, OCR, and audio retrieval incremental and explainable without requiring live Premiere UI work.
5. **P2 — Replace the audio-reactive terminal stub with an existing-stack renderer.** Beat analysis, rhythm effects, and deterministic visualizers already ship; the stub should compose those primitives instead of making BeatNet a mandatory dependency.
6. **P2 — Add a small MCP Apps review/progress surface.** The server currently exposes empty resources and text-wrapped JSON. Reviewables, job progress, thumbnails, and artifacts are already suitable for a versioned, sandboxed rich result with a text fallback.

These recommendations are intentionally narrow. They do not repeat the blocked FFmpeg, dependency-stack, live Premiere, release-publication, localization, package-index, or maintainer-decision items in `Roadmap_Blocked.md`. They also do not recommend implementing all 50 terminal AI modules: most are model, hardware, licensing, or upstream-risk placeholders, while audio-reactive processing has a credible path through code already in the repository.

Confidence labels used below: **Verified** means confirmed in the local tree or a primary source; **Likely** means a design opportunity supported by multiple signals but still requiring implementation validation; **Needs live validation** means it cannot be closed by headless repository work.

## Product Map

### Core workflows

- **Media-to-cut:** a user selects a Premiere project or local media, runs silence/beat/scene/transcription/OCR/audio analysis, applies an edit or effect, reviews the result, and writes a sequence, captions, markers, or media changes back through CEP or UXP.
- **Search-to-edit:** files are indexed for transcript, OCR, audio tags, and increasingly visual descriptors; a user or automation client searches the library, inspects timestamps/thumbnails, and turns selected evidence into a reviewable proposal or timeline operation.
- **Caption and delivery:** speech and diarization outputs are corrected, styled, synchronized, rendered, checked against delivery standards, and exported to subtitle, caption, or interchange formats.
- **Review and approval:** long-running processing creates durable jobs and review artifacts with progress, cancellation, redacted payloads, thumbnails/HLS where applicable, notifications, and explicit approval before destructive or host-facing actions.
- **Automation:** the same local service is reached through CLI, REST, MCP, panel calls, plugins, and scripts. The loopback boundary keeps processing local while allowing Premiere to act as the editing host.

### Personas and jobs to be done

- A solo editor or small post-production team wants fast, local rough cuts and search without uploading footage.
- A podcast, education, or social-video producer needs transcript-driven edits, caption corrections, silence removal, beat-aware assembly, and reusable review outputs.
- A caption, accessibility, or delivery operator needs deterministic exports, standards checks, timing repair, and audit-friendly artifacts.
- A technical user needs scriptable, inspectable, cancellable operations that can be orchestrated from the CLI, REST, or MCP without arbitrary code execution.

### Platform and distribution surface

- Python 3.11–3.14 is the core runtime. Windows has WPF installer, PyInstaller, and Inno Setup lanes; Linux has self-contained packaging and Flatpak-oriented metadata; macOS remains source/package constrained by notarization and host acceptance work.
- Premiere integration is split between legacy CEP/ExtendScript and the newer UXP panel. UXP is the strategic path, with a minimum host around 25.6 and a pinned 26.3 Adobe package in the panel build.
- Optional model and media stacks are lazy-loaded. CPU-only Docker is supported; GPU, diarization, separators, caption standards, and depth features remain explicitly conditional.
- The local service owns the data plane: SQLite/job journals, file indexes, review bundles, cache/sidecar directories, FFmpeg artifacts, and configuration. Premiere is the host-side control plane for operations that must mutate a project.

### Data and integration flow

Panels, CLI clients, and MCP/REST callers submit validated commands to Flask routes. Routes call bounded workers for long operations, which persist job state and write artifacts. Search data currently splits across FTS5 transcript/OCR/audio records, an older JSON/BM25 index, and visual model sidecars. CEP and UXP adapters translate approved operations into Premiere calls. FFmpeg and optional engines produce media; review and delivery layers consume the resulting files and metadata.

### Scope boundaries

OpenCut is strongest as a local automation and review layer around Premiere. It is not presently a mobile editor, a cloud collaboration service, a multi-user asset-management system, or a replacement for Premiere’s full timeline UI. Those directions were considered and rejected as poor fits for the current architecture and privacy promise.

## Competitive Landscape

### Adobe Premiere Pro, Media Intelligence, and Media Encoder

Adobe now pairs Premiere’s UXP APIs with Media Intelligence and Media Encoder sidecar analysis (`.prmi`) for visual, audio, and transcript search. Premiere’s host APIs set the compatibility baseline, while UXP 26.2 hybrid plugins and the 26.3 API additions expand what a native extension can eventually do. OpenCut should learn from sidecar-backed indexing and explicit host-version contracts, but should not chase first-party parity or assume that a new UXP API is stable without a fallback.

### FireCut, AutoCut, and AutoPod

These commercial Premiere products package recognizable workflows: silence and filler removal, captions, chapters, B-roll, auto-zoom, multicam, shorts, and beat-aware edits. Their strength is task-oriented UX and clear packaging of a workflow rather than exposing a large route catalog. OpenCut should make its high-value operations similarly coherent through durable jobs, review artifacts, and predictable presets. It should preserve its local-first and inspectable behavior instead of copying subscription, credit, or opaque-cloud assumptions.

### Descript, Riverside, CapCut, and VEED

Transcript-first editing, Magic Clips, Auto Cut, Smart Search, and edit-by-script workflows show demand for context-aware triage and quick repurposing. The useful lesson is to expose evidence and allow the user to adjust what is included, excluded, or promoted. OpenCut already has the local transcript and review primitives for this direction. It should avoid presenting a single unexplained virality score or making cloud inference a prerequisite for a local workflow.

### auto-editor, Kdenlive, Shotcut, OpenShot, and MLT

The open-source desktop ecosystem emphasizes deterministic media processing, proxy workflows, timeline stability, replayable tests, and packaging. Kdenlive’s 2026 roadmap and MLT’s 7.40 security/reliability work reinforce that regressions and safe command handling matter as much as feature count. OpenCut should borrow artifact-level regression tests and explicit interchange/loss reporting. It should not reintroduce legacy or unsafe MLT/FFmpeg assumptions that the current repository has already guarded against.

### Subtitle Edit

Subtitle Edit demonstrates the value of a local, cross-platform caption workflow with many format adapters, waveform-oriented editing, OCR, automatic backups, and optional online services that are clearly optional. OpenCut should continue to favor local fallbacks, recoverable edits, and precise timing contracts. It should not become a second standalone subtitle editor when its differentiator is the Premiere bridge and automation surface.

### OpenTimelineIO and the MLT/interchange ecosystem

OTIO provides a useful neutral interchange model, but adapters, schema versions, nested sequences, and transitions remain compatibility boundaries. OpenCut already pins the split AAF adapter, runs preflight, preserves supported transitions, and marks lossiness. The opportunity is continued compatibility testing, not another roadmap item for a pin or transition implementation that has already shipped.

### Premiere MCP projects

Open-source Premiere MCP projects range from a small UXP file bridge to hundreds or thousands of ExtendScript tools and arbitrary execute-script entry points. They demonstrate demand for agent control, but also show fragmented host support, macOS/Windows gaps, and unsafe code-generation patterns. OpenCut should differentiate with an allowlisted tool catalog, approvals, durable jobs, and the standardized MCP Tasks lifecycle. It should not add an arbitrary script execution tool merely to match tool counts.

### Adjacent media and model projects

Editly, OpenReelio, Remotion, MoviePy, WhisperX, pyannote, audio-separator, faster-whisper, and VAD projects provide useful patterns for inspectable programmatic timelines, diarization, local transcription, separation, and reproducible render pipelines. They are dependency and license inputs, not automatic feature requirements. OpenCut’s optional-adapter policy is the safer way to absorb those capabilities.

### Competitive takeaways

The durable differentiators are local privacy, Premiere-aware mutation, inspectable artifacts, review-before-apply behavior, and a broad automation contract. The common failure modes are opaque scoring, cloud-only processing, brittle host APIs, and media operations that claim to be queued or recoverable without actually persisting work. The recommended roadmap addresses the latter failure mode first.

## Security, Privacy, and Reliability

### New findings

- **False asynchronous contract — Verified, high impact.** `opencut/routes/search.py:auto_index_project` calculates changed files using `footage_index_db.needs_reindex`, then returns a “Queued” response without `@async_job`, a job identifier, worker dispatch, or index mutation. A caller can report success while the library remains unchanged. This is the clearest root-cause reliability defect found in the current pass.
- **Protocol lifecycle mismatch — Verified, medium/high impact.** `opencut/mcp_server.py` implements the 2026-07-28-style request surface and returns modern completion metadata, but advertises an empty extension map and has no `tasks/get`, `tasks/update`, or `tasks/cancel`. Long operations depend on an OpenCut-specific `job_id` tool and polling convention. This is not a protocol vulnerability, but it makes cancellation and client interoperability less reliable than the underlying job engine warrants.
- **Search state fragmentation — Verified, medium impact.** Text search, legacy JSON search, multimodal indexing, and visual sidecars have different schemas and lifecycle rules. Visual search accepts explicit clip paths and optional per-project sidecars; there is no configured library registry that can report stale, moved, deleted, or model-schema-incompatible entries. A future federated index must keep path exposure, cache invalidation, and missing-model states explicit.
- **Host API drift — Verified, medium impact.** `adobe_premierepro_versions.py` checks npm tags and a committed package snapshot, but package-version drift does not capture API signature or sync/async behavior changes. Adobe’s 26.3 changelog changed `Sequence.setSelection` behavior and added track/subclip/transcript APIs. Feature detection in the panel is good defensive practice, but it is not a versioned compatibility contract.
- **Terminal audio route — Verified, medium impact.** `/video/audio-reactive` is synchronous and dependency-gated, while `audio_reactive_fx.py` still raises `NotImplementedError`. The repository already contains deterministic waveform visualization and rhythm-effect primitives, so the current route can mislead clients about a capability that is present in the catalog but not executable.

### Existing positive controls

OpenCut already has CSRF checks on state-changing routes, trusted-host and loopback controls, SSRF/path validation, remote-auth handling, plugin trust/isolation rules, redacted job payloads, bounded workers, durable job journals, cancellation/resource reporting, review approvals, ZIP-slip defenses, backup hashes, and C2PA-related provenance support. The rendered panel suite exercises WCAG 2.2 AA roles, names, focus, keyboard navigation, contrast, themes, reduced motion, errors, empty states, and destructive confirmations without suppressions. These are foundations to preserve, not reasons to add a broad security or accessibility rewrite.

### Known external blockers excluded from this roadmap

The FFmpeg 8.1.2 CVE floor, OpenCV 5 versus Transformers/Hugging Face dependency conflict, live macOS notarization, downloadable release publication, Premiere 26.x live smoke testing, broader localization, PyPI/Homebrew/winget publication, UXP theme/runtime verification, and queue-allowlist reconciliation are already tracked in the local blocked ledger. They require credentials, a live host, hardware, a maintained dependency decision, or human review. Re-adding them would make the active roadmap less actionable.

### Reliability acceptance principles

Every new long-running path should persist its job before returning, expose progress and cooperative cancellation, retain per-input errors, survive a process restart, and make no-op/up-to-date behavior observable. Every new index should carry a content signature and schema/model version, support explicit invalidation, and avoid sending local media off-machine. Every Premiere-facing API should declare a minimum host and a fallback or a clearly blocked capability.

## Architecture Assessment

### Strengths

- The job system is already a suitable foundation for reliable orchestration: it persists an initial record synchronously, runs work through bounded workers, records terminal state, supports cancellation and progress, and exposes both live and persisted status.
- The repository has strong boundary discipline around optional imports, subprocess execution, safe user-data wrappers, route validation, CSRF, and artifact provenance.
- CEP and UXP are separated enough to permit an incremental migration. UXP code already feature-detects several transcript and theme APIs, and generated dashboards make parity gaps visible.
- Search and media analysis have reusable pieces: FTS5 indexing, multimodal workers, content-signature sidecars, deterministic audio visualization, rhythm analysis, and reviewable outputs.
- Verification is unusually broad for a local media tool: hundreds of Python test modules, route/contract tests, generated-manifest checks, panel unit tests, and rendered accessibility scans.

### Main architectural seams

1. **Index lifecycle seam.** `auto_index_project`, transcript indexing, multimodal indexing, FTS5, legacy JSON search, and visual sidecars do not share one lifecycle or job contract. First repair the false queue response. Then introduce a registry/manifest layer for configured roots and modality/schema state; do not begin by rewriting all query code.
2. **Protocol seam.** The job store already contains most MCP task state. A protocol adapter should translate only at the MCP boundary: negotiate the task extension per request, create the durable task before returning, map OpenCut states to MCP states, and retain legacy `job_id` behavior for clients without the extension.
3. **Host compatibility seam.** The Adobe package snapshot is a version feed, not a behavior matrix. A generated matrix should record host minimums, API availability, sync/async expectations, and fallback paths alongside the pinned typings. It can be tested headlessly and should report drift before a release; it must not pretend to replace a live Premiere acceptance run.
4. **Audio composition seam.** `audio_visualizer.py` and `rhythm_effects.py` already provide deterministic analysis and effects. `audio_reactive_fx.py` should become a thin orchestration/rendering layer with an allowlisted filter/keyframe model, rather than introducing BeatNet as a required new runtime.
5. **Rich-result seam.** MCP Apps can be added after task and artifact contracts are stable. A small `ui://` resource for job progress/review artifacts is safer than exposing the entire panel or arbitrary filesystem/network access. Clients without Apps support must receive the existing text result.

### Implementation and test gaps

- There is no route test proving that a non-empty `/search/auto-index` request eventually changes database rows or returns a real job lifecycle.
- There are no MCP fixtures for per-request Tasks negotiation, durable creation-before-response, task state mapping, or legacy-client fallback.
- The Adobe updater lacks a checked-in API behavior/signature matrix and tests for 25.6/26.2/26.3 fallback decisions. Live host checks remain a separate blocked lane.
- The audio-reactive route lacks a successful media fixture, cancellation/progress contract, and proof that missing optional analysis is represented as a capability state rather than a runtime `NotImplementedError`.
- A federated index would need deterministic small media fixtures, path move/delete cases, cache schema migrations, and bounded-resource tests before it should be exposed in the panel.

### Operating constraints

The single-user loopback model, local-only option, optional dependency policy, and current distribution gates are coherent. Multi-user/cloud/mobile expansion would require a different threat model, storage model, and product promise. Accessibility and i18n should remain part of each surface’s acceptance criteria; a broad localization item is blocked by human translation/rendering review, while the existing English/Spanish and RTL scaffolding can support incremental work later.

## Rejected Ideas

- **Repeat the FFmpeg CVE floor, dependency conflict, release publication, live Premiere smoke, localization, PyPI/Homebrew/winget, or installer/UAC work.** These are already in `Roadmap_Blocked.md` and require external state or maintainer decisions.
- **Implement all terminal AI modules.** The 50-module stub inventory mixes research models, unavailable upstreams, hardware-heavy runtimes, and licensing-sensitive adapters. A generic sweep would increase surface area without a supportable acceptance contract. Audio-reactive rendering is the exception because existing deterministic primitives make it bounded.
- **Re-add OTIO adapter pinning, transition preservation, MLT export, or generic interchange work.** The adapter split, version preflight, transition handling, and MLT export are already shipped and tested. Future upstream compatibility is maintenance, not a new feature proposal.
- **Build the UXP hybrid native addon now.** Adobe’s hybrid path is real, but OpenCut’s current package is a static validator with no native binary or live UDT/Premiere acceptance. The caption/QE candidates remain host-gated; the compatibility matrix is the headless precursor.
- **Create a generic accessibility overhaul.** Rendered WCAG 2.2 AA scans, keyboard/focus checks, themes, reduced motion, and panel state coverage already exist. The remaining qualitative UXP language review is correctly blocked for human evaluation.
- **Restore generic GitHub Actions or add a broad CI migration.** The prior research already rejected this direction, and local verification/release gates are not evidence that another CI architecture would solve a current product defect.
- **Add a generic virality or highlight score.** OpenCut already has highlight and local scoring paths, while current long-form editing research still shows evaluation gaps. A benchmark/evaluation program needs a defined dataset license, target user, and quality threshold before it becomes an actionable feature.
- **Pursue mobile, standalone cloud collaboration, or multiplayer editing.** These directions conflict with the current local-first Premiere bridge and would require a new security, storage, synchronization, and distribution architecture.
- **Expose arbitrary Premiere script execution through MCP.** Existing third-party projects show demand, but arbitrary code generation weakens the allowlist, review, and trust model. Standardized task lifecycle and explicitly scoped tools are a better fit.

## Sources

### Host and platform

- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://developer.adobe.com/premiere-pro/uxp/resources/fundamentals/apis/
- https://developer.adobe.com/premiere-pro/uxp/resources/recipes/external-process/
- https://blog.developer.adobe.com/en/publish/2026/04/uxp-hybrid-plugins-now-available-for-premiere
- https://helpx.adobe.com/premiere/desktop/whats-new/whats-new.html?mv=product
- https://helpx.adobe.com/media-encoder/using/whats-new/2026.html

### Commercial products

- https://firecut.ai/pricing/premiere-pro/
- https://www.autocut.com/en/download/
- https://www.autopod.fm/pricing
- https://help.descript.com/hc/en-us/articles/27252457732237-AI-Tools-Overview
- https://support.riverside.com/hc/en-us/articles/12124048765981-About-Magic-Clips
- https://www.capcut.com/help/auto-cut-in-capcut

### Open-source competitors and adjacent projects

- https://github.com/wyattblue/auto-editor/releases
- https://github.com/mifi/lossless-cut/releases
- https://github.com/SubtitleEdit/subtitleedit
- https://subtitleedit.github.io/subtitleedit/faq.html
- https://kdenlive.org/news/2026/state-2026/
- https://kdenlive.org/roadmap/
- https://shotcut.com/roadmap/
- https://github.com/OpenShot/openshot-qt/releases
- https://github.com/mltframework/mlt/releases
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://editly.in/changelog
- https://openreelio.com/
- https://github.com/nepfaff/premiere-pro-mcp
- https://github.com/ayushozha/AdobePremiereProMCP

### Awesome lists

- https://github.com/wentianli/awesome-video-editing
- https://github.com/sindresorhus/awesome-whisper
- https://project-awesome.org/r/awesome-ffmpeg
- https://github.com/ad-si/awesome-video-production

### Community signal

- https://community.adobe.com/feature-requests-730/uxp-caption-track-editing-apis-1557329
- https://community.adobe.com/bug-reports-728/trackitem-createsetinpointaction-does-not-persist-across-save-reopen-for-video-tracks-in-uxp-in-points-revert-to-the-source-clip-s-original-in-point-1624188
- https://news.ycombinator.com/item?id=46270519
- https://news.ycombinator.com/item?id=46434092
- https://lobste.rs/s/nbh5uw/what_are_you_working_on_this_week
- https://stackoverflow.com/questions/tagged/adobe-premiere?tab=Active
- https://www.reddit.com/r/premiere/comments/1tywbj8/building_a_premiere_pro_plugin_to_auto_fix_captions_using_ai/
- https://www.reddit.com/r/editors/comments/1vd48jr/aug_2026_open_source_video_tools_devs/

### MCP and interchange standards

- https://blog.modelcontextprotocol.io/posts/2026-07-28/
- https://modelcontextprotocol.io/seps/2663-tasks-extension
- https://modelcontextprotocol.io/extensions/tasks/overview
- https://modelcontextprotocol.io/extensions/apps/overview
- https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/
- https://modelcontextprotocol.io/extensions/client-matrix
- https://modelcontextprotocol.io/seps/2575-stateless-mcp
- https://www.w3.org/TR/ttml-imsc1.2/
- https://tech.ebu.ch/timedtext
- https://www.smpte.org/blog/new-and-revised-smpte-standards

### Dependency and security advisories

- https://pypi.org/project/opencv-python/5.0.0.93/
- https://github.com/opencv/opencv/wiki/OpenCV-5
- https://pypi.org/project/transformers/
- https://huggingface.co/blog/huggingface-hub-v1
- https://github.com/modelcontextprotocol/python-sdk/security
- https://nvd.nist.gov/vuln/detail/CVE-2026-64832

### Academic and engineering research

- https://arxiv.org/abs/2606.06926
- https://openaccess.thecvf.com/content/CVPR2026/html/Qiu_TVHighlights_LLM-Guided_Human-Free_Collaborative_Training_for_Video_Highlight_Detection_in_CVPR_2026_paper.html
- https://arxiv.org/abs/2605.03276
- https://arxiv.org/abs/2606.08415
- https://arxiv.org/abs/2608.05049
- https://arxiv.org/abs/2603.25750

## Open Questions

- Which three MCP tools should receive the first rich Apps resources after the task adapter exists: job progress, reviewable approval, and search result inspection are the strongest candidates, but client support and display value should be measured before broadening the surface.
- For the federated index, should configured roots be user-selected project directories only, or may a future project registry discover them? The first implementation should choose explicit roots to preserve privacy and avoid surprising filesystem scans.
- Which visual/audio model schemas are stable enough to become a persisted cross-project contract? The index must treat model and schema versions as data and should not silently mix embeddings or labels from incompatible engines.
- Which Premiere 26.x host behaviors can be proven only with a licensed live host? The UXP compatibility matrix can cover package/API evidence, but caption editing, track mutations, `document.theme`, and hybrid/native candidates still require the blocked host lane.
- What retention policy should apply to search thumbnails, transcript/OCR text, and job artifacts when a source file is deleted? This is a privacy/product choice for the federated index and should be settled before persistent cross-project caching is enabled by default.
- How should downstream MCP clients discover an OpenCut task’s final media artifact and review approval state without relying on a second proprietary polling convention? The task adapter should answer this from existing job/artifact metadata, but the exact resource/result shape should be fixed with compatibility fixtures.
