# Research — OpenCut
Date: 2026-07-14 — replaces all prior research.

## Executive Summary

**[Verified]** OpenCut is a local-first Adobe Premiere automation platform: a Python/Flask backend coordinates FFmpeg, optional local ML, jobs, journals, and interchange through CEP/UXP panels, REST, CLI, MCP, and plugins. Its strongest current shape is broad automation backed by explicit readiness, privacy, rollback, and release controls. The 2026-07-14 hardening sequence closed the previously verified FFmpeg floor, uninstall-data, UXP locking, container-secret, plugin-origin/isolation, migration-retry, rendered-panel, theme-token, caption-conformance, and panel-shell gaps (`75364eca` through `39e3baef`). The highest-value direction is now to simplify the oversized panel implementations and finish a small set of coherent editor workflows instead of adding more disconnected feature cards.

Top opportunities, in priority order:

1. **[Verified]** Decompose the CEP/UXP monoliths behind the rendered and source-contract gates that now ship.
2. **[Verified]** Complete local semantic media search with portable sidecars, relink/invalidation behavior, ranking fixtures, and panel integration; the CLIP index and route already exist.
3. **[Verified]** Extend the existing script-to-rough-cut planner with alternate takes, deterministic regeneration, preview, and journal-backed write-back.
4. **[Needs live validation]** Adopt the final MCP 2026-07-28 revision for stateless discovery, long-running render tasks, MRTR prompts, and cache metadata.
5. **[Verified]** Implement the stubbed Parakeet adapter and route supported languages between Parakeet v3 and Whisper turbo with explicit fallback.
6. **[Verified]** Localize backend/CLI strings, add panel locales beyond en/es, and verify an RTL layout.
7. **[Verified]** Revisit Depth Anything 3 only through an isolated runtime that preserves OpenCut's supported NumPy/OpenCV stack, allowlists permissive checkpoints, and normalizes depth direction.

## Product Map

- **[Verified] Core workflow — analyze:** ingest local media, transcribe, index, detect scenes/subjects/silence, evaluate quality, and generate reviewable metadata or plans.
- **[Verified] Core workflow — edit:** create cuts, captions, effects, markers, multicam/script assemblies, and reversible Premiere timeline mutations.
- **[Verified] Core workflow — deliver:** render or proxy through FFmpeg/AME, export captions and NLE interchange, package review artifacts, and optionally publish or dispatch to remote nodes.
- **[Verified] Core workflow — automate:** run the same capabilities through CEP/UXP, REST, CLI, MCP, scheduled jobs, webhooks, and plugins with job/journal diagnostics.
- **[Verified] Personas:** solo and professional editors, assistant editors, podcast/social teams, accessibility/localization operators, and technical users automating repeatable Premiere work.
- **[Verified] Platforms and distribution:** `pyproject.toml` requires Python 3.11+ on Windows/macOS/Linux; the CEP and UXP manifests target Premiere 13+ and 25.6+ respectively; `OpenCut.iss`, `installer/`, `Dockerfile`, and `packaging/linux/` provide Windows, source, Docker, Flatpak, AppImage, and release-automation surfaces. Credential/hardware-gated notarization, store publishing, and live Premiere validation remain correctly separated in `Roadmap_Blocked.md`.
- **[Verified] Integrations and data flows:** Premiere host APIs ↔ panel ↔ local Flask API ↔ FFmpeg/ffprobe, local/optional remote models, SQLite/JSON state under `~/.opencut`, OTIO/XML/AAF/EDL/caption files, remote render nodes, review/social APIs, MCP clients, and plugin routes.

## Competitive Landscape

- **[Verified] Adobe Premiere Pro / UXP:** local Media Intelligence and portable `.prmi` sidecars make private, relinkable semantic search table-stakes, while the 26.3 changelog shows that host action contracts can change at stable release. Learn portable index metadata, per-version capability detection, and official lint rules; avoid cloud-only indexing and undocumented host assumptions.
- **[Verified] DaVinci Resolve:** IntelliScript and alternate-take workflows keep AI output editable inside a professional timeline. Apply reviewable alternatives and reversible write-back to OpenCut's existing script-assembly row; avoid competing as a standalone NLE.
- **[Verified] Descript:** transcript editing, filler/repetition removal, review, and export limits are presented as coherent workflows rather than isolated models. Learn visible plans, correction loops, and compatibility preflights; avoid mandatory cloud processing, credits, and proprietary project lock-in.
- **[Verified] FireCut / TimeBolt:** commercial Premiere tools charge for reliable silence/filler removal, captions, chapters, shorts, and repeatable presets. Learn workflow packaging, saved presets, and honest progress; avoid duplicating narrow feature cards OpenCut already ships.
- **[Verified] LosslessCut / auto-editor:** fast deterministic edits, explicit keyframe/codec limitations, keyboard accessibility, versioned edit data, and OTIO export build trust. Learn reproducible plans and boundary diagnostics; avoid a second desktop editor.
- **[Verified] Kdenlive / Shotcut / OpenShot:** releases repeatedly prioritize proxy queues, cancellation, crash recovery, translation width, HiDPI, and relinking. Learn that queue state and rendered UI regressions are core product work; avoid mature timeline/editor duplication.
- **[Verified] OpenCut-app / PySceneDetect / OpenTimelineIO:** integer media time, explicit migrations, waveform virtualization, rational/PTS timestamps, VFR correctness, and adapter boundaries are strong adjacent patterns. Learn stable time representations, portable sidecars, and small interchange cores; avoid browser-storage fragility and coupling optional adapters into the base install.
- **[Verified] VS Code extension host / HandBrake queue:** isolate third-party failure domains, activate lazily, declare capabilities, disable unsupported options, and persist resumable queue state. Apply those boundaries to plugins and long media work; do not label a Python subprocess a security sandbox unless the OS enforces it.

## Security, Privacy, and Reliability

- **[Verified] The audited trust gaps are closed.** `opencut/core/ffmpeg_provenance.py` now enforces FFmpeg 8.1.2; both Windows uninstall paths preserve user data by default; UXP action creation runs inside `project.lockedAccess()`; container auth accepts permission-checked secret files; plugin packages are authenticated and third-party routes execute behind a worker boundary; JSON migrations retain retryable source state. Coverage lives in the associated tests and commits `f05e5835`–`7c03216b`.
- **[Verified] Existing network and artifact boundaries remain in force.** `opencut/core/url_safety.py` resolves and revalidates outbound targets, limits streamed downloads, and is used by URL ingestion/webhooks; update, C2PA, smart-render, remote-result, and proxy paths validate or transact their artifacts before replacement. Do not create duplicate roadmap items for these completed controls.
- **[Verified] Recovery is now part of the product contract.** Uninstall offers separate application/data removal, migrations are retry-safe, plugin activation can roll back, plugin workers time out independently, proxy batches resume, and host-side panel failures surface instead of being silently treated as success.
- **[Verified] Remaining roadmap work does not require a new security initiative.** Semantic search must stay local and make sidecar invalidation explicit; script assembly must preview and journal write-back; MCP tasks must preserve current auth/cancellation boundaries; ASR and localization must report optional-dependency readiness honestly. These guardrails are included in each owning roadmap item rather than duplicated as generic hardening work.
- **[Needs live validation]** Signed/notarized/store distribution and live Premiere host acceptance still require external credentials or host hardware and remain in `Roadmap_Blocked.md`; they are not actionable `ROADMAP.md` work.

## Architecture Assessment

- **[Verified] Decompose panels by responsibility, not by visual card.** CEP `client/main.js`/`style.css` and UXP `main.js`/`style.css` remain oversized shared boundaries. Extract state, API, i18n, job, timeline, token, layout, and bootstrap responsibilities while preserving public IDs, host-action names, route payloads, and the rendered baselines added in `6a44b951`.
- **[Verified] Treat existing implementations as the starting point.** `opencut/core/semantic_video_search.py` and `/search/semantic` already provide local CLIP search; implement portable sidecars, relink/invalidation, ranking fixtures, and panel discovery rather than a second index. `opencut/core/script_to_roughcut.py` already plans a rough cut; add alternate takes, deterministic regeneration, preview, and reversible journal write-back rather than another planner.
- **[Verified] Keep optional model adapters execution-truthful.** `opencut/registry.py::resolved_state()` distinguishes terminal stubs from dependency readiness. `opencut/core/asr_parakeet.py` and `asr_canary.py` still raise `NotImplementedError`; implement Parakeet before routing traffic to it, keep Whisper as the coverage fallback, and test both dependency absence and language routing.
- **[Needs live validation] Upgrade MCP at its protocol boundary.** After the 2026-07-28 final specification publishes, adapt `opencut/mcp_server.py` and `mcp_extended_tools.py` to stateless `server/discover`, Tasks, MRTR, and resource-cache metadata without leaking protocol details into render/transcode core modules; discovery and long-task conformance tests should own the compatibility contract.
- **[Verified] Keep incompatible model stacks out of process.** `depth-anything-3==0.1.1` requires NumPy 1.x and regular `opencv-python`, while OpenCut's supported video extras resolve NumPy 2 with headless OpenCV 4.13+. Any DA3 adapter needs a separate worker environment, an Apache-2.0 Small/Base allowlist, and explicit conversion from direct depth to the near/far convention CineFocus currently consumes from DA2 disparity.
- **[Verified] Localize through one backend boundary.** Add a backend gettext/Babel layer with English fallback, keep locale-key parity tests for CEP/UXP, and make directionality a shell/layout concern so RTL support does not fork individual components.
- **[Verified] Tests and docs follow owning boundaries.** Rendered panel regression, IMSC 1.3 conformance, plugin isolation, UXP locking, migration retry, uninstall preservation, and FFmpeg provenance now have direct gates. Each remaining item must extend those gates and update its setup/recovery copy in the same batch; no standalone test/docs roadmap duplicate is justified.
- **[Verified] Category disposition:** security, accessibility, observability, testing, docs, distribution/packaging, plugin isolation, offline/resilience, migration, and upgrade strategy are covered by completed controls or blocked release lanes; i18n and MCP remain actionable; mobile and hosted multi-user work remain rejected below.

## Rejected Ideas

- **[Verified] Standalone NLE or browser-editor parity — Rejected.** Kdenlive, Shotcut, OpenShot, and OpenCut-app already solve timeline editing; OpenCut's differentiated boundary is Premiere-native local automation.
- **[Verified] Mobile editing client — Rejected.** The primary workflows require Premiere host APIs, large local media, and keyboard/timeline interaction; a remote monitor would add a second security/distribution surface without a verified core need.
- **[Verified] Hosted simultaneous editing — Rejected.** Commercial demand exists, but mandatory server state conflicts with the single-user local-first design; OTIO, markers, review bundles, and sidecars are the fitting migration/collaboration boundary.
- **[Verified] Cloud semantic search — Rejected.** Adobe's local `.prmi` model and OpenCut's existing local CLIP index support private portable search without upload latency or new data governance.
- **[Verified] Bulk implementation of six HTTP stubs or 52 terminal model adapters — Rejected.** `opencut/_generated/route_manifest.json`, `opencut/_generated/feature_readiness.json`, `opencut/core/stub_scan.py`, and `opencut/tools/dump_feature_readiness.py` report these honestly; route count is not user value and optional heavyweight/licensing costs are unproven.
- **[Verified] Capability manifests described as a Python sandbox — Rejected.** VS Code's extension-host pattern supports failure isolation, but capability declarations alone cannot stop arbitrary Python filesystem/network access; use precise trust copy and OS enforcement before claiming sandboxing.
- **[Verified] Premiere 26.2 TrackItem-specific workaround — Under consideration, not roadmap work.** Adobe reports a save/reopen regression for video `TrackItem.createSetInPointAction`, but OpenCut's current helper mutates `Sequence` in/out points; the separate 26.3 `lockedAccess()` violation is verified and should land first.
- **[Verified] Generic dependency bump campaign — Rejected.** `pyproject.toml`, `requirements-lock.txt`, `docs/PYTHON_ADVISORIES.md`, and `docs/NODE_ADVISORIES.md` already track the audited Flask/Click/keyring/PyInstaller and ML floors; only the concrete FFmpeg floor produces a net-new security item.
- **[Verified] Academic generative editor/agent rewrite — Rejected.** Project Blink and newer long-video research are useful evidence for interactive review and retrieval, not justification to replace the deterministic jobs/journal/timeline architecture.
- **[Verified] In-process Depth Anything 3 adapter — Rejected.** Official DA3 0.1.1 and OpenCut's supported OpenCV/NumPy constraints are mutually unsatisfiable, both OpenCV wheels provide `cv2`, and DA3 Large is non-commercial; only an isolated Small/Base worker is under consideration.

## Sources

### Open source and adjacent projects

- https://github.com/mifi/lossless-cut/releases
- https://github.com/WyattBlue/auto-editor/releases
- https://github.com/mltframework/shotcut/releases/
- https://github.com/OpenShot/openshot-qt/releases
- https://github.com/Breakthrough/PySceneDetect/releases
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases
- https://github.com/OpenCut-app/OpenCut/releases
- https://github.com/leancoderkavy/premiere-pro-mcp
- https://handbrake.fr/docs/en/latest/technical/official-presets.html
- https://code.visualstudio.com/api/advanced-topics/extension-host

### Commercial products and platform APIs

- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/media-intelligence-and-search-panel.html
- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/manage-media-intelligence-metadata.html
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://community.adobe.com/bug-reports-728/trackitem-createsetinpointaction-does-not-persist-across-save-reopen-for-video-tracks-in-uxp-in-points-revert-to-the-source-clip-s-original-in-point-audio-trackitems-unaffected-regression-in-26-2-1624188
- https://github.com/AdobeDocs/uxp-premiere-pro-samples
- https://www.blackmagicdesign.com/products/davinciresolve/whatsnew
- https://www.descript.com/pricing
- https://firecut.ai/pricing/premiere-pro/
- https://www.timebolt.io/features

### Models and package metadata

- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results
- https://pypi.org/pypi/depth-anything-3/0.1.1/json
- https://pypi.org/pypi/opencv-python-headless/4.13.0.92/json

### Standards, research, dependencies, and security

- https://www.w3.org/TR/WCAG22/
- https://www.w3.org/TR/ttml-imsc1.3/
- https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html
- https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate/
- https://research.adobe.com/video/project-blink/
- https://nvd.nist.gov/vuln/detail/CVE-2026-8461
- https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6

## Open Questions

- **[Needs live validation]** Does the final MCP 2026-07-28 specification preserve the release candidate's stateless `server/discover`, Tasks, MRTR, and cache-metadata contracts? Validate the published specification before changing OpenCut's protocol dependency or conformance tests. Packaged Docker and live Premiere acceptance are implementation verification lanes; credential/hardware-only release checks remain in `Roadmap_Blocked.md`.
