# Research — OpenCut
Date: 2026-07-21 — replaces all prior research.

## Executive Summary

**[Verified]** OpenCut is a local-first Adobe Premiere automation platform whose Flask/FFmpeg backend is exposed through CEP and UXP panels, REST, CLI, MCP, plugins, jobs, and interchange files. Its strongest shape is breadth with unusually strong review, rollback, local-processing, and release gates; its highest-value direction is now to make those trust guarantees true on every path before adding more surface area. The immediate priorities are the vulnerable Pillow floor, secret-bearing issue reports, broken CEP destructive confirmations, release artifacts whose license/SBOM gates do not inspect resolved composition, and a process-memory queue that loses pending work. After those, the best investments are accurate runtime/dependency contracts, ASR provenance and boundary validation, native UXP theme response, one accessible onboarding state machine, real NeMo adapters, targetable OTIO compatibility, and checkpoint-style recovery.

Top opportunities, in priority order:

1. **[Verified]** Raise Pillow to 12.3.0 on every static and dynamic install path; 13 vendor advisories published in 2026-07 affect the admitted `<=12.2.0` range.
2. **[Verified]** Redact credentials and tokens before logs are embedded in public GitHub issue URLs.
3. **[Verified]** Repair CEP preset/model/queue/workflow deletion so it performs the backend's preview → confirmation-token → commit protocol.
4. **[Verified]** Generate release locks, license evidence, third-party notices, and the SBOM from resolved artifacts, including bundled GPL FFmpeg.
5. **[Verified]** Persist the legacy `/queue` and recover interrupted work after server restarts.
6. **[Verified]** Make ASR engine/model/alignment provenance and boundary accuracy explicit, then implement the Parakeet/Canary adapters against that contract.
7. **[Verified]** Synchronize UXP tokens with Premiere's real host theme and consolidate duplicate CEP first-run dialogs.
8. **[Verified]** Add an OTIO compatibility target and promote the operation journal into previewable crash-recovery checkpoints.
9. **[Verified]** Make documentation claims and runtime support executable facts rather than permissive prose.

## Product Map

- **[Verified] Core workflows:** ingest/analyze local media; transcribe, caption, search, and assemble; apply reviewable Premiere timeline mutations; render/export NLE, caption, social, and review artifacts; automate the same work through REST, CLI, MCP, jobs, and plugins.
- **[Verified] Personas:** solo and professional editors, assistant editors, podcast/social teams, accessibility/localization operators, and technical users building repeatable Premiere workflows.
- **[Verified] Platforms and distribution:** Python 3.11+ source installs on Windows/macOS/Linux; a bundled Windows installer/WPF shell; Docker and Linux packaging; CEP for Premiere 13+ and UXP for Premiere 25.6+.
- **[Verified] Integrations and data flows:** Premiere host APIs ↔ CEP/UXP ↔ localhost Flask/Waitress ↔ FFmpeg/ffprobe and optional local/cloud engines; persistent jobs/journal/settings under `~/.opencut`; OTIO/XML/AAF/EDL/caption sidecars; review links, render nodes, social providers, and plugin workers.
- **[Verified] Product philosophy:** augment Premiere with local, inspectable, reversible automation rather than become a second NLE; optional egress must be explicit and core workflows must remain usable offline.

## Competitive Landscape

- **[Verified] Adobe Premiere / UXP:** Media Intelligence persists local `.prmi` analysis and AI Assistant shows tool steps, asks permission, and keeps actions undoable. Learn host-native themes, visible plans, and consent before mutation; avoid assuming undocumented host behavior or cloud-only workflows.
- **[Verified] LosslessCut:** deterministic edit plans, explicit export review, undo/redo, durable project data, and editable NLE interchange keep automation inspectable. Learn plan-first commits and versioned interchange; avoid building a duplicate timeline editor.
- **[Verified] Descript:** non-destructive text edits, boundary restoration, and version preview create a consistent recovery model. Learn visible recovery; avoid cloud-only history and account coupling.
- **[Verified] TimeBolt:** provides rejected-material preview, splits-only modes, and exact padding. Learn safe previews and curated workflows; avoid transformations that cannot be previewed or restored.
- **[Verified] Kdenlive:** automatic save checkpoints and a time-tiered restore browser make recovery discoverable. Learn named checkpoints and crash recovery; avoid opaque backup files with no preview.
- **[Verified] StoryToolkitAI / WhisperX:** local ingest-to-story workflows and explicit separation of ASR text accuracy from alignment accuracy fit OpenCut's positioning. Learn engine provenance, timing diagnostics, and honest standalone/source parity; avoid silent model fallback.
- **[Verified] Frame.io:** immutable asset versions, synchronized comparison, and comments scoped to a selected version make review migrations safe. Learn stable review identity and version-bound feedback; avoid mandatory hosted collaboration.

## Security, Privacy, and Reliability

- **[Verified] Current dependency exposure:** `pyproject.toml`, `requirements.txt`, `install.py`, and `opencut/core/styled_captions.py` can resolve Pillow `<=12.2.0`; Pillow's 2026-07 advisories include memory-safety, decompression-denial, information-disclosure, and Windows command-injection issues fixed in 12.3.0. The installed environment's `pip-audit` returned no known vulnerabilities on 2026-07-21, but that does not cover all advertised or dynamic install paths.
- **[Verified] Secret egress:** `opencut/core/issue_report.py` scrubs home paths only, then embeds crash/log tails in a public GitHub issue URL. Authorization headers, API keys, OAuth tokens, URL credentials, and known secret environment names can survive; OWASP explicitly says access tokens and primary secrets must be removed, masked, sanitized, hashed, or encrypted before logging/export.
- **[Verified] Destructive control drift:** CEP calls `/presets/delete`, `/models/delete`, `/queue/clear`, and `/workflow/delete` without dry-run or `confirm_token` (`extension/com.opencut.panel/client/main.js`), while their routes require that protocol and return 409 otherwise. Several callbacks suppress the resulting error. WCAG 2.2 SC 3.3.4 requires reversibility, checking, or review/confirmation for user-data deletion.
- **[Verified] Release composition blind spot:** `opencut/tools/license_gate.py::_read_requirements()` is unused by `lint()` and reports zero requirement findings; `scripts/sbom.py` is declaration-only. The Windows bundle and Docker image include GPL-enabled FFmpeg, but installer/release outputs do not establish exact corresponding source, build configuration, or complete third-party notices. The safe claim is missing redistribution evidence—not that a separate FFmpeg process relicenses OpenCut.
- **[Verified] Non-durable pending work:** `opencut/routes/jobs_routes.py:151` stores `/queue` entries in a module-level list; queued work disappears on restart and in-flight state has no interrupted recovery classification.
- **[Verified] Recovery is fragmented:** tombstones and the operation journal exist, but mutations are not grouped into durable pre-commit checkpoints that can detect and recover an interrupted multi-step host write.

## Architecture Assessment

- **[Verified] Dependency/runtime contract:** the advertised `torch-stack` combines WhisperX's Torch 2.8 family with OpenCut's Torch 2.10/torchvision 0.25 floor and is resolver-incompatible; `requires-python >=3.11` admits Python 3.14 while installer/classifier/support paths disagree. Each extra needs resolver-smoke coverage and an explicit OS/Python matrix.
- **[Verified] ASR boundary:** `opencut/core/asr_parakeet.py:47` and `opencut/core/asr_canary.py:47` terminate with `NotImplementedError`, while the router exposes them as preferred engines when ready. `TranscriptionResult` preserves confidence/cache identity but not engine/model/alignment version. Official NeMo APIs support pretrained restore/transcribe and structured timestamps; Parakeet/Canary v2 Windows viability still needs live validation.
- **[Verified] UXP theming:** `extension/com.opencut.uxp/command-center-tokens.css` defines `html.theme-light`, but production JavaScript never calls Premiere UXP's `document.theme.getCurrent()` or `theme.onUpdated`; the rendered “light” test emulates browser media and remains visually dark.
- **[Verified] First-run accessibility:** CEP has a static wizard using `activateOverlay()` and a separately generated server-backed onboarding dialog. The generated dialog lacks an accessible name, focus trap, Escape handling, and focus restoration; rendered tests pre-dismiss onboarding and never exercise the real first-run path.
- **[Verified] Interchange compatibility:** `opencut/export/otio_export.py` writes current schemas without a requested downgrade target or adapter preflight, although OTIO supports target schema versions and downgrade maps.
- **[Verified] Maintainability:** `opencut/routes/system.py`, `captions.py`, and `wave_l_routes.py` remain multi-thousand-line blueprints despite `CONTRIBUTING.md` advising decomposition above 600 lines. Flask blueprints provide the existing semantic split mechanism; route-manifest and response-contract tests can hold URLs stable.
- **[Verified] Documentation truth:** README claims “No `eval`/`exec`/`pickle`” despite guarded uses in `expression_engine.py`, `scripting_console.py`, and `model_quantization.py`; it also makes absolute locality and PyPI-install claims contradicted by optional cloud providers and an unpublished `opencut-ppro` project. Several README links point to ignored, untracked local docs, and `CLAUDE.md` retains v1.33.1/1,152-route facts. The existing semantic fact gate should cover trust, install, platform, and tracked-link claims.
- **[Verified] Test strategy:** browser coverage is broad, but confirmation and state semantics are partly synthetic DOM fixtures. New work should drive actual production controls, real callback/error paths, restart recovery, theme events, and first-run focus behavior.

## Rejected Ideas

- **[Rejected] Standalone NLE, mobile editor, or real-time multi-user editing:** conflicts with the Premiere-augmentation/local-first product boundary; LosslessCut and Frame.io patterns can be adopted without replacing the host editor.
- **[Rejected] Another semantic-search backend:** Adobe Media Intelligence validates the direction, but OpenCut's backend/sidecar already shipped and remaining panel work is already in `Roadmap_Blocked.md`.
- **[Rejected] Mandatory cloud collaboration/accounts:** Frame.io mechanics are useful, but hosted identity and storage are unnecessary for versioned local review bundles.
- **[Rejected] Additional locales or RTL expansion in this pass:** the work requires human translation/review and already lives in `Roadmap_Blocked.md`; onboarding and theme fixes must preserve the current i18n contract without duplicating that item.
- **[Rejected] Code signing or notarization as an active task:** blocked by external credentials and intentionally excluded from this pass; unsigned recovery paths remain valid.
- **[Rejected] Restore GitHub Actions:** commit history documents an intentional local-build policy; improve local release gates instead.
- **[Rejected] Wholesale Spectrum Web Components migration:** Adobe still supports vanilla HTML and documents current Premiere SWC theming/RTL constraints; use targeted components only for proven gaps.
- **[Rejected] Plugin-host or generic observability rewrite:** `opencut/core/plugin_worker.py`, `opencut/core/job_diagnostics.py`, and `opencut/core/issue_report.py` already provide the relevant boundaries; fix their concrete recovery and secret-egress gaps instead of replacing those systems.
- **[Rejected] Generic dependency bumps, OpenCV 5, or Ruff churn:** Flask/keyring/PyInstaller are already at safe/current floors, OpenCV 5 is a breaking major without a needed capability, and developer-tool drift is not a product feature.
- **[Rejected] Remote C2PA soft-binding repository:** C2PA guidance requires an external privacy-sensitive service and human verification; OpenCut intentionally has no soft binding today.
- **[Rejected] AutoCut-style XP, ranks, or activity profiles:** engagement gamification conflicts with professional, local-first workflows.
- **[Rejected] More readiness-only AI stubs:** awesome-list breadth is not evidence of value; finish and validate the existing adapters before expanding providers.

## Sources

### Adobe and standards

- https://developer.adobe.com/premiere-pro/uxp/resources/recipes/css-styling/
- https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/review-guidelines/
- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/media-intelligence-and-search-panel.html
- https://helpx.adobe.com/premiere/desktop/premiere-ai-assistant/overview.html
- https://www.w3.org/TR/WCAG22/
- https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/

### Competitors

- https://github.com/mifi/lossless-cut/blob/master/docs/index.md
- https://docs.kdenlive.org/en/project_and_asset_management/file_management/backup.html
- https://github.com/octimot/StoryToolkitAI
- https://github.com/m-bain/whisperX
- https://help.descript.com/hc/en-us/articles/10164106619405-Version-history
- https://www.timebolt.io/features
- https://help.frame.io/en/articles/9952618-comparison-viewer

### Security, dependencies, and distribution

- https://github.com/python-pillow/Pillow/security/advisories
- https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html
- https://ffmpeg.org/legal.html
- https://www.gyan.dev/ffmpeg/builds/
- https://cyclonedx.org/specification/overview/
- https://pip.pypa.io/en/stable/topics/secure-installs/
- https://packaging.python.org/en/latest/specifications/pylock-toml/
- https://pypi.org/project/opencut-ppro/
- https://docs.nvidia.com/nemo/speech/nightly/asr/inference.html
- https://github.com/NVIDIA-NeMo/Speech/releases
- https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- https://huggingface.co/nvidia/canary-1b-v2
- https://dotnet.microsoft.com/en-us/platform/support/policy

### Adjacent projects, research, and community

- https://opentimelineio.readthedocs.io/en/v0.16.0/tutorials/versioning-schemas.html
- https://handbrake.fr/docs/en/1.2.0/advanced/queue.html
- https://flask.palletsprojects.com/en/stable/blueprints/
- https://community.adobe.com/bug-reports-728/known-issue-inaccuracies-in-pause-and-filler-filtering-and-language-detection-in-premiere-26-2-26-3-1629693

## Open Questions

None. Parakeet/Canary Windows throughput and timestamp behavior require live validation during implementation, but do not block prioritization.
