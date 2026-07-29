# Research — OpenCut
Date: 2026-07-29 — replaces all prior research.
Confidence: all claims are Verified unless explicitly labeled otherwise.

## Executive Summary

OpenCut is a local-first automation and interchange layer for Adobe Premiere, not a replacement editor. Its strongest current shape is the breadth of its inspectable surfaces—1,542 shipped Flask routes, CLI/REST/MCP access, CEP and UXP panels, durable jobs and recovery, and local media processing—combined with a philosophy of previewable, reversible host changes. The highest-value direction is to make those boundaries fail closed before adding more AI breadth. **Confidence: Verified** from the live tree, generated manifests, executable probes, the last 200 commits, rendered UI coverage, dependency audits, and the sources below.

Top opportunities, in priority order:

1. Reject untrusted `Host` values before `/health` can issue a CSRF token.
2. Replace the FFmpeg 8.1.2 security floor with patch- and capability-aware provenance.
3. Remove unnecessary CEP Node/mixed-context privileges.
4. Make generated readiness and contract inspection truthful and side-effect free.
5. Finish Sequence Index as the accessible, searchable table its UI promises.
6. Consolidate typed OpenAPI and migrate MCP across the legacy/2026-07-28 protocol boundary.
7. Gate IMF, IMSC, loudness, and media claims with reference validators and real fixtures.
8. Prove upgrades from the public installer and compile workflows into previewable, resumable plans.
9. Test production renderers at real breakpoint boundaries instead of synthetic state markup.
10. Turn the existing benchmark registry into a reproducible, provenance-rich runner.

## Product Map

- **Core workflows:** select Premiere media; run local cut, caption, audio, video, search, or delivery automation; review a proposed result; write approved changes through CEP/UXP; export media, captions, timelines, review artifacts, or standards-oriented packages.
- **Personas:** solo and small-team Premiere editors, caption/transcript operators, podcast and social-video editors, delivery/QC operators, and technical users who need repeatable local automation.
- **Platforms and distribution:** Python 3.11–3.14; Windows WPF installer and PyInstaller bundle; source launchers for Windows/macOS/Linux; Docker/Linux packaging scaffolds; CEP for Premiere 2019+ and UXP for Premiere 25.6+. The project is MIT-licensed; bundled FFmpeg obligations are handled separately.
- **Integrations and data flow:** panels, CLI, and MCP call a Flask service that is loopback-only by default; FFmpeg and optional local/cloud engines process media; Premiere receives approved host actions; SQLite and user-data stores persist queue, journal, review, index, plugin, settings, and diagnostic state.
- **Product boundary:** augment Premiere while keeping core inference local, network use explicit, destructive work checkpointed, and unavailable dependencies honestly unavailable.

## Competitive Landscape

- **Adobe Premiere / UXP** — Does well: host-native transcript editing, direct timeline actions, theme-aware UI, and the 26.3 Sequence Index’s searchable/sortable spreadsheet, filters, jump, and CSV export. Learn: expose real host capabilities with persistent table state and precise version checks. Avoid: Adobe-cloud dependence and undocumented host behavior.
- **DaVinci Resolve** — Does well: project libraries, proxy/relink workflows, timeline comparison, collaboration review, and explicit recovery. Learn: make migration, relink, compare, and rollback executable contracts. Avoid: turning OpenCut into another NLE or mandatory shared-cloud product.
- **Descript** — Does well: text-first editing, version restoration, recoverable failures, and clear paid boundaries around storage, AI hours, and collaboration. Learn: durable recovery/update states and conflict-safe history are product features. Avoid: cloud-required media retention and usage-credit gating.
- **LosslessCut / auto-editor** — Do well: focused automation, conservative overwrite behavior, scriptable workflows, and inspectable timing semantics. Learn: test damaged timestamps, metadata, stream layouts, and export timing with real media. Avoid: treating a preview as proof of final A/V correctness.
- **StoryToolkitAI / WhisperX / Subtitle Edit** — Do well: footage search, transcript review, waveform alignment, diarization, and correction-oriented caption workflows. Learn: preserve human corrections and record model/alignment provenance. Avoid: silently rewriting approved text or allowing dependency drift to change results without notice.
- **Kdenlive / OpenShot / OpenTimelineIO** — Do well: backups, proxy recovery, documented project migrations, and versioned interchange schemas. Learn: fixture-driven migrations and explicit unknown-schema behavior. Avoid: inventing a proprietary timeline format when Premiere/OTIO/XML remain the boundary.
- **Frame.io / FireCut** — Do well: low-friction review/version actions and focused in-host automation. Learn: make updates, review state, and recovery actions persistent and actionable. Avoid: opaque upload defaults, subscription-only core recovery, and trend features without deterministic review.
- **OpenCut-app/OpenCut** — Does well: a plugin-first standalone editor, portable project data, and browser/desktop reach. Learn: version plugin contracts before ecosystem growth. Avoid: product convergence; this repository’s defensible scope is Premiere-native local automation.

## Security, Privacy, and Reliability

- **Untrusted Host reaches loopback authority — Verified:** Flask has no `TRUSTED_HOSTS` policy, `/health` compares `Origin` with attacker-controlled `request.host_url`, and loopback authentication exempts `/health` (`opencut/server.py:387-402,682-716`, `opencut/routes/system.py:378-428`, `opencut/auth.py:402-439`). A live testing-client request with `Host: attacker.invalid:5680`, matching `Origin`, and loopback peer returned HTTP 200 plus a CSRF token. This creates a DNS-rebinding path to authenticated mutations.
- **Accepted FFmpeg floor is newly unsafe — Verified:** OpenCut accepts any tagged build `>=8.1.2` and pins `8.1.2-essentials_build-www.gyan.dev` (`opencut/core/ffmpeg_provenance.py:45-85`, installer constants/scripts, `README.md:90-105`). NVD lists 8.1.2 in the affected range for CVE-2026-64832, CVE-2026-64833, CVE-2026-64835, and CVE-2026-66041. The repository’s local `8.1.2-full_build-www.gyan.dev` exposes NVDEC, S/PDIF, ADX, and `quirc`, so version-only acceptance is demonstrably insufficient. FFmpeg still lists 8.1.2 as the latest stable release on 2026-07-29, requiring an audited post-fix snapshot or a build that proves affected components absent until a fixed point release exists.
- **CEP privilege exceeds need — Verified:** the production manifest enables `--enable-nodejs` and `--mixed-context`; the only production Node use is `openLogs()`, which aliases `require` and launches an OS process (`extension/com.opencut.panel/CSXS/manifest.xml:32-33`, `extension/com.opencut.panel/client/main.js:8783-8833`). Adobe disables Node by default for security. Existing tests require the privilege and miss the aliased import (`tests/test_platform_ux.py:426`, `tests/test_cep_external_launch_boundary.py:16`).
- **Generated readiness can advertise terminal stubs — Verified:** `auto.deblur-motion`, `auto.searaft`, and `auto.track-cutie` resolved as `available` locally while their adapters terminate in `NotImplementedError`. Generated records omit `impl_module`, bypassing the stub scanner (`opencut/registry.py:145-166,1052-1069`, `opencut/tools/dump_feature_readiness.py:263-316`, the three adapters under `opencut/core/`).
- **Read-only contract generation can mutate runtime state — Verified:** `dump_route_manifest.build_manifest()` calls non-testing `create_app()`, which can run stale-job cleanup, workers, credential migration, and plugin loading (`opencut/tools/dump_route_manifest.py:289-302`, `opencut/server.py:335-375,463-471`).
- **Current direct dependency audits are green — Verified on 2026-07-29:** `npm run audit:check`, runtime `pip-audit`, and build-lock `pip-audit` found no known vulnerable packages. Generic dependency bumps are therefore lower value than the concrete FFmpeg and boundary fixes.
- **Recovery foundations are strong — Verified:** durable queue/journal state, corruption quarantine, immutable review versions, artifact hashes, redacted diagnostics, request correlation, SSRF controls, plugin trust/isolation, and explicit offline states already exist. The active gaps are upgrade proof, truthful readiness, conformance, and resumable composite workflows—not another telemetry or storage subsystem.

## Architecture Assessment

- **Breadth needs a stronger truth layer — Verified:** the committed manifest reports 1,567 routes across 107 blueprints: 1,513 implemented, 29 dependency-gated, and 25 stubs. REST, panels, MCP, docs, and readiness must consume one fail-closed capability model rather than independently inferring availability.
- **API contracts are split — Verified:** `opencut/openapi.py` emits OpenAPI 3.0.3 at `/openapi.json`; `opencut/core/openapi_spec.py` emits 3.1.0 at `/api/openapi.json`; `opencut/core/fastapi_app.py:558` maintains another generation adapter; most operations remain weakly typed. `opencut/mcp_server.py:1641,1787` hardcodes `2024-11-05`, while `pyproject.toml:147` excludes MCP SDK 2. The final 2026-07-28 specification and SDK 2.0.0 remove the date gate recorded in the ignored blocker ledger. The existing typed-OpenAPI roadmap item should establish one schema source consumed by every adapter and by a dual-era MCP migration.
- **Workflow validation stops too early — Verified:** `validate_workflow_steps()` checks endpoint names, then `run_workflow()` executes sequentially and can wait one hour per child job (`opencut/core/workflow.py:163-180,186-345,378-388`). It does not preflight typed parameters, dependencies, input streams, disk/output collisions, network use, side-effect class, or resumability.
- **Sequence Index backend outpaces its UI — Verified:** the backend returns rows and supports filtering/sorting (`opencut/core/sequence_index.py:343-525`, `opencut/routes/sequence_index_routes.py:51-179`), but UXP renders only a summary (`extension/com.opencut.uxp/index.html:1502-1512`, `extension/com.opencut.uxp/main.js:8460-8504`). Adobe 26.3 establishes the expected table, search/filter, issue highlighting, jump, column persistence, and export behavior.
- **Standards claims need independent proof — Verified:** `opencut/core/imf_package.py` hand-builds ST 2067 XML/MXF without an external IMP validator; `README.md:289` says “validated IMSC 1.3,” while `opencut/core/caption_interchange.py:361-759` implements a selected internal ruleset, not the W3C conformance corpus and HRM. Loudness and VMAF similarly need reference fixtures plus model/standard provenance.
- **Rendered coverage has blind spots — Verified:** the shared accessibility helper accepts placeholder/value as an accessible name, semantic state coverage inserts synthetic DOM instead of invoking production renderers, and width matrices skip actual CSS boundaries such as 620, 700, 820/821, 980, 1020, and 1050 (`panel-regression.spec.mjs:12-20,602-626,1692-1701`; panel CSS).
- **Performance machinery is declarative only — Verified:** `opencut/core/performance_benchmarks.py` defines specs, budgets, and a measurement primitive, but no runner captures fixtures, model/dependency versions, hardware, warm-up, repeats, quality, baselines, or a release receipt.
- **Churn supports root-cause work — Verified:** the last 200 commits include 84 `fix`, 29 `feat`, and 19 `refactor` subjects, with repeated feature-then-repair sequences in panel, registry, release, and media boundaries. Existing controller-split, upgrade, media-corpus, accessibility, i18n, packaging, plugin-versioning, and update items remain correctly prioritized.

## Rejected Ideas

- **Standalone editor or mobile client — Rejected (OpenCut-app, CapCut):** conflicts with the Premiere-extension boundary and duplicates mature product shapes.
- **Mandatory cloud media or broad multi-user collaboration — Rejected (Resolve, Frame.io, Descript):** valuable commercially, but incompatible with local-first defaults and disproportionate to the current trust gaps.
- **Prompt-only autonomous timeline mutation — Rejected for now (VideoAgent, CutVerse, Crayotter):** research is promising, but OpenCut first needs typed, previewable, checkpointed, and resumable workflow plans.
- **More readiness-only AI adapters — Rejected (awesome-video lists):** three existing adapters already demonstrate that breadth without executable truth damages trust.
- **Restore GitHub Actions — Rejected (repository history):** workflows were deliberately removed in favor of local builds; strengthen the local fail-closed release path instead.
- **Restore in-process DA3 execution — Rejected (repository history):** the experiment landed and was retracted because the isolation boundary was not acceptable.
- **Code signing or notarization — Rejected (repository operating rules):** prohibited; docs and packaging must not treat signing as a release gate.
- **Additional locales or RTL claims without human QA — Rejected from the active roadmap:** UXP English/Spanish and CEP English exist, while human language validation remains explicitly environment-gated in `Roadmap_Blocked.md`.
- **PyPI, Homebrew, winget, or live-Premiere validation as active coding work — Rejected from this roadmap:** credentials, external review, hardware, or a signed-in host gate them; keep them in `Roadmap_Blocked.md`.
- **Generic framework upgrades — Rejected (Flask, Vite, Vitest, Playwright, PyInstaller changelogs):** no current capability or advisory justifies a broad migration; adopt dependency changes only behind a specific contract or security need.

## Sources

### Open-source competitors and adjacent projects

- https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.25.1
- https://github.com/mifi/lossless-cut
- https://github.com/mifi/lossless-cut/issues/1027
- https://github.com/mifi/lossless-cut/issues/2930
- https://github.com/WyattBlue/auto-editor
- https://github.com/octimot/StoryToolkitAI
- https://github.com/m-bain/whisperX
- https://github.com/SubtitleEdit/subtitleedit
- https://github.com/KDE/kdenlive
- https://www.openshot.org/static/files/user-guide/OpenShotVideoEditor.pdf
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO
- https://github.com/OpenCut-app/OpenCut
- https://github.com/Netflix/photon
- https://github.com/wentianli/awesome-video-editing
- https://github.com/ebu/awesome-broadcasting

### Commercial products and host platform

- https://helpx.adobe.com/premiere/desktop/organize-media/file-organization/navigate-timelines-with-sequence-index.html
- https://helpx.adobe.com/premiere/desktop/whats-new/whats-new.html
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://github.com/Adobe-CEP/CEP-Resources/blob/master/CEP_12.x/Documentation/CEP%2012%20HTML%20Extension%20Cookbook.md
- https://www.blackmagicdesign.com/products/davinciresolve/whatsnew
- https://www.blackmagicdesign.com/products/davinciresolve/collaboration
- https://feedback.descript.com/changelog
- https://help.descript.com/hc/en-us/articles/10164106619405-Version-history
- https://www.descript.com/pricing
- https://help.frame.io/en/articles/9859849-adobe-premiere-frame-io-v4-comments-panel-overview
- https://firecut.ai/changelog

### Community signal

- https://www.reddit.com/r/editors/comments/1u6jue5/anybody_got_better_caption_editing_tools_than/
- https://www.reddit.com/r/editors/comments/1rvqmu6/are_there_better_alternatives_to_captioneer_im/
- https://www.reddit.com/r/CapCut/comments/19e43x6/capcut_auto_captions_are_impossible_to_use_and/
- https://stackoverflow.com/questions/55794240/how-to-concatenate-two-or-more-videos-with-different-framerates-in-ffmpeg
- https://stackoverflow.com/questions/35416110/ffmpeg-concat-video-and-audio-out-of-sync/35520705

### Standards, security, and dependencies

- https://flask.palletsprojects.com/en/stable/web-security/
- https://nvd.nist.gov/vuln/detail/CVE-2026-64832
- https://nvd.nist.gov/vuln/detail/CVE-2026-64833
- https://nvd.nist.gov/vuln/detail/CVE-2026-64835
- https://nvd.nist.gov/vuln/detail/CVE-2026-66041
- https://ffmpeg.org/download.html
- https://modelcontextprotocol.io/specification/2026-07-28
- https://modelcontextprotocol.io/specification/2026-07-28/changelog
- https://py.sdk.modelcontextprotocol.io/v2/
- https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0
- https://spec.openapis.org/oas/v3.1.1.html
- https://www.w3.org/TR/ttml-imsc1.3/
- https://www.w3.org/TR/imsc-hrm/
- https://github.com/w3c/imsc-tests
- https://github.com/w3c/imsc-hrm-tests
- https://github.com/sandflow/imscHRM
- https://www.w3.org/TR/WCAG22/
- https://tech.ebu.ch/publications/ebu_loudness_test_set
- https://www.itu.int/rec/R-REC-BS.1770/
- https://opentimelineio.readthedocs.io/en/latest/tutorials/versioning-schemas.html
- https://vite.dev/blog/announcing-vite8-1
- https://vitest.dev/blog/vitest-4
- https://pyinstaller.org/en/stable/CHANGES.html

### Research and engineering

- https://arxiv.org/abs/2606.23327
- https://arxiv.org/abs/2606.07636
- https://arxiv.org/abs/2605.03276
- https://arxiv.org/abs/2605.19484
- https://arxiv.org/abs/2302.14117
- https://engineering.fb.com/2026/03/02/video-engineering/ffmpeg-at-meta-media-processing-at-scale/
- https://github.com/Netflix/vmaf/blob/master/resource/doc/models.md

## Open Questions

- None that block prioritization or implementation. Credential-, hardware-, human-language-, publication-, and live-Premiere-only work remains isolated in `Roadmap_Blocked.md`.
