# Research — OpenCut for Premiere Pro

Date: 2026-09-04 — replaces all prior research.

Confidence labels: **Verified** (confirmed from repository state, a built artifact, a first-party
specification, or a directly inspected tracker record), **Likely** (several credible sources agree,
but OpenCut's exact behavior still needs a fixture), **Needs live validation** (the risk is real but
the result must be measured in a Premiere host or a released artifact).

## Executive Summary

OpenCut v1.55.1 is a local-first Premiere automation system: a large Flask media backend, CEP and UXP
panels, reviewable timeline mutations, delivery tooling, interchange, and agent-facing MCP operations.
The 2026-08-23 pass concluded that breadth was no longer the useful target and that provable trust
was. That was right, and several of its P0s have since shipped: embedded decoders are now attested
against the FFmpeg 8.1.2 floor and `opencv_videoio_ffmpeg*.dll` is stripped from frozen artifacts
(`opencut/core/embedded_media_provenance.py:30`, `opencut_server.spec:126`), and the release lock has
moved to `huggingface-hub==1.28.0`, past the 1.26.0 path-traversal fix
(`requirements-release-lock.txt:459`). **Verified.**

Two things changed since that pass, and both outrank everything currently queued.

First, **the product now has real users and they are filing real bugs.** Issues #7 (2026-08-30) and #8
(2026-09-01) are the first substantive external bug reports against a released artifact. Between them
they expose four independent defects, three of which reproduce locally without a Premiere host. The
prior roadmap contains nothing that fixes any of them.

Second, **Adobe's ExtendScript support in Premiere Pro is scheduled to end in September 2026** — this
month ([Adobe community thread](https://community.adobe.com/questions-729/extendscript-to-uxp-for-premiere-pro-1553924),
corroborated by [Agent-Driven Editing 2026](https://github.com/ismael-joffroy-chandoutis/open-source-cinema/blob/master/Agent-Driven-Editing-2026.md)).
OpenCut's CEP panel routes every host mutation through a 167 KB ExtendScript file
(`extension/com.opencut.panel/CSXS/manifest.xml:30`, 27 `evalScript` call sites in
`extension/com.opencut.panel/client/main.js`), and **no installer lane can deploy the UXP panel** —
`Install.ps1`, `OpenCut.iss`, `install.py` and the WPF installer under `installer/src` contain zero
UXP references between them. **Verified.** The 2026-08-23 conclusion that "Adobe has not published an
exact CEP removal version… CEP stays a tested fallback until Adobe publishes a firm cutoff" is now
stale and should not be relied on again.

Top opportunities, in priority order:

| # | Opportunity | Tier | Impact | Effort | Evidence |
|---:|---|---|---:|---|---|
| 1 | Ship the 17 generated manifests inside the frozen build | Now, P0 | 5 | S | Issue #8; `opencut_server.spec:85`; `dist/OpenCut-Server/_internal/opencut/_generated/` is empty |
| 2 | Stop importing a foreign interpreter's site-packages into the frozen server | Now, P0 | 5 | M | Issue #8; `opencut/server.py:158` |
| 3 | Fix Windows single-instance detection so two servers cannot share a port | Now, P0 | 5 | M | Issue #8 log (two PIDs, one port); `opencut/pid.py:98` |
| 4 | Report GPU usability from executable arch support, not adapter presence | Now, P0 | 5 | M | Issue #7; `opencut/gpu.py:333`; `requirements.txt:3` |
| 5 | Move the plugin registry off a namespace the project does not own | Now, P0 | 4 | M | `opencut/core/plugin_marketplace.py:39`; `github.com/opencut` exists, `opencut/plugin-registry` 404s |
| 6 | Make the UXP panel installable from every lane before ExtendScript EOL | Now, P0 | 5 | L | ExtendScript EOL Sept 2026; zero UXP refs in all four installers |
| 7 | Capture native crashes so a dying server leaves evidence | Next, P1 | 4 | S | Issue #8 log ends with no traceback; zero `faulthandler` references repo-wide |
| 8 | Correct the GPU install guidance that sends users to a CUDA index without sm_120 | Next, P1 | 4 | S | `requirements.txt:3` names cu121; RTX 50-series is sm_120 |
| 9 | Gate the staleness of tracked Adobe platform snapshots | Later, P2 | 3 | S | `opencut/_generated/adobe_premierepro_versions.json` `recorded_at` 2026-06-25 |
| 10 | Decide what the loopback WSGI server is, and stop shipping the dev-server warning | Later, P2 | 2 | M | Issue #8 console output; `opencut/server.py:719` |

## Product Map

### Core workflows

- Analyze local media, transcribe, detect silence and filler, propose edits, stage reviewable changes
  before Premiere write-back (`opencut/core/transcript_edit.py`, `opencut/core/cut_review.py`). **Verified.**
- Captions, audio repair, reframes, multicam, highlights and delivery variants as durable background
  jobs (`opencut/routes/jobs_routes.py`, `opencut/core/captions.py`). **Verified.**
- Local review versions, comments, drawings and portable bundles with no hosted account
  (`opencut/core/review_links.py`, `opencut/core/review_bundle.py`). **Verified.**
- Export through FFmpeg, OTIO, AAF, MLT, FCP XML, caption sidecars, broadcast checks and C2PA
  provenance (`opencut/core/delivery_validate.py`, `opencut/core/c2pa_sidecar.py`). **Verified.**
- Controlled operations exposed through panels, CLI, REST and MCP (`opencut/cli.py`,
  `opencut/mcp_server.py`). **Verified.**

### User personas

- Premiere editors automating repetitive work without handing the final cut to an opaque service.
- Privacy-sensitive creators who want local models, local project data and explicit network boundaries.
- Technical operators driving batch, CLI, REST or MCP access instead of panel controls.

### Platforms and distribution

- Windows 10/11, macOS and Linux declared. Windows has a WPF installer plus Inno Setup script; source
  launchers serve macOS and Linux; Docker, Flatpak and AppImage lanes exist. **Verified.**
- CEP targets Premiere 2019+; UXP targets 25.6+ with typings pinned to 26.3. Only CEP is installable
  by any shipped lane. **Verified.**
- 44 GitHub stars, no PyPI/Homebrew/winget publish. Distribution is effectively "download the Windows
  installer from Releases." **Verified.**

### Key integrations and data flows

- Both panels call a loopback Flask backend. CEP mutations cross ExtendScript; UXP mutations cross
  Premiere UXP actions. Media processing uses external FFmpeg plus Python-native libraries. **Verified.**
- Generated manifests under `opencut/_generated/` bind code to public claims — and are absent from the
  frozen build, so every packaged install runs with those bindings silently degraded. **Verified.**

## Competitive Landscape

**Adobe Premiere Pro itself.** Adobe shipped native Text-Based Editing in 2024, which commoditizes the
plain "cut the silences from a transcript" pitch inside the host OpenCut plugs into
([Adobe idea thread](https://community.adobe.com/t5/premiere-pro-ideas/innovative-feature-suggestion-ai-powered-transcript-editing-and-scene-auto-cutting-in-premiere-pro/idi-p/15333734)).
Learn: the host will keep absorbing single-verb features. Avoid: positioning on silence removal alone.
OpenCut's defensible ground is the reviewable multi-step pipeline, local models, and agent access —
none of which Adobe offers.

**AutoCut, FireCut, TimeBolt, Cutback.** Commercial Premiere silence removers, subscription-priced,
with polished single-purpose UX ([AutoCut comparison](https://www.autocut.com/en/blogs/best-tool-remove-silences-2026/),
[Cutback](https://cutback.video/blog/the-best-auto-silence-removal-plugin-for-premiere-pro)). Learn:
their onboarding is one screen and one button; OpenCut's install path is a multi-component server plus
extension plus FFmpeg plus optional models. Avoid: their metered pricing, which is exactly the thing
OpenCut's README positions against.

**PremiereCopilot "Claude Cut".** Descript-style natural-language transcript editing performed inside
Premiere ([product page](https://www.premierecopilot.com/en/blog/descript-alternative-premiere-pro)).
This is the closest direct competitor to OpenCut's agent story. Learn: they lead with plain-English
instructions over the timeline rather than a feature grid. Avoid: cloud-only inference.

**Descript and Cutsio.** Text-first editors that own the transcript surface; Cutsio exports XML/EDL
into the NLE rather than living inside it ([Cutsio](https://cutsio.com/blog/top-descript-alternatives)).
Learn: the transcript is the primary UI, not a tab. This supports the already-queued F422 workbench.
Avoid: round-tripping through interchange when a live host binding exists.

**DaVinci Resolve MCP servers.** `samuelgursky/davinci-resolve-mcp` (~485 stars, 202 claimed features)
and `apvlv/davinci-resolve-mcp` are mature agent bridges for a competing NLE. Learn: agent control of
an NLE has proven demand and Resolve is where it has consolidated. Avoid: their gap — Resolve's API
deliberately withholds primary color controls, so agents cannot close the grading loop.

**Premiere MCP.** `hetpatel-11/Adobe_Premiere_Pro_MCP` is early-stage with many tools unimplemented.
This is the most important competitive fact in the whole landscape: **the Premiere agent-control niche
is effectively unoccupied**, and OpenCut already ships a 2,790-line MCP server with CSRF, auth and
non-loopback bind gating (`opencut/mcp_server.py`). Learn: this is the leapfrog position. Avoid:
letting the CEP/ExtendScript dependency take the whole product down before that position is claimed.

**OpenChatCut, `OpenCut-app/OpenCut`.** Browser-based local-first editors with MCP integration and
their own timelines. Learn: they own the standalone-editor framing and the bare "OpenCut" search term.
Avoid: competing there — the `-ppro` naming decision already recorded in README is correct.

**auto-editor, LosslessCut, Shotcut, Kdenlive.** Mature OSS media tools solving adjacent problems
(Shotcut 26.2.26, LosslessCut 3.69.0 as of 2026-06-04). Learn: LosslessCut's timeline-memory work on
multi-hour files is the kind of bounded-resource engineering F426 needs. Avoid: their scope — they are
editors, OpenCut is an automation layer.

**AutoSubs.** A peer Premiere/Resolve extension whose tracker carries a 2026 report that its CEP
extension no longer loads in Premiere Pro 2026 ([issue #571](https://github.com/tmoroney/auto-subs/issues/571)).
Single reporter, no maintainer confirmation — **Likely, needs live validation** — but it is a direct
warning shot for OpenCut's identical architecture.

## Reported Issues

The tracker is `SysAdminDoc/OpenCut`: 2 open issues, 4 closed, 0 open PRs, discussions enabled with
nothing actionable. Both open issues are against released v1.55.1 on Windows 11 and both are real.

**#8 — route_manifest.json missing, then the server dies (2026-09-01).** This one report contains four
separable defects:

1. *Generated manifests are not packaged.* `opencut_server.spec:85` collects data files from
   `opencut.data`, `ctranslate2` and `faster_whisper` only. It never collects `opencut._generated`.
   Reproduced locally: `dist/OpenCut-Server/_internal/opencut/data/` is populated and
   `_internal/opencut/_generated/` contains **zero** JSON files. All 17 manifests are missing, not just
   the one the user saw — including `feature_readiness.json`, `openapi_contract.json`,
   `mcp_extended_tools.json`, `model_cards.json` and `project_facts.json`. Consumers that degrade
   silently: `opencut/cli.py:32`, `opencut/core/agent_skills.py:22`,
   `opencut/core/feature_readiness.py:11`, `opencut/core/workflow.py:21`,
   `opencut/core/surface_ratchet.py:34`, `opencut/mcp_extended_tools.py:20`. **Verified.**
2. *The frozen server imports a foreign interpreter's site-packages.* `_setup_system_site_packages()`
   (`opencut/server.py:158`) shells out to the first `python`/`python3`/`py` on PATH and appends its
   `site.getsitepackages()` to `sys.path`. The reporter's log shows it adopting `C:\Python312`. The
   packaged build ships its own runtime, so native extension modules built for a different CPython
   minor version are now importable — an ABI mismatch that crashes the interpreter rather than raising.
   It is also a code-execution ingress: any writable PATH directory containing `python.exe` is executed
   at startup, and anything in its site-packages can shadow bundled modules. No version check, no
   allowlist, no signature. This contradicts the project's own ingress posture
   (`opencut/trusted_hosts.py`, `opencut/network_policy.py`, model attestation). **Verified for the
   code path; Likely for it being the specific crash cause here.**
3. *Two servers bind one port.* `_check_port()` (`opencut/pid.py:98`) sets `SO_REUSEADDR` before
   binding. On Windows `SO_REUSEADDR` permits binding over a socket that is *actively* bound, not just
   one in `TIME_WAIT`, so the check returns "available" while a live server holds the port. The
   reporter's log shows pid 26164 writing the PID file for port 5679 at 17:49:39 and pid 10352 writing
   the same port at 17:50:20. The second overwrites the PID file and orphans the first, which is
   consistent with "bridge tells OK and then everything is killed, server unavailable". The correct
   Windows primitive is `SO_EXCLUSIVEADDRUSE`, and `_is_opencut_on_port()` already exists two functions
   below but is never consulted by `_check_port`. **Verified for the defect; Likely for it being this
   user's failure.**
4. *A dying server leaves no evidence.* The log simply stops. There is no `faulthandler`, no
   `sys.excepthook` and no `threading.excepthook` anywhere in the tree. A native-level crash — exactly
   what defect 2 would produce — is unobservable. **Verified.**

**#7 — "GPU index 0 is not available. Available CUDA devices: 0: NVIDIA GeForce RTX 5070" (2026-08-30).**
The error message contradicts itself, and the causal chain is fully traceable:

- `requirements.txt:3` tells GPU users to install torch from `https://download.pytorch.org/whl/cu121`.
  CUDA 12.1 wheels carry no `sm_120` kernels; RTX 50-series Blackwell is `sm_120` and needs cu128 or
  newer ([pytorch#159207](https://github.com/pytorch/pytorch/issues/159207),
  [pytorch#164342](https://github.com/pytorch/pytorch/issues/164342)).
- `list_gpu_devices()` (`opencut/gpu.py:173`) prefers `nvidia-smi`, which reports every physically
  present adapter regardless of whether the installed torch build can execute on it. Devices from that
  path carry **no** `compute_capability` key at all — only the torch fallback at `gpu.py:237` sets one,
  so `faster_whisper_compute_recommendation` (`gpu.py:114`) grades the primary path on a missing field.
- `activate_selected_gpu()` finds index 0 in the device set, calls `torch.cuda.set_device(0)`
  (`gpu.py:333`), catches the resulting `RuntimeError`, discards it, and re-raises
  `GPUSelectionError(0, devices)`. That constructor (`gpu.py:26-35`) renders "index 0 is not available"
  from the very list that contains index 0.
- Nothing in the tree checks `torch.cuda.get_arch_list()` — zero repo-wide hits for `get_arch_list`,
  `sm_120`, or kernel-image errors. So Settings shows the GPU as healthy while every job fails.
- Related: `selected_onnx_providers()` (`gpu.py:358`) pins `CUDAExecutionProvider` whenever an index
  exists, without consulting `onnxruntime.get_available_providers()`. The reporter installed the
  CPU-only `onnxruntime` wheel, so that provider can never be satisfied. **Verified.**

**Closed and judged handled.** #6 (installer `NullReferenceException`, closed 2026-08-25) shipped as
v1.55.1. #5 (CSRF token) is closed in code; the remaining maintainer reply is already tracked as F359
in `Roadmap_Blocked.md`. #1 and #2 are 2026-06 bot/self-reports with no residue. Not re-proposed.

## Security, Privacy, and Reliability

**Plugin registry points at a namespace the project does not own.** `REGISTRY_URL` in
`opencut/core/plugin_marketplace.py:39` is
`https://raw.githubusercontent.com/opencut/plugin-registry/main/registry.json`. Checked 2026-09-04: the
GitHub organization `opencut` **exists** (created 2022-12-03) and is not this project's namespace —
this project is `SysAdminDoc/OpenCut` — while `opencut/plugin-registry` returns 404. Whoever controls
that org can create the repo at any time and become the authoritative plugin index for every
installation. The registry document itself carries no signature. Because publisher trust is TOFU
(`_trusted_publisher`/`_trust_publisher`, `opencut/core/plugin_installation.py:253-273`), a hostile
registry can introduce a new `publisher_id` with its own Ed25519 key and have it pinned silently on
first install. **Verified.**

**Foreign-interpreter import path.** See issue #8 defect 2 above. This is the single largest
unreviewed code-execution ingress in the product, and it exists only in packaged builds — the exact
configuration least likely to be exercised by the test suite. **Verified.**

**Checked and found sound — recorded so the next pass does not re-derive it.** The plugin trust model
is genuinely well built: Ed25519 publisher signatures over `plugin_id\nversion\nartifact_sha256`
(`plugin_installation.py:133-174`), a registry-pinned artifact digest, a local TOFU trust store at
`~/.opencut/trusted-plugin-publishers.json`, per-file lock hashing (`plugin_manifest.py:367`), an
explicit `OPENCUT_PLUGIN_ALLOW_UNSIGNED` opt-in, subprocess worker isolation with a watchdog and an
honest `"security_boundary": "availability isolation; not an OS sandbox"` self-description
(`plugin_runtime.py:386`), and download URLs validated through `opencut/core/url_safety.py`. The MCP
server gates non-loopback binds behind auth and carries CSRF token handling with TTL refresh
(`opencut/mcp_server.py:84-165`). Embedded decoder attestation fails closed against the FFmpeg 8.1.2
floor for CVE-2026-8461 and the frozen build strips `opencv_videoio_ffmpeg*.dll`
(`opencut/core/embedded_media_provenance.py:30`, `opencut_server.spec:126`) — the 2026-08-23 P0 is
resolved and is **not** re-queued here.

**Recovery and rollback.** Nothing regressed, but the packaging defects above mean a released build
cannot be trusted to have the same behavior as the source tree it was built from. Until the frozen
artifact is exercised in the suite, "the tests pass" is a statement about the source checkout only.

## Architecture Assessment

- **The frozen artifact is untested.** 371 test files and ~14,500 tests all run against the source
  tree. Every defect in issue #8 lives exclusively in the packaged build. A smoke test that boots
  `dist/OpenCut-Server` and asserts manifest presence, `sys.path` hygiene and single-instance behavior
  would have caught three of the four.
- **`opencut/gpu.py` conflates three questions** — is an adapter present, is it selectable, can the
  installed runtime execute on it — into one integer-membership test. The fix is a resolved capability
  record per adapter, mirroring the pattern `opencut/registry.py` already uses for feature readiness
  (and which queued item F411 is extending).
- **Installer lanes have no shared contract.** Four independent implementations (`Install.ps1`,
  `OpenCut.iss`, `install.py`, `installer/src`) each hand-roll CEP deployment, and all four omit UXP.
  There is no generated manifest describing what an install must place, which is why the omission is
  uniform and invisible.
- **Documentation gap:** `README.md:70` and `README.md:558` advertise the UXP panel to users who have
  no supported way to install it.
- **Test gap:** no fixture asserts that `opencut/_generated/*.json` survives packaging, and
  `scripts/lint_subprocess_timeouts.py` still scans only `opencut/core` and `opencut/routes`, leaving
  `opencut/helpers.py` — which holds the two most-used `Popen` sites — correct by inspection rather
  than by enforcement (carried forward from the 2026-08-22 pass, still true).

## Rejected Ideas

- **Re-queue the OpenCV/FFmpeg CVE-2026-8461 removal** (source: 2026-08-23 RESEARCH.md P0 #1). Shipped;
  attestation fails closed at the 8.1.2 floor and the DLL is stripped from frozen builds.
- **Re-queue huggingface-hub hardening** (source: 2026-08-23 P0 #2). The lock is at 1.28.0, past the
  1.26.0 fix.
- **Bump PyInstaller / ONNX Runtime / PyAV** (source: PyPI currency check — 6.22.2, 1.29.0, 18.1.0 are
  current as of 2026-09-04). Already exactly what queued item F424 specifies. Not duplicated.
- **Adopt OTIO's experimental editing commands** (source: Agent-Driven Editing 2026). OpenCut mutates
  through the live host, not through interchange; routing edits via OTIO would lose host fidelity for
  no gain.
- **Build a headless NLE** (source: the "no true headless NLE" gap in Agent-Driven Editing 2026).
  Contradicts the project's premise of automating the editor the user already owns.
- **Compete on the bare "OpenCut" name** (source: `OpenCut-app/OpenCut`, ~85K stars). The `-ppro`
  distribution decision in README is correct and should stand.
- **Add a hosted review service** (source: Kitsu/Frame.io comparison). Contradicts the local-first,
  no-account philosophy that portable review bundles exist to preserve.
- **Chase Adobe's native Text-Based Editing feature-for-feature.** The host will always win a
  single-verb race; the pipeline and agent surface is the defensible ground.

Categories deliberately carrying no new item this pass: **accessibility** (the 72-case rendered axe
matrix over both panels, themes and widths is clean since v1.55.0, and F399 already walks the full
scroll container), **i18n** (`scripts/i18n_lint.py` and `scripts/lint_locales.py` are clean; backend
localization beyond en/es is already tracked in `Roadmap_Blocked.md`), **mobile** and **multi-user**
(a single-operator desktop plugin has neither surface), and **dependency currency** (F424 already
names the exact versions that a 2026-09-04 PyPI check confirms are current: PyInstaller 6.22.2, ONNX
Runtime 1.29.0, PyAV 18.1.0).

## Sources

Tracker and repository
- https://github.com/SysAdminDoc/OpenCut/issues/7
- https://github.com/SysAdminDoc/OpenCut/issues/8
- https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.55.1

Adobe platform
- https://community.adobe.com/questions-729/extendscript-to-uxp-for-premiere-pro-1553924
- https://community.adobe.com/questions-628/migration-from-cep-to-uxp-685825
- https://github.com/Adobe-CEP/Samples/blob/master/PProPanel/ReadMe.md
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/
- https://github.com/tmoroney/auto-subs/issues/571

GPU and dependencies
- https://github.com/pytorch/pytorch/issues/159207
- https://github.com/pytorch/pytorch/issues/164342
- https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509
- https://pypi.org/project/torch/
- https://pypi.org/project/onnxruntime/
- https://pypi.org/project/pyinstaller/
- https://app.opencve.io/cve/CVE-2026-8461
- https://github.com/opencv/opencv-python/releases

Windows platform behavior
- https://learn.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse

Competitive landscape
- https://github.com/ismael-joffroy-chandoutis/open-source-cinema/blob/master/Agent-Driven-Editing-2026.md
- https://github.com/samuelgursky/davinci-resolve-mcp
- https://github.com/hetpatel-11/Adobe_Premiere_Pro_MCP
- https://github.com/0xsline/OpenChatCut
- https://www.premierecopilot.com/en/blog/descript-alternative-premiere-pro
- https://www.autocut.com/en/blogs/best-tool-remove-silences-2026/
- https://cutback.video/blog/the-best-auto-silence-removal-plugin-for-premiere-pro
- https://cutsio.com/blog/top-descript-alternatives
- https://community.adobe.com/t5/premiere-pro-ideas/innovative-feature-suggestion-ai-powered-transcript-editing-and-scene-auto-cutting-in-premiere-pro/idi-p/15333734

## Open Questions

- Which Premiere major versions still auto-load CEP panels after the September 2026 ExtendScript
  cutoff? The AutoSubs report says 2026 does not; Adobe's own statements say CEP continues for
  "several years". Only a live 26.x host settles it, and the answer decides whether the CEP panel is
  a supported fallback or dead weight. Tracked as blocked work (F386).
- Does the maintainer control, or can the maintainer obtain, a GitHub namespace suitable for the
  plugin registry? The fix for the dangling-registry finding differs depending on whether the answer
  is "move it under `SysAdminDoc`" or "sign the registry document and pin the key".
- Is the reporter in issue #8 running the bundled runtime or a source install? The log shows both a
  frozen-build marker and a `C:\Python312` adoption, and the fix ordering depends on which.
