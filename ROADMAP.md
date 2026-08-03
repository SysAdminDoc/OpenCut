# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-29 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) are tracked separately in a
maintainer-local `Roadmap_Blocked.md`, which is deliberately untracked — it is
a working file, not part of a clone. This file is the tracked queue.

## Research-Driven Additions

### P0 — 2026-08-02 (research pass)

- [ ] P0 — Enable PlayerDebugMode for CSXS 13–18 in the WPF installer
  Why: The README's recommended Windows install path cannot make the CEP panel load on any Premiere newer than CC 2022, and reports success anyway.
  Evidence: `installer/src/OpenCut.Installer/Models/AppConstants.cs:38` sets `CsxsVersions = { 7,8,9,10,11,12 }`, consumed at `Services/RegistryManager.cs:65-81`; Premiere CC 2023+/2025 use CSXS 13–18. `Install.ps1:555-561` and `OpenCut.iss:105-116` already cover 7–18, and the PowerShell comment names this exact regression.
  Touches: `installer/src/OpenCut.Installer/Models/AppConstants.cs`, `Services/RegistryManager.cs` (two log strings hardcode "7-12"), `installer/tests`.
  Acceptance: A test asserts the installer's CSXS set equals the `Install.ps1`/`OpenCut.iss` set; a smoke run writes `PlayerDebugMode` under CSXS.13–18 and a drift test fails if the three lists diverge again.
  Complexity: S

- [ ] P0 — Stop the installer force-killing Premiere
  Why: Re-running install or uninstall while the panel is connected terminates Premiere and loses unsaved project work.
  Evidence: `Install.ps1:138-143` and `:217-229` take the last column of every `netstat -ano | Select-String ":5679 "` row; that matches `ESTABLISHED` rows whose PID is the *client* — i.e. Premiere. `installer/src/OpenCut.Installer/Services/ProcessKiller.cs:93` filters with `findstr LISTENING` and is correct.
  Touches: `Install.ps1` (both port-kill loops), `Uninstall.bat` path, installer smoke scripts.
  Acceptance: A test feeds recorded `netstat -ano` output containing an ESTABLISHED row for the port and asserts only the LISTENING PID is selected.
  Complexity: S

- [ ] P0 — Normalise and guard the chosen install path before uninstall deletes it
  Why: Uninstall runs `rmdir /s /q` on a raw, unvalidated user string, so choosing a drive root or an existing media folder in the browser destroys it.
  Evidence: `installer/src/OpenCut.Installer/Pages/OptionsPage.xaml.cs:207` stores `PathBox.Text.Trim()`; the `Path.GetFullPath` result at `:238` is used only for a length check and discarded. `Services/UninstallEngine.cs:123-142` deletes recursively and `:175` runs `rmdir /s /q "{installDir}"`. No drive-root rejection, no non-empty-directory warning, no app-name append.
  Touches: `Pages/OptionsPage.xaml.cs`, `Services/UninstallEngine.cs`, `Models/InstallConfig.cs`, installer tests.
  Acceptance: Selecting a drive root or a non-empty directory is refused or warned before install; the stored path is always absolute; a test asserts uninstall refuses a path that is a drive root, a system directory, or the user profile.
  Complexity: S

- [ ] P0 — Make the test suite runnable from a fresh clone
  Why: Nine test modules read markdown that `.gitignore` excludes from the repo, so the advertised green baseline is reproducible only on the maintainer's machine and no contributor can verify a release.
  Evidence: `tests/test_uxp_migration_docs.py`, `test_uxp_macos_http.py`, `test_uxp_webview_scaffold.py`, `test_uxp_webview_permission_split.py`, `test_uxp_filesystem_permission.py`, `test_cep_uxp_parity_catalogue.py`, `test_windows_arm64_doc.py`, `test_roadmap_mirror.py` all `read_text()` untracked `docs/*.md`; `test_local_release_policy.py` reads the gitignored `CLAUDE.md`. `README.md:475` also points readers at the untracked `docs/UXP_MIGRATION.md`. `tests/test_fresh_clone_integrity.py:57` should catch this but its regex `\[[^\]]*\]\(([^)]+)\)` matches markdown links only, so backticked path references pass.
  Touches: `.gitignore` allowlist, `docs/` (track the 16 untracked files or relocate the fixtures), `tests/test_fresh_clone_integrity.py`, `tests/test_local_release_policy.py`, `README.md:475`.
  Acceptance: `git clone` + `pip install -e ".[dev]"` + `pytest` passes with no missing-file errors; the fresh-clone check also scans backticked and quoted path references, in tracked docs *and* in `tests/`, and fails on the current tree before the fix.
  Complexity: M

### P1 — 2026-08-02 (research pass)

- [ ] P1 — Advance the FFmpeg snapshot floor past the July 2026 fixes and ship the `full` build
  Why: The accepted release lane and the current snapshot floor both predate four HIGH-severity crafted-media fixes, and the bundled variant lacks every FFmpeg 8.x capability the project could expose.
  Evidence: CVE-2026-64832/64833/64835/66041 list 8.1.2 as affected; fix commits `4c6217477f`, `6f80e27654`, `1836ef9684`, `4da9812e25` landed on master 2026-07-02…07-05 and are not on `release/8.1` (Debian tracker marks all four unfixed; ffmpeg.org/security.html omits them). `opencut/core/ffmpeg_provenance.py:47,53,72` still sets `RELEASE_FLOOR=(8,1,2)`, `SNAPSHOT_FLOOR_DATE="2026-06-10"`, and pins `8.1.2-essentials_build-www.gyan.dev`. The bundled `ffmpeg/ffmpeg.exe` reports `8.1.2-essentials` with `--enable-nvdec --enable-cuvid` and without `libsvtav1`/`libdav1d`/`whisper`/`libplacebo`/`vulkan`/`libjxl`/`libvvenc`. Cross-reference: `Roadmap_Blocked.md` P0 "Replace the FFmpeg 8.1.2 security floor" is blocked on the *release* lane only — the snapshot lane already exists in code and makes this actionable now.
  Touches: `opencut/core/ffmpeg_provenance.py`, `scripts/verify_ffmpeg_provenance.py`, installer FFmpeg constants (`AppConstants.cs`, `OpenCut.iss`, `Install.ps1`), `Dockerfile` pinned source + SHA-256, `release_licenses/` source archive, `README.md` install instructions.
  Acceptance: The release lane is refused with a named-CVE message until `n8.1.3` exists; the snapshot floor is `>= 2026-07-06` and records the four CVEs plus their fix commits; the bundled and documented build is a `git-full` snapshot pinned by exact commit hash with its source archived beside it; a probe flips the release lane back on when a fixed tag appears.
  Complexity: M

- [ ] P1 — Test against the dependency versions the project declares
  Why: The 10,726-pass baseline runs on a stack that violates four of OpenCut's own constraints, two at major-version boundaries, so users installing per `pyproject.toml` execute code paths the suite has never run.
  Evidence: In the environment that produced the recorded baseline: `opencv-python` 4.11.0.86 vs declared `>=5,<6`; `edge-tts` 7.2.7 vs `<7`; `cryptography` 49.0.0 vs `<49`; `scenedetect` 0.6.7.1 vs `>=0.7.1`. PySceneDetect 0.7 is a documented breaking release (VFR handling, seconds-vs-frames option semantics, `save-fcp`). `scripts/check_dependency_matrix.py` resolves declared lanes but never compares them to what is installed.
  Touches: `scripts/check_dependency_matrix.py`, `scripts/release_smoke.py`, `tests/conftest.py` or a new `tests/test_declared_floors.py`, `opencut/core/scene_detect.py` (0.7 API), `pyproject.toml`/`requirements.txt` if a constraint is wrong rather than the environment.
  Acceptance: A gate compares every installed distribution against the declared specifier for the active extras and fails on mismatch; it fails on the current environment and passes after either the environment or the constraint is corrected; the PySceneDetect path is exercised on 0.7.x.
  Complexity: M

- [ ] P1 — Close the UXP capability gap before ExtendScript support ends
  Why: Adobe states verbatim that ExtendScript is supported "through September 2026" and CEP 12 is the last CEP release, yet 133 user-facing capabilities exist only in the CEP panel — a UXP-only user cannot even install Whisper.
  Evidence: Literal route references resolve to 189 non-stub routes in `extension/com.opencut.panel/client/main.js` and 77 across `extension/com.opencut.uxp/*.js`; the CEP-only set includes `/audio/separate`, `/captions/translate`, `/audio/enhance`, `/captions/animated/render`, `/export/preset`, `/full`, `/install-whisper`. `opencut/_generated/uxp_migration_dashboard.json` measures the 18 ExtendScript host functions (nearly all `direct_uxp`) and therefore reports migration as near-complete; `extension/PANEL_PARITY.json` already records `"$adobe_cep_eol": "approximately 2026-09"`. Host-write itself is sound — `main.js:812-833` feature-detects `project.lockedAccess()` for 26.3. Cross-references the existing P2 "Complete UXP first-run and settings portability", which covers onboarding/settings but not feature reach.
  Touches: `extension/com.opencut.uxp/*`, `opencut/tools/dump_uxp_migration_dashboard.py` (or equivalent generator), `extension/PANEL_PARITY.json`, `tests/test_panel_tab_parity.py`, `tests/test_cep_uxp_parity_catalogue.py`, locale files.
  Acceptance: The migration dashboard reports **route coverage**, not host-function coverage, and a gate fails when a CEP-reachable route has no UXP path or a recorded, justified exclusion; the CEP-only set is driven to zero for capabilities the product claims, starting with dependency installation, stem separation, translation, enhancement, and export presets.
  Complexity: XL

- [ ] P1 — Publish a downloadable release for the current source tree
  Why: The newest artifact anyone can install is 21 versions old, so no user has any fix shipped since 2026-04-20 — including the security work this roadmap tracks.
  Evidence: `gh release list` shows v1.25.1 (2026-04-20) as the latest; `pyproject.toml:13` is 1.46.0; `git rev-parse refs/tags/v1.34.0` … `v1.46.0` all fail — thirteen shipped versions carry no tag.
  Touches: `scripts/release_smoke.py`, `scripts/release_gate.py`, `scripts/release_composition.py`, `installer/InstallerBuilder.ps1`, `scripts/build_linux_packages.sh`, `CHANGELOG.md`, git tags.
  Acceptance: A tag and an unsigned GitHub Release exist for the current version with the Windows installer, `release-composition.json`, artifact SBOM, third-party notices, and FFmpeg provenance attached; a release-gate check fails when `__version__` has no matching tag.
  Complexity: M

- [ ] P1 — Fix the two CLI commands that crash on valid input
  Why: `opencut scene-detect` raises `TypeError` on its default invocation and `opencut podcast` throws away an expensive diarization pass at the last step.
  Evidence: `opencut/cli.py:1425` calls `detect_scenes(input_file, threshold=..., method=method)` but `opencut/core/scene_detect.py:54-59` accepts no `method` kwarg, so both `--method ffmpeg` (default) and `--method pyscenedetect` fail; `cli.py:1430` then normalises a `SceneInfo` dataclass as list-or-dict, so `--method ml` always reports "Scenes found: 0" and writes `[]`. `cli.py:540-545` passes a `List[TimeSegment]` into `generate_multicam_xml(cuts=...)`, which immediately calls `c.get("end", 0)` at `opencut/core/multicam_xml.py:73` — `TimeSegment` has no `.get`.
  Touches: `opencut/cli.py`, `opencut/core/scene_detect.py` dispatch, `opencut/core/multicam_xml.py` input contract, `tests/` CLI coverage.
  Acceptance: A test invokes every CLI subcommand with its documented default arguments against a generated fixture and asserts exit 0 and non-empty structured output; `scene-detect --method ml` reports the real boundary count.
  Complexity: S

- [ ] P1 — Make generated readiness prove implementation for every record
  Why: 27 auto-generated feature records report `available` with no `impl_module`, which is exactly the blind spot that let three terminal-stub adapters advertise as available.
  Evidence: `opencut/_generated/feature_readiness.json` — 27 of 72 records have `source: "generated"` and `impl_module: ""`, all in state `available`, including `audio.demucs`, `video.sam2`, `video.mediapipe`, `editing.auto-editor`, `auto.otio`. The three previously-caught adapters now carry a populated `impl_module`; nothing prevents the next one.
  Touches: `opencut/tools/dump_feature_readiness.py`, `opencut/registry.py`, `tests/test_feature_impl_readiness.py`.
  Acceptance: A record cannot be emitted in `available` without a resolvable `impl_module` that the stub scanner has inspected; the generator fails on the current tree and passes once all 27 are resolved or reclassified.
  Complexity: M

### P2 — 2026-08-02 (research pass)

- [ ] P2 — Triage the routes that no surface reaches
  Why: 1,253 of 1,518 non-stub routes are referenced by no panel, no command palette entry, no CLI command, and no MCP tool — the product's breadth is unreachable by its own users.
  Evidence: Literal-path matching against `opencut/_generated/route_manifest.json` gives 211 routes referenced across all panel JS, 38 in `opencut/core/command_palette.py`, 93 in `opencut/mcp_server.py`, 5 in `opencut/cli.py`. Margin is small: `client/main.js` builds zero route paths by template literal and the UXP panel builds eight. Cross-reference: the existing P3 "Reconcile the queue allowlist with the documented invariant" is the same judgement at one-tenth the scale and should be decided together.
  Touches: `opencut/_generated/route_manifest.json` generator, `opencut/registry.py`, `opencut/core/command_palette.py`, `docs/` API documentation, `scripts/release_smoke.py`.
  Acceptance: Every shipped route carries a declared surface class (panel / palette / CLI / MCP / integration-only) in the generated manifest; a gate fails on an unclassified route; the README's route claim is restated in terms of what a user can reach.
  Complexity: L

- [ ] P2 — Apply timeline cuts as interchange, not per-clip razor operations
  Why: Razoring clip-by-clip through the host is the mechanism that makes a silence pass leave Premiere unusable, and the panel's own success toast miscounts the result.
  Evidence: `ocApplySequenceCuts` in `extension/com.opencut.panel/host/index.jsx` increments per clip removed per track, so `client/main.js:15157` reports "Applied 9 cuts" for a 3-cut apply on a 1V/2A sequence; an Adobe forum report has a comparable tool's >1000-cut pass leaving Premiere "unusably laggy" on a 4090/i9-14900K/64 GB machine. Round-trip risks to cover: OTIO #569 (FCP XML losing trim points on Premiere import) and auto-editor #70 (only the first audio track survives).
  Touches: `extension/com.opencut.panel/host/index.jsx`, `extension/com.opencut.uxp/main.js`, `opencut/export/otio_export.py`, `opencut/core/multicam_xml.py`, panel result rendering and locale strings.
  Acceptance: A cut list above a configurable threshold is written as a timeline interchange import instead of per-clip razors; a fixture with 1,000 cuts across 1V/2A round-trips with in/out points and all audio tracks intact; the toast reports cuts requested, not clip-removals.
  Complexity: L

- [ ] P2 — Replace the archived Demucs pin and declare the separation backend that actually works
  Why: The declared stem-separation dependency is archived upstream, and the maintained backend the code already supports cannot be installed from any extra.
  Evidence: `facebookresearch/demucs` was archived 2024-04-24 and is pinned `demucs>=4.0,<5` in the `audio`, `torch-stack`, and `all` extras. `python-audio-separator` is wired at `opencut/routes/audio.py:498-503` and probed at `opencut/core/engine_registry.py:470`, but appears in no extra and in no `requirements*.txt`; the only install guidance is a runtime `RuntimeError` string.
  Touches: `pyproject.toml` extras, `requirements.txt`, `opencut/core/dependency_support.py`, `opencut/checks.py`, `opencut/routes/system_runtime_routes.py` hint tables, `docs/MODELS.md`.
  Acceptance: `pip install -e ".[audio]"` installs a maintained separation backend; the dependency dashboard names the archived status of Demucs and does not advertise it as the recommended path; the default backend is the maintained one.
  Complexity: S

- [ ] P2 — Retire the dead `auto-editor` pip pin
  Why: The pinned branch is nine months stale and upstream left PyPI, so every 2026 capability — partial-lossless GOP-copy rendering, linked dissolve transitions, Parakeet TDT word timestamps, MLT export — is unreachable.
  Evidence: `pyproject.toml` pins `auto-editor>=29.3,<30`; PyPI's last release is 29.3.1 (2025-11-04). Upstream was rewritten in Nim and now ships prebuilt native binaries. Positioning note for the docs: distributed builds now gate rendering above 3200×1800 and all professional-NLE export behind a licence key while the repository stays Unlicense.
  Touches: `pyproject.toml`, `requirements.txt`, `opencut/core/auto_edit.py` (binary resolution, same pattern as `get_ffmpeg_path()`), `opencut/checks.py`, `docs/MODELS.md`, installer optional-tools step.
  Acceptance: The integration resolves a bundled or system `auto-editor` binary with a version probe and a clear message when absent; the pip pin is removed or documented as legacy-only; a test asserts the resolver prefers the native binary.
  Complexity: M

- [ ] P2 — Make the CEP panel build a real bundle
  Why: The shipped artifact is a byte-identical copy of the 18,360-line source, so a Chromium-99 runtime parses unbundled, unminified source on every panel open.
  Evidence: `extension/com.opencut.panel/client/dist/main.js` and `client/main.js` have identical MD5 (`5282cc69…`) and identical line counts, despite `vite.config.mjs` and a `build` script in `extension/com.opencut.panel/package.json`.
  Touches: `extension/com.opencut.panel/vite.config.js`, `package.json` build scripts, `scripts/verify-build.mjs`, `CSXS/manifest.xml` `MainPath`, packaging steps.
  Acceptance: `npm run build` produces a bundled, minified `dist/` that differs from source and passes `build:verify`; the installer ships `dist/`; a test fails if the built artifact is byte-identical to a source file.
  Complexity: M

- [ ] P2 — Restate the product claims Premiere 26.2/26.3 made first-party
  Why: Several headline features now ship in the host, so the README and panel copy advertise parity work instead of the differentiated capability.
  Evidence: Premiere 26.2 shipped the Sequence Index panel (search, sort, column chooser, filter funnel, jump, CSV export); 26.3 shipped Single-Word Captions; 25.2 shipped Media Intelligence search; 25.6 added bulk bleep/mute; Auto-Match Loudness and Text-Based-Editing Delete Pauses / Delete Filler Words predate both; Adobe's on-device Speechmatics STT (April 2026) claims 12–16% better accuracy than Whisper-powered creative tools. See RESEARCH.md "Rejected Ideas" for the per-feature verdicts.
  Touches: `README.md` feature overview and comparison sections, `extension/*/locales/en.json` descriptions, `docs/` positioning, `opencut/core/command_palette.py` ordering.
  Acceptance: Every claim that overlaps a native 26.x feature is restated as the differentiated part (cross-project scope, exportable artifacts, unlimited/uncapped, headless/REST access, template breadth) or removed; a doc test asserts no claim states the host cannot do something it now does.
  Complexity: S

- [ ] P2 — Ship diarization-driven cutting on pyannote's exclusive-speaker output
  Why: Cutting at speaker changes is the single most-requested Premiere automation with no free implementation, and pyannote 4.x added an output built specifically to reconcile diarization against imprecise ASR timestamps that OpenCut does not use.
  Evidence: Adobe feature request 1555738 asks for speaker-change cuts "similar to Scene Edit Detection" plus auto track placement and speaker colour-coding; multicam-by-speaker is paywalled by AutoPod ($29/mo), FireCut Pro, and AutoCut. pyannote-audio 4.0 adds `exclusive_speaker_diarization` alongside regular diarization and VBx clustering in `speaker-diarization-community-1`; repo-wide search for `exclusive_speaker` returns zero hits while `opencut/core/diarize.py` already references `community-1`.
  Touches: `opencut/core/diarize.py`, `opencut/core/multicam.py`, `opencut/routes/video_core.py` (`/video/multicam-cuts`), panel multicam surfaces, `docs/MODELS.md`.
  Acceptance: Multicam cut generation consumes the exclusive-speaker output when available, cuts land on speaker boundaries rather than ASR segment boundaries, and a fixture with two overlapping speakers produces cuts within one frame of the diarization boundary.
  Complexity: M

- [ ] P2 — Give the cleanup chain one reversible verb
  Why: OpenCut has every component of the standard cleanup pass and no single entry point, while every competitor ships one button for it.
  Evidence: Podcastle "Magic Dust", FireCut "Magic Cut" (2026-07-22), Descript Underlord all collapse silence trim → denoise → loudness → captions into one action. OpenCut exposes them as separate operations plus workflow presets in `opencut/data/workflow_presets.json` that require the user to know which preset to choose.
  Touches: `opencut/core/workflow.py`, `opencut/data/workflow_presets.json`, CEP/UXP quick actions, locale strings, cut review panel.
  Acceptance: One control runs the chain, shows a single preview of every proposed change before anything is written, and is reversible from the journal as one unit; it degrades honestly when an optional dependency in the chain is missing. Depends on the existing P2 "Compile workflows into preflighted resumable plans".
  Complexity: M

- [ ] P2 — Make the highlight score explainable and re-weightable
  Why: The category's incumbent locks an opaque 0–99 score behind its paid tier and users treat it as triage, not verdict — an inspectable score is the differentiator a local tool can own.
  Evidence: Opus Clip's Virality Score is free-tier-locked and blends hook strength, topic-transition density, speaker engagement, and category history; `opencut/core/virality_score.py` already computes a weighted blend of audio energy, transcript hook, and visual salience, and its own documentation warns the absolute numbers are not comparable across video types.
  Touches: `opencut/core/virality_score.py`, `opencut/core/highlights.py`, `opencut/routes/wave_h_routes.py`, panel result rendering, `docs/`.
  Acceptance: The response returns each named component signal with its weight and contribution; the panel renders the breakdown and lets the user re-weight and re-rank without re-analysing; the docs state the score is ordinal within one video, not absolute.
  Complexity: M

### P3 — 2026-08-02 (research pass)

- [ ] P3 — Export MLT projects for Kdenlive and Shotcut
  Why: The two most actively developed open-source NLEs cannot receive an OpenCut edit natively, and neither can produce one — Shotcut's own roadmap lists OTIO import/export and Kdenlive XML export as not done.
  Evidence: Repo-wide search for `mlt`, `kdenlive`, and `shotcut` returns zero hits in `opencut/`; auto-editor 31.1.2–31.3.0 exports MLT with volume/blur ramps as keyframe animations and timewarp producers; OTIO does not cover MLT.
  Touches: new `opencut/export/mlt_export.py`, `opencut/routes/timeline.py`, `opencut/cli.py`, `tests/`.
  Acceptance: A cut list with keyframed volume and a speed change round-trips into Kdenlive and Shotcut with correct in/out points and timing, verified against a checked-in reference `.mlt`.
  Complexity: L

- [ ] P3 — Emit OTIO transitions and bound the caption round-trip claim
  Why: The OTIO export writes clips and markers but no `Transition` items, and OTIO has no caption schema at all — so the interchange claim is broader than the format can carry.
  Evidence: Search for `schema.Transition` in `opencut/` returns zero hits; OpenTimelineIO issue #62 confirms there is no subtitle/caption schema; issues #442/#445/#446 record that the AAF writer supports only cross-dissolves, no markers, no essence import.
  Touches: `opencut/export/otio_export.py`, `opencut/export/otio_diff.py`, `README.md` interchange claims, `docs/DELIVERY_STANDARDS.md`.
  Acceptance: Transitions survive OTIO export and re-import; the documented interchange matrix states per-format exactly what is and is not carried, and a test fails if a claim exceeds what the writer emits.
  Complexity: M

- [ ] P3 — Resync an existing subtitle file to its video
  Why: Editors with subtitle libraries ask for in-NLE resync repeatedly and no free tool wires it into Premiere; OpenCut already has the ASR and alignment pieces.
  Evidence: Adobe feature request 1326702 quantifies the manual cost at 10–15 minutes per episode against a 1,000+ episode library; `ffsubsync` exists as a standalone CLI. Repo-wide search for `subsync` returns zero hits.
  Touches: new `opencut/core/subtitle_resync.py`, `opencut/routes/subtitle_routes.py`, CEP/UXP captions tab, `opencut/cli.py`.
  Acceptance: An SRT offset by a known constant and an SRT with a known drift both realign to within one frame against a fixture, and the result is previewable before it overwrites anything.
  Complexity: M

- [ ] P3 — Add bulk transcript correction
  Why: Per-word click-in correction is the loudest transcript complaint and OpenCut's transcript surfaces have no find/replace or glossary.
  Evidence: Adobe feature request 1329035 — "correcting the transcript by having to click into the text window is cumbersome when there's lots of correcting to be done"; repo-wide search for `find_replace` and `glossary` returns zero hits.
  Touches: `opencut/routes/transcript_edit_routes.py`, new core module, CEP/UXP transcript editor, `opencut/user_data.py` for the persisted glossary.
  Acceptance: Find/replace across a whole transcript with preview and undo, plus a persisted per-project term glossary applied on transcription; corrections survive re-transcription of unchanged regions.
  Complexity: M

- [ ] P3 — Let the user choose the GPU
  Why: Multi-GPU workstations get whatever device the runtime picks first, and there is no way to steer or exclude one.
  Evidence: Repo-wide search for `cuda_device`, `device_index`, `gpu_index`, `CUDA_VISIBLE`, and `multi_gpu` returns zero hits. Shotcut 26.8.1 shipped a Graphics Adapter setting for exactly this; the same request is open on StoryToolkitAI (#124), ClearerVoice (#62), and python-audio-separator (#180).
  Touches: `opencut/config.py`, `opencut/core/gpu_semaphore.py`, `opencut/core/resource_monitor.py`, ML core modules, settings UI, `/system/gpu`.
  Acceptance: A configured device index is honoured by every GPU-backed operation and reported in `/system/status`; an invalid index fails with a structured error listing the available devices.
  Complexity: M

- [ ] P3 — Run a WCAG rule engine against the rendered panel
  Why: Accessibility coverage is a hand-rolled contrast ratio plus ad-hoc role assertions, so whole rule classes — landmarks, focus order, name-from-content, duplicate ids — are unchecked.
  Evidence: `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:887-947` computes relative luminance and asserts ≥ 4.5:1 in both themes; 34 aria/role assertions exist across the file; `package.json` has no accessibility rule engine in `devDependencies`.
  Touches: `extension/com.opencut.panel/package.json`, `tests/rendered/panel-regression.spec.mjs`, panel markup fixes the scan surfaces.
  Acceptance: An automated WCAG 2.2 AA rule scan runs over each production panel state in both themes with an explicit, reviewed suppression list, and fails on a new violation. Complements the existing P2 "Test production UI states at real breakpoint boundaries".
  Complexity: M

- [ ] P3 — Finish MCP conformance for the transport OpenCut actually serves
  Why: The server advertises the `2026-07-28` revision but the HTTP transport omits headers the revision requires and one error code was not renumbered.
  Evidence: `opencut/mcp_server.py:1657` sets `LATEST_PROTOCOL_VERSION = "2026-07-28"` and the server exposes `--http` on 5681; searches for `Mcp-Method` and `Mcp-Name` return zero hits, and `-32021` (`MissingRequiredClientCapability`) is absent while `-32020`/`-32022` are present. `server/discover`, `resultType`, `ttlMs`, and `cacheScope` are implemented; `subscriptions/listen` is legitimately absent because the server declares `subscribe: false` and `listChanged: false`.
  Touches: `opencut/mcp_server.py` HTTP request/response path, `tests/test_mcp_*`, `docs/MCP_SERVER.md`, `pyproject.toml` (`mcp>=1.26` spans the 1.x→2.x protocol break unbounded; bound it).
  Acceptance: A conformance test drives both protocol eras over the HTTP transport and asserts required headers and error codes; the `mcp` extra is bounded to one major line.
  Complexity: M

- [ ] P3 — Make the Docker `gpu` and `mcp` profiles work as documented
  Why: Both profiles fail out of the box for anyone following the README.
  Evidence: `docker-compose.yml:36-64` reserves an NVIDIA device but the image is `python:3.12-slim-bookworm` with no CUDA runtime, the bundled FFmpeg is configured without NVENC/CUDA (`Dockerfile:53-72`), and the locked set has CPU-only `onnxruntime` and no torch; both services also publish `5679:5679`, so `docker compose --profile gpu up` fails on port allocation. `docker-compose.yml:96-97` points the MCP sidecar at `http://opencut-server:5679` while `OPENCUT_TRUSTED_HOSTS` defaults empty, so `opencut/server.py:738-748` returns 400 `UNTRUSTED_HOST` on every call.
  Touches: `docker-compose.yml`, `Dockerfile` (a real CUDA target or removal of the profile), `README.md` Docker section.
  Acceptance: `docker compose --profile mcp up` completes an MCP→backend round trip with no manual environment editing; the GPU profile either provides working acceleration on a distinct port or is removed.
  Complexity: M

- [ ] P3 — Fix the example plugins so they load
  Why: Two of the three shipped examples are rejected by OpenCut's own plugin validator, so copying the documented reference produces a plugin that never loads.
  Evidence: `opencut/data/example_plugins/clip-notes/plugin.json` and `.../timecode-watermark/plugin.json` declare no `api_version`; `_declared_api_range` returns `None` (`opencut/core/plugin_manifest.py:117-119`) and `check_api_compatibility` then reports `compatible=False` (`:169-183`). Separately `timecode-watermark/routes.py:77` hardcodes `rate=25` for `drawtext=timecode=` and `:34` rejects any frame component above 24, so 29.97/30 fps footage drifts ~20% and a valid `00:00:00:29` start timecode is refused.
  Touches: both `plugin.json` manifests, `opencut/data/example_plugins/timecode-watermark/routes.py`, `docs/PLUGIN_AUTHORING.md`, plugin tests.
  Acceptance: A test installs each shipped example into a temp plugin dir and asserts it loads and `opencut plugins doctor` exits 0; the timecode plugin probes source fps.
  Complexity: S

- [ ] P3 — Make `auto-zoom --apply` use the tracking it computed
  Why: The CLI runs face detection and then renders a static top-left crop, usually cropping out the face it found.
  Evidence: `opencut/cli.py:1170-1179` uses only `keyframes[0]["zoom"]` (commented "simplified: use first keyframe zoom for now") and builds `zoompan=z={zoom_val}:d=1:s={w}x{h}:fps={fps}` with no `x`/`y` expressions, so `zoompan` defaults to `x=0:y=0`.
  Touches: `opencut/cli.py`, `opencut/core/auto_zoom.py` filter construction (share it with `/video/auto-zoom`), tests.
  Acceptance: A fixture with a face in the lower-right renders a zoom centred on the face; the CLI and the route build the filter through the same helper.
  Complexity: S

- [ ] P3 — Complete the `Install.ps1` uninstall path and stop using a bare `pip`
  Why: Uninstall leaves working launchers pointing at a removed package, and the "old package removed" step silently no-ops on multi-Python machines.
  Evidence: `Install.ps1:124-183` removes the CEP extension, pip package, and model caches but leaves `Start-OpenCut.bat` and `Start-OpenCut-Hidden.vbs` (created at `:597-623`), the desktop shortcut (`:626-638`), and every `PlayerDebugMode` key (`:559-583`). `:253-255` and `:161` call bare `pip` before `$pythonCmd` is resolved at `:370-389`, while installs use `& $pythonCmd -m pip` at `:412-439`. `:536` and `:546` call `Remove-Item -Recurse -Force` under `$ErrorActionPreference = "Stop"`, so a Premiere-held handle aborts mid-install with a raw exception.
  Touches: `Install.ps1` uninstall branch, interpreter resolution order, CEP folder removal.
  Acceptance: Uninstall removes every artifact install created and reports what it could not remove; all pip calls go through the resolved interpreter; a locked CEP folder yields a "close Premiere and retry" message rather than a stack trace.
  Complexity: S

- [ ] P3 — Give the WPF installer rollback and upgrade detection
  Why: A failure mid-install leaves PATH mutated and files half-copied with no uninstaller registered, and reinstalling to a new directory orphans the previous copy.
  Evidence: `installer/src/OpenCut.Installer/Services/InstallEngine.cs:198-205` catches every failure, deletes only the temp dir, and rethrows — PATH was already mutated at `:120` and steps 13–14 (uninstaller registration) never run. Nothing reads the existing `HKCU\Software\OpenCut\InstallPath` before overwriting it (`RegistryManager.cs:95`) or the single-GUID HKLM uninstall key (`AppConstants.cs:33`); `FileInstaller.cs:33` is copy-with-overwrite, so stale files from the previous version are never pruned.
  Touches: `Services/InstallEngine.cs`, `Services/UninstallEngine.cs`, `Services/RegistryManager.cs`, `Services/FileInstaller.cs`, installer tests.
  Acceptance: A simulated failure at each step leaves the machine in its pre-install state; installing over an existing install detects the prior path, runs its uninstaller first, and prunes files the new version no longer ships.
  Complexity: M

- [ ] P3 — Launch installer helpers by absolute path and stop expanding the user's PATH
  Why: An elevated installer resolving `python`/`cmd.exe`/`powershell.exe` by bare name searches its own directory first, and the PATH writer bakes environment-variable expansions into the user's PATH.
  Evidence: `Services/DependencyInstaller.cs:11,59` (`"python"`/`"python3"`/`"py"`), `Services/ProcessKiller.cs:92,131`, and `Services/UninstallEngine.cs:179` all use bare filenames with `UseShellExecute=false`, so `CreateProcess` searches the application directory — normally Downloads — before `PATH`. `Services/RegistryManager.cs:17` reads `Path` without `RegistryValueOptions.DoNotExpandEnvironmentNames` and writes it back as `ExpandString` at `:30`, permanently rewriting entries like `%USERPROFILE%\bin` to a literal expansion computed from the elevated account.
  Touches: `Services/DependencyInstaller.cs`, `Services/ProcessKiller.cs`, `Services/UninstallEngine.cs`, `Services/RegistryManager.cs`.
  Acceptance: All helper launches use fully-qualified `%SystemRoot%\System32` paths and a validated absolute Python path; a test asserts a PATH containing `%USERPROFILE%\bin` round-trips unexpanded.
  Complexity: S

- [ ] P3 — Make `install.py` fail when verification fails
  Why: A missing Flask — the one dependency the script treats as critical — is reported as a successful install to the user and to any calling script.
  Evidence: `install.py:262` calls `verify()` and discards the return value; `:264-273` print the success banner unconditionally and `main()` exits 0. `install.py:246-248` shows `verify()` already classifies Flask as critical.
  Touches: `install.py`.
  Acceptance: A run with a missing critical dependency prints the failure and exits non-zero; a test asserts the exit code.
  Complexity: S

- [ ] P3 — Make the Linux bundles resolvable and honest about their data dir
  Why: The Flatpak and AppImage launchers export a variable the application ignores, and neither bundle provides an FFmpeg for the sandbox to find.
  Evidence: `packaging/linux/appimage/AppRun:8` and `packaging/linux/flatpak/opencut-server:7` set and `mkdir -p` `OPENCUT_HOME=~/.local/share/opencut`, but `OPENCUT_HOME` appears nowhere in the Python tree and the data dir is `~/.opencut` (`opencut/helpers.py:108`). Neither `io.github.sysadmindoc.opencut.yml` nor the AppDir staging in `scripts/build_linux_packages.sh:54-71` provides `ffmpeg`/`ffprobe` or the `org.freedesktop.Platform.ffmpeg-full` extension, so `get_ffmpeg_path()` (`helpers.py:64-76`) has nothing to resolve. Needs live validation for the exact Flatpak runtime contents.
  Touches: `opencut/helpers.py` (honour `OPENCUT_HOME` or drop it), `packaging/linux/*`, `io.github.sysadmindoc.opencut.yml`, `scripts/build_linux_packages.sh`.
  Acceptance: A built AppImage and Flatpak each resolve FFmpeg and write user data to the documented location; a test asserts the launcher's declared data dir matches what the application uses.
  Complexity: M

- [ ] P3 — Warn about the RTX 50-series faster-whisper failure before the job runs
  Why: The default transcription backend fails on current-generation NVIDIA hardware unless compute type is forced, and the user sees a raw cuBLAS error.
  Evidence: Subtitle Edit issue #10180 documents faster-whisper crashing with `cuBLAS_STATUS_NOT_SUPPORTED` on RTX 50-series unless `float16` is used; OpenCut ships faster-whisper as the default engine, and upstream has had no release since 2025-10-31.
  Touches: `opencut/core/captions.py` compute-type selection, `opencut/routes/system.py` GPU detection, dependency dashboard, `docs/MODELS.md`.
  Acceptance: The GPU probe detects the affected architecture and selects a working compute type automatically, reporting the substitution; a forced-failure path returns a structured error naming the fix rather than the raw CUDA message.
  Complexity: S

- [ ] P3 — Get OpenCut into the lists its competitors are already on
  Why: Discovery is a distribution channel the project is absent from while every named competitor is listed.
  Evidence: OpenCut appears in none of `krzemienski/awesome-video` (1.9k★), `sindresorhus/awesome-whisper` (2.4k★), `ebu/awesome-broadcasting`, `transitive-bullshit/awesome-ffmpeg`, `wentianli/awesome-video-editing`, or `ScreenKite/awesome-ai-video-editing`; the first two list auto-editor, LosslessCut, and WhisperX.
  Touches: outbound pull requests only; `README.md` description line used as the submission text.
  Acceptance: A submission exists for each list whose scope OpenCut fits, with the one-line description matching the README. Depends on the P1 release item — submit only once a current artifact is downloadable.
  Complexity: S

### P0 — 2026-07-29

### P1 — 2026-07-25

### P1 — 2026-07-29

### P2 — 2026-07-25

- [ ] P2 — Complete UXP first-run and settings portability
  Why: UXP lacks the CEP panel’s recoverable onboarding, full settings import/export, support-bundle export, and issue-report path.
  Evidence: CEP onboarding/settings implementation and rendered tests; `extension/com.opencut.uxp/index.html` nine-pane Settings surface.
  Touches: UXP onboarding/settings UI, shared settings/support endpoints, locale files, rendered state and keyboard tests.
  Acceptance: A new user can connect, choose media, understand unavailable capabilities, and reach a first successful operation; settings round-trip with schema/version validation and redacted support export; malformed imports are non-destructive and actionable.
  Complexity: L

- [ ] P2 — Split the remaining panel controller hotspots
  Why: CEP and UXP controllers still centralize lifecycle, bridge state, navigation, settings, and result rendering, and were among the highest-churn files in the last 200 commits.
  Evidence: `extension/com.opencut.panel/client/main.js` (~18,200 lines), `extension/com.opencut.uxp/main.js` (~8,700 lines), recent decomposition commits.
  Touches: CEP/UXP controller modules, build/source-safety checks, unit and rendered tests.
  Acceptance: Navigation, update lifecycle, settings/diagnostics, and result-state controllers have explicit imports and teardown contracts; no duplicate global ownership remains; controller size/churn budgets are machine-checked; behavior and rendered snapshots remain unchanged.
  Complexity: L

### P2 — 2026-07-29

- [ ] P2 — Compile workflows into preflighted resumable plans
  Why: Saved workflows validate endpoint names but can run for hours before discovering invalid parameters, unavailable dependencies, incompatible media, output collisions, or a failed later step.
  Evidence: `opencut/core/workflow.py:163-180,186-345,378-388`; OpenCut queue/journal/checkpoint primitives; Descript recovery/version history; auto-editor’s inspectable automation model.
  Touches: workflow schema/compiler/executor, typed OpenAPI/readiness registry, media probe, queue/journal/checkpoints, CEP/UXP plan review and tests.
  Acceptance: Save and Run compile the same immutable plan; preflight validates typed parameters, capabilities, media/streams, space, output policy, network use, and side-effect class; users can preview and explicitly approve destructive/cloud steps; completed idempotent steps persist with artifacts/checksums and a failed or restarted workflow resumes safely without repeating them.
  Complexity: L

- [ ] P2 — Test production UI states at real breakpoint boundaries
  Why: Current rendered coverage can pass synthetic state markup, treat placeholder/value as an accessible name, and miss the exact widths where panel layouts change.
  Evidence: `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:12-20,602-626,1692-1701`; CEP/UXP media queries at 620, 700, 820/821, 980, 1020, and 1050.
  Touches: production state renderers, shared rendered fixtures/helpers, CEP/UXP viewport matrix, accessibility and overflow assertions.
  Acceptance: Loading/empty/offline/permission/error/destructive/success states are produced through production renderers; accessible names follow the platform computation and never pass from placeholder/value alone; each actual breakpoint is exercised at boundary minus one, boundary, and boundary plus one in both themes with overflow, focus, keyboard, and semantic assertions.
  Complexity: M

- [ ] P2 — Turn the benchmark registry into a reproducible runner
  Why: The repository defines benchmark IDs and advisory budgets but cannot execute or compare them with enough provenance to guide releases or backend choices.
  Evidence: `opencut/core/performance_benchmarks.py`, `tests/test_performance_benchmark_registry.py`, `opencut/core/eval_datasets.py`; VEBench and Netflix VMAF methodology.
  Touches: benchmark CLI/runner, pinned opt-in fixtures, backend adapters, JSON receipts/baselines, diagnostics and release-smoke integration.
  Acceptance: A documented opt-in command runs selected registered backends and records fixture hash/license, model/dependency versions, hardware, seed, warm-up, repeats, timing/memory, and quality metrics; JSON receipts compare only compatible environments with declared tolerances; unavailable backends skip truthfully; release checks may consume a same-host baseline without penalizing different hardware.
  Complexity: M

## Audit Findings — 2026-08-02

Baseline recorded before this audit (all green, so every item below is a new
finding, not a pre-existing failure): `py -3.12 -m pytest tests/ -q` →
**10726 passed, 21 skipped, 4656 subtests, 0 failed** (686 s);
`ruff check opencut/ --select E,F,I --ignore E501` → clean;
`scripts/sync_version.py --check` → all files in sync at v1.46.0; all five
generated manifests report in-sync; `npx playwright test` (panel rendered
suite) → 57 passed, 1 skipped; `npm run lint` → 0 errors, 24 warnings.

### P1 — 2026-08-02


### P2 — 2026-08-02



### P3 — 2026-08-02

- [ ] P3 — Route the remaining drawtext call sites through the shared escaper
  Category: correctness
  Where: `opencut/core/watermark.py:150`, `kinetic_type.py:234,465`, `template_assembly.py:577`, `thumbnail_ab.py:357`, `adr_cueing.py:249-254`, `multicam_grid.py:156`, `camera_solver.py:698`, `character_consistency.py:691`, `data_animation.py:981`. Related `%`-escaping: `hook_generator.py:416-421`, `ab_variant.py:149-154`, `programmatic_video.py:94-97`. Helper: `opencut/helpers.py:447` (`escape_drawtext`).
  Problem: Each module carries its own partial escaper handling only `'` and sometimes `:`, never `\`, and none sets `expansion=none`. A Windows path or a backslash in user text breaks the filtergraph, and literal `%{...}` in user text is evaluated as a drawtext expression rather than printed. `adr_cueing.py` additionally embeds an unescaped `cue.cue_id`. The `%`→`%%` escaping in the last three modules is likely wrong under `expansion=normal` and would render a literal double `%%`.
  Evidence: Probed the bundled FFmpeg: `drawtext=text='C:\media'` fails with `Invalid argument`, and `%{eif:...}` is evaluated (exit 0) rather than printed. Reachable via user text on the owning routes (e.g. `/video/watermark`, kinetic-type routes at `opencut/routes/color_mam_routes.py:804-880`).
  Fix: Replace every local escaper with `helpers.escape_drawtext` paired with `expansion=none`, as that helper's docstring mandates. Then add a guard test that fails if any `drawtext=` construction in `opencut/core/` does not go through the helper.
  Acceptance: A parametrised test renders `C:\media\clip`, `Title: Part 2`, and `100%{x}` through each affected entry point, asserting exit 0 and literal output. The guard test fails if a new local escaper is introduced.
  Confidence: Verified (the breaking inputs are confirmed against the bundled binary; per-site reachability varies)
  Effort: M

- [ ] P3 — Use SMPTE drop-frame math in tc-sync
  Category: correctness
  Where: `opencut/core/tc_sync.py:57-69` (`_tc_to_frames`), `:251` (`compute_tc_offsets`), `:302` (`find_common_timecode_range`). Correct implementation already exists at `opencut/core/timecode_utils.py:260-286`.
  Problem: `_tc_to_frames` strips the `;` drop-frame separator and computes `hh*fps_int*3600 + …`, which overcounts 29.97 DF timecode by 2 frames per non-tenth minute (e.g. `01:00:00;02` → 108,002 rather than the correct 107,894). Two cameras striped with DF timecode starting minutes apart get relative offsets wrong by roughly 2 frames per minute of timecode delta — defeating the module's stated frame-accurate purpose. Separately, `compute_tc_offsets` and `find_common_timecode_range` apply `sources[0]`'s fps to every source's frame counts, so mixed 25/50 fps sets produce wrong `offset_seconds`.
  Evidence: The `;` separator is discarded before the arithmetic, and the arithmetic contains no drop-frame correction; `timecode_utils` has the correct algorithm and is not imported here.
  Fix: Delegate to `timecode_utils.timecode_to_frames` (honouring `;`) and convert each source with its own fps before comparing.
  Acceptance: A test with two 29.97 DF sources one hour apart asserts the computed offset is exact, and a mixed 25/50 fps pair asserts correct `offset_seconds`.
  Confidence: Verified
  Effort: M

- [ ] P3 — Consolidate the duplicate filter-path escaper and catch metric timeouts
  Category: correctness
  Where: `opencut/core/quality_metrics.py:106-115` (`_escape_filter_path`) versus `opencut/helpers.py:443` (`escape_filter_path`); and `opencut/core/quality_metrics.py:343-364` (`compare_videos` per-metric loop).
  Problem: Two defects in one module. (1) The local escaper handles `\`→`/` and `:` but not apostrophes, while the shared helper handles `'` with the close/reopen idiom — so VMAF breaks for any user whose profile path contains an apostrophe (the log path comes from `tempfile.mkstemp`, e.g. `C:\Users\O'Brien\AppData\Local\Temp\...`). It also duplicates a consolidated helper, against the repo's own convention. (2) The per-metric loop catches only `RuntimeError`, but `_run_ffmpeg_filter_complex` calls `_sp.run(..., timeout=timeout)`, which raises `subprocess.TimeoutExpired` — so one hung metric (VMAF on long media being the obvious case) aborts the entire report including metrics already measured, instead of degrading into `notes` like every other failure. The docstring promises per-metric isolation.
  Evidence: Both are direct reads of the cited lines; the helper's apostrophe handling is present and the local copy's is absent.
  Fix: Use `helpers.escape_filter_path` and delete the local copy; add `_sp.TimeoutExpired` to the per-metric `except` clause.
  Acceptance: A test with an apostrophe in the temp path measures VMAF successfully; a test where one metric times out still returns the other metrics with a note.
  Confidence: Verified
  Effort: S

- [ ] P3 — Scope the IMSC validator's log capture to ttconv
  Category: correctness
  Where: `opencut/core/standards_validators.py:160-206` (`_CollectingHandler` attachment in `validate_imsc`).
  Problem: The handler is added to both `logging.getLogger("ttconv")` and the root logger, but ttconv propagates to root by default — so every ttconv warning/error is captured twice and appears duplicated in `report.errors`/`report.warnings`. More seriously, any logger in the process that propagates to root (including `"opencut"`, which has handlers but default `propagate=True`, `opencut/server.py:79-84`) contributes records during the validation window, so an unrelated ERROR lands in `report.errors` and `report.passed = not report.errors` becomes a false failure. Currently only test and release-gate callers exercise it (mostly single-threaded), which is what keeps this at P3 — but it becomes a live flake the moment the validator is exposed as a route.
  Evidence: Both `addHandler` calls are present at the cited lines with no filter on `record.name`.
  Fix: Attach only to the `ttconv` logger, or keep the root attachment behind a filter on `record.name.startswith("ttconv")`, and de-duplicate findings before returning.
  Acceptance: A test that logs an unrelated ERROR to the `opencut` logger during validation asserts `report.passed` is unaffected and no duplicate findings are recorded.
  Confidence: Verified
  Effort: S

- [ ] P3 — Return 400, not 500, for malformed sequence-index payloads
  Category: reliability
  Where: `opencut/core/sequence_index.py:513-514` (`filter_rows`) and `:143` (frame conversion); route handler `opencut/routes/sequence_index_routes.py:200`, row rebuild at `:47-49`.
  Problem: Two crash shapes. (1) `filter_rows` calls `t.lower()` / `e.lower()` on `tags` and `effects` elements; `_dict_to_row` preserves non-string list elements, so a round-tripped row with `tags: [1]` plus a `query` raises `AttributeError`, which the route does not catch — a 500 where a 400 belongs. (2) Python's `json.loads` accepts `Infinity`, so an infinite `start`/`end`/`fps` reaches `int(round(seconds * fps))` and raises `OverflowError` → 500. (`NaN` yields a 400, but with the cryptic message "cannot convert float NaN to integer".)
  Evidence: The `.lower()` calls are unguarded and `_dict_to_row` performs no element coercion; `_safe_float` does not reject non-finite values.
  Fix: Coerce list elements to `str` in `_dict_to_row`/`build_index`, and make `_safe_float` reject non-finite values with a clear validation message.
  Acceptance: Both payloads return 400 with an actionable message; a test covers `tags: [1]` with a query and `start: Infinity`.
  Confidence: Verified
  Effort: S

- [ ] P3 — Purge terminal jobs by completion time, not creation time
  Category: correctness
  Where: `opencut/jobs.py:664-667` (`_cleanup_old_jobs`); compare the correct SQLite TTL at `opencut/job_store.py:453-455`.
  Problem: Terminal jobs are deleted from memory when `now - created > JOB_MAX_AGE` (1 h default), so a job that *ran* longer than an hour becomes eligible for purge on the first 5-minute tick after it completes. `/status/<job_id>` — the endpoint the CEP panel polls — then 404s ("Job not found") for a job that finished seconds ago. `/jobs/<job_id>` falls back to SQLite, but `/status` does not. The SQLite TTL already does this correctly with `COALESCE(completed_at, created_at)`.
  Evidence: The in-memory branch keys on `created` only.
  Fix: Use `now - (completed_at or created)` for the terminal-purge branch; leave stuck-job detection keyed on `created`.
  Acceptance: A test with a terminal job created 90 minutes ago but completed 1 minute ago asserts it survives the purge and `/status` still returns it.
  Confidence: Verified
  Effort: S

- [ ] P3 — Verify the port holder is OpenCut before force-killing it
  Category: reliability
  Where: `opencut/pid.py:197-231,238-274` (`_kill_via_netstat`, strategy 3); `_is_opencut_on_port` is defined and re-exported at `opencut/server.py:537` but never called in the kill path.
  Problem: Strategies 1 and 2 are correctly OpenCut-specific (its own endpoint, its own PID file). Strategy 3 runs `taskkill /F /T` against whatever PID is listening on the port — including an unrelated user application and its entire process tree. It fires exactly when the first two fail, which is precisely the case where the holder is *not* OpenCut. The aggressive startup behaviour appears intentional, but killing a foreign process tree is a real hazard on a workstation.
  Evidence: Call-graph check — `_is_opencut_on_port` has no callers in the kill path.
  Fix: Gate strategy 3 on `_is_opencut_on_port(host, port)`; if the holder is not OpenCut, skip straight to the alternate-port search that already exists.
  Acceptance: A test with a non-OpenCut listener on the port asserts no kill is attempted and the server binds an alternate port.
  Confidence: Verified
  Effort: S

- [ ] P3 — Hold the per-file lock across read-modify-write sequences
  Category: correctness
  Where: `opencut/user_data.py:373-379` (`create_user_tombstone`), `:517-528` (`save_assistant_dismissed`), plus route-level load→mutate→save such as `opencut/routes/workflow.py:243-265`.
  Problem: The per-file `RLock` makes each individual `read_user_file`/`write_user_file` atomic, but these helpers release it between the read and the write. Flask is threaded, so two concurrent requests interleave and one update is lost — e.g. two concurrent `/workflows/delete` calls each create a tombstone and one silently vanishes, breaking the reversibility guarantee that the destructive-confirmation flow advertises.
  Evidence: Each helper calls the read wrapper and the write wrapper as separate lock acquisitions.
  Fix: Expose a `with user_file_lock(filename):` context manager (the lock is already an `RLock`, so nesting is safe) and wrap the read-modify-write sequences.
  Acceptance: A concurrency test issuing two simultaneous tombstone-creating deletes asserts both tombstones exist.
  Confidence: Verified (code trace; requires concurrency to observe)
  Effort: M

- [ ] P3 — Let the queue runner wait as long as the job is allowed to run
  Category: correctness
  Where: `opencut/routes/jobs_routes.py:1043-1058` (the `_run` poll loop); related terminal-state check in `opencut/core/workflow.py` `_wait_for_job`.
  Problem: The runner polls a dispatched job for 1800 s then marks the entry `QUEUE_JOB_TIMEOUT`, but the job itself may run for 7200 s (`job_stuck_timeout`) and workflows wait 3600 s per step. The runner then starts the **next** entry while the "timed-out" job is still executing, so two heavy jobs run concurrently despite the queue's one-at-a-time design, and the user sees an error for a job that later completes successfully. Separately, `_wait_for_job` treats only `complete`/`error`/`cancelled` as terminal, so an `interrupted` sub-job spins the full timeout.
  Evidence: The 1800 s constant is independent of, and shorter than, both `_JOB_STUCK_TIMEOUT` and the workflow step budget.
  Fix: Poll until the job reaches a terminal state or `_JOB_STUCK_TIMEOUT` (which already guarantees termination) rather than an independent shorter deadline; add `interrupted` to the terminal set.
  Acceptance: A test with a job running longer than 1800 s asserts the queue does not start the next entry and does not report a timeout error.
  Confidence: Verified
  Effort: S

- [ ] P3 — Make the shutdown WAL checkpoint actually run
  Category: reliability
  Where: `opencut/job_store.py:159-182` (`close_all_connections`) and `:145-155`; same pattern in `opencut/journal.py:95-104,158-172`.
  Problem: Connections are created with sqlite3's default `check_same_thread=True`. `close_all_connections()` runs at exit on the main thread and calls `execute("PRAGMA wal_checkpoint(TRUNCATE)")` plus `close()` on connections created by `_io_pool`/worker threads — both raise `ProgrammingError` ("SQLite objects created in a thread can only be used in that same thread") and both are swallowed. Since nearly all `save_job` writes happen on `_io_pool` threads, the documented "checkpoints WAL before closing to avoid orphaned -wal/-shm files" never happens for the connections that matter. The same swallow hides failed closes in the dead-thread pruning paths. Impact is limited (process exit releases the handles) but the hygiene the code claims is not occurring — another check that always appears to pass.
  Evidence: Reproduced the `ProgrammingError` for a cross-thread `close()`/`PRAGMA` against these modules.
  Fix: Open connections with `check_same_thread=False` (each is already thread-confined by design), or have each pool thread close its own connection via an executor-shutdown hook.
  Acceptance: A test asserts that after `close_all_connections()` the `-wal` file is truncated and no exception was swallowed.
  Confidence: Verified
  Effort: M

- [ ] P3 — Wire or remove the versioned-config migration framework
  Category: maintainability
  Where: `opencut/user_data.py:161-257` (`CONFIG_SCHEMAS`, `register_config_schema`, `read_user_file_versioned`, `_MIGRATION_BACKUP_SUFFIX`).
  Problem: `CONFIG_SCHEMAS` has zero production registrations and `read_user_file_versioned` has zero production callers — the only caller is `tests/test_config_and_userdata.py:187-221`. All real reads go through plain `read_user_file`, so a schema migration registered tomorrow would never run in production while appearing to be supported. ~100 lines of framework verified only against itself.
  Evidence: Repo-wide call-site search returns only the test module.
  Fix: Either wire `read_user_file_versioned` into the `load_X()` wrappers (its evident purpose) or delete it with its tests. Given the repo already ships JSON schema migrations elsewhere, wiring is probably correct — but pick one.
  Acceptance: Either a production `load_X()` path is covered by a migration test, or the framework and its tests are gone and the suite still passes.
  Confidence: Verified
  Effort: S (delete) / M (wire in)

- [ ] P3 — Drop `noisereduce` from the declared dependencies
  Category: maintainability
  Where: `requirements.txt` (STANDARD section); `pyproject.toml` `standard` and `audio` extras; install hints at `opencut/core/dependency_support.py:83` and `opencut/routes/system_runtime_routes.py:371,397`.
  Problem: `noisereduce` is never imported anywhere in the repo — no static import, no `import_module`, no `reduce_noise` usage; it appears only in install-hint strings. Everyone installing `[standard]` or `[audio]` pulls the package plus its scipy chain for code that cannot use it. Other spot-checked dependencies (rich, waitress, psutil, keyring, python-json-logger, scenedetect) are all genuinely imported.
  Evidence: Repo-wide import scan finds zero usages.
  Fix: Remove it from the extras and `requirements.txt`, or keep it documented as plugin-only; update the two hint tables either way.
  Acceptance: A fresh `pip install -e ".[audio]"` does not pull `noisereduce`, and the dependency-support table no longer advertises it as a supported backend.
  Confidence: Verified
  Effort: S

- [ ] P3 — Deselect network/integration tests from the default run
  Category: testing
  Where: `tests/test_integration_whisper.py:16-20`; `[tool.pytest.ini_options]` in `pyproject.toml`.
  Problem: The `integration` and `slow` markers are declared but there is no `addopts` filter, so a plain `pytest` downloads a Whisper model over the network and runs real FFmpeg renders. The docstring says "Run manually" but nothing enforces it, and the only guard is a `skipif` on FFmpeg availability.
  Evidence: No `addopts` entry in the pytest config; the marker declarations exist without a default filter.
  Fix: Set `addopts = -m "not integration and not slow"` and document opting in with `-m integration`.
  Acceptance: A plain `pytest` run performs no network I/O; `pytest -m integration` still runs the suite.
  Confidence: Verified
  Effort: S

- [ ] P3 — Cover the workflow between-steps cancellation branch
  Category: testing
  Where: `opencut/core/workflow.py:218-227` (the `_is_cancelled(parent_job_id)` early exit).
  Problem: The branch that returns "Workflow cancelled by user" with partial `step_results` is exercised by no test — the only `parent_job_id` reference in the test suite is against a mocked `run_workflow` (`tests/test_workflow.py:146`), and the repo-root `.coverage` shows these lines unexecuted. Job-level cancellation is well covered (`tests/test_job_cancellation_race.py` is solid), but the workflow-chain contract — partial results and the `steps_completed` count — is unverified. This matters more once the propagation fix above lands.
  Evidence: Coverage data plus the absence of a non-mocked caller.
  Fix: Add a test that flips `_is_cancelled` after the first step and asserts the partial-result shape and `steps_completed`.
  Acceptance: The new test fails if the early-exit branch is removed.
  Confidence: Verified
  Effort: S

- [ ] P3 — Consolidate the duplicated route helpers
  Category: maintainability
  Where: `_json_object_or_400` defined five times — `opencut/routes/dev_scripting_routes.py:26`, `plugins.py:42`, `workflow.py:34`, `workflow_dev_routes.py:34`, `workflow_routes.py:20`; `_stub_503` defined three times — `wave_h_routes.py:73`, `wave_k_routes.py:26`, `wave_l_contract.py:10`.
  Problem: Copies have already drifted: `wave_h_routes._stub_503` has no default for `hint` while the others do, so behaviour depends on which module a route happens to live in. The repo's own "consolidated helpers" convention (CLAUDE.md) exists for exactly this.
  Evidence: Definition counts from a repo-wide search.
  Fix: Move both into `opencut/helpers.py` or a new `opencut/routes/_common.py` and import them; reconcile the `hint` default deliberately.
  Acceptance: Each helper is defined once; all callers import it; the suite passes.
  Confidence: Verified
  Effort: S

- [ ] P3 — Fix the UXP controls that silently do nothing
  Category: correctness
  Where: (a) `extension/com.opencut.uxp/main.js:4693,4701` (Auto Zoom aspect); (b) `:4589-4616` (Loudness Match); (c) `:6704-6708` (chat actions); (d) `:6386-6388` (OTIO export path fallback).
  Problem: Four controls mislead the user about what they do. (a) `zoomAspect` is read into `aspect` and never included in the request payload (`{ filepath, zoom_amount, easing }`), so the user's 9:16 / 1:1 choice has no effect on the output. (b) "Loudness Match" posts `{files: [clipPath, refPath], target_lufs: -14.0}`, so the backend batch-normalises *both* files to a fixed −14 LUFS: the reference's loudness is never measured and a pointless normalised copy of the reference is produced, while the UI ("Matching loudness to reference…", a required reference picker) promises reference-matching. (c) The chat flow toasts "Executing {count} action(s)…" but only counts `r.data.actions` — no dispatch follows. (d) `document.getElementById("clipPathCut")?.value?.trim() ?? document.getElementById("clipPathVideo")?.value?.trim()` uses `??`, but an empty Cut input yields `""`, which is not nullish, so the Video-tab fallback is dead code; meanwhile `updateTimelineReadiness` (`:2999-3001`) uses `||`, so the Export OTIO button *enables* when only `clipPathVideo` is set and then dead-ends on "Select a clip first."
  Evidence: Each is a direct read of the cited lines; the payload objects visibly omit the read values.
  Fix: (a) include `aspect` in the payload (and confirm the backend honours it, else remove the control); (b) either measure the reference first and use its LUFS as the target, or relabel to "Normalize to −14 LUFS" and drop the reference input; (c) wire the actions through the existing NLP apply path or change the copy to "N suggested action(s) — review in the result panel"; (d) change `??` to `||`.
  Acceptance: Each control is covered by a test asserting the request payload or dispatched action reflects the UI state; the OTIO button and its handler agree on which inputs count.
  Confidence: Verified
  Effort: M

- [ ] P3 — Order-guard the Sequence Index filter requests
  Category: correctness
  Where: `extension/com.opencut.uxp/main.js:8954-8981` (debounced filter), `:9061-9065` (facet/sort change handlers).
  Problem: The debounced search (200 ms) and the un-debounced facet/sort handlers each POST `/timeline/sequence-index/filter` with no in-flight cancellation or sequence token, so a slow earlier response landing after a fast later one overwrites `visibleRows` and re-renders headers and sort indicators with stale results. The payload also re-ships the full `rows` array on every debounced keystroke — acceptable at the 250-row page size, heavy on large sequences.
  Evidence: Neither handler tracks a request generation nor holds an `AbortController`.
  Fix: Keep a monotonically increasing request id and apply a response only if it is still the latest, or abort the previous request via `AbortController`.
  Acceptance: A test that resolves two filter responses out of order asserts only the later request's results are rendered.
  Confidence: Likely (race is structural; not reproduced against a live host)
  Effort: S

- [ ] P3 — Tidy the panel copy and the dead onboarding markup
  Category: ux
  Where: `extension/com.opencut.panel/client/locales/en.json`; dead markup at `extension/com.opencut.panel/client/index.html:4145-4207`; count semantics at `client/main.js:15157` versus `host/index.jsx:1850-1870`.
  Problem: Four small quality issues. (1) `audio.effects_desc`, `cut.full_desc`, and `video.style_desc` use a double space plus ASCII `--` where the rest of the file consistently uses an em dash. (2) Terminology is split between "backend" (78 strings) and "server" (13), sometimes within one flow — `conn.dot_disconnected` says "Server disconnected" while `conn.start_hint` says "Start the backend with Start-OpenCut.bat…". (3) The static first-run wizard body — three steps, a Quick Tip, a "Don't show again" checkbox, and an "Open Workspace" button, ~60 lines with 10 live i18n keys — is unreachable: `wizardCloseBtn` and `wizardDontShow` appear in no JS file, and the only consumer of `#wizardOverlay` is the server-backed onboarding, which wipes `card.innerHTML` first (`main.js:18038,18159`). (4) The "Applied {count} cuts" toast reports `r.applied`, which `ocApplySequenceCuts` increments per clip removed *per track*, so a 3-cut apply on a 1V/2A sequence reports "Applied 9 cuts".
  Evidence: String counts from `en.json`; repo-wide search finds no references to the two wizard control ids; the JSX increments inside the per-track loop.
  Fix: Normalise to the em dash; pick one user-facing term (the rest of the UI favours "backend") and apply it consistently; delete the static wizard body (keeping the overlay shell) or wire it as the offline fallback tour; report cuts applied rather than clip-removals, or relabel the toast.
  Acceptance: A lint/test pass asserts no `--` in `en.json` descriptions and a single term for the backend concept; the toast count matches the number of cuts requested.
  Confidence: Verified
  Effort: S

- [ ] P3 — Use function replacement for interpolated error text
  Category: correctness
  Where: pervasive in `extension/com.opencut.panel/client/main.js` — e.g. `:15153`, `:15236`, `:16027`.
  Problem: i18n interpolation uses `String.prototype.replace("{error}", text)` with a string replacement, so backend or FFmpeg error text containing `$&`, `$'`, `` $` ``, or `$$` is mangled by JavaScript's replacement-pattern expansion. Low probability individually, but stderr passthrough makes it reachable and the pattern is repeated widely.
  Evidence: The call sites pass a raw string as the replacement argument.
  Fix: Add one shared interpolation helper that uses a function replacement (`.replace("{error}", function () { return text; })`) and route these call sites through it — closing the whole class rather than the three cited lines.
  Acceptance: A test interpolating an error string containing `$&` and `` $` `` asserts the output is literal.
  Confidence: Likely
  Effort: M

- [ ] P3 — Gate the panel lint warnings
  Category: maintainability
  Where: `npm run lint` in `extension/com.opencut.panel/package.json`; warnings concentrated in `client/main.js`.
  Problem: The lint script exits 0 with 24 warnings (14 `no-redeclare`, 10 `no-unused-vars`), so the count can drift upward unnoticed. Most `no-redeclare` hits are idiomatic ES5 `var i` loop counters re-declared in the same function scope and are harmless — including `editDebounceTimer` at `:128` and `:8030`, which share one binding, so `cleanupTimers()` does clear it. The value here is preventing drift, not fixing the current hits.
  Evidence: `npx eslint client/main.js` lists the 14 `no-redeclare` sites; the two `editDebounceTimer` declarations are both at IIFE top level, so the second is a redundant no-op rather than a bug.
  Fix: Set a warning ceiling (`--max-warnings 24`) so the count cannot grow, then reduce it over time — the unused-vars hits are the ones worth clearing first.
  Acceptance: `npm run lint` fails if a new warning is introduced.
  Confidence: Verified
  Effort: S

- [ ] P3 — Close the UXP teardown asymmetry
  Category: maintainability
  Where: `extension/com.opencut.uxp/main.js:8237-8241` (`beforeunload`) versus `:6722-6852` (`uxpWsDisconnect`).
  Problem: The unload handler closes the SSE stream, theme sync, and the media-scan interval, but never calls `uxpWsDisconnect()`, so `_uxpWs` and `_uxpWsReconnectTimer` survive teardown. UXP host teardown usually reaps them, which keeps this low severity, but the cleanup is inconsistent with the SSE handling in the very same handler and will leak if the panel is reloaded rather than closed.
  Evidence: `uxpWsDisconnect` has no caller in the teardown path.
  Fix: Call `uxpWsDisconnect()` from the `beforeunload` handler alongside the existing cleanup.
  Acceptance: A test asserts the socket and reconnect timer are cleared on teardown.
  Confidence: Verified
  Effort: S

- [ ] P3 — Bring the UXP full-report flow under the single-job contract
  Category: maintainability
  Where: `extension/com.opencut.uxp/main.js:5671-5701` (`runFullReport`).
  Problem: It drives the global processing banner and progress bar through direct `BackendClient.post` loops without calling `markJobStarting`, so a real `JobPoller` job started concurrently (a Settings quick action, or a WebSocket progress event) contends for `progressFill` and `processingMsg`. Impact is low because the deliverables POSTs are fast, but this is the one flow that bypasses the single-job contract every other flow honours.
  Evidence: No `markJobStarting`/`state` interaction in the function.
  Fix: Acquire the job lock via the same controller path the other flows use, or use a scoped progress surface that does not share the global banner.
  Acceptance: A test starting a poller job during a full-report run asserts the banner reflects one owner.
  Confidence: Verified
  Effort: S

- [ ] P3 — Reconcile the queue allowlist with the documented invariant
  Category: docs
  Where: `opencut/routes/jobs_routes.py:179-398` (`_ALLOWED_QUEUE_ENDPOINTS`); the invariant is stated in `CLAUDE.md` Gotchas ("New async routes MUST be added to `_ALLOWED_QUEUE_ENDPOINTS`, or queue operations silently fail").
  Problem: Measured against the live app, 760 parameterless async POST routes exist and 547 of them are not queueable — whole families (`/qc/*`, `/export/gif|prores|dcp`, `/repair/*`, `/privacy/*`, `/rough-cut/*`, `/spectral/*`). Either the invariant is stale and the allowlist is deliberate curation (in which case the gotcha should say so), or this is accumulated omission and hundreds of routes silently return "Endpoint not queueable". Given entries were added wave-by-wave, omission looks more likely — but the intent needs an owner's decision, and right now the documentation and the code disagree.
  Evidence: Route-table enumeration against `_ALLOWED_QUEUE_ENDPOINTS` (the same script used for the sync-route finding above).
  Fix: Decide the intent, then encode it: if curation, rewrite the CLAUDE.md gotcha to say the allowlist is opt-in and explain the criteria; if omission, triage the 547 routes. Either way add the release-gate test that diffs async routes against the allowlist so the two cannot drift silently again.
  Acceptance: The documentation matches the code, and the drift test encodes whichever rule was chosen.
  Confidence: Verified (the numbers); the intent is a judgement call for the maintainer
  Effort: S (decide + test) / M (triage 547 routes)

- [ ] P3 — Unaudited areas needing their own pass
  Category: docs
  Where: repo-wide.
  Problem: This audit did not cover, and no finding above should be read as clearing: the installer (`installer/`, `OpenCut.iss`, `Install.ps1`) and its .NET build; Docker and the Linux packaging lane (`Dockerfile`, `packaging/linux/`, `io.github.sysadmindoc.opencut.yml`); the CLI surface (`opencut/cli.py`, ~1,781 lines) beyond the two commands touched incidentally; the ~130 core modules not sampled (the media pass prioritised FFmpeg-command builders, parsers, and timecode math); the plugin examples under `opencut/data/example_plugins/`; localisation completeness for `es.json` (only key *presence* is machine-checked, not translation quality); and any behaviour requiring a live Adobe Premiere host — every panel finding here was verified by code trace plus the headless rendered suite, never against Premiere itself.
  Evidence: Scope of this pass, recorded honestly.
  Fix: Schedule a pass per area, starting with the installer and Docker lanes since they gate distribution.
  Update (2026-08-02, research pass): the installer, Docker, CLI, Linux packaging, and example-plugin lanes have now had that pass — findings are the P0/P1/P3 items under "Research-Driven Additions" dated 2026-08-02. Remaining unaudited: the ~130 unsampled core modules, `es.json` translation quality, and anything requiring a live Premiere host.
  Acceptance: Each listed area has had a recorded audit pass.
  Confidence: Verified
  Effort: L
