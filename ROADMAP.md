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

### P1 — 2026-08-02 (research pass)

- [ ] P1 — Test against the dependency versions the project declares
  Why: The 10,726-pass baseline runs on a stack that violates four of OpenCut's own constraints, two at major-version boundaries, so users installing per `pyproject.toml` execute code paths the suite has never run.
  Evidence: In the environment that produced the recorded baseline: `opencv-python` 4.11.0.86 vs declared `>=5,<6`; `edge-tts` 7.2.7 vs `<7`; `cryptography` 49.0.0 vs `<49`; `scenedetect` 0.6.7.1 vs `>=0.7.1`. PySceneDetect 0.7 is a documented breaking release (VFR handling, seconds-vs-frames option semantics, `save-fcp`). `scripts/check_dependency_matrix.py` resolves declared lanes but never compares them to what is installed.
  Touches: `scripts/check_dependency_matrix.py`, `scripts/release_smoke.py`, `tests/conftest.py` or a new `tests/test_declared_floors.py`, `opencut/core/scene_detect.py` (0.7 API), `pyproject.toml`/`requirements.txt` if a constraint is wrong rather than the environment.
  Acceptance: A gate compares every installed distribution against the declared specifier for the active extras and fails on mismatch; it fails on the current environment and passes after either the environment or the constraint is corrected; the PySceneDetect path is exercised on 0.7.x.
  Update (2026-08-03): the gate exists — `scripts/check_installed_versions.py`, run by `release_smoke.py` and covered by `tests/test_declared_floors.py`. It found eight violations, not four. Six are resolved: `scenedetect` upgraded to 0.7.1 and its adapter now exercised against a generated clip on the 0.7 API; `cryptography` pulled back to 48.0.1 and `Pillow` raised to 12.3.0 to meet their declared floors; the stale `edge-tts`/`pytest-cov`/`pre-commit` ceilings were raised to admit the versions the suite actually runs on. Two remain, each blocked by a concrete conflict recorded in `KNOWN_ENVIRONMENT_GAPS`: `opencv-python>=5` and `transformers>=5.3`.
  **opencv 5 was installed and measured before being rolled back, and it breaks the product**: OpenCV 5.0 removes `cv2.CascadeClassifier` entirely (and `cv2.objdetect`), which 13 modules call directly (`auto_zoom`, `face_tools`, `ai_reframe_multi`, `deepfake_detect`, `face_tagging`, `morph_cut`, and others) — so the declared floor makes face-tracked auto-zoom and Haar face blur raise `AttributeError` for anyone installing per `pyproject.toml`. `simple-lama-inpainting` separately pins `opencv-python<5.0.0.0`. Closing the opencv half therefore means migrating the Haar call sites onto a detector that exists in OpenCV 5 (YuNet/DNN or MediaPipe) and resolving the LaMA pin — not just a version bump. `transformers>=5.3` forces `huggingface_hub` 1.x, which the pyannote / faster-whisper / diffusers lane does not accept; that half needs the stack qualified on `huggingface_hub` 1.x.
  Note: installing and then downgrading opencv leaves a mixed `site-packages/cv2` (the 5.0 payload survives a downgrade and `cv2.__version__` keeps reporting 5.0.0). Uninstall all three opencv distributions and delete the leftover `cv2/` directory before reinstalling.
  Complexity: M

- [ ] P1 — Publish a downloadable release for the current source tree
  Why: The newest artifact anyone can install is 21 versions old, so no user has any fix shipped since 2026-04-20 — including the security work this roadmap tracks.
  Evidence: `gh release list` shows v1.25.1 (2026-04-20) as the latest; `pyproject.toml:13` is 1.46.0; `git rev-parse refs/tags/v1.34.0` … `v1.46.0` all fail — thirteen shipped versions carry no tag.
  Touches: `scripts/release_smoke.py`, `scripts/release_gate.py`, `scripts/release_composition.py`, `installer/InstallerBuilder.ps1`, `scripts/build_linux_packages.sh`, `CHANGELOG.md`, git tags.
  Acceptance: A tag and an unsigned GitHub Release exist for the current version with the Windows installer, `release-composition.json`, artifact SBOM, third-party notices, and FFmpeg provenance attached; a release-gate check fails when `__version__` has no matching tag.
  Complexity: M

### P2 — 2026-08-02 (research pass)

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
