# Changelog

Notable changes from the June 2026 hardening/audit pass. The authoritative
record also lives in the git commit messages.

## [Unreleased]

### Added — CJK, Bengali, and RTL caption font-fallback

- Added script-aware font fallback chains to `styled_captions.py` for CJK
  (Noto Sans CJK, MS Gothic, SimSun, Malgun, Meiryo, Yu Gothic), Indic
  (Noto Sans Devanagari/Bengali, Mangal, Vrinda, Nirmala), and RTL
  (Noto Sans Arabic/Hebrew, Tahoma) scripts in both Pillow and Skia
  renderers.
- `_load_font()` and `_load_skia_typeface()` now accept `text_hint` to
  detect caption text scripts and try script-capable fonts before falling
  back to Western fonts.
- Added `get_caption_font_info(text, style_name)` for the UI to surface
  which fallback font was selected or warn when no capable font is
  available for the detected scripts.
- Added Bengali fixture (`bengali_indic`) to caption Unicode validation
  (F223), bringing complex-script fixture count from 5 to 6.
- Added `rtl_arabic` burn-in style preset with right-aligned bottom
  positioning (ASS alignment 6) for natural RTL reading direction.

### Added — D3D12VA and Vulkan HW encoder detection

- Added D3D12VA encoders (h264_d3d12va, hevc_d3d12va, av1_d3d12va) to
  hw_accel.py — vendor-agnostic Windows hardware encoding for non-NVIDIA
  users when FFmpeg 8.x is available.
- Added Vulkan Video encoders (h264_vulkan, hevc_vulkan, av1_vulkan) for
  cross-platform GPU encoding via Vulkan Video API.
- Updated priority order: nvenc > qsv > amf > d3d12va > videotoolbox > vulkan.
- Added quality presets and test-encode options for both new HW types.
- Added 5 new tests for D3D12VA/Vulkan parsing, priority, and verification.
- Wired D3D12VA and Vulkan into `/hw/encode` route validation and all HW
  export preset descriptions. Installer version strings updated for FFmpeg 8.1.

### Fixed — Engineering audit (batch 2): VideoCapture validation

- **segment_sam2.py**: Added `cap.isOpened()` check after `cv2.VideoCapture()`
  to fail early with a clear error instead of silently producing corrupt output.
- **multicam_visual.py**: Added `cap.isOpened()` check in `analyze_frames()`.

### Fixed — Engineering audit: correctness, safety, resource leaks

- **segment_sam3.py**: Fixed unreliable `frame_count` scope check
  (`"frame_count" in dir()` → initialized at top), added
  `VideoCapture.isOpened()` / `VideoWriter.isOpened()` validation, moved
  GPU cleanup (`del predictor, state` + `empty_cache()`) into a `finally`
  block to guarantee cleanup on exception paths.
- **semantic_video_search.py**: Fixed unused `fps` variable in
  `_extract_key_frames()`, fixed progress callback always reporting 0%
  (now computes actual percentage), fixed temp directory leak in
  `build_clip_index()` and `semantic_search()` (added `finally:
  shutil.rmtree`), removed dead `family` variable from
  `_compute_frame_embeddings`.
- **object_intel_routes.py**: Fixed 4 instances of `validate_filepath()`
  misused on output directories (should be `validate_path()` — files vs.
  directories use different validation functions).

### Added — Versioned visual-search engine registry

- Added `SEARCH_ENGINES` registry to `semantic_video_search.py` with 4
  engines: CLIP ViT-B/32 (default), CLIP ViT-L/14, SigLIP ViT-B/16,
  and SigLIP 2 ViT-B/16.
- Cache keys now include engine ID and schema version — different models
  use separate cache directories, preventing embedding dimension mismatches.
- `build_clip_index()` and `semantic_search()` accept an `engine` parameter.
- Model loading supports both CLIP (CLIPModel/CLIPProcessor) and SigLIP
  (AutoModel/AutoProcessor) families.
- Added `list_search_engines()` function and `GET /search/semantic/engines`
  route for UI engine discovery.
- Routes `/search/semantic` and `/search/semantic/index` now accept `engine`
  parameter to select among registered models.

### Added — Multicam visual speech cues (audio+visual mode)

- Added `core/multicam_visual.py` with MediaPipe Face Mesh lip movement
  scoring and shot type classification (wide/medium/close by face-to-frame
  ratio) for multicam cut refinement.
- `/video/multicam-cuts` now accepts `mode: "audio+visual"` to combine
  diarization with visual lip movement analysis. Cross-mic bleed is
  resolved by checking which on-camera face is actually speaking.
- Each cut in visual mode gains `visual_confidence`, `shot_type`, and
  optional `visual_override` annotations.
- Added `check_visual_multicam_available()` to `checks.py`.

### Added — SAM 3 text-prompted object removal engine

- Added `core/segment_sam3.py` with text-prompted object segmentation and
  tracking via SAM 3, plus `segment_video_auto()` with auto-fallback to
  CLIP+SAM2 when SAM3 is not installed.
- Added `POST /video/object-remove/sam3` route accepting text queries
  (e.g., "the watermark in the corner") with SAM2 fallback.
- Added `GET /video/object-remove/engines` listing available segmentation
  backends with capability flags (text_prompts, click_prompts, box_prompts).
- Updated `get_removal_capabilities()` to include `sam3` availability.
- Added `check_sam3_available()` to `checks.py`.
- Added `/video/object-remove/sam3` to queue allowlist.

### Changed — Premiere 26 positioning audit

- Updated README tagline to lead with silence-cut-to-timeline, stem separation,
  voice cloning, animated captions, local LLM, and social export.
- Added "What OpenCut adds beyond Premiere 26" comparison table contrasting
  10 OpenCut-unique capabilities against Adobe's native feature set.
- Added Descript to cost comparison table with stem separation and voice clone
  columns across all competitors.
- Documented Premiere 26.x manifest compatibility in UXP panel section
  (CEP [13.0,99.9], UXP minVersion 25.6).

### Changed — onnxruntime floor raise

- Raised onnxruntime floor from `>=1.25` to `>=1.26` in both `[ai]` and
  `[ai-gpu]` extras (pyproject.toml). 1.26.0 hardens multiple OOB/overflow
  scenarios and replaces unrestricted Python `setattr` config with an
  allowlist.
- Documented the floor raise rationale in `docs/PYTHON_ADVISORIES.md`.

### Fixed — UXP i18n

- Added Spanish diacritics to 400 keys in UXP es.json (507 accent/tilde
  characters introduced; zero existed before).
- Replaced hardcoded English "-- Select a clip --" in main.js with i18n
  `t()` call and added corresponding en.json/es.json keys.
- Added `scripts/lint_locales.py` — locale lint checking key uniqueness,
  placeholder parity between en/es, missing keys, and diacritics regression.
  Supports `--check` flag for CI enforcement.

### Fixed — CEP style.css consolidation

- Merged duplicate `@media (prefers-reduced-motion: reduce)` blocks into one.
- Completed light theme variable block with missing accent gradients, mesh
  background effects, and glow-opacity tokens (8 new variables).
- Retokenized 14 hardcoded hex color literals in light theme overrides to use
  CSS custom properties (`--text-primary`, `--text-secondary`, `--bg-surface`,
  `--bg-raised`).

## [1.33.1] — 2026-06-13 — CI hardening, privacy mode, v30 compat

### Security — dependency floors

- Flask floor raised to >=3.1.3 (CVE-2026-27205 Vary:Cookie info disclosure).
- Waitress floor raised to >=3.0.1 (CVE-2024-49768/49769 request smuggling).
- PyInstaller pinned to >=6.0 in CI (CVE-2025-59042 local privilege escalation).

### Changed — CI

- Added Python 3.13 test matrix job running core test suite on Ubuntu.
- Added Python 3.13 classifier to pyproject.toml.

### Fixed — UXP i18n

- Deduplicated 113 duplicate keys in UXP en.json (1497 → 1384 unique keys).
- `formatI18n` now uses a function replacement to prevent `$&` and `$'` in
  substitution values from corrupting output.

### Added — captions

- Caption export preflight (`POST /captions/export/preflight`) checks segment
  validity, timecode range, QC compliance, and host version before export.
  Returns a fallback strategy (native, srt_sidecar, or burnin) when native
  Premiere caption export is risky. Premiere 26.0/26.0.1 flagged based on
  community reports.

### Fixed — CEP panel

- Wizard and audio-preview dialogs now route through the overlay stack
  (activateOverlay/deactivateOverlay) for proper focus trap, inert background,
  and topmost-only Escape dispatch. Removes the waterfall Escape listener that
  closed every visible surface at once. Audio preview gains aria-modal="true".

### Added — auto-editor v30

- `detect_auto_editor_generation()` distinguishes native v30+ Nim binary from
  legacy v29 pip package. Install hints across model cards and feature registry
  now list both install paths.

### Added — privacy

- Local-only privacy mode via `OPENCUT_LOCAL_ONLY=1` env var or
  `/settings/local-only` UI toggle. Blocks cloud LLM providers (keeps
  Ollama), cloud vision APIs, cloud TTS, social uploads, stock search,
  and telemetry with structured local-alternative error messages.

### Fixed — documentation

- Removed stale ROADMAP-NEXT.md and PROJECT_CONTEXT.md references from
  CONTRIBUTING.md and docs/ROADMAP.md pointer file.

### Changed - CEP/UXP polish

- CEP first-run, launcher, disabled-action, and settings copy now frames the
  workflow around clear workspace actions instead of shortcut-first language.
- CEP and UXP panel surfaces now share tighter 8px geometry, clearer focus and
  disabled states, calmer empty-state feedback, and a more cohesive blue/teal
  premium accent system across onboarding, workspace, tabs, cards, and controls.
- UXP tab navigation now binds before backend discovery so the panel remains
  navigable while the local backend is offline or still being detected.

### Fixed — CEP panel

- Settings no longer exposes non-English UI language choices before matching
  locale files ship, preventing a selection that cannot persistently localize
  the panel.
- Settings import now sanitizes local panel preferences before writing
  `localStorage`, ignoring unknown keys and unsupported preference values while
  surfacing storage failures as a warning toast.

### Changed — audio privacy

- Standalone TTS, auto-dubbing, and overdub workflows now default to local-first
  TTS selection; Edge TTS and external API providers run only when explicitly
  selected.

### Fixed — release process

- Release smoke now catches MCP registry version drift, the committed MCP
  registry manifest is back on the package version, and stale UXP harness/i18n
  guardrails have been brought back in sync.
- README test-count badge now syncs to the live suite count so the release
  readiness gate stays green after new coverage lands.
- Release smoke and doc-size drift checks now target only live documentation
  surfaces (`CLAUDE.md` and `README.md`) instead of removed planning files.
- Version sync now covers the security support table, CEP package-lock root
  metadata, and the C2PA claim-generator string so release smoke fails when
  those public version surfaces drift.

### Security — data loading

- Character embeddings and depth-compositor archives now call `np.load` with
  pickle explicitly disabled, with regression coverage for both paths.

### Security — cache safety

- Preview-cache metadata is now scoped to the active cache directory and ignores
  metadata entries whose files resolve outside that directory before eviction,
  invalidation, or flush can delete them.

## [1.33.0] — 2026-06-11 — June hardening & quality pass

### Fixed — Docker/runtime

- Docker HTTP containers now set `OPENCUT_ALLOW_REMOTE=1` alongside the
  intentional `0.0.0.0` bind, preventing startup crash-loops while preserving
  the non-loopback auth gate.

### Documentation

- README planning links now resolve in a fresh clone and name installer
  artifacts by the release version pattern instead of a stale filename.

### Security — dependency policy

- Standalone depth/model-loading installs now require `transformers>=5.3`;
  the lower Transformers floor is confined to the explicit, documented
  `torch-stack`/WhisperX exception and the advisory audit allow-list now covers
  the current Transformers config-injection CVE only in that opt-in lane.

### Security — server boundary

- Default CORS is now closed instead of allowing `null`/`file://`, and
  `/health` withholds CSRF bootstrap tokens from those origins even if they are
  explicitly added back for compatibility.
- Non-loopback server binds now serve through Waitress instead of Flask's
  Werkzeug development server, covering the Docker/remote runtime path while
  keeping local loopback/debug launches on the dev server.
- MCP HTTP sidecar binds on non-loopback interfaces now require the persistent
  `X-OpenCut-Auth` token before serving health, tool-list, or JSON-RPC calls.
- Opt-in generated MCP route tools now run the same fail-fast path validation
  as curated tools, including nested `query` and `body` path fields.
- Remote HTTP startup now fails closed when `OPENCUT_ALLOW_REMOTE=1` is set but
  the auth-token middleware cannot be installed.
- Every mutating Flask request now passes through a global CSRF gate, so new
  POST/PUT/PATCH/DELETE routes stay protected even if a handler omits the
  per-route decorator.
- Kinetic text, data-animation, shape-animation, and IMF export routes now
  clamp render dimensions and font sizes before dispatching to allocation-heavy
  render code.
- The writable `~/.opencut/packages` fallback install directory is now appended
  to `sys.path` instead of taking priority over stdlib and bundled modules.

### Fixed — UXP panel

- UXP job submission now rejects concurrent backend jobs before they overwrite
  the shared tracker/SSE stream, and job-action buttons stay locked while a
  backend job is starting or running.
- The UXP Refresh button now rescans backend ports 5679-5689, and failed
  background health checks retry against a newly detected port before staying
  offline.
- The UXP live-updates WebSocket now consumes the bridge URL reported by the
  backend, uses a capped reconnect backoff, and honors manual disconnects so
  the Stop button actually stops reconnect attempts.
- UXP job polling now tolerates short `/status/<job_id>` interruptions before
  failing a running job, preventing one dropped request from abandoning work.
- Periodic UXP backend health checks no longer repaint the workspace as
  connecting/offline before each background probe completes.
- FCC caption display settings now unwrap BackendClient response payloads before
  populating token selects or applying preview CSS, making the compliance card
  functional again.
- Cancelling an active job now clears any button left in the panel's loading
  state, so the initiating action is usable again without reloading the panel.

### Security — routes layer

- **Closed the unvalidated `output_path` write sinks.** `@async_job` validates
  only the primary input filepath; eight route modules forwarded a
  user-supplied `output_path`/`output_file`/`db_path` straight into an FFmpeg
  `-y` or `open(..., "w")` sink, allowing arbitrary local file overwrite via a
  crafted request (CSRF-protected, but reachable by any local caller). All
  secondary write targets now pass `validate_output_path`/`validate_path`:
  - `audio_production_routes` (declip/dehum/decrackle/dewind/dereverb,
    room-tone generate/fill, M&E mix, fingerprint `db_path` read+write)
  - `integration_routes` (adjustment-layers apply, flight-map, jog-mapping save)
  - `overlay_routes` (safe-zones + the three `output_path_override` routes)
  - `solver_agent_routes` (3D overlay), `timeline_auto_routes` (auto-mix —
    replaced the brittle startswith/colon check that let absolute paths and
    `..\` traversal through), `cloud_distrib_routes` (farm render),
    `editing_workflow_routes` (7 export/assemble routes),
    `color_mam_routes` (22 output sinks via a shared validated helper).

### Fixed — CEP panel (extension/com.opencut.panel)

- **Cut Review panel was completely broken.** A `for (var t = ...)` loop counter
  in `showCutReview()` shadowed the `t()` i18n function in the same scope, so
  every silence / filler / auto-edit / highlights / repeat-detect completion
  threw `TypeError: t is not a function` and the review panel never opened.
  Renamed the loop variable.
- **A throwing job-done listener could lock the UI forever.** The completion
  dispatch loop ran listeners with no isolation before `hideProgress()`, so any
  listener exception left the processing banner and `body.job-active` pointer
  lock stuck on. Each listener is now wrapped in try/catch.
- **Escape no longer silently cancels a running job** when it was pressed only
  to dismiss a dropdown, menu, wizard, modal, or popover. Added a single
  "any dismissible surface open" guard to the cancel-job shortcut.
- **Enter no longer hijacks focused controls.** The "Enter runs the primary
  action" shortcut now bails when focus is on a button/link/tab/dropdown/
  menuitem, so keyboard activation of those controls works and an unrelated
  job isn't launched.
- **ExtendScript string safety:** migrated the five remaining PremiereBridge
  call sites (importFiles, removeSequenceMarkers, unrenameItems,
  removeImportedSequence, removeImportedItem) from the narrow
  `replace(/'/g,"\\'")` escape to `escSingleQuote()`, which also handles
  newlines and U+2028/U+2029 line separators in clip/marker names.
- **Accessibility:** `document.documentElement.lang` now tracks the active
  locale so assistive tech uses correct pronunciation after a language switch.
- Stale per-run time estimate is cleared at job start instead of lingering
  from the previous run until the new estimate resolves.

### Fixed — FFmpeg integration (backend)

- Styled-caption overlay encoding now drains FFmpeg stderr in a background
  thread while raw frames are piped to stdin, preventing stderr backpressure
  from deadlocking long renders.
- The Real-ESRGAN upscale tier now resolves and caches official model weights
  before constructing `RealESRGANer`, instead of passing `model_path=None`.
- **Caption burn-in failed on every Windows drive-letter path.** The
  `subtitles=`/`ass=` filter value dropped colon escaping, so `C:/...` was
  parsed as filename `C`. Added `escape_filter_path()` with the verified
  two-level escaping (drive colons, apostrophes, spaces) and applied it.
  *Verified end-to-end against real ffmpeg.*
- **Large edits crashed on Windows with "filename or extension is too long".**
  Silence-removal export and `speed_up_silences` built `-filter_complex` as a
  single argv string; past ~220 segments this exceeds the 32 KB command-line
  limit (`OSError WinError 206`). Both now spill large graphs to a temp file
  and use `-filter_complex_script`. *Verified: 300-segment graph (43 KB) fails
  inline, succeeds via script file.*
- **Merge/concat broke on filenames with apostrophes and corrupted non-ASCII
  names.** The concat-demuxer escaping used `\'` (rejected by ffmpeg) and the
  default text codec (cp1252 on Windows). Added `write_concat_list()` (UTF-8 +
  the correct `'\''` idiom). *Verified: apostrophe and Cyrillic/CJK names.*
- All core concat-demuxer writers found by the repo-wide scan now use
  `write_concat_list()` or the shared concat-line formatter instead of local
  platform-codec path writes.
- New shared escaping helpers in `opencut/helpers.py`
  (`escape_filter_path`, `escape_drawtext`, `write_concat_list`) with a
  string-level regression test (`tests/test_ffmpeg_escaping.py`) locking the
  ffmpeg-verified behavior for CI.
- `escape_drawtext` documents/requires `expansion=none`: under default
  drawtext expansion a literal `%` cannot be escaped at all ("Stray %") and
  caption text containing `%{...}` would be interpreted as an expression.
- Caption styles, click/keystroke overlays, tickers, quiz cards, telemetry
  overlays, callouts, audiograms, brand watermarks, end screens, and guest
  lower-thirds now use the shared `escape_drawtext()` contract with
  `expansion=none`.
- Audio-only export to `.aac` and `.ogg` no longer fails — those containers
  were handed the mp3 encoder; now mapped to `aac` / `libvorbis`.
- Progress-parsing FFmpeg subprocess now decodes stdout/stderr as UTF-8 with
  replacement, preventing a `UnicodeDecodeError` in the stderr drain thread
  from deadlocking the progress loop on non-ASCII output.

### Fixed — backend correctness

- **Transcription timeout was a no-op.** The `ThreadPoolExecutor` was used as a
  context manager, whose exit blocks on `shutdown(wait=True)` until the worker
  finishes — so a timed-out transcription still pinned the caller for the full
  run. Switched to manual non-blocking shutdown with `cancel_futures`.
- `smart_trim` now detects trailing silence (a clip that ends in silence emits
  an unmatched `silence_start`), so "trim trailing silence" actually works.
- Stem recombination uses `amix=...:normalize=0`; the default `normalize=1`
  scaled each stem by 1/N, making a recombined mix ~12 dB too quiet.
- `loudness_match` clamps non-finite (`-inf`) measurements from silent input
  (previously produced invalid `-Infinity` JSON and out-of-range pass-2 args)
  and anchors on the last loudnorm JSON block.
- `smart_reframe` guards against ZeroDivisionError from a zero aspect-ratio
  component (e.g. "16:0") or a zero source dimension from a probe failure.
- `extract_audio_wav` no longer leaks its temp WAV when extraction fails.
- `speed_up_silences` now treats MP3/M4A cover art as attached pictures, not
  real video streams, so audio-only files stay on the audio concat path.
- Waveform image uses a unique temp name instead of `waveform_<pid>.png`,
  which collided across requests in the long-lived server.
- `waveform_timeline` uses `array.array` instead of Python list for PCM
  samples (~8x less memory for long audio files).
- `smart_trim` now surfaces subprocess timeout/errors instead of silently
  returning an empty result as "no speech detected".
- Export cancel now calls `proc.wait()` after killing FFmpeg before
  unlinking the partial output file, preventing silent failure on Windows.
- `preview_frame` checks FFmpeg return code and output file size before
  returning success.
- Captions temp WAV cleanup uses `_schedule_temp_cleanup` when immediate
  unlink fails from a timed-out Whisper thread on Windows.
- auth.json now applies a restrictive DACL via `icacls` on Windows (was
  POSIX-only chmod).
- Multiview/repurpose routes now validate secondary file paths
  (`video_paths`, `content_path`, `reaction_path`) before forwarding to core.
- Installers prefer `requirements.txt`/`requirements-lock.txt` over loose
  pip specs in fallback paths.
- CLI `loudness-match` resolves bare filenames to absolute paths before
  passing output_dir.
- CLI `auto-zoom` uses `.get()` for width/height to avoid KeyError.
- Fixed placeholder `github.com/opencut` URLs in CLI banner and
  InstallerBuilder.

### Fixed — CEP panel

- Job history dedupe now compares by `job_id` (or `type+createdAt`
  fallback) instead of `(type, status)`.
- NLP auto-execute raised confidence threshold to 0.8 with a confirm step.
- `body.job-active` UI lock now sets `inert` on main content for keyboard
  and AT users (was pointer-events-only).
- `showAlert()` accepts an explicit tone parameter, bypassing English-only
  regex inference for non-English locales.
- Time estimate reads numeric clip duration from state instead of parsing
  rendered DOM text.
- Command palette section labels ("Recent", "Favorites", etc.) are now
  i18n-configurable via `sectionLabels` on the palette context.
- Language dropdown reduced to only `en` (the only shipped locale).

### Fixed — UXP panel

- Await precedence fix: `(await getOutPoint?.())?.seconds` instead of
  `await getOutPoint?.()?.seconds`.
- `addMarkers` now awaits `getFirstMarkerAtTime` so names/colors apply.
- Enter key handlers guarded with `hasActiveJob()` to prevent duplicate
  submissions.
- 10 handlers changed `else` to `else if (!r.ok)` so ok-without-job_id
  synchronous responses aren't reported as failures.
- `udt-smoke.js` gated behind `localStorage opencut_debug=1`.
- `postMessage` targetOrigin restricted from `"*"` to `location.origin`.
- `cep_node.exec` passthrough blocked in the WebView shim.

### Fixed — Installer

- Inno Setup PlayerDebugMode registry writes use `runasoriginaluser` via
  `reg.exe` so keys land in the invoking user's HKCU, not the admin's.
- VBS launcher uses `WshShell.Environment` instead of `cmd /c` string
  concatenation, preventing injection from paths containing `&` or `^`.

### Changed

- Background sweep threads (temp cleanup, disk monitor) are skipped during
  test runs via `create_app(testing=True)`.
- README: added June 2026 cost comparison table vs CapCut Pro, Submagic,
  AutoCut, and FireCut.
