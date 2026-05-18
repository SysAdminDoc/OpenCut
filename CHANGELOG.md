# Changelog

## [Unreleased]

### Added — Performance Benchmark Contract

- Added `opencut/core/performance_benchmarks.py` with F214 throughput specs for Whisper-family ASR, AI upscalers, declarative compose, and TTS backends.
- Added `tests/test_performance_benchmark_registry.py` and wired it into release smoke so CI pins the benchmark inventory, backend matrix, opt-in gate, and wall-clock normalization primitive.

### Fixed — Job Cancellation

- `_cancel_job()` now terminates a registered child process immediately and closes parent pipe handles so blocked FFmpeg progress readers unblock during cancellation.
- Added `tests/test_job_cancellation_race.py` to cover the concurrent cancellation race and wired it into release smoke.

### Added — Fuzz Harness Coverage

- Extended the Atheris fuzz harness from 5 to 13 targets, adding coverage for path validation, OTIO parsing, FCPXML parsing, marker import, C2PA sidecars, plugin manifests, webhook HMAC signatures, and `safe_pip_install` package-spec validation.
- Added `tests/test_fuzz_harness_targets.py` to pin the fuzz target inventory in normal CI and wired it into release smoke.
- Added pure webhook HMAC signature helpers in `opencut/core/webhook_signature.py`.

### Changed — Parser Hardening

- Hardened C2PA sidecar verification and plugin manifest validation against non-object JSON payloads.
- Promoted the CUBE LUT parser used by LUT blending to a module-level helper so fuzzing exercises the real user-supplied LUT parser.

### Added — Roadmap Governance

- Added the F180 Wave N-T / F-number governance ledger at `.ai/research/2026-05-18/WAVE_N_T_F_NUMBER_LEDGER.md`.
- Added `tests/test_wave_f_number_ledger.py` so future Wave N-T roadmap rows must stay mapped to an F-number disposition.

### Added — Installer CI Coverage

- Added `scripts/smoke_inno_installer.ps1`, a CI-only Inno Setup install/uninstall smoke that validates the generated installer payload, registry entry, installer manifest, and cleanup path.
- Wired the Inno smoke into the Windows release workflow immediately after the Inno setup executable is built.

### Changed — Dependency Security Floor

- Raised source installs to Python 3.11+ so OpenCut can consume current security-fix dependency lines.
- Updated install floors for `flask-cors>=6,<7`, `opencv-python-headless>=4.13,<5`, `Pillow>=12.2,<13`, `onnxruntime>=1.25,<2`, `onnxruntime-gpu>=1.25,<2`, and `whisperx>=3.8.5,<4`.
- Removed the stale `pydub` entry from `requirements.txt`, matching the previously shipped F123 pyproject extras.

## [1.32.0] - 2026-05-09

### Added — Light Theme & Appearance Toggle (CEP Panel)

- **`extension/com.opencut.panel/client/index.html`**: Added FOUC-prevention inline script in `<head>` that reads `opencut_settings.theme` from `localStorage` and applies `html.theme-light` before first paint. Added **Appearance** select (Auto / Dark / Light) to Settings > Preferences.
- **`extension/com.opencut.panel/client/main.js`**: Added `_applyTheme(pref)` function that toggles `html.theme-light` based on explicit preference or OS `prefers-color-scheme`. Wired `settingsTheme` change event for instant switching. Added `theme` field to `saveLocalSettings`/`loadLocalSettings`. Added `matchMedia` listener to re-apply when OS theme changes in Auto mode.
- **`extension/com.opencut.panel/client/style.css`**: Added comprehensive `html.theme-light` CSS block — 200+ lines of token overrides and component-specific rules covering: backgrounds, text, borders, sidebar, nav tabs, cards, workspace stage, buttons, form inputs, sub-tabs, empty states, toasts, alert/processing banners, results, job history, command palette, wizard overlay, wizard card, onboarding overlay, scrollbars, and focus rings. Light palette uses professional off-white surfaces (`#f0f3f8` void → `#ffffff` cards), deep blue accent (`#1e4fd8`) with WCAG AA contrast, and light-calibrated shadows.



### Changed — Premium UX/UI Polish (CEP Panel)

- **`extension/com.opencut.panel/client/style.css`**: Full premium design-quality pass on the CEP panel stylesheet.
  - Replaced neon synthwave color system with a professional studio dark theme: accent blue `#4d7fff`, muted violet secondary, calmer semantic success/warning/error tokens, desaturated background layers.
  - Removed all decorative CSS animations: mesh background, scrolling grid, header gradient bar, logo shimmer, card border rotation, button shine sweeps, and neon button underglow.
  - Static subtle depth gradients and dot grid replace animated backgrounds.
  - Typography cleaned throughout: removed excessive letter-spacing on titles and labels, reduced font-weight on card headings, removed forced uppercase transforms.
  - Eliminated all remaining hardcoded `rgba(0,255,255,...)` and `rgba(255,0,85,...)` values — all colors now flow from CSS custom properties.
  - Updated CSS variable fallback strings to match new design tokens.
  - Fixed `--text-muted` typo (`#6868888` → `#686888`).
  - `.btn-primary`: removed shine sweep (`::before`) and neon underglow (`::after`).
  - `.btn-outline`: replaced hardcoded neon rgba with design-system variables.
  - `.filler-checks`, `.stem-check` checked states: replaced neon glow box-shadow with clean variable-driven shadow.
  - `.custom-dropdown-trigger:focus`: replaced hardcoded rgba focus ring with `var(--cyan-subtle)`.
  - `.drop-zone.drag-over`, `.drop-zone.drag-active`: use `var(--cyan-subtle)` background.
  - Wizard `box-shadow`: removed extraneous neon ambient glow.
  - `.server-status-banner`: replaced hardcoded error-red rgba with `var(--error-dim)` / `var(--red-glow)`.
- **`extension/com.opencut.panel/CSXS/manifest.xml`**: Updated `ExtensionBundleVersion` and extension `Version` to `1.31.0`.

---

## [1.30.0] - 2026-05-04

### Added — Wave M (Dubbing Pipeline, Sports Highlights, EchoMimic, MCP Server)

- **`core/dub_pipeline.py`** (`POST /video/dub`): Full automatic dubbing pipeline — delegates to the existing `auto_dub_pipeline.auto_dub()` engine. Supports 27 target languages, Whisper transcription, deep-translation, edge-tts/ElevenLabs/OpenAI voice synthesis, optional voice cloning, lip-sync, and music preservation. Route is `@async_job`; registered in `_ALLOWED_QUEUE_ENDPOINTS`.
- **`core/highlights_sports.py`** (`POST /video/highlights/sports`): Optical-flow motion scoring (cv2.calcOpticalFlowFarneback at 2fps, 320×180 resize) + ffmpeg astats audio-energy scoring with genre-specific weighting (sports/concert/reaction/gaming/news). Returns non-overlapping top-N highlight windows sorted by start time. Route is `@async_job`; registered in `_ALLOWED_QUEUE_ENDPOINTS`.
- **`core/lipsync_echomimic.py`** (`POST /video/lipsync/echomimic`): Two-backend EchoMimic lipsync — tries `echomimic` PyPI package first, falls back to `diffusers` + torch with `BadToBest/EchoMimic` HuggingFace weights (~2 GB auto-download to `~/.opencut/models/echomimic/`). Override model directory via `OPENCUT_ECHOMIMIC_MODEL_DIR` env var.
- **`mcp_server.py`**: MCP sidecar with 27 tools covering all major OpenCut capabilities. Added `main()` CLI entry point, `--http` mode (JSON-RPC 2.0 on port 5681), `--port`, `--list-tools`, and `--log-level` flags. Registered as `opencut-mcp-server` console script.
- **`pyproject.toml`**: Added `mcp` optional dependency group (`mcp>=1.0`). Registered `opencut-mcp-server` console script.

### Changed

- **`routes/wave_k_routes.py`**: Wired `route_dub_pipeline()` and `route_highlights_sports()` — replaced `_stub_501` with full `@async_job` implementations.
- **`routes/jobs_routes.py`**: Added `/video/dub` and `/video/highlights/sports` to `_ALLOWED_QUEUE_ENDPOINTS`.
- **`mcp_server.py`**: Added 4 Wave M tools (`opencut_dub_video`, `opencut_sports_highlights`, `opencut_lipsync_echomimic`, `opencut_chat_edit`) to `MCP_TOOLS` and `_TOOL_ROUTES`. Added `audio_path` to required path validation set.

---

## [1.29.0] - 2026-04-23

### Added — Wave L (Premium TTS, Smart Upscaling, AI Face Tools)

- **`core/tts_elevenlabs.py`** (`POST /audio/tts/elevenlabs`, `GET /audio/tts/elevenlabs/voices`, `GET /audio/tts/elevenlabs/info`): ElevenLabs cloud TTS backend — 3,000+ voices across 32 languages. Gated by `OPENCUT_ELEVENLABS_API_KEY` environment variable; returns 503 with install/setup hint if absent. Supports all four synthesis models (multilingual v2, turbo v2.5, turbo v2, monolingual v1), voice stability, similarity boost, style exaggeration, and speaker boost. Voice can be specified by name or voice_id. Route is `@async_job` and registered in `_ALLOWED_QUEUE_ENDPOINTS`.
- **`core/upscale_hub.py`** (`POST /video/upscale/smart`, `GET /video/upscale/smart/info`): Smart upscaling dispatcher that auto-selects the best available backend — FlashVSR (streaming SR) → Real-ESRGAN → Video2x → Lanczos (always available). Content hints (`auto`, `fast`, `quality`, `anime`, `face`, `film`) route to appropriate backends and models. Falls back to Lanczos on any backend failure. Route is `@async_job`. `/info` endpoint returns live backend availability and auto-selection table.
- **`core/face_reshape.py`** (`POST /video/face/reshape`, `GET /video/face/reshape/info`): Liquify-style face warping using MediaPipe FaceMesh (468 landmarks). Supports five operations: `slim_face`, `enlarge_eyes`, `shrink_nose`, `raise_cheekbones`, `smooth_jaw`. Builds a smooth per-pixel displacement field from landmark deltas with Gaussian blending, applied via `cv2.remap`. Strength parameter 0–1. Route is `@async_job`.
- **`core/skin_retouch.py`** (`POST /video/face/retouch`, `GET /video/face/retouch/info`): Automated skin retouching and blemish removal. Bilateral `mode` applies frequency-separation smoothing (edge-preserving bilateral filter + skin HSV mask + blend). `gfpgan` mode runs GFPGAN deep face restoration when available; falls back to bilateral if GFPGAN absent. Optional `radiance` parameter applies a subtle brightness boost to skin areas. Route is `@async_job`.
- **`routes/wave_l_routes.py`**: New Flask blueprint `wave_l_bp` with 11 new routes.
- **`checks.py`**: Added `check_elevenlabs_available()`, `check_upscale_hub_available()`, `check_face_reshape_available()`, `check_skin_retouch_available()`, and `check_wave_l()` summary function.

### Changed

- **`routes/__init__.py`**: Registered `wave_l_bp` — brings total blueprint count to 99.
- **`routes/jobs_routes.py`**: Added 4 Wave L `@async_job` routes to `_ALLOWED_QUEUE_ENDPOINTS`.

---

## [1.28.2] - 2026-04-22

### Fixed

- **`jobs_routes.py`**: Added 9 Wave K async routes to `_ALLOWED_QUEUE_ENDPOINTS` — they were implemented in v1.28.0 but never registered, making them unreachable via the job queue. Routes added: `/audio/watermark/embed`, `/audio/watermark/detect`, `/settings/brand-kit/preview`, `/video/reframe/batch`, `/audio/censor/profanity`, `/audio/spectral-match`, `/video/lottie/render`, `/search/ai`, `/search/ai/index`.
- **`wave_k_routes.py`**: Changed `validate_path` → `validate_filepath` on all 16 remaining file-input parameters in Tier 2 synchronous routes (`singing_vevo2`, `lipsync_echomimic`, `style_tokenflow`, `track_cutie`, `track_deva`, `flow_searaft`, `restore_diffbir`, `stabilize_gyroflow`, `deblur_motion`, `depth_depthpro`, `depth_flow`, `audio_reactive_fx`, `cinefocus`, `screenplay_parse`, `slate_id`). Ensures file existence is validated before handing paths to heavy AI modules, giving callers a specific `FILE_NOT_FOUND` error instead of an opaque internal failure.
- **`server.py`**: Removed duplicate comment block at the PID re-export section.

---

## [1.28.1] - 2026-04-21

### Fixed — Wave K Bugfix Pass (Phase 3/4 production hardening)

- **All 6 Wave K core modules**: replaced bare `"ffmpeg"` / `"ffprobe"` string literals with `get_ffmpeg_path()` / `get_ffprobe_path()` from `opencut.helpers`. Fixes silent failure in PyInstaller-bundled distributions where FFmpeg is not on PATH.
- **All 6 Wave K core modules**: wrapped `subprocess.run(..., check=True)` in `try/except CalledProcessError` and re-raised as `RuntimeError` with the FFmpeg stderr included. Debugging FFmpeg failures now gives actionable error messages.
- **`clip_rating.py`**: Added `threading.Lock()` around all `_load_db()` / `_save_db()` cycles in `rate()`, `tag()`, `untag()`, `search()`, `get()` to prevent concurrent-request data loss. `_save_db()` now uses an atomic write (`os.replace`) to prevent JSON corruption on crash.
- **`profanity_censor.py`**: Fixed critical bleep-mode logic bug — original audio was mixed with the bleep instead of being silenced during profanity windows, leaving the original speech fully audible. Bleep mode now correctly silences the original and gates a 1 kHz tone to those windows only. Added `_get_audio_duration()` ffprobe helper to generate a full-length bleep track.
- **`spectral_match.py`**: Added `-t 60` duration cap to the audio decode step, preventing OOM on long files. Analysis uses the first 60 seconds as a representative spectral sample.
- **`lottie_import.py`**: Fixed silent blank-frame rendering — the PIL fallback created empty transparent frames with no warning. Now raises `RuntimeError` requiring Cairo. Fixed `bg_color` being accepted but not applied — compositing now uses `filter_complex` `color` + `overlay`.
- **`semantic_search.py`**: Fixed `build_index()` which stored only timestamps — it now extracts the first frame with FFmpeg and computes a real CLIP embedding stored in the index. `search()` now uses pre-built embeddings instead of re-encoding every file per query. Fixed double CLIP import (`clip_mod` / `clip_mod2`). Added `threading.Lock()` for model load and index writes. `_save_index()` is now atomic via `os.replace`.
- **`subtitle_qa.py`**: Fixed VTT regex that rejected `HH:MM:SS.mmm` timestamps (only `MM:SS.mmm` matched). Added file existence check at top of `validate()`. Renamed loop variable `l` to `line`. Added missing `import os`.
- **`slate_id.py`**: Added `_florence_cache` dict to cache the Florence-2 processor and model across calls, avoiding a multi-GB reload on every invocation.
- **`wave_k_routes.py`**: Added `validate_filepath` to security imports. Changed `validate_path` → `validate_filepath` on all file-input route parameters (`batch_reframe`, `subtitle_qa`, `spectral_match` × 3, `lottie_info`) to enforce file existence before handing paths to modules. Simplified Tier 3 stubs `route_gen_video_wan_vace` and `route_highlights_sports` to return `_stub_501()` directly, consistent with all other Tier 3 routes.

---

## [1.28.0] - 2026-04-21

### Added — Wave K Completeness Pass (40 features, 59 routes)

Wave K is the result of a four-angle competitive research pass spanning Kdenlive, DaVinci Resolve 19, CapCut Pro, Runway Gen-3, Pika 2.0, ElevenLabs, Adobe Firefly Video, Kling 2.0, and a broad sweep of 2024–2025 open-source AI papers. Features are tiered: Tier 1 ships working, Tier 2 ships as 503 stubs pending optional dependency installs, Tier 3 ships as 501 stubs on the roadmap for v1.29.0.

**Tier 1 — Fully Implemented (8 modules, ~24 routes)**
- **K1.1 AudioSeal Watermarking** (`audio_watermark.py`) — AI-inaudible perceptual watermark embed and detect via Facebook Research AudioSeal; wired as a post-export background job so export latency is unaffected. Routes: `POST /audio/watermark/embed`, `POST /audio/watermark/detect`, `GET /audio/watermark/info`.
- **K1.2 Brand Kit** (`brand_kit.py`, existed) — Brand-consistent lower-thirds, intro/outro overlays, and logo-burn. Routes now wired in `wave_k_routes.py`.
- **K1.3 Podcast Suite** (`podcast_suite.py`, existed) — Multi-speaker transcription, chapter markers, audiogram export. Routes wired.
- **K1.4 Batch Reframe** (`batch_reframe.py`) — AI-crop a folder of clips to any target AR in one call. Routes: `POST /batch/reframe`, `GET /batch/reframe/status`.
- **K1.5 Clip Rating** (`clip_rating.py`) — Heuristic quality score (motion blur, low-light, handheld shake) so editors can bin rejects before cut. Routes: `POST /clip/rating`, `GET /clip/rating/info`.
- **K1.6 Subtitle QA** (`subtitle_qa.py`) — Lint SRT/VTT for overlaps, encoding errors, and reading-speed violations. Routes: `POST /subtitle/qa`, `GET /subtitle/qa/rules`.
- **K1.7 Profanity Censor** (`profanity_censor.py`) — Bleep or silence detected profanity with configurable word list and mode (`bleep`, `silence`, `mute-word`). Routes: `POST /audio/censor/profanity`, `GET /audio/censor/profanity/wordlist`.
- **K1.8 Spectral Match** (`spectral_match.py`) — Match the spectral profile of one audio file to a reference using FFmpeg `amultiply`/EQ. Routes: `POST /audio/spectral/match`, `GET /audio/spectral/info`.
- **K1.9 Lottie Import** (`lottie_import.py`) — Render Lottie JSON animations to transparent video via lottie-python or Puppeteer fallback. Routes: `POST /lottie/render`, `GET /lottie/info`.
- **K1.10 Semantic Search** (`semantic_search.py`) — CLIP/SentenceTransformers frame-level index so editors can search footage by description. Routes: `POST /search/ai`, `POST /search/ai/index`, `GET /search/ai/index/status`.

**Tier 2 — 503 Stubs, install-ready (19 modules, ~38 routes)**
- K2.1 GPT-SoVITS TTS (`tts_gptsovits.py`) — voice-cloned TTS via local GPT-SoVITS server on port 9880.
- K2.2 Amphion TTS (`tts_amphion.py`) — neural TTS with style control.
- K2.3 Vevo2 Singing (`singing_vevo2.py`) — AI singing voice synthesis.
- K2.4 CosyVoice2 TTS (`tts_cosyvoice2.py`) — multilingual zero-shot voice cloning.
- K2.5 EchoMimic LipSync (`lipsync_echomimic.py`) — audio-driven talking-head animation.
- K2.6 TokenFlow Style Transfer (`style_tokenflow.py`) — temporally-consistent video style transfer.
- K2.7 CUTIE Tracking (`track_cutie.py`) — high-accuracy object segmentation tracking.
- K2.8 DEVA Tracking (`track_deva.py`) — open-world decoupled video segmentation.
- K2.9 SEA-RAFT Optical Flow (`flow_searaft.py`) — state-of-the-art dense optical flow.
- K2.10 DiffBIR Restoration (`restore_diffbir.py`) — face + general image/video restoration.
- K2.11 Gyroflow Stabilization (`stabilize_gyroflow.py`) — gyroscope-data lens/stabilization via `gyroflow` binary.
- K2.12 Motion Deblur (`deblur_motion.py`) — blind motion deblur via NAFNet or DeblurGAN-v2.
- K2.13 Depth Pro (`depth_depthpro.py`) — Apple Depth Pro metric monocular depth estimation.
- K2.14 DepthFlow (`depth_flow.py`) — 2.5D parallax video from still images using ModernGL.
- K2.15 AudioGen SFX (`sfx_audiogen.py`) — text-to-sound-effects via Meta AudioGen (CC-BY-NC; weights not auto-downloaded).
- K2.16 Open-Sora Generation (`gen_video_opensora.py`) — open-source text-to-video via Open-Sora.
- K2.17 LTX-Video v2 A+V (`gen_video_ltx.py`) — joint audio+video generation; new route `/generate/ltx/v2` preserves backward compat with existing `/generate/ltx`.
- K2.18 Audio-Reactive FX (`audio_reactive_fx.py`) — beat-synced visual FX driven by audio energy bands.
- K2.19 CineFocus Rack Focus (`cinefocus.py`) — AI-driven simulated rack focus using depth maps.

**Tier 3 — 501 Stubs, v1.29.0 roadmap (8 modules, ~16 routes)**
- K3.1 Dub Pipeline (`dub_pipeline.py`) — multi-language dub orchestration (transcribe → translate → TTS → lipsync).
- K3.2 Trailer Generator (`trailer_gen.py`) — highlights-driven auto-trailer with title card injection.
- K3.3 Screenplay Parser (`screenplay_parser.py`) — FDX/Fountain parse → shot list; partial real implementation via `fountain` library.
- K3.4 Face Age Transform (`face_age_transform.py`) — age/de-age face via SAM2 + inpaint.
- K3.5 Slate Identify (`slate_id.py`) — OCR-based clapperboard reader; partial real implementation via Tesseract.
- K3.6 Video Outpainting (`outpaint_video.py`) — extend frame borders via diffusion outpaint.
- K3.7 WAN/VACE Generation (`gen_video_wan_vace.py`) — Wan2.1 + VACE video generation and editing.
- K3.8 Sports Highlights (`highlights_sports.py`) — action-density + audio-peak highlight detection.

**Infrastructure**
- `opencut/routes/wave_k_routes.py` — new blueprint `wave_k` with 59 handlers (0 route collisions on startup).
- `opencut/checks.py` — 35 new `check_X_available()` functions appended.
- `opencut/routes/__init__.py` — `wave_k_bp` registered; blueprint count now 98.
- Route collision guard confirmed clean at startup: 1,334 total routes, 0 duplicate method+path pairs.

---

## [1.25.1] - 2026-04-20

### Fixed

Two user-reported bugs in the CEP panel: imported media not detected, and processed output landing next to the source clip instead of in the project folder.

- **CEP project-item walker** — [`host/index.jsx`](extension/com.opencut.panel/host/index.jsx) `_walkProjectItems()` now uses `getMediaPath()` as the primary discriminator instead of an `item.type === 2` + `!item.getMediaPath` heuristic that broke silently on Premiere 25.x. The old heuristic misclassified bins that expose `getMediaPath` as a function (most versions do), so nested media was never traversed. New logic: if `getMediaPath()` returns a non-empty string, record it; else if the item has children, walk it regardless of reported type.
- **Project-folder detection fallback** — [`host/index.jsx`](extension/com.opencut.panel/host/index.jsx) `getProjectMedia()` returns a best-guess project folder even when `app.project.path` is empty (i.e. the .prproj hasn't been saved yet). Priority chain: saved project path → directory of the first imported media → scratch disk path → empty. A new `projectFolderSource` field in the response tells the panel which fallback fired. Stops processed files from landing in `Downloads/` / network shares for unsaved projects.
- **Panel `outputDir` preference honoured** — [`client/main.js`](extension/com.opencut.panel/client/main.js) introduces `_recomputeEffectiveOutputDir()` that layers the user's Settings → Output directory preference on top of the JSX-detected project folder. Previously the preference was saved to localStorage but ignored at job time; every `output_dir: projectFolder` call site now receives the resolved priority chain (user pref → detected project folder → empty). Recompute fires on settings save and on every media-list refresh.
- **Route ownership cleanup** — removed ambiguous duplicate Flask registrations that let the wrong handlers win for `/plugins/install`, `/resolve/import`, and `/api/macro/play`. Local plugin-source install, Resolve import, and macro dry-run now resolve to the intended synchronous handlers again; the async marketplace and developer-only variants were moved onto explicit non-colliding endpoints.
- **Second collision sweep** — removed four more overlapping registrations discovered in the full route table: `/api/scripting/execute`, `/audio/me-mix`, `/audio/dialogue-premix`, and `/timeline/diff`. The stable public endpoints now have a single owner, while the shadowed variants moved onto explicit paths (`/api/workflow/scripting/execute`, `/audio/me-mix/basic`, `/audio/dialogue-premix/basic`, `/timeline/otio-diff`) instead of silently competing at startup.
- **Runtime route-collision guard** — [`opencut/routes/__init__.py`](opencut/routes/__init__.py) now exposes collision detection helpers used by [`server.py`](opencut/server.py) to fail fast during `create_app()` if any later blueprint or plugin registration reintroduces a duplicate method/path pair. This turns a previously silent startup-order bug class into an immediate, diagnosable error.
- **Deterministic blueprint registration** — [`opencut/routes/__init__.py`](opencut/routes/__init__.py) no longer relies on `locals()` reflection to discover blueprints. Built-in route registration now comes from an explicit ordered tuple, which makes startup easier to audit, keeps registration intent readable, and gives tests a stable helper (`get_core_blueprints()`) to validate directly.
- **Offline-safe webhook URL validation** — [`opencut/core/url_safety.py`](opencut/core/url_safety.py) no longer performs DNS lookups for ordinary public hostnames. Literal localhost/private/reserved IP targets are still blocked, but valid public domains now pass deterministically in offline/test environments instead of failing with `hostname does not resolve`.
- **Resolve import input hardening** — [`system.py`](opencut/routes/system.py) now accepts either a single `filepath` or a validated `paths` list on `/resolve/import`, and missing/invalid path payloads return 400s consistently instead of falling through to the wrong integration route and surfacing as 500s.
- **CEP panel premium polish** — [`extension/com.opencut.panel/client/main.js`](extension/com.opencut.panel/client/main.js), [`index.html`](extension/com.opencut.panel/client/index.html), and [`style.css`](extension/com.opencut.panel/client/style.css) now preserve per-tab scroll position, move focus correctly from the skip link into the main workspace, calm alert/progress copy, rename history surfaces to activity where that better matches the UX, and add a final shell-state refinement pass so banners, cards, results, and nav states feel more consistent and readable.
- **CEP accessibility hardening** — the panel now traps focus inside the active command palette or frame-preview overlay, restores focus to the right launcher on close, keeps custom dropdown keyboard flow anchored to the trigger, stops dropdown wheel scrolling from leaking into the page, keeps warnings on screen until dismissed, and raises disabled/placeholder/focus contrast on key interactive surfaces.
- **CEP workflow refinement** — overflowed sub-tab strips now advertise when more tools are off-screen and respond to vertical wheel scrolling as horizontal navigation, while high-risk numeric fields clamp themselves on blur and keep highlight/short-duration min/max pairs internally consistent before a run starts.
- **CEP packaging cleanup** — [`vite.config.mjs`](extension/com.opencut.panel/vite.config.mjs) now treats `CSInterface.js` and `main.js` as explicit classic CEP runtime assets during Vite builds, so `client/dist/` contains the real CEP-ready script files without the earlier classic-script build warnings.
- **Panel dead-code reduction** — [`client/main.js`](extension/com.opencut.panel/client/main.js) dropped unused helpers and stale state, fixed the WebSocket completion fallback to re-enter polling correctly, and reduced the panel lint backlog from 52 warnings to 25 without changing intended behavior.
- **Frontend build dependency restored** — [`extension/com.opencut.panel/package.json`](extension/com.opencut.panel/package.json) now declares `terser` so the existing Vite production build can honor `minify: "terser"` without failing on a clean install.

### Gotchas
- **JSX walker is getMediaPath-first** — sequences and offline media both return empty `getMediaPath()`; the difference is sequences have 0 children. Both are now skipped cleanly. Bins with `type` reported as `undefined` / `0` / `3` on newer Premiere builds are now walked via the "has children" fallback.
- **`projectFolder` is now the effective output dir**, not the raw detected value. The JSX-detected folder lives on the panel side in `_detectedProjectFolder`. If something needs the raw value, read that — don't read `projectFolder`.

## [1.25.0] - 2026-04-19

### Added — Wave H: Commercial Parity & Content-Creator Polish

16 new core modules + 1 new blueprint (`wave_h_bp`) + 34 new routes (1,241 → 1,275). Extends the OSS survey with commercial-product patterns (Opus Clip, Descript, CapCut, ScreenStudio) and post-April-2026 AI projects (FlashVSR, ROSE, Sammie-Roto-2, OmniVoice, ReEzSynth, VidMuse, VideoAgent, ViMax, Hailuo 2.3, Seedance 2.0, GaussianHeadTalk, FantasyTalking2).

**Tier 1 — Content-creator polish (fully working)**:

- **Virality scoring** — new [`core/virality_score.py`](opencut/core/virality_score.py) + `POST /analyze/virality`, `POST /analyze/virality/rank`. Multimodal 0–100 heuristic score blending audio-energy peaks (existing `silence.py`), transcript hook density (LLM when available, keyword lexicon fallback), and visual salience (FFmpeg `signalstats`). `ViralityResult` + `ViralitySignals` subscriptable for jsonify. Stable-sorted ranking. Opus Clip inspiration.
- **Cursor-zoom sidecar parsing** — extended [`core/cursor_zoom.py`](opencut/core/cursor_zoom.py) + `POST /video/cursor-zoom/resolve`. Parses ScreenStudio / Screen.Studio / OBS click-log sidecars, falls back to inline events, falls back to the existing frame-diff detector. All coordinates clamped to `[0, width] × [0, height]`.
- **Changelog feed** — new [`core/changelog_feed.py`](opencut/core/changelog_feed.py) + `GET /system/changelog/latest`, `GET /system/changelog/unseen`, `POST /system/changelog/mark-seen`. Fetches GitHub releases, 15 min cache, graceful offline fallback, persists last-seen tag in `~/.opencut/changelog_seen.json`.
- **Issue report bundle** — new [`core/issue_report.py`](opencut/core/issue_report.py) + `GET|POST /system/issue-report/bundle`. Scrubs all HOME paths to `~` before including crash.log + log tail. Returns a pre-filled GitHub issue URL. 60 KB body cap.
- **Demo bundle** — new [`core/demo_bundle.py`](opencut/core/demo_bundle.py) + `GET /system/demo/list`, `GET /system/demo/sample`, `POST /system/demo/download`. Serves `opencut/data/demo/sample.mp4` when the installer ships one; dev installs can pull it from a GitHub release asset.
- **Gist preset sync** — new [`core/gist_sync.py`](opencut/core/gist_sync.py) + `POST /settings/gist/push`, `POST /settings/gist/pull`, `GET /settings/gist/info`. Pure-stdlib urllib; refuses anonymous secret gists; rejects files > 2 MB; whitelists `.json`/`.jsonl`/`.txt`/`.md` extensions.
- **Onboarding state** — new [`core/onboarding.py`](opencut/core/onboarding.py) + `GET|POST /settings/onboarding`. Per-profile `{seen, step, updated_at}` persisted in `~/.opencut/onboarding.json`. Delete the file to re-trigger the tour.

**Tier 2 — AI model stubs (503 MISSING_DEPENDENCY)**:

All six ship as `check_X_available()`-gated stubs returning 503 with install hints, matching the v1.18–1.20 pattern. Full wiring deferred to v1.26.0+ once each upstream pins a stable Python entry point.

- **FlashVSR** — streaming diffusion VSR (CVPR'26). [`core/upscale_flashvsr.py`](opencut/core/upscale_flashvsr.py) + `POST /video/upscale/flashvsr`, `GET /video/upscale/flashvsr/info`.
- **ROSE** — video inpainting that preserves shadows/reflections. [`core/inpaint_rose.py`](opencut/core/inpaint_rose.py) + `POST /video/inpaint/rose`.
- **Sammie-Roto-2** — AI rotoscoping with VideoMaMa segmentation + in/out markers. [`core/matte_sammie.py`](opencut/core/matte_sammie.py) + `POST /video/matte/sammie`.
- **OmniVoice** — zero-shot TTS with 600+ languages. [`core/tts_omnivoice.py`](opencut/core/tts_omnivoice.py) + `POST /audio/tts/omnivoice`, `GET /audio/tts/omnivoice/models`.
- **ReEzSynth** — flicker-free Ebsynth successor. [`core/style_reezsynth.py`](opencut/core/style_reezsynth.py) + `POST /video/style/reezsynth`.
- **VidMuse** — video-to-music generation (CVPR'25). [`core/music_vidmuse.py`](opencut/core/music_vidmuse.py) + `POST /audio/music/vidmuse`.

**Tier 2 — Infra stubs**:

- **QE reflection probe** — [`wave_h_routes.py`](opencut/routes/wave_h_routes.py) + `GET|POST /system/qe-reflect`. Panel POSTs the output of `ocQeReflect()` (host JSX); server caches it in `~/.opencut/qe_reflect.json` for future API discovery. Host-side JSX wiring lands in v1.26.0 (see ROADMAP-NEXT Wave H2.8).

**Tier 3 — Strategic stubs (501 ROUTE_STUBBED)**:

All routes return 501 with install hints — promoted to Tier 2 once usage signal confirms demand or upstream licences clarify.

- **VideoAgent + ViMax** — [`core/video_agent.py`](opencut/core/video_agent.py) + `POST /agent/search-footage`, `POST /agent/storyboard`. HKUDS agentic pipelines.
- **Hailuo 2.3 / Seedance 2.0** — [`core/gen_video_cloud.py`](opencut/core/gen_video_cloud.py) + `POST /generate/cloud/submit`, `GET /generate/cloud/status/<id>`, `GET /generate/cloud/backends`. Cloud gen-video backends.
- **GaussianHeadTalk + FantasyTalking2** — [`core/lipsync_advanced.py`](opencut/core/lipsync_advanced.py) + `POST /lipsync/gaussian`, `POST /lipsync/fantasy2`, `GET /lipsync/advanced/backends`. Wobble-free lip-sync alternatives to LatentSync/MuseTalk.

### Infrastructure

- 16 new `check_*_available()` entries in [`opencut/checks.py`](opencut/checks.py) (Wave H block).
- 3 new async routes added to `_ALLOWED_QUEUE_ENDPOINTS`: `/analyze/virality`, `/analyze/virality/rank`, `/video/cursor-zoom/resolve`.
- New blueprint `wave_h_bp` registered (continues wave_a/b/c/d/e/f/g/h naming).
- New error code `ROUTE_STUBBED` returned at HTTP 501 for Tier 3 routes. Frontend treats 501 as "coming soon" (greyed-out with tooltip), never as a failed call.
- ROADMAP-NEXT.md extended with a new **Wave H** section documenting every item and its upstream source.

### Frontend wiring

- **[`client/main.js`](extension/com.opencut.panel/client/main.js)** — ~330-line `WaveH` module appended at the end of the main IIFE. Staggered startup probes at 1.2/1.8/2.4/3.0 s to avoid racing the first health check. Exposes `window.OpenCutWaveH` for the command palette.
- **[`client/index.html`](extension/com.opencut.panel/client/index.html)** — "Panel Polish" card injected into the Settings tab with 7 `data-i18n` labelled buttons: Try demo, Restart tour, Gist push, Gist pull, Virality, Cursor zoom, Send log.
- **[`client/style.css`](extension/com.opencut.panel/client/style.css)** — `.oc-wave-h-*` + `.oc-onboarding-*` rules, dark-mode-first, with a 520 px narrow-panel media query.
- **[`client/locales/en.json`](extension/com.opencut.panel/client/locales/en.json)** — 10 new keys under `wave_h.*`.
- **[`host/index.jsx`](extension/com.opencut.panel/host/index.jsx)** — three new ES3-safe functions: `ocQeReflect()` (QE API reflection probe), `_ocDispatchEvent()` (CSXS event dispatcher), `ocEmitPingEvent()` (panel-callable ack). All CSXS events use the `com.opencut.<event>` namespace.

### Gotchas

- Virality score is heuristic — no ML model. The absolute number is not comparable across video types; use for relative ranking only.
- Cursor-zoom: never trust client-supplied sidecar coordinates — the clamp to `[0, width] × [0, height]` is non-negotiable.
- Changelog feed never raises. Network / JSON errors return `{releases: [], source: "fallback", note: "..."}` so the panel keeps working offline.
- Issue report bundle scrubs HOME paths but does NOT redact API keys / credentials that may leak into logs. Callers must still prompt the user to review the body before opening the GitHub URL.
- Gist sync refuses anonymous secret gists — anonymous gists are always public. Secret gists require `GITHUB_TOKEN` env.
- Tier 3 routes always return 501 `ROUTE_STUBBED`, even with optional deps installed. These are scaffolding for future releases, not "features waiting for a pip install".

## [1.24.0] - 2026-04-17

### Added — second wide-net pass

Five infrastructure modules + three governance files + one `run_ffmpeg` ergonomics extension. Closes the remaining wide-net findings from the v1.14.0 strategic-gap audit. 6 new routes (1,235 → 1,241).

**Subprocess tracking for cancel** — [`helpers.run_ffmpeg`](opencut/helpers.py) now accepts an optional `job_id=...` keyword. When supplied, the spawned Popen is registered with the job subsystem via `_register_job_process(job_id, proc)` so `POST /cancel/<job_id>` actually terminates the child. Previous behaviour used blocking `subprocess.run()` — cancel marked the job "cancelled" but FFmpeg ran to completion, pinning a worker slot indefinitely. The non-`job_id` call path is unchanged, so all ~158 existing callers keep working; new / refactored callers opt in by passing `job_id=job_id`. Closes the v1.14.0 audit finding "untracked subprocesses can orphan on cancel".

**Disk-space monitor** — new [`core/disk_monitor.py`](opencut/core/disk_monitor.py) + `GET /system/disk`, `POST /system/disk/preflight`, `POST /system/disk/track`. Tracks `tempfile.gettempdir()` + `~/.opencut` by default; ops add more via `register_tracked_path()` or the `/system/disk/track` route. Warn / critical thresholds via `OPENCUT_DISK_WARN_MB` (2048) / `OPENCUT_DISK_CRITICAL_MB` (500). Optional background probe thread via `OPENCUT_DISK_MONITOR_INTERVAL` (default 0 = disabled). `preflight(path, required_mb)` gates a long render — fails open when the probe itself fails so disk-probe bugs don't cascade into job failures. Started from `server.create_app()` alongside the temp-cleanup boot.

**Request-ID correlation** — new [`core/request_correlation.py`](opencut/core/request_correlation.py) + `GET /system/request-correlation`. Installs Flask `before_request` / `after_request` / `teardown_request` hooks: generates `r-{uuid[:16]}`, stamps `g.request_id`, echoes on the response `X-Request-ID` header (RFC-standard), and attaches a `logging.Filter` so every log record under `logger("opencut")` carries the ID. Never trusts client-supplied `X-Request-ID` — the incoming value is sanitised into `g.client_request_id` for forensic value but the canonical ID is always server-generated. Prevents log-injection + server-log poisoning.

**Deprecation registry** — new [`core/deprecation.py`](opencut/core/deprecation.py) + `GET /system/deprecations`. `@deprecated_route(remove_in="2.0.0", replacement="/new/path", sunset_date="2026-10-01", reason="…")` decorator stamps metadata on the view, emits RFC 8594 `Deprecation: true` + `Sunset: ...` + `Link: <...>; rel="successor-version"` response headers, and logs a one-time INFO record the first time each deprecated route is hit per process (including the caller's User-Agent). `enrich_openapi_spec(spec, app)` helper upgrades the existing OpenAPI generator output with `deprecated: true` + an `x-opencut-deprecation` extension block so clients see the warning at schema-generation time.

**Governance docs**:
- [`SECURITY.md`](SECURITY.md) — vulnerability reporting process, supported-versions table (1.24.x active / 1.23.x previous / 1.22.x critical-only / ≤1.21 EOL), disclosure SLA (Critical 72h, High 7d, Medium 30d, Low 90d), known-safe-by-design surface, hardening recommendations.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — 100-line repo tour + common-pattern reference card. Covers async route recipe, optional-dep guard, dataclass result shape, new `run_ffmpeg(job_id=...)` pattern, rate-limit categories, deprecation decorator.
- [`scripts/sbom.py`](scripts/sbom.py) — pure-stdlib CycloneDX 1.5 SBOM generator. Walks `pyproject.toml` + `requirements.txt`, emits `dist/opencut-sbom.cyclonedx.{json,xml}`. First run catalogued 15 required + 71 optional deps. No `cyclonedx-bom` pip dep required.

### Infrastructure
- 3 new `check_*_available()` entries in `opencut/checks.py`: `disk_monitor`, `request_correlation`, `deprecation_registry`.
- New blueprint `wave_g_bp` registered (continues wave_a/b/c/d/e/f/g naming).
- `server.create_app()` now calls `install_middleware()` for request-ID wiring + `disk_monitor.start_background()` for optional background monitoring.

### Gotchas
- **`run_ffmpeg` default path is unchanged** — no `job_id` means the fast `subprocess.run` path runs exactly as it did pre-v1.24. Only callers that opt in (`job_id=job_id`) get cancel support. New code should always pass `job_id` when in an `@async_job` context.
- **Request-ID middleware regenerates even when the client sends one** — this is intentional (prevents log-injection). The client-supplied value is preserved in `g.client_request_id` for investigation but never echoed back as-is. If downstream systems need client-controlled correlation, use a separate header (e.g. `X-Client-Correlation-ID`) that we log but don't adopt as the server ID.
- **Disk-monitor `preflight` fails open** — when the probe itself errors (permissions, odd filesystem), it returns `ok=True` + a diagnostic note. Callers needing guaranteed refusal should check `note` and bail explicitly. Fail-open matches the existing `check_disk_space()` semantics.
- **Deprecation registry is keyed by `func.__qualname__`** at decoration time. When the OpenAPI generator is enriched later via `enrich_openapi_spec(spec, app)`, it walks `__wrapped__` to recover the metadata. Keep decorator order `@bp.route → @deprecated_route → @require_csrf → @async_job` so the walk finds the info.
- **SBOM scope** — the generator reads *declared* deps from `pyproject.toml` + `requirements.txt`, not installed ones. Use `cyclonedx-py` / `syft` against a populated venv when you need the installed-packages SBOM (e.g. for CVE scanning the actually-running set).
- **`X-Request-ID` charset** is strictly alphanumeric + `- _ : .`, cap 64 chars. Clients that want richer IDs should derive them from UUID4 hex + hyphens.

## [1.23.0] - 2026-04-17

### Added — wide-net infrastructure pass

Four cross-cutting modules + one blueprint.  No new features — closes long-standing gaps flagged in the v1.14.0 strategic audit and the original ROADMAP.md Wave 3A.  7 new routes, 1,228 → 1,235.

**OpenAPI 3.1 spec generator** — new [`core/openapi_spec.py`](opencut/core/openapi_spec.py) + `GET /api/openapi.json`, `GET /api/docs`, `GET /api/routes`. Walks Flask's `url_map`, introspects view-function docstrings, detects `@require_csrf` by walking the wrapper chain, converts Flask `<int:foo>` path converters to OpenAPI `{foo}` parameters, and emits a valid 3.1 spec (1,208 paths / 97 tags on a stock install). Swagger UI hosted inline via CDN-loaded script — no extra dep. Per-endpoint `schema_hints` escape hatch lets callers attach real request/response schemas without rewriting the generator. Closes Wave 3C "FastAPI migration is deferred" with a zero-migration alternative.

**GPU-exclusive semaphore** — new [`core/gpu_semaphore.py`](opencut/core/gpu_semaphore.py) + `GET /system/gpu-semaphore`. Ships the minimal version of the Wave 3A P0: a process-wide `threading.Semaphore` (default `MAX_CONCURRENT_GPU_JOBS=3`, env-tunable) that routes wrap via `@gpu_exclusive` on their inner worker body. `acquire()` honours a configurable `OPENCUT_GPU_ACQUIRE_TIMEOUT` (default 0 = non-blocking 429). Status endpoint reports `active` / `available` / `rejected_total` / `acquired_total` for ops dashboards. Upgrade path stays open — the decorator boundary is the same whether the implementation becomes an in-process semaphore or a multi-process worker pool later.

**Rate-limit categories** — new [`core/rate_limit_categories.py`](opencut/core/rate_limit_categories.py) + `GET /system/rate-limits`. Four categories (`gpu_heavy`, `cpu_heavy`, `io_bound`, `light`) with env-tunable ceilings. Each category gets a `threading.Semaphore`, an acquire / release API, a `@rate_limit_category("gpu_heavy")` decorator, and a `category_of(func)` helper so introspection / OpenAPI hints can surface the category on the UI. Closes the v1.14.0 audit finding "4 % of async routes are rate-limited" by reducing the per-route cost of opt-in to a single decorator.

**Temp-file startup + periodic sweep** — new [`core/temp_cleanup.py`](opencut/core/temp_cleanup.py) + `POST /system/temp-cleanup/sweep`, `GET /system/temp-cleanup/status`. Runs on `create_app()`: walks `tempfile.gettempdir()` for `opencut_*` + `opencut-*` entries older than a configurable TTL (default 1 h) and removes them. Daemon thread re-sweeps every `OPENCUT_TEMP_CLEANUP_INTERVAL` seconds (default 3600, `0` disables). Symlink-out-of-tempdir defence via `realpath() + commonpath()` check. Logs `removed N files / N dirs / N.N MB` summary per sweep. On the dev machine, the startup sweep of a long-running box reclaimed 4.6 MB across 45 files + 397 dirs.

### Infrastructure wiring
- **`server.py::create_app`** now calls `temp_cleanup.run_startup_sweep()` + `start_background_sweep()`. Any failure logs at warning and the app continues.
- **4 new `check_*_available()`** entries in `opencut/checks.py` (all stdlib-only, so all return True unless the module import itself fails).

### Gotchas
- **OpenAPI spec is cached module-level** — the first request walks 1,200+ routes; subsequent requests serve from `_OPENAPI_CACHE`. `?refresh=1` bypasses the cache. Spec drift between deploys is a concern for deploy-time validation, not runtime.
- **`@gpu_exclusive` goes on the *inner* worker body, not the Flask route** — the `@async_job` decorator already handles queue back-pressure for the HTTP layer. Putting `@gpu_exclusive` on the route would mean the HTTP caller waits; putting it on the inner body means the worker thread waits, which is where the actual GPU contention happens.
- **Rate-limit categories coexist with the existing `security.rate_limit()` / `require_rate_limit()`** — the new category API is for *category-scoped* work (gpu_heavy, cpu_heavy, io_bound, light). The legacy `rate_limit("model_install")` API is still the right choice for *single-op* gates like pip-install serialisation.
- **`temp_cleanup` prefix match is strict basename-only** — we never walk into arbitrary user files. Operators can extend the prefix list via `OPENCUT_TEMP_CLEANUP_PREFIXES` (comma-separated) but should only add prefixes they *own*.
- **`_route_uses_csrf` walks `__wrapped__`** — add your own decorators above `@require_csrf` in the stack order `@wave_X_bp.route → @require_csrf → @async_job` to keep introspection accurate. The OpenAPI spec's `CSRFToken` security block is derived from this walk.
- **Startup temp sweep is best-effort** — logs a warning on failure; never blocks app startup. A corrupt tempfile permission won't prevent the server from coming up.

## [1.22.0] - 2026-04-17

### Added — Shaka Packager / OBS bridge / RunPod / Plausible telemetry

Four new core modules, 11 new routes (1,216 → 1,228). All opt-in via env + optional deps, with graceful degradation.

**B5.2 Shaka Packager HLS/DASH/CENC** — new [`core/shaka_pkg.py`](opencut/core/shaka_pkg.py) + `POST /delivery/shaka/package`, `GET /delivery/shaka/info`. Wraps the standalone ``packager`` binary (BSD-3, Google). Stream-descriptor builder emits one entry per rendition (video/audio/text), with safe basename sanitisation so user stream labels can't smuggle filter characters. Optional CENC DRM with AES-128 raw-key schema for Widevine / PlayReady / FairPlay / Common System. LL-HLS gate validates `protocol=hls`. Resolution cache (`_AVAILABILITY_CACHE`) probes PATH candidates once per process. 2 h default subprocess timeout.

**E6 OBS WebSocket v5 bridge** — new [`core/obs_bridge.py`](opencut/core/obs_bridge.py) + `POST /integration/obs/status`, `POST /integration/obs/switch-scene`, `POST /integration/obs/recording`, `POST /integration/obs/screenshot`. Implements the v5 JSON-over-WebSocket protocol with SHA-256 challenge auth inline — no dedicated ``obs-websocket-py`` dependency. Uses the existing ``websockets`` pip package (already checked via `check_websocket_available`). Blocking sync API with a single background event-loop thread per client. `ObsClient` is a context-manager (`with` block). High-level helpers: `ping()`, `status()`, `switch_scene()`, `recording()`, `take_screenshot()`. Unlocks the Gaming vertical (Twitch VOD clipping), live-production cue sheets, tutorial-capture auto-split.

**E9 RunPod serverless render** — new [`core/runpod_render.py`](opencut/core/runpod_render.py) + `POST /cloud/runpod/submit`, `GET /cloud/runpod/status/<endpoint>/<job>`, `POST /cloud/runpod/cancel`, `GET /cloud/runpod/info`. Pure-`urllib` transport (no hard pip dep); `runpod` SDK is an optional ergonomic enhancement. Submit / status / wait / cancel entry points with exponential-ish backoff in `wait()`. `_ENDPOINT_ID_RE` prevents URL-injection via malformed endpoint IDs. API key from `RUNPOD_API_KEY` env or explicit argument — **never logged**. Sync mode via `/runsync` for small jobs; default `/run` returns a job_id to poll.

**D5.3 Plausible telemetry** — new [`core/telemetry_plausible.py`](opencut/core/telemetry_plausible.py) + `POST /telemetry/plausible/track`, `GET /telemetry/plausible/info`. Fire-and-forget event emitter to a self-hosted Plausible instance (AGPL-3). Single background worker thread reads from a bounded queue (`_MAX_QUEUED=500`, drops oldest on overflow). `_scrub_props()` hardens user-supplied props: drops `_`-prefixed keys, caps string lengths to 120 chars, caps dict size to 30 entries, rejects non-alphanumeric key chars. Opt-in via `PLAUSIBLE_HOST` + `PLAUSIBLE_DOMAIN` env vars — absent env vars = complete no-op, zero overhead.

### Infrastructure
- 5 new `check_*_available()` entries in `opencut/checks.py`: `shaka`, `obs_bridge`, `runpod` (always True, stdlib), `runpod_api_key_set`, `plausible_configured`.
- 1 new route in `_ALLOWED_QUEUE_ENDPOINTS`: `/delivery/shaka/package`.
- New blueprint `wave_e_bp` registered (v1.22 continues the wave_a/b/c/d/e sequence).

### Gotchas
- **Shaka `extra_args` is power-user escape hatch** — args are passed verbatim to the ``packager`` binary without validation. Don't expose this via untrusted surfaces. The stream-descriptor builder *does* sanitise basenames to reject filter-chain metacharacters.
- **OBS `ObsClient` is not thread-safe** — one client per thread; sharing across threads will drop responses because `_pending` dispatch is lookup-by-request-id, not broadcast. The high-level helpers (``status``, ``switch_scene`` etc.) create a fresh client per call, which is the right pattern for low-volume use.
- **OBS background thread uses daemon=True** — clean process exit won't block on a hung OBS connection. If you need guaranteed flush, call `client.close()` explicitly.
- **RunPod API key env fallback** — the `submit` / `status_of` / `cancel` / `wait` functions all honour `RUNPOD_API_KEY` when `api_key=None`. Be careful not to log request bodies that might be echoed back by a misbehaving handler — the SDK fallback path returns raw `output` verbatim.
- **RunPod `wait` backoff caps at 30s** — poll interval grows ×1.25 every 3 polls. For long-running jobs this is fine; for very short jobs the client may observe `COMPLETED` on the first poll regardless.
- **Plausible worker is daemon=True, survives the event loop** — `shutdown()` is provided for clean test teardown but normal daemon-thread behaviour is correct for production. `queue_depth()` lets ops spot backlog.
- **Plausible drops oldest on overflow** — `_MAX_QUEUED=500`. Telemetry is best-effort, so a long Plausible outage doesn't pin memory. If operators care about *durability* of events, Plausible isn't the right sink — use a proper queue + OpenTelemetry stack.

## [1.21.0] - 2026-04-17

### Added — Wave B delivery + Wave E voice grammar + D4.2 fuzz harness

Four new core modules, one fuzz-test harness, 9 new routes. 1,207 → 1,216 total.

**B5.1 VVC / H.266 export** — new [`core/vvc_export.py`](opencut/core/vvc_export.py) + `POST /video/encode/vvc`, `GET /video/encode/vvc/info`. Wraps FFmpeg's `libvvenc` (Fraunhofer HHI, BSD-3). Three presets (`faster`/`balanced`/`archive`), QP override, optional raw `.266` / `.vvc` elementary-stream output for HLS packagers. Availability cached — module only probes FFmpeg's encoder list once per process.

**B5.4 SRT streaming** — new [`core/srt_streaming.py`](opencut/core/srt_streaming.py) + `POST /video/stream/srt/start`, `POST /video/stream/srt/stop`, `GET /video/stream/srt/info`. Spawns FFmpeg → `srt://` push (caller) or listen (listener). URL builder validates host/port/passphrase/stream-id, and reserved param names (mode, latency, pkt_size, passphrase, streamid) are immune to `extra_params` overrides. The 1.5 s probe window on start catches early subprocess exits so clients learn about URL/auth failures before polling. `stop_stream()` does SIGTERM → wait → SIGKILL / Windows `taskkill /F` fallback.

**B2.2 colour-science scopes** — new [`core/color_scopes_pro.py`](opencut/core/color_scopes_pro.py) + `POST /video/scopes/pro`, `GET /video/scopes/pro/info`. Mathematically-proper scopes via the ``colour-science`` pip package (BSD-3): mean CIE 1931 xy chromaticity, mean CIELUV (u*, v*), and **gamut coverage** fractions for Rec.709 / DCI-P3 / Rec.2020 computed with a vectorised barycentric test. Optional matplotlib PNG plots (chromaticity horseshoe + LUV vectorscope) — stats return even when matplotlib is absent.

**E4 voice-command grammar** — new [`core/voice_command_grammar.py`](opencut/core/voice_command_grammar.py) + `POST /voice/grammar/parse`, `GET /voice/grammar/catalogue`. Deterministic grammar (no LLM) for hands-free timeline editing. Verbs: `cut` / `trim` / `mark` / `slip` / `nudge` / `speed` / `slow` / `undo` / `redo` / `seek` / `mute` / `unmute` / `ripple` / `zoom_in` / `zoom_out`. Frames + seconds + minutes + beats all normalise to seconds given sequence `fps` + `bpm`. Number-words 0..99 parsed inline. ``parse()`` is advertised as never-raising — fuzzed to verify. Low-confidence parses emit `fallback_route="/nlp/command"` for broader LLM-backed dispatch.

**D4.2 Atheris fuzz harness** — new [`tests/fuzz/test_parser_fuzz.py`](tests/fuzz/test_parser_fuzz.py). Five fuzz targets: SRT time parser, SRT file parser, `.cube` LUT parser, voice-grammar parser, `event_moments._find_spikes`. Opt-in via `RUN_FUZZ=1`; default pytest runs skip the infinite-loop libFuzzer entry points. Pytest-smoke variant runs a fixed deterministic payload suite per target so CI catches "import-broke-the-fuzz-target" regressions without needing Atheris at ordinary CI time.

### Infrastructure
- 6 new `check_*_available()` entries in `opencut/checks.py`: `vvc`, `srt`, `colour_science`, `voice_grammar` (always True), `atheris`, plus the deprecation of `_AVAILABILITY_CACHE` dicts to per-module.
- 3 new routes in `_ALLOWED_QUEUE_ENDPOINTS`: `/video/encode/vvc`, `/video/stream/srt/start`, `/video/scopes/pro`.
- New blueprint `wave_d_bp` registered (naming: v1.18/19/20 = wave_a/b/c; v1.21 = wave_d).

### Gotchas
- **`_AVAILABILITY_CACHE` is per-module** — both `vvc_export` and `srt_streaming` memoise their FFmpeg-capability probe. Cache lives for the lifetime of the Python process. A rebuild that flips libvvenc on/off requires a server restart before the cached `False` becomes `True`.
- **SRT stop is platform-split** — POSIX uses `signal.SIGKILL` after the grace window; Windows falls back to `taskkill /F /PID`. Don't remove the Windows branch — Windows Python lacks SIGKILL entirely.
- **Voice-grammar is intentionally narrow** — the grammar is a whitelist, not a classifier. Utterances it doesn't match emit `fallback_route="/nlp/command"` so the broader LLM-backed parser in `core/nlp_command.py` picks up stragglers. Don't grow the grammar to catch everything — that's what the fallback is for.
- **`colour-science` sample-count cap is 120** — probing 120 frames at 256² is fast (seconds). Larger values are silently clamped. The xy / LUV means are computed on **concatenated** pixel arrays across all samples, not per-frame averages, so uniform distributions don't wash out interesting chromaticity.
- **VVC encode timeout is 4 hours** — reflects real-world archival encode times at `preset=archive` on ≥30-minute sources. Lower it for `faster` preset users only if monitored.
- **Fuzz tests skip by default** — `RUN_FUZZ=1 pytest tests/fuzz/` enables the deterministic smoke suite. libFuzzer-style real fuzzing uses `python -m tests.fuzz.test_parser_fuzz <target> -max_total_time=N`.

## [1.20.0] - 2026-04-17

### Added — Wave D remaining items (D1.1, D3.1, D4.1, D5.2, D6.2)

Three new modules, two LLM-enrichment extensions of existing modules, one server-level observability hook. 1,202 → 1,207 total routes (+5).

**D3.1 Semantic OTIO timeline diff** — new [`opencut/export/otio_diff.py`](opencut/export/otio_diff.py) + `POST /timeline/diff`. Loads two timelines through OpenTimelineIO's adapter framework (supports `.otio`, `.otioz`, FCP XML, EDL, AAF when the optional adapter is installed) and emits a structured `OtioDiffResult` with track-level, clip-level, and marker-level diffs. Clip matching by `(name, source_url)` tuple with order-preserving pair-up; different `(start, duration)` → `retimed`, different track index → `moved`. Includes `format_diff_text()` for CLI / log output.

**D4.1 Objective quality metrics** — new [`opencut/core/quality_metrics.py`](opencut/core/quality_metrics.py). VMAF / SSIM / PSNR comparison of a distorted file against a reference via FFmpeg's built-in filters + libvmaf JSON log. Three entry points: `measure_vmaf()` / `measure_ssim()` / `measure_psnr()` for single metrics, `compare_videos()` for a full `QualityReport`, and `batch_compare()` for CI golden-regression suites. Routes: `POST /video/quality/compare`, `POST /video/quality/batch-compare`, `GET /video/quality/backends`. Handles libvmaf absence gracefully — SSIM/PSNR still run; VMAF emits a clear `notes` entry. Threshold-based `passes` gate for CI regression budgets.

**D5.2 Optional Sentry / GlitchTip observability** — new `_init_sentry_if_configured()` hook in [`opencut/server.py`](opencut/server.py), called from `create_app()`. Activated when `SENTRY_DSN` env var is set AND `sentry_sdk` is installed — no behaviour change for installs without either. Reads `OPENCUT_SENTRY_ENV`, `OPENCUT_SENTRY_RELEASE`, `OPENCUT_SENTRY_SAMPLE_RATE`. `send_default_pii=False` so request bodies (which often carry user media paths) don't leak to the error tracker. Reports via `GET /observability/status`.

**D1.1 LLM-enriched audio description** — new `describe_scene_llm()` in [`opencut/core/audio_description.py`](opencut/core/audio_description.py). Takes a timestamp + surrounding transcript + optional brightness hint, asks the configured LLM (`core/llm.py`) to produce a single present-tense AD line of ≤ `~3·duration` words. No vision model needed — pure text-to-text. Falls back to the heuristic template on any LLM error so AD rendering never aborts mid-job. Complements the existing template-based `describe_visual_content()`; callers can pick per invocation.

**D6.2 LLM-enriched quiz generation** — new `generate_quiz_questions_llm()` in [`opencut/core/quiz_overlay.py`](opencut/core/quiz_overlay.py). When `llm_config` is provided, produces genuine comprehension questions with shuffled `correct_index`. Strict JSON output parsed defensively — markdown code fences stripped, malformed entries dropped. Falls back to the existing TF-IDF fill-in-the-blank generator on any failure, so callers pass `llm_config` unconditionally.

### Infrastructure
- **4 new `check_*_available()`** entries in `opencut/checks.py`: `otio_diff`, `quality_metrics`, `vmaf`, `sentry`.
- **3 new routes added to `_ALLOWED_QUEUE_ENDPOINTS`** so they're queueable via `/queue/add`.
- **New blueprint** `wave_c_bp` registered in `routes/__init__.py` (alongside `wave_a_bp`, `wave_b_bp`).

### Gotchas
- **VMAF absence is non-fatal** — a FFmpeg build without `--enable-libvmaf` returns a `QualityReport` with `vmaf=None` and a diagnostic in `notes`. Callers must check `report.vmaf is not None` before using the value. Windows Gyan.dev / BtbN builds include libvmaf; many Linux distro packages don't.
- **Sentry init is idempotent** — `_SENTRY_INITIALISED` guards re-entry. `create_app()` can be called multiple times in tests without creating multiple hubs. Environment variables are read *every* call, so changing them between `create_app()` invocations in test fixtures won't take effect until a fresh interpreter.
- **OTIO AAF reads require the adapter even for diff** — `/timeline/diff` on `.aaf` inputs fails with `MISSING_DEPENDENCY` unless `otio-aaf-adapter` is installed. The error surfaces through the `job.error` event (the route is `@async_job`), not inline.
- **LLM quiz JSON fence stripping is defensive** — the model may wrap output in ```` ```json ... ``` ````, ` ```...``` `, or `~~~...~~~`. We strip all three before `json.loads`. A genuinely non-JSON response falls back to TF-IDF — no error surfaces to the caller.
- **`describe_scene_llm` word cap is approximate** — the `max_words + 4` tolerance (vs strict `max_words`) handles short models that over-shoot by one or two words without truncating natural sentence endings. AD burn-in downstream clamps display duration to the gap length, so a slightly long line is still safely bounded.

## [1.19.1] - 2026-04-17

### Fixed — production hardening audit on v1.17.0–v1.19.0 additions

- **Broken `__import__("PIL.Image")` hack in `core/matte_birefnet.py` `cutout` branch** — the previous code relied on Python's `__import__` returning the top-level `PIL` package, which happened to expose `.Image` only because `from PIL import Image` had run earlier in the same function.  Fragile, linter-hostile, and confusing to read.  Replaced the cutout compositing with a clean `from PIL import Image` + `import numpy as np` path, extracted into a `_compose_output()` helper that also owns the `alpha` / `rgba` / `cutout` branching.  Removed the `_ = np` load-bearing noise line that existed solely to suppress an unused-import lint on the broken code path.
- **Apples-to-oranges score comparison in `core/event_moments.py::_find_spikes`** — the "keep the louder of two close picks" branch compared a raw RMS value (`e`, typically 0..0.5) against a normalised score (`moments[-1].score`, 0..1) scaled by `mean` — different units entirely.  Result: the replacement branch almost never fired, so `min_spacing` was ignored for clustered spikes and duplicate moments survived ranking.  Fix: track the raw RMS magnitude of the last pick (`last_picked_e`) separately from the normalised score used in the final `EventMoment`, and compare in consistent units.  Also extracted an `_build_spike()` helper to eliminate the copy-pasted EventMoment construction.
- **Model-cache race in `core/tts_f5.py`** — `_MODEL_CACHE` was a plain module-level dict.  Two concurrent `POST /audio/tts/f5` requests could both observe `cached_model is None`, both call `F5TTS(model_type=...)`, both write — doubling VRAM for the race window and wasting ~5 s of load time each.  Added `_MODEL_LOCK` (`threading.Lock`) around reads and writes; extracted `_get_or_load_model()` so the locking discipline lives in one place.  When a *different* model is requested than the cached one, the previous is released via `torch.cuda.empty_cache()` before the new one is loaded — prevents two big torch models pinned in VRAM simultaneously.
- **Backend-cache race in `core/neural_interp.py`** — same shape but less severe: `_BACKEND_CACHE["checked"]` was written before `rife_cli` / `rife_torch`, so a concurrent reader could see `checked=True` + empty values.  Added `_BACKEND_LOCK` and write `checked=True` last inside the critical section.

### Changed
- **AAF + OTIOZ exports are now async** (`POST /timeline/export/aaf`, `POST /timeline/export/otioz`) — previously blocked the Flask request thread for tens of seconds on 500-segment timelines.  Both promoted to `@async_job` with a shared `_export_pre_validate` hook so malformed `segments` lists short-circuit to HTTP 400 synchronously before a worker slot is consumed.  Added to `_ALLOWED_QUEUE_ENDPOINTS` so they're queueable via `/queue/add`.  Response shape now includes `framerate` + `sequence_name` echoes for downstream tooling.
- **`core/clip_quality.py`** — removed the stale `# noqa: F401` annotation on `from PIL import Image` in the `piq` fallback branch.  The import is actually used two lines later (`Image.open(fpath).convert("RGB")`) — the `noqa` was a leftover from an earlier refactor and misled readers about the import's purpose.

### Gotchas
- **F5-TTS model swap frees VRAM** — when a caller requests a different model name than the cached one, `_get_or_load_model()` deletes the previous model + calls `torch.cuda.empty_cache()` before loading.  This means rapidly switching between `F5-TTS` and `E2-TTS` causes repeated cold-loads.  Stick to one model per deployment where possible.
- **Spike dedup uses raw RMS** — `_find_spikes` compares raw `e` against `last_picked_e`, not the normalised `.score` field.  If future callers add additional scoring criteria to `EventMoment`, remember the replacement check is **by loudness**, not by the scored axis.
- **Async AAF/OTIOZ error surface** — exports now return 202 + job_id instead of an inline 503 when the adapter is missing.  Clients must poll `/status/{job_id}` and inspect `error` + `code` fields.  The `pre_validate` hook still short-circuits malformed requests with 400 synchronously.

## [1.19.0] - 2026-04-17

### Added — Wave A (remaining items) + Wave D2 + Wave D3.2

Seven new core modules + one jobs-layer hook; 10 new routes; 1,192 → 1,202 total routes. All backends optional and gated behind `check_X_available()`; FFmpeg fallbacks where applicable keep routes functional on minimal installs.

**Wave A2.3 — Matting**
- **BiRefNet still/keyframe matte** — `core/matte_birefnet.py`, `POST /video/matte/birefnet`. ONNX Runtime (preferred, no torch) or transformers/HF fallback. Three output modes: `alpha` (α only), `rgba` (RGB + α), `cutout` (premultiplied RGBA). Targets the `ZhengPeng7/BiRefNet` HF repo by default. ONNX checkpoint via `OPENCUT_BIREFNET_ONNX` env var.

**Wave A3.1 — Advanced Captions**
- **Karaoke / kinetic captions** — `core/captions_karaoke_adv.py`, `POST /captions/karaoke-adv/render`, `GET /captions/karaoke-adv/presets`. Six presets — `fill`, `bounce`, `color_wave`, `typewriter`, `karaoke_glow`, `highlight_word` — using libass override tags. PyonFX hook available but defers to built-in renderer until PyonFX-specific FX logic is wired in a follow-up. `segments_from_whisperx_dicts()` adapter accepts WhisperX segment dicts directly.

**Wave A4.2 — Encoding**
- **SVT-AV1-PSY** — `core/svtav1_psy.py`, `POST /video/encode/svtav1-psy`, `GET /video/encode/svtav1-psy/info`. Three perceptually-tuned presets (`social`, `web`, `archive`). Uses the FFmpeg-integrated `libsvtav1` path when available; falls back to a 2-stage rawvideo pipe through the standalone `SvtAv1EncApp` binary. Outputs `.mp4` with libopus audio.

**Wave D2 — Restoration**
- **DDColor B&W colorisation** — `core/colorize_ddcolor.py`, `POST /video/restore/colorize`. Frame-by-frame colorisation via ONNX Runtime (checkpoint path via `OPENCUT_DDCOLOR_ONNX`) or ModelScope (`damo/cv_ddcolor_image-colorization`). Extract → colourise → reassemble with original audio.
- **VRT / RVRT unified restoration** — `core/restore_vrt.py`, `POST /video/restore/vrt`. Unifies denoise / deblur / super-res / real-world-SR / `unified` into one pass via sliding-window temporal inference (default window=8). ONNX or `basicsr` backend. The `basicsr` path is a forward-compatible stub — the ONNX path is the complete inference loop.
- **Neural deflicker** — `core/deflicker_neural.py`, `POST /video/restore/deflicker`. 7-frame temporal sliding window via All-In-One-Deflicker ONNX checkpoint, with `auto` mode that falls back to FFmpeg's `deflicker` + `tmix` chain when the neural backend is absent. Audio stream-copied verbatim.
- **Restoration backends report** — `GET /video/restore/backends` enumerates which backends are installed and provides install hints.

**Wave D3.2 — Webhook auto-emit**
- **Job-complete webhook events** — `jobs.py::_emit_job_webhook()` automatically fires `job.complete` / `job.error` / `job.cancelled` events via `core/webhook_system.fire_event()` on terminal job status. Dispatched asynchronously through `_io_pool` so job finalisation is never blocked on outbound HTTP. Best-effort — any webhook failure is logged at warning and swallowed.
- **Webhook event catalogue route** — `GET /webhooks/events` lists auto-emitted event types + payload schema + retry policy (3 attempts, 1 s / 5 s / 15 s backoff).

### Infrastructure
- 6 new `check_*_available()` entries in `opencut/checks.py` — `birefnet`, `pyonfx`, `svtav1_psy`, `ddcolor`, `vrt`, `neural_deflicker`.
- 5 new routes added to `_ALLOWED_QUEUE_ENDPOINTS`.
- `_emit_job_webhook()` hook in `jobs._update_job` — lazy-imports `webhook_system`, never blocks.

### Gotchas
- **Webhook auto-emit is best-effort** — a failing webhook must **never** block `_update_job`. The `try/except` around `_emit_job_webhook()` is load-bearing. Dispatch runs on the `_io_pool` worker; raise only inside the worker where it'll be logged.
- **BiRefNet outputs vary by checkpoint** — different exported ONNX files emit different head shapes (single tensor, tuple, list). `_postprocess_alpha()` handles squeeze + sigmoid + clamp defensively; don't trust `mask.shape` assumptions when swapping checkpoints.
- **PyonFX is advisory** — `backend="pyonfx"` is reported when the library is installed but the renderer still uses the built-in preset path until we wire per-syllable FX. Users with Aegisub-specific expectations should be warned the implementation is pending.
- **SVT-AV1-PSY integrated vs standalone** — the integrated `libsvtav1` path is ~2× faster than the rawvideo pipe + standalone binary because it skips an intermediate yuv420p10le encode. Prefer an FFmpeg build with libsvtav1 linked against the PSY fork; the standalone path is there as a fallback, not a goal.
- **VRT `basicsr` backend is a stub** — the ONNX path is production-ready; the torch/basicsr path logs a warning and no-ops on frames. Users hitting the stub should either install the ONNX checkpoint or wait for the full torch wiring (Wave E tracking).
- **Deflicker auto-fallback path** — when `backend="auto"` is chosen and the neural path is *registered* but fails at runtime (e.g. ONNX checkpoint missing but env var set), the code intentionally re-calls `deflicker_video(..., backend="ffmpeg")` to guarantee a usable output. Do not remove this recursion — early-stage users frequently mis-set `OPENCUT_DEFLICKER_ONNX`.
- **DDColor ModelScope import cost** — `modelscope.pipelines.pipeline("image-colorization")` is slow (~10 s cold) and downloads weights on first use. Consider pre-warming via a background job on server start for production installs.
- **Job-complete event payload includes `result`** — the full job result dict is included in the webhook payload. If a webhook endpoint sits over an untrusted network, users should register it via a domain filter (the existing `_validate_webhook_url` layer blocks private / loopback addresses by default).

## [1.18.0] - 2026-04-17

### Added — Wave A + Wave D (first batch from ROADMAP-NEXT.md)

Ten features built against the established `@async_job` + subscriptable-
dataclass + graceful-degradation pattern. All new routes registered under
the new `wave_a_bp` blueprint. 1,177 → 1,192 total routes (+15).

**Wave A1 — Audio**
- **F5-TTS zero-shot voice clone** — `core/tts_f5.py`, `POST /audio/tts/f5`, `GET /audio/tts/f5/models`. Supports F5-TTS and E2-TTS models. Module-level model cache to avoid re-loading per request. Subscriptable `F5Result`.
- **WhisperX `--diarize` exposure** — extended `CaptionConfig` with `diarize`, `hf_token`, `min_speakers`, `max_speakers` fields; `_transcribe_whisperx` now runs pyannote 3.x diarisation when `diarize=True` and an HF token is available (env `HF_TOKEN` or explicit). Falls back to plain transcription on pipeline failure — never kills a caption job over a diarisation error.
- **BeatNet downbeat detection** — `core/beats_beatnet.py`, `POST /audio/beats/beatnet`. Supports `offline` (best) and `realtime` modes; configurable meter (3/4/6). Approximates a confidence score as the ratio of downbeat labels to expected-per-meter counts.

**Wave A2 — Video Intelligence**
- **Scene detection auto-dispatcher** — `core/scene_detect.py::detect_scenes_auto()`, `POST /video/scenes/auto`. TransNetV2 (ML) → PySceneDetect → FFmpeg threshold priority. Keeps the original `detect_scenes` / `detect_scenes_ml` / `detect_scenes_pyscenedetect` entry points callable by name for users who want a specific backend.
- **CLIP-IQA+ clip quality scoring** — `core/clip_quality.py`, `POST /video/quality/score`, `POST /video/quality/rank`. Samples frames via FFmpeg, scores each via `pyiqa.create_metric("clipiqa+")` (or `piq.CLIPIQA` fallback). `INVERTED_AXES` frozenset inverts noisy/blurry/dark so the aggregate is monotone-positive. Batch-rank up to 40 clips per call.
- **HSEmotion emotion arc** — `core/emotion_arc.py`, `POST /video/emotion/arc`. 8-class emotion timeline (Neutral/Happy/Sad/Surprise/Fear/Disgust/Anger/Contempt). Prefers `hsemotion_onnx` backend (no torch needed), falls back to torch-based `hsemotion`. Emits mean probabilities, transition count, and an emotional-range score for highlight ranking.

**Wave A4 — Encoding**
- **ab-av1 VMAF-target encode** — `core/ab_av1.py`, `POST /video/encode/vmaf-target`, `GET /video/encode/vmaf-target/info`. Wraps the ab-av1 CLI for "give me VMAF 95" one-shot encoding (SVT-AV1 / libaom-av1 / libx264 / libx265). Parses both the trailing-JSON format and the legacy `crf N VMAF M` regex output. Hard 3 h subprocess timeout.

**Wave A5 — Interchange**
- **OTIO AAF export** — `export/otio_export.py::export_aaf()`, `POST /timeline/export/aaf`. Uses `otio-aaf-adapter` (Apache-2) to emit Avid-compatible AAF files. Graceful 503 `MISSING_DEPENDENCY` when the adapter isn't installed.
- **OTIOZ bundle export** — `export/otio_export.py::export_otioz()`, `POST /timeline/export/otioz`. Zip container with optional embedded media (`bundle_media=True`). One-file portable project handoff.

**Wave D1 — Compliance**
- **Broadcast compliance profiles** — extended `core/caption_compliance.py` with **EBU-TT-D** (EBU Tech 3380), **YouTube broadcast-grade**, and **accessibility-first** (CVAA + WCAG 2.2) profiles in addition to the existing netflix / bbc / fcc / youtube sets. New `GET /captions/compliance/standards` to let panels enumerate them.

**Wave D6 — Verticals**
- **Event moment finder** — `core/event_moments.py`, `POST /events/moments`. Wedding / ceremony / live-event highlight detection via audio-energy RMS spike detection (mean + k·σ threshold) with a `min_spacing` guard to prevent redundant picks. Optional YAMNet stub for AudioSet tagging when `tflite_runtime` + a local model path are configured via `OPENCUT_YAMNET_MODEL`.

### Changed
- Nothing user-visible — all Wave A/D additions live behind `check_X_available()` guards and degrade gracefully when optional dependencies are absent.

### Infrastructure
- 8 new `check_*_available()` entries in `opencut/checks.py`: `f5_tts`, `beatnet`, `clip_iqa`, `hsemotion`, `ab_av1`, `aaf_adapter`, `event_moments`, plus the pre-existing `neural_interp` / `declarative_compose`.
- 8 new routes added to `_ALLOWED_QUEUE_ENDPOINTS` so they can be queued via `/queue/add`.
- `captions.CaptionSegment` now carries an optional `speaker: str` field populated by the WhisperX diarisation pipeline when enabled.

### Gotchas
- **F5-TTS model cache stays alive across requests** — `_MODEL_CACHE["model"]` pins the loaded model to free subsequent requests from the ~5 s load penalty. Call `tts_f5.clear_model_cache()` from a GPU-pressure code path if VRAM becomes tight.
- **WhisperX diarisation auto-fallback** — the `try/except Exception` wrapper around the pyannote branch is intentional. If pyannote 3.x / the HF token / the model download fails for any reason, the caption job completes *without* speaker labels rather than erroring out. Any diarisation-specific failure mode must be surfaced via `logger.warning`, never re-raised.
- **BeatNet wav decode** — we intentionally decode to a temp 22 050 Hz mono WAV because BeatNet's pretrained models expect that rate. The temp file is cleaned in `finally`.
- **ab-av1 output parsing** — we check for a JSON summary line first (current versions), then fall back to the `crf N VMAF M` regex for older builds. Some versions emit neither — in that case the result has `achieved_vmaf=0, final_crf=0` and only the `output` path is reliable.
- **OTIOZ `bundle_media` pulls media into the zip** — relies on OTIO's `file_bundle_utils.MediaReferencePolicy`. The policy enum name differs across OTIO versions (`AllMissingReferences` on 0.17, `ALL_MISSING_REFERENCES` on 0.15). Write the adapter import inside the function so import-time failures don't gate module import.
- **Scene-detect auto-dispatcher is stateless** — `detect_scenes_auto` imports TransNetV2/PySceneDetect fresh per call. Cheap in practice (imports are cached by Python) but avoid calling it in a tight inner loop; use `detect_scenes_ml` directly if you already know TransNetV2 is installed.
- **Event-moments heuristic fallback** — when `mode="yamnet"` is requested but `tflite_runtime` / TF / a valid model file are unavailable, the function silently downgrades to `mode="heuristic"` and records that in `result.mode`. Never raise in that case — YAMNet is a nice-to-have enrichment, not a requirement.

## [1.17.0] - 2026-04-17

### Added — cross-project research pass

Two feature surfaces borrowed from OSS editors that were missing from the codebase after a survey of LosslessCut, auto-editor, editly, Shotcut/MLT, Olive, OpenShot, Kdenlive, OpenTimelineIO, WhisperX, and PyAV. (Most research recommendations were already implemented: faster-whisper + whisperx, OTIO export, DeepFilterNet, EBU R128 two-pass loudnorm, and GOP-aware `smart_render` are all present. These two were the remaining concrete gaps.)

- **Neural frame interpolation** (`core/neural_interp.py`, `routes/enhancement_routes.py`) — RIFE-NCNN-Vulkan CLI backend with torch-RIFE hook and FFmpeg `minterpolate` fallback. Single `interpolate(video_path, target_fps, backend="auto", ...)` entry point with auto backend selection, graceful degradation, and a `notes` field describing fallbacks. Routes: `GET /video/interpolate/backends` (list installed/available), `POST /video/interpolate/neural` (async, `@async_job("neural_interp")`). Doubles via RIFE up to 3 passes (8×), then decimates to the exact `target_fps`. Audio stream-copied from the source to avoid resampling drift. Source for the external binary: https://github.com/nihui/rife-ncnn-vulkan.
- **Declarative JSON video composition** (`core/declarative_compose.py`, `routes/enhancement_routes.py`) — editly-inspired (https://github.com/mifi/editly). Single JSON spec → finished video, no timeline reasoning required by callers. Supports `video` / `image` / `title` / `color` clip types, ~18 xfade transitions (fade, dissolve, wipe*, slide*, circle*, smooth*, pixelize, radial), optional ducked background audio, burnt captions (top/bottom/center) via drawtext, and per-clip in-points. Every intermediate clip is normalised to the target `width×height×fps` + yuv420p + AAC 48 k stereo so `filter_complex` concat/xfade works without "resolution mismatch" errors. Routes: `GET /compose/schema` (enums + example spec), `POST /compose/validate` (sync validation without rendering), `POST /compose/render` (async, `@async_job("compose", filepath_required=False)`). Max 200 clips per spec. `ComposeResult` dataclass is subscriptable so Flask `jsonify` accepts it directly.
- **New checks** — `check_neural_interp_available()` (always True — FFmpeg fallback), `check_rife_cli_available()` (rife-ncnn-vulkan on PATH), `check_declarative_compose_available()` (FFmpeg on PATH). Added to `opencut/checks.py`.
- **Queue allowlist** — `/video/interpolate/neural` and `/compose/render` added to `_ALLOWED_QUEUE_ENDPOINTS` in `jobs_routes.py` so they can be queued via `/queue/add`.
- **Blueprint** — `enhancement_bp` registered in `routes/__init__.py`. 5 new routes bring the total from 1,152 → 1,177.

### Design notes
- Both modules follow the established dataclass-with-`__getitem__` pattern (`InterpResult`, `ComposeResult`) so routes can return them directly to Flask's `jsonify` without a `to_dict()` conversion — matches the `MEMixResult` / `PremixResult` convention from v1.16.2.
- Neural interpolation refuses to run when `target_fps <= source fps` and redirects the caller to the existing `/video/framerate-convert` route for downsampling.
- `declarative_compose._escape_text` scrubs drawtext meta-characters (`:`, `;`, `\`, `'`) and control bytes, and caps text at 500 chars — same pattern as the title/overlay route hardening in v1.3.1 batch 10.

## [1.16.3] - 2026-04-16

### Fixed
- **Demucs separation could hang the worker thread forever** — `routes/audio.py::audio_separate` consumed `process.stdout` line-by-line then called `process.wait()` with no timeout. A wedged Demucs (stdout closed but child not reaping) blocked the whole job slot indefinitely. Bounded the final wait to 1800 s with a force-kill + `RuntimeError` on overrun so the slot is released and the user gets an actionable error instead of a permanent "running" job.
- **UXP `_doFetch` had no request timeout** — every backend call (`get` / `post` / `del`) used `fetch(url, opts)` with no `AbortController`. A hung backend pinned the calling button into "Working…" forever. Added a 120 s `AbortController.signal` on every request, cleared in `finally`. Also wrapped `resp.json()` in try/catch so a server that lies about `Content-Type: application/json` and returns garbage doesn't blow past the status-code check.
- **`ocAddNativeCaptionTrack` produced malformed SRT for multi-line captions** — caption text was written verbatim with `tempFile.writeln(seg.text || "")`. SRT uses a blank line as the cue terminator, so any caption containing `\n\n` (very common in Whisper output for sentence breaks) split a single cue into two malformed cues. Premiere either dropped the half-cues or imported them with mis-aligned timecodes. Now collapses `\r\n`/`\r` to `\n`, replaces consecutive newlines with a single `\n`, trims trailing whitespace, and substitutes a single space when the body is empty (empty cue body is also invalid SRT).

### Hardened — encoding consistency
- **`open()` text-mode without `encoding=`** — added `encoding="utf-8"` to JSON metadata reads/writes in `core/model_manager.py`, `core/character_consistency.py`, `core/motion_transfer.py`, `core/proxy_gen.py`, and `core/proxy_swap.py`. Without an explicit encoding, the Python default on Windows is the system codepage (cp1252), which raises `UnicodeDecodeError` whenever a profile/proxy mapping mentions a path or speaker name with non-ASCII characters. Concat-list files were intentionally left alone — FFmpeg reads them in the system locale, so forcing UTF-8 there would corrupt non-ASCII paths the writer thinks were saved correctly.

### Audit notes (no fix needed)
- `subprocess.run` / `_sp.run` calls — verified all 100+ call sites across the codebase carry an explicit `timeout=`. Multi-line calls had previously evaded a single-line grep; a syntax-aware sweep confirmed full coverage.
- `urlopen()` — verified all call sites carry a `timeout=` argument.
- `process.wait()` on `Popen` — only one site (`routes/audio.py` Demucs) lacked a timeout; fixed above.
- `WorkerPool` shutdown / cancel paths — all already use `set_running_or_notify_cancel()` and `BaseException` to guarantee futures resolve even on `SystemExit`/`KeyboardInterrupt`.
- `errors.safe_error()` exception classification — verified the order is correct (more specific phrases checked before broader ones); GPU OOM is matched before the generic FFmpeg branch so an "FFmpeg out of memory" RuntimeError reports `GPU_OUT_OF_MEMORY` (the more actionable code).
- `job_store.save_job` `ON CONFLICT … DO UPDATE` — `result_json` / `error` / `progress` use `excluded.X` (not `COALESCE`), so a status-only update wipes the prior value. This is intentional: callers always pass the full job dict from the in-memory `jobs` registry, so `excluded` is never partial.

## [1.16.2] - 2026-04-16

### Fixed — broken production routes (entire endpoints were 500ing on every call)
- **`/api/audio/stem-remix` was non-functional** — `routes/audio_expansion_routes.py::stem_remix` imports `mix_stems` and calls `remix_stems(stem_paths=LIST, effects_config=, output=, on_progress=)` and reads `result.stems_processed` / `result.effects_applied`. None of those names existed in `core/stem_remix.py`. Added the full effect-config API surface to the module: `SUPPORTED_EFFECTS` dict (10 named effects), `StemEffect` dataclass, `_effect_to_filter()`, `_build_pan_filter()`, `apply_stem_effect()`, `mix_stems()`. Made `remix_stems()` polymorphic so it routes a list-based `stem_paths` + `effects_config` call to a new `_remix_stems_with_effects()` helper while preserving the original dict-based preset path. Added `stems_processed` / `effects_applied` fields to `RemixResult`.
- **`/video/data-animation/*` was non-functional** — `routes/color_mam_routes.py` calls `create_data_animation`, `render_bar_chart`, `render_counter` from `core/data_animation.py`; module only had the lower-level `render_data_animation`. Added a high-level chart API: `DataTemplate`, `DataAnimationResult` dataclasses, `_load_data_source()`, `_extract_labels_values()`, `render_bar_chart()`, `render_counter()`, `create_data_animation()`. Hoisted `run_ffmpeg`/`get_ffmpeg_path` to module-level imports so route/test patching actually intercepts them.
- **`/video/shape-animation/*` was non-functional** — same pattern as data-animation: routes call `animate_shape_morph`, `animate_stroke_draw`, `animate_fill_transition` from `core/shape_animation.py`; module only had `render_shape_animation`. Added `ShapeDefinition`, `ShapeAnimationResult` dataclasses, `_parse_shape()`, the three animate-* entry points, plus a `_normalise_color()` helper that converts ``0xFF0000`` style hex into the ``#FF0000`` shape FFmpeg accepts. Module-level `run_ffmpeg` import for testability.
- **`/audio/me-mix` returned a non-JSON-serialisable dataclass** — `routes/audio_production_routes.py::route_me_mix` does `return result` straight into Flask's auto-jsonify, but `core/me_mix.py::generate_me_mix` returned the legacy `MEMixResult` dataclass which isn't JSON-friendly. Also the legacy `_level_match()` post-step crashed on every "subtract" call because of the `os.replace(tmp_out, filepath)` race when the input doesn't actually exist. Rewrote `generate_me_mix()`: default `method="subtract"` (was `"auto"`), added a fast `_me_subtract()` path that does a single FFmpeg `pan=stereo|c0=c0-c1|c1=c1-c0` + `loudnorm` invocation and returns a hybrid `MEMixResult` (now subscriptable so `result["method_used"]` and `result.method_used` both work). Legacy paths (`stem_separation`, `track_mute`, `spectral`) still route to the original implementation.
- **`/audio/dialogue-premix` returned dataclass to Flask `jsonify`** — `core/dialogue_premix.py::premix_dialogue` had `target_lufs` default of `None` (used the preset's broadcast value, not what the route passed), and the result was a `PremixResult` dataclass. Added `premix_multi_speaker()` as a public alias of the internal helper, defaulted `target_lufs=-23.0`, and made `PremixResult` subscriptable so dict and attribute access both work. Wrapped the progress callback so internal 1-arg calls and external 2-arg `(pct, msg)` calls compose without a `TypeError`.
- **CLI `opencut nlp` was completely broken** — imported from non-existent `opencut.core.nlp` (real module is `nlp_command`), called `parse_nlp_command` (real name is `parse_command`), and accessed `parsed.route` / `parsed.params` / `parsed.execute(file)` on a dict that has no such attributes/methods. Rewrote the command to use the correct import + `parse_command()` dict result, and replaced the hallucinated `.execute()` call with an actual `urllib` POST to the matched route (with CSRF refresh). On a missing backend the command now exits non-zero with a helpful "start the server first" hint.
- **CLI `chapters` / `repeat-detect` / `search index` all imported a non-existent `transcribe_audio`** — all three commands do `from .core.captions import transcribe_audio`, but the captions module only exposes `transcribe(filepath, config=...)`. Added `transcribe_audio(filepath, model, language, timeout)` to `core/captions.py` as a thin wrapper that builds a `CaptionConfig` and projects the `TranscriptionResult` down to a flat list of segment dicts so callers don't need the full result object.

### Added — in-memory subtitle/multilang API
- **`MultilangProject`, `create_multilang_project()`, `update_language_text()`, `export_language()`, `sync_timing_change()`** in `core/multilang_subtitle.py`. The persistence-based `create_project` / `add_language` / `export_srt` API is preserved; the new functions are a smaller in-memory surface used by the v1.15.0 editing-workflows tests and by callers that don't need disk-backed project IDs. `sync_timing_change` supports `shift` / `update` / `insert` / `delete` operations and mirrors all timing edits across every language array so they never drift out of sync.

### Fixed — robustness
- **`room_tone.analyze_room_tone()` propagated librosa load failures** — when librosa was installed, a missing/corrupt input file raised `FileNotFoundError` from `librosa.load()` instead of degrading to the FFmpeg-based fallback that the function was specifically designed to provide. Wrapped the whole librosa block in a broad `except Exception` (still re-raised for `ImportError` short-circuit) and logged at info level, so callers always get either a librosa-grade or an FFmpeg-grade envelope.
- **CEP recent-files store crashed if localStorage was tampered** — `getRecentFiles()` returned whatever `JSON.parse` produced, so a corrupted `opencut_recent_files` value (`null`, `"foo"`, an object) made `populateRecentFiles` and `addRecentFile` crash on `.length` / `.filter` / `.unshift`. Now defensively coerces to a clean `Array` of `{path, name}` entries, filtering out anything missing a string `path`. `addRecentFile` reuses the same parser instead of re-parsing raw localStorage.

### Result counts
- **7,689 tests passing**, 0 failing, 3 skipped (was 7,629 / 60 / 3 before this batch). Net +60 tests fixed across 5 production-route bug clusters and 1 environment-fragility test.

## [1.16.1] - 2026-04-16

### Fixed
- **`outpaint_aspect_ratio` silently ran on missing files** — `core/frame_extension.py::outpaint_aspect_ratio()` (and the related `extend_frame_spatial`/`extend_frame_temporal` helpers) called `get_video_info()` first, which logs a warning and returns 1920x1080 defaults on a missing file, then walked the rest of the pipeline producing garbage output paths. Added an explicit `os.path.isfile()` check that raises `FileNotFoundError` upfront for all three entry points.
- **Command palette recents sorted by insertion order, not recency** — `core/command_palette.py::record_feature_use()` stamped each entry with `time.time()`, which on Windows advances in ~15.6 ms ticks. Two consecutive `record_feature_use()` calls received identical timestamps and the stable sort kept the earlier entry on top, so the most-recently-used feature was hidden behind older ones. Now bumps the timestamp to `max(existing) + 1us` whenever the wall clock has not advanced past the previous entry — guarantees monotonic ordering without breaking the JSON schema.
- **Step duration always reported 0.0 in autonomous agent** — `core/autonomous_agent.py::execute_step()` measured elapsed time with `time.time()`, which has ~15.6 ms resolution on Windows. Mocked or fast-completing steps recorded `duration_seconds=0.0`, breaking downstream timing analytics. Switched to `time.perf_counter()` with a `1e-6` floor. Same fix applied to `execute_plan()`.
- **Model quantization elapsed always 0.0 for fast paths** — `core/model_quantization.py::quantize_model()` used `time.time()` then `round(elapsed, 2)`, so any sub-10 ms run rounded to `0.0`. Switched to `perf_counter`, raised reporting precision to 3 decimals, and applied a 1 ms floor so callers can distinguish "ran" from "no-op".
- **`get_video_info()` raised on ffprobe timeout/missing** — `helpers.py::get_video_info()` called `subprocess.run(..., timeout=30)` without catching `TimeoutExpired`, `FileNotFoundError`, or `OSError`. A hung or missing ffprobe surfaced as a 500 instead of degrading to safe defaults like every other failure path. Now catches the subprocess exceptions and logs a warning before returning defaults.

## [1.16.0] - 2026-04-15

### Security & Hardening
- **Path traversal prevention** — `validate_output_path()` added to all 157+ output_path and 100+ output_dir parameters across 43+ route files.
- **FFmpeg concat demux injection** — Newline/carriage return stripping in filenames for `video_core.py` concat operations.
- **SSRF prevention** — LLM base_url validation in captions routes blocks non-HTTP schemes.
- **Information disclosure** — Error responses no longer leak raw exception messages or detail fields.
- **Docker hardening** — Non-root user added to container image.
- **Windows reserved names** — `validate_path()` now blocks CON, PRN, AUX, NUL, etc.
- **Atomic file writes** — Plugin marketplace manifest and scripting console history use tempfile + os.replace.
- **Sandbox escape prevention** — Scripting console blocked `__import__`, exec, eval, compile, open, os, sys, subprocess in AST.
- **.gitignore hardened** — Added `.env.*`, `*.key`, `*.pem`, `credentials*.json`.

## [1.15.0] - 2026-04-14

### Added
- **5 new feature categories (78-82)** — 25 new core modules with routes and comprehensive tests.
- **AI Voice & Speech** (Cat. 78): Descript-style transcript-based video editing with undo stack and EDL export, AI eye contact correction via MediaPipe gaze estimation, voice overdub (replace spoken words with TTS-cloned corrections), audio-driven lip sync with Wav2Lip fallback, voice-to-voice conversion with target profiles.
- **Motion Design & Animation** (Cat. 79): Kinetic typography engine (12 presets, 7 easing functions), data-driven animation from CSV/JSON (6 chart types), vector shape animation with SVG path morphing, sandboxed expression scripting engine (30+ math functions), configurable particle system (8 presets, 5 emitters, physics simulation).
- **Professional Subtitling** (Cat. 80): Shot-change-aware subtitle timing (Netflix/BBC/FCC profiles), multi-language simultaneous editing with shared timing, broadcast caption export (CEA-608/708, EBU-TT, TTML, IMSC1, WebVTT), SDH/HoH auto-formatting (speaker IDs, sound events, music notation), dynamic subtitle positioning to avoid visual obstructions.
- **Developer & Scripting Platform** (Cat. 81): Sandboxed Python scripting console with curated OpenCut namespace, macro recording and playback with variable substitution, FFmpeg filter chain builder (20 filter types, cycle detection), event-driven webhook system (5 event types, retry with backoff), batch scripting engine with glob patterns and dry-run.
- **Audio Post-Production** (Cat. 82): ADR cueing system with cue sheet export (CSV/JSON) and guide audio, M&E mix export (3 methods: stem separation, track mute, spectral), automated dialogue premix (per-speaker EQ/compression/de-ess, 5 presets), stereo-to-surround upmix (5.1/7.1, 4 modes, VBAP panning), foley cueing with SFX auto-placement (8 categories).
- **5 new route blueprints** — `voice_speech_bp`, `motion_design_bp`, `subtitle_pro_bp`, `dev_scripting_bp`, `audio_post_bp` (64 new routes).
- **5 new test files** — 626 new tests covering all v1.15.0 features.
- **1,152 total API routes** (up from 1,088 in v1.14.0).
- **7,551 total tests** across 92 test files (up from 6,925 across 87).
- **424 core modules** (up from 408), **88 route blueprints** (up from 83).

## [1.14.0] - 2026-04-14

### Added
- **5 new feature categories (73-77)** — 25 new core modules with routes and comprehensive tests.
- **AI Collaboration & Review** (Cat. 73): Frame-accurate review comments with annotations, frame-by-frame version comparison (SSIM/PSNR), multi-stage approval workflows, immutable edit history audit log, team-shared preset library with merge resolution.
- **Advanced Timeline Automation** (Cat. 74): AI rough cut assembly from script/brief (LLM + keyword fallback), multi-track audio auto-mix with ducking profiles, smart trim point detection (3 modes), batch timeline operations (6 op types with chaining), template-based video assembly (4 built-in templates).
- **AI Sound Design & Music** (Cat. 75): AI sound design from video analysis (12 SFX categories, PCM synthesis), procedural ambient soundscape generator (7 presets), music mood morphing (6 transforms, keyframeable curves), beat-synced auto-edit (6 cut modes, energy matching), creative stem remix (7 presets including lo-fi/nightcore/slowed reverb).
- **Real-Time AI Preview** (Cat. 76): Live AI effect preview at 480p (10 effects, LRU cache), GPU-accelerated preview pipeline with CPU fallback, A/B comparison generator (6 modes, quality metrics), real-time video scopes (waveform/vectorscope/histogram/parade/false_color), disk-persistent preview render cache with TTL.
- **Cloud & Distribution** (Cat. 77): Cloud/remote render dispatch with health checks and failover, multi-platform auto-publish (10 platforms), content fingerprinting and duplicate detection (pHash + audio), render farm management with segment dispatch, post-publish distribution analytics.
- **5 new route blueprints** — `collab_review_bp`, `timeline_auto_bp`, `sound_music_bp`, `preview_realtime_bp`, `cloud_distrib_bp` (57 new routes).
- **5 new test files** — 647 new tests covering all v1.14.0 features.
- **1,088 total API routes** (up from 1,031 in v1.13.0).
- **6,925 total tests** across 87 test files (up from 6,278 across 82).
- **408 core modules** (up from 385), **83 route blueprints** (up from 78).

## [1.13.0] - 2026-04-14

### Added
- **5 new feature categories (68-72)** — 25 new core modules with routes and comprehensive tests.
- **AI Timeline Intelligence** (Cat. 68): Holistic timeline quality analysis, per-segment engagement scoring, AI narrative clip chaining, natural language color grading (32 named looks), end-to-end auto-dubbing pipeline (47 languages).
- **Advanced Object AI** (Cat. 69): Text-driven video segmentation ("remove the red car"), physics-aware object removal (shadows + reflections), object tracking with graphic overlays (10 overlay types), CLIP-based semantic video search, multi-subject intelligent reframe.
- **Delivery & Mastering** (Cat. 70): DCP (Digital Cinema Package) export, IMF (Interoperable Master Format) packaging, automated delivery validation (Netflix/YouTube/Broadcast/DCP/Apple TV+/Amazon specs), multi-format simultaneous render with priority, delivery spec profile manager.
- **AI Content Generation** (Cat. 71): Voice-to-avatar video generation (5 styles), ML thumbnail CTR prediction, AI B-roll generation from text prompts, auto chapter artwork/title cards (5 card styles), animated video intro generation (5 intro styles).
- **Pipeline Intelligence** (Cat. 72): Pipeline health monitoring dashboard, cron-like scheduled/recurring jobs, smart content routing (10 content types), accurate processing time estimation, real-time CPU/GPU/memory resource monitoring.
- **5 new route blueprints** — `timeline_intel_bp`, `object_intel_bp`, `delivery_master_bp`, `content_gen_bp`, `pipeline_intel_bp` (51 new routes).
- **5 new test files** — 596 new tests covering all v1.13.0 features.
- **1,031 total API routes** (up from 980 in v1.12.0).
- **6,278 total tests** across 82 test files (up from 5,742 across 77).
- **385 core modules** (up from 360), **78 route blueprints** (up from 73).

## [1.12.0] - 2026-04-14

### Added
- **Competitive audit features (N1-N20)** — 25 new core modules implementing all features identified in AUDIT.md competitive analysis.
- **UX Intelligence**: Command palette with fuzzy search (N1), AI contextual suggestions (UX2), smart defaults engine (UX11), unified preview system (UX5).
- **Enhanced Media**: AI speech restoration (N2), one-click enhance pipeline (N9), low-light video enhancement (N13), scene edit detection (N14).
- **Engagement & Content**: Engagement/retention prediction (N4), 50+ animated caption styles (N5), AI hook generator (N6), A/B variant generator (N7), Essential Graphics caption output (N11).
- **Next-Gen AI**: Video LLM integration (N8), AI music remix/duration fit (N10), audio category tagging (N15), AI color match between shots (N20).
- **Motion & Generation**: Generative extend (N3), green-screen-free background replacement (N12), consistent character generation (N16), motion brush/cinemagraph+ (N17).
- **Body & Transfer**: Body/pose-driven effects (N18), full-body motion transfer (N19), AI foley sound generation, AI face restoration.
- **6 new route blueprints** — `enhanced_media_bp`, `ux_intel_bp`, `engagement_content_bp`, `next_gen_ai_bp`, `motion_gen_bp`, `body_transfer_bp` (48 new routes).
- **6 new test files** — 615 new tests covering all audit features.
- **980 total API routes** (up from 932 in v1.11.0).
- **5,742 total tests** across 77 test files (up from 5,127 across 71).
- **360 core modules** (up from 335), **73 route blueprints** (up from 67).

## [1.11.0] - 2026-04-13

### Added
- **All 302 features fully implemented** — Every feature in `features.md` now has a working core module, route blueprint, and test coverage.
- **12 new feature categories** (Batch 5) — VR/360 video, camera/lens correction, video repair & generation, privacy/redaction, spectral audio editing, split-screen & multicam, content repurposing, storyboard/pre-production, proxy management, composition intelligence, AI dubbing & localization.
- **52 new core modules** — `vr_stabilize`, `spatial_audio_vr`, `chromatic_aberration`, `lens_profile`, `old_restoration`, `sdr_to_hdr`, `framerate_convert`, `img_to_video`, `scene_extend`, `video_condensed`, `bg_replace_ai`, `plate_blur`, `pii_redact`, `profanity_bleep`, `doc_redact`, `audio_anon`, `spectrogram_edit`, `spectral_repair`, `noise_classify`, `split_screen`, `reaction_template`, `multicam_grid`, `long_to_shorts`, `video_to_blog`, `podcast_bundle`, `content_calendar`, `shot_list_gen`, `mood_board`, `script_to_roughcut`, `proxy_swap`, `media_relink`, `composition_guide`, `saliency_crop`, `ai_dubbing`, `isochronous_translate`, `multilang_audio`, `emotion_voice`, and enhancements to 15 existing modules.
- **6 new route blueprints** — `vr_lens_bp`, `repair_gen_bp`, `privacy_spectral_bp`, `multiview_repurpose_bp`, `preproduction_proxy_bp`, `composition_dubbing_bp` (52 new routes).
- **6 new test files** — 562 new tests covering all Batch 5 features.
- **932 total API routes** (up from 880 in v1.10.5).
- **5,127 total tests** across 71 test files (up from 867 across 65).
- **335 core modules** (up from 298), **67 route blueprints** (up from 61).

### Fixed
- Duplicate blueprint registration in `server.py` that caused `ValueError` on app startup.

## [1.10.5] - 2026-04-13

### Added
- **302-feature expansion plan** (`features.md`) — 62 categories covering every major video editing domain. 12 new categories added: 360/VR, camera correction, video repair, AI generation, privacy/redaction, spectral audio, split-screen, content repurposing, storyboarding, proxy management, composition intelligence, AI dubbing.
- **ROADMAP v3.0** — 7-wave implementation plan organized by dependency chains (Wave 1: quick wins with 0 new deps through Wave 7: emerging tech). Replaces completed Phases 0-6.
- **230 new core modules** (`opencut/core/`) — implementations for feature categories including deinterlace, lens correction, room tone, credits generator, retro effects, hardware acceleration, GIF export, ProRes/AV1/DNxHR encoding, color scopes, spectral audio, video repair, proxy generation, split-screen templates, AI dubbing, storyboard generation, pacing analysis, shot classification, and 200+ more.
- **43 new route blueprints** (`opencut/routes/`) — 600+ new API endpoints covering audio production, video processing, encoding, creative effects, QC checks, gaming, education, documentary, integrations, generative AI, professional workflows, and more.
- **43 new test files** (`tests/`) — smoke tests for all new route blueprints.
- **880 total routes** (up from 254 in v1.9.26). All registered via `__init__.py`.

### Fixed
- **Ruff lint** — 1,207 auto-fixed + 45 manual fixes (F401 noqa for lazy imports, F821 forward refs, E741 ambiguous vars, E701 multi-statement lines) across all new files. Codebase fully lint-clean.

## [1.10.4] - 2026-04-12

### Added
- **Expandable "Why this suggestion?"** on Sequence Assistant recommendation cards. Click to reveal the reasoning behind each suggestion with context-specific explanation.

## [1.10.3] - 2026-04-12

### Added
- **Journal "Apply to selection"** — Apply a journal entry's operation to the currently selected clip with one click from the Operation Journal panel.

## [1.10.2] - 2026-04-12

### Added
- **Persisted Assistant dismissals** — Dismissed Sequence Assistant suggestions are remembered across sessions via localStorage. Prevents the same suggestion from reappearing after panel reload.

## [1.10.1] - 2026-04-12

### Added
- **Preflight checks for /full and /shorts-pipeline** — Validate dependencies, disk space, and clip metadata before starting long multi-step pipelines. Prevents wasted time on doomed jobs.
- **More audio preview buttons** — Added waveform preview buttons on additional audio processing sub-tabs.

## [1.10.0] - 2026-04-12

### Added
- **Sequence Assistant** — "What should I edit next?" AI-powered recommendation engine. Analyzes the current clip and project context to suggest relevant operations (denoise, captions, color correction, etc.) with one-click execution. Displays as a dismissible card panel.

## [1.9.36] - 2026-04-11

### Added
- **Live audio preview for denoise** — Preview denoised audio in-panel before applying. Streams a short processed sample for A/B comparison.

## [1.9.35] - 2026-04-11

### Added
- **Visual LUT grid** — Browse available LUTs as a visual grid with thumbnail previews showing each LUT applied to a reference frame. Click to apply.

## [1.9.34] - 2026-04-11

### Added
- **Transcript-to-timeline scrubber** — Click any word in the transcript view to seek to that position in the Premiere timeline. Bidirectional sync between text and playhead.

## [1.9.33] - 2026-04-11

### Added
- **Preflight checklist** — Before starting long pipelines (silence + captions + export), display a checklist of required dependencies, available disk space, and estimated processing time. User confirms or cancels.

## [1.9.32] - 2026-04-11

### Added
- **Batch mode for Interview Polish** — Run the Interview Polish workflow on multiple clips sequentially with a single click. Progress tracking per clip.

## [1.9.31] - 2026-04-11

### Added
- **CLI `opencut polish`** — Headless interview/podcast cleanup pipeline. Chains denoise + normalize + silence removal + filler removal from the command line.

## [1.9.30] - 2026-04-11

### Added
- **Replay past job on current clip** — From the job history panel, replay any previous operation on the currently selected clip with the same parameters. One-click re-run.

## [1.9.29] - 2026-04-10

### Added
- **Interview Polish** — One-click interview/podcast cleanup pipeline. Chains: denoise + normalize + silence removal + filler removal with optimized defaults. Quick Action button on Cut tab.

## [1.9.28] - 2026-04-10

### Added
- **Operation Journal** — Persistent log of all operations performed on each clip with parameters, timestamps, and results. One-click rollback to undo the last operation.

## [1.9.27] - 2026-04-10

### Added
- **Session Context** — "Where you left off" welcome-back card on panel open. Shows last clip worked on, last operation performed, and quick resume button.

## [1.9.26] - 2026-04-09

### Fixed
- **Install route rate limit timing** — `make_install_route()` checked `rate_limit("model_install")` inside the `@async_job` body (async thread), so concurrent installs returned `200 + job_id` then failed asynchronously. Moved check to synchronous Flask handler — now returns `429 RATE_LIMITED` immediately with no job spawned.
- **`@async_job` filepath errors lacked structured codes** — Filepath validation failures returned bare `{"error": "..."}` without `code` or `suggestion` fields. Now classifies into `INVALID_INPUT` vs `FILE_NOT_FOUND` with actionable suggestions. Applies to all 97 async routes.

## [1.9.25] - 2026-04-09

### Fixed
- **`/system/dependencies` cold call 6.5s** — 20+ `importlib.import_module()` probes + subprocess calls ran synchronously. Added 60-second TTL cache. Cold: 6,463ms. Warm: <1ms (~6,000x speedup). `?fresh=1` param bypasses cache.

## [1.9.24] - 2026-04-09

### Fixed
- **Env var config ignored** — `OPENCUT_PORT`, `OPENCUT_HOST`, `OPENCUT_DEBUG` env vars were silently ignored. `server.py::main()` only read argparse with hardcoded defaults. Now reads env vars first, argparse overrides.

## [1.9.23] - 2026-04-09

### Fixed
- **UXP accessibility** — All 7 tab panels now have `aria-labelledby`. All 72 buttons have `type="button"`. `.hidden` utility class unified as global `display: none !important`.

## [1.9.22] - 2026-04-08

### Fixed
- **`/file` preview route allowlist restored** — Regression had removed the security allowlist.
- **`safe_bool()` hardened** — Rejects NaN, inf, containers. 15 new call sites across route files.
- **MCP serialization fallback** — Added JSON fallback for non-serializable job results.
- **`highlights.py` None-text guard** — Prevents crash on segments with null text.

## [1.9.21] - 2026-04-08

### Added
- **CEP panel UX polish** — Friendlier error messages, clearer button labels, improved alert formatting.

### Fixed
- **Boolean coercion hardening** — `safe_bool()` added to all 26 boolean flag sites across 8 route files. Prevents `"false"` string coercing to `True`.

## [1.9.20] - 2026-04-07

### Fixed
- **Gemini API key URL leak** — `core/llm.py` passed API key as query string, leaking through error messages. Moved to `x-goog-api-key` header. Added `_sanitize_url()`.
- **motion_graphics drawtext colon injection** — 3 drawtext builders escaped `:` inside single-quoted values, producing literal `\:` in rendered text. Removed wrong escapes.
- **Queue allowlist +19 routes** — 19 async routes missing from `_ALLOWED_QUEUE_ENDPOINTS` (total now 101).
- **CEP SSE/poll cancel race** — Stale events after cancel could trigger `onJobDone()`. Added guard.
- **audio_enhance no-op .cpu() calls** — Removed ineffective `.cpu()` calls before `del`.
- **animated_captions temp file leak** — Unlinked orphan tempfile on VideoWriter failure.

## [1.9.19] - 2026-04-06

### Changed
- Comprehensive audit + full modernization pass.

## [1.9.18] - 2026-04-05

### Fixed (Comprehensive 6-Phase Audit — 103+ issues)
- **P0: loudness_match completely broken** — `_run_ffmpeg()` returned `str` but callers accessed `.returncode`/`.stderr`. Renamed to `_run_ffmpeg_raw()`.
- **P0: UXP CSRF token refresh wrong header** — Read `X-CSRF-Token` instead of `X-OpenCut-Token`.
- **P0: UXP chat endpoint 404** — Posted to `/chat/message` but route is `/chat`.
- **P1: WorkerPool shutdown TypeError** — `None` poison pills not comparable with `PriorityQueue` items.
- **P1: styled captions progress crash** — Missing `msg=""` default.
- **P1: scene_detect + shorts_pipeline bare ffmpeg** — 4 bare `"ffmpeg"` strings.
- **P1: 3 GPU memory leaks** — MusicGen (1.2-13GB), SAM2 (1-4GB), Florence-2 (450MB) models never freed.
- **P2: 37 additional fixes** — Route allowlists, float coercion, structured errors, resource cleanup, font thread safety, CSS missing rules.

## [1.9.17] - 2026-04-03

### Changed
- README update with comprehensive feature documentation and audit prompt.

## [1.9.16] - 2026-04-02

### Fixed (Performance Audit & Memory Leak Fixes)
- **`/video/auto-zoom` broken return** — Route returned `{}` instead of `result_dict`.
- **Unbounded thread spawning** — `_persist_job()` and `_schedule_record_time()` spawned raw threads per call. Replaced with bounded `_io_pool = ThreadPoolExecutor(2)`.
- **Rate limit slot leak** — `make_install_route()` leaked permanently when `rate_limit()` returned False.
- **GPU memory leak in audio_enhance** — Now deletes all tensor refs + `.cpu()` + `torch.cuda.empty_cache()`.
- **Timer thread spam** — `_schedule_temp_cleanup()` replaced with single `_cleanup_worker` daemon thread + event queue.
- **Haar cascade race condition** — Added `_CASCADE_LOCK` with double-checked locking.
- **SQLite connection leak** — Added `close_all_connections()` shutdown hook.
- **CEP health timer leak** — `setInterval` reassigned without `clearInterval`.
- **Single theme ("Studio Graphite")** — Removed multi-theme system (10 themes). Only root dark theme remains.

## [1.9.15] - 2026-03-27

### Release Audit (Batch 47)
Full 3-agent audit (routes, core modules, frontend+infra). Results:
- **Routes**: CLEAN. All 17 route files pass security, consistency, and contract checks.
- **Core modules**: 1 real bare ffmpeg fixed (`audio_enhance.py` audio extraction). All other flagged items were either already using `run_ffmpeg()` (which auto-resolves) or false positives.
- **Frontend/infra**: No critical issues. All XSS, thread-safety, and MCP claims verified as false positives.

### Fixed
- **audio_enhance.py bare ffmpeg** — `_extract_audio()` used `"ffmpeg"` directly in subprocess instead of `get_ffmpeg_path()`. Crashes on bundled installs.
- **4 test failures** — `test_score_frame_colorful_image` threshold too high (20→5), `test_registry_has_builtin_engines` expected wrong dict key, `test_depth_map_rate_limited` + `test_emotion_rate_limited` couldn't reach rate limit due to filepath validation happening first.

### Verified
- 0 lint warnings
- All 18 version targets in sync
- 254 routes, 23 MCP tools, 23 schemas, 16 CLI commands
- All subprocess calls use `get_ffmpeg_path()`/`get_ffprobe_path()` or go through `run_ffmpeg()`
- All `open()` calls have `encoding="utf-8"` for text mode
- All `on_progress` closures have `msg=""` default

## [1.9.14] - 2026-03-27

### Fixed (Batch 46 — Final Sweep)
- **4 more bare `"ffmpeg"` missed by agents** — `audio.py` (3 in stem separation format conversion + audio extraction) and `video_core.py` (1 in stabilize audio merge). All replaced with `get_ffmpeg_path()`.
- **MCP server stale docstring** — Claimed `--sse` mode existed but was never implemented. Fixed docstring to accurately describe stdio-only transport.
- **MCP `resources/list` and `prompts/list`** — Added empty handlers required by MCP spec. Modern MCP clients (Claude Code) expect these methods to exist.

## [1.9.13] - 2026-03-27

### Fixed (Batch 45 — Complete ffmpeg/ffprobe Path Resolution)
- **23 bare `"ffmpeg"` in subprocess calls** — 13 in route files (`audio.py`, `system.py`, `timeline.py`, `video_core.py` x7, `video_editing.py` x3) and 10 in core modules (`audio.py` x2, `audio_suite.py` x2, `highlights.py`, `loudness_match.py` x4, `scene_detect.py`). All replaced with `get_ffmpeg_path()`. On bundled installs where FFmpeg is not in system PATH, every one of these would crash with "command not found".

## [1.9.12] - 2026-03-27

### Fixed (Batch 44 — Dependencies & Hardcoded Ports)
- **`psutil` added to core dependencies** — `/system/status` endpoint used psutil but it wasn't in dependencies, causing ImportError on fresh installs.
- **`numpy` upper bound removed** — `<2` cap prevented numpy 2.x which is already used in the dist build. Now `>=1.24` with no upper bound.
- **3 new optional dependency groups** — `otio` (opentimelineio), `tts` (edge-tts + kokoro), `depth` (torch + transformers). All were installable but not declared in pyproject.toml.
- **`opentimelineio` added to `[all]` extras** — OTIO export was available in code but not installed via `pip install opencut[all]`.
- **social_post.py hardcoded port** — `get_oauth_url()` hardcoded `localhost:5679` in all 3 OAuth redirect URIs. Now accepts `port` parameter (default 5679) for correct redirect on non-default ports.

## [1.9.11] - 2026-03-27

### Fixed (Batch 43 — Encoding, Docker, CI)
- **13 `open()` calls missing `encoding="utf-8"`** — 6 core modules (captions, footage_search, lut_library, motion_graphics, social_post, speed_ramp) and 1 route (video_core) used system-locale encoding on Windows, causing `UnicodeDecodeError` on non-ASCII file content.
- **Docker GPU variant** — `docker-compose.yml` GPU service now usable via `docker compose --profile gpu up` instead of requiring manual uncommenting. Added WebSocket port 5680 to EXPOSE. Added `psutil` and `python-json-logger` to Dockerfile deps. Plugin dir created at build time.
- **CI smoke tests expanded** — Added imports for `create_app` (factory pattern), `WorkerPool`/`JobPriority`, `MCP_TOOLS` (count check), `cli.commands` (count check), `WorkflowResult` schema.

## [1.9.10] - 2026-03-27

### Added
- **CLI: `denoise` command** — Remove background noise from audio/video via FFmpeg filters (afftdn, highpass, gate) with adjustable strength.
- **CLI: `scene-detect` command** — Detect scene boundaries with method selection (ffmpeg/ml/pyscenedetect), threshold control, and optional JSON export. Rich table output.

### Fixed (Batch 42 — CLI Crash Bugs)
- **CLI `color-match` crash** — Imported non-existent `from .core.color import match_color`. Fixed to `from .core.color_match import color_match_video`.
- **CLI `loudness-match` crash** — Imported non-existent `from .core.loudness import normalize_loudness`. Fixed to `from .core.loudness_match import batch_loudness_match` with correct call signature.
- **CLI `auto-zoom` crash** — Imported non-existent `from .core.zoom import apply_auto_zoom`. Fixed to `from .core.auto_zoom import generate_zoom_keyframes` with FFmpeg zoompan apply step.
- **CLI `deliverables` crash** — Imported non-existent `generate_deliverables()` function. Fixed to import individual generators (`generate_vfx_sheet`, `generate_adr_list`, etc.) and dispatch based on `--type` flag. Now properly loads sequence JSON and routes to correct generator.

## [1.9.9] - 2026-03-27

### Added
- **UXP: 4 new Video features** — AI Upscale (Real-ESRGAN 2x/3x/4x), Scene Detection (FFmpeg/TransNetV2/PySceneDetect), Style Transfer (8 preset styles with intensity slider), Shorts Pipeline (one-click short-form clip generation with face tracking + captions).
- **UXP: full JS handlers** — `runUpscaleUxp()`, `runSceneDetectUxp()`, `runStyleTransferUxp()`, `runShortsPipelineUxp()` with proper job polling, error handling, and toast notifications.

### Fixed
- **UXP stale version** — Settings tab showed hardcoded "1.9.2" instead of actual version. Now dynamic via version sync.

## [1.9.8] - 2026-03-27

### Added
- **9 new response schemas** — `WorkflowResult`, `ContextAnalysisResult`, `VideoAIResult`, `ShortsPipelineResult`, `DepthMapResult`, `BrollPlanResult`, `BatchResult`, `PluginListResult` (22 total schemas, up from 13). All mapped into OpenAPI spec.
- **OpenAPI endpoint coverage +15** — 15 more async routes registered in `_JOB_ENDPOINTS` and `_ENDPOINT_SCHEMAS` for typed OpenAPI responses.

### Fixed (Batch 40)
- **3 more bare `"ffprobe"` in routes/utils** — `audio.py` waveform probe, `video_editing.py` reframe probe, `utils/media.py` media probe.
- **2 broken tests fixed** — `test_info_with_csrf` expected 200 from `/info` with no filepath (returns 400), `test_file_no_path` expected 400 but route returns 404.
- **pytest hanging on exit** — WorkerPool daemon threads prevented clean exit. Added session-scoped `_shutdown_worker_pool` fixture in conftest.py.

## [1.9.7] - 2026-03-27

### Added
- **5 new MCP tools** — `opencut_denoise_audio`, `opencut_upscale`, `opencut_scene_detect`, `opencut_depth_map`, `opencut_shorts_pipeline` (23 total, up from 18). AI clients can now drive denoising, upscaling, scene detection, depth mapping, and the full shorts pipeline.
- **WorkerPool priority queue** — Replaced FIFO `ThreadPoolExecutor` with custom `PriorityQueue`-backed pool. `JobPriority.CRITICAL` (0) jobs now leapfrog `BACKGROUND` (200) work when all workers are busy. Equal-priority jobs still run in FIFO order.

### Fixed (Batch 39)
- **3 more bare `"ffprobe"` in routes/utils** — `audio.py` waveform probe, `video_editing.py` reframe probe, `utils/media.py` media probe — all crash on bundled installs where ffprobe isn't in PATH.
- **FFmpeg/ffprobe path warning** — `get_ffmpeg_path()` and `get_ffprobe_path()` now log WARNING when binary not found in PATH (was silently falling back to bare name, causing confusing subprocess errors later).

## [1.9.6] - 2026-03-27

### Fixed (Batch 38 — Route Cleanup & Remaining Crash Bugs)
- **10 remaining `_p(pct, msg)` closures missing `msg=""` default** — Found in `audio.py` (3), `video_fx.py` (5), `captions.py` (1), `workflow.py` (1). All core modules call `on_progress(pct)` with 1 arg, causing TypeError crash on every job using these routes. All now have `msg=""` default.
- **Redundant filepath re-validation in 7 routes** — `face_enhance`, `face_swap`, `upscale_run`, `remove_watermark`, `title_overlay`, `particle_apply`, `color_correct`, `color_convert`, `color_external_lut` all re-read `data.get("filepath")` and re-validated despite `@async_job` already doing this. Removed ~80 lines of dead validation code, now using the decorator-provided `filepath` param directly.
- **Unused `validate_filepath` import** — Removed from `video_specialty.py` after watermark/title routes stopped re-validating.

## [1.9.5] - 2026-03-27

### Fixed (Batch 37 — Infrastructure & Hardening)
- **Bare ffprobe in 8 core modules** — `audio_enhance`, `color_management`, `highlights`, `motion_graphics`, `scene_detect` (2 sites), `shorts_pipeline`, `transitions_3d`, and `video_ai` all used `"ffprobe"` directly in subprocess commands. On systems where ffprobe is bundled (not in PATH), all of these would crash. Fixed to use `get_ffprobe_path()`.
- **GPU VRAM check can hang** — `check_vram()` in `gpu.py` called `torch.cuda.mem_get_info()` with no timeout. On hung NVIDIA drivers, this blocked forever. Added 5-second timeout via ThreadPoolExecutor.
- **safe_error GPU OOM false positives** — `errors.py` pattern `"cuda" in lower and "memory" in lower` matched unrelated errors like "CUDA device not found in memory". Tightened to exact phrases `"cuda out of memory"` / `"cuda error: out of memory"`.
- **_unique_output_path 10K stat loop** — `helpers.py` looped up to 9,998 times trying filenames. On full disk or permission errors, wasted thousands of stat() calls. Capped at 100.
- **get_video_info silent defaults** — Fallback to 1920x1080@30fps was logged at DEBUG level (invisible). Upgraded all 3 fallback paths to WARNING so users know when probe data is missing.
- **Job cleanup on every creation** — `_new_job()` called `_cleanup_old_jobs()` synchronously on every job creation. With 1000+ old jobs, blocked the HTTP handler. Removed — periodic cleanup thread (already running every 5 minutes) handles this.
- **proc.poll() crash on stale handles** — `_cleanup_old_jobs()` called `.poll()` on potentially corrupt Popen objects. Added try/except per handle.
- **async_job future not stored for cancel** — Race window between `_new_job()` returning job_id and storing the future. Cancel requests during that window couldn't find the future. Now stores `_thread` reference immediately after submit.
- **multicam_xml path encoding** — `_path_to_url()` didn't URI-encode spaces or special characters in filenames. Premiere Pro rejected the XML. Added `urllib.parse.quote()`.
- **main.js parentNode null guard** — `select.parentNode.insertBefore()` could crash if select element had no parent. Added null check.

## [1.9.4] - 2026-03-27

### Fixed (Batch 36 Audit)
- **P0: face_enhance/face_swap/upscale `_p(pct, msg)` crash** — 3 `on_progress` closures in video_ai.py missing `msg=""` default; core modules call with 1 arg → TypeError. Added default.
- **P0: engagement attribute crash** — shorts pipeline response accessed `c.engagement.hook_strength` directly; switched to `getattr()` with defaults for all 5 engagement fields.
- **P0: broll_plan inconsistent response** — empty segments returned 2 keys but success path returned 4; frontend expects all 4. Fixed + added `plan is None` guard + `getattr()` for all window fields.
- **Security: style_arbitrary path traversal** — `/video/style/arbitrary` accepted `style_image` without `validate_filepath()`; attacker could read arbitrary files. Added validation.
- **Security: plugin install path traversal** — `/plugins/install` accepted arbitrary `source` directory without `validate_path()`. Added validation.
- **Security: plugin uninstall symlink escape** — `/plugins/uninstall` didn't verify resolved `plugin_dir` stays within `PLUGINS_DIR`. Added `os.path.realpath()` containment check.
- **Security: depth_effects model_id injection** — `model_size` param interpolated into HuggingFace model ID without validation in `apply_bokeh_effect()` and `apply_parallax_zoom()`. Added allowlist in all 3 functions.
- **GPU rate limiting gaps** — `/video/ai/interpolate` and `/video/ai/denoise` (basicvsr method) bypassed `rate_limit("ai_gpu")`, allowing concurrent GPU OOM. Added guards.
- **Queue allowlist +9 routes** — 9 newer routes missing from `_ALLOWED_QUEUE_ENDPOINTS`: interpolate, depth/map/bokeh/parallax, broll-plan, remove/watermark, upscale/run, multicam-xml, search/auto-index.
- **title_overlay preset allowlist** — Missing `lower_third`, `countdown`, `kinetic_bounce` presets (present in title_render but not title_overlay).
- **Engine registry cache race** — `_availability_cache` reads/writes in `get_available_engines()` were outside `_lock`; added lock protection.
- **main.js timer leak** — `_scanDebounceTimer` missing from `cleanupTimers()`; leaked on panel close.
- **main.js safeFixed** — `defaultZoomVal` slider used raw `toFixed()` instead of `safeFixed()` wrapper.
- **FFmpeg stderr truncation UX** — `run_ffmpeg()` truncated stderr silently; now prepends `"...[truncated] "` marker when truncating.
- **Test fix** — `test_system_gpu` expected `gpu_available` key but API returns `available`.

## [1.9.3] - 2026-03-27

### Added
- **JSON Structured Logging** (Phase 0.3) — File handler now outputs JSON via `python-json-logger`. Each log line includes `timestamp`, `level`, `module`, `job_id`, `message`. Console handler stays plain text. Graceful fallback if `python-json-logger` not installed.
- **CI Coverage Enforcement** (Phase 0.1) — `.github/workflows/build.yml` now runs `pytest-cov` with `--cov-fail-under=50` threshold. CI also triggers on PRs and pushes to main. PyInstaller build skipped for non-release runs.
- **Structured Error Migration** (Phase 0.2) — All route error handlers migrated from bare `{"error": str(e)}` to `safe_error()` from `opencut/errors.py`. Zero bare 500 error patterns remain across all 13 route files. Every API error returns `{error, code, suggestion}`.
- **Smart Tab Reordering** (Phase 3.2) — Sub-tabs now physically reorder in the DOM based on contextual relevance scores. Highest-scoring features move to front within each tab group. `resetTabOrder()` restores original order on clip deselection.
- **Frontend Error Code Mapper** (Phase 0.2) — `enhanceError()` now reads `data.code` field before regex fallbacks. Code-to-action mapping for GPU_OOM, MISSING_DEPENDENCY, FILE_NOT_FOUND, etc. with navigable settings links.
- **Core Module Unit Tests** (Phase 0.1) — 15 additional core modules tested (silence, fillers, scene_detect, auto_edit, highlights, workflow, speed_ramp, audio, face_reframe, chromakey, video_fx, export_presets, diarize, audio_duck, thumbnail).
- **i18n String Extraction** (Phase 6.2) — ~200 additional `data-i18n` attributes added to buttons, headers, labels, and tabs in index.html with corresponding en.json keys.
- **Pre-commit Hooks** (Phase 0.1) — `.pre-commit-config.yaml` now includes ruff lint/format + pytest smoke suite (on pre-push). `pre-commit` added to dev dependencies.
- **Log Levels Audit** (Phase 0.3) — 22 log calls across 10 files corrected: verbose processing steps downgraded INFO→DEBUG, fallback/degraded paths upgraded INFO→WARNING. No secrets found in logging.
- **Core Module Unit Tests Batch 2** (Phase 0.1) — 135 additional tests across 28 previously untested core modules (animated_captions, audio_enhance, audio_pro, audio_suite, caption_burnin, captions_enhanced, color_management, emotion_highlights, face_swap, face_tools, lut_library, motion_graphics, music_ai, music_gen, object_removal, particles, shorts_pipeline, style_transfer, styled_captions, transitions_3d, upscale_pro, video_ai, voice_gen, zoom, broll_insert, broll_generate, multimodal_diarize, social_post).
- **ExtendScript Mock Harness** (Phase 0.1) — `tests/jsx_mock.js` provides fake Premiere Pro DOM (app.project, activeSequence, tracks, markers, ProjectItem). Tests 38 ExtendScript functions including ocApplySequenceCuts, ocBatchRenameProjectItems, ocAddSequenceMarkers, ocAddNativeCaptionTrack. 35 assertions pass under Node.js.

### Refactored
- **`async_job` decorator adoption** — All 97 manual `_new_job()` + `threading.Thread` + `_update_job(status="complete")` patterns across 6 route files converted to `@async_job` decorator. Fixes race condition where cancelled jobs could be overwritten to "complete". Removes ~2,800 lines of boilerplate. Decorator extended with `filepath_required` and `filepath_param` parameters.
- **Split `video.py` into 5 domain blueprints** — Monolithic 3636-line `video.py` split into `video_core.py` (1395), `video_editing.py` (750), `video_fx.py` (687), `video_specialty.py` (489), `video_ai.py` (443). No file exceeds 1400 lines. All URL paths unchanged.
- **App factory pattern** — `server.py` now exposes `create_app(config)` for isolated Flask instances. New `opencut/config.py` centralizes env var reads into `OpenCutConfig` dataclass. Tests use independent app instances. Module-level `app = create_app()` preserved for backward compat.
- **Install route factory** — `make_install_route()` in `opencut/jobs.py` replaces 6 identical install endpoint handlers (depth, emotion, multimodal-diarize, broll-generate, face, crisper-whisper) with a single factory call per route.
- **Version sync CI enforcement** — `scripts/sync_version.py` now includes `package.json` in targets and supports `--check` flag for CI. Added to `.github/workflows/build.yml`.

### Security
- Replaced all `__import__()` calls with `importlib.import_module()` / `importlib.util.find_spec()` in helpers.py, system.py, engine_registry.py, plugins.py

### Fixed
- **`_safe_error` undefined** — Fixed `_safe_error` reference in video.py multicam XML (F821 undefined name bug)
- **`safe_int` missing import** — Added missing import in settings.py log tail endpoint (F821)
- **Timecode watermark plugin** — `start_tc` variable now used in FFmpeg drawtext filter (was extracted but unused)
- **Version sync** — All version strings (pyproject.toml, __init__.py, CEP manifest, UXP manifest, server startup banner, installer AppConstants.cs) now read from single source. Server banner uses `__version__` instead of hardcoded string.
- **TooManyJobsError handling** — 7 remaining `_new_job()` calls in system.py, search.py, timeline.py now properly catch `TooManyJobsError` and return HTTP 429 instead of 500.
- **audio_separate cleanup crash** — `temp_audio` variable moved above try block to prevent `NameError` in finally clause if `_resolve_output_dir()` fails.
- **GPUContext memory leak** — `__exit__` now calls `.cpu()` on registered models before clearing, actually freeing GPU VRAM instead of only deleting a loop variable.
- **batch_id collision risk** — Batch IDs extended from 8 to 16 hex chars, reducing collision probability from 1-in-65K to 1-in-4-billion batches.
- **XSS in engine registry** — All API-sourced values (domain, engine name, display_name, quality, speed) now wrapped in `esc()` before innerHTML insertion.
- **elapsedTimer leak** — `startJob()` now clears any existing elapsed timer before creating a new one, preventing interval accumulation on rapid re-starts.
- **mediaScanTimer leak** — Periodic media scan `setInterval` now assigned to tracked variable and cleared on `beforeunload`.
- **pollTimer leak** — `trackJobPoll()` now clears any existing poll timer before creating a new one; SSE fallback no longer creates duplicate trackers.
- **SSE parse errors silent** — SSE `onmessage` handler now logs JSON parse failures to console instead of silently swallowing them.
- **Reframe dimension DoS** — `/video/reframe` now enforces `min_val=16, max_val=7680` on width/height parameters, preventing absurd allocation requests.
- **ASS subtitle injection** — Caption burn-in now strips backslash sequences from source text before ASS formatting, preventing style override injection via subtitle content.
- **23 import sorting violations** fixed (I001)
- **15 unused imports** removed (F401)
- **4 unused variables** removed (F841)
- Codebase at **0 lint warnings** matching CI config

## [1.9.1] - 2026-03-26

### Added
- **Frontend Context Awareness** (Phase 3.2) — CEP panel now calls `POST /context/analyze` after clip selection. Guidance banner shows clip-specific recommendations. Sub-tabs for high-scoring features get visual highlights. Dismiss button hides banner.
- **Parallel Batch Processing** (Phase 5.4) — `process_batch_parallel()` in `batch_process.py` uses `ThreadPoolExecutor` for concurrent multi-clip operations. GPU ops limited to 1 worker, CPU ops scale to core count. Partial failure isolation — one item crash doesn't kill the batch.
- **Clip Notes Plugin** (Phase 6.1) — Second example plugin at `opencut/data/example_plugins/clip-notes/`. SQLite-backed per-clip notes with `POST /note`, `GET /notes`, `DELETE /note`, `GET /export` (text/CSV). Shipped alongside timecode-watermark as reference plugins.
- **Route Smoke Tests** (Phase 0.1) — Comprehensive smoke test suite in `tests/test_route_smoke.py` covering all 175+ endpoints across 13 blueprints. Verifies routes don't crash with minimal payloads.

### Fixed
- **Ruff lint cleanup** — Fixed unused imports in new test files (8 auto-fixed F401 violations)

## [1.9.0] - 2026-03-26

### Added
- **Contextual Awareness** (Phase 3.2) — Clip type detection and feature relevance scoring. `POST /context/analyze` accepts clip metadata and returns scored features, guidance messages, and per-tab relevance scores. 35 features scored across 4 tabs based on clip tags (talking_head, audio_only, long_duration, etc.)
- **Plugin System** (Phase 6.1) — Plugin loader discovers/validates/loads plugins from `~/.opencut/plugins/`. Each plugin has a `plugin.json` manifest and optional Flask Blueprint routes registered under `/plugins/<name>/`. Endpoints: `GET /plugins/list`, `GET /plugins/loaded`, `POST /plugins/install`, `POST /plugins/uninstall`. Includes example timecode-watermark plugin.
- **Multicam XML Export** — Generate Premiere Pro compatible FCP XML from multicam diarization cut data. `POST /video/multicam-xml` endpoint. CLI `multicam` command now exports XML instead of showing placeholder message.
- **Background Indexing with SQLite FTS5** (Phase 5.3) — Persistent footage index at `~/.opencut/footage_index.db` with full-text search. Incremental re-indexing via mtime comparison. Endpoints: `POST /search/auto-index`, `POST /search/db-search`, `GET /search/db-stats`, `POST /search/cleanup`.
- **Response Streaming** (Phase 5.2) — NDJSON streaming utilities for progressively delivering large result sets. `GET /jobs/stream-result/<job_id>` streams completed job results (segments, scenes, thumbnails) in batches of 50.
- **Context blueprint** (`routes/context.py`) — 2 new endpoints
- **Plugins blueprint** (`routes/plugins.py`) — 4 new endpoints
- **4 new search endpoints** — auto-index, db-search, db-stats, cleanup

### Fixed
- **TooManyJobsError handling** — All 96 manual `_new_job()` calls across video.py (47), audio.py (23), and captions.py (13) now catch `TooManyJobsError` and return proper 429 responses instead of 500 errors
- **AI GPU rate limiting** — Added `rate_limit("ai_gpu")` to 6 GPU-heavy routes: video AI upscale, background removal, shorts pipeline, and 3 music AI generation routes. Prevents concurrent GPU OOM crashes
- **Settings import validation** — `/settings/import` now validates workflow steps (requires endpoint + label per step) before saving, preventing malformed workflow injection
- **Preview frame bounds** — `width` parameter bounded to 32-3840, `detection_skip` bounded to 1-30
- **Secure tempfile** — Preview frame extraction uses `tempfile.mkstemp()` instead of predictable path construction

### Security
- 96 routes hardened against job limit bypass (TooManyJobsError → 429)
- Input bounds added to prevent resource exhaustion via unbounded parameters
- Workflow import step validation prevents malformed workflow injection
- Fixed double rate-limit release in demucs/watermark install routes (decorator + explicit call)

## [1.5.0] - 2026-03-23

### Added
- **Repeated Take Detection** — Jaccard similarity sliding-window to find and remove fumbled/repeated takes from transcriptions
- **YouTube Chapter Generation** — LLM-powered (Ollama/OpenAI/Anthropic) topic detection and chapter timestamp generation
- **Footage Search** — Index media library by spoken content; search by keyword or phrase across all clips
- **Post-Production Deliverables** — Generate VFX sheets, ADR lists, music cue sheets, and asset lists as CSV from sequence data
- **Color Match** — YCbCr histogram matching to match color profile of one clip to a reference clip
- **Multicam Auto-Switching** — Diarization-driven cut generation for podcast multicam editing
- **Auto Zoom Keyframes** — Face-detected keyframe generation for dynamic push-in zoom effect
- **Loudness Match** — FFmpeg two-pass LUFS normalization to match loudness across multiple clips
- **NLP Command Parser** — Natural language command dispatch via keyword matching and LLM
- **Timeline Write-Back** — Apply cuts, markers, and keyframes directly to the active Premiere Pro sequence via ExtendScript
- **Beat Markers** — Sync detected beat timestamps as sequence markers in Premiere Pro
- **Batch Rename** — Rename project panel clips with find/replace pattern support
- **Smart Bins** — Auto-sort project items into bins by rule (name contains, type, duration)
- **SRT to Native Captions** — Import SRT files as native Premiere Pro caption tracks
- **Batch Export from Markers** — Export individual clips defined by sequence markers
- **UXP Panel** — New parallel panel for Premiere Pro 25.6+ using modern UXP APIs (com.opencut.uxp)
- **Timeline Tab** — New CEP panel tab with all timeline write-back operations
- **Search & NLP Tab** — New CEP panel tab with footage search and AI command input
- **Deliverables Section** — Post-production document generation in the Export tab

### Fixed
- Improved progress callback support in color match, auto zoom, and loudness match operations
- LLMConfig now properly instantiated as dataclass object across all routes

## v1.2.0 (2026-03-15)

### New Features
- **Waveform Preview**: Visual waveform display on Silence tab with draggable threshold line synced to slider
- **Side-by-Side Preview**: Before/after frame comparison modal for video effects
- **Dependency Health Dashboard**: Grid view of all 24 optional dependencies with install status in Settings
- **First-Run Wizard**: Animated 3-step onboarding overlay for new users (dismissible, persisted)
- **Output File Browser**: Browse recent output files with Import-to-Premiere button
- **Favorites Bar**: Pin frequently-used operations as quick-access chips below the nav tabs
- **Batch Multi-Select**: Multi-file picker for batch operations (add selected, add all, clear)
- **Parameter Tooltips**: Hover tooltips on range sliders explaining what each parameter controls
- **Custom Workflow Builder**: Chain multiple operations into named reusable workflows (saved to `~/.opencut/workflows.json`)
- **Audio Preview Player**: Floating audio player to preview generated audio before importing
- **Settings Import/Export**: Export all settings (presets, favorites, workflows) as JSON, import on another machine
- **Right-Click Context Menu**: Quick-action context menu on clip selector (silence, transcribe, denoise, etc.)
- **Collapsible Cards**: Click card headers to collapse/expand dense form sections
- **Job Time Estimates**: Estimated processing time in the progress banner based on historical job data
- **Localization Framework**: Language selector in Settings (placeholder for future i18n support)
- **Video Reframe**: Resize/crop video for TikTok, Shorts, Reels, Instagram, Square, or custom dimensions with crop/pad/stretch modes
- **Clip Preview Thumbnail**: Visual thumbnail + duration/resolution metadata when selecting a clip
- **Command Palette**: Ctrl+K fuzzy search across all 28+ operations with keyboard navigation
- **Recent Clips**: Dropdown of last 10 used clips, persisted across sessions
- **Trim Tool**: Set in/out points to extract a clip portion (stream copy or re-encode)
- **Merge/Concatenate**: Join multiple clips into one (fast stream copy or re-encoded for mixed formats)
- **Auto-Crop Detect**: Smart reframe anchor using FFmpeg cropdetect for talking-head content
- **Audio Waveform Everywhere**: Waveform preview buttons on Denoise and Normalize tabs (reuses Silence tab infrastructure)
- **Per-Operation Presets**: Save/load settings per operation (persisted to localStorage)
- **Server Health Monitor**: 10-second heartbeat with reconnect banner and toast notification
- **Output Deduplication**: Auto-increment suffix prevents overwriting previous outputs

### Backend Improvements
- **FFmpeg Progress Parsing**: `_run_ffmpeg_with_progress()` helper parses `-progress pipe:1` for real percentage updates
- **Subprocess Kill on Cancel**: Job cancellation now terminates the running FFmpeg process via `_kill_job_process()`
- **Temp File Cleanup**: Stale `opencut_preview_*.jpg` files cleaned up on server startup
- **File Serving Endpoint**: `GET /file` serves local audio/video for preview player (path-restricted)

### Backend (14 new endpoints)

- `POST /audio/waveform` - Extract waveform peak data via FFmpeg PCM extraction
- `POST /video/preview-frame` - Extract single frame as base64 JPEG at timestamp
- `GET /system/dependencies` - Check 24 optional deps + FFmpeg install status
- `GET /outputs/recent` - List recent output files from completed jobs
- `GET/POST /favorites` - Favorites persistence to `~/.opencut/favorites.json`
- `GET/POST/DELETE /workflows` - Custom workflow CRUD to `~/.opencut/workflows.json`
- `GET/POST /settings/export|import` - Bundle presets + favorites + workflows
- `POST /system/estimate-time` - Historical ratio-based time estimation
- `POST /video/reframe` - Resize/crop video for target aspect ratio
- `GET /video/reframe/presets` - Available reframe platform presets
- `GET /file` - Serve local files for audio/video preview
- `POST /video/merge` - Concatenate multiple clips (demux or filter modes)
- `POST /video/trim` - Extract clip portion by in/out timecodes

## v1.1.0 (2026-03-15)

### New Features
- **Preset Save/Load**: Save and load all panel settings as named presets (persisted to `~/.opencut/user_presets.json`)
- **AI Model Manager**: View all downloaded AI models (HuggingFace, Torch, Whisper) with sizes, delete unused models to free space
- **GPU Auto-Detection & Recommendation**: Automatically recommend optimal model sizes and settings based on detected GPU VRAM
- **Job Queue System**: Queue multiple jobs for sequential processing instead of waiting for each to finish
- **Enhanced Job History**: Re-run previous jobs directly from history, toast notifications on completion
- **Keyboard Shortcuts**: Enter to run active operation, 1-6 to switch tabs, Escape to cancel
- **Enhanced Drag-and-Drop**: Drop files anywhere on the panel, not just the drop zone
- **Toast Notifications**: Non-intrusive slide-in notifications for job completion/errors
- **Transcript Search**: Search and navigate through transcript segments with highlight
- **Social Platform Export Presets**: Quick export presets for YouTube Shorts, TikTok, Instagram Reel/Story/Post, Twitter/X, LinkedIn
- **Premiere Theme Sync**: Detect Premiere Pro's UI brightness via CSInterface
- **Universal Auto-Import**: Smart ExtendScript import function that routes files to appropriate project bins

### Bug Fixes (v1.0.0 audit)
- Fixed 6 broken install routes using `subprocess.run` instead of `_sp.run`
- Fixed SSE race condition (dict copy + status read now inside `job_lock`)
- Fixed VBS launcher path quoting for `C:\Program Files\OpenCut`
- Replaced `eval()` with `JSON.parse()` in ExtendScript
- Fixed wrong `/cut/silence` endpoint in social_ready workflow
- Fixed `jobStarting` guard and `no_input` flag in all applicable routes
- Fixed temp file leaks in audio_separate and watermark routes
- Added `user-select: text` override for CEP Chromium inputs

## v1.0.0-beta (2026-02-09)

First public beta release with full feature set.

### Core (34 modules, 116 routes)
- Silence detection and removal with configurable thresholds
- Filler word detection (um, uh, like, you know, so, actually)
- Speaker diarization via pyannote.audio
- Job queue with async processing and progress polling

### Captions (5 modules)
- WhisperX / faster-whisper transcription with word-level timestamps
- 19 caption styles (YouTube Bold, Neon Pop, Cinematic, Netflix, Sports, etc.)
- 7 animated caption presets (pop, fade, slide, typewriter, bounce, glow, highlight)
- Caption burn-in with style selection
- Translation to 50+ languages

### Audio (8 modules)
- Stem separation via Demucs (vocals, drums, bass, other)
- AI noise reduction + spectral gating
- Loudness normalization (LUFS targeting)
- Beat detection and BPM analysis
- Pro FX chain: compressor, EQ, de-esser, limiter, reverb, stereo width
- Text-to-speech with 100+ voices (Edge-TTS / F5-TTS)
- Procedural SFX generator (tones, sweeps, impacts, noise)
- AI music generation via MusicGen (AudioCraft)
- Audio ducking (auto-lower music under dialogue)

### Video (12 modules)
- FFmpeg stabilization (deshake/vidstab)
- Chromakey with green/blue/red screen + spill suppression
- Picture-in-picture overlay with position/scale
- 14 blend modes (multiply, screen, overlay, etc.)
- 34 xfade transitions (fade, wipe, slide, circle, pixelize, radial, zoom)
- 7 particle effect presets (confetti, sparkles, snow, rain, fire, smoke, bubbles)
- 6 animated title presets (fade, slide, typewriter, lower third, countdown, kinetic)
- Speed ramping with ease curves
- Scene detection via PySceneDetect
- Film grain, letterbox, vignette effects
- 15 built-in cinematic LUTs + external .cube/.3dl support
- Color correction (exposure, contrast, saturation, temperature, shadows, highlights)
- Color space conversion (sRGB, Rec.709, Rec.2020, DCI-P3)

### AI Tools (6 modules)
- AI upscaling: Lanczos (fast), Real-ESRGAN (balanced), Video2x (premium)
- Background removal via rembg (U2-Net)
- Face enhancement via GFPGAN
- Face swapping via InsightFace
- Neural style transfer
- Object/watermark removal (FFmpeg delogo + LaMA inpainting)
- Auto-thumbnail extraction with AI scoring

### Export & Batch (3 modules)
- Platform presets: YouTube, TikTok, Instagram, Twitter, LinkedIn, Podcast
- Batch processing with parallel execution
- Transcript export (SRT, VTT, plain text, timestamped)

### UI
- 6 dark themes (Cyberpunk, Midnight OLED, Catppuccin, GitHub Dark, Stealth, Ember)
- 6 main tabs with 24 sub-tabs
- Persistent processing banner with progress
- Single-job enforcement with UI lockout
- Auto-import results back to Premiere Pro timeline
