# Changelog

Notable changes from the June 2026 hardening/audit pass. The authoritative
record also lives in the git commit messages.

## [Unreleased]

### Fixed - Harden three low-cost reliability/security defaults

- `/health` now exposes the bootstrap CSRF token on an allowlist basis
  (no-Origin, exact same-origin, or a configured CORS origin) instead of a
  two-value denylist, so an arbitrary cross-origin page can no longer harvest
  the loopback mutation token.
- `check_disk_space` now probes the nearest existing ancestor directory and
  fails **closed** on a local-path probe error (surfacing a full/inaccessible
  disk as a clean rejection) while still failing open for un-probeable
  network/UNC paths.
- Job admission is clamped to the worker-pool size, so raising
  `OPENCUT_MAX_CONCURRENT_JOBS` above the pool's worker count can no longer
  over-admit "running" jobs onto a queue that cannot execute them.

### Added - Wire UXP batch rename and smart bins to the host bridge

- The UXP timeline panel's Batch Rename and Smart Bins buttons now run their
  already-implemented `PProBridge` host actions instead of showing a "runs
  through CEP" handoff. Buttons enable only when the UXP host bridge is
  available (with an honest disabled reason otherwise). Batch rename previews
  the affected items on first click and applies on the second, records an
  inverse rename so the same button offers a one-click undo, and reports partial
  failures; smart bins report the number of bins created.
- Extracted the pure rename/bin logic (`expandRenamePattern`,
  `computeInverseRenames`, `buildSmartBinRules`, `summarizeRenamePreview`) into
  `uxp-utils.js` with a new `uxp-batch-ops` vitest suite, and added the
  supporting UXP locale strings (en/es).

### Fixed - Semantic project-facts documentation gate

- Added `test_project_facts`, a gate that derives facts from the authoritative
  source and fails when a committed doc/comment contradicts it: UXP tab
  count/names (README vs `extension/com.opencut.uxp/index.html`), the CSRF
  endpoint/header (UXP `main.js` comment vs implementation), and the Vite major
  (`docs/NODE_ADVISORIES.md` vs panel `package.json`).
- Fixed the contradictions it surfaced: the README under-counted the UXP panel
  as 8 tabs (it has 9 — the Agent tab was omitted), and the UXP `fetchCsrf`
  comment still described a removed `/csrf` endpoint instead of the live
  `/health` + `X-OpenCut-Token` flow. Repaired the stale `test_check_doc_sizes`
  suite left broken by the earlier CLAUDE.md size-check removal.

### Fixed - Feature readiness proves implementation, not just dependencies

- Added `opencut.core.stub_scan`, which statically (via AST) identifies
  `opencut.core` modules whose entrypoints still raise `NotImplementedError`.
  `FeatureRecord` gained an `impl_module` link and an implementation-aware
  `resolved_state()`/`state_reason()`: a feature whose adapter is an unfinished
  stub now stays `stub` even when its optional dependency is installed, so
  readiness never advertises a route that ends in `NotImplementedError`.
- Corrected three features that reported `available` while backed by stub
  adapters (`captions.nemo-asr`, `video.relight.iclight`,
  `video.upscale.seedvr2`). Panels and MCP now see the same state and reason via
  `as_dict`. New `test_feature_impl_readiness` adds a release gate that fails if
  any `available` feature is backed by a terminal-stub module.

### Security - Establish a model-weight supply-chain boundary (CVE-2026-24747)

- Raised the Torch floor to `>=2.10.0` and torchvision to `>=0.25.0` in the
  `depth` and `torch-stack` extras, closing CVE-2026-24747 (GHSA-63cw-57p8-fm3p)
  — a `torch.load(weights_only=True)` unpickler heap-corruption/RCE fixed in
  torch 2.10.0.
- Added `opencut.core.model_safety.safe_torch_load`, which scans pickle-format
  checkpoints with picklescan (`>=1.0.3`, now shipped in the `ai`/`ai-gpu`/
  `depth`/`torch-stack` extras) before loading and rejects flagged files;
  `.safetensors` skips scanning. Wired the three `torch.load` call sites
  (`face_swap`, `video_ai`, `model_quantization`) through it. New
  `test_model_safety` covers scan, rejection, safetensors skip, and graceful
  degradation when picklescan is absent.

### Security - Unify bounded, transactional archive ingestion

- Added `opencut.core.archive_safety`, a shared bounded/path-safe ZIP helper:
  member-count, expanded-byte (zip-bomb), and per-member size ceilings;
  path-traversal, absolute-path, and special-entry (symlink/device) rejection;
  bounded single-member reads; and staged-then-atomic extraction.
- `project_archive.restore_archive` now validates every member and extracts into
  a staging directory promoted atomically, so a rejected or failed restore
  leaves no partial destination. `lottie_import.info` reads the animation JSON
  through a hard byte ceiling, and the plugin installer delegates to the shared
  inspector (gaining symlink rejection it previously lacked). New
  `test_archive_safety` covers traversal, absolute paths, oversized members,
  compression bombs, symlink entries, cleanup, and atomic promotion.

### Security - Guard URL ingest and outbound fetches against SSRF

- Routed `opencut.core.url_ingest` through a shared connect-time SSRF guard in
  `opencut.core.url_safety`: the direct-download path now rejects
  private/loopback/link-local hosts (including numeric-encoding bypasses),
  resolves and re-checks the host before connecting, re-validates every redirect
  hop, honors local-only mode via `require_network_allowed`, enforces a
  configurable byte ceiling (`OPENCUT_MAX_INGEST_BYTES`, default 8 GiB), and
  verifies the downloaded bytes are real media (ffprobe) before caching.
- Added `open_validated_url`, `stream_to_file`, and `build_guarded_opener`
  helpers; webhook delivery now re-resolves the target and follows redirects
  only through hops that pass the guard. New `test_url_ingest_ssrf` covers
  private-IP, redirect-to-private, DNS-rebinding, oversized, non-media, and
  local-only-blocked cases.

### Security - Patch Click command-injection advisory (CVE-2026-7246)

- Raised the Click floor to `>=8.3.3,<9` across `pyproject.toml` and
  `requirements.txt`, and repinned `requirements-lock.txt` from `8.3.1` to
  `8.4.1`, closing CVE-2026-7246 / PYSEC-2026-2132 (command injection in
  `click.edit()`). Documented the floor raise in `docs/PYTHON_ADVISORIES.md`
  and added a `test_dependency_surface` regression asserting no lane admits the
  vulnerable range. `pip-audit -r requirements-lock.txt` is clean.

### Changed - Generated-doc release smoke gate

- Added a `generated-docs` release-smoke step that groups README product facts,
  MCP registry and extended-tool catalogues, model cards, and feature-readiness
  checks so stale generated documentation fails before push.

### Fixed - UXP Spanish locale warnings

- Corrected the remaining Spanish UXP locale diacritic warnings and promoted
  locale-lint warnings to `--check` failures, with a release-gate test that
  blocks future Spanish diacritic regressions.

### Fixed - FastAPI job WebSocket

- Replaced the `/ws/jobs` "coming soon" echo with a real job snapshot and
  command endpoint that mirrors the existing live-updates bridge contract for
  `ping`, `get_jobs`, `get_status`, `subscribe`, and `unsubscribe`.

### Fixed - Model readiness placeholder anchors

- Replaced dead `roadmap H*` model-card and feature-readiness install hints
  with stable generated-doc references, regenerated model/readiness artifacts,
  and added invariants that block future dead roadmap anchor references.

### Changed - MCP catalogue count drift guards

- Refreshed the MCP server guide to match the generated 86 curated and
  1,450 extended tool catalogues, removed stale bridge comment counts, and
  added tests that compare public MCP docs to the generated manifests.

### Changed - Social publish upload handoff

- Made `/publish/upload` return an explicit dry-run export-prep/upload-handoff
  response instead of implying a completed platform publish, and clarified the
  command-palette copy so direct OAuth uploads point to the Social Post flow.

### Fixed - Remote render-node trust boundary

- Redacted remote render-node API keys from register/list responses while
  preserving raw keys only in the local node registry file, and added remote
  node URL validation for schemes, embedded credentials, malformed ports, and
  local/private network targets before health checks or downloads run.

### Fixed - Resumable job invariant check

- Replaced the checkpointable route test's exact decorator string matching with
  decorator metadata parsing so resumable async jobs can carry additional kwargs
  without weakening the resumable-route guard.

### Fixed - CEP i18n catalog completeness

- Added the missing CEP command-palette and plugin-trust locale keys,
  restored the reverted public-gist confirmation key, and taught the i18n
  drift gate to count dynamic plugin summary label keys so runtime fallback
  strings cannot bypass the locale ledger.

### Changed - Plugin trust visibility

- Added a read-only `/plugins/trust` dashboard route and surfaced loaded,
  skipped, failed, lock-missing, quarantined, and cached marketplace plugin
  state in CEP and UXP Settings with capability badges and destructive-action
  confirmation contract copy.
- Hardened CEP translation-language dropdown rendering so list-shaped backend
  language capability payloads cannot break Settings initialization.

### Changed - Caption font fallback

- Caption burn-in filters now resolve real font files for known families and
  script-aware CJK fallbacks, fall back to a sanitized font-family argument
  instead of `fontfile=''`, and surface font-resolution warnings in CEP/UXP
  caption display controls.

### Changed - Local release policy guard

- Replaced stale GitHub Actions/CI wording in active installer wrappers with
  local release-smoke opt-in variables and added tests that prevent removed
  workflow claims from returning to active build surfaces.

### Changed - UX feature index readiness

- Enriched command-palette feature index and search results with route
  readiness, route validity, runnable state, route methods, and disabled-state
  explanations sourced from the generated route manifest.
- Corrected the C2PA command action to the live `/video/c2pa/embed` route and
  added panel launcher handling for disabled readiness metadata.

### Changed - README generated product facts

- Extended the README sync check beyond badges so shipped route prose,
  architecture route counts, caption style counts, and root test-file prose
  are generated from live project sources instead of drifting manually.

### Changed - CEP panel Vite 8 tooling

- Upgraded the CEP panel build lockfile to Vite 8.1, removed the stale Vite 5
  advisory waiver path, and tightened the npm advisory gate to fail closed on
  any future unwaived panel vulnerability.

### Changed - C2PA signed provenance credentials

- C2PA provenance sidecars now verify Ed25519 signatures with embedded
  public keys, classify signed, unsigned, embedded, missing-asset, and
  tampered-manifest states, and can request a `c2patool` embedded
  MP4/JPEG/PNG credential when an operator signing key is configured.

### Changed - UXP caption style catalog

- UXP caption styles now load from the backend `/captions/styles` catalog
  when the server is available, keep a backend-compatible `minimal_clean`
  fallback while offline, and send the selected `caption_style` id with
  transcription jobs.

### Changed - Caption translation license safety

- Added explicit NLLB and SeamlessM4T model-card records, fail-closed
  translation backend selection, and CEP opt-in copy so CC-BY-NC caption
  translation engines cannot be downloaded or used silently.

### Changed - UXP markup safety

- Added a static UXP DOM-sink safety guard that inventories reviewed
  `innerHTML` and HTML accumulator render paths, verifies dynamic markup
  expressions are escaped, and pins script-like backend/user field families
  to escaped rendering before they reach panel markup.

### Changed - GitHub issue seed guard

- GitHub issue seeding now validates active seeds against unchecked
  `ROADMAP.md` rows, skips manifest rows marked `status: shipped` or
  `status: archived`, and fails before issue creation when an active seed has
  gone stale.

### Changed — Local release documentation

- Reconciled active release, installer, signing, notarization, SBOM, and
  contribution docs with the local-build-only release policy and replaced
  obsolete workflow guard tests with local release-policy guards.
- Refreshed MCP, feature-readiness, and model/license generated contracts so
  the local `pytest-fast` release smoke passes against the current source.

### Changed — Bundled FFmpeg provenance

- Refreshed the local bundled FFmpeg/ffprobe binaries to Gyan
  `8.1.2-essentials_build-www.gyan.dev` and updated installer/provenance pins
  plus release provenance metadata for the current security floor.

### Changed — CEP Node advisory gate

- Patched the transitive `js-yaml` advisory through an npm override and
  restored the CEP advisory gate with explicit Vite dev-server-only waivers
  tied to the documented HGFS-blocked major upgrade path.

### Changed — Model-card generated artifact freshness

- Regenerated the model-card manifest and added cards for the optional
  AutoShot, DA3, IC-Light, LatentSync, NeMo ASR/Sortformer, OCR, SAM 3,
  SeedVR2, pyannote diarization, and visual multicam checks so generated
  model/license truth gates pass again.

### Changed — Installer accessibility polish

- Bound installer version badges to the shared app version constant and added
  accessible names/help text for icon-only window controls, progress bars,
  activity logs, and key install options. Shared WPF focus states now cover
  link, checkbox, and radio controls.

### Changed — UXP panel quick-action polish

- Replaced Settings-tab keyboard shortcut hints with visible quick-action
  buttons that focus the destination control before running, keep cancel
  availability synced with active jobs, and explain backend-locked actions
  through localized titles/ARIA state.

### Changed — CEP panel confirmation polish

- Replaced remaining CEP native browser dialogs with panel-local confirmation,
  choice, and prompt flows for journal clearing, footage-index clearing,
  natural-language route execution, issue reports, and gist sync. The new
  dialogs use the shared overlay focus trap, localized copy, validation,
  reduced-motion handling, and light/dark theme styling.
- Added a CEP guard test that blocks reintroducing native `alert`,
  `confirm`, or `prompt` globals in panel source.

### Added — 537 more coverage expansion tests (658 total, 30+ modules)

- New test files: `test_coverage_infra.py` (169 tests covering errors,
  schemas, workers, config, helpers, checks), `test_coverage_core_logic.py`
  (137 tests covering timecode_utils, metadata_tools, safe_zones, hdr_tools,
  dead_time, stream_chapters, timelapse, highlight_detect, soft_subtitles,
  adr_cueing), `test_coverage_core_extra.py` (159 tests covering analytics,
  selects_bin, show_notes, ffmpeg_builder, ab_av1, ab_compare, cinemagraph,
  surround_mix, gpu_dashboard, display_calibration, apple_silicon, amd_gpu),
  `test_coverage_schemas_mcp.py` (72 tests covering schemas, MCP tool
  catalog validation, error factories, config, helpers, checks, workers).
  CI coverage floor raised from 55% to 58%.

### Added — Centralized config schema registry with versioned migration

- New `register_config_schema()` and `read_user_file_versioned()` in
  `opencut/user_data.py`. Config files can declare a schema version
  and register migration functions. `read_user_file_versioned()` auto-
  migrates old configs on read, stamps `_schema_version`, and persists
  the upgraded data. Unknown keys are preserved, not stripped.

### Added — 121 coverage expansion tests across 10 untested core modules

- New `tests/test_coverage_expansion.py` adds focused unit tests for:
  url_ingest (16), multimodal_index (7), edl_aaf (14), media_relink (9),
  auto_update (14), structured_ingest (12), batch_conform (5),
  project_organizer (15), storage_tiering (10), render_queue (9).
  All tests are fast (<1s total), require no external services, and run
  without optional ML/GPU dependencies. CI coverage floor raised to 55%.

### Changed — Docker compose hardened + MCP sidecar profile

- Added CPU/memory resource limits (CPU: 4c/4G, GPU: 8c/16G), JSON-file
  log rotation (50-100MB × 3-5 files), and an opt-in MCP sidecar profile
  (`docker compose --profile mcp up`). GPU service now reads
  `NVIDIA_VISIBLE_DEVICES` from environment instead of hardcoding `all`.

### Changed — Default Whisper model upgraded to large-v3-turbo

- `CaptionConfig.model` default changed from `base` to `large-v3-turbo`.
  Within 1-2% WER of large-v3 at 5.4x speed. Existing users keep their
  configured model.

### Added — CLI config export/import/validate commands

- `opencut config export [-o backup.json]` dumps all `~/.opencut/*.json`
  settings to a single JSON bundle. `opencut config import backup.json`
  restores them. `opencut config validate` checks all config files for
  JSON parse errors (exits non-zero on failure).

### Added — 12 new animated caption presets (19 → 31)

- Neon Glow, Slide Up, Word Pop, Emoji React, Speaker Colored, Retro VHS,
  Podcast, Gaming, Luxury, TikTok Bold, Handwritten, and RTL Arabic.
  Caption style count now exceeds 30, matching paid competitor expectations.

### Added — Smart pause detection for silence removal

- New `smart_pause=true` option on `/silence` route and `detect_speech()`
  that preserves dramatic pauses (0.8-3.0s silences preceded by
  sentence-ending speech) instead of cutting them. When a transcript is
  available, uses sentence-final punctuation as context; without transcript,
  falls back to a duration heuristic that preserves short mid-speech pauses.
- New `filter_smart_pauses()` function in `opencut/core/silence.py` with 2
  unit tests.

### Added — MCP tool catalog expanded from 39 to 86 curated tools

- Added 47 new curated MCP tools covering video editing (trim, merge, concat,
  reframe, stabilize, speed, PiP, blend), video effects (chromakey, LUT, FX,
  denoise, interpolation, transitions), captions (burn-in, translate, styled,
  animated, karaoke, SRT import), audio (effects, duck, isolate, normalize,
  filler removal, beat markers, enhance), timeline/workflow (run, presets, batch
  rename, smart bins, beat cut, export), deliverables (VFX sheet, ADR list,
  music cue sheet), search (NLP command, URL ingest), and system (info, GPU,
  dependencies, feature state, social upload).
- Agent clients (Claude, Cursor, etc.) now discover 86 OpenCut capabilities
  vs 39 previously. Extended tools (1,482 route-level) remain opt-in.

### Security — Werkzeug floor raised to >=3.1.6 (CVE-2026-27199)

- Bumped Werkzeug floor from >=3.1.5 to >=3.1.6 in both `pyproject.toml`
  and `requirements.txt`. CVE-2026-27199 is a `safe_join()` Windows
  device-name bypass that causes hanging file reads in `send_from_directory`.

### Fixed — Route manifest + README badge drift

- Regenerated `route_manifest.json` and `feature_readiness.json` to reflect
  recent url_ingest, multimodal_index, and AutoShot additions. README badges
  and inline counts updated: 1,545 shipped routes, 6 stubs.

### Changed — Feature-state now reflects route readiness

- Generated feature-readiness records now cross-reference the route
  manifest's `readiness` field instead of defaulting to `"available"`.
  Routes classified as `dependency-gated` produce `"missing_dependency"`
  features; stub routes produce `"stub"` features. Panels disable buttons
  for these features before the user clicks them (was: all generated
  features showed as available regardless of route status).

### Added — Import-by-link footage ingest

- New `url_ingest` module (`opencut/core/url_ingest.py`): fetches video from
  a URL to `~/.opencut/media_ingest/` so existing routes consume it as a local
  file. Uses yt-dlp when installed (YouTube, Vimeo, Zoom Clips, Medal, etc.);
  falls back to plain HTTP download for direct-link URLs. SHA-256 content-
  addressed caching avoids re-downloads.
- New routes: `POST /search/ingest` (async job), `GET /search/ingest/cache`,
  `DELETE /search/ingest/cache`.

### Added — Local multimodal footage search (OCR + audio event index)

- Footage index schema upgraded to v2: `ocr_text` and `audio_tags` columns
  added to the `footage` table, FTS5 index rebuilt to cover all four fields
  (file_path, transcript, ocr_text, audio_tags). Existing v1 databases
  auto-migrate on first access.
- New `multimodal_index` module (`opencut/core/multimodal_index.py`):
  extracts on-screen text from key frames via pytesseract or easyocr, and
  classifies audio events (speech, music, silence, tempo) via librosa
  spectral heuristics with a basic fallback.
- New `POST /search/multimodal-index` route: indexes files with transcript +
  OCR text + audio event tags in a single pass. OCR and audio tags are
  opt-in (default on) via `ocr` and `audio_tags` body params.
- FTS5 search now matches across transcript, on-screen text, and audio
  event tags — a query for "music" or "logo" hits audio events and OCR
  text respectively, not just spoken words.
- `check_ocr_available()` added to `checks.py`.

### Added — AutoShot scene-detection engine

- Added AutoShot as a selectable scene-detection engine with priority 85
  (above TransNetV2's 70). `detect_scenes_auto()` now tries AutoShot first
  for gradual-transition-aware detection, falling back to TransNetV2 →
  PySceneDetect → FFmpeg threshold. `check_autoshot_available()`, engine
  registry entry, and 5 new tests added.

### Added — Model privacy/license posture in feature-state

- Feature-state entries now expose `privacy` (local-only / cloud), `license`
  (SPDX token), and `advisory_notes` from the model-card registry. The CEP
  panel annotates gated buttons with data-feature-privacy/license attributes,
  privacy chip badges ("Local" / "Cloud"), and advisory notes in tooltips.

### Changed — Collapsed 4 duplicate 501 stubs onto canonical pipelines

- `/video/trailer/generate` now delegates to `trailer_gen.generate()`.
- `/video/outpaint` now delegates to `frame_extension.outpaint_aspect_ratio()`.
- `/agent/search-footage` now delegates to
  `semantic_video_search.semantic_search()`.
- `/agent/storyboard` now delegates to
  `ai_storyboard.generate_storyboard_from_script()`.
- Shipped route count rises from 1,536 to 1,541. Users no longer hit 501
  for capabilities that already have working local implementations.

### Changed — Default depth backend upgraded to Depth Anything 3 Small

- New `depth_anything_3` module + a `depth-anything/DA3-SMALL` (Apache-2.0,
  single-transformer SOTA, ~+25% geometry) default for CineFocus bokeh/parallax,
  with `Depth-Anything-V2-Small` retained as the automatic fallback. CineFocus
  now routes both depth-load sites through a central `_load_depth_backend()` that
  resolves DA3 first and degrades to DA2 on any load failure.
- New `depth_anything_3` engine-registry entry (priority 85 > DA2's 80) +
  `check_depth_anything_3_available()`, a `da3-small` model entry, and
  `tests/test_depth_engines.py`.

### Changed — Re-aimed the IC-Light relight engine at v1 (Apache-2.0)

- `relight_iclight.py` was titled "IC-Light V2" but IC-Light v2 is non-commercial
  and its full weights were never publicly released — un-shippable under MIT.
  Re-targeted it at **IC-Light v1** (`lllyasviel/IC-Light`, Apache-2.0, real public
  weights). Removed the v2 framing from docstrings/hints, added `MODEL_ID/NAME/
  LICENSE`, `check_iclight_available()`, an `ic-light-v1` model entry, and a new
  `relight` engine-registry domain. The `/video/relight/iclight` route now reports
  the v1 Apache-2.0 licence. Registry test suite added.

### Changed — Default speaker diarization upgraded to pyannote community-1

- `diarize()` now defaults to `pyannote/speaker-diarization-community-1`
  (CC-BY-4.0, "always freely accessible", lower DER) and automatically falls
  back to legacy `speaker-diarization-3.1` when the community model can't be
  loaded (terms not accepted, offline cache miss). The model is configurable via
  `DiarizeConfig.model`.
- New `diarization` engine-registry domain: `pyannote_community1` (default,
  priority 80) > `pyannote_legacy` (60, fallback) > `sortformer` (50, optional,
  NVIDIA NeMo-gated). Added `check_diarization_available()` /
  `check_sortformer_available()`, a `pyannote-community-1` model entry, and
  `tests/test_diarization_engines.py`.

### Added — Normalized 0–100 virality score on highlights/clips

- `EngagementScore.compute_virality()` maps the existing engagement dimensions
  (hook / emotional / pacing / quotability) to a single deterministic, hook-forward
  **0–100 virality score** — the headline number Opus Clip paywalls — so creators
  can sort and threshold clips at a glance. It is a heuristic, not a performance
  guarantee, and is distinct from the editing-relevance `overall` composite.
- Surfaced as a top-level `virality` field (plus inside the `engagement` block)
  on `POST /video/highlights` and the magic-clips/shorts response. Deterministic
  unit tests added.

### Added — LatentSync lip-sync as the dub pipeline's opt-in final stage

- Wired a new `lipsync_latentsync` engine into a new `lip_sync` engine-registry
  domain. The dub pipeline (transcribe→translate→voice-clone→render) can now
  re-animate the speaker's mouth to the new audio — the visual lip-sync stage
  Rask/HeyGen/sync.so paywall — via `DubConfig.lip_sync_backend="latentsync"` and
  `POST /ai/lip-sync` (`backend` param).
- **Opt-in by design** (RESEARCH Open Question 3): LatentSync's *code* is
  Apache-2.0 but its *checkpoint* licence is unconfirmed, so the engine is
  registered below the always-available heuristic and is never auto-selected.
  The dub stage degrades gracefully: LatentSync → heuristic jaw-overlay →
  audio-only dub, so it never hard-fails.
- Added `check_latentsync_available()`, a `latentsync-1.6` model-download entry,
  and `tests/test_lipsync_engines.py` (registry, opt-in ordering, availability
  gating, license hygiene, dub-config plumbing). Full diffusion forward activates
  once local weights are present.

### Added — SeedVR2 upscaling engine (Apache-2.0, one-step diffusion)

- Wired the previously-dark `upscale_seedvr2` stub into the upscaling engine
  registry as `seedvr2`, gated by `check_seedvr2_available()` (diffusers + torch).
  It is registered at higher priority than Real-ESRGAN, so it auto-selects when
  its Apache-2.0 SeedVR2-3B weights are installed and **falls back to Real-ESRGAN**
  otherwise. `POST /video/ai/upscale` gained an `engine` param (`auto`/`seedvr2`/
  `realesrgan`) that tries SeedVR2 first and degrades to Real-ESRGAN on any
  unavailability (recorded in the response `notes`), never hard-failing.
- Added a `seedvr2-3b` model-download catalog entry (Apache-2.0, ~6 GB) and a
  registry test suite (`tests/test_upscaling_engines.py`). The heavy diffusion
  forward activates once the local weights are present; the entry point raises a
  structured error when absent (matching the NeMo ASR engine contract).

### Added — Beat-synced cuts to the active sequence (`POST /timeline/beat-cut`)

- New timeline-native beat-cut path: detect beats in a clip's audio and emit
  beat-aligned clip boundaries as a cut list the panel reviews and ripple-applies
  to the **active Premiere sequence**. Unlike `/audio/beat-sync` (which renders a
  new montage file), this produces cut points for the sequence already open.
- `core/beat_sync_edit.py::plan_timeline_beat_cuts()` — pure planner over detected
  beats + a clip duration, with mode selection (every beat / 2 beats / bar / 2 bars
  / accents) and `min_cut_duration` merging. Returns `{start, end, duration,
  beat_number, strength, label}` segments (the shape `ocApplySequenceCuts` and the
  cut-review panel already consume).
- CEP panel: a "Beat-Synced Cuts" card on the Timeline → Beat Markers sub-tab,
  wired through the shared cut-review preview + ripple-apply pipeline (the same one
  silence/filler/auto-edit use). Added to the queue allowlist and i18n.
- Tests: 4 planner unit tests + 3 route smoke tests; i18n/a11y/parity green.

### Changed — Brand/namespace disambiguation: distribution token `opencut-ppro`

- Resolved the carried brand/namespace question (RESEARCH Open Question 1). The
  product keeps the name **OpenCut**, but the package-manager distribution token
  is now **`opencut-ppro`** to stay unambiguous against the unrelated browser
  editor opencut.app (~48K stars) across PyPI/Homebrew/winget/SEO. The `-ppro`
  suffix marks the Premiere Pro integration that uniquely identifies this tool.
- `pyproject.toml` `[project].name` → `opencut-ppro`; SBOM root component and
  README `pip install` examples updated to match. **Unchanged:** the import
  package (`import opencut`), CLI commands (`opencut`, `opencut-server`,
  `opencut-mcp-server`), and CEP/UXP extension IDs. Decision recorded in the
  README "Naming & distribution" section. (All candidate names were free on
  PyPI; the actual publish remains a credential-gated roadmap item.)

### Security — Bundled FFmpeg 8.1.1 + June-2026 security-patch provenance

- Reconciled the bundled-FFmpeg version pin to `8.1.1-essentials_build-www.gyan.dev`
  across `AppConstants.cs`, `OpenCut.iss`, and the manifest test (the Inno pin had
  drifted to `8.1`, leaving `test_inno_installer_writes_ffmpeg_manifest` red).
- New `opencut/core/ffmpeg_provenance.py` is the single source of truth for the
  acceptable bundled build. It parses `ffmpeg -version` (gyan.dev release, gyan
  git-master, BtbN `git describe`, distro `n`-prefixed) and grades it against two
  lanes: a release floor (`>= 8.1.1`) or a git-master snapshot dated `>= 2026-06-10`
  (reference commit `b29bdd3715`). This asserts a *security patch level* — the
  June-2026 FFmpeg zero-days (CVE-2026-6385 + CVE-2026-39210..39218, crafted-media
  heap/stack overflows) landed as post-release master commits, so a bare `8.1.x`
  tag is insufficient on its own.
- `GET /system/capabilities` now carries `ffmpeg.security` (graded result) and a
  `ffmpeg_below_security_floor` finding so a stale bundled binary is visible at runtime.
- `scripts/verify_ffmpeg_provenance.py` is a stdlib build/CI gate that fails closed
  when the bundled binary is below the floor and records ground-truth provenance
  (version, git commit/snapshot date, lane, CVE list) to a manifest.
- Installer manifests (`installer.json`, both WPF + Inno) now record
  `bundled_ffmpeg_security_floor`. Provenance requirements documented in
  `docs/RELEASE_PROVENANCE.md`.

### Added — NVIDIA NeMo ASR engines (Parakeet TDT 0.6B v3 + Canary 1B Flash)

- Wired the previously-dark `asr_parakeet` (streaming) and `asr_canary` (batch)
  stubs into the transcription engine registry as `parakeet_tdt` and
  `canary_1b_flash`. Both are gated by a new `check_nemo_asr_available()` check
  (accepts the `nemo` / `nemo_toolkit[asr]` install), retain faster-whisper and
  CrisperWhisper as fallbacks, and ship `parakeet-tdt-0.6b-v3` /
  `canary-1b-flash` model-catalog entries for on-demand download. Engines report
  unavailable (never crash) when NeMo is absent.

### Changed — Route manifest tags readiness; advertised count excludes stubs

- `dump_route_manifest` now classifies every route as
  `implemented` / `dependency-gated` / `stub` via static handler inspection,
  and records `readiness_counts` plus a `shipped_route_count` (total minus the
  10 HTTP-501 strategic stubs). Manifest schema bumped to version 3.
- New `GET /system/route-readiness` endpoint surfaces the shipped count and the
  explicit stub / dependency-gated rule lists so panels and tooling stop
  presenting 501 stubs as shipped capabilities. (Panels already gate
  `data-feature-id` actions via F100 `feature-state.js`; the Tier-3 stub
  actions are not surfaced as panel buttons, so none are presented as shipped.)
- README route badge and route-count claims now advertise the **shipped** count
  (1,536) instead of the raw total; `scripts/sync_badges.py` binds the badge to
  `shipped_route_count`.

### Changed — Structured error responses across video/audio routes

- Migrated the remaining 35 raw `jsonify({"error": ...})` responses in
  `routes/video_editing.py`, `routes/video_core.py`, and `routes/audio.py` to
  the structured `error_response()` helper. Every error now returns a
  machine-readable `code`, a `suggestion`, and request-id correlation, and is
  logged through the typed-error path — matching the rest of the API surface
  the frontend already consumes.

### Security — AST whitelist for the expression sandbox

- Replaced the deny-list AST check in `expression_engine.py` with a closed
  node **whitelist**: only arithmetic/boolean/comparison nodes, calls to known
  sandbox functions, and ternaries are permitted. Every `ast.Attribute` node is
  now rejected outright, so dunder-walk escapes (`().__class__.__subclasses__`)
  and obfuscated variants cannot be expressed at all. Call targets are
  constrained to the sandbox's callable table; unknown calls (`breakpoint()`,
  `eval()`) are rejected at validation time.

### Security — AST-level dunder/builtin guard for the scripting console

- Added an obfuscation-proof structural check to `scripting_console.py` that
  parses the script and rejects any dunder attribute access (`__base__`,
  `__flags__`, `__delattr__`, ... — including escape vectors absent from the
  hand-maintained substring list) and any reference to a blocked builtin
  (`getattr`, `eval`, `globals`, ...), regardless of source formatting. The
  lowercased substring scan is retained as a first-pass net.

### Security — Transitive web-dependency floor pins (RA-23)

- Floored Werkzeug (`>=3.1.5`) and Jinja2 (`>=3.1.6`) in core
  `pyproject.toml` dependencies — they ship transitively with flask and are
  always installed, so a clean resolver (no `requirements-lock.txt`) can no
  longer pull Werkzeug CVE-2026-21860 / CVE-2025-66221 (`safe_join` Windows
  device-name DoS) or Jinja2 CVE-2025-27516 (sandbox breakout).
- Floored `requests>=2.33.0` and `urllib3>=2.6.3` in the `standard` and `all`
  extras, where they enter via faster-whisper's huggingface-hub fetch path —
  closing urllib3 CVE-2026-21441 (decompression-bomb DoS) and requests
  CVE-2026-25645 in the release-audited lane.
- Synced `requirements.txt` flask/waitress floors to the pyproject values and
  added the new security pins; `tests/test_dependency_surface.py` now asserts
  the transitive floors and the resolver lane matches the lockfile.

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

### Fixed — Engineering audit (batch 4): multicam kwarg mismatch

- **video_editing.py**: Fixed `speaker_map=` → `speaker_to_track=` in
  `generate_multicam_cuts()` call. The route passed a keyword argument
  that didn't match the function signature, causing a TypeError at
  runtime on every multicam cut request.

### Fixed — Engineering audit (batch 3): SQLite connection safety

- **job_store.py**: Wrapped PRAGMA setup in try/except to close the
  connection on failure instead of leaking it (lines 134-140).
- **job_store.py**: Added `PRAGMA wal_checkpoint(TRUNCATE)` before
  closing connections in `close_all_connections()` to prevent orphaned
  WAL/SHM files on Windows shutdown.

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
