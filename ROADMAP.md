# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

## P1 — Release blocking

- [ ] P1 — External F202 macOS notarization live acceptance is credential-gated
  Why: release wiring exists, but the first live Apple acceptance still needs configured Apple credentials, notarization secrets, and a macOS release run before the claim is complete.
  Where: docs/MACOS_NOTARIZATION.md, .github/workflows/build.yml
  Blocked: Apple credentials / notarization secrets / live macOS release runner.

- [ ] P1 — External F252 UXP WebView cutover still needs live Premiere UDT evidence
  Why: WebView scaffolding and validators exist, but the cutover cannot be claimed until an in-Premiere UDT capture passes the strict validator and residual CEP-only paths are accounted for.
  Where: docs/UXP_MIGRATION.md:153-155, extension/com.opencut.uxp/bolt-webview/, opencut/tools/validate_uxp_udt_results.py
  Blocked: live Premiere UDT capture.

## Research-Driven Additions

- [ ] P1 — Decide and execute brand/namespace disambiguation (gates all distribution work)
  Why: carried from the 2026-06-09 research pass with no roadmap entry; opencut.app (48K stars) is mid-rewrite (May 2026) and will re-dominate search when it relaunches; PyPI/winget/Homebrew/SEO all need an unambiguous token first.
  Evidence: github.com/OpenCut-app/OpenCut; opencut.app/roadmap; RESEARCH.md Open Question 1
  Touches: pyproject.toml project name, README tagline/badges, GitHub repo description/topics
  Acceptance: written decision in README or CONTRIBUTING; chosen PyPI name registered (placeholder publish acceptable).
  Complexity: M

- [ ] P2 — Replace bundled FFmpeg 8.0.1 binary with 8.1.x
  Why: Code-side D3D12VA/Vulkan encoder detection, route validation, export presets, and installer version strings are all wired; only the binary download remains.
  Where: ffmpeg/ (bundled binaries)
  Acceptance: bundled ffmpeg -version reports 8.1.x.
  Complexity: S


- [ ] P2 — Run documented CEP+UXP smoke pass on Premiere 26.x
  Why: README positioning now leads with OpenCut-unique capabilities vs Adobe 26.x. Manifests mathematically cover 26.x (CEP [13.0,99.9], UXP minVersion 25.6). Remaining: live smoke test on an actual Premiere 26.x install.
  Blocked: Premiere 26.x license/installation.
  Acceptance: documented smoke test pass on Premiere 26.0 or 26.2.x for both CEP and UXP panels.
  Complexity: S

- [ ] P3 — Publish the server package to PyPI via trusted publishing (after brand decision)
  Why: pip install of the server/CLI is the cheapest distribution surface and currently impossible (no PyPI package); trusted publishing avoids long-lived tokens in CI.
  Evidence: docs.pypi.org/trusted-publishers/; RESEARCH.md distribution assessment
  Touches: .github/workflows/build.yml (publish job), pyproject.toml metadata
  Acceptance: pip install of the chosen name installs the opencut CLI/server from PyPI; releases publish automatically on tag.
  Complexity: M

- [ ] P2 — Move CEP panel off unsupported Vite 5 with HGFS-safe regression evidence
  Why: Vite 5.4.21 remains pinned with a documented advisory waiver because Vite 6+ regressed VMware HGFS paths, but Vite's maintained release line has moved to 8.x with 7.3/6.4 backports.
  Evidence: extension/com.opencut.panel/package.json:22; docs/NODE_ADVISORIES.md:13,27-34,66-74; tests/test_panel_node_entrypoints.py; vite.dev/releases
  Touches: extension/com.opencut.panel/package.json, package-lock.json, scripts/panel-node-gate.ps1, docs/NODE_ADVISORIES.md, extension panel build/audit tests, CI release smoke
  Acceptance: panel build/audit uses a supported Vite line, the documented Vite advisory waiver is removed, and Windows UNC/HGFS-safe `*:win` entrypoints plus Linux CI build/verify still pass.
  Complexity: M

- [ ] P3 — Add Homebrew tap for macOS CLI distribution (after brand decision + PyPI)
  Why: macOS has no package-manager install path — users must clone + pip-install manually. A Homebrew tap gives macOS users `brew install <name>` for the CLI/server. Depends on brand decision (P1) and PyPI publish (existing P3).
  Evidence: docs.brew.sh/Python-for-Formula-Authors; Homebrew accepts Python apps even without PyPI; no existing tap
  Touches: new homebrew-<name> tap repo, formula file, CI publish workflow
  Acceptance: `brew install <tap>/<name>` installs the CLI/server on macOS; formula auto-updates on new PyPI releases.
  Complexity: M

- [ ] P3 — Add winget package manifest for Windows distribution (after release parity + code signing)
  Why: Windows users discover software via winget; no manifest exists. Requires a stable installer URL (GitHub Release .exe) and ideally a code-signed binary for SmartScreen reputation. Depends on release parity (P1) and code signing budget ($216-575/yr OV/EV certificate).
  Evidence: github.com/microsoft/winget-pkgs (12,850+ packages); no existing OpenCut manifest; signmycode.com pricing
  Touches: winget manifest YAML (submitted as PR to microsoft/winget-pkgs), CI release workflow (signed installer upload)
  Acceptance: `winget install <name>` installs OpenCut on Windows; manifest auto-updates via GitHub Release URLs.
  Complexity: M

## Audit-Driven — Security hardening

- [ ] P1 — Harden expression_engine.py eval() sandbox with AST-based validation
  Why: User expressions are compiled+eval'd. Current regex-based validation is bypassable. Needs AST node whitelist or RestrictedPython.
  Where: opencut/core/expression_engine.py

- [ ] P1 — Harden scripting_console.py exec() sandbox against dunder bypass
  Why: Dunder block uses lowercased string search which is bypassable with obfuscation. Needs AST-level Name node validation.
  Where: opencut/core/scripting_console.py

- [ ] P2 — Migrate 35 raw jsonify({"error":...}) responses to structured error_response()
  Why: Frontend expects structured {code, message, suggestion} responses; raw strings lack machine-readable error codes.
  Where: opencut/routes/video_editing.py (13), video_core.py (18), audio.py (4)

## Research-Driven Additions (2026-06-14)

### P1

- [ ] P1 — Floor-pin transitive web deps in pyproject.toml to match requirements-lock.txt
  Why: requirements-lock.txt is safe (urllib3 2.7.0, Werkzeug 3.1.7, requests 2.33.0, Jinja2 3.1.6) but pyproject [dependencies]/[all] only pin flask/flask-cors/waitress; `pip install opencut[all]` without the lock resolves transitively and can pull vulnerable urllib3/Werkzeug/Jinja2.
  Evidence: pyproject.toml:35-43 (no transitive floors); urllib3 CVE-2026-21441 (GHSA-38jv-5279-wg99, fixed 2.6.3); Werkzeug CVE-2026-21860 / CVE-2025-66221 (fixed 3.1.5, Windows safe_join DoS); Jinja2 CVE-2025-27516 (fixed 3.1.6); requests CVE-2026-25645 (fixed 2.33.0)
  Touches: pyproject.toml (dependencies + all/standard extras), requirements.txt, dependency-floor test
  Acceptance: `pip install opencut[all]` in a clean resolver (no lockfile) cannot select urllib3<2.6.3, Werkzeug<3.1.5, Jinja2<3.1.6, or requests<2.33.0; a test asserts the floors.
  Complexity: S

- [ ] P1 — Assert bundled-FFmpeg security patch level, not just version 8.1.x
  Why: complements the existing "bump bundled FFmpeg to 8.1.x" item — the June-2026 FFmpeg zero-days (heap/stack overflows reachable via crafted media, the first untrusted-input path a media tool hits) landed as post-release master commits, so an 8.1.x release tag may predate them.
  Evidence: CVE-2026-6385 (GHSA-q22x-99q7-fr6w, CVSS 6.5) + CVE-2026-39210..39218 (reserved); thehackernews.com/2026/06/ai-agent-uncovers-21-zero-days-in.html; ffmpeg/ffmpeg.exe (bundled)
  Touches: ffmpeg/ (bundled binary provenance), scripts/ FFmpeg build/verify, docs/PYTHON_ADVISORIES.md or a new FFmpeg provenance note
  Acceptance: bundled `ffmpeg -version` build is documented to include the June-2026 security commits (commit hash or post-fix snapshot recorded), not merely reporting 8.1.x.
  Complexity: S

### P2

- [ ] P2 — Add a capability/stub manifest so docs and panels stop presenting Tier-3 501 stubs as shipped
  Why: 145 NotImplementedError core stubs and ~22 routes returning HTTP 501 (wave_h/wave_k) are counted in the 1,539-route badge and feature framing; honest at the route level but misleading as distribution widens.
  Evidence: opencut/routes/wave_h_routes.py, wave_k_routes.py (501 handlers); opencut/_generated/route_manifest.json (counts stubs); README route-count badge
  Touches: opencut/tools/dump_route_manifest.py (tag implemented/gated/stub), /capabilities or /health extension, extension/com.opencut.panel + com.opencut.uxp (grey out stubbed actions), README badge generation
  Acceptance: manifest tags every route implemented/dependency-gated/stub; advertised route-count counts only shipped routes; both panels disable Tier-3 stub actions with a "planned" hint.
  Complexity: M

- [ ] P2 — Beat-synced auto-edit (assemble cuts to detected beats)
  Why: OpenCut detects beats (librosa) and writes Premiere markers but cannot assemble to them; aescripts "Automated Video Editing", Cardboard (`sync_to_beats`), and CapCut all ship music-driven beat-cutting.
  Evidence: opencut/core/ beat detection + beat-marker export already present; aescripts.com/automated-video-editing-for-premiere-pro/; startuphub.ai Cardboard (sync_to_beats op)
  Touches: new core beat-assembly module, a /timeline beat-cut route, ExtendScript/UXP apply path, panel Cut/Timeline tab
  Acceptance: given a clip + audio track, the route emits ripple cuts / clip boundaries aligned to detected beats and applies them to the active sequence (with a cut-review preview).
  Complexity: M

- [ ] P2 — Wire SeedVR2 as a clean-license upscaling engine
  Why: SeedVR2 (ByteDance, ICLR 2026, Apache-2.0) is one-step diffusion video restoration, >4x faster than multi-step VSR and beats Real-ESRGAN; its stub already exists and the license is MIT-distribution-safe.
  Evidence: opencut/core/upscale_seedvr2.py (NotImplementedError stub); github.com/IceClear/SeedVR2; huggingface.co/ByteDance-Seed/SeedVR2-3B (Apache-2.0)
  Touches: opencut/core/upscale_seedvr2.py, engine_registry.py (upscaling domain), checks.py availability flag, model_manager.py, upscaling route/preset
  Acceptance: SeedVR2 appears as a selectable upscaling engine, gated by availability check, with a model download entry and registry test; falls back to Real-ESRGAN when unavailable.
  Complexity: L

### P3

- [ ] P3 — Upgrade depth backend to Depth Anything 3 Small (Apache-2.0)
  Why: DA3 (Nov 2025) is single-transformer SOTA (~+25% geometry vs prior) and DA3-Small is Apache-2.0; cinefocus/depth effects currently default to Depth-Anything-V2-Small.
  Evidence: opencut/core/cinefocus.py:176 (DA2-Small default); github.com/ByteDance-Seed/depth-anything-3; huggingface.co/depth-anything/DA3-SMALL
  Touches: opencut/core/cinefocus.py, compose_depth_segment.py, depth engine selection, model_manager.py
  Acceptance: DA3-Small is the default depth backend (DA2 retained as fallback) for bokeh/parallax/depth routes, with availability check and test.
  Complexity: M

- [ ] P3 — Local multimodal footage search (object + on-screen-text + sound-event index)
  Why: Premiere 26 Media Intelligence does NL search across visuals, transcripts, and described sounds on-device; OpenCut indexes only speech + CLIP frames. Object-detection/OCR/audio-event indexes would match it locally — a defensible OSS differentiator.
  Evidence: opencut/core/semantic_video_search.py (speech + CLIP only); helpx.adobe.com Media Intelligence; twelvelabs.io/product/video-search
  Touches: semantic_video_search.py, FTS5 index schema, footage indexing pipeline, search route + panel Search tab
  Acceptance: footage search returns hits for on-screen objects, on-screen text, and sound events (not just spoken words), with the index built locally and incrementally.
  Complexity: L

- [ ] P3 — Add scene-detection AutoShot engine (beats TransNetV2 on gradual transitions)
  Why: AutoShot beats TransNetV2 by ~4.2% F1 and handles gradual transitions PySceneDetect misses; scene_detect.py registry is only threshold + TransNetV2.
  Evidence: opencut/core/scene_detect.py (threshold + transnetv2); engine_registry.py scene_detection domain; github.com/wentaozhu/AutoShot
  Touches: opencut/core/scene_detect.py, engine_registry.py, checks.py, scene-detection route/preset
  Acceptance: AutoShot is a selectable scene-detection engine with availability check and registry test; TransNetV2/threshold retained.
  Complexity: M

- [ ] P3 — Import-by-link footage ingest connectors
  Why: Opus Clip ingests directly from Zoom Clips / Apple Podcasts / Medal links with no re-upload; OpenCut requires a local file path for every operation.
  Evidence: coldiq.com/tools/opus-clip (May 2026 import-by-link); OpenCut routes require local filepath
  Touches: new ingest module (URL → local cache via yt-dlp-class fetch), watch_folder/scheduled_jobs integration, panel input affordance
  Acceptance: a supported URL is fetched to the local media cache and becomes usable by any existing route exactly like a local file; failures return structured errors.
  Complexity: M

## Research-Driven Additions (2026-06-14, second pass)

### P2

- [ ] P2 — Wire NVIDIA Parakeet TDT 0.6B v3 (+ Canary-1B-Flash) as ASR engines
  Why: Both stubs already exist but are dark and unregistered. Parakeet TDT 0.6B v3 tops the HF Open ASR Leaderboard (~6.3% mean WER vs Whisper Large-v3 at 4× the params), is ~4× faster on CPU, now covers 25 EU languages, and ships under a permissive code+weight licence — a strict upgrade over the bundled Whisper path for clean-audio English/EU work.
  Evidence: opencut/core/asr_parakeet.py + asr_canary.py (NotImplementedError stubs, INSTALL_HINT "nemo_toolkit[asr] # Apache-2.0 + CC-BY-4.0 model"); huggingface.co/nvidia/parakeet-tdt-0.6b-v3 (CC-BY-4.0); northflank.com 2026 ASR benchmarks
  Touches: opencut/core/asr_parakeet.py, asr_canary.py, transcription engine registry, checks.py availability flag, model_manager.py, transcription route/preset, registry availability test
  Acceptance: Parakeet (streaming) and Canary (batch) appear as selectable transcription engines gated by availability check, with a model download entry and registry test; faster-whisper/WhisperX retained as fallback.
  Complexity: M

- [ ] P2 — Wire LatentSync lip-sync onto the existing dub pipeline (local dubbing-with-lip-sync)
  Why: OpenCut already ships the full dub pipeline (transcribe→translate→voice-clone→render) but stops before re-animating the mouth — the exact visual lip-sync stage Rask AI ($120/mo), HeyGen, and sync.so ($0.04–0.133/sec) paywall. Wiring an Apache-2.0 lip-sync model as the dub pipeline's optional final stage converts a paywalled flagship into a local/MIT feature.
  Evidence: opencut/core/lipsync_advanced.py + lipsync_echomimic.py (stubs return 501 ROUTE_STUBBED); live dub pipeline in dub_pipeline.py / auto_dub_pipeline.py / isochronous_translate.py / multilang_audio.py; github.com/bytedance/LatentSync (Apache-2.0 code; 1.6 trained 512×512); RESEARCH.md Competitive Landscape (Rask/HeyGen/sync.so)
  Touches: new opencut/core/lipsync_latentsync.py engine, engine_registry.py (lip-sync domain), checks.py, model_manager.py, dub-pipeline final-stage hook, a lip-sync route, cut-review/preview gate, both panels
  Acceptance: after a dub render, a lip-sync stage re-animates the speaker's mouth to the new audio behind an availability check and preview gate; falls back to audio-only dub when unavailable. Confirm the LatentSync *checkpoint* licence at the model-card level before defaulting it (Open Question 3); ship as opt-in if the weight licence proves restrictive.
  Complexity: L

### P3

- [ ] P3 — Re-aim the IC-Light relight stub at v1 (Apache-2.0), not v2
  Why: relight_iclight.py is titled "IC-Light V2 Per-Frame Relight", but IC-Light v2 is non-commercial and its full weights were never publicly released (HF Space demo only) — un-shippable under MIT. IC-Light v1 is Apache-2.0 and ships real weights, so the stub can become a real engine only by retargeting v1.
  Evidence: opencut/core/relight_iclight.py:1-4 ("IC-Light V2" stub); github.com/lllyasviel/IC-Light (v1 Apache-2.0); IC-Light/discussions/98 (v2 non-commercial, weights unreleased)
  Touches: opencut/core/relight_iclight.py, relight_video_lav.py / relight_diffrenderer.py (relight domain), engine_registry.py, checks.py, model_manager.py, relight route/preset
  Acceptance: IC-Light v1 is a selectable relight engine gated by availability check with a model download entry and registry test; the v2 framing is removed from docstrings/hints.
  Complexity: M

- [ ] P3 — Normalized 0–100 virality/hook score over existing engagement scoring
  Why: Opus Clip's headline paywalled feature is a single normalized 0–100 virality score; OpenCut already computes multi-dimensional engagement scoring for highlights but surfaces no comparable normalized number creators can sort/threshold on.
  Evidence: opusclip.canny.io (virality score 0–100); OpenCut engagement scoring (README Highlight & Shorts Generation: "hook strength, emotional intensity, pacing, quotability")
  Touches: highlight/engagement scoring module, highlights route response schema, panel Shorts/Highlights tab display
  Acceptance: each highlight/clip returns a normalized 0–100 score (deterministic mapping from existing engagement dimensions) shown in the panel and sortable; documented as a heuristic, not a guarantee.
  Complexity: S

- [ ] P3 — Upgrade speaker diarization default to pyannote community-1 (evaluate Sortformer)
  Why: OpenCut diarizes via pyannote.audio; pyannote/speaker-diarization-community-1 (CC-BY-4.0, "always freely accessible") is the new default with lower DER, and NVIDIA Sortformer reportedly more than halves DER vs the legacy pipeline — a low-risk accuracy bump for podcast/multicam workflows.
  Evidence: huggingface.co/pyannote/speaker-diarization-community-1 (CC-BY-4.0); emergentmind.com Sortformer; OpenCut pyannote diarization (README Captions & Transcription)
  Touches: diarization module(s), engine_registry.py (diarization domain), checks.py, model_manager.py, diarization availability test
  Acceptance: community-1 is the default diarization pipeline (legacy retained as fallback) with availability check and test; Sortformer evaluated and added as an optional engine if licence/availability permit.
  Complexity: S

