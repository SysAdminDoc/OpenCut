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

- [ ] P2 — Run documented CEP+UXP smoke pass on Premiere 26.x
  Why: README positioning now leads with OpenCut-unique capabilities vs Adobe 26.x. Manifests mathematically cover 26.x (CEP [13.0,99.9], UXP minVersion 25.6). Remaining: live smoke test on an actual Premiere 26.x install.
  Blocked: Premiere 26.x license/installation.
  Acceptance: documented smoke test pass on Premiere 26.0 or 26.2.x for both CEP and UXP panels.
  Complexity: S

- [ ] P3 — Publish the server package to PyPI via trusted publishing
  Why: pip install of the server/CLI is the cheapest distribution surface and currently impossible (no PyPI package); trusted publishing avoids long-lived tokens in CI. Brand decision is DONE — distribution name is `opencut-ppro` (pyproject.toml + README "Naming & distribution").
  Evidence: docs.pypi.org/trusted-publishers/; RESEARCH.md distribution assessment
  Touches: .github/workflows/build.yml (publish job), pyproject.toml metadata
  Blocked: needs the maintainer's PyPI account / trusted-publisher config (credential-gated).
  Acceptance: `pip install opencut-ppro` installs the opencut CLI/server from PyPI; releases publish automatically on tag.
  Complexity: M

- [ ] P2 — Move CEP panel off unsupported Vite 5 with HGFS-safe regression evidence
  Why: Vite 5.4.21 remains pinned with a documented advisory waiver because Vite 6+ regressed VMware HGFS paths, but Vite's maintained release line has moved to 8.x with 7.3/6.4 backports.
  Evidence: extension/com.opencut.panel/package.json:22; docs/NODE_ADVISORIES.md:13,27-34,66-74; tests/test_panel_node_entrypoints.py; vite.dev/releases
  Touches: extension/com.opencut.panel/package.json, package-lock.json, scripts/panel-node-gate.ps1, docs/NODE_ADVISORIES.md, extension panel build/audit tests, CI release smoke
  Acceptance: panel build/audit uses a supported Vite line, the documented Vite advisory waiver is removed, and Windows UNC/HGFS-safe `*:win` entrypoints plus Linux CI build/verify still pass.
  Complexity: M

- [ ] P3 — Add Homebrew tap for macOS CLI distribution (after PyPI)
  Why: macOS has no package-manager install path — users must clone + pip-install manually. A Homebrew tap gives macOS users `brew install opencut-ppro` for the CLI/server. Depends on PyPI publish (existing P3). Brand decision is DONE (`opencut-ppro`).
  Evidence: docs.brew.sh/Python-for-Formula-Authors; Homebrew accepts Python apps even without PyPI; no existing tap
  Touches: new homebrew-opencut-ppro tap repo, formula file, CI publish workflow
  Acceptance: `brew install <tap>/opencut-ppro` installs the CLI/server on macOS; formula auto-updates on new PyPI releases.
  Complexity: M

- [ ] P3 — Add winget package manifest for Windows distribution (after release parity + code signing)
  Why: Windows users discover software via winget; no manifest exists. Requires a stable installer URL (GitHub Release .exe) and ideally a code-signed binary for SmartScreen reputation. Depends on release parity (P1) and code signing budget ($216-575/yr OV/EV certificate).
  Evidence: github.com/microsoft/winget-pkgs (12,850+ packages); no existing OpenCut manifest; signmycode.com pricing. Brand decision is DONE — publisher token `SysAdminDoc.OpenCut` / dist token `opencut-ppro`.
  Touches: winget manifest YAML (submitted as PR to microsoft/winget-pkgs), CI release workflow (signed installer upload)
  Acceptance: `winget install SysAdminDoc.OpenCut` installs OpenCut on Windows; manifest auto-updates via GitHub Release URLs.
  Complexity: M

## Research-Driven Additions (2026-06-14)

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

