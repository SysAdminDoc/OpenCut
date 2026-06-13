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

- [ ] P2 — Add SAM 3 engine for text-prompted object removal and tracking
  Why: SAM 3 (Meta, 2025-11-19, commercial-permissive SAM License) segments and tracks by concept text prompt — removes the click-and-track friction of the SAM2+ProPainter pipeline and enables "remove every watermark/logo instance" workflows; Premiere 26's native Object Mask raises user expectations here.
  Evidence: blog.roboflow.com/what-is-sam3; marktechpost SAM 3 release post; no sam3 references in opencut/core/
  Touches: opencut/core/ (new sam3 module beside the SAM2 object-removal module), engine registry, opencut/routes/video_specialty.py, panel Video tab
  Acceptance: object-removal accepts a text prompt when engine=sam3; registry lists sam3 with auto-fallback to sam2; model manager can download/delete the checkpoint.
  Complexity: L

- [ ] P2 — Multicam: add visual speech cues (lip movement, shot type) to cut decisions
  Why: DaVinci Resolve 20 Multicam SmartSwitch and Phantom Wraith combine audio with lip-movement and wide/close shot detection; opencut/core/multicam.py is diarization-only and mis-cuts on cross-mic bleed; multimodal diarization (face+voice mapping) already exists as building blocks.
  Evidence: cined.com Resolve 20 release coverage; phantomeditor.video Wraith; opencut/core/multicam.py (no visual input)
  Touches: opencut/core/multicam.py, multimodal diarization modules, /video/multicam-cuts route, panel Timeline tab
  Acceptance: multicam cut generation offers an "audio+visual" mode that prefers the on-camera active speaker on a test clip with audio bleed.
  Complexity: L

- [ ] P2 — Premiere 26 compatibility and positioning audit
  Why: Adobe rebranded to "Premiere" 26.x (Jan 2026) and shipped native Object Mask, Generative Extend, Media Intelligence, and 90+ Film Impact transitions — README comparisons predate this; CEP HostList [13.0,99.9] should cover 26.x but no documented test pass exists.
  Evidence: blog.adobe.com Jan 2026 announcement; extension/com.opencut.panel/CSXS/manifest.xml:15; README.md feature tables
  Touches: README.md, docs/UXP_MIGRATION.md, extension manifests, .github/workflows/adobe-premierepro-versions.yml
  Acceptance: documented CEP+UXP smoke pass on Premiere 26.x; README positioning updated to lead with capabilities Adobe does not bundle (silence-cut-to-timeline, stems, TTS/voice clone, local agent, social pipeline).
  Complexity: M

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

- [ ] P2 — Add versioned visual-search engines beyond fixed CLIP ViT-B/32
  Why: Adobe Media Intelligence makes on-device visual retrieval table-stakes, while OpenCut's semantic search is hard-coded to `openai/clip-vit-base-patch32` and a single cache schema.
  Evidence: Adobe Media Intelligence docs; Hugging Face SigLIP 2; opencut/core/semantic_video_search.py:43; tests/test_object_intel.py semantic search coverage
  Touches: opencut/core/semantic_video_search.py, footage index/cache schema, model cards/registry, search routes, CEP/UXP Search UI, semantic-search tests
  Acceptance: semantic search supports a versioned engine registry for CLIP/OpenCLIP/SigLIP-style models, cache keys include engine/model/schema version, and a benchmark fixture compares retrieval quality against the current CLIP baseline.
  Complexity: L


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
