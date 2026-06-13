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

## P2 — Correctness / reliability

- [ ] P2 — CEP style.css is eight stacked re-skins, including two conflicting light themes
  Why: six full :root token redefinitions (12, 4462, 5386, 13215, 15466, 17214), two divergent html.theme-light blocks (16701 vs 17628), three different :focus-visible rules, and a triplicated prefers-reduced-motion block — the effective theme is "whatever the last pass overrode" and ~⅓ of 18k lines is dead weight. Consolidate to one token block per theme; then retokenize the ~340 stray hex literals.
  Where: extension/com.opencut.panel/client/style.css

## P3 — Lower-severity correctness, UX, packaging

- [ ] P3 — CEP: wizard and audio-preview dialogs bypass the overlay stack
  Why: aria-modal dialogs without focus trap or inert background (palette/preview modal do it right); route all four dialogs through activateOverlay/deactivateOverlay, give the context menu arrow-key support and focus restoration, and make one Escape close only the topmost surface.
  Where: extension/com.opencut.panel/client/main.js:11381-11527 (wizard/audioPreview/escape chain), index.html:4039, 4064

- [ ] P3 — UXP: i18n gaps and locale hygiene from the recent es push
  Why: es.json has zero Spanish diacritics across 1,381 keys and a {plural} hack that breaks agreement; en.json has 113 duplicated keys (uxp.agent.runtime.* / uxp.captions.runtime.* pasted twice); toast headings/dismiss label/status-tone regexes/shortcut labels are hardcoded English; formatI18n's unescaped string replace corrupts values containing $& or $'. Add a locale lint (key uniqueness, placeholder parity) to the workflow.
  Where: extension/com.opencut.uxp/locales/en.json, es.json; main.js:238-244, 2061-2117, 5541-5548, 7551-7557


## Research-Driven Additions

- [ ] P1 — Decide and execute brand/namespace disambiguation (gates all distribution work)
  Why: carried from the 2026-06-09 research pass with no roadmap entry; opencut.app (48K stars) is mid-rewrite (May 2026) and will re-dominate search when it relaunches; PyPI/winget/Homebrew/SEO all need an unambiguous token first.
  Evidence: github.com/OpenCut-app/OpenCut; opencut.app/roadmap; RESEARCH.md Open Question 1
  Touches: pyproject.toml project name, README tagline/badges, GitHub repo description/topics
  Acceptance: written decision in README or CONTRIBUTING; chosen PyPI name registered (placeholder publish acceptable).
  Complexity: M

- [ ] P2 — Upgrade bundled FFmpeg to 8.1 and expose D3D12/Vulkan encoders
  Why: FFmpeg 8.1 (Mar 2026) adds D3D12 H.264/AV1 encoding (vendor-agnostic Windows hardware encode for non-NVIDIA users) and Vulkan ProRes encode/decode; bundled binary is 8.0.1-essentials and core/hw_accel.py knows only nvenc/qsv/amf/videotoolbox.
  Evidence: 9to5linux.com FFmpeg 8.1 release notes; ffmpeg/ffmpeg.exe -version; opencut/core/hw_accel.py:70-78
  Touches: ffmpeg/ (bundled binaries), installer/InstallerBuilder.ps1, OpenCut.iss, opencut/core/hw_accel.py, export routes/presets
  Acceptance: hw_accel detection lists D3D12 H.264/AV1 encoders when the runtime supports them and /video/export can select them; bundled ffmpeg -version reports 8.1.x.
  Complexity: M

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

- [ ] P3 — Track onnxruntime 1.26 hardening release and raise the floor when GA
  Why: onnxruntime 1.26 (in development) hardens multiple OOB/overflow scenarios (Attention mask OOB write, MaxPoolGrad bounds, SVM/TreeEnsemble, RNN sequence_lens); 15+ core modules import onnxruntime via the ai/insightface/rembg stack with floor >=1.25,<2.
  Evidence: github.com/microsoft/onnxruntime/releases (1.26 notes); pyproject.toml:83,90
  Touches: pyproject.toml, requirements files, docs/PYTHON_ADVISORIES.md
  Acceptance: floor raised to >=1.26 once released and advisory doc records the hardening rationale.
  Complexity: S

- [ ] P3 — Publish the server package to PyPI via trusted publishing (after brand decision)
  Why: pip install of the server/CLI is the cheapest distribution surface and currently impossible (no PyPI package); trusted publishing avoids long-lived tokens in CI.
  Evidence: docs.pypi.org/trusted-publishers/; RESEARCH.md distribution assessment
  Touches: .github/workflows/build.yml (publish job), pyproject.toml metadata
  Acceptance: pip install of the chosen name installs the opencut CLI/server from PyPI; releases publish automatically on tag.
  Complexity: M

- [ ] P2 — Normalize live planning-file references to ROADMAP.md and RESEARCH.md
  Why: live user-facing code still points at removed ROADMAP-NEXT.md/PROJECT_CONTEXT.md files and past "ships in v1.28.x/v1.29.0" promises, creating misleading install and release guidance.
  Evidence: CONTRIBUTING.md:106,124; opencut/registry.py:648-689; opencut/routes/wave_h_routes.py:77,437; opencut/routes/wave_k_routes.py:40; opencut/core/audio_reactive_fx.py:64
  Touches: CONTRIBUTING.md, scripts/check_doc_sizes.py, scripts/release_smoke.py, opencut/registry.py, Wave H/K route modules, optional-engine stub messages, related tests
  Acceptance: `rg "ROADMAP-NEXT|PROJECT_CONTEXT|TODO.md|RESEARCH_FEATURE_PLAN" . --glob "!docs/archive/**" --glob "!tests/**"` returns no user-facing planning references except intentional historical comments; missing-engine messages point to live capability metadata or ROADMAP.md.
  Complexity: S

- [ ] P2 — Move CEP panel off unsupported Vite 5 with HGFS-safe regression evidence
  Why: Vite 5.4.21 remains pinned with a documented advisory waiver because Vite 6+ regressed VMware HGFS paths, but Vite's maintained release line has moved to 8.x with 7.3/6.4 backports.
  Evidence: extension/com.opencut.panel/package.json:22; docs/NODE_ADVISORIES.md:13,27-34,66-74; tests/test_panel_node_entrypoints.py; vite.dev/releases
  Touches: extension/com.opencut.panel/package.json, package-lock.json, scripts/panel-node-gate.ps1, docs/NODE_ADVISORIES.md, extension panel build/audit tests, CI release smoke
  Acceptance: panel build/audit uses a supported Vite line, the documented Vite advisory waiver is removed, and Windows UNC/HGFS-safe `*:win` entrypoints plus Linux CI build/verify still pass.
  Complexity: M

- [ ] P1 — Restore public release channel parity for v1.33.x
  Why: package metadata is at v1.33.1, README still displays v1.33.0, the local repo only has tag v1.33.0, and the public GitHub Releases page still reports v1.25.1 as latest, leaving users without current installer/source artifacts for recent security and reliability work.
  Evidence: `gh release list --repo SysAdminDoc/OpenCut --limit 10`; `git tag --list "v1.33*"`; README.md:3,89; pyproject.toml:7; .github/workflows/build.yml:237-323
  Touches: .github/workflows/build.yml, scripts/release_smoke.py, scripts/sync_version.py, README.md release guidance, CHANGELOG.md release headings
  Acceptance: a current v1.33.x GitHub Release/tag or documented pre-release policy exists; README version text is covered by the sync gate; release artifacts include server package, Windows installer, Linux packages, and SBOM/provenance artifacts or explicit skipped-platform notes.
  Complexity: M

- [ ] P2 — Add a local-only privacy mode that fails closed across cloud-capable providers
  Why: OpenCut's local-first promise is central, but LLM, video-LLM, cloud TTS, stock search, telemetry, and social upload code paths can contact external services when configured without a single global guardrail.
  Evidence: README.md local/no-cloud positioning; opencut/core/llm.py; opencut/core/video_llm.py; opencut/core/voice_overdub.py; opencut/core/social_post.py; opencut/core/stock_search.py; opencut/core/telemetry_aptabase.py
  Touches: opencut/config.py, cloud-capable core modules, social/LLM/TTS/stock routes, CEP/UXP settings, release-smoke or pytest no-network checks
  Acceptance: `OPENCUT_LOCAL_ONLY=1` and the matching UI setting hide or disable cloud controls, route calls fail with structured local-alternative errors, and tests prove external-provider functions are not invoked in local-only mode.
  Complexity: M

- [ ] P2 — Add native auto-editor v30 compatibility canarying
  Why: OpenCut's adapter prefers the native auto-editor binary, but packaging/model-card guidance still points to the legacy v29 Python package while upstream v30.5.0 is the active Nim line.
  Evidence: github.com/WyattBlue/auto-editor release 30.5.0; pyproject.toml:103,172; opencut/core/auto_edit.py; opencut/model_cards.py:345-352; opencut/registry.py:459-464
  Touches: opencut/core/auto_edit.py, opencut/model_cards.py, opencut/registry.py, generated feature readiness/model cards, tests/test_auto_editor_json.py or equivalent fixture tests
  Acceptance: a fixture or smoke test validates current v30 `auto-editor --export json` output, v29 fallback remains covered, and install hints distinguish native v30 binary usage from the legacy pip fallback.
  Complexity: M

- [ ] P2 — Add Premiere caption-export preflight and recovery
  Why: Premiere 26 caption export/burn settings have community-reported failure modes; OpenCut's caption workflows should warn before a failed host handoff and preserve a sidecar/burn-in fallback.
  Evidence: https://www.reddit.com/r/premiere/comments/1s672iq/premiere_pro26_captions_and_export_settings/?tl=en; Adobe caption export docs; opencut/routes/captions.py; opencut/core/caption_burnin.py; extension/com.opencut.panel/client/index.html caption burn-in UI
  Touches: CEP host bridge, UXP host bridge, caption export/burn-in routes, caption status UI, tests for host-status mapping and fallback sidecar generation
  Acceptance: panel preflight reports caption track/export readiness before host export, unknown or unsafe host states produce a deterministic warning plus SRT/sidecar or OpenCut burn-in fallback, and tests cover success/warning/error status mapping.
  Complexity: M

- [ ] P2 — Add versioned visual-search engines beyond fixed CLIP ViT-B/32
  Why: Adobe Media Intelligence makes on-device visual retrieval table-stakes, while OpenCut's semantic search is hard-coded to `openai/clip-vit-base-patch32` and a single cache schema.
  Evidence: Adobe Media Intelligence docs; Hugging Face SigLIP 2; opencut/core/semantic_video_search.py:43; tests/test_object_intel.py semantic search coverage
  Touches: opencut/core/semantic_video_search.py, footage index/cache schema, model cards/registry, search routes, CEP/UXP Search UI, semantic-search tests
  Acceptance: semantic search supports a versioned engine registry for CLIP/OpenCLIP/SigLIP-style models, cache keys include engine/model/schema version, and a benchmark fixture compares retrieval quality against the current CLIP baseline.
  Complexity: L

- [ ] P3 — Add CJK, Bengali, and RTL caption font-fallback render fixtures
  Why: current release gates preserve caption Unicode/text shaping, but community requests around CJK and Bengali captions show users need proof that rendered captions have real glyph fallback and line breaking, not only round-trip strings.
  Evidence: https://github.com/OpenCut-app/OpenCut/issues/817; https://github.com/OpenCut-app/OpenCut/issues/790; scripts/release_smoke.py caption-unicode/text-shaping gates; opencut/core/caption_unicode_validation.py; opencut/core/styled_captions.py
  Touches: caption renderer/font resolver, opencut/core/styled_captions.py, opencut/core/caption_burnin.py, tests/test_text_shaping_gate.py, tests/test_caption_unicode_validation.py, CEP/UXP caption style settings
  Acceptance: automated fixtures render CJK, Bengali, and RTL/Arabic captions with non-missing glyphs and stable line breaking; UI surfaces the selected fallback font or a warning when no capable font is available.
  Complexity: M

- [ ] P2 — Pin PyInstaller >=6.0 in CI build workflow
  Why: CVE-2025-59042 (CVSS 7.0) is a local privilege escalation during PyInstaller bootstrap on Linux/macOS via filenames containing `?`. CI installs PyInstaller without a version pin (`pip install pyinstaller` at .github/workflows/build.yml:59), so Linux Flatpak/AppImage build artifacts could be produced by a vulnerable version.
  Evidence: github.com/advisories/GHSA-p2xp-xx3r-mffc; .github/workflows/build.yml:59
  Touches: .github/workflows/build.yml (pip install line), pyproject.toml dev extras (optional)
  Acceptance: CI installs `pyinstaller>=6.0`; Linux/macOS build artifacts are produced by a patched version.
  Complexity: S

- [ ] P2 — Add Python 3.13 to CI test matrix
  Why: CI runs only Python 3.12. Flask 3.1.3 and core deps (click, rich, waitress, psutil) support 3.13. ML deps may lag on cp313 wheels, but a matrix entry for the core test suite catches breakage before users hit it. Python 3.11 reaches EOL October 2027; 3.13 testing is forward-looking.
  Evidence: .github/workflows/build.yml:36 (single python-version: '3.12'); pypi.org/project/Flask/ (3.13 tested)
  Touches: .github/workflows/build.yml (matrix strategy), pyproject.toml classifiers
  Acceptance: CI runs the core test suite on Python 3.13; known ML-dep skips are documented, not silent failures.
  Complexity: S

- [ ] P2 — Raise Flask and Waitress dependency floors for recent CVEs
  Why: pyproject.toml pins flask>=3.0 (CVE-2026-27205 Vary:Cookie info disclosure fixed in 3.1.3) and waitress>=3.0 (CVE-2024-49768/49769 request smuggling/socket exhaustion fixed in 3.0.1). Both are low-severity for localhost, but the permissive floors let pip resolve vulnerable versions and break the pip-audit gate.
  Evidence: CVE-2026-27205 (sentinelone.com); CVE-2024-49768/49769 (stack.watch/product/agendaless/waitress); pyproject.toml:35,39
  Touches: pyproject.toml (flask>=3.1.3, waitress>=3.0.1)
  Acceptance: `pip audit` passes with no Flask/Waitress findings; pip resolves patched versions on fresh install.
  Complexity: S

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
