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

- [ ] P3 — CEP: --text-faint at 10-11px fails AA contrast; no prefers-contrast support
  Why: ~3.5:1 against card surfaces; bump small text to --text-muted as part of the style.css consolidation.
  Where: extension/com.opencut.panel/client/style.css (e.g. command-palette-hint ~3950)

- [ ] P3 — UXP: i18n gaps and locale hygiene from the recent es push
  Why: es.json has zero Spanish diacritics across 1,381 keys and a {plural} hack that breaks agreement; en.json has 113 duplicated keys (uxp.agent.runtime.* / uxp.captions.runtime.* pasted twice); toast headings/dismiss label/status-tone regexes/shortcut labels are hardcoded English; formatI18n's unescaped string replace corrupts values containing $& or $'. Add a locale lint (key uniqueness, placeholder parity) to the workflow.
  Where: extension/com.opencut.uxp/locales/en.json, es.json; main.js:238-244, 2061-2117, 5541-5548, 7551-7557


## Research-Driven Additions

- [ ] P1 — Cut and publish release v1.33.0 with the June hardening pass
  Why: latest GitHub release is v1.25.1 (2026-04-20) while the tree is v1.32.0 plus unreleased security fixes (arbitrary file overwrite via output_path sinks, FFmpeg escaping, CEP cut-review repair) — users on releases are seven versions behind known fixes.
  Evidence: github.com/SysAdminDoc/OpenCut/releases; CHANGELOG.md [Unreleased]; commits 88bdf62, b7434f6, 81a8a85, 7521f61
  Touches: scripts/sync_version.py (all 19 targets), CHANGELOG.md, .github/workflows/build.yml (tag v1.33.0)
  Acceptance: GitHub release v1.33.0 exists with Windows installer + server bundles attached and CHANGELOG section dated; README badge matches.
  Complexity: S

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

- [ ] P2 — Expand animated caption template library toward market parity
  Why: 6 animation presets (core/animated_captions.py ANIMATION_PRESETS) and 19 static styles vs Submagic 35+ and CapCut 50+; template breadth is the most visible caption differentiator, and CapCut now paywalls auto-captions entirely.
  Evidence: opencut/core/animated_captions.py:27; submagic.co; eesel.ai CapCut pricing 2026
  Touches: opencut/core/animated_captions.py, caption_styles.py, panel Captions tab, locale files
  Acceptance: >=20 animation presets selectable with preview, defined via a data-driven preset schema (JSON) so presets can be added without code changes.
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
