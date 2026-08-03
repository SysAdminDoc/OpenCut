# Research — OpenCut
Date: 2026-08-02 — replaces all prior research (previous pass: 2026-07-29).
Confidence: claims are Verified unless labelled otherwise.

## Executive Summary

OpenCut is a local-first automation and interchange layer for Adobe Premiere — 1,544 shipped
Flask routes across 107 blueprints, 623 core modules, CEP and UXP panels, CLI/REST/MCP access,
durable jobs, and a strong culture of retracting claims it cannot prove. The 2026-07-29 research
cycle closed its top items: trusted-host policy now gates the CSRF bootstrap
(`opencut/trusted_hosts.py`), CEP Node/mixed-context privileges are gone from
`extension/com.opencut.panel/CSXS/manifest.xml`, MCP speaks both `2026-07-28` and the legacy
revision, and generated readiness no longer advertises the three terminal-stub adapters.

The shape of the problem has now changed. Breadth is no longer the constraint — **delivery is**.
Three independent measurements say the same thing: the product the maintainer tests is not the
product a user receives. The recommended Windows installer cannot make the panel load on any
Premiere newer than CC 2022. The test suite cannot run from a clone. The tested dependency set
violates four of the project's own declared constraints. Meanwhile Adobe has put a published
expiry — September 2026 — on the ExtendScript layer that 133 of OpenCut's user-facing
capabilities depend on, and shipped several OpenCut features as first-party in Premiere 26.2/26.3.

Top opportunities, in priority order:

1. Fix the WPF installer's CSXS range (7–12) so the CEP panel loads on Premiere CC 2023+.
2. Make the UXP panel carry the product before ExtendScript support ends in September 2026.
3. Make the green test baseline reproducible from a fresh clone.
4. Test against the dependency versions the project actually declares.
5. Advance the FFmpeg snapshot floor past the July 2026 fixes and ship the `full` build.
6. Publish a release — the newest downloadable artifact is 21 versions behind the source tree.
7. Stop `Install.ps1` force-killing Premiere, and stop the uninstaller `rmdir /s /q`-ing an
   unnormalised user-chosen path.
8. Give the 1,253 routes with no UI affordance either a surface or a sunset.
9. Prune the capabilities Premiere 26.2/26.3 made first-party; redirect that effort to the
   diarization-driven editing and interchange gaps the whole commercial field meters.
10. Close the interchange gaps that no one else has: MLT (Kdenlive/Shotcut) export and OTIO
    `Transition` emission.

## Product Map

- **Core workflows:** select Premiere media → run local cut / caption / audio / video / search /
  delivery automation → review a proposed result → write approved changes through CEP or UXP →
  export media, captions, timelines, review artifacts, or standards packages.
- **Personas:** solo and small-team Premiere editors; podcast and social-video editors;
  caption/transcript operators; delivery/QC operators; technical users automating via CLI/REST/MCP.
- **Platforms and distribution:** Python 3.11–3.14; Windows WPF installer + PyInstaller bundle;
  source launchers for Windows/macOS/Linux; Docker; Flatpak/AppImage scaffolds
  (`io.github.sysadmindoc.opencut.yml`, `packaging/linux/`); CEP for Premiere 2019+, UXP for 25.6+.
  MIT; bundled FFmpeg obligations handled separately.
- **Integrations and data flow:** panels, CLI, and MCP call a loopback-only Flask service; FFmpeg
  and optional local/cloud engines process media; Premiere receives approved host actions; SQLite
  and `~/.opencut` persist queue, journal, review, index, plugin, settings, and diagnostic state.
- **Product boundary:** augment Premiere; keep core inference local, network use explicit,
  destructive work checkpointed, and unavailable dependencies honestly unavailable.

## Competitive Landscape

- **Adobe Premiere (the host, and now the competitor).** 26.2 shipped the **Sequence Index** panel —
  searchable/sortable clip spreadsheet with column chooser, filter funnel, click-to-jump, CSV export.
  26.3 shipped **Single-Word Captions** (word-synced captions generated from the transcript, styled
  with native caption tools). 25.2 shipped **Media Intelligence** + natural-language Search; 25.6
  added audio search and bulk bleep/mute; Auto-Match Loudness and Text-Based-Editing "Delete Pauses /
  Delete Filler Words" already existed; April 2026 brought on-device Speechmatics STT that Adobe
  claims is 12–16% better than Whisper-powered creative tools. **Learn:** the host now defines the
  baseline — compete on breadth of caption templates, cross-project scope, and exportable artifacts,
  not on the capability itself. **Avoid:** rebuilding Sequence Index parity, consumer loudness,
  profanity bleeping, or basic pause removal as headline features.
- **auto-editor** (github.com/WyattBlue/auto-editor, 4.6k★). Best-in-class NLE round-trip: Premiere
  XML, FCP, Resolve, **Kdenlive and Shotcut MLT** with keyframed effects preserved; 31.3.0 added
  linked A/V dissolve transitions that survive into those exports; 31.4.0 added partial-lossless
  rendering (complete GOPs copied). **Learn:** interchange fidelity is the moat. **Avoid:** its new
  direction — distributed builds now gate rendering above 3200×1800 and all professional-NLE export
  behind a licence key while the repo stays Unlicense. That is OpenCut's clearest positioning opening.
- **premiere-pro-mcp** (github.com/leancoderkavy/premiere-pro-mcp, 162★, MIT). The closest
  architectural rival: 280 tools, capability profiles with unsafe scripting off by default, and
  **safe edit plans bound to a SHA-256 plan token** (preview → approve → apply). **Learn:** bind the
  approval to a hash of the exact plan, and ship an MCPB bundle for one-click agent install.
  **Avoid:** nothing — this is convergent design done well.
- **Subtitle Edit** (13.7k★, MIT). A full Avalonia rewrite now ships Windows/macOS/Linux + Flatpak
  from one codebase, with a selectable OCR engine (Tesseract/nOCR/Ollama/PaddleOCR), container input
  straight from `.mkv`/`.mp4`, and waveform + spectrogram + shot-change tools in one visualiser.
  **Learn:** bitmap-subtitle (PGS/VobSub) OCR is a real gap. Its issue #10180 (faster-whisper cuBLAS
  failure on RTX 50-series unless float16) is a hardware trap OpenCut inherits verbatim.
- **Kdenlive / Shotcut** (both pushed 2026-08-01/02). Kdenlive 26.04 rewrote OTIO import/export
  against the C++ library; Shotcut's *published roadmap* still lists "OpenTimelineIO import/export",
  "CMX EDL import", and "Kdenlive XML export" as not done, alongside "automatic silence detection",
  "multi-camera editing", and "background removal" — i.e. Shotcut's roadmap is largely OpenCut's
  shipped feature set. **Learn:** OpenCut can be the bridge before either side builds it.
- **OpenTimelineIO** (0.18.1, 2025-11-09; the 1.0 milestone was due 2026-04-10 and is overdue).
  Issue #62 — **OTIO has no caption/subtitle schema at all**; #446/#442/#445 — the AAF writer supports
  only cross-dissolves, no markers, no essence. **Learn:** bound the interchange claim to what the
  schema can actually carry, and emit `Transition` items rather than bare cuts.
- **FireCut / AutoCut / AutoPod / PremiereCopilot.** The paid Premiere field meters exactly what
  OpenCut can give away: FireCut gates everything but silence cutting and prices transcription by
  the hour (25h Pro / 100h Max); AutoCut's $9.90 tier is silence-only of ten tools; AutoPod charges
  $29/mo for multicam-by-speaker; PremiereCopilot ships a free tier with daily quotas at a $7.99
  floor. **Learn:** the transcription-hour meter is the sharpest wedge, and AutoCut's ten named verbs
  ("AutoZoom", "AutoChapters") communicate better than "1,544 API routes". **Avoid:** competing on
  price framing against a funded SEO operation at a $7.99 floor.
- **pyannote.audio 4.x / Podcastle / Riverside.** pyannote 4.0 swapped AHC for VBx clustering in
  `speaker-diarization-community-1` and added an **`exclusive_speaker_diarization` output built
  specifically to reconcile fine-grained diarization against imprecise ASR timestamps** — OpenCut
  references `community-1` in `opencut/core/diarize.py` but has zero hits for `exclusive_speaker`, so
  multicam cuts still land on ASR segment boundaries. Podcastle's "Magic Dust" and Riverside's Magic
  Audio collapse the whole cleanup chain into one button; OpenCut has every component
  (`opencut/data/workflow_presets.json`) and no single verb. **Learn:** both are small, high-leverage.
  **Avoid:** Riverside's lifetime-not-monthly free allocation, the most-complained-about pattern found.
- **Descript / Opus Clip.** Descript exposes an **"AI Models" settings section** and per-project model
  choice; Opus Clip locks its 0–99 virality score behind the paid tier and users treat it as triage,
  not verdict. **Learn:** ship an *explainable* highlight score with named component signals the user
  can re-weight, and tell users exactly which model touched their media. **Avoid:** media-minute +
  credit hybrids and clip-expiry mechanics — the most-complained-about patterns in the category.

## Security, Privacy, and Reliability

- **The WPF installer cannot make the panel load on modern Premiere — Verified.**
  `installer/src/OpenCut.Installer/Models/AppConstants.cs:38` sets `CsxsVersions = { 7..12 }`,
  consumed by `Services/RegistryManager.cs:65-81`. Premiere CC 2023+/2025 use CSXS 13–18. Both
  `Install.ps1:555-561` and `OpenCut.iss:105-116` cover 7–18, and the PowerShell script carries a
  comment naming this exact regression ("without PlayerDebugMode set, so the panel never loaded").
  Only the shipping installer — the README's recommended Windows path — was missed.
- **`Install.ps1` force-kills Premiere — Verified.** `Install.ps1:138-143` and `:217-229` take the
  last column of every `netstat -ano | Select-String ":5679 "` row. That regex matches `ESTABLISHED`
  rows, where the PID belongs to the *client* — with the CEP panel connected, that PID is Premiere.
  Re-running the installer or uninstaller kills it and loses unsaved project work.
  `installer/.../ProcessKiller.cs:93` gets this right with `| findstr LISTENING`.
- **Uninstall can delete a user directory — Verified.**
  `installer/.../Pages/OptionsPage.xaml.cs:207` stores `PathBox.Text.Trim()` raw; the
  `Path.GetFullPath` result computed at `:238` is used only for a length check and discarded. No
  drive-root rejection, no non-empty-directory warning, no app-name append. `UninstallEngine.cs:175`
  then runs `rmdir /s /q "{installDir}"`. Choosing `D:\` or an existing `D:\Videos` in the folder
  browser makes uninstall wipe it.
- **The green baseline is not reproducible from a clone — Verified.** Nine test modules read markdown
  that `.gitignore`'s blanket `*.md` rule excludes from the repo:
  `test_uxp_migration_docs.py`, `test_uxp_macos_http.py`, `test_uxp_webview_scaffold.py`,
  `test_uxp_webview_permission_split.py`, `test_uxp_filesystem_permission.py`,
  `test_cep_uxp_parity_catalogue.py`, `test_windows_arm64_doc.py`, `test_roadmap_mirror.py`, and
  `test_local_release_policy.py` (which reads the gitignored `CLAUDE.md`). All use unguarded
  `read_text()`. `README.md:475` also points readers at the untracked `docs/UXP_MIGRATION.md`.
  `tests/test_fresh_clone_integrity.py:57` exists to catch exactly this but its regex is
  `\[[^\]]*\]\(([^)]+)\)` — markdown link syntax only — so the README's backticked reference passes.
  Sixteen of 28 `docs/*.md` files are untracked, including `PLUGIN_AUTHORING.md`,
  `SKILL_AUTHORING.md`, `TELEMETRY.md`, and `UXP_MIGRATION.md`.
- **The tested stack violates four declared constraints — Verified.** In this machine's CPython
  3.12 environment — the interpreter the recorded baseline names:
  `opencv-python` 4.11.0.86 against a declared `>=5,<6`;
  `edge-tts` 7.2.7 against `<7`; `cryptography` 49.0.0 against `<49`; `scenedetect` 0.6.7.1 against
  `>=0.7.1`. Two are major-version boundaries, and PySceneDetect 0.7 is a documented breaking release
  (VFR handling, seconds-vs-frames option semantics). Anyone installing per `pyproject.toml` runs code
  paths the suite has never executed. `scripts/check_dependency_matrix.py` resolves declared lanes; it
  does not compare them to what is installed.
- **FFmpeg: wrong variant, and the floor predates the fixes — Verified.** The bundled binary reports
  `8.1.2-essentials_build-www.gyan.dev` with `--enable-nvdec --enable-cuvid` and without
  `libsvtav1`, `libdav1d`, `whisper`, `libplacebo`, `vulkan`, `libjxl`, or `libvvenc` (all `full`-only).
  CVE-2026-64832 (NVDEC double-free), -64833 (S/PDIF DTS OOB read), -64835 (ADX OOB), and -66041
  (`vf_quirc` heap OOB write) list 8.1.2 as affected; their fix commits (`4c6217477f`, `6f80e27654`,
  `1836ef9684`, `4da9812e25`) landed on master 2026-07-02…07-05 and have **not** been backported to
  `release/8.1` — three independent signals agree (ffmpeg.org/security.html lists only CVE-2026-8461
  and -30999 for the 8.1.2/8.0.3 point releases; the Debian tracker marks all four unfixed across every
  suite; each fix hash resolves to exactly one commit, on master). There is no 8.1.3 and no 8.2. `opencut/core/ffmpeg_provenance.py` already implements
  a git-master lane, but `SNAPSHOT_FLOOR_DATE = "2026-06-10"` predates all four fixes and
  `RELEASE_FLOOR = (8,1,2)` still accepts the vulnerable release lane. **This item is filed as blocked
  in `Roadmap_Blocked.md` on the grounds that no fixed release exists — that is true of the release
  lane only. The snapshot lane makes it actionable today.** NVDEC is the highest-reachability of the
  four given the build's own configuration.
- **Readiness still has 27 unverifiable records — Verified.** `opencut/_generated/feature_readiness.json`
  contains 27 auto-generated records reporting `state: "available"` with `impl_module: ""` — including
  `audio.demucs`, `video.sam2`, `video.mediapipe`, `editing.auto-editor`, `auto.otio`. An empty
  `impl_module` is precisely what let `auto.deblur-motion`, `auto.searaft`, and `auto.track-cutie`
  advertise as available while terminating in `NotImplementedError`; that was fixed for those three
  and the blind spot remains for 27 more.
- **No published artifact since 2026-04-20 — Verified.** The newest GitHub Release is v1.25.1; the
  source tree is v1.46.0. v1.34.0 through v1.46.0 carry no git tag. The README already discloses this,
  which is honest, but it means every user is 21 versions behind including on the security fixes above.
- **Positive, and worth preserving:** trusted-host policy, CSRF, path validation, SSRF controls,
  durable queue/journal/checkpoints, corruption quarantine, immutable review versions, redacted
  diagnostics, request correlation, plugin trust/isolation, ZIP-slip defence in `PayloadExtractor.cs`,
  hash-verified user-data backup in `UserDataRemovalService.cs`, and a Dockerfile with pinned FFmpeg
  source + SHA-256, `--require-hashes` pip, and a non-root uid — all verified sound this pass.

## Architecture Assessment

- **82% of the product has no user surface — Verified.** Of 1,518 non-stub routes, literal-path
  references resolve to 189 from `client/main.js`, 77 from the UXP panel (211 across all panel JS),
  38 from `core/command_palette.py`, 93 from `mcp_server.py`, and 5 from `cli.py`. **1,253 routes are
  referenced by none of them.** Method caveat: substring matching against literal paths — but there
  are zero template-literal path constructions in `client/main.js` and eight in the UXP panel, so the
  margin is small. This is the dominant architectural fact and it should drive triage: a route with no
  surface is either an integration API that should be documented as such, or dead weight.
- **The CEP→UXP migration is measuring the wrong thing — Verified.** Adobe's Premiere scripting guide
  states verbatim that ExtendScript integrations are supported "through September 2026", that no
  further ExtendScript API work is planned, and that CEP 12 is the last CEP release;
  `extension/PANEL_PARITY.json` already records `"$adobe_cep_eol": "approximately 2026-09"`.
  `opencut/_generated/uxp_migration_dashboard.json` tracks the **18 ExtendScript host functions** and
  reports nearly all as `direct_uxp` — so the dashboard reads as "almost done". Measured by feature
  surface instead, **133 routes are CEP-only**, including `/audio/separate`, `/captions/translate`,
  `/audio/enhance`, `/captions/animated/render`, `/export/preset`, `/full`, and `/install-whisper`.
  A UXP-only user cannot install Whisper. The host-write layer itself is in good shape — the panel
  already feature-detects `project.lockedAccess()` for 26.3 and falls back correctly
  (`extension/com.opencut.uxp/main.js:812-833`) — the gap is the 133 features, not the bridge.
- **Applying cuts clip-by-clip through the host is the wrong mechanism — Likely.**
  `ocApplySequenceCuts` razors per clip per track. A documented Adobe forum report has a FireCut
  silence pass producing "probably more than 1000" cuts and leaving Premiere "unusably laggy" on a
  4090/i9-14900K/64 GB machine. OpenCut's own panel copy already mis-reports this (`r.applied` counts
  clip-removals per track, so 3 cuts on a 1V/2A sequence toasts "Applied 9 cuts"). Building the cut
  list and importing it as timeline interchange is the mechanism that scales; it also needs testing
  against OTIO #569 (FCP XML round-trip losing trim points in Premiere) and auto-editor #70
  (multi-audio-track XML).
- **The CEP "build" is a copy — Verified.** `extension/com.opencut.panel/client/dist/main.js` is
  md5-identical to `client/main.js` (18,360 lines). Vite is configured but the shipped artifact is
  unbundled and unminified, so the panel loads ~18k lines of source into a Chromium-99 runtime.
- **`opencut/core/` is 623 sibling modules with no sub-packaging — Verified.** 256k LOC, no
  subdirectories, mean 411 lines. Individually healthy (largest is 1,662 lines); collectively this is
  the reason a "one module per adapter" bet produced 50 terminal-stub adapters (8% of `core/`) that
  are one coherent unwired investment in third-party AI models.
- **Dependency posture has two dead ends — Verified.** `demucs` (pinned `>=4.0,<5` in the `audio`,
  `torch-stack`, and `all` extras) was **archived upstream on 2024-04-24**; the maintained successor
  `python-audio-separator` is already wired as a backend (`opencut/routes/audio.py:498-503`,
  `core/engine_registry.py:470`) but is declared in **no** extra, so the live backend is the one users
  cannot install. `auto-editor` is pinned `>=29.3,<30`; upstream was rewritten in Nim and left PyPI
  after 29.3.1 (2025-11-04), shipping native binaries since — the pin is a nine-month-dead branch, and
  everything worth having (partial-lossless GOP copy, linked dissolve transitions, Parakeet TDT word
  timestamps, MLT export) is on the other side of it.
- **Churn confirms the diagnosis — Verified.** The last 200 commits are 96 `fix` to 37 `feat`
  (2.6:1), with `client/main.js` (23 fix touches) and the UXP `main.js` (17) the top non-doc sites.
  `opencut/core/` shows 140 fix touches spread over 118 distinct files — broad and shallow, not a
  hotspot. The existing "Split the remaining panel controller hotspots" item remains correctly aimed.
- **The distribution lanes had never been audited, and they carry the worst defects — Verified.**
  The installer, Docker, CLI, Linux packaging, and example plugins were explicitly out of scope for the
  2026-08-02 audit. Audited here for the first time, they produced three of this pass's four P0s plus
  eleven further defects, all itemised in ROADMAP.md: two CLI commands crash on their documented
  default invocation (`opencut/cli.py:1425` passes a `method=` kwarg `detect_scenes` does not accept;
  `cli.py:540-545` feeds `TimeSegment` objects to a function that calls `.get()` on them);
  `auto-zoom --apply` discards its own tracking data and renders a top-left crop
  (`cli.py:1170-1179`); the Docker `gpu` profile reserves a GPU an image with no CUDA runtime cannot
  use and collides on port 5679, while the `mcp` profile is rejected by the trusted-host guard out of
  the box (`docker-compose.yml:36-64,96-97`); two of three shipped example plugins fail OpenCut's own
  manifest validator for want of an `api_version` key; the WPF installer has no rollback and no
  upgrade detection; and the Linux launchers export an `OPENCUT_HOME` the Python tree ignores while
  neither bundle provides an FFmpeg for the sandbox to resolve. This discharges the installer, Docker,
  CLI, and example-plugin portions of the existing "Unaudited areas needing their own pass" item; the
  ~130 unsampled core modules and localisation quality remain unaudited.
- **Test and docs gaps.** Nine clone-breaking test modules (above); no automated WCAG rule engine —
  the rendered suite hand-rolls a WCAG-AA contrast ratio check at
  `tests/rendered/panel-regression.spec.mjs:887-947` and asserts roles/names ad hoc, so rule classes
  like landmark structure, focus order, and name-from-content go unchecked; sixteen untracked
  `docs/*.md`; `pyproject.toml` declares `integration`/`slow` markers with no `addopts` filter (already
  on the roadmap). Hardware selection is absent too: repo-wide searches for `cuda_device`,
  `device_index`, `gpu_index`, `CUDA_VISIBLE`, and `multi_gpu` return zero hits, so a multi-GPU
  workstation gets whatever device each runtime picks first.
- **Categories deliberately not carried into this roadmap.** *Observability* — Sentry/GlitchTip init,
  request correlation, disk monitoring, structured JSON logs, and opt-in Plausible telemetry already
  exist and no gap surfaced worth an item. *i18n/l10n* — CEP ships `en`, UXP ships `en`/`es`;
  additional locales and RTL are human-QA-gated and correctly parked in `Roadmap_Blocked.md`.
  *Mobile* — outside the Premiere-extension boundary. *Multi-user* — rejected above. *Offline
  resilience*, *workflow resumability*, and *plugin contract versioning* are covered by existing
  roadmap items and are not duplicated here.

## Rejected Ideas

- **In-browser / WASM runtime — Rejected** (auto-editor Online, Rescript, Remotion client-side render,
  Mediabunny): three competitors have it, but it is a second full runtime that a Python/Flask +
  PyTorch backend cannot follow, with zero leverage on the Premiere-panel core.
- **Rebuilding Sequence Index parity — Rejected** (Premiere 26.2 ships it natively with search, sort,
  filter funnel, jump, and CSV export). Supersedes the 2026-07-29 item "Finish Sequence Index as the
  accessible, searchable table its UI promises". Only cross-project/cross-sequence indexing, headless
  CLI/REST query, and sequence-diffing survive as differentiated.
- **Consumer loudness, profanity bleeping, basic pause/filler removal as headline features — Rejected**
  (Auto-Match Loudness with EBU/ATSC/Spotify/Netflix presets; 25.6 bulk bleep/mute; Text-Based Editing
  Delete Pauses/Delete Filler Words). Keep the code; stop leading with it. Standards-grade IMF/IMSC
  loudness validation is the part Adobe does not serve.
- **"Local transcription" as the differentiator — Rejected** (Adobe/Speechmatics on-device STT,
  April 2026, claims 12–16% better than Whisper-powered creative tools; Premiere STT has been
  on-device since 2023). The surviving claim is *unlimited, uncapped, 99 languages, exportable
  artifacts* — which is exactly what every commercial competitor meters.
- **Restore GitHub Actions — Rejected** (repository policy; workflows were deliberately removed).
  `.github/workflows/` is empty and should stay empty; the gap to close is that the quality gates only
  run on a machine with `pre-commit` installed.
- **Code signing / notarization — Rejected** (repository operating rules). Note the consequence:
  macOS Gatekeeper quarantine of unsigned builds is the #2 abandonment cause in the community data
  (StoryToolkitAI #11, open four years, recurring per release). The answer is in-product first-run
  guidance, not a certificate. Adobe explicitly permits unsigned, self-hosted UXP `.ccx` distribution
  with no review, so the panel itself is unaffected.
- **More readiness-only AI adapters — Rejected.** 50 terminal-stub adapters already exist; adding a
  51st damages trust further.
- **Standalone editor, mobile client, mandatory cloud media, broad multi-user collaboration —
  Rejected** (OpenCut-app, CapCut, Resolve, Frame.io): outside the Premiere-extension boundary, and
  Frame.io review is bundled free in Premiere 26.0.
- **A black-box virality score — Rejected** (Opus Clip): copy the idea, not the opacity. Ship named
  component signals the user can re-weight.
- **PyPI / Homebrew / winget publish, macOS notarization, live-Premiere validation, additional
  locales — Rejected from this roadmap:** credential-, hardware-, or human-gated; they belong in
  `Roadmap_Blocked.md` where they already are.

## Sources

### Host platform and commercial
- https://ppro-scripting.docsforadobe.dev/print_page/
- https://medium.com/adobetech/updates-for-creative-cloud-desktop-extensibility-0dd5c663563e
- https://developer.adobe.com/premiere-pro/uxp/changelog/
- https://developer.adobe.com/premiere-pro/uxp/plugins/distribution/overview
- https://blog.developer.adobe.com/en/publish/2026/04/uxp-hybrid-plugins-now-available-for-premiere
- https://community.adobe.com/announcements-727/welcome-to-premiere-26-2-1557825
- https://community.adobe.com/announcements-727/what-s-new-in-adobe-premiere-26-3-june-2026-1628369
- https://helpx.adobe.com/premiere/desktop/add-text-images/insert-captions/create-single-word-captions.html
- https://helpx.adobe.com/premiere/desktop/edit-projects/edit-video-using-text-based-editing/detect-and-delete-pauses-in-transcripts.html
- https://podnews.net/press-release/adobe-speechmatics-on-device
- https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/
- https://firecut.ai/changelog · https://www.autocut.com/en/pricing/ · https://www.premierecopilot.com/en/changelog
- https://descript.canny.io/changelog · https://autopod.fm

### Open source
- https://github.com/WyattBlue/auto-editor/releases · https://basswood.io/blog/auto-editor-is-now-online
- https://github.com/leancoderkavy/premiere-pro-mcp
- https://github.com/mifi/lossless-cut/releases/tag/v3.69.0 · https://github.com/mifi/lossless-cut/issues/126
- https://github.com/SubtitleEdit/subtitleedit/releases/tag/v5.1.0 · https://github.com/SubtitleEdit/subtitleedit/issues/10180
- https://github.com/AcademySoftwareFoundation/OpenTimelineIO/issues/62 · .../issues/446 · .../releases/tag/v0.18.0
- https://github.com/PixarAnimationStudios/OpenTimelineIO/issues/569
- https://kdenlive.org/news/releases/26.04.0/ · https://www.shotcut.com/roadmap/
- https://github.com/pyannote/pyannote-audio/releases/tag/4.0.0
- https://github.com/nomadkaraoke/python-audio-separator · https://github.com/facebookresearch/demucs
- https://github.com/SYSTRAN/faster-whisper/releases · https://huggingface.co/blog/open-asr-leaderboard
- https://github.com/tmoroney/auto-subs/issues/571 · https://github.com/octimot/StoryToolkitAI/issues/11

### Standards, security, dependencies
- https://www.ffmpeg.org/download.html · https://www.ffmpeg.org/security.html · https://github.com/FFmpeg/FFmpeg/tags
- https://nvd.nist.gov/vuln/detail/CVE-2026-64832 · /CVE-2026-64833 · /CVE-2026-64835 · /CVE-2026-66041
- https://security-tracker.debian.org/tracker/source-package/ffmpeg · https://www.gyan.dev/ffmpeg/builds/
- https://modelcontextprotocol.io/specification/2026-07-28/changelog · https://github.com/modelcontextprotocol/python-sdk/releases
- https://www.w3.org/TR/ttml-imsc1.3/ · https://www.w3.org/TR/imsc-hrm/ · https://www.smpte.org/blog/new-and-revised-smpte-standards
- https://gitlab.com/AOMediaCodec/SVT-AV1/-/tags · https://www.phoronix.com/news/FFmpeg-8.0-Released
- https://www.scenedetect.com/docs/0.7/api/migration_guide.html · https://github.com/microsoft/onnxruntime/releases
- https://peps.python.org/pep-0790/ · https://docs.python.org/3/howto/free-threading-python.html

### Community signal
- https://community.adobe.com/bug-reports-733/premiere-pro-incorrectly-detects-silences-in-transcript-and-text-based-editing-1554235
- https://community.adobe.com/t5/premiere-pro-discussions/using-firecut-plugin-to-cut-silence-and-after-i-do-my-premiere-pro-is-unusably-laggy/m-p/14765054
- https://community.adobe.com/feature-requests-730/speaker-detection-editing-tool-1555738
- https://community.adobe.com/t5/premiere-pro-ideas/premiere-pro-quot-multicam-auto-switch-quot-plugin-search/m-p/10307266
- https://community.adobe.com/feature-requests-730/feature-request-auto-sync-existing-caption-subtitle-file-to-video-1326702
- https://community.adobe.com/feature-requests-730/workflow-for-correcting-ai-transcripts-using-text-window-1329035
- https://community.adobe.com/t5/premiere-pro-ideas/auto-transcribed-captions-needs-improvement/idi-p/15217800
- https://community.adobe.com/t5/premiere-pro-ideas/automatically-remove-silence-from-a-video/idc-p/15482855

Note on method: Reddit was **not** mined this cycle. Anthropic's user agent is robots.txt-blocked on
reddit.com and the repository rule forbids crawling or proxying it, so Reddit-targeted searches
returned no usable content. Treat that axis as unmined, not as absent demand.

## Open Questions

- **Does OpenCut's CEP panel currently load in Premiere 26.x?** Adobe's published statement covers
  ExtendScript "through September 2026", but field reports (auto-subs #571; an unanswered Adobe forum
  thread on 25.6.4/Mac) claim CEP extensions are not being enumerated at all in 2026 builds. The
  answer changes whether the UXP work is a migration or an emergency. Requires a live Premiere 26.x
  host, so it belongs with the existing `Roadmap_Blocked.md` "Run documented CEP+UXP smoke pass on
  Premiere 26.x" entry — but it now gates prioritisation, not just verification.
- **Is the 1,253-route no-surface set an intentional integration API or accumulated breadth?** The
  same judgement the existing "Reconcile the queue allowlist" item needs, at ten times the scale.
  Owner decision; the triage cannot start without it.
