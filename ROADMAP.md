# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10 and extended 2026-08-11 from the research passes recorded in `RESEARCH.md`. IDs continue
the existing F-number scheme (highest prior allocation before 2026-08-11: F318).

### P1

- [ ] P1 — F322 — Stop classifying hardcoded-501 handlers as `dependency-gated`
  Why: Five routes return 503 when their dependency is absent and a hardcoded 501 when it is present, so installing the dependency never makes them work — yet the manifest labels them `dependency-gated` (a class its own comment defines as "fully implemented but require an optional dependency") and counts them inside the 1,568 shipped-route total the README advertises.
  Evidence: `opencut/routes/wave_h_routes.py:491` (`/video/upscale/flashvsr`), `:522` (`/video/inpaint/rose`), `:540` (`/video/matte/sammie`), `:558` (`/audio/tts/omnivoice`), `:582` (`/video/style/reezsynth`) — each `_stub_503(...)` then `error_response("NOT_IMPLEMENTED", …, status=501)`; `opencut/tools/dump_route_manifest.py:62-65` where `_DEPENDENCY_MARKERS` matches `_stub_503(` and wins over the inline 501; the same file's comments already state that handlers delegating to a terminal `NotImplementedError` adapter are stubs
  Touches: `opencut/tools/dump_route_manifest.py`, `opencut/_generated/route_manifest.json`, `opencut/_generated/feature_readiness.json`, `opencut/model_cards.py`, `README.md` route counts, `tests/`
  Acceptance: A handler whose success path is an unconditional 501 classifies as `stub` regardless of any earlier dependency marker; the shipped-route count and every advertised total are regenerated from the corrected classification; a test asserts no route is simultaneously `dependency-gated` and unconditionally 501; `feature_readiness.json` regenerates in the same gate as `route_manifest.json` so the two manifests cannot drift on separate clocks.
  Complexity: S

- [ ] P1 — F323 — Bias the ASR decoder with the project glossary instead of correcting after the fact
  Why: The project glossary is applied as a post-hoc find/replace over finished transcripts, so a mis-recognised proper noun that does not match the replacement rule survives, while the transcription backend accepts decoder-level term biasing that would prevent the error — and every commercial competitor sells exactly this as "custom vocabulary".
  Evidence: `opencut/core/captions.py:740` (`_apply_project_glossary`) and `:745` (`apply_glossary_to_result`); `model.transcribe(...)` at `opencut/core/captions.py:930,1227,1241,1353` passes no `initial_prompt` or `hotwords`; `faster-whisper>=1.1,<2` (`pyproject.toml:69`) supports both; https://www.submagic.co/pricing (custom vocabulary, Business tier); https://news.ycombinator.com/item?id=44886647 (proper-noun and homophone complaints)
  Touches: `opencut/core/captions.py`, `opencut/core/transcript_corrections.py`, `opencut/utils/config.py` (`CaptionConfig`), `opencut/core/asr_router.py`, caption routes, `tests/`
  Acceptance: Glossary terms are passed to the decoder as `hotwords`/`initial_prompt` where the active backend supports it, with a documented length cap and safe truncation; the post-hoc correction pass remains as a second layer; backends without biasing degrade silently to today's behaviour and report which layer was used; a fixture proves a glossary term is recognised at the decoder rather than repaired afterwards.
  Complexity: S

- [ ] P1 — F324 — Unify the two panels' i18n key namespaces
  Why: The CEP panel's `en.json` holds 2,868 keys and the UXP panel's 1,927, of which exactly 26 are shared — so the panels maintain two independent translation namespaces for largely the same product, and the only Spanish locale belongs to the panel no installer ships, meaning any future locale must be translated twice.
  Evidence: measured key counts across `extension/com.opencut.panel/client/locales/en.json`, `extension/com.opencut.uxp/locales/en.json`, `extension/com.opencut.uxp/locales/es.json` (26-key intersection); 1,773 `data-i18n` attributes in the CEP markup vs 875 in UXP; `command-center.css` ships as two unrelated files under one name in both panels while `studio-workbench-v2.css`/`.js` are byte-identical copies with no generator
  Touches: `extension/com.opencut.panel/client/locales/`, `extension/com.opencut.uxp/locales/`, `scripts/i18n_lint.py`, `scripts/lint_locales.py`, both `index.html` files, `tests/`
  Acceptance: Shared concepts resolve to one canonical key namespace consumed by both panels, with panel-specific keys explicitly namespaced and justified; the locale lint fails on a key that exists in one panel and has an unnamespaced twin in the other; adding a locale requires translating each string once; the duplicated design-system assets are either generated from a single source or covered by a drift test.
  Note: distinct from the blocked "Localize the Python/CLI backend and add panel locales beyond en/es" item in `Roadmap_Blocked.md` — this ships no new translations and needs no human translator; it is the static refactor that makes that blocked item affordable when it unblocks.
  Complexity: M

- [ ] P1 — F332 — Complete the FFmpeg per-CVE acceptance matrix for the whole 2026-07 batch
  Why: F304 replaced the snapshot date heuristic with per-CVE fix-commit grading, but the matrix grades four CVEs out of roughly sixteen High-severity FFmpeg advisories published 2026-07-22→24 against "FFmpeg through 8.1.2" — so the gate reports a per-CVE verdict while being silent on twelve disclosures from the same batch, including the Vulkan HEVC and NVDEC hardware decode paths this product actually drives.
  Evidence: `opencut/core/ffmpeg_provenance.py` names CVE-2026-64832, -64833, -64835, -66041 from the July batch (plus the earlier CVE-2026-39210…39218 and CVE-2026-6385/-8461 entries); a repo-wide grep finds no mention of CVE-2026-64830 (VobSub demuxer heap overflow), -64831 (Vulkan HEVC hwaccel stack overflow), -64834 (RTP/ASF infinite loop), -65703, -65704, -65705, -65706 (`vf_swaprect` OOB write), -66036 (`vf_hqdn3d` OOB write), -66037, -66038, -66039, or -66040 (native PNG/APNG encoder heap OOB write); NVD `cpe:2.3:a:ffmpeg:ffmpeg`, 2026-07 publication window
  Touches: `opencut/core/ffmpeg_provenance.py`, `scripts/verify_ffmpeg_provenance.py`, `opencut/_generated/project_facts.json`, `docs/RELEASE_PROVENANCE.md`, `SECURITY.md`, `tests/`
  Acceptance: Every FFmpeg advisory the project claims coverage for is enumerated with its upstream fix commit and the ancestry check that proves the pinned build contains it; the gate distinguishes "graded and clear", "graded and vulnerable", and "not yet graded" rather than reporting a clear verdict from partial coverage; adding a new advisory to the matrix without a fix commit fails the gate; the advertised claim in `SECURITY.md` and the README matches the matrix's actual scope.
  Complexity: M

- [ ] P1 — F334 — Raise the urllib3 floor above the two High-severity 2026-05 advisories
  Why: The declared floor is `urllib3>=2.6.3`, annotated for CVE-2026-21441, but two further High-severity advisories published 2026-05-11 are fixed only in 2.7.0 — so the pin the project chose deliberately for security reasons is itself vulnerable, and one of them forwards sensitive headers across origins on proxied redirects, which matters for a service that makes outbound model and update requests.
  Evidence: `pyproject.toml:86` (`urllib3>=2.6.3`) with the CVE-2026-21441 rationale at `:81-85`; GitHub Advisory API (`/advisories?ecosystem=pip&affects=urllib3`): GHSA-qccp-gfcp-xxvc / CVE-2026-44431 "sensitive headers forwarded across origins in proxied low-level redirects", High, `>=1.23, <2.7.0`, patched 2.7.0, published 2026-05-11; GHSA-mf9v-mfxr-j63j / CVE-2026-44432 "decompression-bomb safeguards bypassed in parts of the streaming API", High, `>=2.6.0, <2.7.0`, patched 2.7.0, same date
  Touches: `pyproject.toml`, `requirements.txt`, `requirements-*-lock.txt`, `scripts/check_installed_versions.py`, `docs/PYTHON_ADVISORIES.md`, `tests/test_declared_floors.py`
  Acceptance: The floor is `urllib3>=2.7.0` with both CVE ids in the inline rationale following the file's existing convention; the resolved lane still installs under the declared extras and the installed-version gate passes; the advisory policy doc records both ids so the next audit does not re-derive them.
  Complexity: S

### P2

- [ ] P2 — F314 — Make caption burn-in incremental
  Why: The most-repeated Premiere captioning complaint is that a small caption change forces a full timeline re-render; OpenCut burns captions with a whole-file FFmpeg re-encode, so it inherits the same cost and has an unclaimed differentiator available.
  Evidence: https://community.adobe.com/feature-requests-730/overhaul-captioning-workflow-1555697; `opencut/core/caption_burnin.py`, `opencut/core/styled_captions.py`; existing segment machinery in `opencut/core/smart_render.py`
  Touches: `opencut/core/caption_burnin.py`, `opencut/core/smart_render.py`, `opencut/routes/captions.py`, `tests/test_smart_render_transactional.py`
  Acceptance: Re-burning after a caption edit re-encodes only the affected segments and stream-copies the remainder, with the unchanged regions bit-identical to the prior render; a changed-caption job on a multi-segment fixture measurably beats the full re-encode; falling back to a whole-file render is automatic and reported when segment boundaries cannot be honoured.
  Complexity: L

- [ ] P2 — F325 — Scope silence detection to in/out points and add a tighten-don't-delete mode
  Why: `detect_silences()` takes no time range, so silence work always spans the whole file with no way to honour an in/out selection, and the only outcomes are hard cut or speed-up — while the most-voted open Premiere idea in this area asks for exactly range scoping plus "shorten pauses to N seconds", and Adobe's staff answer ("use Text-Based Editing") was rejected by the requester.
  Evidence: `opencut/core/silence.py:37` (`detect_silences(filepath, threshold_db, min_duration, file_duration)`), `:139` (`detect_silences_vad`), `:580` (`speed_up_silences`); `:428` (`filter_smart_pauses`) already covers keeping dramatic pauses but not scoping or tightening; https://community.adobe.com/t5/premiere-pro-ideas/feature-request-apply-silence-detection-and-removal-only-between-in-out-points/idi-p/15448602 (2025-08-07, Open for Voting); https://community.adobe.com/t5/premiere-pro-ideas/automatically-remove-silence-from-a-video/idi-p/13577633
  Touches: `opencut/core/silence.py`, `opencut/routes/audio.py`, `opencut/cli.py`, both panels' cut surfaces, `extension/com.opencut.panel/host/index.jsx` (in/out read), `tests/`
  Acceptance: Detection accepts an optional `[start, end]` range and returns segments only within it, with timestamps still absolute to the source; a `tighten` mode shortens each detected silence to a target duration instead of removing it; the panels pass the sequence's current in/out points when a selection exists; existing whole-file behaviour is unchanged when no range is supplied.
  Complexity: M

- [ ] P2 — F326 — Guard long-file ASR repetition loops
  Why: Whisper-family decoders degrade on hour-plus audio into looping a single phrase for the remainder of the file, and OpenCut passes no decoder threshold and runs no post-hoc detection — so the failure produces a plausible-looking, silently wrong transcript that the next stage happily deletes footage from.
  Evidence: `opencut/core/captions.py` contains no `compression_ratio_threshold`, `no_speech_threshold`, `condition_on_previous_text`, or repetition check (grep returns nothing); https://lobste.rs/s/ddssxd/captioning_all_my_youtube_videos_with_ai (2-hour talk repeating one phrase after ~45 minutes); https://news.ycombinator.com/item?id=44886647 (chunk-boundary and long-form degradation); https://community.adobe.com/bug-reports-733/transcription-has-stopped-working-properly-after-latest-update-1627635 (same failure shape in Premiere 26.x)
  Touches: `opencut/core/captions.py`, `opencut/core/asr_router.py`, `opencut/core/asr_provenance.py`, `opencut/utils/config.py`, `tests/`
  Acceptance: Backends that expose decoder thresholds receive explicit values rather than defaults; a backend-independent post-pass flags runs of near-identical consecutive segments and marks the affected span as low-confidence in the result and in ASR provenance; a synthetic long-form fixture with an induced loop is detected and reported; detection never silently discards transcript content.
  Complexity: M

- [ ] P2 — F327 — Record upstream maintenance status per engine and stop auto-installing abandoned packages
  Why: A runtime code path pip-installs DeepFilterNet, whose newest PyPI release is 2023-08-31 and whose repository has not been pushed since 2024-10-17, and neither the engine registry nor the model cards carry any field that could tell a user they are installing dead software — while the project already reasons this way informally in route comments.
  Evidence: `opencut/core/audio_pro.py:529` (`ensure_package("df", "deepfilternet")`); https://pypi.org/pypi/deepfilternet/json (0.5.6, 2023-08-31); https://github.com/Rikorose/DeepFilterNet (last push 2024-10-17); `opencut/routes/audio.py:457` already comments that the separation default was changed to "the maintained backend"; `opencut/core/engine_registry.py` has no maintenance/upstream field and `opencut/model_cards.py` has no DeepFilterNet card; https://pypi.org/pypi/faster-whisper/json (1.2.1, 2025-10-31); https://github.com/modelscope/ClearerVoice-Studio (last push 2025-08-14)
  Touches: `opencut/core/engine_registry.py`, `opencut/model_cards.py`, `opencut/_generated/model_cards.json`, `opencut/core/audio_pro.py`, `opencut/checks.py`, `docs/MODELS.md`, `tests/`
  Acceptance: Every third-party engine entry records last upstream release, last repository activity, and a dated maintenance verdict with its source URL; engines marked unmaintained are never a default, warn before an on-demand install, and say so in `/system/dependencies` and the panel; a test fails when an engine is a default while marked unmaintained; the dates are re-verifiable from the recorded URLs rather than asserted.
  Complexity: M

- [ ] P2 — F328 — Ratchet the direct-surface ratio so new routes cannot ship unreachable
  Why: The repo's own manifest reports 280 of 1,568 shipped routes reachable from any first-party surface (17.9%), 1,288 integration-only, and zero routes whose primary surface is the CLI — so every wave adds API faster than it adds product, and the ratio is measured but nothing stops it falling.
  Evidence: `opencut/_generated/route_manifest.json` → `surface_coverage.summary` (`direct_surface_routes: 280`, `integration_only_routes: 1288`, `coverage_percent: 17.9`, `primary_counts.cli: 0`); 19 CLI commands in `opencut/cli.py`; 88 MCP tools in `opencut/_generated/mcp_server_registry.json`; the gate at `surface_coverage.gate` only asserts every route is classified, never that the ratio holds
  Touches: `opencut/tools/dump_route_manifest.py`, `scripts/release_smoke.py`, `opencut/cli.py`, `opencut/core/mcp_tools.py`, `opencut/core/command_palette.py`, `tests/`
  Acceptance: The release gate fails when `coverage_percent` falls below the value recorded at the time the ratchet lands; a new route must either declare a surface or carry an explicit `integration-only` justification that the gate records; the report names the largest integration-only route families so a triage or deprecation decision has data behind it.
  Complexity: M

- [ ] P2 — F333 — Decide the FFmpeg 9.0 release lane so users are not forced onto git-master snapshots
  Why: `RELEASE_FLOOR` is `(8, 1, 3)` with the comment "the release lane remains closed until upstream publishes 8.1.3" — a version that never shipped, because upstream moved to 9.0 instead — so today every source installer must fetch a dated git-master snapshot, which is a harder and less auditable ask than a tagged release and leaves the documented release lane permanently dead.
  Evidence: `opencut/core/ffmpeg_provenance.py:56-57` (`RELEASE_FLOOR = (8, 1, 3)`) and `:62` (`SNAPSHOT_FLOOR_DATE = "2026-07-06"`); https://ffmpeg.org/download.html — FFmpeg 9.0 "Lei" released 2026-08-04, branch cut from master 2026-06-26; 8.1.2 (2026-06-17) is the last 8.1 release and 8.0.3 (2026-06-18) closed the 8.0 branch; `README.md:110-118` and `Install.ps1` currently instruct users toward the gyan.dev git-master snapshot
  Touches: `opencut/core/ffmpeg_provenance.py`, `scripts/verify_ffmpeg_provenance.py`, `README.md`, `Install.ps1`, `install.py`, `Dockerfile`, `docs/RELEASE_PROVENANCE.md`, `tests/`
  Acceptance: The release lane either opens on a specific 9.x version proven by F332's per-CVE ancestry checks to carry every graded fix, or is documented as deliberately closed with the reason recorded in the module rather than a comment about an unshipped 8.1.3; the 9.0 branch point (2026-06-26, before the July fix commits landed on master) is explicitly accounted for rather than assumed, since a later release date does not by itself imply the fixes were backported; installation docs name whichever lane is supported.
  Depends on: F332
  Complexity: M

### P3

- [ ] P3 — F316 — Add Mel-Band RoFormer to the separator engine registry
  Why: The engine registry offers Demucs, BS-RoFormer, and MDX-Net; Mel-Band RoFormer reports higher separation quality than BS-RoFormer on vocals and drums and is already reachable through the pinned `audio-separator` dependency, so this is a registry entry rather than a new dependency.
  Evidence: `opencut/core/engine_registry.py:466-469` (BS-RoFormer entry, no Mel-Band variant); `audio-separator>=0.44,<1` in `pyproject.toml`; https://arxiv.org/abs/2310.01809
  Touches: `opencut/core/engine_registry.py`, `opencut/routes/audio.py`, `opencut/checks.py`, `tests/`
  Acceptance: The model is selectable through the existing `backend`/engine parameter, availability is probed rather than assumed, current defaults are unchanged, and the registry entry records the model identifier and licence alongside the existing entries.
  Complexity: S

- [ ] P3 — F317 — Adopt PEP 751 `pylock.toml` and PEP 639 SPDX licence metadata
  Why: Four hand-maintained `requirements-*-lock.txt` files (one of them 126 KB) now have a standard replacement that pip installs directly, and the current `license` plus classifier form is the deprecated pre-PEP-639 spelling; consolidating removes a recurring version-sync surface that already carries dozens of `fix:` commits.
  Evidence: https://peps.python.org/pep-0751/ (pip 26.1 installs from `pylock.toml`); `requirements-lock.txt`, `requirements-build-lock.txt`, `requirements-release-lock.txt`; `pyproject.toml` licence block
  Touches: `pyproject.toml`, `requirements*.txt`, `scripts/sync_version.py`, `scripts/check_dependency_matrix.py`, `Dockerfile`, `docs/`
  Acceptance: A generated `pylock.toml` reproduces the release environment and is verified in release smoke; the bespoke lockfiles are either removed or generated from it; `project.license` uses an SPDX expression with the deprecated classifier removed; the version-sync target count is updated to match.
  Complexity: M

- [ ] P3 — F318 — Document the unsigned-install experience and publish artifact digests
  Why: Standing policy forbids code signing, and Microsoft's current guidance is that unsigned files rebuild SmartScreen reputation from zero on every update and that signing no longer guarantees a bypass; users therefore need an explicit, permanent instruction plus a way to verify what they downloaded.
  Evidence: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation; `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`; `scripts/sbom.py`
  Touches: `README.md`, `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`, `scripts/release_gate.py`
  Acceptance: Installation documentation states plainly that artifacts are unsigned, shows the exact SmartScreen path to proceed, and gives the verification command; the release gate emits SHA-256 digests for every artifact into the release metadata so the published digests can be checked against the download.
  Complexity: S

- [ ] P3 — F329 — Document the Premiere 2026 "Extensions (Legacy)" location and probe it at install time
  Why: Premiere 2026 moved CEP panels under a separate **Extensions (Legacy)** menu, and neither the README nor `docs/` mentions it anywhere — this is the most predictable support question for a CEP-primary product and it has already stranded two comparable projects publicly with "extension not loading" reports that were really menu-location confusion.
  Evidence: grep for "Legacy" across `README.md` and `docs/*.md` returns nothing; `README.md:65` claims CEP support for "Premiere Pro 2019 or later" with no version-specific note; https://github.com/tmoroney/auto-subs/issues/571 (2026-06-06, extension absent from Window > Extensions on PPro 2026); https://github.com/leancoderkavy/premiere-pro-mcp/issues/14
  Touches: `README.md`, `docs/` (installation guidance), `Install.ps1`, `install.py`, `OpenCut.iss`, `installer/src/OpenCut.Installer/Services/InstallEngine.cs`
  Acceptance: Installation documentation states where the panel appears on Premiere 2026+ versus earlier versions; the installer reports the detected Premiere major version and the expected menu path in its completion output; the same sentence appears in the panel's own connection-failure guidance.
  Complexity: S

- [ ] P3 — F330 — Make the release-gate lint and test configuration the one the editor applies
  Why: `[tool.ruff]` declares only `line-length` and `target-version` with no `lint` section, so an editor-integrated ruff applies its default rule set while the gate applies a different, narrower one defined in two other files — and `[tool.pytest.ini_options]` sets no `testpaths` and no `--strict-markers` across 10,751 tests in 346 modules, so a mistyped marker silently selects nothing.
  Evidence: `pyproject.toml:267-269`; `.pre-commit-config.yaml` (ruff `--select E,F,I --ignore E501,E402`); `scripts/release_smoke.py` repeats the same selection independently; `[tool.pytest.ini_options]` sets only `addopts` and two markers
  Touches: `pyproject.toml`, `.pre-commit-config.yaml`, `scripts/release_smoke.py`, `DEVELOPMENT.md`, `CONTRIBUTING.md`
  Acceptance: The ruff rule selection lives in `pyproject.toml` and both the pre-commit hook and the release smoke read it rather than restating it; pytest declares `testpaths` and `--strict-markers`; running the editor's ruff and the gate's ruff on the same tree produces the same findings.
  Complexity: S

- [ ] P3 — F331 — Pin OpenTimelineIO against the 0.19 bundle rewrite and contract-test the export path
  Why: The declared ceiling `opentimelineio>=0.17,<1` will silently admit 0.19, which moves `otioz`/`otiod` bundle handling out of Python into the C++ core — a behaviour change to a shipped export path — and the project has already been bitten once by an OTIO minor bump renaming `MediaReferencePolicy` enum members between 0.15 and 0.17.
  Evidence: `pyproject.toml:170,230`; `opencut/export/otio_export.py:547,566-567` (the enum-naming workaround); `opencut/export/otio_compat.py` reports the runtime version but pins and asserts nothing; https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases (0.18.1, 2025-11-09, still flagged prerelease, no release in 9 months)
  Touches: `pyproject.toml`, `requirements*.txt`, `opencut/export/otio_compat.py`, `opencut/export/otio_export.py`, `scripts/check_dependency_matrix.py`, `tests/`
  Acceptance: The OTIO specifier bounds the tested minor line rather than an open `<1`; a contract test round-trips an OTIOZ bundle and asserts the media-reference policy and bundle layout, failing on an untested OTIO minor; `otio_compat` records the verified-against version alongside the runtime version.
  Complexity: S
