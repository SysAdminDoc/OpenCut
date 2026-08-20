# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

### P1

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

- [ ] P1 — F337 — Map the high-risk CEP-only host functions onto the Premiere 26.3 UXP API surface
  Why: The UXP migration dashboard's route-level gate fails on the high-risk CEP-only functions (`ocAddNativeCaptionTrack`, `ocQeReflect`; `ocApplySequenceCuts` partial), and Adobe's 26.2/26.3 UXP releases added Hybrid Plugins, encoder and batch-encode control, `ProjectConverter.exportAAF`, `ObjectMaskUtils`, marker `guid`s, and Transcript APIs that plausibly cover part of that gap — the typings are already pinned at 26.3, so this audit is headless work that shrinks the project's largest tracked liability (119 uxp-pending routes against a ~2026-11 CEP horizon, with conflicting 2026 field reports of earlier breakage).
  Evidence: `opencut/_generated/uxp_migration_dashboard.json` (2 high-risk `cep_only` functions, failing route gate); https://developer.adobe.com/premiere-pro/uxp/changelog (26.2 Hybrid Plugins; 26.3 launchEncoder/startBatchEncode, exportAAF, Transcript.querySupportedLanguages/hasTranscript, ObjectMaskUtils, createSubClipAction); pinned `@adobe/premierepro` 26.3 typings (CLAUDE.md Learned 2026-08-12); https://hyperbrew.co/blog/uxp-plugins-in-premiere-2026/ and https://github.com/tmoroney/auto-subs/issues/571 (breakage reports raising urgency)
  Touches: `opencut/tools/adobe_uxp_compatibility.py` (`API_CATALOGUE`), `opencut/_generated/adobe_uxp_compatibility.json`, the `uxp_migration_dashboard` generator, `extension/com.opencut.uxp/main.js` where a typed equivalent exists, `extension/PANEL_PARITY.json`, `tests/`
  Acceptance: Every 26.2/26.3 UXP API relevant to the three functions is catalogued with typed evidence per the established adobe_uxp_compatibility workflow; each high-risk function carries a dated verdict (portable now / needs live validation / no UXP equivalent); functions judged portable get a UXP implementation or an explicit blocked entry with the missing piece named; the migration dashboard reclassifies from the new catalogue.
  Complexity: M

- [ ] P1 — F338 — Prove the CSRF bootstrap fix closes the issue #5 first-contact failure
  Why: The only open external bug (issue #5, 2026-08-10) is a real user blocked at the panel's first mutation by "Invalid or missing CSRF Token"; an Unreleased CHANGELOG entry records CSRF-bootstrap work, but nothing ties it to the reported scenario, no regression test names the failure shape, and the README troubleshooting section never mentions the error string — for a 39-star project, one first-contact failure costs more than any missing feature.
  Evidence: https://github.com/SysAdminDoc/OpenCut/issues/5; CHANGELOG.md `## Unreleased` CSRF bootstrap entry; grep for "CSRF" in README.md troubleshooting returns nothing (2026-08-20)
  Touches: `opencut/security.py`, `opencut/routes/system.py` (`/health` bootstrap), `tests/test_integration.py`, `README.md` (troubleshooting), both panels' `ERROR_CODE_ACTIONS` hint tables
  Acceptance: A regression test reproduces the reported failure shape (panel connects, then mutates with a missing/stale token or opaque-origin bootstrap) and passes against the fix; README troubleshooting names the exact error string with recovery steps; the panel hint for the CSRF error code is actionable; the issue can be answered with a pinned commit and a workaround for the published v1.25.1 build.
  Complexity: S

- [ ] P1 — F335 — Apply micro audio fades at silence and filler cut boundaries
  Why: Hard razor joins at detected boundaries produce audible clicks and pops on exported media, the silence/filler export path applies no boundary fade anywhere, and the closest CLI competitor queued exactly this fix in June — it is the cheapest audible-quality win available on the headline feature.
  Evidence: grep for afade/crossfade/fade in `opencut/core/silence.py` returns nothing (2026-08-20); https://github.com/WyattBlue/auto-editor/issues/1272 (audio fade handling across split edges, 2026-06-25)
  Touches: `opencut/core/silence.py` (export path), `opencut/routes/audio.py`, `opencut/helpers.py` (`FFmpegCmd`), `opencut/cli.py`, `tests/`
  Acceptance: Exported cut media applies a configurable micro-fade (default a few milliseconds) at each join, off-switchable per request; a fixture with a tone crossing a cut boundary shows bounded sample discontinuity with the fade on and a click with it off; timeline write-back docs state whether host-side audio transitions are applied or deferred to a later item.
  Complexity: S

- [ ] P1 — F336 — Route long-file transcription through faster-whisper batched inference
  Why: Upstream's `BatchedInferencePipeline` delivers roughly 4x on long files and `captions.py` only ever calls sequential `model.transcribe(...)` — long-file speed is simultaneously the top user complaint against paid competitors (AutoCut's 15-minute multi-track runs) and the dominant wall-clock cost of every transcript-driven OpenCut workflow.
  Evidence: grep for `BatchedInferencePipeline` in `opencut/core/captions.py` returns nothing (2026-08-20); https://github.com/SYSTRAN/faster-whisper/releases (batched inference ~4x, VAD speedups); https://www.capterra.com/p/10036511/AutoCut/ (slowness complaints)
  Touches: `opencut/core/captions.py`, `opencut/core/asr_router.py`, ASR provenance recording, `opencut/utils/config.py`, `tests/`
  Acceptance: Files above a documented duration threshold route through the batched pipeline when the active backend supports it, preserving word timestamps and segment shape; provenance records batched vs sequential; a long fixture shows a measurable speedup; an opt-out parameter restores sequential decoding; unsupported backends are unaffected.
  Complexity: M

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
  Note (2026-08-20): upstream released 9.0.1 on 2026-08-12, so the 9.0 lane decision now has a patch release to grade against; the branch-point caveat (9.0 cut 2026-06-26, before the July fix commits) applies to 9.0.x until ancestry-checked.
  Complexity: M

- [ ] P2 — F339 — Add a "disable instead of delete" mode to cut-review write-back
  Why: Editors distrust destructive auto-cuts, and the reviewer-tested FireCut comparison specifically calls out the inability to disable or mark clips instead of removing them — while Premiere track items expose a settable disabled state and OpenCut's host layers currently have no disable path at all, so the review panel's only outcomes are delete or skip.
  Evidence: https://cutback.video/blog/the-best-auto-silence-removal-plugin-for-premiere-pro (FireCut "can't disable instead of delete"); https://www.freevisuals.net/post/firecut-ai-review (2026-07-20, no non-destructive option); grep for `.disabled`/disable in `extension/com.opencut.panel/host/index.jsx` returns nothing (2026-08-20)
  Touches: `extension/com.opencut.panel/host/index.jsx` (new host function or `ocApplySequenceCuts` mode), `extension/com.opencut.uxp/main.js`, both panels' cut-review UI, `client/host-write-verification.js` + `uxp-host-write-verification.js` contracts, `tests/jsx_mock.js`
  Acceptance: The cut review panel offers apply-as-disable alongside apply-as-delete; disabled ranges pass the same read-back verification contract as deletions; re-running with delete after a disable pass works; the docs state the mode is non-destructive and reversible in Premiere.
  Complexity: M

- [ ] P2 — F340 — Guard timeline performance when applying large cut batches
  Why: Applying thousands of razor cuts makes a Premiere sequence unusably laggy — reported verbatim on Adobe's forum against silence-cutting plugins — and OpenCut has no cut-count warning, no merge-small-gaps option at write-back time, and no consolidation alternative, so a 2-hour recording with aggressive thresholds produces exactly that failure.
  Evidence: https://community.adobe.com/questions-729/using-a-plugin-to-cut-out-silence-and-the-sequence-becomes-unusably-laggy-1411061; grep for consolidation/nesting in `extension/com.opencut.panel/host/index.jsx` returns nothing (2026-08-20); `opencut/core/smart_render.py` already renders consolidated media that could serve as the escape hatch
  Touches: both panels' cut-review surfaces, `opencut/core/silence.py` (merge segments separated by sub-threshold gaps), `extension/com.opencut.panel/host/index.jsx`, `docs/`
  Acceptance: Before write-back, the review panel reports the edit count and warns above a documented threshold; the user can merge cuts separated by less than N frames (provably reducing edit count on a fixture), switch to the F325 tighten mode, or render consolidated media via smart_render instead of cutting the sequence; the warning threshold and rationale are documented.
  Complexity: M

- [ ] P2 — F341 — Add multicam cutting grammar controls and benchmark the mixed-track workflow
  Why: `generate_multicam_cuts` exposes a single knob (min_cut_duration), while the category leader AutoPod is distrusted for its wide-shot handling, cannot work from mixed/shared audio tracks, and leaves silence padding — and OpenCut's diarization-driven path already works from one mixed track but has no cutting grammar (wide-shot cadence, cut-on-interruption, per-speaker weighting) and nothing documents or proves the mixed-track advantage.
  Evidence: `opencut/core/multicam.py:88` (only `min_cut_duration`); https://diyai.io/ai-tools/video-generation/reviews/autopod-review/ and https://vidpros.com/autopod-review-ai-editing-for-podcasts-worth-it/ (mixed-track failure, silence padding); `docs/RESEARCH_COMPETITIVE_TEARDOWN_2026-06-10.md` multicam-v2 gap (tunable wide-shot frequency), never regraded until 2026-08-20
  Touches: `opencut/core/multicam.py`, `opencut/routes/video_core.py` (`/video/multicam-cuts`, `/video/multicam-xml`), multicam settings in `~/.opencut/multicam_config.json` wrappers, both panels' multicam surfaces, `tests/test_new_modules.py`
  Acceptance: Cut generation accepts wide-shot cadence (a wide angle every N cuts or T seconds), a cut-on-interruption toggle, and per-speaker track weighting, with current defaults unchanged; a fixture proves cuts generate from a single mixed-audio track and the README/docs state it; grammar settings flow through both the cut list and the multicam XML export.
  Complexity: M

- [ ] P2 — F342 — Add de-reverb and denoise separation checkpoints through the pinned audio-separator
  Why: The pinned `audio-separator` dependency exposes UVR-family de-reverb and denoise checkpoints beyond music stems, competitors sell the equivalent as metered "studio sound", and OpenCut's separator registry lists music-stem models only — so this is a registry entry on an existing dependency, the same shape as F316.
  Evidence: https://github.com/nomadkaraoke/python-audio-separator (de-reverb/denoise/karaoke checkpoint registry, pushed 2026-07-20); `opencut/core/engine_registry.py` separator entries (music stems only); F316 precedent
  Touches: `opencut/core/engine_registry.py`, `opencut/routes/audio.py`, `opencut/checks.py`, `opencut/model_cards.py`, `tests/`
  Acceptance: De-reverb and denoise checkpoints are selectable through the existing backend/engine parameter with probed availability; model identifiers and licences are recorded on cards per the existing convention; current defaults are unchanged.
  Complexity: S

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

- [ ] P3 — F343 — Refresh the naming and positioning section against the OpenCut-app relaunch
  Why: The README naming section still cites ~48K stars and says "when it relaunches", but the unrelated web OpenCut reached 85K stars, merged its ground-up rewrite around 2026-07-14 with a plugin store, headless rendering, and MCP direction, and carries an active name-infringement thread — the discoverability risk the section exists to manage grew while its facts went stale.
  Evidence: `README.md` naming section ("~48K stars", "when it relaunches"); GitHub API star count 85,234 (2026-08-20); https://explainx.ai/blog/opencut-rewrite-plugins-headless-mcp-2026; https://github.com/OpenCut-app/OpenCut (issue #192, name infringement)
  Touches: `README.md` (naming section and lede), repo description, `pyproject.toml` description
  Acceptance: The section states the current facts with dates, keeps the `opencut-ppro` distribution decision, and the disambiguating phrase ("OpenCut for Premiere Pro") appears in the repo description and README lede; the prose follows the project's public-writing voice rules.
  Complexity: S

- [ ] P3 — F344 — Rank repeat clusters with a best-take recommendation
  Why: `repeat_detect` finds repeated sentences but ranks nothing, so review shows "these repeat" without "keep this one" — while AutoCut Repeat and Gling sell exactly the keep-best-take selection, and a heuristic (filler count, WPM stability, completion) with an optional LLM verdict layers cleanly on the existing detection output.
  Evidence: `opencut/core/repeat_detect.py` (detection and range merging only); https://github.com/rafcopy/auto-cut-agent (LLM-based take dedup in a UXP+local-server design, 2026-08-13); https://opentools.ai/tools/gling-ai (bad-take marking)
  Touches: `opencut/core/repeat_detect.py`, `opencut/core/llm.py` consumers, `opencut/routes/captions.py` (`/captions/repeat-detect`), both panels' cut-review surfaces, `tests/test_new_modules.py`
  Acceptance: Each repeat cluster carries a ranked keep-candidate with per-take signals (filler count, speech-rate stability, sentence completion, optional LLM verdict with recorded fallback); the review UI preselects the keep and cuts the rest; the heuristic path works with no LLM configured; existing detect-only output remains available.
  Complexity: M

- [ ] P3 — F345 — Generate a packaged agent skill for the MCP server
  Why: Remotion ships Agent Skills so coding agents drive it correctly on first contact, and OpenCut's 88-tool MCP server has no packaged skill or conventions document, so every agent session rediscovers the review-first and durable-jobs patterns from raw tool schemas.
  Evidence: https://www.remotion.dev/blog (Agent Skills, 2026-01); `opencut/_generated/mcp_server_registry.json` (88 tools, no companion skill artifact)
  Touches: a new generated skill document (tool map, review-first conventions, durable-jobs pattern, safety rules), its dump tool under `opencut/tools/`, `docs/MCP_SERVER.md`, drift test
  Acceptance: A versioned skill document generates from the MCP registry, ships in-repo, and regenerates with the manifest under a drift test like other `_generated` artifacts; it teaches the review-before-mutate and task-polling conventions; a coldstart agent following only the skill can run a transcribe-review-export flow.
  Complexity: S

- [ ] P3 — F346 — Activate the /analyze/video/qwen3vl lane through local Ollama vision models
  Why: Content-aware editing ("cut the boring parts" from semantic video understanding, not just audio) is where the closest CLI competitor and the agentic wave are heading, and OpenCut already has the route stubbed (`/analyze/video/qwen3vl`, 501) plus an LLM layer that fronts Ollama — which serves Qwen-VL-class models locally — so one stub activation delivers per-segment semantic relevance scoring with no cloud key.
  Evidence: `opencut/_generated/route_manifest.json` (qwen3vl/internvl3 stubs); https://github.com/WyattBlue/auto-editor/issues/1273 (content-aware edit method, 2026-06-25); `opencut/core/llm.py` (Ollama support); the text-first economy pattern from browser-use/video-use recorded in the 2026-08-11 pass
  Touches: `opencut/core/multimodal_qwen3vl.py` (remove terminal NotImplementedError per readiness rules), `opencut/core/llm.py`, the wave_qrs route, highlights integration, `opencut/model_cards.py`, `tests/`
  Acceptance: The route leaves stub state through the established readiness flow (stub_scan reclassifies it once the terminal raise is gone and `check_X_available()` gates it); frame-sampled scoring returns per-segment relevance keeping transcript as the primary signal and pixels at decision points; it runs against a local Ollama vision model with no API key; the manifest and README counts regenerate.
  Complexity: L
