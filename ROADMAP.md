# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

### P1

- [ ] P1 — F324 — Unify the two panels' i18n key namespaces
  Why: The CEP panel's `en.json` holds 2,880 keys and the UXP panel's 1,927, of which exactly 26 are shared — so the panels maintain two independent translation namespaces for largely the same product, and the only Spanish locale belongs to the panel no installer ships, meaning any future locale must be translated twice.
  Evidence: measured key counts across `extension/com.opencut.panel/client/locales/en.json`, `extension/com.opencut.uxp/locales/en.json`, `extension/com.opencut.uxp/locales/es.json` (26-key intersection); 1,773 `data-i18n` attributes in the CEP markup vs 875 in UXP
  Touches: `extension/com.opencut.panel/client/locales/`, `extension/com.opencut.uxp/locales/`, `scripts/i18n_lint.py`, `scripts/lint_locales.py`, both `index.html` files, `tests/`
  Acceptance: Shared concepts resolve to one canonical key namespace consumed by both panels, with panel-specific keys explicitly namespaced and justified; the locale lint fails on a key that exists in one panel and has an unnamespaced twin in the other; adding a locale requires translating each string once.
  Note: distinct from the blocked "Localize the Python/CLI backend and add panel locales beyond en/es" item in `Roadmap_Blocked.md` — this ships no new translations and needs no human translator; it is the static refactor that makes that blocked item affordable when it unblocks. The duplicated design-system assets that were part of this item are now covered by `tests/test_shared_panel_assets.py`, and the lint's file-scan boundary is fixed.
  Complexity: M

### P2

- [ ] P2 — F348 — Grade the twelve remaining July 2026 FFmpeg advisories
  Why: F332 made the gate honest about its scope — it now reports "5 graded of 17 known" and names the twelve it did not check — but those twelve are still ungraded, so the pinned snapshot's status against the VobSub, Vulkan HEVC, RTP/ASF, `vf_swaprect`, `vf_hqdn3d`, and PNG/APNG encoder paths is unknown rather than clear.
  Evidence: `UNGRADED_ADVISORIES` in `opencut/core/ffmpeg_provenance.py` (CVE-2026-64830, -64831, -64834, -65703, -65704, -65705, -65706, -66036, -66037, -66038, -66039, -66040); `tests/test_ffmpeg_cve_matrix.py` enforces that a graded entry carries a real 40-character fix commit, which is why these cannot be moved without one
  Touches: `opencut/core/ffmpeg_provenance.py` (move entries from `UNGRADED_ADVISORIES` into `SECURITY_ADVISORIES`), `README.md`, `tests/test_ffmpeg_cve_matrix.py`
  Acceptance: Each advisory has its upstream fix commit, the date it landed on master, its affected component, and capability tokens that allow "component not compiled in" to be established; the pinned snapshot is graded against every one of them; `advisory_coverage()["complete"]` becomes true when the list empties and the README paragraph about ungraded advisories is removed in the same commit.
  Note: needs upstream commit archaeology per CVE (git log over the fix window in each affected file). Do not guess a hash — the test rejects a malformed one, and a plausible-but-wrong hash would make the gate lie in the dangerous direction.
  Complexity: M

- [ ] P2 — F349 — Port the ripple-delete cut path onto SequenceEditor.createRemoveItemsAction
  Why: The F337 audit confirmed a typed 26.3 successor for the cut path — `SequenceEditor.getEditor(sequence).createRemoveItemsAction(selection, ripple, mediaType)` returning an undoable `Action` — which is both the UXP migration path for `ocApplySequenceCuts` and a likely fix for the failure mode premiere-pro-mcp #21 measured, where `sequence.rippleDelete()` returns success and changes nothing on 26.3.
  Evidence: `@adobe/premierepro` 26.3.0 typings `src/premierepro.d.ts:3203-3232` (`SequenceEditorStatic.getEditor`, `SequenceEditor.createRemoveItemsAction`); catalogued as `fixture_only` in `opencut/tools/adobe_uxp_compatibility.py`; `opencut/core/cep_uxp_parity.py` (`ocApplySequenceCuts`, `partial_uxp`); https://github.com/leancoderkavy/premiere-pro-mcp/issues/21
  Touches: `extension/com.opencut.uxp/main.js` (cut application), `uxp-host-write-verification.js`, `opencut/tools/adobe_uxp_compatibility.py` (drop `fixture_only` once consumed), `opencut/core/cep_uxp_parity.py`, `tests/`
  Acceptance: The UXP cut path builds a `TrackItemSelection`, creates the remove-items Action, and runs it inside `Project.executeTransaction` so the edit is undoable; the existing host-write read-back contract verifies the sequence actually changed; the capability drops `fixture_only` and the migration dashboard reclassifies `ocApplySequenceCuts` from `partial_uxp`; the CEP path stays as fallback until the live-host lane confirms the UXP one.
  Note: writing the code is headless, but confirming it mutates a real 26.x sequence needs the blocked live-Premiere lane — land it behind the existing verification contract so a user's bug report answers it.
  Complexity: M

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

### P3

- [ ] P3 — F350 — Rename the co-named per-panel assets so one filename means one file
  Why: `backend-client.js`, `command-center.css`, `command-center-layout.css`, and `command-center-tokens.css` each exist under the same name in both panels while being genuinely different implementations (for example `command-center.css` is 2,416 lines in CEP and 2,627 in UXP) — one name for two files at the same cascade position invites edits landing in the wrong panel, which is the failure the shared-asset drift gate cannot catch because these were never copies.
  Evidence: `tests/test_shared_panel_assets.py` records all four in `panel_specific` with byte comparisons; RESEARCH.md 2026-08-11 named `command-center.css` shipping "as two unrelated files under one name and cascade position"
  Touches: `extension/com.opencut.panel/client/`, `extension/com.opencut.uxp/`, both `index.html` link/script tags, `scripts/i18n_lint.py` and any build/verification file lists, `tests/test_shared_panel_assets.py`
  Acceptance: Each per-panel asset carries a name that identifies its panel (or lives under a panel-scoped directory), every referencing tag and tool list is updated, and the entries are removed from `panel_specific` because the collision no longer exists.
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

- [ ] P3 — F331 — Pin OpenTimelineIO against the 0.19 bundle rewrite and contract-test the export path
  Why: The declared ceiling `opentimelineio>=0.17,<1` will silently admit 0.19, which moves `otioz`/`otiod` bundle handling out of Python into the C++ core — a behaviour change to a shipped export path — and the project has already been bitten once by an OTIO minor bump renaming `MediaReferencePolicy` enum members between 0.15 and 0.17.
  Evidence: `pyproject.toml:170,230`; `opencut/export/otio_export.py:547,566-567` (the enum-naming workaround); `opencut/export/otio_compat.py` reports the runtime version but pins and asserts nothing; https://github.com/AcademySoftwareFoundation/OpenTimelineIO/releases (0.18.1, 2025-11-09, still flagged prerelease, no release in 9 months)
  Touches: `pyproject.toml`, `requirements*.txt`, `opencut/export/otio_compat.py`, `opencut/export/otio_export.py`, `scripts/check_dependency_matrix.py`, `tests/`
  Acceptance: The OTIO specifier bounds the tested minor line rather than an open `<1`; a contract test round-trips an OTIOZ bundle and asserts the media-reference policy and bundle layout, failing on an untested OTIO minor; `otio_compat` records the verified-against version alongside the runtime version.
  Complexity: S

- [ ] P3 — F347 — Normalize line endings so one-line edits stop producing whole-file diffs
  Why: `core.autocrlf=false` and a `.gitattributes` that only covers `*.sh`/`*.command` mean git stores whatever byte sequence the writing tool emitted, and different editors push the same file in opposite directions — a one-key addition to `client/locales/en.json` landed as a 5,745-line diff and a README troubleshooting paragraph as 1,983 lines, which destroys reviewability and hides real changes inside mechanical churn.
  Evidence: commit 72a795b6 (7 files, 4,137 insertions for ~70 lines of real content; `git diff --ignore-cr-at-eol --stat` reports 15/1/49); `.gitattributes` covers only 4 shell-script patterns; `git config core.autocrlf` is `false`; CLAUDE.md already records "CEP `main.js` is mixed CRLF/LF"; the 2026-08-12 in-flight refactor showed the same flip on `model_cards.py` and `dump_feature_readiness.py`
  Touches: `.gitattributes`, then a single mechanical `git add --renormalize .` pass
  Acceptance: `.gitattributes` declares the text/eol policy for the source types this repo actually edits (`*.py`, `*.js`, `*.json`, `*.md`, `*.css`, `*.html`, `*.jsx`) with binary types excluded; one renormalization commit lands separately from any content change and is labelled as mechanical; after it, editing one line in `en.json` produces a one-line diff; run it only when no other session has uncommitted work.
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
