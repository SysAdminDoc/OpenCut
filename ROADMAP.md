# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

F359-F365 came from a different source: an audit run on 2026-08-20 over the F335-F358 drain, checking
each closed item against its own acceptance text rather than against the research. Highest prior
allocation before that audit: F358. F359 needed a maintainer's voice on a public issue and moved to
Roadmap_Blocked.md; the rest landed.

### P0

- [ ] P0 — F411 — Make resolved feature readiness authoritative across every machine surface
  Why: Eighteen generated readiness records disagree with terminal-stub resolution, and queue, extended MCP, and OpenAPI exposure can present operations that installing a dependency cannot make runnable.
  Evidence: Verified: `opencut/registry.py:148`, `opencut/_generated/feature_readiness.json`, `opencut/routes/jobs_routes.py:184`, `opencut/mcp_extended_tools.py:138`, and `opencut/core/openapi_source.py:231`.
  Touches: `opencut/registry.py`, readiness generators, `opencut/routes/jobs_routes.py`, `opencut/mcp_extended_tools.py`, `opencut/core/openapi_source.py`, install guidance, and catalogue tests.
  Acceptance: One per-adapter resolved state distinguishes available, dependency-gated, and terminal-stub implementations; generated manifests equal runtime resolution; queue admission rejects terminal stubs before job creation; extended MCP hides or explicitly marks them; OpenAPI carries readiness metadata; a new declared/runtime disagreement fails the suite.
  Complexity: L

- [ ] P0 — F412 — Apply feature readiness to real CEP and UXP controls
  Why: CEP's gating helper has no production `data-feature-id` bindings, UXP does not consume `/system/feature-state`, and UXP reconnect enables primary controls without restoring semantic prerequisites.
  Evidence: Verified: `extension/com.opencut.panel/client/feature-state.js:224`, `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.uxp/main.js:6819`, and `opencut/routes/system_diagnostics_routes.py:363`.
  Touches: CEP and UXP markup and controllers, shared readiness data, `extension/com.opencut.panel/tests`, and rendered panel tests.
  Acceptance: Every dependency-sensitive control has a canonical feature ID; unavailable controls are disabled with one actionable reason in both panels; a refresh re-enables a newly available feature; reconnect never enables a control whose dependency or selection prerequisite is still false; rendered tests cover disable, explanation, refresh, and re-enable in both themes.
  Complexity: M

### P1

- [ ] P1 — F414 — Generate one runnable command catalog and count only real user surfaces
  Why: The backend command catalog has 225 entries but only 43 runnable routes, 182 missing routes are accepted as speculative, and route accounting treats that backend list as direct user exposure.
  Evidence: Verified: `opencut/core/command_palette.py:400`, `tests/test_ux_intelligence.py:68`, `opencut/tools/dump_route_manifest.py:390`, and `extension/com.opencut.panel/client/main.js:13888`.
  Touches: `opencut/core/command_palette.py`, route and UX manifest generators, CEP and UXP command discovery, CLI and curated MCP registries, and surface-ratchet tests.
  Acceptance: One generated catalog separates live and aspirational commands; every live entry has method, payload schema, prerequisites, navigation target, and invoker; missing-route entries never appear in user search; direct-surface counts include only literal panel, CLI, or curated MCP consumers; deleting an invoker lowers the count and fails the ratchet.
  Complexity: L

- [ ] P1 — F416 — Unify visible media-library indexing behind one recursive incremental contract
  Why: Visible folder indexing is nonrecursive, stops at 100 files, and retranscribes unchanged media, while JSON, SQLite, and federated indexes expose inconsistent status and clear behavior.
  Evidence: Verified: `opencut/routes/search.py:71`, `opencut/routes/search.py:184`, `opencut/core/footage_search.py:314`, `opencut/core/footage_index_db.py`, and `opencut/core/federated_media_index.py`.
  Touches: Search routes, all three index stores, saved footage settings, CEP and UXP indexing flows, migrations, and index integration tests.
  Acceptance: A root-folder request recursively preflights file count, bytes, estimated transcription cost, and configured caps; unchanged media is skipped by signature; one versioned adapter supplies index, search, status, and clear semantics across stores; clear failures propagate; no nested lock is acquired; interruption resumes without duplicate work.
  Complexity: L

- [ ] P1 — F417 — Make every persisted editing setting authoritative or remove it with a migration
  Why: Footage-index options are stored but unused, UXP hardcodes the base transcription model, chapter and multicam execution bypass saved defaults, and color-profile persistence has no production consumer.
  Evidence: Verified: `opencut/user_data.py:815`, `opencut/user_data.py:860`, `opencut/user_data.py:875`, `extension/com.opencut.uxp/main.js:5956`, `opencut/routes/caption_analysis_routes.py:128`, and `opencut/routes/video_editing.py:636`.
  Touches: `opencut/user_data.py`, settings routes, indexing, chapter and multicam routes, both panels, settings migration, and conformance tests.
  Acceptance: Every persisted key has exactly one validated consumer or a documented removal migration; selected transcription model and language reach the job; chapter and multicam defaults affect execution; unknown and noncanonical keys are rejected; a generated settings-consumer test fails when a stored key is orphaned or bypassed.
  Complexity: M

- [ ] P1 — F418 — Issue a common validation receipt before any media output is published
  Why: Smart render proves staged probing and atomic promotion, but many media producers stop at process success or file existence, and declarative compose trusts planned duration after rendering.
  Evidence: Verified: `opencut/core/smart_render.py:430`, `opencut/core/delivery_validate.py`, `opencut/core/declarative_compose.py:493`, plus output-loss reports across LosslessCut, Kdenlive, Shotcut, and OpenShot trackers.
  Touches: `opencut/helpers.py`, async job completion, `opencut/core/delivery_validate.py`, media-producing route metadata, output staging helpers, and golden-media tests.
  Acceptance: Every media-producing route is classified against one output contract; completed jobs return a receipt covering expected streams, container and codec, duration tolerance, geometry, frame rate, audio layout, timestamp continuity, sampled beginning and end decode, and atomic promotion; an unclassified producer or injected corrupt, black, truncated, silent, or wrong-layout artifact fails before replacing an existing destination.
  Complexity: L

- [ ] P1 — F419 — Add a reviewed silent-failure budget and no-growth gate
  Why: The source contains 1,817 broad Python catches across 457 files and 165 empty JavaScript or JSX catches, with no machine-readable distinction between compatibility probes and lost failures.
  Evidence: Verified: repository-wide searches over `opencut/**/*.py`, `extension/**/*.js`, and `extension/**/*.jsx`; existing structured patterns live in `opencut/errors.py`, `opencut/checks.py`, and `opencut/core/pipeline_health.py`.
  Touches: A static analysis script, `opencut/errors.py`, logging and diagnostics, CEP and UXP controllers, release smoke, and fault-injection tests.
  Acceptance: Every retained broad or empty catch has an explicit reviewed suppression category; new unsuppressed catches fail the gate; host bridge, job, file I/O, output validation, and model-loading failures produce a typed user result plus structured log evidence; the committed baseline can only decrease unless an allowlist entry carries a reason and owner.
  Complexity: L

- [ ] P1 — F420 — Test agent-authored edits against executable editorial briefs
  Why: The model evaluation system measures individual inference calls, but no gate executes an edit brief and checks the resulting timeline for safety, constraint compliance, or deterministic invariants.
  Evidence: Verified: `opencut/core/ai_eval_harness.py:1`, `opencut/core/eval_datasets.py:56`, https://arxiv.org/abs/2509.10761, and https://arxiv.org/abs/2607.25300.
  Touches: `opencut/core/ai_eval_harness.py`, agent planner and executor modules, declarative compose, timeline fixtures, and evaluation reports.
  Acceptance: Fixture briefs cover source selection, duration, supported tools, duplicate clips, bounds, mid-word cuts, required and forbidden content, undo, and deterministic replay; injected violations fail the corresponding invariant; human or model preference scores are secondary and cannot override a deterministic failure; results record source hashes, plan, environment, and output receipt.
  Complexity: M

- [ ] P1 — F421 — Complete the backend-independent ASR integrity gate
  Why: OpenCut flags repeated-phrase loops, but it does not gate dropped windows, regressing word times, unexplained coverage gaps, overlapping stitches, or long spans of low-confidence output before transcript-driven mutations.
  Evidence: Verified for the local gap: `opencut/core/captions.py:1097`, `tests/test_asr_repetition_guard.py`, and `opencut/core/asr_provenance.py`; Likely demand from https://github.com/m-bain/whisperX/issues and https://github.com/SYSTRAN/faster-whisper/issues.
  Touches: Caption engines, ASR provenance, transcript cache, transcript-edit routes, panel warnings, and long-form audio fixtures.
  Acceptance: Every backend reports monotonic segment and word timing, decoded-audio coverage, gap and overlap anomalies, repetition, confidence distribution, and batch-window lineage; suspect spans are preserved and highlighted; transcript-driven cuts require review or an explicit recorded override; sequential and batched fixtures detect a deleted window, shifted timestamps, and a repeated tail.
  Complexity: M

- [ ] P1 — F422 — Combine transcript timing, waveform, shot boundaries, and live skip audition
  Why: CEP renders editable transcript segments and a waveform in separate workflows, so editors cannot judge word boundaries, cuts, confidence, and visual transitions in one place.
  Evidence: Verified: `extension/com.opencut.panel/client/index.html:466`, `extension/com.opencut.panel/client/main.js:7864`, `extension/com.opencut.panel/client/main.js:12159`, `opencut/core/transcript_timeline_edit.py`, and Subtitle Edit's waveform workflow.
  Touches: CEP and UXP transcript surfaces, `opencut/core/waveform_timeline.py`, transcript mapping modules, shot and confidence data, host audition calls, and rendered accessibility tests.
  Acceptance: One workbench aligns words, speakers, confidence, waveform, silence, shot boundaries, and proposed cuts; boundary drags snap without invalid ordering; selecting or deleting text auditions the kept result immediately without mutating Premiere; the final change set remains reviewable and reversible; keyboard, screen-reader, narrow-width, and reduced-motion tests pass.
  Complexity: L

- [ ] P1 — F423 — Audit caption accuracy, completeness, synchronization, and placement against source media
  Why: Current caption QC checks format compliance, glyphs, overlaps, and reading rules, but not whether spoken content, speaker identity, or meaningful sounds are represented.
  Evidence: Verified: `opencut/core/caption_qc.py:220`; standards: https://docs.fcc.gov/public/attachments/FCC-14-12A1_Rcd.pdf, https://www.section508.gov/create/captions-transcripts/, and https://www.w3.org/TR/WCAG22/.
  Touches: Caption QC, ASR, diarization, audio-event detection, obstruction analysis, export preflight, panel diagnostics, and multilingual accessibility fixtures.
  Acceptance: A report scores accuracy, completeness, synchronization, and placement; it identifies uncovered speech, unresolved speakers, missing meaningful sounds, timing drift, and obstruction; thresholds are language-aware and operator-overridable; uncertain findings remain advisory; export can fail only on configured deterministic rules; English and one non-English fixture prove coverage.
  Complexity: L

- [ ] P1 — F424 — Run dependency and packaging upgrades through one executable compatibility matrix
  Why: PyInstaller 6.20.0 trails a security fix, ONNX Runtime 1.27.0 and PyAV 18.0.0 trail hardened releases, Werkzeug locks disagree, OTIO documents Python only through 3.12, and the Docker base is mutable.
  Evidence: Verified: `requirements-build.txt:3`, `requirements-release-lock.txt:64`, `:841`, `:1381`, `requirements-lock.txt:30`, `Dockerfile`, https://github.com/pyinstaller/pyinstaller/security/advisories/GHSA-9fxf-4qw3-ghmr, https://github.com/microsoft/onnxruntime/releases/tag/v1.29.0, and https://github.com/PyAV-Org/PyAV/releases/tag/v18.1.0.
  Touches: Dependency declarations and locks, `Dockerfile`, `opencut_server.spec`, release smoke, capability reporting, model and media corpora, and packaging tests.
  Acceptance: PyInstaller is at least 6.22.2 and the canonical artifact is asserted onedir; Werkzeug locks converge; PyAV 18.1.0 and ONNX Runtime 1.29.0 pass CPU plus available GPU providers; Python 3.11.16, 3.12.14, 3.13.15, and 3.14.7 are exercised; free-threaded builds are explicitly excluded until green; the Docker base uses a patch tag and digest; OTIO is tested on 3.13 and 3.14 or reports unavailable with a reason; nonnative keyring backends fail closed for production secrets.
  Complexity: XL

### P2
- [ ] P2 — F445 — Compare the blocked-work ledger against the state it describes
  Why: F413 made the release-provenance and advisory documents answer to executable sources, but left the third source it named untouched. `Roadmap_Blocked.md` still carries entries whose stated evidence is provably stale, and nothing detects it: the release-publish entry claimed "the newest artifact anyone can install is 21 versions old" and cited v1.25.1 as latest while v1.55.1 had shipped, tagged, with assets. A ledger that describes the past as the present is worse than no ledger, because it is read as the current blocker list.
  Evidence: Verified 2026-09-05: `gh release list` shows v1.55.1 (2026-08-25) latest with `OpenCut-Setup-1.55.1.exe`, `payload.zip` and `release-digests.json` attached; `git tag -l v1.55.1` resolves and matches `opencut/__init__.py`; the entry's own Evidence line names v1.25.1 and thirteen untagged versions. `opencut/tools/check_provenance_docs.py` covers documents but not this file. Release versions and counts are separately covered by `dump_project_facts --check` and `sync_badges --check`, already in `GENERATED_DOC_CHECKS`.
  Touches: `opencut/tools/check_provenance_docs.py` or a sibling checker, `Roadmap_Blocked.md`, `scripts/release_smoke.py`.
  Acceptance: A check reads each blocked entry's machine-checkable claims (referenced tags, released versions, cited file paths and line anchors) and fails naming any that no longer hold; an entry whose blocker has cleared is reported rather than silently carried; the check runs in the release gate and passes on a ledger corrected in the same commit. `Roadmap_Blocked.md` is gitignored, so the check must tolerate its absence rather than failing a clean checkout.
  Complexity: M

- [ ] P2 — F442 — Make the FFmpeg banner parser reject a non-text banner instead of raising TypeError
  Why: `parse_ffmpeg_banner` assumes `str` and dies with `TypeError: cannot use a string pattern on a bytes-like object` when handed `bytes`, so a caller that captures FFmpeg output without `text=True` gets an unhandled crash inside the security guard rather than an unverified verdict. Two tests already hit it.
  Evidence: Verified 2026-09-05: `opencut/core/ffmpeg_provenance.py:446` (`_VERSION_RE.search(first_line)`); `tests/test_remote_realtime.py::TestFrameExtraction::test_extract_success` and `::test_extract_ffmpeg_failure` fail on a clean checkout, both because `@patch("opencut.core.realtime_ai.subprocess.run")` returns `stdout=b"..."` and that module-wide patch also reaches `_probe_bundled_banner`.
  Touches: `opencut/core/ffmpeg_provenance.py`, `tests/test_remote_realtime.py`.
  Acceptance: A bytes banner is decoded or refused with the module's typed unverified result, never a `TypeError`; the two frame-extraction tests pass without loosening what they assert about extraction; a fixture passing bytes directly to `parse_ffmpeg_banner` proves the path.
  Complexity: S


- [ ] P2 — F425 — Carry unresolved review comments across cut versions with confidence
  Why: Review comments are bound to immutable versions, so a recut strands unresolved feedback even when the same dialogue or shot survives at a new time.
  Evidence: Verified local model: `opencut/core/review_links.py:35`, `tests/test_review_versions.py`; Likely demand: https://www.reddit.com/r/editors/comments/q419v1 and https://kitsu.cg-wire.com/review/.
  Touches: Review versions, OTIO diffing, transcript anchors, content fingerprints, review portal and bundle UI, notifications, and review fixtures.
  Acceptance: Creating a version proposes mappings using timeline diff, transcript anchors, time warps, and perceptual hashes; each proposal preserves the original anchor and exposes method plus confidence; high-confidence mappings can be accepted in bulk; low-confidence mappings remain quarantined; deleted material stays unresolved on its original version; shifted, split, deleted, and duplicate-shot fixtures pass.
  Complexity: M

- [ ] P2 — F426 — Query federated visual embeddings with explicit resource budgets
  Why: Federated search imports visual sidecars but intentionally refuses text queries against them, while editor reports show uncontrolled background analysis can consume a workstation for hours.
  Evidence: Verified local gap: `opencut/core/federated_media_index.py:1184`; Likely resource demand: https://www.reddit.com/r/AdobePremiere/comments/1vqw9mb/media_intelligence_analysis/ and https://www.reddit.com/r/premiere/comments/1qpqbzz/media_intelligence_analysis_question/.
  Touches: Federated and multimodal indexes, embedding sidecars, scheduler and GPU semaphore, settings, search panels, and performance fixtures.
  Acceptance: Text-to-visual search works across selected roots using versioned reusable sidecars; users choose scope, schedule, CPU or GPU budget, and pause or resume; unchanged media is not re-embedded; foreground jobs preempt background indexing; results expose model revision and matched frames; a large-library fixture proves bounded memory and no duplicate indexing.
  Complexity: L

- [ ] P2 — F427 — Rank audio-repair variants with quality evidence and require audition
  Why: OpenCut has several repair engines, but choosing a model by name gives no evidence that denoising preserved speech or avoided new artifacts.
  Evidence: Verified local engine breadth: `opencut/core/audio_enhance.py`, `opencut/core/ab_compare.py`; research: https://github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/README.md, https://github.com/microsoft/Distill-MOS, and https://arxiv.org/abs/2603.04710.
  Touches: Audio enhancement, damaged-region detection, A/B comparison, ASR stability, loudness and artifact metrics, panel audition, and audio fixtures.
  Acceptance: The system renders conservative variants only for detected damaged regions; ranks them by no-reference speech quality, ASR stability, loudness, clipping, and artifact checks; explains every score; never auto-commits the winner; an A/B audition preserves the original and selected variant; fixtures include speech where the cleanest-sounding output has worse recognition.
  Complexity: M

- [ ] P2 — F428 — Exchange review notes and drawings through ORI OTIO annotations
  Why: Review bundles encode comments as ordinary OTIO markers and drawings as OpenCut SVG assets, so other review systems cannot recover standardized annotation semantics.
  Evidence: Verified local format: `opencut/core/review_bundle.py:368`; specification: https://lf-aswf.atlassian.net/wiki/spaces/PRWG/pages/605814827/OTIO%2B2D-Annotations%2BInterchange%2Bspecification.
  Touches: Review bundle import and export, OTIO and OTIOZ adapters, drawing assets, marker metadata, version migrations, and round-trip fixtures.
  Acceptance: OpenCut imports and exports ORI `ANNOTATION_1.0` notes and drawings in OTIOZ while retaining current marker and SVG compatibility; unsupported fields survive in namespaced metadata; version and schema validation fail clearly; standardized and legacy bundles round-trip without losing time range, author, status, text, color, or drawing geometry.
  Complexity: M

- [ ] P2 — F429 — Add stable asset identity and ASC MHL fixity to relink and provenance
  Why: Path-based media breaks across machines, NAS moves, proxies, and archives, while OpenCut already computes hashes separately for deduplication, sidecars, and C2PA ingredients.
  Evidence: Verified local seams: `opencut/core/content_fingerprint.py`, `opencut/core/federated_media_index.py`, and `opencut/core/c2pa_sidecar.py`; standards: https://github.com/OpenAssetIO/OpenAssetIO and https://github.com/ascmitc/mhl-specification.
  Touches: Ingest, asset identity and resolver adapters, proxy lineage, relink, federated index, C2PA ingredients, review and interchange manifests, and migration tests.
  Acceptance: An optional adapter assigns a stable asset ID independent of path, resolves current location and version, imports and exports ASC MHL records, reuses verified hashes instead of recomputing them, links proxies to originals, and feeds verified transfer history into C2PA ingredients; missing adapters fall back to existing paths; moved, renamed, proxy, tampered, and offline fixtures pass.
  Complexity: L

### P3
- [ ] P3 — F443 — Give `chapter_defaults.naming_style` a consumer or drop it with a migration
  Why: `load_chapter_defaults` persists `naming_style` ("descriptive" / "numbered" / "timecode") and the settings API serves it, but `generate_chapters` has no parameter for it, so the value cannot reach chapter titles by any path. It is the one key left in a file whose other two keys are now applied.
  Evidence: Verified 2026-09-05: `opencut/user_data.py:925` declares it; `opencut/core/chapter_gen.py:244-249` takes only `segments`, `llm_config`, `max_chapters` and `min_chapter_duration`; `opencut/routes/caption_analysis_routes.py` reads the defaults and deliberately skips this key.
  Touches: `opencut/core/chapter_gen.py`, `opencut/routes/caption_analysis_routes.py`, `opencut/core/settings_registry.py`, `tests/test_settings_consumers.py`.
  Acceptance: Either `generate_chapters` accepts a naming style and the three documented values produce visibly different chapter titles under test, or the key is removed from the saved defaults with a migration and the registry records why; the settings-consumer test covers whichever was chosen at key level rather than file level.
  Complexity: S

- [ ] P3 — F444 — Retire the colour-profile and auto-zoom settings surfaces
  Why: `color_profiles.json` and `auto_zoom_presets.json` are written, served by REST, and read by nothing: no backend module, and zero references in either panel. `opencut/core/settings_registry.py` already classifies them as removed and `migrate_removed_settings` deletes them, but the loaders and endpoints are still live, so the migration cannot be enabled without deleting what a user had just saved.
  Evidence: Verified 2026-09-05: zero readers of either filename outside `user_data.py` and `routes/settings.py`; zero matches for `color-profiles` or `auto-zoom-presets` in `extension/com.opencut.panel/client/main.js` and `extension/com.opencut.uxp/main.js`; endpoints at `opencut/routes/settings.py:1005` and `:1014`, plus the colour-profile loader at `opencut/user_data.py:860`.
  Touches: `opencut/user_data.py`, `opencut/routes/settings.py`, `opencut/core/settings_registry.py`, server startup, and the generated route, readiness and extended-MCP manifests.
  Acceptance: The loaders, save functions and REST endpoints are gone; `migrate_removed_settings` runs once at startup and is covered by a test that plants both files and asserts they are deleted while a live setting survives; the generated manifests, route counts and README badges are regenerated in the same commit and the surface ratchet passes.
  Complexity: M


- [ ] P3 — F430 — Import bitmap subtitles through OCR and a reviewable typesetting lane
  Why: OpenCut supports broad text-caption import and export but has no PGS, VobSub, or burned-subtitle OCR workflow, while professional subtitle tools treat image-subtitle recovery as a standard ingest need.
  Evidence: Verified local absence: no bitmap-subtitle route or surface outside FFmpeg advisory text; competitor evidence: https://github.com/SubtitleEdit/subtitleedit/releases and https://github.com/TypesettingTools/Aegisub.
  Touches: Subtitle stream extraction, OCR adapters, caption confidence and timing, ASS style mapping, transcript workbench, export preflight, and hostile-subtitle fixtures.
  Acceptance: PGS and VobSub streams can be extracted without trusting a vulnerable embedded decoder; sampled burned captions can be detected and OCRed; each cue retains source image, timing, language, text confidence, and style hints; low-confidence cues require review; edited results export to SRT, WebVTT, and ASS; duplicate, overlapping, vertical, and forced-caption fixtures pass.
  Complexity: L


## Audit Findings — 2026-08-22

Deep audit pass over v1.53.0 (HEAD `bb3e4102`). IDs continue the F-number scheme; highest prior
allocation across `ROADMAP.md`, `Roadmap_Blocked.md`, `CHANGELOG.md` and `RESEARCH.md` was F388.

Baseline recorded before any finding was logged, so nothing below is a regression from this pass:
`py -3.13 -m pytest -q` = **11401 passed, 45 skipped, 57 deselected, 4620 subtests passed** (921 s, exit 0);
`ruff check opencut/` and `ruff check opencut/ --select E,F,I --ignore E501` = clean;
`py -3.13 scripts/sync_version.py --check` = all targets at 1.53.0;
`py -3.13 scripts/check_doc_sizes.py` = all within tolerance;
`py -3.13 scripts/i18n_lint.py --check` and `scripts/lint_locales.py` = clean;
`py -3.13 scripts/lint_subprocess_timeouts.py` = clean;
`npm test` (panel vitest) = 255 passed; `npx playwright test` (rendered) = 69 passed, 1 skipped.
Three commands were NOT green on the same clean checkout: `node tests/jsx_mock.js` (exit 1),
`py -3.13 -m pytest -m "integration or slow"` (1 failed), and `npm run lint` (4 ungated warnings).
Those are F394, F398 and F395 below.

Checked and found clean, recorded so the next pass does not re-derive it. **Performance**: panel tab
switches measured 2.4-17.8 ms across two full rounds of all eight tabs, theme toggle 8.6-13.9 ms, and
the document held a steady 4,989 nodes before and after 16 tab switches, so the lazy-tab system is not
leaking. Backend GETs against a live server on 5688: `/health` 9.7 ms, `/jobs` 1.7 ms, `/presets` 1.9 ms,
`/export/presets` 6.7 ms, `/caption-styles` 6.5 ms, `/models/list` 8.6 ms, `/api/routes` 30.5 ms for
284 KB, `/queue/coverage` 3.7 ms. `/system/dependencies` costs 4.3 s cold, which is the documented
behaviour behind its 60 s TTL cache. **Security scanners**: `gitleaks detect` over 1,670 commits returns
only historical hits on obviously synthetic test fixtures (`tests/test_integration.py:256`,
`tests/test_telemetry_aptabase.py:200` — `"A-US-1234567890"`); `bandit -r opencut -ll` reports 0 high and
133 medium, all of which are `B310`/`B615`/`B104` advisory categories on reviewed call sites, no
injection or unsafe-deserialization findings. **CSRF coverage**: all 1,190 mutating URL rules carry the
`_opencut_requires_csrf` marker — zero gaps. **Subprocess timeouts**: `scripts/lint_subprocess_timeouts.py`
is clean, and an independent AST sweep of all of `opencut/**` found no `subprocess.run`/`call`/
`check_output` without `timeout=` and no unbounded process `wait()`/`communicate()`. One caveat worth
fixing opportunistically: that linter's `SCAN_DIRS` covers only `opencut/core` and `opencut/routes`, so
`opencut/helpers.py` — which holds the two most-used `Popen` sites in the codebase — is never scanned.
It is correct today by inspection, not by enforcement.

### P1

### P2

### P3

- [ ] P3 — F406 — Unaudited surfaces from the 2026-08-22 pass
  Category: quality
  Where: listed below.
  Problem: these areas were not examined in the 2026-08-22 audit and carry no verdict either way. Recording them so the next pass starts from an honest map rather than assuming coverage.
  Evidence: the 2026-08-22 pass drove the CEP panel in a browser across all eight tabs in both themes, traced the CSRF/bootstrap path end to end, ran every gate listed in this section's baseline, and AST-scanned `opencut/routes/**` for boolean-flag and CSRF invariants. It did not touch the areas below.
  Checked since this item was written: the UXP panel's rendered light-theme contrast. F399 made the axe gate walk the whole scroll container and resolve the results axe could not decide, and it runs over both surfaces at every tab, theme and width, so UXP light is now covered by `npx playwright test`. It found and fixed one real failure there (the FCC source link at 4.33:1).
  The CEP and UXP workbench shells were also checked at 480 and 1200 pixels across dark, light, automatic, reduced-motion and forced-colours states. The 72-case rendered matrix is clean after the v1.55.0 typography and density pass; no unresolved visual finding was left from that pass.
  Not audited: the WPF installer under `installer/src` and `Install.ps1` / `OpenCut.iss` / `install.py`; the Docker and Flatpak/AppImage packaging lanes; `opencut/mcp_server.py` and the MCP tool surface; the plugin runtime and trust model in `opencut/core/plugins.py` and `plugin_runtime.py` beyond a read of the loader's shape; `opencut/core/**` (roughly 600 modules) except where a route traced into it; the CLI beyond `--help`, three error paths and exit codes; SSE and WebSocket streaming under load; and any behaviour that needs a live Premiere host, which is already tracked as F386 in `Roadmap_Blocked.md`.
  Fix: pick one area per pass and give it the same treatment — run it, probe its error paths, and either log findings or record here that it was checked and is clean.
  Acceptance: each line above is either replaced by findings or moved to a "checked clean on <date>" note with the command that proved it.
  Confidence: Verified
  Effort: L

## Research-Driven Additions — 2026-09-04

Added 2026-09-04 from the research pass recorded in `RESEARCH.md`. IDs continue the F-number scheme;
highest prior allocation across `ROADMAP.md`, `Roadmap_Blocked.md`, `CHANGELOG.md` and `RESEARCH.md`
was F430.

Two drivers: the first substantive external bug reports against a released artifact (issues #7 and #8,
both v1.55.1 on Windows 11), and Adobe's scheduled September 2026 end of ExtendScript support in
Premiere Pro. The 2026-08-23 conclusion that Adobe had published no firm CEP cutoff is stale.

Not re-queued because they shipped since 2026-08-23: embedded-decoder attestation against the FFmpeg
8.1.2 floor for CVE-2026-8461, and the huggingface-hub upgrade past the 1.26.0 path-traversal fix.

### P0


### P1


### P2

