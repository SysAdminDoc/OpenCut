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

### P1

### P2

### P3


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

- [ ] P2 — F398 — `tests/test_batched_asr.py` asserts word-for-word equality between sequential and batched Whisper and fails (pre-existing baseline)
  Category: testing
  Where: `tests/test_batched_asr.py:255-257`, in `TestBatchedDecodingKeepsTheTranscriptCaptionsAreBuiltFrom::test_words_survive_batching_even_though_segments_do_not` (class marked `@pytest.mark.slow` at `:217`).
  Problem: the test asserts `[w.word for w in sequential] == [w.word for w in batched]`. Whisper's batched pipeline legitimately re-punctuates at re-segmentation boundaries, so this is an assertion about model determinism that the model does not offer. Because `pyproject.toml:275` sets `addopts = '-m "not integration and not slow"'`, the default suite never runs it, and it has rotted unnoticed.
  Evidence: `py -3.13 -m pytest -q -m "integration or slow"` → `1 failed, 55 passed, 2 skipped, 11445 deselected in 57.51s`, exit 1. The failure is `AssertionError: batching must not change the words themselves ... At index 69 diff: ' second.' != ' second'`. Run context printed by the test itself: `[F361] 31.2s speech, CPU int8 Systran/faster-whisper-medium.en: sequential 12.6s / 8 segments, batched 10.6s / 2 segments`. The default suite is fully green on the same checkout (11401 passed, 0 failed), so this is only visible when the deselected markers are run.
  Fix: assert the property the caption workflow actually depends on rather than byte equality — compare the concatenated transcripts after normalising trailing punctuation and whitespace, or require a token-level similarity above a stated threshold (the surrounding timing assertions at `:259-268` already use this shape with explicit tolerances and a comment explaining the measured numbers). Keep the exact-equality check only if it is narrowed to the alphabetic content of each word.
  Acceptance: `py -3.13 -m pytest -q -m "integration or slow"` exits 0 on this machine, and the relaxed assertion still fails if a word is genuinely dropped or reordered (prove it with a unit-level fixture that deletes one word).
  Confidence: Verified
  Effort: S

- [ ] P2 — F399 — Three separate reasons the contrast gates report green while 26 AA violations render
  Category: testing
  Where: `opencut/tools/contrast_audit.py` (`TOKEN_BLOCK_RE` at `:129-132`, `PANEL_PAIRS` at `:69-106`); `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs` — `assertWcagCompliance()` at `:22-32`, `LIGHT_THEME_PROBES` at `:1124-1137`, and the light-theme probe loop at `:1192-1196`.
  Problem: three gates all point at contrast and all miss F390. (1) `contrast_audit.py` parses only `:root` / `html.theme-light` custom-property blocks and compares named token pairs, so a rule writing `color: rgba(244,239,229,.94)` directly is outside its model by construction. (2) `assertWcagCompliance()` runs axe with the full WCAG 2.2 AA tag set and an empty `WCAG_SUPPRESSIONS`, and it does visit the Settings tab in light theme at 900 px — but `#mainContent` is `overflow-y: auto` with `scrollHeight` 7610 against `clientHeight` 828, and the test never scrolls, so axe only ever analyses the top 11% of the page. (3) the assertion reads `results.violations` and discards `results.incomplete`, which is where axe parks the cases it cannot decide.
  Evidence: an unscrolled axe run scoped to the Settings tab in light theme returns 0 violations and 38 passes. Re-running the same analysis while scrolling `#mainContent` in 700 px steps returns **26 unique `color-contrast` violations**, including all five `.hint-title` / `.hint-copy` / `.hint-kicker` triples, all three `.about-link` items, all four `.checkbox-row label` rows and both `.param-label`s. The `kbd` chips and the two Save buttons never appear in `violations` at any scroll position — axe files them under `incomplete` with "Element has a 1:1 contrast ratio with the background" and "background color could not be determined due to a background gradient", so even a scroll fix leaves those three green. Separately, `LIGHT_THEME_PROBES.cep` covers only `.quick-action-icon`, `.workspace-stage-card-icon`, `.content-subtitle`, `.quick-action-meta`, `.card-desc` and three surfaces, and the loop at `:1192` never leaves the default tab. `py -3.13 -m pytest tests/test_contrast_audit.py -q` and `npx playwright test` both pass on a tree where `.hint-title` renders at 1.05:1.
  Fix: fix all three, smallest first. Make `assertWcagCompliance()` scroll the tallest scroll container to the bottom in viewport-sized steps and union the results across positions before asserting. Then assert on `results.incomplete` too — either fail on it, or allow-list specific rule/target pairs with a stated reason so a new one cannot appear silently. Then widen `LIGHT_THEME_PROBES.cep.text` and drive the probe across every tab, not just the default one. Keep `contrast_audit.py` as the cheap token gate; just stop treating it as the contrast gate.
  Acceptance: with the scroll fix alone, `npx playwright test` fails on the current tree naming the F390 selectors, and passes once F390, F392 and F393 land. With the `incomplete` handling added, the `kbd` and Save-button cases are either failures or explicitly allow-listed. `scripts/release_smoke.py` gains a step that runs the rendered suite so the release gate depends on it.
  Confidence: Verified
  Effort: M

- [ ] P2 — F400 — The forced-colors "disabled controls" test can never run on CEP
  Category: testing
  Where: `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:3040-3066`, the `${surfaceName} distinguishes disabled controls without a tint` test, and its `test.skip(...)` guard at `:3048-3051`.
  Problem: the test needs one enabled and one disabled `button.quick-action-btn` in the initial view to compare like with like, and skips when either is missing. In the CEP surface the initial view is the backend-disconnected state, where all nine quick-action buttons are disabled and none is enabled, so the guard fires every time. The result is a permanent skip, not a transient one: whether CEP disabled controls stay distinguishable under Windows High Contrast is simply unverified, while the UXP twin of the same test does run and passes.
  Evidence: `npx playwright test --config playwright.config.mjs` → `69 passed, 1 skipped`, and the skipped entry is `- 64 [chromium] › panel-regression.spec.mjs:3040:5 › cep forced-colors › cep distinguishes disabled controls without a tint`. Measured in the rendered panel at that width: `document.querySelectorAll('button.quick-action-btn')` → 9 total, `:disabled` → 9, `:not(:disabled)` → 0.
  Fix: give the CEP case a pair it can actually find. Either drive the surface into a connected state before the assertion using the existing `backendFixtures` mock in the same spec (the fixture server at `:237` already serves `/settings/onboarding`, so a health fixture that reports connected is in reach), or compare a disabled `.quick-action-btn` against an enabled control of the same class elsewhere in the document by explicitly enabling one in the page before measuring. Do not widen the class comparison — the test's own comment explains why comparing across classes proves nothing.
  Acceptance: `npx playwright test` reports 70 passed and 0 skipped, and the CEP variant fails if the disabled cue is reduced to a background tint.
  Confidence: Verified
  Effort: S

- [ ] P2 — F401 — 552 of 769 async routes cannot be queued and nothing pins the intended exclusions
  Category: ux
  Where: `opencut/routes/jobs_routes.py:180` `_ALLOWED_QUEUE_ENDPOINTS` (217 entries, with the comment "Only processing-oriented routes may be invoked via the queue"), the coverage reporter at `:720-766` `queue_coverage()`, and `tests/test_queue_coverage.py`.
  Problem: the allowlist has fallen roughly four-fold behind route growth. Routes that are unambiguously "processing-oriented" by the list's own stated criterion — `/video/repair`, `/video/relight`, `/video/proxy/generate`, `/adr/sync`, `/agent/auto-edit` — return `400 ENDPOINT_NOT_QUEUEABLE` from `POST /queue/add` while working fine when called directly. The reporter was built deliberately to surface the gap without pre-empting the curation decision (its docstring says so), but the decision has never been made, and no test pins the intentional exclusions, so a newly added async route falls out of the queue silently rather than visibly.
  Evidence: `GET /queue/coverage` against a live server returns `async_post_routes: 769, queueable: 217, not_queueable: 552, coverage_percent: 28.2, allowlist_size: 217, stale_allowlist_entries: []`. `POST /queue/add` with a valid CSRF token → `/video/repair`, `/video/stabilize-advanced` and `/video/proxy/generate` all return 400 `ENDPOINT_NOT_QUEUEABLE`, while the allowlisted `/silence` returns 200 with a queue position. The two existing tests only check the reverse direction: `tests/test_hardening.py:725` asserts no phantom entries, and `tests/test_queue_coverage.py::test_missing_entries_carry_enough_context_to_act_on` returns early when `missing` is non-empty, so it passes at any coverage level.
  Fix: make the curation decision the reporter was written to prompt, and encode it. Group the 552 by blueprint from the `/queue/coverage` output, add the processing-oriented ones to `_ALLOWED_QUEUE_ENDPOINTS`, and move the deliberate exclusions into a named `_QUEUE_EXCLUDED_ENDPOINTS` frozenset alongside it with a one-line reason per entry. Do not auto-derive the allowlist from the `_opencut_async_job` marker — that would make non-processing routes queueable and discards the criterion the list exists to express.
  Acceptance: `tests/test_queue_coverage.py` gains `assert client.get("/queue/coverage").get_json()["missing"] == []`, and a second assertion that every entry in `_QUEUE_EXCLUDED_ENDPOINTS` still resolves to a live async POST route. Adding a new `@async_job` POST route without listing it in either set fails the suite.
  Confidence: Verified
  Effort: L

### P3

- [ ] P3 — F407 — The stage session card is too dense to read at the panel's default width
  Category: ux
  Where: `extension/com.opencut.panel/client/command-center.css` — `html body .workspace-stage-session-item` (the grid declared for label/value) and the three-column `.workspace-stage-session` track list it sits inside.
  Problem: at a 900px viewport the session card splits into three ~110px cells, so after the label and gap each value gets 33-47px. Every value ellipsises to two or three characters: "SOURCE Aw…", "SUITE Cut & …", "STATUS Rec…". The card technically fits and truncates honestly, but it conveys nothing.
  Evidence: measured after F391 at vw=900 with the backend disconnected — item clientWidth 110 and scrollWidth 110 for all three (so no overflow), value widths 33 / 47 / 35 px, and `value.scrollWidth > value.clientWidth` true on all three. Screenshot of the Cut tab at 900x800 shows the three truncated values. Before F391 the same card read "Reconnect backer", clipped with no ellipsis; F391 made the truncation honest but did not make it readable.
  Fix: stop forcing three columns at this width. Give `.workspace-stage-session` an `auto-fit` track list with a sensible minimum (`repeat(auto-fit, minmax(180px, 1fr))`) so the cells stack into one or two rows instead of shrinking below a legible width, or move the label above its value at narrow widths so the value gets the full cell.
  Acceptance: at viewport widths 701, 900 and 1200 with the backend disconnected, no `.workspace-stage-session-value` has `scrollWidth > clientWidth` — every value renders in full — and the card still shows no overflow (`scrollWidth <= clientWidth`).
  Confidence: Verified
  Effort: S

- [ ] P3 — F402 — 19 em dashes and several banned words in user-facing CEP panel copy, with no gate
  Category: docs
  Where: `extension/com.opencut.panel/client/locales/en.json` — keys `audio.deepfilter_desc`, `audio.effects_desc`, `assistant.empty_good`, `cut.filler_backend_crisper`, `cut.filler_backend_whisper`, `onboarding.ready`, `silence.waveform_label`, `timeline.otio_saved`, `toast.changelog_released`, `toast.demo_loaded`, `toast.issue_report_opened`, `toast.job_replay_missing_params`, `toast.job_rerun_missing_params`, `toast.update_available`, `toast.watermark_region_autofilled`, `video.style_desc`, `video.watermark_detected_region`, `wave_h.send_log`, `ws.settings_desc`. Plus `assistant.sequence_subtitle` ("highest-leverage"), `interview.source_required_hint`, `media.source_empty_copy`, `whisper.install_status_title` (all "unlock").
  Problem: this is text an end user reads inside the panel, and the project's writing standard for user-facing copy rules out em dashes and the "unlock / leverage" register. The CEP locale is the only file affected — `extension/com.opencut.uxp/locales/en.json`, `es.json` and `extension/shared-locales/en.json` all contain zero em or en dashes — so the inconsistency is also internal.
  Evidence: a scan of all 2,904 strings in the CEP `en.json` finds 19 containing `—`, 0 containing `–`, 1 spaced-hyphen (`settings.plugin_publisher_identity`), and 7 hits on the banned-vocabulary list. The same scan over the other four locale files returns 0 dashes each. No gate covers it: `scripts/i18n_lint.py` checks dead keys, HTML fallback drift and `t(key, fallback)` drift, but not punctuation or register.
  Fix: rewrite the 19 strings with a period, a comma, or parentheses in place of the em dash, and replace "unlock" and "highest-leverage" with plain verbs. Remember that `scripts/i18n_lint.py --check` will name the HTML `data-i18n` fallback text and the JS `t(key, fallback)` literals that must move with each value — change those in the same edit.
  Acceptance: a new check in `scripts/i18n_lint.py` fails on any locale value containing `—` or `–`; `py -3.13 scripts/i18n_lint.py --check` passes after the rewrite, with no new fallback drift.
  Confidence: Verified
  Effort: S

- [ ] P3 — F403 — The panel stylesheet stack is an override war, and it is producing the visual bugs
  Category: maintainability
  Where: `extension/com.opencut.panel/client/index.html:8-12` loads five stylesheets in order — `style.css` (19,010 lines), `command-center-tokens.css` (52), `command-center-layout.css` (232), `command-center.css` (2,426), `studio-workbench-v2.css` (1,763).
  Problem: the later sheets do not extend the earlier ones, they fight them. `studio-workbench-v2.css` uses `!important` 682 times across 1,763 lines, and `command-center.css` 71 times; `command-center.css:2134` is introduced by the comment "Final composition overrides live last so legacy layout layers cannot reintroduce wrapping", which is a description of the problem rather than a fix. `.workspace-stage-actions` alone matches 53 rules across three sheets and changes `display` twice. This is not a style preference — both P1 visual defects in this pass originate in exactly that layering: F391's clipping comes from a "final override" that removed wrapping without removing the clip, and F396's three dead `grid-template-columns` declarations are neutralised by a `display: flex !important` in a different file.
  Evidence: `grep -c '!important'` per file — `style.css` 22, `command-center-tokens.css` 0, `command-center-layout.css` 0, `command-center.css` 71, `studio-workbench-v2.css` 682. `grep -c` for selector occurrences across all five sheets — `.workspace-stage-actions` 53, `.stage-action` 66, `.btn-primary` 63, `.card-title` 28. Runtime CSSOM inspection of `.workspace-stage-actions` shows six rules setting `flex-wrap` or `overflow`, three of which are unconditional and contradict each other.
  Fix: take one component at a time rather than attempting a rewrite. For `.workspace-stage-actions` and `.stage-action`, collapse the 53 rules into a single owning block in `command-center.css` (the layout layer), delete the `!important` duplicates in `studio-workbench-v2.css`, and keep only genuinely conditional rules in media queries. Land it behind the rendered screenshot suite, which will show any unintended change. Record the component-by-component plan in `CLAUDE.md` so the next pass continues rather than restarts.
  Acceptance: after the first component is consolidated, `.workspace-stage-actions` matches at most 6 rules across all sheets with no `!important`, the rendered suite passes with regenerated baselines, and F391's clipping assertion holds.
  Confidence: Verified
  Effort: L

- [ ] P3 — F404 — `CLAUDE.md`'s documented panel file sizes are stale and unchecked
  Category: docs
  Where: `CLAUDE.md` "Frontend (CEP Panel)" and "UXP Panel" sections; the checker is `scripts/check_doc_sizes.py`, whose targets cover `README.md` only.
  Problem: `CLAUDE.md` is the file every agent reads first, and its size claims are the cheapest way to judge whether a note is current. They are all understated: `main.js` "~15263 lines as of 2026-05-25" is actually 18,814; `style.css` "~17870" is 19,010; `index.html` "~4061" is 4,302; `host/index.jsx` "~2736" is 4,128; UXP `main.js` "~5568" is 10,169 (an 83% understatement); UXP `style.css` "~3863" is 5,256; UXP `index.html` "~1466" is 2,134. `check_doc_sizes.py` verifies the same claims in `README.md` and reports them all within tolerance, so the drift is confined to the file that has no checker.
  Evidence: `py -3.13 scripts/check_doc_sizes.py` → "All documented sizes within tolerance" for 13 README targets. `wc -l` on the seven panel files gives the actual numbers above.
  Fix: refresh the seven numbers in `CLAUDE.md` and add the same targets to `scripts/check_doc_sizes.py` so it checks `CLAUDE.md` alongside `README.md`. The existing target model already carries a per-target tolerance, so this is a data addition rather than new logic.
  Acceptance: `py -3.13 scripts/check_doc_sizes.py` reports the `CLAUDE.md` panel-size targets and fails when one drifts past its tolerance.
  Confidence: Verified
  Effort: S

- [ ] P3 — F405 — The panel ships defaulting to Dark, so it ignores Premiere's skin out of the box
  Category: ux
  Where: `extension/com.opencut.panel/client/index.html:3780-3784` — the `#settingsTheme` select carries `selected` on `<option value="dark">`, not on `<option value="auto">`. The consumers are `main.js:8942` `_currentThemePref()`, `main.js:13672` `_applyTheme(_currentThemePref())` at init, and `extension/com.opencut.panel/client/cep-theme.js` `resolveTheme()`.
  Problem: `cep-theme.js` exists to make the panel follow Premiere's skin — its module docstring says so, it classifies all four Premiere skins, and it subscribes to `THEME_COLOR_CHANGED`. But `resolveTheme(pref, ...)` short-circuits on an explicit `"dark"` or `"light"` preference and only consults the host when the preference is `"auto"`. On a fresh install there is no saved setting, so `_currentThemePref()` returns the markup default `"dark"`, `_applyTheme("dark")` runs at init, and none of the host-skin machinery ever fires. A user running Premiere's light skin gets a dark panel that clashes with the rest of the application until they find Settings → Appearance and pick Auto. The one control that would fix it is the same control that, today, walks them into F390.
  Evidence: `index.html:3782` is `<option value="dark" selected data-i18n="settings.theme_dark">Dark</option>` while `:3781` is the unselected `auto`. `main.js:9019-9022` applies a theme only `if (settings.theme && ...)`, so a fresh `localStorage` leaves the markup default in force. `main.js:13672` calls `_applyTheme(_currentThemePref())` unconditionally at init. `cep-theme.js` `resolveTheme` returns `{isLight:false, premiereTheme:"dark", source:"user"}` for `pref === "dark"` before it looks at `hostTheme`.
  Fix: move `selected` to the `auto` option at `index.html:3781` so a fresh install follows the host, which is what the rest of the theme system is built for. Land it after F390, F392 and F393 — until those are fixed, defaulting more users into light theme makes the contrast problem worse, not better.
  Acceptance: with `localStorage` cleared and a host skin reporting light, the panel renders in light theme without the user touching Settings; with a dark host skin it renders dark; an explicit Light or Dark choice still overrides the host. A vitest over `resolveTheme` plus a rendered assertion on the default `#settingsTheme` value covers both halves.
  Confidence: Verified
  Effort: S

- [ ] P3 — F406 — Unaudited surfaces from the 2026-08-22 pass
  Category: quality
  Where: listed below.
  Problem: these areas were not examined in the 2026-08-22 audit and carry no verdict either way. Recording them so the next pass starts from an honest map rather than assuming coverage.
  Evidence: the 2026-08-22 pass drove the CEP panel in a browser across all eight tabs in both themes, traced the CSRF/bootstrap path end to end, ran every gate listed in this section's baseline, and AST-scanned `opencut/routes/**` for boolean-flag and CSRF invariants. It did not touch the areas below.
  Not audited: the UXP panel's own rendered light-theme contrast (the browser session repeatedly executed against the CEP document instead of the UXP one, so the single "clean" reading taken for UXP is not trustworthy and should be redone); the WPF installer under `installer/src` and `Install.ps1` / `OpenCut.iss` / `install.py`; the Docker and Flatpak/AppImage packaging lanes; `opencut/mcp_server.py` and the MCP tool surface; the plugin runtime and trust model in `opencut/core/plugins.py` and `plugin_runtime.py` beyond a read of the loader's shape; `opencut/core/**` (roughly 600 modules) except where a route traced into it; the CLI beyond `--help`, three error paths and exit codes; SSE and WebSocket streaming under load; and any behaviour that needs a live Premiere host, which is already tracked as F386 in `Roadmap_Blocked.md`.
  Fix: pick one area per pass and give it the same treatment — run it, probe its error paths, and either log findings or record here that it was checked and is clean.
  Acceptance: each line above is either replaced by findings or moved to a "checked clean on <date>" note with the command that proved it.
  Confidence: Verified
  Effort: L
