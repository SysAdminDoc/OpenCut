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

- [ ] P3 — F408 — `scripts/` is outside the ruff gate and has drifted
  Category: maintainability
  Where: `scripts/fix_es_diacritics.py:5` (unused `sys` import) and `:178` (f-string with no placeholder); the gate is `scripts/release_smoke.py` `step_ruff`, which runs `ruff check opencut/`.
  Problem: the release gate lints `opencut/` only, so the 28 files under `scripts/` — including the release gate itself, the version syncer and the i18n linter — are never checked. Three ruff findings have accumulated there unnoticed. None is a live defect, which is exactly why nothing surfaced them.
  Evidence: `py -3.13 -m ruff check opencut/` passes; `py -3.13 -m ruff check scripts/` reports `Found 3 errors` (2 auto-fixable). Confirmed pre-existing: `git diff --stat scripts/fix_es_diacritics.py` is empty on a tree where the other `scripts/` files were edited.
  Fix: clear the three findings, then widen `step_ruff` to `ruff check opencut/ scripts/` so the tooling holds the same bar as the package. Check `tests/test_release_smoke.py` for an assertion pinning the ruff argument list before changing it.
  Acceptance: `py -3.13 -m ruff check opencut/ scripts/` exits 0, and `py -3.13 scripts/release_smoke.py --json --only ruff` fails when a new finding is introduced under `scripts/`.
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
