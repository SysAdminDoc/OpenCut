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
