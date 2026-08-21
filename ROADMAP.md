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

- [ ] P2 — F383 — Rendered goldens do not prove the light theme
  Category: quality
  Where: `extension/com.opencut.panel/tests/rendered/__screenshots__/chromium` (42 goldens), `tests/rendered/panel-regression.spec.mjs` `SURFACES`/`THEMES`
  Problem: The rendered suite runs all three themes for the tab-matrix cases, but the 42 committed goldens are dark-theme captures of a handful of surfaces. A light-theme regression like F369 or F370 lands with the whole suite green, and the F368 pass found the goldens never capture a UXP primary button at all, so an accent-token change is invisible to them.
  Evidence: 2026-08-21. `ls extension/com.opencut.panel/tests/rendered/__screenshots__/chromium` returns 42 files, every name ending `-dark-<width>.png` or with no theme segment. The three theme-flip regressions closed on 2026-08-21 (F368, F369, F370) were all found by `getComputedStyle` sweeps, never by this suite.
  Fix: Either add light-theme `toHaveScreenshot` companions for the surfaces that already have dark goldens, or replace the theme claim with computed-style assertions on the tokens that actually flip (the studio timeline surface, chips, progress track, focus ring). Say in the spec file which one the suite is promising.
  Acceptance: A deliberate light-theme token regression fails `npm run test:rendered`. Add the repro used to prove it to the commit message.
  Confidence: Verified
  Effort: M

- [ ] P2 — F384 — Disabled CEP controls do not say why they are disabled
  Category: ux
  Where: `extension/com.opencut.panel/client/index.html` — 109 `disabled` attributes; the ones with no `title`/`data-i18n-title` include `#polishInterviewBtn`, `#loadWaveformBtn`, and the Timeline, Deliverables and Rename action buttons
  Problem: A disabled button with no tooltip gives a user nothing to act on. The three quick actions do carry a title, but it describes what the button does rather than what is missing (`#quickCleanInterview` says "Preview one reversible cleanup pass" while it is greyed out for want of a clip). Screen readers get `aria-disabled` and silence.
  Repro: Open the panel with no clip selected. Tab to Timeline, Deliverables, or Rename; every action is disabled and nothing states the precondition. Compare with the Cut tab, which sets a status line.
  Fix: Give each disabled family one `data-i18n-title` naming the precondition ("Choose a clip first", "Connect Premiere first"), set from the same state the disable is driven by. `feature-state.js` already owns the gating and is the natural home; CEP `main.js` has 2 lines of budget left, so the wiring has to land there or in a new module.
  Acceptance: Every disabled control in Timeline, Deliverables and Rename has a title naming its precondition, and a rendered test asserts one of each.
  Confidence: Verified
  Effort: M

### P3

- [ ] P3 — F385 — MCP extended tool catalogue is checked but never exercised
  Category: quality
  Where: `opencut/tools/dump_mcp_extended_tools.py`, `tests/test_mcp_extended_tools.py`, the `--check` step in `scripts/release_smoke.py:574`
  Problem: The gate compares the committed manifest to the generated one. Nothing calls the tools it lists, so a tool can be catalogued, shipped, and broken at the same time without any gate noticing.
  Repro: `python -m opencut.tools.dump_mcp_extended_tools --check` passes on the current tree; there is no counterpart that starts the MCP server and invokes a tool.
  Fix: Add a smoke that starts the MCP surface against the test backend and calls a representative tool from each family, asserting the response shape the manifest claims.
  Acceptance: A tool whose handler is deleted fails the new smoke, not just the manifest diff.
  Confidence: Verified
  Effort: M

## Audit Findings — 2026-08-20 (deep audit pass, v1.50.0 at 5588bd4a)

Full-repo engineering/UX/visual audit. Baseline this pass: `python -m pytest tests/ -q` = 11,354 passed,
45 skipped, 0 failures; `ruff check opencut/` clean; panel vitest 225 passed. IDs continue from F365.
Scanner sweeps dismissed as false positives after tracing: all 10 gitleaks hits (dummy fixture tokens in
tests), bandit B307 eval (AST-whitelisted expression sandbox, `ast.Attribute` forbidden), all B608 SQL
hits (parameterised or literal WHERE fragments), B603/B607 subprocess (list-form argv throughout). The
one open GitHub issue (#5) is already tracked as F359 in Roadmap_Blocked.md.

### P1

### P2

### P3
