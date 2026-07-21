# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-14 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

### P2

- [ ] P2 — Decompose the CEP and UXP panel monoliths behind contract tests
  Why: Controllers/styles have grown far beyond the repository's own decomposition guidance, making parity, review, and safe UI changes harder; the rendered regression gate now provides the contract needed to extract them safely.
  Evidence: `CONTRIBUTING.md`; `extension/com.opencut.panel/client/main.js` (16,497 lines); CEP `style.css` (16,278 lines); `extension/com.opencut.uxp/main.js` (8,107 lines); UXP `style.css` (4,247 lines); rendered gate in commit `6a44b951`.
  Touches: CEP/UXP state, backend client, i18n, job, timeline, component, token, layout, and bootstrap modules; Vite build; parity/source/release tests.
  Acceptance: shared responsibility boundaries are extracted without changing public IDs, host-action names, route payloads, or visual baselines; entrypoints contain bootstrap/orchestration rather than feature implementations; state/API/i18n/job/timeline modules have focused tests; panel build, parity, browser, i18n, and release-smoke gates pass.
  Complexity: XL

### P3

- [ ] P3 — Flatten the CEP command-center stylesheet into a single layer
  Why: `command-center.css` is three stacked authoring passes; the second `:root` block fully redefines the first and ~500 lines (sidebar width, radius, title sizing, duplicated media queries) are overridden wholesale later in file order. Future edits must reason through the dead cascade, and the `html.theme-light` token block only stays coherent by luck.
  Where: `extension/com.opencut.panel/client/command-center.css`.
  Acceptance: one token layer and one rule per selector; rendered CEP/UXP visual baselines and geometry/contrast tests still pass unchanged.

