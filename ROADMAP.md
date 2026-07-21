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

