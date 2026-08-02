# Roadmap

Single task tracker for known issues and planned work. Items below come from
verified engineering/product audits through 2026-07-29 (with file locations);
fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) are tracked separately in a
maintainer-local `Roadmap_Blocked.md`, which is deliberately untracked — it is
a working file, not part of a clone. This file is the tracked queue.

## Research-Driven Additions

### P0 — 2026-07-29

### P1 — 2026-07-25

### P1 — 2026-07-29

### P2 — 2026-07-25

- [ ] P2 — Complete UXP first-run and settings portability
  Why: UXP lacks the CEP panel’s recoverable onboarding, full settings import/export, support-bundle export, and issue-report path.
  Evidence: CEP onboarding/settings implementation and rendered tests; `extension/com.opencut.uxp/index.html` nine-pane Settings surface.
  Touches: UXP onboarding/settings UI, shared settings/support endpoints, locale files, rendered state and keyboard tests.
  Acceptance: A new user can connect, choose media, understand unavailable capabilities, and reach a first successful operation; settings round-trip with schema/version validation and redacted support export; malformed imports are non-destructive and actionable.
  Complexity: L

- [ ] P2 — Make update notices persistent and actionable
  Why: Both panels reduce update availability to a short-lived “visit GitHub” toast with no release notes, retry, or durable destination.
  Evidence: `extension/com.opencut.uxp/main.js:8099-8110`; CEP update-check implementation; Descript changelog; Frame.io version-update UX.
  Touches: CEP/UXP settings and status surfaces, update endpoint/client, locale strings, rendered tests.
  Acceptance: Available updates persist in Settings/About with current/available versions, release notes and a validated browser action; dismissed state is version-scoped; offline/error states retain retry guidance; no update launches automatically.
  Complexity: M

- [ ] P2 — Add forced-colors and high-contrast regression coverage
  Why: Existing light/dark contrast tests do not cover Windows forced-colors behavior, where custom surfaces and focus indicators can disappear.
  Evidence: CEP/UXP CSS lacks `forced-colors` rules; WCAG 2.2 non-text contrast and focus criteria; Playwright forced-colors emulation.
  Touches: shared panel tokens/styles, icon/status semantics, Playwright rendered matrix.
  Acceptance: Both panels remain navigable with `forced-colors: active`; text, focus, selected, disabled, error, and success states remain distinguishable without color alone; screenshots and existing semantic/keyboard checks cover the mode.
  Complexity: M

- [ ] P2 — Split the remaining panel controller hotspots
  Why: CEP and UXP controllers still centralize lifecycle, bridge state, navigation, settings, and result rendering, and were among the highest-churn files in the last 200 commits.
  Evidence: `extension/com.opencut.panel/client/main.js` (~18,200 lines), `extension/com.opencut.uxp/main.js` (~8,700 lines), recent decomposition commits.
  Touches: CEP/UXP controller modules, build/source-safety checks, unit and rendered tests.
  Acceptance: Navigation, update lifecycle, settings/diagnostics, and result-state controllers have explicit imports and teardown contracts; no duplicate global ownership remains; controller size/churn budgets are machine-checked; behavior and rendered snapshots remain unchanged.
  Complexity: L

### P2 — 2026-07-29

- [ ] P2 — Compile workflows into preflighted resumable plans
  Why: Saved workflows validate endpoint names but can run for hours before discovering invalid parameters, unavailable dependencies, incompatible media, output collisions, or a failed later step.
  Evidence: `opencut/core/workflow.py:163-180,186-345,378-388`; OpenCut queue/journal/checkpoint primitives; Descript recovery/version history; auto-editor’s inspectable automation model.
  Touches: workflow schema/compiler/executor, typed OpenAPI/readiness registry, media probe, queue/journal/checkpoints, CEP/UXP plan review and tests.
  Acceptance: Save and Run compile the same immutable plan; preflight validates typed parameters, capabilities, media/streams, space, output policy, network use, and side-effect class; users can preview and explicitly approve destructive/cloud steps; completed idempotent steps persist with artifacts/checksums and a failed or restarted workflow resumes safely without repeating them.
  Complexity: L

- [ ] P2 — Test production UI states at real breakpoint boundaries
  Why: Current rendered coverage can pass synthetic state markup, treat placeholder/value as an accessible name, and miss the exact widths where panel layouts change.
  Evidence: `extension/com.opencut.panel/tests/rendered/panel-regression.spec.mjs:12-20,602-626,1692-1701`; CEP/UXP media queries at 620, 700, 820/821, 980, 1020, and 1050.
  Touches: production state renderers, shared rendered fixtures/helpers, CEP/UXP viewport matrix, accessibility and overflow assertions.
  Acceptance: Loading/empty/offline/permission/error/destructive/success states are produced through production renderers; accessible names follow the platform computation and never pass from placeholder/value alone; each actual breakpoint is exercised at boundary minus one, boundary, and boundary plus one in both themes with overflow, focus, keyboard, and semantic assertions.
  Complexity: M

- [ ] P2 — Turn the benchmark registry into a reproducible runner
  Why: The repository defines benchmark IDs and advisory budgets but cannot execute or compare them with enough provenance to guide releases or backend choices.
  Evidence: `opencut/core/performance_benchmarks.py`, `tests/test_performance_benchmark_registry.py`, `opencut/core/eval_datasets.py`; VEBench and Netflix VMAF methodology.
  Touches: benchmark CLI/runner, pinned opt-in fixtures, backend adapters, JSON receipts/baselines, diagnostics and release-smoke integration.
  Acceptance: A documented opt-in command runs selected registered backends and records fixture hash/license, model/dependency versions, hardware, seed, warm-up, repeats, timing/memory, and quality metrics; JSON receipts compare only compatible environments with declared tolerances; unavailable backends skip truthfully; release checks may consume a same-host baseline without penalizing different hardware.
  Complexity: M
