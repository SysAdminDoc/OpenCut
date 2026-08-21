# Roadmap — OpenCut

Actionable work only. Historical and completed roadmap material is archived in CHANGELOG.md; blocked work is kept in Roadmap_Blocked.md.

## Research-Driven Additions

Added 2026-08-10, extended 2026-08-11 and 2026-08-20 from the research passes recorded in `RESEARCH.md`.
IDs continue the existing F-number scheme (highest prior allocation before 2026-08-11: F318; before
2026-08-20: F334).

F359-F365 came from a different source: an audit run on 2026-08-20 over the F335-F358 drain, checking
each closed item against its own acceptance text rather than against the research. Highest prior
allocation before that audit: F358. Each entry below names the item it is a follow-up to, so the
original acceptance criteria stay traceable.

### P1

- [ ] P1 — F360 — Make the release gate reproduce, or stop calling it a gate
  Why: `panel-rendered` passed inside the receipt that backs the v1.49.0 release and then failed on a re-run against that identical commit roughly two hours later, with no source change in between. F358 was queued to resolve exactly this step and its acceptance required the failure be handled "so the gate stops failing on it"; the step is failing again, so whatever `7e3d9237` fixed was not that. A release gate that passes or fails depending on when you run it teaches everyone to re-run until it goes green, which is the same as not having one.
  Evidence: `build/release-receipt.json` (commit `a90757b7`, 43 steps, `panel-rendered` ok in 339299 ms, generated 2026-08-20T23:20:05Z) against a same-commit re-run returning `release smoke failed: panel-rendered; no receipt was written`; the machine was running an unrelated 16-worker job during the failing run, so contention is a live hypothesis and not yet excluded; F358 note already recorded 2-3% golden drift on this machine with no CSS change
  Touches: `extension/com.opencut.panel/tests/rendered/`, `extension/com.opencut.panel/playwright.config.mjs`, `scripts/release_smoke.py`
  Acceptance: The failure is reproduced deliberately and attributed to one of load-driven flake, environmental golden drift, or a real regression, with the expected/actual/diff artifacts inspected before anything is regenerated; whichever it is gets fixed at its cause (bounded comparison threshold, serialised or load-aware execution, or a code fix); the same commit then passes the step on three consecutive runs including one under deliberate CPU load. Do not update goldens before the diff artifacts have been read.
  Complexity: M

### P2

- [ ] P2 — F362 — Clear or justify the CEP routes failing the UXP route-level gate
  Why: F337 delivered what it promised, dated verdicts and typed evidence for both high-risk CEP-only functions, but it opened by citing a failing route-level gate and that gate is still failing. `route_coverage.gate.passes` is false on a list of CEP routes with no UXP path and no recorded justification. The release does not block on it, which is why it has stayed red; that also means nothing will ever force it green, and the CEP horizon does not move because the gate is ignored.
  Evidence: `opencut/_generated/uxp_migration_dashboard.json` → `route_coverage.gate.passes: false`, one error listing CEP routes without a UXP path or justified exclusion (`/analyze/virality`, `/assistant/suggest`, `/audio/beats`, `/audio/duck-video`, and others); `opencut/core/cep_uxp_parity.py:307,392` (the two cep_only entries, audited 2026-08-20 and correctly pinned)
  Touches: `opencut/core/cep_uxp_parity.py`, `extension/com.opencut.uxp/main.js`, the `uxp_migration_dashboard` generator, `tests/`
  Acceptance: Every route in the gate error either gains a UXP path or carries a written exclusion with a reason, on the same dated-verdict pattern F337 established for functions; the gate passes, or its remaining failures are reduced to a named set that a separate item owns; whichever it is, the gate becomes something whose colour means something.
  Complexity: L

### P3
