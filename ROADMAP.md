# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

_No actionable items remaining. All open work is either shipped or
blocked (see `Roadmap_Blocked.md`)._

## Research-Driven Additions

- [ ] P0 — Restore model-card generated artifact freshness
  Why: Release/model-license truth is stale today and `dump_model_cards --check` fails.
  Evidence: `py -3.12 -m opencut.tools.dump_model_cards --check`; `opencut/model_cards.py`; `opencut/tools/dump_model_cards.py`; `docs/MODELS.md`; `opencut/_generated/model_cards.json`.
  Touches: `docs/MODELS.md`, `opencut/_generated/model_cards.json`, `tests/test_model_cards.py`, `scripts/release_smoke.py` if the guard needs tightening.
  Acceptance: `py -3.12 -m opencut.tools.dump_model_cards --check` and `py -3.12 -m pytest tests/test_model_cards.py tests/test_sbom_completeness.py` pass with reviewed generated diffs.
  Complexity: S

- [ ] P0 — Restore the CEP Node advisory gate while Vite major upgrade remains blocked
  Why: `npm run audit:check -- --json` fails on unwaived `js-yaml` and Vite advisories, so release trust is red even though production CEP does not run a Vite dev server.
  Evidence: `extension/com.opencut.panel/scripts/check-advisories.mjs`; `docs/NODE_ADVISORIES.md`; `extension/com.opencut.panel/package-lock.json`; GHSA-h67p-54hq-rp68, GHSA-4w7w-66w2-5vf9, GHSA-v6wh-96g9-6wx3, GHSA-fx2h-pf6j-xcff; blocked Vite HGFS item in `Roadmap_Blocked.md`.
  Touches: `extension/com.opencut.panel/package.json`, `extension/com.opencut.panel/package-lock.json`, `extension/com.opencut.panel/scripts/check-advisories.mjs`, `docs/NODE_ADVISORIES.md`, `tests/test_node_advisories.py`, `tests/test_panel_node_entrypoints.py`.
  Acceptance: `npm run audit:check:win -- --json`, `npm run build:verify:win`, and `py -3.12 -m pytest tests/test_node_advisories.py tests/test_panel_node_entrypoints.py` pass; any remaining Vite waiver is documented as dev-server-only and tied to the blocked HGFS upgrade.
  Complexity: M

- [ ] P1 — Replace CEP native dialogs with panel-local confirmation and input flows
  Why: UXP forbids `alert`/`confirm`/`prompt` and uses inline confirmation, but CEP still relies on native dialogs for journal clear, search clear, route execution, issue description, and gist push/pull.
  Evidence: `extension/com.opencut.panel/client/main.js:9632`, `15119`, `15399`, `16565`, `16635`, `16667`; `tests/test_uxp_confirmation_guard.py`; `extension/com.opencut.uxp/main.js:3578`.
  Touches: `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/locales/en.json`, CEP panel tests.
  Acceptance: CEP source has no `window.confirm`, bare `confirm(`, `window.prompt`, or bare `prompt(` usage outside tests/fixtures; destructive flows use inline second-click confirmation, toast/status feedback, and localized copy; a CEP guard test matching UXP coverage passes.
  Complexity: M
