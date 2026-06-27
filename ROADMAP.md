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

- [ ] P0 — Refresh bundled FFmpeg and provenance floor
  Why: The bundled binary is below OpenCut's own release provenance floor, so release security/trust is already red before packaging.
  Evidence: `ffmpeg\ffmpeg.exe -version` reports `8.0.1-essentials_build-www.gyan.dev`; `scripts\verify_ffmpeg_provenance.py` exits `RESULT: BELOW FLOOR`; `docs/RELEASE_PROVENANCE.md:38-46`; FFmpeg security advisories and LosslessCut issue #2943.
  Touches: `ffmpeg\ffmpeg.exe`, `ffmpeg\ffprobe.exe`, `opencut/core/ffmpeg_provenance.py`, `scripts/verify_ffmpeg_provenance.py`, `docs/RELEASE_PROVENANCE.md`, installer provenance/version tests.
  Acceptance: bundled `ffmpeg.exe -version` reports a compliant release or snapshot lane; `py -3.12 scripts\verify_ffmpeg_provenance.py` passes; focused release-provenance tests pass; docs record the updated version, lane, source URL, and CVE floor.
  Complexity: M

- [ ] P1 — Reconcile docs with local-build-only release policy
  Why: GitHub Actions workflows and Dependabot config were removed, but active docs still promise CI/workflow builds and Dependabot monitoring.
  Evidence: `git log --oneline -1 592ec577`; absent `.github/workflows` and `.github/dependabot.yml`; `SECURITY.md:47`; `CONTRIBUTING.md:11`, `95`, `108`; `docs/MACOS_NOTARIZATION.md:3`; `docs/WINDOWS_CODESIGNING.md:4`, `61`; `docs/INSTALLER_POLICY.md`.
  Touches: `SECURITY.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `docs/INSTALLER_POLICY.md`, `docs/MACOS_NOTARIZATION.md`, `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`, focused doc guard tests.
  Acceptance: active docs describe local build/test/release commands instead of nonexistent CI workflows; `rg "GitHub Actions|Dependabot|\.github/workflows|CI"` in active docs returns only explicitly historical/archive references or local-build caveats; a focused guard test prevents reintroducing workflow/Dependabot claims.
  Complexity: M

- [ ] P1 — Guard GitHub issue seeding against stale roadmap IDs
  Why: The issue seeder can still create shipped or obsolete F097-F116 issues from `ROADMAP.md v4.3`, which would pollute the public tracker with work no longer present in the active roadmap.
  Evidence: `.github/issue-seeds.yml`; `py -3.12 scripts\seed_github_issues.py --dry-run --once`; `tests/test_seed_github_issues.py:57`; active `ROADMAP.md`.
  Touches: `.github/issue-seeds.yml`, `scripts/seed_github_issues.py`, `tests/test_seed_github_issues.py`, `CONTRIBUTING.md`.
  Acceptance: dry-run skips or fails seeds whose `roadmap_id` is absent from active `ROADMAP.md` unless explicitly marked archived/shipped; shipped seeds cannot create issues; tests cover stale IDs, shipped IDs, good-first filtering, and one valid active roadmap seed.
  Complexity: M
