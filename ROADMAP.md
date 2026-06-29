# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

## Research-Driven Additions

- [ ] P2 - Surface plugin trust, lock, and quarantine status in panel Settings
  Why: Plugin routes now support install, marketplace, lock validation, uninstall, quarantine, restore, and delete, but users need a visible trust dashboard before running or managing third-party code.
  Evidence: `docs/PLUGIN_AUTHORING.md:23`, `opencut/routes/plugins.py:151`, `opencut/routes/plugins.py:357`, `opencut/routes/platform_infra_routes.py:369`, `opencut/core/command_palette.py:270`.
  Touches: `opencut/routes/plugins.py`, `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py`, `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/index.html`, `extension/com.opencut.uxp/main.js`, `tests/test_plugins.py`, `tests/test_uxp_plugins.py`.
  Acceptance: Settings lists loaded, skipped, failed, unsigned/lock-missing, marketplace, and quarantined plugins with capability badges; destructive plugin actions require the existing typed confirmation route contract; panel tests cover lock-missing warnings, quarantine restore/delete, and failed plugin error display.
  Complexity: M

- [ ] P1 — Make the UX feature index readiness-aware before exposing actions
  Why: `/ux/feature-index` mixes shipped and speculative routes without readiness state, so command/search users can be offered non-runnable actions.
  Evidence: `opencut/core/command_palette.py:120`, `opencut/routes/ux_intelligence_routes.py:257`, `opencut/_generated/route_manifest.json`, local check found 183 of 215 command-palette route targets absent from the generated manifest including stale `/platform/c2pa`.
  Touches: `opencut/core/command_palette.py`, `opencut/routes/ux_intelligence_routes.py`, `opencut/registry.py`, `opencut/core/feature_readiness.py`, `extension/com.opencut.panel/client/main.js`, `tests/test_ux_intelligence.py`, `tests/test_feature_readiness_generator.py`.
  Acceptance: `/ux/feature-index` and `/ux/search` return `readiness`, `route_valid`, and `runnable`; planned or missing-route entries are visibly disabled/explained in the panel; every `runnable=true` route exists in `opencut/_generated/route_manifest.json`; C2PA search routes to a live provenance/export route or is marked not runnable.
  Complexity: M

- [ ] P2 — Resolve caption font paths and CJK fallback for styled and burned captions
  Why: Caption styles define fonts and CJK wrapping is tested, but the FFmpeg drawtext path emits an empty `fontfile` and does not prove CJK glyph rendering.
  Evidence: `opencut/core/caption_styles.py:32`, `opencut/core/caption_styles.py:387`, `tests/test_caption_line_breaks.py:12`, https://github.com/OpenCut-app/OpenCut/issues/817.
  Touches: `opencut/core/caption_styles.py`, `opencut/routes/captions.py`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/main.js`, `tests/test_engagement_content.py`, `tests/test_caption_line_breaks.py`.
  Acceptance: Styled caption filters use a resolved font path or safe font-family fallback instead of `fontfile=''`; CJK sample captions render in a focused FFmpeg smoke when FFmpeg is available; panel font controls explain fallback behavior; tests cover missing preferred fonts and CJK text.
  Complexity: M

- [ ] P2 — Remove stale GitHub Actions and CI claims from active local-build docs
  Why: Active working notes and helper text still reference non-existent workflows even though this repo's current policy and `.github` state are local-build only.
  Evidence: `CLAUDE.md:200`, `CLAUDE.md:208`, `docs/UXP_MIGRATION.md:94`, `scripts/build_wpf_installer_ci.ps1:3`, `.github/` contains no workflow files.
  Touches: `CLAUDE.md`, `docs/UXP_MIGRATION.md`, `scripts/build_wpf_installer_ci.ps1`, `scripts/smoke_wpf_installer.ps1`, `scripts/smoke_inno_installer.ps1`, `tests/test_release_smoke.py`.
  Acceptance: Active docs and script descriptions say local release/smoke/build instead of GitHub Actions or CI where no workflow exists; archived research docs are exempt; a local grep/test guards against reintroducing `.github/workflows` claims outside archived files and intentionally named compatibility scripts.
  Complexity: S
