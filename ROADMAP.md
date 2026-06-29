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
