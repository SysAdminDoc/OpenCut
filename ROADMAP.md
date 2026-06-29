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

- [ ] P2 — Resolve caption font paths and CJK fallback for styled and burned captions
  Why: Caption styles define fonts and CJK wrapping is tested, but the FFmpeg drawtext path emits an empty `fontfile` and does not prove CJK glyph rendering.
  Evidence: `opencut/core/caption_styles.py:32`, `opencut/core/caption_styles.py:387`, `tests/test_caption_line_breaks.py:12`, https://github.com/OpenCut-app/OpenCut/issues/817.
  Touches: `opencut/core/caption_styles.py`, `opencut/routes/captions.py`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/main.js`, `tests/test_engagement_content.py`, `tests/test_caption_line_breaks.py`.
  Acceptance: Styled caption filters use a resolved font path or safe font-family fallback instead of `fontfile=''`; CJK sample captions render in a focused FFmpeg smoke when FFmpeg is available; panel font controls explain fallback behavior; tests cover missing preferred fonts and CJK text.
  Complexity: M
