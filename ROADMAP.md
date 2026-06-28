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

- [ ] P2 - Generate UXP caption styles from the shared style catalog
  Why: The backend exposes 55 built-in caption styles, and the CEP panel exposes grouped style choices, but UXP still shows six hard-coded caption styles. Caption templates are a visible paid-competitor feature, so UXP parity matters before CEP deprecation pressure increases.
  Evidence: `opencut/core/caption_styles.py:43`, `opencut/routes/engagement_content_routes.py:46`, `extension/com.opencut.uxp/index.html:292`, `extension/com.opencut.panel/client/index.html:813`, `tests/test_engagement_content.py:342`.
  Touches: `opencut/core/caption_styles.py`, `opencut/routes/engagement_content_routes.py`, `extension/com.opencut.uxp/index.html`, `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/locales/en.json`, `tests/test_uxp_caption_styles.py`, `tests/test_panel_parity.py`.
  Acceptance: UXP loads caption styles from the same backend/generated catalog as CEP; all 55 backend styles are reachable or intentionally hidden by explicit metadata; the UXP preview uses the selected style id; tests fail when CEP, UXP, and backend style ids drift.
  Complexity: M

- [ ] P2 - Promote C2PA provenance to signed embedded export credentials
  Why: OpenCut has C2PA sidecars and a metadata embed helper, but the documented sidecar path says real C2PA verifiers will not accept unsigned sidecars as trust credentials. Adobe Content Credentials and C2PA workflows make verifiable export provenance a differentiator for AI-assisted local editing.
  Evidence: `opencut/core/c2pa_sidecar.py:3`, `opencut/core/c2pa_sidecar.py:21`, `opencut/core/c2pa_embed.py:229`, `opencut/registry.py:710`, `tests/test_c2pa_sidecar.py:49`.
  Touches: `opencut/core/c2pa_sidecar.py`, `opencut/core/c2pa_embed.py`, `opencut/routes/timeline.py`, `opencut/routes/music_safety_routes.py`, `opencut/registry.py`, `tests/test_c2pa_sidecar.py`, `tests/test_c2pa_embed.py`.
  Acceptance: export provenance can use `c2patool` or an equivalent local adapter to embed a signed manifest into supported MP4/JPEG/PNG outputs when a test key/operator key is configured; unsigned sidecar fallback remains explicit and warning-bearing; verify routes distinguish embedded, signed sidecar, unsigned sidecar, missing asset, and tampered manifest cases.
  Complexity: L

- [ ] P2 - Surface plugin trust, lock, and quarantine status in panel Settings
  Why: Plugin routes now support install, marketplace, lock validation, uninstall, quarantine, restore, and delete, but users need a visible trust dashboard before running or managing third-party code.
  Evidence: `docs/PLUGIN_AUTHORING.md:23`, `opencut/routes/plugins.py:151`, `opencut/routes/plugins.py:357`, `opencut/routes/platform_infra_routes.py:369`, `opencut/core/command_palette.py:270`.
  Touches: `opencut/routes/plugins.py`, `opencut/core/plugins.py`, `opencut/core/plugin_manifest.py`, `extension/com.opencut.panel/client/index.html`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/index.html`, `extension/com.opencut.uxp/main.js`, `tests/test_plugins.py`, `tests/test_uxp_plugins.py`.
  Acceptance: Settings lists loaded, skipped, failed, unsigned/lock-missing, marketplace, and quarantined plugins with capability badges; destructive plugin actions require the existing typed confirmation route contract; panel tests cover lock-missing warnings, quarantine restore/delete, and failed plugin error display.
  Complexity: M
