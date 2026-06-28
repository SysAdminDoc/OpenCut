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

- [ ] P1 — Guard GitHub issue seeding against stale roadmap IDs
  Why: The issue seeder can still create shipped or obsolete F097-F116 issues from `ROADMAP.md v4.3`, which would pollute the public tracker with work no longer present in the active roadmap.
  Evidence: `.github/issue-seeds.yml`; `py -3.12 scripts\seed_github_issues.py --dry-run --once`; `tests/test_seed_github_issues.py:57`; active `ROADMAP.md`.
  Touches: `.github/issue-seeds.yml`, `scripts/seed_github_issues.py`, `tests/test_seed_github_issues.py`, `CONTRIBUTING.md`.
  Acceptance: dry-run skips or fails seeds whose `roadmap_id` is absent from active `ROADMAP.md` unless explicitly marked archived/shipped; shipped seeds cannot create issues; tests cover stale IDs, shipped IDs, good-first filtering, and one valid active roadmap seed.
  Complexity: M

- [ ] P1 - Harden UXP DOM rendering sinks
  Why: The UXP panel is the strategic Premiere surface, and it still has many `innerHTML` render paths fed by backend/user-facing result data. Current call sites usually escape interpolated values, but there is no UXP-specific static allowlist gate to keep new markup sinks from regressing.
  Evidence: `extension/com.opencut.uxp/main.js:2263`, `extension/com.opencut.uxp/main.js:3842`, `extension/com.opencut.uxp/main.js:5041`, `extension/com.opencut.uxp/main.js:6641`, `tests/test_uxp_confirmation_guard.py`, `tests/test_i18n_hardcoded_migration.py:9138`.
  Touches: `extension/com.opencut.uxp/main.js`, `extension/com.opencut.uxp/uxp-utils.js`, `tests/test_uxp_markup_safety.py`.
  Acceptance: a static test enumerates allowed literal-only UXP `innerHTML` assignments; backend/user data must use `textContent`, DOM builders, or `UIController.escapeHtml`; fixtures with script-like clip names, paths, search excerpts, engine names, and migration rows render as inert text; the test fails on a new unreviewed sink.
  Complexity: M

- [ ] P1 - Reclassify caption translation engines by license and default safety
  Why: Caption translation code promotes SeamlessM4T and NLLB as the automatic local translation path, but those engines are not separately represented in the generated model-card/license table. That is a release-trust gap for a local tool used by commercial editors.
  Evidence: `opencut/core/captions_enhanced.py:7`, `opencut/core/captions_enhanced.py:434`, `opencut/_generated/model_cards.json`, `docs/MODELS.md`, `extension/com.opencut.panel/client/main.js:6636`.
  Touches: `opencut/core/captions_enhanced.py`, `opencut/model_cards.py`, `opencut/_generated/model_cards.json`, `docs/MODELS.md`, `opencut/routes/captions.py`, `extension/com.opencut.panel/client/main.js`, `extension/com.opencut.uxp/main.js`, `tests/test_model_cards.py`, `tests/test_captions_translate_srt.py`.
  Acceptance: SeamlessM4T and NLLB have explicit model-card records with license, privacy, install, and redistribution notes; any non-commercial or unclear engine is opt-in and visibly labeled before download; the default translation path prefers a commercial-safe local backend or returns a clear install/licensing prompt; tests assert restricted engines cannot be silently auto-selected.
  Complexity: M

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
