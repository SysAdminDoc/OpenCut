# OpenCut Research Feature Plan - 2026-06-04 Cycle 8

Planning-only researcher artifact. This file captures net-new UXP runtime
permission findings discovered after Cycle 7. It does not modify app source,
tests, generated manifests, or release assets.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-18, F001-F272, and Waves L-T in `ROADMAP.md`.
- Primary evidence: Adobe Premiere UXP clipboard, manifest, and UXP API
  changelog documentation plus the live UXP manifest and panel source.

## Researcher Queue (Cycle 8 - 2026-06-04)

- [x] `uxp-clipboard-permission-refresh-2026-06-04` - checked Adobe's current
  UXP clipboard and manifest docs against the live UXP copy path. Adobe says
  clipboard access needs a valid manifest entry from manifest v5 onward, and
  the manifest docs default clipboard permission to no access. OpenCut calls
  `navigator.clipboard.writeText(body.value)` in `extension/com.opencut.uxp/main.js`
  but neither the live UXP manifest nor the dormant WebView scaffold declares
  `requiredPermissions.clipboard`. Promoted RA-19.
- [x] `uxp-alert-feature-flag-refresh-2026-06-04` - checked Adobe's UXP API
  changelog against the live UXP destructive-confirmation path. Adobe moved
  `alert`, `prompt`, and `confirm` support back to beta behind the `enableAlerts`
  feature flag, while OpenCut calls `window.confirm(...)` in the UXP panel and
  the manifest has no `featureFlags.enableAlerts`. Promoted RA-20.

## Quick Wins

- [ ] **P2 - RA-19 Declare UXP clipboard permission and centralize copy
  fallback** - Why: Adobe's current UXP clipboard docs state that a valid
  manifest entry is required from manifest version 5 onward, and the manifest
  docs default `requiredPermissions.clipboard` to no clipboard access. OpenCut's
  UXP panel writes text with `navigator.clipboard.writeText(body.value)` but the
  live manifest and WebView scaffold omit `clipboard`. Evidence: Adobe UXP
  Clipboard docs
  (`https://developer.adobe.com/premiere-pro/uxp/uxp-api/reference-js/global-members/data-transfers/clipboard`);
  Adobe UXP Manifest docs
  (`https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest/`);
  local `extension/com.opencut.uxp/main.js` clipboard call; local
  `extension/com.opencut.uxp/manifest.json` and
  `extension/com.opencut.uxp/bolt-webview/uxp.config.ts` permissions. Not a
  duplicate of RA-18 because RA-18 bans deprecated Clipboard APIs; this item
  aligns the supported current Clipboard API with the required manifest
  permission and fallback UX. Touches: `extension/com.opencut.uxp/manifest.json`,
  `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`, the UXP copy helper in
  `main.js`, UXP manifest/source guard tests, and `docs/UXP_MIGRATION.md`.
  Acceptance: if the UXP panel keeps any `navigator.clipboard.*` call, base and
  generated manifests declare the narrowest required clipboard permission
  (`readAndWrite` for copy); copy calls route through one wrapper that reports
  permission denial separately from unsupported Clipboard APIs; tests fail if a
  clipboard call exists without matching manifest permission. Verify:
  `py -3.12 -m pytest tests/test_uxp_clipboard_permission.py -q` plus a UDT copy
  smoke covering success and denied/unavailable fallback. Complexity: S-M.
- [ ] **P2 - RA-20 Replace or explicitly gate UXP `window.confirm` usage** -
  Why: Adobe's UXP API changelog says `alert`, `prompt`, and `confirm` were
  moved back to beta and require `featureFlags.enableAlerts`; OpenCut's UXP panel
  calls `window.confirm(confirmMessage)` before clearing the search index, while
  the live manifest has no `featureFlags.enableAlerts`. Evidence: Adobe UXP API
  changelog
  (`https://developer.adobe.com/premiere-pro/uxp/uxp-api/changelog3-p`);
  local `extension/com.opencut.uxp/main.js` confirmation call; local
  `extension/com.opencut.uxp/manifest.json` feature flags. Touches: the UXP
  search-index clear flow, a reusable in-panel confirmation modal or explicit
  beta-alert manifest decision, UXP static guard tests, and `docs/UXP_MIGRATION.md`.
  Acceptance: destructive UXP confirmations use an OpenCut in-panel modal with
  keyboard focus/escape handling and localized copy, or the manifest explicitly
  opts into `enableAlerts` with documented UDT evidence; tests fail on raw
  `window.alert`, `window.prompt`, or `window.confirm` in UXP code outside the
  approved wrapper. Verify:
  `py -3.12 -m pytest tests/test_uxp_confirm_guard.py -q` plus UDT smoke for the
  clear-index cancel/confirm paths. Complexity: S-M.

## Self-Audit

- Net-new check: RA-19 is permission alignment for the currently supported
  clipboard API, not the RA-18 deprecated API sentinel.
- Net-new check: RA-20 is beta-alert/confirmation runtime behavior, not WebView
  bridge permissions or external-launch consent.
- Local-first check: both items preserve local-only behavior and add no cloud
  dependency.
