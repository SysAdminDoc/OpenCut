# OpenCut Research Feature Plan - 2026-06-04 Cycle 7

Planning-only researcher artifact. This file captures net-new Cycle 7 items
discovered while the canonical `ROADMAP.md`, `TODO.md`, and
`RESEARCH_REPORT.md` already contain uncommitted Cycle 6 edits from another
lane. Once those edits land, insert this cycle after Cycle 6 in the
Research-Driven Additions section and renumber nothing else.

## Scope

- Lane: researcher / planning only.
- Source files changed: none.
- Allowed implementation surfaces named below are for the implementer lane.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-16, F001-F272, and Waves L-T in `ROADMAP.md`.

## Researcher Queue (Cycle 7 - 2026-06-04)

- [x] `uxp-manifest-schema-drift-refresh-2026-06-04` - rechecked Adobe's
  current Premiere UXP manifest documentation against the live UXP manifest and
  dormant Bolt/WebView scaffold. Adobe lists `manifestVersion` as a required
  manifest property and says Premiere supports manifest version `"5"`, while
  `extension/com.opencut.uxp/manifest.json` omits `manifestVersion` entirely and
  `extension/com.opencut.uxp/bolt-webview/uxp.config.ts` hardcodes
  `manifestVersion: 6`. Promoted RA-17 so the build lane explicitly chooses and
  validates the supported live schema version before Marketplace, enterprise, or
  UDT package claims.
- [x] `uxp-deprecation-sentinel-refresh-2026-06-04` - checked Adobe's current
  UXP API changelog against the UXP source. OpenCut does not currently use the
  deprecated `Clipboard.setContent`, `Clipboard.getContent`,
  `Clipboard.clearContent`, object-form `Clipboard.writeText`, or legacy
  `uxpvideo*` events, but no static guard prevents those names from entering the
  UXP/WebView cutover path. Promoted RA-18 as a low-cost regression sentinel.

## Quick Wins

- [ ] **P2 - RA-17 Add UXP manifest schema drift guard and explicit live
  `manifestVersion`** - Why: Adobe's current Premiere UXP manifest docs list
  `manifestVersion` as a required property and state that Premiere supports
  version `"5"`; OpenCut's live UXP manifest omits that required key, while the
  dormant Bolt/WebView scaffold declares `manifestVersion: 6`. That split can
  let UDT/package validation depend on host tolerance instead of an explicit
  OpenCut release decision. Evidence: Adobe UXP Manifest docs
  (`https://developer.adobe.com/premiere-pro/uxp/plugins/concepts/manifest/`);
  local `extension/com.opencut.uxp/manifest.json` has no `manifestVersion`; local
  `extension/com.opencut.uxp/bolt-webview/uxp.config.ts` types and emits
  `manifestVersion: 6`. Not a duplicate of RA-11/RA-13/RA-14 because those cover
  permission scope, external-launch consent, and WebView bridge split; this is
  manifest schema correctness. Touches:
  `extension/com.opencut.uxp/manifest.json`,
  `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`,
  `tests/test_uxp_manifest_schema.py`, `tests/test_uxp_webview_scaffold.py`, and
  `docs/UXP_MIGRATION.md`. Acceptance: the live manifest declares an explicit
  Premiere-supported `manifestVersion` (use `"5"` unless a documented UDT/Premiere
  smoke proves v6 is accepted for this bundle); the WebView scaffold either
  aligns or documents why generated packages intentionally use a different
  schema; a static test validates required manifest keys/types and fails on
  unsupported schema drift between live and generated manifests. Verify:
  `py -3.12 -m pytest tests/test_uxp_manifest_schema.py tests/test_uxp_webview_scaffold.py -q`
  plus a UDT package/load smoke before claiming Marketplace readiness.
  Complexity: S-M.
- [ ] **P3 - RA-18 Add a UXP API deprecation sentinel before F252 cutover** -
  Why: Adobe's UXP API changelog deprecates older Clipboard APIs and legacy
  HTMLVideoElement `uxpvideo*` event names. OpenCut's UXP source currently uses
  `navigator.clipboard.writeText` and no deprecated `Clipboard.*` or
  `uxpvideo*` names, so the right action is not a migration but a static guard
  that keeps future F252/WebView work from regressing. Evidence: Adobe UXP API
  changelog (`https://developer.adobe.com/premiere-pro/uxp/uxp-api/changelog3-p`);
  local scans of `extension/com.opencut.uxp/main.js`,
  `extension/com.opencut.uxp/bolt-webview/`, and UXP tests found no deprecated
  names. Touches: a focused UXP deprecation test, `docs/UXP_MIGRATION.md`, and
  any WebView generated bundle source path once F252 starts. Acceptance: tests
  fail if UXP/WebView code uses `Clipboard.setContent`, `Clipboard.getContent`,
  `Clipboard.clearContent`, object-form `Clipboard.writeText`, `uxpvideoload`,
  `uxpvideoplay`, `uxpvideocomplete`, or `uxpvideopause`; docs record the
  current clipboard path and any required manifest permission. Verify:
  `py -3.12 -m pytest tests/test_uxp_deprecation_sentinel.py -q`. Complexity: S.

## Self-Audit

- Net-new check: RA-17 is schema drift, not permission posture or external
  launch; RA-18 is a regression sentinel, not an implementation of a deprecated
  API migration.
- Local-first check: both items are static/package validation only and do not add
  cloud services or live telemetry.
- Lane-separation check: no source, tests, generated manifests, or release
  assets were modified by this researcher pass.
