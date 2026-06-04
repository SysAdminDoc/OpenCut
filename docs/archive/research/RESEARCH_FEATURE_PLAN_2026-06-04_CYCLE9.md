# OpenCut Research Feature Plan - 2026-06-04 Cycle 9

Planning-only duplicate-check artifact. This file records a follow-up UXP
runtime-permission pass that re-discovered the same clipboard and confirmation
findings already consolidated from Cycle 8. It does not modify app source,
tests, generated manifests, or release assets.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-20, F001-F272, and Waves L-T in `ROADMAP.md`.
- Primary evidence: Adobe Premiere UXP clipboard, manifest, and UXP API
  changelog documentation plus the live UXP manifest and panel source.

## Researcher Queue (Cycle 9 - 2026-06-04)

- [x] `uxp-clipboard-permission-refresh-2026-06-04` - checked Adobe's current
  UXP clipboard and manifest docs against the live UXP copy path. This matches
  Cycle 8 RA-19 exactly, so no new RA row was promoted.
- [x] `uxp-alert-feature-flag-refresh-2026-06-04` - checked Adobe's UXP API
  changelog against the live UXP destructive-confirmation path. This matches
  Cycle 8 RA-20 exactly, so no new RA row was promoted.

## Disposition

- No new quick wins were promoted from Cycle 9.
- RA-19 remains the canonical clipboard-permission/fallback item.
- RA-20 remains the canonical beta alert/confirmation handling item.
- Do not create RA-21 from this duplicate pass.

## Self-Audit

- Net-new check: the clipboard finding is already covered by RA-19.
- Net-new check: the `window.confirm` finding is already covered by RA-20.
- Local-first check: both canonical items preserve local-only behavior and add
  no cloud dependency.
