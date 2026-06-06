# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: Release provenance attestation follow-up and RA-36 shared-folder Node command hardening are closed; E15 rolling i18n migration and remaining UXP permission-split work remain open.
- Shipped this cycle: Release Full packages server release assets before upload, generates GitHub artifact attestations for the uploaded server/Linux/Windows/SBOM paths with a pinned `actions/attest` action, documents `gh attestation verify` commands, and CEP panel npm advisory/build gates have Windows-safe `:win` aliases through `panel-node-gate.ps1`.
- Verification: `actions/attest` v4 resolved to `281a49d4cbb0a72c9575a50d18f6deb515a11deb` via `git ls-remote`, action metadata confirmed `subject-path` support, focused panel/provenance/workflow tests passed, focused Ruff passed for touched tests and `scripts/release_smoke.py`, Windows-safe panel npm advisory/esbuild/build aliases passed, generated route/API-alias/feature-readiness/MCP manifests passed `--check`, badge/doc checks passed, roadmap source lint passed with existing appendix warnings, `py -3.12 scripts\release_smoke.py --only pytest-fast --json` passed (782 tests), and `rtk git diff --check` passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue the compact queue with E15, RA-04, RA-01/RA-02/RA-03, or another remaining UXP permission-split item.
- The next open queue items include E15 rolling i18n migration, RA-04 request ID error bodies, RA-03 direct typed error logging, RA-01/RA-02 dependency and lint alignment, and RA-11/RA-13/RA-14 UXP permission-split work.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
