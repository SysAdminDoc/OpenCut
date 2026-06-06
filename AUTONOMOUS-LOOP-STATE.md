# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-28 closed; E15 rolling i18n migration and broader Docker hardening remain open.
- Shipped this cycle: `scripts/check_doc_sizes.py` now checks README non-badge route, module, blueprint, panel line-count, and root test-file claims against generated manifests and the live filesystem. README generated-count claims are refreshed, `tests/test_check_doc_sizes.py` covers the README route-count regex, and release-smoke doc-size wording now covers size/count drift.
- Verification: `py -3.12 scripts/check_doc_sizes.py --check`, `py -3.12 scripts/sync_badges.py --check`, `py -3.12 -m pytest -q tests/test_check_doc_sizes.py tests/test_sync_badges.py`, `py -3.12 scripts/release_smoke.py --only doc-sizes --json`, and `py -3.12 scripts/release_smoke.py --only badges --json` passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: inspect generated SBOM fidelity and lockfile coverage after RA-34/RA-35, then continue E15 or RA-15/RA-17+ if the SBOM item is already covered.
- The next open queue items include E15 rolling i18n migration, RA-15 optional `[all]` advisory policy, RA-17+ UXP trust work, RA-25/RA-26/RA-29/RA-30 Docker hardening, and RA-35 SBOM fidelity.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
