# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-35 closed; E15 rolling i18n migration, local DB hardening, and broader Docker hardening remain open.
- Shipped this cycle: release SBOM workflow paths and artifact names now state the declared-SBOM contract, and generated CycloneDX metadata carries `declared-only` fidelity, dependency-source, exclusion, and lockfile advisory-audit properties.
- Verification: `py -3.12 scripts/sbom.py --format json --output $env:TEMP\opencut-declared-sbom-cycle19.json`, `py -3.12 -m pytest -q tests/test_release_sbom.py tests/test_sbom_completeness.py`, `py -3.12 -m ruff check scripts/sbom.py tests/test_release_sbom.py tests/test_sbom_completeness.py --select E,F,I --ignore E501,E402`, and `py -3.12 scripts/release_smoke.py --only pytest-fast --json` passed (753 tests).

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: inspect local DB migration implementation shape and test fixture needs for RA-37 through RA-40.
- The next open queue items include E15 rolling i18n migration, RA-15 optional `[all]` advisory policy, RA-17+ UXP trust work, RA-25/RA-26/RA-29/RA-30 Docker hardening, and RA-37 through RA-40 local DB hardening.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
