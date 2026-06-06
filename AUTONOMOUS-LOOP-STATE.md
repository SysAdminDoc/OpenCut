# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-27 closed; E15 rolling i18n migration and broader Docker hardening remain open.
- Shipped this cycle: Docker README and compose comments now use the committed `gpu` profile service command, Docker run examples persist to `/home/opencut/.opencut`, the obsolete Compose `version` key is removed, and `tests/test_docker_distribution_docs.py` pins the docs contract in release-smoke.
- Verification: `py -3.12 -m pytest -q tests/test_docker_distribution_docs.py` passed (4 tests), `py -3.12 scripts/release_smoke.py --only pytest-fast --json` passed (750 tests), and `docker compose --profile gpu config` parsed without warnings.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: prepare a minimal test-environment repair plan for the broken `.venv` path, then continue E15 or RA-15/RA-17+ if the environment item is already covered.
- The next open queue items include E15 rolling i18n migration, RA-15 optional `[all]` advisory policy, RA-17+ UXP trust work, and RA-25/RA-26/RA-29/RA-30 Docker hardening.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
