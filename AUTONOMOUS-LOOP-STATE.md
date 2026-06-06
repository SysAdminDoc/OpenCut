# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-38/RA-07 closed; E15 rolling i18n migration, remaining local DB hardening, and broader Docker hardening remain open.
- Shipped this cycle: oversized job-result and journal inverse/forward JSON payloads now spill to content-addressed files under `.opencut/payload_spills`, and job/journal reads return structured spill metadata instead of large SQLite rows.
- Verification: focused job/journal payload pytest passed (43 tests), Ruff passed for the touched Python files, `py -3.12 scripts\check_doc_sizes.py --check` passed, `py -3.12 scripts\release_smoke.py --only pytest-fast --json` passed (753 tests), and `rtk git diff --check` passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue local DB hardening with RA-39 maintenance diagnostics and RA-40 backup-before-wipe policy.
- The next open queue items include E15 rolling i18n migration, RA-15 optional `[all]` advisory policy, RA-17+ UXP trust work, RA-25/RA-26/RA-29/RA-30 Docker hardening, and RA-39/RA-40 local DB hardening.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
