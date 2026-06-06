# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-16, RA-31, RA-32, and RA-33 closed; E15 rolling i18n migration remains open.
- Shipped this cycle: `@adobe/premierepro` tracking now includes `release-*` dist-tags with schema v2 `tracked_dist_tags`, the snapshot was refreshed to `beta=26.3.0-beta.85` and `release-26.2=26.2.1`, the weekly tracker workflow captures probe exit codes before notification logic, tracker issue labels are seeded and shared, and label dry-runs no longer require GitHub CLI.
- Verification: `py -3.12 -m pytest -q tests/test_adobe_premierepro_versions.py tests/test_adobe_premierepro_versions_workflow.py tests/test_seed_github_issues.py` passed (28 tests), `py -3.12 scripts/release_smoke.py --only adobe-premierepro-versions --json` passed, and `py -3.12 -m opencut.tools.adobe_premierepro_versions --check --json` reported no drift. Final `git diff --check`, commit, and push still pending in this turn.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: refresh distribution competitor/package install docs, then prepare a minimal test-environment repair plan for the broken `.venv` path.
- The next open queue items include E15 rolling i18n migration, RA-15 optional `[all]` advisory policy, and RA-17+ UXP trust work.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
