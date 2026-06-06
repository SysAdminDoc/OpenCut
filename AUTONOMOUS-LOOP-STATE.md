# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-15 closed; E15 rolling i18n migration, UXP trust work, and Docker hardening remain open.
- Shipped this cycle: `opencut[all]` is now the audited convenience install lane and no longer includes Torch/Transformers-backed stacks while WhisperX 3.8.x pins Torch 2.8; those backends remain available through explicit `opencut[torch-stack]` or narrower feature extras.
- Verification: `py -3.12 -m opencut.tools.pip_audit_extras --json --extra all` passed with `pyproject[all]=0 unallowed/0 allowed`, focused dependency/release-smoke pytest passed (36 tests), Ruff passed for the touched Python files, generated route/API-alias/feature-readiness/MCP manifests passed `--check`, `py -3.12 scripts\sync_badges.py --check` passed, `py -3.12 scripts\check_doc_sizes.py --check` passed, `py -3.12 scripts\release_smoke.py --only pip-audit --json` passed, `py -3.12 scripts\release_smoke.py --only pytest-fast --json` passed (755 tests), and `rtk git diff --check` passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue the compact queue with RA-17 UXP manifest schema guard or RA-25 Docker dependency surface.
- The next open queue items include E15 rolling i18n migration, RA-17+ UXP trust work, and RA-25/RA-26/RA-29/RA-30 Docker hardening.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
