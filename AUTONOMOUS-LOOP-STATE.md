# OpenCut Autonomous Loop State

Last updated: 2026-06-07

## Current Project

- Project: `C:\Users\--\repos\OpenCut`
- Branch: `main`
- Cycle result: Cycle 83 scripting-console resource limit is shipped. External F202 notarization and F252 WebView cutover evidence remain open.
- Shipped this cycle: Added a 100 KiB (102,400-byte) scripting-console source limit enforced in the core sandbox API plus the `/api/scripting/execute` and `/api/workflow/scripting/execute` routes, returning HTTP 400 `CODE_TOO_LARGE` before compile/exec for oversized HTTP payloads.
- Verification: `py -3.12 -m pytest -q tests\test_dev_scripting.py tests\test_workflow_dev.py -k "oversized or scripting_route_rejects_oversized_code or workflow_execute_rejects_oversized_code"` (5 passed), `py -3.12 -m ruff check opencut/core/scripting_console.py opencut/routes/dev_scripting_routes.py opencut/routes/workflow_dev_routes.py tests/test_dev_scripting.py tests/test_workflow_dev.py`, `py -3.12 -m pytest -q tests\test_dev_scripting.py tests\test_workflow_dev.py` (282 passed), `py -3.12 -m pytest -q tests\test_roadmap_mirror.py tests\test_roadmap_lint.py` (15 passed), `py -3.12 scripts\check_doc_sizes.py --check`, `py -3.12 -m opencut.tools.lint_roadmap_sources` (known unreferenced appendix warnings only), `py -3.12 scripts\sync_badges.py --check`, `py -3.12 -m opencut.tools.dump_route_manifest --check --quiet`, `rtk git diff --check`, `py -3.12 scripts\release_smoke.py --only ruff --json`, and `py -3.12 scripts\release_smoke.py --only pytest-fast --json` (101 gate tests / 840 pytest cases) passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: return to E15 hardcoded-shell/scanner cleanup, then continue remaining release-trust and UX hardening.
- The next open queue items include E15 hardcoded-shell/scanner cleanup and the external F202/F252 evidence gates.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
