# OpenCut Autonomous Loop State

Last updated: 2026-06-07

## Current Project

- Project: `C:\Users\--\repos\OpenCut`
- Branch: `main`
- Cycle result: Cycle 91 async rate-limit migration is shipped. External F202 notarization and F252 WebView cutover evidence remain open.
- Shipped this cycle: Added worker-lifetime `async_job(rate_limit_key=...)` support for static and conditional async route locks, migrated model-install/GPU-heavy async routes to the shared wrapper, converted the MCP bridge to `rate_limit_slot()` for dynamic per-tool keys, and added a release-smoke regression that blocks direct route-level `rate_limit()` / `rate_limit_release()` primitive calls.
- Verification: `rg -n "\brate_limit\(|\brate_limit_release\(" opencut\routes` (no matches); `py -3.12 -m ruff check opencut\jobs.py opencut\security.py opencut\routes\audio.py opencut\routes\captions.py opencut\routes\system.py opencut\routes\video_ai.py opencut\routes\video_editing.py opencut\routes\video_specialty.py opencut\routes\mcp_bridge_routes.py tests\test_async_job_rate_limit.py tests\test_boolean_coercion.py tests\test_magic_clips.py tests\test_disk_preflight.py scripts\release_smoke.py`; `py -3.12 -m pytest -q tests\test_async_job_rate_limit.py tests\test_mcp_bridge.py tests\test_boolean_coercion.py tests\test_magic_clips.py tests\test_disk_preflight.py` (88 passed); `py -3.12 -m pytest -q tests\test_route_manifest.py tests\test_route_collisions.py` (14 passed); `py -3.12 -m pytest -q tests\test_release_smoke.py tests\test_ci_workflow_split.py` (17 passed); route-smoke touched endpoint checks (4 passed); `py -3.12 scripts\sync_badges.py --check`; `py -3.12 scripts\check_doc_sizes.py --check`; `py -3.12 -m pytest -q tests\test_roadmap_mirror.py tests\test_roadmap_lint.py` (15 passed); `git diff --check`; `py -3.12 scripts\release_smoke.py --only ruff --json`; `py -3.12 scripts\release_smoke.py --only pytest-fast --json` (104 gate tests / 853 passed).

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue E15 hardcoded-shell/scanner cleanup, CEP empty-state audit, or another release-trust/UX gap from the June 6 plan.
- The next open queue items include E15 hardcoded-shell/scanner cleanup, CEP empty-state components, UXP i18n parity, and the external F202/F252 evidence gates.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
