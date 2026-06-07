# OpenCut Autonomous Loop State

Last updated: 2026-06-07

## Current Project

- Project: `C:\Users\--\repos\OpenCut`
- Branch: `main`
- Cycle result: Cycle 81 `open-path` allowlist hardening is shipped. External F202 notarization and F252 WebView cutover evidence remain open.
- Shipped this cycle: Replaced `/system/open-path` direct-open extension blocklisting with an explicit safe media/document allowlist while leaving reveal mode available for validated files. Regression coverage now rejects `.bat`, `.msc`, `.cpl`, `.settingcontent-ms`, and `.url` payloads.
- Verification: `py -3.12 -m pytest -q tests\test_hardening.py -k "open_path" tests\test_new_features.py -k "TestOpenPath"` (6 passed / 110 deselected), `py -3.12 -m ruff check opencut\routes\system.py tests\test_hardening.py tests\test_new_features.py`, `py -3.12 scripts\sync_badges.py --check`, `py -3.12 scripts\check_doc_sizes.py --check`, `py -3.12 -m pytest -q tests\test_roadmap_mirror.py tests\test_roadmap_lint.py` (15 passed), `git diff --check`, `py -3.12 scripts\release_smoke.py --only ruff --json`, and `py -3.12 scripts\release_smoke.py --only pytest-fast --json` (101 gate tests / 840 pytest cases) passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue the security hardening queue with safe CLIP cache deserialization or scripting-console resource limits, then return to E15 hardcoded-shell/scanner cleanup.
- The next open queue items include safe CLIP cache deserialization, scripting-console resource limits, E15 hardcoded-shell/scanner cleanup, and the external F202/F252 evidence gates.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
