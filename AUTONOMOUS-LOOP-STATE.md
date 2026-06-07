# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: Cycle 65 UXP WebView permission profiles are shipped; RA-14 is closed. E15 rolling work and external F252 WebView cutover evidence remain open.
- Shipped this cycle: the dormant WebView config now exports development and release manifest profiles, keeps Vite/hot-reload domains in the development profile, removes remote WebView domains from the release profile, and uses `localOnly` messaging for locally rendered release WebView content.
- Verification: focused UXP WebView/permission/schema pytest passed (24 tests), focused Ruff passed for the UXP WebView permission test slice, UXP `node --check` passed, badge sync passed, doc-size checks passed within tolerance, roadmap source lint exited 0 with existing appendix warnings, `rtk git diff --check` passed, `py -3.12 scripts\release_smoke.py --only ruff --json` passed, and `py -3.12 scripts\release_smoke.py --only pytest-fast` passed (100 gate tests).

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue E15 batch 155 or another remaining release-trust item after the UXP permission bundle.
- The next open queue items include E15 batch 155 plus the external F202/F252 evidence gates.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
