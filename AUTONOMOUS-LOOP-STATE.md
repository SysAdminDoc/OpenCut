# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: Cycle 63 UXP external launch permission is shipped; RA-13 is closed. RA-11/RA-14, E15 rolling work, and external F252 WebView cutover evidence remain open.
- Shipped this cycle: live UXP and dormant WebView manifests now declare HTTPS-only `launchProcess` schemes with an empty extension allowlist, social OAuth browser handoff uses HTTPS normalization and manual fallback, WebView `openURL()` rejects non-HTTPS URLs, and static guards block broad external schemes or file-launch APIs.
- Verification: focused UXP permission/schema/docs pytest passed (25 tests), focused Ruff passed for the UXP permission test slice, UXP `node --check` passed, badge sync passed, doc-size checks passed within tolerance, roadmap source lint exited 0 with existing appendix warnings, `rtk git diff --check` passed, `py -3.12 scripts\release_smoke.py --only ruff --json` passed, and `py -3.12 scripts\release_smoke.py --only pytest-fast` passed (100 gate tests).

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue UXP trust work around RA-11/RA-14, then continue the compact queue with E15 batch 155 or another remaining release-trust item.
- The next open queue items include RA-11/RA-14 UXP permission-split work and E15 batch 155.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
