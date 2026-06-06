# OpenCut Autonomous Loop State

Last updated: 2026-06-06

## Current Project

- Project: `\\vmware-host\Shared Folders\repos\OpenCut`
- Branch: `main`
- Cycle result: RA-46 caption sidecar schema is closed under RA-09; remaining E15 rolling work, RA-47 through RA-50, and UXP permission-split work remain open.
- Shipped this cycle: Caption exports now write versioned `.opencut-captions.json` sidecars, caption export routes return sidecar metadata, and `/timeline/srt-to-captions` can preserve metadata from sidecars while warning on SRT-only loss.
- Verification: focused caption sidecar tests passed (11 tests), `py -3.12 scripts\release_smoke.py --only pytest-fast --json` passed (797 tests), `py -3.12 scripts\release_smoke.py --only ruff --json` passed, README badges are in sync, doc-size checks passed within tolerance, roadmap source lint passed with existing appendix warnings, and `rtk git diff --check` passed.

## Next Work

- Continue this same project on the next cycle.
- Next cycle focus: continue the compact queue with RA-47 caption diff/apply, E15 batch 155, Magic Clips, or another remaining UXP permission-split item.
- The next open queue items include RA-47 caption diff/apply, E15 batch 155, RA-10 Magic Clips, and RA-11/RA-13/RA-14 UXP permission-split work.
- External F202 notarization and F252 UXP WebView cutover remain blocked on external evidence.
