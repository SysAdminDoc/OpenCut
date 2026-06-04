# OpenCut Research Feature Plan - 2026-06-04 Cycle 17

Planning-only researcher artifact for the autonomous loop. This note records a
docs/generated-surface gap found while implementation work was active in the
main roadmap and panel files; it intentionally does not edit source, tests,
build assets, or canonical planning ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `README.md`, `PROJECT_CONTEXT.md`, `ROADMAP.md`,
  `scripts/sync_badges.py`, `scripts/release_smoke.py`, `tests/test_sync_badges.py`,
  `opencut/_generated/route_manifest.json`, and generated-route CLI output.
- Verification commands:
  - `py -3.12 -m opencut.tools.dump_route_manifest --check`
  - `py -3.12 -m opencut.tools.dump_api_aliases --check`
  - PowerShell JSON count of `route_manifest.json` blueprint properties
  - targeted `rg` searches for the stale README numbers and shipped Q4/E8/F099 coverage

## Researcher Queue (Cycle 17 - 2026-06-04)

- [x] `readme-non-badge-count-drift` - Checked whether the shipped README badge
  generator covers the remaining hand-edited route/module/blueprint counts.

## Quick Wins

- [ ] **P2 - Candidate RA-28 README non-badge generated-count gate** - Why:
  Q4/E8 closed the visible badge drift, but the README still contains several
  hand-edited count claims outside the badge block. Evidence: README line 79
  still says `v1.28.0` and `1,334 API routes`; the architecture diagram says
  `980 API routes`, `360 core modules`, and `73 route blueprints`; the project
  structure comments still say `360 processing modules` and `73 route blueprints`.
  Live verification reports `1523 routes across 107 blueprints`, and the current
  `PROJECT_CONTEXT.md` source-of-truth table reports 1,523 API routes and 599
  core modules. `scripts/sync_badges.py` only defines regexes for the Routes and
  Tests badges, and `release_smoke.py` runs only `sync_badges.py --check` for the
  `badges` step, so release smoke can pass while the higher-visibility prose and
  architecture diagram remain stale. A small follow-up should either extend the
  generated README sync/check to cover these non-badge claims or replace exact
  prose/diagram counts with source-of-truth wording.

## Evidence Notes

- `py -3.12 -m opencut.tools.dump_route_manifest --check` passed with
  `1523 routes across 107 blueprints`.
- `py -3.12 -m opencut.tools.dump_api_aliases --check` passed with
  `17 aliases, 217 canonical /api routes`.
- `route_manifest.json` currently reports `total_routes: 1523`; counting
  `blueprints` object properties yields 107.
- `Get-ChildItem opencut/core -Recurse -Filter *.py | Measure-Object` reports
  599 Python files under `opencut/core`, which matches the current
  `PROJECT_CONTEXT.md` core-module table and is far from README's 360 claim.
- Existing Q4/E8 and F099 records justify generated route truth, but the shipped
  test and release-smoke coverage is badge-scoped. This note is therefore a
  narrower follow-up, not a replacement for the completed badge generator.

## Self-Audit

- Net-new check: older Q4/E8 and F099 entries cover generated route manifests and
  README badge drift; no current note found for the remaining README prose,
  architecture-diagram, and project-structure count claims.
- Lane-separation check: no implementation files were modified; this archive note
  is safe to stage independently while the implementation lane owns current dirty
  panel and canonical roadmap files.
- Risk check: implementation should avoid simply hand-editing the README once.
  The durable fix is a check/sync path or a removal of fragile exact counts from
  non-generated prose.
