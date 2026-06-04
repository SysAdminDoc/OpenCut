# OpenCut Research Feature Plan - 2026-06-04 Cycle 24

Planning-only duplicate-check artifact for the autonomous loop. This file
records a live GitHub issue tracker recheck and intentionally promotes no new
RA row because the empty tracker state is already covered by F182/F097 work.

## Scope

- Lane: researcher/planning only.
- Files inspected: `ROADMAP.md`, `PROJECT_CONTEXT.md`, `.github/issue-seeds.yml`,
  `scripts/seed_github_issues.py`, `tests/test_seed_github_issues.py`, and prior
  Cycle 22-23 GitHub automation notes.
- Verification commands:
  - `gh issue list --repo SysAdminDoc/OpenCut --state all --limit 100 --json number,title,state,labels,createdAt,updatedAt`
  - targeted `rg` searches for `gh issue list`, F097, F182, issue seeding, and
    public tracker references

## Researcher Queue (Cycle 24 - 2026-06-04)

- [x] `github-live-issue-tracker-empty-recheck` - Rechecked whether the public
  GitHub issue tracker has been seeded since the earlier roadmap audit.

## Findings

- `gh issue list --repo SysAdminDoc/OpenCut --state all --limit 100 --json ...`
  returned `[]`.
- This remains important for contributor discoverability, but it is not a new
  planning row: `ROADMAP.md` already lists F182 as the live gh issue seeder run,
  while F097 shipped the seeder code, label seed, templates, and seed manifest.
- Cycle 22 and Cycle 23 already filed narrower follow-ups for automation label
  contracts and `--labels --dry-run` behavior.

## Quick Wins

- No new quick wins were promoted from Cycle 24.
- Do not create RA-34 for the empty issue list unless a later audit finds a
  source-level seeding gap outside F182/F097, RA-32, or RA-33.

## Self-Audit

- Net-new check: empty live issue state is already explicitly called out in the
  older roadmap research and remains actionable through the existing F182 run
  item.
- Lane-separation check: no implementation files were modified; this archive
  note only records the duplicate live recheck.
- Risk check: implementation should avoid conflating source seeder fixes with
  the external act of applying labels/issues to GitHub. Both are needed, but the
  existing backlog already has the live-seeding action.
