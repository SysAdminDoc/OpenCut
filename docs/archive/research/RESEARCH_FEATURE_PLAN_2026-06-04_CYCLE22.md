# OpenCut Research Feature Plan - 2026-06-04 Cycle 22

Planning-only researcher artifact for the autonomous loop. This note records a
GitHub label mismatch in the Adobe `@adobe/premierepro` tracker workflow. It
intentionally does not edit source, tests, workflows, or canonical planning
ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `.github/workflows/adobe-premierepro-versions.yml`,
  `.github/labels.yml`, `.github/issue-seeds.yml`, `.github/ISSUE_TEMPLATE/`,
  `scripts/seed_github_issues.py`, `ROADMAP.md`, `TODO.md`,
  `RESEARCH_REPORT.md`, `PROJECT_CONTEXT.md`, and prior F251/RA-16 research.
- Verification commands:
  - `gh label list --repo SysAdminDoc/OpenCut --limit 200`
  - targeted `rg` searches for `f251`, `uxp`, `tracking`, `labels`, and
    GitHub issue notification paths
  - targeted reads of workflow notification label usage and repository label
    seed definitions

## Researcher Queue (Cycle 22 - 2026-06-04)

- [x] `adobe-tracker-label-contract` - Checked whether the F251 weekly workflow
  uses labels that exist in the repo label seed and on the live GitHub repo.

## Quick Wins

- [ ] **P2 - Candidate RA-32 Adobe tracker issue-label contract guard** - Why:
  the weekly F251 tracker can detect drift but fail or duplicate notifications
  when its issue labels are not guaranteed to exist. Evidence:
  `.github/workflows/adobe-premierepro-versions.yml` searches open issues with
  `labels: 'f251'` and creates new issues with `labels: ['f251', 'uxp',
  'tracking']`. `.github/labels.yml` defines labels such as `area:uxp-plugin`,
  `area:integrations`, `roadmap:*`, and `priority:*`, but not `f251`, `uxp`, or
  `tracking`. A live `gh label list --repo SysAdminDoc/OpenCut --limit 200`
  returned only the default GitHub labels (`bug`, `documentation`, `duplicate`,
  `enhancement`, `good first issue`, `help wanted`, `invalid`, `question`, and
  `wontfix`), so the workflow's labels are absent both from the source seed and
  the live repo. A small follow-up should either add the exact tracker labels to
  `.github/labels.yml` and seed them before the workflow relies on them, or
  change the workflow to use existing seeded labels such as `area:uxp-plugin`
  and `type:chore`; add a workflow/label-shape test that fails if `github-script`
  references labels absent from `.github/labels.yml`.

## Evidence Notes

- This is separate from RA-31, which covers exit-code capture into
  `$GITHUB_OUTPUT`. RA-32 covers the issue deduplication/creation label contract
  after the notification step decides to run.
- The live repo label state may change after someone runs
  `scripts/seed_github_issues.py --labels`, but the source seed still does not
  contain `f251`, `uxp`, or `tracking`, so the workflow remains source-drifting
  even after the current label seed is applied.
- Issue templates also use seeded labels like `type:bug` and `type:feature`;
  this audit focused only on the automated F251 workflow because it creates
  issues through the GitHub API.

## Self-Audit

- Net-new check: existing F097/F251/RA-16 records cover issue seeding and Adobe
  package drift tracking, but no current item found for verifying that
  workflow-created issue labels exist in `.github/labels.yml` and live GitHub.
- Lane-separation check: no implementation files were modified; this archive
  note is safe to stage independently while the implementation lane owns current
  panel i18n edits.
- Risk check: implementation should avoid hardcoding a second label vocabulary.
  The durable fix is to make workflow label names derive from or be guarded by
  the same label seed used for issue seeding.
