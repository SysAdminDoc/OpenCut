# OpenCut Research Feature Plan - 2026-06-04 Cycle 13

Planning-only researcher artifact. This file captures one GitHub Actions
least-privilege gap in the release workflow. It does not modify source, tests,
workflows, generated files, or canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-23, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 12.
- Primary evidence: GitHub Actions workflow permission blocks, release upload
  steps, existing workflow guard tests, and GitHub's current GITHUB_TOKEN
  permission guidance.

## Researcher Queue (Cycle 13 - 2026-06-04)

- [x] `release-full-github-token-permission-refresh-2026-06-04` - checked
  workflow permissions against GitHub's least-privilege GITHUB_TOKEN guidance.
  PR Fast correctly scopes workflow permissions to `contents: read`, and the
  Adobe Premiere typings tracker scopes its job to `contents: read` plus
  `issues: write`. Release Full, however, declares `permissions:
  contents: write` at workflow scope. That grants write-capable repository
  contents access to the full three-OS build/test/package/notarization/signing
  matrix even though only the tag-only `gh release upload` steps need release
  write access. Candidate RA-24 should split Release Full into read-default jobs
  plus a narrow release-upload permission boundary.

## Quick Wins

- [ ] **P2 - Candidate RA-24 Scope Release Full `GITHUB_TOKEN` permissions by
  job** - Why: Release Full runs lint, tests, package builds, code signing, and
  notarization before artifact/release upload. A workflow-level `contents:
  write` token unnecessarily broadens the blast radius of every earlier step
  and every third-party action in that workflow. GitHub's GITHUB_TOKEN docs say
  workflows should grant the least required access and that `contents: write`
  allows release creation while `contents: read` is sufficient for checkout and
  source reads. Evidence: `.github/workflows/build.yml` declares workflow-level
  `permissions: contents: write`; upload-to-release steps are tag-only and use
  `gh release upload` with `GH_TOKEN`; `.github/workflows/pr-fast.yml` uses
  `permissions: contents: read`; `.github/workflows/adobe-premierepro-versions.yml`
  scopes permissions to the one job that writes issues; GitHub docs:
  `https://docs.github.com/en/actions/using-jobs/assigning-permissions-to-jobs`.
  Touches: `.github/workflows/build.yml`, workflow-shape guard tests, and
  release-maintenance docs. Acceptance: Release Full defaults to read-only
  permissions for build/test/package jobs; only tag/manual release upload jobs
  or steps receive `contents: write`; guard tests fail if a workflow-wide
  `contents: write` returns without an explicit allowlisted reason. Verify:
  focused workflow-permission pytest plus one Release Full tag/manual dry run
  confirming uploads still work. Complexity: S-M.

## Self-Audit

- Net-new check: RA-23 is about immutable action references; this item is about
  the `GITHUB_TOKEN` privilege granted to jobs and steps.
- Net-new check: PR Fast and the Adobe tracker already use narrower permissions,
  so the gap is specific to Release Full's workflow-level write scope.
- Risk calibration: the broad token does not break builds today; it increases
  the supply-chain blast radius of Release Full.
- Lane-separation check: no implementation files or canonical docs were changed
  by this research pass.
