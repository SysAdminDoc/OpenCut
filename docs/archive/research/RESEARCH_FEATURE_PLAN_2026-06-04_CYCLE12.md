# OpenCut Research Feature Plan - 2026-06-04 Cycle 12

Planning-only researcher artifact. This file captures one GitHub Actions
supply-chain hardening gap found while the implementation lane had dirty panel
source files. It does not modify source, tests, workflows, generated files, or
canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-22, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 11.
- Primary evidence: GitHub Actions workflow `uses:` references, workflow guard
  tests, and GitHub's current Actions secure-use guidance.

## Researcher Queue (Cycle 12 - 2026-06-04)

- [x] `github-actions-sha-pin-refresh-2026-06-04` - checked every workflow
  action reference against GitHub's current secure-use guidance. GitHub says
  full-length commit SHAs are the immutable-release option for Actions and that
  tags can be moved or deleted. OpenCut workflows currently use tag references:
  `actions/checkout@v4`, `actions/setup-python@v5`,
  `actions/setup-node@v4`, `actions/upload-artifact@v4`, and
  `actions/github-script@v7`. Existing CI-shape tests cover the PR/Release
  split and panel Node setup but do not fail on mutable action tags. Candidate
  RA-23 should pin workflow actions by full SHA, preserve Dependabot update
  comments, and add a guard test so release/signing workflows are not exposed
  to silent tag movement.

## Quick Wins

- [ ] **P2 - Candidate RA-23 Pin GitHub Actions to full-length SHAs** - Why:
  OpenCut's Release Full workflow handles artifacts, release uploads, code
  signing, SBOM upload, and issue creation, so mutable action tags are part of
  the release trust boundary. GitHub's secure-use docs state that pinning to a
  full-length commit SHA is the way to use an Action as an immutable release and
  warn that tags can move or be deleted. Evidence: `.github/workflows/build.yml`
  uses `actions/checkout@v4` and `actions/upload-artifact@v4`;
  `.github/workflows/pr-fast.yml` uses `actions/checkout@v4`,
  `actions/setup-python@v5`, and `actions/setup-node@v4`;
  `.github/workflows/adobe-premierepro-versions.yml` uses
  `actions/checkout@v4`, `actions/setup-python@v5`,
  `actions/upload-artifact@v4`, and `actions/github-script@v7`; GitHub secure
  use guidance recommends full-length SHA pins and repository/org policies for
  requiring them (`https://docs.github.com/en/actions/reference/security/secure-use`).
  Touches: `.github/workflows/*.yml`, `tests/test_ci_workflow_split.py` or a
  new workflow-security test, and release-maintenance docs. Acceptance: every
  non-local workflow `uses:` reference is pinned to a full-length SHA with an
  adjacent version comment that Dependabot can update; a guard test fails on
  unpinned mutable tags or branches; the repo documents the update workflow for
  refreshing pins. Verify: focused workflow-security pytest plus one PR Fast
  and one Release Full run using the pinned actions. Complexity: S-M.

## Self-Audit

- Net-new check: RA-22 is about selecting a deterministic Node runtime inside
  Release Full; this item is about immutable action references across all
  workflows.
- Net-new check: existing F095/F131/F264 npm advisory work does not inspect
  workflow `uses:` references.
- Risk calibration: official `actions/*` tags are widely used and trusted, so
  this is a release-trust hardening item rather than a current build failure.
- Lane-separation check: no implementation files or canonical docs were changed
  by this research pass.
