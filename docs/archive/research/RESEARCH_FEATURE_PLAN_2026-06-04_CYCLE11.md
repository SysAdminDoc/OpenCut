# OpenCut Research Feature Plan - 2026-06-04 Cycle 11

Planning-only researcher artifact. This file captures one release-CI
reproducibility gap found while canonical roadmap files were dirty from the
implementation lane. It does not modify source, tests, workflows, generated
files, or canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-21, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 10.
- Primary evidence: GitHub Actions workflows, CEP panel package metadata, npm
  package engine metadata, and current GitHub Actions Node runner guidance.

## Researcher Queue (Cycle 11 - 2026-06-04)

- [x] `release-full-node-version-pin-refresh-2026-06-04` - checked Release Full
  and PR Fast Node setup against the current CEP panel test/build toolchain.
  PR Fast pins `actions/setup-node@v4` with `node-version: '22'`, but Release
  Full runs `npm ci`, `npm run audit:check`, `npm test`, `npm run build:verify`,
  and `npm run build` on the Linux leg without any `setup-node` step. The
  current Ubuntu 24.04 runner image lists Node.js 22.22.3, so this is not an
  immediate failure today, but GitHub documents `setup-node` as the consistent
  way to configure Node across runners and versions, and runner images update on
  a weekly cadence. Candidate RA-22 should pin Release Full to the same Node
  major as PR Fast before relying on panel tests/builds as release evidence.

## Quick Wins

- [ ] **P2 - Candidate RA-22 Pin Node explicitly in Release Full panel gates** -
  Why: Release Full is the release-trust workflow that runs the CEP panel npm
  install, advisory check, Vitest suite, source verifier, and Vite build, but it
  currently inherits whatever `node` binary the Linux runner image places first
  on `PATH`. PR Fast already pins Node 22, so the two CI lanes can silently test
  different Node majors. Evidence: `.github/workflows/pr-fast.yml` sets up
  Node 22 before `npm ci`; `.github/workflows/build.yml` has no `setup-node`
  step before its Linux-only panel gate; `extension/com.opencut.panel/package-lock.json`
  records Vitest's engine as `^20.0.0 || ^22.0.0 || >=24.0.0`; live
  `npm view vitest@4.1.6 engines --json` returned the same range; GitHub's
  Node CI docs say `setup-node` ensures consistent behavior across runners and
  Node versions; the runner-images Ubuntu 24.04 readme currently lists Node.js
  22.22.3 and notes weekly image updates. Touches:
  `.github/workflows/build.yml`, `tests/test_ci_workflow_split.py`, and possibly
  `PROJECT_CONTEXT.md` / `ROADMAP.md` during canonical consolidation.
  Acceptance: Release Full has an explicit `actions/setup-node` step with the
  same supported Node major used by PR Fast before any panel `npm` command; a
  workflow-shape test fails if PR Fast and Release Full diverge on the panel
  Node major or if Release Full runs panel npm commands without prior Node
  setup. Verify: `py -3.12 -m pytest tests/test_ci_workflow_split.py -q` plus
  one Release Full workflow run showing the pinned Node version in the panel
  gate. Complexity: S.

## Self-Audit

- Net-new check: F210 added Vitest panel tests and wired them into CI; this note
  is about the release workflow's unpinned Node runtime, not test coverage.
- Net-new check: F095/F131/F264 govern npm advisories and esbuild pins; this
  note is about CI runtime determinism.
- Current-state check: Ubuntu 24.04 runner images currently include a compatible
  Node 22 build, so this is a reproducibility guardrail rather than a live red
  build report.
- Lane-separation check: no implementation or canonical docs were changed while
  the build lane had dirty source/planning files.
