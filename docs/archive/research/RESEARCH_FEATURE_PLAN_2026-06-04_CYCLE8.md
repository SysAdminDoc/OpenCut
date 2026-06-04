# OpenCut Research Feature Plan - 2026-06-04 Cycle 8

Planning-only researcher artifact. This file captures a net-new Cycle 8 item
while the build lane is actively editing the canonical roadmap/context docs and
CEP E15 source files. Reconcile this item into `ROADMAP.md`, `TODO.md`,
`RESEARCH_REPORT.md`, and `PROJECT_CONTEXT.md` once those shared docs are clean.

## Scope

- Lane: researcher / planning only.
- Source files changed: none.
- Allowed implementation surfaces named below are for the build lane.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-18, F001-F272, and Waves L-T in `ROADMAP.md`.

## Researcher Queue (Cycle 8 - 2026-06-04)

- [x] `python-313-ci-claim-refresh-2026-06-04` - checked OpenCut's Python
  support metadata against the live CI workflows and current GitHub/Python
  platform docs. `pyproject.toml` advertises Python 3.13 support, and F123
  added an audioop/pydub 3.13 shim, but `.github/workflows/build.yml`,
  `.github/workflows/pr-fast.yml`, and the weekly Adobe tracker workflow all
  pin `actions/setup-python` to 3.12. GitHub's setup-python docs support
  `python-version: '3.13'`, and Python.org marks 3.13 as a stable release.
  Promoted RA-19 so advertised Python support is backed by a real CI lane or
  an explicit classifier downgrade.

## Quick Win

- [ ] **P2 - RA-19 Add Python 3.13 CI coverage for the advertised classifier**
  - Why: OpenCut publishes `Programming Language :: Python :: 3.13` and keeps
    `requires-python = ">=3.11"`, but CI only exercises Python 3.12. The
    current 3.13 confidence comes from targeted F123 shim tests, not an install
    and release-gate run on the advertised interpreter.
  - Evidence: local `pyproject.toml` classifiers include Python 3.11, 3.12, and
    3.13; `.github/workflows/build.yml` and `.github/workflows/pr-fast.yml`
    configure `actions/setup-python` with `python-version: '3.12'` only;
    GitHub's setup-python docs show `python-version: '3.13'` as supported
    (`https://github.com/actions/setup-python`); Python.org describes Python
    3.13.0 as the stable 3.13 release
    (`https://www.python.org/downloads/release/python-3130/`).
  - Not a duplicate: RA-01 aligns Ruff's static target version; RA-02 aligns
    dependency metadata; F123 fixed the audioop/pydub compatibility slice. RA-19
    is the missing CI proof for the packaging classifier.
  - Touches: `.github/workflows/pr-fast.yml`, `.github/workflows/build.yml`,
    `tests/test_dependency_surface.py`, `tests/test_ci_workflow_split.py`, and
    optionally a lightweight `python-compat.yml` workflow if Release Full should
    remain artifact-focused on Python 3.12.
  - Acceptance: CI runs at least bootstrap/version-sync/dependency-surface
    checks and focused release-gate pytest on Python 3.11, 3.12, and 3.13, or
    the 3.13 classifier is removed until that lane exists. A static test fails
    when advertised Python classifiers drift from the CI compatibility matrix.
  - Verify: `py -3.12 -m pytest tests/test_dependency_surface.py tests/test_ci_workflow_split.py -q`
    plus one successful GitHub Actions run showing the Python 3.13 lane.
  - Complexity: S.

## Self-Audit

- Net-new check: no existing F/RA item owns "advertised Python classifier must
  have a matching CI lane." RA-01/RA-02 are metadata alignment work, and F123 is
  a runtime compatibility shim.
- Release-risk check: this can be implemented as a lightweight PR compatibility
  matrix and does not require changing release artifacts or installer builds.
- Lane-separation check: no source, tests, workflow YAML, generated manifests,
  or release assets were modified by this researcher pass.
