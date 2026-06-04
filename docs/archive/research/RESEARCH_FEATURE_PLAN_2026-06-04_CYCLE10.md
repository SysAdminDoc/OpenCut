# OpenCut Research Feature Plan - 2026-06-04 Cycle 10

Planning-only researcher artifact. This file captures one net-new release-trust
gap outside the UXP packaging cluster.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-20, F001-F272, and Waves L-T in `ROADMAP.md`.
- Primary evidence: OpenCut Python classifiers and GitHub Actions workflows,
  plus current GitHub setup-python and Python 3.13 release documentation.

## Researcher Queue (Cycle 10 - 2026-06-04)

- [x] `python-313-ci-classifier-refresh-2026-06-04` - checked OpenCut's Python
  support metadata against the current CI workflows and official setup-python /
  Python 3.13 documentation. `pyproject.toml` advertises
  `Programming Language :: Python :: 3.13`, and F123 already added the
  `audioop` compatibility shim, but `.github/workflows/build.yml`,
  `.github/workflows/pr-fast.yml`, and
  `.github/workflows/adobe-premierepro-versions.yml` still run only Python 3.12.
  Promoted RA-21 so release metadata is either proven by CI or narrowed.

## Quick Wins

- [ ] **P2 - RA-21 Prove or retract the advertised Python 3.13 classifier** -
  Why: release metadata claims Python 3.13 support, but no committed GitHub
  Actions lane installs or tests OpenCut under 3.13. F123 fixed the known
  `audioop`/pydub compatibility issue, yet dependency resolution, optional
  extras, generated manifest tooling, and release-smoke scripts are still only
  exercised on 3.12. Evidence: `pyproject.toml` classifiers include
  `Programming Language :: Python :: 3.13`; `.github/workflows/build.yml`,
  `.github/workflows/pr-fast.yml`, and
  `.github/workflows/adobe-premierepro-versions.yml` all set
  `python-version` to 3.12; GitHub's setup-python docs show 3.13 as a supported
  workflow version (`https://docs.github.com/actions/language-and-framework-guides/using-python-with-github-actions`);
  Python 3.13 is a stable maintenance release in current python.org docs
  (`https://docs.python.org/3.13/whatsnew/changelog.html`). Touches:
  `.github/workflows/build.yml`, `.github/workflows/pr-fast.yml`,
  `.github/workflows/adobe-premierepro-versions.yml`,
  `tests/test_dependency_surface.py`, and release-smoke docs. Acceptance:
  either CI has a 3.13 lane covering dependency install plus the fast manifest /
  smoke checks, or the 3.13 classifier is removed until a passing lane exists;
  tests fail if classifiers advertise a Python minor that no CI workflow covers.
  Verify: focused dependency-surface test plus one GitHub Actions run with the
  3.13 matrix entry. Complexity: S-M.

## Self-Audit

- Net-new check: this is not F123's `audioop` shim; it is classifier-vs-CI proof.
- Local-first check: CI metadata only; no runtime cloud dependency is added.
- Lane-separation check: no source, tests, workflows, or generated files were
  modified by this research pass.
