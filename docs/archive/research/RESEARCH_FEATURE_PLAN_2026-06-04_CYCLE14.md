# OpenCut Research Feature Plan - 2026-06-04 Cycle 14

Planning-only researcher artifact. This file captures one Docker install-surface
drift gap found during the container packaging pass. It does not modify source,
tests, workflows, generated files, or canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-24, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 13.
- Primary evidence: `Dockerfile`, `pyproject.toml`, `requirements.txt`,
  dependency-surface tests, and the shipped F094/F123 dependency decisions.

## Researcher Queue (Cycle 14 - 2026-06-04)

- [x] `docker-dependency-surface-refresh-2026-06-04` - checked the Docker image
  dependency layer against the current Python install-surface policy. F094
  removed vulnerable `deep-translator` from install surfaces, and F123 retired
  `pydub` from OpenCut extras/requirements because OpenCut does not import it
  and Python 3.13 removed stdlib `audioop`. `Dockerfile` still installs both
  `pydub` and `deep-translator` in its "standard optional dependencies" layer.
  The current `tests/test_dependency_surface.py` deep-translator guard checks
  `pyproject.toml`, `requirements.txt`, README, `routes/system.py`, and
  `core/dub_pipeline.py`, but not `Dockerfile`, so the stale container path can
  reintroduce dependencies that the release install surface intentionally
  removed. Candidate RA-25 should make the Docker dependency layer consume the
  canonical extras/requirements or add a Dockerfile drift guard.

## Quick Wins

- [ ] **P1 - Candidate RA-25 Align Docker dependency installs with tracked
  Python install surfaces** - Why: Docker is a user-facing packaging path, but
  its handwritten dependency list has drifted from the audited install surface.
  It still installs `deep-translator` after F094 removed that known no-fix
  advisory path, and it still installs `pydub` after F123 removed the unused
  dependency for Python 3.13 compatibility. Evidence: `Dockerfile` installs
  `pydub` and `deep-translator`; `requirements.txt` comments that pydub is
  retired; `pyproject.toml` extras omit pydub; `tests/test_audioop_shim.py`
  asserts OpenCut has no pydub imports and extras do not pin it;
  `tests/test_dependency_surface.py` forbids deep-translator in several install
  surfaces but omits Dockerfile. Touches: `Dockerfile`,
  `tests/test_dependency_surface.py`, and release/container docs. Acceptance:
  Docker installs from the same canonical requirements/extras used by source
  installs or has a guard-tested minimal list that excludes `deep-translator`
  and `pydub`; tests fail if Dockerfile reintroduces dependency names banned
  from the audited install surfaces. Verify: focused dependency-surface test
  plus `docker build` or a parse-only Dockerfile smoke in release CI.
  Complexity: S.

## Self-Audit

- Net-new check: RA-02 covers `requirements.txt` vs `pyproject.toml`; this item
  is specifically the Docker image install surface.
- Net-new check: F094 and F123 closed the source/requirements paths but did not
  update the Dockerfile dependency layer.
- Risk calibration: this is a concrete install-surface regression because the
  stale packages are listed in Dockerfile today.
- Lane-separation check: no implementation files or canonical docs were changed
  by this research pass.
