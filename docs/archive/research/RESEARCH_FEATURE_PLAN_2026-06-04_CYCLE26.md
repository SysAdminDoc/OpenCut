# OpenCut Research Feature Plan - 2026-06-04 Cycle 26

Planning-only researcher artifact for the autonomous loop. This note records a
lockfile advisory regression found while checking the SBOM and dependency audit
surfaces. It intentionally does not edit source, tests, dependency pins, or
canonical planning ledgers because the build lane had concurrent ROADMAP/TODO
changes in flight.

## Scope

- Lane: researcher/planning only.
- Files inspected: `requirements-lock.txt`, `requirements.txt`,
  `pyproject.toml`, `scripts/release_smoke.py`,
  `opencut/tools/pip_audit_extras.py`, `tests/test_pip_audit_extras.py`,
  `scripts/bootstrap_check.py`, `tests/test_bootstrap_check.py`,
  `docs/PYTHON_ADVISORIES.md`, `PROJECT_CONTEXT.md`, `ROADMAP.md`, `TODO.md`,
  and prior archive research notes.
- Verification commands:
  - `py -3.12 -m pip_audit --version` -> `pip-audit 2.10.0`
  - `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off`
    -> exit 1, one unallowed advisory in `idna==3.11`
  - `py -3.12 -m opencut.tools.pip_audit_extras --json --extra all` -> exit
    1 for the existing RA-15 optional Torch/Transformers findings; its
    `requirements.txt` target resolved `idna==3.18` cleanly
  - live advisory lookup for
    `https://github.com/kjd/idna/security/advisories/GHSA-65pc-fj4g-8rjx`
    and `https://advisories.gitlab.com/pypi/idna/CVE-2026-45409/`

## Researcher Queue (Cycle 26 - 2026-06-04)

- [x] `requirements-lock-advisory-gap` - Checked whether F094's lockfile audit
  evidence is still represented in the active release-smoke dependency gate.

## Quick Wins

- [ ] **P1 - Candidate RA-34 Restore `requirements-lock.txt` advisory coverage
  and refresh vulnerable lock pins** - Why: F094 and `PROJECT_CONTEXT.md` still
  present `requirements-lock.txt` as an audited trust surface, but active
  release smoke now audits only `requirements.txt` plus `pyproject[all]`.
  Evidence: `scripts/release_smoke.py` invokes
  `opencut.tools.pip_audit_extras --json --extra all`; the wrapper's default
  `build_targets()` returns only `requirements.txt` and `pyproject[all]`; and
  `tests/test_pip_audit_extras.py` asserts that two-target default contract.
  `scripts/bootstrap_check.py` still checks only that the lockfile exists, has
  active lines, and does not pin the `opencut` self-package. A fresh
  `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off`
  reported `idna==3.11` affected by `CVE-2026-45409` /
  `GHSA-65pc-fj4g-8rjx`; upstream advisories mark versions before 3.15 as
  affected and 3.15 as patched. Touches: `requirements-lock.txt`,
  `opencut/tools/pip_audit_extras.py`, `scripts/release_smoke.py`,
  `tests/test_pip_audit_extras.py`, `tests/test_release_smoke.py`,
  `tests/test_bootstrap_check.py`, and dependency-governance docs. Acceptance:
  `requirements-lock.txt` no longer pins vulnerable `idna<3.15`; the active
  pip-audit wrapper/release-smoke path has an explicit lockfile target or
  equivalent fail-closed lockfile audit step; tests fail if the release gate
  drops the lockfile target; bootstrap wording stops implying "auditable" means
  vulnerability-clean when only shape checks ran. Verify:
  `py -3.12 -m pip_audit -r requirements-lock.txt --format json --progress-spinner off`,
  `py -3.12 -m opencut.tools.pip_audit_extras --json --extra all`
  with the new lockfile target enabled, and focused pytest for
  `tests/test_pip_audit_extras.py`, `tests/test_release_smoke.py`, and
  `tests/test_bootstrap_check.py`. Complexity: S-M.

## Evidence Notes

- This is separate from RA-15. RA-15 covers optional `pyproject[all]`
  Torch/Transformers advisories; the wrapper run for this pass still reports
  those known unallowed optional findings, while its `requirements.txt` target
  resolves `idna==3.18` with no advisory.
- This is also separate from the old F094 burn-down. F094 removed the stale
  self-package lock entry and once documented a clean lockfile audit, but the
  current recurring release-smoke command no longer exercises
  `requirements-lock.txt`.
- The advisory is time-sensitive: GitHub published
  `GHSA-65pc-fj4g-8rjx` on 2026-05-12, and GitLab's advisory page was generated
  on 2026-06-04. The local lockfile still pins `idna==3.11`.

## Self-Audit

- Net-new check: no current RA item covers the active release-smoke loss of
  `requirements-lock.txt` advisory coverage or the new `idna==3.11`
  lockfile finding. Existing F094/RA-15 entries are adjacent but do not close
  this specific recurring-audit gap.
- Lane-separation check: no implementation, dependency, source, test, workflow,
  or canonical planning files were modified by this pass.
- Risk check: implementation should avoid broad lockfile churn unless the lock
  can be regenerated deterministically; a minimal pin refresh plus a recurring
  fail-closed audit target is enough to restore the trust claim.
