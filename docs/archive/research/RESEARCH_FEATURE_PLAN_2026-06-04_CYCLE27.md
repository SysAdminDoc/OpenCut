# OpenCut Research Feature Plan - 2026-06-04 Cycle 27

Planning-only researcher artifact for the autonomous loop. This note records a
release SBOM fidelity gap found after the Cycle 26 lockfile advisory audit. It
intentionally does not edit source, tests, workflows, dependency files, or
canonical planning ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `scripts/sbom.py`, `tests/test_sbom_completeness.py`,
  `tests/test_release_sbom.py`, `.github/workflows/build.yml`,
  `requirements-lock.txt`, `requirements.txt`, `pyproject.toml`, `ROADMAP.md`,
  `PROJECT_CONTEXT.md`, `RESEARCH_REPORT.md`, `TODO.md`, and Cycle 26 research.
- Verification commands:
  - `py -3.12 scripts/sbom.py --format json --output dist/research-cycle27-sbom.json`
    -> generated 13 required components, 82 optional components, and 55 model
    cards
  - PowerShell JSON inspection of that generated SBOM ->
    `components=95 required=13 idna=0 flaskVersion= flaskPurl=pkg:pypi/flask`
  - targeted `rg`/`Select-String` checks for the SBOM generator boundary,
    release workflow attachment, F204/F219 docs, and lockfile references
  - official CycloneDX reference checks:
    `https://cyclonedx.org/use-cases/identify-known-vulnerabilities/` and
    `https://cyclonedx.org/guides/sbom/external-references`

## Researcher Queue (Cycle 27 - 2026-06-04)

- [x] `release-sbom-fidelity-boundary` - Checked whether the tagged release
  SBOM artifact is a resolved dependency SBOM or only a declared-surface
  inventory.

## Quick Wins

- [ ] **P2 - Candidate RA-35 Publish a resolved release SBOM or label the
  current artifact declared-only** - Why: the release workflow uploads
  `dist/opencut-sbom.cyclonedx.json` as `OpenCut-SBOM-CycloneDX`, and roadmap
  text says F219 made the release SBOM "complete enough to audit", but the
  artifact is not a resolved package inventory. Evidence: `scripts/sbom.py`
  explicitly reads only `pyproject.toml` and `requirements.txt` and says an
  installed-package SBOM needs `cyclonedx-py` or `syft`; `build_sbom()` records
  only declared direct dependencies and optional extras; `tests/test_sbom_completeness.py`
  asserts only declared `pyproject.toml`/`requirements.txt` names plus model
  cards and graph references. A generated SBOM on this pass reported 95
  components but `idna=0`, even though `requirements-lock.txt` pins
  `idna==3.11` and Cycle 26 found a current advisory there; direct components
  such as `flask` are emitted as unversioned `pkg:pypi/flask` because the
  declarations use ranges. CycloneDX's known-vulnerability use case relies on
  identifiers such as PURL to map components to advisory databases, and the
  authoritative guide describes dependency graphs that can represent direct and
  transitive relationships plus component identity evidence including version
  and PURL. Touches: `scripts/sbom.py`, `.github/workflows/build.yml`,
  SBOM/release-smoke tests, release docs, and possibly a resolved-SBOM tool
  install step. Acceptance: release artifacts either include a resolved
  transitive package SBOM with versions for the environment being released, or
  the current declaration-only artifact is renamed/labeled with explicit
  metadata and docs so downstream scanners do not treat it as a vulnerability
  inventory. Tests fail if the release SBOM omits the chosen fidelity marker or
  if the resolved mode drops transitive packages from the lock/install
  evidence. Verify: focused SBOM pytest, release workflow smoke, generated
  CycloneDX JSON inspection for transitive packages such as `idna`, and one
  release-smoke run using the selected SBOM mode. Complexity: M.

## Evidence Notes

- This is separate from RA-34. RA-34 covers the fail-closed advisory gate for
  `requirements-lock.txt` and the current vulnerable `idna==3.11` pin; this item
  covers the release SBOM artifact's fidelity and naming contract.
- The current generator's boundary is technically honest inside `scripts/sbom.py`,
  but release-facing wording and artifact naming are broader than the artifact's
  contents. A consumer receiving `OpenCut-SBOM-CycloneDX` would not see the
  vulnerable transitive lockfile package found in Cycle 26.
- The scratch SBOM was generated under `dist/research-cycle27-sbom.json`,
  inspected, and removed before committing this planning note.

## Self-Audit

- Net-new check: no current RA item or archive note mentions resolved,
  installed-package, or declared-only SBOM fidelity. F204/F219 cover generation,
  upload, declared coverage, model-card coverage, and dependency graph presence.
- Lane-separation check: no implementation, dependency, workflow, test, or
  canonical planning files were modified by this pass.
- Risk check: implementation should choose one clear contract. A declared-only
  SBOM is still useful for policy review, but vulnerability scanning needs
  resolved versions/transitives or explicit downstream documentation that this
  artifact is not the vulnerability inventory.
