# OpenCut Research Feature Plan - 2026-06-04 Cycle 25

Planning-only duplicate-check artifact for the autonomous loop. This file
records a packaging metadata recheck and intentionally promotes no new RA row
because the findings are already covered by RA-01 and RA-02.

## Scope

- Lane: researcher/planning only.
- Files inspected: `pyproject.toml`, `requirements.txt`,
  `tests/test_dependency_surface.py`, `ROADMAP.md`, `TODO.md`,
  `RESEARCH_REPORT.md`, `PROJECT_CONTEXT.md`, and prior RA-01/RA-02 references.
- Verification commands:
  - targeted reads of `pyproject.toml` project metadata and Ruff config
  - targeted reads of `requirements.txt` dependency floors
  - targeted reads of `tests/test_dependency_surface.py`
  - targeted `rg` searches for RA-01, RA-02, Ruff target-version, and
    requirements/pyproject drift

## Researcher Queue (Cycle 25 - 2026-06-04)

- [x] `python-packaging-metadata-drift-recheck` - Rechecked whether Ruff target
  version and requirements/pyproject dependency drift still need new planning
  rows.

## Findings

- `pyproject.toml` still declares `requires-python = ">=3.11"` while
  `[tool.ruff] target-version = "py39"`.
- `requirements.txt` still lists `faster-whisper>=1.0`, while
  `pyproject.toml` `[project.optional-dependencies].standard` lists
  `faster-whisper>=1.1,<2`.
- `tests/test_dependency_surface.py` guards selected security floors, but it
  does not yet enforce full requirements-vs-pyproject floor parity or Ruff target
  alignment.
- These are already active as RA-01 and RA-02 in the canonical queue, with clear
  touches and acceptance criteria.

## Quick Wins

- No new quick wins were promoted from Cycle 25.
- Do not create RA-34 for this pass; implement RA-01/RA-02 instead.

## Self-Audit

- Net-new check: the exact evidence appears in `ROADMAP.md` RA-01 and RA-02.
- Lane-separation check: no implementation files were modified; this archive
  note only records the duplicate recheck.
- Risk check: implementation should add guard assertions after fixing the
  metadata, otherwise the drift can return through either file.
