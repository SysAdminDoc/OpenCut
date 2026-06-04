# OpenCut Research Feature Plan - 2026-06-04 Cycle 18

Planning-only duplicate-check artifact for the autonomous loop. This file
records a Docker CI/release-smoke coverage audit and intentionally promotes no
new RA row because the discovered coverage shape is already captured by RA-25,
RA-26, and RA-27 acceptance criteria.

## Scope

- Lane: researcher/planning only.
- Files inspected: `.github/workflows/build.yml`, `.github/workflows/pr-fast.yml`,
  `.github/workflows/adobe-premierepro-versions.yml`, `scripts/release_smoke.py`,
  `tests/test_release_smoke.py`, `Dockerfile`, `docker-compose.yml`, `README.md`,
  `TODO.md`, `RESEARCH_REPORT.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`, and prior
  Cycle 14-16 Docker research notes.
- Verification commands:
  - `rg -n "docker|compose|container|image" .github/workflows scripts/release_smoke.py tests/test_release_smoke.py`
  - `rg --files tests | rg -i "docker|compose|container|image"`
  - targeted reads of Release Full, PR Fast, Adobe typings tracker, Dockerfile,
    compose, and `release_smoke.py` step registration.

## Researcher Queue (Cycle 18 - 2026-06-04)

- [x] `docker-ci-smoke-coverage-recheck` - Checked whether CI or release smoke
  builds the Docker image, validates compose config, or runs a container health
  smoke outside the already-filed Docker RA items.

## Findings

- Release Full builds and tests the Python package, panel, desktop packages,
  Windows installers, Linux packages, and release smoke, but it does not run
  `docker build`, `docker compose config`, or a container `/health` smoke.
- PR Fast runs `scripts/release_smoke.py --json` with selected skips, but the
  release-smoke step registry has no Docker/Dockerfile/compose step.
- No test filename under `tests/` is Docker- or Compose-specific.
- This absence is not a new roadmap gap by itself. RA-25 already asks for a
  Dockerfile dependency-surface guard plus real `docker build` evidence when
  Docker is available. RA-26 already asks for a container config drift test and
  compose health/WebSocket posture smoke. RA-27 already asks for a docs/config
  guard plus `docker compose config --profile gpu` evidence when Docker Compose
  is available.

## Quick Wins

- No new quick wins were promoted from Cycle 18.
- Do not create RA-29 from this duplicate pass unless a future audit finds a
  Docker risk outside dependency surface, runtime/port posture, GPU compose
  launch docs, or their requested CI smoke coverage.

## Self-Audit

- Net-new check: existing RA-25, RA-26, and RA-27 already cover the practical
  Docker build/config/health evidence this audit would otherwise request.
- Lane-separation check: no implementation files were modified; this archive
  note only documents the duplicate audit.
- Risk check: implementation should still add Docker/Compose validation, but it
  should land under the existing Docker RA items rather than fragmenting the
  backlog with a redundant umbrella row.
