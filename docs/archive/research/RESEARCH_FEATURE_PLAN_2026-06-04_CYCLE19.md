# OpenCut Research Feature Plan - 2026-06-04 Cycle 19

Planning-only researcher artifact for the autonomous loop. This note records a
Docker dependency-install reliability gap discovered after the Cycle 18 Docker
CI duplicate check. It intentionally does not edit source, tests, build assets,
or canonical planning ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `Dockerfile`, `pyproject.toml`, `requirements.txt`,
  `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, `PROJECT_CONTEXT.md`, and prior
  Docker research notes for Cycles 14-18.
- Verification commands:
  - targeted `rg` searches for `optional deps failed`, `fail-open`, package
    specifiers, and Docker RA coverage
  - targeted reads of the Dockerfile dependency layer and canonical Python
    extras in `pyproject.toml`
  - availability checks for `docker`, `bash`, and WSL before attempting live
    container/shell reproduction

## Researcher Queue (Cycle 19 - 2026-06-04)

- [x] `docker-dependency-install-fail-open` - Checked whether the Docker image
  dependency layer can silently succeed after optional dependency failures, and
  whether the shell-form package specifiers are robust.

## Quick Wins

- [ ] **P2 - Candidate RA-29 Docker dependency install fail-closed and quoted
  specifiers** - Why: Docker is advertised as a supported launch path, but its
  dependency layer can produce a partially provisioned image while the build
  still exits successfully. Evidence: `Dockerfile` uses shell-form `RUN pip
  install --no-cache-dir ...` with unquoted `faster-whisper>=1.1`; in a POSIX
  shell-form `RUN`, `>` is shell redirection syntax unless the requirement is
  quoted or supplied from a requirements/extras file. The same layer ends with
  `|| echo "Some optional deps failed -- continuing"`, so any failed optional
  install is converted into a successful Docker build. `pyproject.toml` already
  has canonical extras with quoted/parseable requirement strings such as
  `faster-whisper>=1.1,<2`, `opencv-python-headless>=4.13,<5`, and
  `scenedetect[opencv]>=0.6,<1`; RA-25 covers stale Docker package names, but it
  does not explicitly require fail-closed install behavior or a guard against
  shell-form requirement parsing. A small follow-up should make the Docker
  dependency layer install from canonical extras/requirements or quote every
  specifier, remove the fail-open `|| echo`, and add a static Dockerfile guard
  that rejects unquoted comparison specifiers and unconditional failure masking.

## Evidence Notes

- Docker is not available on this host, so no real `docker build` was run in
  this pass.
- `bash` is not installed on this Windows host, and `wsl.exe` is present only as
  the Windows launcher; it reports that WSL is not installed. No live POSIX shell
  reproduction was available locally.
- The risk is still source-evident because Linux Dockerfile shell-form `RUN`
  commands are interpreted by the image shell, and this file uses shell syntax
  rather than JSON exec-form invocation.
- Existing RA-25 acceptance should still remove the stale `pydub` and
  `deep-translator` names. This candidate is narrower: fail the build on
  dependency-install errors and prevent shell parsing from weakening or dropping
  requirement constraints.

## Self-Audit

- Net-new check: RA-25 covers Docker dependency-surface drift, RA-26 covers
  runtime/port posture, RA-27 covers GPU compose docs, and Cycle 18 promoted no
  new row for generic Docker CI smoke coverage. None explicitly covers
  fail-open Docker dependency installation or unquoted shell-form comparison
  specifiers.
- Lane-separation check: no implementation files were modified; this archive
  note is safe to stage independently while the implementation lane owns current
  panel i18n edits.
- Risk check: implementation should avoid only deleting `|| echo`; it should also
  source dependencies from canonical extras/requirements or quote package
  constraints so the Dockerfile cannot silently install different versions than
  the Python package metadata.
