# OpenCut Research Feature Plan - 2026-06-04 Cycle 20

Planning-only researcher artifact for the autonomous loop. This note records a
Docker build-context hygiene gap discovered while auditing the container path.
It intentionally does not edit source, tests, build assets, or canonical
planning ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `.dockerignore`, `.gitignore`, `Dockerfile`, `ROADMAP.md`,
  `TODO.md`, `RESEARCH_REPORT.md`, `PROJECT_CONTEXT.md`, and prior Docker
  research notes.
- Verification commands:
  - `Get-Content .dockerignore`
  - `.gitignore` search for `*.log`, `.env`, `.env.*`, build outputs, and caches
  - `rg --files` checks for root log artifacts and Docker-ignore coverage
  - `git ls-files "*.log"` and `git status --short -- "*.log"`

## Researcher Queue (Cycle 20 - 2026-06-04)

- [x] `docker-context-secret-log-hygiene` - Checked whether `.dockerignore`
  excludes the same local-only logs/secrets that `.gitignore` excludes before
  `Dockerfile` runs `COPY . /app`.

## Quick Wins

- [ ] **P2 - Candidate RA-30 Docker build-context secret/log hygiene guard** -
  Why: Docker builds include untracked local files unless `.dockerignore` blocks
  them, and this image copies the filtered build context into `/app`.
  Evidence: `.gitignore` excludes `.env`, `.env.*`, and `*.log`; `.dockerignore`
  excludes `.git`, caches, `dist/`, `build/`, `installer/`, docs, tests,
  extension, virtualenvs, and `ffmpeg/`, but does not exclude `.env`, `.env.*`,
  or `*.log`. The current local checkout has untracked root logs
  (`build104.log`, `build35.log`, `pt.log`, `pytest-audit.log`,
  `pytest-first-failure.log`, and `pytest-full.log`) that Git ignores but a
  local Docker build would still send in the build context and copy into the
  final image via `COPY . /app`. A small follow-up should align `.dockerignore`
  with `.gitignore` for secrets/logs and add a static guard test that fails when
  Git-ignored secret/log patterns are missing from `.dockerignore`.

## Evidence Notes

- `git ls-files "*.log"` returned no tracked log files, so this is not a
  repository-tracked artifact leak.
- `Get-ChildItem -File -Filter *.log` found six local root logs totaling roughly
  104 KB. The exact local files are not the durable issue; the durable issue is
  the missing `.dockerignore` patterns that would allow similar local artifacts
  into any developer-built image.
- This is separate from RA-25/RA-29 dependency-layer fixes and RA-26/RA-27
  runtime/compose fixes because it concerns build-context contents before the
  Dockerfile executes dependency or runtime instructions.

## Self-Audit

- Net-new check: existing Docker RA items cover dependency surface, runtime
  ports/volumes, GPU launch docs, CI smoke duplication, and fail-open dependency
  installs. No current item found for `.dockerignore` alignment with Git-ignored
  logs/secrets.
- Lane-separation check: no implementation files were modified; this archive
  note is safe to stage independently while the implementation lane owns current
  panel i18n edits.
- Risk check: implementation should avoid relying only on current local file
  names. The durable guard should compare pattern classes such as `.env*`,
  `*.log`, caches, virtualenvs, and build outputs across `.gitignore` and
  `.dockerignore`.
