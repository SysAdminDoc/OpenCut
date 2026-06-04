# OpenCut Research Feature Plan - 2026-06-04 Cycle 16

Planning-only researcher artifact. This file captures one Docker onboarding
command drift found after the container runtime parity pass. It does not modify
source, tests, workflows, generated files, or canonical planning docs.

## Scope

- Lane: researcher / planning only.
- De-duplication baseline: active queue E15, external F202/F252, RA-01 through
  RA-26, F001-F272, Waves L-T, and `docs/archive/research/` through Cycle 15.
- Primary evidence: README Docker launch snippet, repo file inventory, and
  `docker-compose.yml` GPU profile configuration.

## Researcher Queue (Cycle 16 - 2026-06-04)

- [x] `docker-gpu-compose-command-refresh-2026-06-04` - checked README Docker
  launch commands against the actual compose files. README tells users to run
  `docker-compose -f docker-compose.gpu.yml up` for GPU mode, but the repo has
  no `docker-compose.gpu.yml`. The only compose file defines the GPU service
  under `profiles: [gpu]`, and its inline comment says to use
  `docker compose --profile gpu up`. Candidate RA-27 should align the README,
  compose comments, and a docs/config guard so Docker GPU onboarding does not
  point users at a missing file.

## Quick Wins

- [ ] **P3 - Candidate RA-27 Fix Docker GPU compose launch command drift** -
  Why: Docker is presented as an easy install path, but the documented GPU
  command currently references a file that is not tracked. Users following the
  README will fail before reaching the existing GPU compose profile. Evidence:
  `README.md` says `docker-compose -f docker-compose.gpu.yml up`; `rg --files`
  shows only `docker-compose.yml`; `docker-compose.yml` defines
  `opencut-server-gpu` with `profiles: [gpu]` and comments
  `docker compose --profile gpu up`. Touches: README Docker launch docs,
  `docker-compose.yml` comments if needed, and a lightweight docs/config drift
  test. Acceptance: README and compose agree on one GPU launch command, either
  by adding the missing file or using the existing profile syntax; tests fail if
  README references a compose file that is absent from the repo. Verify:
  focused docs/config pytest plus `docker compose config --profile gpu` when
  Docker Compose is available. Complexity: S.

## Self-Audit

- Net-new check: RA-26 covers Docker runtime volume/port posture; this item is
  specifically the missing GPU compose filename in README.
- Net-new check: no existing tracked research item mentions `docker-compose.gpu.yml`.
- Risk calibration: this is an onboarding break, not a runtime failure for users
  who already know to use the profile command.
- Lane-separation check: no implementation files or canonical docs were changed
  by this research pass.
