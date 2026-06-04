# OpenCut Research Feature Plan - 2026-06-04 Cycle 23

Planning-only researcher artifact for the autonomous loop. This note records a
GitHub issue-seeder dry-run bug found after checking the live label state. It
intentionally does not edit source, tests, workflows, or canonical planning
ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `scripts/seed_github_issues.py`,
  `tests/test_seed_github_issues.py`, `.github/labels.yml`,
  `.github/issue-seeds.yml`, `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`,
  `PROJECT_CONTEXT.md`, and prior F097 research.
- Verification commands:
  - `py -3.12 scripts/seed_github_issues.py --labels --dry-run --json`
  - inline module probe that monkeypatched `_gh_available = lambda: False` and
    called `apply_labels(..., dry_run=True)`
  - targeted `rg` search for documented dry-run usage and `apply_labels`

## Researcher Queue (Cycle 23 - 2026-06-04)

- [x] `github-seeder-label-dry-run-without-gh` - Checked whether the documented
  `--labels --dry-run` path really avoids requiring the GitHub CLI.

## Quick Wins

- [ ] **P3 - Candidate RA-33 Let issue-label dry-runs run without `gh`** - Why:
  the issue seeder advertises dry-run as the safe inspection path, but label
  dry-runs still require the GitHub CLI to be installed. Evidence:
  `scripts/seed_github_issues.py` documents
  `python scripts/seed_github_issues.py --labels --dry-run`; `main()` calls
  `apply_labels(..., dry_run=args.dry_run)` for that path; `apply_labels()`
  checks `_gh_available()` before it checks `dry_run`, so a no-gh environment
  raises `RuntimeError: gh CLI not found on PATH -- install GitHub CLI or run
  --dry-run only` even when `dry_run=True`. A live run succeeds on this machine
  only because `gh` is installed. A small follow-up should move the gh
  availability check behind the dry-run branch and add a unit test that
  monkeypatches `_gh_available` false while asserting `--labels --dry-run --json`
  still emits planned label commands.

## Evidence Notes

- This is separate from RA-32. RA-32 covers label names used by an automated
  workflow. RA-33 covers the manual seeder's documented dry-run ergonomics.
- `tests/test_seed_github_issues.py` already verifies seed dry-runs do not
  invoke `gh`, but there is no equivalent test for label dry-runs.
- The bug matters because F097's label seed is optional and local-first; users
  should be able to inspect the intended label operations before installing or
  authenticating GitHub CLI.

## Self-Audit

- Net-new check: no current F097/RA item found for `--labels --dry-run`
  requiring `gh`.
- Lane-separation check: no implementation files were modified; this archive
  note is safe to stage independently.
- Risk check: implementation should preserve the existing fail-closed behavior
  for `--labels --apply` while making dry-run paths side-effect-free and
  dependency-light.
