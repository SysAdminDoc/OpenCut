# Roadmap

Single task tracker for known issues and planned work. Items below come from
the June 2026 engineering/product audit (verified findings, with file
locations); fixes already shipped are recorded in CHANGELOG.md and git
history, not here.

Blocked items (credential/license/hardware-gated) live in
[`Roadmap_Blocked.md`](Roadmap_Blocked.md).

_No actionable items remaining. All open work is either shipped or
blocked (see `Roadmap_Blocked.md`)._

## Research-Driven Additions

- [ ] P1 — Guard GitHub issue seeding against stale roadmap IDs
  Why: The issue seeder can still create shipped or obsolete F097-F116 issues from `ROADMAP.md v4.3`, which would pollute the public tracker with work no longer present in the active roadmap.
  Evidence: `.github/issue-seeds.yml`; `py -3.12 scripts\seed_github_issues.py --dry-run --once`; `tests/test_seed_github_issues.py:57`; active `ROADMAP.md`.
  Touches: `.github/issue-seeds.yml`, `scripts/seed_github_issues.py`, `tests/test_seed_github_issues.py`, `CONTRIBUTING.md`.
  Acceptance: dry-run skips or fails seeds whose `roadmap_id` is absent from active `ROADMAP.md` unless explicitly marked archived/shipped; shipped seeds cannot create issues; tests cover stale IDs, shipped IDs, good-first filtering, and one valid active roadmap seed.
  Complexity: M
