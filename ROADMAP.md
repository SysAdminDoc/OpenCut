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

- [ ] P1 — Reconcile docs with local-build-only release policy
  Why: GitHub Actions workflows and Dependabot config were removed, but active docs still promise CI/workflow builds and Dependabot monitoring.
  Evidence: `git log --oneline -1 592ec577`; absent `.github/workflows` and `.github/dependabot.yml`; `SECURITY.md:47`; `CONTRIBUTING.md:11`, `95`, `108`; `docs/MACOS_NOTARIZATION.md:3`; `docs/WINDOWS_CODESIGNING.md:4`, `61`; `docs/INSTALLER_POLICY.md`.
  Touches: `SECURITY.md`, `CONTRIBUTING.md`, `DEVELOPMENT.md`, `docs/INSTALLER_POLICY.md`, `docs/MACOS_NOTARIZATION.md`, `docs/WINDOWS_CODESIGNING.md`, `docs/RELEASE_PROVENANCE.md`, focused doc guard tests.
  Acceptance: active docs describe local build/test/release commands instead of nonexistent CI workflows; `rg "GitHub Actions|Dependabot|\.github/workflows|CI"` in active docs returns only explicitly historical/archive references or local-build caveats; a focused guard test prevents reintroducing workflow/Dependabot claims.
  Complexity: M

- [ ] P1 — Guard GitHub issue seeding against stale roadmap IDs
  Why: The issue seeder can still create shipped or obsolete F097-F116 issues from `ROADMAP.md v4.3`, which would pollute the public tracker with work no longer present in the active roadmap.
  Evidence: `.github/issue-seeds.yml`; `py -3.12 scripts\seed_github_issues.py --dry-run --once`; `tests/test_seed_github_issues.py:57`; active `ROADMAP.md`.
  Touches: `.github/issue-seeds.yml`, `scripts/seed_github_issues.py`, `tests/test_seed_github_issues.py`, `CONTRIBUTING.md`.
  Acceptance: dry-run skips or fails seeds whose `roadmap_id` is absent from active `ROADMAP.md` unless explicitly marked archived/shipped; shipped seeds cannot create issues; tests cover stale IDs, shipped IDs, good-first filtering, and one valid active roadmap seed.
  Complexity: M
