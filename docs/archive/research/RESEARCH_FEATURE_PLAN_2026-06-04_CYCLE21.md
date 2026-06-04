# OpenCut Research Feature Plan - 2026-06-04 Cycle 21

Planning-only researcher artifact for the autonomous loop. This note records a
GitHub Actions notification-path gap in the Adobe `@adobe/premierepro` tracker.
It intentionally does not edit source, tests, workflows, or canonical planning
ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `.github/workflows/adobe-premierepro-versions.yml`,
  `opencut/tools/adobe_premierepro_versions.py`,
  `tests/test_adobe_premierepro_versions.py`, `scripts/release_smoke.py`,
  `ROADMAP.md`, `TODO.md`, `RESEARCH_REPORT.md`, `PROJECT_CONTEXT.md`, and prior
  RA-16/F251 research notes.
- Verification commands:
  - targeted read of the Adobe tracker workflow
  - targeted `rg` searches for `exit_code`, `GITHUB_OUTPUT`, `continue-on-error`,
    F251, and RA-16 coverage
  - targeted reads of the tracker CLI and its tests

## Researcher Queue (Cycle 21 - 2026-06-04)

- [x] `adobe-tracker-output-capture-guard` - Checked whether the weekly Adobe
  typings tracker reliably captures its nonzero drift exit code before the
  downstream notification condition evaluates.

## Quick Wins

- [ ] **P2 - Candidate RA-31 Adobe tracker exit-code capture guard** - Why:
  F251 is intentionally fail-open, but the weekly workflow's notification path
  depends on a step output that is written after a command whose drift path
  intentionally exits nonzero. Evidence: `.github/workflows/adobe-premierepro-versions.yml`
  runs `python -m opencut.tools.adobe_premierepro_versions --check --json >
  premierepro-drift.json`, then writes `echo "exit_code=$?" >> $GITHUB_OUTPUT`.
  The tracker CLI documents and tests exit code `2` for drift. The workflow step
  sets `continue-on-error: true`, but it does not locally disable shell
  fail-fast behavior around the probe command or capture the exit code with an
  explicit `set +e` / `rc=$?` block. That means the output contract can depend
  on hosted-runner shell behavior instead of being pinned in source, and the
  downstream `if: steps.probe.outputs.exit_code != '0'` has no guard for a
  missing or malformed output. A small follow-up should make the probe step
  explicitly capture the command exit code, always write the JSON/output when
  possible, and add a workflow-shape test that fails if the F251 notification
  path can regress to implicit `continue-on-error` semantics.

## Evidence Notes

- This is not a replacement for RA-16. RA-16 covers which Adobe `release-*`
  dist-tags F251 tracks. This candidate covers whether the weekly workflow
  reliably propagates the probe result into the notification condition.
- `tests/test_adobe_premierepro_versions.py` covers CLI JSON and exit-code
  behavior, but no current test inspects the GitHub Actions wrapper that turns
  those exit codes into issue notifications.
- `scripts/release_smoke.py` has its own F251 warning-step path; this finding is
  scoped to `.github/workflows/adobe-premierepro-versions.yml`.

## Self-Audit

- Net-new check: existing RA-16/F251 records mention fail-open release-smoke and
  dist-tag policy, but no current item found for the scheduled workflow's
  `$GITHUB_OUTPUT` capture contract.
- Lane-separation check: no implementation files were modified; this archive
  note is safe to stage independently while the implementation lane owns current
  panel i18n edits.
- Risk check: implementation should preserve fail-open behavior while making the
  notification decision explicit. A network/parse failure should not be confused
  with a normal Adobe package drift unless the JSON payload says that is the
  intended policy.
