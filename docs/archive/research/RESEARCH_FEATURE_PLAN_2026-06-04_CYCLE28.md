# OpenCut Research Feature Plan - 2026-06-04 Cycle 28

Planning-only researcher artifact for the active research queue. This note records a
Windows UNC/HGFS package-script gap found while refreshing the CEP panel Node
advisory surface. It intentionally does not edit source, package metadata,
tests, workflows, or canonical planning ledgers.

## Scope

- Lane: researcher/planning only.
- Files inspected: `extension/com.opencut.panel/package.json`,
  `extension/com.opencut.panel/scripts/check-advisories.mjs`,
  `extension/com.opencut.panel/scripts/check-esbuild-pin.mjs`,
  `extension/com.opencut.panel/scripts/verify-build.mjs`,
  `docs/NODE_ADVISORIES.md`, `scripts/release_smoke.py`,
  `tests/test_node_advisories.py`, `tests/test_esbuild_pin.py`,
  `tests/test_release_smoke.py`, `.github/workflows/build.yml`, `ROADMAP.md`,
  `PROJECT_CONTEXT.md`, `RESEARCH_REPORT.md`, `TODO.md`, and prior research
  notes.
- Verification commands:
  - `npm run audit:check -- --json` from
    `\\vmware-host\Shared Folders\repos\OpenCut\extension\com.opencut.panel`
    -> fails before the checker runs, looking for
    `C:\Windows\scripts\check-advisories.mjs`
  - `npm run audit:esbuild -- --json` from the same UNC path -> fails looking
    for `C:\Windows\scripts\check-esbuild-pin.mjs`
  - `npm run build:verify` from the same UNC path -> fails looking for
    `C:\Windows\scripts\verify-build.mjs`
  - `node scripts/check-advisories.mjs --json` -> succeeds with one allowed
    Vite advisory and zero unwaived findings
  - `node scripts/check-esbuild-pin.mjs --json` -> succeeds with two esbuild
    instances, both satisfying `>=0.25.0`
  - `node scripts/verify-build.mjs` -> succeeds
  - `python scripts/release_smoke.py --json --only npm-advisory --only esbuild-pin --only panel-source --skip pytest-fast`
    -> succeeds because release smoke invokes the Node scripts directly

## Researcher Queue (Cycle 28 - 2026-06-04)

- [x] `panel-npm-scripts-on-unc` - Checked whether the documented panel npm
  script entry points work from this workspace's UNC/HGFS path.

## Quick Wins

- [ ] **P3 - Candidate RA-36 Make CEP panel Node command entry points
  UNC/HGFS-safe** - Why: OpenCut already carries Windows/HGFS workarounds inside
  the advisory and esbuild scripts, and `docs/NODE_ADVISORIES.md` documents
  `npm run audit:check`, `npm run build:verify`, and related npm commands as
  the operational path. On this actual checkout path, however, `npm run` starts
  through `cmd.exe`, which prints "UNC paths are not supported. Defaulting to
  Windows directory." and then resolves relative package-script targets from
  `C:\Windows`. Evidence: `npm run audit:check -- --json`,
  `npm run audit:esbuild -- --json`, and `npm run build:verify` all fail with
  `MODULE_NOT_FOUND` for `C:\Windows\scripts\*.mjs`; the direct `node
  scripts/*.mjs` commands and the Python release-smoke Node steps succeed from
  the same UNC tree. Touches: `extension/com.opencut.panel/package.json`, panel
  helper scripts or a Windows-safe wrapper, `docs/NODE_ADVISORIES.md`,
  `tests/test_node_advisories.py`, `tests/test_esbuild_pin.py`, and
  `tests/test_release_smoke.py`. Acceptance: documented local commands either
  work from UNC/HGFS paths or explicitly route Windows users through the
  validated direct-node/PowerShell wrapper path; tests cover the package-script
  contract so future relative `node scripts/*.mjs` entries do not regress on
  UNC. Verify: `node scripts/check-advisories.mjs --json`,
  `node scripts/check-esbuild-pin.mjs --json`, `node scripts/verify-build.mjs`,
  the chosen documented npm/wrapper commands from a UNC path, and focused
  Python tests for the command-contract docs. Complexity: S.

## Evidence Notes

- This is separate from RA-22. RA-22 covers Release Full Node runtime pinning;
  this item covers local Windows/HGFS command invocation after Node is already
  available.
- This is separate from F095/F131/F264 advisory correctness. The underlying
  advisory, esbuild, and panel-source checks all pass when invoked directly;
  the failure is in the npm script launch path on UNC.
- The Linux Release Full workflow still has a valid `npm run` path because
  GitHub Actions checks out onto a normal Linux filesystem. The local Windows
  docs are the mismatch.

## Self-Audit

- Net-new check: existing docs mention HGFS path issues for Vite/Rollup and the
  checker scripts already contain PowerShell inner-command workarounds, but no
  RA item covers `npm run` itself failing before those workarounds execute.
- Lane-separation check: no source, tests, workflows, package files, or
  canonical planning files were modified by this pass.
- Risk check: implementation should avoid weakening the advisory checks. The
  direct-node path already gives the desired fail-closed behavior; the fix is
  to make the documented local command entry point reach that same code.
