# F205 Interrupted Coverage Reattempt Note

**Date:** 2026-05-17  
**Purpose:** preserve the wrap-up evidence for the interrupted F205 coverage-floor measurement attempt.

**Superseded:** Pass 82 completed the full F205 coverage command on 2026-05-19 and raised the CI floor to `--cov-fail-under=54`; see `F205_COVERAGE_FLOOR_SUCCESS.md`. This file remains as historical evidence for the earlier interrupted run.

## Attempted Command

```powershell
python -m pytest tests/ -q --tb=short --cov=opencut --cov-report=term-missing --cov-report=json:dist\coverage-f205.json --cov-fail-under=0 -n auto --dist worksteal
```

The command was interrupted after **2,206.6 seconds** (36m46s), before pytest completed. F205 therefore remains open.

## Partial Artifact

The ignored `dist\coverage-f205.json` file parsed as valid JSON, but it is **partial** because the pytest session did not complete.

| Field | Value |
|---|---|
| coverage.py version | 7.14.0 |
| file size | 5,490,775 bytes |
| SHA256 | `63DD45BF6C617BB05A7944911DEFF735A528F37F96CAD4CCC10F6E93CF59A6F9` |
| files reported | 670 |
| statements | 126,421 |
| covered lines | 65,890 |
| missing lines | 60,531 |
| reported coverage | 52.11950546190902% |

Because the run was interrupted, **do not use this percentage to change the CI coverage floor**.

## Cleanup

Wrap-up process inspection found one leftover pytest process:

```text
python.exe -m pytest tests sidecar/tests -q
```

It was stopped with `Stop-Process -Id 15924 -Force`, and a follow-up Python/pytest process list was empty.

After recording the hash and totals above, the stale `.coverage` and `dist\coverage-f205.json` files were removed so the next coverage run starts cleanly. The existing ignored `dist\opencut-sbom.cyclonedx.json` release artifact was left untouched.

## Decision

Leave `.github/workflows/build.yml` unchanged at `--cov-fail-under=50`. Resume F205 only on a runner where the complete CI-style coverage command can finish cleanly.
