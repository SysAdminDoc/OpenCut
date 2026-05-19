# F205 Coverage Floor Success Note

**Date:** 2026-05-19  
**Purpose:** preserve the completed coverage measurement that closed F205 and raised the Release Full CI coverage floor.

## Completed Command

```powershell
python -m pytest tests/ -q --tb=short --cov=opencut --cov-report=term-missing --cov-report=json:dist\coverage-f205.json --cov-fail-under=0 -n auto --dist worksteal
```

The command completed successfully in **132.73 seconds** with **8,540 passed**, **16 skipped**, and **7 warnings**.

## Coverage Totals

The ignored `dist\coverage-f205.json` file was parsed only for evidence and was not committed.

| Field | Value |
|---|---|
| coverage.py version | 7.14.0 |
| file size | 5,736,478 bytes |
| SHA256 | `C3044F261073964E868FED338B7B09114F0115DA16F6EAF0C34005146576F318` |
| statements | 131,130 |
| covered lines | 70,935 |
| missing lines | 60,195 |
| excluded lines | 30 |
| reported coverage | 54.09517272935255% |

## Decision

Raise `.github/workflows/build.yml` from `--cov-fail-under=50` to `--cov-fail-under=54`.

The earlier 75-80% estimate in the test-coverage planning notes was not supported by the completed run. The new floor is the conservative integer floor of the measured baseline, so CI catches regression below the current complete local measurement without depending on a fractional threshold.

## Cleanup

After recording the hash and totals above, remove the ignored `.coverage` database and `dist\coverage-f205.json` so future coverage runs start cleanly. Leave any unrelated ignored release artifacts, such as `dist\opencut-sbom.cyclonedx.json`, untouched.
