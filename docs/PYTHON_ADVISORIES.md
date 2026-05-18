# Python pip-audit advisory policy

OpenCut release smoke audits both `requirements.txt` and the combined
`pyproject[all]` optional install surface. Any Python advisory not listed here
causes `python -m opencut.tools.pip_audit_extras --json --extra all` and the
`pip-audit` release-smoke step to fail.

## Allow-list

| Advisory | Package | Status | Justification |
|----------|---------|--------|---------------|
| [CVE-2024-27763](https://github.com/advisories/GHSA-86w8-vhw6-q9qq) / GHSA-86w8-vhw6-q9qq | `basicsr` | **waived** | Pulled transitively by optional local RealESRGAN/GFPGAN enhancement paths. The upstream issue is a local BasicSR SLURM environment/scontrol execution edge case, OpenCut does not set `SLURM_NODELIST` or expose BasicSR as a network service, and no fixed BasicSR release exists. Remove this waiver when BasicSR publishes a fix or OpenCut replaces the dependency. |
| [CVE-2026-1839](https://github.com/advisories/GHSA-69w3-r845-3855) / GHSA-69w3-r845-3855 | `transformers` | **waived** | The upstream fix is in Transformers 5.x, but the current `pyproject[all]` stack is constrained by WhisperX 3.8.5 requiring `huggingface-hub<1.0.0` while Transformers 5 requires `huggingface-hub>=1.3.0`. OpenCut does not use `transformers.Trainer` checkpoint resume, and the audited `[all]` resolution uses Torch 2.8. Remove this waiver when WhisperX supports the Transformers 5 dependency stack or Transformers 4 receives a backport. |

To add a new entry, update both `ALLOWED_ADVISORIES` in
`opencut/tools/pip_audit_extras.py` and this table in the same commit.

## Operational commands

```sh
python -m opencut.tools.pip_audit_extras --json --extra all
python scripts/release_smoke.py --json --only pip-audit
```

The JSON payload reports `allowed_vulnerability_count` and
`unallowed_vulnerability_count` per audit target. Release smoke passes only when
every finding is either absent or explicitly documented above.
