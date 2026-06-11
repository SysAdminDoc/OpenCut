# Python pip-audit advisory policy

OpenCut release smoke audits `requirements.txt`, `requirements-lock.txt`, and
the combined `pyproject[all]` optional install surface. The `[all]` extra is the
release-audited convenience lane; optional stacks with unresolved upstream
advisory or resolver conflicts stay in explicit extras such as `torch-stack`,
`captions-whisperx`, `music`, and `enhance`. Any Python advisory not listed here
causes `python -m opencut.tools.pip_audit_extras --json --extra all` and the
`pip-audit` release-smoke step to fail.

## Allow-list

| Advisory | Package | Status | Justification |
|----------|---------|--------|---------------|
| [CVE-2024-27763](https://github.com/advisories/GHSA-86w8-vhw6-q9qq) / GHSA-86w8-vhw6-q9qq | `basicsr` | **waived for explicit `torch-stack` lane** | Pulled transitively by optional local RealESRGAN/GFPGAN enhancement paths. The upstream issue is a local BasicSR SLURM environment/scontrol execution edge case, OpenCut does not set `SLURM_NODELIST` or expose BasicSR as a network service, and no fixed BasicSR release exists. The audited `[all]` extra excludes this stack. Remove this waiver when BasicSR publishes a fix or OpenCut replaces the dependency. |
| [CVE-2026-1839](https://github.com/advisories/GHSA-69w3-r845-3855) / GHSA-69w3-r845-3855 | `transformers` | **waived for explicit `torch-stack` lane** | The upstream fix is in Transformers 5.x, but WhisperX 3.8.x requires `huggingface-hub<1.0.0` while Transformers 5 requires `huggingface-hub>=1.3.0`. OpenCut does not use `transformers.Trainer` checkpoint resume. The audited `[all]` extra excludes this stack so it can pass with zero advisories; remove this waiver when WhisperX supports the Transformers 5 dependency stack or Transformers 4 receives a backport. |
| [CVE-2026-4372](https://nvd.nist.gov/vuln/detail/CVE-2026-4372) | `transformers` | **waived for explicit `torch-stack` lane** | The upstream fix is Transformers 5.3.0+, and standalone OpenCut model-loading extras now require `transformers>=5.3`. The explicit `torch-stack` lane still carries the lower floor only because WhisperX 3.8.x requires `huggingface-hub<1.0.0`, which conflicts with the Transformers 5 dependency stack. The audited `[all]` extra excludes Torch/Transformers-backed stacks; remove this waiver when WhisperX supports the Transformers 5 resolver posture. |

To add a new entry, update both `ALLOWED_ADVISORIES` in
`opencut/tools/pip_audit_extras.py` and this table in the same commit.

## Explicit Torch stack

`opencut[torch-stack]` restores the larger Torch/Transformers-backed feature
surface for users who need WhisperX, Demucs, RealESRGAN/GFPGAN, pyannote.audio,
TransNetV2, or depth models. It is not part of the default release-smoke audit
because the live resolver can still report unwaived Torch-stack and Transformers
advisories. The standalone `depth` extra uses `transformers>=5.3`; only
`torch-stack` keeps the lower `transformers>=4.30` floor because of WhisperX's
current `huggingface-hub<1.0.0` constraint. The declared Torch floor stays at
`torch>=2.6` / `torchvision>=0.21` so known `torch.load` deserialization
advisories from older Torch releases are not admitted by OpenCut extras. Keep
those packages out of `[all]` until the dedicated `torch-stack` audit command
below returns no unallowed findings, or until each remaining finding has a
documented project-specific waiver.

## Operational commands

```sh
python -m opencut.tools.pip_audit_extras --json --extra all
python -m opencut.tools.pip_audit_extras --json --no-requirements --no-lockfile --extra torch-stack
python scripts/release_smoke.py --json --only pip-audit
```

The JSON payload reports `allowed_vulnerability_count` and
`unallowed_vulnerability_count` per audit target. Release smoke passes only when
every finding is either absent or explicitly documented above.
