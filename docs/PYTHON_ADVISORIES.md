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

## Floor raises

| Package | Old floor | New floor | Date | Rationale |
|---------|-----------|-----------|------|-----------|
| `onnxruntime` / `onnxruntime-gpu` | `>=1.25,<2` | `>=1.26,<2` | 2026-06-13 | 1.26.0 hardens multiple out-of-bounds and overflow scenarios across ML and core ops (Attention mask OOB write, MaxPoolGrad bounds, SVM/TreeEnsemble, RNN sequence_lens) and replaces unrestricted Python `setattr` configuration with an allowlist. 15+ core modules import onnxruntime via the ai/insightface/rembg stack. |
| `torch` / `torchvision` | `>=2.6` / `>=0.21` | `>=2.10.0` / `>=0.25.0` | 2026-07-14 | [CVE-2026-24747](https://github.com/advisories/GHSA-63cw-57p8-fm3p) — `torch.load(weights_only=True)` unpickler heap corruption / RCE on a crafted checkpoint (affected `<2.10.0`, fixed 2.10.0); torchvision floored to the matching 0.25.0 release. Applies to the `depth` and `torch-stack` extras. `model_safety.safe_torch_load` additionally scans pickle weights with picklescan before load. |
| `click` | `>=8.0,<9` (lock `8.3.1`) | `>=8.3.3,<9` (lock `8.4.1`) | 2026-07-14 | [CVE-2026-7246](https://nvd.nist.gov/vuln/detail/CVE-2026-7246) / PYSEC-2026-2132 — command injection in `click.edit()` via the editor shell invocation; fixed in Click 8.3.3. Click is a core dependency of the CLI/server entrypoints. Floored above the fix in `pyproject.toml`, `requirements.txt`, and pinned to `8.4.1` in `requirements-lock.txt`. No waiver required. |
| `urllib3` | `>=2.6.3` (lock `2.7.0`) | `>=2.7.0` (lock `2.7.0`) | 2026-08-20 | [CVE-2026-44431](https://github.com/advisories/GHSA-qccp-gfcp-xxvc) / GHSA-qccp-gfcp-xxvc — sensitive headers forwarded across origins on proxied low-level redirects (affected `>=1.23,<2.7.0`), and [CVE-2026-44432](https://github.com/advisories/GHSA-mf9v-mfxr-j63j) / GHSA-mf9v-mfxr-j63j — decompression-bomb safeguards bypassed in parts of the streaming API (affected `>=2.6.0,<2.7.0`). Both High severity, published 2026-05-11, fixed only in 2.7.0, so the previous `>=2.6.3` floor chosen for CVE-2026-21441 was itself vulnerable. urllib3 enters transitively through faster-whisper's huggingface-hub fetch path, which makes outbound model and update requests. The lockfiles already pinned 2.7.0; this raises the declared floor to match. No waiver required. |
| `cryptography` | `>=48.0.1,<49` (lock `48.0.1`) | `>=50.0.0,<51` (lock `50.0.0`) | 2026-08-20 | [CVE-2026-69247](https://github.com/advisories/GHSA-g6cj-pr64-35w5) / PYSEC-2026-3552, distinguishable PKCS#7 RSA decryption failures; [CVE-2026-69249](https://github.com/advisories/GHSA-jwv3-5hgf-82ww) / PYSEC-2026-3553, duplicate-certificate chain validation blowup; and [CVE-2026-69248](https://github.com/advisories/GHSA-m2h6-j472-rp4c) / PYSEC-2026-3554, DNS constraint escape through wildcard SANs. The authenticated plugin verifier uses PyCA for Ed25519 signatures. The core floor and both dependency locks now admit only the fixed 50.x line. No waiver required. |

## Explicit Torch stack

`opencut[torch-stack]` restores the larger Torch/Transformers-backed feature
surface for users who need WhisperX, Demucs, RealESRGAN/GFPGAN, pyannote.audio,
TransNetV2, or depth models. It is not part of the default release-smoke audit
because the live resolver can still report unwaived Torch-stack and Transformers
advisories. The standalone `depth` extra uses `transformers>=5.3`; only
`torch-stack` keeps the lower `transformers>=4.30` floor because of WhisperX's
current `huggingface-hub<1.0.0` constraint. The declared Torch floor is
`torch>=2.10.0` / `torchvision>=0.25.0` so known `torch.load` deserialization
advisories from older Torch releases — including CVE-2026-24747
(GHSA-63cw-57p8-fm3p), a `weights_only=True` unpickler heap-corruption/RCE fixed
in torch 2.10.0 — are not admitted by OpenCut extras. As defense in depth,
`opencut.core.model_safety.safe_torch_load` scans pickle-format checkpoints with
picklescan (`>=1.0.3`, shipped in the `ai`/`ai-gpu`/`depth`/`torch-stack`
extras) before every load. Keep
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
