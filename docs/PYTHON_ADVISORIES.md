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

To add a new entry, update both `ALLOWED_ADVISORIES` in
`opencut/tools/pip_audit_extras.py` and this table in the same commit.

## Floor raises

| Package | Old floor | New floor | Date | Rationale |
|---------|-----------|-----------|------|-----------|
| `onnxruntime` / `onnxruntime-gpu` | `>=1.25,<2` | `>=1.26,<2` | 2026-06-13 | 1.26.0 hardens multiple out-of-bounds and overflow scenarios across ML and core ops (Attention mask OOB write, MaxPoolGrad bounds, SVM/TreeEnsemble, RNN sequence_lens) and replaces unrestricted Python `setattr` configuration with an allowlist. 15+ core modules import onnxruntime via the ai/insightface/rembg stack. |
| `torch` / `torchvision` | `>=2.6` / `>=0.21` | `>=2.10.0` / `>=0.25.0` | 2026-07-14 | [CVE-2026-24747](https://github.com/advisories/GHSA-63cw-57p8-fm3p) — `torch.load(weights_only=True)` unpickler heap corruption / RCE on a crafted checkpoint (affected `<2.10.0`, fixed 2.10.0); torchvision floored to the matching 0.25.0 release. Applies to the `depth` and `torch-stack` extras. `model_safety.safe_torch_load` additionally scans pickle weights with picklescan before load. |
| `click` | `>=8.0,<9` (lock `8.3.1`) | `>=8.3.3,<9` (lock `8.4.1`) | 2026-07-14 | [CVE-2026-7246](https://nvd.nist.gov/vuln/detail/CVE-2026-7246) / PYSEC-2026-2132 — command injection in `click.edit()` via the editor shell invocation; fixed in Click 8.3.3. Click is a core dependency of the CLI/server entrypoints. Floored above the fix in `pyproject.toml`, `requirements.txt`, and pinned to `8.4.1` in `requirements-lock.txt`. No waiver required. |
| `urllib3` | `>=2.7.0` (lock `2.7.0`) | `>=2.8.0` (lock `2.8.0`) | 2026-11-15 | [GHSA-8988-9cw3-xx77](https://github.com/advisories/GHSA-8988-9cw3-xx77) (CVE-2026-97687) lets HTTPS proxy TLS settings be ignored or overridden. [GHSA-vxq7-64xx-v4gw](https://github.com/advisories/GHSA-vxq7-64xx-v4gw) (CVE-2026-97689) buffers an unbounded chunk-size line into memory in `stream()` and `read_chunked()`, and [GHSA-gh4c-6fx4-qh6g](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g) (CVE-2026-97688) can spin chunked deflate streaming in an infinite loop. All three are fixed only in 2.8.0. The 2026-08-20 raise from `>=2.6.3` to `>=2.7.0` covered CVE-2026-44431 (GHSA-qccp-gfcp-xxvc) and CVE-2026-44432 (GHSA-mf9v-mfxr-j63j). urllib3 comes in through huggingface-hub's model download path. No waiver required. |
| `cryptography` | `>=48.0.1,<49` (lock `48.0.1`) | `>=50.0.0,<51` (lock `50.0.0`) | 2026-08-20 | [CVE-2026-69247](https://github.com/advisories/GHSA-g6cj-pr64-35w5) / PYSEC-2026-3552, distinguishable PKCS#7 RSA decryption failures; [CVE-2026-69249](https://github.com/advisories/GHSA-jwv3-5hgf-82ww) / PYSEC-2026-3553, duplicate-certificate chain validation blowup; and [CVE-2026-69248](https://github.com/advisories/GHSA-m2h6-j472-rp4c) / PYSEC-2026-3554, DNS constraint escape through wildcard SANs. The authenticated plugin verifier uses PyCA for Ed25519 signatures. The core floor and both dependency locks now admit only the fixed 50.x line. No waiver required. |
| `huggingface-hub` | transitive, lock `1.24.0` | `>=1.26,<2` (lock `1.28.0`) | 2026-08-23 | [CVE-2026-15717](https://nvd.nist.gov/vuln/detail/CVE-2026-15717) allowed absolute, UNC, drive-relative, and traversal filenames to escape a model `local_dir`. OpenCut now preflights repository filenames under both POSIX and Windows path rules before download. Direct downloads and executable model code also require full commit hashes. NeMo 2.7.3 requires Hub below 1.0, so the `nemo-asr` extra is explicitly unavailable until NVIDIA publishes a compatible lane. |
| `Pillow` | `>=12.3.0,<13` (lock `12.3.0`) | `>=13.0.0,<14` (lock `13.0.0`) | 2026-11-15 | 13.0.0 fixes 43 advisories published after 12.3.0, and five of them are critical: heap overflows while decoding JPEG ([GHSA-8g5g-8vmr-9782](https://github.com/advisories/GHSA-8g5g-8vmr-9782)) and FlashPix JPEG ([GHSA-5x8w-8jj5-q52h](https://github.com/advisories/GHSA-5x8w-8jj5-q52h)) or writing TIFF to a buffer ([GHSA-332p-7hrc-92wf](https://github.com/advisories/GHSA-332p-7hrc-92wf)), argument injection into the Ghostscript call ([GHSA-w889-685w-c8fh](https://github.com/advisories/GHSA-w889-685w-c8fh)), and multi-frame loaders reusing stale native storage ([GHSA-v9pv-5gc8-w2m6](https://github.com/advisories/GHSA-v9pv-5gc8-w2m6)). OpenCut opens user images and video frames with Pillow for thumbnails, captions, mattes and LUT previews. Pillow 13 also refuses `fromarray(mode=...)` when the mode would change the array's data type. Every call site passes uint8 arrays, and `tests/test_pillow13_mask_writers.py` pins the mask, matte and RGBA output byte for byte. No waiver required. |
| `Werkzeug` | `>=3.1.6` (release lock `3.1.8`) | `>=3.1.9` (lock `3.1.9`) | 2026-11-15 | [GHSA-g6x2-hccm-hh4m](https://github.com/advisories/GHSA-g6x2-hccm-hh4m) (CVE-2026-102598): `safe_join()` still let some Windows device names through. OpenCut's Flask server runs on Windows, so this one applies directly. No waiver required. |
| `certifi` | transitive, lock `2026.7.22` | `>=2026.10.18` (lock `2026.10.23`) | 2026-11-15 | [GHSA-6q48-6c7w-9gw2](https://github.com/advisories/GHSA-6q48-6c7w-9gw2): the bundle still trusted emSign roots whose intermediate CA keys were compromised. certifi arrives through huggingface-hub and backs every HTTPS model download, so it's now a declared core dependency with its own floor. |
| `filelock` | transitive, lock `3.32.0` | `>=4.2.1` (lock `4.2.10`) | 2026-11-15 | [GHSA-8mx5-7m5m-r77g](https://github.com/advisories/GHSA-8mx5-7m5m-r77g) (CVE-2026-112910): `WindowsFileLock` ignored `mode`, so a lock file meant to be `0o600` inherited whatever the directory ACL handed down. huggingface-hub takes these locks in the model cache. Declared in core so the floor holds. |
| `setuptools` | `>=77.0` (locks `83.0.0`) | `>=84.1.0` (locks `84.2.1`) | 2026-11-15 | [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w): a Windows path escape through `namespace_packages.txt` during legacy wheel installs. It's rated low, but the frozen build installs packages on Windows. Raised in `[build-system]` and `requirements-build.txt`, and both the release and build locks carry 84.2.1. |
| `mcp` | `>=1.26,<2` | `>=1.30.0,<2` | 2026-11-15 | Three medium advisories fixed on the 1.x line in 1.30.0. Sessions in the StreamableHTTP transport could pile up without a bound ([GHSA-84m7-p3x7-pcfv](https://github.com/advisories/GHSA-84m7-p3x7-pcfv)), server bearer auth accepted tokens issued for a different resource server ([GHSA-w4fh-qvv9-3v23](https://github.com/advisories/GHSA-w4fh-qvv9-3v23)), and the HTTP client transports followed cross-origin redirects ([GHSA-5h93-6whr-6q8j](https://github.com/advisories/GHSA-5h93-6whr-6q8j)). Applies to the `mcp` extra. |
| `rembg` | `>=2.0,<3` | `>=2.0.82,<3` | 2026-11-15 | [GHSA-qr6x-7v33-6qmg](https://github.com/advisories/GHSA-qr6x-7v33-6qmg) (CVE-2026-115924): rembg's own `/api/remove` endpoint let an unauthenticated caller plant a malicious SAM ONNX model and read local files back through the returned PNG. OpenCut calls rembg as a library and never starts that server, so the floor keeps a known-vulnerable release out of the extras rather than closing a live hole. |

## Explicit Torch stack

`opencut[torch-stack]` restores the larger Torch/Transformers-backed feature
surface for users who need Demucs, RealESRGAN/GFPGAN, pyannote.audio,
TransNetV2, or depth models. It is not part of the default release-smoke audit
because the live resolver can still report unwaived Torch-stack and Transformers
advisories. Both `depth` and `torch-stack` require `transformers>=5.10`.
WhisperX is not in either extra because its current dependency range conflicts
with OpenCut's security floors. The declared Torch floor is
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
