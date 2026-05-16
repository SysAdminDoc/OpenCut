# Windows ARM64 packaging evaluation (F101)

**Status**: evaluation, not yet shipped.
**Owner**: release engineering.
**Decision summary**: ship Windows x64 reliability first. ARM64 packaging
is feasible end-to-end today but the cost/value ratio doesn't clear the
release-trust bar — revisit once F098/F099/F112 are stable across two
releases.

## Why this exists

OpenCut already ships an Inno-Setup installer + a PyInstaller bundle
for Windows x64 (`installer/`, `OpenCut.iss`). Issue signal on
competitor projects (LosslessCut #N — see ROADMAP `[V43-S57]`) suggests
ARM64 demand exists, particularly on Surface Pro X / Snapdragon X Elite
laptops. This document records the current technical posture, the gaps,
and the trigger that moves ARM64 from *Next* to *Now*.

## Component-by-component compatibility (May 2026)

| Layer | x64 today | ARM64 status | Notes |
|-------|-----------|--------------|-------|
| Python interpreter | 3.10 / 3.11 / 3.12 official wheels | 3.11+ official ARM64 builds since CPython 3.11 | Use `python.org` ARM64 MSI or `pyenv-win`'s ARM64 entries. |
| Flask / werkzeug / flask-cors | pure-Python | works as-is | Nothing to do. |
| `cryptography` | wheels on PyPI | ARM64 Linux wheels exist; Windows ARM64 wheels still missing | Need to build from source via Cargo + Rust ARM64 toolchain. |
| `numpy` / `scipy` | manylinux + win_amd64 wheels | ARM64 wheels exist on PyPI ≥ 1.24 / 1.11 | Confirmed at this writing. |
| `pyav` | win_amd64 wheels | no published ARM64 wheel | Build against FFmpeg locally; cost is in FFmpeg cross-compile. |
| `torch` (optional) | win_amd64 CUDA wheels | no official Windows ARM64 wheel | Local-only / CPU fallback works; GPU AI features will be unavailable until PyTorch ships ARM64 Windows. |
| `whisperx` / `faster-whisper` | win_amd64 + CUDA | torch dep is the constraint | Same status as torch. |
| FFmpeg | bundled `ffmpeg-x86_64` | Need to bundle `ffmpeg-arm64` from gyan.dev or a reproducible build | gyan.dev publishes ARM64 nightlies but not signed releases. |
| Premiere CEP panel | Adobe ships CEP host as x64 | Adobe is x64-only through Premiere 2025. ARM64 emulated via Prism. | CEP runs under x64 emulation on Surface Pro X. |
| Premiere UXP plugin | Adobe ships UXP host as x64 today | Adobe 2026+ UXP is ARM64-aware on Apple Silicon; Windows UXP ARM64 schedule unclear | Track Adobe public roadmap. |
| Inno Setup installer | `OpenCut.iss` builds an x86 installer that runs x64 binaries | Inno Setup 6.3 supports ARM64 architecture detection | Bump `ArchitecturesAllowed=arm64` / `ArchitecturesInstallIn64BitMode=arm64` once the bundled binaries are ARM64. |
| .NET WPF installer (`installer/src/OpenCut.Installer`) | Targets `net9.0-windows` x64 | `net9.0-windows` supports `RuntimeIdentifier=win-arm64` | Need an ARM64 publish step in CI. |
| PyInstaller spec (`opencut_server.spec`) | targets x64 | PyInstaller 6.x supports ARM64 host since v6.4 | Need an ARM64 GitHub runner; `windows-latest` is x64. |

## What would have to change

1. **CI runners.** GitHub Actions only ships `windows-2022` (x64) and a
   *preview* `windows-11-arm` runner that's currently slot-limited.
   We'd need either the preview runner or a self-hosted ARM64 Windows
   VM. The build workflow (`.github/workflows/build.yml`) would gain a
   third matrix row with `runs-on: windows-11-arm` plus a guard that
   the PyInstaller spec resolves the right interpreter.
2. **Bundled FFmpeg.** Pull `ffmpeg-arm64.exe` + `ffprobe-arm64.exe`
   from gyan.dev (or build from upstream) and place them under
   `ffmpeg/win-arm64/` so `helpers.get_ffmpeg_path()` can pick the
   right binary at runtime. Update the Inno script's `Source:` paths
   and `Check: IsArchitectureSupported('arm64')` guards.
3. **PyInstaller spec.** The hidden-imports list and `binaries=[...]`
   block already work cross-arch; verify by running
   `pyinstaller opencut_server.spec` on an ARM64 host.
4. **Cryptography wheel.** Until PyPI ships a wheel,
   `OPENCUT_C2PA_SIGNING_KEY` and TLS-secured remote binds need
   manual install: `pip install cryptography --no-binary :all:`
   (requires Rust + clang). Document this in
   `docs/NODE_ADVISORIES.md` / `SECURITY.md` rather than auto-install.
5. **Telemetry / capability probe.** `system/capabilities` already
   reports `platform.machine()`, so the panel can grey out features
   that don't have an ARM64 dependency.
6. **Code-signing certificate.** We currently sign x64 with one cert;
   ARM64 signing uses the same cert but the SignTool invocation must
   add `/fd SHA256 /td SHA256 /tr <ts>` for the dual-binary signing
   the Microsoft Store accepts. Inno's `SignTool` directive can do
   both binaries in one pass.

## Cost estimate

| Task | Effort |
|------|--------|
| CI matrix row + self-hosted runner setup | 2-3 days |
| Bundled FFmpeg ARM64 verification + Inno guard | 1 day |
| PyInstaller spec + smoke test | 1 day |
| Cryptography / TLS dep documentation | 0.5 day |
| Code-signing pipeline update | 1 day |
| Manual QA on a Surface Pro X / SQ3 device | 2 days |
| **Total** | **~7-8 days** |

## Acceptance criteria for promotion to *Now*

We will move F101 out of *Next* when **all** of the following are true:

1. `scripts/release_smoke.py` runs green on a Windows ARM64 host with
   `--skip` removed from no more steps than the x64 host.
2. The Inno installer produces a signed `OpenCut-Setup-arm64.exe`
   that installs and launches `opencut-server` end-to-end on a clean
   Surface Pro X / SQ3 device.
3. `python -m opencut.tools.dump_route_manifest --check` and
   `python scripts/sync_version.py --check` both pass on the same
   host.
4. We have at least one third-party tester confirming the install
   path is friction-free.

Until then this document is the source of truth: link any ARM64-related
issue at the existing seed in `.github/issue-seeds.yml#F101`.
