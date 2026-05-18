# OpenCut — Installer Policy

> **F200 status (2026-05-17):** repository ships two parallel Windows
> installer paths today. This document captures the policy decision
> and the retirement plan so future contributors don't add work to
> both pipelines by accident.
>
> **F213 status (2026-05-18):** the Inno fallback now has a CI-only
> install/uninstall smoke script (`scripts/smoke_inno_installer.ps1`)
> wired after the Windows Inno build step.
>
> **Tracking F-number:** F200 (Now-tier doc deliverable).
> Related F-numbers: **F201** (CI for the recommended path),
> **F203** (Authenticode code-signing renewal), **F207** (bundled
> FFmpeg version embedded in installer manifest, DONE Pass 12),
> **F212** (WPF installer xUnit test suite, deferred to Later).

---

## 1. The two paths today

| Path | Source | Output | When used |
|---|---|---|---|
| **WPF / .NET 9 installer** (`installer/`) | `installer/src/OpenCut.Installer/*` + `installer/InstallerBuilder.ps1` | `installer/dist/OpenCut-Setup-X.Y.Z.exe` | Default download from the Releases page. Custom UI, signed by the OpenCut Authenticode cert, embeds the bundled FFmpeg + manifest writer (F207). |
| **Inno Setup installer** (`OpenCut.iss` + `Install.bat`) | `OpenCut.iss` + the helper `Install.bat` / `Install.ps1` | `dist/OpenCut-Setup-X.Y.Z.exe` (legacy filename collision) | Available as a fallback for builders without the .NET 9 SDK. Pre-dates the WPF path; was the only installer through v1.10.x. |

Both installers produce the same on-disk layout (FFmpeg under
`{app}\ffmpeg`, server entry under `{app}\server`, registry CEP
extension keys, optional shortcut), so the end-user experience is
identical — the difference is purely in the build pipeline.

---

## 2. Policy decision

**The WPF installer is the recommended path going forward.** The
Inno Setup script (`OpenCut.iss`) is **deprecated** but kept in the
repo for two specific use cases:

1. **No-.NET-9 build environments.** A contributor who only has the
   PyInstaller dist available can still produce an installer
   binary with Inno Setup 6, which is single-binary, no-SDK.
   F201 (WPF in CI) closes the gap; until then the Inno path is the
   fallback.
2. **Emergency hot-fix releases.** If an issue blocks the WPF path
   (e.g. signing cert expiry, F203 not renewed yet), the Inno path
   can still produce a self-signed installer for users who already
   trust the OpenCut publisher fingerprint.

Both installer outputs ship under the **same** filename pattern
(`OpenCut-Setup-X.Y.Z.exe`) and the **same** install-tree layout.
Only one binary is published per release.

### What "recommended" means

* All new installer features (UI strings, new bundled assets,
  registry keys, telemetry opt-outs, etc.) land in the WPF path
  first.
* The Inno script gets **mirror-only** updates — every change to
  `installer/src/OpenCut.Installer/AppConstants.cs` that affects
  user-visible behaviour must have a matching entry in
  `OpenCut.iss`. Test `tests/test_installer_policy.py` enforces
  this for the items the policy contract names.
* CI (`F201`) covers the WPF path. The Inno path has F213
  install/uninstall smoke coverage on tag/manual Windows builds, but
  remains a deprecated fallback rather than the recommended release
  path.

### Retirement timeline

| Milestone | Trigger | Action |
|---|---|---|
| **Now** | (this document) | Designate WPF as recommended; Inno as deprecated-but-supported fallback. |
| **F201 close** | WPF in CI | Mark Inno as "deprecated; build locally only" in the release-notes template. |
| **F213 close** | Inno install/uninstall smoke in CI | DONE — keep Inno as a deprecated-but-supported fallback with CI smoke coverage until F212 decides retirement. |
| **F212 close** | WPF xUnit test suite | If WPF coverage meets the bar set in `TEST_COVERAGE_GAPS.md §3.6`, formally retire Inno (move `OpenCut.iss` to `archive/`). |

The retirement gate is **not** a calendar date — it is the WPF
coverage milestone. Don't retire the Inno script until WPF can
catch the regressions the Inno path historically caught.

---

## 3. What must stay in lockstep

The two installers diverge on UI but **must agree** on the items
that affect the install tree or the running server's environment.
`tests/test_installer_policy.py` pins the following invariants:

1. **Bundled FFmpeg version string** — both installers must write
   the same `~/.opencut/installer.json` `ffmpeg_version` value
   (F207).
2. **App display name** — both installers register the same
   `DisplayName` (`OpenCut`) under
   `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OpenCut`.
3. **Install root** — both installers default to
   `C:\Program Files\OpenCut` and accept the same per-user override.
4. **CEP extension folder layout** — both installers deposit the
   panel under
   `%APPDATA%\Adobe\CEP\extensions\com.opencut.panel`.

When you add a fifth invariant, document it here and add a test in
`tests/test_installer_policy.py`.

---

## 4. Why we didn't retire Inno yet

Three risks force keeping the fallback for now:

1. **No WPF CI** — F201 (WPF in CI) is still Next-tier. Until then
   only the maintainer's local Windows machine builds the WPF
   installer.
2. **Authenticode renewal cliff** — F203 flags that the OpenCut
   signing cert validity drops to 458 days starting March 2026,
   and SmartScreen no longer auto-trusts new certs. A botched
   renewal would block WPF releases; the Inno path stays as a
   self-signed fallback.
3. **No WPF tests yet** — F212 (WPF xUnit suite) is Later-tier.
   Inno's behaviour is now covered by the F213 install/uninstall
   smoke on disposable Windows CI workers, while WPF still relies on
   manual QA plus lockstep invariant tests.

When any one of those three resolves, reassess the retirement
schedule in §2.

---

## 5. Common contributor questions

> **Q: Which installer do I run as a tester?**
>
> A: `installer/dist/OpenCut-Setup-X.Y.Z.exe` (WPF). The Releases
> page download is always the WPF build.

> **Q: I'm hacking on the install flow. Which file do I edit?**
>
> A: WPF (`installer/src/OpenCut.Installer/*.cs`). Then mirror the
> change in `OpenCut.iss` if it touches one of the invariants in
> §3, and add a regression entry in `tests/test_installer_policy.py`.

> **Q: Can I add a registry key only to one installer?**
>
> A: No. Either both, or neither. The whole point of F200 is that
> the two installers produce the same on-disk + registry state.

> **Q: How do I check the policy without reading this doc?**
>
> A: Run `python -m pytest tests/test_installer_policy.py`. It
> spells out the invariants and fails with the exact mismatch.

---

## 6. References

* `installer/InstallerBuilder.ps1` — WPF build orchestrator.
* `installer/src/OpenCut.Installer/AppConstants.cs` — WPF constants;
  shipped FFmpeg version (F207) lives here.
* `OpenCut.iss` — Inno Setup script.
* `Install.bat` / `Install.ps1` — dev-mode bootstrap helpers (not
  end-user installers; install Python deps + register CEP for local
  testing).
* `docs/MACOS_NOTARIZATION.md` — macOS counterpart (F202).
* `docs/WINDOWS_ARM64_PACKAGING.md` — Windows-on-ARM64 build notes
  (F101).
* `.ai/research/2026-05-17/INSTALLER_AUDIT.md` — Pass-2 audit that
  surfaced F200.
