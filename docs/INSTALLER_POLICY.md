# OpenCut — Installer Policy

> **F200 status (2026-05-17):** repository ships two parallel Windows
> installer paths today. This document captures the policy decision
> and the retirement plan so future contributors don't add work to
> both pipelines by accident.
>
> **F213 status (2026-05-18):** the Inno fallback now has a guarded
> install/uninstall smoke script (`scripts/smoke_inno_installer.ps1`)
> for disposable local release validation after the Windows Inno build step.
>
> **F201 status (2026-05-18):** the recommended WPF installer now has
> a reusable Windows build wrapper via `scripts/build_wpf_installer_ci.ps1`
> and publishes separately from the Inno fallback.
>
> **F203 status (2026-05-18):** Windows release scripts now call
> `scripts/sign_windows_artifacts.ps1` to Authenticode-sign WPF and
> Inno installer outputs when signing variables are configured. See
> `docs/WINDOWS_CODESIGNING.md`.
>
> **F212 status (2026-05-18):** the recommended WPF installer now has
> `installer/tests/OpenCut.Installer.Tests` xUnit coverage plus guarded
> quiet install/uninstall smoke validation via `scripts/smoke_wpf_installer.ps1`.
>
> **Tracking F-number:** F200 (Now-tier doc deliverable).
> Related F-numbers: **F201** (local build wrapper for the recommended path, DONE),
> **F203** (Authenticode code-signing renewal, DONE tooling), **F207** (bundled
> FFmpeg version embedded in installer manifest, DONE Pass 12),
> **F212** (WPF installer xUnit + headless smoke, DONE).

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
   F201 now closes the official local build-wrapper gap; the Inno path remains
   a local fallback for contributors without the .NET SDK.
2. **Emergency hot-fix releases.** If an issue blocks the WPF path
   (e.g. signing cert expiry or missing F203 secrets), the Inno path
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
* Local release validation (`F201`) covers the WPF path on Windows builds. The Inno path has F213
  install/uninstall smoke coverage on disposable local Windows builds, but
  remains a deprecated fallback rather than the recommended release
  path.

### Retirement timeline

| Milestone | Trigger | Action |
|---|---|---|
| **Now** | (this document) | Designate WPF as recommended; Inno as deprecated-but-supported fallback. |
| **F201 close** | WPF local wrapper | DONE — WPF builds through the Windows wrapper and archives `OpenCut-WPF-Setup-X.Y.Z.exe`; Inno stays deprecated-but-supported until signed WPF release verification is complete. |
| **F213 close** | Inno install/uninstall smoke | DONE — keep Inno as a deprecated-but-supported fallback with guarded smoke coverage until signed WPF release verification is complete. |
| **F212 close** | WPF xUnit + headless smoke | DONE — WPF now has xUnit coverage and guarded quiet install/uninstall smoke. Keep Inno as an emergency fallback until at least one signed WPF release completes with the smoke green. |

The retirement gate is **not** a calendar date — it is the WPF
coverage milestone plus one signed-release verification. Don't retire
the Inno script until WPF can catch the regressions the Inno path
historically caught and the signed artifact path has been exercised.

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

## 4. Why Inno still remains as fallback

One risk forces keeping the fallback for now:

1. **Operational signing credentials** — F203 tooling is wired, but
   production releases still require valid `WINDOWS_CODESIGN_*`
   signing variables and timely certificate renewal. A botched renewal
   would block signed WPF releases; the Inno path stays as a fallback.
   F212 now covers the WPF behavior gap with xUnit tests and quiet
   install/uninstall smoke wiring, but a signed WPF release still needs
   live GitHub secrets to validate the final production path.

When the signed WPF release path completes cleanly, reassess the
retirement schedule in §2.

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
* `scripts/build_wpf_installer_ci.ps1` — F201 Windows wrapper that prepares
  the PyInstaller + FFmpeg payload and archives the WPF output before
  the Inno fallback can overwrite the legacy filename.
* `scripts/sign_windows_artifacts.ps1` — F203 Authenticode signing
  helper for WPF and Inno installer outputs.
* `docs/WINDOWS_CODESIGNING.md` — required secrets and renewal policy.
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
