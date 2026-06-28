# Windows Authenticode Signing

OpenCut release builds can Authenticode-sign Windows installer artifacts from a
local Windows release shell. The signing helper signs the recommended WPF
installer artifact and the Inno fallback when signing environment variables are
configured.

## Required Environment Variables

| Variable | Purpose |
|---|---|
| `WINDOWS_CODESIGN_PFX_BASE64` | Base64-encoded Authenticode `.pfx` certificate. |
| `WINDOWS_CODESIGN_PFX_PASSWORD` | Password for the `.pfx` certificate. |

Optional:

| Variable | Purpose |
|---|---|
| `WINDOWS_CODESIGN_TIMESTAMP_URL` | RFC3161 timestamp server. Defaults to `http://timestamp.digicert.com`. |
| `WINDOWS_CODESIGN_CERT_EXPIRES_AT` | ISO date used for renewal warnings, for example `2027-03-01T00:00:00Z`. |

## Local Release Path

The local release path:

1. Builds `dist/OpenCut-Server` with PyInstaller.
2. Builds the WPF installer and archives `installer/dist/wpf/OpenCut-WPF-Setup-*.exe`.
3. Runs the WPF quiet install/uninstall smoke against the self-extracting artifact.
4. Builds the Inno fallback installer at `installer/dist/OpenCut-Setup-*.exe`.
5. Runs `scripts/sign_windows_artifacts.ps1` over both installer paths.
6. Verifies each signed artifact with `signtool verify /pa /v`.
7. Uploads the selected release-facing installer with `gh release upload`.

If the signing variables are absent, the signing step exits successfully with a
warning and leaves artifacts unsigned. That keeps dry-run local builds usable
while making the production signing path explicit.

## Renewal Policy

Set `WINDOWS_CODESIGN_CERT_EXPIRES_AT` whenever the signing certificate is
configured. The helper warns when the date is inside the 90-day renewal window.
For a release freeze, rerun with `-FailOnExpiringCert` to make the warning fatal.

Keep the renewed certificate and password in a local secret store. Do not commit
`.pfx` files, DER/PEM exports, or password material to the repository.

## Local Check

On a Windows machine with the Windows SDK and secrets exported:

```powershell
pyinstaller opencut_server.spec
./scripts/build_wpf_installer_ci.ps1
./scripts/smoke_wpf_installer.ps1 -AllowLocalProfileMutation
./scripts/sign_windows_artifacts.ps1 -FailOnExpiringCert
```

References:

- `scripts/sign_windows_artifacts.ps1` — Authenticode signing helper.
- `docs/INSTALLER_POLICY.md` — WPF vs Inno policy and retirement gates.
