# Windows Authenticode Signing

OpenCut release builds can Authenticode-sign Windows installer artifacts on
GitHub Actions tag builds and manual release builds. The workflow signs the
recommended WPF installer artifact and the Inno fallback when signing secrets
are configured.

## Required GitHub Secrets

| Secret | Purpose |
|---|---|
| `WINDOWS_CODESIGN_PFX_BASE64` | Base64-encoded Authenticode `.pfx` certificate. |
| `WINDOWS_CODESIGN_PFX_PASSWORD` | Password for the `.pfx` certificate. |

Optional:

| Secret | Purpose |
|---|---|
| `WINDOWS_CODESIGN_TIMESTAMP_URL` | RFC3161 timestamp server. Defaults to `http://timestamp.digicert.com`. |
| `WINDOWS_CODESIGN_CERT_EXPIRES_AT` | ISO date used for CI renewal warnings, for example `2027-03-01T00:00:00Z`. |

## Workflow

The release workflow:

1. Builds `dist/OpenCut-Server` with PyInstaller on `windows-latest`.
2. Builds the WPF installer and archives `installer/dist/wpf/OpenCut-WPF-Setup-*.exe`.
3. Builds the Inno fallback installer at `installer/dist/OpenCut-Setup-*.exe`.
4. Runs `scripts/sign_windows_artifacts.ps1` over both installer paths.
5. Verifies each signed artifact with `signtool verify /pa /v`.
6. Uploads the WPF and Inno artifacts to the workflow run; tag releases still publish the release-facing installer path.

If the signing secrets are absent, the signing step exits successfully with a
warning and leaves artifacts unsigned. That keeps fork builds and dry-run
workflow dispatches usable while making the production signing path explicit.

## Renewal Policy

Set `WINDOWS_CODESIGN_CERT_EXPIRES_AT` whenever the signing certificate is
configured. The helper warns when the date is inside the 90-day renewal window.
For a release freeze, rerun with `-FailOnExpiringCert` to make the warning fatal.

Keep the renewed certificate and password in GitHub Secrets only. Do not commit
`.pfx` files, DER/PEM exports, or password material to the repository.

## Local Check

On a Windows machine with the Windows SDK and secrets exported:

```powershell
pyinstaller opencut_server.spec
./scripts/build_wpf_installer_ci.ps1
./scripts/sign_windows_artifacts.ps1 -FailOnExpiringCert
```

References:

- `scripts/sign_windows_artifacts.ps1` — Authenticode signing helper.
- `.github/workflows/build.yml` — release workflow wiring.
- `docs/INSTALLER_POLICY.md` — WPF vs Inno policy and retirement gates.
