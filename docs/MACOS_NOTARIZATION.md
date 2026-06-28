# macOS Notarization

OpenCut release builds notarize the macOS PyInstaller bundle from a local macOS
release shell. The helper uses Apple's `xcrun notarytool`; `altool` is
intentionally not used because Apple no longer accepts it for notarization
uploads.

## Required GitHub Secrets

Export these variables before cutting a macOS release:

| Variable | Purpose |
|---|---|
| `MACOS_CERTIFICATE_P12_BASE64` | Base64-encoded Developer ID Application `.p12` certificate. |
| `MACOS_CERTIFICATE_PASSWORD` | Password for the `.p12` certificate. |
| `APPLE_API_KEY_ID` | App Store Connect API key ID. |
| `APPLE_API_ISSUER_ID` | App Store Connect issuer ID. |
| `APPLE_API_PRIVATE_KEY` | Contents of the `AuthKey_*.p8` private key. |

Optional:

| Secret | Purpose |
|---|---|
| `MACOS_SIGNING_IDENTITY` | Explicit Developer ID Application identity if multiple identities are present. |
| `MACOS_KEYCHAIN_PASSWORD` | Temporary keychain password override. |

## Local Release Path

The local release path:

1. Builds `dist/OpenCut-Server` with PyInstaller on macOS.
2. Imports the Developer ID certificate into a temporary keychain.
3. Signs Mach-O files with hardened runtime and timestamping.
4. Archives the bundle as `dist/OpenCut-Server-macOS.zip`.
5. Submits the ZIP to Apple's notary service with `xcrun notarytool submit --wait`.
6. Uploads the notarized ZIP with `gh release upload`.

ZIP archives can be submitted for notarization, but they cannot be stapled directly. If OpenCut later ships a `.app`, `.pkg`, or `.dmg`, add stapling for that distributable in `scripts/notarize_macos.sh`.

## Local Checks

On a macOS machine with secrets exported:

```sh
scripts/notarize_macos.sh --check-env
pyinstaller opencut_server.spec
scripts/notarize_macos.sh dist/OpenCut-Server dist/OpenCut-Server-macOS.zip
```

References:

- [Apple: Notarizing macOS software before distribution](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- [Apple: Customizing the notarization workflow](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution/customizing_the_notarization_workflow)
- [Apple: TN3147 Migrating to the latest notarization tool](https://developer.apple.com/documentation/technotes/tn3147-migrating-to-the-latest-notarization-tool)
