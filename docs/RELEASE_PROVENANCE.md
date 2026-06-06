# Release Provenance

Tagged Release Full runs generate GitHub artifact attestations before uploading
release assets. The attested subjects are the same file paths used by the
release upload job:

- `release-upload-artifacts/server/*`
- `release-artifacts/OpenCut-Linux-Desktop-Packages/*`
- `release-artifacts/OpenCut-Setup-Windows/*`
- `release-artifacts/OpenCut-Declared-Dependency-SBOM-CycloneDX/opencut-declared-sbom.cyclonedx.json`

After downloading a release asset, verify its provenance with GitHub CLI:

```bash
gh attestation verify OpenCut-Server-Linux.tar.gz -R SysAdminDoc/OpenCut
gh attestation verify opencut-declared-sbom.cyclonedx.json -R SysAdminDoc/OpenCut
```

Use the local filename for the downloaded artifact you are checking. Server
directory bundles are packaged in `release-upload-artifacts/server/` before the
attestation step so the generated provenance matches the uploaded tarball or
macOS ZIP rather than the transient downloaded artifact directory.
