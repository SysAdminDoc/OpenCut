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

## Bundled FFmpeg — version + security patch level

The FFmpeg/ffprobe binaries are bundled by the installer (the `ffmpeg/` directory
is gitignored and fetched at build time, not committed). The bundled build must
clear a **security floor**, not merely a version string: the June-2026 automated
FFmpeg audit disclosed ~21 zero-days — `CVE-2026-6385` (confirmed, CVSS 6.5) plus
`CVE-2026-39210..39218` (reserved) — several heap/stack overflows reachable via
crafted media, which is the first untrusted-input path a media tool hits. Those
fixes landed as post-release master commits, so an `8.1.x` *release tag* can
predate them.

`opencut/core/ffmpeg_provenance.py` is the single source of truth. A bundled build
is acceptable on **either** lane:

- **Release lane** — a tagged release `>= 8.1.1` (gyan.dev's current stable point
  release; point releases carry the backported security branch). This satisfies the
  D3D12VA/Vulkan encoder routes that expect `8.1.x`.
- **Snapshot lane** — a gyan.dev / BtbN git-master snapshot dated `>= 2026-06-10`.
  The recorded reference snapshot is gyan.dev `git-full` **commit `b29bdd3715`**
  (`2026-06-10`), the guaranteed-clean fallback if a release tag is ever found to
  predate a specific June-2026 fix.

The current bundled release pin is
`8.1.2-essentials_build-www.gyan.dev` from
<https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip>
(updated `2026-06-27`, SHA256
`db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec`).
The version pin is mirrored in
`installer/src/OpenCut.Installer/Models/AppConstants.cs` and `OpenCut.iss`, and both
installers record `bundled_ffmpeg_security_floor` into `~/.opencut/installer.json`.

### Verify at build / release time

```bash
# Fails closed (exit 1) when the bundled binary is below the floor:
python scripts/verify_ffmpeg_provenance.py
# Record ground-truth provenance (version, git commit/date, lane, CVEs) to JSON:
python scripts/verify_ffmpeg_provenance.py --manifest dist/ffmpeg-provenance.json
```

At runtime, `GET /system/capabilities` carries `ffmpeg.security` and emits a
`ffmpeg_below_security_floor` finding when a stale binary is detected.

Fetch a compliant build with `winget install Gyan.FFmpeg` (release lane) or from
<https://www.gyan.dev/ffmpeg/builds/> (`ffmpeg-git-full` for the snapshot lane).
