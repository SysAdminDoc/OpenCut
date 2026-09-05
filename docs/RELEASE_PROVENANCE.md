# Release Provenance

Release provenance is generated and checked locally before artifacts are
attached to a release. Run the local release smoke first, then generate the
declared SBOM and FFmpeg provenance manifest beside the artifacts:

```bash
python scripts/release_smoke.py --json
python scripts/sbom.py --format json --output dist/opencut-declared-sbom.cyclonedx.json
python scripts/verify_ffmpeg_provenance.py --manifest dist/ffmpeg-provenance.json
python scripts/verify_embedded_media_provenance.py --artifact dist/OpenCut-Server --manifest dist/embedded-media-provenance.json
```

Keep the generated manifest files with the server bundle, Linux packages, and
Windows installer that were built from the same commit. Use the local filenames
when verifying hashes or attaching assets with `gh release create` /
`gh release upload`.

## Embedded media decoders

OpenCV and PyAV each carry their own FFmpeg libraries. They are checked
separately from the external `ffmpeg` and `ffprobe` executables. The release
gate records the library versions, native filenames, sizes, and SHA-256 hashes
in `embedded-media-provenance.json`.

OpenCut pins OpenCV to the reviewed `4.14.0.94` wheel and PyAV to a reviewed
18.x wheel. PyAV must report the FFmpeg 8.1.2 library floor on every platform.
Linux and macOS require the same floor from OpenCV. The Windows OpenCV wheel
still carries older FFmpeg libraries, so OpenCut disables that backend before
`cv2` loads and removes its `opencv_videoio_ffmpeg` plugin from the packaged
server. The release gate fails if that plugin reappears, if a decoder version
cannot be read, or if an unattributed FFmpeg library enters an artifact.

Run the lane-specific check against the assembled payload:

```bash
python scripts/verify_embedded_media_provenance.py --lane windows --artifact dist/OpenCut-Server --manifest dist/embedded-media-provenance.json
```

## Bundled FFmpeg version and security patch level

The FFmpeg/ffprobe binaries are bundled by the installer (the `ffmpeg/` directory
is gitignored and fetched at build time, not committed). The bundled build must
clear a **security floor**, not merely a version string: the June-2026 automated
FFmpeg audit disclosed ~21 zero-days — `CVE-2026-6385` (confirmed, CVSS 6.5) plus
`CVE-2026-39210..39218` (reserved) — several heap/stack overflows reachable via
crafted media, which is the first untrusted-input path a media tool hits. Those
fixes landed as post-release master commits, so an `8.1.x` *release tag* can
predate them.

`opencut/core/ffmpeg_provenance.py` is the single source of truth, and
`python -m opencut.tools.check_provenance_docs --check` fails the release gate if
anything below drifts from it. There are two lanes, and only one of them is open:

- **Release lane is closed.** `RELEASE_LANE_OPEN` is `False`. No published FFmpeg
  release clears the advisory matrix: 8.1.2 is affected by the July 2026 batch,
  the 8.1.3 that would carry the fix was never published, and the 9.0 series was
  branched on 2026-06-26, before those fixes landed on master. `RELEASE_FLOOR` is
  `8.1.3` because that is where a qualifying release would have to appear, not
  because such a build exists. A higher version number is not evidence; the
  branch point is.
- **Snapshot lane** is the only way through. A gyan.dev or BtbN git-master
  snapshot dated `2026-07-06` or later, which is `SNAPSHOT_FLOOR_DATE`.

The bundled build is therefore a snapshot, not a release:
`2026-08-03-git-01a25f74cc-full_build-www.gyan.dev` from
<https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-2026-08-03-git-01a25f74cc-full_build.7z>
(SHA256 `8c32ed9800ff421bbcfda96beb0a66783a64a7cd98869b87ec1b494d3c855fcc`).
That pin lives in `installer/src/OpenCut.Installer/Models/AppConstants.cs` and
`OpenCut.iss`, and both installers record `bundled_ffmpeg_security_floor` into
`~/.opencut/installer.json`.

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
