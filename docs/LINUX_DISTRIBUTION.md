# Linux Distribution

OpenCut's Linux release path is Flatpak-first, with AppImage as the
fallback for users who cannot or do not want to install through Flatpak.

## Package IDs

- Flatpak app ID: `io.github.sysadmindoc.opencut`
- Desktop file: `packaging/linux/io.github.sysadmindoc.opencut.desktop`
- MetaInfo file: `packaging/linux/io.github.sysadmindoc.opencut.metainfo.xml`
- Primary release artifacts:
  - `dist/linux-packages/io.github.sysadmindoc.opencut-<version>.flatpak`
  - `dist/linux-packages/OpenCut-<version>-x86_64.AppImage`

The app ID follows the Flatpak reverse-DNS convention for GitHub-hosted
projects (`io.github.<owner>.<project>`), and the domain portion is
lowercase. The desktop, MetaInfo, and icon names intentionally match the
same ID.

## Build

Linux packages are built from the PyInstaller one-folder server bundle:

```bash
pyinstaller opencut_server.spec
bash scripts/build_linux_packages.sh
```

The script always stages an AppDir at
`dist/linux-packages/appimage/OpenCut.AppDir`. If `appimagetool` is
available, it also emits the `.AppImage`. If `flatpak-builder` and
`flatpak` are available, it builds a local OSTree repository and exports
a single-file `.flatpak` bundle.

## Flathub Boundary

The repository now carries the upstream Linux desktop metadata required
for Flathub review: app ID, desktop file, MetaInfo, icon install path,
runtime/Sdk selection, and `flathub.json` architecture policy. The release
workflow produces self-contained Flatpak bundles for GitHub releases.

Flathub's hosted submission repository must still obey Flathub's
build-from-source and no-network-at-build rules. If OpenCut is submitted
to Flathub, the PyInstaller bundle source in `io.github.sysadmindoc.opencut.yml`
should be replaced in the Flathub submission repository with a source build
module plus generated Python dependency manifests.

## Source Notes

- Flatpak current first-build tutorial uses `org.freedesktop.Platform` /
  `org.freedesktop.Sdk` 25.08.
- Flathub requires hosted runtimes to be current at submission time and
  rejects network access during build.
- Flatpak desktop integration expects the app ID to match the desktop,
  MetaInfo, and icon filenames.
- AppImage packages are built from an AppDir with `AppRun`, a desktop
  file, an app icon, and a `usr/` layout.
