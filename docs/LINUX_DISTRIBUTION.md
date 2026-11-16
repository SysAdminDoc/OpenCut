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

## Flathub

OpenCut isn't published on Flathub, and no submission is planned. The
Flatpak is a self-contained bundle built locally from the PyInstaller
output, so the manifest wouldn't meet Flathub's build-from-source and
no-network-at-build rules anyway. The build only uses Flathub as the
place it downloads the freedesktop runtime and SDK from.

The app ID, desktop file, MetaInfo and icon still follow the Flatpak
conventions, since GNOME Software, KDE Discover and `flatpak` itself read
them from an installed bundle.

## Source Notes

- Flatpak current first-build tutorial uses `org.freedesktop.Platform` /
  `org.freedesktop.Sdk` 25.08.
- Flathub requires hosted runtimes to be current at submission time and
  rejects network access during build.
- Flatpak desktop integration expects the app ID to match the desktop,
  MetaInfo, and icon filenames.
- AppImage packages are built from an AppDir with `AppRun`, a desktop
  file, an app icon, and a `usr/` layout.
