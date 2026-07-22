#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ID="io.github.sysadmindoc.opencut"
APP_NAME="OpenCut"
ARCH="${ARCH:-x86_64}"
DIST_DIR="${1:-${OPENCUT_DIST_DIR:-$REPO_ROOT/dist/OpenCut-Server}}"
OUT_DIR="${2:-${OPENCUT_LINUX_PACKAGE_DIR:-$REPO_ROOT/dist/linux-packages}}"
VERSION="${OPENCUT_VERSION:-}"

if [[ -z "$VERSION" ]]; then
  VERSION="$(grep -m1 '^version = ' "$REPO_ROOT/pyproject.toml" | sed -E 's/version = "([^"]+)"/\1/')"
fi

SERVER_BIN="$DIST_DIR/OpenCut-Server"
if [[ ! -f "$SERVER_BIN" ]]; then
  echo "Missing PyInstaller server binary: $SERVER_BIN" >&2
  echo "Run: pyinstaller opencut_server.spec" >&2
  exit 2
fi

if [[ ! -x "$SERVER_BIN" ]]; then
  chmod +x "$SERVER_BIN"
fi

mkdir -p "$OUT_DIR"

RELEASE_METADATA="$OUT_DIR/release-metadata"
python "$REPO_ROOT/scripts/release_composition.py" \
  --lane linux \
  --artifact "$DIST_DIR" \
  --build-lock "$REPO_ROOT/requirements-build-lock.txt" \
  --output-dir "$RELEASE_METADATA"

DESKTOP_FILE="$REPO_ROOT/packaging/linux/$APP_ID.desktop"
METAINFO_FILE="$REPO_ROOT/packaging/linux/$APP_ID.metainfo.xml"
ICON_FILE="$REPO_ROOT/img/icon.png"

if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$DESKTOP_FILE"
else
  echo "desktop-file-validate not found; skipping desktop metadata validation"
fi

if command -v appstreamcli >/dev/null 2>&1; then
  if ! appstreamcli validate --no-net "$METAINFO_FILE"; then
    echo "appstreamcli reported metadata issues; fix before Flathub submission" >&2
  fi
else
  echo "appstreamcli not found; skipping MetaInfo validation"
fi

APPDIR="$OUT_DIR/appimage/$APP_NAME.AppDir"
rm -rf "$APPDIR"
mkdir -p \
  "$APPDIR/usr/lib/opencut" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/metainfo" \
  "$APPDIR/usr/share/icons/hicolor/256x256/apps" \
  "$APPDIR/usr/share/opencut/release-metadata"

cp -a "$DIST_DIR" "$APPDIR/usr/lib/opencut/OpenCut-Server"
install -Dm755 "$REPO_ROOT/packaging/linux/appimage/AppRun" "$APPDIR/AppRun"
install -Dm644 "$DESKTOP_FILE" "$APPDIR/usr/share/applications/$APP_ID.desktop"
install -Dm644 "$METAINFO_FILE" "$APPDIR/usr/share/metainfo/$APP_ID.metainfo.xml"
install -Dm644 "$ICON_FILE" "$APPDIR/usr/share/icons/hicolor/256x256/apps/$APP_ID.png"
cp -a "$RELEASE_METADATA/." "$APPDIR/usr/share/opencut/release-metadata/"
ln -sf "usr/share/applications/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
ln -sf "usr/share/icons/hicolor/256x256/apps/$APP_ID.png" "$APPDIR/$APP_ID.png"
ln -sf "$APP_ID.png" "$APPDIR/.DirIcon"

if command -v appimagetool >/dev/null 2>&1; then
  APPIMAGE_OUT="$OUT_DIR/$APP_NAME-$VERSION-$ARCH.AppImage"
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" VERSION="$VERSION" APPIMAGETOOL_APP_NAME="$APP_NAME" \
    appimagetool "$APPDIR" "$APPIMAGE_OUT"
else
  echo "appimagetool not found; AppDir staged at $APPDIR"
fi

if command -v flatpak-builder >/dev/null 2>&1 && command -v flatpak >/dev/null 2>&1; then
  FLATPAK_BUILD_DIR="$OUT_DIR/flatpak-build"
  FLATPAK_REPO_DIR="$OUT_DIR/flatpak-repo"
  FLATPAK_BUNDLE="$OUT_DIR/$APP_ID-$VERSION.flatpak"
  rm -rf "$FLATPAK_BUILD_DIR" "$FLATPAK_REPO_DIR"
  flatpak-builder \
    --force-clean \
    --install-deps-from=flathub \
    --repo="$FLATPAK_REPO_DIR" \
    "$FLATPAK_BUILD_DIR" \
    "$REPO_ROOT/$APP_ID.yml"
  flatpak build-bundle \
    "$FLATPAK_REPO_DIR" \
    "$FLATPAK_BUNDLE" \
    "$APP_ID" \
    --runtime-repo=https://flathub.org/repo/flathub.flatpakrepo
else
  echo "flatpak-builder or flatpak not found; skipping Flatpak bundle"
fi

find "$OUT_DIR" -maxdepth 2 -type f \( -name '*.flatpak' -o -name '*.AppImage' \) -print
