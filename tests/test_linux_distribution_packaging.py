"""F249 Linux distribution packaging contract tests."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ID = "io.github.sysadmindoc.opencut"
MANIFEST = REPO_ROOT / f"{APP_ID}.yml"
DESKTOP = REPO_ROOT / "packaging" / "linux" / f"{APP_ID}.desktop"
METAINFO = REPO_ROOT / "packaging" / "linux" / f"{APP_ID}.metainfo.xml"
FLATPAK_RUNNER = REPO_ROOT / "packaging" / "linux" / "flatpak" / "opencut-server"
APPIMAGE_RUNNER = REPO_ROOT / "packaging" / "linux" / "appimage" / "AppRun"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_linux_packages.sh"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_flatpak_manifest_uses_current_freedesktop_runtime_and_matching_id():
    text = _read(MANIFEST)

    assert "app-id: io.github.sysadmindoc.opencut" in text
    assert "runtime: org.freedesktop.Platform" in text
    assert "runtime-version: '25.08'" in text
    assert "sdk: org.freedesktop.Sdk" in text
    assert "command: opencut-server" in text
    assert f"path: packaging/linux/{APP_ID}.desktop" in text
    assert f"path: packaging/linux/{APP_ID}.metainfo.xml" in text
    assert "path: dist/OpenCut-Server" in text
    assert "path: img/icon.png" in text


def test_flatpak_manifest_sandbox_permissions_are_explicit():
    text = _read(MANIFEST)

    assert "--share=network" in text
    assert "--device=dri" in text
    assert "--filesystem=home" in text
    assert "--filesystem=xdg-videos" in text
    assert "--filesystem=/run/media" in text
    assert "--env=OPENCUT_HOST=127.0.0.1" in text
    assert "--env=OPENCUT_PORT=5679" in text


def test_desktop_file_matches_flatpak_id_and_console_server_shape():
    entries: dict[str, str] = {}
    for line in _read(DESKTOP).splitlines():
        if "=" in line and not line.startswith("["):
            key, value = line.split("=", 1)
            entries[key] = value

    assert entries["Type"] == "Application"
    assert entries["Name"] == "OpenCut"
    assert entries["Exec"] == "opencut-server"
    assert entries["Icon"] == APP_ID
    assert entries["Terminal"] == "true"
    assert "AudioVideo;" in entries["Categories"]
    assert "Video;" in entries["Categories"]


def test_metainfo_matches_flatpak_id_and_release_metadata():
    root = ET.parse(METAINFO).getroot()

    assert root.attrib["type"] == "desktop-application"
    assert root.findtext("id") == APP_ID
    assert root.findtext("metadata_license") == "CC0-1.0"
    assert root.findtext("project_license") == "MIT"
    assert root.find("launchable").attrib["type"] == "desktop-id"
    assert root.findtext("launchable") == f"{APP_ID}.desktop"
    assert root.find("developer").attrib["id"] == "io.github.sysadmindoc"
    assert root.find("releases/release").attrib["version"] == "1.32.0"


def test_linux_package_launchers_pin_user_data_and_loopback_defaults():
    for path in (FLATPAK_RUNNER, APPIMAGE_RUNNER):
        text = _read(path)
        assert text.startswith("#!/bin/sh")
        assert "OPENCUT_HOST" in text
        assert "127.0.0.1" in text
        assert "OPENCUT_PORT" in text
        assert "OPENCUT_HOME" in text
        assert "${XDG_DATA_HOME:-$HOME/.local/share}/opencut" in text
        assert "OPENCUT_FFMPEG_DIR" in text
        assert "export PATH=\"$OPENCUT_FFMPEG_DIR" in text
        assert "OpenCut-Server" in text
        assert '"$@"' in text


def test_helpers_honor_launcher_data_dir_and_prefer_packaged_media(monkeypatch, tmp_path):
    helpers_source = _read(REPO_ROOT / "opencut" / "helpers.py")
    assert 'os.environ.get("OPENCUT_HOME"' in helpers_source
    assert 'os.environ.get("OPENCUT_FFMPEG_DIR"' in helpers_source
    assert "_bundled_binary_path" in helpers_source

    data_dir = tmp_path / "opencut-data"
    env = os.environ.copy()
    env["OPENCUT_HOME"] = str(data_dir)
    probe = subprocess.run(
        [sys.executable, "-c", "import opencut.helpers as h; print(h.OPENCUT_DIR)"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert Path(probe.stdout.strip()) == data_dir

    import opencut.helpers as helpers

    monkeypatch.setattr(helpers, "_bundled_binary_path", lambda binary: f"/bundled/{binary}")
    monkeypatch.setattr(helpers, "_require_safe_media_binary", lambda binary: binary)
    monkeypatch.setattr(helpers.shutil, "which", lambda binary: f"/path/{binary}")
    monkeypatch.setattr(helpers, "_ffmpeg_path", None)
    monkeypatch.setattr(helpers, "_ffprobe_path", None)
    assert helpers.get_ffmpeg_path() == "/bundled/ffmpeg"
    assert helpers.get_ffprobe_path() == "/bundled/ffprobe"


def _fake_media_binaries(folder):
    folder.mkdir(parents=True)
    found = {}
    for binary in ("ffmpeg", "ffprobe"):
        exe = folder / (f"{binary}.exe" if os.name == "nt" else binary)
        exe.write_bytes(b"")
        exe.chmod(0o755)
        found[binary] = os.path.normcase(os.path.abspath(exe))
    return found


def test_ffmpeg_dir_names_the_folder_that_holds_the_binaries(monkeypatch, tmp_path):
    """The launchers and the build script set OPENCUT_FFMPEG_DIR to .../OpenCut-Server/ffmpeg."""
    import opencut.helpers as helpers

    media = _fake_media_binaries(tmp_path / "OpenCut-Server" / "ffmpeg")
    monkeypatch.setenv("OPENCUT_FFMPEG_DIR", str(tmp_path / "OpenCut-Server" / "ffmpeg"))
    for binary, expected in media.items():
        assert os.path.normcase(helpers._bundled_binary_path(binary) or "") == expected


def test_ffmpeg_dir_still_accepts_a_parent_with_an_ffmpeg_folder(monkeypatch, tmp_path):
    import opencut.helpers as helpers

    media = _fake_media_binaries(tmp_path / "OpenCut-Server" / "ffmpeg")
    monkeypatch.setenv("OPENCUT_FFMPEG_DIR", str(tmp_path / "OpenCut-Server"))
    for binary, expected in media.items():
        assert os.path.normcase(helpers._bundled_binary_path(binary) or "") == expected


def test_ffmpeg_dir_parent_never_resolves_to_the_ffmpeg_folder_itself(monkeypatch, tmp_path):
    """A parent layout holds a folder named like the binary; only the binary inside may win.

    Export tests patch os.path.isfile to True, and Windows os.access accepts a
    folder, so probing <dir>/ffmpeg before <dir>/ffmpeg/ffmpeg handed the
    folder to subprocess.
    """
    import opencut.helpers as helpers

    media = _fake_media_binaries(tmp_path / "OpenCut-Server" / "ffmpeg")
    monkeypatch.setenv("OPENCUT_FFMPEG_DIR", str(tmp_path / "OpenCut-Server"))
    real_isfile = os.path.isfile
    monkeypatch.setattr(os.path, "isfile", lambda path: os.path.isdir(path) or real_isfile(path))
    assert os.path.normcase(helpers._bundled_binary_path("ffmpeg") or "") == media["ffmpeg"]


def test_build_script_builds_appdir_appimage_and_flatpak_bundle():
    text = _read(BUILD_SCRIPT)

    assert "dist/OpenCut-Server" in text
    assert "desktop-file-validate" in text
    assert "appstreamcli validate --no-net" in text
    assert "appimagetool" in text
    assert "flatpak-builder" in text
    assert "flatpak build-bundle" in text
    assert "OPENCUT_FFMPEG_DIR" in text
    assert "verify_ffmpeg_provenance.py" in text
    assert '--ffprobe "$DIST_DIR/ffmpeg/ffprobe"' in text
    assert '--ffmpeg-provenance "$FFMPEG_PROVENANCE"' in text
    assert "https://flathub.org/repo/flathub.flatpakrepo" in text
    assert "$APP_ID-$VERSION.flatpak" in text
    assert "$APP_NAME-$VERSION-$ARCH.AppImage" in text


def test_build_script_has_lf_line_endings_and_executable_bit():
    raw = BUILD_SCRIPT.read_bytes()
    assert b"\r\n" not in raw

    on_disk_exec = bool(BUILD_SCRIPT.stat().st_mode & stat.S_IXUSR)
    if on_disk_exec:
        return
    result = subprocess.run(
        ["git", "ls-files", "--stage", "scripts/build_linux_packages.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    first_field = (result.stdout.split("\t", 1)[0] or "").split()
    mode = first_field[0] if first_field else ""
    assert mode == "100755"


def test_release_smoke_includes_linux_packaging_contract():
    text = _read(RELEASE_SMOKE)

    assert '"tests/test_linux_distribution_packaging.py"' in text


def test_docs_record_official_flatpak_and_appimage_boundaries():
    text = _read(REPO_ROOT / "docs" / "LINUX_DISTRIBUTION.md")

    assert "AppDir" in text
    assert re.search(r"org\.freedesktop\.Platform.*25\.08", text, re.DOTALL)
