"""Install.ps1 installs the pinned FFmpeg snapshot itself.

winget and Chocolatey only carry tagged releases, and the security floor
refuses every tagged release today, so the source installer downloads the
archive the packaged installers ship and checks it before use.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from functools import partial
from pathlib import Path

import pytest

from opencut.core import ffmpeg_provenance as fp

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "Install.ps1"

_RUNNER = r"""
param([string]$InstallScript, [string]$Destination, [string]$Url,
      [string]$ArchiveSha, [string]$ZipUrl, [string]$ZipSha,
      [string]$FfmpegSha, [string]$FfprobeSha, [string]$TarExe)
$ErrorActionPreference = "Stop"
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($InstallScript, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw "Install.ps1 does not parse" }
$fn = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Install-PinnedFFmpeg" }, $true) | Select-Object -First 1
. ([scriptblock]::Create($fn.Extent.Text))
Install-PinnedFFmpeg -Destination $Destination -ArchiveUrl $Url -ArchiveSha256 $ArchiveSha -ZipArchiveUrl $ZipUrl -ZipArchiveSha256 $ZipSha -TarPath $TarExe -BinarySha256 @{ "ffmpeg.exe" = $FfmpegSha; "ffprobe.exe" = $FfprobeSha }
"""

SYSTEM_TAR = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"

# What tar --version prints on Windows 10 and Windows 11 before 23H2. No
# liblzma, so it can't unpack the LZMA2 inside the pinned .7z.
OLD_WINDOWS_TAR = (
    "@echo off\r\n"
    'if "%~1"=="--version" (\r\n'
    "  echo bsdtar 3.5.2 - libarchive 3.5.2 zlib/1.2.5.f-ipp\r\n"
    "  exit /b 0\r\n"
    ")\r\n"
    '"%SystemRoot%\\System32\\tar.exe" %*\r\n'
    "exit /b %errorlevel%\r\n"
)


def _system_tar_reads_7z() -> bool:
    if not SYSTEM_TAR.is_file():
        return False
    completed = subprocess.run(
        [str(SYSTEM_TAR), "--version"], capture_output=True, text=True, timeout=30, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return "liblzma/" in completed.stdout


def _function_source() -> str:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    start = text.index("function Install-PinnedFFmpeg {")
    return text[start : text.index("\nfunction ", start + 1)]


def test_pinned_archive_defaults_match_the_provenance_pin():
    body = _function_source()
    assert f'$ArchiveUrl = "{fp.PINNED_INSTALLER_ARCHIVE_URL}"' in body
    assert f'$ArchiveSha256 = "{fp.PINNED_INSTALLER_ARCHIVE_SHA256}"' in body
    assert f'$ZipArchiveUrl = "{fp.PINNED_INSTALLER_ZIP_ARCHIVE_URL}"' in body
    assert f'$ZipArchiveSha256 = "{fp.PINNED_INSTALLER_ZIP_ARCHIVE_SHA256}"' in body
    pinned = dict(re.findall(r'"(ff\w+\.exe)"\s*=\s*"([0-9a-f]{64})"', body))
    assert pinned == fp.PINNED_INSTALLER_BINARY_SHA256


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass


@pytest.fixture
def archive_server(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_QuietHandler, directory=str(served))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield served, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _build_archive(folder: Path, binaries: dict[str, bytes]) -> Path:
    archive = folder / "ffmpeg-test-full_build.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, payload in binaries.items():
            bundle.writestr(f"ffmpeg-test-full_build/bin/{name}", payload)
        bundle.writestr("ffmpeg-test-full_build/doc/ffmpeg.exe.txt", b"not a binary")
    return archive


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_install(
    tmp_path, url, archive_sha, ffmpeg_sha, ffprobe_sha, destination,
    zip_url=None, zip_sha=None, tar=SYSTEM_TAR,
):
    shell = shutil.which("powershell")
    assert shell is not None
    runner = tmp_path / "run_install.ps1"
    runner.write_text(_RUNNER, encoding="ascii")
    return subprocess.run(
        [
            shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(runner),
            "-InstallScript", str(INSTALL_PS1),
            "-Destination", str(destination),
            "-Url", url,
            "-ArchiveSha", archive_sha,
            # Either archive works for the tests that don't pick, so a tar
            # without liblzma never reaches for the real download.
            "-ZipUrl", zip_url or url,
            "-ZipSha", zip_sha or archive_sha,
            "-TarExe", str(tar),
            "-FfmpegSha", ffmpeg_sha,
            "-FfprobeSha", ffprobe_sha,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="Install.ps1 runs under Windows PowerShell with Windows tar"
)

FFMPEG = b"MZ fake ffmpeg"
FFPROBE = b"MZ fake ffprobe"


@windows_only
def test_verified_archive_lands_both_binaries_and_the_marker(tmp_path, archive_server):
    served, base = archive_server
    archive = _build_archive(served, {"ffmpeg.exe": FFMPEG, "ffprobe.exe": FFPROBE})
    destination = tmp_path / "install" / "ffmpeg"

    completed = _run_install(
        tmp_path, f"{base}/{archive.name}", _sha(archive.read_bytes()),
        _sha(FFMPEG), _sha(FFPROBE), destination,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (destination / "ffmpeg.exe").read_bytes() == FFMPEG
    assert (destination / "ffprobe.exe").read_bytes() == FFPROBE
    assert (destination / ".opencut-installed").is_file()


@windows_only
def test_archive_with_the_wrong_hash_never_reaches_the_destination(tmp_path, archive_server):
    served, base = archive_server
    archive = _build_archive(served, {"ffmpeg.exe": FFMPEG, "ffprobe.exe": FFPROBE})
    destination = tmp_path / "install" / "ffmpeg"

    completed = _run_install(
        tmp_path, f"{base}/{archive.name}", "0" * 64,
        _sha(FFMPEG), _sha(FFPROBE), destination,
    )

    assert completed.returncode != 0
    assert "SHA-256" in completed.stdout + completed.stderr
    assert not destination.exists()


@windows_only
def test_swapped_binary_inside_a_matching_archive_is_refused(tmp_path, archive_server):
    served, base = archive_server
    archive = _build_archive(served, {"ffmpeg.exe": b"MZ swapped", "ffprobe.exe": FFPROBE})
    destination = tmp_path / "install" / "ffmpeg"

    completed = _run_install(
        tmp_path, f"{base}/{archive.name}", _sha(archive.read_bytes()),
        _sha(FFMPEG), _sha(FFPROBE), destination,
    )

    assert completed.returncode != 0
    assert "ffmpeg.exe in the FFmpeg archive" in completed.stdout + completed.stderr
    assert not destination.exists()


def _decoy(folder: Path, name: str) -> Path:
    """An archive that matches its hash but can't be unpacked."""
    decoy = folder / name
    decoy.write_bytes(b"not an archive at all")
    return decoy


@windows_only
def test_a_tar_without_liblzma_takes_the_zip(tmp_path, archive_server):
    served, base = archive_server
    sevenz = _decoy(served, "ffmpeg-test-full_build.7z")
    archive = _build_archive(served, {"ffmpeg.exe": FFMPEG, "ffprobe.exe": FFPROBE})
    old_tar = tmp_path / "old-windows-tar.cmd"
    old_tar.write_text(OLD_WINDOWS_TAR, encoding="ascii")
    destination = tmp_path / "install" / "ffmpeg"

    completed = _run_install(
        tmp_path, f"{base}/{sevenz.name}", _sha(sevenz.read_bytes()),
        _sha(FFMPEG), _sha(FFPROBE), destination,
        zip_url=f"{base}/{archive.name}", zip_sha=_sha(archive.read_bytes()), tar=old_tar,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (destination / "ffmpeg.exe").read_bytes() == FFMPEG


@pytest.mark.skipif(
    sys.platform != "win32" or not _system_tar_reads_7z(),
    reason="needs a Windows tar built with liblzma (Windows 11 23H2 or later)",
)
def test_a_tar_with_liblzma_keeps_the_smaller_7z(tmp_path, archive_server):
    served, base = archive_server
    root = tmp_path / "tree"
    (root / "ffmpeg-test-full_build" / "bin").mkdir(parents=True)
    (root / "ffmpeg-test-full_build" / "bin" / "ffmpeg.exe").write_bytes(FFMPEG)
    (root / "ffmpeg-test-full_build" / "bin" / "ffprobe.exe").write_bytes(FFPROBE)
    sevenz = served / "ffmpeg-test-full_build.7z"
    subprocess.run(
        [str(SYSTEM_TAR), "--format", "7zip", "--options", "7zip:compression=lzma2",
         "-cf", str(sevenz), "-C", str(root), "ffmpeg-test-full_build"],
        check=True, capture_output=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    zip_decoy = _decoy(served, "ffmpeg-test-full_build.zip")
    destination = tmp_path / "install" / "ffmpeg"

    completed = _run_install(
        tmp_path, f"{base}/{sevenz.name}", _sha(sevenz.read_bytes()),
        _sha(FFMPEG), _sha(FFPROBE), destination,
        zip_url=f"{base}/{zip_decoy.name}", zip_sha=_sha(zip_decoy.read_bytes()),
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (destination / "ffprobe.exe").read_bytes() == FFPROBE
