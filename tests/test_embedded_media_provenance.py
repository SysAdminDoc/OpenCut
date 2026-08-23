"""Security gates for FFmpeg copies embedded by OpenCV and PyAV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opencut.core import embedded_media_provenance as provenance

FIXED_LIBRARIES = {
    "avcodec": (62, 28, 102),
    "avformat": (62, 12, 102),
    "avutil": (60, 26, 102),
}


def _runtime_fixture(*, ok: bool = True, opencv_status: str = "disabled") -> dict:
    return {
        "ok": ok,
        "errors": [] if ok else ["PyAV FFmpeg is below the security floor"],
        "security": {
            "cve": "CVE-2026-8461",
            "fixed_ffmpeg": "8.1.2",
        },
        "opencv": {
            "installed": True,
            "distribution": "opencv-python",
            "version": "4.14.0.94",
            "status": opencv_status,
            "ok": True,
        },
        "pyav": {
            "installed": True,
            "distribution": "av",
            "version": "18.1.0",
            "status": "verified",
            "ok": ok,
        },
    }


def test_ffmpeg_812_library_floor_is_the_executable_contract():
    assert provenance.FIXED_FFMPEG_VERSION == "8.1.2"
    assert provenance.FIXED_LIBRARY_FLOORS == FIXED_LIBRARIES

    fixed = provenance.grade_library_versions(FIXED_LIBRARIES)
    assert fixed["ok"] is True
    assert fixed["below_floor"] == {}


def test_library_floor_rejects_the_opencv_5_windows_abi():
    result = provenance.grade_library_versions(
        {
            "avcodec": (61, 19, 100),
            "avformat": (61, 7, 100),
            "avutil": (59, 39, 100),
        }
    )

    assert result["ok"] is False
    assert set(result["below_floor"]) == {"avcodec", "avformat", "avutil"}


def test_library_floor_fails_closed_when_a_library_is_not_reported():
    result = provenance.grade_library_versions({"avcodec": (62, 28, 102)})

    assert result["ok"] is False
    assert result["missing"] == ["avformat", "avutil"]


def test_parse_opencv_build_information_captures_real_abi_values():
    parsed = provenance.parse_opencv_build_information(
        """
        Video I/O:
          FFMPEG:                      YES (prebuilt binaries)
            avcodec:                   YES (61.19.100)
            avformat:                  YES (61.7.100)
            avutil:                    YES (59.39.100)
            swscale:                   YES (8.3.100)
        """
    )

    assert parsed == {
        "enabled": True,
        "build": "prebuilt binaries",
        "libraries": {
            "avcodec": (61, 19, 100),
            "avformat": (61, 7, 100),
            "avutil": (59, 39, 100),
        },
    }


def test_windows_guard_disables_opencv_ffmpeg_before_cv2_import(monkeypatch):
    monkeypatch.delenv(provenance.OPENCV_FFMPEG_PRIORITY_ENV, raising=False)

    result = provenance.install_runtime_guard(
        platform_name="win32",
        installed_versions={"opencv-python": "4.14.0.94", "av": "18.1.0"},
        loaded_cv2=None,
    )

    assert result["opencv_ffmpeg"] == "disabled"
    assert provenance.os.environ[provenance.OPENCV_FFMPEG_PRIORITY_ENV] == "0"


def test_package_sets_windows_priority_before_eager_core_imports():
    package_source = (
        Path(__file__).resolve().parents[1] / "opencut" / "__init__.py"
    ).read_text(encoding="utf-8")

    priority_assignment = 'os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"'
    core_import = "from opencut.core.embedded_media_provenance import"
    assert package_source.index(priority_assignment) < package_source.index(core_import)


def test_non_windows_guard_allows_only_the_reviewed_opencv_wheel(monkeypatch):
    monkeypatch.delenv(provenance.OPENCV_FFMPEG_PRIORITY_ENV, raising=False)

    allowed = provenance.install_runtime_guard(
        platform_name="linux",
        installed_versions={"opencv-python": "4.14.0.94", "av": "18.0.0"},
    )

    assert allowed["opencv_ffmpeg"] == "candidate"
    assert provenance.OPENCV_FFMPEG_PRIORITY_ENV not in provenance.os.environ

    blocked = provenance.install_runtime_guard(
        platform_name="linux",
        installed_versions={"opencv-python": "5.0.0.93", "av": "18.0.0"},
        loaded_cv2=None,
    )

    assert blocked["opencv_ffmpeg"] == "disabled"
    assert provenance.os.environ[provenance.OPENCV_FFMPEG_PRIORITY_ENV] == "0"


def test_runtime_guard_rejects_unreviewed_pyav_before_media_decode():
    with pytest.raises(RuntimeError, match="unreviewed PyAV"):
        provenance.install_runtime_guard(
            platform_name="linux",
            installed_versions={"opencv-python": "4.14.0.94", "av": "17.0.0"},
        )


def test_artifact_scan_hashes_embedded_decoder_files(tmp_path):
    plugin = tmp_path / "server" / "_internal" / "cv2" / "opencv_videoio_ffmpeg4140_64.dll"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"opencv decoder")
    pyav = tmp_path / "server" / "_internal" / "av.libs" / "avcodec-62-fixture.dll"
    pyav.parent.mkdir(parents=True)
    pyav.write_bytes(b"pyav decoder")

    records = provenance.scan_artifact_decoder_files([tmp_path], include_hashes=True)

    assert {record["provider"] for record in records} == {"opencv", "pyav"}
    expected = hashlib.sha256(b"opencv decoder").hexdigest()
    assert next(record for record in records if record["provider"] == "opencv")["sha256"] == expected
    assert all(not Path(record["path"]).is_absolute() for record in records)


def test_windows_release_rejects_an_opencv_ffmpeg_plugin(tmp_path):
    plugin = tmp_path / "OpenCut-Server" / "_internal" / "cv2" / "opencv_videoio_ffmpeg4140_64.dll"
    plugin.parent.mkdir(parents=True)
    plugin.write_bytes(b"unsafe")

    inventory = provenance.build_release_inventory(
        lane="windows",
        artifact_paths=[tmp_path],
        runtime_inventory=_runtime_fixture(),
    )

    assert inventory["ok"] is False
    assert any("opencv_videoio_ffmpeg" in error for error in inventory["errors"])


def test_windows_release_accepts_disabled_backend_when_plugin_is_absent(tmp_path):
    server = tmp_path / "OpenCut-Server.exe"
    server.write_bytes(b"server")

    inventory = provenance.build_release_inventory(
        lane="windows",
        artifact_paths=[server],
        runtime_inventory=_runtime_fixture(),
    )

    assert inventory["ok"] is True
    assert inventory["artifact_files"] == []


def test_non_windows_release_requires_verified_opencv_ffmpeg(tmp_path):
    inventory = provenance.build_release_inventory(
        lane="linux",
        artifact_paths=[tmp_path],
        runtime_inventory=_runtime_fixture(opencv_status="disabled"),
    )

    assert inventory["ok"] is False
    assert any("verified OpenCV FFmpeg" in error for error in inventory["errors"])


def test_manifest_writer_uses_lf_and_round_trips(tmp_path):
    destination = tmp_path / "embedded-media-provenance.json"
    payload = provenance.write_manifest(_runtime_fixture(), destination)

    assert payload == destination
    raw = destination.read_bytes()
    assert b"\r\n" not in raw
    assert json.loads(raw)["security"]["cve"] == "CVE-2026-8461"
