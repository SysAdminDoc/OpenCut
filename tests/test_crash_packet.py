"""Tests for the crash + recovery diagnostic packet (F066)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from opencut.core import crash_packet as cp


def test_build_packet_produces_zip_with_environment(tmp_path):
    out = tmp_path / "packet.zip"
    result = cp.build_packet(output_path=out, include_jobs=False)

    assert Path(result.output_path).exists()
    with zipfile.ZipFile(result.output_path) as zf:
        names = sorted(zf.namelist())
        env_text = zf.read("environment.txt").decode("utf-8")

    assert "environment.txt" in names
    assert "manifest.json" in names
    assert "python_version" in env_text
    assert "platform" in env_text


def test_packet_records_per_entry_hash(tmp_path):
    out = tmp_path / "packet.zip"
    result = cp.build_packet(output_path=out, include_jobs=False)

    arcnames = {entry.arcname for entry in result.entries}
    assert "manifest.json" in arcnames
    assert all(len(entry.sha256) == 64 for entry in result.entries)


def test_packet_includes_crash_log_when_present(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_data = fake_home / ".opencut"
    fake_data.mkdir(parents=True)
    (fake_data / "crash.log").write_text("traceback line\nanother line\n", encoding="utf-8")
    (fake_data / "opencut.log").write_text("info line\n", encoding="utf-8")

    monkeypatch.setattr(cp, "_scrub_paths", lambda s: s)
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home) if p == "~" else p)

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        names = zf.namelist()
        assert "crash.log" in names
        assert "opencut.log" in names
        crash = zf.read("crash.log").decode("utf-8")
        assert "traceback line" in crash


def test_packet_scrubs_home_paths(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_data = fake_home / ".opencut"
    fake_data.mkdir(parents=True)
    (fake_data / "opencut.log").write_text(
        f"reading from {fake_home}/projects/clip.mp4\n", encoding="utf-8"
    )

    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home) if p == "~" else p)
    monkeypatch.setattr(cp, "_HOME" if hasattr(cp, "_HOME") else "DOES_NOT_EXIST", str(fake_home), raising=False)
    # Patch the issue_report symbols the module already imported.
    import re

    import opencut.core.issue_report as ir

    monkeypatch.setattr(ir, "_HOME", str(fake_home))
    monkeypatch.setattr(
        ir,
        "_HOME_PATTERNS",
        [re.compile(re.escape(str(fake_home)), re.IGNORECASE)],
    )

    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        if "opencut.log" in zf.namelist():
            log_text = zf.read("opencut.log").decode("utf-8")
            assert str(fake_home) not in log_text
            assert "~/projects/clip.mp4" in log_text


def test_packet_manifest_lists_only_pre_entries(tmp_path):
    """manifest.json should describe every file *other than itself*."""
    result = cp.build_packet(output_path=tmp_path / "packet.zip", include_jobs=False)
    with zipfile.ZipFile(result.output_path) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    arcnames = {entry["arcname"] for entry in manifest["entries"]}
    assert "manifest.json" not in arcnames
    assert manifest["version"] == cp.PACKET_VERSION


def test_route_smoke(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    out = tmp_path / "packet.zip"
    resp = client.post(
        "/system/crash-packet",
        json={
            "output_path": str(out),
            "log_tail_lines": 100,
            "crash_tail_bytes": 4000,
            "include_jobs": False,
        },
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    assert payload["bundle_sha256"]
    assert Path(payload["output_path"]).exists()


def test_route_requires_output_path(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/system/crash-packet",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
