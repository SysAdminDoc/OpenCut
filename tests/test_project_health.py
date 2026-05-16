"""Tests for the local project + media health report (F011)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from opencut.core import project_health as ph


def test_build_report_flags_missing_root(tmp_path):
    report = ph.build_report(tmp_path / "does_not_exist")

    rules = {c.rule for c in report.checks}
    assert "project_root_missing" in rules
    assert report.error_count() >= 1


def test_build_report_scans_media_and_sidecars(tmp_path):
    (tmp_path / "render.mp4").write_bytes(b"\x00" * 65536)
    (tmp_path / "render.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    report = ph.build_report(tmp_path)

    assert report.media_count == 1
    assert report.sidecar_count == 2  # srt + json
    assert report.free_bytes > 0


def test_build_report_flags_zero_byte_media(tmp_path):
    bad = tmp_path / "bad.mov"
    bad.write_bytes(b"")

    report = ph.build_report(tmp_path)

    findings = [c for c in report.checks if c.rule == "media_empty"]
    assert findings and findings[0].path == str(bad)


def test_build_report_flags_suspiciously_small_media(tmp_path):
    small = tmp_path / "small.mov"
    small.write_bytes(b"\x00" * 64)  # < 4 KB

    report = ph.build_report(tmp_path)

    findings = [c for c in report.checks if c.rule == "media_suspiciously_small"]
    assert findings


def test_build_report_flags_stale_sidecar(tmp_path):
    import os

    media = tmp_path / "clip.mp4"
    sidecar = tmp_path / "clip.en.srt"
    sidecar.write_text("placeholder", encoding="utf-8")
    media.write_bytes(b"\x00" * 65536)

    # Force the sidecar mtime to be a clean 5 minutes older than media so the
    # +1 second slack in the check doesn't absorb our delta on coarse FS
    # clocks (e.g. FAT32 / VMware HGFS).
    media_mtime = media.stat().st_mtime
    os.utime(sidecar, (media_mtime - 300, media_mtime - 300))

    report = ph.build_report(tmp_path)

    findings = [c for c in report.checks if c.rule == "stale_sidecar"]
    assert findings and findings[0].path == str(sidecar)


def test_build_report_warns_on_low_free_space(tmp_path, monkeypatch):
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 8192)

    def _fake_free(_):
        return 16 * 1024 * 1024, 1024 * 1024 * 1024  # 16 MB free

    monkeypatch.setattr(ph, "_free_space", _fake_free)

    report = ph.build_report(tmp_path, min_free_mb=2048)
    rules = {c.rule for c in report.checks}
    assert "low_free_space" in rules


def test_build_report_flags_explicit_missing_media(tmp_path):
    report = ph.build_report(tmp_path, media_paths=[str(tmp_path / "nope.mp4")])
    findings = [c for c in report.checks if c.rule == "media_missing"]
    assert findings


def test_build_report_dict_round_trips(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"\x00" * 8192)
    report = ph.build_report(tmp_path)
    payload = report.as_dict()

    assert payload["project_root"] == str(tmp_path)
    assert "checks" in payload
    assert isinstance(payload["checks"], list)


def test_route_smoke(client, csrf_token, tmp_path):
    from tests.conftest import csrf_headers

    (tmp_path / "x.mp4").write_bytes(b"\x00" * 16384)

    resp = client.post(
        "/system/project-health",
        json={"project_root": str(tmp_path)},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["media_count"] == 1


def test_route_requires_project_root(client, csrf_token):
    from tests.conftest import csrf_headers

    resp = client.post(
        "/system/project-health",
        json={},
        headers=csrf_headers(csrf_token),
    )
    assert resp.status_code == 400
