"""Tests for the capability probe + ``/system/capabilities`` route (F106)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from opencut.core import capability_profile as cp


def test_extract_version_handles_typical_ffmpeg_banner():
    banner = (
        "ffmpeg version 6.1.1-essentials_build-www.gyan.dev Copyright (c)\n"
        "  built with gcc 12.2.0 (Rev10)\n"
    )
    assert cp._extract_version(banner) == "6.1.1-essentials_build-www.gyan.dev"


def test_extract_version_returns_empty_on_no_match():
    assert cp._extract_version("") == ""
    assert cp._extract_version("nothing useful here") == ""


def test_probe_ffmpeg_handles_missing_binary(monkeypatch):
    monkeypatch.setattr(cp, "_resolve_ffmpeg_bin", lambda: None)
    payload = cp._probe_ffmpeg()
    assert payload["available"] is False
    assert payload["path"] == ""
    assert payload["encoders"] == []


def test_probe_ffmpeg_filters_to_allowlist(monkeypatch, tmp_path):
    """When ffmpeg returns a long list we surface only the curated names."""
    fake_bin = tmp_path / "ffmpeg"
    fake_bin.write_text("")

    def _fake_run(cmd, *args, **kwargs):
        if "-version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "ffmpeg version 6.1\n", "")
        if "-hwaccels" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "Hardware acceleration methods:\ncuda\nvideotoolbox\n", "")
        if "-encoders" in cmd:
            stdout = " V..... libx264              H.264\n V..... h264_nvenc           NVENC\n V..... obscure_codec        unused\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")
        if "-decoders" in cmd:
            stdout = " V..... h264                 H.264\n V..... mjpeg                MJPEG\n"
            return subprocess.CompletedProcess(cmd, 0, stdout, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cp, "_resolve_ffmpeg_bin", lambda: str(fake_bin))
    monkeypatch.setattr(cp, "_run_capturing", _fake_run)

    payload = cp._probe_ffmpeg()
    assert payload["available"] is True
    assert "libx264" in payload["encoders"]
    assert "h264_nvenc" in payload["encoders"]
    assert "obscure_codec" not in payload["encoders"]
    assert "cuda" in payload["hwaccel"]


def test_derive_findings_flags_cpu_only_environment():
    profile = cp.CapabilityProfile(
        ffmpeg={"available": True, "encoders": ["libx264"], "hwaccel": []},
        ffprobe={"available": True},
        gpu={"device": "cpu", "vram_total_mb": 0},
        disk={"temp": {"free_mb": 50000, "path": "/tmp"}},
    )
    findings = cp._derive_findings(profile)

    rules = {f.rule for f in findings}
    assert "cpu_only" in rules
    assert "no_hw_encoder" in rules


def test_derive_findings_warns_on_low_disk():
    profile = cp.CapabilityProfile(
        ffmpeg={"available": True, "encoders": ["libx264", "h264_nvenc"], "hwaccel": []},
        ffprobe={"available": True},
        gpu={"device": "cuda", "vram_total_mb": 16384},
        disk={"temp": {"free_mb": 512, "path": "/tmp"}},
    )
    findings = cp._derive_findings(profile)

    low = [f for f in findings if f.rule == "low_disk"]
    assert low and "512" in low[0].message


def test_derive_findings_flags_low_vram_only_when_gpu_detected():
    profile = cp.CapabilityProfile(
        ffmpeg={"available": True, "encoders": ["libx264", "h264_nvenc"], "hwaccel": []},
        ffprobe={"available": True},
        gpu={"device": "cuda", "vram_total_mb": 4096},
        disk={"temp": {"free_mb": 99999, "path": "/tmp"}},
    )
    findings = cp._derive_findings(profile)
    rules = {f.rule for f in findings}
    assert "low_vram" in rules
    assert "cpu_only" not in rules


def test_build_profile_returns_versioned_payload():
    payload = cp.build_profile()
    assert payload["version"] == 1
    assert set(payload).issuperset({"python", "ffmpeg", "ffprobe", "gpu", "disk", "findings"})


def test_route_returns_capability_payload(client):
    resp = client.get("/system/capabilities")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["version"] == 1
    for required in ("python", "ffmpeg", "gpu", "disk", "findings"):
        assert required in payload, f"missing field {required!r}"
