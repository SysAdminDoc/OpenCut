"""Fail-closed FFmpeg runtime and distribution security-floor coverage."""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

from opencut import helpers
from opencut import checks
from opencut import preflight
from opencut.core import capability_profile as cp
from opencut.core import ffmpeg_provenance as fp


REPO_ROOT = Path(__file__).resolve().parents[1]
SAFE_BANNER = "ffmpeg version 8.1.2-essentials_build-www.gyan.dev\n"
UNSAFE_BANNER = "ffmpeg version 8.1.1-essentials_build-www.gyan.dev\n"


@pytest.fixture(autouse=True)
def _reset_helper_caches():
    previous_ffmpeg = helpers._ffmpeg_path
    previous_ffprobe = helpers._ffprobe_path
    previous_security = dict(helpers._media_binary_security_cache)
    helpers._ffmpeg_path = None
    helpers._ffprobe_path = None
    helpers._media_binary_security_cache.clear()
    yield
    helpers._ffmpeg_path = previous_ffmpeg
    helpers._ffprobe_path = previous_ffprobe
    helpers._media_binary_security_cache.clear()
    helpers._media_binary_security_cache.update(previous_security)


def test_shared_resolver_rejects_release_811(monkeypatch, tmp_path):
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"unsafe")
    grade = fp.check_security_floor(UNSAFE_BANNER)
    monkeypatch.setattr(helpers.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(
        fp,
        "require_security_floor",
        lambda path: (_ for _ in ()).throw(fp.FfmpegSecurityError(path, grade)),
    )

    with pytest.raises(fp.FfmpegSecurityError, match="8.1.2"):
        helpers.get_ffmpeg_path()


def test_shared_resolver_accepts_and_caches_release_812(monkeypatch, tmp_path):
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"safe")
    calls = []
    monkeypatch.setattr(helpers.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(
        fp,
        "require_security_floor",
        lambda path: calls.append(path) or fp.check_security_floor(SAFE_BANNER),
    )

    assert helpers.get_ffmpeg_path() == str(binary)
    assert helpers.get_ffmpeg_path() == str(binary)
    assert calls == [str(binary)]

    binary.write_bytes(b"safe-but-replaced")
    assert helpers.get_ffmpeg_path() == str(binary)
    assert calls == [str(binary), str(binary)]


def test_media_processing_is_blocked_before_subprocess_start(monkeypatch, tmp_path):
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"unsafe")
    grade = fp.check_security_floor(UNSAFE_BANNER)
    subprocess_started = False

    monkeypatch.setattr(helpers.shutil, "which", lambda _name: str(binary))
    monkeypatch.setattr(
        fp,
        "require_security_floor",
        lambda path: (_ for _ in ()).throw(fp.FfmpegSecurityError(path, grade)),
    )

    def _unexpected_run(*_args, **_kwargs):
        nonlocal subprocess_started
        subprocess_started = True
        raise AssertionError("unsafe media binary reached subprocess.run")

    monkeypatch.setattr(helpers._sp, "run", _unexpected_run)
    with pytest.raises(fp.FfmpegSecurityError):
        helpers.run_ffmpeg(["ffmpeg", "-version"])
    assert subprocess_started is False


def test_shared_runner_guards_literal_ffprobe(monkeypatch):
    observed = []
    monkeypatch.setattr(helpers, "get_ffprobe_path", lambda: "verified-ffprobe")
    monkeypatch.setattr(
        helpers._sp,
        "run",
        lambda cmd, **_kwargs: observed.append(cmd) or subprocess.CompletedProcess(cmd, 0, b"", b""),
    )

    helpers.run_ffmpeg(["ffprobe", "-version"])
    assert observed[0][0] == "verified-ffprobe"


def test_feature_checks_and_preflight_report_unsafe_ffmpeg_unavailable(monkeypatch):
    grade = fp.check_security_floor(UNSAFE_BANNER)
    monkeypatch.setattr(
        helpers,
        "get_ffmpeg_path",
        lambda: (_ for _ in ()).throw(fp.FfmpegSecurityError("ffmpeg", grade)),
    )

    assert checks.ffmpeg_security_available() is False
    result = preflight._probe_check("ffmpeg", True)
    assert result["ok"] is False
    assert "8.1.2+" in result["fix"]


def test_capability_probe_reports_unsafe_binary_unavailable(monkeypatch, tmp_path):
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"unsafe")

    def _fake_run(cmd, *_args, **_kwargs):
        if "-version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, UNSAFE_BANNER, "")
        raise AssertionError("unsafe FFmpeg must not be probed for codecs or acceleration")

    monkeypatch.setattr(cp, "_resolve_ffmpeg_bin", lambda: str(binary))
    monkeypatch.setattr(cp, "_run_capturing", _fake_run)

    payload = cp._probe_ffmpeg()
    assert payload["detected"] is True
    assert payload["available"] is False
    assert payload["blocked_reason"] == "security_floor"
    assert payload["security"]["version"].startswith("8.1.1")
    findings = cp._derive_findings(cp.CapabilityProfile(ffmpeg=payload))
    blocked = [item for item in findings if item.rule == "ffmpeg_below_security_floor"]
    assert blocked and blocked[0].severity == "error"
    assert "CVE-2026-8461" in blocked[0].message


def test_no_core_subprocess_bypasses_the_shared_ffmpeg_resolver():
    bypasses = []
    for path in (REPO_ROOT / "opencut").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            first_arg = node.args[0]
            if not isinstance(first_arg, (ast.List, ast.Tuple)) or not first_arg.elts:
                continue
            executable = first_arg.elts[0]
            if isinstance(executable, ast.Constant) and executable.value in {"ffmpeg", "ffprobe"}:
                func_name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if func_name in {"run", "Popen", "check_call", "check_output"}:
                    bypasses.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert bypasses == []


def test_distribution_lanes_pin_and_verify_the_same_floor():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    install_py = (REPO_ROOT / "install.py").read_text(encoding="utf-8")
    install_ps1 = (REPO_ROOT / "Install.ps1").read_text(encoding="utf-8")
    builder = (REPO_ROOT / "installer" / "InstallerBuilder.ps1").read_text(encoding="utf-8")

    assert "FFMPEG_VERSION=8.1.2" in dockerfile
    assert "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c" in dockerfile
    assert "verify_ffmpeg_provenance.py /opt/ffmpeg/bin/ffmpeg" in dockerfile
    assert "    ffmpeg \\\n" not in dockerfile
    assert "probe_binary_security" in install_py
    assert "Continue without FFmpeg" not in install_py
    assert "-SkipFFmpeg skips auto-install only; it cannot bypass" in install_ps1
    assert "verify_ffmpeg_provenance.py" in builder


def test_source_installer_rejects_unsafe_ffmpeg(monkeypatch):
    source_installer = importlib.import_module("install")
    monkeypatch.setattr(source_installer.shutil, "which", lambda _name: "unsafe-ffmpeg")
    monkeypatch.setattr(
        fp,
        "probe_binary_security",
        lambda *_args, **_kwargs: fp.check_security_floor(UNSAFE_BANNER),
    )

    with pytest.raises(SystemExit) as exc:
        source_installer.check_ffmpeg()
    assert exc.value.code == 1


def test_source_installer_accepts_release_812(monkeypatch, capsys):
    source_installer = importlib.import_module("install")
    monkeypatch.setattr(source_installer.shutil, "which", lambda _name: "safe-ffmpeg")
    monkeypatch.setattr(
        fp,
        "probe_binary_security",
        lambda *_args, **_kwargs: fp.check_security_floor(SAFE_BANNER),
    )

    source_installer.check_ffmpeg()
    assert "clears the security floor" in capsys.readouterr().out
