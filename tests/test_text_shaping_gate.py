import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from opencut.tools import text_shaping_gate as gate

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def _release_smoke_module():
    spec = importlib.util.spec_from_file_location("release_smoke_under_test", RELEASE_SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner(version_stdout: str, filters_stdout: str):
    def _fake(cmd):
        if "-version" in cmd:
            return subprocess.CompletedProcess(cmd, 0, version_stdout, "")
        if "-filters" in cmd:
            return subprocess.CompletedProcess(cmd, 0, filters_stdout, "")
        raise AssertionError(f"unexpected command: {cmd}")

    return _fake


def test_ffmpeg_gate_requires_libass_harfbuzz_fribidi_and_exact_filters():
    version = (
        "ffmpeg version 8.0\n"
        "configuration: --enable-libass --enable-libharfbuzz --enable-libfribidi\n"
    )
    filters = (
        " ..C ass               V->V       Render ASS subtitles\n"
        " TS greyedge          V->V       grey edge assumption\n"
        " ..C subtitles         V->V       Render text subtitles\n"
    )

    check = gate.inspect_ffmpeg_text_shaping("ffmpeg", runner=_runner(version, filters))

    assert check.status == "ok"
    assert check.details["required_config_flags"] == {
        "libass": True,
        "libharfbuzz": True,
        "libfribidi": True,
    }
    assert check.details["required_filters"] == {"ass": True, "subtitles": True}


def test_ffmpeg_gate_fails_without_harfbuzz_even_when_ass_filter_exists():
    version = "configuration: --enable-libass --enable-libfribidi\n"
    filters = " ..C ass V->V Render ASS subtitles\n ..C subtitles V->V Render text subtitles\n"

    check = gate.inspect_ffmpeg_text_shaping("ffmpeg", runner=_runner(version, filters))

    assert check.status == "fail"
    assert "libharfbuzz" in check.details["missing_config_flags"]


def test_pillow_raqm_warning_can_be_promoted_to_failure(monkeypatch):
    monkeypatch.setattr(
        gate,
        "inspect_ffmpeg_text_shaping",
        lambda *args, **kwargs: gate.CapabilityCheck(
            "ffmpeg-libass",
            "ok",
            True,
            "ok",
        ),
    )
    monkeypatch.setattr(
        gate,
        "inspect_skia_text_shaping",
        lambda *, require_skia=False: gate.CapabilityCheck(
            "skia-shaping",
            "skipped",
            require_skia,
            "not installed",
        ),
    )

    def _fake_pillow(*, require_raqm=False):
        return gate.CapabilityCheck(
            "pillow-raqm",
            "fail" if require_raqm else "warning",
            require_raqm,
            "missing RAQM",
        )

    monkeypatch.setattr(gate, "inspect_pillow_text_shaping", _fake_pillow)

    advisory_report = gate.build_text_shaping_report()
    strict_report = gate.build_text_shaping_report(require_pillow_raqm=True)

    assert advisory_report["status"] == "ok"
    assert advisory_report["summary"]["warnings"] == 1
    assert strict_report["status"] == "fail"
    assert strict_report["summary"]["failures"] == 1


def test_release_smoke_runs_text_shaping_gate(monkeypatch):
    module = _release_smoke_module()
    payload = {"status": "ok", "summary": {"warnings": 1}, "checks": []}

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        assert cmd[:3] == [sys.executable, "-m", "opencut.tools.text_shaping_gate"]
        assert "--json" in cmd
        assert cwd == module.REPO_ROOT
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_text_shaping(argparse.Namespace())

    assert result.status == "ok"
    assert result.message == "hard shaping gates passed (1 advisory warnings)"
    assert any(step.name == "text-shaping" for step in module.STEPS)
    assert "tests/test_text_shaping_gate.py" in module.RELEASE_GATE_TESTS

