from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from opencut.tools import contrast_audit

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def _release_smoke_module():
    spec = importlib.util.spec_from_file_location("release_smoke_under_test", RELEASE_SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target(path: Path, foreground: str = "fg", background: str = "bg") -> contrast_audit.ContrastTarget:
    return contrast_audit.ContrastTarget(
        path,
        (
            contrast_audit.ContrastPair(
                foreground,
                background,
                minimum_ratio=contrast_audit.AA_NORMAL_TEXT,
                usage="fixture normal text",
            ),
        ),
    )


def test_contrast_ratio_matches_wcag_black_white_baseline():
    ratio = contrast_audit.contrast_ratio((0, 0, 0), (255, 255, 255))

    assert round(ratio, 2) == 21.0


def test_contrast_audit_passes_high_contrast_fixture(tmp_path):
    css = tmp_path / "style.css"
    css.write_text(":root { --fg: #ffffff; --bg: #111111; }\n", encoding="utf-8")

    report = contrast_audit.build_contrast_report((_target(css),), repo_root=tmp_path)

    assert report["status"] == "ok"
    assert report["summary"]["audited_pairs"] == 1
    assert report["findings"][0]["ratio"] >= contrast_audit.AA_NORMAL_TEXT


def test_contrast_audit_fails_low_contrast_fixture(tmp_path):
    css = tmp_path / "style.css"
    css.write_text(":root { --fg: #777777; --bg: #777777; }\n", encoding="utf-8")

    report = contrast_audit.build_contrast_report((_target(css),), repo_root=tmp_path)

    assert report["status"] == "fail"
    assert report["summary"]["failures"] == 1
    finding = report["findings"][0]
    assert finding["foreground"] == "fg"
    assert finding["background"] == "bg"
    assert finding["ratio"] < contrast_audit.AA_NORMAL_TEXT


def test_contrast_audit_includes_explicit_light_theme_tokens(tmp_path):
    css = tmp_path / "style.css"
    css.write_text(
        ":root { --fg: #ffffff; --bg: #111111; }\n"
        "html.theme-light { --fg: #eeeeee; --bg: #ffffff; }\n",
        encoding="utf-8",
    )

    report = contrast_audit.build_contrast_report((_target(css),), repo_root=tmp_path)

    assert report["status"] == "fail"
    assert report["summary"]["audited_pairs"] == 2
    assert report["summary"]["failures"] == 1
    assert report["findings"][1]["selector"] == "html.theme-light[1]"


def test_default_cep_and_uxp_theme_tokens_pass_wcag_gate():
    report = contrast_audit.build_contrast_report()

    assert report["status"] == "ok"
    assert report["summary"]["targets"] == 2
    assert report["summary"]["audited_pairs"] >= 30
    assert report["summary"]["failures"] == 0


def test_flagged_cep_control_colors_use_semantic_tokens():
    css = (
        REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "style.css"
    ).read_text(encoding="utf-8")

    for literal in (
        "color: #f4c7bd",
        "color: #fff5f2",
        "color: #fff1ec",
        "color: #f2d6a7",
        "color: #fff8ee",
        "color: #fff5e8",
        "color: #ffe2d9",
        "color: #fff8ed",
        "accent-color: #d4b17a",
    ):
        assert literal not in css
    for token in (
        "--text-danger-control",
        "--text-accent-control",
        "--text-warning-control-hover",
        "--text-action-hover",
        "--text-on-danger-surface",
    ):
        assert css.count(token) >= 3


def test_release_smoke_runs_contrast_audit_gate(monkeypatch):
    module = _release_smoke_module()
    payload = {"status": "ok", "summary": {"audited_pairs": 42, "failures": 0}, "findings": []}

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        assert cmd[:3] == [sys.executable, "-m", "opencut.tools.contrast_audit"]
        assert "--json" in cmd
        assert cwd == module.REPO_ROOT
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_contrast_audit(argparse.Namespace())

    assert result.status == "ok"
    assert result.message == "WCAG AA contrast token pairs passed (42 pairs)"
    assert any(step.name == "contrast-audit" for step in module.STEPS)
    assert "tests/test_contrast_audit.py" in module.RELEASE_GATE_TESTS

