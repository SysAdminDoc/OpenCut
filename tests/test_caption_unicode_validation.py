import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from opencut.core import caption_unicode_validation as unicode_gate

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def _release_smoke_module():
    spec = importlib.util.spec_from_file_location("release_smoke_under_test", RELEASE_SMOKE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_classifier_detects_rtl_mixed_bidi_cjk_and_indic():
    assert unicode_gate.classify_caption_text("OpenCut גרסה 1.32 מוכנה") == [
        "latin",
        "mixed_bidi",
        "rtl",
    ]
    assert "cjk" in unicode_gate.classify_caption_text("これは改行検証用の字幕です")
    assert "indic" in unicode_gate.classify_caption_text("नमस्ते दुनिया")


def test_cjk_line_breaker_limitation_is_explicit_f242_followup():
    assert unicode_gate.needs_cjk_line_breaker("これは改行検証用の長い字幕テキストです")
    assert not unicode_gate.needs_cjk_line_breaker("これは 改行 検証")
    assert not unicode_gate.needs_cjk_line_breaker("Hello world")


def test_default_unicode_validation_report_preserves_all_export_text():
    report = unicode_gate.build_caption_unicode_report()

    assert report["status"] == "ok"
    assert report["summary"]["case_count"] == 5
    assert report["summary"]["complex_shaping_cases"] >= 3
    assert report["summary"]["cjk_cases"] == 2
    assert report["summary"]["failures"] == 0
    assert report["summary"]["warnings"] == 0
    assert report["known_followups"]["F241"]
    assert report["known_followups"]["F242"]

    cases = {case["case_id"]: case for case in report["cases"]}
    assert set(cases) == {
        "arabic_rtl",
        "hebrew_latin_bidi",
        "hindi_devanagari",
        "japanese_no_space",
        "chinese_no_space",
    }
    for case in cases.values():
        assert case["srt_roundtrip"] is True
        assert case["ass_roundtrip"] is True
        assert case["burnin_ass_roundtrip"] is True
        assert case["utf8_without_bom"] is True
        assert case["cjk_line_break_supported"] is True
        assert not case["failures"]

    assert cases["japanese_no_space"]["cjk_line_break_required"] is True
    assert cases["japanese_no_space"]["cjk_line_break_supported"] is True
    assert cases["chinese_no_space"]["cjk_line_break_required"] is True
    assert cases["chinese_no_space"]["cjk_line_break_supported"] is True
    assert cases["arabic_rtl"]["requires_complex_shaping"] is True


def test_cli_json_and_check_modes(capsys):
    from opencut.tools import caption_unicode_validation as cli

    assert cli.main(["--check"]) == 0
    capsys.readouterr()
    assert cli.main(["--json", "--check"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["summary"]["case_count"] == 5


def test_release_smoke_runs_caption_unicode_gate(monkeypatch):
    module = _release_smoke_module()
    payload = {
        "status": "ok",
        "summary": {"case_count": 5, "warnings": 0, "failures": 0},
        "cases": [],
    }

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        assert cmd[:3] == [sys.executable, "-m", "opencut.tools.caption_unicode_validation"]
        assert "--json" in cmd
        assert "--check" in cmd
        assert cwd == module.REPO_ROOT
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_caption_unicode(argparse.Namespace())

    assert result.status == "ok"
    assert result.message == "5 complex-script fixtures preserved (0 advisory warnings)"
    assert any(step.name == "caption-unicode" for step in module.STEPS)
    assert "tests/test_caption_unicode_validation.py" in module.RELEASE_GATE_TESTS
