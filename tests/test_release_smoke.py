"""Tests for the release smoke matrix runner (F098)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("release_smoke_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_steps_have_unique_names_and_callable_runners():
    module = load_module()
    names = [step.name for step in module.STEPS]

    assert len(names) == len(set(names)), "duplicate step names defined"
    for step in module.STEPS:
        assert callable(step.runner), f"step {step.name} runner not callable"


def test_step_result_serialises_to_dict():
    module = load_module()
    res = module.StepResult("demo", "ok", message="hi", duration_ms=42)
    payload = json.dumps(res.__dict__)
    parsed = json.loads(payload)
    assert parsed["name"] == "demo"
    assert parsed["status"] == "ok"
    assert parsed["duration_ms"] == 42
    assert "[PASS] demo (42 ms) — hi" in res.as_line()


def _stub_run(rc: int, out: str = "", err: str = ""):
    def _fake(cmd, *args, **kwargs):  # noqa: ANN001
        return subprocess.CompletedProcess(cmd, rc, out, err)
    return _fake


def test_step_version_sync_reports_ok_when_script_passes(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_run", _stub_run(0, out="OK"))

    result = module.step_version_sync(argparse.Namespace())

    assert result.status == "ok"
    assert result.exit_code == 0
    assert "aligned" in result.message


def test_step_version_sync_reports_failure_on_drift(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module, "_run", _stub_run(1, out="drift!"))

    result = module.step_version_sync(argparse.Namespace())

    assert result.status == "fail"
    assert result.exit_code == 1


def test_step_dependency_matrix_runs_contract_check(monkeypatch):
    module = load_module()
    calls = []

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"status":"ok"}', "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_dependency_matrix(argparse.Namespace())

    assert result.status == "ok"
    assert any("scripts/check_dependency_matrix.py" in " ".join(cmd) for cmd in calls)


def test_step_generated_docs_runs_all_doc_generators(monkeypatch):
    module = load_module()
    calls = []

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_generated_docs(argparse.Namespace())

    assert result.status == "ok"
    assert result.message == "6 generated-doc checks passed"
    joined = [" ".join(cmd).replace("\\", "/") for cmd in calls]
    assert any("opencut.tools.dump_project_facts --check" in cmd for cmd in joined)
    assert any("scripts/sync_badges.py --check" in cmd for cmd in joined)
    assert any("opencut.tools.dump_mcp_registry_manifest --check" in cmd for cmd in joined)
    assert any("opencut.tools.dump_mcp_extended_tools --check" in cmd for cmd in joined)
    assert any("opencut.tools.dump_model_cards --check" in cmd for cmd in joined)
    assert any("opencut.tools.dump_feature_readiness --check" in cmd for cmd in joined)


def test_step_generated_docs_reports_failed_generator(monkeypatch):
    module = load_module()

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        joined = " ".join(cmd)
        if "dump_mcp_extended_tools" in joined:
            return subprocess.CompletedProcess(cmd, 1, "", "extended drift")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_generated_docs(argparse.Namespace())

    assert result.status == "fail"
    assert result.exit_code == 1
    assert "generated-doc drift" in result.message
    assert "MCP extended tool catalogue failed" in result.stderr_tail
    assert "extended drift" in result.stderr_tail


def test_step_ruff_skips_when_binary_missing(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    result = module.step_ruff(argparse.Namespace())

    assert result.status == "skipped"
    assert "ruff" in result.skipped_reason


def test_step_pip_audit_skips_when_module_missing(monkeypatch):
    module = load_module()

    def _missing_module(cmd, *args, **kwargs):  # noqa: ANN001
        assert "opencut.tools.pip_audit_extras" in cmd
        return subprocess.CompletedProcess(cmd, 1, "", "No module named pip_audit")

    monkeypatch.setattr(module, "_run", _missing_module)

    result = module.step_pip_audit(argparse.Namespace())

    assert result.status == "skipped"
    assert "pip-audit" in result.skipped_reason


def test_step_pip_audit_runs_pyproject_all_wrapper(monkeypatch):
    module = load_module()

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        assert cmd[:3] == [sys.executable, "-m", "opencut.tools.pip_audit_extras"]
        assert "--extra" in cmd
        assert "all" in cmd
        payload = {
            "status": "ok",
            "message": "no unallowed advisories",
            "targets": [
                {
                    "name": "requirements.txt",
                    "vulnerability_count": 0,
                    "allowed_vulnerability_count": 0,
                    "unallowed_vulnerability_count": 0,
                },
                {
                    "name": "requirements-lock.txt",
                    "vulnerability_count": 0,
                    "allowed_vulnerability_count": 0,
                    "unallowed_vulnerability_count": 0,
                },
                {
                    "name": "pyproject[all]",
                    "vulnerability_count": 0,
                    "allowed_vulnerability_count": 0,
                    "unallowed_vulnerability_count": 0,
                },
            ],
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_pip_audit(argparse.Namespace())

    assert result.status == "ok"
    assert "pyproject[all]=0 unallowed/0 allowed" in result.message


def test_step_npm_advisory_requires_machine_parseable_json(monkeypatch, tmp_path):
    module = load_module()
    panel_dir = tmp_path / "panel"
    (panel_dir / "scripts").mkdir(parents=True)
    (panel_dir / "scripts" / "check-advisories.mjs").write_text("// ok", encoding="utf-8")
    (panel_dir / "node_modules").mkdir()

    monkeypatch.setattr(module, "PANEL_DIR", panel_dir)
    monkeypatch.setattr(module.shutil, "which", lambda name: "node" if name == "node" else None)

    def _fake_run(cmd, cwd=None, env=None):  # noqa: ANN001
        assert cmd[-1] == "--json"
        assert cwd == panel_dir
        payload = {"status": "ok", "summary": {"allowed": 1, "unwaived": 0}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps(payload), "")

    monkeypatch.setattr(module, "_run", _fake_run)

    result = module.step_npm_advisory(argparse.Namespace())

    assert result.status == "ok"
    assert result.message == "advisories on allow-list (1 allowed)"


def test_step_panel_rendered_runs_headless_browser_script(monkeypatch, tmp_path):
    module = load_module()
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    (panel_dir / "playwright.config.mjs").write_text("export default {};", encoding="utf-8")
    (panel_dir / "node_modules").mkdir()
    monkeypatch.setattr(module, "PANEL_DIR", panel_dir)
    calls = []

    def _fake_npm(*args, cwd):  # noqa: ANN001
        calls.append((args, cwd))
        return module.StepResult("panel-script", "ok", exit_code=0)

    monkeypatch.setattr(module, "_npm_command", _fake_npm)
    result = module.step_panel_rendered(argparse.Namespace())

    assert result.status == "ok"
    assert result.name == "panel-rendered"
    assert "headless CEP/UXP" in result.message
    assert calls == [(('run', 'test:rendered'), panel_dir)]


def test_step_npm_advisory_fails_on_unparseable_json(monkeypatch, tmp_path):
    module = load_module()
    panel_dir = tmp_path / "panel"
    (panel_dir / "scripts").mkdir(parents=True)
    (panel_dir / "scripts" / "check-advisories.mjs").write_text("// ok", encoding="utf-8")
    (panel_dir / "node_modules").mkdir()

    monkeypatch.setattr(module, "PANEL_DIR", panel_dir)
    monkeypatch.setattr(module.shutil, "which", lambda name: "node" if name == "node" else None)
    monkeypatch.setattr(module, "_run", _stub_run(0, out="[advisory-check] ok"))

    result = module.step_npm_advisory(argparse.Namespace())

    assert result.status == "fail"
    assert "parseable JSON" in result.message


def test_overall_status_handles_mixed_results():
    module = load_module()
    make = module.StepResult
    assert module.overall_status([make("a", "ok"), make("b", "skipped")]) == "ok"
    assert module.overall_status([make("a", "ok"), make("b", "fail")]) == "fail"
    assert module.overall_status([make("a", "skipped")]) == "skipped"
    assert module.overall_status([]) == "skipped"


def test_overall_status_strict_fails_on_skipped_critical_steps():
    module = load_module()
    make = module.StepResult
    results = [make("a", "ok"), make("ruff", "skipped", skipped_reason="ruff missing")]

    # Default behavior unchanged: skips never fail the run.
    assert module.overall_status(results) == "ok"
    # Strict mode: a skipped critical step is a failure.
    assert module.overall_status(results, strict=True) == "fail"
    # Strict mode with non-critical skips only: still ok.
    assert module.overall_status([make("a", "ok"), make("b", "skipped")], strict=True) == "ok"
    assert module.skipped_critical_steps(results) == ["ruff"]


def test_critical_step_names_exist_in_step_matrix():
    module = load_module()
    step_names = {step.name for step in module.STEPS}
    assert module.CRITICAL_STEP_NAMES <= step_names


def test_main_json_lists_skipped_critical_steps_and_strict_fails(monkeypatch, capsys):
    module = load_module()

    def _skipping_runner(_args):
        return module.StepResult("pip-audit", "skipped", skipped_reason="pip-audit missing")

    def _ok_runner(_args):
        return module.StepResult("noop", "ok", message="fine")

    monkeypatch.setattr(
        module,
        "STEPS",
        [
            module.StepDefinition("noop", _ok_runner),
            module.StepDefinition("pip-audit", _skipping_runner),
        ],
    )

    # Default: skip is prominent in JSON but does not fail the run.
    rc = module.main(["--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["strict"] is False
    assert payload["skipped_critical_steps"] == ["pip-audit"]

    # Strict: the same skip fails the run.
    rc = module.main(["--json", "--strict"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["status"] == "fail"
    assert payload["strict"] is True
    assert payload["skipped_critical_steps"] == ["pip-audit"]


def test_run_release_smoke_honours_only_filter(monkeypatch):
    module = load_module()
    sentinel: List[str] = []

    def _record(name: str):
        def runner(_args):
            sentinel.append(name)
            return module.StepResult(name, "ok", message="ok")
        return runner

    monkeypatch.setattr(
        module,
        "STEPS",
        [
            module.StepDefinition("alpha", _record("alpha")),
            module.StepDefinition("beta", _record("beta")),
        ],
    )

    results = module.run_release_smoke(
        argparse.Namespace(pytest_extra=[]),
        only=["beta"],
    )

    assert [r.name for r in results] == ["beta"]
    assert sentinel == ["beta"]


def test_main_emits_json_and_returns_zero_on_pass(monkeypatch, capsys):
    module = load_module()

    def _stub_runner(_args):
        return module.StepResult("noop", "ok", message="trivial pass")

    monkeypatch.setattr(
        module,
        "STEPS",
        [module.StepDefinition("noop", _stub_runner)],
    )

    rc = module.main(["--json"])

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["steps"][0]["name"] == "noop"


def test_main_returns_nonzero_when_any_step_fails(monkeypatch):
    module = load_module()

    def _stub_runner(_args):
        return module.StepResult("broken", "fail", exit_code=2, message="broken")

    monkeypatch.setattr(
        module,
        "STEPS",
        [module.StepDefinition("broken", _stub_runner)],
    )

    rc = module.main([])

    assert rc == 1
