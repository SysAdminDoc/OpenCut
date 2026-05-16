import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap_check.py"


def load_bootstrap_check():
    spec = importlib.util.spec_from_file_location("bootstrap_check_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lockfile_check_rejects_self_package(tmp_path):
    bootstrap_check = load_bootstrap_check()
    (tmp_path / "requirements-lock.txt").write_text(
        "Flask==3.1.3\nopencut==1.4.0\npytest==9.0.2\n",
        encoding="utf-8",
    )

    result = bootstrap_check.check_lockfile(tmp_path)

    assert result.ok is False
    assert result.name == "requirements-lock"
    assert "opencut==1.4.0" in result.detail


def test_lockfile_check_accepts_dependency_only_lock(tmp_path):
    bootstrap_check = load_bootstrap_check()
    (tmp_path / "requirements-lock.txt").write_text(
        "Flask==3.1.3\npytest==9.0.2\n",
        encoding="utf-8",
    )

    result = bootstrap_check.check_lockfile(tmp_path)

    assert result.ok is True
    assert "2 auditable dependency lines" in result.detail


def test_runtime_import_check_reports_missing_modules(monkeypatch):
    bootstrap_check = load_bootstrap_check()

    def fake_find_spec(module_name):
        return None if module_name == "flask" else object()

    monkeypatch.setattr(bootstrap_check.importlib.util, "find_spec", fake_find_spec)

    result = bootstrap_check.check_runtime_imports({
        "click": "click",
        "flask": "flask",
    })

    assert result.ok is False
    assert result.name == "runtime-imports"
    assert "flask" in result.detail


def test_json_output_uses_aggregate_status(monkeypatch, capsys):
    bootstrap_check = load_bootstrap_check()

    monkeypatch.setattr(
        bootstrap_check,
        "run_checks",
        lambda metadata_only=False: [
            bootstrap_check.CheckResult("one", True, "ok"),
            bootstrap_check.CheckResult("two", False, "bad", "fix it"),
        ],
    )

    exit_code = bootstrap_check.main(["--json", "--metadata-only"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["checks"][1]["hint"] == "fix it"
