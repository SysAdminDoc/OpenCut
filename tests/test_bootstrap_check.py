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


def test_dev_import_check_reports_missing_test_tooling(monkeypatch):
    bootstrap_check = load_bootstrap_check()

    def fake_find_spec(module_name):
        return None if module_name in {"pytest", "ruff"} else object()

    monkeypatch.setattr(bootstrap_check.importlib.util, "find_spec", fake_find_spec)

    result = bootstrap_check.check_dev_imports({
        "pytest": "pytest",
        "pytest-cov": "pytest_cov",
        "ruff": "ruff",
    })

    assert result.ok is False
    assert result.name == "dev-imports"
    assert "pytest" in result.detail
    assert "ruff" in result.detail
    assert '".[dev]"' in result.hint


def test_python_version_floor_is_3_11():
    bootstrap_check = load_bootstrap_check()
    assert bootstrap_check.MIN_PYTHON == (3, 11)
    assert bootstrap_check.MAX_PYTHON == (3, 14)


def test_json_output_uses_aggregate_status(monkeypatch, capsys):
    bootstrap_check = load_bootstrap_check()

    monkeypatch.setattr(
        bootstrap_check,
        "run_checks",
        lambda metadata_only=False, dev=False: [
            bootstrap_check.CheckResult("one", True, "ok"),
            bootstrap_check.CheckResult("two", False, "bad", "fix it"),
        ],
    )

    exit_code = bootstrap_check.main(["--json", "--metadata-only"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["checks"][1]["hint"] == "fix it"


def test_json_output_can_include_dev_import_check(monkeypatch, capsys):
    bootstrap_check = load_bootstrap_check()
    calls = []

    def fake_run_checks(metadata_only=False, repo_root=None, dev=False):
        calls.append({"metadata_only": metadata_only, "dev": dev})
        return [bootstrap_check.CheckResult("dev-imports", True, "ok")]

    monkeypatch.setattr(bootstrap_check, "run_checks", fake_run_checks)

    exit_code = bootstrap_check.main(["--json", "--metadata-only", "--dev"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert calls == [{"metadata_only": True, "dev": True}]


# ---------------------------------------------------------------------------
# F181 — UV trampoline / broken Python fallback
# ---------------------------------------------------------------------------


def test_resolve_python_returns_sys_executable_when_working():
    bootstrap_check = load_bootstrap_check()
    # The current Python is obviously working (we're running it).
    resolved = bootstrap_check._resolve_python_for_subprocess()
    assert resolved
    assert isinstance(resolved, str)


def test_resolve_python_falls_back_when_sys_executable_broken(monkeypatch):
    """F181 — when ``sys.executable`` cannot spawn a child, try PATH.

    Simulate a UV trampoline by patching subprocess.run to raise
    FileNotFoundError on the sys.executable probe, then succeed on
    a different candidate from shutil.which.
    """
    bootstrap_check = load_bootstrap_check()
    import shutil as real_shutil
    import subprocess as real_subprocess

    fake_system_python = real_shutil.which("python") or real_shutil.which("python3")
    if not fake_system_python:
        # No system Python on PATH — skip the assertion-on-fallback test.
        import pytest
        pytest.skip("no system Python on PATH to use as F181 fallback target")

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        # First invocation is the sys.executable probe — pretend trampoline.
        if cmd[0] == sys.executable and len(calls) == 1:
            raise FileNotFoundError(2, "trampoline failed to spawn")
        # Subsequent invocations succeed.
        completed = real_subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout="", stderr=""
        )
        return completed

    monkeypatch.setattr(bootstrap_check.subprocess, "run", fake_run)
    resolved = bootstrap_check._resolve_python_for_subprocess()
    # We should NOT have returned the broken trampoline.
    assert resolved != sys.executable or resolved == fake_system_python


def test_check_version_sync_returns_actionable_hint_on_trampoline(monkeypatch):
    """F181 — broken Python trampoline must surface a clear remediation hint."""
    bootstrap_check = load_bootstrap_check()

    monkeypatch.setattr(
        bootstrap_check,
        "_resolve_python_for_subprocess",
        lambda: "/definitely/not/a/real/python",
    )

    def fake_run(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(bootstrap_check.subprocess, "run", fake_run)
    result = bootstrap_check.check_version_sync(REPO_ROOT)
    assert result.ok is False
    assert "UV trampoline" in result.hint
    assert "recreate" in result.hint.lower() or "venv" in result.hint.lower()


def test_check_version_sync_handles_timeout_gracefully(monkeypatch):
    bootstrap_check = load_bootstrap_check()

    monkeypatch.setattr(
        bootstrap_check,
        "_resolve_python_for_subprocess",
        lambda: sys.executable,
    )

    def fake_run(*args, **kwargs):
        raise bootstrap_check.subprocess.TimeoutExpired(cmd=["python"], timeout=60)

    monkeypatch.setattr(bootstrap_check.subprocess, "run", fake_run)
    result = bootstrap_check.check_version_sync(REPO_ROOT)
    assert result.ok is False
    assert "timed out" in result.detail.lower()
