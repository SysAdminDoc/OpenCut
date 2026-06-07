"""F263 — pip-audit must cover optional extras, not only requirements.txt."""

from __future__ import annotations

import subprocess
from pathlib import Path

from opencut.tools import pip_audit_extras

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_all_target_includes_base_and_optional_dependencies():
    requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "all",
    )

    assert "flask>=3.0,<4" in requirements
    assert "faster-whisper>=1.1,<2" in requirements
    assert "otio-aaf-adapter>=2.0,<3" in requirements
    advisory_heavy_prefixes = (
        "audiocraft",
        "demucs",
        "gfpgan",
        "pyannote.audio",
        "realesrgan",
        "resemble-enhance",
        "torch",
        "torchvision",
        "transformers",
        "transnetv2-pytorch",
        "whisperx",
    )
    for prefix in advisory_heavy_prefixes:
        assert not any(req.startswith(prefix) for req in requirements)
    assert len(requirements) == len(set(req.lower() for req in requirements))


def test_auto_editor_extra_tracks_available_v29_pip_line():
    requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "auto-edit",
    )

    assert "auto-editor>=29.3,<30" in requirements
    assert all("auto-editor>=24.0,<25" not in req for req in requirements)


def test_audiocraft_stays_separate_from_combined_all_extra():
    music_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "music",
    )
    all_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "all",
    )

    assert "audiocraft>=1.3,<2; python_version < '3.12'" in music_requirements
    assert not any(req.startswith("audiocraft") for req in all_requirements)


def test_resemble_enhance_stays_separate_from_combined_all_extra():
    enhance_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "enhance",
    )
    all_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "all",
    )

    assert "resemble-enhance>=0.0.1,<1; python_version < '3.12'" in enhance_requirements
    assert not any(req.startswith("resemble-enhance") for req in all_requirements)


def test_whisperx_stays_separate_from_combined_all_extra():
    whisperx_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "captions-whisperx",
    )
    all_requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "all",
    )

    assert "whisperx>=3.8.5,<4" in whisperx_requirements
    assert not any(req.startswith("whisperx") for req in all_requirements)


def test_torch_stack_collects_advisory_heavy_backends():
    requirements = pip_audit_extras.load_pyproject_requirements(
        REPO_ROOT / "pyproject.toml",
        "torch-stack",
    )

    expected = {
        "whisperx>=3.8.5,<4",
        "demucs>=4.0,<5",
        "realesrgan>=0.3,<1",
        "gfpgan>=1.3,<2",
        "pyannote.audio>=4.0,<5",
        "transnetv2-pytorch>=1.0.5,<2",
        "torch>=2.6",
        "torchvision>=0.21",
        "transformers>=4.30",
    }
    assert expected.issubset(set(requirements))


def test_default_targets_include_requirements_and_pyproject_all():
    targets = pip_audit_extras.build_targets()

    assert [target.name for target in targets] == ["requirements.txt", "requirements-lock.txt", "pyproject[all]"]
    assert targets[0].kind == "requirements"
    assert targets[1].kind == "lockfile"
    assert targets[1].no_deps is True
    assert targets[2].kind == "pyproject-extra"
    assert targets[2].extra == "all"
    assert any(req.startswith("idna==") for req in targets[1].requirements)
    assert len(targets[2].requirements) > len(targets[0].requirements)


def test_lockfile_target_can_be_disabled_for_diagnostics():
    targets = pip_audit_extras.build_targets(include_lockfile=False)

    assert [target.name for target in targets] == ["requirements.txt", "pyproject[all]"]
    assert all(target.kind != "lockfile" for target in targets)


def test_pyproject_extra_targets_can_be_disabled_for_diagnostics():
    targets = pip_audit_extras.build_targets(extras=())

    assert [target.name for target in targets] == ["requirements.txt", "requirements-lock.txt"]
    assert all(target.kind != "pyproject-extra" for target in targets)


def test_cli_no_extras_audits_only_committed_requirements(monkeypatch):
    captured_targets = []

    def _fake_run_audits(targets, **kwargs):  # noqa: ANN001, ARG001
        captured_targets.extend(targets)
        return {
            "allowed_vulnerability_count": 0,
            "message": "ok",
            "status": "ok",
            "target_count": len(targets),
            "targets": [],
            "unallowed_vulnerability_count": 0,
            "vulnerability_count": 0,
        }

    monkeypatch.setattr(pip_audit_extras, "run_audits", _fake_run_audits)

    assert pip_audit_extras.cli(["--json", "--no-extras"]) == 0
    assert [target.name for target in captured_targets] == ["requirements.txt", "requirements-lock.txt"]


def test_all_extras_cli_target_builder_knows_every_optional_extra():
    extras = pip_audit_extras._optional_extra_names(REPO_ROOT / "pyproject.toml")
    targets = pip_audit_extras.build_targets(
        extras=extras,
        include_requirements=False,
        include_lockfile=False,
    )

    assert "all" in extras
    assert {target.extra for target in targets} == set(extras)
    assert all(target.name.startswith("pyproject[") for target in targets)


def test_run_audits_reports_per_target_advisory_state(monkeypatch):
    target = pip_audit_extras.AuditTarget(
        name="pyproject[all]",
        kind="pyproject-extra",
        extra="all",
        source="pyproject.toml",
        requirements=["flask>=3.0,<4"],
    )

    monkeypatch.setattr(
        pip_audit_extras.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip_audit" else None,
    )

    def _fake_run(cmd, cwd, timeout, env=None):  # noqa: ANN001
        assert "-r" in cmd
        assert "--progress-spinner" in cmd
        assert "--cache-dir" in cmd
        assert env and env.get("PIP_CACHE_DIR")
        payload = {
            "dependencies": [
                {"name": "flask", "version": "3.1.3", "vulns": []},
            ],
            "fixes": [],
        }
        return subprocess.CompletedProcess(cmd, 0, pip_audit_extras.json.dumps(payload), "")

    monkeypatch.setattr(pip_audit_extras, "_run", _fake_run)

    result = pip_audit_extras.run_audits([target])

    assert result["status"] == "ok"
    assert result["target_count"] == 1
    assert result["targets"][0]["name"] == "pyproject[all]"
    assert result["targets"][0]["resolved_dependency_count"] == 1
    assert result["targets"][0]["vulnerability_count"] == 0
    assert result["targets"][0]["unallowed_vulnerability_count"] == 0


def test_run_audits_uses_no_deps_for_lockfile_targets(monkeypatch):
    target = pip_audit_extras.AuditTarget(
        name="requirements-lock.txt",
        kind="lockfile",
        source="requirements-lock.txt",
        requirements=["idna==3.16"],
        no_deps=True,
    )

    monkeypatch.setattr(
        pip_audit_extras.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip_audit" else None,
    )

    def _fake_run(cmd, cwd, timeout, env=None):  # noqa: ANN001
        assert "--no-deps" in cmd
        payload = {
            "dependencies": [
                {"name": "idna", "version": "3.16", "vulns": []},
            ],
            "fixes": [],
        }
        return subprocess.CompletedProcess(cmd, 0, pip_audit_extras.json.dumps(payload), "")

    monkeypatch.setattr(pip_audit_extras, "_run", _fake_run)

    result = pip_audit_extras.run_audits([target])

    assert result["status"] == "ok"
    assert result["targets"][0]["no_deps"] is True
    assert result["targets"][0]["resolved_dependency_count"] == 1


def test_run_audits_allows_documented_optional_dependency_advisories(monkeypatch):
    target = pip_audit_extras.AuditTarget(
        name="pyproject[all]",
        kind="pyproject-extra",
        extra="all",
        source="pyproject.toml",
        requirements=["realesrgan>=0.3,<1"],
    )

    monkeypatch.setattr(
        pip_audit_extras.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip_audit" else None,
    )

    def _fake_run(cmd, cwd, timeout, env=None):  # noqa: ANN001
        payload = {
            "dependencies": [
                {
                    "name": "basicsr",
                    "version": "1.4.2",
                    "vulns": [
                        {
                            "id": "CVE-2024-27763",
                            "aliases": ["GHSA-86w8-vhw6-q9qq"],
                            "fix_versions": [],
                        }
                    ],
                },
                {
                    "name": "transformers",
                    "version": "4.57.6",
                    "vulns": [
                        {
                            "id": "CVE-2026-1839",
                            "aliases": ["GHSA-69w3-r845-3855"],
                            "fix_versions": ["5.0.0rc3"],
                        }
                    ],
                },
            ],
            "fixes": [],
        }
        return subprocess.CompletedProcess(cmd, 1, pip_audit_extras.json.dumps(payload), "")

    monkeypatch.setattr(pip_audit_extras, "_run", _fake_run)

    result = pip_audit_extras.run_audits([target])
    target_result = result["targets"][0]

    assert result["status"] == "ok"
    assert result["vulnerability_count"] == 2
    assert result["allowed_vulnerability_count"] == 2
    assert result["unallowed_vulnerability_count"] == 0
    assert target_result["status"] == "ok"
    assert all(vulnerability["allowed"] for vulnerability in target_result["vulnerabilities"])
    assert {vulnerability["waiver"]["docs"] for vulnerability in target_result["vulnerabilities"]} == {
        "docs/PYTHON_ADVISORIES.md"
    }


def test_run_audits_fails_unlisted_advisories(monkeypatch):
    target = pip_audit_extras.AuditTarget(
        name="pyproject[all]",
        kind="pyproject-extra",
        extra="all",
        source="pyproject.toml",
        requirements=["example>=1"],
    )

    monkeypatch.setattr(
        pip_audit_extras.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip_audit" else None,
    )

    def _fake_run(cmd, cwd, timeout, env=None):  # noqa: ANN001
        payload = {
            "dependencies": [
                {
                    "name": "example",
                    "version": "1.0",
                    "vulns": [{"id": "CVE-2099-0001", "aliases": [], "fix_versions": ["1.1"]}],
                }
            ],
            "fixes": [],
        }
        return subprocess.CompletedProcess(cmd, 1, pip_audit_extras.json.dumps(payload), "")

    monkeypatch.setattr(pip_audit_extras, "_run", _fake_run)

    result = pip_audit_extras.run_audits([target])

    assert result["status"] == "fail"
    assert result["targets"][0]["status"] == "fail"
    assert result["targets"][0]["unallowed_vulnerability_count"] == 1
    assert result["targets"][0]["vulnerabilities"][0]["allowed"] is False


def test_python_advisory_doc_documents_every_allowed_entry():
    doc = (REPO_ROOT / "docs" / "PYTHON_ADVISORIES.md").read_text(encoding="utf-8")
    for advisory_id, advisory in pip_audit_extras.ALLOWED_ADVISORIES.items():
        assert advisory_id in doc
        for alias in advisory.aliases:
            assert alias in doc


def test_run_audits_converts_subprocess_timeout_to_target_failure(monkeypatch):
    target = pip_audit_extras.AuditTarget(
        name="pyproject[all]",
        kind="pyproject-extra",
        extra="all",
        source="pyproject.toml",
        requirements=["flask>=3.0,<4"],
    )

    monkeypatch.setattr(
        pip_audit_extras.importlib.util,
        "find_spec",
        lambda name: object() if name == "pip_audit" else None,
    )

    def _timeout(cmd, cwd, timeout, env=None):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(pip_audit_extras, "_run", _timeout)

    result = pip_audit_extras.run_audits([target], process_timeout=1)

    assert result["status"] == "fail"
    assert result["targets"][0]["exit_code"] == -1
    assert "timed out" in result["targets"][0]["parse_error"]


def test_run_audits_skips_when_pip_audit_is_missing(monkeypatch):
    monkeypatch.setattr(pip_audit_extras.importlib.util, "find_spec", lambda name: None)

    result = pip_audit_extras.run_audits([])

    assert result["status"] == "skipped"
    assert "pip-audit" in result["message"]
