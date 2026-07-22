from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

from opencut import dependency_support

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_matrix_script():
    path = REPO_ROOT / "scripts" / "check_dependency_matrix.py"
    spec = importlib.util.spec_from_file_location("test_dependency_matrix_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pyproject_and_runtime_publish_one_python_range():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["requires-python"] == dependency_support.PYTHON_REQUIRES
    assert dependency_support.PYTHON_VERSIONS == ("3.11", "3.12", "3.13", "3.14")
    assert "Programming Language :: Python :: 3.14" in project["classifiers"]
    dependency_support.assert_extra_names(project["optional-dependencies"])


def test_platform_specific_gpu_support_is_explicit():
    assert dependency_support.extra_support("ai-gpu", platform_name="win32")["supported"] is True
    assert dependency_support.extra_support("ai-gpu", platform_name="linux")["supported"] is True
    mac = dependency_support.extra_support("ai-gpu", platform_name="darwin")
    assert mac["supported"] is False
    assert "win32, linux" in mac["reason"]


def test_macos_python_314_caption_wheel_gap_is_explicit():
    for extra in ("standard", "captions"):
        status = dependency_support.extra_support(
            extra,
            version=(3, 14),
            platform_name="darwin",
        )
        assert status["supported"] is False
        assert "onnxruntime cp314 wheel" in status["reason"]

    video = dependency_support.extra_support(
        "video",
        version=(3, 14),
        platform_name="darwin",
    )
    assert video["supported"] is True


def test_macos_generic_target_excludes_onnx_ai_lanes():
    for version in ((3, 11), (3, 12), (3, 13), (3, 14)):
        for extra in ("ai", "all"):
            status = dependency_support.extra_support(
                extra,
                version=version,
                platform_name="darwin",
            )
            assert status["supported"] is False
            assert "onnxruntime >=1.26" in status["reason"]


def test_python_ceiling_rejects_unverified_future_runtime():
    assert dependency_support.python_supported((3, 11)) is True
    assert dependency_support.python_supported((3, 14)) is True
    assert dependency_support.python_supported((3, 10)) is False
    assert dependency_support.python_supported((3, 15)) is False
    result = dependency_support.extra_support("standard", version=(3, 15))
    assert result["supported"] is False
    assert "Python 3.11-3.14" in result["reason"]


def test_whisperx_conflict_never_emits_an_install_command():
    status = dependency_support.dependency_support("whisperx")

    assert status["supported"] is False
    assert "torchvision <0.24" in status["reason"]
    assert "Torch >=2.10" in status["reason"]
    assert status["install_hint"] == ""


def test_dependency_dashboard_uses_canonical_support_contract(client):
    response = client.get("/system/dependencies?fresh=1")
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["whisperx"]["supported"] is False
    assert payload["whisperx"].get("install_hint", "") == ""
    assert "Torch >=2.10" in payload["whisperx"]["support_reason"]
    assert payload["pyannote.audio"]["supported"] is True
    if not payload["pyannote.audio"]["installed"]:
        assert payload["pyannote.audio"]["install_hint"] == 'pip install "opencut-ppro[diarize]"'


def test_matrix_script_reports_machine_readable_contract(capsys):
    module = _load_matrix_script()

    exit_code = module.main(["--json", "--extra", "standard"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["results"] == [{"extra": "standard", "reason": "", "status": "supported"}]


def test_local_matrix_covers_every_os_and_python_lane():
    module = _load_matrix_script()

    assert set(module.matrix_lanes()) == {
        (version, platform_name)
        for platform_name in dependency_support.PLATFORMS
        for version in dependency_support.PYTHON_VERSIONS
    }

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/check_dependency_matrix.py --matrix" in readme
