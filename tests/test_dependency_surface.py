import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _dep_names(dependencies):
    names = {}
    for dep in dependencies:
        name = dep.split("[", 1)[0].split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip().lower()
        names[name] = dep
    return names


def _active_requirements_txt() -> set[str]:
    requirements = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if requirement and not requirement.startswith("#"):
            requirements.add(requirement)
    return requirements


def _python_bounds(requires_python: str) -> tuple[tuple[int, int], tuple[int, int]]:
    match = re.fullmatch(r">=(\d+)\.(\d+),<(\d+)\.(\d+)", requires_python)
    assert match, f"unsupported requires-python range: {requires_python}"
    min_major, min_minor, max_major, max_minor = (int(value) for value in match.groups())
    return (min_major, min_minor), (max_major, max_minor)


def _python_floor_target(requires_python: str) -> str:
    (major, minor), _ = _python_bounds(requires_python)
    return f"py{major}{minor}"


def _classifier_versions(classifiers: list[str]) -> set[str]:
    prefix = "Programming Language :: Python :: "
    versions = set()
    for classifier in classifiers:
        if not classifier.startswith(prefix):
            continue
        suffix = classifier.removeprefix(prefix)
        if re.fullmatch(r"\d+\.\d+", suffix):
            versions.add(suffix)
    return versions


def test_deep_translator_removed_from_install_surfaces():
    """PYSEC-2022-252 has no fixed deep-translator release; keep it out of installs."""
    checked = [
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "README.md",
        REPO_ROOT / "opencut" / "routes" / "system_runtime_routes.py",
        REPO_ROOT / "opencut" / "core" / "dub_pipeline.py",
    ]

    offenders = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        if "deep-translator" in text or "deep_translator" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []


def test_python_floor_tracks_security_dependency_floor():
    """F121/F133/F135 require a Python 3.11+ install surface."""
    project = _pyproject()["project"]
    assert project["requires-python"] == ">=3.11,<3.15"
    classifiers = set(project["classifiers"])
    assert "Programming Language :: Python :: 3.9" not in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.14" in classifiers


def test_every_source_install_and_launch_surface_tracks_python_floor():
    """Keep user-facing entry paths aligned with canonical package metadata."""
    project_range = _pyproject()["project"]["requires-python"]
    (min_major, min_minor), (max_major, max_minor) = _python_bounds(project_range)
    project_floor = f"{min_major}.{min_minor}"
    project_ceiling = f"{max_major}.{max_minor - 1}"
    surfaces = [
        "README.md",
        "install.py",
        "Install.ps1",
        "OpenCut-Server.bat",
        "OpenCut-Server.vbs",
        "OpenCut-Server.sh",
        "scripts/bootstrap_check.py",
        "opencut/__init__.py",
    ]
    tuple_floor = re.compile(rf"\({min_major},\s*{min_minor}\)")
    tuple_ceiling = re.compile(rf"\({max_major},\s*{max_minor - 1}\)")
    for relative_path in surfaces:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert project_floor in text or tuple_floor.search(text), (
            f"{relative_path} must advertise or enforce Python {project_floor}"
        )
        assert project_ceiling in text or tuple_ceiling.search(text), (
            f"{relative_path} must advertise or enforce Python through {project_ceiling}"
        )

    executable_surfaces = surfaces[1:]
    for relative_path in executable_surfaces:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8").lower()
        assert "require" in text, f"{relative_path} must state the required floor"
        assert "python.org/downloads" in text or "python 3.11-3.14" in text, (
            f"{relative_path} must provide Python upgrade remediation"
        )


def test_runtime_guard_rejects_unsupported_python_with_actionable_error():
    import opencut

    with pytest.raises(RuntimeError) as exc_info:
        opencut._require_supported_python((3, 10, 14))

    message = str(exc_info.value)
    assert "requires Python 3.11-3.14" in message
    assert "detected Python 3.10.14" in message
    assert "python.org/downloads" in message

    opencut._require_supported_python((3, 11, 0))
    opencut._require_supported_python((3, 14, 0))
    with pytest.raises(RuntimeError, match="Python 3.11-3.14"):
        opencut._require_supported_python((3, 15, 0))


def test_ruff_target_tracks_python_floor():
    data = _pyproject()
    project = data["project"]
    floor_target = _python_floor_target(project["requires-python"])
    advertised_versions = _classifier_versions(project["classifiers"])

    assert data["tool"]["ruff"]["target-version"] == floor_target
    assert "3.11" in advertised_versions
    assert all(tuple(map(int, version.split("."))) >= (3, 11) for version in advertised_versions)


def test_core_dependency_security_floor_pins():
    deps = _dep_names(_pyproject()["project"]["dependencies"])
    assert deps["flask-cors"] == "flask-cors>=6.0,<7"
    # RA-23 — Werkzeug/Jinja2 ship transitively with flask; their resolver
    # floor must match the lockfile security level even without the lockfile.
    assert deps["werkzeug"] == "Werkzeug>=3.1.6"
    assert deps["jinja2"] == "Jinja2>=3.1.6"


def test_click_floor_patches_command_injection_advisory():
    """CVE-2026-7246 / PYSEC-2026-2132 — click.edit() command injection fixed 8.3.3.

    Click is a core CLI/server dependency; every declared and locked floor must
    sit at or above the fixed release so no install lane admits the vulnerable
    8.3.0-8.3.2 range.
    """
    deps = _dep_names(_pyproject()["project"]["dependencies"])
    assert deps["click"] == "click>=8.3.3,<9"

    requirements = _dep_names(_active_requirements_txt())
    assert requirements["click"] == "click>=8.3.3,<9"

    lock = {}
    for line in (REPO_ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        pinned = line.strip()
        if pinned and not pinned.startswith("#") and "==" in pinned:
            name, version = pinned.split("==", 1)
            lock[name.strip().lower()] = version.strip()
    locked = tuple(int(part) for part in lock["click"].split("."))
    assert locked >= (8, 3, 3), f"locked click {lock['click']} is below the CVE-2026-7246 fix"


def test_transitive_web_dep_floors_match_lockfile():
    """RA-23 — clean-resolver installs cannot select vulnerable web deps.

    requirements-lock.txt holds urllib3 2.7.0, Werkzeug 3.1.7, requests 2.33.0,
    Jinja2 3.1.6. The pyproject resolver lane must floor the same packages so
    `pip install opencut-ppro[all]` without the lock cannot regress below the
    advisory-fixed versions.
    """
    project = _pyproject()["project"]
    core = _dep_names(project["dependencies"])
    extras = project["optional-dependencies"]

    # Always-present web deps (via flask) floored in core.
    assert core["werkzeug"] == "Werkzeug>=3.1.6"  # CVE-2026-21860 / CVE-2025-66221
    assert core["jinja2"] == "Jinja2>=3.1.6"  # CVE-2025-27516

    # Transitive HTTP stack floored in every lane that pulls it.
    for extra in ("standard", "all"):
        deps = _dep_names(extras[extra])
        assert deps["requests"] == "requests>=2.33.0"  # CVE-2026-25645
        assert deps["urllib3"] == "urllib3>=2.6.3"  # CVE-2026-21441


def test_os_credential_vault_dependency_is_core_and_locked():
    core = _dep_names(_pyproject()["project"]["dependencies"])
    assert core["keyring"] == "keyring>=25.7,<26"
    lock = (REPO_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "keyring==25.7.0" in lock.splitlines()


def test_plugin_signature_verifier_dependency_is_core_and_locked():
    core = _dep_names(_pyproject()["project"]["dependencies"])
    assert core["cryptography"] == "cryptography>=48.0.1,<49"
    requirements = _dep_names(_active_requirements_txt())
    assert requirements["cryptography"] == "cryptography>=48.0.1,<49"
    lock = (REPO_ROOT / "requirements-lock.txt").read_text(encoding="utf-8")
    assert "cryptography==48.0.1" in lock.splitlines()


def test_optional_dependency_security_floor_pins():
    extras = _pyproject()["project"]["optional-dependencies"]

    for extra in ("standard", "video", "all"):
        deps = _dep_names(extras[extra])
        assert deps["opencv-python"] == "opencv-python>=5,<6"
        assert deps["pillow"] == "Pillow>=12.3.0,<13"

    for extra in ("captions", "all"):
        deps = _dep_names(extras[extra])
        assert deps["pillow"] == "Pillow>=12.3.0,<13"

    assert "captions-whisperx" not in extras
    assert "whisperx" not in _dep_names(extras["all"])
    assert "whisperx" not in _dep_names(extras["torch-stack"])
    assert "music" not in extras
    assert "enhance" not in extras

    for extra in ("ai", "all"):
        deps = _dep_names(extras[extra])
        assert deps["onnxruntime"] == "onnxruntime>=1.26,<2"

    deps = _dep_names(extras["ai-gpu"])
    assert deps["onnxruntime-gpu"] == "onnxruntime-gpu>=1.26,<2"

    # CVE-2026-24747 — torch.load weights_only RCE fixed in 2.10.0.
    torch_stack = _dep_names(extras["torch-stack"])
    assert torch_stack["torch"] == "torch>=2.10.0"
    assert torch_stack["torchvision"] == "torchvision>=0.25.0"
    assert torch_stack["transformers"] == "transformers>=5.3"

    depth = _dep_names(extras["depth"])
    assert depth["torch"] == "torch>=2.10.0"
    assert depth["torchvision"] == "torchvision>=0.25.0"
    assert depth["transformers"] == "transformers>=5.3"

    # picklescan supply-chain scanner must ship in every pickle-weight lane.
    for extra in ("ai", "ai-gpu", "depth", "torch-stack"):
        assert _dep_names(extras[extra])["picklescan"] == "picklescan>=1.0.3"

    nemo = _dep_names(extras["nemo-asr"])
    assert nemo["picklescan"] == "picklescan>=1.0.3; sys_platform == 'linux'"
    assert nemo["nemo_toolkit"] == "nemo_toolkit[asr]>=2.7.3,<2.8; sys_platform == 'linux'"
    assert nemo["torch"] == "torch>=2.10.0; sys_platform == 'linux'"
    assert nemo["huggingface-hub"] == "huggingface-hub>=0.36,<1; sys_platform == 'linux'"
    assert "nemo_toolkit" not in _dep_names(extras["all"])


def test_transformers_security_floor_covers_every_declared_extra():
    extras = _pyproject()["project"]["optional-dependencies"]
    transformers_specs = {
        extra: _dep_names(requirements).get("transformers")
        for extra, requirements in extras.items()
    }

    assert transformers_specs["torch-stack"] == "transformers>=5.3"
    assert transformers_specs["depth"] == "transformers>=5.3"
    assert {
        extra: spec
        for extra, spec in transformers_specs.items()
        if spec and extra not in {"depth", "torch-stack"}
    } == {}
    assert "transformers" not in _dep_names(extras["all"])


def test_requirements_txt_matches_security_floor():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    required = [
        "flask-cors>=6.0,<7",
        "opencv-python>=5,<6",
        "Pillow>=12.3.0,<13",
        "# onnxruntime-gpu>=1.26",
        "WhisperX is unsupported until it accepts torchvision >=0.25 / Torch >=2.10.",
    ]
    for needle in required:
        assert needle in text
    assert "pydub>=0.25" not in text


def test_runtime_security_floors_cover_bootstrap_installers():
    """Runtime installs must normalize vulnerable or conflicting package specs."""
    from opencut.security import (
        OPENCV_RUNTIME_REQUIREMENT,
        PILLOW_RUNTIME_REQUIREMENT,
        runtime_security_requirement,
    )

    expected = "Pillow>=12.3.0,<13"
    assert PILLOW_RUNTIME_REQUIREMENT == expected
    assert runtime_security_requirement("Pillow") == expected
    assert runtime_security_requirement("Pillow>=10.0") == expected
    assert runtime_security_requirement("Pillow==12.2.0") == expected
    assert OPENCV_RUNTIME_REQUIREMENT == "opencv-python>=5,<6"
    assert runtime_security_requirement("opencv-python") == OPENCV_RUNTIME_REQUIREMENT
    assert runtime_security_requirement("opencv-python-headless") == OPENCV_RUNTIME_REQUIREMENT
    with pytest.raises(RuntimeError, match="No safe OpenCut install lane"):
        runtime_security_requirement("whisperx")
    assert runtime_security_requirement("numpy>=1.24") == "numpy>=1.24"

    install_source = (REPO_ROOT / "install.py").read_text(encoding="utf-8")
    assert "requirements-release-lock.txt" in install_source
    assert '"--require-hashes"' in install_source
    assert "Pillow>=10.0" not in install_source

    dashboard_source = (REPO_ROOT / "opencut" / "routes" / "system_runtime_routes.py").read_text(encoding="utf-8")
    assert f'pip install "{expected}"' in dashboard_source

    unsafe_runtime_installs = []
    unsafe_copy_commands = []
    for path in (REPO_ROOT / "opencut").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"safe_pip_install\(\s*[\"']Pillow[\"']", source):
            unsafe_runtime_installs.append(str(path.relative_to(REPO_ROOT)))
        if re.search(r"pip install(?: [^\r\n\"']+)* Pillow(?:[\"'\s]|$)", source):
            unsafe_copy_commands.append(str(path.relative_to(REPO_ROOT)))

    assert unsafe_runtime_installs == []
    assert unsafe_copy_commands == []


def test_requirements_txt_matches_core_and_standard_pyproject_deps():
    project = _pyproject()["project"]
    expected = set(project["dependencies"])
    expected.update(project["optional-dependencies"]["standard"])

    assert expected <= _active_requirements_txt()
