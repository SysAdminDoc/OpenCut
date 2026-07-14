import re
import tomllib
from pathlib import Path

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


def _python_floor_target(requires_python: str) -> str:
    match = re.fullmatch(r">=(\d+)\.(\d+)", requires_python)
    assert match, f"unsupported requires-python floor: {requires_python}"
    major, minor = match.groups()
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
        REPO_ROOT / "opencut" / "routes" / "system.py",
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
    assert project["requires-python"] == ">=3.11"
    classifiers = set(project["classifiers"])
    assert "Programming Language :: Python :: 3.9" not in classifiers
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers


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


def test_optional_dependency_security_floor_pins():
    extras = _pyproject()["project"]["optional-dependencies"]

    for extra in ("standard", "video", "all"):
        deps = _dep_names(extras[extra])
        assert deps["opencv-python-headless"] == "opencv-python-headless>=4.13,<5"
        assert deps["pillow"] == "Pillow>=12.2,<13"

    for extra in ("captions", "all"):
        deps = _dep_names(extras[extra])
        assert deps["pillow"] == "Pillow>=12.2,<13"

    deps = _dep_names(extras["captions-whisperx"])
    assert deps["whisperx"] == "whisperx>=3.8.5,<4"
    assert "whisperx" not in _dep_names(extras["all"])
    assert _dep_names(extras["torch-stack"])["whisperx"] == "whisperx>=3.8.5,<4"

    for extra in ("ai", "all"):
        deps = _dep_names(extras[extra])
        assert deps["onnxruntime"] == "onnxruntime>=1.26,<2"

    deps = _dep_names(extras["ai-gpu"])
    assert deps["onnxruntime-gpu"] == "onnxruntime-gpu>=1.26,<2"

    # CVE-2026-24747 — torch.load weights_only RCE fixed in 2.10.0.
    torch_stack = _dep_names(extras["torch-stack"])
    assert torch_stack["torch"] == "torch>=2.10.0"
    assert torch_stack["torchvision"] == "torchvision>=0.25.0"
    assert torch_stack["transformers"] == "transformers>=4.30"

    depth = _dep_names(extras["depth"])
    assert depth["torch"] == "torch>=2.10.0"
    assert depth["torchvision"] == "torchvision>=0.25.0"
    assert depth["transformers"] == "transformers>=5.3"

    # picklescan supply-chain scanner must ship in every pickle-weight lane.
    for extra in ("ai", "ai-gpu", "depth", "torch-stack"):
        assert _dep_names(extras[extra])["picklescan"] == "picklescan>=1.0.3"


def test_transformers_floor_exception_is_confined_to_torch_stack():
    extras = _pyproject()["project"]["optional-dependencies"]
    transformers_specs = {
        extra: _dep_names(requirements).get("transformers")
        for extra, requirements in extras.items()
    }

    assert transformers_specs["torch-stack"] == "transformers>=4.30"
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
        "opencv-python-headless>=4.13,<5",
        "Pillow>=12.2,<13",
        "# onnxruntime-gpu>=1.25",
        "# whisperx>=3.8.5",
    ]
    for needle in required:
        assert needle in text
    assert "pydub>=0.25" not in text


def test_requirements_txt_matches_core_and_standard_pyproject_deps():
    project = _pyproject()["project"]
    expected = set(project["dependencies"])
    expected.update(project["optional-dependencies"]["standard"])

    assert expected <= _active_requirements_txt()
