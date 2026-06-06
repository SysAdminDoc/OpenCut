from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _dep_names(dependencies):
    names = {}
    for dep in dependencies:
        name = dep.split("[", 1)[0].split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip().lower()
        names[name] = dep
    return names


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


def test_python_313_classifier_requires_ci_lane():
    """RA-21: do not advertise Python 3.13 until a workflow tests it."""
    classifiers = set(_pyproject()["project"]["classifiers"])
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(WORKFLOW_DIR.glob("*.yml"))
    )

    if "Programming Language :: Python :: 3.13" in classifiers:
        assert "python-version: '3.13'" in workflow_text or 'python-version: "3.13"' in workflow_text
    else:
        assert "Programming Language :: Python :: 3.13" not in classifiers


def test_core_dependency_security_floor_pins():
    deps = _dep_names(_pyproject()["project"]["dependencies"])
    assert deps["flask-cors"] == "flask-cors>=6.0,<7"


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
        assert deps["onnxruntime"] == "onnxruntime>=1.25,<2"

    deps = _dep_names(extras["ai-gpu"])
    assert deps["onnxruntime-gpu"] == "onnxruntime-gpu>=1.25,<2"

    torch_stack = _dep_names(extras["torch-stack"])
    assert torch_stack["torch"] == "torch>=2.0"
    assert torch_stack["transformers"] == "transformers>=4.30"


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
