from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
