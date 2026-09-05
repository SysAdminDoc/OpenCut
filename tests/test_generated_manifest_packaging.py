"""Generated manifests must ship with the package, source tree or frozen build.

v1.55.1 shipped a PyInstaller artifact with no ``opencut/_generated`` at all:
``collect_data_files`` is per-subpackage, the spec named only ``opencut.data``,
and nothing noticed because the whole suite runs against the source tree where
the files exist. A user found it (issue #8).

These tests cover the three places that failure could return from: the declared
manifest set drifting from the source tree, the spec dropping a data-bearing
subpackage, and a built artifact missing files that the source tree has.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from opencut._generated import (
    GENERATED_DIR,
    REQUIRED_MANIFESTS,
    GeneratedManifestMissing,
    missing_manifests,
    require_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "opencut_server.spec"
PACKAGE_ROOT = REPO_ROOT / "opencut"


def _data_bearing_subpackages(root: Path) -> set[str]:
    """Return dotted names of ``opencut.*`` packages that carry non-Python files."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        if "__init__.py" not in filenames:
            continue
        if not any(not name.endswith((".py", ".pyc")) for name in filenames):
            continue
        rel = Path(dirpath).relative_to(root.parent)
        found.add(str(rel).replace(os.sep, "."))
    return found


def test_required_manifests_match_the_source_tree():
    """A newly generated manifest cannot be left out of the declared set."""
    on_disk = {path.name for path in GENERATED_DIR.glob("*.json")}
    assert on_disk == set(REQUIRED_MANIFESTS), (
        "opencut/_generated/__init__.py REQUIRED_MANIFESTS has drifted from the "
        "source tree. Add or remove the manifest names listed in the diff.\n"
        f"  only on disk: {sorted(on_disk - set(REQUIRED_MANIFESTS))}\n"
        f"  only declared: {sorted(set(REQUIRED_MANIFESTS) - on_disk)}"
    )


def test_source_tree_is_not_missing_any_required_manifest():
    assert missing_manifests() == []


def test_require_manifest_raises_a_named_error_instead_of_degrading(tmp_path, monkeypatch):
    monkeypatch.setattr("opencut._generated.GENERATED_DIR", tmp_path)
    with pytest.raises(GeneratedManifestMissing) as excinfo:
        require_manifest("route_manifest.json")
    message = str(excinfo.value)
    assert "route_manifest.json" in message
    assert excinfo.value.names == ["route_manifest.json"]
    # The message has to say what to do about it; a bare FileNotFoundError sent
    # the reporter of issue #8 looking for a corrupted install.
    assert "opencut/_generated" in message


def test_missing_manifests_reports_every_absent_file(tmp_path):
    (tmp_path / "route_manifest.json").write_text("{}", encoding="utf-8")
    missing = missing_manifests(tmp_path)
    assert "route_manifest.json" not in missing
    assert set(missing) == set(REQUIRED_MANIFESTS) - {"route_manifest.json"}


def test_spec_collects_every_data_bearing_subpackage():
    """The spec must derive data packages, not name one and drift from the rest."""
    spec_source = SPEC_PATH.read_text(encoding="utf-8")
    expected = _data_bearing_subpackages(PACKAGE_ROOT)
    assert "opencut._generated" in expected and "opencut.data" in expected

    hardcoded = set(re.findall(r"collect_data_files\(\s*['\"](opencut[\w.]*)['\"]", spec_source))
    if hardcoded:
        missing = expected - hardcoded
        assert not missing, (
            "opencut_server.spec names data subpackages literally and is missing "
            f"{sorted(missing)}. Derive them from the source tree instead."
        )
    else:
        assert "_opencut_data_subpackages" in spec_source, (
            "opencut_server.spec must either name every data-bearing opencut "
            "subpackage or derive them; it now does neither, so a new data "
            "package would silently not ship."
        )


def test_built_artifact_contains_every_packaged_data_file():
    """When a build exists, prove the artifact actually carries the data files."""
    internal = REPO_ROOT / "dist" / "OpenCut-Server" / "_internal" / "opencut"
    if not internal.is_dir():
        pytest.skip("no dist/OpenCut-Server build present")

    missing: list[str] = []
    for package in sorted(_data_bearing_subpackages(PACKAGE_ROOT)):
        source_dir = REPO_ROOT / Path(package.replace(".", os.sep))
        target_dir = internal.parent / Path(package.replace(".", os.sep))
        for source_file in source_dir.rglob("*"):
            if not source_file.is_file():
                continue
            if source_file.suffix in {".py", ".pyc"} or "__pycache__" in source_file.parts:
                continue
            relative = source_file.relative_to(source_dir)
            if not (target_dir / relative).is_file():
                missing.append(f"{package}/{relative.as_posix()}")

    assert not missing, (
        "The built artifact is missing data files that the source tree has. "
        "Rebuild after the spec change; collect_data_files is per-subpackage.\n"
        + "\n".join(f"  {name}" for name in missing[:25])
    )


# ---------------------------------------------------------------------------
# Loaders must not invent answers when the manifest did not ship
# ---------------------------------------------------------------------------

def test_workflow_metadata_does_not_claim_implemented_without_a_manifest(tmp_path, caplog):
    """The worst silent degrade: readiness fabricated as "implemented".

    A packaged build with no opencut/_generated hit this on every call, so the
    system reported all 53 workflow endpoints as runnable on the strength of a
    file it could not read.
    """
    from opencut.core import workflow

    missing = tmp_path / "route_manifest.json"
    with caplog.at_level("ERROR"):
        metadata = workflow._workflow_route_metadata(missing)

    assert metadata, "the endpoint fallback itself should survive"
    readiness = {entry["readiness"] for entry in metadata.values()}
    assert readiness == {"unknown"}, (
        f"readiness was fabricated as {readiness} from a manifest that does not exist"
    )
    assert "opencut/_generated" in caplog.text


def test_extended_mcp_loader_names_the_missing_manifest(tmp_path):
    from opencut._generated import GeneratedManifestMissing
    from opencut.mcp_extended_tools import load_route_manifest

    with pytest.raises(GeneratedManifestMissing) as excinfo:
        load_route_manifest(tmp_path / "route_manifest.json")
    assert "route_manifest.json" in str(excinfo.value)
    assert "opencut/_generated" in str(excinfo.value)


def test_surface_ratchet_loader_names_the_missing_manifest(tmp_path):
    from opencut._generated import GeneratedManifestMissing
    from opencut.core.surface_ratchet import load_manifest

    with pytest.raises(GeneratedManifestMissing):
        load_manifest(tmp_path / "route_manifest.json")


def test_cli_route_manifest_error_names_the_packaging_cause(tmp_path):
    import click

    from opencut.cli import _load_cli_route_manifest

    with pytest.raises(click.ClickException) as excinfo:
        _load_cli_route_manifest(tmp_path / "route_manifest.json")
    assert "opencut/_generated" in str(excinfo.value)


def test_agent_skill_validation_reports_a_missing_manifest(monkeypatch, tmp_path, caplog):
    from opencut.core import agent_skills

    monkeypatch.setattr(agent_skills, "ROUTE_MANIFEST_PATH", tmp_path / "route_manifest.json")
    with caplog.at_level("ERROR"):
        assert agent_skills._load_route_method_pairs() == set()
    assert "opencut/_generated" in caplog.text
