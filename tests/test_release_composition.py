from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from opencut.tools import license_gate

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _composition_fixture() -> dict:
    return {
        "$schema": "https://opencut.dev/schemas/release-composition/v1",
        "schema_version": 1,
        "generated_at": "2026-07-21T12:00:00Z",
        "application": {"name": "opencut-ppro", "version": "1.41.0"},
        "lane": "windows",
        "python": {
            "components": [
                {
                    "name": "flask",
                    "version": "3.1.3",
                    "direct": True,
                    "purl": "pkg:pypi/flask@3.1.3",
                    "license": "BSD-3-Clause",
                    "source_url": "https://github.com/pallets/flask",
                    "download_sha256": ["1" * 64],
                    "installed_tree_sha256": "2" * 64,
                    "dependencies": ["werkzeug"],
                    "license_documents": [{"path": "LICENSE.txt", "sha256": "3" * 64, "text": "Flask license"}],
                },
                {
                    "name": "werkzeug",
                    "version": "3.1.8",
                    "direct": False,
                    "purl": "pkg:pypi/werkzeug@3.1.8",
                    "license": "BSD-3-Clause",
                    "source_url": "https://github.com/pallets/werkzeug",
                    "download_sha256": ["4" * 64],
                    "installed_tree_sha256": "5" * 64,
                    "dependencies": [],
                    "license_documents": [{"path": "LICENSE.txt", "sha256": "6" * 64, "text": "Werkzeug license"}],
                },
            ]
        },
        "bundled_components": [
            {
                "name": "ffmpeg",
                "bundled": {"ok": True, "version": "8.1.2"},
                "source": {"url": "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz", "sha256": "7" * 64},
                "build": {"origin": "fixture builder", "configuration": "--enable-gpl"},
                "redistribution": {"license": "GPL-3.0-or-later", "corresponding_source": "download and verify"},
                "artifacts": [{"path": "ffmpeg", "sha256": "8" * 64}],
            }
        ],
        "artifacts": [
            {"path": "OpenCut-Server", "kind": "directory", "size": 10, "sha256": "9" * 64, "file_count": 2}
        ],
    }


def test_committed_release_lock_is_exact_and_hash_locked():
    module = _load_script("release_composition")
    entries = module.parse_hashed_lock(REPO_ROOT / "requirements-release-lock.txt")
    direct = module.direct_requirement_names(REPO_ROOT / "requirements.txt")

    assert len(entries) > len(direct) >= 10
    assert direct <= set(entries)
    assert all(entry.version and entry.hashes for entry in entries.values())
    assert any(entry.marker for entry in entries.values())


def test_release_lock_markers_select_only_the_target_lane():
    module = _load_script("release_composition")
    entries = {
        "common": module.LockEntry("common", "1", ("a" * 64,)),
        "windows": module.LockEntry("windows", "1", ("b" * 64,), "sys_platform == 'win32'"),
        "linux": module.LockEntry("linux", "1", ("c" * 64,), "sys_platform == 'linux'"),
    }

    active = module.active_lock_entries(entries, environment={"sys_platform": "darwin"})

    assert set(active) == {"common"}


def test_release_lock_rejects_overlapping_conditional_versions():
    module = _load_script("release_composition")
    entries = {
        "package": module.LockEntry("package", "1", ("a" * 64,), "python_version >= '3.11'"),
        "package#2": module.LockEntry("package", "2", ("b" * 64,), "python_version >= '3.12'"),
    }

    with pytest.raises(module.CompositionError, match="overlapping lock markers"):
        module.active_lock_entries(entries, environment={"python_version": "3.12"})


def test_release_lock_rejects_unhashed_or_unpinned_inputs(tmp_path):
    module = _load_script("release_composition")
    lock = tmp_path / "lock.txt"
    lock.write_text("flask>=3\nrequests==2.0\n", encoding="utf-8")

    with pytest.raises(module.CompositionError, match="not exactly pinned|no SHA-256"):
        module.parse_hashed_lock(lock)


def test_tree_record_changes_when_artifact_changes(tmp_path):
    module = _load_script("release_composition")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "server.bin"
    payload.write_bytes(b"one")
    first = module.tree_record(artifact)
    payload.write_bytes(b"two")
    second = module.tree_record(artifact)

    assert first["kind"] == "directory"
    assert first["file_count"] == 1
    assert first["sha256"] != second["sha256"]


def test_ffmpeg_provenance_rejects_hash_drift(tmp_path):
    module = _load_script("release_composition")
    binary = tmp_path / "ffmpeg"
    binary.write_bytes(b"binary")
    manifest = tmp_path / "ffmpeg.json"
    manifest.write_text(
        json.dumps(
            {
                "bundled": {"ok": True},
                "source": {"url": "https://example.com/source.tar.xz", "sha256": "a" * 64},
                "build": {"origin": "fixture", "configuration": "--enable-gpl"},
                "redistribution": {"license": "GPL-3.0-or-later", "corresponding_source": "download"},
                "artifacts": [{"path": str(binary), "sha256": "b" * 64}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(module.CompositionError, match="artifact hash"):
        module.validate_ffmpeg_provenance(manifest)


def test_resolved_sbom_contains_transitive_graph_and_bundled_hashes():
    sbom = _load_script("sbom").build_resolved_sbom(_composition_fixture())
    properties = {item["name"]: item["value"] for item in sbom["metadata"]["properties"]}
    components = {item["name"]: item for item in sbom["components"]}
    dependencies = {item["ref"]: item["dependsOn"] for item in sbom["dependencies"]}

    assert properties["opencut:sbom:fidelity"] == "resolved-artifact"
    assert components["flask"]["hashes"] == [{"alg": "SHA-256", "content": "2" * 64}]
    assert components["ffmpeg"]["licenses"] == [{"license": {"name": "GPL-3.0-or-later"}}]
    assert dependencies["pkg:pypi/flask@3.1.3"] == ["pkg:pypi/werkzeug@3.1.8"]


def test_notices_include_exact_ffmpeg_source_and_package_license_text():
    notices = _load_script("release_composition").render_notices(_composition_fixture())

    assert "Flask license" in notices
    assert "https://ffmpeg.org/releases/ffmpeg-8.1.2.tar.xz" in notices
    assert "download and verify" in notices


def test_license_gate_covers_every_direct_requirement():
    expected = license_gate._read_requirements(REPO_ROOT / "requirements.txt")
    report = license_gate.lint()
    findings = [item for item in report.findings if item.surface == "requirements"]

    assert len(findings) == len(expected)
    assert findings
    assert all(item.severity == "info" for item in findings)


def test_release_assembly_paths_require_resolved_evidence():
    docker = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    inno = (REPO_ROOT / "OpenCut.iss").read_text(encoding="utf-8")
    wpf = (REPO_ROOT / "installer" / "InstallerBuilder.ps1").read_text(encoding="utf-8")
    linux = (REPO_ROOT / "scripts" / "build_linux_packages.sh").read_text(encoding="utf-8")

    assert "--require-hashes --requirement requirements-release-lock.txt" in docker
    assert "scripts/release_composition.py" in docker
    assert r'dist\release-metadata\*"; DestDir: "{app}\release-metadata' in inno
    assert "release_composition.py" in wpf
    assert "PythonExe" in wpf
    assert 'Path.Combine(tempDir, "release-metadata")' in (
        REPO_ROOT / "installer" / "src" / "OpenCut.Installer" / "Services" / "InstallEngine.cs"
    ).read_text(encoding="utf-8")
    assert "release_composition.py" in linux
